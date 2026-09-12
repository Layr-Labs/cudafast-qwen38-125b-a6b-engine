/* Host-only state-machine spy. No CUDA execution or mixer parity claim.
 * Build with -ffunction-sections -fdata-sections -Wl,--gc-sections and mtp.c;
 * including the binding file exercises the actual private default hook. */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../ds4_qwen4exp_mtp_hooks.c"

struct ds4_gpu_tensor { uint64_t bytes; unsigned char *data; };
static unsigned mixer_calls, copies, captures, replays, retires, syncs;
static int capturing, supported = 1, fail_end, fail_mixer;
static ds4_decode_graph_key current;
typedef struct {
    ds4_gpu_tensor *dst, *src, *mixed, *hyper;
    uint64_t offset, bytes;
    uint32_t embd, hc, rows;
} job;
static job pending;
static struct { ds4_decode_graph_key key; int state; job work; } slots[4];

uint64_t ds4_gpu_tensor_bytes(const ds4_gpu_tensor *t) { return t ? t->bytes : 0; }
ds4_gpu_tensor *ds4_gpu_tensor_alloc(uint64_t bytes) {
    ds4_gpu_tensor *t = calloc(1, sizeof(*t)); assert(t);
    t->bytes = bytes; t->data = calloc(1, bytes); assert(t->data); return t;
}
void ds4_gpu_tensor_free(ds4_gpu_tensor *t) { if(t) { free(t->data); free(t); } }
int ds4_gpu_synchronize(void) { ++syncs; return 1; }
static void run(job *j) {
    if (j->dst) memcpy(j->dst->data, j->src->data + j->offset, j->bytes);
    float *out = (float *)j->mixed->data, *in = (float *)j->hyper->data;
    for (unsigned r=0; r<j->rows; ++r) for(unsigned i=0; i<j->embd; ++i) {
        float sum=0; for(unsigned k=0;k<j->hc;++k) sum+=in[(r*j->hc+k)*j->embd+i];
        out[r*j->embd+i]=sum;
    }
}
int ds4_gpu_tensor_copy(ds4_gpu_tensor *dst, uint64_t off,
        const ds4_gpu_tensor *src, uint64_t src_off, uint64_t bytes) {
    ++copies;
    if (!dst || !src || off>dst->bytes || bytes>dst->bytes-off ||
        src_off>src->bytes || bytes>src->bytes-src_off) return 0;
    assert(off==0);
    if(capturing) { pending.dst=dst; pending.src=(ds4_gpu_tensor *)src;
                   pending.offset=src_off; pending.bytes=bytes; }
    else memcpy(dst->data,src->data+src_off,bytes);
    return 1;
}
int ds4_gpu_qwen4exp_hc_mixer_tensor(ds4_gpu_tensor *mixed, ds4_gpu_tensor *inject,
        ds4_gpu_tensor *normed, ds4_gpu_tensor *lowrank, ds4_gpu_tensor *wide,
        const ds4_gpu_tensor *hyper, const ds4_gpu_qwen4exp_slab *norm,
        const ds4_gpu_qwen4exp_slab *down, const ds4_gpu_qwen4exp_slab *up,
        const ds4_gpu_qwen4exp_slab *iw, uint32_t embd, uint32_t hc,
        uint32_t lr, uint32_t rows, float eps, float bias, int round) {
    (void)inject;(void)normed;(void)lowrank;(void)wide;(void)norm;(void)down;
    (void)up;(void)iw;(void)lr;(void)eps;(void)bias;(void)round;
    ++mixer_calls; if(fail_mixer) return 0;
    job j={0}; j.mixed=mixed;j.hyper=(ds4_gpu_tensor *)hyper;
    j.embd=embd;j.hc=hc;j.rows=rows;
    if(capturing) { pending.mixed=j.mixed;pending.hyper=j.hyper;
                   pending.embd=embd;pending.hc=hc;pending.rows=rows; }
    else run(&j);
    return 1;
}
int ds4_gpu_decode_graphs_supported(void) { return supported; }
static unsigned slot(const ds4_decode_graph_key *key) {
    for(unsigned i=0;i<4;++i) if(slots[i].state&&!memcmp(key,&slots[i].key,sizeof(*key))) return i;
    for(unsigned i=0;i<4;++i) if(!slots[i].state) { slots[i].key=*key;return i; }
    abort();
}
int ds4_gpu_decode_graph_begin(const ds4_decode_graph_key *key) {
    unsigned i=slot(key);current=*key;
    if(!slots[i].state) { slots[i].state=1;return -1; }
    if(slots[i].state==2) { run(&slots[i].work);++replays;return 1; }
    capturing=1;memset(&pending,0,sizeof(pending));return 0;
}
int ds4_gpu_decode_graph_end(const ds4_decode_graph_key *key) {
    assert(capturing);capturing=0;unsigned i=slot(key);
    if(fail_end) { fail_end=0;slots[i].state=0;return -1; }
    slots[i].work=pending;slots[i].state=2;run(&pending);++captures;return 0;
}
void ds4_gpu_decode_graph_abort(const ds4_decode_graph_key *key) {
    assert(capturing);capturing=0;slots[slot(key)].state=0;
}
void ds4_gpu_decode_graphs_invalidate(void) {
    assert(!capturing);++retires;memset(slots,0,sizeof(slots));
}
static int custom_calls;
static int custom(ds4_gpu_tensor *a,ds4_gpu_tensor *b,ds4_gpu_tensor *c,
 ds4_gpu_tensor *d,ds4_gpu_tensor *e,const ds4_gpu_tensor *f,
 const ds4_gpu_qwen4exp_slab *g,const ds4_gpu_qwen4exp_slab *h,
 const ds4_gpu_qwen4exp_slab *i,const ds4_gpu_qwen4exp_slab *j,
 uint32_t k,uint32_t l,uint32_t m,uint32_t n,float o,float p,int q) {
    ++custom_calls; return ds4_gpu_qwen4exp_hc_mixer_tensor(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q);
}
int main(void) {
    ds4_qwen4exp_mtp_head h={0};h.n_embd=2560;h.n_hc=4;h.n_lowrank=128;
    h.max_tokens=2;h.block_index=48;h.hooks.hc_mixer=ds4_gpu_qwen4exp_hc_mixer_tensor;
    h.t_hyper=ds4_gpu_tensor_alloc(81920);h.t_h_normed=ds4_gpu_tensor_alloc(81920);
    h.t_sample=ds4_gpu_tensor_alloc(20480);h.t_mix_normed=ds4_gpu_tensor_alloc(81920);
    h.t_mix_lowrank=ds4_gpu_tensor_alloc(1024);h.t_mix_wide=ds4_gpu_tensor_alloc(81920);
    for(unsigned width=1;width<=2;++width) for(unsigned pass=0;pass<4;++pass) {
        float value=(float)(width*10+pass);
        for(unsigned i=0;i<20480;++i)((float *)h.t_hyper->data)[i]=value+(i>=10240);
        unsigned before=mixer_calls;
        assert(mtp_mix_tail(&h,width-1,1,width==2));
        assert(((float *)h.t_sample->data)[0]==4*(value+(width==2)));
        if(pass>=2)assert(before==mixer_calls);
    }
    assert(captures==2&&replays==4&&copies==2);
    unsigned old=retires;h.weight_bias=1;
    assert(mtp_mix_tail(&h,1,1,1));assert(retires==old+1);
    fail_end=1;assert(mtp_mix_tail(&h,1,1,1));assert(!capturing);
    assert(mtp_mix_tail(&h,1,1,1)); /* Warm after failed finalization. */
    unsigned failed_before=mixer_calls;
    fail_mixer=1;assert(!mtp_mix_tail(&h,1,1,1));fail_mixer=0;
    assert(!capturing&&mixer_calls==failed_before+1); /* No error retry. */
    assert(mtp_mix_tail(&h,1,1,1));
    h.hooks.hc_mixer=custom;assert(mtp_mix_tail(&h,1,1,1));assert(custom_calls==1);
    h.hooks.hc_mixer=ds4_gpu_qwen4exp_hc_mixer_tensor;
    unsigned before=mixer_calls;assert(mtp_mix_tail(&h,0,2,0));assert(mixer_calls==before+1);
    setenv("DS4_MTP_NO_TAIL_GRAPH","1",1);before=mixer_calls;
    assert(mtp_mix_tail(&h,0,1,0));assert(mixer_calls==before+1);unsetenv("DS4_MTP_NO_TAIL_GRAPH");
    setenv("DS4_CUDA_NO_Q8_DP4A","1",1);before=mixer_calls;
    assert(mtp_mix_tail(&h,0,1,0));assert(mixer_calls==before+1);unsetenv("DS4_CUDA_NO_Q8_DP4A");
    supported=0;before=mixer_calls;assert(mtp_mix_tail(&h,0,1,0));assert(mixer_calls==before+1);supported=1;
    old=retires;ds4_qwen4exp_mtp_head_free(&h);
    assert(retires==old+1&&!h.tail_graph_state&&!h.tail_graph_release&&syncs>=2);
    puts("MTP tail graph host state-machine contracts: PASS");return 0;
}
