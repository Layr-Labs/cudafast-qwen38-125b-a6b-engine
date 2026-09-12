/* Prepared CUDA test; requires the actual backend and GPU. No model files.
 * Include the default binding to observe real begin/end return values without
 * adding production instrumentation. Link mtp.c and the usual GPU_TEST_OBJS
 * with function sections and --gc-sections; do not also link mtp_hooks.o. */
#include "ds4_qwen4exp_mtp.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
static unsigned graph_replays, graph_captures;
static int traced_begin(const ds4_decode_graph_key *key) {
    int rc=ds4_gpu_decode_graph_begin(key);if(rc==1)++graph_replays;return rc;
}
static int traced_end(const ds4_decode_graph_key *key) {
    int rc=ds4_gpu_decode_graph_end(key);if(rc==0)++graph_captures;return rc;
}
#define ds4_gpu_decode_graph_begin traced_begin
#define ds4_gpu_decode_graph_end traced_end
#include "../ds4_qwen4exp_mtp_hooks.c"
#undef ds4_gpu_decode_graph_begin
#undef ds4_gpu_decode_graph_end

bool ds4_log_is_tty(FILE *fp) { (void)fp; return false; }
enum { DIM=2560, HC=4, WIDE=DIM*HC, LR=320,
       DOWN=WIDE*4, UP=DOWN+LR*(WIDE/32)*34,
       BYTES=UP+WIDE*(LR/32)*34 };
static uint32_t rng=137;
static uint32_t next(void) { rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;return rng; }
static void q8(unsigned char *p,unsigned blocks,uint16_t scale) {
    for(unsigned b=0;b<blocks;++b,p+=34) {
        memcpy(p,&scale,2);
        for(unsigned j=0;j<32;++j)p[2+j]=(unsigned char)(int8_t)((int)(next()%31)-15);
    }
}
static void init_head(ds4_qwen4exp_mtp_head *h,void *map) {
    memset(h,0,sizeof(*h));h->head_map=map;h->head_size=BYTES;
    h->hc_head_down_offset=DOWN;h->hc_head_up_offset=UP;
    h->n_embd=DIM;h->n_hc=HC;h->n_lowrank=LR;h->max_tokens=2;
    h->block_index=48;h->rms_eps=1e-6f;h->weight_bias=1;h->round_bf16=1;
    h->hooks.hc_mixer=ds4_gpu_qwen4exp_hc_mixer_tensor;
    h->t_hyper=ds4_gpu_tensor_alloc(2ull*WIDE*4);
    h->t_h_normed=ds4_gpu_tensor_alloc(2ull*WIDE*4);
    h->t_mix_normed=ds4_gpu_tensor_alloc(2ull*WIDE*4);
    h->t_mix_lowrank=ds4_gpu_tensor_alloc(2ull*LR*4);
    h->t_mix_wide=ds4_gpu_tensor_alloc(2ull*WIDE*4);
    h->t_sample=ds4_gpu_tensor_alloc(2ull*DIM*4);
    assert(h->t_hyper&&h->t_h_normed&&h->t_mix_normed&&h->t_mix_lowrank&&h->t_mix_wide&&h->t_sample);
}
static void parity(ds4_qwen4exp_mtp_head *h,unsigned width,unsigned pass) {
    float input[2*WIDE];
    for(unsigned i=0;i<2*WIDE;++i)input[i]=((int)(next()%8191)-4095)*(1.0f/1024)+(float)pass/32;
    ds4_gpu_tensor *t[]={h->t_h_normed,h->t_mix_normed,h->t_mix_lowrank,
                        h->t_mix_wide,h->t_sample,h->t_hyper};
    void *actual[6];
    assert(ds4_gpu_tensor_write(h->t_hyper,0,input,sizeof(input)));
    for(unsigned j=0;j<5;++j)assert(ds4_gpu_tensor_fill_f32(t[j],0.375f,ds4_gpu_tensor_bytes(t[j])/4));
    assert(mtp_mix_tail(h,width-1,1,width==2));
    for(unsigned j=0;j<6;++j) {
        size_t n=ds4_gpu_tensor_bytes(t[j]);actual[j]=malloc(n);assert(actual[j]);
        assert(ds4_gpu_tensor_read(t[j],0,actual[j],n));
    }
    assert(!memcmp(actual[5],input,sizeof(input)));
    assert(ds4_gpu_tensor_write(h->t_hyper,0,input,sizeof(input)));
    for(unsigned j=0;j<5;++j)assert(ds4_gpu_tensor_fill_f32(t[j],0.375f,ds4_gpu_tensor_bytes(t[j])/4));
    assert(ds4_qwen4exp_mtp_head_mix_eager(h,width-1,1,width==2));
    for(unsigned j=0;j<6;++j) {
        size_t n=ds4_gpu_tensor_bytes(t[j]);void *expected=malloc(n);assert(expected);
        assert(ds4_gpu_tensor_read(t[j],0,expected,n));assert(!memcmp(actual[j],expected,n));
        free(actual[j]);free(expected);
    }
}
int main(void) {
    assert(ds4_gpu_init());
    if(!ds4_gpu_decode_graphs_supported()) { puts("CUDA decode graphs required");return 77; }
    unsigned char *map=mmap(NULL,BYTES,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    assert(map!=MAP_FAILED);
    for(unsigned i=0;i<WIDE;++i)((float *)map)[i]=(int)(next()%31)*(1.0f/128);
    q8(map+DOWN,LR*(WIDE/32),0x0800);q8(map+UP,WIDE*(LR/32),0x2400);
    assert(ds4_gpu_set_model_map(map,BYTES));
    ds4_qwen4exp_mtp_head h;init_head(&h,map);
    for(unsigned width=1;width<=2;++width)for(unsigned pass=0;pass<5;++pass)parity(&h,width,pass);
    assert(graph_captures>=2&&graph_replays>=6); /* Cannot silently test eager only. */
    h.weight_bias=0;for(unsigned pass=0;pass<3;++pass)parity(&h,2,pass+10);
    unsigned before=graph_captures;
    /* A live head must not replay cached device pointers after map retirement. */
    ds4_gpu_forget_model_map(map);assert(ds4_gpu_set_model_map(map,BYTES));
    for(unsigned pass=0;pass<3;++pass)parity(&h,2,pass+20);
    assert(graph_captures>before);
    before=graph_captures;ds4_gpu_tensor *old=h.t_mix_normed;
    h.t_mix_normed=ds4_gpu_tensor_alloc(2ull*WIDE*4);assert(h.t_mix_normed);
    for(unsigned pass=0;pass<3;++pass)parity(&h,1,pass+30);
    assert(graph_captures>before);ds4_gpu_tensor_free(old);
    ds4_qwen4exp_mtp_head_free(&h);assert(!h.tail_graph_state&&!h.tail_graph_release);
    init_head(&h,map);before=graph_captures;
    for(unsigned pass=0;pass<3;++pass)parity(&h,1,pass+40);
    assert(graph_captures>before);ds4_qwen4exp_mtp_head_free(&h);
    ds4_gpu_forget_model_map(map);assert(munmap(map,BYTES)==0);ds4_gpu_cleanup();
    puts("MTP tail CUDA eager/replay and lifetime parity: PASS");return 0;
}
