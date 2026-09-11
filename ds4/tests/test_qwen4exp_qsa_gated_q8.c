/* CUDA regression: complete Q8 activation, scale, and allocation canaries
 * against separate attention/gate/quantize; no model weights are required. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "ds4_gpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned checks;
static uint32_t seed = 0x9182abcd;
static float randf(void) {
    seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
    return ((int)(seed % 20001u)-10000)*0.0001f;
}
static void must(int ok, const char *what) {
    if (!ok) { fprintf(stderr,"QSA gated Q8: %s\n",what); exit(1); }
}
typedef struct {
    ds4_gpu_tensor *q,*k,*v,*gate,*sel,*counts,*pos,*out,*packed[2];
    uint32_t nh,nkv,hd,rows,p0,cap,maxsel;
    uint64_t so,bytes;
    int sparse,dpos;
} data;
static void run(data *a,unsigned fused) {
    if (fused) {
        must(ds4_gpu_qwen4exp_qsa_attention_gated_q8_dpos_tensor(
            a->packed[1],64,a->so,a->gate,a->q,a->k,a->v,
            a->sparse?a->sel:NULL,a->sparse?a->counts:NULL,
            a->rows,a->nh,a->nkv,a->hd,a->dpos?0:a->p0,a->cap,a->maxsel,
            1.0f/sqrtf((float)a->hd),a->dpos?a->pos:NULL),"fused call");
    } else {
        must(ds4_gpu_qwen4exp_qsa_attention_dpos_tensor(
            a->out,a->q,a->k,a->v,a->sparse?a->sel:NULL,
            a->sparse?a->counts:NULL,a->rows,a->nh,a->nkv,a->hd,
            a->dpos?0:a->p0,a->cap,a->maxsel,1.0f/sqrtf((float)a->hd),
            a->dpos?a->pos:NULL),"attention");
        must(ds4_gpu_qwen4exp_qsa_output_gate_tensor(
            a->out,a->gate,a->rows*a->nh*a->hd),"gate");
        must(ds4_gpu_quantize_q8_0_decode_rows_exact_tensor(
            a->packed[0],64,a->so,a->out,a->nh*a->hd,a->rows),"quantize");
    }
}
static void graph_run(data *a,unsigned fused,const ds4_decode_graph_key *key) {
    int state=ds4_gpu_decode_graph_begin(key);
    must(state>=0,"graph begin");
    if (state==0) { run(a,fused);must(ds4_gpu_decode_graph_end(key)==0,"graph end"); }
}
static void compare(data *a,unsigned char *b0,unsigned char *b1) {
    must(ds4_gpu_tensor_read(a->packed[0],0,b0,a->bytes),"read reference");
    must(ds4_gpu_tensor_read(a->packed[1],0,b1,a->bytes),"read candidate");
    if (memcmp(b0,b1,a->bytes)) {
        for (uint64_t i=0;i<a->bytes;i++) if (b0[i]!=b1[i]) {
            fprintf(stderr,"mismatch nh=%u hd=%u rows=%u pos=%u sparse=%d dpos=%d byte=%llu old=%02x new=%02x\n",
                a->nh,a->hd,a->rows,a->p0,a->sparse,a->dpos,
                (unsigned long long)i,b0[i],b1[i]);exit(1);
        }
    }
    checks++;
}
static void shape(uint32_t nh,uint32_t nkv,uint32_t hd) {
    data a={.nh=nh,.nkv=nkv,.hd=hd,.cap=4096,.maxsel=257};
    const uint64_t ne=(uint64_t)7*nh*hd,ce=(uint64_t)a.cap*nkv*hd;
    a.so=64+ne+64;a.bytes=a.so+(ne/32)*4+64;
    float *host=malloc((ce>ne?ce:ne)*4);
    int32_t *ids=malloc(7*a.maxsel*4),cnt[7];
    unsigned char *b0=malloc(a.bytes),*b1=malloc(a.bytes),*poison=malloc(a.bytes);
    must(host&&ids&&b0&&b1&&poison,"host allocation");memset(poison,0xa5,a.bytes);
    a.q=ds4_gpu_tensor_alloc(ne*4);a.gate=ds4_gpu_tensor_alloc(ne*4);
    a.out=ds4_gpu_tensor_alloc(ne*4);a.k=ds4_gpu_tensor_alloc(ce*4);
    a.v=ds4_gpu_tensor_alloc(ce*4);a.sel=ds4_gpu_tensor_alloc(7*a.maxsel*4);
    a.counts=ds4_gpu_tensor_alloc(7*4);a.pos=ds4_gpu_tensor_alloc(4);
    a.packed[0]=ds4_gpu_tensor_alloc(a.bytes);a.packed[1]=ds4_gpu_tensor_alloc(a.bytes);
    must(a.q&&a.gate&&a.out&&a.k&&a.v&&a.sel&&a.counts&&a.pos&&a.packed[0]&&a.packed[1],"device allocation");
    for(uint64_t i=0;i<ce;i++)host[i]=randf()*.25f;
    must(ds4_gpu_tensor_write(a.k,0,host,ce*4),"K write");
    const float magnitudes[]={1.0f,1.0e-30f,1.0e-37f,1.0e20f};
    const uint32_t widths[]={1,2,3,7},positions[]={0,31,255,1023,2047,4088};
    for(unsigned m=0;m<4;m++) {
        for(uint64_t i=0;i<ce;i++)host[i]=randf()*magnitudes[m];
        must(ds4_gpu_tensor_write(a.v,0,host,ce*4),"V write");
        for(unsigned wi=0;wi<4;wi++)for(unsigned pi=0;pi<6;pi++)for(int sparse=0;sparse<2;sparse++) {
            a.rows=widths[wi];a.p0=positions[pi];a.sparse=sparse;a.dpos=(wi+pi+sparse+m)&1;
            for(uint64_t i=0;i<ne;i++)host[i]=randf();
            must(ds4_gpu_tensor_write(a.q,0,host,ne*4),"Q write");
            for(uint64_t i=0;i<ne;i++)host[i]=randf()*16.0f;
            must(ds4_gpu_tensor_write(a.gate,0,host,ne*4),"gate write");
            for(uint32_t r=0;r<7;r++) {
                const uint32_t counts[]={0,1,31,255,257,64,129};cnt[r]=counts[(r+pi)%7];
                for(uint32_t j=0;j<a.maxsel;j++)
                    ids[r*a.maxsel+j]=(j%11==0)?-1:((j%13==0)?(int32_t)a.cap:(int32_t)((j*37u)%(a.p0+r+1u)));
            }
            must(ds4_gpu_tensor_write(a.sel,0,ids,7*a.maxsel*4),"selected write");
            must(ds4_gpu_tensor_write(a.counts,0,cnt,7*4),"count write");
            must(ds4_gpu_tensor_write(a.pos,0,&a.p0,4),"position write");
            for(unsigned f=0;f<2;f++) {
                must(ds4_gpu_tensor_write(a.packed[f],0,poison,a.bytes),"poison");run(&a,f);
            }
            compare(&a,b0,b1);
        }
    }
    if(nh==24&&hd==256) {
        for(uint64_t i=0;i<ce;i++)host[i]=randf();
        must(ds4_gpu_tensor_write(a.v,0,host,ce*4),"timing V");
        for(unsigned rows=1;rows<=2;rows++)for(unsigned pos=1023;pos<=2047;pos+=1024) {
            a.rows=rows;a.p0=pos;a.dpos=1;a.sparse=0;
            must(ds4_gpu_tensor_write(a.pos,0,&a.p0,4),"timing pos");
            ds4_gpu_decode_graphs_invalidate();
            ds4_decode_graph_key keys[2]={{.il=0,.island=0,.variant=0},{.il=0,.island=1,.variant=1}};
            for(unsigned f=0;f<2;f++){must(ds4_gpu_decode_graph_begin(&keys[f])==-1,"initial graph warmup");run(&a,f);graph_run(&a,f,&keys[f]);}
            compare(&a,b0,b1);
            /* Replay with fresh Q/gate values and device positions. The
             * reference is encoded eagerly for each new input. */
            for(unsigned replay=0;replay<4;replay++) {
                a.p0=pos+replay+1;
                must(ds4_gpu_tensor_write(a.pos,0,&a.p0,4),"replay position");
                for(uint64_t i=0;i<ne;i++)host[i]=randf();
                must(ds4_gpu_tensor_write(a.q,0,host,ne*4),"replay Q");
                for(uint64_t i=0;i<ne;i++)host[i]=randf()*16.0f;
                must(ds4_gpu_tensor_write(a.gate,0,host,ne*4),"replay gate");
                for(unsigned f=0;f<2;f++)
                    must(ds4_gpu_tensor_write(a.packed[f],0,poison,a.bytes),"replay poison");
                run(&a,0);graph_run(&a,1,&keys[1]);compare(&a,b0,b1);
            }
        }
        ds4_gpu_decode_graphs_invalidate();
    }
    ds4_gpu_tensor_free(a.q);ds4_gpu_tensor_free(a.gate);ds4_gpu_tensor_free(a.out);
    ds4_gpu_tensor_free(a.k);ds4_gpu_tensor_free(a.v);ds4_gpu_tensor_free(a.sel);
    ds4_gpu_tensor_free(a.counts);ds4_gpu_tensor_free(a.pos);
    ds4_gpu_tensor_free(a.packed[0]);ds4_gpu_tensor_free(a.packed[1]);
    free(host);free(ids);free(b0);free(b1);free(poison);
}
int main(void) {
    setenv("DS4_CUDA_DECODE_GRAPHS","1",1);
    must(ds4_gpu_init(),"init");
    shape(24,2,256);shape(6,1,64);shape(2,1,512);
    ds4_gpu_cleanup();printf("QSA gated Q8: %u full quant/scale/canary comparisons pass\n",checks);return 0;
}
