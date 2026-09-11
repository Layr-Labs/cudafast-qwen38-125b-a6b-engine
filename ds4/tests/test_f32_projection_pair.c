/* CUDA exact GDN projection-pair regression, synthetic weights only. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "ds4_gpu.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <math.h>
static void must(int a,const char *s){if(!a){fprintf(stderr,"vector pair: %s\n",s);exit(1);}}
static uint32_t rng=0x98bfac32u;
static float val(void){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;return ((int)(rng%20001)-10000)*.0001f;}
enum {IN=2560,OUT=48,STEPS=32};
static const uint64_t stride=IN*OUT*4;
static unsigned char *maps[2];
static uint64_t mapbytes;
static ds4_gpu_tensor *x,*y[2][2];
static void run(unsigned f,unsigned rows,unsigned step,unsigned unaligned){
    uint64_t o0=64+step*stride,o1=128+step*stride+unaligned;
    if(f)must(ds4_gpu_matmul_f32_pair_decode_rows_exact_tensor(y[f][0],y[f][1],maps[0],mapbytes,o0,maps[1],mapbytes,o1,IN,OUT,x,rows),"pair");
    else {
        must(ds4_gpu_matmul_f32_decode_rows_exact_tensor(y[f][0],maps[0],mapbytes,o0,IN,OUT,x,rows),"first ordinary");
        must(ds4_gpu_matmul_f32_decode_rows_exact_tensor(y[f][1],maps[1],mapbytes,o1,IN,OUT,x,rows),"second ordinary");
    }
}
static void graph_run(unsigned f,unsigned rows,unsigned many,ds4_decode_graph_key *key){
    int s=ds4_gpu_decode_graph_begin(key);must(s>=0,"graph begin");
    if(!s){for(unsigned i=0;i<STEPS;i++)run(f,rows,many?i:0,0);must(ds4_gpu_decode_graph_end(key)==0,"graph end");}
}
int main(void){
    setenv("DS4_CUDA_DECODE_GRAPHS","1",1);
    mapbytes=128+STEPS*stride+16;
    for(unsigned b=0;b<2;b++){
        maps[b]=mmap(NULL,mapbytes,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);must(maps[b]!=MAP_FAILED,"map");
        float *w=(float *)maps[b];for(uint64_t i=0;i<mapbytes/4;i++)w[i]=val();
    }
    must(ds4_gpu_init(),"init");must(ds4_gpu_set_model_map(maps[0],mapbytes),"map0");
    must(ds4_gpu_set_aux_model_map_range(maps[1],mapbytes,0,mapbytes),"map1");
    x=ds4_gpu_tensor_alloc(IN*2*4);must(x!=NULL,"x");
    const unsigned bytes=OUT*3*4;
    for(unsigned f=0;f<2;f++)for(unsigned b=0;b<2;b++){y[f][b]=ds4_gpu_tensor_alloc(bytes);must(y[f][b]!=NULL,"y");}
    must(!ds4_gpu_matmul_f32_pair_decode_rows_exact_tensor(y[0][0],y[0][0],maps[0],mapbytes,64,maps[1],mapbytes,128,IN,OUT,x,1),"output alias refusal");
    float host[IN*2],mag[]={1,1e-30f,1e-37f,1e10f};
    unsigned char poison[OUT*3*4],a[OUT*3*4],b[OUT*3*4];memset(poison,0xa5,sizeof(poison));unsigned checks=0;
    for(unsigned m=0;m<4;m++)for(unsigned rows=1;rows<=2;rows++)for(unsigned align=0;align<2;align++)for(unsigned step=0;step<STEPS;step++){
        for(unsigned i=0;i<IN*2;i++)host[i]=val()*mag[m];must(ds4_gpu_tensor_write(x,0,host,sizeof(host)),"input");
        for(unsigned f=0;f<2;f++){
            for(unsigned z=0;z<2;z++)must(ds4_gpu_tensor_write(y[f][z],0,poison,bytes),"reset");
            run(f,rows,step,align*4);
        }
        for(unsigned z=0;z<2;z++){
            must(ds4_gpu_tensor_read(y[0][z],0,a,bytes),"read reference");must(ds4_gpu_tensor_read(y[1][z],0,b,bytes),"read candidate");
            if(memcmp(a,b,bytes)){fprintf(stderr,"pair mismatch m%u rows%u align%u step%u output%u\n",m,rows,align,step,z);return 1;}
        }checks++;
    }

    /* Replayed graphs must read fresh activations, not the captured values. */
    for(unsigned rows=1;rows<=2;rows++) {
        ds4_gpu_decode_graphs_invalidate();
        ds4_decode_graph_key keys[2]={{.il=1,.island=0,.variant=0},{.il=1,.island=1,.variant=1}};
        for(unsigned f=0;f<2;f++) {
            must(ds4_gpu_decode_graph_begin(&keys[f])==-1,"warm graph");
            run(f,rows,0,0); graph_run(f,rows,1,&keys[f]);
        }
        for(unsigned step=0;step<4;step++) {
            for(unsigned i=0;i<IN*2;i++)host[i]=val()*mag[step];
            must(ds4_gpu_tensor_write(x,0,host,sizeof(host)),"replay input");
            for(unsigned f=0;f<2;f++)for(unsigned z=0;z<2;z++)
                must(ds4_gpu_tensor_write(y[f][z],0,poison,bytes),"graph output reset");
            for(unsigned f=0;f<2;f++)graph_run(f,rows,1,&keys[f]);
            /* Compare the replay with an eager call using this new input. */
            run(0,rows,STEPS-1,0);
            for(unsigned z=0;z<2;z++) {
                must(ds4_gpu_tensor_read(y[0][z],0,a,bytes),"graph reference");
                must(ds4_gpu_tensor_read(y[1][z],0,b,bytes),"graph candidate");
                must(memcmp(a,b,bytes)==0,"changed-input graph mismatch");
            }
            checks++;
        }
    }
    printf("F32 vector pair: %u complete paired-output/canary comparisons pass\n",checks);
    ds4_gpu_decode_graphs_invalidate();ds4_gpu_tensor_free(x);
    for(unsigned f=0;f<2;f++)for(unsigned z=0;z<2;z++)ds4_gpu_tensor_free(y[f][z]);
    ds4_gpu_cleanup();for(unsigned z=0;z<2;z++)munmap(maps[z],mapbytes);return 0;
}
