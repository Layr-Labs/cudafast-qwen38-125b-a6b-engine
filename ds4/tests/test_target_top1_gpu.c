/* Prepared CUDA operator test, no model assets. Requires built libds4qwen.
 * cc -O2 -std=c11 -D_GNU_SOURCE -Ids4 ds4/tests/test_target_top1_gpu.c \
 *   -L.build/ds4 -lds4qwen -lm -o /tmp/test-target-top1
 * LD_LIBRARY_PATH=.build/ds4 /tmp/test-target-top1
 * Run with argument "diagnostic" in a fresh process to test cached NO_TOP1. */
#include "ds4_gpu.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
static void need(int x,const char*s){if(!x){fprintf(stderr,"target top1: %s\n",s);exit(1);}}
static uint32_t seed=42123;
static uint32_t rnd(void){seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;return seed;}
static void run(uint32_t width,uint32_t rows,int diagnostic) {
 uint64_t count=(uint64_t)width*rows;float *data=malloc(count*4),*after=malloc(count*4);
 ds4_gpu_tensor *scores=ds4_gpu_tensor_alloc(count*4),*old=ds4_gpu_tensor_alloc(16),*out=ds4_gpu_tensor_alloc(16),*scratch=ds4_gpu_tensor_alloc(528);
 need(data&&after&&scores&&old&&out&&scratch,"allocations");
 uint32_t sentinel[4]={0xdeadbeef,0xdeadbeef,0xdeadbeef,0xdeadbeef},a[4],b[4];unsigned char canary[528],got[528];memset(canary,0xa5,528);
 for(unsigned mode=0;mode<9;mode++) {
  for(uint64_t i=0;i<count;i++){uint32_t bits=rnd();memcpy(&data[i],&bits,4);}
  if(mode==1)for(uint64_t i=0;i<count;i++)data[i]=NAN;
  if(mode==2)for(uint64_t i=0;i<count;i++)data[i]=-INFINITY;
  if(mode==3)for(uint64_t i=0;i<count;i++)data[i]=INFINITY;
  if(mode==4)for(uint64_t i=0;i<count;i++)data[i]=i&1?0.f:-0.f;
  if(mode==5){for(uint64_t i=0;i<count;i++)data[i]=-1;data[0]=NAN;data[width-1]=7;}
  if(mode==6){for(uint64_t i=0;i<count;i++)data[i]=-1;data[1023]=7;data[8192]=7;}
  if(mode==7)for(uint64_t i=0;i<count;i++){uint32_t bits=i&1?1u:0x80000001u;memcpy(&data[i],&bits,4);}
  if(mode==8)for(uint64_t i=0;i<count;i++){uint32_t bits=i&1?0x800000u:0x7fffffu;memcpy(&data[i],&bits,4);}
  need(ds4_gpu_tensor_write(scores,0,data,count*4)&&ds4_gpu_tensor_write(old,0,sentinel,16)&&ds4_gpu_tensor_write(out,0,sentinel,16)&&ds4_gpu_tensor_write(scratch,0,canary,528),"uploads");
  need(ds4_gpu_indexer_topk_tensor(old,scores,width,rows,1)&&ds4_gpu_target_top1_tensor(out,scores,scratch,width,rows),"old/new launch");
  need(ds4_gpu_tensor_read(old,0,a,16)&&ds4_gpu_tensor_read(out,0,b,16)&&!memcmp(a,b,16),"complete winner parity/canary");
  need(ds4_gpu_tensor_read(scores,0,after,count*4)&&!memcmp(data,after,count*4),"scores unchanged");
  need(ds4_gpu_tensor_read(scratch,0,got,528),"scratch witness read");
  int active=width>=65536&&rows<=2&&!diagnostic;
  need(active?memcmp(got,canary,rows*256)!=0:!memcmp(got,canary,528),"actual dispatch witness");
  if(active)need(!memcmp(got+rows*256,canary+rows*256,528-rows*256),"scratch tail unchanged");
 }
 /* Explicit old path and NULL scratch keep the same whole output. */
 setenv("DS4_NO_HIERARCHICAL_TARGET_TOP1","1",1);
 need(ds4_gpu_target_top1_tensor(out,scores,scratch,width,rows),"opt-out");unsetenv("DS4_NO_HIERARCHICAL_TARGET_TOP1");
 need(ds4_gpu_target_top1_tensor(out,scores,NULL,width,rows),"NULL scratch fallback");
 need(ds4_gpu_tensor_read(out,0,b,16)&&!memcmp(a,b,16),"fallback winners");
 if(width>=65536&&rows<=2&&!diagnostic){
  ds4_gpu_tensor*tiny=ds4_gpu_tensor_alloc(4);
  need(!ds4_gpu_target_top1_tensor(out,scores,tiny,width,rows),"scratch bound");ds4_gpu_tensor_free(tiny);
  /* Alias routing preserves old behavior without claiming arbitrary alias-safe input. */
  need(ds4_gpu_target_top1_tensor(out,scores,scores,width,rows),"scratch/scores old route");
  need(ds4_gpu_tensor_read(out,0,b,16)&&!memcmp(a,b,16),"alias route winners");
 }
 if(width==248320&&rows==2&&!diagnostic){
  ds4_gpu_decode_graphs_invalidate();ds4_decode_graph_key key={.il=60,.island=0,.variant=1};
  need(ds4_gpu_decode_graph_begin(&key)==-1,"warm graph key");
  need(ds4_gpu_target_top1_tensor(out,scores,scratch,width,rows),"eager warm");
  need(ds4_gpu_decode_graph_begin(&key)==0,"capture begin");
  need(ds4_gpu_target_top1_tensor(out,scores,scratch,width,rows),"capture kernels");
  need(ds4_gpu_decode_graph_end(&key)==0,"capture end");
  for(unsigned replay=0;replay<3;replay++){
   for(uint64_t i=0;i<count;i++)data[i]=-1;data[100+replay]=10;data[width+200+replay]=20;
   need(ds4_gpu_tensor_write(scores,0,data,count*4)&&ds4_gpu_indexer_topk_tensor(old,scores,width,rows,1),"changed eager oracle");
   need(ds4_gpu_tensor_read(old,0,a,16),"oracle read");
   need(ds4_gpu_decode_graph_begin(&key)==1,"actual graph replay");
   need(ds4_gpu_tensor_read(out,0,b,16)&&!memcmp(a,b,16),"changed graph parity");
  }ds4_gpu_decode_graphs_invalidate();
 }
 ds4_gpu_tensor_free(scores);ds4_gpu_tensor_free(old);ds4_gpu_tensor_free(out);ds4_gpu_tensor_free(scratch);free(data);free(after);
}
int main(int argc,char**argv){(void)argv;int diagnostic=argc>1;if(diagnostic)setenv("DS4_CUDA_NO_TOP1","1",1);setenv("DS4_CUDA_DECODE_GRAPHS","1",1);need(ds4_gpu_init(),"init");run(65535,1,diagnostic);run(65536,1,diagnostic);run(65537,2,diagnostic);run(248320,2,diagnostic);run(65536,3,diagnostic);ds4_gpu_cleanup();puts("target hierarchical top1 GPU checks pass");return 0;}
