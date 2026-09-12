/* Original BF16 arithmetic on one full grid: complete outputs and replay. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "ds4_gpu.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
enum{SETS=4,MODES=2,CAP=1032};
static uint32_t rng=0x84fa7613u;
static uint32_t word(void){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;return rng;}
static void must(int ok,const char *s){if(!ok){fprintf(stderr,"BF16 launch grid: %s\n",s);exit(1);}}
static void *alloc(size_t n){void*p=malloc(n);must(p!=NULL,"allocation");return p;}
static uint32_t in_dim,out_dim;
static uint64_t mb,offsets[SETS];
static size_t xb,yb;
static unsigned char *map,*xh[SETS],*ref[SETS],*poison;
static ds4_gpu_tensor *xt[SETS],*yt[MODES][SETS];
static void inputs(float scale){
 for(unsigned i=0;i<SETS;i++){
  float *x=(float *)xh[i];for(size_t j=0;j<xb/4;j++)x[j]=((int)(word()%2001)-1000)*scale*.001f;
  must(ds4_gpu_tensor_write(xt[i],0,xh[i],xb),"write input");
 }
}
static void reset(unsigned m){for(unsigned i=0;i<SETS;i++)must(ds4_gpu_tensor_write(yt[m][i],0,poison,yb),"reset complete output");}
static void operate(unsigned m,unsigned rows){
 for(unsigned i=0;i<SETS;i++){
  if(m)must(ds4_gpu_qwen4exp_bf16_prefill_exact_tensor(yt[m][i],map,mb,offsets[i],in_dim,out_dim,xt[i],rows),"single row grid");
  else for(unsigned at=0;at<rows;at+=8u){
   unsigned take=rows-at<8u?rows-at:8u;
   ds4_gpu_tensor *xv=ds4_gpu_tensor_view(xt[i],(uint64_t)at*in_dim*4u,(uint64_t)take*in_dim*4u);
   ds4_gpu_tensor *ov=ds4_gpu_tensor_view(yt[m][i],(uint64_t)at*out_dim*4u,(uint64_t)take*out_dim*4u);
   must(xv&&ov,"reference views");
   must(ds4_gpu_glm53_matmul_bf16(ov,map,mb,offsets[i],in_dim,out_dim,xv,take),"original chunk");
   ds4_gpu_tensor_free(xv);ds4_gpu_tensor_free(ov);
  }
 }
}
static unsigned compare(unsigned m){
 unsigned char *got=alloc(xb>yb?xb:yb);
 for(unsigned i=0;i<SETS;i++){
  must(ds4_gpu_tensor_read(yt[m][i],0,m?got:ref[i],yb),"read full result");
  if(m)must(memcmp(got,ref[i],yb)==0,"full output and canary equality");
  must(ds4_gpu_tensor_read(xt[i],0,got,xb),"read input");must(memcmp(got,xh[i],xb)==0,"input immutable");
 }
 free(got);return m?SETS:0;
}
static int graph(unsigned m,unsigned rows,ds4_decode_graph_key *key){
 int state=ds4_gpu_decode_graph_begin(key);must(state>=0,"begin graph");
 if(!state){operate(m,rows);must(ds4_gpu_decode_graph_end(key)==0,"capture graph");}return state;
}
static void check_shape(unsigned in,unsigned out,unsigned offset){
 in_dim=in;out_dim=out;xb=(size_t)CAP*in*4u+64;yb=(size_t)CAP*out*4u+64;
 const uint64_t stride=((uint64_t)in*out*2u+63u)&~63ull;mb=offset+SETS*stride+64;
 map=mmap(NULL,mb,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);must(map!=MAP_FAILED,"mapping");
 for(unsigned i=0;i<SETS;i++){
  offsets[i]=offset+i*stride;uint16_t*w=(uint16_t *)(map+offsets[i]);
  for(uint64_t j=0;j<(uint64_t)in*out;j++){float f=((int)(word()%2001)-1000)*.0002f;uint32_t bits;memcpy(&bits,&f,4);w[j]=(uint16_t)(bits>>16);}
 }
 must(ds4_gpu_init(),"GPU init");must(ds4_gpu_set_model_map(map,mb),"model map");poison=alloc(yb);memset(poison,0xa5,yb);
 for(unsigned i=0;i<SETS;i++){
  xt[i]=ds4_gpu_tensor_alloc(xb);must(xt[i]!=NULL,"input tensor");xh[i]=alloc(xb);ref[i]=alloc(yb);
  for(unsigned m=0;m<MODES;m++){yt[m][i]=ds4_gpu_tensor_alloc(yb);must(yt[m][i]!=NULL,"output tensor");}
 }
 inputs(.2f);reset(0);operate(0,1);compare(0);
 must(!ds4_gpu_qwen4exp_bf16_prefill_exact_tensor(yt[1][0],map,mb,offsets[0],in,out,xt[0],0),"zero rows rejected");
 must(!ds4_gpu_qwen4exp_bf16_prefill_exact_tensor(yt[1][0],map,mb,offsets[0],in,out,xt[0],65536),"grid overflow rejected");
 must(!ds4_gpu_qwen4exp_bf16_prefill_exact_tensor(yt[1][0],map,mb,offsets[0],0,out,xt[0],1),"zero input dimension rejected");
 must(!ds4_gpu_qwen4exp_bf16_prefill_exact_tensor(yt[1][0],map,mb,offsets[0],in,65537,xt[0],1),"large output dimension rejected");
 must(!ds4_gpu_qwen4exp_bf16_prefill_exact_tensor(yt[1][0],map,mb,mb,in,out,xt[0],1),"short weights rejected");
 ds4_gpu_tensor *tiny=ds4_gpu_tensor_view(xt[0],0,4);must(tiny!=NULL,"tiny view");
 must(!ds4_gpu_qwen4exp_bf16_prefill_exact_tensor(yt[1][0],map,mb,offsets[0],in,out,tiny,1),"short input rejected");
 must(!ds4_gpu_qwen4exp_bf16_prefill_exact_tensor(tiny,map,mb,offsets[0],in,out,xt[1],1),"short output rejected");ds4_gpu_tensor_free(tiny);
 must(!ds4_gpu_qwen4exp_bf16_prefill_exact_tensor(xt[0],map,mb,offsets[0],in_dim,out_dim,xt[0],1),"same input/output rejected");
 ds4_gpu_tensor *over=ds4_gpu_tensor_view(xt[0],4,xb-4);must(over!=NULL,"overlap view");
 must(!ds4_gpu_qwen4exp_bf16_prefill_exact_tensor(over,map,mb,offsets[0],in_dim,out_dim,xt[0],1),"partial input/output rejected");ds4_gpu_tensor_free(over);
 ds4_gpu_tensor *adj=ds4_gpu_tensor_alloc(xb+yb);must(adj!=NULL,"adjacent allocation");
 ds4_gpu_tensor *ax=ds4_gpu_tensor_view(adj,0,xb),*ay=ds4_gpu_tensor_view(adj,xb,yb);must(ax&&ay,"adjacent views");
 must(ds4_gpu_tensor_write(ax,0,xh[0],xb)&&ds4_gpu_tensor_write(ay,0,poison,yb),"adjacent write");
 must(ds4_gpu_qwen4exp_bf16_prefill_exact_tensor(ay,map,mb,offsets[0],in_dim,out_dim,ax,1),"adjacent exact projection");
 unsigned char *got=alloc(xb>yb?xb:yb);must(ds4_gpu_tensor_read(ay,0,got,yb),"adjacent readback");
 must(memcmp(got,ref[0],yb)==0,"adjacent whole output and tail");
 must(ds4_gpu_tensor_read(ax,0,got,xb),"adjacent input read");must(memcmp(got,xh[0],xb)==0,"adjacent input immutable");
 free(got);ds4_gpu_tensor_free(ax);ds4_gpu_tensor_free(ay);ds4_gpu_tensor_free(adj);
 const unsigned widths[]={1,2,7,8,9,16,64,1017,1024};const float scales[]={.2f,1e-20f,8.0f,0.0f};
 unsigned eager=0,captured=0,replayed=0;
 for(unsigned wi=0;wi<sizeof(widths)/sizeof(widths[0]);wi++){
  unsigned rows=widths[wi];ds4_gpu_decode_graphs_invalidate();ds4_decode_graph_key keys[MODES]={{.il=1},{.il=2}};
  for(unsigned trial=0;trial<4;trial++){
   inputs(scales[trial]);for(unsigned m=0;m<MODES;m++){reset(m);operate(m,rows);eager+=compare(m);}
   for(unsigned m=0;m<MODES;m++){
    if(!trial){must(ds4_gpu_decode_graph_begin(&keys[m])==-1,"warm capture");operate(m,rows);}
    reset(m);must(graph(m,rows,&keys[m])==(trial?1:0),"actual capture/replay");if(trial)replayed++;captured+=compare(m);
   }
  }
 }
 printf("BF16_LAUNCH_GRID_CHECK in=%u out=%u offset=%u sets=4 eager_buffers=%u graph_buffers=%u changed_replays=%u PASS\n",in,out,offset,eager,captured,replayed);fflush(stdout);
 ds4_gpu_decode_graphs_invalidate();
 for(unsigned i=0;i<SETS;i++){
  ds4_gpu_tensor_free(xt[i]);free(xh[i]);free(ref[i]);for(unsigned m=0;m<MODES;m++)ds4_gpu_tensor_free(yt[m][i]);
 }
 ds4_gpu_cleanup();munmap(map,mb);free(poison);
}
int main(void){
 setenv("DS4_CUDA_DECODE_GRAPHS","1",1);
 check_shape(97,65,66);check_shape(2560,128,64);check_shape(2560,128,66);check_shape(2560,512,64);
 puts("BF16_LAUNCH_GRID_ALL PASS");return 0;
}
