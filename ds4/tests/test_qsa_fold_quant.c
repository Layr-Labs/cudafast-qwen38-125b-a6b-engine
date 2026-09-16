/* Native CUDA parity of the split fold/gate/Q8 fusion, eager and graph replay.
 * Synthetic activations only. Build off-box; execute only with a CUDA device. */
#include "ds4_gpu.h"
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

enum { H=24, KV=2, D=256, CAP=4096, SEL=1024 };
bool ds4_log_is_tty(FILE *f){(void)f;return false;}
static void need(int ok,const char *msg){if(!ok){fprintf(stderr,"qsa fold: %s\n",msg);exit(1);}}
static uint32_t rng=0x853efbu;
static float random_float(void){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;return ((int)(rng%20001)-10000)*0.0001f;}
static ds4_gpu_tensor *upload(const void *p,uint64_t bytes){ds4_gpu_tensor *t=ds4_gpu_tensor_alloc(bytes);need(t&&ds4_gpu_tensor_write(t,0,p,bytes),"upload");return t;}
static void identical(ds4_gpu_tensor *a,ds4_gpu_tensor *b,uint64_t bytes,const char *why){
 void *x=malloc(bytes),*y=malloc(bytes);need(x&&y,"host compare allocation");
 need(ds4_gpu_tensor_read(a,0,x,bytes)&&ds4_gpu_tensor_read(b,0,y,bytes)&&!memcmp(x,y,bytes),why);free(x);free(y);
}
static void unchanged(ds4_gpu_tensor *t,const void *expected,uint64_t bytes){
 void *got=malloc(bytes);need(got&&ds4_gpu_tensor_read(t,0,got,bytes)&&!memcmp(got,expected,bytes),"input/guard unchanged");free(got);
}
static unsigned cases;
static void run(unsigned rows,unsigned pos,int sparse,int mode){
 const uint64_t n=(uint64_t)rows*H*D,qo=16,so=(n+32+15)&~15ull,pbytes=so+n/32*4+32,obytes=n*4+64;
 float *q=malloc(n*4),*cache=malloc((uint64_t)CAP*KV*D*4),*doubled=malloc(n*8);
 int32_t *selected=malloc((uint64_t)rows*SEL*4),counts[3];
 unsigned char *canary=malloc(pbytes),*out_canary=malloc(obytes);
 need(q&&cache&&doubled&&selected&&canary&&out_canary,"host buffers");
 for(uint64_t i=0;i<n;i++)q[i]=random_float();
 for(uint64_t i=0;i<(uint64_t)CAP*KV*D;i++)cache[i]=random_float();
 for(uint64_t i=0;i<n*2;i++)doubled[i]=mode==2?(i%2?1000:-1000):random_float();
 for(unsigned r=0;r<rows;r++){
  counts[r]=mode==1?0:mode==2?257:SEL;
  for(unsigned i=0;i<SEL;i++)selected[r*SEL+i]=mode==2?(i%11==0?-1:i%13==0?CAP+1:(int32_t)(SEL-i-1)):(int32_t)i;
 }
 memset(canary,0x6b,pbytes);memset(out_canary,0x39,obytes);
 ds4_gpu_tensor *tq=upload(q,n*4),*tk=upload(cache,(uint64_t)CAP*KV*D*4),*tv=upload(cache,(uint64_t)CAP*KV*D*4),
  *td=upload(doubled,n*8),*ts=upload(selected,(uint64_t)rows*SEL*4),*tc=upload(counts,rows*4),*tp=upload(&pos,4),
  *oa=upload(out_canary,obytes),*ob=upload(out_canary,obytes),*qa=upload(canary,pbytes),*qb=upload(canary,pbytes);
 ds4_gpu_tensor *a=ds4_gpu_tensor_view(oa,32,n*4),*b=ds4_gpu_tensor_view(ob,32,n*4);
 uint64_t sb=ds4_gpu_qwen4exp_qsa_split_scratch_bytes(rows,H,D,CAP);
 ds4_gpu_tensor *scratch=ds4_gpu_tensor_alloc(sb);need(a&&b&&scratch,"views/scratch");
 const ds4_gpu_tensor *sel=sparse?ts:NULL,*cnt=sparse?tc:NULL;
#define FUSED() ds4_gpu_qwen4exp_qsa_attention_fold_q8_dpos_tensor(b,tq,tk,tv,sel,cnt,rows,H,KV,D,0,CAP,SEL,0.0625f,tp,scratch,CAP,qb,qo,so,td)
#define ORIGINAL() ds4_gpu_qwen4exp_qsa_attention_dpos_tensor(a,tq,tk,tv,sel,cnt,rows,H,KV,D,0,CAP,SEL,0.0625f,tp,scratch,CAP)
 need(ORIGINAL(),"original attention");need(ds4_gpu_qwen4exp_qsa_output_gate_doubled_q8_tensor(qa,qo,so,a,td,n,D),"original gate");
 need(FUSED()==1,"fused attention/gate");identical(oa,ob,obytes,"float and float guards");identical(qa,qb,pbytes,"Q8 scales and packing guards");
 // Unsupported/fallback must leave the output untouched.
 setenv("DS4_QWEN4EXP_NO_QSA_FOLD_QUANT","1",1);need(FUSED()==0,"negative valve decline");unsetenv("DS4_QWEN4EXP_NO_QSA_FOLD_QUANT");
 identical(qa,qb,pbytes,"decline leaves output");
 if(rows==2&&!sparse&&mode==0&&pos==1023){
  ds4_gpu_decode_graphs_invalidate();ds4_decode_graph_key key={.il=2,.island=2,.variant=12};
  need(ds4_gpu_decode_graph_begin(&key)==-1,"graph warmup");need(FUSED()==1,"warm fused");
  need(ds4_gpu_decode_graph_begin(&key)==0,"graph capture");need(FUSED()==1,"captured fused");need(ds4_gpu_decode_graph_end(&key)==0,"graph end");
  for(unsigned step=0;step<3;step++){
   unsigned next=pos+step;need(ds4_gpu_tensor_write(tp,0,&next,4),"dynamic graph position");
   need(ORIGINAL(),"replay original");need(ds4_gpu_qwen4exp_qsa_output_gate_doubled_q8_tensor(qa,qo,so,a,td,n,D),"replay gate");
   need(ds4_gpu_decode_graph_begin(&key)==1,"graph replay");identical(oa,ob,obytes,"graph float");identical(qa,qb,pbytes,"graph Q8");
  }
  ds4_gpu_decode_graphs_invalidate();need(ds4_gpu_tensor_write(tp,0,&pos,4),"restore position");
 }
 unchanged(tq,q,n*4);unchanged(tk,cache,(uint64_t)CAP*KV*D*4);unchanged(tv,cache,(uint64_t)CAP*KV*D*4);
 unchanged(td,doubled,n*8);unchanged(ts,selected,(uint64_t)rows*SEL*4);unchanged(tc,counts,rows*4);unchanged(tp,&pos,4);
 ds4_gpu_tensor *all[]={a,b,tq,tk,tv,td,ts,tc,tp,oa,ob,qa,qb,scratch};for(unsigned i=0;i<sizeof(all)/sizeof(*all);i++)ds4_gpu_tensor_free(all[i]);
 free(q);free(cache);free(doubled);free(selected);free(canary);free(out_canary);cases++;
#undef FUSED
#undef ORIGINAL
}
int main(void){
 need(ds4_gpu_init(),"GPU init");unsigned positions[]={0,254,255,256,1023,2045};
 for(unsigned rows=1;rows<=3;rows++){
  for(unsigned i=0;i<sizeof positions/sizeof *positions;i++)run(rows,positions[i],0,0);
  for(unsigned mode=0;mode<3;mode++)run(rows,1023,1,mode);
 }
 printf("PASS %u native fold/gate/Q8 cases plus dynamic-position graph replays\n",cases);return 0;
}
