"""Actual precise shared-down + HC inject versus coalesced fused epilogue.
PDL macros are no-ops on SM86; this does not validate GB10 dependent launches.
Run with NVCC pointing to CUDA 13 and CUDA_TEST_ARCH=sm_86 (default).
"""
from pathlib import Path
import os,re,subprocess,tempfile
repo=Path(__file__).resolve().parents[2]
s=(repo/'ds4/ds4_cuda_qwen4exp.cu').read_text()
def body(name,templ=False):
 m=re.search(r'^.*\b'+name+r'\([^;]*?\)\s*\{',s,re.M);assert m,name
 a=m.start();b=s.index('{',a);e=b+1;d=1
 while d:d+=(s[e]=='{')-(s[e]=='}');e+=1
 if templ:a=s.rfind('template <',0,a)
 return s[a:e]
def structure(name):
 m=re.search(r'typedef struct \{[^}]*\} '+name+r';',s);assert m,name
 return m.group()
source=r'''
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#include "ds4_qwen4exp_moe_types.h"
#define CUDA_QK_K 256
#define QWEN4EXP_PDL_TRIGGER() ((void)0)
#define QWEN4EXP_PDL_SYNC() ((void)0)
#define CK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d\n",cudaGetErrorString(e),__LINE__);exit(1);}}while(0)
enum {
#define DS4_QWEN4EXP_TYPE_ENUM(name,id) DS4_QWEN4EXP_TY_ ## name = id,
DS4_QWEN4EXP_MOE_TYPES(DS4_QWEN4EXP_TYPE_ENUM)
#undef DS4_QWEN4EXP_TYPE_ENUM
};
'''
source+='\n'.join(structure(n) for n in ['cuda_block_q4_K','cuda_block_q5_1','cuda_block_q5_K','cuda_block_q6_K'])
source+='\n'.join(body(n) for n in ['dev_f16_to_f32','dev_q4_K_get_scale_min','warp_sum_f32','qwen4exp_load_i8x4','qwen4exp_word_aligned'])
source+=body('qwen4exp_dp4a',True)
source+='\n'.join(body(n) for n in ['dev_qwen4exp_group_decode','qwen4exp_group_accumulate','qwen4exp_shared_vector_accumulate','qwen4exp_hc_inject_kernel'])
source+=body('qwen4exp_shared_down_q_kernel',True)+body('qwen4exp_shared_down_hc_kernel',True)
source+=r'''
struct B{float*p;size_t n;std::vector<float>h;
 B(size_t n):n(n),h(n+8){CK(cudaMalloc(&p,h.size()*4));}~B(){cudaFree(p);}float*d(){return p+4;}
 void up(){CK(cudaMemcpy(p,h.data(),h.size()*4,cudaMemcpyHostToDevice));}
 void seed(std::mt19937&r,float scale){for(auto&v:h)v=scale==0.f?0.f:(int(r()%2001)-1000)*scale;for(unsigned i=0;i<4;i++)h[i]=h[n+4+i]=12345.25f;up();}
 std::vector<uint32_t>read(){std::vector<uint32_t>x(h.size());CK(cudaMemcpy(x.data(),p,x.size()*4,cudaMemcpyDeviceToHost));return x;}
 void same(B&b,const char*l){if(read()!=b.read()){fprintf(stderr,"DIFF %s\n",l);exit(2);}}
 void intact(){auto x=read();if(memcmp(x.data(),h.data(),h.size()*4))exit(3);}
 void guard(){auto x=read();uint32_t v;float f=12345.25f;memcpy(&v,&f,4);for(unsigned i=0;i<4;i++)if(x[i]!=v||x[n+4+i]!=v)exit(4);}
};

int main(){std::mt19937 r(680071);unsigned eager=0,replays=0;cudaStream_t st;CK(cudaStreamCreate(&st));
 for(unsigned rows:{1u,2u})for(unsigned mode=0;mode<10;mode++){
 B w(2560*680/4),mq(rows*640/4),ms(rows*20),sum(rows*20),gate(rows),inj(rows*4),seed(rows*2560),hs(rows*10240),o0(rows*2560),o1(rows*2560),h0(rows*10240),h1(rows*10240);
 auto publish=[&](unsigned phase){
  for(B*b:{&ms,&gate,&inj,&seed,&hs})b->seed(r,.0001f*(phase+1));
  auto wb=(unsigned char*)(w.h.data()+4);auto qb=(int8_t*)(mq.h.data()+4);
  for(unsigned i=0;i<2560*20;i++){uint16_t v=__half_as_ushort(__float2half_rn((int(r()%2001)-1000)*.0001f));if(mode==7)v=(i&1)?0x7c00:0xfc00;if(mode==8)v=0x7e00;memcpy(wb+i*34,&v,2);for(unsigned k=0;k<32;k++)wb[i*34+2+k]=(unsigned char)r();}
  for(unsigned i=0;i<rows*20;i++){int32_t z=0;for(unsigned k=0;k<32;k++){int8_t v=(int8_t)r();qb[i*32+k]=v;z+=v;}memcpy(sum.h.data()+4+i,&z,4);}
  if(mode==1)for(unsigned i=0;i<ms.n;i++)ms.h[4+i]=i&1?1e-40f:-1e-40f;
  if(mode==2)for(unsigned i=0;i<hs.n;i++)hs.h[4+i]=i&1?-0.f:0.f;
  if(mode==3)for(unsigned i=0;i<seed.n;i++)seed.h[4+i]=i&1?INFINITY:-INFINITY;
  if(mode==4)for(unsigned i=0;i<inj.n;i++)inj.h[4+i]=i&1?INFINITY:-INFINITY;
  if(mode==5)for(unsigned i=0;i<gate.n;i++)gate.h[4+i]=NAN;
  if(mode==6)for(unsigned i=0;i<hs.n;i++)hs.h[4+i]=NAN;
  if(mode==9)for(unsigned i=0;i<seed.n;i++)seed.h[4+i]=i&1?1e30f:-1e30f;
  for(B*b:{&w,&mq,&sum}){for(unsigned j=0;j<4;j++)b->h[j]=b->h[b->n+4+j]=12345.25f;}
  for(B*b:{&w,&mq,&ms,&sum,&gate,&inj,&seed,&hs})b->up();
  for(B*b:{&o0,&o1,&h0,&h1})b->seed(r,0.f);
 };
 auto launch=[&](){
  for(B*b:{&o0,&o1})CK(cudaMemcpyAsync(b->d(),seed.d(),seed.n*4,cudaMemcpyDeviceToDevice,st));
  for(B*b:{&h0,&h1})CK(cudaMemcpyAsync(b->d(),hs.d(),hs.n*4,cudaMemcpyDeviceToDevice,st));
  if(rows==1){
   qwen4exp_shared_down_q_kernel<1,DS4_QWEN4EXP_TY_q8_0,false,true><<<320,256,5440,st>>>(o0.d(),(char*)w.d(),(int8_t*)mq.d(),ms.d(),(int32_t*)sum.d(),gate.d(),680,8,20,2560,rows);
   qwen4exp_hc_inject_kernel<<<dim3(10,4,rows),256,0,st>>>(h0.d(),h0.d(),o0.d(),inj.d(),2560,4,rows);
   qwen4exp_shared_down_hc_kernel<1,false><<<320,256,5440,st>>>(o1.d(),h1.d(),inj.d(),(char*)w.d(),(int8_t*)mq.d(),ms.d(),(int32_t*)sum.d(),gate.d(),680,8,20,2560,rows);
  }else{
   qwen4exp_shared_down_q_kernel<2,DS4_QWEN4EXP_TY_q8_0,true,true><<<320,256,5440,st>>>(o0.d(),(char*)w.d(),(int8_t*)mq.d(),ms.d(),(int32_t*)sum.d(),gate.d(),680,8,20,2560,rows);
   qwen4exp_hc_inject_kernel<<<dim3(10,4,rows),256,0,st>>>(h0.d(),h0.d(),o0.d(),inj.d(),2560,4,rows);
   qwen4exp_shared_down_hc_kernel<2,true><<<320,256,5440,st>>>(o1.d(),h1.d(),inj.d(),(char*)w.d(),(int8_t*)mq.d(),ms.d(),(int32_t*)sum.d(),gate.d(),680,8,20,2560,rows);
  }CK(cudaGetLastError());
 };
 auto check=[&](){CK(cudaStreamSynchronize(st));o0.same(o1,"block output");h0.same(h1,"hyper output");for(B*b:{&o0,&o1,&h0,&h1})b->guard();for(B*b:{&w,&mq,&ms,&sum,&gate,&inj,&seed,&hs})b->intact();};
 publish(0);launch();check();eager++;
 cudaGraph_t g;cudaGraphExec_t ex;CK(cudaStreamBeginCapture(st,cudaStreamCaptureModeGlobal));launch();CK(cudaStreamEndCapture(st,&g));CK(cudaGraphInstantiate(&ex,g,nullptr,nullptr,0));
 for(unsigned phase=1;phase<=4;phase++){publish(phase);CK(cudaGraphLaunch(ex,st));check();replays++;}
 CK(cudaGraphExecDestroy(ex));CK(cudaGraphDestroy(g));
 }CK(cudaStreamDestroy(st));printf("PASS shared down HC: %u eager + %u fresh-input graph replays; full block/hyper bits, guards and input/weight immutability\n",eager,replays);
}
'''
with tempfile.TemporaryDirectory(prefix='shared-down-hc-') as t:
 p=Path(t);(p/'test.cu').write_text(source)
 subprocess.run([os.environ.get('NVCC','nvcc'),'-O3','-ftz=false','-prec-div=true','-prec-sqrt=true','-arch='+os.environ.get('CUDA_TEST_ARCH','sm_86'),'-I'+str(repo/'ds4'),str(p/'test.cu'),'-o',str(p/'test')],check=True)
 subprocess.run([str(p/'test')],check=True)
