"""Actual old fast-math HC up object + precise mix/dual versus precise fusion.
Separate CUDA translation units; no device LTO. SM86 PDL macros are no-ops.
"""
from pathlib import Path
import os,re,subprocess,tempfile
repo=Path(__file__).resolve().parents[2]
main=(repo/'ds4/ds4_cuda.cu').read_text();qw=(repo/'ds4/ds4_cuda_qwen4exp.cu').read_text()
def body(src,name,templ=False):
 for m in re.finditer(r'\b'+name+r'\(',src):
  a=src.rfind('\n',0,m.start())+1;b=src.index('{',m.end())
  if ';' in src[m.end():b]:continue
  e=b+1;d=1
  while d:d+=(src[e]=='{')-(src[e]=='}');e+=1
  if templ:a=src.rfind('template',0,a)
  return src[a:e]
 raise AssertionError(name)
headers=r'''
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#define QWEN4EXP_PDL_SYNC() ((void)0)
#define QWEN4EXP_PDL_TRIGGER() ((void)0)
#define QWEN4EXP_HC_THREADS 256u
#define QWEN4EXP_HC_STAGED_STEPS 10u
#define DS4_QWEN4EXP_TY_f32 0
#define DS4_QWEN4EXP_TY_q8_0 8
#include "ds4_qwen4exp_hc_types.h"
#define CK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d\n",cudaGetErrorString(e),__LINE__);exit(1);}}while(0)
'''
old=headers+body(main,'matmul_q8_hc_warp_pair_kernel',True)+r'''
extern "C" void old_up(float*out,const unsigned char*w,const int8_t*q,const float*s,unsigned rows,cudaStream_t st){
 matmul_q8_hc_warp_pair_kernel<2><<<2560,128,0,st>>>(out,w,q,s,10240,rows);
}
'''
helpers=['dev_f16_to_f32','dev_qwen4exp_f32_value','dev_qwen4exp_q8_0_value','dev_qwen4exp_inject_value','warp_sum_all_f32','qwen4exp_block_sum_f32','qwen4exp_sigmoid','qwen4exp_round_bf16','qwen4exp_hc_normed_value','qwen4exp_fmul_ftz','qwen4exp_fma_ftz','qwen4exp_hc_add_ftz']
precise=headers+'\n'.join(body(qw,n) for n in helpers)+'\n'+body(qw,'qwen4exp_hc_inject_value_staged',True)+'\n'+body(qw,'qwen4exp_hc_mix_renorm_kernel')+'\n'+body(qw,'qwen4exp_hc_mix_inject_dual_kernel',True)+'\n'+body(qw,'qwen4exp_hc_up_mix_short_probe_kernel',True)+r'''
extern "C" void old_up(float*,const unsigned char*,const int8_t*,const float*,unsigned,cudaStream_t);
struct B{unsigned char*p;size_t n;std::vector<unsigned char>h;
 B(size_t n):n(n),h(n+32,0xa5){CK(cudaMalloc(&p,h.size()));}~B(){cudaFree(p);}unsigned char*d(){return p+16;}float*f(){return (float*)d();}
 void up(){CK(cudaMemcpy(p,h.data(),h.size(),cudaMemcpyHostToDevice));}
 void poison(){std::fill(h.begin(),h.end(),0xa5);up();}
 void immutable(){std::vector<unsigned char>v(h.size());CK(cudaMemcpy(v.data(),p,v.size(),cudaMemcpyDeviceToHost));if(v!=h){fprintf(stderr,"IMMUTABLE\n");exit(3);}}
 void same(B&o,const char*name){std::vector<unsigned char>a(h.size()),b(h.size());CK(cudaMemcpy(a.data(),p,a.size(),cudaMemcpyDeviceToHost));CK(cudaMemcpy(b.data(),o.p,b.size(),cudaMemcpyDeviceToHost));if(a!=b){for(size_t i=16;i<n+16;i+=4)if(memcmp(a.data()+i,b.data()+i,4)){uint32_t x,y;memcpy(&x,a.data()+i,4);memcpy(&y,b.data()+i,4);fprintf(stderr,"DIFF %s element%zu %08x %08x\n",name,(i-16)/4,x,y);break;}exit(2);}for(size_t i=0;i<16;i++)if(a[i]!=0xa5||a[n+16+i]!=0xa5)exit(4);}
};
static void put(B&b,size_t i,uint32_t v){memcpy(b.h.data()+16+4*i,&v,4);}
static void floats(B&b,std::mt19937&r,float scale){for(size_t i=0;i<b.n/4;i++){float v=(int(r()%2001)-1000)*scale;memcpy(b.h.data()+16+4*i,&v,4);}}
static void qweights(B&b,unsigned offset,unsigned blocks,std::mt19937&r,unsigned mode){for(unsigned i=0;i<blocks;i++){auto p=b.h.data()+16+offset+i*34;uint16_t s=__half_as_ushort(__float2half_rn((int(r()%2001)-1000)*.0001f));if(mode==6)s=(i&1)?0x7c00:0xfc00;if(mode==7)s=0x7e01;memcpy(p,&s,2);for(unsigned j=0;j<32;j+=4){uint32_t v=r();memcpy(p+2+j,&v,4);}}}
int main(){std::mt19937 rng(690073);cudaStream_t st;CK(cudaStreamCreate(&st));unsigned eager=0,graphs=0;
 for(unsigned rows:{1u,2u})for(unsigned offset:{0u,2u})for(unsigned bf:{0u,1u})for(unsigned bias:{0u,1u})for(unsigned ty:{0u,8u,99u})for(unsigned mode=0;mode<10;mode++){
 const unsigned iwrow=ty==8?320*34:10240*4;
 B up(10240*340+offset),q(rows*320),xs(rows*10*4),hyper(rows*10240*4),ns(rows*4*4),nw(10240*4),iw(4*iwrow),wide(rows*10240*4),m0(rows*2560*4),m1(rows*2560*4),j0(rows*4*4),j1(rows*4*4);
 qweights(up,offset,10240*10,rng,mode);
 if(ty==8)qweights(iw,0,4*320,rng,mode==8?7:0);else floats(iw,rng,.0001f);
 auto publish=[&](unsigned phase){
  for(size_t i=0;i<q.n;i++)q.h[16+i]=(unsigned char)rng();
  floats(xs,rng,.0001f*(phase+1));floats(hyper,rng,.001f*(phase+1));floats(ns,rng,.0001f);floats(nw,rng,.0001f);
  if(mode==1)for(size_t i=0;i<xs.n/4;i++)put(xs,i,(i&1)?0x80000001:1);
  if(mode==2)for(size_t i=0;i<hyper.n/4;i++)put(hyper,i,(i&1)?0x80000001:1);
  if(mode==3)for(size_t i=0;i<hyper.n/4;i++)put(hyper,i,(i&1)?0x80000000:0);
  if(mode==4)for(size_t i=0;i<hyper.n/4;i++)put(hyper,i,(i&1)?0xff800000:0x7f800000);
  if(mode==5)for(size_t i=0;i<nw.n/4;i++)put(nw,i,0x7fc00001+(i&31));
  if(mode==8&&ty!=8)for(size_t i=0;i<iw.n/4;i++)put(iw,i,0x7fc00001+(i&31));
  if(mode==9)for(size_t i=0;i<xs.n/4;i++)put(xs,i,(i&1)?0xde800000:0x5e800000);
  // Refresh a live weight payload at the same address on every replay.
  up.h[16+offset+2]=(unsigned char)(phase*31+rows);
  for(B*b:{&up,&q,&xs,&hyper,&ns,&nw,&iw})b->up();for(B*b:{&wide,&m0,&m1,&j0,&j1})b->poison();
 };
 auto launch=[&](){
  old_up(wide.f(),up.d()+offset,(int8_t*)q.d(),xs.f(),rows,st);
  if(ty==99){
   qwen4exp_hc_mix_renorm_kernel<<<dim3(10,rows),256,0,st>>>(m0.f(),hyper.f(),ns.f(),nw.f(),wide.f(),2560,4,rows,(float)bias,bf);
   qwen4exp_hc_up_mix_short_probe_kernel<-1,false><<<1280,256,0,st>>>(m1.f(),nullptr,up.d()+offset,(int8_t*)q.d(),xs.f(),hyper.f(),ns.f(),nw.f(),nullptr,rows,(float)bias,bf,0,0);
  }else if(ty==0){
   qwen4exp_hc_mix_inject_dual_kernel<0><<<dim3(14,rows),256,0,st>>>(m0.f(),j0.f(),hyper.f(),ns.f(),nw.f(),wide.f(),(char*)iw.d(),2560,4,rows,(float)bias,bf,ty,iwrow);
   qwen4exp_hc_up_mix_short_probe_kernel<0,true><<<1280+4*rows,256,0,st>>>(m1.f(),j1.f(),up.d()+offset,(int8_t*)q.d(),xs.f(),hyper.f(),ns.f(),nw.f(),(char*)iw.d(),rows,(float)bias,bf,ty,iwrow);
  }else{
   qwen4exp_hc_mix_inject_dual_kernel<8><<<dim3(14,rows),256,0,st>>>(m0.f(),j0.f(),hyper.f(),ns.f(),nw.f(),wide.f(),(char*)iw.d(),2560,4,rows,(float)bias,bf,ty,iwrow);
   qwen4exp_hc_up_mix_short_probe_kernel<8,true><<<1280+4*rows,256,0,st>>>(m1.f(),j1.f(),up.d()+offset,(int8_t*)q.d(),xs.f(),hyper.f(),ns.f(),nw.f(),(char*)iw.d(),rows,(float)bias,bf,ty,iwrow);
  }CK(cudaGetLastError());
 };
 auto check=[&](){CK(cudaStreamSynchronize(st));m0.same(m1,"mixed");j0.same(j1,"inject");for(B*b:{&up,&q,&xs,&hyper,&ns,&nw,&iw})b->immutable();};
 publish(0);launch();check();eager++;
 cudaGraph_t gr;cudaGraphExec_t ex;CK(cudaStreamBeginCapture(st,cudaStreamCaptureModeGlobal));launch();CK(cudaStreamEndCapture(st,&gr));CK(cudaGraphInstantiate(&ex,gr,nullptr,nullptr,0));
 for(unsigned phase=1;phase<=2;phase++){publish(phase);CK(cudaGraphLaunch(ex,st));check();graphs++;}
 CK(cudaGraphExecDestroy(ex));CK(cudaGraphDestroy(gr));
 }
 CK(cudaStreamDestroy(st));printf("PASS HC up/mix separate-TU oracle: %u eager + %u changed-input graph replays; mixed/inject bits, guards, immutable sources; SM86 PDL disabled\n",eager,graphs);
}
'''
with tempfile.TemporaryDirectory(prefix='hc-up-mix-gpu-') as t:
 p=Path(t);(p/'old.cu').write_text(old);(p/'precise.cu').write_text(precise)
 nvcc=os.environ.get('NVCC','nvcc');common=[nvcc,'-O3','-arch='+os.environ.get('CUDA_TEST_ARCH','sm_86'),'-I'+str(repo/'ds4')]
 subprocess.run(common+['--use_fast_math','-c',str(p/'old.cu'),'-o',str(p/'old.o')],check=True)
 subprocess.run(common+['-ftz=false','-prec-div=true','-prec-sqrt=true','-c',str(p/'precise.cu'),'-o',str(p/'precise.o')],check=True)
 subprocess.run([nvcc,str(p/'old.o'),str(p/'precise.o'),'-o',str(p/'test')],check=True)
 subprocess.run([str(p/'test')],check=True)
