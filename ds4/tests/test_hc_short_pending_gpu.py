"""Actual precise CUDA inject+staged norm versus fused short pending norm.
PDL macros are no-ops on this SM86 synthetic arithmetic/graph test.
"""
from pathlib import Path
import os,re,subprocess,tempfile
s=(Path(__file__).resolve().parents[1]/'ds4_cuda_qwen4exp.cu').read_text()
def body(name,templ=False):
 m=re.search(r'^.*\b'+name+r'\([^;]*?\)\s*\{',s,re.M);assert m,name
 a=m.start();b=s.index('{',a);e=b+1;d=1
 while d:d+=(s[e]=='{')-(s[e]=='}');e+=1
 if templ:a=s.rfind('template <',0,a)
 return s[a:e]
names=['warp_sum_all_f32','qwen4exp_round_bf16','qwen4exp_block_sum_f32','qwen4exp_hc_norm_scale','qwen4exp_hc_norm_scale_staged','qwen4exp_hc_normed_value','qwen4exp_q8_ftz','qwen4exp_q8_rcp_approx','qwen4exp_hc_inject_kernel']
source=r'''
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#define QWEN4EXP_HC_THREADS 256u
#define QWEN4EXP_HC_STAGED_STEPS 10u
#define QWEN4EXP_Q8_RCP127 0x1.020408p-7f
#define QWEN4EXP_PDL_TRIGGER() ((void)0)
#define QWEN4EXP_PDL_SYNC() ((void)0)
#define CK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d\n",cudaGetErrorString(e),__LINE__);exit(1);}}while(0)
'''+ '\n'.join(body(n) for n in names)+'\n'+body('qwen4exp_hc_norm_quant_kernel',True)+'\n'+body('qwen4exp_hc_norm_quant_pending_short_kernel')+r'''
struct B{float*p;size_t n;std::vector<float>h;
 B(size_t n):n(n),h(n+8){CK(cudaMalloc(&p,h.size()*4));}~B(){cudaFree(p);}float*d(){return p+4;}
 void up(){CK(cudaMemcpy(p,h.data(),h.size()*4,cudaMemcpyHostToDevice));}
 void seed(std::mt19937&r,float scale){for(auto&v:h)v=scale==0.f?0.f:(int(r()%2001)-1000)*scale;for(unsigned i=0;i<4;i++)h[i]=h[n+4+i]=12345.25f;up();}
 std::vector<uint32_t>read(){std::vector<uint32_t>x(h.size());CK(cudaMemcpy(x.data(),p,x.size()*4,cudaMemcpyDeviceToHost));return x;}
 void same(B&b,const char*l){if(read()!=b.read()){fprintf(stderr,"DIFF %s\n",l);exit(2);}}
 void intact(){auto x=read();if(memcmp(x.data(),h.data(),h.size()*4))exit(3);}
 void guard(){auto x=read();uint32_t v;float f=12345.25f;memcpy(&v,&f,4);for(unsigned i=0;i<4;i++)if(x[i]!=v||x[n+4+i]!=v)exit(4);}
};
int main(){std::mt19937 r(670013);unsigned eager=0,replays=0;cudaStream_t st;CK(cudaStreamCreate(&st));
 for(unsigned rows:{1u,2u})for(unsigned bf:{0u,1u})for(unsigned bias:{0u,1u})for(unsigned mode=0;mode<8;mode++){
 const unsigned n=rows*10240,qn=n/4,sc=n/32,ns=rows*4;
 B seed(n),block(rows*2560),inject(ns),weight(10240),h0(n),h1(n),q0(qn),q1(qn),s0(sc),s1(sc),n0(ns),n1(ns);
 auto publish=[&](unsigned phase){
  for(B*b:{&seed,&block,&inject,&weight})b->seed(r,mode==1?1e-40f:.0001f*(phase+1));
  if(mode==2)for(size_t i=0;i<seed.n;i++)seed.h[4+i]=i&1?-0.f:0.f;
  if(mode==3)for(size_t i=0;i<seed.n;i++)seed.h[4+i]=i&1?INFINITY:-INFINITY;
  if(mode==4)for(size_t i=0;i<block.n;i++)block.h[4+i]=NAN;
  if(mode==5)for(size_t i=0;i<inject.n;i++)inject.h[4+i]=i&1?INFINITY:-INFINITY;
  if(mode==6)for(size_t i=0;i<weight.n;i++)weight.h[4+i]=NAN;
  if(mode==7)for(size_t i=0;i<block.n;i++)block.h[4+i]=i&1?1e18f:-1e18f;
  seed.up();block.up();inject.up();weight.up();for(B*b:{&h0,&h1,&q0,&q1,&s0,&s1,&n0,&n1})b->seed(r,0.f);
 };
 auto launch=[&](){for(B*b:{&h0,&h1})CK(cudaMemcpyAsync(b->d(),seed.d(),n*4,cudaMemcpyDeviceToDevice,st));
  qwen4exp_hc_inject_kernel<<<dim3(10,4,rows),256,0,st>>>(h0.d(),h0.d(),block.d(),inject.d(),2560,4,rows);
  qwen4exp_hc_norm_quant_kernel<1><<<dim3(4,rows,1),256,0,st>>>((int8_t*)q0.d(),s0.d(),n0.d(),h0.d(),weight.d(),10240,2560,rows,1e-6f,(float)bias,bf);
  qwen4exp_hc_norm_quant_pending_short_kernel<<<dim3(4,rows,1),256,0,st>>>((int8_t*)q1.d(),s1.d(),n1.d(),h1.d(),weight.d(),block.d(),inject.d(),10240,2560,rows,1e-6f,(float)bias,bf);CK(cudaGetLastError());
 };
 auto check=[&](){CK(cudaStreamSynchronize(st));h0.same(h1,"materialized hyper");q0.same(q1,"quant bytes");s0.same(s1,"quant scales");n0.same(n1,"norm scales");for(B*b:{&h0,&h1,&q0,&q1,&s0,&s1,&n0,&n1})b->guard();for(B*b:{&seed,&block,&inject,&weight})b->intact();};
 publish(0);launch();check();eager++;cudaGraph_t g;cudaGraphExec_t ex;CK(cudaStreamBeginCapture(st,cudaStreamCaptureModeGlobal));launch();CK(cudaStreamEndCapture(st,&g));CK(cudaGraphInstantiate(&ex,g,nullptr,nullptr,0));
 for(unsigned phase=1;phase<=4;phase++){publish(phase);CK(cudaGraphLaunch(ex,st));check();replays++;}
 CK(cudaGraphExecDestroy(ex));CK(cudaGraphDestroy(g));
 }CK(cudaStreamDestroy(st));printf("PASS HC short pending: %u eager + %u changed-input graph replays; hyper/Q8/scales/nscale bitwise, input immutability and canaries\n",eager,replays);
}
'''
with tempfile.TemporaryDirectory(prefix='hc-short-pending-') as t:
 p=Path(t);(p/'test.cu').write_text(source)
 subprocess.run([os.environ.get('NVCC','nvcc'),'-O3','-ftz=false','-prec-div=true','-prec-sqrt=true','-arch='+os.environ.get('CUDA_TEST_ARCH','sm_86'),str(p/'test.cu'),'-o',str(p/'test')],check=True)
 subprocess.run([str(p/'test')],check=True)
