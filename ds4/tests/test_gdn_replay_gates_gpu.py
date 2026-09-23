"""Extracted precise convolution -> replay -> float/Q8 norm CUDA parity.
Synthetic operators and changed-input graphs; not model or GB10 performance.
Run: NVCC=/path/to/nvcc python3 ds4/tests/test_gdn_replay_gates_gpu.py
"""
from pathlib import Path
import os, re, subprocess, tempfile
s=(Path(__file__).resolve().parents[1]/'ds4_cuda_qwen4exp.cu').read_text()
def body(name):
 for m in re.finditer(r'\b'+name+r'\(',s):
  a=s.rfind('\n',0,m.start())+1;b=s.index('{',m.end())
  if ';' in s[m.end():b]:continue
  e=b+1;d=1
  while d:d+=(s[e]=='{')-(s[e]=='}');e+=1
  return s[a:e]
 raise AssertionError(name)
names=['warp_sum_f32','warp_sum_all_f32','dot4_f32','qwen4exp_gdn_silu','qwen4exp_gdn_sigmoid','qwen4exp_gdn_softplus','qwen4exp_q8_ftz','qwen4exp_q8_rcp_approx','qwen4exp_gdn_conv_kernel','qwen4exp_gdn_conv_replay_gates_kernel','qwen4exp_gdn_replay_kernel','qwen4exp_gdn_replay_gates_kernel','qwen4exp_gdn_output_kernel','qwen4exp_gdn_output_quant_kernel']
source=r'''
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#define QWEN4EXP_GDN_DIM 128u
#define QWEN4EXP_GDN_HISTORY 3u
#define DS4_QWEN4EXP_GDN_REPLAY_ROWS 2u
#include <cstddef>
#define QWEN4EXP_Q8_RCP127 0x1.020408p-7f
#define QWEN4EXP_PDL_SYNC() ((void)0)
#define QWEN4EXP_PDL_TRIGGER() ((void)0)
#define CK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d\n",cudaGetErrorString(e),__LINE__);exit(1);}}while(0)
'''+ '\n'.join(body(n) for n in names)+r'''
struct B{
 float*p;size_t n;std::vector<float>h;
 B(size_t count):n(count),h(count+8){CK(cudaMalloc(&p,h.size()*4));}
 ~B(){cudaFree(p);} float*d(){return p+4;}
 void up(){CK(cudaMemcpy(p,h.data(),h.size()*4,cudaMemcpyHostToDevice));}
 void seed(std::mt19937&r,float scale){for(auto&v:h)v=scale == 0.f ? 0.f : (int(r()%2001)-1000)*scale;for(unsigned i=0;i<4;i++)h[i]=h[n+4+i]=12345.25f;up();}
 std::vector<uint32_t>read(){std::vector<uint32_t>x(h.size());CK(cudaMemcpy(x.data(),p,x.size()*4,cudaMemcpyDeviceToHost));return x;}
 void same(B&b,const char*l){if(read()!=b.read()){fprintf(stderr,"DIFF %s\n",l);exit(2);}}
 void intact(){auto x=read();if(memcmp(x.data(),h.data(),h.size()*4))exit(3);}
 void guard(){auto x=read();uint32_t v;float f=12345.25f;memcpy(&v,&f,4);for(unsigned i=0;i<4;i++)if(x[i]!=v||x[n+4+i]!=v)exit(4);}
};
int main(){std::mt19937 r(650031);unsigned eager=0,replays=0;cudaStream_t st;CK(cudaStreamCreate(&st));
 const unsigned nk=16,nv=48,kd=nk*128,vd=nv*128,cd=2*kd+vd,ts=(kd+vd+2*nv+3)&~3u;const size_t cells=(size_t)nv*128*128;
 for(unsigned defer:{0u,6u})for(unsigned layout=0;layout<2;layout++)for(unsigned mode=0;mode<9;mode++)for(unsigned quant=0;quant<2;quant++){
 const unsigned tr=defer?2*defer:2;
 B input(2*cd),hist(3*cd),snap(6*3*cd),base(cells),tapeseed(tr*ts),alpha(2*nv),beta(2*nv),coeff(nv),bias(nv),cw(cd*4),gate(2*vd),norm(128);
 B q0(2*cd),q1(2*cd),h0(3*cd),h1(3*cd),ss0(6*3*cd),ss1(6*3*cd),s0(cells),s1(cells),c0(cells),c1(cells),t0(tr*ts),t1(tr*ts),o0(2*vd),o1(2*vd),pairs(4*nv+16),z0(2*vd/4),z1(2*vd/4),sc0(2*vd/32),sc1(2*vd/32);
 uint32_t*control,*adopt;CK(cudaMalloc(&control,4));CK(cudaMalloc(&adopt,4));
 auto publish=[&](unsigned phase){
  for(B*b:{&input,&hist,&snap,&base,&tapeseed,&alpha,&beta,&coeff,&bias,&cw,&gate,&norm})b->seed(r,mode==1?1e-40f:.0001f);
  for(size_t i=0;i<coeff.n;i++)coeff.h[4+i]=mode==8?(i%3==0?NAN:i%3==1?INFINITY:-INFINITY):-.1f;coeff.up();
  for(size_t i=0;i<alpha.n;i++){if(mode>=2&&mode<=5)alpha.h[4+i]=mode==2?-80.f:mode==3?80.f:mode==4?INFINITY:NAN;beta.h[4+i]=mode==2?-80.f:mode==3?80.f:mode==6?NAN:mode==7?(i%2?INFINITY:-INFINITY):beta.h[4+i];}alpha.up();beta.up();
  for(B*b:{&q0,&q1,&h0,&h1,&ss0,&ss1,&s0,&s1,&c0,&c1,&t0,&t1,&o0,&o1,&pairs,&z0,&z1,&sc0,&sc1})b->seed(r,0.f);
  // Deferred: prefix | parity << 8 over prefixes 0..defer (flush at >= defer-1).
  uint32_t prefix=defer?(phase%(defer+1))|(((phase/2)&1u)<<8):phase%3,a=(phase/3)%2;CK(cudaMemcpy(control,&prefix,4,cudaMemcpyHostToDevice));CK(cudaMemcpy(adopt,&a,4,cudaMemcpyHostToDevice));
 };
 auto copy=[&](B&dst,B&src){CK(cudaMemcpyAsync(dst.d(),src.d(),src.n*4,cudaMemcpyDeviceToDevice,st));};
 auto launch=[&](){
  for(B*b:{&q0,&q1})copy(*b,input);for(B*b:{&h0,&h1})copy(*b,hist);for(B*b:{&ss0,&ss1})copy(*b,snap);for(B*b:{&s0,&s1,&c0,&c1})copy(*b,base);for(B*b:{&t0,&t1})copy(*b,tapeseed);
  // Real runtime validation queries also execute while the stream is captured.
  int dev;cudaPointerAttributes at={};CK(cudaGetDevice(&dev));CK(cudaPointerGetAttributes(&at,pairs.d()));if(at.device!=dev||at.type!=cudaMemoryTypeDevice)exit(5);
  qwen4exp_gdn_conv_kernel<<<dim3(2*nk+nv,1),128,0,st>>>(q0.d(),h0.d(),cw.d(),ss0.d(),nk,nv,1,2,1,1e-6f,adopt);
  qwen4exp_gdn_conv_replay_gates_kernel<<<dim3(2*nk+nv,1),128,0,st>>>(q1.d(),h1.d(),cw.d(),ss1.d(),nk,nv,1,2,1,1e-6f,adopt,(float2*)pairs.d(),alpha.d(),beta.d(),coeff.d(),bias.d());
  qwen4exp_gdn_replay_kernel<<<dim3(nv,32),128,0,st>>>(o0.d(),s0.d(),c0.d(),t0.d(),q0.d(),alpha.d(),beta.d(),coeff.d(),bias.d(),nk,nv,2,layout,control,0,defer);
  qwen4exp_gdn_replay_gates_kernel<<<dim3(nv,32),128,0,st>>>(o1.d(),s1.d(),c1.d(),t1.d(),q1.d(),alpha.d(),beta.d(),(float2*)pairs.d(),nk,nv,2,layout,control,0,defer);
  if(quant){
   qwen4exp_gdn_output_quant_kernel<<<dim3(2,nv),128,0,st>>>((int8_t*)z0.d(),sc0.d(),o0.d(),gate.d(),norm.d(),nv,2,1e-6f);
   qwen4exp_gdn_output_quant_kernel<<<dim3(2,nv),128,0,st>>>((int8_t*)z1.d(),sc1.d(),o1.d(),gate.d(),norm.d(),nv,2,1e-6f);
  }else{
   qwen4exp_gdn_output_kernel<<<dim3(2,nv,1),128,0,st>>>(o0.d(),gate.d(),norm.d(),nv,1,2,1e-6f);
   qwen4exp_gdn_output_kernel<<<dim3(2,nv,1),128,0,st>>>(o1.d(),gate.d(),norm.d(),nv,1,2,1e-6f);
  }CK(cudaGetLastError());
 };
 auto check=[&](){CK(cudaStreamSynchronize(st));
 q0.same(q1,"convolution");h0.same(h1,"history");ss0.same(ss1,"snapshots");s0.same(s1,"state");c0.same(c1,"checkpoint");t0.same(t1,"tape");o0.same(o1,"output");z0.same(z1,"q8 bytes");sc0.same(sc1,"q8 scales");
 for(B*b:{&q0,&q1,&h0,&h1,&ss0,&ss1,&s0,&s1,&c0,&c1,&t0,&t1,&o0,&o1,&pairs,&z0,&z1,&sc0,&sc1})b->guard();
 for(B*b:{&input,&hist,&snap,&base,&tapeseed,&alpha,&beta,&coeff,&bias,&cw,&gate,&norm})b->intact();
 auto gp=pairs.read();if(!memcmp(gp.data()+4,pairs.h.data()+4,4*nv*4))exit(6);for(unsigned i=4*nv;i<pairs.n;i++)if(gp[4+i]!=0)exit(7);
 };
 publish(0);launch();check();eager++;
 cudaGraph_t gr;cudaGraphExec_t ex;CK(cudaStreamBeginCapture(st,cudaStreamCaptureModeGlobal));launch();CK(cudaStreamEndCapture(st,&gr));CK(cudaGraphInstantiate(&ex,gr,nullptr,nullptr,0));
 for(unsigned phase=1;phase<=(defer?2*defer+2:6);phase++){publish(phase);CK(cudaGraphLaunch(ex,st));check();replays++;}
 CK(cudaGraphExecDestroy(ex));CK(cudaGraphDestroy(gr));CK(cudaFree(control));CK(cudaFree(adopt));
 }
 CK(cudaStreamDestroy(st));printf("PASS gate publication pipeline: %u eager + %u changed-input/prefix/adopt graph replays; float/Q8, full convolution/history/snapshot/state/checkpoint/tape/output, immutable inputs and canaries\n",eager,replays);
}
'''
with tempfile.TemporaryDirectory(prefix='gdn-replay-gates-') as t:
 p=Path(t);(p/'test.cu').write_text(source)
 subprocess.run([os.environ.get('NVCC','nvcc'),'-O3','-ftz=false','-prec-div=true','-prec-sqrt=true','-arch='+os.environ.get('CUDA_TEST_ARCH','sm_86'),str(p/'test.cu'),'-o',str(p/'test')],check=True)
 subprocess.run([str(p/'test')],check=True)
