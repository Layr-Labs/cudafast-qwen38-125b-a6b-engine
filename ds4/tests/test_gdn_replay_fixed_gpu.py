"""Actual precise CUDA fixed geometry replay vs unchanged replay and ordinary recurrence.
Synthetic SM86 operator/graph tests; no model, GB10 or performance claim.
"""
from pathlib import Path
import os, subprocess, tempfile
s=(Path(__file__).resolve().parents[1]/'ds4_cuda_qwen4exp.cu').read_text()
def body(name, template=False):
 a=s.index(name+'(');a=s.rfind('\n',0,a)+1
 if template:a=s.rfind('\n',0,a-1)+1
 b=s.index('{',a);e=b+1;d=1
 while d:d+=(s[e]=='{')-(s[e]=='}');e+=1
 return s[a:e]
code='\n'.join(body(n) for n in ['warp_sum_all_f32','dot4_f32','qwen4exp_gdn_sigmoid','qwen4exp_gdn_softplus'])
code+='\n'+body('qwen4exp_gdn_recurrence_kernel',True)+'\n'+body('qwen4exp_gdn_replay_kernel')+'\n'+body('qwen4exp_gdn_replay_fixed_kernel')
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
#define DS4_QWEN4EXP_GDN_REPLAY_ROWS 2u
#define CK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d\n",cudaGetErrorString(e),__LINE__);exit(1);}}while(0)
'''+code+r'''
struct Buffer {
 float *p; size_t n;std::vector<float> host;
 Buffer(size_t count):n(count),host(count+8){CK(cudaMalloc(&p,(n+8)*4));}
 ~Buffer(){cudaFree(p);}
 float* data(){return p+4;}
 void upload(){CK(cudaMemcpy(p,host.data(),host.size()*4,cudaMemcpyHostToDevice));}
 void fill(std::mt19937&r,float scale){for(auto&v:host)v=(int(r()%2001)-1000)*scale;for(unsigned i=0;i<4;i++)host[i]=host[n+4+i]=12345.25f;upload();}
 std::vector<uint32_t> read(){std::vector<uint32_t>x(n+8);CK(cudaMemcpy(x.data(),p,x.size()*4,cudaMemcpyDeviceToHost));return x;}
 void same(Buffer&b,const char*label){if(read()!=b.read()){fprintf(stderr,"DIFF %s\n",label);exit(2);}}
 void intact(){auto x=read();if(memcmp(x.data(),host.data(),host.size()*4)){fprintf(stderr,"input modified\n");exit(3);}}
 void guard(){auto x=read();uint32_t bits;float f=12345.25f;memcpy(&bits,&f,4);for(unsigned i=0;i<4;i++)if(x[i]!=bits||x[n+4+i]!=bits)exit(4);}
};
int main(){std::mt19937 rng(620019);unsigned eager=0,replay_count=0;cudaStream_t st;CK(cudaStreamCreate(&st));
 const unsigned nk=16,nv=48,kd=nk*128,vd=nv*128,cd=2*kd+vd,stride=(kd+vd+2*nv+3)&~3u;const size_t cells=(size_t)nv*128*128;
 for(unsigned layout=1;layout<2;layout++)for(unsigned mode=0;mode<8;mode++){
  Buffer base(cells),tape_seed(2*stride),qkv(2*cd),alpha(2*nv),beta(2*nv),coeff(nv),bias(nv),pq(2*cd),pairs(4*nv);
  Buffer s0(cells),s1(cells),sr(cells),c0(cells),c1(cells),t0(2*stride),t1(2*stride),o0(2*vd),o1(2*vd),orr(2*vd),unused(2*vd),snapshot(cells);
  uint32_t*ctl;CK(cudaMalloc(&ctl,4));unsigned prefix=0;
  auto publish=[&](unsigned phase){
   prefix=phase%3;float scale=mode==1?1e-40f:mode==2?.02f:.0001f;
   for(Buffer*b:{&base,&tape_seed,&qkv,&alpha,&beta,&coeff,&bias,&pq,&pairs})b->fill(rng,scale);
   for(auto&v:coeff.host)v=-.1f;for(auto&v:bias.host)v=.01f;
   // Keep coefficient/input guard values consistent with every upload.
   for(unsigned i=0;i<4;i++){coeff.host[i]=coeff.host[nv+4+i]=12345.25f;bias.host[i]=bias.host[nv+4+i]=12345.25f;}
   for(size_t i=0;i<alpha.n;i++) {alpha.host[i+4]=mode==3?-80.f:mode==4?80.f:mode==5?INFINITY:mode==6?-INFINITY:mode==7?NAN:alpha.host[i+4];beta.host[i+4]=mode==3?-80.f:mode==4?80.f:beta.host[i+4];}
   alpha.upload();beta.upload();coeff.upload();bias.upload();
   // Construct the scalar prefix oracle from the exact saved K/V/gate bits.
   for(unsigned t=0;t<2;t++){
    auto*saved=tape_seed.host.data()+4+t*stride;
    for(unsigned j=0;j<kd;j++)pq.host[4+t*cd+kd+j]=saved[j];
    for(unsigned j=0;j<vd;j++)pq.host[4+t*cd+2*kd+j]=saved[kd+j];
    for(unsigned j=0;j<2*nv;j++)pairs.host[4+t*2*nv+j]=saved[kd+vd+j];
   }pq.upload();pairs.upload();
   for(Buffer*b:{&s0,&s1,&sr,&c0,&c1,&t0,&t1,&o0,&o1,&orr,&unused,&snapshot})b->fill(rng,0.f);
   CK(cudaMemcpy(ctl,&prefix,4,cudaMemcpyHostToDevice));
  };
  auto launch=[&](){
   for(Buffer*b:{&s0,&s1,&sr,&c0,&c1})CK(cudaMemcpyAsync(b->data(),base.data(),cells*4,cudaMemcpyDeviceToDevice,st));
   for(Buffer*b:{&t0,&t1})CK(cudaMemcpyAsync(b->data(),tape_seed.data(),2*stride*4,cudaMemcpyDeviceToDevice,st));
   qwen4exp_gdn_replay_kernel<<<dim3(nv,32),128,0,st>>>(o0.data(),s0.data(),c0.data(),t0.data(),qkv.data(),alpha.data(),beta.data(),coeff.data(),bias.data(),nk,nv,2,layout,ctl,0);
   qwen4exp_gdn_replay_fixed_kernel<<<dim3(nv,32),128,0,st>>>(o1.data(),s1.data(),c1.data(),t1.data(),qkv.data(),alpha.data(),beta.data(),coeff.data(),bias.data(),ctl);
   CK(cudaGetLastError());
  };
  auto oracle=[&](){
   if(prefix)qwen4exp_gdn_recurrence_kernel<true><<<dim3(nv,32),128,0,st>>>(unused.data(),sr.data(),pq.data(),alpha.data(),beta.data(),coeff.data(),bias.data(),(const float2*)pairs.data(),nullptr,nk,nv,1,prefix,layout,0,1,nullptr);
   qwen4exp_gdn_recurrence_kernel<false><<<dim3(nv,32),128,0,st>>>(orr.data(),sr.data(),qkv.data(),alpha.data(),beta.data(),coeff.data(),bias.data(),nullptr,snapshot.data(),nk,nv,1,2,layout,1,1,nullptr);
   CK(cudaGetLastError());CK(cudaStreamSynchronize(st));
  };
  auto compare=[&](){
   s0.same(s1,"state fixed geometry");c0.same(c1,"checkpoint fixed geometry");t0.same(t1,"tape fixed geometry");o0.same(o1,"output fixed geometry");s0.same(sr,"ordinary state");o0.same(orr,"ordinary output");
   if(prefix==2)c0.same(snapshot,"row0 checkpoint");else c0.same(base,"checkpoint untouched");
   for(Buffer*b:{&s0,&s1,&sr,&c0,&c1,&t0,&t1,&o0,&o1,&orr,&snapshot})b->guard();
   for(Buffer*b:{&base,&tape_seed,&qkv,&alpha,&beta,&coeff,&bias,&pq,&pairs})b->intact();
  };
  publish(0);launch();oracle();compare();eager++;
  cudaGraph_t g;cudaGraphExec_t ex;CK(cudaStreamBeginCapture(st,cudaStreamCaptureModeGlobal));launch();CK(cudaStreamEndCapture(st,&g));CK(cudaGraphInstantiate(&ex,g,nullptr,nullptr,0));
  for(unsigned phase=1;phase<=5;phase++){publish(phase);CK(cudaGraphLaunch(ex,st));oracle();compare();replay_count++;}
  CK(cudaGraphExecDestroy(ex));CK(cudaGraphDestroy(g));CK(cudaFree(ctl));
 }
 CK(cudaStreamDestroy(st));printf("PASS replay fixed geometry: %u eager + %u changed-input/control graph replays; old replay and ordinary recurrence; state/checkpoint/tape/output/guards/input bits\n",eager,replay_count);
}
'''
with tempfile.TemporaryDirectory(prefix='gdn-replay-fixed-') as t:
 p=Path(t);(p/'test.cu').write_text(source)
 subprocess.run([os.environ.get('NVCC','nvcc'),'-O3','-ftz=false','-prec-div=true','-prec-sqrt=true','-arch='+os.environ.get('CUDA_TEST_ARCH','sm_86'),str(p/'test.cu'),'-o',str(p/'test')],check=True)
 subprocess.run([str(p/'test')],check=True)
