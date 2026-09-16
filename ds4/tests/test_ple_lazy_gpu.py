"""Actual precise-math CUDA PLE kernels: eager-copy oracle versus lazy adoption.
No model; NVCC selects compiler, CUDA_TEST_ARCH defaults sm_86. This measures
functional equality only, not GB10 performance or the full engine API.
"""
from pathlib import Path
import os,subprocess,tempfile
s=(Path(__file__).resolve().parents[1]/'ds4_cuda_qwen4exp.cu').read_text()
def fn(name,template=False):
 a=s.index(name+'(');a=s.rfind('\n',0,a)+1
 if template:a=s.rfind('template <',0,a)
 b=s.index('{',a);e=b+1;d=1
 while d:d+=(s[e]=='{')-(s[e]=='}');e+=1
 return s[a:e]
source=r'''
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#include <cmath>
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s line%d\n",cudaGetErrorString(e),__LINE__);exit(1);}}while(0)
typedef struct {void *ptr; uint64_t bytes; int device_id;} ds4_gpu_tensor;
static cudaStream_t test_stream;
static int ds4_tensor_device_idx(const ds4_gpu_tensor*t){return t->device_id;}
static int cuda_ok(cudaError_t e,const char*){return e==cudaSuccess;}
static cudaStream_t cuda_decode_stream(){return test_stream;}
/* Only weight registry is substituted: synthetic weights are already onGPU. */
static const float *cuda_resolve_weight_ptr(const void*m,uint64_t off,uint64_t,int,const char*){return (const float*)((const char*)m+off);}
'''+fn('qwen4exp_sigmoid')+'\n'+fn('qwen4exp_ple_conv_kernel')+'\n'+fn('qwen4exp_ple_conv_adopt_kernel',True)+'\n'+fn('qwen4exp_ple_conv_impl')+'\n'+fn('ds4_gpu_qwen4exp_ple_conv_adopt_tensor')+r'''
int main(){
 std::mt19937 rng(510091);unsigned checks=0,eager=0,replays=0;
 cudaStream_t stream;CK(cudaStreamCreate(&stream));
 for(unsigned C:{1u,257u,10240u})for(unsigned R:{1u,2u})for(unsigned nr=0;nr<R;nr++){
  const size_t H=(size_t)R*C,S=(size_t)9*C,P=S*6,W=(size_t)C*4;
  float *dh[2],*ds[2],*dp[2],*dg,*dx,*dw;uint32_t*df;
  for(int i=0;i<2;i++){CK(cudaMalloc(&dh[i],(H+2)*4));CK(cudaMalloc(&ds[i],(S+2)*4));CK(cudaMalloc(&dp[i],(P+2)*4));}
  CK(cudaMalloc(&dg,H*4));CK(cudaMalloc(&dx,H*4));CK(cudaMalloc(&dw,W*4));CK(cudaMalloc(&df,4));
  std::vector<float> h(H),live(S),snap(P),g(H),x(H),w(W),old(S);
  std::vector<uint32_t>a,b;
  unsigned flag=0,mode=0;
  auto setup=[&](){
   auto fill=[&](std::vector<float>&v){for(auto&t:v)t=(int(rng()%20001)-10000)*0.0001f;};
   fill(h);fill(live);fill(snap);fill(g);fill(x);fill(w);
   auto specials=[&](std::vector<float>&v){if(mode==0)return;uint32_t bits=mode==1?1u:mode==2?0x80000000u:mode==3?0x7f800000u:mode==4?0xff800000u:mode==5?0x7fc12345u:mode==6?0x00800000u:0x7f7fffffu;float q;memcpy(&q,&bits,4);for(size_t j=0;j<v.size();j+=31)v[j]=q;};
   specials(live);specials(snap);specials(g);specials(x);specials(w);
   old=live;if(flag>=1&&flag<=6)memcpy(old.data(),snap.data()+(flag-1)*S,S*4);
   for(int i=0;i<2;i++){
    CK(cudaMemset(dh[i],0xa5,(H+2)*4));CK(cudaMemset(ds[i],0xa5,(S+2)*4));CK(cudaMemset(dp[i],0xa5,(P+2)*4));
    CK(cudaMemcpy(dh[i]+1,h.data(),H*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(ds[i]+1,i?live.data():old.data(),S*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(dp[i]+1,snap.data(),P*4,cudaMemcpyHostToDevice));
   }
   CK(cudaMemcpy(dg,g.data(),H*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(dx,x.data(),H*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(dw,w.data(),W*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(df,&flag,4,cudaMemcpyHostToDevice));
  };
  auto launch=[&](cudaStream_t st){
   qwen4exp_ple_conv_kernel<<<(C+255)/256,256,0,st>>>(dh[0]+1,ds[0]+1,dg,dx,dw,dp[0]+1,C,4,3,9,R,nr);
   test_stream=st;
   ds4_gpu_tensor th={dh[1]+1,H*4,0},ts={ds[1]+1,S*4,0},tp={dp[1]+1,P*4,0},tg={dg,H*4,0},tx={dx,H*4,0},tf={df,4,0};
   if(!ds4_gpu_qwen4exp_ple_conv_adopt_tensor(&th,&ts,&tp,nr,&tg,&tx,dw,W*4,0,C,4,3,R,&tf,6)){fprintf(stderr,"extracted API refused valid case\n");exit(5);}
   CK(cudaGetLastError());
  };
  auto compare=[&](){for(unsigned which=0;which<3;which++){
   size_t n=(which==0?H:which==1?S:P)+2;float*p0=which==0?dh[0]:which==1?ds[0]:dp[0],*p1=which==0?dh[1]:which==1?ds[1]:dp[1];
   a.resize(n);b.resize(n);CK(cudaMemcpy(a.data(),p0,n*4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(b.data(),p1,n*4,cudaMemcpyDeviceToHost));
   if(a!=b){size_t at=0;while(at<n&&a[at]==b[at])at++;fprintf(stderr,"PLE mismatch C%u R%u nr%u flag%u mode%u buffer%u at%zu old%08x new%08x\n",C,R,nr,flag,mode,which,at,a[at],b[at]);exit(2);}
   if(a.front()!=0xa5a5a5a5u||a.back()!=0xa5a5a5a5u)exit(3);
  }checks++;};
  for(mode=0;mode<8;mode++)for(unsigned seedflag=0;seedflag<=6;seedflag++){
   flag=seedflag;setup();launch(stream);CK(cudaStreamSynchronize(stream));compare();eager++;
   cudaGraph_t graph;cudaGraphExec_t exec;CK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));launch(stream);CK(cudaStreamEndCapture(stream,&graph));CK(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
   for(unsigned replay=0;replay<2;replay++){flag=(seedflag+replay+1)%7;setup();CK(cudaGraphLaunch(exec,stream));CK(cudaStreamSynchronize(stream));compare();replays++;}
   CK(cudaGraphExecDestroy(exec));CK(cudaGraphDestroy(graph));
  }
  // Invalid calls must not launch or alter any output. Resolver is synthetic;
  // real current-device and pointer-attribute checks still execute.
  std::vector<uint32_t> before(H+2),after(H+2);CK(cudaMemcpy(before.data(),dh[1],(H+2)*4,cudaMemcpyDeviceToHost));
  for(unsigned bad=0;bad<9;bad++){
   ds4_gpu_tensor th={dh[1]+1,H*4,0},ts={ds[1]+1,S*4,0},tp={dp[1]+1,P*4,0},tg={dg,H*4,0},tx={dx,H*4,0},tf={df,4,0};
   unsigned slots=6,nr_bad=nr,conv=4;
   if(bad==0)tg.ptr=nullptr;if(bad==1)th.ptr=nullptr;if(bad==2)tf.bytes=3;
   if(bad==3)tf.ptr=(char*)df+1;if(bad==4)tg.device_id=1;
   if(bad==5)ts.ptr=th.ptr;if(bad==6)slots=7;if(bad==7)nr_bad=R;if(bad==8)conv=3;
   if(ds4_gpu_qwen4exp_ple_conv_adopt_tensor(&th,&ts,&tp,nr_bad,&tg,&tx,dw,W*4,0,C,conv,3,R,&tf,slots))exit(6);
  }
  CK(cudaMemcpy(after.data(),dh[1],(H+2)*4,cudaMemcpyDeviceToHost));if(before!=after)exit(7);
  for(int i=0;i<2;i++){CK(cudaFree(dh[i]));CK(cudaFree(ds[i]));CK(cudaFree(dp[i]));}CK(cudaFree(dg));CK(cudaFree(dx));CK(cudaFree(dw));CK(cudaFree(df));
 }
 CK(cudaStreamDestroy(stream));printf("PASS precise PLE lazy: %u eager + %u changed-data/flag graph replays; all hyper/live/six snapshot buffers and canaries (%u checks)\n",eager,replays,checks);
}
'''
with tempfile.TemporaryDirectory(prefix='ple-lazy-gpu-') as t:
 p=Path(t);(p/'test.cu').write_text(source)
 subprocess.run([os.environ.get('NVCC','nvcc'),'-O3','-ftz=false','-prec-div=true','-prec-sqrt=true','-arch='+os.environ.get('CUDA_TEST_ARCH','sm_86'),str(p/'test.cu'),'-o',str(p/'test')],check=True)
 subprocess.run([str(p/'test')],check=True)
