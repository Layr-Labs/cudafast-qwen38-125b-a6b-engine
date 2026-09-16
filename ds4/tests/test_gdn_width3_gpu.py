"""Actual CUDA R3 fusion versus existing Q8 R4 and F32 R2+R1 kernels.
Synthetic operator tests only; PDL is disabled for local SM86, as the production
three-row path has no PDL producer. No model or GB10 performance is measured.
"""
from pathlib import Path
import os, subprocess, tempfile
repo=Path(__file__).resolve().parents[2]
s=(repo/'ds4/ds4_cuda.cu').read_text()
def body(marker, start=None):
 a=s.index(marker) if start is None else start
 b=s.index('{',a);e=b+1;depth=1
 while depth: depth+=(s[e]=='{')-(s[e]=='}');e+=1
 return s[a:e]
pair=s.index('__global__ static void matmul_q8_0_preq_pair_lanes_kernel(')
pair=body('',s.rfind('template <',0,pair))
helper=body('template<int C>\n__device__ __forceinline__ void qwen_f32_vector_read')
f32=body('template<int R, int C, int U>\n__global__ __launch_bounds__(256/C)')
args=body('struct qwen_gdn_projection_args')+';'
fused=body('template<int R, bool Stage=false>\n__global__ QW_GDN_PROJ_ATTR')
source=r'''
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#define QWEN4EXP_PDL_SYNC() ((void)0)
#define QW_GDN_PROJ_ATTR __maxnreg__(40)
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s line%d\n",cudaGetErrorString(e),__LINE__);exit(1);}}while(0)
'''+pair+'\n'+helper+'\n'+f32+'\n'+args+'\n'+fused+r'''
int main(){
 std::mt19937 rng(590017);unsigned eager=0,replays=0;cudaStream_t st;CK(cudaStreamCreate(&st));
 for(unsigned width:{516u,1024u,513u,10240u})for(unsigned off:{0u,2u,4u})for(unsigned mode=0;mode<12;mode++){
  unsigned od[4]={width,width+4,48,48}; if(width==513)od[1]=515; if(width==10240)od[1]=6144;
  qwen_gdn_projection_args a{};a.od[0]=od[0];a.od[1]=od[1];a.blocks=80;a.n_rows=3;
  unsigned char*dw[4];float *old[4],*fresh[4],*dx,*ds;int8_t*dq;
  std::vector<unsigned char>w[4];std::vector<float>x(3*2560),sc(3*80);std::vector<int8_t>q(3*2560);
  for(auto&v:x)v=(int(rng()%2001)-1000)*.001f;for(auto&v:sc)v=(rng()%100+1)/127.f;for(auto&v:q)v=(int8_t)rng();
  if(mode==1)for(auto&v:q)v=-128;if(mode==2)for(auto&v:q)v=127;if(mode==3)for(auto&v:x)v=-0.f;
  CK(cudaMalloc(&dx,x.size()*4));CK(cudaMalloc(&ds,sc.size()*4));CK(cudaMalloc(&dq,q.size()));
  CK(cudaMemcpy(dx,x.data(),x.size()*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(ds,sc.data(),sc.size()*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(dq,q.data(),q.size(),cudaMemcpyHostToDevice));a.x=dx;a.xscale=ds;a.xq=dq;
  for(unsigned k=0;k<4;k++){
   size_t wb=k<2?(size_t)od[k]*80*34+off:48*2560*4;
   w[k].resize(wb);for(auto&v:w[k])v=(unsigned char)rng();
   if(k<2){for(size_t b=0;b<(size_t)od[k]*80;b++){
    uint16_t h=mode==4?1:mode==5?0x03ff:mode==6?0x8000:mode==7?0x7c00:mode==8?0xfc00:mode==9?0x7e00:0x1800;
    memcpy(w[k].data()+off+b*34,&h,2);
   }}else{for(size_t b=0;b<48*2560;b++){float v=(int(rng()%2001)-1000)*.0002f;if(mode==10&&b%127==0)v=__builtin_inff();if(mode==11&&b%127==0)v=__builtin_nanf("");memcpy(w[k].data()+4*b,&v,4);}}
   CK(cudaMalloc(&dw[k],wb));CK(cudaMemcpy(dw[k],w[k].data(),wb,cudaMemcpyHostToDevice));a.weights[k]=dw[k]+(k<2?off:0);
   size_t ob=((size_t)3*od[k]+8)*4;CK(cudaMalloc(&old[k],ob));CK(cudaMalloc(&fresh[k],ob));a.out[k]=fresh[k]+4;
  }
  const bool stage=off!=2 && ((od[0]|od[1])&3)==0;
  auto launch=[&](){
   for(unsigned k=0;k<4;k++){size_t ob=((size_t)3*od[k]+8)*4;CK(cudaMemsetAsync(old[k],0xa5,ob,st));CK(cudaMemsetAsync(fresh[k],0xa5,ob,st));}
   for(unsigned k=0;k<2;k++)matmul_q8_0_preq_pair_lanes_kernel<4,false><<<(od[k]+3)/4,256,0,st>>>(old[k]+4,a.weights[k],dq,ds,od[k],3,80);
   for(unsigned k=2;k<4;k++){
    qwen_f32_vector_tree_kernel<2,2,10><<<48,128,0,st>>>(old[k]+4,(float*)a.weights[k],dx,48);
    qwen_f32_vector_tree_kernel<1,2,10><<<48,128,0,st>>>(old[k]+4+96,(float*)a.weights[k],dx+5120,48);
   }
   unsigned grid=(od[0]+3)/4+(od[1]+3)/4+96;
   if(stage)qwen_gdn_projection_kernel<3,true><<<grid,256,4*80*34+16,st>>>(a);
   else qwen_gdn_projection_kernel<3,false><<<grid,256,0,st>>>(a);
   CK(cudaGetLastError());
  };
  auto compare=[&](){for(unsigned k=0;k<4;k++){
   size_t n=(size_t)3*od[k]+8;std::vector<uint32_t>ref(n),got(n);
   CK(cudaMemcpy(ref.data(),old[k],n*4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(got.data(),fresh[k],n*4,cudaMemcpyDeviceToHost));
   if(ref!=got){fprintf(stderr,"parity width%u off%u mode%u output%u stage%d\n",width,off,mode,k,stage);exit(2);}
   for(size_t j=0;j<n;j++)if((j<4||j>=n-4)&&got[j]!=0xa5a5a5a5u)exit(3);
  }};
  launch();CK(cudaStreamSynchronize(st));compare();eager++;
  cudaGraph_t g;cudaGraphExec_t ex;CK(cudaStreamBeginCapture(st,cudaStreamCaptureModeGlobal));launch();CK(cudaStreamEndCapture(st,&g));CK(cudaGraphInstantiate(&ex,g,nullptr,nullptr,0));
  for(unsigned rep=0;rep<2;rep++){for(auto&v:x)v=(int(rng()%2001)-1000)*.001f;for(auto&v:q)v=(int8_t)rng();CK(cudaMemcpy(dx,x.data(),x.size()*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(dq,q.data(),q.size(),cudaMemcpyHostToDevice));CK(cudaGraphLaunch(ex,st));CK(cudaStreamSynchronize(st));compare();replays++;}
  CK(cudaGraphExecDestroy(ex));CK(cudaGraphDestroy(g));
  for(unsigned k=0;k<4;k++){std::vector<unsigned char>after(w[k].size());CK(cudaMemcpy(after.data(),dw[k],after.size(),cudaMemcpyDeviceToHost));if(after!=w[k])exit(4);CK(cudaFree(dw[k]));CK(cudaFree(old[k]));CK(cudaFree(fresh[k]));}
  CK(cudaFree(dx));CK(cudaFree(ds));CK(cudaFree(dq));
 }
 CK(cudaStreamDestroy(st));printf("PASS GDN R3: %u eager, %u changed-input graph replays, all four complete outputs/canaries/weights\n",eager,replays);
}
'''
with tempfile.TemporaryDirectory(prefix='gdn-r3-') as tmp:
 p=Path(tmp);(p/'test.cu').write_text(source)
 subprocess.run([os.environ.get('NVCC','nvcc'),'-O3','--use_fast_math','-arch='+os.environ.get('CUDA_TEST_ARCH','sm_86'),str(p/'test.cu'),'-o',str(p/'test')],check=True)
 subprocess.run([str(p/'test')],check=True)
