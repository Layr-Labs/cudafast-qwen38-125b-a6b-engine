"""Actual CUDA QSA triple R3 versus Q8 R4 and narrow row-tile R2 kernels.
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
helpers='\n'.join(body(marker) for marker in [
 '__device__ static float warp_sum_f32(',
 '__device__ __forceinline__ static int32_t load_i8x4_i32_unaligned(',
 '__device__ __forceinline__ static int32_t q8_0_tail_word(',
 '__device__ __forceinline__ static int32_t dot_i8x32_dp4a(',
 '__device__ __forceinline__ static int32_t dot_i8_block(',
 '__device__ __forceinline__ static void q8_0_group_words(',
 '__device__ __forceinline__ static int32_t dot_i8x32_dp4a_words('])
tile=body('template <int R>\n__global__ static void matmul_q8_0_preq_rows_exact_tile_kernel(')
fused=body('template<int R>\n__global__ static void qwen_q8_projection_triple_kernel(')
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
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s line%d\n",cudaGetErrorString(e),__LINE__);exit(1);}}while(0)
'''+pair+'\n'+helpers+'\n'+tile+'\n'+fused+r'''
int main(){
 std::mt19937 rng(600017);unsigned eager=0,replays=0;cudaStream_t st;CK(cudaStreamCreate(&st));
 for(unsigned in_dim:{32u,64u,2560u,2592u})for(unsigned width:{1u,511u,513u,1024u,12288u})for(unsigned off:{0u,2u,4u})for(unsigned mode=0;mode<12;mode++){
  if(in_dim!=2560&&width!=511)continue;
  const unsigned blocks=in_dim/32;
  unsigned od[3]={width,width+2,width+4}; if(width==12288)od[1]=od[2]=512;
  unsigned char*dw[3];float *old[3],*fresh[3],*ds;int8_t*dq;
  std::vector<unsigned char>w[3];std::vector<float>sc(3*blocks);std::vector<int8_t>q(3*in_dim);
  for(auto&v:sc)v=(rng()%100+1)/127.f;for(auto&v:q)v=(int8_t)rng();
  if(mode==1)for(auto&v:q)v=-128;if(mode==2)for(auto&v:q)v=127;if(mode==3)for(auto&v:sc)v=-0.f; if(mode==10)for(auto&v:sc)v=__builtin_inff(); if(mode==11)for(auto&v:sc)v=__builtin_nanf("");
  CK(cudaMalloc(&ds,sc.size()*4));CK(cudaMalloc(&dq,q.size()));
  CK(cudaMemcpy(ds,sc.data(),sc.size()*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(dq,q.data(),q.size(),cudaMemcpyHostToDevice));
  for(unsigned k=0;k<3;k++){
   size_t wb=(size_t)od[k]*blocks*34+off;
   w[k].resize(wb);for(auto&v:w[k])v=(unsigned char)rng();
   for(size_t b=0;b<(size_t)od[k]*blocks;b++){
    uint16_t h=mode==4?1:mode==5?0x03ff:mode==6?0x8000:mode==7?0x7c00:mode==8?0xfc00:mode==9?0x7e00:0x1800;
    memcpy(w[k].data()+off+b*34,&h,2);
   }
   CK(cudaMalloc(&dw[k],wb));CK(cudaMemcpy(dw[k],w[k].data(),wb,cudaMemcpyHostToDevice));
   size_t ob=((size_t)3*od[k]+8)*4;CK(cudaMalloc(&old[k],ob));CK(cudaMalloc(&fresh[k],ob));
  }
  auto launch=[&](){
   for(unsigned k=0;k<3;k++){size_t ob=((size_t)3*od[k]+8)*4;CK(cudaMemsetAsync(old[k],0xa5,ob,st));CK(cudaMemsetAsync(fresh[k],0xa5,ob,st));}
   for(unsigned k=0;k<3;k++) {
    if(od[k]>512)matmul_q8_0_preq_pair_lanes_kernel<4,false><<<(od[k]+3)/4,256,0,st>>>(old[k]+4,dw[k]+off,dq,ds,od[k],3,blocks);
    else matmul_q8_0_preq_rows_exact_tile_kernel<2><<<dim3(od[k],2),32,0,st>>>(old[k]+4,dw[k]+off,dq,ds,in_dim,od[k],3,blocks,1);
   }
   unsigned grid=(od[0]+3)/4+(od[1]+3)/4+(od[2]+3)/4;
   qwen_q8_projection_triple_kernel<3><<<grid,256,0,st>>>(fresh[0]+4,fresh[1]+4,fresh[2]+4,dw[0]+off,dw[1]+off,dw[2]+off,dq,ds,od[0],od[1],od[2],3,blocks);
   CK(cudaGetLastError());
  };
  auto compare=[&](){for(unsigned k=0;k<3;k++){
   size_t n=(size_t)3*od[k]+8;std::vector<uint32_t>ref(n),got(n);
   CK(cudaMemcpy(ref.data(),old[k],n*4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(got.data(),fresh[k],n*4,cudaMemcpyDeviceToHost));
   if(ref!=got){fprintf(stderr,"parity width%u off%u mode%u output%u\n",width,off,mode,k);exit(2);}
   for(size_t j=0;j<n;j++)if((j<4||j>=n-4)&&got[j]!=0xa5a5a5a5u)exit(3);
  }};
  launch();CK(cudaStreamSynchronize(st));compare();eager++;
  cudaGraph_t g;cudaGraphExec_t ex;CK(cudaStreamBeginCapture(st,cudaStreamCaptureModeGlobal));launch();CK(cudaStreamEndCapture(st,&g));CK(cudaGraphInstantiate(&ex,g,nullptr,nullptr,0));
  for(unsigned rep=0;rep<2;rep++){for(auto&v:q)v=(int8_t)rng();CK(cudaMemcpy(dq,q.data(),q.size(),cudaMemcpyHostToDevice));CK(cudaGraphLaunch(ex,st));CK(cudaStreamSynchronize(st));compare();replays++;}
  CK(cudaGraphExecDestroy(ex));CK(cudaGraphDestroy(g));
  for(unsigned k=0;k<3;k++){std::vector<unsigned char>after(w[k].size());CK(cudaMemcpy(after.data(),dw[k],after.size(),cudaMemcpyDeviceToHost));if(after!=w[k])exit(4);CK(cudaFree(dw[k]));CK(cudaFree(old[k]));CK(cudaFree(fresh[k]));}
  std::vector<int8_t>qafter(q.size());std::vector<float>safter(sc.size());
  CK(cudaMemcpy(qafter.data(),dq,q.size(),cudaMemcpyDeviceToHost));CK(cudaMemcpy(safter.data(),ds,sc.size()*4,cudaMemcpyDeviceToHost));
  if(qafter!=q||memcmp(safter.data(),sc.data(),sc.size()*4)!=0)exit(5);
  CK(cudaFree(ds));CK(cudaFree(dq));
 }
 CK(cudaStreamDestroy(st));printf("PASS QSA triple R3: %u eager, %u changed-input graph replays, all three complete outputs/canaries/weights\n",eager,replays);
}
'''
with tempfile.TemporaryDirectory(prefix='qsa-r3-') as tmp:
 p=Path(tmp);(p/'test.cu').write_text(source)
 subprocess.run([os.environ.get('NVCC','nvcc'),'-O3','--use_fast_math','-arch='+os.environ.get('CUDA_TEST_ARCH','sm_86'),str(p/'test.cu'),'-o',str(p/'test')],check=True)
 subprocess.run([str(p/'test')],check=True)
