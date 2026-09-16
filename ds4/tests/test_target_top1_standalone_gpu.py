"""Extract unchanged production reduction bodies; standalone CUDA parity, no model.

Usage: python3 target_top1_standalone_gpu.py REPO OUTPUT.cu
Compile with nvcc -O3 --use_fast_math -arch=sm_86 OUTPUT.cu -o OUTPUT.
This exercises kernels, not the engine API/session or GB10 performance.
"""
from pathlib import Path
import sys

s = (Path(sys.argv[1]) / 'ds4/ds4_cuda.cu').read_text()
def function(signature):
    begin = s.index(signature)
    end = s.index('{', begin) + 1
    depth = 1
    while depth:
        depth += (s[end] == '{') - (s[end] == '}')
        end += 1
    return s[begin:end]

code = r'''
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#define CK(expr) do {cudaError_t e=(expr);if(e!=cudaSuccess){fprintf(stderr,"%s:%d %s: %s\n",__FILE__,__LINE__,#expr,cudaGetErrorString(e));exit(1);}}while(0)
#define NEED(expr) do {if(!(expr)){fprintf(stderr,"FAIL line%d %s\n",__LINE__,#expr);exit(2);}}while(0)
struct target_top1_pair {float value; uint32_t id;};
'''
for signature in [
    '__device__ __forceinline__ static bool topk_score_better',
    '__global__ static void indexer_top1_kernel',
    '__global__ static void target_top1_partition_kernel',
    '__global__ static void target_top1_finish_kernel',
]:
    code += function(signature) + '\n'
code += r'''
int main() {
 cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop,0));
 printf("GPU %s sm_%d%d; standalone production-source kernels only\n",prop.name,prop.major,prop.minor);
 cudaStream_t stream; CK(cudaStreamCreate(&stream));
 std::mt19937 rng(44044); unsigned cases=0,replays=0;
 for(uint32_t width:{65536u,65537u,65543u,248320u,262145u}) for(uint32_t rows:{1u,2u}) {
  size_t n=(size_t)width*rows;
  std::vector<float> x(n),after(n); float *d;
  uint32_t *old,*out; target_top1_pair *partial;
  CK(cudaMalloc(&d,n*4)); CK(cudaMalloc(&old,16)); CK(cudaMalloc(&out,16)); CK(cudaMalloc(&partial,528));
  cudaGraph_t graph; cudaGraphExec_t exec;
  CK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
  target_top1_partition_kernel<<<dim3(32,rows),256,0,stream>>>(partial,d,width,rows);
  target_top1_finish_kernel<<<rows,32,0,stream>>>(out,partial);
  CK(cudaStreamEndCapture(stream,&graph)); CK(cudaGraphInstantiate(&exec,graph,0));
  for(unsigned mode=0;mode<20;mode++) {
   for(auto &v:x){uint32_t bits=rng();std::memcpy(&v,&bits,4);}
   if(mode==1)std::fill(x.begin(),x.end(),NAN);
   if(mode==2)std::fill(x.begin(),x.end(),-INFINITY);
   if(mode==3)std::fill(x.begin(),x.end(),INFINITY);
   if(mode==4)for(size_t i=0;i<n;i++)x[i]=(i&1)?0.f:-0.f;
   if(mode==5){std::fill(x.begin(),x.end(),-1.f);for(unsigned r=0;r<rows;r++){x[(size_t)r*width]=NAN;x[(size_t)(r+1)*width-1]=7.f;}}
   if(mode==6){std::fill(x.begin(),x.end(),-1.f);for(unsigned r=0;r<rows;r++)for(unsigned i:{255u,256u,1023u,1024u,8191u,8192u})x[(size_t)r*width+i]=7.f;}
   if(mode==7){std::fill(x.begin(),x.end(),-INFINITY);x[0]=NAN;x[width-1]=INFINITY;}
   if(mode==8||mode==9)for(size_t i=0;i<n;i++){uint32_t bits=(i&1?0x80000000u:0)|((mode==8)?1u:(i%3?0x7fffffu:0x800000u));std::memcpy(&x[i],&bits,4);}
   if(mode==10){std::fill(x.begin(),x.end(),-1.f);for(unsigned r=0;r<rows;r++)x[(size_t)r*width+100+r]=10.f;}
   CK(cudaMemcpyAsync(d,x.data(),n*4,cudaMemcpyHostToDevice,stream));
   CK(cudaMemsetAsync(old,0xa5,16,stream)); CK(cudaMemsetAsync(out,0xa5,16,stream)); CK(cudaMemsetAsync(partial,0xa5,528,stream));
   indexer_top1_kernel<<<rows,1024,0,stream>>>(old,d,width,rows);
   target_top1_partition_kernel<<<dim3(32,rows),256,0,stream>>>(partial,d,width,rows);
   target_top1_finish_kernel<<<rows,32,0,stream>>>(out,partial);
   CK(cudaGetLastError()); CK(cudaStreamSynchronize(stream));
   uint32_t a[4],b[4]; unsigned char scratch[528];
   CK(cudaMemcpy(a,old,16,cudaMemcpyDeviceToHost)); CK(cudaMemcpy(b,out,16,cudaMemcpyDeviceToHost));
   if(memcmp(a,b,16)){fprintf(stderr,"width%u rows%u mode%u old%u,%u new%u,%u\n",width,rows,mode,a[0],a[1],b[0],b[1]);return 3;}
   CK(cudaMemcpy(after.data(),d,n*4,cudaMemcpyDeviceToHost)); NEED(!memcmp(x.data(),after.data(),n*4));
   CK(cudaMemcpy(scratch,partial,528,cudaMemcpyDeviceToHost));
   for(unsigned i=rows*256;i<528;i++)NEED(scratch[i]==0xa5);
   for(unsigned i=rows;i<4;i++)NEED(b[i]==0xa5a5a5a5);
   CK(cudaMemsetAsync(out,0xa5,16,stream)); CK(cudaGraphLaunch(exec,stream)); CK(cudaStreamSynchronize(stream));
   CK(cudaMemcpy(b,out,16,cudaMemcpyDeviceToHost)); NEED(!memcmp(a,b,16));
   cases++;replays++;
  }
  CK(cudaGraphExecDestroy(exec)); CK(cudaGraphDestroy(graph)); CK(cudaFree(partial)); CK(cudaFree(out)); CK(cudaFree(old)); CK(cudaFree(d));
 }
 CK(cudaStreamDestroy(stream)); printf("PASS %u eager parity cases, %u changed-input captured replays; input/output/scratch canaries\n",cases,replays);
}
'''
Path(sys.argv[2]).write_text(code)
