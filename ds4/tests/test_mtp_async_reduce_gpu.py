"""Generate actual-kernel CUDA parity test; no model or engine API.
Usage: python3 test_mtp_async_reduce_gpu.py REPO OUTPUT.cu
Compile: nvcc -O3 --use_fast_math -arch=sm_86 OUTPUT.cu -o OUTPUT
"""
from pathlib import Path
import sys
repo=Path(sys.argv[1]);main=(repo/'ds4/ds4_cuda.cu').read_text();native=(repo/'ds4/ds4_cuda_mtp_native.cuh').read_text()
def function(s,marker):
 a=s.index(marker);b=s.index('{',a)+1;depth=1
 while depth:depth+=(s[b]=='{')-(s[b]=='}');b+=1
 return s[a:b]
code=r'''
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(e));exit(1);}}while(0)
#define NEED(x) do{if(!(x)){fprintf(stderr,"failed line %d\n",__LINE__);exit(2);}}while(0)
'''
code+='\n'+function(main,'__device__ __forceinline__ static bool topk_score_better(')
code+='\n'+function(main,'__global__ static void indexer_top1_kernel(')
code+='\n'+function(native,'__global__ static void mtp_native_map(')
code+='\n'+function(native,'__global__ static void mtp_native_reduce_pending(')
code+=r'''
int main(){constexpr unsigned N=2048;std::mt19937 rng(46001);
 float *x,*sorted;uint32_t *ids,*ordered,*flag,*out,*old;
 CK(cudaMalloc(&x,N*4));CK(cudaMalloc(&sorted,N*4));CK(cudaMalloc(&ids,N*4));CK(cudaMalloc(&ordered,N*4));CK(cudaMalloc(&flag,4));CK(cudaMalloc(&out,16));CK(cudaMalloc(&old,16));
 cudaStream_t stream;CK(cudaStreamCreate(&stream));cudaGraph_t graph;cudaGraphExec_t exec;
 CK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
 mtp_native_reduce_pending<<<1,1024,0,stream>>>(out,x,ids,N,UINT32_MAX,flag);
 CK(cudaStreamEndCapture(stream,&graph));CK(cudaGraphInstantiate(&exec,graph,0));
 std::vector<float> scores(N),ref(N);std::vector<uint32_t> orig(N),order(N),refid(N);
 unsigned count=0;
 for(unsigned trial=0;trial<256;trial++){
  orig[0]=0;for(unsigned i=1;i<N;i++)orig[i]=UINT32_MAX-3*i;
  std::shuffle(orig.begin()+1,orig.end(),rng);
  for(unsigned i=0;i<N;i++){uint32_t bits=rng();memcpy(&scores[i],&bits,4);}
  switch(trial%8){case 0:std::fill(scores.begin(),scores.end(),-INFINITY);break;case 1:std::fill(scores.begin(),scores.end(),NAN);break;case 2:std::fill(scores.begin(),scores.end(),0.f);for(unsigned i=0;i<N;i+=2)scores[i]=-0.f;break;case 3:scores[0]=NAN;break;case 4:std::fill(scores.begin(),scores.end(),INFINITY);break;case 5:for(unsigned i=0;i<N;i++){uint32_t bits=i&1?1u:0x80000001u;memcpy(&scores[i],&bits,4);}break;}
  std::iota(order.begin(),order.end(),0);std::sort(order.begin(),order.end(),[&](unsigned a,unsigned b){return orig[a]<orig[b];});
  for(unsigned i=0;i<N;i++){ref[i]=scores[order[i]];refid[i]=orig[order[i]];}
  uint32_t bad=trial%13==0?UINT32_MAX:0;
  CK(cudaMemcpy(x,scores.data(),N*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(sorted,ref.data(),N*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(ids,orig.data(),N*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(ordered,refid.data(),N*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(flag,&bad,4,cudaMemcpyHostToDevice));
  CK(cudaMemset(old,0xa5,16));indexer_top1_kernel<<<1,1024>>>(old,sorted,N,1);CK(cudaGetLastError());mtp_native_map<<<1,1>>>(old,sorted,ordered,N,UINT32_MAX);CK(cudaGetLastError());uint32_t expected[4];CK(cudaMemcpy(expected,old,16,cudaMemcpyDeviceToHost));if(bad)expected[0]=UINT32_MAX;
  for(unsigned replay=0;replay<2;replay++){
   CK(cudaMemsetAsync(out,0xa5,16,stream));
   if(replay)CK(cudaGraphLaunch(exec,stream));else mtp_native_reduce_pending<<<1,1024,0,stream>>>(out,bad?nullptr:x,bad?nullptr:ids,N,UINT32_MAX,flag);
   CK(cudaGetLastError());CK(cudaStreamSynchronize(stream));uint32_t actual[4];CK(cudaMemcpy(actual,out,16,cudaMemcpyDeviceToHost));NEED(!memcmp(expected,actual,16));count++;
  }
  std::vector<float> after(N);std::vector<uint32_t> afterid(N);CK(cudaMemcpy(after.data(),x,N*4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(afterid.data(),ids,N*4,cudaMemcpyDeviceToHost));NEED(!memcmp(after.data(),scores.data(),N*4)&&afterid==orig);
 }
 CK(cudaGraphExecDestroy(exec));CK(cudaGraphDestroy(graph));CK(cudaStreamDestroy(stream));CK(cudaFree(x));CK(cudaFree(sorted));CK(cudaFree(ids));CK(cudaFree(ordered));CK(cudaFree(flag));CK(cudaFree(out));CK(cudaFree(old));
 printf("PASS %u actual-kernel original-ID parity cases (256 eager + 256 changed-input graph replay), flag-first null inputs, output canaries and input immutability\n",count);
}
'''
Path(sys.argv[2]).write_text(code)
