"""Actual extracted target Q8 kernels, bitwise parity on synthetic CUDA data.
NVCC chooses compiler; CUDA_TEST_ARCH defaults sm_86. PDL is disabled only in
this standalone arithmetic test. Production sm_121a compilation tests PDL code,
not runtime scheduling. No model or performance score is measured.
"""
from pathlib import Path
import os, subprocess, tempfile
repo=Path(__file__).resolve().parents[2]
s=(repo/'ds4/ds4_cuda.cu').read_text()
def kernel(name):
 a=s.index('__global__ static void '+name+'(');a=s.rfind('template <',0,a)
 b=s.index('{',a);e=b+1;depth=1
 while depth:depth+=(s[e]=='{')-(s[e]=='}');e+=1
 return s[a:e]
source=r'''
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#include <algorithm>
#define QWEN4EXP_PDL_SYNC() ((void)0)
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s line%d\n",cudaGetErrorString(e),__LINE__);exit(1);}}while(0)
'''+kernel('matmul_q8_0_preq_pair_lanes_kernel')+'\n'+kernel('matmul_q8_target_fixed_kernel')+r'''
int main(){
 std::mt19937 rng(490019);unsigned eager=0,replays=0;
 cudaStream_t stream;CK(cudaStreamCreate(&stream));
 for(unsigned rows:{1u,2u})for(unsigned width:{1u,3u,4u,5u,513u,1025u})for(unsigned offset:{0u,2u}){
  size_t wb=(size_t)width*80*34+4,ob=((size_t)rows*width+4)*4;
  unsigned char*dw;int8_t*dx;float*ds,*do0,*do1;
  CK(cudaMalloc(&dw,wb));CK(cudaMalloc(&dx,rows*2560));CK(cudaMalloc(&ds,rows*80*4));CK(cudaMalloc(&do0,ob));CK(cudaMalloc(&do1,ob));
  std::vector<unsigned char>w(wb),after(wb);std::vector<int8_t>x(rows*2560);std::vector<float>scales(rows*80);
  std::vector<uint32_t>a(ob/4),b(ob/4);
  for(unsigned mode=0;mode<16;mode++){
   for(auto&v:w)v=(unsigned char)rng();for(auto&v:x)v=(int8_t)rng();
   for(auto&v:scales)v=mode==8?-0.f:mode==9?0.f:(float)(rng()%100+1)/127.f;
   for(unsigned row=0;row<width;row++)for(unsigned g=0;g<80;g++){
    uint16_t h=(uint16_t)rng();if(mode<4)h=0x3c00;if(mode==4)h=1;if(mode==5)h=0x03ff;
    if(mode==6)h=0x0400;if(mode==7)h=0x8000;if(mode==10)h=0x7c00;if(mode==11)h=0xfc00;if(mode==12)h=0x7e00;
    memcpy(w.data()+offset+(size_t)row*80*34+g*34,&h,2);
    if(mode==1||mode==2)memset(w.data()+offset+(size_t)row*80*34+g*34+2,mode==1?128:127,32);
   }
   if(mode==1)std::fill(x.begin(),x.end(),-128);if(mode==2)std::fill(x.begin(),x.end(),127);if(mode==3)std::fill(x.begin(),x.end(),0);
   CK(cudaMemcpy(dw,w.data(),wb,cudaMemcpyHostToDevice));CK(cudaMemcpy(dx,x.data(),x.size(),cudaMemcpyHostToDevice));CK(cudaMemcpy(ds,scales.data(),scales.size()*4,cudaMemcpyHostToDevice));
   auto launch=[&](cudaStream_t st){
    CK(cudaMemsetAsync(do0,0xa5,ob,st));CK(cudaMemsetAsync(do1,0xa5,ob,st));
    if(rows==1){
     matmul_q8_0_preq_pair_lanes_kernel<1,false><<<(width+3)/4,256,0,st>>>(do0+1,dw+offset,dx,ds,width,rows,80);
     matmul_q8_target_fixed_kernel<1><<<(width+3)/4,256,0,st>>>(do1+1,dw+offset,dx,ds,width);
    }else{
     matmul_q8_0_preq_pair_lanes_kernel<2,false><<<(width+3)/4,256,0,st>>>(do0+1,dw+offset,dx,ds,width,rows,80);
     matmul_q8_target_fixed_kernel<2><<<(width+3)/4,256,0,st>>>(do1+1,dw+offset,dx,ds,width);
    }CK(cudaGetLastError());
   };
   auto compare=[&](){CK(cudaMemcpy(a.data(),do0,ob,cudaMemcpyDeviceToHost));CK(cudaMemcpy(b.data(),do1,ob,cudaMemcpyDeviceToHost));if(a!=b){fprintf(stderr,"parity rows%u width%u offset%u mode%u\n",rows,width,offset,mode);exit(2);}if(a[0]!=0xa5a5a5a5u)exit(3);for(size_t i=1+(size_t)rows*width;i<a.size();i++)if(a[i]!=0xa5a5a5a5u)exit(3);};
   launch(stream);CK(cudaStreamSynchronize(stream));compare();eager++;
   cudaGraph_t graph;cudaGraphExec_t exec;CK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));launch(stream);CK(cudaStreamEndCapture(stream,&graph));CK(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
   for(unsigned repeat=0;repeat<2;repeat++){for(auto&v:x)v=(int8_t)rng();CK(cudaMemcpy(dx,x.data(),x.size(),cudaMemcpyHostToDevice));CK(cudaGraphLaunch(exec,stream));CK(cudaStreamSynchronize(stream));compare();replays++;}
   CK(cudaGraphExecDestroy(exec));CK(cudaGraphDestroy(graph));CK(cudaMemcpy(after.data(),dw,wb,cudaMemcpyDeviceToHost));if(after!=w)exit(4);
  }
  CK(cudaFree(dw));CK(cudaFree(dx));CK(cudaFree(ds));CK(cudaFree(do0));CK(cudaFree(do1));
 }
 CK(cudaStreamDestroy(stream));printf("PASS target fixed Q8: %u eager + %u changed-input graph replays; bitwise full outputs/canaries/weight immutability\n",eager,replays);
}
'''
with tempfile.TemporaryDirectory(prefix='target-q8-fixed-') as tmp:
 p=Path(tmp);(p/'test.cu').write_text(source)
 subprocess.run([os.environ.get('NVCC','nvcc'),'-O3','--use_fast_math','-arch='+os.environ.get('CUDA_TEST_ARCH','sm_86'),str(p/'test.cu'),'-o',str(p/'test')],check=True)
 subprocess.run([str(p/'test')],check=True)
