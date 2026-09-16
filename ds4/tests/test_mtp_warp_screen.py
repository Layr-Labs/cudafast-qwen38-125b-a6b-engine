"""Compile and execute extracted old/new coarse kernels on a CUDA GPU.
No model weights. NVCC selects the compiler, CUDA_TEST_ARCH defaults to sm_86.
This is an operator test, not a GB10 performance measurement.
"""
from pathlib import Path
import os, subprocess, tempfile
repo=Path(__file__).resolve().parents[2]
main=(repo/'ds4/ds4_cuda.cu').read_text();native=(repo/'ds4/ds4_cuda_mtp_native.cuh').read_text()
def function(s,name):
 a=s.index(name);a=s.rfind('\n',0,a)+1;b=s.index('{',a);e=b+1;d=1
 while d:d+=(s[e]=='{')-(s[e]=='}');e+=1
 return s[a:e]
helpers='\n'.join(function(main,n) for n in ['__device__ static float warp_sum_f32(', '__device__ __forceinline__ static uint32_t q8_top1_float_ordered_key(', '__device__ __forceinline__ static uint64_t q8_top1_pack_key('])
a=native.index('static constexpr uint32_t MTP_NATIVE_CAP');b=native.index('struct mtp_native_layout')
instrumented=function(native,'__global__ static void mtp_native_warp_screen_kernel(')
instrumented=instrumented.replace('mtp_native_warp_screen_kernel','mtp_native_warp_screen_values_probe').replace('uint64_t *keys,uint32_t *invalid)', 'uint64_t *keys,uint32_t *invalid,float *raw)')
instrumented=instrumented.replace('const float value=valid?total:-INFINITY;', 'const float value=valid?total:-INFINITY; raw[row]=value;')
source=r'''
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#include <cmath>
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s line%d\n",cudaGetErrorString(e),__LINE__);exit(1);}}while(0)
'''+helpers+'\n'+native[a:b]+'\n'+instrumented+r'''
int main(){
 constexpr uint32_t vocab=259,prefix=251,tail=7,width=prefix+tail;
 constexpr size_t wb=(size_t)vocab*80*34+4;
 unsigned char *dw;int8_t*dx;float*ds,*dr0,*dr1;uint64_t*dk0,*dk1;uint32_t*df0,*df1;
 CK(cudaMalloc(&dw,wb));CK(cudaMalloc(&dx,2560));CK(cudaMalloc(&ds,80*4));
 CK(cudaMalloc(&dr0,(width+2)*4));CK(cudaMalloc(&dr1,(width+2)*4));CK(cudaMalloc(&dk0,(width+2)*8));CK(cudaMalloc(&dk1,(width+2)*8));CK(cudaMalloc(&df0,4));CK(cudaMalloc(&df1,4));
 std::vector<unsigned char>w(wb),after(wb);std::vector<int8_t>x(2560);float scales[80];
 std::vector<uint64_t>k0(width+2),k1(width+2);std::vector<uint32_t>r0(width+2),r1(width+2);std::mt19937 rng(450019);unsigned cases=0;
 cudaStream_t stream;CK(cudaStreamCreate(&stream));
 for(unsigned offset:{0u,2u})for(unsigned mode=0;mode<16;mode++){
  for(auto&v:w)v=(unsigned char)rng();for(auto&v:x)v=(int8_t)rng();
  for(unsigned b=0;b<80;b++)scales[b]=mode==8?-0.f:mode==9?0.f:(float)(rng()%100+1)/127.f;
  for(unsigned row=0;row<vocab;row++)for(unsigned b=0;b<80;b++){
   uint16_t h=(uint16_t)rng();
   if(mode<4)h=0x3c00; if(mode==4)h=0x0001; if(mode==5)h=0x03ff;
   if(mode==6)h=0x0400; if(mode==7)h=0x8000;
   if(mode==10)h=0x7c00;if(mode==11)h=0xfc00;if(mode==12)h=0x7e00;
   memcpy(w.data()+offset+(size_t)row*80*34+b*34,&h,2);
  }
  if(mode==1){std::fill(x.begin(),x.end(),-128);for(unsigned row=0;row<vocab;row++)for(unsigned b=0;b<80;b++)memset(w.data()+offset+(size_t)row*80*34+b*34+2,128,32);}
  if(mode==2){std::fill(x.begin(),x.end(),127);for(unsigned row=0;row<vocab;row++)for(unsigned b=0;b<80;b++)memset(w.data()+offset+(size_t)row*80*34+b*34+2,127,32);}
  if(mode==3)std::fill(x.begin(),x.end(),0);
  CK(cudaMemcpy(dw,w.data(),wb,cudaMemcpyHostToDevice));CK(cudaMemcpy(dx,x.data(),2560,cudaMemcpyHostToDevice));CK(cudaMemcpy(ds,scales,320,cudaMemcpyHostToDevice));
  auto launch=[&](cudaStream_t st){
   CK(cudaMemsetAsync(dr0,0xa5,(width+2)*4,st));CK(cudaMemsetAsync(dr1,0xa5,(width+2)*4,st));
   CK(cudaMemsetAsync(dk0,0xa5,(width+2)*8,st));CK(cudaMemsetAsync(dk1,0xa5,(width+2)*8,st));CK(cudaMemsetAsync(df0,0,4,st));CK(cudaMemsetAsync(df1,0,4,st));
   mtp_native_projection_kernel<true,true><<<(width+3)/4,256,0,st>>>(nullptr,dw+offset,dx,ds,width,nullptr,vocab,prefix,tail,dk0,df0);CK(cudaGetLastError());
   mtp_native_warp_screen_kernel<<<(width+3)/4,128,0,st>>>(dw+offset,dx,ds,width,vocab,prefix,tail,dk1,df1);CK(cudaGetLastError());
   mtp_native_projection_kernel<true,false><<<(width+3)/4,256,0,st>>>(dr0,dw+offset,dx,ds,width,nullptr,vocab,prefix,tail);CK(cudaGetLastError());
   mtp_native_warp_screen_values_probe<<<(width+3)/4,128,0,st>>>(dw+offset,dx,ds,width,vocab,prefix,tail,dk1,df1,dr1);CK(cudaGetLastError());
  };
  auto compare=[&](){uint32_t f0,f1;CK(cudaMemcpy(k0.data(),dk0,(width+2)*8,cudaMemcpyDeviceToHost));CK(cudaMemcpy(k1.data(),dk1,(width+2)*8,cudaMemcpyDeviceToHost));CK(cudaMemcpy(&f0,df0,4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(&f1,df1,4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(r0.data(),dr0,(width+2)*4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(r1.data(),dr1,(width+2)*4,cudaMemcpyDeviceToHost));if(k0!=k1||f0!=f1||r0!=r1){fprintf(stderr,"mismatch offset%u mode%u\n",offset,mode);exit(2);}for(unsigned i=width;i<width+2;i++)if(k0[i]!=0xa5a5a5a5a5a5a5a5ull)exit(3);cases++;};
  launch(stream);CK(cudaStreamSynchronize(stream));compare();
  cudaGraph_t graph;cudaGraphExec_t exec;CK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));launch(stream);CK(cudaStreamEndCapture(stream,&graph));CK(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
  for(unsigned replay=0;replay<3;replay++){for(auto&v:x)v=(int8_t)rng();CK(cudaMemcpy(dx,x.data(),2560,cudaMemcpyHostToDevice));CK(cudaGraphLaunch(exec,stream));CK(cudaStreamSynchronize(stream));compare();}
  CK(cudaGraphExecDestroy(exec));CK(cudaGraphDestroy(graph));CK(cudaMemcpy(after.data(),dw,wb,cudaMemcpyDeviceToHost));if(after!=w)exit(4);
 }
 CK(cudaStreamDestroy(stream));CK(cudaFree(dr0));CK(cudaFree(dr1));CK(cudaFree(dw));CK(cudaFree(dx));CK(cudaFree(ds));CK(cudaFree(dk0));CK(cudaFree(dk1));CK(cudaFree(df0));CK(cudaFree(df1));
 printf("PASS %u actual old/new CUDA comparisons, eager and changed-input graph replay, offsets0/2, key parents/flags/raw coarse values including mandatory IDs/weight immutability\n",cases);
}
'''
with tempfile.TemporaryDirectory(prefix='mtp-warp-screen-') as d:
 p=Path(d)/'test.cu';p.write_text(source);binary=Path(d)/'test'
 subprocess.run([os.environ.get('NVCC','nvcc'),'-O3','--use_fast_math','-arch='+os.environ.get('CUDA_TEST_ARCH','sm_86'),str(p),'-o',str(binary)],check=True)
 subprocess.run([str(binary)],check=True)
