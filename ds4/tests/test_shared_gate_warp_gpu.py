#!/usr/bin/env python3
"""Actual precise F32 shared gate old/new parity; PDL disabled on SM86."""
import os,subprocess,tempfile
from pathlib import Path
R=Path(__file__).resolve().parents[2];s=(R/'ds4/ds4_cuda_qwen4exp.cu').read_text()
def block(marker):
 a=s.index(marker);p=s.index('{',a);i=p+1;n=1
 while n:n+=(s[i]=='{')-(s[i]=='}');i+=1
 return s[a:i]
parts=[block('__device__ __forceinline__ static float '+n+'(') for n in ('dev_qwen4exp_f32_value','dev_qwen4exp_weight_value','dev_qwen4exp_block_sum')]
parts+=['template<int RouterType=-1>\n'+block('__global__ static void qwen4exp_shared_gate_kernel('),block('__global__ static void qwen4exp_shared_gate_short_kernel(')]
c=r'''
#include <cuda_runtime.h>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define QWEN4EXP_PDL_TRIGGER() ((void)0)
#define QWEN4EXP_SHARED_GATE_STEPS 10u
#define DS4_QWEN4EXP_TY_f32 0
#define DS4_QWEN4EXP_MOE_TYPES(X) X(f32,0)
#define CK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));abort();}}while(0)
'''+ '\n'.join(parts)+r'''
int main(){float *x,*w,*a,*b;CK(cudaMalloc(&x,5120*4));CK(cudaMalloc(&w,2560*4));CK(cudaMalloc(&a,64));CK(cudaMalloc(&b,64));cudaStream_t st;CK(cudaStreamCreate(&st));unsigned eager=0,replays=0;
 std::vector<uint32_t>hx(5120),hw(2560),rx(5120),rw(2560),oa(16),ob(16);
 for(unsigned rows=1;rows<=2;rows++){
  cudaGraph_t gr;cudaGraphExec_t ex;CK(cudaStreamBeginCapture(st,cudaStreamCaptureModeGlobal));
  qwen4exp_shared_gate_kernel<0><<<rows,256,1024,st>>>(a,(const char*)w,x,0,2560,rows);
  qwen4exp_shared_gate_short_kernel<<<rows,256,1024,st>>>(b,(const char*)w,x,0,2560,rows);
  CK(cudaStreamEndCapture(st,&gr));CK(cudaGraphInstantiate(&ex,gr,0));
  for(unsigned pat=0;pat<12;pat++)for(unsigned trial=0;trial<8;trial++)for(unsigned replay=0;replay<3;replay++){
   for(unsigned i=0;i<5120;i++){float f=((int)((i*37+trial*11+replay*7)%127)-63)*0.003f;memcpy(&hx[i],&f,4);if(pat==1)hx[i]=(i&1?0x80000001:1);if(pat==2)hx[i]=(i&1?0x80000000:0);if(pat==3)hx[i]=(i==trial*256?0x7f800000:hx[i]);if(pat==4)hx[i]=(i%256==trial?0x7fc00123:hx[i]);if(pat==5)hx[i]=(i%256==trial?0xff800000:hx[i]);if(pat==9)hx[i]=(i&1?0x7f7fffff:0xff7fffff);}
   for(unsigned i=0;i<2560;i++){float f=((int)((i*19+trial*5+replay*13)%63)-31)*0.007f;memcpy(&hw[i],&f,4);if(pat==6)hw[i]=(i%256==trial?0x7fc12345:hw[i]);if(pat==7)hw[i]=(i%256==trial?0x7f800000:hw[i]);if(pat==8)hw[i]=(i&1?0x80000001:1);if(pat==10)hw[i]=(i%256==trial?0xffc23456:hw[i]);if(pat==11)hw[i]=(i&1?0x80000000:0);}
   CK(cudaMemcpy(x,hx.data(),hx.size()*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(w,hw.data(),hw.size()*4,cudaMemcpyHostToDevice));CK(cudaMemset(a,0xa5,64));CK(cudaMemset(b,0xa5,64));
   if(replay){CK(cudaGraphLaunch(ex,st));replays++;}else{qwen4exp_shared_gate_kernel<0><<<rows,256,1024>>>(a,(const char*)w,x,0,2560,rows);qwen4exp_shared_gate_short_kernel<<<rows,256,1024>>>(b,(const char*)w,x,0,2560,rows);eager++;}
   CK(cudaDeviceSynchronize());CK(cudaMemcpy(oa.data(),a,64,cudaMemcpyDeviceToHost));CK(cudaMemcpy(ob.data(),b,64,cudaMemcpyDeviceToHost));assert(oa==ob);for(unsigned i=rows;i<16;i++)assert(ob[i]==0xa5a5a5a5);
   CK(cudaMemcpy(rx.data(),x,hx.size()*4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(rw.data(),w,hw.size()*4,cudaMemcpyDeviceToHost));assert(rx==hx&&rw==hw);
  }CK(cudaGraphExecDestroy(ex));CK(cudaGraphDestroy(gr));
 }printf("PASS shared gate: %u eager + %u changed-input/weight graph replays; bitwise output, tail guards, immutable inputs; SM86 PDL disabled\n",eager,replays);
}
'''
with tempfile.TemporaryDirectory(prefix='shared-gate-') as d:
 p=Path(d)/'test.cu';p.write_text(c);e=Path(d)/'test';subprocess.run([os.environ.get('NVCC','nvcc'),'-O3','-ftz=false','-prec-div=true','-prec-sqrt=true','-arch=sm_86',str(p),'-o',str(e)],check=True);subprocess.run([str(e)],check=True)
