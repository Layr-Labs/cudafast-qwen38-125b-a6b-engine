#!/usr/bin/env python3
"""Actual CUDA embedding kernels/API with synthetic weights and a resolver shim.
Old gather+repeat and new fused seed use the same production fastmath flags.
Graph replay captures only a consumer; fresh by-value IDs are enqueued eagerly.
"""
import os, subprocess, tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
s=(ROOT/'ds4/ds4_cuda.cu').read_text()
def block(marker):
 a=s.index(marker);p=s.index('{',a);n=1;i=p+1
 while n:n+=(s[i]=='{')-(s[i]=='}');i+=1
 return s[a:i]
parts=[block('__global__ static void '+n+'(') for n in ('glm_embed_tokens_q8_0_kernel','repeat_hc_rows_kernel','qwen4exp_embed_short_kernel')]
parts += [block('static bool qwen4exp_embed_disjoint('),block('extern "C" int ds4_gpu_qwen4exp_embed_short_tensor(')]
code=r'''
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));abort();}}while(0)
struct ds4_gpu_tensor {void*ptr;uint64_t bytes;int device_id;};
static int g_n_gpus=1;static struct {int device_id;} g_gpu[1]={{0}};
static int ds4_tensor_device_idx(const ds4_gpu_tensor*t){return t->device_id<0?0:t->device_id;}
static bool cuda_ok(cudaError_t e,const char*){return e==cudaSuccess;}
static const char*cuda_resolve_weight_ptr(const void*m,uint64_t o,uint64_t,int,const char*){return (const char*)m+o;}
'''+ '\n'.join(parts)+r'''
__global__ void consumer(uint32_t*out,const uint32_t*h,const uint32_t*r,const int32_t*t){unsigned i=blockIdx.x*256+threadIdx.x;if(i<20480)out[i]=h[i];if(i<5120)out[20480+i]=r[i];if(i<2)out[25600+i]=(uint32_t)t[i];}
int main(){
 CK(cudaSetDevice(0));unsigned char*wb;CK(cudaMalloc(&wb,8*2720+16));
 float *ha,*hb,*ra,*rb;int32_t *ta,*tb;uint32_t*seen;
 CK(cudaMalloc(&ha,81920+64));CK(cudaMalloc(&hb,81920+64));CK(cudaMalloc(&ra,20480+64));CK(cudaMalloc(&rb,20480+64));CK(cudaMalloc(&ta,64));CK(cudaMalloc(&tb,64));CK(cudaMalloc(&seen,25602*4));
 cudaStream_t stream;CK(cudaStreamCreate(&stream));cudaGraph_t graph;cudaGraphExec_t exec;
 CK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));consumer<<<80,256,0,stream>>>(seen,(uint32_t*)hb,(uint32_t*)rb,tb);CK(cudaStreamEndCapture(stream,&graph));CK(cudaGraphInstantiate(&exec,graph,0));
 unsigned cases=0, controls=0;uint16_t scales[]={0x3c00,1,0x8000,0x7c00,0xfc00,0x7e13,0x7d05,0x7bff};
 std::vector<unsigned char>w(8*2720+16),back(w.size());std::vector<unsigned char>a(81920+64),b(a.size()),ar(20480+64),br(ar.size()),at(64),bt(64);std::vector<uint32_t>obs(25602);
 for(unsigned rows=1;rows<=2;rows++)for(unsigned off: {0u,2u})for(unsigned pat=0;pat<8;pat++)for(unsigned ids=0;ids<4;ids++)for(unsigned replay=0;replay<3;replay++){
  for(unsigned i=0;i<w.size();i++)w[i]=(i*37+pat*13+replay*19)&255;
  for(unsigned row=0;row<8;row++)for(unsigned g=0;g<80;g++){uint16_t h=scales[(pat+g)%8];memcpy(w.data()+off+row*2720+g*34,&h,2);}
  CK(cudaMemcpy(wb,w.data(),w.size(),cudaMemcpyHostToDevice));
  int32_t tok[2]={(int32_t)((ids==0?0:ids==1?7:ids+replay)%8),(int32_t)((ids==2?ids+replay:7-replay)%8)};if(ids==2)tok[1]=tok[0];
  CK(cudaMemset(ha,0xa5,a.size()));CK(cudaMemset(hb,0xa5,b.size()));CK(cudaMemset(ra,0xa5,ar.size()));CK(cudaMemset(rb,0xa5,br.size()));CK(cudaMemset(ta,0xa5,64));CK(cudaMemset(tb,0xa5,64));
  CK(cudaMemcpy(ta,tok,rows*4,cudaMemcpyHostToDevice));
  glm_embed_tokens_q8_0_kernel<<<rows*10,256>>>(ra,ta,wb+off,rows,2560);repeat_hc_rows_kernel<<<rows*40,256>>>(ha,ra,rows,2560,4);
  ds4_gpu_tensor H{hb,81920+64,0},R{rb,20480+64,0},T{tb,64,0};
  assert(ds4_gpu_qwen4exp_embed_short_tensor(&H,&R,&T,wb,w.size(),off,8,8,rows,tok[0],tok[1]));
  if(replay)CK(cudaGraphLaunch(exec,stream));CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(a.data(),ha,a.size(),cudaMemcpyDeviceToHost));CK(cudaMemcpy(b.data(),hb,b.size(),cudaMemcpyDeviceToHost));assert(a==b);
  CK(cudaMemcpy(ar.data(),ra,ar.size(),cudaMemcpyDeviceToHost));CK(cudaMemcpy(br.data(),rb,br.size(),cudaMemcpyDeviceToHost));assert(ar==br);
  CK(cudaMemcpy(at.data(),ta,64,cudaMemcpyDeviceToHost));CK(cudaMemcpy(bt.data(),tb,64,cudaMemcpyDeviceToHost));assert(at==bt);
  CK(cudaMemcpy(back.data(),wb,w.size(),cudaMemcpyDeviceToHost));assert(w==back);
  if(replay){CK(cudaMemcpy(obs.data(),seen,obs.size()*4,cudaMemcpyDeviceToHost));assert(!memcmp(obs.data(),b.data(),81920));assert(!memcmp(obs.data()+20480,br.data(),20480));assert(!memcmp(obs.data()+25600,bt.data(),8));}
  if(cases==0){
   auto fail=[&](ds4_gpu_tensor h,ds4_gpu_tensor r,ds4_gpu_tensor t,uint32_t nr,int32_t id,uint64_t size,uint64_t offset,uint32_t ty){assert(!ds4_gpu_qwen4exp_embed_short_tensor(&h,&r,&t,wb,size,offset,ty,8,nr,id,7));controls++;};
   auto q=H;q.bytes=1;fail(q,R,T,1,0,w.size(),0,8);q=R;q.bytes=1;fail(H,q,T,1,0,w.size(),0,8);q=T;q.bytes=1;fail(H,R,q,1,0,w.size(),0,8);
   q=H;q.device_id=1;fail(q,R,T,1,0,w.size(),0,8);q=H;q.ptr=(char*)H.ptr+1;fail(q,R,T,1,0,w.size(),0,8);
   q=R;q.ptr=H.ptr;fail(H,q,T,1,0,w.size(),0,8);q=T;q.ptr=R.ptr;fail(H,R,q,1,0,w.size(),0,8);q=T;q.ptr=H.ptr;fail(H,R,q,1,0,w.size(),0,8);
   q=H;q.ptr=wb;fail(q,R,T,1,0,w.size(),0,8);q=R;q.ptr=wb;fail(H,q,T,1,0,w.size(),0,8);q=T;q.ptr=wb;fail(H,R,q,1,0,w.size(),0,8);
   fail(H,R,T,0,0,w.size(),0,8);fail(H,R,T,3,0,w.size(),0,8);fail(H,R,T,1,-1,w.size(),0,8);fail(H,R,T,1,8,w.size(),0,8);fail(H,R,T,1,0,4,0,8);fail(H,R,T,1,0,w.size(),UINT64_MAX,8);fail(H,R,T,1,0,w.size(),0,1);
   CK(cudaDeviceSynchronize());CK(cudaMemcpy(b.data(),hb,b.size(),cudaMemcpyDeviceToHost));assert(a==b);
  }
  cases++;
 }
 printf("PASS short embedding: %u eager, %u eager-to-captured-consumer executions, %u API refusals; full outputs/token publication/guards/weights\n",cases,cases*2/3,controls);
 CK(cudaGraphExecDestroy(exec));CK(cudaGraphDestroy(graph));CK(cudaStreamDestroy(stream));
}
'''
with tempfile.TemporaryDirectory(prefix='short-embed-') as d:
 p=Path(d)/'test.cu';p.write_text(code);exe=Path(d)/'test'
 subprocess.run([os.environ.get('NVCC','nvcc'),'-O3','--use_fast_math','-arch=sm_86',str(p),'-o',str(exe)],check=True)
 subprocess.run([str(exe)],check=True)
