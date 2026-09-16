"""Actual old/new BF16 indexer-K CUDA kernels at production shape.
Main-unit fast math; SM86 arithmetic/graph parity, not SM121 PDL scheduling.
Run with NVCC=/path/to/nvcc; CUDA_TEST_ARCH defaults to sm_86.
"""
from pathlib import Path
import os,re,subprocess,tempfile
s=(Path(__file__).resolve().parents[1]/'ds4_cuda.cu').read_text()
def body(name):
 m=re.search(r'^.*\b'+name+r'\([^;]*?\)\s*\{',s,re.M)
 assert m,name
 a=m.start();b=s.index('{',m.start());e=b+1;d=1
 while d:d+=(s[e]=='{')-(s[e]=='}');e+=1
 return s[a:e]
source=r'''
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#define QWEN4EXP_PDL_SYNC() ((void)0)
#define CK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s line%d\n",cudaGetErrorString(e),__LINE__);exit(1);}}while(0)
'''+ '\n'.join(body(n) for n in ['warp_sum_f32','glm53_matvec_bf16_f32_kernel','qwen4exp_indexer_k_bf16_r2_kernel'])+r'''
int main(){unsigned eager=0,replays=0;std::mt19937 rng(660043);cudaStream_t st;CK(cudaStreamCreate(&st));
 for(unsigned offset:{0u,2u})for(unsigned mode=0;mode<12;mode++){
 const size_t wb=128*2560*2+offset+16,xb=(2*2560+8)*4,ob=(2*128+8)*4;
 unsigned char *dw;float *dx,*a,*b;CK(cudaMalloc(&dw,wb));CK(cudaMalloc(&dx,xb));CK(cudaMalloc(&a,ob));CK(cudaMalloc(&b,ob));
 std::vector<unsigned char>w(wb,0x5a),wa(wb);std::vector<uint32_t>x(xb/4,0x46f11200),xa(x.size()),aa(ob/4),bb(ob/4);
 auto publish=[&](unsigned phase){
  for(unsigned col=0;col<128;col++)for(unsigned k=0;k<2560;k++){
   uint16_t bits=(uint16_t)((rng()%2?0x8000:0)|(0x3d00+(rng()%0x300)));
   if(mode==1)bits=(k&1)?0x8000:0;if(mode==2)bits=(uint16_t)(1+rng()%127);if(mode==3)bits=0x0080;
   if(mode==4)bits=0x7f80;if(mode==5)bits=0xff80;if(mode==6)bits=(uint16_t)(0x7fc0|(rng()%63));if(mode==7)bits=(uint16_t)(0x7f81|(rng()%63));
   memcpy(w.data()+offset+((size_t)col*2560+k)*2,&bits,2);
  }
  for(unsigned row=0;row<2;row++)for(unsigned k=0;k<2560;k++){
   float f=(int(rng()%2001)-1000)*(.0001f*(phase+row+1));uint32_t bits;memcpy(&bits,&f,4);
   if(mode==1)bits=(k+row+phase)&1?0x80000000:0;if(mode==2)bits=1+rng()%0x007fffff;
   if(mode==8)bits=(k&1)?0x7f800000:0xff800000;if(mode==9)bits=0x7fc00000|(rng()&0x3fffff);
   if(mode==10)bits=0x7f800001|(rng()&0x3fffff);if(mode==11)bits=(k&1)?0x00800000:0x80800000;
   x[4+row*2560+k]=bits;
  }
  CK(cudaMemcpy(dw,w.data(),wb,cudaMemcpyHostToDevice));CK(cudaMemcpy(dx,x.data(),xb,cudaMemcpyHostToDevice));
 };
 auto launch=[&](){CK(cudaMemsetAsync(a,0xa5,ob,st));CK(cudaMemsetAsync(b,0xa5,ob,st));
  glm53_matvec_bf16_f32_kernel<<<dim3(16,2),256,0,st>>>(a+4,(const uint16_t*)(dw+offset),dx+4,2560,128);
  qwen4exp_indexer_k_bf16_r2_kernel<<<32,128,0,st>>>(b+4,(const uint16_t*)(dw+offset),dx+4);CK(cudaGetLastError());
 };
 auto check=[&](){CK(cudaStreamSynchronize(st));CK(cudaMemcpy(aa.data(),a,ob,cudaMemcpyDeviceToHost));CK(cudaMemcpy(bb.data(),b,ob,cudaMemcpyDeviceToHost));
  if(aa!=bb){fprintf(stderr,"DIFF offset%u mode%u\n",offset,mode);exit(2);}
  for(unsigned i=0;i<4;i++)if(aa[i]!=0xa5a5a5a5||aa[260+i]!=0xa5a5a5a5)exit(3);
  CK(cudaMemcpy(wa.data(),dw,wb,cudaMemcpyDeviceToHost));CK(cudaMemcpy(xa.data(),dx,xb,cudaMemcpyDeviceToHost));if(wa!=w||xa!=x)exit(4);
 };
 publish(0);launch();check();eager++;
 cudaGraph_t g;cudaGraphExec_t ex;CK(cudaStreamBeginCapture(st,cudaStreamCaptureModeGlobal));launch();CK(cudaStreamEndCapture(st,&g));CK(cudaGraphInstantiate(&ex,g,nullptr,nullptr,0));
 for(unsigned phase=1;phase<=4;phase++){publish(phase);CK(cudaGraphLaunch(ex,st));check();replays++;}
 CK(cudaGraphExecDestroy(ex));CK(cudaGraphDestroy(g));CK(cudaFree(dw));CK(cudaFree(dx));CK(cudaFree(a));CK(cudaFree(b));
 }
 CK(cudaStreamDestroy(st));printf("PASS BF16 indexer-K R2: %u eager + %u changed-input graph replays, bitwise outputs/canaries and complete input/weight immutability\n",eager,replays);
}
'''
with tempfile.TemporaryDirectory(prefix='indexer-k-r2-gpu-') as t:
 p=Path(t);(p/'test.cu').write_text(source)
 subprocess.run([os.environ.get('NVCC','nvcc'),'-O3','--use_fast_math','-arch='+os.environ.get('CUDA_TEST_ARCH','sm_86'),str(p/'test.cu'),'-o',str(p/'test')],check=True)
 subprocess.run([str(p/'test')],check=True)
