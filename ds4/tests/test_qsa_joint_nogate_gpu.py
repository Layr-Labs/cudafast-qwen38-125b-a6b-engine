"""Actual precise-CUDA joint Part2/Part4 parity plus gate-consumer equivalence.
Synthetic GPU operators only, not the model or complete backend API.
"""
from pathlib import Path
import os,subprocess,tempfile
s=(Path(__file__).resolve().parents[1]/'ds4_cuda_qwen4exp.cu').read_text()
def body(marker):
 a=s.index(marker);b=s.index('{',a);e=b+1;depth=1
 while depth:depth+=(s[e]=='{')-(s[e]=='}');e+=1
 return s[a:e]
parts=[body(x) for x in ['__device__ __forceinline__ static float qwen4exp_blk_sum(',
 '__device__ __forceinline__ static float qwen4exp_q8_ftz(',
 '__device__ __forceinline__ static float qwen4exp_q8_rcp_approx(',
 'template<int Part, bool KVFirst=false>\n__global__ static void qwen4exp_qsa_prep_joint_kernel(',
 '__global__ static void qwen4exp_qsa_output_gate_quant_kernel(',
 '__global__ static void qwen4exp_qsa_output_gate_doubled_quant_kernel(']]
macro=next(x for x in s.splitlines() if x.startswith('#define QWEN4EXP_Q8_RCP127 '))
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
'''+macro+'\n'+'\n'.join(parts)+r'''
int main(){
 std::mt19937 rng(610017);unsigned eager=0,replay=0,quant=0;cudaStream_t st;CK(cudaStreamCreate(&st));
 for(unsigned dim:{64u,256u})for(unsigned rows:{1u,2u})for(unsigned mode=0;mode<8;mode++)for(bool optional:{false,true}){
  unsigned qh=dim==256?24:3,kh=dim==256?2:1,rot=64,cap=7,pos=0;
  size_t qe=(size_t)rows*qh*dim,ke=(size_t)rows*kh*dim,ce=(size_t)cap*kh*dim;
  size_t ni[6]={2*qe,ke,ke,dim,dim,rot/2},no[5]={qe,qe,ce,ce,ke};
  float *in[6],*out[3][5];std::vector<float>host[6];uint32_t*dp;
  for(unsigned i=0;i<6;i++){host[i].resize(ni[i]);for(size_t j=0;j<ni[i];j++)host[i][j]=(int(rng()%2001)-1000)*.001f;if(i==5)for(auto&v:host[i])v*=.01f;CK(cudaMalloc(&in[i],ni[i]*4));}
  for(unsigned m=0;m<3;m++)for(unsigned j=0;j<5;j++)CK(cudaMalloc(&out[m][j],(no[j]+8)*4));CK(cudaMalloc(&dp,4));
  int8_t*quantized[2];float*scale[2];for(unsigned m=0;m<2;m++){CK(cudaMalloc(&quantized[m],qe+32));CK(cudaMalloc(&scale[m],(qe/32+8)*4));}
  auto publish=[&](unsigned phase){
   for(size_t j=0;j<ni[0];j++){float v=(int(rng()%2001)-1000)*.001f;if(mode==1)v=-0.f;if(mode==2)v=ldexpf(1.f,-140);if(mode==3&&j%137==0)v=INFINITY;if(mode==4&&j%137==0)v=-INFINITY;if(mode==5&&j%137==0)v=NAN;if(mode==6)v=j&1?80.f:-80.f;host[0][j]=v;}
   pos=phase==0?0:phase==1?cap-1:phase==2?cap:UINT32_MAX;
   for(unsigned i=0;i<6;i++)CK(cudaMemcpy(in[i],host[i].data(),ni[i]*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(dp,&pos,4,cudaMemcpyHostToDevice));
  };
  auto launch=[&](){
   for(unsigned m=0;m<3;m++)for(unsigned j=0;j<5;j++)CK(cudaMemsetAsync(out[m][j],0xa5,(no[j]+8)*4,st));
   qwen4exp_qsa_prep_joint_kernel<2><<<dim3(qh+kh,rows),dim,dim*4,st>>>(in[0],in[1],in[2],in[3],in[4],in[5],out[0][0]+4,out[0][1]+4,out[0][2]+4,out[0][3]+4,optional?out[0][4]+4:nullptr,rows,qh,kh,dim,rot,0,cap,1e-6f,1.f,1.f,dp);
   qwen4exp_qsa_prep_joint_kernel<4><<<dim3(qh+kh,rows),dim,dim*4,st>>>(in[0],in[1],in[2],in[3],in[4],in[5],out[1][0]+4,mode&1?nullptr:out[1][1]+4,out[1][2]+4,out[1][3]+4,optional?out[1][4]+4:nullptr,rows,qh,kh,dim,rot,0,cap,1e-6f,1.f,1.f,dp);
   qwen4exp_qsa_prep_joint_kernel<3><<<dim3(qh,rows),dim,dim*4,st>>>(in[0],nullptr,nullptr,in[3],nullptr,in[5],out[2][0]+4,nullptr,nullptr,nullptr,nullptr,rows,qh,0,dim,rot,0,0,1e-6f,1.f,0.f,dp);
   qwen4exp_qsa_prep_joint_kernel<1><<<dim3(kh,rows),dim,dim*4,st>>>(nullptr,in[1],in[2],nullptr,in[4],in[5],nullptr,nullptr,out[2][2]+4,out[2][3]+4,optional?out[2][4]+4:nullptr,rows,0,kh,dim,rot,0,cap,1e-6f,0.f,1.f,dp);
   if(dim==256){
    for(unsigned m=0;m<2;m++){CK(cudaMemsetAsync(quantized[m],0xa5,qe+32,st));CK(cudaMemsetAsync(scale[m],0xa5,(qe/32+8)*4,st));}
    qwen4exp_qsa_output_gate_quant_kernel<<<qe/256,256,0,st>>>(quantized[0],scale[0],out[0][1]+4,out[0][0]+4,qe);
    qwen4exp_qsa_output_gate_doubled_quant_kernel<<<qe/256,256,0,st>>>(quantized[1],scale[1],in[0],out[1][0]+4,qe);
   }CK(cudaGetLastError());
  };
  auto compare=[&](){
   for(unsigned variant=1;variant<=2;variant++)for(unsigned j=0;j<5;j++){
    std::vector<uint32_t>a(no[j]+8),b(no[j]+8);CK(cudaMemcpy(a.data(),out[0][j],a.size()*4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(b.data(),out[variant][j],b.size()*4,cudaMemcpyDeviceToHost));
    if(j!=1){if(a!=b){fprintf(stderr,"prep dim%u rows%u mode%u pos%u output%u\n",dim,rows,mode,pos,j);exit(2);}}
    else {for(auto v:b)if(v!=0xa5a5a5a5u)exit(3);for(size_t k=0;k<qe;k++){size_t tok=k/(qh*dim),head=k/dim%qh,d=k%dim;uint32_t want;memcpy(&want,&host[0][tok*2*qh*dim+head*2*dim+d+dim],4);if(a[k+4]!=want)exit(4);}}
    for(size_t k=0;k<a.size();k++)if(k<4||k>=no[j]+4)if(a[k]!=0xa5a5a5a5u||b[k]!=0xa5a5a5a5u)exit(5);
   }
   if(dim==256){std::vector<unsigned char>a(qe+32),b(qe+32);std::vector<uint32_t>sa(qe/32+8),sb(qe/32+8);CK(cudaMemcpy(a.data(),quantized[0],a.size(),cudaMemcpyDeviceToHost));CK(cudaMemcpy(b.data(),quantized[1],b.size(),cudaMemcpyDeviceToHost));CK(cudaMemcpy(sa.data(),scale[0],sa.size()*4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(sb.data(),scale[1],sb.size()*4,cudaMemcpyDeviceToHost));if(a!=b||sa!=sb)exit(6);for(size_t i=qe;i<a.size();i++)if(a[i]!=0xa5)exit(7);for(size_t i=qe/32;i<sa.size();i++)if(sa[i]!=0xa5a5a5a5u)exit(7);quant++;}
  };
  publish(0);launch();CK(cudaStreamSynchronize(st));compare();eager++;
  cudaGraph_t graph;cudaGraphExec_t ex;CK(cudaStreamBeginCapture(st,cudaStreamCaptureModeGlobal));launch();CK(cudaStreamEndCapture(st,&graph));CK(cudaGraphInstantiate(&ex,graph,nullptr,nullptr,0));
  for(unsigned p=1;p<=3;p++){publish(p);CK(cudaGraphLaunch(ex,st));CK(cudaStreamSynchronize(st));compare();replay++;}
  CK(cudaGraphExecDestroy(ex));CK(cudaGraphDestroy(graph));
  for(unsigned i=0;i<6;i++){std::vector<float>after(ni[i]);CK(cudaMemcpy(after.data(),in[i],ni[i]*4,cudaMemcpyDeviceToHost));if(memcmp(after.data(),host[i].data(),ni[i]*4))exit(8);CK(cudaFree(in[i]));}
  for(unsigned m=0;m<3;m++)for(unsigned j=0;j<5;j++)CK(cudaFree(out[m][j]));for(unsigned m=0;m<2;m++){CK(cudaFree(quantized[m]));CK(cudaFree(scale[m]));}CK(cudaFree(dp));
 }
 CK(cudaStreamDestroy(st));printf("PASS joint no-gate: %u eager + %u changed-input/position graph replays; %u downstream quantized comparisons; joint/standalone no-gate exact, gate untouched, full buffers/canaries/input immutability\n",eager,replay,quant);
}
'''
with tempfile.TemporaryDirectory(prefix='qsa-joint-nogate-') as t:
 p=Path(t);(p/'test.cu').write_text(source)
 subprocess.run([os.environ.get('NVCC','nvcc'),'-O3','-ftz=false','-prec-div=true','-prec-sqrt=true','-arch='+os.environ.get('CUDA_TEST_ARCH','sm_86'),str(p/'test.cu'),'-o',str(p/'test')],check=True)
 subprocess.run([str(p/'test')],check=True)
