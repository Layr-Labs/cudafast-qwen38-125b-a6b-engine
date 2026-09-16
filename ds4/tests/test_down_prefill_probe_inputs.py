"""Execute private-input generators and diagnostic formatters as host C++.

Checks Q5_1 headers, signed sums, zero/subnormal patterns, full/tail coverage,
allocation guards and the real identity formatter's capacity. Not GPU timing.
"""
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
s = (root/'ds4_cuda_down_prefill_tune.cuh').read_text()
init = s[s.index('__global__ static void'):s.index('extern "C" const char *')]
cu = (root/'ds4_cuda.cu').read_text()
at = cu.index('extern "C" const char *ds4_gpu_hw_limits(void)')
hw = cu[at:cu.index('\n}',at)+2]
resident = (root.parent/'harness/protocol-adapter/ds4_shim/ds4_resident.c').read_text()
at = resident.index('    char ident_buf[')
fmt = resident[at:resident.index('\n\n',at)]
code = r'''
#include <cassert>
#include <cmath>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>
#include <string>
#define __global__
struct Index{unsigned x;};static Index blockIdx,threadIdx,gridDim,blockDim;
using cudaDeviceAttr=int;
constexpr int cudaSuccess=0,cudaDevAttrMaxSharedMemoryPerBlockOptin=0,
 cudaDevAttrMultiProcessorCount=1,cudaDevAttrComputeCapabilityMajor=2,
 cudaDevAttrComputeCapabilityMinor=3,cudaDevAttrIntegrated=4,cudaDevAttrCooperativeLaunch=5;
static int cudaGetDevice(int *p){*p=0;return 0;}
static int cudaGetLastError(){return 0;}
static int cudaDeviceGetAttribute(int *p,int,int){*p=INT_MAX;return 0;}
static std::string pt(319,'p'),kl(383,'k');
static const char *ds4_gpu_qwen4exp_down_prefill_tune(){return pt.c_str();}
static const char *ds4_gpu_qwen4exp_kernel_limits(){return kl.c_str();}
static void log_line(const char *,...){ }
'''+init+hw+r'''
static const char *ds4s_hw_limits(){return ds4_gpu_hw_limits();}
int main(){
 unsigned checks=0;
 for(unsigned n:{0u,1u,31u,257u,81920u})for(unsigned threads:{1u,32u,127u}){
  gridDim.x=7;blockDim.x=threads;
  std::vector<uint32_t>w((size_t)n*6+32,0xdeadbeef);
  for(blockIdx.x=0;blockIdx.x<gridDim.x;blockIdx.x++)for(threadIdx.x=0;threadIdx.x<blockDim.x;threadIdx.x++)
   qwen4exp_down_probe_weights(w.data()+16,n);
  for(unsigned i=0;i<16;i++)assert(w[i]==0xdeadbeef&&w[w.size()-1-i]==0xdeadbeef);
  for(unsigned i=0;i<n;i++){
   assert((w[16+i*6]&0xffffu)==0x1400u);
   assert((w[16+i*6]>>16)==(i&1u?0x1000u:0x9000u));
   for(unsigned j=1;j<6;j++)assert(w[16+i*6+j]!=0xdeadbeef);
  }
  checks++;
  for(unsigned pattern=0;pattern<6;pattern++){
   std::vector<int8_t>q((size_t)n*32+32,99);
   std::vector<float>scale(n+32,123.0f);std::vector<int32_t>sum(n+32,123);
   for(blockIdx.x=0;blockIdx.x<gridDim.x;blockIdx.x++)for(threadIdx.x=0;threadIdx.x<blockDim.x;threadIdx.x++)
    qwen4exp_down_probe_activation(q.data()+16,scale.data()+16,sum.data()+16,n,pattern);
   for(unsigned i=0;i<16;i++){
    assert(q[i]==99&&q[q.size()-1-i]==99);
    assert(scale[i]==123&&scale[scale.size()-1-i]==123&&sum[i]==123&&sum[sum.size()-1-i]==123);
   }
   for(unsigned i=0;i<n;i++){
    int total=0;for(unsigned j=0;j<32;j++){int v=q[16+i*32+j];assert(v>=-127);total+=v;if(pattern==1)assert(v==0);}
    assert(total==sum[16+i]&&std::isfinite(scale[16+i])&&scale[16+i]>0);
    if(pattern==2)assert(scale[16+i]<1e-19f&&std::isnormal(scale[16+i]));
    if(pattern==3)assert(scale[16+i]==65536.0f);
    if(pattern==4)assert(!std::isnormal(scale[16+i]));
   }
   checks++;
  }
 }
 std::string prefix(255,'i');const char *ident=prefix.c_str();
'''+fmt+r'''
 assert(std::string(limits).find(pt)!=std::string::npos&&std::string(limits).find(kl)!=std::string::npos);
 assert(std::string(ident)==prefix+" "+limits);
 std::cout<<checks<<" private-input cases, actual hardware/identity formatter "<<strlen(limits)<<"/"<<strlen(ident)<<" bytes PASS\n";
}
'''
with tempfile.TemporaryDirectory(prefix='prefill-inputs-') as d:
    d=Path(d);p=d/'test.cpp';e=d/'test';p.write_text(code)
    subprocess.run(['c++','-std=c++17','-O2','-fsanitize=undefined',
                    '-fno-sanitize-recover=all',str(p),'-o',str(e)],check=True)
    subprocess.run([str(e)],check=True)
