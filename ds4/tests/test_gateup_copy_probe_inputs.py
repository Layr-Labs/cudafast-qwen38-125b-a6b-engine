"""Execute private-input generators and diagnostic formatters as host C++.

Checks Q5_1 headers, signed sums, zero/subnormal patterns, full/tail coverage,
allocation guards and the real identity formatter's capacity. Not GPU timing.
"""
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
s = (root/'ds4_cuda_gateup_copy_tune.cuh').read_text()
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
static const char *ds4_gpu_qwen4exp_gateup_copy_tune(){return pt.c_str();}
static const char *ds4_gpu_qwen4exp_kernel_limits(){return kl.c_str();}
static void log_line(const char *,...){ }
'''+init+hw+r'''
static const char *ds4s_hw_limits(){return ds4_gpu_hw_limits();}
int main(){
 unsigned checks=0;
 for(unsigned n:{0u,1u,31u,257u,81920u})for(unsigned threads:{1u,32u,127u}){
  gridDim.x=7;blockDim.x=threads;
  std::vector<uint32_t>w((size_t)n*36+32,0xdeadbeef);
  for(blockIdx.x=0;blockIdx.x<gridDim.x;blockIdx.x++)for(threadIdx.x=0;threadIdx.x<blockDim.x;threadIdx.x++)
   qwen4exp_gateup_probe_weights(w.data()+16,n);
  for(unsigned i=0;i<16;i++)assert(w[i]==0xdeadbeef&&w[w.size()-1-i]==0xdeadbeef);
  for(unsigned i=0;i<n;i++){
   assert((w[16+i*36]&0xffffu)==0x1400u);
   assert((w[16+i*36]>>16)==0x1000u);
   for(unsigned j=1;j<36;j++)assert(w[16+i*36+j]!=0xdeadbeef);
  }
  checks++;
  std::vector<uint32_t>pressure(n+32,0xdeadbeef);
  for(blockIdx.x=0;blockIdx.x<gridDim.x;blockIdx.x++)for(threadIdx.x=0;threadIdx.x<blockDim.x;threadIdx.x++)
   qwen4exp_gateup_cache_pressure(pressure.data()+16,n);
  for(unsigned i=0;i<16;i++)assert(pressure[i]==0xdeadbeef&&pressure[pressure.size()-1-i]==0xdeadbeef);
  for(unsigned i=0;i<n;i++)assert(pressure[16+i]==uint32_t(0xdeadbeefu*1664525u+1013904223u));
  checks++;
 }
 std::string prefix(255,'i');const char *ident=prefix.c_str();
'''+fmt+r'''
 assert(std::string(limits).find(pt)!=std::string::npos&&std::string(limits).find(kl)!=std::string::npos);
 assert(std::string(ident)==prefix+" "+limits);
 std::cout<<checks<<" private-input cases, actual hardware/identity formatter "<<strlen(limits)<<"/"<<strlen(ident)<<" bytes PASS\n";
}
'''
with tempfile.TemporaryDirectory(prefix='gateup-copy-inputs-') as d:
    d=Path(d);p=d/'test.cpp';e=d/'test';p.write_text(code)
    subprocess.run(['c++','-std=c++17','-O2','-fsanitize=undefined',
                    '-fno-sanitize-recover=all',str(p),'-o',str(e)],check=True)
    subprocess.run([str(e)],check=True)
