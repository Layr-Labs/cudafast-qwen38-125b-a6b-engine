#!/usr/bin/env python3
"""Execute the actual convolution allocator and hardware identity formatter."""
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
s = (root/'ds4_cuda_qwen4exp.cu').read_text()
at = s.index('static void *g_qwen4exp_conv_scratch[16]')
allocator = s[at:s.index('\n/* Snapshot slots', at)]
s = (root/'ds4_cuda.cu').read_text()
at = s.index('extern "C" const char *ds4_gpu_hw_limits(void)')
formatter = s[at:s.index('\n}',at)+2]
s = (root.parent/'harness/protocol-adapter/ds4_shim/ds4_resident.c').read_text()
at = s.index('    char ident_buf[')
resident = s[at:s.index('\n\n',at)]
code = r'''
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <string>
#include <vector>
static bool fail_alloc;static std::vector<int> events;static unsigned calls;
static bool glm53_cuda_mul_u64(uint64_t a,uint64_t b,uint64_t*out){if(a>UINT64_MAX/b)return false;*out=a*b;return true;}
static int cudaMalloc(void**p,size_t n){events.push_back(1);calls++;if(fail_alloc)return 1;*p=malloc(n?n:1);return *p?0:1;}
static int cudaFree(void*p){assert(events.size()>=2&&events[events.size()-1]==2);events.push_back(3);free(p);return 0;}
static bool cuda_ok(int rc,const char*){return rc==0;}
static void ds4_gpu_decode_graphs_invalidate(){events.push_back(2);}
'''+allocator+r'''
using cudaDeviceAttr=int;
constexpr int cudaSuccess=0,cudaDevAttrMaxSharedMemoryPerBlockOptin=0,
 cudaDevAttrMultiProcessorCount=1,cudaDevAttrComputeCapabilityMajor=2,
 cudaDevAttrComputeCapabilityMinor=3,cudaDevAttrIntegrated=4,cudaDevAttrCooperativeLaunch=5;
static int cudaGetDevice(int*p){*p=0;return 0;}static int cudaGetLastError(){return 0;}
static int cudaDeviceGetAttribute(int*p,int,int){*p=INT_MAX;return 0;}
static void log_line(const char*,...){ }
constexpr unsigned CUDA_DECODE_GRAPH_LAYERS=64,CUDA_DECODE_GRAPH_VARIANTS=8;
struct cuda_decode_graph_entry{struct{uint32_t _pad;}key;int state;uint64_t hits;};
static cuda_decode_graph_entry g_decode_graphs[64][4][8];
static std::string kl(230,'k');
static const char* ds4_gpu_qwen4exp_kernel_limits(){return kl.c_str();}
'''+formatter+r'''
static const char *ds4s_hw_limits(){return ds4_gpu_hw_limits();}
int main(){unsigned checks=0;
 for(int t=0;t<16;t++){
  events.clear();void*p=qwen4exp_conv_scratch(t,100);assert(p&&events==std::vector<int>{1});checks++;
  events.clear();assert(qwen4exp_conv_scratch(t,99)==p&&events.empty());checks++;
  fail_alloc=true;assert(!qwen4exp_conv_scratch(t,101));assert(g_qwen4exp_conv_scratch[t]==p&&g_qwen4exp_conv_bytes[t]==400);checks++;
  fail_alloc=false;events.clear();void*q=qwen4exp_conv_scratch(t,102);assert(q&&q!=p&&(events==std::vector<int>{1,2,3}));checks++;
  free(q);g_qwen4exp_conv_scratch[t]=nullptr;
 }
 unsigned before=calls;assert(!qwen4exp_conv_scratch(-1,10));assert(!qwen4exp_conv_scratch(16,10));assert(!qwen4exp_conv_scratch(0,UINT64_MAX));assert(calls==before);checks+=3;
 for(unsigned il=56;il<64;il++)for(unsigned v=0;v<8;v++)g_decode_graphs[il][3][v]={{0x51575043u},v%2?3:2,5};
 // Wrong row, island and tag are excluded from diagnostics.
 g_decode_graphs[0][3][0]={{0x51575043u},2,900};g_decode_graphs[56][0][0]={{0x51575043u},2,900};
 g_decode_graphs[56][3][0]={{0},2,900};
 const char*hw=ds4_gpu_hw_limits();assert(strstr(hw,"pfGraph[c=31 r=315 d=32]"));assert(strstr(hw,kl.c_str()));
 std::string prefix(255,'i');const char *ident=prefix.c_str();
'''+resident+r'''
 assert(std::string(ident)==prefix+" "+limits);checks++;
 printf("Actual prefill scratch/formatter: %u checks PASS; identity %zu/%zu bytes\n",checks,strlen(hw),strlen(ident));
}
'''
with tempfile.TemporaryDirectory(prefix='prefill-scratch-') as d:
    d = Path(d)
    (d/'test.cpp').write_text(code)
    subprocess.run(['c++', '-std=c++17', '-O1', '-g', '-fsanitize=undefined',
                    '-fno-sanitize-recover=all', str(d/'test.cpp'), '-o', str(d/'test')], check=True)
    subprocess.run([str(d/'test')], check=True)
