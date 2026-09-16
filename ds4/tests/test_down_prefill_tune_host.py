"""Run the actual startup tuner with CUDA stubs: policy, order, errors, cleanup.

The stubs are not a GPU speed or arithmetic test. The separate fiber test
executes the real three down tile bodies; native correctness remains required.
"""
from pathlib import Path
import re
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
source = (root / 'ds4_cuda_down_prefill_tune.cuh').read_text()
source = source[source.index('extern "C" const char *'):]
source = re.sub(r'<<<.*?>>>', '', source, flags=re.S)
for name in ('getenv', 'malloc', 'free'):
    source = source.replace(name + '(', 'probe_' + name + '(')
stubs = r'''
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <string>
#include <vector>
using std::isfinite;
constexpr unsigned DS4_QWEN4EXP_TY_q5_1=7;
static int qwen4exp_down_prefill_choice=0,qwen4exp_down_prefill_device=-1;
struct GPU{int device_id;};static GPU g_gpu[1]={{0}};static int g_n_gpus=1;
using cudaStream_t=int;using cudaStreamCaptureStatus=int;
constexpr int cudaSuccess=0,cudaStreamCaptureStatusNone=0,cudaDevAttrL2CacheSize=1;
constexpr int cudaMemcpyHostToDevice=1,cudaMemcpyDeviceToHost=2;
struct Event{double us=0;};using cudaEvent_t=Event*;
static std::string scenario,fail_name;
static unsigned fail_n=1,allocations,frees,host_allocs,host_frees,events,destroyed;
static unsigned elapsed_count,api_calls,weight_inits,activation_inits;
static std::map<std::string,unsigned> counts;
static std::vector<int> order;
static double gpu_us=0;
static bool in_span=false;
static unsigned pattern;
static bool fail(const char *name){api_calls++;return ++counts[name]==fail_n && fail_name==name;}
static const char *probe_getenv(const char *name){
    return (scenario=="disabled"&&!strcmp(name,"DS4_QWEN4EXP_NO_DOWN_PREFILL_TUNE")) ||
      (scenario=="nopipe"&&!strcmp(name,"DS4_QWEN4EXP_NO_DOWN_RAW_PIPE")) ||
      (scenario=="nodq"&&!strcmp(name,"DS4_QWEN4EXP_NO_DOWN_DQ")) ||
      (scenario=="nowide"&&!strcmp(name,"DS4_QWEN4EXP_NO_Q51_WIDE_LOAD")) ? "1" : nullptr;
}
static void *probe_malloc(size_t n){if(fail("hostalloc"))return nullptr;auto p=malloc(n);assert(p);host_allocs++;return p;}
static void probe_free(void *p){assert(p);host_frees++;free(p);}
static int cuda_decode_stream(){return 0;}
static int cudaGetDevice(int *p){if(fail("device"))return 1;*p=scenario=="wrongdevice"?1:0;return 0;}
static int cudaStreamIsCapturing(int,int *p){if(fail("capture"))return 1;*p=scenario=="capturing"?1:0;return 0;}
static int cudaDeviceGetAttribute(int *p,int attr,int dev){assert(attr==1&&dev==0);if(fail("attribute"))return 1;*p=scenario=="zerol2"?0:scenario=="bigl2"?64*1024*1024:24*1024*1024;return 0;}
static int cudaDeviceSynchronize(){return fail("sync")?1:0;}
static int cudaStreamSynchronize(int){return fail("streamsync")?1:0;}
static int cudaMalloc(void **p,size_t n){if(fail("malloc"))return 1;*p=malloc(n);assert(*p);allocations++;return 0;}
static int cudaFree(void *p){assert(p);free(p);frees++;return 0;}
static int cudaEventCreate(cudaEvent_t *p){if(fail("event"))return 1;*p=new Event;events++;return 0;}
static int cudaEventDestroy(cudaEvent_t p){assert(p);delete p;destroyed++;return 0;}
static int cudaGetLastError(){return fail("last")?1:0;}
static int cudaMemcpy(void *dst,const void *src,size_t n,int kind){if(fail(kind==1?"h2d":"d2h"))return 1;memcpy(dst,src,n);return 0;}
static int cudaMemsetAsync(void *p,int v,size_t n,int){if(fail("memset"))return 1;memset(p,v,n);return 0;}
static int cudaEventRecord(cudaEvent_t p,int){if(fail("record"))return 1;p->us=gpu_us;in_span=!in_span;return 0;}
static int cudaEventSynchronize(cudaEvent_t){return fail("wait")?1:0;}
static int cudaEventElapsedTime(float *ms,cudaEvent_t a,cudaEvent_t b){
    if(fail("elapsed"))return 1;
    *ms=scenario=="zero"?0:scenario=="nan"?NAN:scenario=="huge"?1001:(float)((b->us-a->us)/1000.0);
    elapsed_count++;return 0;
}
static void qwen4exp_down_probe_weights(uint32_t *,size_t n){assert(!in_span&&n==64u*2560u*20u);weight_inits++;}
static void qwen4exp_down_probe_activation(int8_t *,float *,int32_t *,unsigned n,unsigned p){assert(!in_span&&n==4096u*20u);pattern=p;activation_inits++;}
static void down(int tile,float *out,const char *,const int8_t *,const float *,const int32_t *,
 const int32_t *pairs,const int32_t *cnt,const int32_t *offsets,const int32_t *active,
 uint64_t eb,uint64_t rb,unsigned type,unsigned groups,unsigned od,unsigned dq){
    assert(eb==1228800&&rb==480&&type==7&&groups==20&&od==2560&&dq==1);
    assert(offsets==cnt+64&&active==cnt+128&&pairs==cnt+193&&active[0]==64);
    unsigned np=0;const unsigned mixed[]={1,3,9,17,33,63};
    for(unsigned e=0;e<64;e++){
        assert(active[e+1]==63-(int)e&&offsets[e]==(int)np);
        assert(cnt[e]==(int)(pattern==0?1:pattern==1?20:pattern==2?32:pattern==3?mixed[e%6]:64));
        np+=cnt[e];
    }
    for(unsigned i=0;i<np;i++)assert(pairs[i]==(int)(np-1-i));
    out[0]=(float)np;
    if(scenario=="bad64"&&tile==64)out[0]+=1;
    if(scenario=="bad32"&&tile==32)out[0]+=1;
    if(scenario=="late_mismatch"&&tile==32&&pattern==5)out[4096u*2560u+63u]+=1e16f;
    if(scenario=="nonfinite")out[0]=NAN;
    double cost=tile==64?80:70;
    if(scenario=="only64"&&tile==32)cost=105;
    if(scenario=="only32"&&tile==64)cost=105;
    if(scenario=="marginal")cost=98.5;
    if(scenario=="slower")cost=110;
    if(scenario=="one_slow"&&pattern==2)cost=101;
    if(scenario=="noisy")cost=elapsed_count/2%7==6?80:96;
    if(scenario=="five_wins")cost=elapsed_count/2%7<5?90:101;
    gpu_us+=tile?cost:100;order.push_back(tile);
}
template<int T,bool W,class...A>static void qwen4exp_moe_down_mma_kernel(A...a){assert(T==7&&W);down(0,a...);}
template<int T,bool W,class...A>static void qwen4exp_moe_down_raw64_kernel(A...a){assert(T==7&&W);down(64,a...);}
template<int T,bool W,class...A>static void qwen4exp_moe_down_raw_pipe_kernel(A...a){assert(T==7&&W);down(32,a...);}
'''
main = r'''
int main(int argc,char **argv){
 assert(argc==2);scenario=argv[1];
 if(scenario.find(':')!=std::string::npos){auto at=scenario.find(':');fail_name=scenario.substr(0,at);fail_n=std::stoul(scenario.substr(at+1));}
 if(scenario=="multi")g_n_gpus=2;
 const std::string report=ds4_gpu_qwen4exp_down_prefill_tune();
 assert(allocations==frees&&events==destroyed&&host_allocs==host_frees);
 unsigned done=api_calls;assert(report==ds4_gpu_qwen4exp_down_prefill_tune());assert(api_calls==done);
 bool success=scenario=="fast"||scenario=="only64"||scenario=="only32"||scenario=="marginal"||scenario=="slower"||scenario=="one_slow"||scenario=="noisy"||scenario=="five_wins";
 if(success){
  assert(qwen4exp_down_prefill_device==0&&elapsed_count==112&&allocations==6&&events==2&&host_allocs==2);
  assert(weight_inits==1&&activation_inits==14&&order.size()==930);
  assert(qwen4exp_down_prefill_choice==(scenario=="fast"||scenario=="only32"?32:scenario=="only64"?64:0));
  for(unsigned p=0;p<6;p++)for(unsigned v=0;v<3;v++)assert(order[p*3+v]==(v==0?0:v==1?64:32));
  for(unsigned state=0;state<8;state++){
   unsigned start=18+state*114;int tile=state<4?64:32;
   assert(order[start]==0&&order[start+1]==tile);
   for(unsigned pair=0;pair<7;pair++)for(unsigned leg=0;leg<2;leg++)for(unsigned rep=0;rep<8;rep++)
    assert(order[start+2+pair*16+leg*8+rep]==(((pair+leg)&1)?tile:0));
  }
  assert(report.find("pfTune[l2=25165824")!=std::string::npos&&report.back()==']'&&report.size()<320);
 }else{
  assert(qwen4exp_down_prefill_choice==0&&qwen4exp_down_prefill_device==-1);
  assert(report=="pfTune[unmeasured]"||report=="pfTune[disabled]"||report.find("pfTune[failed=")==0);
 }
 std::cout<<scenario<<" "<<report<<" PASS\n";
}
'''
scenarios = ['fast', 'only64', 'only32', 'marginal', 'slower', 'one_slow',
             'noisy', 'five_wins', 'disabled', 'nopipe', 'nodq', 'nowide',
             'multi', 'wrongdevice', 'capturing', 'zerol2', 'bigl2',
             'bad64', 'bad32', 'late_mismatch', 'nonfinite', 'zero', 'nan', 'huge']
for name, times in {'device':[1], 'capture':[1], 'attribute':[1], 'sync':[1],
                    'malloc':range(1,7), 'event':[1,2], 'hostalloc':[1,2],
                    'h2d':[1,6,7,14], 'd2h':[1,2,3,18], 'memset':[1,18],
                    'last':[1,2,20,40,100,941], 'streamsync':[1,8],
                    'record':[1,2,223,224], 'wait':[1,112], 'elapsed':[1,112]}.items():
    scenarios.extend(f'{name}:{n}' for n in times)
with tempfile.TemporaryDirectory(prefix='down-prefill-tune-') as d:
    d = Path(d); p = d/'test.cpp'; e = d/'test'
    p.write_text(stubs + source + main)
    subprocess.run(['c++','-std=c++17','-O2','-fsanitize=undefined',
                    '-fno-sanitize-recover=all',str(p),'-o',str(e)],check=True)
    for scenario in scenarios:
        subprocess.run([str(e),scenario],check=True)
print(f'{len(scenarios)} actual startup tuner scenarios PASS')
