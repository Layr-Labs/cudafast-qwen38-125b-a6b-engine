"""Execute the actual startup policy with synthetic CUDA/event stubs.

This verifies control flow and cleanup, not GPU arithmetic or performance.
The separate lane scheduler executes the production down kernels themselves.
"""
from pathlib import Path
import re
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
source = (root / 'ds4_cuda_down_tune.cuh').read_text()
source = source[source.index('extern "C" const char *'):]
source = re.sub(r'<<<.*?>>>', '', source, flags=re.S)
for name in ('getenv', 'malloc', 'free'):
    source = source.replace(name + '(', 'probe_' + name + '(')
stubs = r'''
#include <algorithm>
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
constexpr unsigned QW_DOWN_REUSE_PANELS=6,DS4_QWEN4EXP_TY_q5_1=7;
static int qwen4exp_down_tuned_device=-1;
static bool qwen4exp_down_prefer_reuse=false;
struct GPU{int device_id;};static GPU g_gpu[1]={{0}};static int g_n_gpus=1;
using cudaStream_t=int;using cudaStreamCaptureStatus=int;
constexpr int cudaSuccess=0,cudaStreamCaptureStatusNone=0,cudaDevAttrL2CacheSize=1;
constexpr int cudaMemcpyHostToDevice=1,cudaMemcpyDeviceToHost=2;
struct Event{double us=0;};using cudaEvent_t=Event*;
static std::string scenario,fail_name;
static unsigned fail_n=1,allocations,frees,host_allocs,host_frees,events,destroyed;
static unsigned elapsed_count,api_calls,pressure_calls;
static std::map<std::string,unsigned> counts;
static std::vector<bool> order;
static double gpu_us=0;
static bool in_span=false,next_group=false;
static const int32_t *last_routes=nullptr;
static bool fail(const char *name){api_calls++;return ++counts[name]==fail_n && fail_name==name;}
static const char *probe_getenv(const char *name){
    return (scenario=="disabled"&&!strcmp(name,"DS4_QWEN4EXP_NO_DOWN_AUTOTUNE")) ||
      (scenario=="noreuse"&&!strcmp(name,"DS4_QWEN4EXP_NO_DOWN_PANEL_REUSE")) ||
      (scenario=="nopanel"&&!strcmp(name,"DS4_QWEN4EXP_NO_DOWN_PANEL")) ||
      (scenario=="noasync"&&!strcmp(name,"DS4_QWEN4EXP_NO_DOWN_ASYNC")) ? "1" : nullptr;
}
static void *probe_malloc(size_t n){if(fail("hostalloc"))return nullptr;auto p=malloc(n);assert(p);host_allocs++;return p;}
static void probe_free(void *p){assert(p);host_frees++;free(p);}
static int cuda_decode_stream(){return 0;}
static int cudaGetDevice(int *p){if(fail("device"))return 1;*p=scenario=="wrongdevice"?1:0;return 0;}
static int cudaStreamIsCapturing(int,int *p){if(fail("capture"))return 1;*p=scenario=="capturing"?1:0;return 0;}
static int cudaDeviceGetAttribute(int *p,int attr,int dev){assert(attr==1&&dev==0);if(fail("attribute"))return 1;*p=scenario=="zerol2"?0:scenario=="bigl2"?64*1024*1024:1024*1024;return 0;}
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
static void group(bool mode,int32_t *counts,int32_t *offsets,int32_t *cursor,
    int32_t *active,int32_t *pairs,float *,const int32_t *selected,
    unsigned ne,unsigned np,unsigned used,unsigned md,unsigned stride){
    assert(!next_group&&offsets==counts+512&&cursor==counts+1024&&active==counts+1536&&pairs==active+513);
    assert(ne==512&&np==20&&used==10&&md==640&&stride==6400);
    next_group=true;last_routes=selected;cursor[0]=(int32_t)mode;gpu_us+=1;
}
template<class...A>static void qwen4exp_moe_group_plan_kernel(A...a){group(true,a...);}
template<class...A>static void qwen4exp_moe_group_small_kernel(A...a){group(false,a...);}
static void down(float *out,bool reuse){
    assert(next_group&&last_routes);next_group=false;
    for(unsigned i=0;i<5120;i++)out[i]=(float)(last_routes[i%20]+(int)(i%97));
    if(reuse&&scenario=="bad_result")out[0]+=1;
    if(scenario=="nonfinite")out[0]=NAN;
    double cost=8;
    if(scenario=="marginal")cost=9.85;
    if(scenario=="slower")cost=11;
    if(scenario=="one_slow"&&elapsed_count/112==7)cost=11;
    if(scenario=="noisy")cost=elapsed_count/16%7==6?7:9;
    if(scenario=="five_wins")cost=elapsed_count/16%7<5?9.6:10.001;
    gpu_us+=(reuse?cost:10)-1;order.push_back(reuse);
}
template<unsigned P>static void qwen4exp_moe_down_reuse_kernel(float *out,const char *,const int32_t *plan,const int8_t *,const float *,const int32_t *){
    assert(P==6&&plan[0]==1);down(out,true);
}
template<int R,int T,bool V,bool S,bool A>static void qwen4exp_moe_down_q_kernel(float *out,const char *,const int32_t *,const int8_t *,const float *,const int32_t *,
    uint64_t eb,uint64_t rb,unsigned type,unsigned groups,unsigned od,unsigned nt,unsigned ne,unsigned used){
    assert(R==2&&T==7&&V&&S&&A&&eb==1228800&&rb==480&&type==7&&groups==20&&od==2560&&nt==2&&ne==512&&used==10);
    down(out,false);
}
static void qwen4exp_down_cache_pressure(volatile uint32_t *,size_t n){assert(!in_span&&n==1024*1024);pressure_calls++;gpu_us+=100;}
'''
main = r'''
int main(int argc,char **argv){
    assert(argc==2);scenario=argv[1];
    if(scenario.find(':')!=std::string::npos){auto at=scenario.find(':');fail_name=scenario.substr(0,at);fail_n=std::stoul(scenario.substr(at+1));}
    if(scenario=="multi")g_n_gpus=2;
    const std::string report=ds4_gpu_qwen4exp_down_tune();
    assert(allocations==frees&&events==destroyed&&host_allocs==host_frees);
    unsigned done=api_calls;assert(report==ds4_gpu_qwen4exp_down_tune());assert(api_calls==done);
    bool success=scenario=="fast"||scenario=="marginal"||scenario=="slower"||scenario=="one_slow"||scenario=="noisy"||scenario=="five_wins";
    if(success){
        assert(qwen4exp_down_tuned_device==0&&elapsed_count==896&&allocations==9&&events==2&&pressure_calls==448);
        assert(qwen4exp_down_prefer_reuse==(scenario=="fast"));
        assert(order.size()==922);
        for(unsigned state=0;state<8;state++){
            unsigned base=10+state*114;assert(!order[base]&&order[base+1]);
            for(unsigned pair=0;pair<7;pair++)for(unsigned leg=0;leg<2;leg++)for(unsigned repeat=0;repeat<8;repeat++)
                assert(order[base+2+pair*16+leg*8+repeat]==bool((pair+leg)&1));
        }
        if(scenario=="fast")assert(report.find("a=10,10 b=8,8")!=std::string::npos&&report.find("win=7 use=1")!=std::string::npos);
    }else{
        assert(!qwen4exp_down_prefer_reuse&&qwen4exp_down_tuned_device==-1);
        if(!fail_name.empty())assert(counts[fail_name]>=fail_n);
    }
    assert(report.size()<256);
    std::cout<<scenario<<": "<<report<<" device="<<allocations<<"/"<<frees<<" host="<<host_allocs<<"/"<<host_frees<<" event="<<events<<"/"<<destroyed<<"\n";
}
'''
cases = ['fast', 'marginal', 'slower', 'one_slow', 'noisy', 'five_wins', 'disabled',
         'noreuse', 'nopanel', 'noasync', 'multi', 'wrongdevice', 'capturing',
         'zerol2', 'bigl2', 'bad_result', 'nonfinite', 'zero', 'nan', 'huge',
         'device:1', 'capture:1', 'attribute:1', 'sync:1', 'hostalloc:1',
         *[f'malloc:{i}' for i in range(1, 10)], 'event:1', 'event:2',
         'h2d:1', 'h2d:4', 'h2d:5', 'h2d:9', 'd2h:1', 'd2h:2', 'd2h:10',
         'memset:1', 'last:1', 'last:2', 'last:3', 'last:40', 'last:1000',
         'streamsync:1', 'streamsync:8', 'record:1', 'record:2', 'record:100',
         'wait:1', 'elapsed:1', 'elapsed:896']
with tempfile.TemporaryDirectory(prefix='down-autotune-') as directory:
    cpp = Path(directory) / 'test.cpp'
    exe = cpp.with_suffix('')
    cpp.write_text(stubs + source + main)
    subprocess.run(['c++', '-O2', '-std=c++17', '-fsanitize=undefined',
                    '-fno-sanitize-recover=all', str(cpp), '-o', str(exe)], check=True)
    for case in cases:
        subprocess.run([str(exe), case], check=True, timeout=20)
print(f'PASS {len(cases)} extracted tuner scenarios under UBSan; simulated times are not GPU measurements')
