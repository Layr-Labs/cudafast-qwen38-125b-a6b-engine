"""Execute the actual startup policy with synthetic CUDA/event stubs.

This verifies control flow and cleanup, not GPU arithmetic or performance.
The separate lane scheduler executes the production down kernels themselves.
"""
from pathlib import Path
import re
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
source = (root / 'ds4_cuda_gateup_copy_tune.cuh').read_text()
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
constexpr unsigned QW_GU_COOP_ROWS=4,DS4_QWEN4EXP_TY_q4_K=12;
static int qwen4exp_gateup_copy_device=-1;
static bool qwen4exp_gateup_copy_prefer=false;
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
static bool in_span=false;
static unsigned route_calls;

static bool fail(const char *name){api_calls++;return ++counts[name]==fail_n && fail_name==name;}
static const char *probe_getenv(const char *name){
 return (scenario=="disabled"&&!strcmp(name,"DS4_QWEN4EXP_NO_GATEUP_COPY_TUNE")) ||
        (scenario=="noasync"&&!strcmp(name,"DS4_QWEN4EXP_NO_GATEUP_ASYNC_COPY")) ? "1" : nullptr;
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
static int cudaMemcpy(void *dst,const void *src,size_t n,int kind){if(fail(kind==1?"h2d":"d2h"))return 1;memcpy(dst,src,n);if(kind==1&&n==1557u*4u)route_calls++;return 0;}
static int cudaMemsetAsync(void *p,int v,size_t n,int){if(fail("memset"))return 1;memset(p,v,n);return 0;}
static int cudaEventRecord(cudaEvent_t p,int){if(fail("record"))return 1;p->us=gpu_us;in_span=!in_span;return 0;}
static int cudaEventSynchronize(cudaEvent_t){return fail("wait")?1:0;}
static int cudaEventElapsedTime(float *ms,cudaEvent_t a,cudaEvent_t b){
    if(fail("elapsed"))return 1;
    *ms=scenario=="zero"?0:scenario=="nan"?NAN:scenario=="huge"?1001:(float)((b->us-a->us)/1000.0);
    elapsed_count++;return 0;
}
static void gateup(bool reuse,float *out,const char *gate,const char *up,const int8_t *xq,
 const float *xs,const int32_t *xsum,const int32_t *pairs,const int32_t *counts,
 const int32_t *offsets,const int32_t *active,const float *router,
 uint64_t ge,uint64_t gr,uint64_t ue,uint64_t ur,unsigned gt,unsigned ut,
 unsigned groups,unsigned md,unsigned stride,unsigned used){
 assert(ge==921600&&ue==ge&&gr==1440&&ur==gr&&gt==12&&ut==12);
 assert(groups==80&&md==640&&stride==6416&&used==10&&up==gate+20u*921600u);
 assert(offsets==counts+512&&active==counts+1024&&pairs==counts+1537);
 unsigned p=route_calls<=5?route_calls-1:(route_calls-6)%4,np=p==0?10:20;
 int ids[20];for(unsigned i=0;i<10;i++){ids[i]=i;ids[10+i]=p==1?i+10:p==2?9-i:i<5?4-i:i+5;}
 if(p==4){ids[0]=-1;ids[6]=512;ids[14]=-7;}
 unsigned total=0,na=0;
 for(unsigned e=0;e<512;e++){
  unsigned want=0;for(unsigned i=0;i<np;i++)want+=ids[i]==(int)e;
  assert(counts[e]==(int)want&&offsets[e]==(int)total);na+=want>0;
  for(unsigned j=0;j<want;j++){
   int slot=pairs[total+j];assert(slot>=0&&slot<(int)np&&ids[slot]==(int)e);
   for(unsigned row=0;row<640;row++)out[(slot/10)*6416+(slot%10)*640+row]=(float)(slot*100+row+e);
  }
  total+=want;
 }
 assert(active[0]==(int)na);
 for(unsigned i=1;i<=na;i++){assert(active[i]>=0&&active[i]<20&&counts[active[i]]>0);if(i>1)assert(active[i]<active[i-1]);}
 for(unsigned g=0;g<160;g++){int sum=0;for(unsigned j=0;j<32;j++)sum+=xq[g*32+j];assert(xsum[g]==sum&&xs[g]>0&&isfinite(xs[g]));}
 for(unsigned i=0;i<20;i++)assert(router[i]>0&&router[i]<1);
 if(reuse&&scenario=="bad_result")out[0]+=1;
 if(reuse&&scenario=="late_result"&&p==4)out[2*6416+63]+=1e16f;
 if(scenario=="nonfinite")out[0]=NAN;
 double cost=8;
 if(scenario=="marginal")cost=9.85;
 if(scenario=="slower")cost=11;
 if(scenario=="one_slow"&&elapsed_count/112==7)cost=11;
 if(scenario=="noisy")cost=elapsed_count/16%7==6?8:9.6;
 if(scenario=="five_wins")cost=elapsed_count/16%7<5?9.6:10.001;
 gpu_us+=reuse?cost:10;order.push_back(reuse);
}
template<int R,int T,bool V,unsigned P,bool C,class...A>static void qwen4exp_moe_gateup_split_kernel(A...a){assert(R==2&&T==12&&V&&P==4&&C);gateup(false,a...);}
template<int R,int T,bool V,unsigned P,bool C,class...A>static void qwen4exp_moe_gateup_async_kernel(A...a){assert(R==2&&T==12&&V&&P==4&&C);gateup(true,a...);}
static void qwen4exp_gateup_probe_weights(uint32_t *,size_t n){assert(!in_span&&n==2u*20u*640u*10u);}
static void qwen4exp_gateup_cache_pressure(volatile uint32_t *,size_t n){assert(!in_span&&n==1024*1024);pressure_calls++;gpu_us+=100;}
'''
main = r'''
int main(int argc,char **argv){
    assert(argc==2);scenario=argv[1];
    if(scenario.find(':')!=std::string::npos){auto at=scenario.find(':');fail_name=scenario.substr(0,at);fail_n=std::stoul(scenario.substr(at+1));}
    if(scenario=="multi")g_n_gpus=2;
    const std::string report=ds4_gpu_qwen4exp_gateup_copy_tune();
    assert(allocations==frees&&events==destroyed&&host_allocs==host_frees);
    unsigned done=api_calls;assert(report==ds4_gpu_qwen4exp_gateup_copy_tune());assert(api_calls==done);
    bool success=scenario=="fast"||scenario=="marginal"||scenario=="slower"||scenario=="one_slow"||scenario=="noisy"||scenario=="five_wins";
    if(success){
        assert(qwen4exp_gateup_copy_device==0&&elapsed_count==896&&allocations==8&&events==2&&pressure_calls==448);
        assert(qwen4exp_gateup_copy_prefer==(scenario=="fast"));
        assert(order.size()==922);
        for(unsigned state=0;state<8;state++){
            unsigned base=10+state*114;assert(!order[base]&&order[base+1]);
            for(unsigned pair=0;pair<7;pair++)for(unsigned leg=0;leg<2;leg++)for(unsigned repeat=0;repeat<8;repeat++)
                assert(order[base+2+pair*16+leg*8+repeat]==bool((pair+leg)&1));
        }
        if(scenario=="fast")assert(report.find("a=10,10 b=8,8")!=std::string::npos&&report.find("win=7 use=1")!=std::string::npos);
    }else{
        assert(!qwen4exp_gateup_copy_prefer&&qwen4exp_gateup_copy_device==-1);
        if(!fail_name.empty())assert(counts[fail_name]>=fail_n);
    }
    assert(report.size()<320);
    std::cout<<scenario<<": "<<report<<" device="<<allocations<<"/"<<frees<<" host="<<host_allocs<<"/"<<host_frees<<" event="<<events<<"/"<<destroyed<<"\n";
}
'''
cases = ['fast', 'marginal', 'slower', 'one_slow', 'noisy', 'five_wins', 'disabled',
         'noasync', 'multi', 'wrongdevice', 'capturing',
         'zerol2', 'bigl2', 'bad_result', 'late_result', 'nonfinite', 'zero', 'nan', 'huge',
         'device:1', 'capture:1', 'attribute:1', 'sync:1',
         *[f'malloc:{i}' for i in range(1, 9)], 'event:1', 'event:2',
         'h2d:1', 'h2d:4', 'h2d:5', 'h2d:9', 'd2h:1', 'd2h:2', 'd2h:10',
         'memset:1', 'last:1', 'last:2', 'last:3', 'last:40', 'last:1000',
         'streamsync:1', 'streamsync:8', 'record:1', 'record:2', 'record:100',
         'wait:1', 'elapsed:1', 'elapsed:896']
with tempfile.TemporaryDirectory(prefix='gateup-copy-tune-') as directory:
    cpp = Path(directory) / 'test.cpp'
    exe = cpp.with_suffix('')
    cpp.write_text(stubs + source + main)
    subprocess.run(['c++', '-O2', '-std=c++17', '-fsanitize=undefined',
                    '-fno-sanitize-recover=all', str(cpp), '-o', str(exe)], check=True)
    for case in cases:
        subprocess.run([str(exe), case], check=True, timeout=20)
print(f'PASS {len(cases)} extracted tuner scenarios under UBSan; simulated times are not GPU measurements')

print(f"{len(cases)} actual gate/up startup-tuner scenarios PASS")
