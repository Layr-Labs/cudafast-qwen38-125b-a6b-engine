"""Execute the actual startup tuner with deterministic CUDA/event stubs.

Tests paired ordering, conservative selection, idempotence and cleanup/error
paths. Stub timings are deliberately synthetic, never GPU performance data.
"""
from pathlib import Path
import re, subprocess, tempfile
repo=Path(__file__).resolve().parents[2]
source=(repo/'ds4/ds4_cuda_mtp_id_tune.cuh').read_text()
source=re.sub(r'<<<.*?>>>','',source,flags=re.S).replace('getenv(', 'probe_getenv(')
code=r'''
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
constexpr unsigned MTP_NATIVE_CAP=2048;
struct GPU{int device_id;};static GPU g_gpu[1]={{0}};static int g_n_gpus=1;
using cudaStream_t=int;using cudaStreamCaptureStatus=int;
constexpr int cudaSuccess=0,cudaStreamCaptureStatusNone=0;
constexpr int cudaMemcpyHostToDevice=1,cudaMemcpyDeviceToHost=2;
struct Event{double us=0;};using cudaEvent_t=Event*;
static std::string scenario,fail_name;
static unsigned fail_n=1,allocations,frees,events,destroyed,elapsed_count,api_calls;
static std::map<std::string,unsigned> counts;
static std::vector<bool> order;
static double gpu_us=0;
static bool fail(const char *name){api_calls++;unsigned n=++counts[name];return fail_name==name&&n==fail_n;}
static const char *probe_getenv(const char *name){
    return (scenario=="disabled"&&!strcmp(name,"DS4_MTP_NO_ID_AUTOTUNE")) ||
           (scenario=="noblock"&&!strcmp(name,"DS4_MTP_NO_BLOCK_ID_SORT"))?"1":nullptr;
}
static int cuda_decode_stream(){return 0;}
static int cudaGetDevice(int *p){if(fail("device"))return 1;*p=scenario=="wrongdevice"?1:0;return 0;}
static int cudaStreamIsCapturing(int,int *p){if(fail("capture"))return 1;*p=scenario=="capturing"?1:0;return 0;}
static int cudaDeviceSynchronize(){return fail("sync")?1:0;}
static int cudaStreamSynchronize(int){return fail("streamsync")?1:0;}
static int cudaMalloc(void **p,size_t n){if(fail("malloc"))return 1;*p=malloc(n);assert(*p);allocations++;return 0;}
static int cudaFree(void *p){assert(p);free(p);frees++;return 0;}
static int cudaEventCreate(cudaEvent_t *p){if(fail("event"))return 1;*p=new Event;events++;return 0;}
static int cudaEventDestroy(cudaEvent_t p){assert(p);delete p;destroyed++;return 0;}
static int cudaGetLastError(){return fail("last")?1:0;}
static int cudaMemcpy(void *dst,const void *src,size_t n,int kind){if(fail(kind==1?"h2d":"d2h"))return 1;memcpy(dst,src,n);return 0;}
static int cudaEventRecord(cudaEvent_t p,int){if(fail("record"))return 1;p->us=gpu_us;return 0;}
static int cudaEventSynchronize(cudaEvent_t){return fail("wait")?1:0;}
static int cudaEventElapsedTime(float *ms,cudaEvent_t a,cudaEvent_t b){
    if(fail("elapsed"))return 1;
    *ms=scenario=="zero"?0:scenario=="nan"?NAN:(float)((b->us-a->us)/1000.0);
    elapsed_count++;return 0;
}
static void mtp_native_unpack_ids(uint32_t *ids,const uint64_t *keys){
    for(unsigned i=0;i<MTP_NATIVE_CAP;i++)ids[i]=UINT32_MAX-(uint32_t)keys[i];gpu_us+=3;
}
static void mtp_native_unpack_sort_ids(uint32_t *ids,const uint64_t *keys,int bits){
    assert(bits==18);
    for(unsigned i=0;i<MTP_NATIVE_CAP;i++)ids[i]=UINT32_MAX-(uint32_t)keys[i];
    std::sort(ids,ids+MTP_NATIVE_CAP);if(scenario=="bad_result")ids[0]^=1;
    double cost=8;
    if(scenario=="marginal")cost=9.85;
    if(scenario=="slower")cost=11;
    if(scenario=="noisy")cost=elapsed_count/2==6?7:9;
    if(scenario=="five_wins")cost=elapsed_count/2<5?9.6:10.001;
    gpu_us+=cost;order.push_back(true);
}
namespace cub{struct DeviceRadixSort{
    static int SortKeys(void *temp,size_t &bytes,const uint32_t *in,uint32_t *out,
            unsigned n,int lo,int hi,int){
        assert(n==2048&&lo==0&&hi==18);
        if(!temp){bytes=4096;return fail("query")?1:0;}
        if(fail("sort"))return 1;
        assert(bytes>=4096);std::copy(in,in+n,out);std::sort(out,out+n);gpu_us+=7;
        order.push_back(false);return 0;
    }
};}
'''+source+r'''
int main(int argc,char **argv){
    assert(argc==2);scenario=argv[1];
    if(scenario.find(':')!=std::string::npos){auto at=scenario.find(':');fail_name=scenario.substr(0,at);fail_n=std::stoul(scenario.substr(at+1));}
    if(scenario=="multi")g_n_gpus=2;
    const std::string report=mtp_native_id_sort_tune();
    assert(allocations==frees&&events==destroyed);
    unsigned done=api_calls;assert(report==mtp_native_id_sort_tune());assert(api_calls==done);
    bool success=scenario=="fast"||scenario=="marginal"||scenario=="slower"||scenario=="noisy"||scenario=="five_wins";
    if(success){
        assert(mtp_id_tuned_device==0 && elapsed_count==14 && allocations==4 && events==2);
        assert(mtp_id_prefer_block18==(scenario=="fast"));
        assert(order.size()==6+8+7*128);
        for(unsigned pair=0;pair<7;pair++)for(unsigned leg=0;leg<2;leg++)for(unsigned repeat=0;repeat<64;repeat++)
            assert(order[14+pair*128+leg*64+repeat]==bool((pair+leg)&1));
        if(scenario=="fast")assert(report.find("a_us=10 b_us=8")!=std::string::npos&&report.find("win=7 use=1")!=std::string::npos);
    }else{
        assert(!mtp_id_prefer_block18&&mtp_id_tuned_device==-1);
        if(!fail_name.empty())assert(counts[fail_name]>=fail_n);
    }
    assert(report.size()<128);
    std::cout<<scenario<<": "<<report<<" alloc/free="<<allocations<<"/"<<frees<<" events/destroy="<<events<<"/"<<destroyed<<"\n";
}
'''
cases=['fast','marginal','slower','noisy','five_wins','disabled','noblock','multi',
       'wrongdevice','capturing','bad_result','zero','nan','device:1','capture:1',
       'sync:1','query:1','malloc:1','malloc:2','malloc:3','malloc:4','event:1','event:2',
       'h2d:1','h2d:3','d2h:1','d2h:2','last:1','last:2','last:50','sort:1','sort:50',
       'streamsync:1','record:1','record:2','record:5','wait:1','elapsed:1','elapsed:14']
with tempfile.TemporaryDirectory(prefix='mtp-id-autotune-') as directory:
    cpp=Path(directory)/'tune.cpp';exe=cpp.with_suffix('');cpp.write_text(code)
    subprocess.run(['c++','-O2','-std=c++17','-fsanitize=undefined',
                    '-fno-sanitize-recover=all',str(cpp),'-o',str(exe)],check=True)
    for case in cases:subprocess.run([str(exe),case],check=True,timeout=20)
print(f'PASS {len(cases)} extracted tuner scenarios under UBSan; timings and kernels are deterministic host stubs')
