"""Run the real startup selector against deterministic CUDA/event stubs.

The separate partial-selection oracle checks kernel indexing and ordering.
Here synthetic times test policy and failure cleanup, not native performance.
"""
from pathlib import Path
import re, subprocess, tempfile
root=Path(__file__).resolve().parents[1]
source=(root/'ds4_cuda_mtp_partial_tune.cuh').read_text()
source=re.sub(r'<<<.*?>>>','',source,flags=re.S)
for n in ('getenv','malloc','free'): source=source.replace(n+'(','probe_'+n+'(')
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
constexpr unsigned MTP_NATIVE_CAP=2048,MTP_PARTIAL_SAMPLES=1024,MTP_PARTIAL_LIMIT=4096;
static uint32_t mtp_partial_registered_width=8192,mtp_partial_tuned_width=0;
static int mtp_partial_tuned_device=-1;static bool mtp_partial_prefer=false;
struct GPU{int device_id;};static GPU g_gpu[1]={{0}};static int g_n_gpus=1;
using cudaStream_t=int;using cudaStreamCaptureStatus=int;
constexpr int cudaSuccess=0,cudaStreamCaptureStatusNone=0,cudaMemcpyHostToDevice=1,cudaMemcpyDeviceToHost=2;
struct Event{double us=0;};using cudaEvent_t=Event*;
static std::string scenario,fail_name;static unsigned fail_n=1,api_calls,allocs,frees,ha,hf,events,destroyed,elapsed_count;
static std::map<std::string,unsigned> counts;static std::vector<bool> order;
static std::vector<uint64_t> ranked;static std::vector<uint32_t> expected;
static double gpu_us=0;static bool attempted;static unsigned pattern;
static bool fail(const char *s){api_calls++;return ++counts[s]==fail_n&&fail_name==s;}
static const char *probe_getenv(const char *s){return (scenario=="disabled"&&!strcmp(s,"DS4_MTP_NO_PARTIAL_TUNE"))||(scenario=="no_partial"&&!strcmp(s,"DS4_MTP_NO_PARTIAL_SELECT"))?"1":nullptr;}
static void *probe_malloc(size_t n){if(fail("hostalloc"))return nullptr;auto p=malloc(n);assert(p);ha++;return p;}
static void probe_free(void *p){assert(p);free(p);hf++;}
static int cuda_decode_stream(){return 0;}
static int cudaGetDevice(int *p){if(fail("device"))return 1;*p=scenario=="wrongdevice"?1:0;return 0;}
static int cudaStreamIsCapturing(int,int *p){if(fail("capture"))return 1;*p=scenario=="capturing"?1:0;return 0;}
static int cudaDeviceSynchronize(){return fail("sync")?1:0;}
static int cudaStreamSynchronize(int){return fail("streamsync")?1:0;}
static int cudaMalloc(void **p,size_t n){if(fail("malloc"))return 1;*p=malloc(n);assert(*p);allocs++;return 0;}
static int cudaFree(void *p){assert(p);free(p);frees++;return 0;}
static int cudaEventCreate(cudaEvent_t *p){if(fail("event"))return 1;*p=new Event;events++;return 0;}
static int cudaEventDestroy(cudaEvent_t p){assert(p);delete p;destroyed++;return 0;}
static int cudaGetLastError(){return fail("last")?1:0;}
static int cudaMemcpy(void *dst,const void *src,size_t n,int kind){
 if(fail(kind==1?"h2d":"d2h"))return 1;memcpy(dst,src,n);
 if(kind==1){
  assert(n==8192*8);ranked.assign((const uint64_t*)src,(const uint64_t*)src+n/8);
  std::sort(ranked.begin(),ranked.end(),std::greater<uint64_t>());expected.clear();
  for(unsigned i=0;i<2048;i++)expected.push_back(UINT32_MAX-(uint32_t)ranked[i]);std::sort(expected.begin(),expected.end());
  pattern=counts["h2d"]-1;
 }
 return 0;
}
static int cudaMemsetAsync(void *p,int v,size_t n,int){if(fail("memset"))return 1;assert(n==8);memset(p,v,n);attempted=false;return 0;}
static int cudaEventRecord(cudaEvent_t p,int){if(fail("record"))return 1;p->us=gpu_us;return 0;}
static int cudaEventSynchronize(cudaEvent_t){return fail("wait")?1:0;}
static int cudaEventElapsedTime(float *ms,cudaEvent_t a,cudaEvent_t b){
 if(fail("elapsed"))return 1;*ms=scenario=="zero"?0:scenario=="nan"?NAN:scenario=="huge"?8001:(float)((b->us-a->us)/1000.0);elapsed_count++;return 0;
}
static bool mtp_partial_count_fits(uint32_t n){return n>=2048&&n<=4096;}
static void finish(){
 double cost=8;
 if(scenario=="marginal")cost=9.85;if(scenario=="slower")cost=11;
 if(scenario=="one_slow"&&elapsed_count/14==3)cost=11;
 if(scenario=="noisy")cost=elapsed_count/2%7==6?7:9.5;
 if(scenario=="large_noisy_win")cost=elapsed_count/2%7==6?8.8:5.3;
 if(scenario=="one_noisy_pattern"&&elapsed_count/14==3)cost=elapsed_count/2%7==6?7:9.5;
 if(scenario=="lucky_outlier")cost=elapsed_count/2%7==6?1:9.9;
 if(scenario=="big_negative_outlier")cost=elapsed_count/2%7==6?20:5;
 if(scenario=="safe_six_wins")cost=elapsed_count/2%7==6?10.01:5;
 if(scenario=="policy_edge")cost=9.8;
 if(scenario=="one_pair_ab")cost=(elapsed_count%14)<2?4:10.001;
 if(scenario=="five_wins")cost=elapsed_count/2%7<5?9.6:10.001;
 gpu_us+=attempted?cost:10;order.push_back(attempted);
}
static void mtp_partial_pivot(uint64_t *pivot,const uint64_t *,uint32_t width){
 assert(width==8192);attempted=true;unsigned n=pattern==4?5000:pattern==5?1000:3072;*pivot=ranked[n-1];
}
static void mtp_partial_filter(uint64_t *out,uint32_t *count,const uint64_t *keys,const uint64_t *pivot,uint32_t width){
 assert(width==8192&&attempted);for(unsigned i=0;i<width;i++)if(keys[i]>=*pivot){auto at=(*count)++;if(at<4096)out[at]=keys[i];}
}
static void mtp_partial_sort_ids(uint32_t *ids,const uint64_t *,uint32_t count,int bits){
 assert(bits==18&&mtp_partial_count_fits(count));std::copy(expected.begin(),expected.end(),ids);
 if(scenario=="bad_result")ids[0]^=1;finish();
}
static void mtp_native_unpack_ids(uint32_t *out,const uint64_t *keys){for(unsigned i=0;i<2048;i++)out[i]=UINT32_MAX-(uint32_t)keys[i];}
namespace cub{struct DeviceRadixSort{
 static int SortKeysDescending(void *temp,size_t &bytes,const uint64_t *,uint64_t *out,unsigned width,int lo,int hi,int){
  assert(width==8192&&lo==32&&hi==64);if(!temp){bytes=32768;return fail("score_query")?1:0;}
  if(fail("score_sort"))return 1;assert(bytes>=32768);std::copy(ranked.begin(),ranked.end(),out);return 0;
 }
 static int SortKeys(void *temp,size_t &bytes,const uint32_t *,uint32_t *out,unsigned n,int lo,int hi,int){
  assert(n==2048&&lo==0&&hi==18);if(!temp){bytes=4096;return fail("id_query")?1:0;}
  if(fail("id_sort"))return 1;assert(bytes>=4096);std::copy(expected.begin(),expected.end(),out);finish();return 0;
 }
};}
'''+source+r'''
int main(int argc,char **argv){
 assert(argc==2);scenario=argv[1];auto at=scenario.find(':');if(at!=std::string::npos){fail_name=scenario.substr(0,at);fail_n=std::stoul(scenario.substr(at+1));}
 if(scenario=="multi")g_n_gpus=2;
 if(scenario=="unset")mtp_partial_registered_width=0;
 if(scenario=="small")mtp_partial_registered_width=2048;
 if(scenario=="large")mtp_partial_registered_width=131073;
 const std::string report=mtp_partial_tune();assert(allocs==frees&&ha==hf&&events==destroyed);
 unsigned done=api_calls;assert(report==mtp_partial_tune()&&api_calls==done);
 bool success=scenario=="fast"||scenario=="marginal"||scenario=="slower"||scenario=="one_slow"||scenario=="noisy"||scenario=="five_wins"||scenario=="large_noisy_win"||scenario=="one_noisy_pattern"||scenario=="lucky_outlier"||scenario=="big_negative_outlier"||scenario=="safe_six_wins"||scenario=="policy_edge"||scenario=="one_pair_ab";
 if(success){
  assert(mtp_partial_tuned_width==8192&&mtp_partial_tuned_device==0&&elapsed_count==56&&allocs==6&&events==2&&ha==2);
  assert(mtp_partial_prefer==(scenario=="fast"||scenario=="large_noisy_win"||scenario=="safe_six_wins"));assert(order.size()==470);
  for(unsigned p=0;p<4;p++){unsigned base=14+p*114;assert(!order[base]&&order[base+1]);
   for(unsigned pair=0;pair<7;pair++)for(unsigned leg=0;leg<2;leg++)for(unsigned r=0;r<8;r++)assert(order[base+2+pair*16+leg*8+r]==bool((pair+leg)&1));
  }
  if(scenario=="fast")assert(report.find("a_us=10 b_us=8")!=std::string::npos&&report.find("win=7 use=1")!=std::string::npos);
 }else{assert(!mtp_partial_prefer&&mtp_partial_tuned_width==0&&mtp_partial_tuned_device==-1);if(!fail_name.empty())assert(counts[fail_name]>=fail_n);}
 assert(report.size()<192);std::cout<<scenario<<": "<<report<<" alloc="<<allocs<<"/"<<frees<<" host="<<ha<<"/"<<hf<<" events="<<events<<"/"<<destroyed<<"\n";
}
'''
cases=['fast','marginal','slower','one_slow','noisy','five_wins','large_noisy_win','one_noisy_pattern','lucky_outlier','big_negative_outlier','safe_six_wins','policy_edge','one_pair_ab','disabled','no_partial',
 'multi','unset','small','large','wrongdevice','capturing','bad_result','zero','nan','huge',
 'device:1','capture:1','sync:1','score_query:1','id_query:1',
 *[f'malloc:{i}' for i in range(1,7)],'hostalloc:1','hostalloc:2','event:1','event:2',
 'h2d:1','h2d:7','h2d:11','d2h:1','d2h:2','d2h:4','d2h:484',
 'memset:1','memset:30','last:1','last:2','last:3','last:800',
 'score_sort:1','score_sort:30','id_sort:1','id_sort:30',
 'streamsync:1','streamsync:4','record:1','record:2','record:100','wait:1','elapsed:1','elapsed:56']
with tempfile.TemporaryDirectory(prefix='mtp-partial-tune-') as d:
    cpp=Path(d)/'test.cpp';exe=cpp.with_suffix('');cpp.write_text(code)
    subprocess.run(['c++','-O2','-std=c++17','-fsanitize=undefined','-fno-sanitize-recover=all',str(cpp),'-o',str(exe)],check=True)
    for case in cases:subprocess.run([str(exe),case],check=True,timeout=20)
print(f'PASS {len(cases)} actual tuner scenarios under UBSan; native kernels and elapsed times are simulated')
