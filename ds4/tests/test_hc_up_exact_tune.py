"""Execute the actual native startup policy with simulated CUDA operations.
Kernel arithmetic is checked separately; these cases check choice and cleanup.
"""
from pathlib import Path
import subprocess,tempfile
root=Path(__file__).resolve().parents[1]
source=(root/'ds4_cuda_hc_up_tune.cuh').read_text()
for name in ('getenv','malloc','free'): source=source.replace(name+'(', 'probe_'+name+'(')
code=r'''
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>
using std::isfinite;
static int hc_up_tuned_device=-1;static bool hc_up_prefer_mma=false;
struct GPU{int device_id;};static GPU g_gpu[1]={{0}};static int g_n_gpus=1;
using cudaStream_t=int;using cudaStreamCaptureStatus=int;
enum{cudaSuccess=0,cudaStreamCaptureStatusNone=0,cudaMemcpyHostToDevice=1,cudaMemcpyDeviceToHost=2,
 cudaDevAttrComputeCapabilityMajor=8,cudaDevAttrL2CacheSize=9};
struct Event{double time=0;};using cudaEvent_t=Event*;
static std::string scenario,fail_name;static unsigned fail_n=1,calls,allocs,frees,hosts,host_frees,events,destroyed,elapsed;
static std::map<std::string,unsigned> count;static double gpu_us=0;
struct Launch{bool mma;unsigned rows;};static std::vector<Launch> order;
static bool fail(const char *s){calls++;return ++count[s]==fail_n&&fail_name==s;}
static const char *probe_getenv(const char *s){return scenario==s?"1":nullptr;}
static void *probe_malloc(size_t n){if(fail("host"))return nullptr;auto p=malloc(n);assert(p);hosts++;return p;}
static void probe_free(void *p){if(p){free(p);host_frees++;}}
static int cuda_decode_stream(){return 0;}
static int cudaGetDevice(int *p){if(fail("device"))return 1;*p=scenario=="wrongdevice"?1:0;return 0;}
static int cudaDeviceGetAttribute(int *p,int a,int){if(fail("attr"))return 1;
 *p=a==8?(scenario=="old"?7:12):(scenario=="bad_l2"?0:scenario=="huge_l2"?129*1024*1024:64);return 0;}
static int cudaStreamIsCapturing(int,int *p){if(fail("capture"))return 1;*p=scenario=="capturing"?1:0;return 0;}
static int cudaDeviceSynchronize(){return fail("sync")?1:0;}
static int cudaStreamSynchronize(int){return fail("streamsync")?1:0;}
static int cudaMalloc(void **p,size_t n){if(fail("malloc"))return 1;*p=malloc(n);assert(*p);allocs++;return 0;}
static int cudaFree(void *p){assert(p);free(p);frees++;return 0;}
static int cudaEventCreate(cudaEvent_t *p){if(fail("event"))return 1;*p=new Event;events++;return 0;}
static int cudaEventDestroy(cudaEvent_t p){assert(p);delete p;destroyed++;return 0;}
static int cudaGetLastError(){return fail("last")?1:0;}
static int cudaMemcpy(void *dst,const void *src,size_t n,int kind){if(fail(kind==1?"h2d":"d2h"))return 1;memcpy(dst,src,n);return 0;}
static int cudaEventRecord(cudaEvent_t p,int){if(fail("record"))return 1;p->time=gpu_us;return 0;}
static int cudaEventSynchronize(cudaEvent_t){return fail("wait")?1:0;}
static int cudaEventElapsedTime(float *ms,cudaEvent_t a,cudaEvent_t b){
 if(fail("elapsed"))return 1;*ms=scenario=="zero"?0:scenario=="nan"?NAN:scenario=="huge"?10001:(float)((b->time-a->time)/1000);elapsed++;return 0;}
static void launch(bool mma,float *out,const unsigned char *w,const int8_t *x,const float *xs,unsigned rows,unsigned n){
 assert((rows==1||rows==2)&&n==10240);float v=(float)((int8_t)w[2]*(int)x[0])+xs[0];
 std::fill(out,out+(size_t)rows*n,v);if(mma&&scenario=="bad_result")out[17]+=1;
 double cost=8;unsigned pair=elapsed/2%7,mode=elapsed/14;
 if(scenario=="marginal")cost=9.85;if(scenario=="slower")cost=11;
 if(scenario=="one_slow"&&mode==3)cost=11;
 if(scenario=="noisy")cost=pair==6?7:9.5;
 if(scenario=="large_noisy")cost=pair==6?8.8:5.3;
 if(scenario=="lucky_outlier")cost=pair==6?1:9.9;
 if(scenario=="big_bad_outlier")cost=pair==6?20:5;
 if(scenario=="six_wins")cost=pair==6?10.01:5;
 if(scenario=="five_wins")cost=pair<5?9.6:10.001;
 if(scenario=="edge")cost=9.8;
 gpu_us+=mma?cost:10;order.push_back({mma,rows});
}
static void matmul_q8_hc_up_exact_mma_kernel(float *o,const unsigned char *w,const int8_t *x,const float *s,unsigned r,unsigned n){launch(true,o,w,x,s,r,n);}
template<int R> static void matmul_q8_hc_warp_pair_kernel(float *o,const unsigned char *w,const int8_t *x,const float *s,uint64_t n,unsigned r){assert(R==2);launch(false,o,w,x,s,r,n);}
#define QWEN4EXP_LAUNCH_PDL(K,G,B,S,T,...) K(__VA_ARGS__)
'''+source+r'''
int main(int argc,char **argv){assert(argc==2);scenario=argv[1];auto sep=scenario.find(':');
 if(sep!=std::string::npos){fail_name=scenario.substr(0,sep);fail_n=std::stoul(scenario.substr(sep+1));}
 if(scenario=="multi")g_n_gpus=2;
 std::string report=hc_up_exact_tune();assert(allocs==frees&&hosts==host_frees&&events==destroyed);
 auto c=calls;assert(report==hc_up_exact_tune()&&calls==c);
 bool success=scenario=="fast"||scenario=="marginal"||scenario=="slower"||scenario=="one_slow"||scenario=="noisy"||scenario=="large_noisy"||scenario=="lucky_outlier"||scenario=="big_bad_outlier"||scenario=="six_wins"||scenario=="five_wins"||scenario=="edge";
 if(success){assert(hc_up_tuned_device==0&&elapsed==56&&allocs==5&&hosts==1&&events==2);
  assert(hc_up_prefer_mma==(scenario=="fast"||scenario=="large_noisy"||scenario=="six_wins"));
  unsigned index=40;
  for(unsigned mode=0;mode<4;mode++)for(unsigned pair=0;pair<7;pair++)for(unsigned leg=0;leg<2;leg++)
   for(unsigned repeat=0;repeat<(mode<2?32:4);repeat++){
    assert(order[index].mma==bool((pair+leg)&1)&&order[index].rows==(mode%2+1));index++;
   }
  assert(index==order.size());
  if(scenario=="fast")assert(report.find("a=10,10,10,10 b=8,8,8,8")!=std::string::npos);
 }else{assert(!hc_up_prefer_mma&&hc_up_tuned_device==-1);if(!fail_name.empty())assert(count[fail_name]>=fail_n);}
 assert(report.size()<192);printf("%s: %s allocations=%u/%u events=%u/%u\n",scenario.c_str(),report.c_str(),allocs,frees,events,destroyed);
}
'''
cases=['fast','marginal','slower','one_slow','noisy','large_noisy','lucky_outlier','big_bad_outlier','six_wins','five_wins','edge',
 'DS4_Q8_NO_HC_UP_MMA','DS4_Q8_NO_HC_UP_TUNE','DS4_QWEN4EXP_NO_ROW_TILE','DS4_Q8_NO_STREAM_LOADS','DS4_Q8_NO_HC_WARP_PAIR',
 'multi','wrongdevice','old','bad_l2','huge_l2','capturing','bad_result','zero','nan','huge',
 'device:1','attr:1','attr:2','capture:1','host:1','sync:1',*[f'malloc:{i}' for i in range(1,6)],'event:1','event:2',
 'h2d:1','h2d:2','h2d:13','d2h:1','d2h:24','last:1','last:2','last:800','streamsync:1','streamsync:13',
 'record:1','record:2','record:100','wait:1','elapsed:1','elapsed:56']
with tempfile.TemporaryDirectory(prefix='hc-up-tune-') as d:
 d=Path(d);p=d/'test.cpp';exe=d/'test';p.write_text(code)
 subprocess.run(['c++','-std=c++17','-O2','-fsanitize=undefined','-fno-sanitize-recover=all',str(p),'-o',str(exe)],check=True)
 for case in cases:subprocess.run([str(exe),case],check=True,timeout=20)
print(f'PASS {len(cases)} actual tuner cases; native kernels and timings simulated')
