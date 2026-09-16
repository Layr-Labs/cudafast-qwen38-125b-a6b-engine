"""Actual startup selector with CUDA kernel/event stubs; native times are simulated."""
from pathlib import Path
import subprocess,tempfile,re
root=Path(__file__).resolve().parents[1]
source=(root/'ds4_cuda_mtp_screen_warp_tune.cuh').read_text()
launch=re.search(r'mtp_native_screen_warp_kernel<<<([^,]+),([^,]+), 0, stream>>>',source)
assert launch
source=source.replace(launch[0],f'probe_geometry({launch[1]}, {launch[2]}); mtp_native_screen_warp_kernel')
source=re.sub(r'<<<.*?>>>','',source,flags=re.S)
for name in ('getenv','malloc','free'):source=source.replace(name+'(', 'probe_'+name+'(')
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
static int mtp_screen_tuned_device=-1;static bool mtp_screen_prefer_warp=false;
static uint32_t mtp_screen_registered_width=4096,mtp_screen_tuned_width=0;
constexpr unsigned MTP_NATIVE_SCREEN_GROUPS=24,MTP_NATIVE_DIM=2560;
struct GPU{int device_id;};static GPU g_gpu[1]={{0}};static int g_n_gpus=1;
using cudaStream_t=int;using cudaStreamCaptureStatus=int;
enum{cudaSuccess=0,cudaStreamCaptureStatusNone=0,cudaMemcpyHostToDevice=1,cudaMemcpyDeviceToHost=2,
 cudaDevAttrComputeCapabilityMajor=8,cudaDevAttrL2CacheSize=9};
struct Event{double time=0;};using cudaEvent_t=Event*;
static std::string scenario,fail_name;static unsigned fail_n=1,calls,allocs,frees,hosts,host_frees,events,destroyed,elapsed;
static std::map<std::string,unsigned> count;static double gpu_us=0;
static std::vector<bool> order;
static void probe_geometry(unsigned blocks,unsigned threads){assert(blocks==(4096u+3u)/4u && threads==128u);}
static bool cuda_q8_use_dp4a(){return scenario!="no_dp4a";}
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
static int cudaMemsetAsync(void *p,int v,size_t n,int){if(fail("memset"))return 1;assert(n==4&&v==0);memset(p,v,n);return 0;}
static int cudaEventRecord(cudaEvent_t p,int){if(fail("record"))return 1;p->time=gpu_us;return 0;}
static int cudaEventSynchronize(cudaEvent_t){return fail("wait")?1:0;}
static int cudaEventElapsedTime(float *ms,cudaEvent_t a,cudaEvent_t b){
 if(fail("elapsed"))return 1;*ms=scenario=="zero"?0:scenario=="nan"?NAN:scenario=="huge"?10001:(float)((b->time-a->time)/1000);elapsed++;return 0;}

static void launch(bool warp,uint64_t *keys,uint32_t *flag,const unsigned char *w,const int8_t *x,const float *xs,unsigned width,unsigned vocab,unsigned prefix,unsigned tail){
 assert(width==4096&&vocab==4115&&prefix==3820&&tail==276);
 for(unsigned i=0;i<width;i++)keys[i]=((uint64_t)(i+(int)x[0]+w[2])<<32)|i;
 if(warp&&scenario=="bad_result")keys[17]^=1;if(warp&&scenario=="bad_flag")*flag=1;
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
 gpu_us+=warp?cost:10;order.push_back(warp);
}
static void mtp_native_screen_warp_kernel(uint64_t *k,uint32_t *f,const unsigned char *w,const int8_t *x,const float *s,uint32_t width,uint32_t v,uint32_t p,uint32_t t){launch(true,k,f,w,x,s,width,v,p,t);}
template<bool S,bool E>static void mtp_native_projection_kernel(float *,const unsigned char *w,const int8_t *x,const float *s,uint32_t width,const uint32_t *,uint32_t v,uint32_t p,uint32_t t,uint64_t *k,uint32_t *f){assert(S&&E);launch(false,k,f,w,x,s,width,v,p,t);}
'''+source+r'''
int main(int argc,char **argv){assert(argc==2);scenario=argv[1];auto sep=scenario.find(':');
 if(sep!=std::string::npos){fail_name=scenario.substr(0,sep);fail_n=std::stoul(scenario.substr(sep+1));}
 if(scenario=="multi")g_n_gpus=2;
 if(scenario=="unset")mtp_screen_registered_width=0;
 if(scenario=="small")mtp_screen_registered_width=4095;
 if(scenario=="large")mtp_screen_registered_width=131073;
 std::string report=mtp_screen_warp_tune();assert(allocs==frees&&hosts==host_frees&&events==destroyed);
 auto c=calls;assert(report==mtp_screen_warp_tune()&&calls==c);
 bool success=scenario=="fast"||scenario=="marginal"||scenario=="slower"||scenario=="one_slow"||scenario=="noisy"||scenario=="large_noisy"||scenario=="lucky_outlier"||scenario=="big_bad_outlier"||scenario=="six_wins"||scenario=="five_wins"||scenario=="edge";
 if(success){assert(mtp_screen_tuned_device==0&&mtp_screen_tuned_width==4096&&elapsed==56&&allocs==6&&hosts==3&&events==2);
  assert(mtp_screen_prefer_warp==(scenario=="fast"||scenario=="large_noisy"||scenario=="six_wins"));
  for(unsigned mode=0;mode<4;mode++){
   unsigned index=12+mode*226;assert(!order[index]&&order[index+1]);index+=2;
   for(unsigned pair=0;pair<7;pair++)for(unsigned leg=0;leg<2;leg++)for(unsigned repeat=0;repeat<16;repeat++){
    assert(order[index]==bool((pair+leg)&1));index++;
   }
  }
  assert(order.size()==916);
  if(scenario=="fast")assert(report.find("a_us=10 b_us=8")!=std::string::npos);
 }else{assert(!mtp_screen_prefer_warp&&mtp_screen_tuned_device==-1&&mtp_screen_tuned_width==0);if(!fail_name.empty())assert(count[fail_name]>=fail_n);}
 assert(report.size()<192);printf("%s: %s allocations=%u/%u host=%u/%u events=%u/%u\n",scenario.c_str(),report.c_str(),allocs,frees,hosts,host_frees,events,destroyed);
}
'''

cases=['no_dp4a','fast','marginal','slower','one_slow','noisy','large_noisy','lucky_outlier','big_bad_outlier','six_wins','five_wins','edge',
 'DS4_MTP_NO_SCREEN_WARP','DS4_MTP_NO_SCREEN_WARP_TUNE','DS4_MTP_NO_FUSED_SCREEN_KEYS','DS4_QWEN4EXP_NO_ROW_TILE','DS4_QWEN4EXP_PAIR_LANES_R2',
 'multi','unset','small','large','wrongdevice','old','bad_l2','huge_l2','capturing','bad_result','bad_flag','zero','nan','huge',
 'device:1','attr:1','attr:2','capture:1','host:1','host:2','host:3','sync:1',*[f'malloc:{i}' for i in range(1,7)],'event:1','event:2',
 'h2d:1','h2d:2','h2d:21','d2h:1','d2h:24','memset:1','memset:600','last:1','last:2','last:800','streamsync:1','streamsync:16',
 'record:1','record:2','record:100','wait:1','elapsed:1','elapsed:56']
with tempfile.TemporaryDirectory(prefix='mtp-screen-tune-') as d:
 d=Path(d);p=d/'test.cpp';exe=d/'test';p.write_text(code)
 subprocess.run(['c++','-std=c++17','-O2','-fsanitize=undefined','-fno-sanitize-recover=all',str(p),'-o',str(exe)],check=True)
 for case in cases:subprocess.run([str(exe),case],check=True,timeout=20)
print(f'PASS {len(cases)} actual tuner scenarios; kernel results and native timings simulated')
