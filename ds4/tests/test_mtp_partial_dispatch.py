"""Compile actual native-screen control flow with logged CUDA launch stubs."""
from pathlib import Path
import re, subprocess, tempfile, argparse
root=Path(__file__).resolve().parents[1]
native=(root/'ds4_cuda_mtp_native.cuh').read_text()
parser=argparse.ArgumentParser();parser.add_argument('--partial-source',type=Path,default=root/'ds4_cuda_mtp_partial.cuh')
partial=parser.parse_args().partial_source.read_text()
def definition(text,needle):
    start=text.index(needle);begin=text.index('{',start);end=begin+1;depth=1
    while depth:
        depth+=(text[end]=='{')-(text[end]=='}');end+=1
    return text[start:end]
layout=native[native.index('struct mtp_native_layout'):native.index('extern "C" int ds4_gpu_mtp_native_screen_init')]
body=definition(native,'extern "C" int ds4_gpu_mtp_native_screen(')
body=re.sub(r'<<<.*?>>>','',body,flags=re.S)
gate=definition(partial,'static bool mtp_partial_eligible(')+definition(native,'static bool mtp_screen_warp_use(')
count=definition(partial,'static bool mtp_partial_count_fits(')
disjoint=definition(native,'static bool mtp_native_key_range_disjoint(')
code=r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <set>
#include <string>
#include <vector>
constexpr uint32_t MTP_NATIVE_DIM=2560,MTP_NATIVE_CAP=2048,MTP_NATIVE_MAX_WIDTH=1<<20,MTP_PARTIAL_LIMIT=4096,MTP_NATIVE_SCREEN_GROUPS=24;
static int mtp_partial_tuned_device=0;static uint32_t mtp_partial_tuned_width=98580;
static bool mtp_partial_prefer=true;
static bool mtp_screen_prefer_warp=false;
static int mtp_screen_tuned_device=0;static uint32_t mtp_screen_tuned_width=98580;
static std::set<std::string> env;
static const char *probe_getenv(const char *s){return env.count(s)?"1":nullptr;}
#define getenv probe_getenv
struct ds4_gpu_tensor{void *ptr;uint64_t bytes;};
struct GPU{int device_id;};static GPU g_gpu[1]={{0}};static int g_n_gpus=1;
using cudaStreamCaptureStatus=int;
constexpr int cudaSuccess=0,cudaStreamCaptureStatusNone=0;
static uint32_t supplied_count=3072,invalid=0;static unsigned read_bytes;
static std::vector<std::string> launches;static std::string failure,last;
static int device=0,capture=0;static bool resolve=true,odd=false;
static void mark(const char *s){last=s;launches.push_back(s);}
static int cuda_decode_stream(){return 0;}
static bool cuda_q8_use_dp4a(){return true;}
static int ds4_tensor_device_idx(const ds4_gpu_tensor*){return 0;}
static int cudaGetDevice(int *p){*p=device;return failure=="device"?1:0;}
static int cudaStreamIsCapturing(int,int *p){*p=capture;return failure=="capture"?1:0;}
static const char *cuda_resolve_weight_ptr(const void *,uint64_t,uint64_t,int,const char*){return resolve?(const char*)(uintptr_t)(0x70000000u+(odd?1:0)):nullptr;}
static bool cuda_ok(int e,const char*){return e==0;}
static int cudaGetLastError(){return failure==last?1:0;}
static int cudaMemsetAsync(void*,int n,size_t bytes,int){assert(n==0&&(bytes==4||bytes==8));mark("clear");return failure==last?1:0;}
template<class... A>static void quantize_q8_0_f32_rows_warp_kernel(A...){mark("quant");}
template<bool S,bool E=false,class... A>static void mtp_native_projection_kernel(A...){mark(S?(E?"fused":"coarse"):"refine");}
template<class... A>static void mtp_native_keys(A...){mark("keys");}
template<class... A>static void mtp_native_screen_warp_kernel(A...){mark("warp");}
template<class... A>static void mtp_partial_pivot(A...){mark("pivot");}
template<class... A>static void mtp_partial_filter(A...){mark("filter");}
static void mtp_partial_sort_ids(uint32_t*,const uint64_t*,uint32_t n,int bits){assert(n>=2048&&n<=4096&&bits>=1&&bits<=32);mark("partial_ids");}
template<class... A>static void mtp_native_unpack_ids(A...){mark("unpack");}
static int ds4_gpu_tensor_read(const ds4_gpu_tensor*,uint64_t,void *p,size_t n){mark("read");read_bytes=n;assert(n==4||n==8);uint32_t v[]={invalid,supplied_count};memcpy(p,v,n);return failure==last?0:1;}
namespace cub{struct DeviceRadixSort{
 template<class... A>static int SortKeysDescending(A...){mark("score_sort");return failure==last?1:0;}
 template<class... A>static int SortKeys(A...){mark("id_sort");return failure==last?1:0;}
};}
'''+layout+gate+count+disjoint+body+r'''
int main(){
 constexpr uint32_t width=98580,prefix=98304,tail=276;
 auto l=mtp_native_offsets(width);assert(l.temporary-l.flag>=16);
 std::vector<unsigned char> workspace(l.temporary+4096),xbuf(2560*4),obuf(2048*4),ibuf(2048*4);
 ds4_gpu_tensor scratch{workspace.data(),workspace.size()},x{xbuf.data(),xbuf.size()},out{obuf.data(),obuf.size()},ids{ibuf.data(),ibuf.size()};
 unsigned checks=0;
 auto run=[&](int want,int path,uint32_t vocab=248320){
  launches.clear();last.clear();read_bytes=0;
  int r=ds4_gpu_mtp_native_screen(&out,&ids,&scratch,(void*)0x70000000u,1048576ull*2720,0,2560,vocab,prefix,tail,&x);
  assert(r==want);
  auto has=[&](const char*s){return std::find(launches.begin(),launches.end(),s)!=launches.end();};
  if(want==2048){
   assert(has("refine")&&has("quant")&&std::count(launches.begin(),launches.end(),"quant")==1);
   assert(has("warp")==bool(mtp_screen_prefer_warp&&!has("coarse")));
   assert(has("fused")==bool(!mtp_screen_prefer_warp&&!has("coarse")));
   assert(has("partial_ids")==bool(path==1));assert(has("score_sort")==bool(path!=1));
   assert(has("unpack")==bool(path!=1)&&has("id_sort")==bool(path!=1));
   assert(read_bytes==(path==2||path==1?8:4));
   if(path==1||path==2)assert(has("pivot")&&has("filter"));else assert(!has("pivot")&&!has("filter"));
  }else assert(!has("refine")||failure=="refine");
  checks++;
 };
 for(bool screen:{false,true}){
 mtp_screen_prefer_warp=screen;
 for(uint32_t n:{2048u,2049u,3072u,4096u}){supplied_count=n;run(2048,1);}
 for(uint32_t n:{0u,2047u,4097u,width,UINT32_MAX}){supplied_count=n;run(2048,2);}
 supplied_count=3072;invalid=1;run(0,-1);invalid=0;
 mtp_partial_prefer=false;run(2048,0);mtp_partial_prefer=true;
 mtp_partial_tuned_width=width+1;run(2048,0);mtp_partial_tuned_width=width;
 mtp_partial_tuned_device=1;run(2048,0);mtp_partial_tuned_device=0;
 run(2048,0,524288);
 env.insert("DS4_MTP_NO_PARTIAL_SELECT");run(2048,0);
 env.insert("DS4_MTP_FORCE_PARTIAL_SELECT");run(2048,0);env.erase("DS4_MTP_NO_PARTIAL_SELECT");
 mtp_partial_prefer=false;run(2048,1);mtp_partial_prefer=true;env.clear();
 env.insert("DS4_MTP_NO_FUSED_SCREEN_KEYS");run(2048,0);env.clear();
 for(auto *t:{&x,&out,&ids}){void *p=t->ptr;t->ptr=workspace.data()+128;run(2048,0);t->ptr=p;}
 device=1;run(0,-1);device=0;capture=1;run(0,-1);capture=0;g_n_gpus=2;run(0,-1);g_n_gpus=1;
 odd=true;run(0,-1);odd=false;resolve=false;run(-1,-1);resolve=true;
 for(const char *s:{"device","capture","clear","quant",screen?"warp":"fused","pivot","filter","read","partial_ids","refine"}){failure=s;run(-1,-1);}
 supplied_count=4097;
 for(const char *s:{"score_sort","unpack","id_sort"}){failure=s;run(-1,-1);}failure.clear();
 for(uint32_t n:{0u,2047u,2048u,4096u,4097u,UINT32_MAX})assert(mtp_partial_count_fits(n)==(n>=2048&&n<=4096));
 for(uint32_t w:{2049u,4095u,4096u,4097u,1048576u,1048577u}){
  env.insert("DS4_MTP_FORCE_PARTIAL_SELECT");assert(mtp_partial_eligible(true,w,0,18)==(w>=4096&&w<=1048576));
  assert(!mtp_partial_eligible(false,w,0,18));env.clear();
 }
 }
 std::cout<<"PASS "<<checks<<" extracted native-screen scenarios plus count/width/alias policy boundaries under UBSan\n";
}
'''
with tempfile.TemporaryDirectory(prefix='mtp-partial-dispatch-') as d:
    cpp=Path(d)/'test.cpp';exe=cpp.with_suffix('');cpp.write_text(code)
    subprocess.run(['c++','-O2','-std=c++17','-fsanitize=undefined','-fno-sanitize-recover=all',str(cpp),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True)
