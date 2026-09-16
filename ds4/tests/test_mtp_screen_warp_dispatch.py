"""Execute the actual whole native-screen entry with logged CUDA stubs."""
from pathlib import Path
import re,subprocess,tempfile
root=Path(__file__).resolve().parents[1]
native=(root/'ds4_cuda_mtp_native.cuh').read_text()
def definition(text,needle):
 a=text.index(needle);b=text.index('{',a);end=b+1;depth=1
 while depth:depth+=(text[end]=='{')-(text[end]=='}');end+=1
 return text[a:end]
layout=native[native.index('struct mtp_native_layout'):native.index('extern "C" int ds4_gpu_mtp_native_screen_init')]
body=definition(native,'extern "C" int ds4_gpu_mtp_native_screen(')
launch=re.search(r'mtp_native_screen_warp_kernel<<<([^,]+),([^,]+),0,cuda_decode_stream\(\)>>>',body)
assert launch
body=body.replace(launch.group(0),f'record_grid({launch[1]}, {launch[2]}); mtp_native_screen_warp_kernel')
body=re.sub(r'<<<.*?>>>','',body,flags=re.S)
gate=definition(native,'static bool mtp_screen_warp_use(')
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
static int mtp_screen_tuned_device=0;static uint32_t mtp_screen_tuned_width=98580;
static bool mtp_screen_prefer_warp=true;
static bool mtp_partial_eligible(bool,unsigned,int,int){return false;}
static bool mtp_partial_count_fits(unsigned n){return n>=2048&&n<=4096;}
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
static unsigned observed_grid=0,observed_block=0;
static void record_grid(unsigned g,unsigned b){observed_grid=g;observed_block=b;}
static void mark(const char *s){last=s;launches.push_back(s);}
static int cuda_decode_stream(){return 0;}
static bool dp4a_available=true;
static bool cuda_q8_use_dp4a(){return dp4a_available;}
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
'''+layout+gate+disjoint+body+r'''
int main(){
 constexpr uint32_t width=98580,prefix=98304,tail=276;
 auto l=mtp_native_offsets(width);
 std::vector<unsigned char> workspace(l.temporary+4096),xbuf(2560*4),obuf(2048*4),ibuf(2048*4);
 ds4_gpu_tensor scratch{workspace.data(),workspace.size()},x{xbuf.data(),xbuf.size()},out{obuf.data(),obuf.size()},ids{ibuf.data(),ibuf.size()};
 unsigned checks=0;
 auto run=[&](int want,int path){
  launches.clear();last.clear();read_bytes=0;observed_grid=observed_block=0;
  int r=ds4_gpu_mtp_native_screen(&out,&ids,&scratch,(void*)0x70000000u,1048576ull*2720,0,2560,248320,prefix,tail,&x);
  assert(r==want);
  auto has=[&](const char*s){return std::find(launches.begin(),launches.end(),s)!=launches.end();};
  if(want==2048){
   assert(has("refine")&&has("quant")&&std::count(launches.begin(),launches.end(),"quant")==1);
   assert(has("warp")==bool(path==1));if(path==1)assert(observed_grid==(width+3)/4&&observed_block==128);assert(has("fused")==bool(path==0));assert(has("coarse")==bool(path==2));
   assert(has("keys")==bool(path==2));assert(read_bytes==4&&has("score_sort")&&has("unpack")&&has("id_sort"));
  }else assert(!has("refine")||failure=="refine");
  checks++;
 };
 for(int prefer:{0,1})for(int capable:{0,1})for(int force:{0,1})for(int no:{0,1}){
  env.clear();mtp_screen_prefer_warp=prefer;dp4a_available=capable;
  if(force)env.insert("DS4_MTP_FORCE_SCREEN_WARP");if(no)env.insert("DS4_MTP_NO_SCREEN_WARP");
  run(capable?2048:0,capable&&!no&&(force||prefer)?1:0);
 }
 env.clear();mtp_screen_prefer_warp=true;dp4a_available=true;
 mtp_screen_tuned_width=width+1;run(2048,0);mtp_screen_tuned_width=width;
 mtp_screen_tuned_device=1;run(2048,0);mtp_screen_tuned_device=0;
 env.insert("DS4_MTP_NO_FUSED_SCREEN_KEYS");run(2048,2);env.clear();
 for(auto *t:{&x,&out,&ids}){void *p=t->ptr;t->ptr=workspace.data()+128;run(2048,2);t->ptr=p;}
 invalid=1;run(0,-1);invalid=0;
 device=1;run(0,-1);device=0;capture=1;run(0,-1);capture=0;g_n_gpus=2;run(0,-1);g_n_gpus=1;
 odd=true;run(0,-1);odd=false;resolve=false;run(-1,-1);resolve=true;
 for(const char *s:{"device","capture","clear","quant","warp","read","score_sort","unpack","id_sort","refine"}){failure=s;run(-1,-1);}failure.clear();
 env.insert("DS4_MTP_NO_SCREEN_WARP");failure="fused";run(-1,-1);failure.clear();env.clear();
 env.insert("DS4_MTP_NO_FUSED_SCREEN_KEYS");failure="coarse";run(-1,-1);failure="keys";run(-1,-1);failure.clear();env.clear();
 for(const char *s:{"DS4_QWEN4EXP_NO_ROW_TILE","DS4_QWEN4EXP_PAIR_LANES_R2"}){env.insert(s);run(0,-1);env.clear();}
 std::cout<<"PASS "<<checks<<" actual native-screen dispatch scenarios under UBSan; alias, selector, capability, existing fallbacks and errors\n";
}
'''

with tempfile.TemporaryDirectory(prefix='mtp-screen-dispatch-') as d:
 d=Path(d);p=d/'test.cpp';e=d/'test';p.write_text(code)
 subprocess.run(['c++','-std=c++17','-O2','-fsanitize=undefined','-fno-sanitize-recover=all',str(p),'-o',str(e)],check=True)
 subprocess.run([str(e)],check=True)
