"""Compile the actual fused-entry and split dispatcher against launch stubs."""
from pathlib import Path
import re,subprocess,tempfile
root=Path(__file__).resolve().parents[1];s=(root/'ds4_cuda_qwen4exp.cu').read_text()
def fn(marker):
 st=s.index(marker);b=s.index('{',st);e=b+1;d=1
 while d:d+=(s[e]=='{')-(s[e]=='}');e+=1
 return re.sub(r'<<<.*?>>>','',s[st:e],flags=re.S)
code=r'''
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <tuple>
#include <vector>
struct ds4_gpu_tensor{void *ptr;uint64_t bytes;int device;};
struct dim3{unsigned x,y,z;dim3(unsigned a,unsigned b=1,unsigned c=1):x(a),y(b),z(c){}};
static unsigned scores,probs,plain,fused;static int error;static bool disabled,split_disabled;
static const ds4_gpu_tensor *expected_q8,*expected_doubled;static uint64_t expected_qoff,expected_soff;
static const char *test_env(const char *n){return !strcmp(n,"DS4_QWEN4EXP_NO_QSA_FOLD_QUANT")&&disabled?"1":nullptr;}
static int ds4_tensor_device_idx(const ds4_gpu_tensor *t){return t->device;}
static bool glm53_cuda_tensor_has(const ds4_gpu_tensor *t,uint64_t n,uint64_t b){return t&&t->ptr&&n<=UINT64_MAX/b&&t->bytes>=n*b;}
static uint32_t qwen4exp_cuda_threads(uint32_t){return 256;}
static unsigned qwen4exp_qsa_split_width(unsigned n,unsigned,unsigned,unsigned){return split_disabled?0:n==1?2:4;}
static int cuda_decode_stream(){return 0;}static int cudaGetLastError(){return error;}
static bool cuda_ok(int x,const char*){return !x;}
#define QWEN4EXP_QSA_SPLIT_KPITCH 36u
#define QWEN4EXP_QSA_GROUP_SHARED_CAP (48u*1024u)
#define QWEN4EXP_QSA_SPLIT_VSTEP 16u
'''+(root/'ds4_qwen4exp_qsa_scratch.h').read_text()+r'''
static uint64_t ds4_gpu_qwen4exp_qsa_split_scratch_bytes(uint32_t n,uint32_t h,uint32_t d,uint32_t c){return ds4_qwen4exp_qsa_split_bytes(n,h,d,c);}
template<unsigned G,class... A>static void qwen4exp_qsa_split_scores_kernel(A...){scores++;}
template<unsigned G,unsigned V,class... A>static void qwen4exp_qsa_split_probs_kernel(A...){probs++;}
template<class... A>static void qwen4exp_qsa_split_fold_kernel(A...){plain++;}
template<class... A>static void qwen4exp_qsa_split_fold_quant_kernel(A... args){
 fused++;auto a=std::make_tuple(args...);assert(std::get<13>(a)==(int8_t*)expected_q8->ptr+expected_qoff);
 assert(std::get<14>(a)==(float*)((char*)expected_q8->ptr+expected_soff));assert(std::get<15>(a)==expected_doubled->ptr);
}
'''+fn('static int qwen4exp_hc_ranges_disjoint(')+'\n'+fn('static int qwen4exp_qsa_attention_split(')+'\n'+fn('extern "C" int ds4_gpu_qwen4exp_qsa_attention_fold_q8_dpos_tensor(').replace('getenv(', 'test_env(')+r'''
int main(){
 unsigned cases=0;
 auto check=[&](int scenario,unsigned variant){
  ds4_gpu_tensor t[10];for(unsigned i=0;i<10;i++)t[i]={(void*)(uintptr_t)(0x10000000ull+i*0x10000000ull),0x10000000ull,0};
  // out,q,k,v,selected,counts,dpos,scratch,q8,doubled
  auto *out=&t[0],*q=&t[1],*k=&t[2],*v=&t[3],*sel=&t[4],*cnt=&t[5],*dp=&t[6],*scratch=&t[7],*q8=&t[8],*dbl=&t[9];
  unsigned rows=1,heads=24,kv=2,dim=256,pos=511,cap=4096,selected=1024,maximum=1024;
  uint64_t qo=16,so=3*6144+32;int want=1;
  scores=probs=plain=fused=0;error=0;disabled=split_disabled=false;
  if(scenario==0){rows=variant%3+1;if(variant<3)dp=nullptr;if(variant%2)sel=cnt=nullptr;}
  if(scenario==1){rows=variant;want=0;}
  if(scenario==2){heads=variant;want=0;}
  if(scenario==3){kv=variant;want=0;}
  if(scenario==4){dim=variant;want=0;}
  if(scenario==5){disabled=true;want=0;}
  if(scenario==6){split_disabled=true;want=0;}
  if(scenario==7){t[variant].bytes=0;want=0;}
  if(scenario==8){t[variant].ptr=nullptr;want=0;}
  if(scenario==9){t[variant].device=1;want=0;}
  if(scenario==10){t[variant].ptr=q8->ptr;want=0;}
  if(scenario==11){dbl->ptr=out->ptr;want=0;}
  if(scenario==12){qo=variant?UINT64_MAX:1;want=0;}
  if(scenario==13){so=variant?UINT64_MAX:1;want=0;}
  if(scenario==14){so=qo+variant;want=0;}
  if(scenario==15){t[8].bytes=so+192*4-1;want=0;}
  if(scenario==16){dp=nullptr;pos=cap;sel=cnt=nullptr;want=0;}
  if(scenario==17){selected=0;want=0;}
  if(scenario==18){cnt=nullptr;want=0;}
  if(scenario==19){maximum=0;want=0;}
  if(scenario==20){maximum=512;want=0;}
  if(scenario==21){sel=cnt=dp=nullptr;pos=1024;want=0;}
  if(scenario==22){error=1;want=-1;}
  if(scenario==23){std::swap(qo,so);}
  expected_q8=q8;expected_doubled=dbl;expected_qoff=qo;expected_soff=so;
  const int got=ds4_gpu_qwen4exp_qsa_attention_fold_q8_dpos_tensor(out,q,k,v,sel,cnt,
   rows,heads,kv,dim,pos,cap,selected,0.0625f,dp,scratch,maximum,q8,qo,so,dbl);
  if(got!=want){fprintf(stderr,"scenario %d variant %u got %d expected %d\n",scenario,variant,got,want);abort();}
  if(want==0)assert(!scores&&!probs&&!plain&&!fused);else assert(scores==1&&probs==1&&fused==1&&!plain);
  cases++;
 };
 for(unsigned i=0;i<6;i++)check(0,i);
 for(unsigned v:{0u,4u,64u,UINT32_MAX})check(1,v);
 for(unsigned v:{0u,1u,12u,25u,UINT32_MAX})check(2,v);
 for(unsigned v:{0u,1u,3u,UINT32_MAX})check(3,v);
 for(unsigned v:{0u,128u,512u,UINT32_MAX})check(4,v);
 check(5,0);check(6,0);
 for(int s=7;s<=10;s++)for(unsigned v=0;v<10;v++)if(s!=10||v!=8)check(s,v);
 for(int s=11;s<=23;s++)check(s,0);
 check(12,1);check(13,1);check(14,16);check(14,32);
 printf("PASS %u actual fused-entry/split-dispatch scenarios with recorded launches\n",cases);
}
'''
with tempfile.TemporaryDirectory(prefix='qsa-fold-dispatch-') as d:
 src=Path(d)/'test.cpp';exe=src.with_suffix('');src.write_text(code)
 subprocess.run(['c++','-O2','-std=c++17','-fsanitize=undefined','-fno-sanitize-recover=all',str(src),'-o',str(exe)],check=True)
 subprocess.run([str(exe)],check=True)
