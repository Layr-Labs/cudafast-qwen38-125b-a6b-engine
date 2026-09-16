"""Compile the actual new decode-rung branch with launch stubs."""
from pathlib import Path
import re,subprocess,tempfile
root=Path(__file__).resolve().parents[1]
s=(root/'ds4_cuda.cu').read_text();key='if (in_dim == 320u && out_dim == 10240u &&'
a=s.index('{',s.index(key,s.index('static int cuda_matmul_q8_0_preq_rows_exact')));b=a+1;depth=1
while depth:depth+=(s[b]=='{')-(s[b]=='}');b+=1
branch=s[a+1:b-1];branch=re.sub(r'<<<.*?>>>','',branch,flags=re.S)
h=(root/'ds4_cuda_hc_up_exact.cuh').read_text();helper=h[h.index('static bool hc_up_exact_use'):]
for name in ('getenv',):branch=branch.replace(name+'(', 'test_'+name+'(');helper=helper.replace(name+'(', 'test_'+name+'(')
code=r'''
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>
#include <cstdio>
struct Tensor{void *ptr;int tier;};struct GPU{int device_id;};static GPU g_gpu[2]={{3},{7}};static int g_n_gpus=2;
static int hc_up_tuned_device=3;static bool hc_up_prefer_mma=true,available=true;
static std::set<std::string> env;static std::vector<std::string> calls;static int launch_error;
static const char *test_getenv(const char *s){return env.count(s)?"1":nullptr;}
static int ds4_tensor_device_idx(Tensor *t){return t->tier;}
static bool cuda_q8_mma_available(){return available;}
static int cuda_decode_stream(){return 0;}
static int cudaGetLastError(){return launch_error;}
static int cuda_ok(int e,const char *){return !e;}
struct dim3{unsigned x,y,z;dim3(unsigned a,unsigned b=1,unsigned c=1):x(a),y(b),z(c){}};
#define QWEN4EXP_LAUNCH_PDL(K,G,B,S,T,...) K(__VA_ARGS__)
static void matmul_q8_hc_up_exact_mma_kernel(float *,const unsigned char *,const int8_t *,const float *,uint32_t r,uint32_t n){assert(r>=1&&r<=2&&n==10240);calls.push_back("mma");}
template<int R,bool S=true>static void matmul_q8_0_preq_pair_lanes_kernel(float *,const unsigned char *,const int8_t *,const float *,uint64_t n,uint32_t r,uint64_t b){assert(n==10240&&b==10);calls.push_back("pair"+std::to_string(R));}
template<int R>static void matmul_q8_hc_warp_pair_kernel(float *,const unsigned char *,const int8_t *,const float *,uint64_t n,uint32_t r){assert(R==2&&n==10240&&(r==1||r==2));calls.push_back("warp");}
'''+helper+'\n'+r'''
static int dispatch(Tensor *out,uint32_t n_rows){
 const char *wptr=nullptr;const int8_t *xq=nullptr;const float *xscale=nullptr;uint64_t out_dim=10240,blocks=10;
'''+branch+r'''
 return cuda_ok(cudaGetLastError(),"parent launch");
}
int main(){float storage[20480];Tensor out{storage,0};unsigned tests=0;
 auto run=[&](unsigned rows,std::vector<std::string> want,int ok=1){calls.clear();assert(dispatch(&out,rows)==ok);assert(calls==want);tests++;};
 for(int tier:{-1,0,1,2})for(int prefer:{0,1})for(int capable:{0,1})for(int force:{0,1})for(int no:{0,1}){
  env.clear();out.tier=tier;hc_up_prefer_mma=prefer;available=capable;
  if(force)env.insert("DS4_Q8_FORCE_HC_UP_MMA");if(no)env.insert("DS4_Q8_NO_HC_UP_MMA");
  bool use=capable&&!no&&(tier==0||tier==1)&&(force||(prefer&&tier==0));
  run(1,{use?"mma":"warp"});run(2,{use?"mma":"warp"});run(4,{"pair4"});
 }
 env.clear();out.tier=0;hc_up_prefer_mma=true;available=true;
 run(3,{"pair4"});env.insert("DS4_QWEN4EXP_WIDE_VERIFY_R2");run(3,{"warp","warp"});
 env.insert("DS4_Q8_NO_HC_WARP_PAIR");run(1,{"pair2"});run(2,{"pair2"});run(3,{"pair2"});
 env.clear();launch_error=1;run(1,{"mma"},0);env.insert("DS4_Q8_NO_HC_UP_MMA");run(2,{"warp"},0);
 assert(!hc_up_exact_use(0,0)&&!hc_up_exact_use(0,3));
 printf("PASS %u actual branch scenarios: device, widths, force/optouts, original fallback and CUDA error\n",tests);
}
'''
with tempfile.TemporaryDirectory(prefix='hc-up-dispatch-') as d:
 d=Path(d);p=d/'test.cpp';e=d/'test';p.write_text(code)
 subprocess.run(['c++','-std=c++17','-O2','-fsanitize=undefined','-fno-sanitize-recover=all',str(p),'-o',str(e)],check=True)
 subprocess.run([str(e)],check=True)
