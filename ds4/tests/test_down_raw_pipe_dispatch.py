"""Exercise the actual prefill eligibility and down launch block with CUDA stubs."""
from pathlib import Path
import re,subprocess,tempfile
root=Path(__file__).resolve().parents[1]
s=(root/'ds4_cuda_qwen4exp.cu').read_text()
pos=s.index('extern "C" int ds4_gpu_qwen4exp_routed_moe_tensor(')
s=s[pos:]
a=s.index('    const int use_mma =');b=s.index('    const int moe_epilogue',a)
eligibility=s[a:b]
a=s.index('    if (down_mma) {');i=s.index('{',a);depth=1;i+=1
while depth:
 depth+=(s[i]=='{')-(s[i]=='}');i+=1
body=s[a:i]
body=re.sub(r'<<<.*?>>>','',body,flags=re.S)
code=r'''
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <set>
#include <string>
#include <vector>
static std::set<std::string> env;
static const char *probe_getenv(const char *s){return env.count(s)?"1":nullptr;}
#define getenv probe_getenv
constexpr unsigned QW_MMA_BM=32,QW_MMA_G=4,QW_DOWN_MMA_BM=64,QW_DOWN_MMA_THREADS=128;
constexpr unsigned DS4_QWEN4EXP_TY_q5_1=7,DS4_QWEN4EXP_TY_q8_0=8,DS4_QWEN4EXP_TY_q6_K=14;
struct Tensor{void *ptr=nullptr;};
struct Slab{uint64_t expert_bytes=2560u*480,row_bytes=480;unsigned type=7;};
struct Scratch{int8_t *mq=nullptr;float *ms=nullptr;int32_t *msum=nullptr,*pairs=nullptr,*counts=nullptr,*offsets=nullptr;};
static int path,combines;static std::string fail_at,last;
static bool cuda_ok(int e,const char *){return e==0;}
static int cudaGetLastError(){return last==fail_at?1:0;}
template<int Type,bool Wide,class...A>static void qwen4exp_moe_down_mma_kernel(A...){
 path=Type==7?(Wide?1:2):Type==8?3:4;last="tile";
}
template<int Type,bool Wide>static void qwen4exp_moe_down_raw_pipe_kernel(float *,const char *p,
 const int8_t *,const float *,const int32_t *,const int32_t *,const int32_t *,const int32_t *,const int32_t *,
 uint64_t eb,uint64_t rb,unsigned type,unsigned groups,unsigned out,unsigned dq){
 assert(Type==7 && Wide && type==7 && dq==1 && (groups&3u)==0);
 assert((((uintptr_t)p)|eb|rb)%16==0 && out%64==0);path=5;last="tile";
}
template<class...A>static void qwen4exp_moe_down_combine_grid_kernel(A...){combines++;last="combine";}
template<class...A>static void qwen4exp_moe_down_combine_kernel(A...){combines++;last="combine";}
static int run(unsigned n_tokens,unsigned mid_dim,unsigned out_dim,Slab dn,uintptr_t address,
 bool specialize=true,unsigned gate_type=12,unsigned xgroups=80){
 unsigned mgroups=mid_dim/32,n_expert_used=10,n_total_expert=512;
 Tensor ot,pt,st;Tensor *out=&ot,*down_partial=&pt,*selected=&st;
 Slab gu;gu.type=gate_type;Slab *gate_slab=&gu,*up_slab=&gu,*down_slab=&dn;
 const char *down=(const char *)address;Scratch sc;int32_t *gu_active=nullptr;
 unsigned threads=256;
'''+eligibility+body+r'''
 return 2;
}
int main(){
 unsigned checks=0;
 auto check=[&](int want,int result,unsigned nt=1024,unsigned md=640,unsigned od=2560,
     Slab d=Slab{},uintptr_t addr=0x100000,bool special=true,unsigned gt=12,unsigned xg=80){
  path=combines=0;last.clear();
  assert(run(nt,md,od,d,addr,special,gt,xg)==result);assert(path==want);
  assert(combines==(result==1 || fail_at=="combine"?1:0));checks++;
 };
 for(unsigned nt: {1u,2u,3u,4u,7u}) check(0,2,nt);
 for(unsigned nt: {8u,63u,64u,65u,256u,1024u}) check(5,1,nt);
 for(unsigned md: {128u,256u,640u,1024u}) {Slab d;d.row_bytes=md/32*24;d.expert_bytes=2560*d.row_bytes;check(5,1,64,md,2560,d);}
 for(unsigned md: {32u,96u,608u,672u}) {Slab d;d.row_bytes=md/32*24;d.expert_bytes=2560*d.row_bytes;check(1,1,64,md,2560,d);}
 for(uintptr_t off: {2u,4u,8u}) check(1,1,1024,640,2560,Slab{},0x100000+off);
 {Slab d;d.row_bytes+=8;check(1,1,1024,640,2560,d);d=Slab{};d.expert_bytes+=8;check(1,1,1024,640,2560,d);}
 for(const char *key:{"DS4_QWEN4EXP_NO_DOWN_RAW_PIPE","DS4_QWEN4EXP_NO_DOWN_DQ"}){env.insert(key);check(1,1);env.clear();}
 env.insert("DS4_QWEN4EXP_NO_Q51_WIDE_LOAD");check(2,1);env.clear();
 for(unsigned dt:{7u,8u,14u}){Slab d;d.type=dt;check(dt==7?5:dt==8?3:0,dt==14?2:1,1024,640,2560,d);}
 check(4,1,1024,640,2560,Slab{},0x100000,false);
 check(0,2,1024,640,2561);check(0,2,1024,640,2560,Slab{},0x100000,true,14);
 check(0,2,1024,640,2560,Slab{},0x100000,true,12,79);
 env.insert("DS4_QWEN4EXP_NO_MMA");check(0,2);env.clear();
 env.insert("DS4_QWEN4EXP_NO_COMBINE_GRID");check(5,1);env.clear();
 fail_at="tile";check(5,0);fail_at="combine";check(5,0);fail_at.clear();
 std::cout<<checks<<" actual prefill dispatch scenarios PASS\n";
}
'''
with tempfile.TemporaryDirectory(prefix='down-raw-dispatch-') as d:
 d=Path(d);p=d/'test.cpp';e=d/'test';p.write_text(code)
 subprocess.run(['c++','-std=c++17','-O2','-fsanitize=undefined','-fno-sanitize-recover=all',str(p),'-o',str(e)],check=True)
 subprocess.run([str(e)],check=True)
