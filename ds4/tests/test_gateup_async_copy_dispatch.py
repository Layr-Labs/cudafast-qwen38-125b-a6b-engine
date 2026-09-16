"""Compile the actual public split-gate/up branch against launch stubs."""
from pathlib import Path
import subprocess
import tempfile

root=Path(__file__).resolve().parents[1]
s=(root/'ds4_cuda_qwen4exp.cu').read_text()
s=s[s.index('extern "C" int ds4_gpu_qwen4exp_routed_moe_tensor('):]
a=s.index('    const int use_mma =');b=s.index('\n\n',a);eligibility=s[a:b]
a=s.index('    else if ((n_tokens <= 2u || wide_verify)');i=s.index('{',a)+1;depth=1
while depth:
    depth+=(s[i]=='{')-(s[i]=='}');i+=1
body=s[a:i].replace('    else if (','    if (',1)
code=r'''
#include <cassert>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <set>
#include <string>
#include <iostream>
static std::set<std::string> env;
static const char *probe_getenv(const char *s){return env.count(s)?(!strcmp(s,"DS4_GATEUP_COOP")?"0":"1"):nullptr;}
#define getenv probe_getenv
constexpr unsigned QW_MMA_BM=32,QW_MMA_G=4,DS4_QWEN4EXP_TY_q4_K=12,DS4_QWEN4EXP_TY_q6_K=14;
constexpr unsigned QW_GU_COOP_ROWS=4,QW_GU_COOP_GROUPS=80,QW_GU_COOP_ROW_U4=90,DS4_GATEUP_COOP_BUILD=1;
static bool qwen4exp_gateup_copy_prefer=true;
static int qwen4exp_gateup_copy_device=0,g_n_gpus=1;
struct GPU{int device_id;};static GPU g_gpu[1]={{0}};
struct Slab{uint64_t row_bytes=1440,expert_bytes=921600;unsigned type=12;};
struct Tensor{void *ptr=nullptr;};
struct Scratch{int8_t *xq=(int8_t *)0x300000;float *xs=nullptr;int32_t *xsum=nullptr,*pairs=nullptr,*counts=nullptr,*offsets=nullptr;};
struct dim3{unsigned x,y,z;dim3(unsigned a,unsigned b,unsigned c):x(a),y(b),z(c){}};
static unsigned gx,gy,bx,launches;static int path;
static void record(dim3 g,unsigned b){gx=g.x;gy=g.y;bx=b;launches++;}
#define QWEN4EXP_LAUNCH_PDL(K,G,B,S,STREAM,...) do{record(G,B);(K)(__VA_ARGS__);}while(0)
template<int R,int T,bool V,unsigned P,bool C,class...A>static void qwen4exp_moe_gateup_split_kernel(A...){
 assert(R==2&&T==12&&((C&&V&&P==4)||(!C&&V&&P==1)||(!C&&!V&&P==4)));
 path=C?1:V?3:4;
}
template<int R,int T,bool V,unsigned P,bool C,class...A>static void qwen4exp_moe_gateup_async_kernel(A...){
 assert(R==2&&T==12&&V&&P==4&&C);path=2;
}
struct Args{unsigned nt=2,groups=80,mid=640,used=10,experts=512;Slab gate,up;uintptr_t g=0x100000,u=0x200000,x=0x300000;bool wide=false,special=true;int tile=2,tier=0;};
static void run(Args v){
 unsigned n_tokens=v.nt,xgroups=v.groups,mid_dim=v.mid,mid_token_stride=v.mid*10+16,n_expert_used=v.used,n_total_expert=v.experts,gu_rows=20;
 int tile=v.tile,logical_tier=v.tier;bool wide_verify=v.wide,specialize=v.special;
 Slab *gate_slab=&v.gate,*up_slab=&v.up;const char *gate=(const char *)v.g,*up=(const char *)v.u;
 Tensor mt,wt;Tensor *mid=&mt,*weights=&wt;Scratch sc;sc.xq=(int8_t *)v.x;int32_t *gu_active=nullptr;
'''+eligibility+r'''
 if(use_mma)return;
'''+body+r'''
}
int main(){
 unsigned checks=0;
 auto check=[&](int want,Args v=Args{}){path=0;launches=0;run(v);assert(path==want&&launches==(want?1u:0u));
  if(want){unsigned rows=want==3?1:4;assert(gx==(v.mid+rows-1)/rows&&gy==20&&bx==rows*64);}checks++;};
 check(2);Args v;v.nt=1;check(2,v);
 v=Args{};v.used=3;check(1,v);v=Args{};v.experts=20;check(1,v);
 qwen4exp_gateup_copy_prefer=false;check(1);qwen4exp_gateup_copy_prefer=true;
 qwen4exp_gateup_copy_device=-1;check(1);qwen4exp_gateup_copy_device=0;
 g_gpu[0].device_id=1;check(1);g_gpu[0].device_id=0;
 g_n_gpus=2;check(1);g_n_gpus=1;
 v=Args{};v.tier=1;check(1,v);
 for(unsigned nt:{3u,4u,5u,8u,64u}){v=Args{};v.nt=nt;check(0,v);v.wide=true;check(nt<8?1:0,v);}
 for(unsigned md:{4u,7u,128u,639u}){v=Args{};v.mid=md;check(1,v);}
 for(unsigned offset:{2u,4u,8u}){v=Args{};v.g+=offset;check(3,v);v=Args{};v.u+=offset;check(3,v);v=Args{};v.x+=offset;check(4,v);}
 for(unsigned which=0;which<2;which++){
  v=Args{};(which?v.up:v.gate).row_bytes+=16;check(3,v);
  v=Args{};(which?v.up:v.gate).expert_bytes+=8;check(3,v);
  v=Args{};(which?v.up:v.gate).expert_bytes+=16;check(1,v);
  v=Args{};(which?v.up:v.gate).type=8;check(0,v);
 }
 for(unsigned groups:{1u,79u,81u}){v=Args{};v.groups=groups;check(3,v);}
 v=Args{};v.special=false;check(0,v);v=Args{};v.tile=4;check(0,v);
 env.insert("DS4_GATEUP_COOP");check(3);env.clear();
 env.insert("DS4_QWEN4EXP_NO_SPLIT_GATEUP");check(0);env.clear();
 env.insert("DS4_QWEN4EXP_NO_SPLIT_VECTOR");check(4);env.clear();
 env.insert("DS4_QWEN4EXP_NO_GATEUP_ASYNC_COPY");check(1);
 env.insert("DS4_QWEN4EXP_FORCE_GATEUP_ASYNC_COPY");check(1);env.clear();
 env.insert("DS4_QWEN4EXP_FORCE_GATEUP_ASYNC_COPY");
 qwen4exp_gateup_copy_prefer=false;qwen4exp_gateup_copy_device=-1;check(2);
 v=Args{};v.mid=7;check(2,v);v=Args{};v.nt=3;v.wide=true;check(2,v);
 v=Args{};v.g+=2;check(3,v);v=Args{};v.groups=79;check(3,v);
 std::cout<<checks<<" actual gate/up dispatch scenarios PASS\n";
}
'''
with tempfile.TemporaryDirectory(prefix='gateup-dispatch-') as d:
    d=Path(d);p=d/'test.cpp';e=d/'test';p.write_text(code)
    subprocess.run(['c++','-std=c++17','-O2','-fsanitize=undefined','-fno-sanitize-recover=all',str(p),'-o',str(e)],check=True)
    subprocess.run([str(e)],check=True)
