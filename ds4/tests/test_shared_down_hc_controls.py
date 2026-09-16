"""Actual shared-HC validation prefix and final panel-arm host controls.
CUDA queries/resolution/launches are facades. This is not a full shared API or
CUDA integration test; the intervening gate/up/quantizer and earlier MMA/stage
routing remain outside these extracted slices.
"""
from pathlib import Path
import re,subprocess,tempfile
s=(Path(__file__).resolve().parents[1]/'ds4_cuda_qwen4exp.cu').read_text()
def body(name):
 m=re.search(r'^.*\b'+name+r'\([^;]*?\)\s*\{',s,re.M);assert m,name
 a=m.start();e=s.index('{',a)+1;d=1
 while d:d+=(s[e]=='{')-(s[e]=='}');e+=1
 return s[a:e]
f=body('qwen4exp_shared_expert_impl')
prefix=f[:f.index('    cudaStream_t stream')]+ '\nreturn 1;\n}'
# Retain the exact final eligibility and launch/error segment. Setup variables
# stand for the already-selected plain panel arm, not a simulated MMA policy.
start=f.index('    const uint64_t sd_panel')
end=f.index('#define QWEN4EXP_SH_DOWN_IMPL',start)
tail=f[start:end]+'return 1;'
# Existing actual specialized row predicates, including all their diagnostics.
a=f.index('    const bool single_q8');b=f.index('    const uint32_t tiles',a)
preds=f[a:b]
wrappers=body('ds4_gpu_qwen4exp_shared_expert_preq_tensor')+body('ds4_gpu_qwen4exp_shared_expert_preq_hc_tensor')
source=r'''
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cassert>
struct ds4_gpu_tensor { void*ptr; uint64_t bytes; int tier; };
struct ds4_gpu_qwen4exp_slab { void*map; uint64_t offset,row_bytes; uint32_t type; };
constexpr int DS4_QWEN4EXP_TY_q8_0=8;
constexpr uint64_t QW_DOWN_PANEL_MAX_BYTES=16384;
static unsigned queries,resolves,launches,cases; static int device,query_error,resolve_fail,launch_error;
int ds4_tensor_device_idx(const ds4_gpu_tensor*t){return t->tier;}
int cudaGetDevice(int*t){queries++;*t=device;return query_error;}
int cuda_current_tier(){return device;}
int cuda_ok(int e,const char*){return e==0;}
int cudaGetLastError(){return launch_error;}
bool cuda_qwen4exp_moe_type_supported(unsigned t){return t==0||t==8;}
const char*cuda_resolve_weight_ptr(const void*p,uint64_t o,uint64_t,int,const char*){resolves++;return resolves==(unsigned)resolve_fail?nullptr:(const char*)((uintptr_t)p+o);}
int qwen4exp_moe_tile(unsigned n){return n<=4?2:8;}
#define QWEN4EXP_LAUNCH_PDL(...) do { launches++; } while(0)
'''+prefix+wrappers+r'''
struct Fixture{
 ds4_gpu_tensor out{(void*)0x100000,20480,0},mid{(void*)0x200000,5120,0},gs{(void*)0x300000,8,0},x{(void*)0x400000,20480,0},h{(void*)0x500000,81920,0},inj{(void*)0x600000,32,0};
 ds4_gpu_qwen4exp_slab r{(void*)0x1000000,0,10240,0},g{(void*)0x2000000,0,2720,8},u{(void*)0x3000000,0,2720,8},d{(void*)0x4000000,0,680,8};
 unsigned rows=2,id=2560,md=640,od=2560; bool legacy=false,nullh=false,nulli=false;
 int run(){return legacy?ds4_gpu_qwen4exp_shared_expert_preq_tensor(&out,&mid,&gs,&r,&g,&u,&d,id,md,od,&x,rows,1):ds4_gpu_qwen4exp_shared_expert_preq_hc_tensor(&out,&mid,&gs,&r,&g,&u,&d,id,md,od,&x,rows,1,nullh?nullptr:&h,nulli?nullptr:&inj);}
 int select(){
 auto*out=&this->out;auto*mid=&this->mid;auto*gate_scale=&gs;auto*x=&this->x;
 auto*terminal_hyper=nullh?nullptr:&h;auto*terminal_inject=nulli?nullptr:&inj;
 auto*gate_slab=&g;auto*up_slab=&u;auto*down_slab=&d;
 unsigned in_dim=id,mid_dim=md,out_dim=od,n_tokens=rows;
 bool specialize_shared=getenv("DS4_QWEN4EXP_GENERIC_EXPERTS")==nullptr;
 const char*router=(char*)r.map;const char*gate=(char*)g.map;const char*up=(char*)u.map;const char*down=(char*)d.map;
 char*base=(char*)0x700000;uint64_t xq_bytes=6400,mq_bytes=1600;
 int8_t*mq=(int8_t*)(base+xq_bytes);
'''+preds+tail+r'''
 }
};
void reset(){queries=resolves=launches=0;device=query_error=resolve_fail=launch_error=0;}
void ck(bool c){cases++;if(!c){fprintf(stderr,"FAIL case%u\n",cases);abort();}}
int main(){
 reset();Fixture f;ck(f.run()==1&&queries==1&&resolves==4);ck(f.select()==2&&launches==1);
 for(int k=0;k<16;k++){reset();Fixture a;
 switch(k){case0:break;
 case 1:a.nullh=true;break;case 2:a.nulli=true;break;case 3:a.h.ptr=nullptr;break;case 4:a.inj.ptr=nullptr;break;
 case 5:a.h.bytes--;break;case 6:a.inj.bytes--;break;case 7:a.h.ptr=(void*)0x500002;break;case 8:a.inj.ptr=(void*)0x600002;break;
 case 9:a.h.tier=1;break;case 10:a.inj.tier=1;break;case 11:device=1;break;case 12:query_error=1;break;
 case 13:a.rows=0;break;case 14:a.md=639;break;case 15:a.out.bytes--;break;}
 if(k)ck(a.run()==0&&launches==0&&resolves==0);
 }
 for(int k=1;k<=4;k++){reset();Fixture a;resolve_fail=k;ck(a.run()==0&&launches==0);}
 reset();{Fixture a;a.legacy=true;query_error=1;ck(a.run()==1&&queries==0);}
 reset();{Fixture a;launch_error=1;ck(a.select()==0&&launches==1);}
 // Every prohibited hyper view, each mutable view vs coefficient, pool, weights.
 for(int k=0;k<15;k++){reset();Fixture a;
 switch(k){case 0:a.h.ptr=a.out.ptr;break;case 1:a.h.ptr=(char*)a.mid.ptr+4;break;case 2:a.h.ptr=a.gs.ptr;break;case 3:a.h.ptr=a.x.ptr;break;case 4:a.h.ptr=a.inj.ptr;break;
 case 5:a.inj.ptr=a.out.ptr;break;case 6:a.inj.ptr=(char*)a.mid.ptr+4;break;case 7:a.inj.ptr=a.gs.ptr;break;
 case 8:a.h.ptr=(void*)0x700004;break;case 9:a.inj.ptr=(void*)0x700004;break;
 case 10:a.h.ptr=a.r.map;break;case 11:a.h.ptr=(char*)a.g.map+4;break;case 12:a.h.ptr=a.u.map;break;case 13:a.h.ptr=a.d.map;break;
 case 14:a.h.ptr=(void*)(UINTPTR_MAX-3);break;}
 ck(a.select()==1&&launches==0);
 }
 reset();{Fixture a;a.h.ptr=(char*)a.out.ptr+a.out.bytes;ck(a.select()==2&&launches==1);}
 for(const char*n:{"DS4_QWEN4EXP_GENERIC_EXPERTS","DS4_QWEN4EXP_MOE_R","DS4_QWEN4EXP_NO_SHARED_VECTOR","DS4_QWEN4EXP_NO_SHARED_DOWN_PANEL","DS4_QWEN4EXP_NO_SHARED_HC_INJECT"}){reset();Fixture a;setenv(n,"1",1);ck(a.select()==1&&launches==0);unsetenv(n);}
 reset();{Fixture a;a.rows=1;ck(a.select()==2&&launches==1);setenv("DS4_QWEN4EXP_NO_SHARED_R1","1",1);launches=0;ck(a.select()==1&&launches==0);unsetenv("DS4_QWEN4EXP_NO_SHARED_R1");}
 for(int k=0;k<7;k++){reset();Fixture a;switch(k){case 0:a.rows=3;break;case 1:a.rows=48;break;case 2:a.id=2592;break;case 3:a.md=672;break;case 4:a.od=2559;break;case 5:a.d.type=0;break;case 6:a.d.map=(void*)0x4000002;break;}ck(a.select()==1&&launches==0);}
 printf("PASS shared HC prefix/panel controls: %u cases; no GPU or intervening projections executed\n",cases);
}
'''
actual_limit=re.search(r'^#define QW_DOWN_PANEL_MAX_BYTES\s+(\d+)u?',s,re.M)
assert actual_limit and int(actual_limit[1])==16384
source=source.replace('printf("PASS shared HC', 'reset();{Fixture a;a.d.row_bytes=2048;ck(a.select()==2&&launches==1);a.d.row_bytes=2050;launches=0;ck(a.select()==1&&launches==0);}'+'printf("PASS shared HC')
source=source.replace('case0:break;','case 0:break;').replace('#include <cassert>','#include <cassert>\n#include <initializer_list>')
with tempfile.TemporaryDirectory(prefix='shared-hc-controls-') as t:
 p=Path(t);(p/'test.cpp').write_text(source)
 subprocess.run(['c++','-std=c++17','-O2',str(p/'test.cpp'),'-o',str(p/'test')],check=True)
 subprocess.run([str(p/'test')],check=True)
