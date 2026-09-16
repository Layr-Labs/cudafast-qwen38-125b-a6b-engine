"""Actual short-pending eligibility/launch prefix and graph-island controls.
Host facades replace CUDA launches and backend operations; no GPU API claim.
"""
from pathlib import Path
import re,subprocess,tempfile
root=Path(__file__).resolve().parents[1];s=(root/'ds4_cuda_qwen4exp.cu').read_text();g=(root/'ds4_qwen4exp_graph.inc').read_text()
def body(s,mark):
 a=s.index(mark);b=s.index('{',a);e=b+1;d=1
 while d:d+=(s[e]=='{')-(s[e]=='}');e+=1
 return s[a:e]
api=body(s,'static int qwen4exp_hc_mixer_fused_cuda(')
end='    if (!cuda_ok(cudaGetLastError(), "qwen4exp_hc_norm_quant launch")) return 0;'
api=api[:api.index(end)+len(end)]+'\n    return 1;\n}'
api,n=re.subn(r'(qwen4exp_\w+)(?:<[^>]+>)?<<<.*?>>>(\s*)\(',lambda m:'spy("'+m[1]+'", ',api,flags=re.S);assert n==4,n
api=api.replace('QWEN4EXP_LAUNCH_PDL(\n                    (qwen4exp_hc_norm_quant_kernel<1>),','spy("pdl_norm",')
pre=r'''
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <initializer_list>
struct ds4_gpu_tensor{void*ptr;uint64_t bytes;int tier;};
struct ds4_gpu_qwen4exp_slab{const void*map;uint64_t map_size,offset,row_bytes;uint32_t type;};
struct dim3{dim3(unsigned,unsigned,unsigned){}};
#define QWEN4EXP_HC_THREADS 256u
#define QWEN4EXP_HC_MAX_STREAMS 16u
#define QWEN4EXP_HC_UP_MIX_NT 4u
#define QWEN4EXP_HC_FUSE_MIX_MIN_ROWS 48u
static int g_n_gpus=1,fail_launch,fail_resolve,staged=1;static std::vector<std::string>calls;
static int ds4_tensor_device_idx(const ds4_gpu_tensor*t){return t->tier;}
static int qwen4exp_hc_staged_ok(unsigned emb,unsigned){return staged&&emb==2560;}
static int ds4_qwen4exp_hc_wide_off(){return 0;}static int ds4_cuda_qwen4exp_q8_mma_active(unsigned){return 0;}
static const char*cuda_resolve_weight_ptr(const void*p,uint64_t o,uint64_t,int,const char*){return fail_resolve?nullptr:(const char*)((uintptr_t)p+o);}
static int cuda_decode_stream(){return 9;}static int cudaGetLastError(){return fail_launch&&(int)calls.size()==fail_launch;}static int cuda_ok(int e,const char*){return !e;}
template<class...T>static void spy(const char*n,T...){calls.push_back(n);}
template<class...T>static void qwen4exp_hc_norm_quant_inject_launch(T...){assert(false);}
'''
source=pre+body(s,'static int qwen4exp_hc_ranges_disjoint(')+'\n'+api+r'''
int main(int argc,char**){bool disabled=argc>1;if(disabled)setenv("DS4_QWEN4EXP_NO_HC_SHORT_PENDING","1",1);else unsetenv("DS4_QWEN4EXP_NO_HC_SHORT_PENDING");unsigned cases=0;
 ds4_gpu_tensor t[8],saved[8];uint64_t z[]={2*2560*4,2*4*4,2*10240*4,2*320*4,2*10240*4,2*10240*4,2*2560*4,2*4*4};
 for(unsigned i=0;i<8;i++)saved[i]={(void*)(uintptr_t)(0x10000000ull+i*0x10000000ull),z[i],0};
 ds4_gpu_qwen4exp_slab w[4],sw[4];for(unsigned i=0;i<4;i++)sw[i]={(void*)(uintptr_t)(0x200000000ull+i*0x10000000ull),10000000,0,0,0};
 auto reset=[&](){for(unsigned i=0;i<8;i++)t[i]=saved[i];for(unsigned i=0;i<4;i++)w[i]=sw[i];calls.clear();fail_launch=fail_resolve=0;staged=1;};
 auto run=[&](unsigned rows=2,unsigned emb=2560,unsigned hc=4,bool pending=true){cases++;return qwen4exp_hc_mixer_fused_cuda(&t[0],&t[1],&t[2],&t[3],&t[4],&t[5],&w[0],&w[1],&w[2],&w[3],emb,hc,320,rows,1e-6f,0.f,1,pending?&t[6]:nullptr,pending?&t[7]:nullptr);};
 auto expect=[&](bool fused){if(fused&&!disabled){assert(calls.size()==1&&calls[0]=="qwen4exp_hc_norm_quant_pending_short_kernel");}else{assert(calls.size()==2&&calls[0]=="qwen4exp_hc_inject_kernel");assert(calls[1]=="pdl_norm"||calls[1]=="qwen4exp_hc_norm_quant_kernel");}};
 for(unsigned rows:{1u,2u}){reset();assert(run(rows)==1);expect(true);}
 reset();t[7]=t[1];assert(run()==1);expect(true); // future output safely reuses pending head
 reset();assert(run(2,2560,4,false)==1);assert(calls.size()==1&&calls[0]=="pdl_norm");
 reset();staged=0;assert(run()==1);expect(false);
 reset();assert(run(2,2560,2)==1);expect(false);
 reset();assert(run(2,2304,4)==1);expect(false);
 // Four new early-write/source overlap classes plus scratch/weight classes.
 for(unsigned which=0;which<7;which++){reset();switch(which){case 0:t[6].ptr=(char*)t[5].ptr+4;break;case 1:t[7].ptr=(char*)t[5].ptr+4;break;case 2:t[2].ptr=(char*)t[5].ptr+4;break;case 3:w[0].map=(char*)t[5].ptr+4;break;case 4:t[6].ptr=(char*)t[2].ptr+4;break;case 5:t[7].ptr=(char*)t[2].ptr+4;break;case 6:w[0].map=(char*)t[2].ptr+4;break;}assert(run()==1);expect(false);}
 reset();t[6].ptr=(char*)t[5].ptr+t[5].bytes;assert(run()==1);expect(true);
 reset();t[5].ptr=(void*)(UINTPTR_MAX-7);assert(run()==1);expect(false);
 reset();t[2].bytes=1;assert(run()==-1);assert(calls.empty());
 reset();t[6].bytes=1;assert(run()==-1);assert(calls.empty());
 reset();t[7].tier=1;assert(run()==-1);assert(calls.empty());
 reset();fail_resolve=1;assert(!run());assert(calls.empty());
 reset();fail_launch=1;assert(!run());assert(calls.size()==1);
 if(disabled){reset();fail_launch=2;assert(!run());assert(calls.size()==2);}
 printf("PASS actual HC launch-prefix controls: %u cases diagnostic=%d\n",cases,disabled);
}
'''
# Actual graph helpers and island body with per-operation spies.
graph=body(g,'static bool qwen4exp_graph_residual_impl(')+'\n'+body(g,'static bool qwen4exp_graph_residual(')+'\n'+body(g,'static bool qwen4exp_graph_layer_island_encode(')
gsource=r'''
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
struct ds4_gpu_tensor{};struct ds4_model{};struct ds4_tensor{};struct ds4_gpu_qwen4exp_slab{int unused;};
struct ds4_qwen4exp_session{bool hc_pending=false;ds4_gpu_tensor*hyper,*mixed,*inject,*normed,*lowrank,*wide,*block_out;};
struct ds4_qwen4exp_layer_weights{const ds4_tensor *hc_attn_norm,*hc_attn_down,*hc_attn_up,*hc_attn_inject,*hc_ffn_norm,*hc_ffn_down,*hc_ffn_up,*hc_ffn_inject;float hc_attn_norm_offset=0,hc_ffn_norm_offset=0;bool is_full_attention=false;};
#define QW_SLAB(m,t) ds4_gpu_qwen4exp_slab{}
#define DS4_N_EMBD 2560
#define DS4_N_HC 4
#define DS4_N_HC_LOWRANK 320
#define DS4_HC_EPS 1e-6f
#define QW_TRACE(x) ((void)0)
static std::vector<std::string>events;static int fail_at=0,hb=0;
static int op(const char*n){events.push_back(n);return (int)events.size()!=fail_at;}
static int qw_hb_time_on(){return hb;}static bool qwen4exp_hc_defer_ok(uint32_t n){return n>=48;}
template<class...T>static int ds4_gpu_qwen4exp_hc_inject_tensor(T...){return op("inject");}
template<class...T>static int ds4_gpu_qwen4exp_hc_mixer_pending_tensor(T...){return op("pending_mix");}
template<class...T>static int ds4_gpu_qwen4exp_hc_mixer_tensor(T...){return op("mix");}
template<class...T>static bool qwen4exp_graph_gdn_block(T...){return op("gdn");}
template<class...T>static bool qwen4exp_graph_moe_block(T...){return op("moe");}
'''+graph+r'''
int main(int argc,char**argv){bool disabled=argc>1&&argv[1][0]=='d',timing=argc>1&&argv[1][0]=='t';if(disabled)setenv("DS4_QWEN4EXP_NO_HC_SHORT_PENDING","1",1);if(timing)setenv("DS4_QWEN4EXP_TIME_SLICES","1",1);ds4_qwen4exp_session s{};ds4_qwen4exp_layer_weights l{};ds4_model m;unsigned count=0;
#if !defined(DS4_NO_GPU) && !defined(__APPLE__) && !defined(DS4_ROCM_BUILD)
 bool cuda=true;
#else
 bool cuda=false;
#endif
 for(unsigned rows:{1u,2u,3u})for(unsigned heart:{0u,1u}){hb=heart;events.clear();fail_at=0;s.hc_pending=false;bool fused=cuda&&!disabled&&!timing&&!hb&&rows<=2;
  assert(qwen4exp_graph_layer_island_encode(&s,&l,&m,0,rows,1));assert(!s.hc_pending);
  std::vector<std::string>expected=fused?std::vector<std::string>{"pending_mix","moe","inject"}:std::vector<std::string>{"inject","mix","moe","inject"};assert(events==expected);
  for(unsigned failure=1;failure<=expected.size();failure++){events.clear();fail_at=failure;s.hc_pending=false;assert(!qwen4exp_graph_layer_island_encode(&s,&l,&m,0,rows,1));assert(events.size()==failure);assert(!s.hc_pending);}
  count++;
 }
 hb=0;fail_at=0;events.clear();s.hc_pending=false;assert(qwen4exp_graph_layer_island_encode(&s,&l,&m,0,48,1));assert((events==std::vector<std::string>{"pending_mix","moe"}));assert(s.hc_pending);
 events.clear();s.hc_pending=false;assert(qwen4exp_graph_layer_island_encode(&s,&l,&m,0,2,0));assert((events==std::vector<std::string>{"mix","gdn"}));assert(!s.hc_pending);
 printf("PASS actual HC graph-island controls: %u width/heartbeat cases plus every failure stage; local pending only, terminal inject and prefill retained\n",count);
}
'''
with tempfile.TemporaryDirectory(prefix='hc-short-controls-') as t:
 p=Path(t)
 for name,code in [('dispatch',source),('graph',gsource)]:
  (p/'test.cpp').write_text(code)
  for backend in ([None] if name=='dispatch' else [None,'DS4_NO_GPU','__APPLE__','DS4_ROCM_BUILD']):
   subprocess.run(['c++','-O2','-std=c++17']+(['-D'+backend] if backend else [])+[str(p/'test.cpp'),'-o',str(p/'test')],check=True)
   for arg in (['','disabled'] if name=='dispatch' else ['','disabled','timing']):subprocess.run([str(p/'test')]+([arg] if arg else []),check=True)
