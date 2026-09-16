"""Actual MoE/HC graph statements with host backend facades, not GPU execution.
Checks status ownership, one terminal apply, diagnostics and failure short-circuit.
"""
from pathlib import Path
import subprocess, tempfile
s=(Path(__file__).resolve().parents[1]/'ds4_qwen4exp_graph.inc').read_text()
def body(mark):
 a=s.index(mark);b=s.index('{',a);e=b+1;d=1
 while d:d+=(s[e]=='{')-(s[e]=='}');e+=1
 return s[a:e]
code=r'''
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
struct ds4_gpu_tensor{};struct ds4_tensor{};struct ds4_model{};
struct ds4_gpu_qwen4exp_slab{int unused;};
struct ds4_qwen4exp_session{bool hc_pending=false;ds4_gpu_tensor *moe_logits,*moe_selected,*moe_weights,*mixed,*block_out,*moe_mid,*moe_down_partial,*shexp_mid,*shexp_gate,*hyper,*inject;};
struct ds4_qwen4exp_layer_weights{ds4_tensor *ffn_gate_inp,*ffn_gate_exps,*ffn_up_exps,*ffn_down_exps,*ffn_gate_inp_shexp,*ffn_gate_shexp,*ffn_up_shexp,*ffn_down_shexp,*hc_attn_norm,*hc_attn_down,*hc_attn_up,*hc_attn_inject,*hc_ffn_norm,*hc_ffn_down,*hc_ffn_up,*hc_ffn_inject;float hc_attn_norm_offset=0,hc_ffn_norm_offset=0;bool is_full_attention=false;};
#define QW_MAP(m,t) nullptr
#define QW_MAPSZ(m,t) 0
#define QW_OFF(t) 0
#define QW_SLAB(m,t) ds4_gpu_qwen4exp_slab{}
#define DS4_N_EMBD 2560
#define DS4_N_EXPERT 512
#define DS4_N_EXPERT_USED 10
#define DS4_N_FF_EXP 640
#define DS4_N_FF_SHEXP 640
#define DS4_N_HC 4
#define QW_MOE_TICK(x) ((void)0)
#define QW_TRACE(x) ((void)0)
static int qw_moe_in_head_block=0;static uint64_t qw_moe_stage_calls[2]={};
static std::vector<std::string> events;static int failure=0,status=2,hb=0,applies=0;
static int qw_hb_time_on(){return hb;}static uint64_t qw_hb_now_ns(){return 0;}
static bool op(const char*n){events.push_back(n);return (int)events.size()!=failure;}
template<class...T>static int ds4_qwen4exp_matmul_f32(T...){return op("router");}
template<class...T>static int ds4_gpu_qwen4exp_router_select_tensor(T...){return op("select");}
template<class...T>static int ds4_gpu_qwen4exp_routed_moe_tensor(T...){return op("routed");}
template<class...T>static int ds4_gpu_qwen4exp_shared_expert_preq_tensor(T...){return op("shared");}
template<class...T>static int ds4_gpu_qwen4exp_shared_expert_preq_hc_tensor(T...){if(!op("shared_hc"))return 0;if(status==2)applies++;return status;}
template<class...T>static int ds4_gpu_qwen4exp_hc_inject_tensor(T...){if(!op("inject"))return 0;applies++;return 1;}
static bool qwen4exp_hc_defer_ok(unsigned n){return n>=48;}
template<class...T>static bool qwen4exp_graph_residual(ds4_qwen4exp_session*s,T...){s->hc_pending=false;return op("mix");}
template<class...T>static bool qwen4exp_graph_gdn_block(T...){return op("gdn");}
'''+body('static bool qwen4exp_graph_moe_block_impl(')+'\n'+body('static bool qwen4exp_graph_moe_block(')+'\n'+body('static bool qwen4exp_graph_layer_island_encode(')+r'''
int main(int argc,char**argv){const std::string mode=argc>1?argv[1]:"default";
 if(mode=="disabled")setenv("DS4_QWEN4EXP_NO_SHARED_HC_INJECT","1",1);
 if(mode=="timing")setenv("DS4_QWEN4EXP_TIME_SLICES","1",1);
 ds4_qwen4exp_session s{};ds4_qwen4exp_layer_weights l{};ds4_model m;unsigned cases=0,faults=0;
#if !defined(DS4_NO_GPU) && !defined(__APPLE__) && !defined(DS4_ROCM_BUILD)
 const bool cuda=true;
#else
 const bool cuda=false;
#endif
 auto reset=[&](){events.clear();failure=0;applies=0;s.hc_pending=false;};
 for(unsigned rows:{0u,1u,2u,3u,48u})for(int heartbeat:{0,1})for(int result:{0,1,2}){
  hb=heartbeat;status=result;const bool native=cuda&&mode=="default"&&!hb&&rows>=1&&rows<=2;
  reset();bool done=true;bool ok=qwen4exp_graph_moe_block_impl(&s,&l,&m,0,rows,&done);
  assert(ok==(!native||result!=0));assert(done==(native&&result==2));
  assert((events==std::vector<std::string>{"router","select","routed",native?"shared_hc":"shared"}));
  assert(applies==(native&&result==2));cases++;
  // A stale true from an earlier call never survives any failed stage.
  for(int at=1;at<=4;at++){reset();failure=at;done=true;assert(!qwen4exp_graph_moe_block_impl(&s,&l,&m,0,rows,&done));assert(!done);assert(events.size()==(unsigned)at);assert(!applies);faults++;}
  // Legacy wrapper has no authority to apply terminal injection.
  reset();assert(qwen4exp_graph_moe_block(&s,&l,&m,0,rows));assert(events.back()=="shared"&&!applies);cases++;
  if(rows==0)continue; // Backend facade accepts zero; this only proves routing above.
  reset();ok=qwen4exp_graph_layer_island_encode(&s,&l,&m,0,rows,1);
  std::vector<std::string> expected;
  if(rows<48)expected.push_back("inject");expected.push_back("mix");
  for(auto e:{"router","select","routed"})expected.push_back(e);
  expected.push_back(native?"shared_hc":"shared");
  if((!native||result==1)&&rows<48)expected.push_back("inject");
  assert(ok==(!native||result!=0));assert(events==expected);
  assert(applies==(rows>=48?0:(native&&result==0?1:2)));
  assert(s.hc_pending==(rows>=48));cases++;
  // Every actual operation failure stops without replaying shared work or injection.
  for(unsigned at=1;at<=expected.size();at++){reset();failure=at;assert(!qwen4exp_graph_layer_island_encode(&s,&l,&m,0,rows,1));assert(events.size()==at);assert(!s.hc_pending);faults++;}
 }
 reset();hb=0;l.is_full_attention=false;assert(qwen4exp_graph_layer_island_encode(&s,&l,&m,0,2,0));assert((events==std::vector<std::string>{"mix","gdn"})&&!applies);cases++;
 printf("PASS actual shared-down HC graph controls: %u cases, %u injected failure stages, mode=%s; status/reset/single apply/backend/prefill\n",cases,faults,mode.c_str());
}
'''
with tempfile.TemporaryDirectory(prefix='shared-down-hc-graph-') as t:
 p=Path(t);(p/'test.cpp').write_text(code)
 for backend in [None,'DS4_NO_GPU','__APPLE__','DS4_ROCM_BUILD']:
  subprocess.run(['c++','-O2','-std=c++17']+(['-D'+backend] if backend else [])+[str(p/'test.cpp'),'-o',str(p/'test')],check=True)
  for mode in ['default','disabled','timing']:
   r=subprocess.run([str(p/'test'),mode],capture_output=True,text=True)
   if r.returncode: print(r.stderr);r.check_returncode()
   print(r.stdout,end='')
