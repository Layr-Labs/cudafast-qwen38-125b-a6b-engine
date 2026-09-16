"""Exercise the actual graph call site: one attention, one gate, errors stop."""
from pathlib import Path
import subprocess,tempfile
root=Path(__file__).resolve().parents[1];s=(root/'ds4_qwen4exp_graph.inc').read_text()
b=s.index('    const float scale = 1.0f / sqrtf((float)head_dim);',s.index('const bool indexed ='))
e=s.index('    if (!(gate_q8\n',b);body=s[b:e]
code=r'''
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
struct ds4_gpu_tensor{};
struct Session {ds4_gpu_tensor *qsa_out,*qsa_q,*qsa_k[1],*qsa_v[1],*d_pos,*qsa_split,*gdn_out_q8,*qsa_doubled,*qsa_gate;struct{unsigned n_ctx;}plan;};
constexpr unsigned DS4_N_HEAD=24,DS4_N_HEAD_KV=2,DS4_N_INDEXER_TOP_K=2048;
static bool q8on,doubledon;static int fuse_rc,attn_rc,gate_rc;static unsigned attempts,attentions,gates,kind,projection;
static bool qw_qsa_gate_q8(unsigned,unsigned){return q8on;}
static bool qw_qsa_gate_from_doubled(unsigned,unsigned,unsigned){return q8on&&doubledon;}
template<class...A>static int ds4_gpu_qwen4exp_qsa_attention_fold_q8_dpos_tensor(A...){attempts++;return fuse_rc;}
template<class...A>static int ds4_gpu_qwen4exp_qsa_attention_dpos_tensor(A...){attentions++;return attn_rc;}
template<class...A>static int ds4_gpu_qwen4exp_qsa_output_gate_doubled_q8_tensor(A...){gates++;kind=1;return gate_rc;}
template<class...A>static int ds4_gpu_qwen4exp_qsa_output_gate_q8_tensor(A...){gates++;kind=2;return gate_rc;}
template<class...A>static int ds4_gpu_qwen4exp_qsa_output_gate_tensor(A...){gates++;kind=3;return gate_rc;}
static bool run(unsigned n_tokens,bool indexed){
 Session storage={};auto *s=&storage;unsigned head_dim=256,q_width=6144,pos0=512,max_selected=2048,il=0;const void *cfg=nullptr;
 const ds4_gpu_tensor *sel=nullptr,*counts=nullptr;
'''+body+r'''
 projection++;return true;
}
int main(){unsigned cases=0;
 for(bool q8:{false,true})for(bool doubled:{false,true})for(int fr:{-1,0,1})for(bool fail_attention:{false,true})for(bool fail_gate:{false,true})for(bool indexed:{false,true}){
  q8on=q8;doubledon=doubled;fuse_rc=fr;attn_rc=!fail_attention;gate_rc=!fail_gate;
  attempts=attentions=gates=kind=projection=0;
  const bool try_new=q8&&doubled,fused=try_new&&fr>0,error=try_new&&fr<0;
  bool want=!error&&(fused||(!fail_attention&&!fail_gate));
  assert(run(2,indexed)==want);
  assert(attempts==unsigned(try_new));assert(attentions==unsigned(!error&&!fused));
  assert(gates==unsigned(!error&&!fused&&!fail_attention));assert(projection==unsigned(want));
  if(gates)assert(kind==(q8?(doubled?1:2):3));cases++;
 }
 printf("PASS %u actual graph call-site scenarios: successful fusion skips both old stages, decline runs both, errors stop\n",cases);
}
'''
code=code.replace('#include <cstdio>','#include <cstdio>\n#include <initializer_list>')
with tempfile.TemporaryDirectory(prefix='qsa-fold-graph-') as d:
 src=Path(d)/'test.cpp';exe=src.with_suffix('');src.write_text(code)
 subprocess.run(['c++','-O2','-std=c++17','-fsanitize=undefined','-fno-sanitize-recover=all',str(src),'-o',str(exe)],check=True)
 subprocess.run([str(exe)],check=True)
