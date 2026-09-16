"""Actual short preparation branch and forward scheduling controls with mocks."""
from pathlib import Path
import subprocess,tempfile
s=(Path(__file__).resolve().parents[1]/'ds4_qwen4exp_graph.inc').read_text()
a=s.index('    if (gather_t0 == 0.0');b=s.index('\n#endif',a);short=s[a:b]
a=s.index('    ds4_qwen4exp_ple_direct_rows ple_prepared = {0};');b=s.index('    /* Layer-0 seed:',a);schedule=s[a:b]
common=r'''
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#define DS4_N_PLE_HEAD 16u
#define DS4_N_PLE_ROW_DIM 160u
#define DS4_QWEN4EXP_MTP_MAX_COMMIT 7u
typedef struct{uint32_t row_count;}ds4_qwen4exp_ple_direct_rows;
typedef struct{bool has_ple;}ds4_qwen4exp_layer_weights;
typedef struct{ds4_qwen4exp_layer_weights layer[1];struct{const uint8_t*base;uint64_t bytes,rows,row_bytes;}ple;}ds4_qwen4exp_weights;
typedef struct{int ple_layer;}config;
typedef struct{uint32_t spec_snapshot_rows,ple_constants,ple_history,ple_history_rows[2],adopt_state,adopt_conv,adopt_device,pos;uint64_t ple_ids[32];float ple_rows_host[5120];void*ple_rows,*d_adopt,*d_pos;}session;
'''
shortcode=common+r'''
static int hashes,decoded,uploaded,decision=1;
static void ds4_ple_row_ids(uint32_t*c,uint32_t*h,const int32_t*t,uint32_t n,uint64_t*out){(void)c;hashes++;for(unsigned r=0;r<n;r++){*h+=t[r];for(unsigned j=0;j<16;j++)out[r*16+j]=*h+j;}}
static int ds4_gpu_qwen4exp_ple_direct_prepare(ds4_qwen4exp_ple_direct_rows*p,void*out,const uint8_t*base,uint64_t bytes,uint64_t rows,uint64_t stride,const uint64_t*ids,uint32_t n){(void)out;(void)base;(void)bytes;(void)rows;(void)stride;assert(ids[0]==1);p->row_count=decision>0?n:0;return decision;}
static const uint8_t*ds4_qwen4exp_ple_row(const void*t,uint64_t id){(void)t;assert(id<=18);static uint8_t v[90];return v;}
static void ds4_ple_dequant_iq4_nl(const void*q,uint32_t n,float*out){(void)q;(void)out;assert(n==5);decoded++;}
static bool run(session*s,const ds4_qwen4exp_weights*w,const int32_t*tokens,uint32_t n_tokens,ds4_qwen4exp_ple_direct_rows*prepared){double gather_t0=0;uint32_t blocks_per_row=5;
'''+short+r'''
return false;
ple_upload:uploaded++;return true;}
int main(){ds4_qwen4exp_weights w={0};int32_t tok[2]={1,2};for(unsigned n=1;n<=2;n++)for(unsigned snap=0;snap<3;snap++)for(decision=-1;decision<=1;decision++){
session x={0};x.spec_snapshot_rows=snap;ds4_qwen4exp_ple_direct_rows p={0};hashes=decoded=uploaded=0;bool ok=run(&x,&w,tok,n,&p);assert(ok==(decision>=0));assert(hashes==(snap?n:1));assert(x.ple_history==(n==1?1:3));if(snap)assert(x.ple_history_rows[0]==1);if(snap==2&&n==2)assert(x.ple_history_rows[1]==3);assert(decoded==(decision==0?16*n:0));assert(uploaded==(decision==0));assert(p.row_count==(decision>0?16*n:0));}
puts("PASS actual short branch: widths1/2 snapshot modes, single hash/history, CPU fallback without rehash, prepare refusal");}
'''
sched=common+r'''
static int fail,events[16],ne;static void event(int e){events[ne++]=e;}
static bool qwen4exp_graph_ple_gather(session*s,const ds4_qwen4exp_weights*w,const int32_t*t,uint32_t n,uint32_t il,ds4_qwen4exp_ple_direct_rows*p){(void)s;(void)w;(void)t;(void)n;(void)il;event(1);p->row_count=16;return fail!=1;}
static bool qwen4exp_session_copy_layers(session*s,uint32_t r,bool a,bool b){(void)s;(void)r;(void)a;(void)b;event(2);return fail!=2;}
static int ds4_gpu_qwen4exp_update_dpos(void*p,uint32_t v){(void)p;(void)v;event(3);return fail!=3;}
static int ds4_gpu_begin_commands(){event(4);return fail!=4;}
static int ds4_gpu_qwen4exp_ple_direct_gather(void*out,const ds4_qwen4exp_ple_direct_rows*p){(void)out;assert(p->row_count==16);event(5);return fail!=5;}
static bool qwen4exp_forward_fail(){event(6);return false;}
static bool run(session*s,const ds4_qwen4exp_weights*w,const int32_t*tokens,uint32_t n_tokens){config c={0};const config*cfg=&c;
'''+schedule+r'''
return true;}
int main(){ds4_qwen4exp_weights w={0};w.layer[0].has_ple=true;int32_t tok=1;for(fail=0;fail<=5;fail++){session x={0};x.adopt_state=1;x.adopt_conv=0;x.d_adopt=&x;x.d_pos=&x;ne=0;assert(run(&x,&w,&tok,1)==(fail==0));int launched=0,drained=0,begun=0;for(int i=0;i<ne;i++){if(events[i]==4)begun=1;if(events[i]==5){assert(begun);launched++;}if(events[i]==6)drained++;}assert(launched==(fail==0||fail==5));assert(drained==(fail==5));}
puts("PASS actual forward prebatch schedule: prepare/materialize/metadata/begin failures never enqueue; launch failure drains");}
'''
with tempfile.TemporaryDirectory(prefix='ple-direct-host-') as t:
 p=Path(t)
 for name,code in [('short',shortcode),('schedule',sched)]:
  (p/(name+'.c')).write_text(code);subprocess.run(['cc','-O2','-std=c11',str(p/(name+'.c')),'-o',str(p/name)],check=True);subprocess.run([str(p/name)],check=True)
