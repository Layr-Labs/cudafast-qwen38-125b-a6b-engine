#!/usr/bin/env python3
"""Actual forward-prefix host controls for short by-value embedding.

Runs source through the embedding trace point with GPU/upload/PLE/completion
spies and the real GDN replay planner. Old source is an independent routing
oracle; GPU/PLE work is mocked, including completion rather than a real drain. No CUDA, model, network, layer graph or
full-forward execution. GPU consumer-capture parity is a separate test.
"""
from pathlib import Path
import os
import subprocess
import tempfile
ROOT=Path(__file__).resolve().parents[2]
BASE='6c7e8c2ff69c5e025e3d64dfd8b8045e90170876'

def function(text,marker):
    a=text.index(marker); op=text.index('{',a); n=1;b=op+1
    while n:
        n+=(text[b]=='{')-(text[b]=='}');b+=1
    return text[a:b]

def prefix(text,name):
    f=function(text,'bool ds4_qwen4exp_graph_forward(')
    end=f.index('QW_TRACE(QW_SLICE_EMBED);')+len('QW_TRACE(QW_SLICE_EMBED);')
    return f[:end].replace('ds4_qwen4exp_graph_forward(',name+'(',1)+'\nreturn true;\n}\n'

old=subprocess.check_output(['git','show',f'{BASE}:ds4/ds4_qwen4exp_graph.inc'],cwd=ROOT,text=True)
new=(ROOT/'ds4/ds4_qwen4exp_graph.inc').read_text()
source=r'''
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ds4_qwen4exp_gdn_replay.h"
#ifndef DS4_N_EMBD
#define DS4_N_EMBD 2560u
#endif
#ifndef DS4_N_HC
#define DS4_N_HC 4u
#endif
#define DS4_TENSOR_Q8_0 8u
#define DS4_QWEN4EXP_MTP_MAX_COMMIT 2u
#define QW_TRACE(x) ((void)0)
#define QW_MAP(m,w) ((const void*)(m))
#define QW_MAPSZ(m,w) 65536u
#define QW_OFF(w) 128u
#define QW_SLICE_EMBED 0
#define fprintf(...) ((int)0)
typedef struct {int id;int32_t data[8];} ds4_gpu_tensor;
typedef struct {uint32_t type;} ds4_tensor;
typedef struct {bool has_ple;} ds4_qwen4exp_layer_weights;
typedef struct {ds4_qwen4exp_layer_weights layer[2];ds4_tensor *token_embd;} ds4_qwen4exp_weights;
typedef struct {int dummy;} ds4_model;
typedef struct {uint32_t n_vocab,n_layer,ple_layer;} ds4_qwen4exp_config;
static ds4_qwen4exp_config g_ds4_qwen4exp={8,2,0};
typedef struct {
 struct {uint32_t n_batch,n_ctx;} plan;
 uint32_t spec_top1_rows,pos,spec_snapshot_rows,adopt_state,adopt_conv,adopt_device;
 uint32_t gdn_replay_prefix,gdn_replay_phase;
 bool state_dirty,hc_pending,gdn_replay_enabled,gdn_replay_previous,gdn_replay_active;
 ds4_gpu_tensor *tokens,*hyper,*embed_rows,*d_gdn_replay,*d_adopt,*d_pos;
 ds4_gpu_tensor *gdn_state[2],*gdn_checkpoint[2];
} ds4_qwen4exp_session;
static char events[128],fail_at;static unsigned nevents;
static unsigned fused_calls,old_calls,upload_calls,drains;
static int32_t recorded[2];static unsigned recorded_rows;
static int event(char c){assert(nevents<127);events[nevents++]=c;events[nevents]=0;return c!=fail_at;}
static double now_sec(void){return 1;}
static int ds4_gpu_end_commands(void){drains++;return event('E');}
static int ds4_gpu_tensor_write(ds4_gpu_tensor *dst,uint64_t off,const void *src,uint64_t bytes){
 assert(off==0 && bytes<=sizeof(dst->data));upload_calls++;if(!event('U'))return 0;memcpy(dst->data,src,bytes);return 1;
}
static bool qwen4exp_graph_ple_gather(ds4_qwen4exp_session*s,const ds4_qwen4exp_weights*w,const int32_t*t,uint32_t n,uint32_t il){
 (void)s;(void)w;(void)t;(void)n;assert(il==0);return event('P');
}
static bool qwen4exp_session_copy_layers(ds4_qwen4exp_session*s,uint32_t row,bool recurrent,bool conv){
 (void)s;assert(row<2 && recurrent!=conv);return event(recurrent?'S':'C');
}
static int ds4_gpu_qwen4exp_update_dpos(ds4_gpu_tensor*t,uint32_t v){
 char c=t->id==4?'R':(t->id==5?'A':'D');if(!event(c))return 0;t->data[0]=(int32_t)v;return 1;
}
static int ds4_gpu_begin_commands(void){return event('B');}
static int ds4_gpu_qwen4exp_embed_tokens_hc_tensor(ds4_gpu_tensor*h,ds4_gpu_tensor*r,const ds4_gpu_tensor*t,const void*m,uint64_t size,uint64_t off,uint32_t type,uint32_t vocab,uint32_t n,uint32_t dim,uint32_t hc){
 (void)h;(void)r;(void)m;assert(size==65536 && off==128 && vocab==8 && dim==DS4_N_EMBD && hc==DS4_N_HC);(void)type;
 old_calls++;recorded_rows=n;recorded[0]=t->data[0];recorded[1]=n>1?t->data[1]:-99;return event('O');
}
'''
# The additive API facade is supplied below after its actual signature is stable.
API_FACADE=r'''
static int ds4_gpu_qwen4exp_embed_short_tensor(ds4_gpu_tensor*h,ds4_gpu_tensor*r,ds4_gpu_tensor*t,const void*m,uint64_t size,uint64_t off,uint32_t type,uint32_t vocab,uint32_t n,int32_t t0,int32_t t1){
 (void)h;(void)r;(void)m;assert(size==65536&&off==128&&type==8&&vocab==8&&n>=1&&n<=2);
 fused_calls++;recorded_rows=n;recorded[0]=t0;recorded[1]=n==2?t1:-99;
 if(!event('F'))return 0;t->data[0]=t0;if(n==2)t->data[1]=t1;return 1;
}
''' 
source+=API_FACADE+'\n'+function(new,'static bool qwen4exp_forward_fail(void)')+'\n'
# Include the real new eligibility helper, when introduced outside forward.
HELPER='' # Eligibility is inside the actual forward prefix.
source+=HELPER+'\n'+prefix(old,'old_forward')+prefix(new,'new_forward')
source+=r'''
static unsigned cases,fault_cases;
static void reset_spies(char fail){events[0]=0;nevents=0;fail_at=fail;fused_calls=old_calls=upload_calls=drains=0;recorded[0]=recorded[1]=-99;recorded_rows=0;}
static ds4_qwen4exp_session session(ds4_gpu_tensor *b,unsigned rows){
 for(unsigned i=0;i<11;i++){memset(&b[i],0,sizeof(b[i]));b[i].id=i+1;for(unsigned j=0;j<8;j++)b[i].data[j]=-77;}
 ds4_qwen4exp_session s={0};s.plan.n_batch=8;s.plan.n_ctx=64;s.spec_top1_rows=rows;s.pos=9;s.spec_snapshot_rows=1;
 s.hc_pending=true;s.gdn_replay_enabled=true;s.adopt_state=1;s.adopt_conv=2;s.adopt_device=1;
 s.tokens=&b[0];s.hyper=&b[1];s.embed_rows=&b[2];s.d_gdn_replay=&b[3];s.d_adopt=&b[4];s.d_pos=&b[5];
 s.gdn_state[0]=&b[6];s.gdn_state[1]=&b[7];s.gdn_checkpoint[0]=&b[8];s.gdn_checkpoint[1]=&b[9];return s;
}
static void check(unsigned rows,int disabled,int type,char failure,int bad,int ple,int reuse){
 ds4_gpu_tensor b[11];ds4_qwen4exp_session s=session(b,rows);if(reuse){s.gdn_replay_previous=true;s.adopt_conv=1;s.adopt_device=0;}ds4_tensor emb={type};
 ds4_qwen4exp_weights w={.token_embd=&emb};w.layer[0].has_ple=ple;ds4_model m={0};
 int32_t tokens[8]={2,5,1,7,0,6,4,3};if(bad==1)tokens[0]=-1;if(bad==2)tokens[rows-1]=8;
 if(disabled)setenv("DS4_QWEN4EXP_NO_SHORT_EMBED","1",1);else unsetenv("DS4_QWEN4EXP_NO_SHORT_EMBED");
 const bool eligible=TEST_CUDA && DS4_N_EMBD==2560u && DS4_N_HC==4u && rows<=2 && type==8 && !disabled;
 ds4_gpu_tensor ref_b[11];ds4_qwen4exp_session ref=session(ref_b,rows);if(reuse){ref.gdn_replay_previous=true;ref.adopt_conv=1;ref.adopt_device=0;}
 char ref_fail=failure;
 if(eligible){if(failure=='F')ref_fail='O';else if(failure=='U'||failure=='O')ref_fail=0;}
 else if(failure=='F')ref_fail=0;
 reset_spies(ref_fail);bool ref_ok=old_forward(&ref,&w,&m,tokens,rows,NULL);
 char reference_events[128];unsigned ri=0;
 for(unsigned j=0;j<nevents;j++){if(eligible&&events[j]=='U')continue;reference_events[ri++]=(eligible&&events[j]=='O')?'F':events[j];}reference_events[ri]=0;
 reset_spies(failure);bool ok=new_forward(&s,&w,&m,tokens,rows,NULL);
 assert(ok==ref_ok && !strcmp(events,reference_events));
 assert(s.state_dirty==ref.state_dirty && s.hc_pending==ref.hc_pending);
 assert(s.adopt_state==ref.adopt_state && s.adopt_conv==ref.adopt_conv && s.adopt_device==ref.adopt_device);
 assert(s.gdn_replay_prefix==ref.gdn_replay_prefix && s.gdn_replay_phase==ref.gdn_replay_phase && s.gdn_replay_active==ref.gdn_replay_active);
 for(unsigned j=0;j<2;j++)assert(s.gdn_state[j]->id==ref.gdn_state[j]->id && s.gdn_checkpoint[j]->id==ref.gdn_checkpoint[j]->id);
 char expect[128]="";if(!bad){if(!eligible)strcat(expect,"U");if(ple)strcat(expect,"P");if(!reuse||rows!=2)strcat(expect,"SC");if(rows==2)strcat(expect,"R");if(!reuse||rows==2)strcat(expect,"A");strcat(expect,"DB");strcat(expect,eligible?"F":"O");}
 bool expected_ok=!bad;
 if(failure && strchr(expect,failure)){
  char*p=strchr(expect,failure);p[1]=0;expected_ok=false;if(failure=='F'||failure=='O')strcat(expect,"E");fault_cases++;
 }
 assert(ok==expected_ok);if(strcmp(events,expect)){printf("events %s expected %s rows%u eligible%d failure%c bad%d\n",events,expect,rows,eligible,failure,bad);abort();}
 if(bad){assert(!upload_calls&&!old_calls&&!fused_calls&&!drains);}
 if(ok){assert(recorded_rows==rows && recorded[0]==tokens[0]);if(rows>1)assert(recorded[1]==tokens[1]);assert(!s.hc_pending&&s.state_dirty);assert(s.adopt_state==(reuse&&rows==2?1u:0u)&&s.adopt_conv==s.adopt_state&&s.adopt_device==s.adopt_state);assert(s.gdn_replay_active==(rows==2));assert(s.gdn_replay_phase==(rows==2&&!reuse));assert(s.pos==9);assert(s.tokens->data[0]==tokens[0]);}
 assert(fused_calls<=1 && old_calls<=1 && !(fused_calls&&old_calls));
 if(eligible)assert(upload_calls==0);
 // A new invocation must publish fresh scalar IDs; changing caller memory after
 // return cannot mutate values already copied into the API facade.
 if(ok){int32_t saved0=recorded[0];tokens[0]=7;assert(recorded[0]==saved0);reset_spies(0);assert(new_forward(&s,&w,&m,tokens,rows,NULL));assert(recorded[0]==7);}
 cases++;
}
static void early_controls(void){
 for(unsigned which=0;which<8;which++){
  ds4_gpu_tensor b[11];ds4_qwen4exp_session s=session(b,2);ds4_tensor emb={8};ds4_qwen4exp_weights w={.token_embd=&emb};ds4_model m={0};int32_t ids[2]={1,2};
  unsigned n=2;if(which==0)n=0;if(which==5)s.spec_top1_rows=0;if(which==6)s.plan.n_batch=1;if(which==7)s.pos=64;
  reset_spies(0);bool ok=new_forward(which==1?NULL:&s,which==2?NULL:&w,which==3?NULL:&m,which==4?NULL:ids,n,NULL);
  assert(!ok && nevents==0 && !s.state_dirty);cases++;
 }
}
int main(void){
 early_controls();
 for(unsigned rows=1;rows<=3;rows++)for(int off=0;off<2;off++)for(int type=0;type<2;type++)for(int ple=0;ple<2;ple++)for(int reuse=0;reuse<2;reuse++){
  check(rows,off,type?8:1,0,0,ple,reuse);
  check(rows,off,type?8:1,0,1,ple,reuse);check(rows,off,type?8:1,0,2,ple,reuse);
  const char *faults="UPSCRADBFO";for(const char*p=faults;*p;p++)check(rows,off,type?8:1,*p,0,ple,reuse);
 }
 printf("PASS %u forward-prefix cases, %u reached injected failures, backend CUDA=%d embd=%u hc=%u\n",cases,fault_cases,TEST_CUDA,DS4_N_EMBD,DS4_N_HC);return 0;
}
'''
with tempfile.TemporaryDirectory(prefix='short-embed-graph-') as tmp:
    src=Path(tmp)/'test.c';src.write_text(source)
    for name,flags in [('cuda',['-DTEST_CUDA=1']),('other_embd',['-DTEST_CUDA=1','-DDS4_N_EMBD=1280']),('other_hc',['-DTEST_CUDA=1','-DDS4_N_HC=2']),('metal',['-D__APPLE__','-DTEST_CUDA=0']),('rocm',['-DDS4_ROCM_BUILD','-DTEST_CUDA=0'])]:
        exe=Path(tmp)/name
        subprocess.run([os.environ.get('CC','cc'),'-O2','-std=c11','-D_GNU_SOURCE','-I'+str(ROOT/'ds4'),*flags,str(src),'-o',str(exe)],check=True)
        subprocess.run([str(exe)],check=True)
