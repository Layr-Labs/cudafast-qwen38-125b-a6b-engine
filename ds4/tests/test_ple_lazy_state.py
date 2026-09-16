"""Compile actual PLE lazy session helper bodies with counted backend mocks."""
from pathlib import Path
import subprocess,tempfile
s=(Path(__file__).resolve().parents[1]/'ds4_qwen4exp_graph.inc').read_text()
def fn(name):
 a=s.index(name+'(');a=s.rfind('\n',0,a)+1;b=s.index('{',a);e=b+1;d=1
 while d:d+=(s[e]=='{')-(s[e]=='}');e+=1
 return s[a:e]
body='\n'.join(fn(n) for n in ['qwen4exp_ple_materialize','qwen4exp_ple_prepare','qwen4exp_rb_ple_conv_select','ds4_qwen4exp_session_reset','qwen4exp_ple_checksum'])
code=r'''
#include <stdint.h>
#include <stdbool.h>
#include <assert.h>
#include <stdio.h>
#include <string.h>
#define DS4_QWEN4EXP_IMPLEMENTED_DEPTH 6
#define DS4_MAX_LAYER 2
struct tensor {uint64_t bytes; unsigned id;};typedef struct tensor ds4_gpu_tensor;
typedef struct {ds4_gpu_tensor *d_ple_adopt,*ple_conv_state,*ple_conv_snapshot;
uint32_t ple_adopt_state,ple_adopt_device,adopt_state,adopt_conv,head_cache_pos,pos;
bool state_dirty;ds4_gpu_tensor *gdn_conv[2],*gdn_state[2],*qsa_k[2],*qsa_v[2],*idx_tape[2],*idx_pool[2];int ple_constants,ple_history;} ds4_qwen4exp_session;
static int copies,uploads,ends,syncs,begins,zeros,fail_copy,fail_upload,fail_end,fail_sync,fail_begin;static uint64_t copy_offset;static uint32_t uploaded;
static uint64_t ds4_gpu_tensor_bytes(const ds4_gpu_tensor*t){return t?t->bytes:0;}
static int ds4_gpu_begin_commands(void){begins++;return !fail_begin;}
static int ds4_gpu_end_commands(void){ends++;return !fail_end;}
static int ds4_gpu_synchronize(void){syncs++;return !fail_sync;}
static int ds4_gpu_tensor_copy(ds4_gpu_tensor*d,uint64_t doff,const ds4_gpu_tensor*s,uint64_t off,uint64_t n){assert(d&&s&&doff==0&&n==d->bytes);copies++;copy_offset=off;return !fail_copy;}
static int ds4_gpu_qwen4exp_update_dpos(ds4_gpu_tensor*t,uint32_t n){assert(t);uploads++;uploaded=n;return !fail_upload;}
static void qwen4exp_zero_tensor(ds4_gpu_tensor*t){if(t)zeros++;}
static void ds4_ple_history_reset(const int*c,int*h){(void)c;*h=0;}
static double qwen4exp_state_checksum(const ds4_gpu_tensor*t){return t->id*100.;}
static double qwen4exp_state_checksum_range(const ds4_gpu_tensor*t,uint64_t off,uint64_t n){assert(n==36);return t->id*100.+off;}
'''+body+r'''
int main(){
 ds4_gpu_tensor flag={4,1},live={36,2},snap={216,3};ds4_qwen4exp_session x={0};x.d_ple_adopt=&flag;x.ple_conv_state=&live;x.ple_conv_snapshot=&snap;
 assert(qwen4exp_rb_ple_conv_select(&x,5)==0&&x.ple_adopt_state==6&&copies==0);
 assert(qwen4exp_ple_checksum(&x)==480.);
 assert(qwen4exp_rb_ple_conv_select(&x,6)==-1&&x.ple_adopt_state==6);
 assert(qwen4exp_ple_prepare(&x,1)&&uploaded==6&&uploads==1&&copies==0);
 assert(qwen4exp_ple_prepare(&x,2)&&uploads==1);
 assert(qwen4exp_rb_ple_conv_select(&x,0)==0&&x.ple_adopt_state==1);
 fail_upload=1;assert(!qwen4exp_ple_prepare(&x,2)&&x.ple_adopt_state==1&&x.ple_adopt_device==UINT32_MAX);fail_upload=0;
 assert(qwen4exp_ple_prepare(&x,2)&&uploaded==1&&x.ple_adopt_device==1);
 for(int mode=0;mode<4;mode++){
  fail_begin=mode==0;fail_copy=mode==1;fail_end=mode==2;fail_sync=mode==3;
  assert(!qwen4exp_ple_prepare(&x,3)&&x.ple_adopt_state==1);
 }
 fail_begin=fail_copy=fail_end=fail_sync=0;
 assert(qwen4exp_ple_prepare(&x,7)&&x.ple_adopt_state==0&&copy_offset==0&&uploaded==0&&x.ple_adopt_device==0);
 assert(qwen4exp_ple_checksum(&x)==200.);
 assert(qwen4exp_rb_ple_conv_select(&x,4)==0);assert(qwen4exp_ple_prepare(&x,1)&&uploaded==5);
 ds4_qwen4exp_session_reset(&x);assert(x.ple_adopt_state==0&&x.ple_adopt_device==5&&!x.state_dirty&&x.pos==0);
 assert(qwen4exp_ple_prepare(&x,1)&&uploaded==0);int count=copies;
 x.state_dirty=false;x.ple_adopt_state=2;ds4_qwen4exp_session_reset(&x);assert(x.ple_adopt_state==0&&copies==count);
 x.d_ple_adopt=0;assert(qwen4exp_rb_ple_conv_select(&x,3)==0&&copies==count+1&&copy_offset==108&&x.ple_adopt_state==0);
 fail_copy=1;assert(qwen4exp_rb_ple_conv_select(&x,1)==-1&&x.ple_adopt_state==2);fail_copy=0;
 puts("PASS actual lazy-PLE host helpers: select/reselect/reset/inspection/mirror/error/wider/eager controls");
}
'''
with tempfile.TemporaryDirectory(prefix='ple-lazy-state-') as t:
 p=Path(t);(p/'test.c').write_text(code)
 subprocess.run(['cc','-O2','-std=c11',str(p/'test.c'),'-o',str(p/'test')],check=True)
 subprocess.run([str(p/'test')],check=True)
