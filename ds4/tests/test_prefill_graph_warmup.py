#!/usr/bin/env python3
"""Run actual shim open with mocked engine calls, including every warm failure."""
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parents[2]
s = (root/'harness/protocol-adapter/ds4_shim/ds4_shim.c').read_text()
opening = s[s.index('ds4s_handle *ds4s_open('):s.index('\nvoid ds4s_close(')]
code = r'''
#define _POSIX_C_SOURCE 200809L
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef struct {int unused;} ds4_engine;
typedef struct {int unused;} ds4_session;
typedef struct {const char *model_path,*mtp_path;int backend,n_threads,context_size,mtp_draft_tokens;float mtp_margin;} ds4_engine_options;
typedef struct {ds4_engine *engine;ds4_session *session;char err[512];} ds4s_handle;
enum {DS4_BACKEND_CUDA=1};
static ds4_engine engine;static ds4_session session;static char g_open_err[512];
static int failure,fail_sync,fail_spec,live,nsync,nspec,prepared,dirty;
static char events[80];static unsigned nevents;
static void event(char c){assert(nevents<79);events[nevents++]=c;events[nevents]=0;}
static void set_open_err(const char *m){snprintf(g_open_err,sizeof g_open_err,"%s",m);}
static void *mock_calloc(size_t a,size_t b){if(failure==1)return NULL;void*p=calloc(a,b);if(p)live++;return p;}
static void *mock_malloc(size_t n){if(failure==8)return NULL;void*p=malloc(n);if(p)live++;return p;}
static void mock_free(void *p){if(p)live--;free(p);}
static int ds4_engine_open(ds4_engine **out,const ds4_engine_options *o){assert(o->backend==DS4_BACKEND_CUDA);*out=failure==3?NULL:&engine;return failure==2?-1:0;}
static int ds4_session_create(ds4_session **out,ds4_engine *e,int ctx){assert(e==&engine&&ctx==2048);*out=failure==5?NULL:&session;return failure==4?-1:0;}
static void ds4_engine_close(ds4_engine *e){assert(e==&engine);}
static void ds4_session_set_progress(ds4_session*s,void*a,void*b){assert(s==&session&&!a&&!b);}
static void ds4_session_set_display_progress(ds4_session*s,void*a,void*b){assert(s==&session&&!a&&!b);}
static int ds4_session_qwen4exp_spec_prepare(ds4_session*s,char*err,size_t n){assert(s==&session);prepared++;if(failure==6){snprintf(err,n,"injected");return -1;}return 0;}
static int ds4s_vocab_size(ds4s_handle*h){assert(h);return failure==7?16:248320;}
static int ds4s_sync(ds4s_handle*h,const int32_t*ids,size_t n){
 assert(h&&prepared&&n==1024&&!dirty);
 for(size_t i=0;i<n;i++)assert(ids[i]==(int32_t)i+1);
 if(nsync)assert(nevents&&events[nevents-1]=='R');
 event('S');dirty=1;nsync++;return nsync==fail_sync?-1:0;
}
static int ds4s_eval(ds4s_handle*h,int32_t id){assert(h&&dirty&&id==1024);event('E');return 0;}
static int32_t ds4s_argmax(ds4s_handle*h){assert(h&&dirty);return 7;}
static int ds4s_eval_speculative(ds4s_handle*h,int32_t t,int budget,int32_t*out,int cap){
 assert(h&&dirty&&t==7&&budget==2&&cap==8);event('P');nspec++;out[0]=7;
 return nspec==fail_spec?-1:1;
}
static void ds4s_invalidate(ds4s_handle*h){assert(h);event('R');dirty=0;}
#define calloc mock_calloc
#define malloc mock_malloc
#define free mock_free
'''+opening+r'''
#undef calloc
#undef malloc
#undef free
static void one(int f,int fs,int fp,int head,int warm,int enable){
 failure=f;fail_sync=fs;fail_spec=fp;live=nsync=nspec=prepared=dirty=0;nevents=0;events[0]=0;
 if(warm)unsetenv("DS4_SHIM_NO_WARMUP");else setenv("DS4_SHIM_NO_WARMUP","1",1);
 if(enable)unsetenv("DS4_QWEN4EXP_NO_PREFILL_GRAPHS");else setenv("DS4_QWEN4EXP_NO_PREFILL_GRAPHS","1",1);
 ds4s_handle*h=ds4s_open("model.gguf",head?"head.gguf":NULL,head?2:0,2048,3);
 if(f>=1&&f<=5){assert(!h&&live==0&&!prepared&&!nevents);return;}
 assert(h&&prepared==1&&!dirty);
 if(!warm)assert(!nevents&&!nsync&&!nspec);
 else{
  assert(nevents&&events[nevents-1]=='R');
  if(f==7||f==8)assert(!strcmp(events,"R"));
  else if(fs==1)assert(!strcmp(events,"SR"));
  else if(!fs&&!fp){
   const char *want=enable?(head?"SEPPPPRS RSPRSRSR":"SERSRSR"):(head?"SEPPPPR":"SER");
   char compact[80];unsigned at=0;for(unsigned i=0;want[i];i++)if(want[i]!=' ')compact[at++]=want[i];compact[at]=0;
   assert(!strcmp(events,compact));
  }
  if(fs&&nsync>=fs)assert(nsync==fs); // Failed prefill stops all later prefill passes.
  if(!enable)assert(nsync<=1&&nspec<=4);
  if(!head)assert(!nspec&&nsync<=3);else assert(nsync<=5&&nspec<=5);
 }
 mock_free(h);assert(!live);
}
int main(void){unsigned cases=0;
 for(int head=0;head<2;head++)for(int warm=0;warm<2;warm++)for(int en=0;en<2;en++){
  for(int f=0;f<=8;f++){one(f,0,0,head,warm,en);cases++;}
  for(int fs=1;fs<=5;fs++){one(0,fs,0,head,warm,en);cases++;}
  for(int fp=1;fp<=5;fp++){one(0,0,fp,head,warm,en);cases++;}
 }
 assert(!ds4s_open(NULL,NULL,0,2048,3));assert(!ds4s_open("",NULL,0,2048,3));assert(!ds4s_open("x",NULL,0,0,3));
 printf("Actual shim prefill warmup: %u scenarios + 3 invalid arguments PASS; reset/input/failure/cleanup\n",cases);
}
'''
with tempfile.TemporaryDirectory(prefix='prefill-warmup-') as d:
    d = Path(d)
    (d/'test.c').write_text(code)
    subprocess.run(['cc', '-std=c11', '-O1', '-g', '-fsanitize=undefined',
                    '-fno-sanitize-recover=all', str(d/'test.c'), '-o', str(d/'test')], check=True)
    subprocess.run([str(d/'test')], check=True)
