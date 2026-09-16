#!/usr/bin/env python3
"""Actual shim open body: commit native choices before any graph capture."""
import argparse
import subprocess
import tempfile
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--source', type=Path)
args = p.parse_args()
root = Path(__file__).resolve().parents[2]
source = (args.source or root/'harness/protocol-adapter/ds4_shim/ds4_shim.c').read_text()
opening = source[source.index('ds4s_handle *ds4s_open('):source.index('\nvoid ds4s_close(')]
report = source[source.index('const char *ds4s_hw_limits('):source.index('\nint ds4s_vocab_size(')]
code = r'''
#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
typedef struct { int value; } ds4_engine;
typedef struct { int value; } ds4_session;
typedef struct { const char *model_path, *mtp_path; int backend,n_threads,context_size,mtp_draft_tokens; float mtp_margin; } ds4_engine_options;
typedef struct { ds4_engine *engine; ds4_session *session; char err[512]; } ds4s_handle;
enum { DS4_BACKEND_CUDA = 1 };
static ds4_engine engine;
static ds4_session session;
static char g_open_err[512];
static int failure, live, head, prepared, registered, built, chosen, enabled;
static int engine_opens, session_opens, closes, hw_calls, probe_calls, captures, invalidates;
static void set_open_err(const char *s) { snprintf(g_open_err,sizeof g_open_err,"%s",s); }
static void *sim_calloc(size_t a,size_t b) { if(failure==1)return NULL; void *p=calloc(a,b); if(p)live++; return p; }
static void *sim_malloc(size_t n) { if(failure==8)return NULL; void *p=malloc(n); if(p)live++; return p; }
static void sim_free(void *p) { if(p)live--; free(p); }
static int ds4_engine_open(ds4_engine **out,const ds4_engine_options *o) {
    engine_opens++; head=o->mtp_path!=NULL; assert(o->backend==DS4_BACKEND_CUDA);
    assert(o->context_size==2048 && o->n_threads==3 && o->mtp_margin==3.0f);
    *out=failure==3?NULL:&engine; return failure==2?-1:0;
}
static int ds4_session_create(ds4_session **out,ds4_engine *e,int ctx) {
    session_opens++; assert(e==&engine && ctx==2048);
    *out=failure==5?NULL:&session; return failure==4?-1:0;
}
static void ds4_engine_close(ds4_engine *e) { assert(e==&engine); closes++; }
static void ds4_session_set_progress(ds4_session *s,void *a,void *b) { assert(s==&session && !a && !b); }
static void ds4_session_set_display_progress(ds4_session *s,void *a,void *b) { assert(s==&session && !a && !b); }
static int ds4_session_qwen4exp_spec_prepare(ds4_session *s,char *err,size_t n) {
    assert(s==&session && !built); prepared=1;
    if(failure==6) { snprintf(err,n,"injected prepare error"); return -1; }
    registered=head?98584:0; return 0;
}
static const char *ds4_gpu_hw_limits(void) {
    hw_calls++;
    if(!built) {
        assert(prepared && captures==0);
        assert(registered==(head && failure!=6?98584:0));
        built=1; probe_calls++; chosen=enabled;
    }
    return failure==7?NULL:"native choices ready";
}
const char *ds4s_hw_limits(void);
static void capture(void) { assert(built && probe_calls==1 && chosen==enabled); captures++; }
static int ds4s_vocab_size(const ds4s_handle *h) { assert(h && h->engine==&engine); return failure==11?16:248320; }
static int ds4s_sync(ds4s_handle *h,const int32_t *ids,size_t n) {
    assert(h && n==1024); for(size_t i=0;i<n;i++)assert(ids[i]==(int32_t)i+1);
    capture(); return failure==9?-1:0;
}
static int ds4s_eval(ds4s_handle *h,int32_t id) { assert(h && id==1024); capture(); return 0; }
static int32_t ds4s_argmax(ds4s_handle *h) { assert(h); return 7; }
static int ds4s_eval_speculative(ds4s_handle *h,int32_t t,int budget,int32_t *out,int cap) {
    assert(h && head && t==7 && budget==2 && cap==8); capture(); out[0]=7; return failure==10?-1:1;
}
static void ds4s_invalidate(ds4s_handle *h) { assert(h); invalidates++; }
#define calloc sim_calloc
#define malloc sim_malloc
#define free sim_free
'''+opening+'\n'+report+r'''
#undef calloc
#undef malloc
#undef free
static void one(int f,int arm,int warm,int enable) {
    failure=f; live=head=prepared=registered=built=chosen=0; enabled=enable;
    engine_opens=session_opens=closes=hw_calls=probe_calls=captures=invalidates=0;
    if(warm)unsetenv("DS4_SHIM_NO_WARMUP"); else setenv("DS4_SHIM_NO_WARMUP","1",1);
    ds4s_handle *h=ds4s_open("synthetic.gguf",arm?"pinned.gguf":NULL,arm,2048,3);
    if(f>=1 && f<=5) {
        assert(!h && !prepared && !built && !probe_calls && !captures && live==0);
        assert(engine_opens==(f==1?0:1));
        assert(session_opens==(f<=3?0:1));
        assert(closes==(f>=4?1:0)); return;
    }
    assert(h && prepared && built && probe_calls==1 && hw_calls==1);
    assert(invalidates==warm);
    if(!warm || f==8 || f==11)assert(captures==0);
    else if(f==9)assert(captures==1);
    else assert(captures==2+(arm?(f==10?1:4):0));
    int before=captures;
    const char *cached=ds4s_hw_limits();
    assert(cached && hw_calls==2 && probe_calls==1 && captures==before);
    if(f==7)assert(!cached[0]);
    ds4_engine_close(h->engine); sim_free(h); assert(live==0);
}
int main(void) {
    unsigned cases=0;
    for(int arm=0;arm<=1;arm++)for(int warm=0;warm<=1;warm++)for(int en=0;en<=1;en++) {
        for(int f=0;f<=11;f++) { one(f,arm,warm,en); cases++; }
    }
    assert(!ds4s_open(NULL,NULL,0,2048,3));
    assert(!ds4s_open("",NULL,0,2048,3));
    assert(!ds4s_open("x",NULL,0,0,3));
    printf("Actual shim startup order PASS: %u scenarios + 3 invalid arguments, preparation before choice before capture, cached report, cleanup\n",cases);
}
'''
with tempfile.TemporaryDirectory(prefix='exact-mma-startup-') as tmp:
    tmp=Path(tmp)
    (tmp/'check.c').write_text(code)
    subprocess.run(['cc','-std=c11','-O1','-g','-fsanitize=undefined','-fno-sanitize-recover=all',str(tmp/'check.c'),'-o',str(tmp/'check')],check=True)
    subprocess.run([str(tmp/'check')],check=True)
