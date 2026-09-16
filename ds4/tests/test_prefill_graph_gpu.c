/* Native, full-engine fixture. Requires a CUDA GPU and a compatible checkpoint.
 * Build/link is possible off-box; it is NOT evidence of a native test run.
 * Usage: test_prefill_graph_gpu first-shard.gguf [mtp-head.gguf]
 * Holds one model/session and compares all logits against eager prefill.
 */
#define _POSIX_C_SOURCE 200809L
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ds4.h"

extern const char *ds4_gpu_hw_limits(void);
extern void ds4_gpu_decode_graphs_invalidate(void);

static char err[512];
static void check(int ok,const char *what){
    if(!ok){fprintf(stderr,"FAIL %s: %s\n",what,err);exit(1);}
}
static void logits(ds4_session *s,float *out,int vocab){
    for(int i=0;i<vocab;i++)out[i]=NAN;
    check(ds4_session_copy_logits(s,out,vocab)==vocab,"complete logits");
    for(int i=0;i<vocab;i++)check(isfinite(out[i]),"finite logits");
}
static void run(ds4_session *s,const ds4_tokens *tokens,unsigned prefix,
                float *out,int vocab){
    ds4_session_invalidate(s);
    if(prefix){ds4_tokens first=*tokens;first.len=(int)prefix;
        check(ds4_session_sync(s,&first,err,sizeof err)==0,"initial prefix");}
    check(ds4_session_sync(s,tokens,err,sizeof err)==0,"prefill/suffix");
    logits(s,out,vocab);
    check(ds4_session_eval(s,7,err,sizeof err)==0,"decode after prefill");
    logits(s,out+vocab,vocab);
}
int main(int argc,char **argv){
    if(argc<2||argc>3){fprintf(stderr,"usage: %s first-shard.gguf [mtp-head.gguf]\n",argv[0]);return 2;}
    ds4_engine_options opt={0};opt.model_path=argv[1];opt.mtp_path=argc==3?argv[2]:NULL;
    opt.backend=DS4_BACKEND_CUDA;opt.context_size=4096;opt.prefill_chunk=1024;
    opt.mtp_draft_tokens=argc==3?2:1;opt.mtp_margin=3.0f;opt.n_threads=4;
    ds4_engine *engine=NULL;ds4_session *session=NULL;
    check(ds4_engine_open(&engine,&opt)==0&&engine,"open engine");
    check(ds4_session_create(&session,engine,4096)==0&&session,"create session");
    check(ds4_session_qwen4exp_spec_prepare(session,err,sizeof err)==0,"prepare head");
    int vocab=ds4_engine_vocab_size(engine);check(vocab>16,"vocabulary");
    size_t bytes=(size_t)2*vocab*sizeof(float);
    float *reference=malloc(bytes),*candidate=malloc(bytes);check(reference&&candidate,"host logits buffers");
    const unsigned widths[]={8,47,48,64,127,1024,64};
    const unsigned prefixes[]={0,64,1024,2048};
    unsigned checks=0;
    for(unsigned shape=0;shape<sizeof widths/sizeof *widths;shape++){
        // Also tests invalidation followed by growing convolution scratch.
        ds4_gpu_decode_graphs_invalidate();
        for(unsigned pass=0;pass<4;pass++)for(unsigned at=0;at<4;at++){
            unsigned n=prefixes[at]+widths[shape];ds4_tokens tokens={0};
            for(unsigned i=0;i<n;i++)ds4_tokens_push(&tokens,1+(int)((i*13+pass*37+shape*19)%(unsigned)(vocab-8)));
            setenv("DS4_QWEN4EXP_NO_PREFILL_GRAPHS","1",1);
            run(session,&tokens,prefixes[at],reference,vocab);
            unsetenv("DS4_QWEN4EXP_NO_PREFILL_GRAPHS");
            if(argc==3){int out[8];int t=ds4_session_argmax(session);
                check(ds4_session_eval_speculative_argmax(session,t,2,-1,out,8,err,sizeof err)>0,"change physical state parity");}
            run(session,&tokens,prefixes[at],candidate,vocab);
            check(!memcmp(reference,candidate,bytes),"eager/captured/replayed full logits and decode state");
            ds4_tokens_free(&tokens);checks++;
        }
    }
    const char *hw=ds4_gpu_hw_limits();unsigned ready=0,dead=0;unsigned long long replays=0;
    const char *p=hw?strstr(hw,"pfGraph["):NULL;
    check(p&&sscanf(p,"pfGraph[c=%u r=%llu d=%u]",&ready,&replays,&dead)==3,"graph diagnostic");
    check(ready>0&&replays>0&&dead==0,"native prefill capture/replay actually ran");
    printf("Native full-engine prefill graphs PASS: %u comparisons; %s\n",checks,hw);
    free(reference);free(candidate);ds4_session_free(session);ds4_engine_close(engine);
    return 0;
}
