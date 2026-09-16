/* Host integration/lifetime/error contract only; mock selection does not test
 * CUDA numerics. Reuses existing independent head algebra and GPU mocks.
 * cc -O3 -ffast-math -fno-finite-math-only -std=c11 -D_GNU_SOURCE -Ids4 \
 * ds4/tests/test_mtp_native_hooks.c ds4/ds4_qwen4exp_mtp.c -lm -o /tmp/test-native-hooks
 */
#define main original_mtp_main
#include "test_qwen4exp_mtp.c"
#undef main
static unsigned calls,maps;
static int mode;
static float observed[HEAD_N_EMBD];
static int screen_mock(ds4_gpu_tensor *out,ds4_gpu_tensor *ids,ds4_gpu_tensor *scratch,
        const void *map,uint64_t bytes,uint64_t offset,uint32_t dim,uint32_t vocab,
        uint32_t prefix,uint32_t tail,const ds4_gpu_tensor *x) {
    calls++;
    CHECK(scratch && dim==HEAD_N_EMBD && vocab==HEAD_N_VOCAB && prefix==5 && tail==2,"screen inputs");
    memcpy(observed,x->data,sizeof observed);
    if(mode) return mode==10?0:mode;
    float full[HEAD_N_VOCAB];ds4_gpu_tensor view={sizeof full,(unsigned char *)full,1};
    CHECK(stub_matmul(&view,map,bytes,offset,dim,vocab,x,1),"oracle projection");
    /* Compact exact rows in original ID order; mandatory0/tail and one other. */
    const uint32_t chosen[]={0,2,6,7};
    memcpy(ids->data,chosen,sizeof chosen);
    for(unsigned i=0;i<4;i++) ((float *)out->data)[i]=full[chosen[i]];
    return 4;
}
static int map_mock(ds4_gpu_tensor *winner,const ds4_gpu_tensor *logits,
                    const ds4_gpu_tensor *ids,uint32_t count,uint32_t vocab) {
    maps++;CHECK(count==4 && vocab==HEAD_N_VOCAB,"map inputs");
    uint32_t bits;memcpy(&bits,logits->data,4);
    uint32_t p=(bits&0x7fffffffu)>0x7f800000u?0:*(uint32_t *)winner->data;
    *(uint32_t *)winner->data=p<count?((uint32_t *)ids->data)[p]:UINT32_MAX;return 1;
}
static void attach(ds4_qwen4exp_mtp_head *h) {
    h->hooks.native_screen=screen_mock;h->hooks.native_map=map_mock;
    h->native_capacity=4;h->t_native_ids=ds4_gpu_tensor_alloc(16);h->t_native_scratch=ds4_gpu_tensor_alloc(32);
}
static int init_mock(uint32_t width,uint64_t *bytes,uint32_t *capacity) {
    CHECK(width==7,"init shortlist width");*bytes=32;*capacity=4;return 1;
}
static unsigned deferred_calls,deferred_maps,block_calls,projection_calls;
static int deferred_invalid,deferred_map_failure,deferred_bad_winner;
static int screen_deferred_mock(ds4_gpu_tensor *out,ds4_gpu_tensor *ids,ds4_gpu_tensor *scratch,
        const void *map,uint64_t bytes,uint64_t offset,uint32_t dim,uint32_t vocab,
        uint32_t prefix,uint32_t tail,const ds4_gpu_tensor *x) {
    deferred_calls++;
    *(uint32_t *)scratch->data=(uint32_t)deferred_invalid;
    return screen_mock(out,ids,scratch,map,bytes,offset,dim,vocab,prefix,tail,x);
}
static int map_deferred_mock(ds4_gpu_tensor *winner,const ds4_gpu_tensor *logits,
        const ds4_gpu_tensor *ids,const ds4_gpu_tensor *scratch,uint32_t width,
        uint32_t count,uint32_t vocab) {
    deferred_maps++;CHECK(width==7 && count==4,"deferred map width/count");
    if(deferred_map_failure)return 0;
    if(*(uint32_t *)scratch->data) {
        *(uint32_t *)winner->data=DS4_MTP_NATIVE_RETRY_ID;return 1;
    }
    if(deferred_bad_winner){*(uint32_t *)winner->data=UINT32_MAX;return 1;}
    return map_mock(winner,logits,ids,count,vocab);
}
static int counted_block(void *graph,void *cache,ds4_gpu_tensor *hyper,
        uint32_t il,uint32_t pos0,uint32_t rows) {
    block_calls++;return stub_block(graph,cache,hyper,il,pos0,rows);
}
static int counted_matmul(ds4_gpu_tensor *out,const void *map,uint64_t bytes,
        uint64_t offset,uint64_t in,uint64_t outdim,const ds4_gpu_tensor *x,uint64_t rows) {
    projection_calls++;return stub_matmul(out,map,bytes,offset,in,outdim,x,rows);
}
static void test_deferred_retry(const int *tokens,const float *input) {
    ds4_qwen4exp_mtp_head h;CHECK(build_shortlist_head(&h)==0,"deferred init");attach(&h);
    h.hooks.native_screen_deferred=screen_deferred_mock;
    h.hooks.native_map_deferred=map_deferred_mock;
    h.hooks.block=counted_block;h.hooks.matmul_q8_0=counted_matmul;h.want_margin=true;
    unsigned comparisons=0;
    for(unsigned last=0;last<2;last++)for(unsigned kind=0;kind<6;kind++) {
        float logits[HEAD_N_VOCAB]={0,2,1,10,3,99,8,7};
        if(kind==1)logits[0]=NAN;
        if(kind==2)logits[3]=INFINITY;
        if(kind==3){logits[0]=-INFINITY;logits[6]=NAN;}
        if(kind==4)for(unsigned i=0;i<HEAD_N_VOCAB;i++)logits[i]=NAN;
        if(kind==5)logits[7]=20;
        g_forced_lm_logits=logits;g_forced_lm_rows=1;
        int got[2]={-1,-1};float multi[2][HEAD_HC_DIM],packed[2][7];
        for(unsigned candidate=0;candidate<2;candidate++) {
            memset(&g_log,0,sizeof g_log);block_calls=projection_calls=0;
            calls=maps=deferred_calls=deferred_maps=0;
            mode=candidate?0:10;deferred_invalid=1;
            if(candidate)unsetenv("DS4_MTP_NO_DEFERRED_SCREEN");
            else setenv("DS4_MTP_NO_DEFERRED_SCREEN","1",1);
            int rc=last?ds4_qwen4exp_mtp_head_forward_last(&h,tokens,input,12,2,&got[candidate],multi[candidate],g_err,sizeof g_err)
                :ds4_qwen4exp_mtp_head_forward(&h,tokens,input,12,1,&got[candidate],multi[candidate],g_err,sizeof g_err);
            CHECK(rc==0,"deferred fallback forward: %s",g_err);
            CHECK(block_calls==1 && projection_calls==3,"retry must not rerun cache block/eh_proj: %u/%u",block_calls,projection_calls);
            CHECK(calls==1 && deferred_calls==candidate && deferred_maps==candidate,"deferred dispatch/fallback counts");
            CHECK(h.last_margin==-1.0f,"static fallback must not publish screen margin");
            memcpy(packed[candidate],h.t_logits->data,sizeof packed[candidate]);
        }
        CHECK(got[0]==got[1],"deferred invalid output matches synchronous fallback");
        CHECK(!memcmp(multi[0],multi[1],sizeof multi[0]),"retry preserves final hyper output");
        CHECK(!memcmp(packed[0],packed[1],sizeof packed[0]),"retry restores full packed logits, including NaN payloads");
        if(!kind)CHECK(got[1]==3,"retry selects winner outside mocked compact candidates");
        if(kind==1||kind==4)CHECK(got[1]==0,"retry preserves original NaN0 contract");
        if(kind==5)CHECK(got[1]==7,"retry restores the full width before selecting a tail winner");
        comparisons++;
    }
    g_forced_lm_logits=NULL;g_forced_lm_rows=0;deferred_invalid=0;mode=0;
    for(unsigned config=0;config<5;config++) {
        calls=maps=deferred_calls=deferred_maps=0;block_calls=projection_calls=0;
        h.hooks.native_screen_deferred=config==2?NULL:screen_deferred_mock;
        h.hooks.native_map_deferred=config==3?NULL:map_deferred_mock;
        if(config==1)setenv("DS4_MTP_NO_DEFERRED_SCREEN","1",1);
        else unsetenv("DS4_MTP_NO_DEFERRED_SCREEN");
        if(config==4)setenv("DS4_MTP_NO_NATIVE_SCREEN","1",1);
        int got=-1;CHECK(ds4_qwen4exp_mtp_head_forward(&h,tokens,input,12,1,&got,NULL,g_err,sizeof g_err)==0,"deferred controls");
        CHECK(deferred_calls==(config==0)&&deferred_maps==(config==0),"deferred pair/valve control");
        CHECK(block_calls==1&&projection_calls==(config==4?3:1),"finite path avoids static retry");
        unsetenv("DS4_MTP_NO_NATIVE_SCREEN");comparisons++;
    }
    h.hooks.native_screen_deferred=screen_deferred_mock;h.hooks.native_map_deferred=map_deferred_mock;
    int got=-1;
    deferred_map_failure=1;CHECK(ds4_qwen4exp_mtp_head_forward(&h,tokens,input,12,1,&got,NULL,g_err,sizeof g_err)<0,"deferred map error propagates");deferred_map_failure=0;
    deferred_bad_winner=1;CHECK(ds4_qwen4exp_mtp_head_forward(&h,tokens,input,12,1,&got,NULL,g_err,sizeof g_err)<0,"malformed winner is error, not nonfinite retry");deferred_bad_winner=0;
    mode=-1;CHECK(ds4_qwen4exp_mtp_head_forward(&h,tokens,input,12,1,&got,NULL,g_err,sizeof g_err)<0,"deferred screen error propagates");
    mode=5;CHECK(ds4_qwen4exp_mtp_head_forward(&h,tokens,input,12,1,&got,NULL,g_err,sizeof g_err)<0,"deferred oversized count refused");
    mode=10;deferred_maps=0;CHECK(ds4_qwen4exp_mtp_head_forward(&h,tokens,input,12,1,&got,NULL,g_err,sizeof g_err)==0&&deferred_maps==0,"deferred decline uses ordinary projection");
    mode=0;ds4_qwen4exp_mtp_head_free(&h);
    printf("deferred native head: %u parity/control cases plus five failure/decline cases\n",comparisons);
}
int main(void) {
    setenv("DS4_QWEN4EXP_DRAFT_VOCAB_PREFIX","5",1);setenv("DS4_QWEN4EXP_DRAFT_VOCAB_TAIL","2",1);
    const int tokens[HEAD_ROWS]={1,2};float input[HEAD_ROWS*HEAD_HC_DIM];
    for(unsigned i=0;i<HEAD_ROWS*HEAD_HC_DIM;i++) input[i]=(int)(i%9)*0.125f;
    ds4_qwen4exp_mtp_head h;CHECK(build_shortlist_head(&h)==0,"init");attach(&h);
    for(unsigned replay=0;replay<3;replay++) {
        input[0]+=0.25f;calls=maps=0;int got=-1;
        CHECK(ds4_qwen4exp_mtp_head_forward(&h,tokens,input,12,1,&got,NULL,g_err,sizeof g_err)==0,"screen forward");
        CHECK(calls==1 && maps==1,"screen dispatch");
        CHECK(!memcmp(observed,h.t_sample->data,sizeof observed),"must read current mixer output");
        float full[HEAD_N_VOCAB];oracle_logits(tokens,input,0,full);
        const unsigned chosen[]={0,2,6,7};unsigned best=0;
        for(unsigned i=0;i<4;i++) {CHECK(((float *)h.t_logits->data)[i]==full[chosen[i]],"selected exact row");if(full[chosen[i]]>full[chosen[best]])best=i;}
        CHECK(got==(int)chosen[best],"original ID winner");
    }
    calls=maps=0;int ids[HEAD_ROWS];
    CHECK(ds4_qwen4exp_mtp_head_forward(&h,tokens,input,12,HEAD_ROWS,ids,NULL,g_err,sizeof g_err)==0 && calls==0,"multirow fallback");
    CHECK(ds4_qwen4exp_mtp_head_forward_last(&h,tokens,input,12,HEAD_ROWS,ids,NULL,g_err,sizeof g_err)==0 && calls==1,"last-only short batch eligible");
    setenv("DS4_MTP_NO_NATIVE_SCREEN","1",1);calls=0;
    CHECK(ds4_qwen4exp_mtp_head_forward(&h,tokens,input,12,1,ids,NULL,g_err,sizeof g_err)==0 && calls==0,"diagnostic fallback");
    unsetenv("DS4_MTP_NO_NATIVE_SCREEN");
    /* Optional backend declines, or reports a real failure/malformed count. */
    mode=0;h.hooks.native_map=NULL;calls=0;
    CHECK(ds4_qwen4exp_mtp_head_forward(&h,tokens,input,12,1,ids,NULL,g_err,sizeof g_err)==0&&calls==0,"unpaired hook fallback");h.hooks.native_map=map_mock;
    mode=10;calls=maps=0;CHECK(ds4_qwen4exp_mtp_head_forward(&h,tokens,input,12,1,ids,NULL,g_err,sizeof g_err)==0&&calls==1&&maps==0,"backend decline uses static projection");
    mode=-1;CHECK(ds4_qwen4exp_mtp_head_forward(&h,tokens,input,12,1,ids,NULL,g_err,sizeof g_err)<0,"backend error");
    mode=5;CHECK(ds4_qwen4exp_mtp_head_forward(&h,tokens,input,12,1,ids,NULL,g_err,sizeof g_err)<0,"oversized count");
    ds4_qwen4exp_mtp_head_free(&h);CHECK(!h.t_native_ids&&!h.t_native_scratch&&!h.native_capacity,"ownership clear");
    /* Init doesn't read weight data: independently exercise native-size scratch
     * ownership with the existing tiny map, without forwarding its wrong shape. */
    h.n_embd=2560;h.eh_proj_in_dim=5120;h.hooks.native_init=init_mock;
    CHECK(ds4_qwen4exp_mtp_head_init(&h,g_err,sizeof g_err)==0,"native-shape init");
    CHECK(h.t_native_ids&&h.t_native_scratch&&h.native_capacity==4,"owned scratch allocation");
    ds4_qwen4exp_mtp_head_free(&h);
    test_deferred_retry(tokens,input);
    printf("native head host contracts: %s\n",g_failures?"FAIL":"PASS");return !!g_failures;
}
