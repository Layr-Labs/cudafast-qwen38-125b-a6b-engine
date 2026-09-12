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

/* Independently compute the selected-row logits from the small head's complete
 * algebra. The wider prefix adds row 5; the original narrow range excludes it. */
static int adaptive_screen_mock(ds4_gpu_tensor *out,ds4_gpu_tensor *ids,ds4_gpu_tensor *scratch,
        const void *map,uint64_t bytes,uint64_t offset,uint32_t dim,uint32_t vocab,
        uint32_t prefix,uint32_t tail,const ds4_gpu_tensor *x) {
    CHECK((prefix==5||prefix==6)&&tail==2&&vocab==HEAD_N_VOCAB,"adaptive screen ranges");
    if(mode==10)return 0;
    float full[HEAD_N_VOCAB];ds4_gpu_tensor view={sizeof full,(unsigned char*)full,1};
    CHECK(stub_matmul(&view,map,bytes,offset,dim,vocab,x,1),"adaptive full oracle");
    const uint32_t chosen[]={0,prefix==5?2u:5u,6,7};
    memcpy(ids->data,chosen,sizeof chosen);
    for(unsigned i=0;i<4;i++)((float*)out->data)[i]=full[chosen[i]];
    return 4;
}
static void adaptive_forward_contract(void) {
    ds4_qwen4exp_mtp_head h;CHECK(build_shortlist_head(&h)==0,"adaptive tiny head init");
    attach(&h);h.hooks.native_screen=adaptive_screen_mock;
    /* Native shape eligibility is tested separately below. This reduced
     * algebra fixture supplies the same preallocated capacity explicitly. */
    ds4_gpu_tensor_free(h.t_logits_prefix);
    h.t_logits_prefix=ds4_gpu_tensor_alloc(HEAD_ROWS*6u*sizeof(float));
    h.draft_vocab_prefix_capacity=6;
    float input[HEAD_ROWS*HEAD_HC_DIM];
    for(unsigned i=0;i<HEAD_ROWS*HEAD_HC_DIM;i++)input[i]=(int)(i%7)*0.125f;
    const int low_or_tail[]={4,6,7,0};int got[HEAD_ROWS];mode=0;
    for(unsigned i=0;i<4;i++) {
        CHECK(ds4_qwen4exp_mtp_head_forward(&h,&low_or_tail[i],input,12+i,1,got,NULL,g_err,sizeof g_err)==0,"narrow adaptive forward");
        CHECK(h.draft_vocab_prefix==5,"prefix boundary and specials keep narrow path");
    }
    const int tokens[HEAD_ROWS]={5,1};
    CHECK(ds4_qwen4exp_mtp_head_forward_last(&h,tokens,input,20,HEAD_ROWS,got,NULL,g_err,sizeof g_err)==0,"excluded seed row expands last-only forward");
    CHECK(h.draft_vocab_prefix==6,"excluded ordinary token expands into reserve");
    float logits[HEAD_N_VOCAB];oracle_logits(tokens,input,HEAD_ROWS-1,logits);
    const unsigned selected[]={0,5,6,7};unsigned best=0;
    for(unsigned i=1;i<4;i++)if(logits[selected[i]]>logits[selected[best]])best=i;
    CHECK(got[0]==(int)selected[best],"expanded original-ID argmax");
    /* Both backend decline and the ordinary multirow fallback must fit the
     * wider preallocated prefix and pack complete rows without overlap. */
    for(unsigned decline=0;decline<2;decline++) {
        mode=decline?10:0;
        unsigned count=decline?1u:HEAD_ROWS;
        CHECK(ds4_qwen4exp_mtp_head_forward(&h,tokens,input,24,count,got,NULL,g_err,sizeof g_err)==0,"expanded fallback forward");
        for(unsigned row=0;row<count;row++) {
            oracle_logits(tokens,input,row,logits);unsigned want=0;
            for(unsigned k=1;k<HEAD_N_VOCAB;k++)if(logits[k]>logits[want])want=k;
            CHECK(got[row]==(int)want,"expanded fallback equals full-vocabulary algebra");
        }
    }
    mode=0;const int low=1;
    CHECK(ds4_qwen4exp_mtp_head_forward(&h,&low,input,30,1,got,NULL,g_err,sizeof g_err)==0&&h.draft_vocab_prefix==6,"request retains expansion");
    ds4_gpu_tensor *scratch=h.t_native_scratch,*prefix=h.t_logits_prefix,*native_ids=h.t_native_ids;
    /* One resident holds the same head across independent requests. Reset
     * must drop coverage history and preserve every allocation, even when
     * called twice or when environment variables changed after init. */
    setenv("DS4_QWEN4EXP_DRAFT_VOCAB_PREFIX","6",1);
    ds4_qwen4exp_mtp_head_reset_vocab(&h);
    ds4_qwen4exp_mtp_head_reset_vocab(&h);
    CHECK(h.draft_vocab_prefix==5&&h.draft_vocab_prefix_capacity==6,"same head restores its initial range");
    CHECK(h.t_native_scratch==scratch&&h.t_logits_prefix==prefix&&h.t_native_ids==native_ids,"reset keeps reserved allocations");
    for(unsigned i=0;i<4;i++) {
        CHECK(ds4_qwen4exp_mtp_head_forward(&h,&low_or_tail[i],input,i,1,got,NULL,g_err,sizeof g_err)==0,"new request narrow forward");
        CHECK(h.draft_vocab_prefix==5,"new low-ID request never inherits expansion");
    }
    CHECK(ds4_qwen4exp_mtp_head_forward_last(&h,tokens,input,4,HEAD_ROWS,got,NULL,g_err,sizeof g_err)==0&&h.draft_vocab_prefix==6,"next excluded token expands again on same head");
    setenv("DS4_QWEN4EXP_DRAFT_VOCAB_PREFIX","5",1);
    ds4_qwen4exp_mtp_head_free(&h);
    CHECK(!h.draft_vocab_prefix_capacity&&!h.draft_vocab_prefix_initial,"free clears reserve ownership");
    CHECK(build_shortlist_head(&h)==0&&h.draft_vocab_prefix==5,"new head restores configured prefix");
    ds4_qwen4exp_mtp_head_free(&h);
}
static uint32_t expected_init_width;
static int adaptive_init_mock(uint32_t width,uint64_t *bytes,uint32_t *capacity) {
    CHECK(width==expected_init_width,"metadata-sized native reserve");
    *bytes=width*16ull+64;*capacity=4;return 1;
}
static void adaptive_allocation_contract(void) {
    ds4_qwen4exp_mtp_head h;CHECK(build_shortlist_head(&h)==0,"allocation template");
    ds4_qwen4exp_mtp_head_free(&h);
    h.n_embd=2560;h.eh_proj_in_dim=5120;h.n_vocab=248320;
    h.hooks.native_init=adaptive_init_mock;
    h.hooks.native_screen=adaptive_screen_mock;h.hooks.native_map=map_mock;
    for(unsigned setting=0;setting<4;setting++) {
        unsetenv("DS4_QWEN4EXP_DRAFT_VOCAB_PREFIX");unsetenv("DS4_QWEN4EXP_DRAFT_VOCAB_TAIL");unsetenv("DS4_MTP_NO_ADAPTIVE_VOCAB");
        if(setting==1)setenv("DS4_MTP_NO_ADAPTIVE_VOCAB","1",1);
        if(setting==2)setenv("DS4_QWEN4EXP_DRAFT_VOCAB_PREFIX","98308",1);
        if(setting==3)setenv("DS4_QWEN4EXP_DRAFT_VOCAB_TAIL","276",1);
        expected_init_width=setting?98584u:248320u;
        CHECK(ds4_qwen4exp_mtp_head_init(&h,g_err,sizeof g_err)==0,"production-metadata adaptive init");
        CHECK(h.draft_vocab_prefix==98308&&h.draft_vocab_tail==276,"initial proposal ranges stay narrow");
        CHECK(h.draft_vocab_prefix_capacity==expected_init_width-276,"override reserve policy");
        CHECK(h.t_logits_prefix->bytes==(uint64_t)HEAD_ROWS*(expected_init_width-276)*sizeof(float),"fallback prefix covers reserve");
        CHECK(h.t_native_scratch->bytes==expected_init_width*16ull+64,"native scratch covers reserve");
        ds4_qwen4exp_mtp_head_reset_vocab(&h);
        CHECK(h.draft_vocab_prefix==98308&&h.draft_vocab_prefix_initial==98308,"reset preserves explicit configuration");
        ds4_qwen4exp_mtp_head_free(&h);
    }
    unsetenv("DS4_QWEN4EXP_DRAFT_VOCAB_PREFIX");unsetenv("DS4_QWEN4EXP_DRAFT_VOCAB_TAIL");unsetenv("DS4_MTP_NO_ADAPTIVE_VOCAB");
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
    adaptive_forward_contract();
    adaptive_allocation_contract();
    printf("native head host contracts: %s\n",g_failures?"FAIL":"PASS");return !!g_failures;
}
