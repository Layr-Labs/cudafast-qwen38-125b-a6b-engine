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
    float best=-INFINITY;uint32_t id=0;
    for(uint32_t i=0;i<count;i++) {float v=((float *)logits->data)[i];uint32_t oi=((uint32_t *)ids->data)[i];if(v>best||(v==best&&oi<id)){best=v;id=oi;}}
    *(uint32_t *)winner->data=(bits&0x7fffffffu)>0x7f800000u?0:id;return 1;
}
static void attach(ds4_qwen4exp_mtp_head *h) {
    h->hooks.native_screen=screen_mock;h->hooks.native_map=map_mock;
    h->native_capacity=4;h->t_native_ids=ds4_gpu_tensor_alloc(16);h->t_native_scratch=ds4_gpu_tensor_alloc(32);
}
static int init_mock(uint32_t width,uint64_t *bytes,uint32_t *capacity) {
    CHECK(width==7,"init shortlist width");*bytes=32;*capacity=4;return 1;
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
    printf("native head host contracts: %s\n",g_failures?"FAIL":"PASS");return !!g_failures;
}
