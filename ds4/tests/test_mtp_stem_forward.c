/* Host integration witness: actual head forwarding must upload current inputs
 * before the optional stem, and timing mode must bypass that hook. */
#define main original_mtp_main
#include "test_qwen4exp_mtp.c"
#undef main
#include <assert.h>
static unsigned stem_calls;
static int stem_fail;
static const int *expected_tokens;
static const float *expected_hyper;
static int observed_stem(ds4_qwen4exp_mtp_head *h,uint32_t rows) {
    ++stem_calls;
    assert(!memcmp(h->t_tokens->data,expected_tokens,rows*sizeof(int)));
    assert(!memcmp(h->t_hyper->data,expected_hyper,rows*HEAD_HC_DIM*sizeof(float)));
    if(stem_fail)return 0;
    return ds4_qwen4exp_mtp_head_stem_eager(h,rows);
}
int main(int argc,char **argv) {
    (void)argv;const int timing=argc>1;
    if(timing)setenv("DS4_MTP_HEAD_TIME","1",1);else unsetenv("DS4_MTP_HEAD_TIME");
    unsetenv("DS4_QWEN4EXP_DRAFT_VOCAB_PREFIX");unsetenv("DS4_QWEN4EXP_DRAFT_VOCAB_TAIL");
    ds4_qwen4exp_mtp_head h;assert(build_shortlist_head(&h)==0);
    int ids[HEAD_ROWS],actual[HEAD_ROWS],oracle[HEAD_ROWS];
    float input[HEAD_ROWS*HEAD_HC_DIM],a[HEAD_ROWS*HEAD_HC_DIM],b[HEAD_ROWS*HEAD_HC_DIM];
    for(unsigned width=1;width<=HEAD_ROWS;++width)for(unsigned pass=0;pass<3;++pass){
        for(unsigned r=0;r<HEAD_ROWS;++r)ids[r]=(int)((pass+r+1)%HEAD_N_VOCAB);
        for(unsigned i=0;i<HEAD_ROWS*HEAD_HC_DIM;++i)input[i]=(float)((int)i-7)*(1.f/32)+(float)pass;
        expected_tokens=ids;expected_hyper=input;h.hooks.stem=observed_stem;
        assert(ds4_qwen4exp_mtp_head_forward(&h,ids,input,12,width,actual,a,g_err,sizeof(g_err))==0);
        h.hooks.stem=NULL;
        assert(ds4_qwen4exp_mtp_head_forward(&h,ids,input,12,width,oracle,b,g_err,sizeof(g_err))==0);
        assert(!memcmp(actual,oracle,width*sizeof(int))&&!memcmp(a,b,width*HEAD_HC_DIM*sizeof(float)));
    }
    assert(stem_calls==(timing?0u:HEAD_ROWS*3u));
    if(!timing){h.hooks.stem=observed_stem;stem_fail=1;
        assert(ds4_qwen4exp_mtp_head_forward(&h,ids,input,12,1,actual,a,g_err,sizeof(g_err))!=0);}
    ds4_qwen4exp_mtp_head_free(&h);assert(!g_failures);
    puts(timing?"MTP stem timing bypass: PASS":"MTP stem current-upload forwarding: PASS");return 0;
}
