/* Actual forward with counted host GPU mocks. No CUDA execution claim. */
#define ds4_gpu_tensor_read base_read
#define ds4_gpu_indexer_topk_tensor base_top1
#define ds4_gpu_begin_commands base_begin
#define ds4_gpu_end_commands base_end
#define main original_mtp_main
#include "test_qwen4exp_mtp.c"
#undef main
#undef ds4_gpu_tensor_read
#undef ds4_gpu_indexer_topk_tensor
#undef ds4_gpu_begin_commands
#undef ds4_gpu_end_commands
#include <assert.h>
static ds4_qwen4exp_mtp_head *head;
static unsigned id_reads,flag_reads,zero_reads,async_calls,old_calls,map_calls,top_calls,begins,ends;
static int mode,fail_read,fail_begin,fail_end,fail_projection,fail_top;
static float sample[HEAD_N_EMBD];
int ds4_gpu_tensor_read(const ds4_gpu_tensor*t,uint64_t off,void*out,uint64_t n) {
    if(t==head->t_top1) { ++id_reads;if(fail_read==1||(fail_read==4&&id_reads==2))return 0; }
    if(t==head->t_native_scratch) { ++flag_reads;if(fail_read==2)return 0; }
    if(t==head->t_logits) { ++zero_reads;if(fail_read==3)return 0; }
    return base_read(t,off,out,n);
}
int ds4_gpu_indexer_topk_tensor(ds4_gpu_tensor*a,const ds4_gpu_tensor*b,uint32_t c,uint32_t d,uint32_t e) {
    ++top_calls;if(fail_top&&begins==2)return 0;return base_top1(a,b,c,d,e);
}
int ds4_gpu_begin_commands(void) { ++begins;return !(fail_begin&&begins==2); }
int ds4_gpu_end_commands(void) { ++ends;return !(fail_end&&ends==2); }
static int fill_native(ds4_gpu_tensor*out,ds4_gpu_tensor*ids,const void*map,
 uint64_t bytes,uint64_t offset,uint32_t dim,uint32_t vocab,const ds4_gpu_tensor*x) {
    float full[HEAD_N_VOCAB];ds4_gpu_tensor view={sizeof full,(unsigned char*)full,1};
    assert(stub_matmul(&view,map,bytes,offset,dim,vocab,x,1));
    uint32_t chosen[]={0,2,6,7};memcpy(ids->data,chosen,sizeof chosen);
    for(unsigned i=0;i<4;++i)((float*)out->data)[i]=full[chosen[i]];
    return 4;
}
static int old_screen(ds4_gpu_tensor*out,ds4_gpu_tensor*ids,ds4_gpu_tensor*scratch,
 const void*map,uint64_t bytes,uint64_t off,uint32_t dim,uint32_t vocab,
 uint32_t prefix,uint32_t tail,const ds4_gpu_tensor*x) {
    (void)scratch;(void)prefix;(void)tail;++old_calls;
    if(mode==1||mode==2)return 0;
    return fill_native(out,ids,map,bytes,off,dim,vocab,x);
}
static int old_map(ds4_gpu_tensor*w,const ds4_gpu_tensor*l,const ds4_gpu_tensor*ids,uint32_t count,uint32_t vocab) {
    (void)vocab;++map_calls;uint32_t bits;memcpy(&bits,l->data,4);
    uint32_t p=(bits&0x7fffffffu)>0x7f800000u?0:*(uint32_t*)w->data;
    *(uint32_t*)w->data=p<count?((uint32_t*)ids->data)[p]:UINT32_MAX;return 1;
}
static int queued(ds4_qwen4exp_mtp_head*h,uint64_t*off) {
    ++async_calls;*off=64;
    if(mode==3)return 0;if(mode==-1)return -1;if(mode==4)return 3;
    if(mode==5){*off=UINT64_MAX;return 4;}
    if(mode==6){*(uint32_t*)h->t_top1->data=h->n_vocab;return 4;}
    memcpy(sample,h->t_sample->data,sizeof sample);
    *(uint32_t*)(h->t_native_scratch->data+64)=mode==1?1:0;
    if(mode==1||mode==2) {
        memset(h->t_logits->data,0xa5,h->t_logits->bytes);
        *(uint32_t*)h->t_top1->data=UINT32_MAX;return 4;
    }
    fill_native(h->t_logits,h->t_native_ids,h->target_map,h->target_size,h->output_offset,h->n_embd,h->n_vocab,h->t_sample);
    assert(ds4_gpu_indexer_topk_tensor(h->t_top1,h->t_logits,4,1,1));
    assert(old_map(h->t_top1,h->t_logits,h->t_native_ids,4,h->n_vocab));
    return 4;
}
static void reset(void) {
    id_reads=flag_reads=zero_reads=async_calls=old_calls=map_calls=top_calls=begins=ends=0;
    fail_read=fail_begin=fail_end=fail_projection=fail_top=0;memset(&g_log,0,sizeof g_log);
}
static int checked_matmul(ds4_gpu_tensor*out,const void*map,uint64_t bytes,
 uint64_t off,uint64_t in,uint64_t dim,const ds4_gpu_tensor*x,uint64_t rows) {
    if(fail_projection&&begins==2&&map==head->target_map)return 0;
    return stub_matmul(out,map,bytes,off,in,dim,x,rows);
}
static unsigned named_calls(const char *name) {
    unsigned count=0;for(int i=0;i<g_log.n_log&&i<16;++i)count+=!strcmp(g_log.log[i],name);return count;
}
#include "../ds4_qwen4exp_mtp_hooks.c"
int main(int argc,char **argv) {
    (void)argv;if(argc>1)setenv("DS4_MTP_HEAD_TIME","1",1);else unsetenv("DS4_MTP_HEAD_TIME");
    ds4_qwen4exp_mtp_head h;assert(build_shortlist_head(&h)==0);head=&h;
    h.t_native_ids=ds4_gpu_tensor_alloc(16);h.t_native_scratch=ds4_gpu_tensor_alloc(128);h.native_capacity=4;
    h.hooks.native_screen=old_screen;h.hooks.native_map=old_map;h.hooks.native_propose_async=queued;
    h.hooks.matmul_q8_0=checked_matmul;
    int tokens[HEAD_ROWS]={1,2},got[HEAD_ROWS],ref[HEAD_ROWS];float input[HEAD_ROWS*HEAD_HC_DIM];
    for(unsigned i=0;i<HEAD_ROWS*HEAD_HC_DIM;++i)input[i]=(float)((int)i-8)/32;
    if(argc>1) {
        mode=0;reset();assert(ds4_qwen4exp_mtp_head_forward(&h,tokens,input,12,1,got,NULL,g_err,sizeof g_err)==0);
        assert(async_calls==0&&old_calls==1);ds4_qwen4exp_mtp_head_free(&h);puts("MTP async timing bypass: PASS");return 0;
    }
    for(unsigned width=1;width<=2;++width)for(unsigned pass=0;pass<3;++pass) {
        input[0]+=.25f;reset();mode=0;
        assert(ds4_qwen4exp_mtp_head_forward_last(&h,tokens,input,12,width,got,NULL,g_err,sizeof g_err)==0);
        assert(async_calls==1&&old_calls==0&&top_calls==1&&map_calls==1&&id_reads==1&&flag_reads==0&&zero_reads==0);
        h.hooks.native_propose_async=NULL;reset();
        assert(ds4_qwen4exp_mtp_head_forward_last(&h,tokens,input,12,width,ref,NULL,g_err,sizeof g_err)==0&&got[0]==ref[0]);
        h.hooks.native_propose_async=queued;
    }
    float actual_multi[HEAD_HC_DIM],oracle_multi[HEAD_HC_DIM];
    mode=1;reset();assert(ds4_qwen4exp_mtp_head_forward_last(&h,tokens,input,12,2,got,actual_multi,g_err,sizeof g_err)==0);
    assert(async_calls==1&&old_calls==0&&id_reads==2&&flag_reads==1&&zero_reads==1&&begins==2&&ends==2&&top_calls==1);
    assert(!memcmp(sample,h.t_sample->data,sizeof sample));
    assert(named_calls("block")==1&&named_calls("hc_mixer")==1);
    h.hooks.native_propose_async=NULL;reset();assert(ds4_qwen4exp_mtp_head_forward_last(&h,tokens,input,12,2,ref,oracle_multi,g_err,sizeof g_err)==0&&got[0]==ref[0]&&!memcmp(actual_multi,oracle_multi,sizeof actual_multi));h.hooks.native_propose_async=queued;
    mode=2;reset();assert(ds4_qwen4exp_mtp_head_forward(&h,tokens,input,12,1,got,NULL,g_err,sizeof g_err)!=0);
    assert(flag_reads==1&&begins==1&&strstr(g_err,"invalid native-screen winner"));
    for(int m=-1;m<=6;++m)if(m==-1||m>=4){mode=m;reset();assert(ds4_qwen4exp_mtp_head_forward(&h,tokens,input,12,1,got,NULL,g_err,sizeof g_err)!=0);assert(old_calls==0&&flag_reads==0);}
    mode=3;reset();assert(ds4_qwen4exp_mtp_head_forward(&h,tokens,input,12,1,got,NULL,g_err,sizeof g_err)==0&&old_calls==1&&async_calls==1);
    mode=0;reset();setenv("DS4_MTP_NO_ASYNC_PROPOSAL","1",1);assert(ds4_qwen4exp_mtp_head_forward(&h,tokens,input,12,1,got,NULL,g_err,sizeof g_err)==0&&async_calls==0&&old_calls==1);unsetenv("DS4_MTP_NO_ASYNC_PROPOSAL");
    reset();assert(ds4_qwen4exp_mtp_head_forward(&h,tokens,input,12,2,got,NULL,g_err,sizeof g_err)==0&&async_calls==0&&old_calls==0);
    for(int which=1;which<=8;++which){mode=1;reset();if(which<=3)fail_read=which;else if(which==4)fail_begin=1;else if(which==5)fail_end=1;else if(which==6)fail_read=4;else if(which==7)fail_projection=1;else fail_top=1;
        assert(ds4_qwen4exp_mtp_head_forward(&h,tokens,input,12,1,got,NULL,g_err,sizeof g_err)!=0);assert(async_calls==1&&old_calls==0);}
    float forced[HEAD_N_VOCAB]={0};g_forced_lm_logits=forced;g_forced_lm_rows=1;
    forced[HEAD_N_VOCAB-1]=5.f;mode=1;reset();
    assert(ds4_qwen4exp_mtp_head_forward(&h,tokens,input,12,1,got,NULL,g_err,sizeof g_err)==0&&got[0]==HEAD_N_VOCAB-1);
    forced[0]=NAN;reset();assert(ds4_qwen4exp_mtp_head_forward(&h,tokens,input,12,1,got,NULL,g_err,sizeof g_err)==0&&got[0]==0);
    g_forced_lm_logits=NULL;
    /* Default optional wrapper must defer to customized legacy native hooks. */
    h.hooks.native_propose_async=mtp_native_propose_async;mode=0;reset();
    assert(ds4_qwen4exp_mtp_head_forward(&h,tokens,input,12,1,got,NULL,g_err,sizeof g_err)==0&&old_calls==1&&async_calls==0);
    ds4_qwen4exp_mtp_head_free(&h);assert(!g_failures);
    puts("MTP async proposal host read/fallback/error contracts: PASS");return 0;
}
