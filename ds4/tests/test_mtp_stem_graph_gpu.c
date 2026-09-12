/* Prepared actual CUDA test: no model files. Include hooks to trace genuine
 * graph capture/replay; link mtp.c and GPU_TEST_OBJS with --gc-sections, without
 * mtp_hooks.o. This test must not be represented as executed off the GPU. */
#include "ds4_qwen4exp_mtp.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
extern void ds4_gpu_enable_q8_dense_mma(void);
static unsigned captures,replays;
static int traced_begin(const ds4_decode_graph_key *k) {
    int rc=ds4_gpu_decode_graph_begin(k);if(rc==1)++replays;return rc;
}
static int traced_end(const ds4_decode_graph_key *k) {
    int rc=ds4_gpu_decode_graph_end(k);if(rc==0)++captures;return rc;
}
#define ds4_gpu_decode_graph_begin traced_begin
#define ds4_gpu_decode_graph_end traced_end
#include "../ds4_qwen4exp_mtp_hooks.c"
#undef ds4_gpu_decode_graph_begin
#undef ds4_gpu_decode_graph_end
bool ds4_log_is_tty(FILE *fp) {(void)fp;return false;}
enum { DIM=2560,HC=4,WIDE=DIM*HC,VOCAB=32,HNORM=DIM*4,
       PROJ=(DIM+WIDE)*4,HEAD_BYTES=PROJ+DIM*160*34,TARGET_BYTES=VOCAB*80*34 };
static uint32_t rng=31;
static uint32_t next(void){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;return rng;}
static void q8(unsigned char *p,unsigned groups) {
    uint16_t scale=0x1800;
    for(unsigned g=0;g<groups;++g,p+=34){memcpy(p,&scale,2);
        for(unsigned j=0;j<32;++j)p[2+j]=(unsigned char)(int8_t)((int)(next()%31)-15);}
}
static void init_head(ds4_qwen4exp_mtp_head *h,void *head,void *target) {
    memset(h,0,sizeof(*h));h->head_map=head;h->head_size=HEAD_BYTES;
    h->target_map=target;h->target_size=TARGET_BYTES;h->token_embd_type=8;
    h->hnorm_offset=HNORM;h->eh_proj_offset=PROJ;
    h->n_embd=DIM;h->n_hc=HC;h->n_vocab=VOCAB;h->max_tokens=2;
    h->block_index=48;h->rms_eps=1e-6f;h->weight_bias=1;h->round_bf16=1;
    h->hooks.embed=ds4_gpu_qwen4exp_embed_tokens_hc_tensor;
    h->hooks.rms_norm=ds4_gpu_qwen4exp_rms_norm_tensor;
    h->hooks.ehx_pack=ds4_gpu_qwen4exp_ehx_pack_tensor;
    h->hooks.matmul_q8_0=mtp_matmul_q8_0_decode_rows;
    h->t_tokens=ds4_gpu_tensor_alloc(8);
    h->t_embed_rows=ds4_gpu_tensor_alloc(2ull*DIM*4);
    h->t_embed_out=ds4_gpu_tensor_alloc(2ull*DIM*4);
    h->t_e_normed=ds4_gpu_tensor_alloc(2ull*DIM*4);
    h->t_h_normed=ds4_gpu_tensor_alloc(2ull*WIDE*4);
    h->t_ehx=ds4_gpu_tensor_alloc(2ull*HC*2*DIM*4);
    h->t_hyper=ds4_gpu_tensor_alloc(2ull*WIDE*4);
    assert(h->t_tokens&&h->t_embed_rows&&h->t_embed_out&&h->t_e_normed&&h->t_h_normed&&h->t_ehx&&h->t_hyper);
}
static void parity(ds4_qwen4exp_mtp_head *h,unsigned width,unsigned pass) {
    float input[2*WIDE];int32_t ids[2]={(int32_t)(pass%VOCAB),(int32_t)((pass*7+3)%VOCAB)};
    for(unsigned i=0;i<2*WIDE;++i)input[i]=((int)(next()%4095)-2047)*(1.f/512);
    ds4_gpu_tensor *t[]={h->t_tokens,h->t_embed_rows,h->t_embed_out,h->t_e_normed,h->t_h_normed,h->t_ehx,h->t_hyper};
    void *actual[7];
    assert(ds4_gpu_tensor_write(h->t_tokens,0,ids,sizeof(ids)));
    assert(ds4_gpu_tensor_write(h->t_hyper,0,input,sizeof(input)));
    for(unsigned j=1;j<6;++j)assert(ds4_gpu_tensor_fill_f32(t[j],0.375f,ds4_gpu_tensor_bytes(t[j])/4));
    assert(mtp_stem(h,width));
    for(unsigned j=0;j<7;++j){size_t n=ds4_gpu_tensor_bytes(t[j]);actual[j]=malloc(n);assert(actual[j]);assert(ds4_gpu_tensor_read(t[j],0,actual[j],n));}
    assert(!memcmp(actual[0],ids,sizeof(ids)));
    if(width==1)assert(!memcmp((float *)actual[6]+WIDE,input+WIDE,WIDE*4));
    assert(ds4_gpu_tensor_write(h->t_tokens,0,ids,sizeof(ids)));
    assert(ds4_gpu_tensor_write(h->t_hyper,0,input,sizeof(input)));
    for(unsigned j=1;j<6;++j)assert(ds4_gpu_tensor_fill_f32(t[j],0.375f,ds4_gpu_tensor_bytes(t[j])/4));
    assert(ds4_qwen4exp_mtp_head_stem_eager(h,width));
    for(unsigned j=0;j<7;++j){size_t n=ds4_gpu_tensor_bytes(t[j]);void *expected=malloc(n);assert(expected);
        assert(ds4_gpu_tensor_read(t[j],0,expected,n));assert(!memcmp(actual[j],expected,n));free(actual[j]);free(expected);}
}
static void cycle(ds4_qwen4exp_mtp_head *h,unsigned width,unsigned pass) {
    unsigned c=captures,r=replays;for(unsigned i=0;i<6;++i)parity(h,width,pass+i);
    assert(captures>c&&replays>=r+2); /* Warm can invalidate itself on scratch growth. */
}
int main(void) {
    assert(ds4_gpu_init());if(!ds4_gpu_decode_graphs_supported())return 77;
    unsigned char *head=mmap(NULL,HEAD_BYTES,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    unsigned char *target=mmap(NULL,TARGET_BYTES,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    assert(head!=MAP_FAILED&&target!=MAP_FAILED);
    for(unsigned i=0;i<DIM+WIDE;++i)((float *)head)[i]=(int)(next()%31)*(1.f/128);
    q8(head+PROJ,DIM*160);q8(target,VOCAB*80);
    assert(ds4_gpu_set_model_map(target,TARGET_BYTES));
    assert(ds4_gpu_set_aux_model_map_range(head,HEAD_BYTES,0,HEAD_BYTES));
    ds4_qwen4exp_mtp_head h;init_head(&h,head,target);
    cycle(&h,1,0);cycle(&h,2,10); /* Default pre-enable narrow tile4/tile8. */
    ds4_gpu_enable_q8_dense_mma();cycle(&h,2,20); /* Same width; enable must retire. */
    h.weight_bias=0;cycle(&h,1,30);h.round_bf16=0;cycle(&h,2,40);
    unsigned r=replays;setenv("DS4_QWEN4EXP_NO_ROW_TILE","1",1);parity(&h,2,50);assert(replays==r);unsetenv("DS4_QWEN4EXP_NO_ROW_TILE");
    ds4_gpu_forget_model_map(head);assert(ds4_gpu_set_aux_model_map_range(head,HEAD_BYTES,0,HEAD_BYTES));cycle(&h,2,60);
    ds4_gpu_forget_model_map(target);assert(ds4_gpu_set_model_map(target,TARGET_BYTES));
    assert(ds4_gpu_set_aux_model_map_range(head,HEAD_BYTES,0,HEAD_BYTES));cycle(&h,1,70);
    ds4_gpu_tensor *old=h.t_e_normed;h.t_e_normed=ds4_gpu_tensor_alloc(2ull*DIM*4);assert(h.t_e_normed);cycle(&h,1,80);ds4_gpu_tensor_free(old);
    /* Grow the shared projection slab beyond both captured stem widths. */
    ds4_gpu_tensor *x=ds4_gpu_tensor_alloc(16ull*5120*4),*out=ds4_gpu_tensor_alloc(16ull*DIM*4);
    assert(x&&out&&ds4_gpu_tensor_fill_f32(x,0.25f,16ull*5120));
    assert(ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(out,head,HEAD_BYTES,PROJ,5120,DIM,x,16));
    cycle(&h,1,90);ds4_gpu_tensor_free(x);ds4_gpu_tensor_free(out);
    ds4_qwen4exp_mtp_head_free(&h);assert(!h.stem_graph_state&&!h.stem_graph_release);
    init_head(&h,head,target);cycle(&h,2,100);ds4_qwen4exp_mtp_head_free(&h);
    ds4_gpu_forget_model_map(head);ds4_gpu_forget_model_map(target);
    assert(!munmap(head,HEAD_BYTES)&&!munmap(target,TARGET_BYTES));ds4_gpu_cleanup();
    puts("MTP stem actual CUDA capture/eager parity: PASS");return 0;
}
