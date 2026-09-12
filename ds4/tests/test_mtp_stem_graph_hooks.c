/* Host graph state-machine spy over the actual stem wrapper/eager helper.
 * No CUDA arithmetic claim. Link mtp.c with function sections/gc-sections. */
#include "ds4_qwen4exp_mtp.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../ds4_qwen4exp_mtp_hooks.c"
struct ds4_gpu_tensor { uint64_t bytes; unsigned char *data; };
typedef struct { int op;ds4_gpu_tensor *a,*b;const ds4_gpu_tensor *x,*y;uint32_t n,rows,hc;float bias; } job;
static job pending[8];static unsigned count,encodes,captures,replays,retires;
static int recording,supported=1,fail_op,fail_end;
static struct { ds4_decode_graph_key key;int state;unsigned n;job work[8]; } slots[4];
ds4_gpu_tensor *ds4_gpu_tensor_alloc(uint64_t bytes){ds4_gpu_tensor*t=calloc(1,sizeof(*t));assert(t);t->bytes=bytes;t->data=calloc(1,bytes);assert(t->data);return t;}
void ds4_gpu_tensor_free(ds4_gpu_tensor*t){if(t){free(t->data);free(t);}}
uint64_t ds4_gpu_tensor_bytes(const ds4_gpu_tensor*t){return t?t->bytes:0;}
int ds4_gpu_synchronize(void){return 1;}
int ds4_gpu_tensor_copy(ds4_gpu_tensor*a,uint64_t off,const ds4_gpu_tensor*b,uint64_t src,uint64_t n){assert(!recording);if(off>a->bytes||n>a->bytes-off||src>b->bytes||n>b->bytes-src)return 0;memcpy(a->data+off,b->data+src,n);return 1;}
static void run(const job*j){float*a=(float*)j->a->data;const float*x=(float*)j->x->data;
 if(j->op==1){for(unsigned r=0;r<j->rows;++r)for(unsigned i=0;i<j->n;++i){float v=((int32_t*)j->x->data)[r]+i*(1.f/1024);a[r*j->n+i]=v;((float*)j->b->data)[r*j->n+i]=v;}}
 if(j->op==2)for(unsigned i=0;i<j->n*j->rows;++i)a[i]=x[i]*(2+j->bias);
 if(j->op==3)for(unsigned r=0;r<j->rows;++r)for(unsigned s=0;s<j->hc;++s)for(unsigned i=0;i<j->n;++i){unsigned o=(r*j->hc+s)*j->n*2;a[o+i]=x[r*j->n+i];a[o+j->n+i]=((float*)j->y->data)[(r*j->hc+s)*j->n+i];}
 if(j->op==4)for(unsigned r=0;r<j->rows;++r)for(unsigned i=0;i<j->n;++i)a[r*j->n+i]=x[r*j->n*2+i]+x[r*j->n*2+j->n+i];
}
static int emit(job j){++encodes;if(fail_op==j.op)return 0;if(recording){assert(count<8);pending[count++]=j;}else run(&j);return 1;}
int ds4_gpu_qwen4exp_embed_tokens_hc_tensor(ds4_gpu_tensor*out,ds4_gpu_tensor*scratch,const ds4_gpu_tensor*tokens,const void*map,uint64_t size,uint64_t off,uint32_t type,uint32_t vocab,uint32_t rows,uint32_t dim,uint32_t hc){(void)map;(void)size;(void)off;(void)type;(void)vocab;assert(hc==1);return emit((job){1,out,scratch,tokens,NULL,dim,rows,0,0});}
int ds4_gpu_qwen4exp_rms_norm_tensor(ds4_gpu_tensor*out,const ds4_gpu_tensor*x,const void*map,uint64_t size,uint64_t off,uint32_t n,uint32_t group,uint32_t rows,float eps,float bias,int round){(void)map;(void)size;(void)off;(void)eps;(void)round;assert(n==group);return emit((job){2,out,NULL,x,NULL,n,rows,0,bias});}
int ds4_gpu_qwen4exp_ehx_pack_tensor(ds4_gpu_tensor*out,const ds4_gpu_tensor*e,const ds4_gpu_tensor*h,uint32_t rows,uint32_t hc,uint32_t dim){return emit((job){3,out,NULL,e,h,dim,rows,hc,0});}
int ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(ds4_gpu_tensor*out,const void*map,uint64_t size,uint64_t off,uint64_t in,uint64_t dim,const ds4_gpu_tensor*x,uint32_t rows){(void)map;(void)size;(void)off;assert(in==2*dim);return emit((job){4,out,NULL,x,NULL,(uint32_t)dim,rows,0,0});}
int ds4_gpu_decode_graphs_supported(void){return supported;}
void ds4_gpu_decode_graphs_invalidate(void){assert(!recording);++retires;memset(slots,0,sizeof(slots));}
static unsigned slot(const ds4_decode_graph_key*k){for(unsigned i=0;i<4;++i)if(slots[i].state&&!memcmp(&slots[i].key,k,sizeof(*k)))return i;for(unsigned i=0;i<4;++i)if(!slots[i].state){slots[i].key=*k;return i;}abort();}
int ds4_gpu_decode_graph_begin(const ds4_decode_graph_key*k){unsigned i=slot(k);if(!slots[i].state){slots[i].state=1;return -1;}if(slots[i].state==2){for(unsigned j=0;j<slots[i].n;++j)run(&slots[i].work[j]);++replays;return 1;}recording=1;count=0;return 0;}
int ds4_gpu_decode_graph_end(const ds4_decode_graph_key*k){assert(recording);recording=0;unsigned i=slot(k);if(fail_end){fail_end=0;slots[i].state=0;return -1;}slots[i].state=2;slots[i].n=count;memcpy(slots[i].work,pending,count*sizeof(job));for(unsigned j=0;j<count;++j)run(&pending[j]);++captures;return 0;}
void ds4_gpu_decode_graph_abort(const ds4_decode_graph_key*k){assert(recording);recording=0;slots[slot(k)].state=0;}
static unsigned custom_calls;
static int custom_pack(ds4_gpu_tensor*a,const ds4_gpu_tensor*b,const ds4_gpu_tensor*c,uint32_t d,uint32_t e,uint32_t f){++custom_calls;return ds4_gpu_qwen4exp_ehx_pack_tensor(a,b,c,d,e,f);}
static void input(ds4_qwen4exp_mtp_head*h,unsigned pass){((int32_t*)h->t_tokens->data)[0]=pass;((int32_t*)h->t_tokens->data)[1]=pass+1;for(unsigned i=0;i<20480;++i)((float*)h->t_hyper->data)[i]=(float)pass+i/10240;}
int main(void){ds4_qwen4exp_mtp_head h={0};h.n_embd=2560;h.n_hc=4;h.n_vocab=128;h.max_tokens=2;h.block_index=48;h.token_embd_type=8;
 h.hooks.embed=ds4_gpu_qwen4exp_embed_tokens_hc_tensor;h.hooks.rms_norm=ds4_gpu_qwen4exp_rms_norm_tensor;h.hooks.ehx_pack=ds4_gpu_qwen4exp_ehx_pack_tensor;h.hooks.matmul_q8_0=mtp_matmul_q8_0_decode_rows;
 h.t_tokens=ds4_gpu_tensor_alloc(8);h.t_embed_rows=ds4_gpu_tensor_alloc(20480);h.t_embed_out=ds4_gpu_tensor_alloc(20480);h.t_e_normed=ds4_gpu_tensor_alloc(20480);h.t_h_normed=ds4_gpu_tensor_alloc(81920);h.t_ehx=ds4_gpu_tensor_alloc(163840);h.t_hyper=ds4_gpu_tensor_alloc(81920);
 for(unsigned width=1;width<=2;++width)for(unsigned p=0;p<5;++p){input(&h,p+1);unsigned before=encodes;assert(mtp_stem(&h,width));assert(((float*)h.t_hyper->data)[0]==4.f*(p+1));if(width==2)assert(((float*)h.t_hyper->data)[10240]==4.f*(p+2));if(p>=2)assert(encodes==before);}
 assert(captures==2&&replays==6);
 unsigned before=retires;h.weight_bias=1;input(&h,10);assert(mtp_stem(&h,2));assert(retires==before+1);
 fail_end=1;input(&h,11);assert(mtp_stem(&h,2));assert(!recording);
 input(&h,12);assert(mtp_stem(&h,2));fail_op=3;unsigned calls=encodes;input(&h,13);assert(!mtp_stem(&h,2));assert(!recording&&encodes==calls+4);fail_op=0;
 h.hooks.ehx_pack=custom_pack;input(&h,14);assert(mtp_stem(&h,1)&&custom_calls==1);h.hooks.ehx_pack=ds4_gpu_qwen4exp_ehx_pack_tensor;
 h.hooks.ehx_pack=NULL;input(&h,15);assert(mtp_stem(&h,1));h.hooks.ehx_pack=ds4_gpu_qwen4exp_ehx_pack_tensor;
 const char*flags[]={"DS4_MTP_NO_STEM_GRAPH","DS4_QWEN4EXP_NO_ROW_TILE","DS4_CUDA_NO_Q8_DP4A","DS4_CUDA_NO_Q8_MMA"};
 for(unsigned i=0;i<4;++i){setenv(flags[i],"1",1);calls=encodes;input(&h,16+i);assert(mtp_stem(&h,2)&&encodes==calls+5);unsetenv(flags[i]);}
 supported=0;input(&h,21);assert(mtp_stem(&h,1));supported=1;
 before=retires;ds4_qwen4exp_mtp_head_free(&h);assert(retires==before+1&&!h.stem_graph_state&&!h.stem_graph_release);
 puts("MTP stem host state-machine contracts: PASS");return 0;}
