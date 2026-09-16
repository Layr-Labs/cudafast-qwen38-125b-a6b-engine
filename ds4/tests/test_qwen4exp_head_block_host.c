/* Actual production encoder/dispatcher over a small stateful host backend.
 * This tests row selection, cache state and graph lifetimes, not CUDA math. */
#include "../ds4_gpu.h"
#include "../ds4_qwen4exp_gdn_replay.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { DS4_MAX_LAYER = 64, DS4_N_INDEXER_TOP_K = 2048,
       DS4_QWEN4EXP_MTP_MAX_COMMIT = 7, DS4_N_EMBD = 2, DS4_N_HC = 2,
       ROW = 4, ROWS = 8, CTX = 4096 };
struct ds4_gpu_tensor { uint64_t bytes; float data[ROWS * ROW]; };
typedef struct { int id; } ds4_tensor;
typedef struct { int id; } ds4_model;
typedef struct {
    bool is_full_attention;
    const ds4_tensor *hc_attn_norm, *hc_attn_down, *hc_attn_up, *hc_attn_inject;
    const ds4_tensor *hc_ffn_norm, *hc_ffn_down, *hc_ffn_up, *hc_ffn_inject;
    float hc_attn_norm_offset, hc_ffn_norm_offset;
} ds4_qwen4exp_layer_weights;
typedef struct {
    struct { uint32_t n_batch, n_ctx; } plan;
    ds4_gpu_tensor *hyper, *d_pos, *qsa_k[DS4_MAX_LAYER];
    ds4_gpu_tensor *mixed, *block_out, *inject;
    uint32_t pos, spec_snapshot_rows, gdn_replay_phase;
    bool gdn_replay_active;
    bool state_dirty;
    float kv[CTX];
    unsigned writes[CTX];
} ds4_qwen4exp_session;
typedef struct {
    const ds4_qwen4exp_layer_weights *l;
    const ds4_model *m;
} ds4_qwen4exp_head_block_ctx;
static int failures, checks, qw_moe_in_head_block;
static unsigned qw_hb_stage_calls;
#define CHECK(c) do { checks++; if (!(c)) { failures++; \
    fprintf(stderr,"line %d: %s\n",__LINE__,#c); } } while (0)
#define QW_HB_TICK(slot) do { (void)tmark; } while (0)
static uint64_t qw_hb_now_ns(void) { return 0; }

enum { MIX_ATTN, QSA, INJECT, COPY, MIX_FFN, MOE };
typedef struct {
    int kind;
    ds4_qwen4exp_session *s;
    ds4_gpu_tensor *in, *out, *aux, *position;
    uint32_t rows, pos;
    uint64_t src_off, dst_off, bytes;
} op;
static struct {
    bool supported, timing, capture, have_exec, fail_end;
    int begin_result;
    unsigned calls, fail_at, updates, fail_update, begins, ends, aborts;
    unsigned attn_rows, ffn_rows, qsa_rows, moe_rows, copies, executions;
    op pending[16], exec[16];
    unsigned n_pending, n_exec;
    ds4_decode_graph_key key, exec_key;
} g;
static int qw_hb_time_on(void) { return g.timing; }
static void reset(void) {
    memset(&g,0,sizeof(g)); g.supported=true; g.begin_result=-1;
    unsetenv("DS4_QWEN4EXP_TIME_SLICES");
}
static void execute(op x) {
    g.executions++;
    if (x.kind==COPY) {
        CHECK(x.src_off+x.bytes<=x.in->bytes && x.dst_off+x.bytes<=x.out->bytes);
        if (x.src_off+x.bytes>x.in->bytes || x.dst_off+x.bytes>x.out->bytes) return;
        memcpy((char *)x.out->data+x.dst_off,(char *)x.in->data+x.src_off,x.bytes);
        return;
    }
    if (x.kind==MIX_ATTN || x.kind==MIX_FFN) {
        CHECK((uint64_t)x.rows*ROW*sizeof(float)<=x.in->bytes);
        if ((uint64_t)x.rows*ROW*sizeof(float)>x.in->bytes) return;
        for (unsigned t=0;t<x.rows;t++) {
            float sum=0;
            for (unsigned d=0;d<ROW;d++) sum+=x.in->data[t*ROW+d];
            x.out->data[t]=sum;
            x.aux->data[t]=x.kind==MIX_ATTN ? 0.25f : 0.125f;
        }
    } else if (x.kind==QSA) {
        const unsigned pos=x.position ? (unsigned)x.position->data[0] : x.pos;
        CHECK((uint64_t)pos+x.rows<=CTX);
        if ((uint64_t)pos+x.rows>CTX) return;
        for (unsigned t=0;t<x.rows;t++) {
            x.s->kv[pos+t]=x.in->data[t]+(float)(pos+t)*0.03125f;
            x.s->writes[pos+t]++;
            float sum=0;
            for (unsigned p=0;p<=pos+t;p++) sum+=x.s->kv[p];
            x.out->data[t]=sum*0.0625f;
        }
    } else if (x.kind==MOE) {
        for (unsigned t=0;t<x.rows;t++) x.out->data[t]=x.in->data[t]*0.0625f+3.0f;
    } else {
        CHECK((uint64_t)x.rows*ROW*sizeof(float)<=x.out->bytes);
        if ((uint64_t)x.rows*ROW*sizeof(float)>x.out->bytes) return;
        for (unsigned t=0;t<x.rows;t++)
            for (unsigned d=0;d<ROW;d++)
                x.out->data[t*ROW+d]+=x.in->data[t]*x.aux->data[t];
    }
}
static bool emit(op x) {
    if (++g.calls==g.fail_at) return false;
    if (g.capture) {
        CHECK(g.n_pending<16);
        if (g.n_pending>=16) return false;
        g.pending[g.n_pending++]=x;
    } else execute(x);
    return true;
}
int ds4_gpu_decode_graphs_supported(void) { return g.supported; }
int ds4_gpu_qwen4exp_update_dpos(ds4_gpu_tensor *p,uint32_t pos) {
    if (++g.updates==g.fail_update) return 0;
    p->data[0]=(float)pos; return 1;
}
int ds4_gpu_decode_graph_begin(const ds4_decode_graph_key *key) {
    g.begins++; g.key=*key;
    if (g.begin_result==0) { g.capture=true; g.n_pending=0; }
    if (g.begin_result==1) {
        CHECK(g.have_exec && !memcmp(key,&g.exec_key,sizeof(*key)));
        if (!g.have_exec) return -1;
        for (unsigned i=0;i<g.n_exec;i++) execute(g.exec[i]);
    }
    return g.begin_result;
}
int ds4_gpu_decode_graph_end(const ds4_decode_graph_key *key) {
    CHECK(g.capture); g.ends++; g.capture=false;
    if (g.fail_end) return -1;
    memcpy(g.exec,g.pending,sizeof(g.exec)); g.n_exec=g.n_pending;
    g.exec_key=*key; g.have_exec=true;
    for (unsigned i=0;i<g.n_exec;i++) execute(g.exec[i]);
    return 0;
}
void ds4_gpu_decode_graph_abort(const ds4_decode_graph_key *key) {
    (void)key; CHECK(g.capture); g.aborts++; g.capture=false; g.n_pending=0;
}
int ds4_gpu_tensor_copy(ds4_gpu_tensor *dst,uint64_t dst_off,
        const ds4_gpu_tensor *src,uint64_t src_off,uint64_t bytes) {
    g.copies++;
    return emit((op){.kind=COPY,.in=(ds4_gpu_tensor *)src,.out=dst,
        .src_off=src_off,.dst_off=dst_off,.bytes=bytes});
}
static bool qwen4exp_graph_residual(ds4_qwen4exp_session *s,const ds4_model *m,
        const ds4_tensor *norm,const ds4_tensor *down,const ds4_tensor *up,
        const ds4_tensor *inject,float offset,uint32_t rows) {
    (void)m;(void)down;(void)up;(void)inject;(void)offset;
    const bool attn=norm->id==1;
    if (attn) g.attn_rows=rows; else g.ffn_rows=rows;
    return emit((op){.kind=attn?MIX_ATTN:MIX_FFN,.in=s->hyper,
        .out=s->mixed,.aux=s->inject,.rows=rows});
}
static bool qwen4exp_graph_qsa_block(ds4_qwen4exp_session *s,
        const ds4_qwen4exp_layer_weights *l,const ds4_model *m,
        uint32_t il,uint32_t rows) {
    (void)l;(void)m;CHECK(il==48);g.qsa_rows=rows;
    return emit((op){.kind=QSA,.s=s,.in=s->mixed,.out=s->block_out,
        .position=s->d_pos,.pos=s->pos,.rows=rows});
}
static bool qwen4exp_graph_moe_block(ds4_qwen4exp_session *s,
        const ds4_qwen4exp_layer_weights *l,const ds4_model *m,
        uint32_t il,uint32_t rows) {
    (void)l;(void)m;(void)il;CHECK(qw_moe_in_head_block==1);g.moe_rows=rows;
    return emit((op){.kind=MOE,.in=s->mixed,.out=s->block_out,.rows=rows});
}
int ds4_gpu_qwen4exp_hc_inject_tensor(ds4_gpu_tensor *out,
        const ds4_gpu_tensor *hyper,const ds4_gpu_tensor *block_out,
        const ds4_gpu_tensor *inject,uint32_t embd,uint32_t hc,uint32_t rows) {
    CHECK(out==hyper && embd*hc==ROW);
    return emit((op){.kind=INJECT,.in=(ds4_gpu_tensor *)block_out,.out=out,
        .aux=(ds4_gpu_tensor *)inject,.rows=rows});
}

#include "../ds4_qwen4exp_head_block.inc"

typedef struct {
    ds4_qwen4exp_session s;
    ds4_gpu_tensor target,input,last,position,kv,mixed,out,inject;
    ds4_tensor attn_norm,ffn_norm;
    ds4_qwen4exp_layer_weights weights;
    ds4_model model;
    ds4_qwen4exp_head_block_ctx ctx;
} fixture;
static void input(fixture *f,float bias) {
    for (unsigned i=0;i<ROWS*ROW;i++) f->input.data[i]=(float)(i%11)*0.25f+bias;
    for (unsigned i=0;i<ROWS*ROW;i++) f->last.data[i]=-999.0f;
}
static void init(fixture *f) {
    memset(f,0,sizeof(*f));
    f->target.bytes=f->input.bytes=f->mixed.bytes=f->out.bytes=f->inject.bytes=sizeof(f->input.data);
    f->last.bytes=ROW*sizeof(float); f->position.bytes=sizeof(float);
    f->s.plan.n_batch=ROWS;f->s.plan.n_ctx=CTX;
    f->s.hyper=&f->target;f->s.pos=37;f->position.data[0]=37;
    f->s.d_pos=&f->position;f->s.qsa_k[48]=&f->kv;
    f->s.mixed=&f->mixed;f->s.block_out=&f->out;f->s.inject=&f->inject;
    f->attn_norm.id=1;f->ffn_norm.id=2;
    f->weights.is_full_attention=true;
    f->weights.hc_attn_norm=&f->attn_norm;f->weights.hc_ffn_norm=&f->ffn_norm;
    f->ctx.l=&f->weights;f->ctx.m=&f->model;
    for (unsigned i=0;i<CTX;i++) f->s.kv[i]=(float)(i%5)*0.03125f;
    input(f,0);
}
static int forward(fixture *f,unsigned pos,unsigned rows,bool last) {
    return last ? ds4_qwen4exp_graph_head_block_last(&f->ctx,&f->s,
        &f->input,&f->last,48,pos,rows) : ds4_qwen4exp_graph_head_block(
        &f->ctx,&f->s,&f->input,48,pos,rows);
}
static void restored(fixture *f) {
    CHECK(f->s.hyper==&f->target && f->s.pos==37 && !qw_moe_in_head_block);
    if (f->s.d_pos) CHECK(f->position.data[0]==37);
}
static void equal_last(fixture *a,fixture *b,unsigned rows) {
    CHECK(!memcmp(a->input.data+(rows-1)*ROW,b->last.data,ROW*sizeof(float)));
    CHECK(!memcmp(a->s.kv,b->s.kv,sizeof(a->s.kv)));
    CHECK(!memcmp(a->s.writes,b->s.writes,sizeof(a->s.writes)));
    for (unsigned i=ROW;i<ROWS*ROW;i++) CHECK(b->last.data[i]==-999.0f);
    restored(a);restored(b);
}
static void test_rows_and_replay(void) {
    const unsigned positions[]={0,3,4,2047,2048};
    for (unsigned rows=1;rows<=7;rows++) for (unsigned p=0;p<5;p++) {
        fixture a,b;init(&a);init(&b);reset();
        CHECK(forward(&a,positions[p],rows,false));
        CHECK(g.attn_rows==rows && g.ffn_rows==rows && g.moe_rows==rows);
        reset();CHECK(forward(&b,positions[p],rows,true));
        CHECK(g.attn_rows==rows && g.qsa_rows==rows && g.ffn_rows==1 && g.moe_rows==1);
        CHECK(g.copies==1);equal_last(&a,&b,rows);
    }
    fixture a,b;init(&a);init(&b);reset();
    for (unsigned round=0;round<3;round++) {
        const unsigned pos=3+round*5;
        input(&a,(float)round+0.5f);input(&b,(float)round+0.5f);
        g.begin_result=-1;CHECK(forward(&a,pos,2,false));
        g.begin_result=round==0 ? -1 : round==1 ? 0 : 1;
        CHECK(forward(&b,pos,2,true));equal_last(&a,&b,2);
    }
    CHECK(g.ends==1 && g.have_exec && g.exec_key._pad==0x4d54504cu);
}
static void test_failures(void) {
    for (unsigned stage=1;stage<=7;stage++) {
        fixture f;init(&f);reset();g.fail_at=stage;
        CHECK(!forward(&f,3,2,true));restored(&f);
        CHECK(g.calls==stage); /* no eager retry after partial execution */
        init(&f);reset();g.fail_at=stage;g.begin_result=0;
        CHECK(forward(&f,3,2,true));restored(&f);
        CHECK(g.aborts==1 && !g.capture);
        CHECK(f.s.writes[3]==1 && f.s.writes[4]==1);
    }
    fixture a,b;init(&a);init(&b);reset();CHECK(forward(&a,3,2,false));
    reset();g.begin_result=0;g.fail_end=true;CHECK(forward(&b,3,2,true));
    CHECK(g.ends==1 && !g.capture);equal_last(&a,&b,2);
    for (unsigned update=1;update<=2;update++) {
        init(&b);reset();g.fail_update=update;
        CHECK(!forward(&b,3,2,true));
        CHECK(b.s.hyper==&b.target && b.s.pos==37);
        CHECK(b.s.writes[3]==(update==1?0u:1u));
    }
}
static void test_keys_and_envelope(void) {
    fixture f;init(&f);reset();CHECK(forward(&f,3,2,true));
    const ds4_decode_graph_key key=g.key;
    CHECK(key.cur_hc==&f.input && key.after_attn_hc==&f.s &&
          key.after_ffn_hc==&f.last && key.attn_norm==&f.ctx);
    ds4_gpu_tensor output=f.last;
    CHECK(ds4_qwen4exp_graph_head_block_last(&f.ctx,&f.s,&f.input,&output,48,8,2));
    CHECK(g.key.after_ffn_hc!=key.after_ffn_hc);
    CHECK(forward(&f,12,2,false));CHECK(g.key._pad!=key._pad);
    /* Retain current-main replay parity and kernel mode in both graph forms. */
    for (unsigned last=0;last<2;last++)
        for (unsigned phase=0;phase<2;phase++)
            for (unsigned active=0;active<2;active++) {
                init(&f);reset();f.s.spec_snapshot_rows=1;
                f.s.gdn_replay_phase=phase;f.s.gdn_replay_active=active;
                CHECK(forward(&f,3,2,last!=0));
                CHECK(g.key.variant==(2u|(1u<<8u)|(phase<<16u)|(active<<17u)));
                restored(&f);
            }
    for (unsigned mode=0;mode<5;mode++) {
        init(&f);reset();unsigned pos=3;
        if (mode==0) f.s.d_pos=NULL;
        if (mode==1) g.supported=false;
        if (mode==2) g.timing=true;
        if (mode==3) setenv("DS4_QWEN4EXP_TIME_SLICES","1",1);
        if (mode==4) pos=2048;
        CHECK(forward(&f,pos,2,true));CHECK(!g.begins);restored(&f);
    }
    init(&f);reset();
    CHECK(!ds4_qwen4exp_graph_head_block_last(&f.ctx,&f.s,&f.input,NULL,48,3,2));
    CHECK(!ds4_qwen4exp_graph_head_block_last(&f.ctx,&f.s,&f.input,&f.input,48,3,2));
    CHECK(!forward(&f,3,8,true));CHECK(!forward(&f,UINT32_MAX,2,true));
    CHECK(!g.calls && !f.s.state_dirty);
    unsetenv("DS4_QWEN4EXP_TIME_SLICES");
}
int main(void) {
    test_rows_and_replay();test_failures();test_keys_and_envelope();
    printf("head-block host: %d checks, %d failures\n",checks,failures);
    return failures ? 1 : 0;
}
