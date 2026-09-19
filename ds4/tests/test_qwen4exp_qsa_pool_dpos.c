#include "../ds4_gpu.h"
#include "../ds4_qwen4exp_indexer_defer.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { CAP = 4096, DIM = 128, POOL = 4, ROT = 64, MAX_ROWS = 1024 };
static void ok(int result, const char *what) {
    if (!result) { fprintf(stderr, "failed: %s\n", what); exit(2); }
}
static ds4_gpu_tensor *alloc(uint64_t bytes) {
    ds4_gpu_tensor *p = ds4_gpu_tensor_alloc(bytes);
    ok(p != NULL, "allocation"); return p;
}
static void write_tensor(ds4_gpu_tensor *p, const void *x, uint64_t n) {
    ok(ds4_gpu_tensor_write(p, 0, x, n), "write");
}

static void policy_cases(void) {
    ds4_qwen4exp_indexer_defer_state s = {0};
    ok(ds4_qwen4exp_indexer_defer_plan(&s,true,0,1024,2048) ==
       DS4_QWEN4EXP_INDEXER_DEFER, "prefill defers");
    ok(ds4_qwen4exp_indexer_defer_commit(&s,0,1024), "prefill commit");
    ok(ds4_qwen4exp_indexer_defer_plan(&s,true,1024,1023,2048) ==
       DS4_QWEN4EXP_INDEXER_DEFER, "2047 defers");
    ok(ds4_qwen4exp_indexer_defer_commit(&s,1024,1023), "2047 commit");
    ok(ds4_qwen4exp_indexer_defer_plan(&s,true,2047,1,2048) ==
       DS4_QWEN4EXP_INDEXER_DEFER, "2048 defers");
    ok(ds4_qwen4exp_indexer_defer_commit(&s,2047,1), "2048 commit");
    ok(ds4_qwen4exp_indexer_defer_plan(&s,true,2048,1,2048) ==
       DS4_QWEN4EXP_INDEXER_MATERIALIZE_EAGER, "2049 materializes");

    /* A rejected verify truncates only the logical staged prefix.  Accepted
     * rows remain contiguous and the rejected rows are overwritten. */
    ds4_qwen4exp_indexer_defer_rollback(&s,2047);
    ok(s.rows == 2047 && !s.materialized, "reject rollback");
    ok(ds4_qwen4exp_indexer_defer_commit(&s,2047,1), "accepted replacement");
    ds4_qwen4exp_indexer_defer_materialized(&s);
    ds4_qwen4exp_indexer_defer_rollback(&s,1024);
    ok(s.materialized, "materialized rollback stays eager");
    ds4_qwen4exp_indexer_defer_reset(&s);
    ok(s.rows == 0 && !s.materialized, "reset rearms defer");

    ok(ds4_qwen4exp_indexer_defer_plan(&s,true,40,5,2048) ==
       DS4_QWEN4EXP_INDEXER_MATERIALIZE_EAGER,
       "noncontiguous head seed falls back");
    ok(ds4_qwen4exp_indexer_defer_plan(&s,false,0,1,2048) ==
       DS4_QWEN4EXP_INDEXER_EAGER, "allocation fallback is eager");

    ds4_qwen4exp_indexer_defer_state outer[6] = {0};
    outer[1].rows = outer[3].rows = outer[5].rows = 7u;
    const uint32_t layers[] = {1u,3u,5u};
    ok(ds4_qwen4exp_indexer_defer_commit_layers(
           outer,6u,layers,3u,7u,2u),
       "outer graph replay commits every QSA layer");
    ok(outer[1].rows==9u && outer[3].rows==9u && outer[5].rows==9u,
       "outer graph replay advances selected layers");
    outer[3].rows = 8u;
    ok(!ds4_qwen4exp_indexer_defer_commit_layers(
           outer,6u,layers,3u,9u,1u),
       "outer graph replay rejects a discontinuous layer");
    ok(outer[1].rows==9u && outer[3].rows==8u && outer[5].rows==9u,
       "outer graph replay failure is atomic");
}

#if !defined(__APPLE__) && !defined(DS4_ROCM_BUILD)
static void compare_prefix(const char *what, ds4_gpu_tensor *a,
                           ds4_gpu_tensor *b, uint64_t bytes) {
    void *av = malloc(bytes), *bv = malloc(bytes);
    ok(av && bv, "comparison allocation");
    ok(ds4_gpu_tensor_read(a,0,av,bytes), "reference read");
    ok(ds4_gpu_tensor_read(b,0,bv,bytes), "deferred read");
    if (memcmp(av,bv,bytes) != 0) {
        fprintf(stderr, "failed: %s differs\n", what);
        exit(2);
    }
    free(av); free(bv);
}

static void deferred_threshold_case(const unsigned *chunks, unsigned nchunks,
                                    unsigned prefix, int append_one) {
    const uint64_t tape_bytes = (uint64_t)CAP * DIM * sizeof(float);
    const uint64_t pool_bytes = tape_bytes / POOL;
    const uint64_t row_bytes = (uint64_t)MAX_ROWS * DIM * sizeof(float);
    float *all = calloc((size_t)CAP * DIM, sizeof(float));
    float *part = malloc(row_bytes);
    float norm[DIM], inv[ROT / 2];
    ok(all && part, "defer host allocation");
    for (size_t i=0;i<(size_t)CAP*DIM;i++)
        all[i] = 0.17f * sinf((float)(i % 8191u) * 0.019f);
    for (unsigned i=0;i<DIM;i++) norm[i] = 0.9f + i*0.0007f;
    for (unsigned i=0;i<ROT/2;i++) inv[i] = powf(10000.0f,-2.0f*i/ROT);

    ds4_gpu_tensor *et=alloc(tape_bytes), *ep=alloc(pool_bytes);
    ds4_gpu_tensor *dt=alloc(tape_bytes), *dp=alloc(pool_bytes);
    ds4_gpu_tensor *deferred=alloc((uint64_t)2048*DIM*sizeof(float));
    ds4_gpu_tensor *rows=alloc(row_bytes), *nw=alloc(sizeof(norm));
    ds4_gpu_tensor *iv=alloc(sizeof(inv)), *dpos=alloc(sizeof(uint32_t));
    ok(ds4_gpu_tensor_fill_f32(et,0.0f,tape_bytes/4), "eager tape zero");
    ok(ds4_gpu_tensor_fill_f32(ep,0.0f,pool_bytes/4), "eager pool zero");
    ok(ds4_gpu_tensor_fill_f32(dt,0.0f,tape_bytes/4), "defer tape zero");
    ok(ds4_gpu_tensor_fill_f32(dp,0.0f,pool_bytes/4), "defer pool zero");
    ok(ds4_gpu_tensor_fill_f32(deferred,0.0f,(uint64_t)2048*DIM), "stage zero");
    write_tensor(nw,norm,sizeof(norm)); write_tensor(iv,inv,sizeof(inv));

    unsigned pos=0;
    for (unsigned ci=0;ci<nchunks;ci++) {
        const unsigned n=chunks[ci];
        ok(n<=MAX_ROWS && pos+n<=prefix, "valid ragged chunk");
        memcpy(part,all+(size_t)pos*DIM,(size_t)n*DIM*sizeof(float));
        write_tensor(rows,part,(uint64_t)n*DIM*sizeof(float));
        ok(ds4_gpu_qwen4exp_qsa_indexer_pool_update_tensor(
            ep,et,rows,nw,iv,pos,n,CAP,DIM,POOL,ROT,1e-6f,0.0f),
            "eager prefix update");
        ok(ds4_gpu_qwen4exp_update_dpos(dpos,pos), "defer position");
        ok(ds4_gpu_qwen4exp_indexer_defer_dpos_tensor(
            deferred,rows,0,n,2048,DIM,dpos), "defer prefix copy");
        pos += n;
    }
    ok(pos==prefix, "ragged chunks cover prefix");
    ok(ds4_gpu_qwen4exp_qsa_indexer_pool_update_tensor(
        dp,dt,deferred,nw,iv,0,prefix,CAP,DIM,POOL,ROT,1e-6f,0.0f),
        "deferred prefix materialization");
    if (append_one) {
        memcpy(part,all+(size_t)prefix*DIM,DIM*sizeof(float));
        write_tensor(rows,part,(uint64_t)DIM*sizeof(float));
        ok(ds4_gpu_qwen4exp_qsa_indexer_pool_update_tensor(
            ep,et,rows,nw,iv,prefix,1,CAP,DIM,POOL,ROT,1e-6f,0.0f),
            "eager crossing row");
        ok(ds4_gpu_qwen4exp_qsa_indexer_pool_update_tensor(
            dp,dt,rows,nw,iv,prefix,1,CAP,DIM,POOL,ROT,1e-6f,0.0f),
            "deferred crossing row");
        prefix++;
    }
    ok(ds4_gpu_synchronize(), "threshold synchronize");
    compare_prefix("materialized tape",et,dt,(uint64_t)prefix*DIM*sizeof(float));
    compare_prefix("materialized pool",ep,dp,
                   (uint64_t)(prefix/POOL)*DIM*sizeof(float));
    ds4_gpu_tensor *ts[]={et,ep,dt,dp,deferred,rows,nw,iv,dpos};
    for(unsigned i=0;i<sizeof(ts)/sizeof(ts[0]);i++) ds4_gpu_tensor_free(ts[i]);
    free(all); free(part);
}

static void deferred_graph_replay_case(void) {
    const uint64_t bytes=(uint64_t)2048*DIM*sizeof(float);
    float row[DIM], got[DIM];
    ds4_gpu_tensor *deferred=alloc(bytes), *src=alloc(sizeof(row));
    ds4_gpu_tensor *dpos=alloc(sizeof(uint32_t));
    ok(ds4_gpu_tensor_fill_f32(deferred,0.0f,bytes/4), "graph stage zero");
    ds4_decode_graph_key key={0};
    key.il=62; key.island=2; key.variant=1;
    key.cur_hc=deferred; key.after_attn_hc=src;
    const unsigned positions[]={0,1,2047};
    unsigned replays=0;
    for(unsigned j=0;j<3;j++) {
        for(unsigned i=0;i<DIM;i++) row[i]=(float)(1000*j+i)*0.003f;
        write_tensor(src,row,sizeof(row));
        ok(ds4_gpu_qwen4exp_update_dpos(dpos,positions[j]), "graph dpos");
        int state=ds4_gpu_decode_graph_begin(&key);
        if(state==1) replays++;
        else {
            ok(state==0 || state==-1, "graph warm/capture state");
            ok(ds4_gpu_qwen4exp_indexer_defer_dpos_tensor(
                deferred,src,99,1,2048,DIM,dpos), "captured defer copy");
            if(state==0) ok(ds4_gpu_decode_graph_end(&key)==0,
                            "defer capture end");
        }
        ok(ds4_gpu_synchronize(), "defer graph synchronize");
        ok(ds4_gpu_tensor_read(deferred,(uint64_t)positions[j]*sizeof(row),
                               got,sizeof(got)), "defer graph read");
        ok(memcmp(row,got,sizeof(row))==0, "dpos graph replay exact");
    }
    ok(replays==1, "defer graph replayed");
    ds4_gpu_decode_graphs_invalidate();
    ds4_gpu_tensor_free(deferred); ds4_gpu_tensor_free(src);
    ds4_gpu_tensor_free(dpos);
}
#endif

int main(void) {
    policy_cases();
    ok(ds4_gpu_init(), "GPU init");
    const size_t tb = (size_t)CAP * DIM * sizeof(float);
    const size_t pb = tb / POOL;
    const size_t rb = (size_t)MAX_ROWS * DIM * sizeof(float);
    float *tape = malloc(tb), *pool = malloc(pb), *raw = malloc(rb);
    float *a = malloc(tb), *b = malloc(tb);
    float norm[DIM], inv[ROT / 2];
    ok(tape && pool && raw && a && b, "host allocation");
    for (size_t i = 0; i < tb / sizeof(float); i++)
        tape[i] = 0.3f * sinf((float)(i % 10007u) * 0.031f);
    for (size_t i = 0; i < pb / sizeof(float); i++) pool[i] = -9.0f;
    for (unsigned i = 0; i < DIM; i++) norm[i] = 0.8f + i * 0.001f;
    for (unsigned i = 0; i < ROT / 2; i++) inv[i] = powf(10000.0f, -2.0f * i / ROT);
    ds4_gpu_tensor *ta = alloc(tb), *t = alloc(tb), *pa = alloc(pb), *p = alloc(pb);
    ds4_gpu_tensor *r = alloc(rb), *n = alloc(sizeof(norm)), *v = alloc(sizeof(inv));
    ds4_gpu_tensor *dpos = alloc(sizeof(uint32_t));
    write_tensor(n, norm, sizeof(norm)); write_tensor(v, inv, sizeof(inv));
    const unsigned widths[] = {1,2,3,4,7,8,63,64,1024};
    const unsigned positions[] = {0,1,2,3,4,7,1023,2047};
    unsigned cases = 0, failed = 0, replays = 0;
    for (unsigned graph = 0; graph < 2; graph++) {
        for (unsigned wi = 0; wi < sizeof(widths)/sizeof(widths[0]); wi++) {
            const unsigned rows = widths[wi];
            if (graph && (rows > 2 || !ds4_gpu_decode_graphs_supported())) continue;
            ds4_decode_graph_key key;
            memset(&key, 0, sizeof(key));
            key.il = 62; key.island = 2; key.variant = rows;
            key.cur_hc = t; key.after_attn_hc = p;
            for (unsigned pi = 0; pi < sizeof(positions)/sizeof(positions[0]); pi++) {
                const unsigned pos = positions[pi];
                for (size_t i = 0; i < rb / sizeof(float); i++)
                    raw[i] = 0.2f * cosf((float)((i + pos * 3) % 997u) * 0.11f);
                write_tensor(ta,tape,tb); write_tensor(t,tape,tb);
                write_tensor(pa,pool,pb); write_tensor(p,pool,pb);
                write_tensor(r,raw,rb);
                ok(ds4_gpu_qwen4exp_update_dpos(dpos,pos), "update position");
                ok(ds4_gpu_qwen4exp_qsa_indexer_pool_update_tensor(
                       pa,ta,r,n,v,pos,rows,CAP,DIM,POOL,ROT,1e-6f,0.0f), "static pool");
                int state = graph ? ds4_gpu_decode_graph_begin(&key) : -1;
                if (state != 1) {
                    /* In graph mode the stale host pos deliberately differs.
                     * Replays must consume the freshly updated device value. */
                    ok(ds4_gpu_qwen4exp_qsa_indexer_pool_update_dpos_tensor(
                           p,t,r,n,v,graph ? 0u : pos,rows,CAP,DIM,POOL,ROT,
                           1e-6f,0.0f,dpos), "device pool");
                    if (state == 0) ok(ds4_gpu_decode_graph_end(&key) == 0, "capture end");
                } else replays++;
                ok(ds4_gpu_synchronize(), "synchronize");
                ok(ds4_gpu_tensor_read(ta,0,a,tb), "reference tape read");
                ok(ds4_gpu_tensor_read(t,0,b,tb), "candidate tape read");
                ok(memcmp(a,b,tb)==0, "exact tape");
                ok(ds4_gpu_tensor_read(pa,0,a,pb), "reference pool read");
                ok(ds4_gpu_tensor_read(p,0,b,pb), "candidate pool read");
                if (memcmp(a,b,pb)) {
                    unsigned changed = 0;
                    for (unsigned i=0;i<CAP/POOL;i++)
                        changed += memcmp(a+(size_t)i*DIM,b+(size_t)i*DIM,DIM*sizeof(float)) != 0;
                    fprintf(stderr,"pool mismatch rows=%u pos=%u graph=%u: %u blocks differ\n",
                            rows,pos,graph,changed);
                    failed++;
                }
                cases++;
            }
        }
    }
    if (ds4_gpu_decode_graphs_supported()) ok(replays == 12u, "all expected graph replays");
    ds4_gpu_decode_graphs_invalidate();
#if !defined(__APPLE__) && !defined(DS4_ROCM_BUILD)
    {
        const unsigned c2047[]={1024,1023};
        const unsigned c2048[]={3,509,1021,515};
        deferred_threshold_case(c2047,2,2047,0);
        deferred_threshold_case(c2048,4,2048,0);
        deferred_threshold_case(c2048,4,2048,1);
        if (ds4_gpu_decode_graphs_supported()) deferred_graph_replay_case();
    }
#endif
    ds4_gpu_tensor *all[] = {ta,t,pa,p,r,n,v,dpos};
    for (unsigned i=0;i<sizeof(all)/sizeof(all[0]);i++) ds4_gpu_tensor_free(all[i]);
    free(tape);free(pool);free(raw);free(a);free(b);
    printf("QSA pool dpos: %u cases, %u failed, %u graph replays\n",cases,failed,replays);
    ds4_gpu_cleanup();
    return failed ? 1 : 0;
}
