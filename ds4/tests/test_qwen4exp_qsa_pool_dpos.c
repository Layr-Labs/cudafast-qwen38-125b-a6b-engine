#include "../ds4_gpu.h"
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
int main(void) {
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
    ds4_gpu_tensor *all[] = {ta,t,pa,p,r,n,v,dpos};
    for (unsigned i=0;i<sizeof(all)/sizeof(all[0]);i++) ds4_gpu_tensor_free(all[i]);
    free(tape);free(pool);free(raw);free(a);free(b);
    printf("QSA pool dpos: %u cases, %u failed, %u graph replays\n",cases,failed,replays);
    ds4_gpu_cleanup();
    return failed ? 1 : 0;
}
