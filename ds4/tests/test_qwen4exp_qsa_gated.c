/* Exact comparison of gated QSA against attention followed by output gating.
 * Synthetic device data only. Link against the normal CUDA engine library:
 * cc -O2 -std=c11 -D_GNU_SOURCE -Ids4 ds4/tests/test_qwen4exp_qsa_gated.c \
 *   -L.build/ds4 -lds4qwen -lm -o /tmp/test-qsa-gated
 * LD_LIBRARY_PATH=.build/ds4 /tmp/test-qsa-gated
 * Includes real graph replay with changed gates, counts and device position. */
#include "ds4_gpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { HEADS = 24, KV_HEADS = 2, DIM = 256, CAP = 1280, WIDTH = 3, GUARD = 16 };
static void ok(int value, const char *what) {
    if (!value) { fprintf(stderr, "QSA gated: %s\n", what); exit(1); }
}
static ds4_gpu_tensor *alloc(size_t bytes) {
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(bytes); ok(t != NULL, "allocate"); return t;
}
static void put(ds4_gpu_tensor *t, const void *p, size_t n) {
    ok(ds4_gpu_tensor_write(t, 0, p, n), "write");
}
static void get(ds4_gpu_tensor *t, void *p, size_t n) {
    ok(ds4_gpu_tensor_read(t, 0, p, n), "read");
}
int main(void) {
    ok(setenv("DS4_CUDA_DECODE_GRAPHS", "1", 1) == 0, "enable graphs");
    ok(unsetenv("DS4_QWEN4EXP_NO_QSA_SPLIT") == 0, "enable split");
    ok(unsetenv("DS4_QWEN4EXP_QSA_SPLIT_GROUP") == 0, "default split width");
    ok(unsetenv("DS4_QWEN4EXP_NO_QSA_FOLD_GATE") == 0, "enable fused fold");
    ok(ds4_gpu_init(), "GPU required");
    const size_t values = WIDTH * HEADS * DIM, bytes = (values + GUARD) * sizeof(float);
    const size_t kv_values = CAP * KV_HEADS * DIM, kv_bytes = kv_values * sizeof(float);
    float *host = malloc(kv_bytes), *gates = malloc(bytes), *a = malloc(bytes), *b = malloc(bytes);
    int32_t selected[WIDTH * CAP], counts[WIDTH];
    ok(host && gates && a && b, "host allocations");
    ds4_gpu_tensor *q = alloc(bytes), *k = alloc(kv_bytes), *v = alloc(kv_bytes);
    ds4_gpu_tensor *ref = alloc(bytes), *cand = alloc(bytes), *gate = alloc(bytes);
    ds4_gpu_tensor *sel = alloc(sizeof(selected)), *cnt = alloc(sizeof(counts)), *pos = alloc(4);
    const uint64_t scratch_bytes = ds4_gpu_qwen4exp_qsa_split_scratch_bytes(WIDTH, HEADS, DIM, CAP);
    ok(scratch_bytes > 0, "split shape");
    ds4_gpu_tensor *scratch = alloc(scratch_bytes);
    for (size_t i = 0; i < kv_values; i++) host[i] = ((int)(i % 101) - 50) * 0.01f;
    put(k, host, kv_bytes);
    for (size_t i = 0; i < kv_values; i++) host[i] = ((int)(i % 79) - 39) * 0.013f;
    put(v, host, kv_bytes);
    for (size_t i = 0; i < values + GUARD; i++) gates[i] = ((int)(i % 53) - 26) * 0.017f;
    put(q, gates, bytes);
    /* Keep one capture alive while device position crosses tile boundaries.
     * The last sparse trial changes its count to zero after multi-tile work. */
    static const uint32_t positions[] = {
        2u, 254u, 255u, 256u, 257u, 510u, 511u, 512u, 513u,
        1022u, 1023u, 1024u, 1025u, 1277u, 1277u
    };
    const unsigned trials = sizeof(positions) / sizeof(positions[0]);
    unsigned replays = 0;
    for (unsigned width = 1; width <= WIDTH; width++) {
        for (unsigned sparse = 0; sparse <= 1; sparse++) {
            ds4_gpu_decode_graphs_invalidate();
            ds4_decode_graph_key key; memset(&key, 0, sizeof(key));
            key.cur_hc = q; key.after_attn_hc = cand;
            for (unsigned trial = 0; trial < trials; trial++) {
                uint32_t position = positions[trial];
                put(pos, &position, sizeof(position));
                for (unsigned t = 0; t < WIDTH; t++) {
                    counts[t] = trial + 1u == trials ? 0 : (int32_t)(position + t + 1);
                    for (unsigned j = 0; j < CAP; j++)
                        selected[t * CAP + j] = j % 7 == 3 ? -1 :
                            j % 13 == 4 ? CAP + 1 : (int32_t)j;
                }
                put(sel, selected, sizeof(selected)); put(cnt, counts, sizeof(counts));
                const float special[] = {0.0f, -0.0f, INFINITY, -INFINITY, NAN, 80.0f, -80.0f};
                for (size_t i = 0; i < values + GUARD; i++)
                    gates[i] = (i % 11 < 7) ? special[(i + trial) % 7] : (float)trial - 2.0f;
                put(gate, gates, bytes);
                memset(a, 0x5a, bytes); put(ref, a, bytes); put(cand, a, bytes);
#define ATTN(OUT) ds4_gpu_qwen4exp_qsa_attention_dpos_tensor(OUT,q,k,v, \
    sparse ? sel : NULL,sparse ? cnt : NULL,width,HEADS,KV_HEADS,DIM,0,CAP,CAP, \
    0.0625f,pos,scratch,CAP)
#define GATED(OUT,GATE) ds4_gpu_qwen4exp_qsa_attention_gated_dpos_tensor(OUT,q,k,v, \
    sparse ? sel : NULL,sparse ? cnt : NULL,width,HEADS,KV_HEADS,DIM,0,CAP,CAP, \
    0.0625f,pos,scratch,CAP,GATE)
                ok(ATTN(ref), "reference attention");
                ok(ds4_gpu_qwen4exp_qsa_output_gate_tensor(ref,gate,width*HEADS*DIM), "reference gate");
                get(ref,a,bytes);
                int state = ds4_gpu_decode_graph_begin(&key);
                if (trial >= 2) ok(state == 1, "actual graph replay required");
                if (state != 1) {
                    ok(state == 0 || state == -1, "graph state");
                    ok(GATED(cand,gate), "gated attention");
                    if (state == 0) ok(ds4_gpu_decode_graph_end(&key) == 0, "capture end");
                } else replays++;
                get(cand,b,bytes); ok(memcmp(a,b,bytes) == 0, "exact output and guards");
                get(gate,b,bytes); ok(memcmp(gates,b,bytes) == 0, "gate unchanged");
                /* In-place gate alias must retain sequential semantics. */
                ok(ATTN(ref), "alias reference attention");
                ok(ds4_gpu_qwen4exp_qsa_output_gate_tensor(ref,ref,width*HEADS*DIM), "alias reference gate");
                ok(GATED(cand,cand), "alias fallback");
                get(ref,a,bytes); get(cand,b,bytes);
                ok(memcmp(a,b,bytes) == 0, "alias fallback parity");
#undef ATTN
#undef GATED
            }
        }
    }
    ds4_gpu_decode_graphs_invalidate();
    ds4_gpu_tensor *all[] = {q,k,v,ref,cand,gate,sel,cnt,pos,scratch};
    for (unsigned i = 0; i < sizeof(all)/sizeof(all[0]); i++) ds4_gpu_tensor_free(all[i]);
    free(host); free(gates); free(a); free(b); ds4_gpu_cleanup();
    printf("QSA gated exact output, guards, alias fallback; %u graph replays PASS\n", replays);
    return 0;
}
