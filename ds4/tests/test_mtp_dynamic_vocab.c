/* CUDA functional tests with generated inputs and original-format Q8 rows.
 * No model file. Link against the built ds4 CUDA library, as for
 * tests/test_q8_decode_pairs.c. This test has not been GPU-executed locally. */
#include "ds4_gpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
static void need(int ok, const char *why) {
    if (!ok) { fprintf(stderr, "dynamic MTP: %s\n", why); exit(1); }
}
static void selection(void) {
    enum { V = 9011, CAP = 8192 };
    float *v = malloc(2u * V * 4u);
    uint32_t got[CAP], expected[CAP];
    ds4_gpu_tensor *l = ds4_gpu_tensor_alloc(2u * V * 4u);
    ds4_gpu_tensor *ids = ds4_gpu_tensor_alloc(CAP * 4u);
    ds4_gpu_tensor *scratch = ds4_gpu_tensor_alloc(((V + 255u)/256u)*4u);
    need(v && l && ids && scratch, "selection allocation");
    for (unsigned pass = 0; pass < 7; pass++) {
        for (unsigned j = 0; j < 2u*V; j++) v[j] = -100.0f;
        const unsigned row = pass & 1u;
        float *p = v + row * V;
        p[500] = 3.0f;
        for (unsigned j = 11; j < V; j += 113) p[j] = -2.0f;
        p[501] = 3.0f - 9.210340371976184f; /* inclusive threshold */
        p[502] = nextafterf(p[501], -INFINITY);
        if (pass == 2) for (unsigned j = 0; j < V; j++) p[j] = 3.0f; /* overflow */
        if (pass == 3) p[1200] = NAN;
        if (pass == 4) p[500] = INFINITY;
        if (pass == 5) p[1200] = 4.0f; /* supplied max is stale/invalid */
        if (pass == 6) p[1200] = -INFINITY; /* allowed excluded value */
        need(ds4_gpu_tensor_write(l, 0, v, 2u*V*4u), "target write");
        uint32_t n = 99;
        need(ds4_gpu_mtp_select_vocab(ids, scratch, l, row, V, 500, 17, CAP, &n), "selection API");
        if (pass >= 2 && pass <= 5) { need(n == 0, "invalid/overflow fallback"); continue; }
        uint32_t want = 0;
        for (unsigned j = 0; j < V; j++)
            if (!j || j >= V-17 || p[j] >= p[500] - 9.210340371976184f)
                expected[want++] = j;
        need(n == want, "candidate count");
        need(ds4_gpu_tensor_read(ids, 0, got, n*4u), "candidate read");
        need(!memcmp(got, expected, n*4u), "sorted unique exact threshold/tail IDs");
    }
    uint32_t n = 7;
    need(ds4_gpu_mtp_select_vocab(ids, scratch, l, 2, V, 0, 0, CAP, &n) && !n,
         "missing row fallback");
    ds4_gpu_tensor_free(l); ds4_gpu_tensor_free(ids); ds4_gpu_tensor_free(scratch); free(v);
}
static void indexed(void) {
    enum { IN = 2560, V = 1031, OFF = 66 };
    const size_t groups = IN/32, bytes = OFF + V*groups*34u;
    unsigned char *w = mmap(NULL, bytes, PROT_READ|PROT_WRITE,
                            MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    need(w != MAP_FAILED, "mapped Q8 allocation");
    for (size_t b = 0; b < V*groups; b++) {
        unsigned char *p = w + OFF + b*34u;
        p[0] = 0; p[1] = (b&1u) ? 0x98 : 0x18;
        for (unsigned j = 2; j < 34; j++) p[j] = (unsigned char)(b*37u+j*19u);
    }
    need(ds4_gpu_set_model_map(w, bytes), "weight map registration");
    float x[IN], ref[V], got[V]; uint32_t ids[V];
    ds4_gpu_tensor *xt = ds4_gpu_tensor_alloc(IN*4u), *yt = ds4_gpu_tensor_alloc(V*4u);
    ds4_gpu_tensor *it = ds4_gpu_tensor_alloc(V*4u);
    need(xt && yt && it, "projection allocation");
    for (unsigned i = 0; i < IN; i++) x[i] = ((int)(i%37)-18)*0.125f;
    need(ds4_gpu_tensor_write(xt, 0, x, IN*4u), "activation write");
    need(ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(yt, w, bytes, OFF, IN, V, xt, 1), "full projection");
    need(ds4_gpu_tensor_read(yt, 0, ref, V*4u), "full read");
    const unsigned counts[] = {1,2,3,4,5,31,1025};
    for (unsigned c = 0; c < sizeof(counts)/sizeof(counts[0]); c++) {
        unsigned n = counts[c];
        for (unsigned i = 0; i < n; i++) ids[i] = (i*97u+3u)%V;
        need(ds4_gpu_tensor_write(it, 0, ids, n*4u), "ID write");
        need(ds4_gpu_mtp_indexed_q8(yt, w, bytes, OFF, IN, V, xt, it, n), "indexed projection");
        need(ds4_gpu_tensor_read(yt, 0, got, n*4u), "indexed read");
        for (unsigned i = 0; i < n; i++)
            need(!memcmp(got+i, ref+ids[i], 4u), "original row arithmetic parity");
    }
    ds4_gpu_decode_graphs_invalidate();
    ds4_decode_graph_key key = {.il = 1u, .island = 0u, .variant = 73u};
    need(ds4_gpu_decode_graph_begin(&key) == -1, "indexed graph warmup");
    need(ds4_gpu_mtp_indexed_q8(yt,w,bytes,OFF,IN,V,xt,it,7), "warm indexed projection");
    need(ds4_gpu_decode_graph_begin(&key) == 0, "indexed capture begin");
    need(ds4_gpu_mtp_indexed_q8(yt,w,bytes,OFF,IN,V,xt,it,7), "capture indexed projection");
    need(ds4_gpu_decode_graph_end(&key) == 0, "indexed capture end");
    for (unsigned pass = 0; pass < 4; pass++) {
        for (unsigned i = 0; i < IN; i++) x[i] = ((int)((i+pass)%41)-20)*0.0625f;
        for (unsigned i = 0; i < 7; i++) ids[i] = (i*73u+pass*11u)%V;
        need(ds4_gpu_tensor_write(xt,0,x,IN*4u), "replay activation update");
        need(ds4_gpu_tensor_write(it,0,ids,7*4u), "replay ID update");
        need(ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(yt,w,bytes,OFF,IN,V,xt,1), "replay reference");
        need(ds4_gpu_tensor_read(yt,0,ref,V*4u), "replay reference read");
        need(ds4_gpu_decode_graph_begin(&key) == 1, "indexed graph replay");
        need(ds4_gpu_tensor_read(yt,0,got,7*4u), "replay output");
        for (unsigned i = 0; i < 7; i++)
            need(!memcmp(got+i,ref+ids[i],4), "capture must read current IDs and activations");
    }
    ds4_gpu_decode_graphs_invalidate();
    ids[0] = V;
    need(ds4_gpu_tensor_write(it, 0, ids, 4), "invalid ID write");
    need(ds4_gpu_mtp_indexed_q8(yt, w, bytes, OFF, IN, V, xt, it, 1), "bounded invalid ID");
    need(ds4_gpu_tensor_read(yt, 0, got, 4) && isinf(got[0]) && got[0]<0, "invalid ID sentinel");
    need(!ds4_gpu_mtp_indexed_q8(yt,w,bytes,OFF,UINT64_MAX,V,xt,it,1), "dimension overflow refusal");
    ds4_gpu_tensor_free(xt); ds4_gpu_tensor_free(yt); ds4_gpu_tensor_free(it);
    ds4_gpu_cleanup(); munmap(w, bytes);
}
int main(void) {
    need(setenv("DS4_CUDA_DECODE_GRAPHS","1",1)==0, "test graphs enabled");
    need(ds4_gpu_init(), "CUDA required");
    selection(); indexed();
    puts("dynamic MTP selection and original Q8 row parity passed");
    return 0;
}
