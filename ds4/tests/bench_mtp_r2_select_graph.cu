/* Isolated target-R2 selector submission benchmark. Timing includes the
 * unchanged quantize/coarse screen and either the eager or graphed exact tail.
 * It is an operator diagnostic, not a model-performance claim.
 *
 * nvcc -O2 -std=c++17 -Ids4 ds4/tests/bench_mtp_r2_select_graph.cu \
 *   -L.build/ds4 -lds4qwen -o /tmp/bench-mtp-r2-select-graph
 */
#include "ds4_gpu.h"
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

static constexpr uint32_t DIM = 2560u;
static constexpr uint32_t VOCAB = 248320u;
static constexpr uint32_t TAIL = 276u;
static constexpr uint32_t PREFIX = VOCAB - TAIL;
static constexpr uint32_t CAP = 16384u;
static constexpr uint64_t ROW = 80u * 34u;

static void need(int ok, const char *what) {
    if (!ok) { fprintf(stderr, "R2 graph bench: %s\n", what); exit(1); }
}

static float timed(ds4_gpu_tensor *out, ds4_gpu_tensor *ids,
        ds4_gpu_tensor *scratch, const void *w, uint64_t bytes,
        ds4_gpu_tensor *x, int iterations) {
    cudaEvent_t a = nullptr, b = nullptr;
    need(cudaEventCreate(&a) == cudaSuccess && cudaEventCreate(&b) == cudaSuccess,
         "event create");
    need(cudaEventRecord(a, 0) == cudaSuccess, "event start");
    for (int i = 0; i < iterations; i++)
        need(ds4_gpu_mtp_native_screen2(out, ids, scratch, w, bytes, 0,
             DIM, VOCAB, PREFIX, TAIL, x, 1) == (int)CAP, "screen2");
    need(cudaEventRecord(b, 0) == cudaSuccess &&
         cudaEventSynchronize(b) == cudaSuccess, "event stop");
    float ms = 0.0f;
    need(cudaEventElapsedTime(&ms, a, b) == cudaSuccess, "event elapsed");
    cudaEventDestroy(a); cudaEventDestroy(b);
    return ms / (float)iterations;
}

int main(void) {
    const uint64_t weight_bytes = (uint64_t)VOCAB * ROW;
    unsigned char *w = (unsigned char *)mmap(nullptr, weight_bytes,
        PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    need(w != MAP_FAILED, "weight mapping");
    /* Touch every source page once before GPU timing. Zero Q8 payloads retain
     * the full address/load pattern and make the exact comparison simple. */
    memset(w, 0, weight_bytes);
    need(ds4_gpu_init(), "GPU init");
    need(ds4_gpu_set_model_map(w, weight_bytes), "register weights");
    uint64_t scratch_bytes = 0; uint32_t capacity = 0;
    need(ds4_gpu_mtp_native_screen2_init(VOCAB, &scratch_bytes, &capacity) == 1 &&
         capacity == CAP, "screen2 init");
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(2ull * DIM * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(2ull * CAP * sizeof(float));
    ds4_gpu_tensor *ids = ds4_gpu_tensor_alloc(2ull * CAP * sizeof(uint32_t));
    ds4_gpu_tensor *scratch = ds4_gpu_tensor_alloc(scratch_bytes);
    need(x && out && ids && scratch, "device allocations");
    float host_x[2u * DIM];
    for (uint32_t i = 0; i < 2u * DIM; i++)
        host_x[i] = (float)((int)(i % 257u) - 128) / 129.0f;
    need(ds4_gpu_tensor_write(x, 0, host_x, sizeof(host_x)), "activation write");

    /* Settle CUB, capture on second sight, then prove replay and eager valve
     * produce identical complete shortlist/value buffers. */
    unsetenv("DS4_MTP_NO_R2_SELECT_GRAPH");
    for (int i = 0; i < 3; i++)
        need(ds4_gpu_mtp_native_screen2(out, ids, scratch, w, weight_bytes, 0,
             DIM, VOCAB, PREFIX, TAIL, x, 1) == (int)CAP, "graph warmup");
    float *graph_out = (float *)malloc(2ull * CAP * sizeof(float));
    uint32_t *graph_ids = (uint32_t *)malloc(2ull * CAP * sizeof(uint32_t));
    float *eager_out = (float *)malloc(2ull * CAP * sizeof(float));
    uint32_t *eager_ids = (uint32_t *)malloc(2ull * CAP * sizeof(uint32_t));
    need(graph_out && graph_ids && eager_out && eager_ids, "host outputs");
    need(ds4_gpu_tensor_read(out, 0, graph_out, 2ull * CAP * sizeof(float)) &&
         ds4_gpu_tensor_read(ids, 0, graph_ids, 2ull * CAP * sizeof(uint32_t)),
         "graph outputs");
    setenv("DS4_MTP_NO_R2_SELECT_GRAPH", "1", 1);
    need(ds4_gpu_mtp_native_screen2(out, ids, scratch, w, weight_bytes, 0,
         DIM, VOCAB, PREFIX, TAIL, x, 1) == (int)CAP, "eager proof");
    need(ds4_gpu_tensor_read(out, 0, eager_out, 2ull * CAP * sizeof(float)) &&
         ds4_gpu_tensor_read(ids, 0, eager_ids, 2ull * CAP * sizeof(uint32_t)),
         "eager outputs");
    need(!memcmp(graph_out, eager_out, 2ull * CAP * sizeof(float)) &&
         !memcmp(graph_ids, eager_ids, 2ull * CAP * sizeof(uint32_t)),
         "graph/eager output parity");

    puts("rep,order,eager_ms,graph_ms,delta_pct");
    const int iterations = 12;
    for (int rep = 0; rep < 9; rep++) {
        float eager_ms, graph_ms;
        if ((rep & 1) == 0) {
            setenv("DS4_MTP_NO_R2_SELECT_GRAPH", "1", 1);
            eager_ms = timed(out, ids, scratch, w, weight_bytes, x, iterations);
            unsetenv("DS4_MTP_NO_R2_SELECT_GRAPH");
            graph_ms = timed(out, ids, scratch, w, weight_bytes, x, iterations);
        } else {
            unsetenv("DS4_MTP_NO_R2_SELECT_GRAPH");
            graph_ms = timed(out, ids, scratch, w, weight_bytes, x, iterations);
            setenv("DS4_MTP_NO_R2_SELECT_GRAPH", "1", 1);
            eager_ms = timed(out, ids, scratch, w, weight_bytes, x, iterations);
        }
        printf("%d,%s,%.6f,%.6f,%.4f\n", rep,
            (rep & 1) ? "graph-first" : "eager-first", eager_ms, graph_ms,
            100.0f * (eager_ms - graph_ms) / eager_ms);
    }
    free(graph_out); free(graph_ids); free(eager_out); free(eager_ids);
    ds4_gpu_tensor_free(x); ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(ids);
    ds4_gpu_tensor_free(scratch); ds4_gpu_cleanup(); munmap(w, weight_bytes);
    return 0;
}
