/* Exact decode scheduling regression test. Requires the CUDA library, no GGUF.
 * Build from the challenge root:
 * cc -O2 -std=c11 -D_GNU_SOURCE -Ids4 ds4/tests/test_q8_decode_pairs.c \
 *    -L.build/ds4 -lds4qwen -lm -o /tmp/test-q8-decode-pairs
 * LD_LIBRARY_PATH=.build/ds4 /tmp/test-q8-decode-pairs
 */
#include "ds4_gpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

static uint32_t rng = 0x9271a503u;
static uint32_t next_word(void) {
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
    return rng;
}
static void require(int ok, const char *what) {
    if (!ok) { fprintf(stderr, "Q8 decode pairs: %s\n", what); exit(1); }
}

static void check_shape(uint64_t in, uint64_t out, uint64_t offset) {
    const uint64_t groups = (in + 31u) / 32u;
    const uint64_t bytes = offset + out * groups * 34u;
    uint8_t *model = mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    require(model != MAP_FAILED, "model allocation");
    for (uint64_t b = 0; b < out * groups; b++) {
        uint8_t *p = model + offset + b * 34u;
        p[0] = 0; p[1] = (b & 1u) ? 0x98 : 0x18;
        for (int j = 2; j < 34; j++) p[j] = (uint8_t)next_word();
    }
    const size_t xn = (size_t)7u * in;
    const size_t yn = (size_t)7u * out;
    const size_t ybytes = yn * sizeof(float);
    float *x = malloc(xn * sizeof(float));
    float *poison = malloc(ybytes), *reference = malloc(ybytes), *got = malloc(ybytes);
    require(x && poison && reference && got, "host allocation");
    for (size_t i = 0; i < xn; i++)
        x[i] = ((int32_t)(next_word() % 2001u) - 1000) / 1000.0f;
    for (size_t i = 0; i < yn; i++)
        poison[i] = ((int)(i % 17u) - 8) * 0.125f;

    require(ds4_gpu_init(), "GPU init");
    require(ds4_gpu_set_model_map(model, bytes), "model registration");
    ds4_gpu_tensor *xt = ds4_gpu_tensor_alloc(xn * sizeof(float));
    ds4_gpu_tensor *yt = ds4_gpu_tensor_alloc(ybytes);
    require(xt && yt, "device allocation");
    require(ds4_gpu_tensor_write(xt, 0, x, xn * sizeof(float)), "input write");
    const uint32_t widths[] = {1, 2, 3, 4, 7};
    for (size_t c = 0; c < sizeof(widths) / sizeof(widths[0]); c++) {
        const uint32_t n = widths[c];
        /* The existing override selects the original per-row kernel, even
         * when the normal path uses a two-lane pair or a wider row tile. */
        for (int pass = 0; pass < 2; pass++) {
            require((pass == 0
                ? setenv("DS4_QWEN4EXP_NO_ROW_TILE", "1", 1)
                : unsetenv("DS4_QWEN4EXP_NO_ROW_TILE")) == 0, "dispatch switch");
            require(ds4_gpu_tensor_write(yt, 0, poison, ybytes), "output reset");
            require(ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(
                yt, model, bytes, offset, in, out, xt, n), "projection");
            require(ds4_gpu_tensor_read(yt, 0, pass == 0 ? reference : got, ybytes),
                    "output read");
        }
        /* Include unused token rows: they must retain the same canaries. */
        for (size_t i = 0; i < yn; i++) {
            if (!isfinite(reference[i]) || !isfinite(got[i]) ||
                memcmp(reference + i, got + i, sizeof(float)) != 0) {
                fprintf(stderr, "Q8 %llu->%llu offset %llu rows %u at %zu: "
                                "reference %.9g, default %.9g\n",
                        (unsigned long long)in, (unsigned long long)out,
                        (unsigned long long)offset, n, i, reference[i], got[i]);
                exit(1);
            }
        }
    }
    /* Captured paired reads must see changed inputs at both decode widths.
     * Compare each replay with a fresh eager one-row-order reference. */
    const float magnitudes[] = {1.0f, 1e-30f, 1e-37f, 1e10f};
    ds4_gpu_decode_graphs_invalidate();
    for (uint32_t n = 1u; n <= 2u; n++) {
        ds4_decode_graph_key key = {.il = 1u, .island = n - 1u, .variant = n};
        require(ds4_gpu_decode_graph_begin(&key) == -1, "graph warmup");
        require(ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(
            yt, model, bytes, offset, in, out, xt, n), "warm projection");
        require(ds4_gpu_decode_graph_begin(&key) == 0, "graph capture");
        require(ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(
            yt, model, bytes, offset, in, out, xt, n), "captured projection");
        require(ds4_gpu_decode_graph_end(&key) == 0, "graph capture end");
        for (unsigned m = 0; m < 4u; m++) {
            for (size_t i = 0; i < xn; i++)
                x[i] = ((int32_t)(next_word() % 2001u) - 1000) *
                       0.001f * magnitudes[m];
            require(ds4_gpu_tensor_write(xt, 0, x, xn * sizeof(float)), "changed input");
            require(setenv("DS4_QWEN4EXP_NO_ROW_TILE", "1", 1) == 0, "reference dispatch");
            require(ds4_gpu_tensor_write(yt, 0, poison, ybytes), "eager canaries");
            require(ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(
                yt, model, bytes, offset, in, out, xt, n), "eager reference");
            require(ds4_gpu_tensor_read(yt, 0, reference, ybytes), "eager read");
            require(unsetenv("DS4_QWEN4EXP_NO_ROW_TILE") == 0, "replay dispatch");
            require(ds4_gpu_tensor_write(yt, 0, poison, ybytes), "replay canaries");
            require(ds4_gpu_decode_graph_begin(&key) == 1, "graph replay");
            require(ds4_gpu_tensor_read(yt, 0, got, ybytes), "replay read");
            require(memcmp(reference, got, ybytes) == 0, "changed-input graph mismatch");
        }
    }
    ds4_gpu_decode_graphs_invalidate();
    printf("Q8 %llu->%llu offset %llu: 5 widths and 8 changed-input graph replays exact\n",
           (unsigned long long)in, (unsigned long long)out,
           (unsigned long long)offset);
    ds4_gpu_tensor_free(xt); ds4_gpu_tensor_free(yt);
    ds4_gpu_cleanup();
    free(x); free(poison); free(reference); free(got);
    munmap(model, bytes);
}

int main(void) {
    require(setenv("DS4_CUDA_DECODE_GRAPHS", "1", 1) == 0, "enable test graphs");
    const char *old = getenv("DS4_QWEN4EXP_NO_ROW_TILE");
    char *saved = old ? strdup(old) : NULL;
    require(!old || saved, "environment copy");
    const uint64_t shapes[][3] = {
        {33, 37, 64}, {63, 19, 66}, {96, 515, 64}, {1056, 519, 66},
        {320, 10240, 64}, {10240, 320, 64}, {2560, 6144, 64},
        {2560, 10240, 66}, {6144, 2560, 64}, {2560, 248320, 64},
    };
    for (size_t i = 0; i < sizeof(shapes) / sizeof(shapes[0]); i++)
        check_shape(shapes[i][0], shapes[i][1], shapes[i][2]);
    if (saved) { setenv("DS4_QWEN4EXP_NO_ROW_TILE", saved, 1); free(saved); }
    else unsetenv("DS4_QWEN4EXP_NO_ROW_TILE");
    puts("Q8 decode pairs: all 50 eager cases and 80 graph replays passed");
    return 0;
}
