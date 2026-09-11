/* Shared activation quantization regression. Link against libds4qwen.
 * The two production-size GDN projections must equal independent ordinary
 * calls. Change both data and row count while reusing one quant scratch. */
#include "ds4_gpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

static uint32_t rng = 0x82a10537u;
static uint32_t word(void) {
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
    return rng;
}
static void require(int ok, const char *what) {
    if (!ok) { fprintf(stderr, "Q8 shared input: %s\n", what); exit(1); }
}

static void check_shape(uint64_t in, uint64_t out0, uint64_t out1,
                        uint64_t offset) {
    enum { MAX_ROWS = 1024 };
    const uint64_t groups = in / 32u;
    const uint64_t offsets[] = {offset, offset + out0 * groups * 34u + 64u};
    const uint64_t outputs[] = {out0, out1};
    const uint64_t bytes = offsets[1] + out1 * groups * 34u;
    uint8_t *model = mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    require(model != MAP_FAILED, "model allocation");
    for (int branch = 0; branch < 2; branch++) {
        for (uint64_t g = 0; g < outputs[branch] * groups; g++) {
            uint8_t *p = model + offsets[branch] + g * 34u;
            p[0] = 0; p[1] = (g & 1u) ? 0x98 : 0x18;
            for (int j = 2; j < 34; j++) p[j] = (uint8_t)word();
        }
    }
    const size_t xn = (size_t)MAX_ROWS * in;
    const size_t max_out = out0 > out1 ? out0 : out1;
    const size_t yn = (MAX_ROWS + 1u) * max_out;
    const size_t qbytes = (size_t)MAX_ROWS * groups * 36u;
    float *x = malloc(xn * sizeof(float));
    float *xread = malloc(xn * sizeof(float));
    float *poison = malloc(yn * sizeof(float));
    float *reference = malloc(yn * sizeof(float));
    float *got = malloc(yn * sizeof(float));
    uint8_t *qbefore = malloc(qbytes), *qafter = malloc(qbytes);
    require(x && xread && poison && reference && got && qbefore && qafter,
            "host allocation");
    for (size_t i = 0; i < yn; i++) poison[i] = ((int)(i % 19u) - 9) * 0.125f;
    require(ds4_gpu_init(), "GPU initialization");
    require(ds4_gpu_set_model_map(model, bytes), "model registration");
    ds4_gpu_tensor *xt = ds4_gpu_tensor_alloc(xn * sizeof(float));
    ds4_gpu_tensor *qt = ds4_gpu_tensor_alloc(qbytes);
    ds4_gpu_tensor *yt = ds4_gpu_tensor_alloc(yn * sizeof(float));
    require(xt && qt && yt, "GPU allocation");
    const uint32_t widths[] = {1, 2, 7, 64, 1024, 2, 1};
    for (size_t c = 0; c < sizeof(widths) / sizeof(widths[0]); c++) {
        const uint32_t rows = widths[c];
        const size_t live_xbytes = (size_t)rows * in * sizeof(float);
        for (size_t i = 0; i < (size_t)rows * in; i++) {
            const float scale = ((i / 32u) % 11u == 0u) ? 0.0f : 0.001f;
            x[i] = ((int32_t)(word() % 2001u) - 1000) * scale;
        }
        require(ds4_gpu_tensor_write(xt, 0, x, live_xbytes), "input write");
        const uint64_t soff = (uint64_t)rows * groups * 32u;
        const size_t live_qbytes = soff + (size_t)rows * groups * sizeof(float);
        require(ds4_gpu_quantize_q8_0_decode_rows_exact_tensor(
            qt, 0, soff, xt, in, rows), "shared quantization");
        require(ds4_gpu_tensor_read(qt, 0, qbefore, live_qbytes), "quant read");
        for (int branch = 0; branch < 2; branch++) {
            const uint64_t out = outputs[branch];
            /* One unused output row also detects over-wide tail stores. */
            const size_t live_ybytes = ((size_t)rows + 1u) * out * sizeof(float);
            require(ds4_gpu_tensor_write(yt, 0, poison, live_ybytes), "reset oracle");
            require(ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(
                yt, model, bytes, offsets[branch], in, out, xt, rows),
                "independent projection");
            require(ds4_gpu_tensor_read(yt, 0, reference, live_ybytes), "oracle read");
            require(ds4_gpu_tensor_write(yt, 0, poison, live_ybytes), "reset shared");
            require(ds4_gpu_matmul_q8_0_preq_rows_exact_tensor(
                yt, model, bytes, offsets[branch], in, out, qt, 0, soff, rows),
                "shared-input projection");
            require(ds4_gpu_tensor_read(yt, 0, got, live_ybytes), "result read");
            for (size_t i = 0; i < live_ybytes / sizeof(float); i++) {
                if (!isfinite(reference[i]) || !isfinite(got[i]) ||
                    memcmp(reference + i, got + i, sizeof(float))) {
                    fprintf(stderr, "shared Q8 %llu->%llu rows %u branch %d "
                                    "at %zu: %.9g != %.9g\n",
                            (unsigned long long)in, (unsigned long long)out,
                            rows, branch, i, reference[i], got[i]);
                    exit(1);
                }
            }
        }
        require(ds4_gpu_tensor_read(qt, 0, qafter, live_qbytes), "quant reread");
        require(!memcmp(qbefore, qafter, live_qbytes), "quant bytes changed");
        require(ds4_gpu_tensor_read(xt, 0, xread, live_xbytes), "input reread");
        require(!memcmp(x, xread, live_xbytes), "float input changed");
    }
    printf("shared Q8 %llu->(%llu,%llu), offset %llu: 14 projections exact\n",
           (unsigned long long)in, (unsigned long long)out0,
           (unsigned long long)out1, (unsigned long long)offset);
    ds4_gpu_tensor_free(xt); ds4_gpu_tensor_free(qt); ds4_gpu_tensor_free(yt);
    ds4_gpu_cleanup();
    free(x); free(xread); free(poison); free(reference); free(got);
    free(qbefore); free(qafter);
    munmap(model, bytes);
}

int main(void) {
    check_shape(96, 515, 129, 64);
    check_shape(2560, 10240, 6144, 64);
    check_shape(2560, 10240, 6144, 66);
    puts("Q8 shared input: all 42 projections passed");
    return 0;
}
