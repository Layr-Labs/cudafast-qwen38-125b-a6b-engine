/* Exact paired-lane decode regression test. Requires the CUDA library. */
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
    float *poison = malloc(ybytes);
    float *reference = malloc(ybytes);
    float *got = malloc(ybytes);
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
        for (int pass = 0; pass < 2; pass++) {
            require((pass == 0
                ? setenv("DS4_QWEN4EXP_NO_ROW_TILE", "1", 1)
                : unsetenv("DS4_QWEN4EXP_NO_ROW_TILE")) == 0,
                "dispatch switch");
            require(ds4_gpu_tensor_write(yt, 0, poison, ybytes), "output reset");
            require(ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(
                yt, model, bytes, offset, in, out, xt, n), "projection");
            require(ds4_gpu_tensor_read(yt, 0,
                pass == 0 ? reference : got, ybytes), "output read");
        }
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
    ds4_gpu_tensor_free(xt); ds4_gpu_tensor_free(yt);
    ds4_gpu_cleanup();
    free(x); free(poison); free(reference); free(got);
    munmap(model, bytes);
}

int main(void) {
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
    puts("Q8 decode pairs: all 50 cases passed");
    return 0;
}
