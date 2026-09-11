#include "ds4_gpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

static uint32_t state = 0x79be5431u;
static uint32_t random_word(void) {
    state ^= state << 13; state ^= state >> 17; state ^= state << 5;
    return state;
}
static void check(int ok, const char *why) {
    if (!ok) { fprintf(stderr, "F32 vector decode: %s\n", why); exit(1); }
}
static void pin(unsigned candidate) {
    check((candidate ? unsetenv("DS4_F32_NO_VECTOR_DECODE") :
                       setenv("DS4_F32_NO_VECTOR_DECODE", "1", 1)) == 0,
          "reference pin");
}
static void shape(uint64_t in, uint64_t out, uint64_t offset) {
    const uint32_t max_rows = 64;
    const uint64_t size = offset + in * out * sizeof(float);
    void *map = mmap(NULL, size, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    check(map != MAP_FAILED, "mapping");
    float *w = (float *)((char *)map + offset);
    for (uint64_t i = 0; i < in * out; i++)
        w[i] = ((int32_t)(random_word() % 20001u) - 10000) * 0.0000127f;
    const size_t xn = (size_t)max_rows * in;
    const size_t yn = (size_t)(max_rows + 1) * out;
    float *x = malloc(xn * sizeof(float));
    float *poison = malloc(yn * sizeof(float));
    float *ref = malloc(yn * sizeof(float));
    float *got = malloc(yn * sizeof(float));
    check(x && poison && ref && got, "host buffers");
    for (size_t i = 0; i < xn; i++)
        x[i] = ((int32_t)(random_word() % 20001u) - 10000) * 0.0001793f;
    for (size_t i = 0; i < yn; i++) poison[i] = (float)(i % 19u) + 17.0f;
    check(ds4_gpu_init(), "init");
    check(ds4_gpu_set_model_map(map, size), "register weights");
    ds4_gpu_tensor *xt = ds4_gpu_tensor_alloc(xn * sizeof(float));
    ds4_gpu_tensor *yt = ds4_gpu_tensor_alloc(yn * sizeof(float));
    check(xt && yt, "device buffers");
    check(ds4_gpu_tensor_write(xt, 0, x, xn * sizeof(float)), "input write");
    const uint32_t widths[] = {1, 2, 3, 4, 7, 8, 64, 2, 1};
    for (unsigned wi = 0; wi < sizeof(widths) / sizeof(widths[0]); wi++) {
        uint32_t rows = widths[wi];
        const float scales[] = {0.0001f, 0.1f, 1.0f, 1000.0f};
        for (size_t i = 0; i < xn; i++)
            x[i] = ((int32_t)(random_word() % 20001u) - 10000) *
                   0.0001793f * scales[(wi + i) % 4u];
        check(ds4_gpu_tensor_write(xt, 0, x, xn * sizeof(float)), "changed input write");
        pin(0);
        check(ds4_gpu_tensor_write(yt, 0, poison, yn * sizeof(float)), "oracle poison");
        check(ds4_gpu_matmul_f32_decode_rows_exact_tensor(
            yt, map, size, offset, in, out, xt, rows), "oracle");
        check(ds4_gpu_tensor_read(yt, 0, ref, yn * sizeof(float)), "oracle read");
        {
            const unsigned mode = 1;
            pin(mode);
            check(ds4_gpu_tensor_write(yt, 0, poison, yn * sizeof(float)), "candidate poison");
            check(ds4_gpu_matmul_f32_decode_rows_exact_tensor(
                yt, map, size, offset, in, out, xt, rows), "candidate");
            check(ds4_gpu_tensor_read(yt, 0, got, yn * sizeof(float)), "candidate read");
            if (memcmp(ref, got, yn * sizeof(float))) {
                for (size_t i = 0; i < yn; i++) if (memcmp(ref+i, got+i, sizeof(float))) {
                    fprintf(stderr, "mismatch in=%llu out=%llu offset=%llu rows=%u mode=%u at=%zu %.9g vs %.9g\n",
                        (unsigned long long)in, (unsigned long long)out,
                        (unsigned long long)offset, rows, mode, i, ref[i], got[i]);
                    exit(1);
                }
            }
        }
    }
    ds4_gpu_tensor_free(xt); ds4_gpu_tensor_free(yt); ds4_gpu_cleanup();
    munmap(map, size); free(x); free(poison); free(ref); free(got);
    printf("F32_VECTOR_DECODE exact in=%llu out=%llu offset=%llu: 9 full-output/canary comparisons pass\n",
           (unsigned long long)in, (unsigned long long)out, (unsigned long long)offset);
}
int main(void) {
    shape(2560, 512, 64); shape(2560, 512, 68);
    shape(2560, 48, 64); shape(2560, 48, 68);
    shape(2560, 49, 68); shape(257, 48, 64);
    unsetenv("DS4_F32_NO_VECTOR_DECODE");
    puts("F32 vector decode: all 54 full-output/canary comparisons pass");
    return 0;
}
