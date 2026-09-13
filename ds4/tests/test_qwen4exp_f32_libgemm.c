/* Host dispatch test for the prefill-only F32 router library GEMM valve.
 *
 * Stubs the two GPU entries. Proves:
 *   - ds4_qwen4exp_matmul_f32 (GDN alpha/beta) is always exact
 *   - libgemm takes cuBLAS only at n_rows >= 8, CUDA, valve unset
 *   - DS4_QWEN4EXP_NO_F32_LIBGEMM presence restores exact at every width
 *   - decode widths 1 and 2 stay exact
 *   - DS4_QWEN4EXP_NO_FP16_DOWN_PARTIAL is independently wired; both
 *     production defaults stay enabled (getenv NULL)
 *
 * No GPU. make test-qwen4exp-f32-libgemm
 */
#include "ds4_qwen4exp_matmul.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VALVE "DS4_QWEN4EXP_NO_F32_LIBGEMM"
#define VALVE_FP16 "DS4_QWEN4EXP_NO_FP16_DOWN_PARTIAL"

enum { FN_NONE = 0, FN_EXACT = 1, FN_TENSOR = 2 };

static int g_fn;
static uint32_t g_rows;
static uint64_t g_in_dim;
static uint64_t g_out_dim;
static int g_calls;

static void reset_stubs(void) {
    g_fn = FN_NONE;
    g_rows = 0;
    g_in_dim = 0;
    g_out_dim = 0;
    g_calls = 0;
}

int ds4_gpu_matmul_f32_decode_rows_exact_tensor(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
        const ds4_gpu_tensor *x, uint32_t n_rows) {
    (void)out;
    (void)model_map;
    (void)model_size;
    (void)weight_offset;
    (void)x;
    g_fn = FN_EXACT;
    g_rows = n_rows;
    g_in_dim = in_dim;
    g_out_dim = out_dim;
    g_calls++;
    return 1;
}

int ds4_gpu_matmul_f32_tensor(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
        const ds4_gpu_tensor *x, uint64_t n_tok) {
    (void)out;
    (void)model_map;
    (void)model_size;
    (void)weight_offset;
    (void)x;
    g_fn = FN_TENSOR;
    g_rows = (uint32_t)n_tok;
    g_in_dim = in_dim;
    g_out_dim = out_dim;
    g_calls++;
    return 1;
}

static int libgemm_live(uint32_t rows) {
#if !defined(DS4_NO_GPU) && !defined(__APPLE__) && !defined(DS4_ROCM_BUILD)
    return rows >= DS4_QWEN4EXP_F32_LIBGEMM_MIN_ROWS &&
           getenv(VALVE) == NULL;
#else
    (void)rows;
    return 0;
#endif
}

static int fail(const char *msg) {
    fprintf(stderr, "FAIL: %s (fn=%d rows=%u in=%llu out=%llu calls=%d)\n",
            msg, g_fn, g_rows, (unsigned long long)g_in_dim,
            (unsigned long long)g_out_dim, g_calls);
    return 1;
}

static int expect(int fn, uint32_t rows, uint64_t in_dim, uint64_t out_dim,
                  const char *msg) {
    if (g_calls != 1) return fail(msg);
    if (g_fn != fn) return fail(msg);
    if (g_rows != rows) return fail(msg);
    if (g_in_dim != in_dim || g_out_dim != out_dim) return fail(msg);
    return 0;
}

static int call_lib(uint32_t rows) {
    reset_stubs();
    if (!ds4_qwen4exp_matmul_f32_libgemm(NULL, (const void *)1, 1, 0,
                                         2560, 512, NULL, rows)) {
        return fail("libgemm returned 0");
    }
    return 0;
}

static int call_exact(uint32_t rows) {
    reset_stubs();
    if (!ds4_qwen4exp_matmul_f32(NULL, (const void *)1, 1, 0,
                                 2560, 48, NULL, rows)) {
        return fail("exact helper returned 0");
    }
    return 0;
}

static int file_has(const char *path, const char *needle) {
    char line[512];
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    while (fgets(line, sizeof line, f)) {
        if (strstr(line, needle)) {
            fclose(f);
            return 1;
        }
    }
    fclose(f);
    return 0;
}

static int check_independent_defaults(void) {
    int want;

    if (unsetenv(VALVE) != 0 || unsetenv(VALVE_FP16) != 0)
        return fail("unsetenv both valves");
    want = libgemm_live(8u) ? FN_TENSOR : FN_EXACT;
    if (call_lib(8u)) return 1;
    if (expect(want, 8u, 2560, 512, "both unset: production libgemm default"))
        return 1;

    if (setenv(VALVE_FP16, "1", 1) != 0) return fail("setenv fp16 valve");
    if (call_lib(8u)) return 1;
    if (expect(want, 8u, 2560, 512, "fp16 valve does not disable libgemm"))
        return 1;

    if (setenv(VALVE, "1", 1) != 0) return fail("setenv f32 with fp16");
    if (call_lib(8u)) return 1;
    if (expect(FN_EXACT, 8u, 2560, 512, "f32 valve still disables with fp16"))
        return 1;

    if (unsetenv(VALVE) != 0 || unsetenv(VALVE_FP16) != 0)
        return fail("restore production defaults");

    if (!file_has("ds4_qwen4exp_matmul.h",
                  "getenv(\"DS4_QWEN4EXP_NO_F32_LIBGEMM\") == NULL"))
        return fail("f32 valve default-on missing");
    if (file_has("ds4_qwen4exp_matmul.h", "NO_FP16_DOWN_PARTIAL"))
        return fail("f32 helper must not read fp16 valve");
    if (!file_has("ds4_cuda_qwen4exp.cu",
                  "getenv(\"DS4_QWEN4EXP_NO_FP16_DOWN_PARTIAL\") == NULL"
                  " ? 1u : 0u"))
        return fail("fp16 valve default-on missing");
    if (file_has("ds4_cuda_qwen4exp.cu", "NO_F32_LIBGEMM"))
        return fail("fp16 launch must not read f32 valve");
    return 0;
}

int main(void) {
    static const uint32_t decode_rows[] = {1u, 2u, 4u, 7u};
    static const uint32_t prefill_rows[] = {8u, 9u, 1024u};
    unsigned i;

    if (unsetenv(VALVE) != 0) return fail("unsetenv valve");
    if (unsetenv(VALVE_FP16) != 0) return fail("unsetenv fp16 valve");

    for (i = 0; i < sizeof(decode_rows) / sizeof(decode_rows[0]); i++) {
        uint32_t rows = decode_rows[i];
        if (call_lib(rows)) return 1;
        if (expect(FN_EXACT, rows, 2560, 512, "decode-width libgemm")) return 1;
        if (call_exact(rows)) return 1;
        if (expect(FN_EXACT, rows, 2560, 48, "decode-width GDN")) return 1;
    }

    for (i = 0; i < sizeof(prefill_rows) / sizeof(prefill_rows[0]); i++) {
        uint32_t rows = prefill_rows[i];
        int want = libgemm_live(rows) ? FN_TENSOR : FN_EXACT;
        if (call_lib(rows)) return 1;
        if (expect(want, rows, 2560, 512, "prefill libgemm default")) return 1;
        if (call_exact(rows)) return 1;
        if (expect(FN_EXACT, rows, 2560, 48, "prefill GDN stays exact")) return 1;
    }

    if (setenv(VALVE, "1", 1) != 0) return fail("setenv valve");
    if (call_lib(1024u)) return 1;
    if (expect(FN_EXACT, 1024u, 2560, 512, "valve 1 restores exact")) return 1;
    if (call_exact(1024u)) return 1;
    if (expect(FN_EXACT, 1024u, 2560, 48, "valve 1 GDN exact")) return 1;

    if (setenv(VALVE, "", 1) != 0) return fail("setenv empty valve");
    if (call_lib(8u)) return 1;
    if (expect(FN_EXACT, 8u, 2560, 512, "empty valve is presence-disable"))
        return 1;

    if (unsetenv(VALVE) != 0) return fail("clear valve");
    if (call_lib(8u)) return 1;
    if (expect(libgemm_live(8u) ? FN_TENSOR : FN_EXACT, 8u, 2560, 512,
               "unset valve re-enables libgemm"))
        return 1;
    if (call_exact(8u)) return 1;
    if (expect(FN_EXACT, 8u, 2560, 48, "GDN never takes libgemm")) return 1;

    if (check_independent_defaults()) return 1;

    printf("qwen4exp f32 libgemm dispatch: PASS (cuda_live=%d min_rows=%u)\n",
           libgemm_live(8u), DS4_QWEN4EXP_F32_LIBGEMM_MIN_ROWS);
    return 0;
}
