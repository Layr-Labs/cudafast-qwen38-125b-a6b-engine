/*
 * Qwen4exp per-layer embedding (PLE) kernel tests.
 *
 * Follows tests/test_qwen4exp_gdn.c: link against ds4_metal.o (or ds4_cuda.o
 * plus ds4_cuda_qwen4exp.o on a CUDA host), drive the kernels with synthetic weights in an anonymous
 * mapping, and compare against ds4_qwen4exp_ple_ref.h -- a plain-C reference
 * accumulated in DOUBLE and written from the MLX forward, not from the
 * kernels, so the band each check reports is the kernel's own error and not a
 * shared rounding order.
 *
 * Shapes are the production per-layer ones: hidden 2560, hyper-connection
 * width 4 (10240 channels), n-gram embedding 2560, convolution kernel 4 at
 * dilation 3, so the rolling window is 9 rows.  Row counts 1, 7, 64 and 128
 * straddle that window: 1 and 7 are shorter than it, 64 and 128 are longer,
 * and the two paths through the state update differ.
 *
 * Checks, in order:
 *   1. the gate against the reference at every row count, and again with the
 *      inner product driven below the 1e-6 floor inside the signed square
 *      root, which nothing else reaches;
 *   2. the gate in place (out == key) equals the out-of-place result, bit for
 *      bit -- the kernel writes the stream it is still reducing;
 *   3. the convolution, its residual add and its carried window against the
 *      reference at every row count, with the window bit exact;
 *   4. chunk invariance, bit exact: 128 rows in one call equals 120 + 8
 *      equals sixteen calls of 8, which is what pins the rolling window;
 *   5. the whole block -- both Q8_0 projections, all three grouped norms, the
 *      gate and the convolution -- against the reference block, under both
 *      norm-weight conventions the binder can classify;
 *   6. two near misses the checks above must SEPARATE the kernels from: a
 *      convolution window shifted by one row, and the norm epsilon moved from
 *      1e-6 to 1e-5.  Both are computed in the REFERENCE, so what they prove
 *      is that the assertion bands are tight enough to notice;
 *   7. a shard boundary inside the block: its six weights split across two
 *      mappings, at offsets neither shares with the one-shard model, must
 *      give the one-shard answer bit for bit.  Each weight is resolved
 *      through its OWN mapping, because a split GGUF can put a boundary
 *      inside one block and one mapping with six offsets would read
 *      plausible numbers out of the wrong file.
 *
 * tests/qwen4exp_ple_mutants.sh proves the same checks bite from the other
 * side, by rewriting one line of the shader at a time.
 */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#include "ds4.h"
#include "ds4_gpu.h"
#include "ds4_qwen4exp_hc_ref.h"
#include "ds4_qwen4exp_ple_ref.h"

bool ds4_log_is_tty(FILE *fp) {
    (void)fp;
    return false;
}

enum {
    N_EMBD = 2560,
    N_HC = 4,
    WIDE = N_EMBD * N_HC,
    PLE_EMBD = 2560,
    CONV_K = 4,
    DILATION = 3,
    STATE_ROWS = (CONV_K - 1) * DILATION,
    ROWS_MAX = 128,

    Q8_0_ROW_BYTES = (PLE_EMBD / 32) * 34,

    NORM_KEY_OFF = 0,                                   /* f32[10240]        */
    NORM_QUERY_OFF = NORM_KEY_OFF + WIDE * 4,           /* f32[10240]        */
    NORM_CONV_OFF = NORM_QUERY_OFF + WIDE * 4,          /* f32[10240]        */
    CONV_OFF = NORM_CONV_OFF + WIDE * 4,                /* f32[10240][4]     */
    KEY_OFF = CONV_OFF + WIDE * CONV_K * 4,             /* Q8_0 10240x2560   */
    VALUE_OFF = KEY_OFF + WIDE * Q8_0_ROW_BYTES,        /* Q8_0 2560x2560    */
    MODEL_BYTES = VALUE_OFF + N_EMBD * Q8_0_ROW_BYTES,
};

/* Half-precision bit patterns for exact Q8_0 scales: a power of two keeps
 * `scale * q` exact in f32 on both sides, so only the accumulation order
 * separates the kernel from the reference. */
enum {
    HALF_2_M12 = 0x0c00, /* 2^-12 */
    HALF_2_M11 = 0x1000, /* 2^-11 */
};

/* Bands.
 *
 * The gate reduces 2560 products in f32 against a double reference, so its
 * error is the reduction's, not the gate function's; the design's gate band
 * is 1e-5 and the kernel sits three orders inside it.  The whole block adds
 * two Q8_0 GEMMs over 2560 inputs, which is the design's dense-GEMM class, so
 * it is asserted as a relative Frobenius error the way that class is, AT that
 * class's 2e-2 ceiling.
 *
 * The ceiling and not a measured multiple, because the two backends do this
 * GEMM differently and a band cut to one of them rejects the other:
 *
 *   Metal   worst 1.43e-3  (128 rows, the epsilon-dominated case)
 *   CUDA    worst 6.04e-3  (64 rows, zero-centered norm weights)
 *
 * The 4x gap is structural, not a defect.  For n_tok > 1 with in_dim a
 * multiple of 256 -- 2560 is -- ds4_gpu_matmul_q8_0_tensor takes the vendored
 * MMQ prefill tier (ds4_cuda.cu, cuda_matmul_q8_0_tensor_labeled), which
 * QUANTIZES THE ACTIVATIONS to int8 per 32-element block through
 * quantize_mmq_q8_1_cuda (cuda/mmq/ds4_mmq.cu:611) and accumulates with dp4a.
 * Metal dequantizes the weights and accumulates the activations in f32.  The
 * block runs two such GEMMs over 2560 inputs, so CUDA carries an activation
 * quantization error Metal does not have at all, and it grows with rows as the
 * reduction lengthens.  Both sit inside 2e-2; a 5e-3 band admitted Metal and
 * rejected CUDA on arithmetic that is correct on both.
 *
 * Widening a band weakens what it proves, so require_separated() now also
 * demands every near miss fail THIS band and not only the measured error.
 *
 * The block's residual is not the PLE kernels': turning the bf16 rounding of
 * the normalized value off drops the single-row block to 8e-8, and what is
 * left at 64 and 128 rows is the Q8_0 matmul's own prefill path.  The two PLE
 * kernels are pinned at 1e-5 by checks 1 and 3, where nothing else is in the
 * way.  Elementwise the block is reported only, for the same reason.
 *
 * Both are the accuracy the kernels actually have and not the design ceiling:
 * a band loose enough to accept a near miss is not a test. */
static const float GATE_BAND = 1e-5f;
static const float CONV_BAND = 1e-5f;
static const double BLOCK_FROBENIUS_BAND = 2e-2;

/* How far the two near misses of check 6 must sit OUTSIDE the band before the
 * test believes it can tell them apart. */
static const double NEAR_MISS_MARGIN = 20.0;

static uint32_t g_rng = 0x2f6e2b1u;

static uint32_t next_u32(void) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 17;
    g_rng ^= g_rng << 5;
    return g_rng;
}

/* Uniform in [-1, 1). */
static float next_unit(void) {
    return (float)((int32_t)(next_u32() >> 8) - 8388608) * (1.0f / 8388608.0f);
}

static void require_ok(int ok, const char *what) {
    if (!ok) {
        fprintf(stderr, "%s failed\n", what);
        exit(1);
    }
}

static void *alloc_floats(uint64_t count) {
    void *p = malloc((size_t)count * sizeof(float));
    require_ok(p != NULL, "host float allocation");
    return p;
}

static double *alloc_doubles(uint64_t count) {
    double *p = malloc((size_t)count * sizeof(double));
    require_ok(p != NULL, "host double allocation");
    return p;
}

static void widen(double *out, const float *in, uint64_t count) {
    for (uint64_t i = 0; i < count; i++) out[i] = (double)in[i];
}

/* Max absolute difference against the double reference. */
static double measure_band(const char *what, const float *actual,
                           const double *expected, uint64_t count,
                           double tolerance, int report) {
    double worst = 0.0;
    uint64_t worst_at = 0;
    for (uint64_t i = 0; i < count; i++) {
        if (!isfinite(actual[i]) || !isfinite(expected[i])) {
            fprintf(stderr, "%s: non-finite at %llu (%g vs %g)\n", what,
                    (unsigned long long)i, (double)actual[i], expected[i]);
            exit(1);
        }
        const double diff = fabs((double)actual[i] - expected[i]);
        if (diff > worst) {
            worst = diff;
            worst_at = i;
        }
    }
    if (tolerance > 0.0 && worst > tolerance) {
        fprintf(stderr, "%s: max abs error %.6g at %llu (%.9g vs %.9g), band %.6g\n",
                what, worst, (unsigned long long)worst_at,
                (double)actual[worst_at], expected[worst_at], tolerance);
        exit(1);
    }
    if (report) {
        printf("  %-58s max abs %.3g (band %.3g)\n", what, worst, tolerance);
    }
    return worst;
}

static double relative_frobenius(const float *actual, const double *expected,
                                 uint64_t count) {
    double num = 0.0, den = 0.0;
    for (uint64_t i = 0; i < count; i++) {
        const double diff = (double)actual[i] - expected[i];
        num += diff * diff;
        den += expected[i] * expected[i];
    }
    return den > 0.0 ? sqrt(num / den) : sqrt(num);
}

static double require_relative_frobenius(const char *what, const float *actual,
                                         const double *expected, uint64_t count,
                                         double tolerance) {
    const double rel = relative_frobenius(actual, expected, count);
    if (!(rel <= tolerance)) {
        fprintf(stderr, "%s: relative Frobenius error %.6g above %.6g\n",
                what, rel, tolerance);
        exit(1);
    }
    printf("  %-58s rel Frobenius %.3g (band %.3g)\n", what, rel, tolerance);
    return rel;
}


static void require_identical(const char *what, const float *a, const float *b,
                              uint64_t count) {
    if (memcmp(a, b, (size_t)count * sizeof(float)) != 0) {
        for (uint64_t i = 0; i < count; i++) {
            if (a[i] != b[i]) {
                fprintf(stderr, "%s: differs at %llu (%.9g vs %.9g)\n", what,
                        (unsigned long long)i, (double)a[i], (double)b[i]);
                exit(1);
            }
        }
    }
    printf("  %-58s bit exact\n", what);
}

/* A near miss must sit well outside the band the kernel is held to, otherwise
 * the assertion above proves nothing about that line of the reference. */
static void require_separated(const char *what, double measured, double band) {
    /* The near miss must also fail the band the block is actually held to.
     * `band` above is the case's own measured error, so it tracks the backend;
     * BLOCK_FROBENIUS_BAND is the fixed ceiling, and a control that separated
     * from the measurement but passed the ceiling would prove nothing about
     * what the assertion accepts. */
    if (!(measured > BLOCK_FROBENIUS_BAND)) {
        fprintf(stderr,
                "%s: the near miss is %.6g, INSIDE the %.6g band the block is "
                "held to -- widening the band has made this control useless\n",
                what, measured, BLOCK_FROBENIUS_BAND);
        exit(1);
    }
    if (!(measured > band * NEAR_MISS_MARGIN)) {
        fprintf(stderr,
                "%s: the near miss is only %.6g from the kernel, inside %.0fx "
                "the %.6g band -- the check cannot separate them\n",
                what, measured, NEAR_MISS_MARGIN, band);
        exit(1);
    }
    printf("  %-58s separated, %.3g away (band %.3g)\n", what, measured, band);
}

static void fill_q8_0(uint8_t *base, uint32_t in_dim, uint32_t out_dim,
                      uint16_t scale) {
    const uint32_t blocks = in_dim / 32u;
    for (uint32_t o = 0; o < out_dim; o++) {
        for (uint32_t b = 0; b < blocks; b++) {
            uint8_t *block = base + ((uint64_t)o * blocks + b) * 34u;
            memcpy(block, &scale, sizeof(scale));
            for (uint32_t i = 0; i < 32u; i++) {
                block[2 + i] = (uint8_t)(int8_t)((int32_t)(next_u32() % 255u) - 127);
            }
        }
    }
}

static ds4_gpu_tensor *upload(const float *host, uint64_t count) {
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(count * sizeof(float));
    require_ok(t != NULL, "tensor allocation");
    require_ok(ds4_gpu_tensor_write(t, 0, host, count * sizeof(float)),
               "tensor upload");
    return t;
}

static void download(const ds4_gpu_tensor *t, float *host, uint64_t count) {
    require_ok(ds4_gpu_tensor_read(t, 0, host, count * sizeof(float)),
               "tensor download");
}

/* One weight's view: the mapping that holds it and where it starts there.
 * `expert_bytes` and `row_bytes` are unused by this block, which has no 3-D
 * tensor, and the type is F32 for the norms and the convolution; the two
 * projections carry their own in the matmul entry point. */
static ds4_gpu_qwen4exp_slab slab_of(const uint8_t *map, uint64_t map_bytes,
                                     uint64_t offset) {
    ds4_gpu_qwen4exp_slab s;
    memset(&s, 0, sizeof(s));
    s.map = map;
    s.map_size = map_bytes;
    s.offset = offset;
    return s;
}

static void write_tensor(ds4_gpu_tensor *t, const float *host, uint64_t count) {
    require_ok(ds4_gpu_tensor_write(t, 0, host, count * sizeof(float)),
               "tensor write");
}

/* ------------------------------------------------------------------ block */

/* The reference block: the op order of Qwen4ExpPLELayer.callAsFunction,
 * every step in double.  `hyper` is read for the query and accumulated into,
 * exactly as `stream = stream + ple(stream, ...)` does. */
static void reference_block_bias(double *hyper, double *state,
                                 const uint8_t *model, const double *ngram_rows,
                                 uint32_t rows, double eps, double bias) {
    const uint64_t wide_count = (uint64_t)rows * WIDE;
    double *key = alloc_doubles(wide_count);
    double *aux = alloc_doubles(wide_count);
    double *value = alloc_doubles((uint64_t)rows * N_EMBD);
    double *gated = alloc_doubles(wide_count);

    ds4_qwen4exp_ple_ref_matmul_q8_0(key, model + KEY_OFF, PLE_EMBD, WIDE,
                                     ngram_rows, rows,
                                     ds4_qwen4exp_ref_half_to_f32);
    ds4_qwen4exp_ple_ref_rms_norm(key, key, (const float *)(model + NORM_KEY_OFF),
                                  WIDE, N_EMBD, rows, eps, bias, 1);
    ds4_qwen4exp_ple_ref_matmul_q8_0(value, model + VALUE_OFF, PLE_EMBD, N_EMBD,
                                     ngram_rows, rows,
                                     ds4_qwen4exp_ref_half_to_f32);
    ds4_qwen4exp_ple_ref_rms_norm(aux, hyper,
                                  (const float *)(model + NORM_QUERY_OFF),
                                  WIDE, N_EMBD, rows, eps, bias, 1);
    ds4_qwen4exp_ple_ref_gate(gated, key, aux, value, N_EMBD, N_HC, rows);
    ds4_qwen4exp_ple_ref_rms_norm(aux, gated,
                                  (const float *)(model + NORM_CONV_OFF),
                                  WIDE, N_EMBD, rows, eps, bias, 1);
    ds4_qwen4exp_ple_ref_conv(hyper, state, gated, aux,
                              (const float *)(model + CONV_OFF), WIDE, CONV_K,
                              DILATION, rows);

    free(gated);
    free(value);
    free(aux);
    free(key);
}

/* The block at the baked convention, which is what the pinned checkpoint and
 * every other check here use. */
static void reference_block(double *hyper, double *state, const uint8_t *model,
                            const double *ngram_rows, uint32_t rows,
                            double eps) {
    reference_block_bias(hyper, state, model, ngram_rows, rows, eps, 0.0);
}

/* The same block with the convolution window shifted by one row: tap k reads
 * `t + dilation * k + 1` instead of `t + dilation * k`.  Written out rather
 * than parameterized so the reference above stays the reference. */
static void reference_conv_shifted(double *hyper, const double *gated,
                                   const double *conv_in, const double *state,
                                   const float *weight, uint32_t rows) {
    for (uint32_t c = 0; c < WIDE; c++) {
        for (uint32_t t = 0; t < rows; t++) {
            double acc = 0.0;
            for (uint32_t k = 0; k < CONV_K; k++) {
                const uint32_t i = t + DILATION * k + 1u;
                const double v = (i < STATE_ROWS)
                    ? state[(uint64_t)i * WIDE + c]
                    : ((i - STATE_ROWS) < rows
                           ? conv_in[(uint64_t)(i - STATE_ROWS) * WIDE + c]
                           : 0.0);
                acc += v * (double)weight[(uint64_t)c * CONV_K + k];
            }
            const uint64_t index = (uint64_t)t * WIDE + c;
            hyper[index] += gated[index] + acc * ds4_qwen4exp_ple_ref_sigmoid(acc);
        }
    }
}

int main(void) {
    uint8_t *model = mmap(NULL, MODEL_BYTES, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANON, -1, 0);
    if (model == MAP_FAILED) {
        perror("mmap");
        return 1;
    }

    float *norm_key = (float *)(model + NORM_KEY_OFF);
    float *norm_query = (float *)(model + NORM_QUERY_OFF);
    float *norm_conv = (float *)(model + NORM_CONV_OFF);
    float *conv_w = (float *)(model + CONV_OFF);
    for (uint32_t i = 0; i < WIDE; i++) {
        norm_key[i] = 1.0f + 0.25f * next_unit();
        norm_query[i] = 1.0f + 0.25f * next_unit();
        norm_conv[i] = 1.0f + 0.25f * next_unit();
    }
    for (uint32_t i = 0; i < WIDE * CONV_K; i++) conv_w[i] = 0.5f * next_unit();
    fill_q8_0(model + KEY_OFF, PLE_EMBD, WIDE, HALF_2_M12);
    fill_q8_0(model + VALUE_OFF, PLE_EMBD, N_EMBD, HALF_2_M11);

    require_ok(ds4_gpu_init(), "GPU initialization");
    require_ok(ds4_gpu_set_model_map(model, MODEL_BYTES),
               "model map registration");

    const uint32_t row_counts[4] = { 1u, 7u, 64u, ROWS_MAX };

    for (uint32_t which = 0; which < 4u; which++) {
        const uint32_t rows = row_counts[which];
        const uint64_t wide_count = (uint64_t)rows * WIDE;
        const uint64_t embd_count = (uint64_t)rows * N_EMBD;
        const uint64_t state_count = (uint64_t)STATE_ROWS * WIDE;

        printf("qwen4exp PLE kernels, %u row%s\n", rows, rows == 1u ? "" : "s");

        float *key = alloc_floats(wide_count);
        float *query = alloc_floats(wide_count);
        float *value = alloc_floats(embd_count);
        float *gated = alloc_floats(wide_count);
        float *conv_in = alloc_floats(wide_count);
        float *hyper = alloc_floats(wide_count);
        float *state = alloc_floats(state_count);
        float *gpu_wide = alloc_floats(wide_count);
        float *gpu_state = alloc_floats(state_count);
        for (uint64_t i = 0; i < wide_count; i++) {
            key[i] = next_unit();
            query[i] = next_unit();
            gated[i] = 0.5f * next_unit();
            conv_in[i] = next_unit();
            hyper[i] = next_unit();
        }
        for (uint64_t i = 0; i < embd_count; i++) value[i] = next_unit();
        for (uint64_t i = 0; i < state_count; i++) state[i] = next_unit();

        double *ref_wide = alloc_doubles(wide_count);
        double *ref_state = alloc_doubles(state_count);
        double *d_key = alloc_doubles(wide_count);
        double *d_query = alloc_doubles(wide_count);
        double *d_value = alloc_doubles(embd_count);
        double *d_gated = alloc_doubles(wide_count);
        double *d_conv_in = alloc_doubles(wide_count);
        widen(d_key, key, wide_count);
        widen(d_query, query, wide_count);
        widen(d_value, value, embd_count);
        widen(d_gated, gated, wide_count);
        widen(d_conv_in, conv_in, wide_count);

        /* ---- 1. the gate ------------------------------------------- */
        ds4_gpu_tensor *key_t = upload(key, wide_count);
        ds4_gpu_tensor *query_t = upload(query, wide_count);
        ds4_gpu_tensor *value_t = upload(value, embd_count);
        ds4_gpu_tensor *out_t = ds4_gpu_tensor_alloc(wide_count * sizeof(float));
        require_ok(out_t != NULL, "gate output allocation");

        require_ok(ds4_gpu_qwen4exp_ple_gate_tensor(out_t, key_t, query_t,
                                                    value_t, N_EMBD, N_HC, rows),
                   "qwen4exp PLE gate");
        download(out_t, gpu_wide, wide_count);
        ds4_qwen4exp_ple_ref_gate(ref_wide, d_key, d_query, d_value,
                                  N_EMBD, N_HC, rows);
        measure_band("PLE gate against the double reference", gpu_wide,
                     ref_wide, wide_count, GATE_BAND, 1);

        /* ---- 1b. the gate at the signed-square-root floor ----------
         * The reference floors the inner product at 1e-6 inside the absolute
         * value before the square root, and nothing else in the block exposes
         * that line: at production scale the product is orders above it.  So
         * drive it there, with query == key so the sum of squares is strictly
         * positive and no cancellation can move the sign the floor keeps.
         * Floored the gate is sigmoid(1e-3); unfloored it is sigmoid(4e-5),
         * which is 1e-4 of output away and outside the band. */
        {
            float *tiny = alloc_floats(wide_count);
            for (uint64_t i = 0; i < wide_count; i++) tiny[i] = key[i] * 1e-5f;
            double *d_tiny = alloc_doubles(wide_count);
            widen(d_tiny, tiny, wide_count);

            ds4_gpu_tensor *tiny_t = upload(tiny, wide_count);
            ds4_gpu_tensor *floor_out =
                ds4_gpu_tensor_alloc(wide_count * sizeof(float));
            require_ok(floor_out != NULL, "floor output allocation");
            require_ok(ds4_gpu_qwen4exp_ple_gate_tensor(
                           floor_out, tiny_t, tiny_t, value_t, N_EMBD, N_HC,
                           rows),
                       "qwen4exp PLE gate at the floor");
            float *floor_gpu = alloc_floats(wide_count);
            download(floor_out, floor_gpu, wide_count);
            ds4_qwen4exp_ple_ref_gate(ref_wide, d_tiny, d_tiny, d_value,
                                      N_EMBD, N_HC, rows);
            measure_band("PLE gate at the signed-square-root floor", floor_gpu,
                         ref_wide, wide_count, GATE_BAND, 1);

            free(floor_gpu);
            ds4_gpu_tensor_free(floor_out);
            ds4_gpu_tensor_free(tiny_t);
            free(d_tiny);
            free(tiny);
        }

        /* ---- 2. the gate in place ---------------------------------- */
        {
            float *in_place = alloc_floats(wide_count);
            require_ok(ds4_gpu_qwen4exp_ple_gate_tensor(key_t, key_t, query_t,
                                                        value_t, N_EMBD, N_HC,
                                                        rows),
                       "qwen4exp PLE gate in place");
            download(key_t, in_place, wide_count);
            require_identical("PLE gate in place equals out of place",
                              in_place, gpu_wide, wide_count);
            free(in_place);
        }

        ds4_gpu_tensor_free(out_t);
        ds4_gpu_tensor_free(value_t);
        ds4_gpu_tensor_free(query_t);
        ds4_gpu_tensor_free(key_t);

        /* ---- 3. the convolution, the add and the window ------------- */
        ds4_gpu_tensor *hyper_t = upload(hyper, wide_count);
        ds4_gpu_tensor *state_t = upload(state, state_count);
        ds4_gpu_tensor *gated_t = upload(gated, wide_count);
        ds4_gpu_tensor *conv_t = upload(conv_in, wide_count);

        require_ok(ds4_gpu_qwen4exp_ple_conv_tensor(
                       hyper_t, state_t, NULL, 0u, gated_t, conv_t, model, MODEL_BYTES,
                       CONV_OFF, WIDE, CONV_K, DILATION, rows, NULL),
                   "qwen4exp PLE convolution");
        download(hyper_t, gpu_wide, wide_count);
        download(state_t, gpu_state, state_count);

        widen(ref_wide, hyper, wide_count);
        widen(ref_state, state, state_count);
        ds4_qwen4exp_ple_ref_conv(ref_wide, ref_state, d_gated, d_conv_in,
                                  conv_w, WIDE, CONV_K, DILATION, rows);
        measure_band("PLE convolution and residual add", gpu_wide, ref_wide,
                     wide_count, CONV_BAND, 1);
        /* The window is a copy of rows the kernel never arithmetically
         * touches, so it must come back bit for bit. */
        require_ok(measure_band("PLE convolution window", gpu_state, ref_state,
                                state_count, 0.0, 0) == 0.0,
                   "PLE convolution window is bit exact");
        printf("  %-58s bit exact\n", "PLE convolution window");

        /* ---- 6a. near miss: the window shifted by one row ----------- */
        {
            double *shifted = alloc_doubles(wide_count);
            widen(shifted, hyper, wide_count);
            double *state_copy = alloc_doubles(state_count);
            widen(state_copy, state, state_count);
            reference_conv_shifted(shifted, d_gated, d_conv_in, state_copy,
                                   conv_w, rows);
            const double away = measure_band(
                "near miss: convolution window shifted one row", gpu_wide,
                shifted, wide_count, 0.0, 0);
            require_separated("near miss: convolution window shifted one row",
                              away, CONV_BAND);
            free(state_copy);
            free(shifted);
        }

        ds4_gpu_tensor_free(conv_t);
        ds4_gpu_tensor_free(gated_t);
        ds4_gpu_tensor_free(state_t);
        ds4_gpu_tensor_free(hyper_t);

        /* ---- 5. the whole block ------------------------------------ */
        {
            float *ngram = alloc_floats((uint64_t)rows * PLE_EMBD);
            for (uint64_t i = 0; i < (uint64_t)rows * PLE_EMBD; i++) {
                ngram[i] = next_unit();
            }
            double *d_ngram = alloc_doubles((uint64_t)rows * PLE_EMBD);
            widen(d_ngram, ngram, (uint64_t)rows * PLE_EMBD);

            ds4_gpu_tensor *block_hyper = upload(hyper, wide_count);
            ds4_gpu_tensor *block_state = upload(state, state_count);
            ds4_gpu_tensor *rows_t = upload(ngram, (uint64_t)rows * PLE_EMBD);
            ds4_gpu_tensor *scratch_key =
                ds4_gpu_tensor_alloc(wide_count * sizeof(float));
            ds4_gpu_tensor *scratch_aux =
                ds4_gpu_tensor_alloc(wide_count * sizeof(float));
            ds4_gpu_tensor *scratch_value =
                ds4_gpu_tensor_alloc(embd_count * sizeof(float));
            require_ok(scratch_key && scratch_aux && scratch_value,
                       "block scratch allocation");

            const ds4_gpu_qwen4exp_slab one_key =
                slab_of(model, MODEL_BYTES, KEY_OFF);
            const ds4_gpu_qwen4exp_slab one_value =
                slab_of(model, MODEL_BYTES, VALUE_OFF);
            const ds4_gpu_qwen4exp_slab one_nkey =
                slab_of(model, MODEL_BYTES, NORM_KEY_OFF);
            const ds4_gpu_qwen4exp_slab one_nquery =
                slab_of(model, MODEL_BYTES, NORM_QUERY_OFF);
            const ds4_gpu_qwen4exp_slab one_nconv =
                slab_of(model, MODEL_BYTES, NORM_CONV_OFF);
            const ds4_gpu_qwen4exp_slab one_conv =
                slab_of(model, MODEL_BYTES, CONV_OFF);

            require_ok(ds4_gpu_qwen4exp_ple_block_tensor(
                           block_hyper, block_state, NULL, 0u, scratch_key, scratch_aux,
                           scratch_value, rows_t, &one_key, &one_value,
                           &one_nkey, &one_nquery, &one_nconv, &one_conv,
                           N_EMBD, N_HC, PLE_EMBD,
                           CONV_K, DILATION, rows, 1e-6f, 0.0f, 0.0f, 0.0f, 1, NULL),
                       "qwen4exp PLE block");
            download(block_hyper, gpu_wide, wide_count);

            widen(ref_wide, hyper, wide_count);
            widen(ref_state, state, state_count);
            reference_block(ref_wide, ref_state, model, d_ngram, rows, 1e-6);
            require_relative_frobenius("PLE block against the double reference",
                                       gpu_wide, ref_wide, wide_count,
                                       BLOCK_FROBENIUS_BAND);
            printf("  %-58s max abs %.3g (reported)\n", "PLE block, elementwise",
                   measure_band("PLE block", gpu_wide, ref_wide, wide_count,
                                0.0, 0));

            /* ---- 5b. the other norm-weight convention --------------
             * The binder classifies each norm weight and hands the block
             * either 0 (the checkpoint baked `1 + w`) or 1 (it stores
             * zero-centered `w`).  Everything above runs the baked
             * convention; run the zero-centered one too, or the parameter is
             * only ever seen at one value. */
            {
                double *biased = alloc_doubles(wide_count);
                double *biased_state = alloc_doubles(state_count);
                widen(biased, hyper, wide_count);
                widen(biased_state, state, state_count);

                ds4_gpu_tensor *bh = upload(hyper, wide_count);
                ds4_gpu_tensor *bs = upload(state, state_count);
                require_ok(ds4_gpu_qwen4exp_ple_block_tensor(
                               bh, bs, NULL, 0u, scratch_key, scratch_aux, scratch_value,
                               rows_t, &one_key, &one_value, &one_nkey,
                               &one_nquery, &one_nconv, &one_conv, N_EMBD,
                               N_HC, PLE_EMBD, CONV_K, DILATION, rows, 1e-6f,
                               1.0f, 1.0f, 1.0f, 1, NULL),
                           "qwen4exp PLE block, zero-centered norm weights");
                float *biased_gpu = alloc_floats(wide_count);
                download(bh, biased_gpu, wide_count);
                reference_block_bias(biased, biased_state, model, d_ngram,
                                     rows, 1e-6, 1.0);
                require_relative_frobenius(
                    "PLE block, zero-centered norm weights", biased_gpu,
                    biased, wide_count, BLOCK_FROBENIUS_BAND);
                free(biased_gpu);
                ds4_gpu_tensor_free(bs);
                ds4_gpu_tensor_free(bh);
                free(biased_state);
                free(biased);
            }

            /* ---- 6b. near miss: the norm epsilon moved -------------
             * At production activation scale the norm's mean square is order
             * one and the epsilon is 1e-6, so moving it to 1e-5 changes the
             * scale by 4e-6 -- a real property of the norm, and one no test at
             * that scale can separate from its own rounding.  So this runs the
             * block again on activations small enough that the EPSILON
             * DOMINATES the mean square, which is where a wrong epsilon
             * actually bites and is the regime tests/test_qwen4exp_gdn.c uses
             * for the same reason.  Then the two epsilons are two orders
             * apart. */
            {
                float *small = alloc_floats((uint64_t)rows * PLE_EMBD);
                float *small_hyper = alloc_floats(wide_count);
                for (uint64_t i = 0; i < (uint64_t)rows * PLE_EMBD; i++) {
                    small[i] = ngram[i] * 6e-4f;
                }
                for (uint64_t i = 0; i < wide_count; i++) {
                    small_hyper[i] = hyper[i] * 1e-3f;
                }
                double *d_small = alloc_doubles((uint64_t)rows * PLE_EMBD);
                widen(d_small, small, (uint64_t)rows * PLE_EMBD);

                ds4_gpu_tensor *small_hyper_t = upload(small_hyper, wide_count);
                ds4_gpu_tensor *small_state_t = upload(state, state_count);
                ds4_gpu_tensor *small_rows_t =
                    upload(small, (uint64_t)rows * PLE_EMBD);
                require_ok(ds4_gpu_qwen4exp_ple_block_tensor(
                               small_hyper_t, small_state_t, NULL, 0u, scratch_key,
                               scratch_aux, scratch_value, small_rows_t,
                               &one_key, &one_value, &one_nkey, &one_nquery,
                               &one_nconv, &one_conv, N_EMBD,
                               N_HC, PLE_EMBD, CONV_K, DILATION, rows, 1e-6f,
                               0.0f, 0.0f, 0.0f, 1, NULL),
                           "qwen4exp PLE block, epsilon-dominant");
                download(small_hyper_t, gpu_wide, wide_count);

                double *small_ref = alloc_doubles(wide_count);
                double *small_ref_state = alloc_doubles(state_count);
                widen(small_ref, small_hyper, wide_count);
                widen(small_ref_state, state, state_count);
                reference_block(small_ref, small_ref_state, model, d_small,
                                rows, 1e-6);
                const double small_rel = require_relative_frobenius(
                    "PLE block where the norm epsilon dominates", gpu_wide,
                    small_ref, wide_count, BLOCK_FROBENIUS_BAND);

                double *moved = alloc_doubles(wide_count);
                double *moved_state = alloc_doubles(state_count);
                widen(moved, small_hyper, wide_count);
                widen(moved_state, state, state_count);
                reference_block(moved, moved_state, model, d_small, rows, 1e-5);
                require_separated("near miss: norm epsilon 1e-5",
                                  relative_frobenius(gpu_wide, moved, wide_count),
                                  small_rel);

                free(moved_state);
                free(moved);
                free(small_ref_state);
                free(small_ref);
                ds4_gpu_tensor_free(small_rows_t);
                ds4_gpu_tensor_free(small_state_t);
                ds4_gpu_tensor_free(small_hyper_t);
                free(d_small);
                free(small_hyper);
                free(small);
            }

            ds4_gpu_tensor_free(scratch_value);
            ds4_gpu_tensor_free(scratch_aux);
            ds4_gpu_tensor_free(scratch_key);
            ds4_gpu_tensor_free(rows_t);
            ds4_gpu_tensor_free(block_state);
            ds4_gpu_tensor_free(block_hyper);
            free(d_ngram);
            free(ngram);
        }

        free(d_conv_in);
        free(d_gated);
        free(d_value);
        free(d_query);
        free(d_key);
        free(ref_state);
        free(ref_wide);
        free(gpu_state);
        free(gpu_wide);
        free(state);
        free(hyper);
        free(conv_in);
        free(gated);
        free(value);
        free(query);
        free(key);
    }

    /* ---- 4. chunk invariance, bit exact ---------------------------- */
    {
        const uint32_t rows = ROWS_MAX;
        const uint64_t wide_count = (uint64_t)rows * WIDE;
        const uint64_t state_count = (uint64_t)STATE_ROWS * WIDE;
        /* Two splits of the same 128 rows.  Both must reproduce the single
         * call bit for bit, output and window alike; only the rolling window
         * carries anything across a boundary, so this is what pins it. */
        const uint32_t plan_a[2] = { 120u, 8u };
        const uint32_t plan_b[16] = { 8u, 8u, 8u, 8u, 8u, 8u, 8u, 8u,
                                      8u, 8u, 8u, 8u, 8u, 8u, 8u, 8u };
        const uint32_t *plans[2] = { plan_a, plan_b };
        const uint32_t plan_steps[2] = { 2u, 16u };
        const char *plan_names[2] = { "120 + 8", "sixteen calls of 8" };

        float *gated = alloc_floats(wide_count);
        float *conv_in = alloc_floats(wide_count);
        float *hyper = alloc_floats(wide_count);
        float *state = alloc_floats(state_count);
        float *whole = alloc_floats(wide_count);
        float *whole_state = alloc_floats(state_count);
        float *split = alloc_floats(wide_count);
        float *split_state = alloc_floats(state_count);
        for (uint64_t i = 0; i < wide_count; i++) {
            gated[i] = 0.5f * next_unit();
            conv_in[i] = next_unit();
            hyper[i] = next_unit();
        }
        for (uint64_t i = 0; i < state_count; i++) state[i] = next_unit();

        ds4_gpu_tensor *hyper_t = upload(hyper, wide_count);
        ds4_gpu_tensor *state_t = upload(state, state_count);
        ds4_gpu_tensor *gated_t = upload(gated, wide_count);
        ds4_gpu_tensor *conv_t = upload(conv_in, wide_count);

        require_ok(ds4_gpu_qwen4exp_ple_conv_tensor(
                       hyper_t, state_t, NULL, 0u, gated_t, conv_t, model, MODEL_BYTES,
                       CONV_OFF, WIDE, CONV_K, DILATION, rows, NULL),
                   "chunk invariance, whole call");
        download(hyper_t, whole, wide_count);
        download(state_t, whole_state, state_count);

        char label[96];
        for (uint32_t p = 0; p < 2u; p++) {
            write_tensor(state_t, state, state_count);
            uint64_t done = 0;
            for (uint32_t step = 0; step < plan_steps[p]; step++) {
                const uint32_t chunk = plans[p][step];
                const uint64_t offset = done * WIDE;
                const uint64_t count = (uint64_t)chunk * WIDE;
                /* Every call reads its inputs from row 0, so a chunk is driven
                 * by uploading the slice it should see. */
                write_tensor(hyper_t, hyper + offset, count);
                write_tensor(gated_t, gated + offset, count);
                write_tensor(conv_t, conv_in + offset, count);
                require_ok(ds4_gpu_qwen4exp_ple_conv_tensor(
                               hyper_t, state_t, NULL, 0u, gated_t, conv_t, model,
                               MODEL_BYTES, CONV_OFF, WIDE, CONV_K, DILATION,
                               chunk, NULL),
                           "chunk invariance, chunked call");
                download(hyper_t, split + offset, count);
                done += chunk;
            }
            download(state_t, split_state, state_count);
            snprintf(label, sizeof(label), "chunk invariance: %s equals 128",
                     plan_names[p]);
            require_identical(label, split, whole, wide_count);
            snprintf(label, sizeof(label), "chunk invariance: window after %s",
                     plan_names[p]);
            require_identical(label, split_state, whole_state, state_count);
        }

        ds4_gpu_tensor_free(conv_t);
        ds4_gpu_tensor_free(gated_t);
        ds4_gpu_tensor_free(state_t);
        ds4_gpu_tensor_free(hyper_t);
        free(split_state);
        free(split);
        free(whole_state);
        free(whole);
        free(state);
        free(hyper);
        free(conv_in);
        free(gated);
    }

    /* ---- 7. a shard boundary inside the block ---------------------- *
     * The production artifact is a four-way split and a boundary can fall
     * inside one block, so the six PLE tensors are resolved against the shard
     * mapping that holds each of them rather than against one mapping with
     * six offsets.  This builds exactly that: shard A carries ple_key and the
     * key and query norms, shard B carries ple_value, the conv norm and
     * ple_conv1d, and neither keeps the offsets the one-shard model uses --
     * so resolving all six against one mapping cannot land on the right bytes
     * by accident.  The answer must be the one-shard answer, bit for bit. */
    {
        enum {
            A_KEY_OFF     = 4096,
            A_NKEY_OFF    = A_KEY_OFF + WIDE * Q8_0_ROW_BYTES,
            A_NQUERY_OFF  = A_NKEY_OFF + WIDE * 4,
            A_BYTES       = A_NQUERY_OFF + WIDE * 4,

            B_NCONV_OFF   = 8192,
            B_CONV_OFF    = B_NCONV_OFF + WIDE * 4,
            B_VALUE_OFF   = B_CONV_OFF + WIDE * CONV_K * 4,
            B_BYTES       = B_VALUE_OFF + N_EMBD * Q8_0_ROW_BYTES,
        };
        const uint32_t shard_rows[2] = { 1u, 64u };

        /* The one-shard answers first: registering the two shards may take
         * the primary mapping's place. */
        float *one_out[2];
        float *one_state_out[2];
        float *shard_hyper[2];
        float *shard_state[2];
        float *shard_rows_in[2];

        for (uint32_t which = 0; which < 2u; which++) {
            const uint32_t rows = shard_rows[which];
            const uint64_t wide_count = (uint64_t)rows * WIDE;
            const uint64_t state_count = (uint64_t)STATE_ROWS * WIDE;

            shard_hyper[which] = alloc_floats(wide_count);
            shard_state[which] = alloc_floats(state_count);
            shard_rows_in[which] = alloc_floats((uint64_t)rows * PLE_EMBD);
            one_out[which] = alloc_floats(wide_count);
            one_state_out[which] = alloc_floats(state_count);
            for (uint64_t i = 0; i < wide_count; i++) {
                shard_hyper[which][i] = next_unit();
            }
            for (uint64_t i = 0; i < state_count; i++) {
                shard_state[which][i] = next_unit();
            }
            for (uint64_t i = 0; i < (uint64_t)rows * PLE_EMBD; i++) {
                shard_rows_in[which][i] = next_unit();
            }

            const ds4_gpu_qwen4exp_slab k = slab_of(model, MODEL_BYTES, KEY_OFF);
            const ds4_gpu_qwen4exp_slab v = slab_of(model, MODEL_BYTES, VALUE_OFF);
            const ds4_gpu_qwen4exp_slab nk = slab_of(model, MODEL_BYTES, NORM_KEY_OFF);
            const ds4_gpu_qwen4exp_slab nq = slab_of(model, MODEL_BYTES, NORM_QUERY_OFF);
            const ds4_gpu_qwen4exp_slab nc = slab_of(model, MODEL_BYTES, NORM_CONV_OFF);
            const ds4_gpu_qwen4exp_slab cw = slab_of(model, MODEL_BYTES, CONV_OFF);

            ds4_gpu_tensor *h = upload(shard_hyper[which], wide_count);
            ds4_gpu_tensor *st = upload(shard_state[which], state_count);
            ds4_gpu_tensor *r =
                upload(shard_rows_in[which], (uint64_t)rows * PLE_EMBD);
            ds4_gpu_tensor *sk = ds4_gpu_tensor_alloc(wide_count * sizeof(float));
            ds4_gpu_tensor *sa = ds4_gpu_tensor_alloc(wide_count * sizeof(float));
            ds4_gpu_tensor *sv =
                ds4_gpu_tensor_alloc((uint64_t)rows * N_EMBD * sizeof(float));
            require_ok(sk && sa && sv, "one-shard scratch allocation");
            require_ok(ds4_gpu_qwen4exp_ple_block_tensor(
                           h, st, NULL, 0u, sk, sa, sv, r, &k, &v, &nk, &nq, &nc, &cw,
                           N_EMBD, N_HC, PLE_EMBD, CONV_K, DILATION, rows,
                           1e-6f, 0.0f, 0.0f, 0.0f, 1, NULL),
                       "qwen4exp PLE block, one shard");
            download(h, one_out[which], wide_count);
            download(st, one_state_out[which], state_count);
            ds4_gpu_tensor_free(sv);
            ds4_gpu_tensor_free(sa);
            ds4_gpu_tensor_free(sk);
            ds4_gpu_tensor_free(r);
            ds4_gpu_tensor_free(st);
            ds4_gpu_tensor_free(h);
        }

        uint8_t *shard_a = mmap(NULL, A_BYTES, PROT_READ | PROT_WRITE,
                                MAP_PRIVATE | MAP_ANON, -1, 0);
        uint8_t *shard_b = mmap(NULL, B_BYTES, PROT_READ | PROT_WRITE,
                                MAP_PRIVATE | MAP_ANON, -1, 0);
        require_ok(shard_a != MAP_FAILED && shard_b != MAP_FAILED,
                   "shard mappings");
        memcpy(shard_a + A_KEY_OFF, model + KEY_OFF,
               (size_t)WIDE * Q8_0_ROW_BYTES);
        memcpy(shard_a + A_NKEY_OFF, model + NORM_KEY_OFF, WIDE * 4);
        memcpy(shard_a + A_NQUERY_OFF, model + NORM_QUERY_OFF, WIDE * 4);
        memcpy(shard_b + B_NCONV_OFF, model + NORM_CONV_OFF, WIDE * 4);
        memcpy(shard_b + B_CONV_OFF, model + CONV_OFF, WIDE * CONV_K * 4);
        memcpy(shard_b + B_VALUE_OFF, model + VALUE_OFF,
               (size_t)N_EMBD * Q8_0_ROW_BYTES);

#if defined(__APPLE__)
        require_ok(ds4_gpu_set_model_map_range(shard_a, A_BYTES, 0, A_BYTES,
                                               WIDE * Q8_0_ROW_BYTES),
                   "shard A registration");
        require_ok(ds4_gpu_set_model_map_range(shard_b, B_BYTES, 0, B_BYTES,
                                               N_EMBD * Q8_0_ROW_BYTES),
                   "shard B registration");
#else
        require_ok(ds4_gpu_set_model_map(shard_a, A_BYTES),
                   "shard A registration");
        require_ok(ds4_gpu_set_aux_model_map_range(shard_b, B_BYTES, 0, B_BYTES),
                   "shard B registration");
#endif

        char label[96];
        for (uint32_t which = 0; which < 2u; which++) {
            const uint32_t rows = shard_rows[which];
            const uint64_t wide_count = (uint64_t)rows * WIDE;
            const uint64_t state_count = (uint64_t)STATE_ROWS * WIDE;

            const ds4_gpu_qwen4exp_slab k = slab_of(shard_a, A_BYTES, A_KEY_OFF);
            const ds4_gpu_qwen4exp_slab nk = slab_of(shard_a, A_BYTES, A_NKEY_OFF);
            const ds4_gpu_qwen4exp_slab nq = slab_of(shard_a, A_BYTES, A_NQUERY_OFF);
            const ds4_gpu_qwen4exp_slab v = slab_of(shard_b, B_BYTES, B_VALUE_OFF);
            const ds4_gpu_qwen4exp_slab nc = slab_of(shard_b, B_BYTES, B_NCONV_OFF);
            const ds4_gpu_qwen4exp_slab cw = slab_of(shard_b, B_BYTES, B_CONV_OFF);

            float *split_out = alloc_floats(wide_count);
            float *split_state_out = alloc_floats(state_count);
            ds4_gpu_tensor *h = upload(shard_hyper[which], wide_count);
            ds4_gpu_tensor *st = upload(shard_state[which], state_count);
            ds4_gpu_tensor *r =
                upload(shard_rows_in[which], (uint64_t)rows * PLE_EMBD);
            ds4_gpu_tensor *sk = ds4_gpu_tensor_alloc(wide_count * sizeof(float));
            ds4_gpu_tensor *sa = ds4_gpu_tensor_alloc(wide_count * sizeof(float));
            ds4_gpu_tensor *sv =
                ds4_gpu_tensor_alloc((uint64_t)rows * N_EMBD * sizeof(float));
            require_ok(sk && sa && sv, "two-shard scratch allocation");
            require_ok(ds4_gpu_qwen4exp_ple_block_tensor(
                           h, st, NULL, 0u, sk, sa, sv, r, &k, &v, &nk, &nq, &nc, &cw,
                           N_EMBD, N_HC, PLE_EMBD, CONV_K, DILATION, rows,
                           1e-6f, 0.0f, 0.0f, 0.0f, 1, NULL),
                       "qwen4exp PLE block, two shards");
            download(h, split_out, wide_count);
            download(st, split_state_out, state_count);

            snprintf(label, sizeof(label),
                     "shard boundary in the block, %u row%s", rows,
                     rows == 1u ? "" : "s");
            require_identical(label, split_out, one_out[which], wide_count);
            snprintf(label, sizeof(label),
                     "shard boundary, window after %u row%s", rows,
                     rows == 1u ? "" : "s");
            require_identical(label, split_state_out, one_state_out[which],
                              state_count);

            ds4_gpu_tensor_free(sv);
            ds4_gpu_tensor_free(sa);
            ds4_gpu_tensor_free(sk);
            ds4_gpu_tensor_free(r);
            ds4_gpu_tensor_free(st);
            ds4_gpu_tensor_free(h);
            free(split_state_out);
            free(split_out);
        }

        munmap(shard_b, B_BYTES);
        munmap(shard_a, A_BYTES);
        for (uint32_t which = 0; which < 2u; which++) {
            free(one_state_out[which]);
            free(one_out[which]);
            free(shard_rows_in[which]);
            free(shard_state[which]);
            free(shard_hyper[which]);
        }
    }

    ds4_gpu_cleanup();
    munmap(model, MODEL_BYTES);
    printf("test_qwen4exp_ple_kernels: ok\n");
    return 0;
}
