/*
 * Qwen4-Exp gated delta net against a plain double-precision reference.
 *
 * tests/test_qwen4exp_gdn.c already carries a double reference, but it checks
 * four lengths (1, 7, 64, 1024) and its other cases compare the kernel with
 * itself at different chunk widths.  A kernel that is CONSISTENTLY wrong --
 * one term dropped, one scale applied twice -- survives every self-comparison,
 * so this file re-states the recurrence from the model semantics and walks the
 * lengths a collapsing prompt actually passes through: 1, 6, 8, 16, 33, 64,
 * 65, 128, 200.  Six tokens is the length the real checkpoint still decodes
 * correctly at; 64 is where it starts to repeat.
 *
 * The reference is a per-token loop in double with no chunking of any kind:
 *
 *   conv[c]  = silu(sum_{w<4} history[w][c] * W[c][w])   K=4 causal, carried
 *   q,k      = x * rsqrt(sum(x^2) + qk_eps), eps on the SUM of squares,
 *              then q *= head_dim^-0.5 and k takes no post scale
 *   g        = exp(a_log[h] * softplus(alpha[h] + dt_bias[h]))
 *              (a_log IS -exp(A_log): the converter stores it exponentiated
 *               and negated, so it is used as-is)
 *   b        = sigmoid(beta[h])
 *   S       := g * S;  S += outer(k, (v - S.k) * b)      per value row
 *   y        = S . q
 *   out      = y * rsqrt(mean(y^2) + norm_eps) * norm * sigmoid(gate)
 *
 * Compared against the kernel: every output row, the carried recurrent state,
 * and the carried convolution history.  Tolerance is relative, not bit exact:
 * the kernel is f32 and the reference is f64.
 */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#include "ds4_gpu.h"

enum {
    D = 128,             /* gdn head dim, DS4_SHAPE_QWEN4EXP.n_gdn_head_dim   */
    KEY_HEADS = 16,      /* n_gdn_key_head                                    */
    VALUE_HEADS = 48,    /* n_gdn_value_head                                  */
    CONV_K = 4,          /* n_gdn_conv                                        */
    HISTORY = CONV_K - 1,
    KEY_DIM = KEY_HEADS * D,
    VALUE_DIM = VALUE_HEADS * D,
    CONV_DIM = 2 * KEY_DIM + VALUE_DIM,
    STATE_ELEMENTS = VALUE_HEADS * D * D,
    MAX_TOKENS = 200,

    CONV_OFFSET = 0,
    CONV_BYTES = CONV_DIM * CONV_K * 4,
    A_LOG_OFFSET = CONV_OFFSET + CONV_BYTES,
    A_LOG_BYTES = 256,
    DT_BIAS_OFFSET = A_LOG_OFFSET + A_LOG_BYTES,
    DT_BIAS_BYTES = 256,
    NORM_OFFSET = DT_BIAS_OFFSET + DT_BIAS_BYTES,
    NORM_BYTES = 1024,
    MODEL_BYTES = NORM_OFFSET + NORM_BYTES,
};

/* DS4_RMS_EPS for this family; the graph passes it as both eps arguments. */
static const float QK_NORM_EPS = 1e-6f;
static const float NORM_EPS = 1e-6f;

/* Relative band, |kernel - reference|_max / rms(reference).  The kernel keeps
 * f32 state through a 200-step recurrence, so this is loose enough for the
 * accumulation and far tighter than any dropped or doubled term. */
static const double BAND = 2.0e-4;

bool ds4_log_is_tty(FILE *fp) { (void)fp; return false; }

static void require(int ok, const char *what) {
    if (!ok) { fprintf(stderr, "test_qwen4exp_gdn_ref: %s failed\n", what); exit(1); }
}

static void *xalloc(size_t bytes) {
    void *p = malloc(bytes);
    require(p != NULL, "allocation");
    return p;
}

static double sample(uint64_t seed) {
    uint64_t z = seed + 0x9e3779b97f4a7c15ull;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    z ^= z >> 31;
    return (double)(z >> 11) * (2.0 / 9007199254740992.0) - 1.0;
}

static ds4_gpu_qwen4exp_slab slab_at(const void *map, uint64_t offset) {
    ds4_gpu_qwen4exp_slab s;
    memset(&s, 0, sizeof(s));
    s.map = map;
    s.map_size = MODEL_BYTES;
    s.offset = offset;
    return s;
}

/* ------------------------------------------------------------- reference */

static double ref_silu(double x)    { return x / (1.0 + exp(-x)); }
static double ref_sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
static double ref_softplus(double x) {
    return (x > 0.0 ? x : 0.0) + log1p(exp(-fabs(x)));
}

typedef struct {
    double *state;    /* [VALUE_HEADS][D][D], row-major over (value, key) */
    double *history;  /* [HISTORY][CONV_DIM], oldest first                */
    double *conv;     /* [CONV_DIM]                                       */
    double *y;        /* [VALUE_DIM]                                      */
} ref_state;

static void ref_open(ref_state *r) {
    r->state   = xalloc((size_t)STATE_ELEMENTS * sizeof(double));
    r->history = xalloc((size_t)HISTORY * CONV_DIM * sizeof(double));
    r->conv    = xalloc((size_t)CONV_DIM * sizeof(double));
    r->y       = xalloc((size_t)VALUE_DIM * sizeof(double));
    memset(r->state, 0, (size_t)STATE_ELEMENTS * sizeof(double));
    memset(r->history, 0, (size_t)HISTORY * CONV_DIM * sizeof(double));
}

static void ref_close(ref_state *r) {
    free(r->state); free(r->history); free(r->conv); free(r->y);
}

static void ref_step(ref_state *r,
                     const float *conv_w, const float *a_log,
                     const float *dt_bias, const float *norm,
                     const float *qkv, const float *alpha, const float *beta,
                     const float *gate, double *out) {
    /* K=4 causal depthwise convolution with a carried 3-row history. */
    for (uint32_t c = 0; c < CONV_DIM; c++) {
        const double raw = (double)qkv[c];
        double acc = raw * (double)conv_w[c * CONV_K + 3u];
        for (uint32_t w = 0; w < HISTORY; w++) {
            acc += r->history[(size_t)w * CONV_DIM + c] *
                   (double)conv_w[c * CONV_K + w];
        }
        for (uint32_t w = 0; w + 1u < HISTORY; w++) {
            r->history[(size_t)w * CONV_DIM + c] =
                r->history[(size_t)(w + 1u) * CONV_DIM + c];
        }
        r->history[(size_t)(HISTORY - 1u) * CONV_DIM + c] = raw;
        r->conv[c] = ref_silu(acc);
    }

    /* l2 norm with the epsilon on the SUM of squares; the query alone then
     * takes head_dim^-0.5. */
    for (uint32_t block = 0; block < 2u * KEY_HEADS; block++) {
        double sumsq = 0.0;
        for (uint32_t i = 0; i < D; i++) {
            const double v = r->conv[block * D + i];
            sumsq += v * v;
        }
        const double post = (block < KEY_HEADS) ? 1.0 / sqrt((double)D) : 1.0;
        const double scale = post / sqrt(sumsq + (double)QK_NORM_EPS);
        for (uint32_t i = 0; i < D; i++) r->conv[block * D + i] *= scale;
    }

    for (uint32_t h = 0; h < VALUE_HEADS; h++) {
        /* TILED order, what a converted GGUF carries. */
        const uint32_t kh = h % KEY_HEADS;
        const double *q = r->conv + (size_t)kh * D;
        const double *k = r->conv + KEY_DIM + (size_t)kh * D;
        const double *v = r->conv + 2u * KEY_DIM + (size_t)h * D;
        /* ssm_a IS ALREADY -exp(A_log) in this checkpoint, so the reference
         * multiplies by it rather than exponentiating it again:
         *     gate  = softplus(alpha + dt_bias) * ssm_a
         *     decay = exp(gate)
         * The old form repeated the kernel's own mistake, which is why this
         * reference passed while the kernel was wrong -- and why the fixture
         * below now feeds LARGE-magnitude negative values, where the two forms
         * are not merely different but opposite: at ssm_a = -78 the correct
         * decay is ~0 and the old one is ~1. */
        const double g = exp((double)a_log[h] *
                             ref_softplus((double)alpha[h] + (double)dt_bias[h]));
        const double b = ref_sigmoid((double)beta[h]);
        for (uint32_t value = 0; value < D; value++) {
            double *row = r->state + ((size_t)h * D + value) * D;
            double sk = 0.0;
            for (uint32_t i = 0; i < D; i++) {
                row[i] *= g;
                sk += row[i] * k[i];
            }
            const double delta = ((double)v[value] - sk) * b;
            double y = 0.0;
            for (uint32_t i = 0; i < D; i++) {
                row[i] += k[i] * delta;
                y += row[i] * q[i];
            }
            r->y[(size_t)h * D + value] = y;
        }
        double sumsq = 0.0;
        for (uint32_t value = 0; value < D; value++) {
            const double y = r->y[(size_t)h * D + value];
            sumsq += y * y;
        }
        const double inv = 1.0 / sqrt(sumsq / (double)D + (double)NORM_EPS);
        for (uint32_t value = 0; value < D; value++) {
            const size_t i = (size_t)h * D + value;
            out[i] = r->y[i] * inv * (double)norm[value] *
                     ref_sigmoid((double)gate[i]);
        }
    }
}

/* ------------------------------------------------------------ comparison */

typedef struct { double max_abs; double rms; double rel; size_t worst; } band;

static band measure(const float *got, const double *want, size_t count) {
    band b = {0.0, 0.0, 0.0, 0};
    double sumsq = 0.0;
    for (size_t i = 0; i < count; i++) {
        if (!isfinite(got[i])) {
            fprintf(stderr, "test_qwen4exp_gdn_ref: element %zu is %g\n", i, got[i]);
            exit(1);
        }
        const double diff = fabs((double)got[i] - want[i]);
        if (diff > b.max_abs) { b.max_abs = diff; b.worst = i; }
        sumsq += want[i] * want[i];
    }
    b.rms = sqrt(sumsq / (double)count);
    b.rel = (b.rms > 0.0) ? b.max_abs / b.rms : 0.0;
    return b;
}

static int g_failures;

static void report(const char *what, uint32_t tokens, band b) {
    const bool ok = b.rel <= BAND;
    if (b.rms < 1e-6) {
        fprintf(stderr, "  %-34s len %4u  REFERENCE DEGENERATE (rms %.3g)\n",
                what, tokens, b.rms);
        g_failures++;
        return;
    }
    printf("  %-34s len %4u  rms %.4g  max abs %.4g  max rel %.4g  %s\n",
           what, tokens, b.rms, b.max_abs, b.rel, ok ? "PASS" : "FAIL");
    if (!ok) {
        fprintf(stderr, "    worst element %zu\n", b.worst);
        g_failures++;
    }
}

int main(void) {
    uint8_t *model = mmap(NULL, MODEL_BYTES, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANON, -1, 0);
    require(model != MAP_FAILED, "model mapping");
    memset(model, 0, MODEL_BYTES);

    float *conv_w  = (float *)(model + CONV_OFFSET);
    float *a_log   = (float *)(model + A_LOG_OFFSET);
    float *dt_bias = (float *)(model + DT_BIAS_OFFSET);
    float *norm    = (float *)(model + NORM_OFFSET);
    for (size_t i = 0; i < (size_t)CONV_DIM * CONV_K; i++)
        conv_w[i] = (float)(0.5 * sample(0xA1000000ull + i));
    for (uint32_t h = 0; h < VALUE_HEADS; h++) {
        /* The artifact's range: blk.0 runs -157.985 to -0.0279.  A fixture
         * clustered near -3 cannot separate the two formulas, because
         * exp(-3) = 0.05 keeps the wrong decay in a plausible band; the large
         * entries are where the wrong form saturates to 1 and stops decaying. */
        a_log[h] = -(0.05f + 3.0f * (float)h);
        dt_bias[h] = (float)(0.5 * sample(0xA2000000ull + h));
    }
    for (uint32_t d = 0; d < D; d++)
        norm[d] = 1.0f + (float)(0.1 * sample(0xA3000000ull + d));

    require(ds4_gpu_init(), "GPU initialisation");
    require(ds4_gpu_set_model_map(model, MODEL_BYTES), "model map registration");

    const size_t qkv_values  = (size_t)MAX_TOKENS * CONV_DIM;
    const size_t head_values = (size_t)MAX_TOKENS * VALUE_HEADS;
    const size_t out_values  = (size_t)MAX_TOKENS * VALUE_DIM;
    float *qkv   = xalloc(qkv_values * sizeof(float));
    float *alpha = xalloc(head_values * sizeof(float));
    float *beta  = xalloc(head_values * sizeof(float));
    float *gate  = xalloc(out_values * sizeof(float));
    float *got   = xalloc(out_values * sizeof(float));
    float *got_state = xalloc((size_t)STATE_ELEMENTS * sizeof(float));
    float *got_conv  = xalloc((size_t)HISTORY * CONV_DIM * sizeof(float));
    double *want = xalloc(out_values * sizeof(double));

    for (size_t i = 0; i < qkv_values; i++)
        qkv[i] = (float)(0.8 * sample(0xB1000000ull + i));
    for (size_t i = 0; i < head_values; i++) {
        alpha[i] = (float)(2.0 * sample(0xB2000000ull + i));
        beta[i]  = (float)(2.0 * sample(0xB3000000ull + i));
    }
    for (size_t i = 0; i < out_values; i++)
        gate[i] = (float)(1.5 * sample(0xB4000000ull + i));

    ds4_gpu_tensor *t_qkv = ds4_gpu_tensor_alloc(qkv_values * sizeof(float));
    ds4_gpu_tensor *t_alpha = ds4_gpu_tensor_alloc(head_values * sizeof(float));
    ds4_gpu_tensor *t_beta = ds4_gpu_tensor_alloc(head_values * sizeof(float));
    ds4_gpu_tensor *t_gate = ds4_gpu_tensor_alloc(out_values * sizeof(float));
    ds4_gpu_tensor *t_out = ds4_gpu_tensor_alloc(out_values * sizeof(float));
    ds4_gpu_tensor *t_conv_state =
        ds4_gpu_tensor_alloc((size_t)HISTORY * CONV_DIM * sizeof(float));
    ds4_gpu_tensor *t_state =
        ds4_gpu_tensor_alloc((size_t)STATE_ELEMENTS * sizeof(float));
    require(t_qkv && t_alpha && t_beta && t_gate && t_out && t_conv_state &&
            t_state, "tensor allocation");

    const ds4_gpu_qwen4exp_slab conv_slab = slab_at(model, CONV_OFFSET);
    const ds4_gpu_qwen4exp_slab a_log_slab = slab_at(model, A_LOG_OFFSET);
    const ds4_gpu_qwen4exp_slab dt_slab = slab_at(model, DT_BIAS_OFFSET);
    const ds4_gpu_qwen4exp_slab norm_slab = slab_at(model, NORM_OFFSET);

    static const uint32_t lengths[] = {1u, 6u, 8u, 16u, 33u, 64u, 65u, 128u, 200u};
    printf("Qwen4-Exp GDN kernel vs double reference "
           "(band: max rel %.1g)\n", BAND);

    for (size_t li = 0; li < sizeof(lengths) / sizeof(lengths[0]); li++) {
        const uint32_t tokens = lengths[li];

        ref_state r;
        ref_open(&r);
        for (uint32_t t = 0; t < tokens; t++) {
            ref_step(&r, conv_w, a_log, dt_bias, norm,
                     qkv + (size_t)t * CONV_DIM,
                     alpha + (size_t)t * VALUE_HEADS,
                     beta + (size_t)t * VALUE_HEADS,
                     gate + (size_t)t * VALUE_DIM,
                     want + (size_t)t * VALUE_DIM);
        }

        require(ds4_gpu_tensor_fill_f32(t_conv_state, 0.0f,
                                        (uint64_t)HISTORY * CONV_DIM),
                "convolution state clear");
        require(ds4_gpu_tensor_fill_f32(t_state, 0.0f, (uint64_t)STATE_ELEMENTS),
                "recurrent state clear");
        require(ds4_gpu_tensor_write(t_qkv, 0, qkv,
                (size_t)tokens * CONV_DIM * sizeof(float)), "qkv write");
        require(ds4_gpu_tensor_write(t_alpha, 0, alpha,
                (size_t)tokens * VALUE_HEADS * sizeof(float)), "alpha write");
        require(ds4_gpu_tensor_write(t_beta, 0, beta,
                (size_t)tokens * VALUE_HEADS * sizeof(float)), "beta write");
        require(ds4_gpu_tensor_write(t_gate, 0, gate,
                (size_t)tokens * VALUE_DIM * sizeof(float)), "gate write");

        const int rc = (tokens > 1u)
            ? ds4_gpu_qwen4exp_gdn_prefill(
                  t_out, t_conv_state, t_state, NULL, NULL, 0u,
                  t_qkv, t_alpha, t_beta, t_gate,
                  &conv_slab, &a_log_slab, &dt_slab, &norm_slab,
                  KEY_HEADS, VALUE_HEADS, tokens,
                  DS4_QWEN4EXP_GDN_HEADS_TILED, QK_NORM_EPS, NORM_EPS)
            : ds4_gpu_qwen4exp_gdn_decode(
                  t_out, t_conv_state, t_state, t_qkv, t_alpha, t_beta, t_gate,
                  &conv_slab, &a_log_slab, &dt_slab, &norm_slab,
                  KEY_HEADS, VALUE_HEADS, 1u,
                  DS4_QWEN4EXP_GDN_HEADS_TILED, QK_NORM_EPS, NORM_EPS);
        require(rc, "GDN kernel");

        require(ds4_gpu_tensor_read(t_out, 0, got,
                (size_t)tokens * VALUE_DIM * sizeof(float)), "output read");
        require(ds4_gpu_tensor_read(t_state, 0, got_state,
                (size_t)STATE_ELEMENTS * sizeof(float)), "state read");
        require(ds4_gpu_tensor_read(t_conv_state, 0, got_conv,
                (size_t)HISTORY * CONV_DIM * sizeof(float)), "conv state read");

        report("output", tokens, measure(got, want, (size_t)tokens * VALUE_DIM));
        report("recurrent state", tokens,
               measure(got_state, r.state, (size_t)STATE_ELEMENTS));
        report("convolution carry", tokens,
               measure(got_conv, r.history, (size_t)HISTORY * CONV_DIM));
        ref_close(&r);
    }

    ds4_gpu_tensor_free(t_state);
    ds4_gpu_tensor_free(t_conv_state);
    ds4_gpu_tensor_free(t_out);
    ds4_gpu_tensor_free(t_gate);
    ds4_gpu_tensor_free(t_beta);
    ds4_gpu_tensor_free(t_alpha);
    ds4_gpu_tensor_free(t_qkv);
    free(want); free(got_conv); free(got_state);
    free(got); free(gate); free(beta); free(alpha); free(qkv);
    ds4_gpu_cleanup();
    munmap(model, MODEL_BYTES);

    if (g_failures) {
        printf("Qwen4-Exp GDN reference tests: FAIL (%d)\n", g_failures);
        return 1;
    }
    puts("Qwen4-Exp GDN reference tests: PASS");
    return 0;
}
