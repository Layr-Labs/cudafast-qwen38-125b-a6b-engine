/*
 * Qwen4exp gated delta net kernel tests.
 *
 * Follows tests/test_glm53_kda.c: link against ds4_metal.o (or ds4_cuda.o on
 * a CUDA host), drive the kernels with synthetic weights in an anonymous
 * mapping, and compare against a CPU reference written from the MLX
 * semantics.  The reference accumulates in double, so the band it reports is
 * the kernel's own error and not a shared rounding order.
 *
 * Checks, in order:
 *   1. output and final recurrent state against the reference, at sequence
 *      lengths 1, 7, 64 and 1024, in the tiled head order a converted GGUF
 *      carries;
 *   2. head layout: the same layer built in grouped order and permuted into
 *      tiled order exactly as llama.cpp's converter permutes it must give
 *      the same answer, bit for bit, once the value heads are unpermuted;
 *   3. two cases that separate the kernel from its near misses -- queries and
 *      keys small enough that the RMS epsilon dominates the sum of squares,
 *      and decays far below KDA's exp(-5) gate floor;
 *   4. chunk invariance, bit exact: 1024 tokens in one call equals 1016 + 8
 *      equals sixteen calls of 64;
 *   5. a decode step continuing a carried state, bit exact against the last
 *      step of the equivalent prefill, single row and two rows;
 *   6. determinism: the same prefill twice, bit exact;
 *   7. gate staging: the recurrence's staged per-(token, head) decay and
 *      write strength against the in-loop recomputation they replace, bit
 *      exact, at the widths the staged path runs at.
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

enum {
    D = 128,
    KEY_HEADS = 16,
    VALUE_HEADS = 48,
    REPEATS = VALUE_HEADS / KEY_HEADS,
    KEY_DIM = KEY_HEADS * D,
    VALUE_DIM = VALUE_HEADS * D,
    CONV_DIM = 2 * KEY_DIM + VALUE_DIM,
    HISTORY = 3,
    STATE_ELEMENTS = VALUE_HEADS * D * D,
    MAX_TOKENS = 1024,

    CONV_BYTES = CONV_DIM * 4 * 4,
    HEAD_BYTES = 256,

    GROUPED_CONV_OFFSET = 0,
    GROUPED_A_LOG_OFFSET = GROUPED_CONV_OFFSET + CONV_BYTES,
    GROUPED_DT_BIAS_OFFSET = GROUPED_A_LOG_OFFSET + HEAD_BYTES,
    TILED_CONV_OFFSET = GROUPED_DT_BIAS_OFFSET + HEAD_BYTES,
    TILED_A_LOG_OFFSET = TILED_CONV_OFFSET + CONV_BYTES,
    TILED_DT_BIAS_OFFSET = TILED_A_LOG_OFFSET + HEAD_BYTES,
    FAST_A_LOG_OFFSET = TILED_DT_BIAS_OFFSET + HEAD_BYTES,
    FAST_DT_BIAS_OFFSET = FAST_A_LOG_OFFSET + HEAD_BYTES,
    NORM_OFFSET = FAST_DT_BIAS_OFFSET + HEAD_BYTES,
    MODEL_BYTES = NORM_OFFSET + 4096,
};

static const float QK_NORM_EPS = 1e-6f;
static const float NORM_EPS = 1e-6f;

/*
 * The kernel lands three orders inside the design's 2e-3 GDN band, and a
 * 2e-3 assertion is too loose to be a test: it cannot tell the reference
 * apart from a kernel that divides the query and key sum of squares by D
 * before adding the epsilon instead of adding it to the sum, which is a
 * 1.8e-4 error.  Assert the accuracy the kernel actually has.
 */
static const float BAND = 2e-5f;

bool ds4_log_is_tty(FILE *fp) {
    (void)fp;
    return false;
}

/* The GDN entry points take one slab per weight, so the test builds them the
 * way the graph does.  Every slab here names the same single test mapping:
 * what is under test is the arithmetic, not the split-file resolution, which
 * tests/test_qwen4exp_graph.c covers with a two-shard model. */
static ds4_gpu_qwen4exp_slab gdn_slab(const void *map, uint64_t offset) {
    ds4_gpu_qwen4exp_slab slab;
    memset(&slab, 0, sizeof(slab));
    slab.map = map;
    slab.map_size = MODEL_BYTES;
    slab.offset = offset;
    return slab;
}

static void require_ok(int ok, const char *what) {
    if (!ok) {
        fprintf(stderr, "%s failed\n", what);
        exit(1);
    }
}

static void *require_alloc(size_t bytes, const char *what) {
    void *p = malloc(bytes);
    if (!p) {
        fprintf(stderr, "%s allocation of %zu bytes failed\n", what, bytes);
        exit(1);
    }
    return p;
}

/* Deterministic input generator: splitmix64 folded to [-1, 1). */
static double sample(uint64_t seed) {
    uint64_t z = seed + 0x9e3779b97f4a7c15ull;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    z ^= z >> 31;
    return (double)(z >> 11) * (2.0 / 9007199254740992.0) - 1.0;
}

/*
 * llama.cpp's converter reorders value heads from the grouped order the
 * reference model stores into tiled order (conversion/qwen.py,
 * `_LinearAttentionVReorderBase._reorder_v_heads`): it views the value axis
 * as [n_key_head][repeats][head_dim] and transposes the first two, so tiled
 * slot j holds grouped head (j % n_key_head) * repeats + j / n_key_head.
 */
static uint32_t grouped_source_head(uint32_t tiled_head) {
    return (tiled_head % KEY_HEADS) * REPEATS + tiled_head / KEY_HEADS;
}

/* --------------------------------------------------------------- weights */

typedef struct {
    const float *conv;      /* [CONV_DIM][4] */
    const float *a_log;     /* [VALUE_HEADS] */
    const float *dt_bias;   /* [VALUE_HEADS] */
    const float *norm;      /* [D] */
    uint64_t     conv_offset;
    uint64_t     a_log_offset;
    uint64_t     dt_bias_offset;
    uint64_t     norm_offset;
    uint32_t     layout;
} weight_set;

static weight_set g_grouped;
static weight_set g_tiled;
static weight_set g_fast_decay;

static void build_weights(uint8_t *model) {
    float *conv = (float *)(model + GROUPED_CONV_OFFSET);
    float *a_log = (float *)(model + GROUPED_A_LOG_OFFSET);
    float *dt_bias = (float *)(model + GROUPED_DT_BIAS_OFFSET);
    float *norm = (float *)(model + NORM_OFFSET);
    for (uint32_t c = 0; c < CONV_DIM; c++) {
        for (uint32_t w = 0; w < 4u; w++) {
            conv[c * 4u + w] =
                (float)(0.5 * sample(0x11000000ull + c * 4u + w));
        }
    }
    for (uint32_t h = 0; h < VALUE_HEADS; h++) {
        /* Decays near one: the long-memory regime, where a chunk boundary or
         * a lost fp32 bit shows up most clearly. */
        /* Negative and spread wide, as the artifact is: blk.0 runs -157.985
         * to -0.0279.  Values clustered near -3 cannot separate ssm_a used
         * as-is from ssm_a exponentiated again. */
        a_log[h] = -(0.05f + 3.0f * (float)h);
        dt_bias[h] = (float)(0.5 * sample(0x22000000ull + h));
    }
    for (uint32_t d = 0; d < D; d++)
        norm[d] = 1.0f + (float)(0.1 * sample(0x33000000ull + d));

    /* The same layer as a converted GGUF stores it: value heads tiled, in the
     * convolution's value channels and in the two per-head parameters. */
    float *tiled_conv = (float *)(model + TILED_CONV_OFFSET);
    float *tiled_a_log = (float *)(model + TILED_A_LOG_OFFSET);
    float *tiled_dt_bias = (float *)(model + TILED_DT_BIAS_OFFSET);
    memcpy(tiled_conv, conv, (size_t)2 * KEY_DIM * 4u * sizeof(float));
    for (uint32_t h = 0; h < VALUE_HEADS; h++) {
        const uint32_t src = grouped_source_head(h);
        memcpy(tiled_conv + ((size_t)2 * KEY_DIM + (size_t)h * D) * 4u,
               conv + ((size_t)2 * KEY_DIM + (size_t)src * D) * 4u,
               (size_t)D * 4u * sizeof(float));
        tiled_a_log[h] = a_log[src];
        tiled_dt_bias[h] = dt_bias[src];
    }

    /* Decays far below KDA's exp(-5) floor: exp(a_log) * softplus(alpha +
     * dt_bias) reaches about 12, so g is around 6e-6 where a floored kernel
     * would hold 6.7e-3. */
    float *fast_a_log = (float *)(model + FAST_A_LOG_OFFSET);
    float *fast_dt_bias = (float *)(model + FAST_DT_BIAS_OFFSET);
    for (uint32_t h = 0; h < VALUE_HEADS; h++) {
        /* -exp(1.1).  The tensor holds -exp(A_log), so the A_log of 1.1 this
         * case used to carry becomes its negated exponential; a positive value
         * here would make the decay exceed one and grow the state. */
        fast_a_log[h] = -3.0041660f;
        fast_dt_bias[h] = 2.0f;
    }

    g_grouped = (weight_set){
        conv, a_log, dt_bias, norm,
        GROUPED_CONV_OFFSET, GROUPED_A_LOG_OFFSET, GROUPED_DT_BIAS_OFFSET,
        NORM_OFFSET, DS4_QWEN4EXP_GDN_HEADS_GROUPED };
    g_tiled = (weight_set){
        tiled_conv, tiled_a_log, tiled_dt_bias, norm,
        TILED_CONV_OFFSET, TILED_A_LOG_OFFSET, TILED_DT_BIAS_OFFSET,
        NORM_OFFSET, DS4_QWEN4EXP_GDN_HEADS_TILED };
    g_fast_decay = (weight_set){
        tiled_conv, fast_a_log, fast_dt_bias, norm,
        TILED_CONV_OFFSET, FAST_A_LOG_OFFSET, FAST_DT_BIAS_OFFSET,
        NORM_OFFSET, DS4_QWEN4EXP_GDN_HEADS_TILED };
}

/* -------------------------------------------------------------- reference */

typedef struct {
    double *state;      /* [VALUE_HEADS][D][D] */
    double *history;    /* [HISTORY][CONV_DIM] */
    double *conv;       /* [CONV_DIM] scratch */
    double *y;          /* [VALUE_DIM] scratch */
} reference;

static void reference_init(reference *ref) {
    ref->state = require_alloc(STATE_ELEMENTS * sizeof(double), "reference state");
    ref->history = require_alloc(HISTORY * CONV_DIM * sizeof(double), "reference history");
    ref->conv = require_alloc(CONV_DIM * sizeof(double), "reference conv");
    ref->y = require_alloc(VALUE_DIM * sizeof(double), "reference y");
    memset(ref->state, 0, STATE_ELEMENTS * sizeof(double));
    memset(ref->history, 0, HISTORY * CONV_DIM * sizeof(double));
}

static void reference_free(reference *ref) {
    free(ref->state);
    free(ref->history);
    free(ref->conv);
    free(ref->y);
}

static double reference_silu(double x) {
    return x / (1.0 + exp(-x));
}

static double reference_sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

static double reference_softplus(double x) {
    const double m = x > 0.0 ? x : 0.0;
    return m + log1p(exp(-fabs(x)));
}

/* One token of the layer: convolution, activation, query and key norm, the
 * delta rule, and the gated output norm. */
static void reference_step(
        reference        *ref,
        const weight_set *ws,
        const float      *qkv,
        const float      *alpha,
        const float      *beta,
        const float      *output_gate,
        float            *out) {
    for (uint32_t c = 0; c < CONV_DIM; c++) {
        const double raw = (double)qkv[c];
        double acc = ref->history[c] * (double)ws->conv[c * 4u + 0u];
        acc += ref->history[CONV_DIM + c] * (double)ws->conv[c * 4u + 1u];
        acc += ref->history[2u * CONV_DIM + c] *
               (double)ws->conv[c * 4u + 2u];
        acc += raw * (double)ws->conv[c * 4u + 3u];
        ref->history[c] = ref->history[CONV_DIM + c];
        ref->history[CONV_DIM + c] = ref->history[2u * CONV_DIM + c];
        ref->history[2u * CONV_DIM + c] = raw;
        ref->conv[c] = reference_silu(acc);
    }
    /* The query and key heads are l2 normalised the way the original model
     * does it -- `x * rsqrt(sum(x^2) + eps)`, the epsilon on the SUM -- and
     * then the query alone is scaled by `head_dim ** -0.5`.  Dividing the sum
     * by D first is the same expression with an epsilon 128 times too large;
     * on a small row that moves the key by more than a factor of two, which is
     * the near miss the tight band and case 3a below catch. */
    for (uint32_t block = 0; block < 2u * KEY_HEADS; block++) {
        double sumsq = 0.0;
        for (uint32_t i = 0; i < D; i++) {
            const double value = ref->conv[block * D + i];
            sumsq += value * value;
        }
        const double post = block < KEY_HEADS
            ? 1.0 / sqrt((double)D)
            : 1.0;
        const double scale =
            post / sqrt(sumsq + (double)QK_NORM_EPS);
        for (uint32_t i = 0; i < D; i++) ref->conv[block * D + i] *= scale;
    }

    for (uint32_t head = 0; head < VALUE_HEADS; head++) {
        const uint32_t key_head =
            ws->layout == DS4_QWEN4EXP_GDN_HEADS_TILED
                ? head % KEY_HEADS
                : head / REPEATS;
        const double *q = ref->conv + key_head * D;
        const double *k = ref->conv + KEY_DIM + key_head * D;
        const double *v = ref->conv + 2u * KEY_DIM + head * D;
        /* ssm_a is ALREADY -exp(A_log) in this checkpoint, so it multiplies
         * the softplus rather than being exponentiated again.  No lower bound
         * either: KDA floors its decay at exp(-5) and this family does not. */
        const double g = exp((double)ws->a_log[head] *
            reference_softplus((double)alpha[head] +
                               (double)ws->dt_bias[head]));
        const double b = reference_sigmoid((double)beta[head]);
        for (uint32_t value = 0; value < D; value++) {
            double *row = ref->state + ((size_t)head * D + value) * D;
            double kv = 0.0;
            for (uint32_t i = 0; i < D; i++) {
                row[i] *= g;
                kv += row[i] * k[i];
            }
            const double delta = (v[value] - kv) * b;
            double y = 0.0;
            for (uint32_t i = 0; i < D; i++) {
                row[i] += k[i] * delta;
                y += row[i] * q[i];
            }
            ref->y[head * D + value] = y;
        }
        double sumsq = 0.0;
        for (uint32_t value = 0; value < D; value++) {
            const double y = ref->y[head * D + value];
            sumsq += y * y;
        }
        const double scale = 1.0 / sqrt(sumsq / (double)D + (double)NORM_EPS);
        for (uint32_t value = 0; value < D; value++) {
            const uint32_t index = head * D + value;
            out[index] = (float)(ref->y[index] * scale *
                (double)ws->norm[value] *
                reference_sigmoid((double)output_gate[index]));
        }
    }
}

/* ----------------------------------------------------------- comparators */

static void require_band(
        const char  *what,
        const float *actual,
        const float *expected,
        size_t       count,
        float        max_abs_allowed) {
    double max_abs = 0.0, dot = 0.0, na = 0.0, nb = 0.0;
    size_t worst = 0;
    for (size_t i = 0; i < count; i++) {
        if (!isfinite(actual[i])) {
            fprintf(stderr, "%s: element %zu is not finite (%g)\n",
                    what, i, actual[i]);
            exit(1);
        }
        const double diff = fabs((double)actual[i] - (double)expected[i]);
        if (diff > max_abs) {
            max_abs = diff;
            worst = i;
        }
        dot += (double)actual[i] * (double)expected[i];
        na += (double)actual[i] * (double)actual[i];
        nb += (double)expected[i] * (double)expected[i];
    }
    const double rms = sqrt(nb / (double)count);
    /* A band comparison of two empty buffers passes for the wrong reason. */
    if (rms < 1e-3) {
        fprintf(stderr, "%s: the reference is degenerate (rms %.6g)\n",
                what, rms);
        exit(1);
    }
    const double cosine = dot / (sqrt(na) * sqrt(nb));
    printf("  %-46s rms %.3g  max abs %.3g  cosine %.9f\n",
           what, rms, max_abs, cosine);
    if (max_abs > (double)max_abs_allowed || cosine < 0.9999) {
        fprintf(stderr,
                "%s: outside the band (max abs %.6g at %zu, got %.9g vs "
                "%.9g; cosine %.9f)\n",
                what, max_abs, worst, actual[worst], expected[worst], cosine);
        exit(1);
    }
}

static void require_identical(
        const char  *what,
        const float *actual,
        const float *expected,
        size_t       count) {
    for (size_t i = 0; i < count; i++) {
        if (memcmp(&actual[i], &expected[i], sizeof(float)) != 0) {
            fprintf(stderr, "%s: element %zu differs (%.9g vs %.9g)\n",
                    what, i, actual[i], expected[i]);
            exit(1);
        }
    }
    printf("  %-46s bit exact over %zu values\n", what, count);
}

/* ---------------------------------------------------------------- harness */

typedef struct {
    ds4_gpu_tensor *qkv;
    ds4_gpu_tensor *alpha;
    ds4_gpu_tensor *beta;
    ds4_gpu_tensor *output_gate;
    ds4_gpu_tensor *out;
    ds4_gpu_tensor *conv_state;
    ds4_gpu_tensor *state;
} gpu_buffers;

static void buffers_alloc(gpu_buffers *b, uint32_t rows, uint32_t tokens) {
    const uint64_t slots = (uint64_t)rows * tokens;
    b->qkv = ds4_gpu_tensor_alloc(slots * CONV_DIM * sizeof(float));
    b->alpha = ds4_gpu_tensor_alloc(slots * VALUE_HEADS * sizeof(float));
    b->beta = ds4_gpu_tensor_alloc(slots * VALUE_HEADS * sizeof(float));
    b->output_gate = ds4_gpu_tensor_alloc(slots * VALUE_DIM * sizeof(float));
    b->out = ds4_gpu_tensor_alloc(slots * VALUE_DIM * sizeof(float));
    b->conv_state = ds4_gpu_tensor_alloc(
        (uint64_t)rows * HISTORY * CONV_DIM * sizeof(float));
    b->state = ds4_gpu_tensor_alloc(
        (uint64_t)rows * STATE_ELEMENTS * sizeof(float));
    require_ok(b->qkv && b->alpha && b->beta && b->output_gate && b->out &&
               b->conv_state && b->state, "tensor allocation");
}

static void buffers_free(gpu_buffers *b) {
    ds4_gpu_tensor_free(b->state);
    ds4_gpu_tensor_free(b->conv_state);
    ds4_gpu_tensor_free(b->out);
    ds4_gpu_tensor_free(b->output_gate);
    ds4_gpu_tensor_free(b->beta);
    ds4_gpu_tensor_free(b->alpha);
    ds4_gpu_tensor_free(b->qkv);
}

static void buffers_clear_state(gpu_buffers *b, uint32_t rows) {
    require_ok(ds4_gpu_tensor_fill_f32(b->conv_state, 0.0f,
                                       (uint64_t)rows * HISTORY * CONV_DIM),
               "convolution state clear");
    require_ok(ds4_gpu_tensor_fill_f32(b->state, 0.0f,
                                       (uint64_t)rows * STATE_ELEMENTS),
               "recurrent state clear");
}

/* The kernels rewrite `qkv` in place, so every call reloads it from the
 * pristine host copy. */
static void run_prefill(
        gpu_buffers      *b,
        const void       *model,
        const weight_set *ws,
        const float      *qkv,
        const float      *alpha,
        const float      *beta,
        const float      *output_gate,
        uint32_t          first,
        uint32_t          tokens,
        float            *out) {
    require_ok(ds4_gpu_tensor_write(b->qkv, 0,
        qkv + (size_t)first * CONV_DIM,
        (size_t)tokens * CONV_DIM * sizeof(float)), "qkv write");
    require_ok(ds4_gpu_tensor_write(b->alpha, 0,
        alpha + (size_t)first * VALUE_HEADS,
        (size_t)tokens * VALUE_HEADS * sizeof(float)), "alpha write");
    require_ok(ds4_gpu_tensor_write(b->beta, 0,
        beta + (size_t)first * VALUE_HEADS,
        (size_t)tokens * VALUE_HEADS * sizeof(float)), "beta write");
    require_ok(ds4_gpu_tensor_write(b->output_gate, 0,
        output_gate + (size_t)first * VALUE_DIM,
        (size_t)tokens * VALUE_DIM * sizeof(float)), "output gate write");
    const ds4_gpu_qwen4exp_slab conv_slab = gdn_slab(model, ws->conv_offset);
    const ds4_gpu_qwen4exp_slab a_log_slab = gdn_slab(model, ws->a_log_offset);
    const ds4_gpu_qwen4exp_slab dt_bias_slab = gdn_slab(model, ws->dt_bias_offset);
    const ds4_gpu_qwen4exp_slab norm_slab = gdn_slab(model, ws->norm_offset);
    require_ok(ds4_gpu_qwen4exp_gdn_prefill(
        b->out, b->conv_state, b->state, NULL, NULL, 0u,
        b->qkv, b->alpha, b->beta,
        b->output_gate, &conv_slab, &a_log_slab, &dt_bias_slab, &norm_slab,
        KEY_HEADS, VALUE_HEADS, tokens, ws->layout, QK_NORM_EPS, NORM_EPS),
        "GDN prefill");
    require_ok(ds4_gpu_tensor_read(b->out, 0, out,
        (size_t)tokens * VALUE_DIM * sizeof(float)), "output read");
}

static void run_reference(
        const weight_set *ws,
        const float      *qkv,
        const float      *alpha,
        const float      *beta,
        const float      *output_gate,
        uint32_t          tokens,
        float            *out,
        float            *final_state) {
    reference ref;
    reference_init(&ref);
    for (uint32_t t = 0; t < tokens; t++) {
        reference_step(&ref, ws, qkv + (size_t)t * CONV_DIM,
                       alpha + (size_t)t * VALUE_HEADS,
                       beta + (size_t)t * VALUE_HEADS,
                       output_gate + (size_t)t * VALUE_DIM,
                       out + (size_t)t * VALUE_DIM);
    }
    if (final_state) {
        for (size_t i = 0; i < STATE_ELEMENTS; i++)
            final_state[i] = (float)ref.state[i];
    }
    reference_free(&ref);
}

int main(void) {
    uint8_t *model = mmap(NULL, MODEL_BYTES, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANON, -1, 0);
    if (model == MAP_FAILED) {
        perror("mmap");
        return 1;
    }
    memset(model, 0, MODEL_BYTES);
    build_weights(model);

    require_ok(ds4_gpu_init(), "GPU initialization");
    require_ok(ds4_gpu_set_model_map(model, MODEL_BYTES),
               "model map registration");

    const size_t qkv_values = (size_t)MAX_TOKENS * CONV_DIM;
    const size_t gate_values = (size_t)MAX_TOKENS * VALUE_HEADS;
    const size_t out_values = (size_t)MAX_TOKENS * VALUE_DIM;
    float *qkv = require_alloc(qkv_values * sizeof(float), "qkv");
    float *alpha = require_alloc(gate_values * sizeof(float), "alpha");
    float *beta = require_alloc(gate_values * sizeof(float), "beta");
    float *output_gate = require_alloc(out_values * sizeof(float), "gate");
    float *actual = require_alloc(out_values * sizeof(float), "actual");
    float *other = require_alloc(out_values * sizeof(float), "other");
    float *expected = require_alloc(out_values * sizeof(float), "expected");
    float *state_actual =
        require_alloc(2u * STATE_ELEMENTS * sizeof(float), "state actual");
    float *state_other =
        require_alloc(2u * STATE_ELEMENTS * sizeof(float), "state other");
    float *state_expected =
        require_alloc(STATE_ELEMENTS * sizeof(float), "state expected");

    /* Inputs in the tiled order a converted GGUF produces: every per-value-
     * head activation comes out of a reordered projection. */
    for (size_t i = 0; i < qkv_values; i++)
        qkv[i] = (float)(0.8 * sample(0x1000000ull + i));
    for (size_t i = 0; i < gate_values; i++) {
        alpha[i] = (float)(2.0 * sample(0x4000000ull + i));
        beta[i] = (float)(2.0 * sample(0x5000000ull + i));
    }
    for (size_t i = 0; i < out_values; i++)
        output_gate[i] = (float)(1.5 * sample(0x6000000ull + i));

    gpu_buffers big;
    buffers_alloc(&big, 1u, MAX_TOKENS);

    /* 1. Reference band at production shape, tiled heads. */
    const uint32_t lengths[] = {1u, 7u, 64u, MAX_TOKENS};
    for (uint32_t li = 0; li < sizeof(lengths) / sizeof(lengths[0]); li++) {
        const uint32_t tokens = lengths[li];
        char label[96];
        run_reference(&g_tiled, qkv, alpha, beta, output_gate, tokens,
                      expected, state_expected);
        buffers_clear_state(&big, 1u);
        run_prefill(&big, model, &g_tiled, qkv, alpha, beta, output_gate, 0u,
                    tokens, actual);
        require_ok(ds4_gpu_tensor_read(big.state, 0, state_actual,
            STATE_ELEMENTS * sizeof(float)), "state read");
        snprintf(label, sizeof(label), "prefill %u tokens vs reference",
                 tokens);
        require_band(label, actual, expected,
                     (size_t)tokens * VALUE_DIM, BAND);
        snprintf(label, sizeof(label), "final state after %u tokens", tokens);
        require_band(label, state_actual, state_expected, STATE_ELEMENTS,
                     BAND);
    }

    /* 2. Head layout.  Take the tiled run above as the reference, undo the
     *    converter's permutation on every per-value-head input, and run the
     *    same layer in grouped order: the answers must agree bit for bit
     *    once the value heads are put back.  This is the check that fails if
     *    the kernel pairs value head j with key head j / 3 on GGUF weights.
     */
    {
        const uint32_t tokens = 64u;
        float *grouped_qkv =
            require_alloc((size_t)tokens * CONV_DIM * sizeof(float),
                          "grouped qkv");
        float *grouped_alpha =
            require_alloc((size_t)tokens * VALUE_HEADS * sizeof(float),
                          "grouped alpha");
        float *grouped_beta =
            require_alloc((size_t)tokens * VALUE_HEADS * sizeof(float),
                          "grouped beta");
        float *grouped_gate =
            require_alloc((size_t)tokens * VALUE_DIM * sizeof(float),
                          "grouped gate");
        float *unpermuted =
            require_alloc((size_t)tokens * VALUE_DIM * sizeof(float),
                          "unpermuted output");
        for (uint32_t t = 0; t < tokens; t++) {
            memcpy(grouped_qkv + (size_t)t * CONV_DIM,
                   qkv + (size_t)t * CONV_DIM,
                   (size_t)2 * KEY_DIM * sizeof(float));
            for (uint32_t h = 0; h < VALUE_HEADS; h++) {
                const uint32_t src = grouped_source_head(h);
                memcpy(grouped_qkv + (size_t)t * CONV_DIM + 2 * KEY_DIM +
                           (size_t)src * D,
                       qkv + (size_t)t * CONV_DIM + 2 * KEY_DIM +
                           (size_t)h * D, D * sizeof(float));
                memcpy(grouped_gate + (size_t)t * VALUE_DIM + (size_t)src * D,
                       output_gate + (size_t)t * VALUE_DIM + (size_t)h * D,
                       D * sizeof(float));
                grouped_alpha[(size_t)t * VALUE_HEADS + src] =
                    alpha[(size_t)t * VALUE_HEADS + h];
                grouped_beta[(size_t)t * VALUE_HEADS + src] =
                    beta[(size_t)t * VALUE_HEADS + h];
            }
        }
        buffers_clear_state(&big, 1u);
        run_prefill(&big, model, &g_tiled, qkv, alpha, beta, output_gate, 0u,
                    tokens, actual);
        require_ok(ds4_gpu_tensor_read(big.state, 0, state_actual,
            STATE_ELEMENTS * sizeof(float)), "tiled state read");
        buffers_clear_state(&big, 1u);
        run_prefill(&big, model, &g_grouped, grouped_qkv, grouped_alpha,
                    grouped_beta, grouped_gate, 0u, tokens, other);
        require_ok(ds4_gpu_tensor_read(big.state, 0, state_other,
            STATE_ELEMENTS * sizeof(float)), "grouped state read");
        /* Undo the permutation the converter applies to `ssm_out`'s columns
         * by putting the grouped run's value heads back in tiled slots. */
        for (uint32_t t = 0; t < tokens; t++) {
            for (uint32_t h = 0; h < VALUE_HEADS; h++) {
                memcpy(unpermuted + (size_t)t * VALUE_DIM + (size_t)h * D,
                       other + (size_t)t * VALUE_DIM +
                           (size_t)grouped_source_head(h) * D,
                       D * sizeof(float));
            }
        }
        require_identical("grouped layout equals tiled layout", unpermuted,
                          actual, (size_t)tokens * VALUE_DIM);
        float *unpermuted_state =
            require_alloc(STATE_ELEMENTS * sizeof(float),
                          "unpermuted state");
        for (uint32_t h = 0; h < VALUE_HEADS; h++) {
            memcpy(unpermuted_state + (size_t)h * D * D,
                   state_other + (size_t)grouped_source_head(h) * D * D,
                   (size_t)D * D * sizeof(float));
        }
        require_identical("grouped layout state equals tiled",
                          unpermuted_state, state_actual, STATE_ELEMENTS);
        free(unpermuted_state);
        free(unpermuted);
        free(grouped_gate);
        free(grouped_beta);
        free(grouped_alpha);
        free(grouped_qkv);
    }

    /* 3a. Queries and keys small enough that the epsilon dominates the sum of
     *     squares.  With rows near 5e-4 the sum is about 3e-5 against a 1e-6
     *     epsilon term, so dividing that sum by D before adding the epsilon --
     *     the 128x-too-large form this kernel used to have -- changes the key
     *     scale by more than a factor of two and this case goes red. */
    {
        const uint32_t tokens = 64u;
        float *small = require_alloc(
            (size_t)tokens * CONV_DIM * sizeof(float), "small qkv");
        memcpy(small, qkv, (size_t)tokens * CONV_DIM * sizeof(float));
        for (uint32_t t = 0; t < tokens; t++) {
            for (uint32_t c = 0; c < 2u * KEY_DIM; c++)
                small[(size_t)t * CONV_DIM + c] *= 1.0e-3f;
        }
        run_reference(&g_tiled, small, alpha, beta, output_gate, tokens,
                      expected, state_expected);
        buffers_clear_state(&big, 1u);
        run_prefill(&big, model, &g_tiled, small, alpha, beta, output_gate,
                    0u, tokens, actual);
        require_ok(ds4_gpu_tensor_read(big.state, 0, state_actual,
            STATE_ELEMENTS * sizeof(float)), "small qk state read");
        require_band("small query and key rows vs reference", actual,
                     expected, (size_t)tokens * VALUE_DIM, BAND);
        require_band("small query and key rows final state", state_actual,
                     state_expected, STATE_ELEMENTS, BAND);
        free(small);
    }

    /* 3b. Decays far below KDA's exp(-5) gate floor. */
    {
        const uint32_t tokens = 64u;
        float *strong = require_alloc(
            (size_t)tokens * VALUE_HEADS * sizeof(float), "strong alpha");
        for (size_t i = 0; i < (size_t)tokens * VALUE_HEADS; i++)
            strong[i] = 2.0f + (float)fabs(sample(0x7000000ull + i));
        run_reference(&g_fast_decay, qkv, strong, beta, output_gate, tokens,
                      expected, state_expected);
        buffers_clear_state(&big, 1u);
        run_prefill(&big, model, &g_fast_decay, qkv, strong, beta,
                    output_gate, 0u, tokens, actual);
        require_ok(ds4_gpu_tensor_read(big.state, 0, state_actual,
            STATE_ELEMENTS * sizeof(float)), "fast decay state read");
        require_band("decay below the KDA floor vs reference", actual,
                     expected, (size_t)tokens * VALUE_DIM, BAND);
        require_band("decay below the KDA floor final state", state_actual,
                     state_expected, STATE_ELEMENTS, BAND);
        free(strong);
    }

    /* 4. Chunk invariance: the 1024-token result must not depend on how the
     *    sequence was cut. */
    buffers_clear_state(&big, 1u);
    run_prefill(&big, model, &g_tiled, qkv, alpha, beta, output_gate, 0u,
                MAX_TOKENS, actual);
    require_ok(ds4_gpu_tensor_read(big.state, 0, state_actual,
        STATE_ELEMENTS * sizeof(float)), "single chunk state read");

    buffers_clear_state(&big, 1u);
    run_prefill(&big, model, &g_tiled, qkv, alpha, beta, output_gate, 0u,
                1016u, other);
    run_prefill(&big, model, &g_tiled, qkv, alpha, beta, output_gate, 1016u,
                8u, other + (size_t)1016 * VALUE_DIM);
    require_ok(ds4_gpu_tensor_read(big.state, 0, state_other,
        STATE_ELEMENTS * sizeof(float)), "split chunk state read");
    require_identical("chunk invariance 1024 = 1016 + 8", other, actual,
                      out_values);
    require_identical("chunk invariance 1016 + 8 final state", state_other,
                      state_actual, STATE_ELEMENTS);

    buffers_clear_state(&big, 1u);
    for (uint32_t chunk = 0; chunk < 16u; chunk++) {
        run_prefill(&big, model, &g_tiled, qkv, alpha, beta, output_gate,
                    chunk * 64u, 64u,
                    other + (size_t)chunk * 64u * VALUE_DIM);
    }
    require_ok(ds4_gpu_tensor_read(big.state, 0, state_other,
        STATE_ELEMENTS * sizeof(float)), "sixteen chunk state read");
    require_identical("chunk invariance 1024 = 16 x 64", other, actual,
                      out_values);
    require_identical("chunk invariance 16 x 64 final state", state_other,
                      state_actual, STATE_ELEMENTS);

    /* SMALL CHUNKS.  The two cases above use 64 and 1016 + 8; nothing checked
     * a width the graph actually prefills at.  A session prefills in chunks of
     * whatever the caller asked for -- two, three, five -- and every one of
     * those must leave the same convolution history and recurrent state as one
     * whole call, or a prompt's meaning depends on how it was fed. */
    {
        float *conv_whole = require_alloc(
            HISTORY * CONV_DIM * sizeof(float), "whole-call convolution carry");
        float *conv_chunked = require_alloc(
            HISTORY * CONV_DIM * sizeof(float), "chunked convolution carry");
        /* The reference carry: the same 1024 tokens in ONE call. */
        buffers_clear_state(&big, 1u);
        run_prefill(&big, model, &g_tiled, qkv, alpha, beta, output_gate, 0u,
                    MAX_TOKENS, other);
        require_ok(ds4_gpu_tensor_read(big.conv_state, 0, conv_whole,
            HISTORY * CONV_DIM * sizeof(float)), "whole conv state read");

        static const uint32_t widths[] = { 1u, 2u, 3u, 4u, 5u, 7u };
        for (size_t w = 0; w < sizeof(widths) / sizeof(widths[0]); w++) {
            const uint32_t width = widths[w];
            char label[96];
            buffers_clear_state(&big, 1u);
            for (uint32_t at = 0; at < MAX_TOKENS; ) {
                const uint32_t take =
                    (MAX_TOKENS - at) < width ? (MAX_TOKENS - at) : width;
                run_prefill(&big, model, &g_tiled, qkv, alpha, beta,
                            output_gate, at, take,
                            other + (size_t)at * VALUE_DIM);
                at += take;
            }
            require_ok(ds4_gpu_tensor_read(big.state, 0, state_other,
                STATE_ELEMENTS * sizeof(float)), "small chunk state read");
            require_ok(ds4_gpu_tensor_read(big.conv_state, 0, conv_chunked,
                HISTORY * CONV_DIM * sizeof(float)),
                "small chunk conv state read");
            snprintf(label, sizeof(label),
                     "chunk invariance 1024 = %u-token chunks", width);
            require_identical(label, other, actual, out_values);
            snprintf(label, sizeof(label),
                     "chunk invariance %u-token chunks, final state", width);
            require_identical(label, state_other, state_actual,
                              STATE_ELEMENTS);
            /* THE CONVOLUTION CARRY, which the two cases above never read.
             * The causal window is HISTORY inputs wide, so a chunk NARROWER
             * than the window has to shift the old carry along and append its
             * own rows rather than replace the carry with what it has.  Widths
             * 1 and 2 are below the window and 3 is exactly it. */
            snprintf(label, sizeof(label),
                     "chunk invariance %u-token chunks, convolution carry",
                     width);
            require_identical(label, conv_chunked, conv_whole,
                              HISTORY * CONV_DIM);
        }
        free(conv_whole);
        free(conv_chunked);
    }

    /* 5. Determinism: the same call twice. */
    buffers_clear_state(&big, 1u);
    run_prefill(&big, model, &g_tiled, qkv, alpha, beta, output_gate, 0u,
                MAX_TOKENS, other);
    require_identical("determinism across runs", other, actual, out_values);

    /* 6. A decode step equals the last step of the equivalent prefill. */
    buffers_clear_state(&big, 1u);
    run_prefill(&big, model, &g_tiled, qkv, alpha, beta, output_gate, 0u, 64u,
                actual);
    require_ok(ds4_gpu_tensor_read(big.state, 0, state_actual,
        STATE_ELEMENTS * sizeof(float)), "prefill 64 state read");

    float *conv_carry = require_alloc(
        HISTORY * CONV_DIM * sizeof(float), "convolution carry");
    float *state_carry =
        require_alloc(STATE_ELEMENTS * sizeof(float), "state carry");
    buffers_clear_state(&big, 1u);
    run_prefill(&big, model, &g_tiled, qkv, alpha, beta, output_gate, 0u, 63u,
                other);
    require_ok(ds4_gpu_tensor_read(big.conv_state, 0, conv_carry,
        HISTORY * CONV_DIM * sizeof(float)), "convolution carry read");
    require_ok(ds4_gpu_tensor_read(big.state, 0, state_carry,
        STATE_ELEMENTS * sizeof(float)), "recurrent carry read");

    /* One row, then two rows sharing the same carried state: the second call
     * exercises the per-row stride of every buffer. */
    for (uint32_t rows = 1u; rows <= 2u; rows++) {
        gpu_buffers step;
        char label[96];
        buffers_alloc(&step, rows, 1u);
        for (uint32_t row = 0; row < rows; row++) {
            require_ok(ds4_gpu_tensor_write(step.conv_state,
                (uint64_t)row * HISTORY * CONV_DIM * sizeof(float),
                conv_carry, HISTORY * CONV_DIM * sizeof(float)),
                "decode convolution carry write");
            require_ok(ds4_gpu_tensor_write(step.state,
                (uint64_t)row * STATE_ELEMENTS * sizeof(float),
                state_carry, STATE_ELEMENTS * sizeof(float)),
                "decode recurrent carry write");
            require_ok(ds4_gpu_tensor_write(step.qkv,
                (uint64_t)row * CONV_DIM * sizeof(float),
                qkv + (size_t)63 * CONV_DIM, CONV_DIM * sizeof(float)),
                "decode qkv write");
            require_ok(ds4_gpu_tensor_write(step.alpha,
                (uint64_t)row * VALUE_HEADS * sizeof(float),
                alpha + (size_t)63 * VALUE_HEADS,
                VALUE_HEADS * sizeof(float)), "decode alpha write");
            require_ok(ds4_gpu_tensor_write(step.beta,
                (uint64_t)row * VALUE_HEADS * sizeof(float),
                beta + (size_t)63 * VALUE_HEADS,
                VALUE_HEADS * sizeof(float)), "decode beta write");
            require_ok(ds4_gpu_tensor_write(step.output_gate,
                (uint64_t)row * VALUE_DIM * sizeof(float),
                output_gate + (size_t)63 * VALUE_DIM,
                VALUE_DIM * sizeof(float)), "decode output gate write");
        }
        const ds4_gpu_qwen4exp_slab d_conv_slab = gdn_slab(model, g_tiled.conv_offset);
        const ds4_gpu_qwen4exp_slab d_a_log_slab = gdn_slab(model, g_tiled.a_log_offset);
        const ds4_gpu_qwen4exp_slab d_dt_bias_slab = gdn_slab(model, g_tiled.dt_bias_offset);
        const ds4_gpu_qwen4exp_slab d_norm_slab = gdn_slab(model, g_tiled.norm_offset);
        require_ok(ds4_gpu_qwen4exp_gdn_decode(
            step.out, step.conv_state, step.state, step.qkv, step.alpha,
            step.beta, step.output_gate, &d_conv_slab, &d_a_log_slab,
            &d_dt_bias_slab, &d_norm_slab,
            KEY_HEADS, VALUE_HEADS, rows, g_tiled.layout,
            QK_NORM_EPS, NORM_EPS), "GDN decode");
        require_ok(ds4_gpu_tensor_read(step.out, 0, other,
            (uint64_t)rows * VALUE_DIM * sizeof(float)),
            "decode output read");
        require_ok(ds4_gpu_tensor_read(step.state, 0, state_other,
            (uint64_t)rows * STATE_ELEMENTS * sizeof(float)),
            "decode state read");
        for (uint32_t row = 0; row < rows; row++) {
            snprintf(label, sizeof(label),
                     "decode row %u of %u equals prefill step 63", row, rows);
            require_identical(label, other + (size_t)row * VALUE_DIM,
                              actual + (size_t)63 * VALUE_DIM, VALUE_DIM);
            snprintf(label, sizeof(label),
                     "decode row %u of %u final state", row, rows);
            require_identical(label,
                              state_other + (size_t)row * STATE_ELEMENTS,
                              state_actual, STATE_ELEMENTS);
        }
        buffers_free(&step);
    }
    /* 7. GATE STAGING.  The recurrence stages its two per-(token, head)
     *    scalars -- the decay and the write strength -- in shared memory
     *    instead of recomputing them in every thread of every token step.
     *    DS4_QWEN4EXP_NO_GDN_GATE_STAGE puts them back in the loop, which is
     *    the kernel the staging replaced and lowers to the same SASS but for
     *    that one branch.  The two are the same arithmetic on the same
     *    operands, so every output and every element of the carried state has
     *    to agree BYTE for byte -- a band here would pass a kernel that
     *    reassociated the delta rule around the staging, which is the mistake
     *    this case exists to catch.
     *
     *    The widths bracket the tile: 8 is the shortest row that stages at
     *    all, 512 is exactly one tile, 513 the first that refills, and 1024
     *    the prefill chunk. */
    {
        static const uint32_t widths[] = { 8u, 64u, 512u, 513u, MAX_TOKENS };
        for (size_t w = 0; w < sizeof(widths) / sizeof(widths[0]); w++) {
            const uint32_t tokens = widths[w];
            char label[96];
            unsetenv("DS4_QWEN4EXP_NO_GDN_GATE_STAGE");
            buffers_clear_state(&big, 1u);
            run_prefill(&big, model, &g_tiled, qkv, alpha, beta, output_gate,
                        0u, tokens, actual);
            require_ok(ds4_gpu_tensor_read(big.state, 0, state_actual,
                STATE_ELEMENTS * sizeof(float)), "staged state read");
            setenv("DS4_QWEN4EXP_NO_GDN_GATE_STAGE", "1", 1);
            buffers_clear_state(&big, 1u);
            run_prefill(&big, model, &g_tiled, qkv, alpha, beta, output_gate,
                        0u, tokens, other);
            require_ok(ds4_gpu_tensor_read(big.state, 0, state_other,
                STATE_ELEMENTS * sizeof(float)), "in-loop state read");
            unsetenv("DS4_QWEN4EXP_NO_GDN_GATE_STAGE");
            snprintf(label, sizeof(label),
                     "staged gates equal in-loop gates, %u tokens", tokens);
            require_identical(label, actual, other,
                              (size_t)tokens * VALUE_DIM);
            snprintf(label, sizeof(label),
                     "staged gates final state, %u tokens", tokens);
            require_identical(label, state_actual, state_other,
                              STATE_ELEMENTS);
        }

        /* The same pair under the decay regime of case 3b, where the softplus
         * runs far out and the exponent underflows the gate: the staged and
         * the in-loop softplus have to land on the same bits there too. */
        const uint32_t tokens = 64u;
        unsetenv("DS4_QWEN4EXP_NO_GDN_GATE_STAGE");
        buffers_clear_state(&big, 1u);
        run_prefill(&big, model, &g_fast_decay, qkv, alpha, beta, output_gate,
                    0u, tokens, actual);
        require_ok(ds4_gpu_tensor_read(big.state, 0, state_actual,
            STATE_ELEMENTS * sizeof(float)), "staged fast decay state read");
        setenv("DS4_QWEN4EXP_NO_GDN_GATE_STAGE", "1", 1);
        buffers_clear_state(&big, 1u);
        run_prefill(&big, model, &g_fast_decay, qkv, alpha, beta, output_gate,
                    0u, tokens, other);
        require_ok(ds4_gpu_tensor_read(big.state, 0, state_other,
            STATE_ELEMENTS * sizeof(float)), "in-loop fast decay state read");
        unsetenv("DS4_QWEN4EXP_NO_GDN_GATE_STAGE");
        require_identical("staged gates equal in-loop gates, fast decay",
                          actual, other, (size_t)tokens * VALUE_DIM);
        require_identical("staged gates fast decay final state",
                          state_actual, state_other, STATE_ELEMENTS);
    }

    buffers_free(&big);

    free(state_carry);
    free(conv_carry);
    free(state_expected);
    free(state_other);
    free(state_actual);
    free(expected);
    free(other);
    free(actual);
    free(output_gate);
    free(beta);
    free(alpha);
    free(qkv);
    ds4_gpu_cleanup();
    munmap(model, MODEL_BYTES);
    puts("Qwen4exp GDN GPU tests: PASS");
    return 0;
}
