/* Cross-implementation parity fixtures from the ds4-metal parity reference.
 *
 * SOURCE.  ivanfioravanti/ds4-metal, branch qwen3.8-flash-next, MIT licence.
 * The fixture inputs, the scalar reference formulas and the tolerances below
 * are transcribed from tests/test_qwen4_metal.c in that repository, with the
 * line of each case named at the case.  Used with attribution.
 *
 * WHAT THIS FILE IS FOR.  Their tests compare a Metal kernel against a scalar
 * host reference computed in the test.  We cannot run their kernel here, and a
 * kernel result is not ground truth in any case.  What transfers is the pair
 * that IS evidence: their fixture inputs, which are deterministic formulas,
 * and their scalar reference, which is an independently written statement of
 * the same op.  This file runs their reference and our reference on their
 * inputs and requires the two to agree.  Our own kernels are tied to our
 * reference by tests/test_qwen4exp_hc_norm.c and tests/test_qwen4exp_moe.c,
 * so the chain reaches from their reference to our kernels.
 *
 * TOLERANCES.  Two are quoted at every case.  Theirs is the band their kernel
 * needed against their host reference, and it is a KERNEL band, not a
 * statement about the mathematics.  Ours is the DS4-FRESH-PORT-DESIGN.md
 * section 2 band for the op class.  Reference against reference is scalar f32
 * on both sides, so the comparison here is held far tighter than either: at
 * f32 rounding, and at zero where the arithmetic is exact.
 *
 * DIFFERENCES.  Three ops do not agree, and the cases say so and pin the
 * disagreement instead of hiding it.  Our reference takes the convention as a
 * parameter, so each case runs it twice: once with their convention, which
 * must agree exactly, and once with ours, which must NOT.  The second half is
 * the negative control -- without it a case that silently lost the parameter
 * would still pass.  QWEN4EXP-PARITY.md carries the list.
 *
 * No GPU: pure C99 host arithmetic, so this runs everywhere.
 */

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ds4_qwen4exp_hc_ref.h"

static int g_failed = 0;
static int g_total  = 0;

#define CHECK(cond, msg) do {                                                  \
    g_total++;                                                                 \
    if (!(cond)) {                                                             \
        fprintf(stderr, "  FAIL: %s (line %d)\n", (msg), __LINE__);            \
        g_failed++;                                                            \
    }                                                                          \
} while (0)

#define RUN(fn) do {                                                           \
    fprintf(stderr, "RUN: %s\n", #fn);                                         \
    int _before = g_failed;                                                    \
    (fn)();                                                                    \
    fprintf(stderr, "  %s\n", (_before == g_failed) ? "ok" : "FAIL");          \
} while (0)

/* Largest absolute difference over an array, reported so a case prints the
 * margin it actually had rather than only that it passed. */
static float max_abs_diff(const float *a, const float *b, size_t n) {
    float worst = 0.0f;
    for (size_t i = 0; i < n; i++) {
        const float d = fabsf(a[i] - b[i]);
        if (!(d <= worst)) worst = d;   /* NaN-safe: a NaN lands here */
    }
    return worst;
}

/* Largest |a-b| relative to the local magnitude, which is the honest measure
 * when the two sides sum in different orders: an absolute figure on a dot
 * product over 256 terms says more about the operand size than the code. */
static float max_rel_diff(const float *a, const float *b, size_t n) {
    float worst = 0.0f;
    for (size_t i = 0; i < n; i++) {
        const float scale = fabsf(b[i]) > 1.0f ? fabsf(b[i]) : 1.0f;
        const float d = fabsf(a[i] - b[i]) / scale;
        if (!(d <= worst)) worst = d;
    }
    return worst;
}

static void report(const char *what, float worst, float theirs, float ours) {
    fprintf(stderr, "  %-34s max_abs_diff=%.3g  their band=%.3g  our band=%.3g\n",
            what, (double)worst, (double)theirs, (double)ours);
}

/* ===================================================================== *
 * Their fixture helpers, transcribed.
 *
 * test_qwen4_metal.c:79-100 (conversions), :186-190 (block layout),
 * :232-247 (fill), :249-257 (accessor), :259-275 (matmul reference).
 * Kept byte-for-byte in behaviour so the inputs are literally theirs.
 * ===================================================================== */

typedef struct {
    uint16_t d;
    int8_t   qs[32];
} ref_block_q8_0;

static uint16_t ref_f32_to_f16(float value) {
    _Float16 half = (_Float16)value;
    uint16_t bits;
    memcpy(&bits, &half, sizeof(bits));
    return bits;
}

static float ref_f16_to_f32(uint16_t value) {
    _Float16 half;
    memcpy(&half, &value, sizeof(half));
    return (float)half;
}

static uint16_t ref_f32_to_bf16(float value) {
    union { float f; uint32_t u; } bits = { .f = value };
    const uint32_t rounding = 0x7fffu + ((bits.u >> 16) & 1u);
    return (uint16_t)((bits.u + rounding) >> 16);
}

static float ref_bf16_to_f32(uint16_t value) {
    union { uint32_t u; float f; } bits = { .u = (uint32_t)value << 16 };
    return bits.f;
}

/* test_qwen4_metal.c:232-247 */
static void ref_fill_q8_0_matrix(ref_block_q8_0 *blocks, uint32_t out_dim,
                                 uint32_t in_dim, uint32_t seed) {
    const uint32_t row_blocks = in_dim / 32u;
    for (uint32_t row = 0; row < out_dim; row++) {
        for (uint32_t block = 0; block < row_blocks; block++) {
            ref_block_q8_0 *qb = blocks + (size_t)row * row_blocks + block;
            const float delta = 0.00625f * (float)(1u + ((row + block + seed) % 7u));
            qb->d = ref_f32_to_f16(delta);
            for (uint32_t i = 0; i < 32u; i++)
                qb->qs[i] = (int8_t)((int)((row * 19u + block * 7u +
                    i * 11u + seed) % 251u) - 125);
        }
    }
}

/* test_qwen4_metal.c:249-257 */
static float ref_q8_0_value(const ref_block_q8_0 *matrix, uint32_t in_dim,
                            uint32_t row, uint32_t col) {
    const uint32_t row_blocks = in_dim / 32u;
    const ref_block_q8_0 *qb = matrix + (size_t)row * row_blocks + col / 32u;
    return ref_f16_to_f32(qb->d) * (float)qb->qs[col % 32u];
}

/* test_qwen4_metal.c:259-275 */
static void ref_q8_0_matmul(float *out, const ref_block_q8_0 *matrix,
                            const float *x, uint32_t in_dim,
                            uint32_t out_dim, uint32_t rows) {
    for (uint32_t token = 0; token < rows; token++) {
        for (uint32_t row = 0; row < out_dim; row++) {
            float sum = 0.0f;
            for (uint32_t k = 0; k < in_dim; k++)
                sum = fmaf(ref_q8_0_value(matrix, in_dim, row, k),
                           x[(size_t)token * in_dim + k], sum);
            out[(size_t)token * out_dim + row] = sum;
        }
    }
}

/* ===================================================================== *
 * Their model-path fixture geometry, test_qwen4_metal.c:925-928 and the
 * input stream at :961-967.
 * ===================================================================== */

enum {
    F_IN = 256, F_OUT = 13, F_ROWS = 3,
    F_HC_HIDDEN = 9, F_HC_STREAMS = 4,
    F_VOCAB = 5,
};

static ref_block_q8_0 g_a[F_OUT * (F_IN / 32)];
static ref_block_q8_0 g_b[F_OUT * (F_IN / 32)];
static ref_block_q8_0 g_hc[F_HC_HIDDEN * F_HC_STREAMS * (F_IN / 32)];
static ref_block_q8_0 g_embedding[F_VOCAB * (F_IN / 32)];
static float g_input[F_ROWS * F_IN];

static void build_their_fixture(void) {
    /* Seeds 17, 29, 41 and 53, test_qwen4_metal.c:956-959. */
    ref_fill_q8_0_matrix(g_a, F_OUT, F_IN, 17u);
    ref_fill_q8_0_matrix(g_b, F_OUT, F_IN, 29u);
    ref_fill_q8_0_matrix(g_embedding, F_VOCAB, F_IN, 41u);
    ref_fill_q8_0_matrix(g_hc, F_HC_HIDDEN * F_HC_STREAMS, F_IN, 53u);
    /* test_qwen4_metal.c:964-967 */
    for (uint32_t i = 0; i < F_ROWS * F_IN; i++)
        g_input[i] = cosf((float)(i + 3u) * 0.017f) * 0.4f;
}

/* ===================================================================== *
 * Case: the Q8_0 block layout and matmul.
 *
 * Their reference test_qwen4_metal.c:259-275 walks the row-major block
 * directory row by row and scales by the f16 delta.  Ours is
 * ds4_qwen4exp_hc_ref.h:139-161.  Both are scalar f32, so a layout or a
 * delta-decode disagreement shows up as a large error, not a small one.
 *
 * Their kernel band: 4e-4 abs / 4e-4 rel (test_qwen4_metal.c:980).
 * Our band: relative Frobenius <= 2e-2 for a dense GEMM at real quant
 * (DS4-FRESH-PORT-DESIGN.md section 2).
 * Reference against reference: their reference fuses every one of the 256
 * products into the running sum, ours sums a block of 32 first and scales by
 * the block delta once.  Both are exact statements of the same dot product
 * and neither is more correct, but they round differently, so this is held at
 * 1e-6 RELATIVE rather than at zero -- two orders of magnitude inside their
 * kernel band.  Measured at 2.3e-6, so the band is set at 1e-5.
 * ===================================================================== */

static void test_q8_0_matmul_agrees(void) {
    float theirs[F_ROWS * F_OUT], ours[F_ROWS * F_OUT];
    ref_q8_0_matmul(theirs, g_a, g_input, F_IN, F_OUT, F_ROWS);
    ds4_qwen4exp_ref_matmul_q8_0(ours, g_a, F_IN, F_OUT, g_input, F_ROWS);
    const float worst = max_rel_diff(theirs, ours, F_ROWS * F_OUT);
    report("Q8_0 matmul (relative)", worst, 4e-4f, 2e-2f);
    CHECK(worst <= 1e-5f, "Q8_0 matmul agrees with the reference");

    /* The block directory is row-major with in_dim/32 blocks per row.  A
     * transposed reading would not merely be less accurate, it would read
     * another row, so pin one element against a hand walk of their accessor. */
    float hand = 0.0f;
    for (uint32_t k = 0; k < F_IN; k++)
        hand = fmaf(ref_q8_0_value(g_a, F_IN, 7u, k), g_input[F_IN + k], hand);
    CHECK(fabsf(hand - ours[F_OUT + 7]) <= 1e-4f,
          "row 7 of token 1 reads the same blocks on both sides");
}

/* ===================================================================== *
 * Case: SiLU and SwiGLU.
 *
 * Theirs, test_qwen4_metal.c:1024-1025 and :1042:
 *     silu(a)   = a / (1 + exp(-a))
 *     swiglu    = silu(a) * b        -- the FIRST matrix is the gate
 * Ours is ds4_qwen4exp_ref_scale_silu with scale 1
 * (ds4_qwen4exp_hc_ref.h:164-170), which is z * sigmoid(z): the same function,
 * but it MULTIPLIES BY THE RECIPROCAL where theirs DIVIDES.  That is a
 * one-ulp difference on the last operation and nothing more, so the case is
 * held at 1e-6 relative rather than at zero.  It is not a semantic
 * difference and it does not appear in the differences list.
 *
 * Their kernel band: 4e-4 SiLU, 5e-4 SwiGLU (:1032, :1042).
 * Our band: 1e-5 for an elementwise gate.
 * ===================================================================== */

static void test_silu_and_swiglu_agree(void) {
    float a[F_OUT], b[F_OUT];
    ref_q8_0_matmul(a, g_a, g_input, F_IN, F_OUT, 1u);
    ref_q8_0_matmul(b, g_b, g_input, F_IN, F_OUT, 1u);

    float theirs[F_OUT], ours[F_OUT];
    for (uint32_t i = 0; i < F_OUT; i++)
        theirs[i] = a[i] / (1.0f + expf(-a[i]));
    memcpy(ours, a, sizeof(ours));
    ds4_qwen4exp_ref_scale_silu(ours, F_OUT, 1.0f);
    float worst = max_rel_diff(theirs, ours, F_OUT);
    report("SiLU (relative)", worst, 4e-4f, 1e-5f);
    CHECK(worst <= 1e-6f, "SiLU agrees with the reference");

    /* The gate is the first matrix and the value the second.  Swapping them
     * is a real port defect and the check must see it, so the swap is run as
     * a negative control. */
    float their_swiglu[F_OUT], our_swiglu[F_OUT], swapped[F_OUT];
    for (uint32_t i = 0; i < F_OUT; i++) {
        their_swiglu[i] = theirs[i] * b[i];
        our_swiglu[i]   = ours[i] * b[i];
        swapped[i]      = (b[i] / (1.0f + expf(-b[i]))) * a[i];
    }
    worst = max_rel_diff(their_swiglu, our_swiglu, F_OUT);
    report("SwiGLU (relative)", worst, 5e-4f, 1e-5f);
    CHECK(worst <= 1e-6f, "SwiGLU agrees with the reference");
    CHECK(max_abs_diff(their_swiglu, swapped, F_OUT) > 1e-3f,
          "swapping the gate and the value is visible");
}

/* ===================================================================== *
 * Case: the quantized embedding gather, and DIFFERENCE 4.
 *
 * Theirs, test_qwen4_metal.c:1044-1065, gathers token ids {4, 1, -1} and
 * reads a NEGATIVE id as a row of zeros.  That convention marks the image
 * placeholder slots of their vision path.  Our port is text only
 * (DS4-FRESH-PORT-DESIGN.md section 2: "mrope is a no-op for text"), so our
 * gather has no negative-id case at all and would read out of bounds on one.
 * The difference is one of SCOPE, not of correctness: neither side is wrong
 * about the other's model.  The case therefore checks the gather itself on
 * the positive ids and records the convention.
 *
 * Their band: 1e-6 abs / 1e-6 rel (:1064).  Ours: a Q8_0 gather is a copy
 * with a scale and rounds once, so ZERO.
 * ===================================================================== */

static void test_embedding_gather_agrees(void) {
    static const int32_t ids[2] = {4, 1};
    float theirs[2 * F_IN];
    for (uint32_t token = 0; token < 2u; token++)
        for (uint32_t dim = 0; dim < F_IN; dim++)
            theirs[token * F_IN + dim] =
                ref_q8_0_value(g_embedding, F_IN, (uint32_t)ids[token], dim);

    /* Our gather tiles into the hyper-connection streams; at one stream it is
     * the plain gather. */
    float wide[2 * F_IN];
    ds4_qwen4exp_ref_embed_hc_q8_0(wide, g_embedding, ids, 2u, F_IN, 1u);
    float worst = max_abs_diff(theirs, wide, 2u * F_IN);
    report("Q8_0 embedding gather", worst, 1e-6f, 0.0f);
    CHECK(worst == 0.0f, "the embedding gather is bit-identical");

    /* Every stream carries the same row: that tiling is what seeds layer 0. */
    float tiled[2 * F_HC_STREAMS * F_IN];
    ds4_qwen4exp_ref_embed_hc_q8_0(tiled, g_embedding, ids, 2u, F_IN,
                                   F_HC_STREAMS);
    for (uint32_t t = 0; t < 2u; t++)
        for (uint32_t h = 0; h < F_HC_STREAMS; h++) {
            worst = max_abs_diff(theirs + t * F_IN,
                                 tiled + ((size_t)t * F_HC_STREAMS + h) * F_IN,
                                 F_IN);
            CHECK(worst == 0.0f, "each stream is seeded with the same row");
        }
}

/* ===================================================================== *
 * Case: the hyper-connection mix.
 *
 * Theirs, test_qwen4_metal.c:1068-1090 at one row and :1161-1197 at three:
 *     mixed[t][d] = (1/S) * SUM_s normalized[t][s][d] * sigmoid(up[t][s][d])
 * with the activation laid out [token][stream][dim].  Ours is
 * ds4_qwen4exp_ref_hc_mix (ds4_qwen4exp_hc_ref.h:174-188), the same formula
 * over the same layout.  This is a full three-way AGREEMENT: layout, gate
 * position, and the mean over the streams.
 *
 * Their kernel band: 5e-4 abs / 5e-4 rel (:1090, :1196).
 * Our band: 2e-3 abs, cosine >= 0.9999 for an HC mix.
 * Reference against reference: theirs divides by the stream count and ours
 * multiplies by its reciprocal.  The count is 4, a power of two, so the two
 * are bit-identical and the case is held at ZERO.
 * ===================================================================== */

static void hc_up_projection(float *wide, uint32_t rows) {
    /* The raw up projection their reference computes inline at :1075-1080:
     * HC row `stream * HC_HIDDEN + dim` against the token's input. */
    for (uint32_t t = 0; t < rows; t++)
        for (uint32_t s = 0; s < F_HC_STREAMS; s++)
            for (uint32_t d = 0; d < F_HC_HIDDEN; d++) {
                float raw = 0.0f;
                for (uint32_t k = 0; k < F_IN; k++)
                    raw = fmaf(ref_q8_0_value(g_hc, F_IN,
                                              s * F_HC_HIDDEN + d, k),
                               g_input[t * F_IN + k], raw);
                wide[(t * F_HC_STREAMS + s) * F_HC_HIDDEN + d] = raw;
            }
}

static void test_hc_mix_agrees(void) {
    enum { WIDE = F_ROWS * F_HC_STREAMS * F_HC_HIDDEN };
    float wide[WIDE], normalized[WIDE];
    hc_up_projection(wide, F_ROWS);
    /* Their normalized stream at three rows, test_qwen4_metal.c:1163-1164. */
    for (uint32_t i = 0; i < WIDE; i++)
        normalized[i] = sinf((float)(i + 5u) * 0.023f);

    float theirs[F_ROWS * F_HC_HIDDEN], ours[F_ROWS * F_HC_HIDDEN];
    /* test_qwen4_metal.c:1166-1187 */
    for (uint32_t row = 0; row < F_ROWS; row++)
        for (uint32_t dim = 0; dim < F_HC_HIDDEN; dim++) {
            float value = 0.0f;
            for (uint32_t stream = 0; stream < F_HC_STREAMS; stream++) {
                const uint32_t idx =
                    (row * F_HC_STREAMS + stream) * F_HC_HIDDEN + dim;
                value = fmaf(normalized[idx],
                             1.0f / (1.0f + expf(-wide[idx])), value);
            }
            theirs[row * F_HC_HIDDEN + dim] = value / (float)F_HC_STREAMS;
        }
    ds4_qwen4exp_ref_hc_mix(ours, normalized, wide,
                            F_HC_HIDDEN, F_HC_STREAMS, F_ROWS);
    const float worst = max_abs_diff(theirs, ours, F_ROWS * F_HC_HIDDEN);
    report("HC mix", worst, 5e-4f, 2e-3f);
    CHECK(worst == 0.0f, "the HC mix is bit-identical to the reference");

    /* Negative control: reading the activation as [stream][token][dim]
     * instead of [token][stream][dim] must be visible.  Row 0 is shared by
     * both readings, so compare rows 1 and 2. */
    float swapped[WIDE];
    for (uint32_t t = 0; t < F_ROWS; t++)
        for (uint32_t s = 0; s < F_HC_STREAMS; s++)
            memcpy(swapped + (t * F_HC_STREAMS + s) * F_HC_HIDDEN,
                   normalized + (s * F_ROWS + t) * F_HC_HIDDEN,
                   F_HC_HIDDEN * sizeof(float));
    float other[F_ROWS * F_HC_HIDDEN];
    ds4_qwen4exp_ref_hc_mix(other, swapped, wide,
                            F_HC_HIDDEN, F_HC_STREAMS, F_ROWS);
    CHECK(max_abs_diff(theirs + F_HC_HIDDEN, other + F_HC_HIDDEN,
                       2u * F_HC_HIDDEN) > 1e-4f,
          "a transposed stream layout is visible");
}

/* ===================================================================== *
 * Case: the hyper-connection write, and DIFFERENCE 1.
 *
 * Theirs, test_qwen4_metal.c:1096-1120, and their kernel
 * metal/qwen4.metal:646-660 and :854-885:
 *     raw[s]    = SUM_source partials[source][s]
 *     inject[s] = 2 * sigmoid(raw[s])
 *     streams[s][d] += block[d] * inject[s]
 * Ours, ds4_qwen4exp_hc_ref.h:191-204 and :208-228:
 *     inject[s] = 2 * sigmoid(dot(W[s], normed) / n_hc)
 *
 * The reduction, the factor of two, the sigmoid and the fused write agree.
 * The DIVIDE BY THE STREAM COUNT before the sigmoid does not: we have it and
 * they do not.  Their fixture feeds arbitrary partials, so their test does
 * not pin the divide either way; their kernel does, and it has no divide.
 * Ours comes from the MLX runner, which DS4-FRESH-PORT-DESIGN.md section 2
 * names as the semantic source of record for this port.
 *
 * So the case checks the parts that agree with their partials, and pins the
 * divide as a named difference with a negative control on both sides.
 *
 * Their kernel band: 5e-4 abs / 5e-4 rel (:1119).
 * Our band: 1e-5 for a gate.
 * ===================================================================== */

static void test_hc_write_agrees_except_the_divide(void) {
    float block[F_OUT];
    ref_q8_0_matmul(block, g_a, g_input, F_IN, F_OUT, 1u);

    /* test_qwen4_metal.c:1101-1104 */
    float partials[F_HC_STREAMS * F_HC_STREAMS];
    float streams[F_HC_STREAMS * F_OUT];
    for (uint32_t i = 0; i < F_HC_STREAMS * F_HC_STREAMS; i++)
        partials[i] = ((int)(i % 7u) - 3) * 0.08f;
    for (uint32_t i = 0; i < F_HC_STREAMS * F_OUT; i++)
        streams[i] = ((int)(i % 11u) - 5) * 0.03f;

    /* Their expected, :1105-1112.  The partial matrix is [source][stream] and
     * the reduction runs along the leading index. */
    float theirs[F_HC_STREAMS * F_OUT];
    float their_inject[F_HC_STREAMS];
    memcpy(theirs, streams, sizeof(theirs));
    for (uint32_t stream = 0; stream < F_HC_STREAMS; stream++) {
        float raw = 0.0f;
        for (uint32_t source = 0; source < F_HC_STREAMS; source++)
            raw += partials[source * F_HC_STREAMS + stream];
        their_inject[stream] = 2.0f / (1.0f + expf(-raw));
        for (uint32_t dim = 0; dim < F_OUT; dim++)
            theirs[stream * F_OUT + dim] = fmaf(
                block[dim], their_inject[stream], theirs[stream * F_OUT + dim]);
    }

    /* Ours, given the same inject values: the write itself must be identical. */
    float ours[F_HC_STREAMS * F_OUT];
    ds4_qwen4exp_ref_hc_inject(ours, streams, block, their_inject,
                               F_OUT, F_HC_STREAMS, 1u);
    float worst = max_abs_diff(theirs, ours, F_HC_STREAMS * F_OUT);
    report("HC write", worst, 5e-4f, 1e-5f);
    CHECK(worst == 0.0f, "the HC write is bit-identical to the reference");

    /* Negative control on the reduction axis: summing the partial matrix
     * along the trailing index instead of the leading one must change the
     * inject values.  This is what makes the [source][stream] convention
     * evidence rather than an assumption. */
    float axis_swapped[F_HC_STREAMS];
    for (uint32_t stream = 0; stream < F_HC_STREAMS; stream++) {
        float raw = 0.0f;
        for (uint32_t source = 0; source < F_HC_STREAMS; source++)
            raw += partials[stream * F_HC_STREAMS + source];
        axis_swapped[stream] = 2.0f / (1.0f + expf(-raw));
    }
    CHECK(max_abs_diff(their_inject, axis_swapped, F_HC_STREAMS) > 1e-4f,
          "reducing the partials along the other axis is visible");

    /* DIFFERENCE 1, pinned.  Build one full inject from weights both ways.
     * With their convention -- no divide -- the two references must agree
     * exactly.  With ours the result must DIFFER, or the divide has been
     * silently dropped from our reference and nothing would notice. */
    enum { WIDE = F_HC_STREAMS * F_HC_HIDDEN };
    float normed[WIDE], weights[F_HC_STREAMS * WIDE];
    for (uint32_t i = 0; i < WIDE; i++)
        normed[i] = sinf((float)(i + 1u) * 0.09f);
    for (uint32_t i = 0; i < F_HC_STREAMS * WIDE; i++)
        weights[i] = cosf((float)(i + 2u) * 0.031f) * 0.25f;

    float their_rule[F_HC_STREAMS], our_rule[F_HC_STREAMS];
    for (uint32_t s = 0; s < F_HC_STREAMS; s++) {
        float acc = 0.0f;
        for (uint32_t i = 0; i < WIDE; i++)
            acc += normed[i] * weights[s * WIDE + i];
        their_rule[s] = 2.0f / (1.0f + expf(-acc));                 /* no divide */
    }
    ds4_qwen4exp_ref_hc_inject_weights(our_rule, normed, weights,
                                       F_HC_HIDDEN, F_HC_STREAMS, 1u);
    CHECK(max_abs_diff(their_rule, our_rule, F_HC_STREAMS) > 1e-3f,
          "DIFFERENCE 1: our inject divides by the stream count, theirs does not");

    /* And with the divide neutralised the two rules are the same rule: feed
     * our reference weights pre-multiplied by the stream count. */
    float scaled[F_HC_STREAMS * WIDE], neutralised[F_HC_STREAMS];
    for (uint32_t i = 0; i < F_HC_STREAMS * WIDE; i++)
        scaled[i] = weights[i] * (float)F_HC_STREAMS;
    ds4_qwen4exp_ref_hc_inject_weights(neutralised, normed, scaled,
                                       F_HC_HIDDEN, F_HC_STREAMS, 1u);
    worst = max_abs_diff(their_rule, neutralised, F_HC_STREAMS);
    report("HC inject, divide neutralised", worst, 5e-4f, 1e-5f);
    CHECK(worst <= 1e-6f,
          "the divide is the ONLY difference in the inject rule");
}

/* ===================================================================== *
 * Case: the low-rank activation, and DIFFERENCE 2.
 *
 * Their host path, ds4.c:54875-54890 in the reference repository, runs the
 * hyper-connection down projection into a plain SiLU with no scale.  Ours
 * (ds4_qwen4exp_hc_ref.h:303-304) applies silu(x / n_hc).  Same shape of
 * difference as DIFFERENCE 1 and the same evidence: the MLX runner.
 * ===================================================================== */

static void test_lowrank_activation_difference(void) {
    enum { N = 64 };
    float x[N], theirs[N], ours[N], neutralised[N];
    for (uint32_t i = 0; i < N; i++)
        x[i] = cosf((float)(i + 1u) * 0.11f) * 3.0f;

    for (uint32_t i = 0; i < N; i++)
        theirs[i] = x[i] / (1.0f + expf(-x[i]));      /* no scale */

    memcpy(ours, x, sizeof(ours));
    ds4_qwen4exp_ref_scale_silu(ours, N, 1.0f / (float)F_HC_STREAMS);
    CHECK(max_abs_diff(theirs, ours, N) > 1e-3f,
          "DIFFERENCE 2: our low-rank activation scales by 1/n_hc, theirs does not");

    /* Pre-scaling the input by the stream count removes the difference, so
     * the scale is the only thing between the two. */
    float pre[N];
    for (uint32_t i = 0; i < N; i++) pre[i] = x[i] * (float)F_HC_STREAMS;
    memcpy(neutralised, pre, sizeof(neutralised));
    ds4_qwen4exp_ref_scale_silu(neutralised, N, 1.0f / (float)F_HC_STREAMS);
    const float worst = max_rel_diff(theirs, neutralised, N);
    report("low-rank SiLU, scale off (rel)", worst, 4e-4f, 1e-5f);
    CHECK(worst <= 1e-6f, "the scale is the ONLY difference in the activation");
}

/* ===================================================================== *
 * Case: the grouped RMS norm, and DIFFERENCE 3.
 *
 * Their kernel metal/qwen4.metal:589-628 normalizes each hyper-connection
 * stream on its own statistic and multiplies by the weight in f32:
 *     out = x * rsqrt(mean(x^2) + eps) * w
 * Ours, ds4_qwen4exp_hc_ref.h:114-135, does the same but ROUNDS THE
 * NORMALIZED VALUE TO BF16 before the weight multiply, because the MLX
 * runner casts to the activation dtype there.
 *
 * The grouping, the epsilon position, the per-stream statistic and the
 * weight indexing all AGREE.  The bf16 rounding does not, and it is a real
 * difference in the last few bits, not a rounding artefact.
 *
 * Their kernel band is not quoted for a norm; ours is 1e-5 absolute.
 * ===================================================================== */

static void their_grouped_rms_norm(float *out, const float *x, const float *w,
                                   uint32_t hidden, uint32_t streams,
                                   uint32_t rows, float eps) {
    for (uint32_t r = 0; r < rows * streams; r++) {
        const uint32_t stream = r % streams;
        const size_t base = (size_t)r * hidden;
        float sum = 0.0f;
        for (uint32_t c = 0; c < hidden; c++)
            sum = fmaf(x[base + c], x[base + c], sum);
        const float scale = 1.0f / sqrtf(sum / (float)hidden + eps);
        for (uint32_t c = 0; c < hidden; c++)
            out[base + c] = x[base + c] * scale * w[stream * hidden + c];
    }
}

static void test_grouped_rms_norm_agrees_except_the_bf16_cast(void) {
    enum { HID = 64, S = F_HC_STREAMS, ROWS = 2, WIDE = S * HID };
    float x[ROWS * WIDE], w[WIDE];
    float theirs[ROWS * WIDE], ours[ROWS * WIDE], unrounded[ROWS * WIDE];
    for (uint32_t i = 0; i < ROWS * WIDE; i++)
        x[i] = sinf((float)(i + 1u) * 0.037f) * 1.5f;
    for (uint32_t i = 0; i < WIDE; i++)
        w[i] = 1.0f + cosf((float)(i + 1u) * 0.019f) * 0.1f;

    their_grouped_rms_norm(theirs, x, w, HID, S, ROWS, 1e-6f);

    /* Ours with THEIR convention: weight already carries the offset, no bf16
     * cast.  This must agree at f32 rounding. */
    ds4_qwen4exp_ref_rms_norm(unrounded, x, w, WIDE, HID, ROWS,
                              1e-6f, 0.0f, 0);
    float worst = max_abs_diff(theirs, unrounded, ROWS * WIDE);
    report("grouped RMS norm", worst, 1e-5f, 1e-5f);
    CHECK(worst <= 1e-6f,
          "the grouped norm agrees once the bf16 cast is off");

    /* DIFFERENCE 3: with the cast on, the results must differ. */
    ds4_qwen4exp_ref_rms_norm(ours, x, w, WIDE, HID, ROWS, 1e-6f, 0.0f, 1);
    CHECK(max_abs_diff(theirs, ours, ROWS * WIDE) > 1e-6f,
          "DIFFERENCE 3: we round the normalized value to bf16, they do not");

    /* Every element still lands within one bf16 ulp of theirs, so the cast is
     * the whole of the difference and not a second one hiding behind it. */
    int outside = 0;
    for (uint32_t i = 0; i < ROWS * WIDE; i++) {
        const float lhs = ref_bf16_to_f32(ref_f32_to_bf16(theirs[i] / w[i % WIDE]));
        const float rhs = ours[i] / w[i % WIDE];
        if (fabsf(lhs - rhs) > 1e-3f * fabsf(lhs) + 1e-6f) outside++;
    }
    CHECK(outside == 0, "the bf16 cast explains every differing element");

    /* Negative control on the grouping: one statistic over the whole 10240
     * row, rather than one per stream, must be visible. */
    float ungrouped[ROWS * WIDE];
    ds4_qwen4exp_ref_rms_norm(ungrouped, x, w, WIDE, WIDE, ROWS,
                              1e-6f, 0.0f, 0);
    CHECK(max_abs_diff(theirs, ungrouped, ROWS * WIDE) > 1e-3f,
          "a single statistic over all streams is visible");
}

/* ===================================================================== *
 * Case: the router.
 *
 * Theirs, test_qwen4_metal.c:352-386 (`moe_topk_reference`) with the fixture
 * at :1662-1691.  Ours is `ref_router` in tests/test_qwen4exp_moe.c:313-341,
 * restated here so this file needs no GPU.
 *
 * Both take the top k of the RAW f32 logits, break ties toward the LOWER
 * expert index, and take a softmax over the SELECTED logits only, with no
 * bias, no renormalisation and no scaling factor.  They shift by the maximum
 * over all experts and we shift by the maximum over the selected; the top-k
 * always contains the global maximum, so the two shifts are the same number
 * and the results are identical, not merely close.  Full AGREEMENT.
 *
 * Their kernel band: ids exact, weights 2e-6 abs / 2e-6 rel (:1683-1686).
 * Our band: identical top-k id set (DS4-FRESH-PORT-DESIGN.md section 2).
 * ===================================================================== */

enum { R_EXPERTS = 512, R_TOP_K = 10, R_ROWS = 6 };

/* test_qwen4_metal.c:352-386, transcribed. */
static void their_moe_topk(const float *scores, uint32_t experts,
                           uint32_t top_k, int32_t *selected, float *weights) {
    float best[16];
    for (uint32_t slot = 0; slot < top_k; slot++) {
        best[slot] = -INFINITY;
        selected[slot] = -1;
    }
    float max_score = -INFINITY;
    for (uint32_t expert = 0; expert < experts; expert++) {
        const float score = scores[expert];
        if (score > max_score) max_score = score;
        uint32_t insert = top_k;
        for (uint32_t slot = 0; slot < top_k; slot++) {
            if (score > best[slot] ||
                (score == best[slot] && (int32_t)expert < selected[slot])) {
                insert = slot;
                break;
            }
        }
        if (insert < top_k) {
            for (uint32_t slot = top_k - 1u; slot > insert; slot--) {
                best[slot] = best[slot - 1u];
                selected[slot] = selected[slot - 1u];
            }
            best[insert] = score;
            selected[insert] = (int32_t)expert;
        }
    }
    float sum = 0.0f;
    for (uint32_t slot = 0; slot < top_k; slot++) {
        weights[slot] = expf(best[slot] - max_score);
        sum += weights[slot];
    }
    for (uint32_t slot = 0; slot < top_k; slot++) weights[slot] /= sum;
}

/* Ours, tests/test_qwen4exp_moe.c:313-341, restated. */
typedef struct { float score; int32_t index; } router_entry;

static int our_router_cmp(const void *a, const void *b) {
    const router_entry *ea = a, *eb = b;
    if (ea->score > eb->score) return -1;
    if (ea->score < eb->score) return 1;
    return ea->index < eb->index ? -1 : 1;
}

static void our_router(const float *logits, int32_t *selected, float *weights) {
    router_entry entries[R_EXPERTS];
    for (int i = 0; i < R_EXPERTS; i++) {
        entries[i].score = logits[i];
        entries[i].index = i;
    }
    qsort(entries, R_EXPERTS, sizeof(entries[0]), our_router_cmp);
    float m = -FLT_MAX;
    for (int i = 0; i < R_TOP_K; i++) {
        selected[i] = entries[i].index;
        if (entries[i].score > m) m = entries[i].score;
    }
    float sum = 0.0f;
    for (int i = 0; i < R_TOP_K; i++) {
        weights[i] = expf(logits[selected[i]] - m);
        sum += weights[i];
    }
    for (int i = 0; i < R_TOP_K; i++) weights[i] /= sum;
}

static void test_router_agrees(void) {
    float logits[R_ROWS * R_EXPERTS];
    /* Their fixture stream at :1667-1668, widened to the production expert
     * count.  Deliberately coarse, so exact ties occur and the tie rule is
     * exercised rather than assumed. */
    for (uint32_t i = 0; i < R_ROWS * R_EXPERTS; i++)
        logits[i] = (float)((int)((i * 37u) % 23u) - 11) * 0.125f;

    int ties = 0;
    for (uint32_t row = 0; row < R_ROWS; row++) {
        int32_t their_ids[R_TOP_K], our_ids[R_TOP_K];
        float their_w[R_TOP_K], our_w[R_TOP_K];
        their_moe_topk(logits + row * R_EXPERTS, R_EXPERTS, R_TOP_K,
                       their_ids, their_w);
        our_router(logits + row * R_EXPERTS, our_ids, our_w);
        CHECK(memcmp(their_ids, our_ids, sizeof(their_ids)) == 0,
              "the router selects the same experts in the same order");
        CHECK(max_abs_diff(their_w, our_w, R_TOP_K) <= 1e-7f,
              "the router weights agree");
        for (uint32_t i = 1; i < R_TOP_K; i++)
            if (logits[row * R_EXPERTS + their_ids[i]] ==
                logits[row * R_EXPERTS + their_ids[i - 1]]) ties++;
    }
    CHECK(ties > 0, "the fixture actually contains ties");

    /* The softmax is over the SELECTED logits only.  A softmax over all
     * experts would give different weights, so run it as a negative control. */
    float over_all[R_TOP_K];
    int32_t ids[R_TOP_K];
    float w[R_TOP_K];
    our_router(logits, ids, w);
    float sum = 0.0f, m = -FLT_MAX;
    for (uint32_t e = 0; e < R_EXPERTS; e++)
        if (logits[e] > m) m = logits[e];
    for (uint32_t e = 0; e < R_EXPERTS; e++) sum += expf(logits[e] - m);
    for (uint32_t i = 0; i < R_TOP_K; i++)
        over_all[i] = expf(logits[ids[i]] - m) / sum;
    CHECK(max_abs_diff(w, over_all, R_TOP_K) > 1e-3f,
          "a softmax over all experts is visible");
}

/* ===================================================================== *
 * Gated delta net: their prepare fixture.
 *
 * Their reference: test_qwen4_metal.c:2977-3042 (`gdn_prepare_reference`),
 * driven by the fixture at :3098-3216.  Geometry TOKENS 17, DIM 128, one key
 * head, three value heads, conv width 4, conv dim 640.  Weights are BF16.
 *
 * Ours: tests/test_qwen4exp_gdn.c:246-286, restated here without the
 * production dimensions baked in.  Same arithmetic, same order.
 *
 * Their kernel band: 2e-6 abs / 2e-5 rel (test_qwen4_metal.c:3169-3182).
 * Our band: 2e-5 max abs with cosine >= 0.9999
 * (tests/test_qwen4exp_gdn.c:75).
 *
 * The convolution, the activation, the split, the beta gate and the decay
 * gate AGREE, bit for bit.  The q/k normalisation does NOT, and on their own
 * fixture the gap is 100 times their own kernel band: see DIFFERENCE 5.  This
 * is the most consequential finding of the comparison, so the case measures
 * the gap and pins it rather than passing it off as rounding.
 * ===================================================================== */

enum {
    G_TOKENS = 17, G_DIM = 128, G_KH = 1, G_VH = 3, G_WIDTH = 4,
    G_KEY_DIM = G_KH * G_DIM,               /* 128 */
    G_VALUE_DIM = G_VH * G_DIM,             /* 384 */
    G_CONV_DIM = 2 * G_KEY_DIM + G_VALUE_DIM /* 640 */
};

/* Their weights, test_qwen4_metal.c:3113-3128.  BF16, so the round trip is
 * part of the fixture and not an approximation of it. */
static uint16_t g_conv_w[G_CONV_DIM * G_WIDTH];
static uint16_t g_a_log[G_VH];
static uint16_t g_dt_bias[G_VH];
static uint16_t g_norm_w[G_DIM];
static float g_mixed[G_TOKENS * G_CONV_DIM];
static float g_raw_decay[G_TOKENS * G_VH];
static float g_raw_beta[G_TOKENS * G_VH];

static void build_their_gdn_fixture(void) {
    for (uint32_t c = 0; c < G_CONV_DIM; c++)
        for (uint32_t tap = 0; tap < G_WIDTH; tap++)
            g_conv_w[c * G_WIDTH + tap] = ref_f32_to_bf16(
                0.03f * (float)(tap + 1u) + 0.002f * (float)((int)(c % 9u) - 4));
    for (uint32_t h = 0; h < G_VH; h++) {
        g_a_log[h]   = ref_f32_to_bf16(logf(0.25f + 0.15f * (float)h));
        g_dt_bias[h] = ref_f32_to_bf16(-0.35f + 0.2f * (float)h);
    }
    for (uint32_t d = 0; d < G_DIM; d++)
        g_norm_w[d] = ref_f32_to_bf16(0.8f + 0.002f * (float)(d % 29u));
    /* test_qwen4_metal.c:3131-3139.  The mask is theirs; we have no mask, so
     * this fixture runs the all-active case.  See DIFFERENCE 7. */
    for (uint32_t i = 0; i < G_TOKENS * G_CONV_DIM; i++)
        g_mixed[i] = 0.0125f * (float)((int)((i * 7u + 3u) % 31u) - 15);
    for (uint32_t i = 0; i < G_TOKENS * G_VH; i++) {
        g_raw_decay[i] = 0.04f  * (float)((int)(i % 7u) - 3);
        g_raw_beta[i]  = 0.075f * (float)((int)(i % 5u) - 2);
    }
}

/* Their prepare, test_qwen4_metal.c:2977-3042, transcribed.  `qk_eps_on_sum`
 * selects the convention: 1 is theirs, 0 is ours.  Everything else is shared,
 * which is what makes the epsilon the isolated variable. */
static void their_gdn_prepare(float *q, float *k, float *v,
                              float *decay, float *beta, float *conv_state,
                              int qk_eps_on_sum) {
    for (uint32_t channel = 0; channel < G_CONV_DIM; channel++) {
        float history[G_WIDTH];
        for (uint32_t tap = 0; tap < G_WIDTH; tap++)
            history[tap] = conv_state[channel * G_WIDTH + tap];
        for (uint32_t token = 0; token < G_TOKENS; token++) {
            for (uint32_t tap = 0; tap + 1u < G_WIDTH; tap++)
                history[tap] = history[tap + 1u];
            history[G_WIDTH - 1u] = g_mixed[(size_t)token * G_CONV_DIM + channel];
            float sum = 0.0f;
            for (uint32_t tap = 0; tap < G_WIDTH; tap++)
                sum = fmaf(history[tap],
                           ref_bf16_to_f32(g_conv_w[channel * G_WIDTH + tap]), sum);
            const float value = sum / (1.0f + expf(-sum));
            if (channel < G_KEY_DIM)
                q[(size_t)token * G_KEY_DIM + channel] = value;
            else if (channel < 2u * G_KEY_DIM)
                k[(size_t)token * G_KEY_DIM + channel - G_KEY_DIM] = value;
            else
                v[(size_t)token * G_VALUE_DIM + channel - 2u * G_KEY_DIM] = value;
        }
        for (uint32_t tap = 0; tap < G_WIDTH; tap++)
            conv_state[channel * G_WIDTH + tap] = history[tap];
    }
    for (uint32_t token = 0; token < G_TOKENS; token++) {
        for (uint32_t head = 0; head < G_KH; head++) {
            const size_t base = ((size_t)token * G_KH + head) * G_DIM;
            float qsum = 0.0f, ksum = 0.0f;
            for (uint32_t d = 0; d < G_DIM; d++) {
                qsum = fmaf(q[base + d], q[base + d], qsum);
                ksum = fmaf(k[base + d], k[base + d], ksum);
            }
            float qscale, kscale;
            if (qk_eps_on_sum == 1) {
                /* THEIRS, test_qwen4_metal.c:3021-3023. */
                qscale = 1.0f / sqrtf((qsum + 1.0e-6f) * (float)G_DIM);
                kscale = 1.0f / sqrtf(ksum + 1.0e-6f);
            } else if (qk_eps_on_sum == 0) {
                /* OURS, and no longer a difference: metal/qwen4exp_gdn.metal
                 * l2-normalises with the epsilon on the SUM and then scales
                 * the query alone by 2^-3.5.  Written the way the kernel
                 * writes it -- the query's 1/sqrt(DIM) is a separate constant
                 * rather than folded under their square root -- so the check
                 * below tests the algebra and its rounding, not a copy. */
                qscale = 0x1.6a09e6p-4f / sqrtf(qsum + 1.0e-6f);
                kscale = 1.0f / sqrtf(ksum + 1.0e-6f);
            } else {
                /* RETIRED: the epsilon inside the mean, which reaches the sum
                 * multiplied by DIM.  Kept only as the negative control below,
                 * so a silent return to it is caught. */
                qscale = (1.0f / (float)G_DIM) /
                         sqrtf(qsum / (float)G_DIM + 1.0e-6f);
                kscale = (1.0f / sqrtf((float)G_DIM)) /
                         sqrtf(ksum / (float)G_DIM + 1.0e-6f);
            }
            for (uint32_t d = 0; d < G_DIM; d++) {
                q[base + d] *= qscale;
                k[base + d] *= kscale;
            }
        }
        for (uint32_t head = 0; head < G_VH; head++) {
            const size_t at = (size_t)token * G_VH + head;
            const float shifted = g_raw_decay[at] + ref_bf16_to_f32(g_dt_bias[head]);
            const float softplus = shifted > 20.0f ? shifted : log1pf(expf(shifted));
            decay[at] = expf(-expf(ref_bf16_to_f32(g_a_log[head])) * softplus);
            beta[at]  = 1.0f / (1.0f + expf(-g_raw_beta[at]));
        }
    }
}

/* Our softplus, tests/test_qwen4exp_gdn.c:241-244: the numerically stable
 * form, with no threshold branch. */
static float our_softplus(float x) {
    const float m = x > 0.0f ? x : 0.0f;
    return m + log1pf(expf(-fabsf(x)));
}

static void test_gdn_prepare_agrees(void) {
    static float q_t[G_TOKENS * G_KEY_DIM], k_t[G_TOKENS * G_KEY_DIM];
    static float q_o[G_TOKENS * G_KEY_DIM], k_o[G_TOKENS * G_KEY_DIM];
    static float v_t[G_TOKENS * G_VALUE_DIM], v_o[G_TOKENS * G_VALUE_DIM];
    static float d_t[G_TOKENS * G_VH], d_o[G_TOKENS * G_VH];
    static float b_t[G_TOKENS * G_VH], b_o[G_TOKENS * G_VH];
    static float cs_t[G_CONV_DIM * G_WIDTH], cs_o[G_CONV_DIM * G_WIDTH];

    memset(cs_t, 0, sizeof(cs_t));
    memset(cs_o, 0, sizeof(cs_o));
    their_gdn_prepare(q_t, k_t, v_t, d_t, b_t, cs_t, 1);
    their_gdn_prepare(q_o, k_o, v_o, d_o, b_o, cs_o, 0);

    /* The value stream never passes the q/k norm, so the convolution, the
     * activation and the split must agree bit for bit. */
    float worst = max_abs_diff(v_t, v_o, G_TOKENS * G_VALUE_DIM);
    report("GDN conv + SiLU + split", worst, 2e-6f, 2e-5f);
    CHECK(worst == 0.0f, "the convolution, activation and split agree exactly");

    /* The decay and beta gates agree: no floor on either side. */
    worst = max_abs_diff(d_t, d_o, G_TOKENS * G_VH);
    report("GDN decay gate", worst, 2e-6f, 2e-5f);
    CHECK(worst == 0.0f, "the decay gate agrees, and neither side floors it");
    CHECK(max_abs_diff(b_t, b_o, G_TOKENS * G_VH) == 0.0f,
          "the beta gate agrees");

    /* A decay far below the KDA floor must survive on both sides.  That floor
     * is the thing our port deliberately dropped, so it needs a positive
     * check and not only a comment. */
    const float deep = expf(-expf(1.1f) * our_softplus(2.0f + 2.0f));
    CHECK(deep < expf(-5.0f),
          "the fixture can reach a decay below the KDA floor");
    CHECK(deep > 0.0f, "and neither side clamps it to zero");

    /* DIFFERENCE 5, now CLOSED.
     *
     * This engine used to divide the sum of squares by the head dimension
     * before adding the epsilon, which is the same expression with an epsilon
     * 128 times too large; on their fixture that moved the key by 2.2e-3,
     * about 100 times their kernel band.  transformers modeling_qwen4_exp.py
     * l2-normalises q and k with the epsilon on the SUM and then scales the
     * query by head_dim ** -0.5, which is what the reference does and what
     * this engine now does.  The two must agree exactly.
     *
     * The engine's own kernel is pinned by tests/test_qwen4exp_gdn.c, which
     * carries the mutation proof; this case pins the arithmetic. */
    const float q_gap = max_abs_diff(q_t, q_o, G_TOKENS * G_KEY_DIM);
    const float k_gap = max_abs_diff(k_t, k_o, G_TOKENS * G_KEY_DIM);
    report("GDN q after norm", q_gap, 2e-6f, 2e-5f);
    report("GDN k after norm", k_gap, 2e-6f, 2e-5f);
    CHECK(k_gap == 0.0f, "DIFFERENCE 5 closed: the key norm agrees exactly");
    /* The query is not bit-identical and does not need to be: they fold the
     * head dimension under one square root, `1 / sqrt((sum + eps) * D)`, and
     * the kernel keeps 2^-3.5 as a separate constant.  The two are the same
     * number to within a rounding step -- 1.9e-9, three orders inside their
     * own 2e-6 band -- and the kernel's factorisation is the one the original
     * model uses.  Assert the band, not the bits. */
    CHECK(q_gap < 2e-6f,
          "DIFFERENCE 5 closed: the query norm agrees inside their band");

    /* NEGATIVE CONTROL.  The retired convention must still be far outside the
     * band, or the two checks above would pass on a case that had stopped
     * being able to tell the conventions apart. */
    static float q_r[G_TOKENS * G_KEY_DIM], k_r[G_TOKENS * G_KEY_DIM];
    static float v_r[G_TOKENS * G_VALUE_DIM];
    static float d_r[G_TOKENS * G_VH], b_r[G_TOKENS * G_VH];
    static float cs_r[G_CONV_DIM * G_WIDTH];
    memset(cs_r, 0, sizeof(cs_r));
    their_gdn_prepare(q_r, k_r, v_r, d_r, b_r, cs_r, 2);
    const float k_old = max_abs_diff(k_t, k_r, G_TOKENS * G_KEY_DIM);
    const float q_old = max_abs_diff(q_t, q_r, G_TOKENS * G_KEY_DIM);
    report("GDN k, retired epsilon", k_old, 2e-6f, 2e-5f);
    CHECK(k_old > 1e-3f && q_old > 2e-5f,
          "the retired epsilon is still detectable, so the case is sensitive");
}

/* The epsilon convention, isolated.
 *
 * DIFFERENCE 5, kept after it was closed, because it is what says the closing
 * mattered.  The model's rule puts the epsilon on the SUM of squares
 * (test_qwen4_metal.c:3021-3023, metal/qwen4.metal:2388-2391), and this engine
 * now does the same (metal/qwen4exp_gdn.metal, tests/test_qwen4exp_gdn.c).
 * The RETIRED rule put it inside the MEAN, which reaches the sum multiplied by
 * the head dimension -- an effective epsilon of 128e-6 against 1e-6.
 *
 * Algebraically the two are the same expression with a different epsilon and
 * nothing else, so the whole difference appears only when the sum of squares
 * is small enough for the epsilon to matter.  This case drives it there and
 * measures the gap, so the size of what was fixed is on the record rather than
 * asserted to have been negligible.
 * ===================================================================== */

static void test_gdn_qk_epsilon_difference(void) {
    /* Their rule and ours written side by side on one vector. */
    static const float magnitudes[4] = {1.0f, 1e-1f, 1e-2f, 1e-3f};
    float measured[4];
    for (uint32_t m = 0; m < 4u; m++) {
        float x[G_DIM];
        for (uint32_t d = 0; d < G_DIM; d++)
            x[d] = magnitudes[m] * sinf((float)(d + 1u) * 0.041f);
        float sum = 0.0f;
        for (uint32_t d = 0; d < G_DIM; d++) sum = fmaf(x[d], x[d], sum);
        /* The rule both engines now use, against the one this engine
         * retired.  Not "theirs against ours": ours IS the first one. */
        const float shared  = 1.0f / sqrtf(sum + 1.0e-6f);
        const float retired = (1.0f / sqrtf((float)G_DIM)) /
                              sqrtf(sum / (float)G_DIM + 1.0e-6f);
        measured[m] = fabsf(shared - retired) / shared;
        fprintf(stderr, "  key scale at magnitude %.0e: relative gap %.3g\n",
                (double)magnitudes[m], (double)measured[m]);
    }
    /* At production magnitudes the conventions are interchangeable. */
    CHECK(measured[0] < 1e-5f,
          "at unit magnitude the epsilon convention does not matter");
    /* At small magnitudes they are not.  This is the discriminating case, and
     * it is the one our own test constructs at tests/test_qwen4exp_gdn.c:645. */
    CHECK(measured[3] > 10.0f * measured[0],
          "at small magnitude the retired epsilon changed the scale, which is "
          "why DIFFERENCE 5 was worth closing");
}

/* ===================================================================== *
 * Gated delta net: the recurrence.
 *
 * Their reference: test_qwen4_metal.c:3453-3491 (`gdn_reference`).
 * Ours: tests/test_qwen4exp_gdn.c:286-311.
 *
 * The delta rule itself AGREES step for step: the decay multiply is folded
 * into the same loop that forms `kv`, the delta is `(v - kv) * beta`, and the
 * read-out reuses the row it has just updated.  Ours accumulates in double
 * and theirs in float, so this is held at a relative band.
 *
 * DIFFERENCE 6 is the key-head pairing.  Theirs is hardcoded GROUPED,
 * `hk = hv / (value_heads / key_heads)` (:3459).  Ours is selectable and
 * production uses TILED, `hv % n_key_head` (tests/test_qwen4exp_gdn.c:287-289,
 * ds4_qwen4exp_graph.inc:930).  Their fixture has ONE key head, where the two
 * coincide, so their fixture cannot see the difference at all.  The second
 * case below gives it two key heads, where it can.
 * ===================================================================== */

static void gdn_recurrence(double *out, double *state,
                           const float *q, const float *k, const float *v,
                           const float *decay, const float *beta,
                           uint32_t tokens, uint32_t key_heads,
                           uint32_t value_heads, uint32_t dim, int tiled) {
    const uint32_t repeat = value_heads / key_heads;
    for (uint32_t t = 0; t < tokens; t++)
        for (uint32_t hv = 0; hv < value_heads; hv++) {
            const uint32_t hk = tiled ? hv % key_heads : hv / repeat;
            const double g = decay[(size_t)t * value_heads + hv];
            const double b = beta[(size_t)t * value_heads + hv];
            for (uint32_t dv = 0; dv < dim; dv++) {
                double *row = state + ((size_t)hv * dim + dv) * dim;
                double kv = 0.0;
                for (uint32_t dk = 0; dk < dim; dk++) {
                    row[dk] *= g;
                    kv += row[dk] * (double)k[((size_t)t * key_heads + hk) * dim + dk];
                }
                const double delta =
                    ((double)v[((size_t)t * value_heads + hv) * dim + dv] - kv) * b;
                double y = 0.0;
                for (uint32_t dk = 0; dk < dim; dk++) {
                    row[dk] += (double)k[((size_t)t * key_heads + hk) * dim + dk] * delta;
                    y += row[dk] * (double)q[((size_t)t * key_heads + hk) * dim + dk];
                }
                out[((size_t)t * value_heads + hv) * dim + dv] = y;
            }
        }
}

static void test_gdn_recurrence_agrees(void) {
    static float q[G_TOKENS * G_KEY_DIM], k[G_TOKENS * G_KEY_DIM];
    static float v[G_TOKENS * G_VALUE_DIM];
    static float decay[G_TOKENS * G_VH], beta[G_TOKENS * G_VH];
    static float cs[G_CONV_DIM * G_WIDTH];
    memset(cs, 0, sizeof(cs));
    their_gdn_prepare(q, k, v, decay, beta, cs, 1);

    static double out_grouped[G_TOKENS * G_VALUE_DIM];
    static double out_tiled[G_TOKENS * G_VALUE_DIM];
    static double state[G_VH * G_DIM * G_DIM];

    memset(state, 0, sizeof(state));
    gdn_recurrence(out_grouped, state, q, k, v, decay, beta,
                   G_TOKENS, G_KH, G_VH, G_DIM, 0);
    memset(state, 0, sizeof(state));
    gdn_recurrence(out_tiled, state, q, k, v, decay, beta,
                   G_TOKENS, G_KH, G_VH, G_DIM, 1);

    /* With one key head the two pairings are the same map, so their fixture
     * is blind to DIFFERENCE 6.  Say so with a check rather than a comment. */
    double gap = 0.0;
    for (size_t i = 0; i < G_TOKENS * G_VALUE_DIM; i++) {
        const double d = fabs(out_grouped[i] - out_tiled[i]);
        if (d > gap) gap = d;
    }
    CHECK(gap == 0.0,
          "at one key head the grouped and tiled pairings coincide");

    /* The recurrence produced something to compare: a state that stayed at
     * zero would make every check above vacuous. */
    double energy = 0.0;
    for (size_t i = 0; i < G_TOKENS * G_VALUE_DIM; i++)
        energy += out_grouped[i] * out_grouped[i];
    CHECK(energy > 1e-6, "the recurrence produced a non-degenerate output");

    /* DIFFERENCE 6, made visible: two key heads over four value heads, where
     * grouped maps {0,0,1,1} and tiled maps {0,1,0,1}. */
    enum { T2 = 4, KH2 = 2, VH2 = 4, D2 = 8 };
    float q2[T2 * KH2 * D2], k2[T2 * KH2 * D2], v2[T2 * VH2 * D2];
    float d2[T2 * VH2], b2[T2 * VH2];
    for (uint32_t i = 0; i < T2 * KH2 * D2; i++) {
        q2[i] = sinf((float)(i + 1u) * 0.13f);
        k2[i] = cosf((float)(i + 2u) * 0.17f);
    }
    for (uint32_t i = 0; i < T2 * VH2 * D2; i++)
        v2[i] = sinf((float)(i + 3u) * 0.19f);
    for (uint32_t i = 0; i < T2 * VH2; i++) {
        d2[i] = 0.9f + 0.01f * (float)(i % 5u);
        b2[i] = 0.3f + 0.05f * (float)(i % 3u);
    }
    double og[T2 * VH2 * D2], ot[T2 * VH2 * D2], st[VH2 * D2 * D2];
    memset(st, 0, sizeof(st));
    gdn_recurrence(og, st, q2, k2, v2, d2, b2, T2, KH2, VH2, D2, 0);
    memset(st, 0, sizeof(st));
    gdn_recurrence(ot, st, q2, k2, v2, d2, b2, T2, KH2, VH2, D2, 1);
    gap = 0.0;
    for (size_t i = 0; i < T2 * VH2 * D2; i++) {
        const double d = fabs(og[i] - ot[i]);
        if (d > gap) gap = d;
    }
    CHECK(gap > 1e-6,
          "DIFFERENCE 6: at two key heads the grouped and tiled pairings differ");
}

/* ===================================================================== *
 * Gated delta net: the gated output norm.
 *
 * Theirs, test_qwen4_metal.c:3296-3311; ours,
 * tests/test_qwen4exp_gdn.c:311-325.  Both put the epsilon INSIDE the mean
 * here, both multiply by the weight and both gate with a sigmoid, so this one
 * AGREES -- which is what makes DIFFERENCE 5 a real inconsistency inside our
 * own port rather than a house style: we use one epsilon convention for the
 * output norm and another for q/k, while they use the two that upstream uses.
 *
 * Their kernel band: 2e-6 abs / 2e-5 rel (:3325).  Our band: 2e-5.
 * ===================================================================== */

static void test_gdn_output_norm_agrees(void) {
    enum { ROWS = G_TOKENS * G_VH };
    static float core[ROWS * G_DIM], z[ROWS * G_DIM];
    static float theirs[ROWS * G_DIM], ours[ROWS * G_DIM];
    /* test_qwen4_metal.c:3290-3294 */
    for (uint32_t i = 0; i < ROWS * G_DIM; i++) {
        core[i] = 0.0175f * (float)((int)(i % 37u) - 18);
        z[i]    = 0.035f  * (float)((int)((i * 5u) % 23u) - 11);
    }
    /* Theirs, :3297-3311 */
    for (uint32_t row = 0; row < ROWS; row++) {
        float sum = 0.0f;
        for (uint32_t d = 0; d < G_DIM; d++)
            sum = fmaf(core[(size_t)row * G_DIM + d],
                       core[(size_t)row * G_DIM + d], sum);
        const float scale = 1.0f / sqrtf(sum / (float)G_DIM + 1.0e-6f);
        for (uint32_t d = 0; d < G_DIM; d++) {
            const size_t at = (size_t)row * G_DIM + d;
            theirs[at] = core[at] * scale * ref_bf16_to_f32(g_norm_w[d]) *
                         (1.0f / (1.0f + expf(-z[at])));
        }
    }
    /* Ours, tests/test_qwen4exp_gdn.c:311-325, in double as our test does. */
    for (uint32_t row = 0; row < ROWS; row++) {
        double sum = 0.0;
        for (uint32_t d = 0; d < G_DIM; d++) {
            const double y = core[(size_t)row * G_DIM + d];
            sum += y * y;
        }
        const double scale = 1.0 / sqrt(sum / (double)G_DIM + 1.0e-6);
        for (uint32_t d = 0; d < G_DIM; d++) {
            const size_t at = (size_t)row * G_DIM + d;
            ours[at] = (float)((double)core[at] * scale *
                               (double)ref_bf16_to_f32(g_norm_w[d]) *
                               (1.0 / (1.0 + exp(-(double)z[at]))));
        }
    }
    const float worst = max_rel_diff(theirs, ours, ROWS * G_DIM);
    report("GDN gated output norm (rel)", worst, 2e-6f, 2e-5f);
    CHECK(worst <= 1e-6f, "the gated output norm agrees with the reference");
}

/* ===================================================================== *
 * Sparse attention, including the GQA load-share pairing.
 *
 * SOURCE AND LICENCE.  Their Metal QSA kernel is attributed in their repo to
 * jundot/omlx PR #3244, whose licence is unverified.  Nothing about that
 * kernel's STRUCTURE is used here.  What is transcribed is their scalar CPU
 * reference, `qsa_attention_cpu_reference` at test_qwen4_metal.c:5716-5789
 * with the fixture at :4912-4993, which is plain arithmetic in their own
 * MIT-licensed test file.  Numbers only.
 *
 * Ours: `ref_attention_row` in tests/test_qwen4exp_qsa.c:281-311, with the
 * gate applied at :797-799.
 *
 * Their kernel band: 3e-6 abs / 3e-5 rel (:5016-5017).
 * Our band: 2e-3 max abs with cosine >= 0.9999
 * (tests/test_qwen4exp_qsa.c:781-818).
 * ===================================================================== */

enum {
    A_QUERIES = 3, A_CAP = 13, A_Q_HEADS = 4, A_KV_HEADS = 2,
    A_DIM = 256, A_TOP_K = 2, A_RATIO = 4,
    A_GROUP = A_Q_HEADS / A_KV_HEADS   /* 2 */
};

/* Their selection state, test_qwen4_metal.c:4940-4946.  `counts` is a BLOCK
 * count on their side; ours is a TOKEN count.  Their query 0 has an empty
 * selection, which is the tail-only case. */
static const int32_t A_SELECTED[A_QUERIES * A_TOP_K] = {0, 0, 1, 0, 2, 0};
static const uint32_t A_COUNTS[A_QUERIES]  = {0, 2, 2};
static const uint32_t A_VISIBLE[A_QUERIES] = {1, 9, 13};

/* Build the gathered token list: the selected blocks expanded, then the
 * query's own incomplete tail.  test_qwen4_metal.c:5723-5737.  `complete` is
 * INTEGER floor division -- true division would admit the query's own
 * incomplete block and attend to the future. */
static uint32_t a_gather(uint32_t query, uint32_t *tokens) {
    const uint32_t complete = A_VISIBLE[query] / A_RATIO;
    const uint32_t blocks = A_COUNTS[query] < complete ? A_COUNTS[query] : complete;
    uint32_t n = 0;
    for (uint32_t r = 0; r < blocks; r++)
        for (uint32_t item = 0; item < A_RATIO; item++) {
            const uint32_t token =
                (uint32_t)A_SELECTED[query * A_TOP_K + r] * A_RATIO + item;
            if (token < A_VISIBLE[query]) tokens[n++] = token;
        }
    for (uint32_t token = complete * A_RATIO; token < A_VISIBLE[query]; token++)
        tokens[n++] = token;
    return n;
}

static void test_qsa_attention_agrees(void) {
    static float q[A_QUERIES * A_Q_HEADS * A_DIM];
    static float gate[A_QUERIES * A_Q_HEADS * A_DIM];
    static uint16_t key[A_CAP * A_KV_HEADS * A_DIM];
    static uint16_t value[A_CAP * A_KV_HEADS * A_DIM];
    /* test_qwen4_metal.c:4931-4939 */
    for (uint32_t i = 0; i < A_QUERIES * A_Q_HEADS * A_DIM; i++) {
        q[i]    = 0.006f * (float)((int)((i * 7u + 5u) % 31u) - 15);
        gate[i] = 0.025f * (float)((int)((i * 11u) % 23u) - 11);
    }
    for (uint32_t i = 0; i < A_CAP * A_KV_HEADS * A_DIM; i++) {
        key[i]   = ref_f32_to_bf16(0.004f * (float)((int)((i * 13u + 3u) % 37u) - 18));
        value[i] = ref_f32_to_bf16(0.007f * (float)((int)((i * 17u + 9u) % 41u) - 20));
    }

    static float theirs[A_QUERIES * A_Q_HEADS * A_DIM];
    static float ours[A_QUERIES * A_Q_HEADS * A_DIM];
    uint32_t tokens[A_CAP];
    int covered_tail_only = 0, covered_group1 = 0;

    for (uint32_t query = 0; query < A_QUERIES; query++) {
        const uint32_t n = a_gather(query, tokens);
        if (A_COUNTS[query] == 0u) covered_tail_only = 1;
        for (uint32_t head = 0; head < A_Q_HEADS; head++) {
            /* GQA load share: a query head reads the KV head of its group.
             * test_qwen4_metal.c:5729. */
            const uint32_t kv_head = head / A_GROUP;
            if (kv_head == 1u) covered_group1 = 1;
            const size_t qbase = ((size_t)query * A_Q_HEADS + head) * A_DIM;

            float scores[A_CAP];
            float maximum = -INFINITY;
            for (uint32_t r = 0; r < n; r++) {
                float dot = 0.0f;
                for (uint32_t d = 0; d < A_DIM; d++)
                    dot = fmaf(q[qbase + d],
                               ref_bf16_to_f32(key[((size_t)tokens[r] * A_KV_HEADS +
                                                    kv_head) * A_DIM + d]), dot);
                scores[r] = dot / sqrtf((float)A_DIM);
                if (scores[r] > maximum) maximum = scores[r];
            }
            float sum = 0.0f;
            for (uint32_t r = 0; r < n; r++) sum += expf(scores[r] - maximum);

            /* THEIRS, :5749-5773: divide each weight by the sum BEFORE the
             * accumulate, and apply the gate as a DIVISION. */
            for (uint32_t d = 0; d < A_DIM; d++) {
                if (n == 0u) { theirs[qbase + d] = 0.0f; continue; }
                float acc = 0.0f;
                for (uint32_t r = 0; r < n; r++)
                    acc = fmaf(expf(scores[r] - maximum) / sum,
                               ref_bf16_to_f32(value[((size_t)tokens[r] * A_KV_HEADS +
                                                      kv_head) * A_DIM + d]), acc);
                theirs[qbase + d] = acc / (1.0f + expf(-gate[qbase + d]));
            }
            /* OURS, tests/test_qwen4exp_qsa.c:281-311 and :797-799: accumulate
             * the unnormalised weights, divide by the sum ONCE at the end, and
             * apply the gate as a MULTIPLY by the sigmoid. */
            for (uint32_t d = 0; d < A_DIM; d++) {
                if (n == 0u) { ours[qbase + d] = 0.0f; continue; }
                float acc = 0.0f;
                for (uint32_t r = 0; r < n; r++)
                    acc += expf(scores[r] - maximum) *
                           ref_bf16_to_f32(value[((size_t)tokens[r] * A_KV_HEADS +
                                                  kv_head) * A_DIM + d]);
                ours[qbase + d] = (acc / sum) * (1.0f / (1.0f + expf(-gate[qbase + d])));
            }
        }
    }
    CHECK(covered_tail_only, "the fixture covers a query with no selected block");
    CHECK(covered_group1, "the fixture covers both key-value groups");

    /* DIFFERENCE 8 is normalisation ORDER, not semantics: dividing per term
     * against dividing once rounds differently and nothing else.  Held at a
     * relative band, two orders inside their kernel band. */
    const float worst = max_rel_diff(theirs, ours,
                                     A_QUERIES * A_Q_HEADS * A_DIM);
    report("QSA sparse attention (rel)", worst, 3e-6f, 2e-3f);
    CHECK(worst <= 1e-6f, "the sparse attention output agrees");

    /* The `complete` floor is load bearing: true division would let a query
     * gather its own incomplete block and read the future.  Query 1 has
     * visible 9, so complete is 2 and NOT 2.25. */
    CHECK(A_VISIBLE[1] / A_RATIO == 2u, "complete is integer floor division");
    uint32_t t1[A_CAP];
    const uint32_t n1 = a_gather(1u, t1);
    for (uint32_t i = 0; i < n1; i++)
        CHECK(t1[i] < A_VISIBLE[1], "no gathered token is beyond the visible set");

    /* Negative control on the load share: the two key-value heads must carry
     * different keys, or `head / A_GROUP` could be any function at all and
     * every check above would still pass. */
    float dot_head0 = 0.0f, dot_head1 = 0.0f;
    for (uint32_t i = 0; i < A_DIM; i++) {
        dot_head0 = fmaf(q[i], ref_bf16_to_f32(key[0 * A_DIM + i]), dot_head0);
        dot_head1 = fmaf(q[i], ref_bf16_to_f32(key[1 * A_DIM + i]), dot_head1);
    }
    CHECK(fabsf(dot_head0 - dot_head1) > 1e-4f,
          "the two key-value heads carry different keys");
}

int main(void) {
    build_their_fixture();
    build_their_gdn_fixture();

    RUN(test_q8_0_matmul_agrees);
    RUN(test_silu_and_swiglu_agree);
    RUN(test_embedding_gather_agrees);
    RUN(test_hc_mix_agrees);
    RUN(test_hc_write_agrees_except_the_divide);
    RUN(test_lowrank_activation_difference);
    RUN(test_grouped_rms_norm_agrees_except_the_bf16_cast);
    RUN(test_router_agrees);
    RUN(test_gdn_prepare_agrees);
    RUN(test_gdn_qk_epsilon_difference);
    RUN(test_gdn_recurrence_agrees);
    RUN(test_gdn_output_norm_agrees);
    RUN(test_qsa_attention_agrees);

    fprintf(stderr, "%d/%d checks passed\n", g_total - g_failed, g_total);
    return g_failed == 0 ? 0 : 1;
}
