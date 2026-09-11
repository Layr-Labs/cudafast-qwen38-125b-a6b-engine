/*
 * qwen4exp MoE GPU tests: router, routed experts, shared expert and combine.
 *
 * Two layers of coverage.  The first runs the router at the production expert
 * count with small rows.  The second runs the expert GEMM at the production
 * row shapes for every quantisation pair the shipped artifacts actually use
 * (UD_Q4_K_XL_EXPERT_TYPES below is the measured table), each against a
 * plain-C double-precision dequantisation of the very same bytes, and then
 * runs one case with the down slab in a SECOND mapping to prove a block whose
 * expert tensors landed in different GGUF shards runs identically.
 *
 * The kernels get synthetic quantised weights and the float32 references
 * dequantise the very same bytes, so the only difference between the two sides
 * is arithmetic order.  Router identifiers and the Q5_1 element layout are
 * checked bit-exactly; the expert outputs are checked against the DESIGN.md
 * section 2 band for an expert GEMM at real quantisation.
 *
 * tests/qwen4exp_moe_mutants.sh proves these checks bite: it rewrites one
 * kernel line at a time and requires this test to fail.  Run it with
 * make test-qwen4exp-moe-mutants.
 */

#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#include "ds4.h"
#include "ds4_gpu.h"

bool ds4_log_is_tty(FILE *fp) {
    (void)fp;
    return false;
}

enum {
    TYPE_F32  = 0,
    TYPE_Q5_1 = 7,
    TYPE_Q8_0 = 8,
    TYPE_Q4_K = 12,
    TYPE_Q5_K = 13,
    TYPE_Q6_K = 14,
};

/* Production geometry: n_embd 2560, n_ff_exp 640.  The expert count is cut to
 * keep the image small; the router is covered at 512 experts above. */
enum {
    PROD_IN_DIM  = 2560,
    PROD_MID_DIM = 640,
    PROD_OUT_DIM = 2560,
    PROD_EXPERTS = 8,
    PROD_USED    = 4,
    PROD_TOKENS  = 2,
};

/* The routed expert quantisation of unsloth/Qwen3.8-Flash-Next-GGUF
 * UD-Q4_K_XL at 38bb39ee, read from the GGUF tensor tables of its four
 * shards (1224 tensors; shard 1 carries metadata only).  gate and up always
 * agree, so one column covers blk.N.ffn_gate_exps.weight and
 * blk.N.ffn_up_exps.weight; down is blk.N.ffn_down_exps.weight. */
typedef struct {
    uint32_t block;
    uint32_t gate_up;
    uint32_t down;
} expert_type_row;

static const expert_type_row UD_Q4_K_XL_EXPERT_TYPES[] = {
    {  0, TYPE_Q4_K, TYPE_Q5_1 }, {  1, TYPE_Q4_K, TYPE_Q5_1 },
    {  2, TYPE_Q5_K, TYPE_Q8_0 }, {  3, TYPE_Q4_K, TYPE_Q5_1 },
    {  4, TYPE_Q4_K, TYPE_Q8_0 }, {  5, TYPE_Q4_K, TYPE_Q5_1 },
    {  6, TYPE_Q4_K, TYPE_Q5_1 }, {  7, TYPE_Q4_K, TYPE_Q5_1 },
    {  8, TYPE_Q4_K, TYPE_Q5_1 }, {  9, TYPE_Q4_K, TYPE_Q5_1 },
    { 10, TYPE_Q4_K, TYPE_Q5_1 }, { 11, TYPE_Q4_K, TYPE_Q5_1 },
    { 12, TYPE_Q4_K, TYPE_Q5_1 }, { 13, TYPE_Q4_K, TYPE_Q5_1 },
    { 14, TYPE_Q4_K, TYPE_Q5_1 }, { 15, TYPE_Q4_K, TYPE_Q5_1 },
    { 16, TYPE_Q4_K, TYPE_Q5_1 }, { 17, TYPE_Q4_K, TYPE_Q5_1 },
    { 18, TYPE_Q4_K, TYPE_Q5_1 }, { 19, TYPE_Q4_K, TYPE_Q5_1 },
    { 20, TYPE_Q4_K, TYPE_Q5_1 }, { 21, TYPE_Q4_K, TYPE_Q5_1 },
    { 22, TYPE_Q4_K, TYPE_Q5_1 }, { 23, TYPE_Q4_K, TYPE_Q5_1 },
    { 24, TYPE_Q4_K, TYPE_Q5_1 }, { 25, TYPE_Q4_K, TYPE_Q5_1 },
    { 26, TYPE_Q4_K, TYPE_Q5_1 }, { 27, TYPE_Q4_K, TYPE_Q5_1 },
    { 28, TYPE_Q4_K, TYPE_Q5_1 }, { 29, TYPE_Q4_K, TYPE_Q5_1 },
    { 30, TYPE_Q4_K, TYPE_Q8_0 }, { 31, TYPE_Q4_K, TYPE_Q5_1 },
    { 32, TYPE_Q4_K, TYPE_Q5_1 }, { 33, TYPE_Q4_K, TYPE_Q5_1 },
    { 34, TYPE_Q4_K, TYPE_Q5_1 }, { 35, TYPE_Q4_K, TYPE_Q5_1 },
    { 36, TYPE_Q4_K, TYPE_Q5_1 }, { 37, TYPE_Q4_K, TYPE_Q5_1 },
    { 38, TYPE_Q4_K, TYPE_Q5_1 }, { 39, TYPE_Q4_K, TYPE_Q5_1 },
    { 40, TYPE_Q4_K, TYPE_Q5_1 }, { 41, TYPE_Q4_K, TYPE_Q5_1 },
    { 42, TYPE_Q4_K, TYPE_Q5_1 }, { 43, TYPE_Q4_K, TYPE_Q5_1 },
    { 44, TYPE_Q4_K, TYPE_Q5_1 }, { 45, TYPE_Q4_K, TYPE_Q5_1 },
    { 46, TYPE_Q4_K, TYPE_Q8_0 }, { 47, TYPE_Q4_K, TYPE_Q8_0 },
};

/* The same tensors in the other artifacts the loader admits, measured the
 * same way.  Baekpica MQ-Q5-SSD-PLE-BF16 stores 44 blocks of gate/up at Q4_K
 * and 4 at Q5_K; MQ-Q6-SSD-PLE-BF16 stores 44 at Q5_K and 4 at Q6_K.  The MTP
 * head ships its whole routed expert set, gate, up and down, at Q8_0. */
static const expert_type_row OTHER_RECIPE_EXPERT_TYPES[] = {
    { 0, TYPE_Q5_K, TYPE_Q5_1 },   /* MQ-Q5 and MQ-Q6 gate/up */
    { 0, TYPE_Q6_K, TYPE_Q5_1 },   /* MQ-Q6 gate/up */
    { 0, TYPE_Q8_0, TYPE_Q8_0 },   /* the MTP head */
};

enum {
    N_EXPERT      = 512,
    N_EXPERT_USED = 10,
    IN_DIM        = 256,
    MID_DIM       = 64,
    OUT_DIM       = 256,
    SHARED_MID    = 64,
    MOE_TOKENS    = 4,
    ROUTER_TOKENS = 11,
};

enum {
    Q4K_ROW_BYTES = (IN_DIM / 256) * 144,
    Q51_ROW_BYTES = (MID_DIM / 32) * 24,
    Q80_IN_ROW    = (IN_DIM / 32) * 34,
    Q80_MID_ROW   = (MID_DIM / 32) * 34,
};

static void fail(const char *what) {
    fprintf(stderr, "%s\n", what);
    exit(1);
}

static void require_ok(int ok, const char *what) {
    if (!ok) fail(what);
}

/* ------------------------------------------------------------------ */
/* float16 helpers.  Weights are built from half bit patterns directly so the
 * reference and the kernel read identical bytes. */

static float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1fu;
    const uint32_t man = h & 0x3ffu;
    union { uint32_t u; float f; } bits;
    if (exp == 0) {
        if (man == 0) { bits.u = sign; return bits.f; }
        bits.u = sign | 0x38800000u | (man << 13);   /* 2^-14 times man/1024 */
        union { uint32_t u; float f; } base;
        base.u = sign | 0x38800000u;
        return bits.f - base.f;
    }
    if (exp == 31) { bits.u = sign | 0x7f800000u | (man << 13); return bits.f; }
    bits.u = sign | ((exp + 112u) << 23) | (man << 13);
    return bits.f;
}

/* ------------------------------------------------------------------ */
/* Deterministic random source. */

static uint64_t rng_state = 0x243f6a8885a308d3ull;

static uint32_t rng_u32(void) {
    rng_state = rng_state * 6364136223846793005ull + 1442695040888963407ull;
    return (uint32_t)(rng_state >> 32);
}

static float rng_unit(void) {
    return (float)(rng_u32() >> 8) * (1.0f / 16777216.0f) * 2.0f - 1.0f;
}

/* A half in [2^-10, 2^-5) in magnitude: large enough to matter, small enough
 * that the dot products stay in a well-conditioned float32 range. */
static uint16_t rng_half_scale(void) {
    const uint32_t sign = (rng_u32() & 1u) << 15;
    const uint32_t exp = 5u + (rng_u32() % 5u);
    const uint32_t man = rng_u32() & 0x3ffu;
    return (uint16_t)(sign | (exp << 10) | man);
}

/* ------------------------------------------------------------------ */
/* float32 reference dequantisers, mirroring ggml-quants.c. */

static void q4_K_scale_min(const uint8_t *q, int j, int *sc, int *m) {
    if (j < 4) {
        *sc = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *sc = (q[j + 4] & 0x0f) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

static double ref_q4_K_value(const uint8_t *block, uint32_t k) {
    const uint8_t *xb = block + (k / 256u) * 144u;
    const uint32_t idx = k % 256u;
    const uint32_t group = idx / 32u;
    const uint32_t l = idx % 32u;
    const float d = f16_to_f32((uint16_t)(xb[0] | ((uint16_t)xb[1] << 8)));
    const float dmin = f16_to_f32((uint16_t)(xb[2] | ((uint16_t)xb[3] << 8)));
    int sc = 0, m = 0;
    q4_K_scale_min(xb + 4, (int)group, &sc, &m);
    const uint8_t byte = xb[4 + 12 + (group >> 1) * 32u + l];
    const uint32_t q = (group & 1u) ? (uint32_t)(byte >> 4) : (uint32_t)(byte & 0x0fu);
    return d * (float)sc * (float)q - dmin * (float)m;
}

/* ggml block_q5_1: 32 elements in 24 bytes.  See dequantize_row_q5_1(): the
 * low nibble of qs[j] is element j and the high nibble is element j + 16; the
 * fifth bit comes from bit j respectively bit j + 16 of the 32-bit qh. */
static double ref_q5_1_value(const uint8_t *block, uint32_t k) {
    const uint8_t *xb = block + (k / 32u) * 24u;
    const uint32_t idx = k % 32u;
    const float d = f16_to_f32((uint16_t)(xb[0] | ((uint16_t)xb[1] << 8)));
    const float m = f16_to_f32((uint16_t)(xb[2] | ((uint16_t)xb[3] << 8)));
    const uint32_t qh = (uint32_t)xb[4] | ((uint32_t)xb[5] << 8) |
                        ((uint32_t)xb[6] << 16) | ((uint32_t)xb[7] << 24);
    const uint32_t j = idx & 15u;
    const uint8_t byte = xb[8 + j];
    uint32_t q;
    if (idx < 16u) q = (uint32_t)(byte & 0x0fu) | (((qh >> j) & 1u) << 4);
    else q = (uint32_t)(byte >> 4) | (((qh >> (j + 16u)) & 1u) << 4);
    return (float)q * d + m;
}

static double ref_q8_0_value(const uint8_t *block, uint32_t k) {
    const uint8_t *xb = block + (k / 32u) * 34u;
    const float d = f16_to_f32((uint16_t)(xb[0] | ((uint16_t)xb[1] << 8)));
    return (double)d * (double)(int8_t)xb[2 + (k % 32u)];
}

/* ggml block_q5_K: 256 elements in 176 bytes.  It is Q4_K plus a high-bit
 * plane: the same packed six-bit scale/min pair, the same nibble, and bit
 * `group` of qh[l] as the fifth bit.  See dequantize_row_q5_K(). */
static double ref_q5_K_value(const uint8_t *block, uint32_t k) {
    const uint8_t *xb = block + (k / 256u) * 176u;
    const uint32_t idx = k % 256u;
    const uint32_t group = idx / 32u;
    const uint32_t l = idx % 32u;
    const float d = f16_to_f32((uint16_t)(xb[0] | ((uint16_t)xb[1] << 8)));
    const float dmin = f16_to_f32((uint16_t)(xb[2] | ((uint16_t)xb[3] << 8)));
    int sc = 0, m = 0;
    q4_K_scale_min(xb + 4, (int)group, &sc, &m);
    const uint8_t *qh = xb + 4 + 12;
    const uint8_t *qs = qh + 32;
    const uint8_t byte = qs[(group >> 1) * 32u + l];
    uint32_t q = (group & 1u) ? (uint32_t)(byte >> 4) : (uint32_t)(byte & 0x0fu);
    if (qh[l] & (uint8_t)(1u << group)) q += 16u;
    return (double)d * (double)sc * (double)q - (double)dmin * (double)m;
}

/* ggml block_q6_K: 256 elements in 210 bytes.  Four quarters per 128
 * elements, each taking a nibble from ql and two high bits from qh, one int8
 * scale per 16 elements and one f16 block scale.  The quant is signed by the
 * fixed -32 bias.  See dequantize_row_q6_K(). */
static double ref_q6_K_value(const uint8_t *block, uint32_t k) {
    const uint8_t *xb = block + (k / 256u) * 210u;
    const uint32_t idx = k % 256u;
    const uint32_t n128 = idx >> 7;
    const uint32_t r = idx & 127u;
    const uint32_t l = r & 31u;
    const uint32_t quarter = r >> 5;
    const uint8_t *ql = xb + n128 * 64u;
    const uint8_t *qh = xb + 128 + n128 * 32u;
    const int8_t *scales = (const int8_t *)(xb + 192) + n128 * 8u;
    const float d = f16_to_f32((uint16_t)(xb[208] | ((uint16_t)xb[209] << 8)));
    const uint32_t hi = ((uint32_t)qh[l] >> (quarter * 2u)) & 3u;
    const uint32_t lo = (quarter & 1u) ? (uint32_t)ql[32u + l] : (uint32_t)ql[l];
    const uint32_t q = (quarter < 2u) ? ((lo & 0x0fu) | (hi << 4))
                                      : ((lo >> 4) | (hi << 4));
    const int sc = scales[l / 16u + quarter * 2u];
    return (double)d * (double)sc * (double)((int32_t)q - 32);
}

static double ref_value(uint32_t type, const uint8_t *row, uint32_t k) {
    switch (type) {
    case TYPE_Q4_K: return ref_q4_K_value(row, k);
    case TYPE_Q5_K: return ref_q5_K_value(row, k);
    case TYPE_Q6_K: return ref_q6_K_value(row, k);
    case TYPE_Q5_1: return ref_q5_1_value(row, k);
    case TYPE_Q8_0: return ref_q8_0_value(row, k);
    default: {
        float v;
        memcpy(&v, row + (size_t)k * sizeof(float), sizeof(v));
        return (double)v;
    }
    }
}

/* Bytes one row of `elems` elements occupies at `type`. */
static uint64_t type_row_bytes(uint32_t type, uint32_t elems) {
    switch (type) {
    case TYPE_Q5_1: return (uint64_t)(elems / 32u) * 24u;
    case TYPE_Q8_0: return (uint64_t)(elems / 32u) * 34u;
    case TYPE_Q4_K: return (uint64_t)(elems / 256u) * 144u;
    case TYPE_Q5_K: return (uint64_t)(elems / 256u) * 176u;
    case TYPE_Q6_K: return (uint64_t)(elems / 256u) * 210u;
    default: return (uint64_t)elems * sizeof(float);
    }
}

static const char *type_name(uint32_t type) {
    switch (type) {
    case TYPE_Q5_1: return "Q5_1";
    case TYPE_Q8_0: return "Q8_0";
    case TYPE_Q4_K: return "Q4_K";
    case TYPE_Q5_K: return "Q5_K";
    case TYPE_Q6_K: return "Q6_K";
    default: return "F32";
    }
}

/* ------------------------------------------------------------------ */
/* Router reference: top-k on the raw float32 logits, ties to the lower expert
 * index, then a softmax over the selected logits only. */

typedef struct { float score; int32_t index; } router_entry;

static int router_cmp(const void *a, const void *b) {
    const router_entry *ea = a, *eb = b;
    if (ea->score > eb->score) return -1;
    if (ea->score < eb->score) return 1;
    return ea->index < eb->index ? -1 : 1;
}

static void ref_router(const float *logits, int32_t *selected, float *weights) {
    router_entry entries[N_EXPERT];
    for (int i = 0; i < N_EXPERT; i++) {
        entries[i].score = logits[i];
        entries[i].index = i;
    }
    qsort(entries, N_EXPERT, sizeof(entries[0]), router_cmp);
    float m = -FLT_MAX;
    for (int i = 0; i < N_EXPERT_USED; i++) {
        selected[i] = entries[i].index;
        if (entries[i].score > m) m = entries[i].score;
    }
    float sum = 0.0f;
    for (int i = 0; i < N_EXPERT_USED; i++) {
        weights[i] = expf(logits[selected[i]] - m);
        sum += weights[i];
    }
    for (int i = 0; i < N_EXPERT_USED; i++) weights[i] /= sum;
}

/* ------------------------------------------------------------------ */

static double rel_frobenius(const float *a, const float *b, size_t n) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < n; i++) {
        const double d = (double)a[i] - (double)b[i];
        num += d * d;
        den += (double)b[i] * (double)b[i];
    }
    if (den == 0.0) return num == 0.0 ? 0.0 : DBL_MAX;
    return sqrt(num / den);
}

/* ------------------------------------------------------------------ */
/* Model image.  Offsets are 64-byte aligned, matching a GGUF tensor layout. */

#define ALIGN64(x) (((x) + 63u) & ~(uint64_t)63u)

static const uint64_t GATE_EXPERT_BYTES = (uint64_t)MID_DIM * Q4K_ROW_BYTES;
static const uint64_t UP_EXPERT_BYTES   = (uint64_t)MID_DIM * Q4K_ROW_BYTES;
static const uint64_t DOWN_EXPERT_BYTES = (uint64_t)OUT_DIM * Q51_ROW_BYTES;

/* ------------------------------------------------------------------ */
/* Production-shape expert GEMM cases.
 *
 * One image holds a gate and an up slab in every type the artifacts use for
 * gate/up, and a down slab in every type they use for down.  Each measured
 * pair then runs at n_embd 2560 and n_ff_exp 640 against a double-precision
 * dequantisation of the same bytes.  A second mapping holds a copy of the
 * Q5_1 down slab so the split-shard path is exercised too. */

static const uint32_t PROD_GATE_UP_TYPES[] = {
    TYPE_Q4_K, TYPE_Q5_K, TYPE_Q6_K, TYPE_Q8_0,
};
static const uint32_t PROD_DOWN_TYPES[] = { TYPE_Q5_1, TYPE_Q8_0 };

static uint32_t prod_type_slot(const uint32_t *types, uint32_t n, uint32_t type) {
    for (uint32_t i = 0; i < n; i++) if (types[i] == type) return i;
    fail("the measured table names a type the production case does not build");
    return 0;
}

/* Give every f16 scale field in one row a sane magnitude.  The quantised
 * payload bytes stay as the random fill left them: the reference reads the
 * same bytes, so the only difference between the two sides is arithmetic. */
static void prod_seed_row_scales(uint8_t *row, uint32_t type, uint32_t elems) {
    switch (type) {
    case TYPE_Q4_K:
    case TYPE_Q5_K: {
        const uint64_t bytes = type == TYPE_Q4_K ? 144u : 176u;
        for (uint32_t b = 0; b < elems / 256u; b++) {
            const uint16_t d = rng_half_scale();
            const uint16_t dmin = rng_half_scale();
            uint8_t *blk = row + (uint64_t)b * bytes;
            blk[0] = (uint8_t)(d & 0xff);    blk[1] = (uint8_t)(d >> 8);
            blk[2] = (uint8_t)(dmin & 0xff); blk[3] = (uint8_t)(dmin >> 8);
        }
        break;
    }
    case TYPE_Q6_K:
        for (uint32_t b = 0; b < elems / 256u; b++) {
            const uint16_t d = rng_half_scale();
            uint8_t *blk = row + (uint64_t)b * 210u;
            blk[208] = (uint8_t)(d & 0xff); blk[209] = (uint8_t)(d >> 8);
        }
        break;
    case TYPE_Q5_1:
        for (uint32_t b = 0; b < elems / 32u; b++) {
            const uint16_t d = rng_half_scale();
            const uint16_t m = rng_half_scale();
            uint8_t *blk = row + (uint64_t)b * 24u;
            blk[0] = (uint8_t)(d & 0xff); blk[1] = (uint8_t)(d >> 8);
            blk[2] = (uint8_t)(m & 0xff); blk[3] = (uint8_t)(m >> 8);
        }
        break;
    case TYPE_Q8_0:
        for (uint32_t b = 0; b < elems / 32u; b++) {
            const uint16_t d = rng_half_scale();
            uint8_t *blk = row + (uint64_t)b * 34u;
            blk[0] = (uint8_t)(d & 0xff); blk[1] = (uint8_t)(d >> 8);
        }
        break;
    default:
        fail("unexpected production expert type");
    }
}

/* The routed expert reference, in double, reading the model bytes directly. */
static void prod_reference(const uint8_t *gate, const uint8_t *up,
                           const uint8_t *down,
                           uint32_t gate_type, uint32_t down_type,
                           const float *x,
                           const int32_t *selected, const float *weights,
                           float *out) {
    const uint64_t gate_row = type_row_bytes(gate_type, PROD_IN_DIM);
    const uint64_t down_row = type_row_bytes(down_type, PROD_MID_DIM);
    const uint64_t gate_expert = (uint64_t)PROD_MID_DIM * gate_row;
    const uint64_t down_expert = (uint64_t)PROD_OUT_DIM * down_row;
    double *mid = calloc((size_t)PROD_USED * PROD_MID_DIM, sizeof(double));
    if (!mid) fail("reference mid allocation");

    for (uint32_t t = 0; t < PROD_TOKENS; t++) {
        const float *token_x = x + (size_t)t * PROD_IN_DIM;
        for (uint32_t slot = 0; slot < PROD_USED; slot++) {
            const int32_t e = selected[t * PROD_USED + slot];
            const double w = weights[t * PROD_USED + slot];
            for (uint32_t r = 0; r < PROD_MID_DIM; r++) {
                const uint8_t *grow = gate + (uint64_t)e * gate_expert + (uint64_t)r * gate_row;
                const uint8_t *urow = up + (uint64_t)e * gate_expert + (uint64_t)r * gate_row;
                double g = 0.0, u = 0.0;
                for (uint32_t k = 0; k < PROD_IN_DIM; k++) {
                    const double xv = token_x[k];
                    g += ref_value(gate_type, grow, k) * xv;
                    u += ref_value(gate_type, urow, k) * xv;
                }
                mid[slot * PROD_MID_DIM + r] = (g / (1.0 + exp(-g))) * u * w;
            }
        }
        for (uint32_t r = 0; r < PROD_OUT_DIM; r++) {
            double acc = 0.0;
            for (uint32_t slot = 0; slot < PROD_USED; slot++) {
                const int32_t e = selected[t * PROD_USED + slot];
                const uint8_t *drow = down + (uint64_t)e * down_expert + (uint64_t)r * down_row;
                const double *y = mid + (size_t)slot * PROD_MID_DIM;
                for (uint32_t k = 0; k < PROD_MID_DIM; k++)
                    acc += ref_value(down_type, drow, k) * y[k];
            }
            out[(size_t)t * PROD_OUT_DIM + r] = (float)acc;
        }
    }
    free(mid);
}

/* The kernels dot an exactly dequantised weight against a Q8_0-QUANTISED
 * activation: the contract's two integer terms per group,
 * wa*xscale*dot + wb*xscale*sum, add up to exactly that, because a group's
 * weight is wa*q + wb element by element.  So the reference does not need the
 * group decomposition -- it needs the same quantised activation.  Quantising
 * it here separates two questions that were previously answered together: is
 * the group arithmetic right (this, to float rounding), and how much does
 * quantising the activation cost (the production cases below, against an exact
 * activation, in the section 2 band).
 *
 * The rule is ds4_cuda_qwen4exp.cu's qwen4exp_quantize_rows_kernel and Metal's
 * twin: amax over 32, d = amax/127, round to nearest, clamp to [-128, 127]. */
static void ref_q8_0_roundtrip(const float *src, float *dst, int n) {
    for (int i0 = 0; i0 < n; i0 += 32) {
        const int len = n - i0 < 32 ? n - i0 : 32;
        float amax = 0.0f;
        for (int i = 0; i < len; i++) {
            const float a = fabsf(src[i0 + i]);
            if (a > amax) amax = a;
        }
        const float d = amax / 127.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        for (int i = 0; i < len; i++) {
            int q = (int)lrintf(src[i0 + i] * id);
            q = q > 127 ? 127 : (q < -128 ? -128 : q);
            dst[i0 + i] = (float)q * d;
        }
    }
}

/* ------------------------------------------------------------------ */
/* The staged shared expert against the per-row shared expert.
 *
 * The staged kernels give one BLOCK one output row: all 256 threads decode
 * that row's groups once into shared memory, and then each of the eight warps
 * takes its own tile of eight tokens, so 64 tokens share one decode where the
 * per-row kernel gave one warp one row and eight tokens.  That is a schedule
 * change and a caching change, not an arithmetic change: for one (row, token)
 * it is still lane l that accumulates groups l, l + 32, l + 64, ... in
 * ascending order into one private float, and still the same warp_sum_f32 that
 * folds the lanes.
 *
 * Nothing but a bit comparison settles that, so this runs the SAME rows
 * through both kernels -- DS4_QWEN4EXP_SHARED_STAGE=0 selects the untouched
 * per-row path -- and requires the gate/up output and the final sum to be
 * identical to the last bit.  Widths are swept across the per-row tile
 * boundary (8), across the staged block's 64-token span and across both
 * tails, because a token's arithmetic must not depend on how many neighbours
 * ride in its warp or its block.
 *
 * Every weight type the decoder returns is run through it: Q8_0 is what the
 * checkpoint's shared expert carries, Q6_K is the only type whose group comes
 * back in two halves, and a group's two halves are the second arm of the fold
 * the staged path has to reproduce. */
/* 1024 is the prefill chunk the tower runs, and it is the width the tile has
 * to be right at; the sweep below it covers both sides of every gate --
 * the per-row tile boundary (8), the staged block's span (64), the tensor-core
 * token tile (32) and the 64-token width gate -- and the tails between them. */
enum { STAGE_AB_TOKENS = 1024 };

static const uint32_t STAGE_AB_WIDTHS[] = {
    1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 21, 31, 32, 33, 63, 64, 65, 127, 128, 129,
    200, 255, 256, 511, 512, 1023, STAGE_AB_TOKENS,
};

/* The tensor-core shared expert against the same per-row oracle.
 *
 * The tile changes the SCHEDULE of the float sum -- one thread now owns a
 * whole (row, token) reduction where thirty-two lanes used to share it -- so
 * unlike the staged kernels it cannot claim exactness by "the launch shape
 * moved and nothing else did".  It claims it by CONSTRUCTION instead: it walks
 * the K axis class major, finishes the ascending chain over the groups
 * c, c + 32, c + 64 ... before it starts the next class, and folds the
 * thirty-two class partials with the streaming pairwise sum that reproduces
 * warp_sum_f32's __shfl_down tree leaf for leaf.  Only a memcmp settles
 * whether the construction holds, and only a launch counter settles whether
 * the tile ran at all -- every reason it can decline is silent, and a
 * comparison of the dp4a path against itself passes for free.
 *
 * The magnitude of any difference is printed whether or not it is zero,
 * because "how far did the numbers move" is the question a golden cares
 * about and pass/fail does not answer it. */
extern unsigned long long ds4_gpu_qwen4exp_shared_mma_launches;

static void diff_stats(const float *a, const float *b, size_t n,
                       size_t *bad, double *max_abs, double *max_rel) {
    *bad = 0; *max_abs = 0.0; *max_rel = 0.0;
    for (size_t i = 0; i < n; i++) {
        if (memcmp(&a[i], &b[i], sizeof(float)) == 0) continue;
        (*bad)++;
        const double d = fabs((double)a[i] - (double)b[i]);
        const double m = fabs((double)a[i]) > fabs((double)b[i])
                       ? fabs((double)a[i]) : fabs((double)b[i]);
        if (d > *max_abs) *max_abs = d;
        if (m > 0.0 && d / m > *max_rel) *max_rel = d / m;
    }
}

/* pass 0 is always the untouched per-row kernels.  pass 1 is whatever
 * (env1, val1) selects -- NULL for "clear the environment and take whatever
 * the shipping default picks at this width", which is the case that proves
 * the sub-gate widths are unchanged. */
static void run_shared_stage_case_env(const ds4_gpu_qwen4exp_slab *router_slab,
                                  const ds4_gpu_qwen4exp_slab *gate_slab,
                                  const ds4_gpu_qwen4exp_slab *up_slab,
                                  const ds4_gpu_qwen4exp_slab *down_slab,
                                  uint32_t in_dim, uint32_t shared_mid,
                                  uint32_t out_dim,
                                  ds4_gpu_tensor *x_t,
                                  const char *what,
                                  const char *env1, const char *val1,
                                  const char *how,
                                  unsigned long long *mma_delta) {
    const size_t out_n = (size_t)STAGE_AB_TOKENS * out_dim;
    const size_t mid_n = (size_t)STAGE_AB_TOKENS * shared_mid;
    ds4_gpu_tensor *out_t = ds4_gpu_tensor_alloc((uint64_t)out_n * sizeof(float));
    ds4_gpu_tensor *mid_t = ds4_gpu_tensor_alloc((uint64_t)mid_n * sizeof(float));
    ds4_gpu_tensor *gsc_t = ds4_gpu_tensor_alloc((uint64_t)STAGE_AB_TOKENS * sizeof(float));
    float *zero = calloc(out_n, sizeof(float));
    /* The gate/up kernel WRITES mid and the down kernel reads it, so a token
     * the gate/up kernel forgot to write would otherwise be compared against
     * the value the previous pass left there.  Poison it before every call. */
    float *poison = malloc(mid_n * sizeof(float));
    if (poison) for (size_t i = 0; i < mid_n; i++) poison[i] = -7777.0f;
    float *out_got[2], *mid_got[2];
    for (int i = 0; i < 2; i++) {
        out_got[i] = calloc(out_n, sizeof(float));
        mid_got[i] = calloc(mid_n, sizeof(float));
    }
    require_ok(out_t && mid_t && gsc_t && zero && poison && out_got[0] &&
               out_got[1] && mid_got[0] && mid_got[1],
               "staged shared expert allocation");

    const size_t n_widths = sizeof(STAGE_AB_WIDTHS) / sizeof(STAGE_AB_WIDTHS[0]);
    for (size_t wi = 0; wi < n_widths; wi++) {
        const uint32_t w = STAGE_AB_WIDTHS[wi];
        for (int pass = 0; pass < 2; pass++) {
            /* pass 0 is the per-row kernels, pass 1 the staged ones. */
            /* "1" asks for the staged kernels at EVERY width, past the
             * token count below which the launcher prefers the per-row pair
             * on speed; the tails this sweep is here to check live below it. */
            unsetenv("DS4_QWEN4EXP_SHARED_STAGE");
            unsetenv("DS4_QWEN4EXP_SHARED_MMA");
            if (pass == 0) {
                setenv("DS4_QWEN4EXP_SHARED_STAGE", "0", 1);
            } else if (env1) {
                setenv(env1, val1, 1);
            }
            const unsigned long long mma_before =
                ds4_gpu_qwen4exp_shared_mma_launches;
            /* The down kernel ADDS into out, so both passes must start from
             * the same zeros or the comparison is meaningless. */
            require_ok(ds4_gpu_tensor_write(out_t, 0, zero,
                                            (uint64_t)w * out_dim * sizeof(float)),
                       "staged shared expert output clear");
            require_ok(ds4_gpu_tensor_write(mid_t, 0, poison,
                                            (uint64_t)w * shared_mid * sizeof(float)),
                       "staged shared expert mid poison");
            require_ok(ds4_gpu_qwen4exp_shared_expert_tensor(
                           out_t, mid_t, gsc_t, router_slab, gate_slab, up_slab,
                           down_slab, in_dim, shared_mid, out_dim, x_t, w),
                       what);
            require_ok(ds4_gpu_tensor_read(out_t, 0, out_got[pass],
                                           (uint64_t)w * out_dim * sizeof(float)),
                       "staged shared expert output read");
            require_ok(ds4_gpu_tensor_read(mid_t, 0, mid_got[pass],
                                           (uint64_t)w * shared_mid * sizeof(float)),
                       "staged shared expert mid read");
            if (pass == 1 && mma_delta)
                *mma_delta += ds4_gpu_qwen4exp_shared_mma_launches - mma_before;
        }
        unsetenv("DS4_QWEN4EXP_SHARED_STAGE");
        unsetenv("DS4_QWEN4EXP_SHARED_MMA");

        size_t bad_mid = 0, bad_out = 0;
        double abs_mid = 0.0, rel_mid = 0.0, abs_out = 0.0, rel_out = 0.0;
        diff_stats(mid_got[0], mid_got[1], (size_t)w * shared_mid,
                   &bad_mid, &abs_mid, &rel_mid);
        diff_stats(out_got[0], out_got[1], (size_t)w * out_dim,
                   &bad_out, &abs_out, &rel_out);
        if (bad_mid != 0) {
            fprintf(stderr, "staged shared gate/up (%s) at width %u: %zu of %zu "
                            "outputs differ, max abs %.6e, max rel %.6e\n",
                    what, w, bad_mid, (size_t)w * shared_mid, abs_mid, rel_mid);
            fail("the staged shared gate/up changed a number");
        }
        if (bad_out != 0) {
            fprintf(stderr, "staged shared down (%s) at width %u: %zu of %zu "
                            "outputs differ, max abs %.6e, max rel %.6e\n",
                    what, w, bad_out, (size_t)w * out_dim, abs_out, rel_out);
            fail("the staged shared down changed a number");
        }
    }
    printf("shared expert %s against per-row, %s (in %u, mid %u, out %u): "
           "%zu widths to %u tokens, 0 of %zu gate/up and 0 of %zu down "
           "outputs differ, max abs 0, max rel 0\n",
           how, what, in_dim, shared_mid, out_dim, n_widths,
           (unsigned)STAGE_AB_TOKENS,
           (size_t)STAGE_AB_TOKENS * shared_mid,
           (size_t)STAGE_AB_TOKENS * out_dim);

    for (int i = 0; i < 2; i++) { free(mid_got[i]); free(out_got[i]); }
    free(poison);
    free(zero);
    ds4_gpu_tensor_free(gsc_t);
    ds4_gpu_tensor_free(mid_t);
    ds4_gpu_tensor_free(out_t);
}

/* The staged kernels, as before: DS4_QWEN4EXP_SHARED_STAGE=1 asks for them at
 * every width and stands the tile down at the same time. */
static void run_shared_stage_case(const ds4_gpu_qwen4exp_slab *router_slab,
                                  const ds4_gpu_qwen4exp_slab *gate_slab,
                                  const ds4_gpu_qwen4exp_slab *up_slab,
                                  const ds4_gpu_qwen4exp_slab *down_slab,
                                  uint32_t in_dim, uint32_t shared_mid,
                                  uint32_t out_dim,
                                  ds4_gpu_tensor *x_t,
                                  const char *what) {
    run_shared_stage_case_env(router_slab, gate_slab, up_slab, down_slab,
                              in_dim, shared_mid, out_dim, x_t, what,
                              "DS4_QWEN4EXP_SHARED_STAGE", "1", "staged", NULL);
}

/* The tile, forced at every width so the sweep reaches the tails it would
 * otherwise never see, and then the shipping default with a cleared
 * environment -- which is what the ranked harness runs and which must pick
 * the OLD kernels below QWEN4EXP_MMA_MIN_TOKENS and the tile above it. */
static void run_shared_mma_case(const ds4_gpu_qwen4exp_slab *router_slab,
                                const ds4_gpu_qwen4exp_slab *gate_slab,
                                const ds4_gpu_qwen4exp_slab *up_slab,
                                const ds4_gpu_qwen4exp_slab *down_slab,
                                uint32_t in_dim, uint32_t shared_mid,
                                uint32_t out_dim,
                                ds4_gpu_tensor *x_t,
                                const char *what, int gu_tile, int dn_tile) {
    unsigned long long forced = 0, dflt = 0, off = 0;
    run_shared_stage_case_env(router_slab, gate_slab, up_slab, down_slab,
                              in_dim, shared_mid, out_dim, x_t, what,
                              "DS4_QWEN4EXP_SHARED_MMA", "1",
                              "tensor-core tile at every width", &forced);
    /* The kill switch: the class-major tile stands down and the call lands on
     * the Q8_0 tile that was here before it, which is the current tip's
     * dispatch exactly.  That path has to hold the same bits too, or the
     * switch is not a switch. */
    run_shared_stage_case_env(router_slab, gate_slab, up_slab, down_slab,
                              in_dim, shared_mid, out_dim, x_t, what,
                              "DS4_QWEN4EXP_SHARED_MMA", "0",
                              "the tile it replaces (SHARED_MMA=0)", &off);
    if (off != 0ull)
        fail("DS4_QWEN4EXP_SHARED_MMA=0 did not stand the class-major tile down");
    run_shared_stage_case_env(router_slab, gate_slab, up_slab, down_slab,
                              in_dim, shared_mid, out_dim, x_t, what,
                              NULL, NULL, "shipping default", &dflt);
    size_t wide = 0;
    for (size_t i = 0; i < sizeof(STAGE_AB_WIDTHS) / sizeof(STAGE_AB_WIDTHS[0]);
         i++) {
        if (STAGE_AB_WIDTHS[i] >= 64u) wide++;
    }
    printf("  tile launches: %llu forced over %zu widths, %llu at the "
           "shipping default over the %zu widths at or above the 64-token "
           "gate (two launches a width: gate/up and down)\n",
           forced, sizeof(STAGE_AB_WIDTHS) / sizeof(STAGE_AB_WIDTHS[0]),
           dflt, wide);
    const unsigned long long per_width =
        (unsigned long long)(gu_tile ? 1 : 0) + (unsigned long long)(dn_tile ? 1 : 0);
    const size_t n_w = sizeof(STAGE_AB_WIDTHS) / sizeof(STAGE_AB_WIDTHS[0]);
    if (forced != per_width * (unsigned long long)n_w)
        fail("the tile did not run at exactly the projections it must take "
             "when it is forced at every width");
    if (dflt != per_width * (unsigned long long)wide)
        fail("the shipping default did not take the tile at exactly the "
             "widths at or above the gate");
}

static void run_shared_stage_cases(const uint8_t *model, uint64_t model_bytes,
                                   uint64_t gate_offset, uint64_t up_offset,
                                   uint64_t down_offset,
                                   uint64_t sh_router_offset,
                                   uint64_t sh_gate_offset,
                                   uint64_t sh_up_offset,
                                   uint64_t sh_down_offset,
                                   uint64_t sh_gate_q5k_offset,
                                   uint64_t sh_up_q5k_offset,
                                   uint64_t sh_gate_q6k_offset,
                                   uint64_t sh_up_q6k_offset,
                                   uint64_t sh_q5k_row, uint64_t sh_q6k_row) {
    float *x = calloc((size_t)STAGE_AB_TOKENS * IN_DIM, sizeof(float));
    if (!x) fail("staged shared expert activation allocation");
    for (size_t i = 0; i < (size_t)STAGE_AB_TOKENS * IN_DIM; i++)
        x[i] = rng_unit() * 0.5f;
    ds4_gpu_tensor *x_t = ds4_gpu_tensor_alloc(
            (uint64_t)STAGE_AB_TOKENS * IN_DIM * sizeof(float));
    require_ok(x_t != NULL, "staged shared expert activation tensor");
    require_ok(ds4_gpu_tensor_write(x_t, 0, x,
                                    (uint64_t)STAGE_AB_TOKENS * IN_DIM * sizeof(float)),
               "staged shared expert activation write");

    const ds4_gpu_qwen4exp_slab router = {
        model, model_bytes, sh_router_offset, 0, IN_DIM * sizeof(float), TYPE_F32 };
    const ds4_gpu_qwen4exp_slab q80_gate = {
        model, model_bytes, sh_gate_offset, 0, Q80_IN_ROW, TYPE_Q8_0 };
    const ds4_gpu_qwen4exp_slab q80_up = {
        model, model_bytes, sh_up_offset, 0, Q80_IN_ROW, TYPE_Q8_0 };
    const ds4_gpu_qwen4exp_slab q80_down = {
        model, model_bytes, sh_down_offset, 0, Q80_MID_ROW, TYPE_Q8_0 };
    /* The routed slabs carry [IN_DIM, MID_DIM] gate/up rows and
     * [MID_DIM, OUT_DIM] down rows, which is exactly the shared expert's
     * shape here, so expert zero doubles as a Q4_K / Q5_1 shared expert. */
    const ds4_gpu_qwen4exp_slab q4k_gate = {
        model, model_bytes, gate_offset, 0, Q4K_ROW_BYTES, TYPE_Q4_K };
    const ds4_gpu_qwen4exp_slab q4k_up = {
        model, model_bytes, up_offset, 0, Q4K_ROW_BYTES, TYPE_Q4_K };
    const ds4_gpu_qwen4exp_slab q51_down = {
        model, model_bytes, down_offset, 0, Q51_ROW_BYTES, TYPE_Q5_1 };
    const ds4_gpu_qwen4exp_slab q5k_gate = {
        model, model_bytes, sh_gate_q5k_offset, 0, sh_q5k_row, TYPE_Q5_K };
    const ds4_gpu_qwen4exp_slab q5k_up = {
        model, model_bytes, sh_up_q5k_offset, 0, sh_q5k_row, TYPE_Q5_K };
    const ds4_gpu_qwen4exp_slab q6k_gate = {
        model, model_bytes, sh_gate_q6k_offset, 0, sh_q6k_row, TYPE_Q6_K };
    const ds4_gpu_qwen4exp_slab q6k_up = {
        model, model_bytes, sh_up_q6k_offset, 0, sh_q6k_row, TYPE_Q6_K };

    run_shared_stage_case(&router, &q80_gate, &q80_up, &q80_down,
                          IN_DIM, SHARED_MID, OUT_DIM, x_t,
                          "Q8_0 gate/up, Q8_0 down");
    run_shared_stage_case(&router, &q4k_gate, &q4k_up, &q51_down,
                          IN_DIM, SHARED_MID, OUT_DIM, x_t,
                          "Q4_K gate/up, Q5_1 down");
    run_shared_stage_case(&router, &q5k_gate, &q5k_up, &q80_down,
                          IN_DIM, SHARED_MID, OUT_DIM, x_t,
                          "Q5_K gate/up, Q8_0 down");
    run_shared_stage_case(&router, &q6k_gate, &q6k_up, &q80_down,
                          IN_DIM, SHARED_MID, OUT_DIM, x_t,
                          "Q6_K gate/up, Q8_0 down");

    /* The same four type pairs through the tensor-core tile.  Q6_K must
     * DECLINE -- its scale changes inside a group, so m16n8k32 cannot take
     * the group whole -- and the other three must run and match to the bit. */
    run_shared_mma_case(&router, &q80_gate, &q80_up, &q80_down,
                        IN_DIM, SHARED_MID, OUT_DIM, x_t,
                        "Q8_0 gate/up, Q8_0 down", 1, 1);
    run_shared_mma_case(&router, &q4k_gate, &q4k_up, &q51_down,
                        IN_DIM, SHARED_MID, OUT_DIM, x_t,
                        "Q4_K gate/up, Q5_1 down", 1, 1);
    run_shared_mma_case(&router, &q5k_gate, &q5k_up, &q80_down,
                        IN_DIM, SHARED_MID, OUT_DIM, x_t,
                        "Q5_K gate/up, Q8_0 down", 1, 1);
    run_shared_mma_case(&router, &q6k_gate, &q6k_up, &q80_down,
                        IN_DIM, SHARED_MID, OUT_DIM, x_t,
                        "Q6_K gate/up, Q8_0 down", 0, 1);

    ds4_gpu_tensor_free(x_t);
    free(x);
}

/* ------------------------------------------------------------------ */
/* Row invariance of the expert GEMMs.
 *
 * The prefill kernels share a decoded weight across a tile of rows, and the
 * routed pair list sorts the (token, slot) pairs by expert so a tile has one
 * expert to share.  Neither is allowed to move a number: a token in an N-row
 * call must get the same BITS it gets in a call of its own, or the tower stops
 * being row invariant and the speculative cycle's verify stops matching a
 * serial decode.
 *
 * The synthetic graph model leaves the routed expert slabs as holes, so the
 * width sweep in tests/test_qwen4exp_graph cannot see this operation at all.
 * This is where the routed MoE's row invariance is asserted.
 *
 * The routing is built so the tile logic is actually exercised:
 *   * tokens 0..15 share one set of logits, so ten experts carry sixteen pairs
 *     each -- two full tiles of eight;
 *   * token 16 repeats one of those, taking an expert to seventeen -- two full
 *     tiles and a tail of one;
 *   * tokens 17..20 route at random, which leaves most experts empty and some
 *     with a single pair.
 * The width, 21, is not a multiple of the tile, so the shared expert's own
 * tail is exercised too.
 */
/* ------------------------------------------------------------------ */
/* The F32 router projection at the production shape, prefill widths.
 *
 * ds4_gpu_matmul_f32_decode_rows_exact_tensor now takes a warp-tile kernel at
 * n_rows >= 8 (ds4_cuda.cu, matmul_f32_warp_tile_kernel) instead of the
 * one-column block tile.  The claim is that the two are BIT-IDENTICAL, not
 * merely close: this entry produces ffn_gate_inp's logits, whose consumer is a
 * top-10-of-512 selection, and a selection is discrete -- one ULP between two
 * nearly tied experts changes which experts run for that token and everything
 * downstream of it.
 *
 * So this sweeps prefill widths at the real shape (in 2560, out 512), runs the
 * projection once with DS4_F32_NO_WARP_TILE set (the kernel that shipped) and
 * once without (the warp tile), and requires
 *
 *   - every logit byte to be equal, and
 *   - the selected expert ids AND THEIR ORDER, and the softmax weights, to be
 *     equal after ds4_gpu_qwen4exp_router_select_tensor has run on each.
 *
 * The second check is not implied by the first for a future change that is
 * only nearly exact, which is the point of running it separately.  The case
 * also reports the tightest logit gap it saw at the top-k boundary, so the
 * margin the exactness has to protect is on the record rather than assumed.
 *
 * It runs at three shapes: the router's 2560 -> 512, the GDN alpha and beta
 * projections' 2560 -> 48 (same entry, same tile), and 2048 -> 512, whose
 * chain length is not the 10 that 2560 gives.  The last one exists because the
 * tile specialises the chain length when in_dim is 2560 and falls back to a
 * runtime count otherwise; without a shape that takes the runtime arm, a
 * mutant of that arm survives -- it did, on the first run of
 * tests/qwen4exp_router_f32_mutants.sh.
 */
enum { RF32_MAXIN = 2560, RF32_MAXOUT = 512, RF32_MAXROWS = 1024 };

static void run_router_f32_prefill_exact_case(int RF32_IN, int RF32_OUT) {
    const uint64_t wbytes = (uint64_t)RF32_OUT * RF32_IN * sizeof(float);
    float *wmap = mmap(NULL, wbytes, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANON, -1, 0);
    if (wmap == MAP_FAILED) fail("router f32 weight mmap");
    for (uint64_t i = 0; i < (uint64_t)RF32_OUT * RF32_IN; i++) {
        wmap[i] = rng_unit() * 0.05f;
    }

    float *x = calloc((size_t)RF32_MAXROWS * RF32_MAXIN, sizeof(float));
    float *la = calloc((size_t)RF32_MAXROWS * RF32_MAXOUT, sizeof(float));
    float *lb = calloc((size_t)RF32_MAXROWS * RF32_MAXOUT, sizeof(float));
    int32_t *sa = calloc((size_t)RF32_MAXROWS * N_EXPERT_USED, sizeof(int32_t));
    int32_t *sb = calloc((size_t)RF32_MAXROWS * N_EXPERT_USED, sizeof(int32_t));
    float *wa = calloc((size_t)RF32_MAXROWS * N_EXPERT_USED, sizeof(float));
    float *wb = calloc((size_t)RF32_MAXROWS * N_EXPERT_USED, sizeof(float));
    if (!x || !la || !lb || !sa || !sb || !wa || !wb) fail("router f32 alloc");
    for (size_t i = 0; i < (size_t)RF32_MAXROWS * RF32_MAXIN; i++) x[i] = rng_unit();

    ds4_gpu_tensor *x_t = ds4_gpu_tensor_alloc((uint64_t)RF32_MAXROWS * RF32_MAXIN * sizeof(float));
    ds4_gpu_tensor *la_t = ds4_gpu_tensor_alloc((uint64_t)RF32_MAXROWS * RF32_MAXOUT * sizeof(float));
    ds4_gpu_tensor *lb_t = ds4_gpu_tensor_alloc((uint64_t)RF32_MAXROWS * RF32_MAXOUT * sizeof(float));
    ds4_gpu_tensor *sa_t = ds4_gpu_tensor_alloc((uint64_t)RF32_MAXROWS * N_EXPERT_USED * sizeof(int32_t));
    ds4_gpu_tensor *sb_t = ds4_gpu_tensor_alloc((uint64_t)RF32_MAXROWS * N_EXPERT_USED * sizeof(int32_t));
    ds4_gpu_tensor *wa_t = ds4_gpu_tensor_alloc((uint64_t)RF32_MAXROWS * N_EXPERT_USED * sizeof(float));
    ds4_gpu_tensor *wb_t = ds4_gpu_tensor_alloc((uint64_t)RF32_MAXROWS * N_EXPERT_USED * sizeof(float));
    require_ok(x_t && la_t && lb_t && sa_t && sb_t && wa_t && wb_t,
               "router f32 tensor allocation");
    require_ok(ds4_gpu_tensor_write(x_t, 0, x,
                                    (uint64_t)RF32_MAXROWS * RF32_MAXIN * sizeof(float)),
               "router f32 activation write");

    static const uint32_t widths[] = {
        1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 11u, 12u, 13u, 15u, 16u, 17u, 18u,
        23u, 24u, 25u, 31u, 32u, 33u, 47u, 63u, 64u, 65u, 96u, 127u, 128u,
        129u, 170u, 171u, 172u, 255u, 256u, 257u, 511u, 512u, 1023u, 1024u,
    };
    const int n_widths = (int)(sizeof(widths) / sizeof(widths[0]));
    uint64_t differing = 0, checked = 0, sel_diff = 0, tiled = 0;
    double tightest = 1e30;

    for (int wi = 0; wi < n_widths; wi++) {
        const uint32_t rows = widths[wi];
        const uint64_t n = (uint64_t)rows * RF32_OUT;

        setenv("DS4_F32_NO_WARP_TILE", "1", 1);
        require_ok(ds4_gpu_matmul_f32_decode_rows_exact_tensor(
                       la_t, wmap, wbytes, 0, RF32_IN, RF32_OUT, x_t, rows),
                   "router f32 shipping-path projection");
        unsetenv("DS4_F32_NO_WARP_TILE");
        require_ok(ds4_gpu_matmul_f32_decode_rows_exact_tensor(
                       lb_t, wmap, wbytes, 0, RF32_IN, RF32_OUT, x_t, rows),
                   "router f32 warp-tile projection");
        if (rows >= 8u) tiled++;

        require_ok(ds4_gpu_tensor_read(la_t, 0, la, n * sizeof(float)), "router f32 read a");
        require_ok(ds4_gpu_tensor_read(lb_t, 0, lb, n * sizeof(float)), "router f32 read b");
        for (uint64_t i = 0; i < n; i++) {
            checked++;
            if (memcmp(&la[i], &lb[i], sizeof(float)) != 0) differing++;
        }

        require_ok(ds4_gpu_qwen4exp_router_select_tensor(sa_t, wa_t, la_t,
                                                         (uint32_t)RF32_OUT,
                                                         N_EXPERT_USED, rows),
                   "router f32 select a");
        require_ok(ds4_gpu_qwen4exp_router_select_tensor(sb_t, wb_t, lb_t,
                                                         (uint32_t)RF32_OUT,
                                                         N_EXPERT_USED, rows),
                   "router f32 select b");
        const uint64_t sn = (uint64_t)rows * N_EXPERT_USED;
        require_ok(ds4_gpu_tensor_read(sa_t, 0, sa, sn * sizeof(int32_t)), "router f32 sel a");
        require_ok(ds4_gpu_tensor_read(sb_t, 0, sb, sn * sizeof(int32_t)), "router f32 sel b");
        require_ok(ds4_gpu_tensor_read(wa_t, 0, wa, sn * sizeof(float)), "router f32 w a");
        require_ok(ds4_gpu_tensor_read(wb_t, 0, wb, sn * sizeof(float)), "router f32 w b");
        for (uint64_t i = 0; i < sn; i++) {
            if (sa[i] != sb[i]) sel_diff++;
            if (memcmp(&wa[i], &wb[i], sizeof(float)) != 0) sel_diff++;
        }

        /* How much room the selection actually had: the gap between the 10th
         * and 11th largest logit of each row, on the shipping path. */
        for (uint32_t r = 0; r < rows; r++) {
            const float *row = la + (size_t)r * RF32_OUT;
            float top[N_EXPERT_USED + 1];
            for (int i = 0; i <= N_EXPERT_USED; i++) top[i] = -FLT_MAX;
            for (int e = 0; e < RF32_OUT; e++) {
                float v = row[e];
                for (int i = 0; i <= N_EXPERT_USED; i++) {
                    if (v > top[i]) { const float t = top[i]; top[i] = v; v = t; }
                }
            }
            const double gap = (double)top[N_EXPERT_USED - 1] - (double)top[N_EXPERT_USED];
            if (gap < tightest) tightest = gap;
        }
    }

    if (differing) fail("router f32 warp tile changed a logit");
    if (sel_diff) fail("router f32 warp tile changed the expert selection");
    printf("router f32 projection (in %d, out %d): %d widths to %u tokens, "
           "%llu warp-tile widths, %llu of %llu logits differ, selected ids and "
           "order identical, tightest top-%d/%d logit gap %.3e\n",
           RF32_IN, RF32_OUT, n_widths, widths[n_widths - 1],
           (unsigned long long)tiled, (unsigned long long)differing,
           (unsigned long long)checked, N_EXPERT_USED, N_EXPERT_USED + 1,
           tightest);

    ds4_gpu_tensor_free(x_t); ds4_gpu_tensor_free(la_t); ds4_gpu_tensor_free(lb_t);
    ds4_gpu_tensor_free(sa_t); ds4_gpu_tensor_free(sb_t);
    ds4_gpu_tensor_free(wa_t); ds4_gpu_tensor_free(wb_t);
    free(x); free(la); free(lb); free(sa); free(sb); free(wa); free(wb);
    munmap(wmap, wbytes);
}

enum { INV_TOKENS = 21 };

static void run_row_invariance_case(const uint8_t *model,
                                    uint64_t model_bytes,
                                    uint64_t gate_offset,
                                    uint64_t up_offset,
                                    uint64_t down_offset,
                                    uint64_t sh_router_offset,
                                    uint64_t sh_gate_offset,
                                    uint64_t sh_up_offset,
                                    uint64_t sh_down_offset) {
    const uint64_t x_bytes = (uint64_t)INV_TOKENS * IN_DIM * sizeof(float);
    const uint64_t out_bytes = (uint64_t)INV_TOKENS * OUT_DIM * sizeof(float);
    const uint64_t mid_bytes =
        (uint64_t)INV_TOKENS * N_EXPERT_USED * MID_DIM * sizeof(float);

    float *x = calloc((size_t)INV_TOKENS * IN_DIM, sizeof(float));
    float *logits = calloc((size_t)INV_TOKENS * N_EXPERT, sizeof(float));
    float *wide = calloc((size_t)INV_TOKENS * OUT_DIM, sizeof(float));
    float *narrow = calloc((size_t)INV_TOKENS * OUT_DIM, sizeof(float));
    if (!x || !logits || !wide || !narrow) fail("invariance allocation");

    for (size_t i = 0; i < (size_t)INV_TOKENS * IN_DIM; i++) x[i] = rng_unit() * 0.5f;
    for (int e = 0; e < N_EXPERT; e++) {
        const float v = rng_unit() * 4.0f;
        for (int t = 0; t < 16; t++) logits[(size_t)t * N_EXPERT + e] = v;
    }
    memcpy(logits + (size_t)16 * N_EXPERT, logits, (size_t)N_EXPERT * sizeof(float));
    for (int t = 17; t < INV_TOKENS; t++) {
        for (int e = 0; e < N_EXPERT; e++) {
            logits[(size_t)t * N_EXPERT + e] = rng_unit() * 4.0f;
        }
    }

    ds4_gpu_tensor *logits_t = ds4_gpu_tensor_alloc((uint64_t)INV_TOKENS * N_EXPERT * sizeof(float));
    ds4_gpu_tensor *sel_t = ds4_gpu_tensor_alloc((uint64_t)INV_TOKENS * N_EXPERT_USED * sizeof(int32_t));
    ds4_gpu_tensor *w_t = ds4_gpu_tensor_alloc((uint64_t)INV_TOKENS * N_EXPERT_USED * sizeof(float));
    ds4_gpu_tensor *x_t = ds4_gpu_tensor_alloc(x_bytes);
    ds4_gpu_tensor *mid_t = ds4_gpu_tensor_alloc(mid_bytes);
    ds4_gpu_tensor *out_t = ds4_gpu_tensor_alloc(out_bytes);
    ds4_gpu_tensor *part_t = ds4_gpu_tensor_alloc(
            (uint64_t)INV_TOKENS * N_EXPERT_USED * OUT_DIM * sizeof(float));
    ds4_gpu_tensor *shmid_t = ds4_gpu_tensor_alloc((uint64_t)INV_TOKENS * SHARED_MID * sizeof(float));
    ds4_gpu_tensor *shgate_t = ds4_gpu_tensor_alloc((uint64_t)INV_TOKENS * sizeof(float));
    require_ok(logits_t && sel_t && w_t && x_t && mid_t && out_t && part_t &&
               shmid_t && shgate_t,
               "invariance tensor allocation");
    require_ok(ds4_gpu_tensor_write(logits_t, 0, logits,
                                    (uint64_t)INV_TOKENS * N_EXPERT * sizeof(float)),
               "invariance logit write");
    require_ok(ds4_gpu_tensor_write(x_t, 0, x, x_bytes), "invariance activation write");
    require_ok(ds4_gpu_qwen4exp_router_select_tensor(sel_t, w_t, logits_t,
                                                     N_EXPERT, N_EXPERT_USED,
                                                     INV_TOKENS),
               "invariance router select");

    const ds4_gpu_qwen4exp_slab gate_slab = {
        model, model_bytes, gate_offset, GATE_EXPERT_BYTES, Q4K_ROW_BYTES, TYPE_Q4_K };
    const ds4_gpu_qwen4exp_slab up_slab = {
        model, model_bytes, up_offset, UP_EXPERT_BYTES, Q4K_ROW_BYTES, TYPE_Q4_K };
    const ds4_gpu_qwen4exp_slab down_slab = {
        model, model_bytes, down_offset, DOWN_EXPERT_BYTES, Q51_ROW_BYTES, TYPE_Q5_1 };
    const ds4_gpu_qwen4exp_slab sh_router_slab = {
        model, model_bytes, sh_router_offset, 0, IN_DIM * sizeof(float), TYPE_F32 };
    const ds4_gpu_qwen4exp_slab sh_gate_slab = {
        model, model_bytes, sh_gate_offset, 0, Q80_IN_ROW, TYPE_Q8_0 };
    const ds4_gpu_qwen4exp_slab sh_up_slab = {
        model, model_bytes, sh_up_offset, 0, Q80_IN_ROW, TYPE_Q8_0 };
    const ds4_gpu_qwen4exp_slab sh_down_slab = {
        model, model_bytes, sh_down_offset, 0, Q80_MID_ROW, TYPE_Q8_0 };

    require_ok(ds4_gpu_qwen4exp_routed_moe_tensor(
                   out_t, mid_t, part_t, &gate_slab, &up_slab, &down_slab,
                   IN_DIM, MID_DIM, OUT_DIM, sel_t, w_t, N_EXPERT, N_EXPERT_USED,
                   x_t, INV_TOKENS, N_EXPERT_USED * MID_DIM),
               "invariance wide routed MoE");
    require_ok(ds4_gpu_qwen4exp_shared_expert_tensor(
                   out_t, shmid_t, shgate_t, &sh_router_slab, &sh_gate_slab,
                   &sh_up_slab, &sh_down_slab, IN_DIM, SHARED_MID, OUT_DIM,
                   x_t, INV_TOKENS),
               "invariance wide shared expert");
    require_ok(ds4_gpu_tensor_read(out_t, 0, wide, out_bytes),
               "invariance wide read");

    ds4_gpu_tensor *x1_t = ds4_gpu_tensor_alloc((uint64_t)IN_DIM * sizeof(float));
    ds4_gpu_tensor *mid1_t = ds4_gpu_tensor_alloc((uint64_t)N_EXPERT_USED * MID_DIM * sizeof(float));
    ds4_gpu_tensor *out1_t = ds4_gpu_tensor_alloc((uint64_t)OUT_DIM * sizeof(float));
    ds4_gpu_tensor *part1_t = ds4_gpu_tensor_alloc(
            (uint64_t)N_EXPERT_USED * OUT_DIM * sizeof(float));
    ds4_gpu_tensor *shmid1_t = ds4_gpu_tensor_alloc((uint64_t)SHARED_MID * sizeof(float));
    ds4_gpu_tensor *shgate1_t = ds4_gpu_tensor_alloc(sizeof(float));
    ds4_gpu_tensor *sel1_t = ds4_gpu_tensor_alloc((uint64_t)N_EXPERT_USED * sizeof(int32_t));
    ds4_gpu_tensor *w1_t = ds4_gpu_tensor_alloc((uint64_t)N_EXPERT_USED * sizeof(float));
    require_ok(x1_t && mid1_t && out1_t && part1_t && shmid1_t && shgate1_t &&
               sel1_t && w1_t,
               "invariance one-row tensor allocation");

    int32_t sel[INV_TOKENS * N_EXPERT_USED];
    float rw[INV_TOKENS * N_EXPERT_USED];
    require_ok(ds4_gpu_tensor_read(sel_t, 0, sel, sizeof(sel)), "invariance selection read");
    require_ok(ds4_gpu_tensor_read(w_t, 0, rw, sizeof(rw)), "invariance weight read");

    for (int t = 0; t < INV_TOKENS; t++) {
        require_ok(ds4_gpu_tensor_write(x1_t, 0, x + (size_t)t * IN_DIM,
                                        (uint64_t)IN_DIM * sizeof(float)),
                   "invariance one-row activation write");
        require_ok(ds4_gpu_tensor_write(sel1_t, 0, sel + (size_t)t * N_EXPERT_USED,
                                        (uint64_t)N_EXPERT_USED * sizeof(int32_t)),
                   "invariance one-row selection write");
        require_ok(ds4_gpu_tensor_write(w1_t, 0, rw + (size_t)t * N_EXPERT_USED,
                                        (uint64_t)N_EXPERT_USED * sizeof(float)),
                   "invariance one-row weight write");
        require_ok(ds4_gpu_qwen4exp_routed_moe_tensor(
                       out1_t, mid1_t, part1_t, &gate_slab, &up_slab, &down_slab,
                       IN_DIM, MID_DIM, OUT_DIM, sel1_t, w1_t, N_EXPERT,
                       N_EXPERT_USED, x1_t, 1u, N_EXPERT_USED * MID_DIM),
                   "invariance one-row routed MoE");
        require_ok(ds4_gpu_qwen4exp_shared_expert_tensor(
                       out1_t, shmid1_t, shgate1_t, &sh_router_slab,
                       &sh_gate_slab, &sh_up_slab, &sh_down_slab,
                       IN_DIM, SHARED_MID, OUT_DIM, x1_t, 1u),
                   "invariance one-row shared expert");
        require_ok(ds4_gpu_tensor_read(out1_t, 0, narrow + (size_t)t * OUT_DIM,
                                       (uint64_t)OUT_DIM * sizeof(float)),
                   "invariance one-row read");
    }

    /* The row tile is a schedule, not an arithmetic.  R = 1, 2, 4 and 8 walk the
     * same weight groups in the same order into the same accumulators; only
     * the number of rows a decoded group serves changes.  This runs the same
     * rows through all four and compares bits, so a tile that starts to matter
     * goes red here rather than at the end of a 48-block tower. */
    {
        float *t1 = calloc((size_t)INV_TOKENS * OUT_DIM, sizeof(float));
        float *other = calloc((size_t)INV_TOKENS * OUT_DIM, sizeof(float));
        if (!t1 || !other) fail("tile comparison allocation");
        const char *tiles[] = {"1", "2", "4", "8"};
        for (int pass = 0; pass < 4; pass++) {
            setenv("DS4_QWEN4EXP_MOE_R", tiles[pass], 1);
            require_ok(ds4_gpu_qwen4exp_routed_moe_tensor(
                           out_t, mid_t, part_t, &gate_slab, &up_slab, &down_slab,
                           IN_DIM, MID_DIM, OUT_DIM, sel_t, w_t, N_EXPERT,
                           N_EXPERT_USED, x_t, INV_TOKENS,
                           N_EXPERT_USED * MID_DIM),
                       "tile comparison routed MoE");
            require_ok(ds4_gpu_qwen4exp_shared_expert_tensor(
                           out_t, shmid_t, shgate_t, &sh_router_slab,
                           &sh_gate_slab, &sh_up_slab, &sh_down_slab,
                           IN_DIM, SHARED_MID, OUT_DIM, x_t, INV_TOKENS),
                       "tile comparison shared expert");
            require_ok(ds4_gpu_tensor_read(out_t, 0, pass == 0 ? t1 : other,
                                           out_bytes),
                       "tile comparison read");
            if (pass > 0) {
                size_t td = 0;
                for (size_t i = 0; i < (size_t)INV_TOKENS * OUT_DIM; i++) {
                    if (memcmp(&t1[i], &other[i], sizeof(float)) != 0) td++;
                }
                printf("MoE row tile R=1 against R=%s at %d rows: %zu of %zu "
                       "outputs differ\n", tiles[pass], (int)INV_TOKENS, td,
                       (size_t)INV_TOKENS * OUT_DIM);
                if (td != 0) fail("the row tile changed a number");
            }
        }
        unsetenv("DS4_QWEN4EXP_MOE_R");
        free(other);
        free(t1);
    }

    /* EXACTNESS AT THE WIDTHS THE CYCLE RUNS.  Decode is one row and a verify
     * at depths one to three is two to four, and those must equal a serial
     * decode bit for bit.  They are all below the tile's width-eight
     * threshold, so they all take the per-row kernels and all must be 0. */
    for (uint32_t w = 1; w <= 4; w++) {
        require_ok(ds4_gpu_qwen4exp_routed_moe_tensor(
                       out_t, mid_t, part_t, &gate_slab, &up_slab, &down_slab,
                       IN_DIM, MID_DIM, OUT_DIM, sel_t, w_t, N_EXPERT,
                       N_EXPERT_USED, x_t, w, N_EXPERT_USED * MID_DIM),
                   "cycle-width routed MoE");
        require_ok(ds4_gpu_qwen4exp_shared_expert_tensor(
                       out_t, shmid_t, shgate_t, &sh_router_slab, &sh_gate_slab,
                       &sh_up_slab, &sh_down_slab, IN_DIM, SHARED_MID, OUT_DIM,
                       x_t, w),
                   "cycle-width shared expert");
        float *got = calloc((size_t)w * OUT_DIM, sizeof(float));
        if (!got) fail("cycle-width allocation");
        require_ok(ds4_gpu_tensor_read(out_t, 0, got,
                                       (uint64_t)w * OUT_DIM * sizeof(float)),
                   "cycle-width read");
        size_t bad = 0;
        for (size_t i = 0; i < (size_t)w * OUT_DIM; i++) {
            if (memcmp(&got[i], &narrow[i], sizeof(float)) != 0) bad++;
        }
        printf("MoE exactness at width %u against one row at a time: "
               "%zu of %zu outputs differ\n", w, bad, (size_t)w * OUT_DIM);
        if (bad != 0) {
            fail("a width the speculative cycle runs is not bit-identical to a "
                 "serial decode");
        }
        float *reference_scan = calloc((size_t)w * OUT_DIM, sizeof(float));
        if (!reference_scan) fail("reference group-scan comparison allocation");
        setenv("DS4_QWEN4EXP_SERIAL_GROUP_SCAN", "1", 1);
        require_ok(ds4_gpu_qwen4exp_routed_moe_tensor(
                       out_t, mid_t, part_t, &gate_slab, &up_slab, &down_slab,
                       IN_DIM, MID_DIM, OUT_DIM, sel_t, w_t, N_EXPERT,
                       N_EXPERT_USED, x_t, w, N_EXPERT_USED * MID_DIM),
                   "reference group-scan routed MoE");
        require_ok(ds4_gpu_qwen4exp_shared_expert_tensor(
                       out_t, shmid_t, shgate_t, &sh_router_slab, &sh_gate_slab,
                       &sh_up_slab, &sh_down_slab, IN_DIM, SHARED_MID, OUT_DIM,
                       x_t, w),
                   "reference group-scan shared expert");
        require_ok(ds4_gpu_tensor_read(
                       out_t, 0, reference_scan,
                       (uint64_t)w * OUT_DIM * sizeof(float)),
                   "reference group-scan read");
        unsetenv("DS4_QWEN4EXP_SERIAL_GROUP_SCAN");
        if (memcmp(got, reference_scan,
                   (size_t)w * OUT_DIM * sizeof(float)) != 0) {
            fail("parallel group scan differs from the reference metadata scan");
        }
        printf("  parallel group scan output matches reference at width %u\n", w);
        free(reference_scan);
        free(got);
    }

    /* THE TILE IS A TOLERANCE, NOT AN IDENTITY.  Above width eight the prefill
     * takes the tensor-core tile, whose fold is one ascending f32 accumulator
     * where the per-row kernel uses lane partials and a butterfly.  Those are
     * different summation orders of the same products, so they differ by
     * rounding.  A prefill is identical across depths by construction, so this
     * is reported and not gated. */
    {
        size_t differing = 0;
        double worst = 0.0, worst_rel = 0.0;
        for (size_t i = 0; i < (size_t)INV_TOKENS * OUT_DIM; i++) {
            if (memcmp(&wide[i], &narrow[i], sizeof(float)) != 0) differing++;
            const double d = fabs((double)wide[i] - (double)narrow[i]);
            const double m = fabs((double)narrow[i]);
            if (d > worst) worst = d;
            if (m > 0.0 && d / m > worst_rel) worst_rel = d / m;
        }
        printf("MoE tile against the per-row path at %d rows: %zu of %zu "
               "outputs differ, max abs %.6g, max rel %.6g\n",
               (int)INV_TOKENS, differing, (size_t)INV_TOKENS * OUT_DIM,
               worst, worst_rel);
        if (!(worst_rel <= 1e-2)) {
            fail("the tile and the per-row path disagree by more than "
                 "rounding");
        }
    }

    ds4_gpu_tensor_free(part1_t);
    ds4_gpu_tensor_free(part_t);
    ds4_gpu_tensor_free(w1_t);
    ds4_gpu_tensor_free(sel1_t);
    ds4_gpu_tensor_free(shgate1_t);
    ds4_gpu_tensor_free(shmid1_t);
    ds4_gpu_tensor_free(out1_t);
    ds4_gpu_tensor_free(mid1_t);
    ds4_gpu_tensor_free(x1_t);
    ds4_gpu_tensor_free(shgate_t);
    ds4_gpu_tensor_free(shmid_t);
    ds4_gpu_tensor_free(out_t);
    ds4_gpu_tensor_free(mid_t);
    ds4_gpu_tensor_free(x_t);
    ds4_gpu_tensor_free(w_t);
    ds4_gpu_tensor_free(sel_t);
    ds4_gpu_tensor_free(logits_t);
    free(narrow);
    free(wide);
    free(logits);
    free(x);
}

/* ------------------------------------------------------------------ */
/* The fused decode-width router tail.
 *
 * The fused path folds three launches into one GEMV and one router-tail
 * kernel: the shared expert's sigmoid gate becomes an extra block-row of the
 * router's F32 GEMV, and the warp top-k, its softmax and the small expert
 * group scan become one 512-thread kernel.  Both paths claim to run the same
 * arithmetic, so this compares them byte for byte at the widths the
 * speculative cycle actually runs, over logits with exact ties at the
 * selection boundary -- the discrete decision a numeric drift would move
 * first.  The logits are compared too: their kernel MOVED translation units
 * (ds4_cuda.cu compiles with --use_fast_math, ds4_cuda_qwen4exp.cu without),
 * so this is where a flag-level difference would surface. */
static void run_router_tail_fuse_case(void) {
#if defined(__APPLE__)
    /* The Metal backend never takes the fuse (its router-tail predicate is a
     * constant zero), so there is no second arm to compare on that link. */
    puts("router tail fuse comparison: skipped, CUDA-only fusion");
    return;
#else
    /* The twenty-expert exact tie of the router section above, restated so
     * this case's logit rows are the same construction. */
    static const int tail_tied[20] = { 3, 9, 17, 40, 63, 100, 128, 129, 200,
                                       255, 256, 257, 300, 333, 400, 401, 450,
                                       500, 510, 511 };
    /* The F32 router slab comes first in the image; the expert and shared
     * slabs follow main's layout recipe. */
    const uint64_t router_off = 0;
    const uint64_t router_bytes = (uint64_t)N_EXPERT * IN_DIM * sizeof(float);
    const uint64_t gate_off = ALIGN64(router_off + router_bytes);
    const uint64_t gate_bytes = (uint64_t)N_EXPERT * GATE_EXPERT_BYTES;
    const uint64_t up_off = ALIGN64(gate_off + gate_bytes);
    const uint64_t up_bytes = (uint64_t)N_EXPERT * UP_EXPERT_BYTES;
    const uint64_t down_off = ALIGN64(up_off + up_bytes);
    const uint64_t down_bytes = (uint64_t)N_EXPERT * DOWN_EXPERT_BYTES;
    const uint64_t sh_router_off = ALIGN64(down_off + down_bytes);
    const uint64_t sh_router_bytes = (uint64_t)IN_DIM * sizeof(float);
    const uint64_t sh_gate_off = ALIGN64(sh_router_off + sh_router_bytes);
    const uint64_t sh_gate_bytes = (uint64_t)SHARED_MID * Q80_IN_ROW;
    const uint64_t sh_up_off = ALIGN64(sh_gate_off + sh_gate_bytes);
    const uint64_t sh_up_bytes = sh_gate_bytes;
    const uint64_t sh_down_off = ALIGN64(sh_up_off + sh_up_bytes);
    const uint64_t sh_down_bytes = (uint64_t)OUT_DIM * Q80_MID_ROW;
    const uint64_t model_bytes = ALIGN64(sh_down_off + sh_down_bytes);

    uint8_t *model = mmap(NULL, model_bytes, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANON, -1, 0);
    if (model == MAP_FAILED) fail("router tail model mmap");
    for (uint64_t i = 0; i < model_bytes; i++) model[i] = (uint8_t)rng_u32();
    for (uint32_t e = 0; e < N_EXPERT; e++) {
        for (uint32_t r = 0; r < MID_DIM; r++) {
            for (int which = 0; which < 2; which++) {
                const uint64_t base = (which ? up_off : gate_off) +
                    (uint64_t)e * GATE_EXPERT_BYTES + (uint64_t)r * Q4K_ROW_BYTES;
                for (uint32_t b = 0; b < IN_DIM / 256u; b++) {
                    const uint16_t d = rng_half_scale();
                    const uint16_t dmin = rng_half_scale();
                    model[base + b * 144 + 0] = (uint8_t)(d & 0xff);
                    model[base + b * 144 + 1] = (uint8_t)(d >> 8);
                    model[base + b * 144 + 2] = (uint8_t)(dmin & 0xff);
                    model[base + b * 144 + 3] = (uint8_t)(dmin >> 8);
                }
            }
        }
        for (uint32_t r = 0; r < OUT_DIM; r++) {
            const uint64_t base = down_off + (uint64_t)e * DOWN_EXPERT_BYTES +
                                  (uint64_t)r * Q51_ROW_BYTES;
            for (uint32_t b = 0; b < MID_DIM / 32u; b++) {
                const uint16_t d = rng_half_scale();
                const uint16_t m = rng_half_scale();
                model[base + b * 24 + 0] = (uint8_t)(d & 0xff);
                model[base + b * 24 + 1] = (uint8_t)(d >> 8);
                model[base + b * 24 + 2] = (uint8_t)(m & 0xff);
                model[base + b * 24 + 3] = (uint8_t)(m >> 8);
            }
        }
    }
    for (uint32_t r = 0; r < SHARED_MID; r++) {
        for (int which = 0; which < 2; which++) {
            const uint64_t base = (which ? sh_up_off : sh_gate_off) +
                                  (uint64_t)r * Q80_IN_ROW;
            for (uint32_t b = 0; b < IN_DIM / 32u; b++) {
                const uint16_t d = rng_half_scale();
                model[base + b * 34 + 0] = (uint8_t)(d & 0xff);
                model[base + b * 34 + 1] = (uint8_t)(d >> 8);
            }
        }
    }
    for (uint32_t r = 0; r < OUT_DIM; r++) {
        const uint64_t base = sh_down_off + (uint64_t)r * Q80_MID_ROW;
        for (uint32_t b = 0; b < MID_DIM / 32u; b++) {
            const uint16_t d = rng_half_scale();
            model[base + b * 34 + 0] = (uint8_t)(d & 0xff);
            model[base + b * 34 + 1] = (uint8_t)(d >> 8);
        }
    }
    /* The two F32 rows: a dot of these against the activation stays in a
     * well-conditioned float32 range on both sides of the sigmoid. */
    for (uint64_t i = 0; i < (uint64_t)N_EXPERT * IN_DIM; i++) {
        const float v = rng_unit() * 0.05f;
        memcpy(model + router_off + i * sizeof(float), &v, sizeof(v));
    }
    for (uint32_t k = 0; k < IN_DIM; k++) {
        const float v = rng_unit() * 0.05f;
        memcpy(model + sh_router_off + (uint64_t)k * sizeof(float), &v, sizeof(v));
    }
    require_ok(ds4_gpu_set_model_map(model, model_bytes),
               "router tail model map");

    /* Three logit rows with exact ties, mirroring the router section above:
     * row 0 has twenty experts sharing one top score, row 1 is all-tied, row 2
     * is random with a six-way tie inside the selection. */
    enum { TAIL_TOKENS = 3 };
    float *logits = calloc((size_t)TAIL_TOKENS * N_EXPERT, sizeof(float));
    float *x = calloc((size_t)TAIL_TOKENS * IN_DIM, sizeof(float));
    if (!logits || !x) fail("router tail allocation");
    for (int e = 0; e < N_EXPERT; e++) logits[e] = -1.0f;
    for (int i = 0; i < 20; i++) logits[tail_tied[i]] = 2.5f;
    for (int e = 0; e < N_EXPERT; e++) logits[(size_t)1 * N_EXPERT + e] = 0.25f;
    for (int e = 0; e < N_EXPERT; e++)
        logits[(size_t)2 * N_EXPERT + e] = rng_unit();
    for (int i = 0; i < 6; i++)
        logits[(size_t)2 * N_EXPERT + 20 + i * 7] = 7.0f;
    for (size_t i = 0; i < (size_t)TAIL_TOKENS * IN_DIM; i++)
        x[i] = rng_unit() * 0.5f;

    ds4_gpu_tensor *logits_t = ds4_gpu_tensor_alloc(
        (uint64_t)TAIL_TOKENS * N_EXPERT * sizeof(float));
    ds4_gpu_tensor *sel_t = ds4_gpu_tensor_alloc(
        (uint64_t)TAIL_TOKENS * N_EXPERT_USED * sizeof(int32_t));
    ds4_gpu_tensor *w_t = ds4_gpu_tensor_alloc(
        (uint64_t)TAIL_TOKENS * N_EXPERT_USED * sizeof(float));
    ds4_gpu_tensor *x_t = ds4_gpu_tensor_alloc(
        (uint64_t)TAIL_TOKENS * IN_DIM * sizeof(float));
    ds4_gpu_tensor *mid_t = ds4_gpu_tensor_alloc(
        (uint64_t)TAIL_TOKENS * N_EXPERT_USED * MID_DIM * sizeof(float));
    ds4_gpu_tensor *out_t = ds4_gpu_tensor_alloc(
        (uint64_t)TAIL_TOKENS * OUT_DIM * sizeof(float));
    ds4_gpu_tensor *part_t = ds4_gpu_tensor_alloc(
        (uint64_t)TAIL_TOKENS * N_EXPERT_USED * OUT_DIM * sizeof(float));
    ds4_gpu_tensor *shmid_t = ds4_gpu_tensor_alloc(
        (uint64_t)TAIL_TOKENS * SHARED_MID * sizeof(float));
    ds4_gpu_tensor *shgate_t = ds4_gpu_tensor_alloc(
        (uint64_t)TAIL_TOKENS * sizeof(float));
    require_ok(logits_t && sel_t && w_t && x_t && mid_t && out_t && part_t &&
               shmid_t && shgate_t,
               "router tail tensor allocation");
    require_ok(ds4_gpu_tensor_write(logits_t, 0, logits,
                                    (uint64_t)TAIL_TOKENS * N_EXPERT * sizeof(float)),
               "router tail logit write");
    require_ok(ds4_gpu_tensor_write(x_t, 0, x,
                                    (uint64_t)TAIL_TOKENS * IN_DIM * sizeof(float)),
               "router tail activation write");

    const ds4_gpu_qwen4exp_slab gate_slab = {
        model, model_bytes, gate_off, GATE_EXPERT_BYTES, Q4K_ROW_BYTES, TYPE_Q4_K };
    const ds4_gpu_qwen4exp_slab up_slab = {
        model, model_bytes, up_off, UP_EXPERT_BYTES, Q4K_ROW_BYTES, TYPE_Q4_K };
    const ds4_gpu_qwen4exp_slab down_slab = {
        model, model_bytes, down_off, DOWN_EXPERT_BYTES, Q51_ROW_BYTES, TYPE_Q5_1 };
    const ds4_gpu_qwen4exp_slab sh_router_slab = {
        model, model_bytes, sh_router_off, 0, IN_DIM * sizeof(float), TYPE_F32 };
    const ds4_gpu_qwen4exp_slab sh_gate_slab = {
        model, model_bytes, sh_gate_off, 0, Q80_IN_ROW, TYPE_Q8_0 };
    const ds4_gpu_qwen4exp_slab sh_up_slab = {
        model, model_bytes, sh_up_off, 0, Q80_IN_ROW, TYPE_Q8_0 };
    const ds4_gpu_qwen4exp_slab sh_down_slab = {
        model, model_bytes, sh_down_off, 0, Q80_MID_ROW, TYPE_Q8_0 };

    /* The switch the graph reads: on at the decode widths, off under the kill
     * variable and at the width the small scan ends. */
    unsetenv("DS4_QWEN4EXP_NO_ROUTER_FUSE");
    if (ds4_gpu_qwen4exp_router_tail_fuse_on(1u) != 1 ||
        ds4_gpu_qwen4exp_router_tail_fuse_on(7u) != 1 ||
        ds4_gpu_qwen4exp_router_tail_fuse_on(8u) != 0) {
        fail("router tail fuse switch is not the small-group envelope");
    }
    setenv("DS4_QWEN4EXP_NO_ROUTER_FUSE", "1", 1);
    if (ds4_gpu_qwen4exp_router_tail_fuse_on(1u) != 0) {
        fail("router tail fuse switch ignores DS4_QWEN4EXP_NO_ROUTER_FUSE");
    }

    const size_t logit_n = (size_t)TAIL_TOKENS * N_EXPERT * sizeof(float);
    const size_t sel_n = (size_t)TAIL_TOKENS * N_EXPERT_USED * sizeof(int32_t);
    const size_t weight_n = (size_t)TAIL_TOKENS * N_EXPERT_USED * sizeof(float);
    const size_t mid_n =
        (size_t)TAIL_TOKENS * N_EXPERT_USED * MID_DIM * sizeof(float);
    const size_t out_n = (size_t)TAIL_TOKENS * OUT_DIM * sizeof(float);
    const size_t gate_n = (size_t)TAIL_TOKENS * sizeof(float);
    int32_t *sel[2] = { calloc(sel_n, 1), calloc(sel_n, 1) };
    float *lg[2] = { calloc(logit_n, 1), calloc(logit_n, 1) };
    float *wt[2] = { calloc(weight_n, 1), calloc(weight_n, 1) };
    float *md[2] = { calloc(mid_n, 1), calloc(mid_n, 1) };
    float *ot[2] = { calloc(out_n, 1), calloc(out_n, 1) };
    /* The routed stage's block_out on its own: the routed MoE OVERWRITES
     * block_out (qwen4exp_moe_down_combine_kernel assigns) and only then does
     * the shared expert add into it, so a check that re-runs the routed stage
     * alone -- the declined arm below -- needs a routed-only reference, not
     * the post-shared-expert bytes ot holds. */
    float *otr[2] = { calloc(out_n, 1), calloc(out_n, 1) };
    float *gt[2] = { calloc(gate_n, 1), calloc(gate_n, 1) };
    if (!sel[0] || !sel[1] || !lg[0] || !lg[1] || !wt[0] || !wt[1] ||
        !md[0] || !md[1] || !ot[0] || !ot[1] || !otr[0] || !otr[1] ||
        !gt[0] || !gt[1]) {
        fail("router tail comparison allocation");
    }

    for (uint32_t width = 1; width <= TAIL_TOKENS; width++) {
        const uint64_t bytes = (uint64_t)width;
        /* The unfused chain, exactly as the graph runs it under the switch. */
        require_ok(ds4_gpu_matmul_f32_decode_rows_exact_tensor(
                       logits_t, model, model_bytes, router_off,
                       IN_DIM, N_EXPERT, x_t, width),
                   "router tail unfused logits");
        require_ok(ds4_gpu_qwen4exp_router_select_tensor(
                       sel_t, w_t, logits_t, N_EXPERT, N_EXPERT_USED, width),
                   "router tail unfused select");
        require_ok(ds4_gpu_qwen4exp_routed_moe_tensor(
                       out_t, mid_t, part_t, &gate_slab, &up_slab, &down_slab,
                       IN_DIM, MID_DIM, OUT_DIM, sel_t, w_t, N_EXPERT,
                       N_EXPERT_USED, x_t, width, N_EXPERT_USED * MID_DIM),
                   "router tail unfused routed MoE");
        require_ok(ds4_gpu_tensor_read(out_t, 0, otr[0],
                                       bytes * OUT_DIM * sizeof(float)),
                   "router tail unfused routed-only output read");
        require_ok(ds4_gpu_qwen4exp_shared_expert_tensor(
                       out_t, shmid_t, shgate_t, &sh_router_slab, &sh_gate_slab,
                       &sh_up_slab, &sh_down_slab, IN_DIM, SHARED_MID, OUT_DIM,
                       x_t, width),
                   "router tail unfused shared expert");
        require_ok(ds4_gpu_tensor_read(logits_t, 0, lg[0],
                                       bytes * N_EXPERT * sizeof(float)),
                   "router tail unfused logit read");
        require_ok(ds4_gpu_tensor_read(sel_t, 0, sel[0],
                                       bytes * N_EXPERT_USED * sizeof(int32_t)),
                   "router tail unfused selection read");
        require_ok(ds4_gpu_tensor_read(w_t, 0, wt[0],
                                       bytes * N_EXPERT_USED * sizeof(float)),
                   "router tail unfused weight read");
        require_ok(ds4_gpu_tensor_read(mid_t, 0, md[0],
                                       bytes * N_EXPERT_USED * MID_DIM * sizeof(float)),
                   "router tail unfused mid read");
        require_ok(ds4_gpu_tensor_read(out_t, 0, ot[0],
                                       bytes * OUT_DIM * sizeof(float)),
                   "router tail unfused output read");
        require_ok(ds4_gpu_tensor_read(shgate_t, 0, gt[0],
                                       bytes * sizeof(float)),
                   "router tail unfused gate read");

        /* The fused chain: two launches, the gate riding the GEMV. */
        unsetenv("DS4_QWEN4EXP_NO_ROUTER_FUSE");
        require_ok(ds4_gpu_qwen4exp_router_tail_fuse_on(width) == 1,
                   "router tail fuse on at the width under test");
        require_ok(ds4_gpu_qwen4exp_router_logits_gate_tensor(
                       logits_t, shgate_t, model, model_bytes, router_off,
                       &sh_router_slab, IN_DIM, N_EXPERT, x_t, width),
                   "router tail fused logits + gate");
        require_ok(ds4_gpu_qwen4exp_router_tail_moe_tensor(
                       out_t, mid_t, part_t, &gate_slab, &up_slab, &down_slab,
                       IN_DIM, MID_DIM, OUT_DIM, sel_t, w_t, logits_t,
                       N_EXPERT, N_EXPERT_USED, x_t, width,
                       N_EXPERT_USED * MID_DIM),
                   "router tail fused MoE");
        require_ok(ds4_gpu_tensor_read(out_t, 0, otr[1],
                                       bytes * OUT_DIM * sizeof(float)),
                   "router tail fused routed-only output read");
        require_ok(ds4_gpu_qwen4exp_shared_expert_preq_tensor(
                       out_t, shmid_t, shgate_t, &sh_router_slab, &sh_gate_slab,
                       &sh_up_slab, &sh_down_slab, IN_DIM, SHARED_MID, OUT_DIM,
                       x_t, width, 0, 1),
                   "router tail fused shared expert");
        require_ok(ds4_gpu_tensor_read(logits_t, 0, lg[1],
                                       bytes * N_EXPERT * sizeof(float)),
                   "router tail fused logit read");
        require_ok(ds4_gpu_tensor_read(sel_t, 0, sel[1],
                                       bytes * N_EXPERT_USED * sizeof(int32_t)),
                   "router tail fused selection read");
        require_ok(ds4_gpu_tensor_read(w_t, 0, wt[1],
                                       bytes * N_EXPERT_USED * sizeof(float)),
                   "router tail fused weight read");
        require_ok(ds4_gpu_tensor_read(mid_t, 0, md[1],
                                       bytes * N_EXPERT_USED * MID_DIM * sizeof(float)),
                   "router tail fused mid read");
        require_ok(ds4_gpu_tensor_read(out_t, 0, ot[1],
                                       bytes * OUT_DIM * sizeof(float)),
                   "router tail fused output read");
        require_ok(ds4_gpu_tensor_read(shgate_t, 0, gt[1],
                                       bytes * sizeof(float)),
                   "router tail fused gate read");

        const char *names[7] = { "logits", "selected", "weights",
                                 "mid", "block_out_routed", "block_out",
                                 "shexp_gate" };
        const void *a[7] = { lg[0], sel[0], wt[0], md[0], otr[0], ot[0], gt[0] };
        const void *b[7] = { lg[1], sel[1], wt[1], md[1], otr[1], ot[1], gt[1] };
        const size_t ns[7] = { (size_t)width * N_EXPERT * sizeof(float),
                               (size_t)width * N_EXPERT_USED * sizeof(int32_t),
                               (size_t)width * N_EXPERT_USED * sizeof(float),
                               (size_t)width * N_EXPERT_USED * MID_DIM * sizeof(float),
                               (size_t)width * OUT_DIM * sizeof(float),
                               (size_t)width * OUT_DIM * sizeof(float),
                               (size_t)width * sizeof(float) };
        for (int i = 0; i < 7; i++) {
            if (memcmp(a[i], b[i], ns[i]) != 0) {
                fprintf(stderr, "router tail %s differs from the unfused "
                        "chain at width %u\n", names[i], width);
                fail("the fused router tail changed a number");
            }
        }
        printf("router tail fused == unfused at width %u: logits, selected, "
               "weights, mid, routed block_out, block_out, shexp_gate all "
               "byte-identical\n", width);

        /* The same entry under the switch must fall back to the standalone
         * kernels and still match: the A/B's own escape hatch is exercised,
         * not just trusted. */
        setenv("DS4_QWEN4EXP_NO_ROUTER_FUSE", "1", 1);
        require_ok(ds4_gpu_qwen4exp_router_tail_moe_tensor(
                       out_t, mid_t, part_t, &gate_slab, &up_slab, &down_slab,
                       IN_DIM, MID_DIM, OUT_DIM, sel_t, w_t, logits_t,
                       N_EXPERT, N_EXPERT_USED, x_t, width,
                       N_EXPERT_USED * MID_DIM),
                   "router tail declined MoE");
        require_ok(ds4_gpu_tensor_read(sel_t, 0, sel[1],
                                       bytes * N_EXPERT_USED * sizeof(int32_t)),
                   "router tail declined selection read");
        require_ok(ds4_gpu_tensor_read(w_t, 0, wt[1],
                                       bytes * N_EXPERT_USED * sizeof(float)),
                   "router tail declined weight read");
        require_ok(ds4_gpu_tensor_read(mid_t, 0, md[1],
                                       bytes * N_EXPERT_USED * MID_DIM * sizeof(float)),
                   "router tail declined mid read");
        require_ok(ds4_gpu_tensor_read(out_t, 0, ot[1],
                                       bytes * OUT_DIM * sizeof(float)),
                   "router tail declined output read");
        /* The declined call stops at the routed stage: the shared expert is
         * not part of the router tail, so its block_out is the ROUTED-ONLY
         * one and the reference is otr[0], read before the unfused arm's
         * shared expert added into block_out.  Comparing against ot[0] here
         * would demand routed == routed + shared and fail at every width. */
        if (memcmp(sel[0], sel[1], ns[1]) != 0 ||
            memcmp(wt[0], wt[1], ns[2]) != 0 ||
            memcmp(md[0], md[1], ns[3]) != 0 ||
            memcmp(otr[0], ot[1], ns[4]) != 0) {
            fail("the declined router tail is not the unfused chain");
        }
        printf("router tail declined under DS4_QWEN4EXP_NO_ROUTER_FUSE at "
               "width %u: still byte-identical\n", width);
    }

    free(gt[1]); free(gt[0]); free(ot[1]); free(ot[0]);
    free(otr[1]); free(otr[0]); free(md[1]); free(md[0]);
    free(wt[1]); free(wt[0]);
    free(lg[1]); free(lg[0]); free(sel[1]); free(sel[0]);
    ds4_gpu_tensor_free(shgate_t);
    ds4_gpu_tensor_free(shmid_t);
    ds4_gpu_tensor_free(part_t);
    ds4_gpu_tensor_free(out_t);
    ds4_gpu_tensor_free(mid_t);
    ds4_gpu_tensor_free(x_t);
    ds4_gpu_tensor_free(w_t);
    ds4_gpu_tensor_free(sel_t);
    ds4_gpu_tensor_free(logits_t);
    free(x);
    free(logits);
    munmap(model, model_bytes);
#endif
}

/* The grouping scan is internal scratch, so exercise it through the routed-MoE
 * API rather than adding a test-only production hook.  Compact and
 * non-compact dispatch are independent consumers of the same counts/offsets;
 * their bitwise equality checks active ids and pair segments.  Comparing a
 * multi-token call with one-token calls checks the same metadata at every
 * expert-count boundary while keeping the floating-point path unchanged. */
enum {
    SCAN_DIM = 32,
    SCAN_MAX_EXPERT = 513,
    SCAN_MAX_USED = 8,
    SCAN_TOKENS = 3,
};

static void run_scan_call(
        ds4_gpu_tensor *out_t,
        ds4_gpu_tensor *mid_t,
        ds4_gpu_tensor *part_t,
        const ds4_gpu_qwen4exp_slab *gate_slab,
        const ds4_gpu_qwen4exp_slab *up_slab,
        const ds4_gpu_qwen4exp_slab *down_slab,
        const ds4_gpu_tensor *selected_t,
        const ds4_gpu_tensor *weights_t,
        uint32_t n_total_expert,
        uint32_t n_expert_used,
        const ds4_gpu_tensor *x_t,
        uint32_t n_tokens,
        const char *what) {
    const size_t mid_count = (size_t)n_tokens * n_expert_used * SCAN_DIM;
    float *poison = malloc(mid_count * sizeof(float));
    if (!poison) fail("scan boundary poison allocation");
    for (size_t i = 0; i < mid_count; i++) poison[i] = 12345.0f;
    require_ok(ds4_gpu_tensor_write(mid_t, 0, poison,
                                    (uint64_t)mid_count * sizeof(float)),
               "scan boundary mid poison write");
    free(poison);

    require_ok(ds4_gpu_qwen4exp_routed_moe_tensor(
                   out_t, mid_t, part_t, gate_slab, up_slab, down_slab,
                   SCAN_DIM, SCAN_DIM, SCAN_DIM, selected_t, weights_t,
                   n_total_expert, n_expert_used, x_t, n_tokens,
                   n_expert_used * SCAN_DIM),
               what);
}

static void scan_boundary_routing(uint32_t n_total_expert,
                                  uint32_t n_expert_used,
                                  int32_t *selected) {
    if (n_total_expert == 1u) {
        selected[0] = 0;
        selected[1] = 0;
        selected[2] = -1;
        return;
    }

    const int32_t last = (int32_t)n_total_expert - 1;
    const int32_t edge30 = last < 30 ? last : 30;
    const int32_t edge31 = last < 31 ? last : 31;
    const int32_t edge32 = last < 32 ? last : 32;
    const int32_t first[SCAN_MAX_USED] = {
        0, edge30, edge31, edge32, last, edge30, edge31, edge32,
    };
    memcpy(selected, first, sizeof(first));

    const int32_t duplicate = n_total_expert > 32u ? 32 : last;
    for (uint32_t i = 0; i < n_expert_used; i++)
        selected[n_expert_used + i] = duplicate;

    for (uint32_t i = 0; i < n_expert_used; i++) {
        const int which = (int)(i % 3u);
        selected[2u * n_expert_used + i] =
            which == 0 ? -1 : (which == 1 ? (int32_t)n_total_expert : INT32_MAX);
    }
}

static void run_group_scan_boundary_cases(void) {
    static const uint32_t expert_counts[] = { 1, 31, 32, 33, 511, 512, 513 };
    const char *compact_env = getenv("DS4_QWEN4EXP_NO_EXPERT_COMPACT");
    char *saved_compact_env = NULL;
    if (compact_env) {
        const size_t bytes = strlen(compact_env) + 1u;
        saved_compact_env = malloc(bytes);
        if (!saved_compact_env) fail("scan boundary environment copy");
        memcpy(saved_compact_env, compact_env, bytes);
    }
    const uint64_t row_bytes = (uint64_t)SCAN_DIM * sizeof(float);
    const uint64_t expert_bytes = (uint64_t)SCAN_DIM * row_bytes;
    const uint64_t slab_bytes = (uint64_t)SCAN_MAX_EXPERT * expert_bytes;
    const uint64_t gate_offset = 0;
    const uint64_t up_offset = ALIGN64(gate_offset + slab_bytes);
    const uint64_t down_offset = ALIGN64(up_offset + slab_bytes);
    const uint64_t image_bytes = ALIGN64(down_offset + slab_bytes);

    uint8_t *image = mmap(NULL, image_bytes, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANON, -1, 0);
    if (image == MAP_FAILED) fail("scan boundary model mmap");
    memset(image, 0, image_bytes);
    for (uint32_t e = 0; e < SCAN_MAX_EXPERT; e++) {
        for (uint32_t r = 0; r < SCAN_DIM; r++) {
            const uint64_t at = (uint64_t)e * expert_bytes +
                                (uint64_t)r * row_bytes +
                                (uint64_t)r * sizeof(float);
            const float gate = 0.015625f * (float)(1u + e % 7u);
            const float up = 0.03125f * (float)(1u + e % 11u);
            const float down = 0.0625f * (float)(1u + e % 13u);
            memcpy(image + gate_offset + at, &gate, sizeof(gate));
            memcpy(image + up_offset + at, &up, sizeof(up));
            memcpy(image + down_offset + at, &down, sizeof(down));
        }
    }
    require_ok(ds4_gpu_set_model_map(image, image_bytes),
               "scan boundary model map");

    const ds4_gpu_qwen4exp_slab gate_slab = {
        image, image_bytes, gate_offset, expert_bytes, row_bytes, TYPE_F32 };
    const ds4_gpu_qwen4exp_slab up_slab = {
        image, image_bytes, up_offset, expert_bytes, row_bytes, TYPE_F32 };
    const ds4_gpu_qwen4exp_slab down_slab = {
        image, image_bytes, down_offset, expert_bytes, row_bytes, TYPE_F32 };

    float x[SCAN_TOKENS * SCAN_DIM];
    for (uint32_t t = 0; t < SCAN_TOKENS; t++)
        for (uint32_t k = 0; k < SCAN_DIM; k++)
            x[t * SCAN_DIM + k] = (float)(1u + t + k % 5u) * 0.125f;

    ds4_gpu_tensor *selected_t = ds4_gpu_tensor_alloc(
        (uint64_t)SCAN_TOKENS * SCAN_MAX_USED * sizeof(int32_t));
    ds4_gpu_tensor *weights_t = ds4_gpu_tensor_alloc(
        (uint64_t)SCAN_TOKENS * SCAN_MAX_USED * sizeof(float));
    ds4_gpu_tensor *x_t = ds4_gpu_tensor_alloc(sizeof(x));
    ds4_gpu_tensor *mid_t = ds4_gpu_tensor_alloc(
        (uint64_t)SCAN_TOKENS * SCAN_MAX_USED * SCAN_DIM * sizeof(float));
    ds4_gpu_tensor *out_t = ds4_gpu_tensor_alloc(
        (uint64_t)SCAN_TOKENS * SCAN_DIM * sizeof(float));
    ds4_gpu_tensor *part_t = ds4_gpu_tensor_alloc(
        (uint64_t)SCAN_TOKENS * SCAN_MAX_USED * SCAN_DIM * sizeof(float));
    ds4_gpu_tensor *selected1_t = ds4_gpu_tensor_alloc(
        (uint64_t)SCAN_MAX_USED * sizeof(int32_t));
    ds4_gpu_tensor *weights1_t = ds4_gpu_tensor_alloc(
        (uint64_t)SCAN_MAX_USED * sizeof(float));
    ds4_gpu_tensor *x1_t = ds4_gpu_tensor_alloc(
        (uint64_t)SCAN_DIM * sizeof(float));
    ds4_gpu_tensor *mid1_t = ds4_gpu_tensor_alloc(
        (uint64_t)SCAN_MAX_USED * SCAN_DIM * sizeof(float));
    ds4_gpu_tensor *out1_t = ds4_gpu_tensor_alloc(
        (uint64_t)SCAN_DIM * sizeof(float));
    ds4_gpu_tensor *part1_t = ds4_gpu_tensor_alloc(
        (uint64_t)SCAN_MAX_USED * SCAN_DIM * sizeof(float));
    require_ok(selected_t && weights_t && x_t && mid_t && out_t && part_t &&
               selected1_t && weights1_t && x1_t && mid1_t && out1_t && part1_t,
               "scan boundary tensor allocation");
    require_ok(ds4_gpu_tensor_write(x_t, 0, x, sizeof(x)),
               "scan boundary activation write");

    for (size_t c = 0; c < sizeof(expert_counts) / sizeof(expert_counts[0]); c++) {
        const uint32_t n_total = expert_counts[c];
        const uint32_t n_used = n_total == 1u ? 1u : SCAN_MAX_USED;
        int32_t selected[SCAN_TOKENS * SCAN_MAX_USED];
        float weights[SCAN_TOKENS * SCAN_MAX_USED];
        float compact[SCAN_TOKENS * SCAN_DIM];
        float reference_scan[SCAN_TOKENS * SCAN_DIM];
        float noncompact[SCAN_TOKENS * SCAN_DIM];
        float reused[SCAN_TOKENS * SCAN_DIM];
        float narrow[SCAN_TOKENS * SCAN_DIM];
        scan_boundary_routing(n_total, n_used, selected);
        for (uint32_t i = 0; i < SCAN_TOKENS * n_used; i++)
            weights[i] = 0.125f * (float)(1u + i % n_used);
        require_ok(ds4_gpu_tensor_write(selected_t, 0, selected,
                    (uint64_t)SCAN_TOKENS * n_used * sizeof(int32_t)),
                   "scan boundary selection write");
        require_ok(ds4_gpu_tensor_write(weights_t, 0, weights,
                    (uint64_t)SCAN_TOKENS * n_used * sizeof(float)),
                   "scan boundary weight write");

        require_ok(unsetenv("DS4_QWEN4EXP_NO_EXPERT_COMPACT") == 0,
                   "scan boundary compact environment");
        run_scan_call(out_t, mid_t, part_t, &gate_slab, &up_slab, &down_slab,
                      selected_t, weights_t, n_total, n_used, x_t, SCAN_TOKENS,
                      "scan boundary compact call");
        require_ok(ds4_gpu_tensor_read(out_t, 0, compact, sizeof(compact)),
                   "scan boundary compact read");

        require_ok(setenv("DS4_QWEN4EXP_SERIAL_GROUP_SCAN", "1", 1) == 0,
                   "scan boundary reference environment");
        run_scan_call(out_t, mid_t, part_t, &gate_slab, &up_slab, &down_slab,
                      selected_t, weights_t, n_total, n_used, x_t, SCAN_TOKENS,
                      "scan boundary reference call");
        require_ok(unsetenv("DS4_QWEN4EXP_SERIAL_GROUP_SCAN") == 0,
                   "scan boundary reference environment restore");
        require_ok(ds4_gpu_tensor_read(out_t, 0, reference_scan,
                                       sizeof(reference_scan)),
                   "scan boundary reference read");
        if (memcmp(compact, reference_scan, sizeof(compact)) != 0)
            fail("parallel group scan differs from reference at boundary");

        /* Change the routing in the shared scratch, then restore the original.
         * Stale counts/cursors/active ids must not survive the intervening call. */
        for (uint32_t i = 0; i < SCAN_TOKENS * n_used; i++)
            selected[i] = (int32_t)((i * 17u + 3u) % n_total);
        require_ok(ds4_gpu_tensor_write(selected_t, 0, selected,
                    (uint64_t)SCAN_TOKENS * n_used * sizeof(int32_t)),
                   "scan boundary changed selection write");
        run_scan_call(out_t, mid_t, part_t, &gate_slab, &up_slab, &down_slab,
                      selected_t, weights_t, n_total, n_used, x_t, SCAN_TOKENS,
                      "scan boundary changed routing call");
        scan_boundary_routing(n_total, n_used, selected);
        require_ok(ds4_gpu_tensor_write(selected_t, 0, selected,
                    (uint64_t)SCAN_TOKENS * n_used * sizeof(int32_t)),
                   "scan boundary restored selection write");
        run_scan_call(out_t, mid_t, part_t, &gate_slab, &up_slab, &down_slab,
                      selected_t, weights_t, n_total, n_used, x_t, SCAN_TOKENS,
                      "scan boundary scratch reuse call");
        require_ok(ds4_gpu_tensor_read(out_t, 0, reused, sizeof(reused)),
                   "scan boundary scratch reuse read");
        if (memcmp(compact, reused, sizeof(compact)) != 0)
            fail("scan metadata survived changed routing");

        require_ok(setenv("DS4_QWEN4EXP_NO_EXPERT_COMPACT", "1", 1) == 0,
                   "scan boundary non-compact environment");
        run_scan_call(out_t, mid_t, part_t, &gate_slab, &up_slab, &down_slab,
                      selected_t, weights_t, n_total, n_used, x_t, SCAN_TOKENS,
                      "scan boundary non-compact call");
        require_ok(unsetenv("DS4_QWEN4EXP_NO_EXPERT_COMPACT") == 0,
                   "scan boundary compact environment restore");
        require_ok(ds4_gpu_tensor_read(out_t, 0, noncompact, sizeof(noncompact)),
                   "scan boundary non-compact read");
        if (memcmp(compact, noncompact, sizeof(compact)) != 0)
            fail("scan active ids or pair segments changed routed output");

        for (uint32_t t = 0; t < SCAN_TOKENS; t++) {
            require_ok(ds4_gpu_tensor_write(selected1_t, 0,
                        selected + (size_t)t * n_used,
                        (uint64_t)n_used * sizeof(int32_t)),
                       "scan boundary one-token selection write");
            require_ok(ds4_gpu_tensor_write(weights1_t, 0,
                        weights + (size_t)t * n_used,
                        (uint64_t)n_used * sizeof(float)),
                       "scan boundary one-token weight write");
            require_ok(ds4_gpu_tensor_write(x1_t, 0, x + (size_t)t * SCAN_DIM,
                        (uint64_t)SCAN_DIM * sizeof(float)),
                       "scan boundary one-token activation write");
            run_scan_call(out1_t, mid1_t, part1_t,
                          &gate_slab, &up_slab, &down_slab,
                          selected1_t, weights1_t, n_total, n_used, x1_t, 1u,
                          "scan boundary one-token call");
            require_ok(ds4_gpu_tensor_read(out1_t, 0,
                        narrow + (size_t)t * SCAN_DIM,
                        (uint64_t)SCAN_DIM * sizeof(float)),
                       "scan boundary one-token read");
        }
        if (memcmp(compact, narrow, sizeof(compact)) != 0)
            fail("scan boundary changed one-token versus multi-token output");
        for (uint32_t r = 0; r < SCAN_DIM; r++) {
            if (compact[2u * SCAN_DIM + r] != 0.0f)
                fail("all-invalid route did not clear poisoned mid rows");
        }
        printf("group scan boundary %u experts: bitwise output parity\n", n_total);
    }

    if (saved_compact_env) {
        require_ok(setenv("DS4_QWEN4EXP_NO_EXPERT_COMPACT",
                          saved_compact_env, 1) == 0,
                   "scan boundary original environment restore");
    } else {
        require_ok(unsetenv("DS4_QWEN4EXP_NO_EXPERT_COMPACT") == 0,
                   "scan boundary absent environment restore");
    }
    free(saved_compact_env);

    ds4_gpu_tensor_free(part1_t);
    ds4_gpu_tensor_free(out1_t);
    ds4_gpu_tensor_free(mid1_t);
    ds4_gpu_tensor_free(x1_t);
    ds4_gpu_tensor_free(weights1_t);
    ds4_gpu_tensor_free(selected1_t);
    ds4_gpu_tensor_free(part_t);
    ds4_gpu_tensor_free(out_t);
    ds4_gpu_tensor_free(mid_t);
    ds4_gpu_tensor_free(x_t);
    ds4_gpu_tensor_free(weights_t);
    ds4_gpu_tensor_free(selected_t);
    munmap(image, image_bytes);
}

static void run_production_expert_cases(void) {
    const uint32_t n_gate_up = (uint32_t)(sizeof(PROD_GATE_UP_TYPES) /
                                          sizeof(PROD_GATE_UP_TYPES[0]));
    const uint32_t n_down = (uint32_t)(sizeof(PROD_DOWN_TYPES) /
                                       sizeof(PROD_DOWN_TYPES[0]));

    uint64_t gate_off[4], up_off[4], down_off[2];
    uint64_t gate_slab_bytes[4], down_slab_bytes[2];
    uint64_t cursor = 0;
    for (uint32_t i = 0; i < n_gate_up; i++) {
        gate_slab_bytes[i] = (uint64_t)PROD_EXPERTS * PROD_MID_DIM *
                             type_row_bytes(PROD_GATE_UP_TYPES[i], PROD_IN_DIM);
        gate_off[i] = cursor; cursor = ALIGN64(cursor + gate_slab_bytes[i]);
        up_off[i] = cursor;   cursor = ALIGN64(cursor + gate_slab_bytes[i]);
    }
    for (uint32_t j = 0; j < n_down; j++) {
        down_slab_bytes[j] = (uint64_t)PROD_EXPERTS * PROD_OUT_DIM *
                             type_row_bytes(PROD_DOWN_TYPES[j], PROD_MID_DIM);
        down_off[j] = cursor; cursor = ALIGN64(cursor + down_slab_bytes[j]);
    }
    /* An F32 router row, so the shared expert can be run at this shape too. */
    const uint64_t sh_router_off = cursor;
    cursor = ALIGN64(cursor + (uint64_t)PROD_IN_DIM * sizeof(float));
    const uint64_t image_bytes = cursor;

    uint8_t *image = mmap(NULL, image_bytes, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANON, -1, 0);
    if (image == MAP_FAILED) fail("production image mmap");
    for (uint64_t i = 0; i < image_bytes; i++) image[i] = (uint8_t)rng_u32();
    for (uint32_t i = 0; i < n_gate_up; i++) {
        const uint32_t ty = PROD_GATE_UP_TYPES[i];
        const uint64_t row = type_row_bytes(ty, PROD_IN_DIM);
        for (uint64_t r = 0; r < (uint64_t)PROD_EXPERTS * PROD_MID_DIM; r++) {
            prod_seed_row_scales(image + gate_off[i] + r * row, ty, PROD_IN_DIM);
            prod_seed_row_scales(image + up_off[i] + r * row, ty, PROD_IN_DIM);
        }
    }
    for (uint32_t j = 0; j < n_down; j++) {
        const uint32_t ty = PROD_DOWN_TYPES[j];
        const uint64_t row = type_row_bytes(ty, PROD_MID_DIM);
        for (uint64_t r = 0; r < (uint64_t)PROD_EXPERTS * PROD_OUT_DIM; r++)
            prod_seed_row_scales(image + down_off[j] + r * row, ty, PROD_MID_DIM);
    }

    for (uint32_t k = 0; k < PROD_IN_DIM; k++) {
        const float v = rng_unit() * 0.05f;
        memcpy(image + sh_router_off + (uint64_t)k * sizeof(float), &v, sizeof(v));
    }

    /* A second mapping carrying a copy of the Q5_1 down slab: the split-shard
     * case reads gate and up from `image` and down from here. */
    const uint32_t q51_slot = prod_type_slot(PROD_DOWN_TYPES, n_down, TYPE_Q5_1);
    const uint64_t shard_bytes = ALIGN64(down_slab_bytes[q51_slot]);
    uint8_t *shard_b = mmap(NULL, shard_bytes, PROT_READ | PROT_WRITE,
                            MAP_PRIVATE | MAP_ANON, -1, 0);
    if (shard_b == MAP_FAILED) fail("second shard mmap");
    memcpy(shard_b, image + down_off[q51_slot], down_slab_bytes[q51_slot]);

    /* Register both mappings the way qwen4exp_session_open registers a split
     * set: Metal accumulates views through set_model_map_range, CUDA takes a
     * primary and then auxiliaries. */
#if defined(__APPLE__)
    require_ok(ds4_gpu_set_model_map_range(image, image_bytes, 0, image_bytes, 0),
               "production model map");
    require_ok(ds4_gpu_set_model_map_range(shard_b, shard_bytes, 0, shard_bytes, 0),
               "second shard map");
#else
    require_ok(ds4_gpu_set_model_map(image, image_bytes), "production model map");
    require_ok(ds4_gpu_set_aux_model_map_range(shard_b, shard_bytes, 0, shard_bytes),
               "second shard map");
#endif

    /* Router over PROD_EXPERTS experts, reused by every case. */
    float logits[PROD_TOKENS * PROD_EXPERTS];
    for (size_t i = 0; i < sizeof(logits) / sizeof(logits[0]); i++)
        logits[i] = rng_unit() * 3.0f;
    ds4_gpu_tensor *logits_t = ds4_gpu_tensor_alloc(sizeof(logits));
    ds4_gpu_tensor *selected_t = ds4_gpu_tensor_alloc(
        (uint64_t)PROD_TOKENS * PROD_USED * sizeof(int32_t));
    ds4_gpu_tensor *weights_t = ds4_gpu_tensor_alloc(
        (uint64_t)PROD_TOKENS * PROD_USED * sizeof(float));
    ds4_gpu_tensor *x_t = ds4_gpu_tensor_alloc(
        (uint64_t)PROD_TOKENS * PROD_IN_DIM * sizeof(float));
    ds4_gpu_tensor *mid_t = ds4_gpu_tensor_alloc(
        (uint64_t)PROD_TOKENS * PROD_USED * PROD_MID_DIM * sizeof(float));
    ds4_gpu_tensor *out_t = ds4_gpu_tensor_alloc(
        (uint64_t)PROD_TOKENS * PROD_OUT_DIM * sizeof(float));
    ds4_gpu_tensor *part_t = ds4_gpu_tensor_alloc(
        (uint64_t)PROD_TOKENS * PROD_USED * PROD_OUT_DIM * sizeof(float));
    require_ok(logits_t && selected_t && weights_t && x_t && mid_t && out_t,
               "production tensor allocation");
    require_ok(ds4_gpu_tensor_write(logits_t, 0, logits, sizeof(logits)),
               "production logit write");
    require_ok(ds4_gpu_qwen4exp_router_select_tensor(selected_t, weights_t, logits_t,
                                                     PROD_EXPERTS, PROD_USED,
                                                     PROD_TOKENS),
               "production router select");
    int32_t selected[PROD_TOKENS * PROD_USED];
    float weights[PROD_TOKENS * PROD_USED];
    require_ok(ds4_gpu_tensor_read(selected_t, 0, selected, sizeof(selected)),
               "production selection read");
    require_ok(ds4_gpu_tensor_read(weights_t, 0, weights, sizeof(weights)),
               "production weight read");

    float *x = calloc((size_t)PROD_TOKENS * PROD_IN_DIM, sizeof(float));
    float *got = calloc((size_t)PROD_TOKENS * PROD_OUT_DIM, sizeof(float));
    float *expected = calloc((size_t)PROD_TOKENS * PROD_OUT_DIM, sizeof(float));
    float *single_shard_q51 = calloc((size_t)PROD_TOKENS * PROD_OUT_DIM, sizeof(float));
    if (!x || !got || !expected || !single_shard_q51) fail("production allocation");
    for (size_t i = 0; i < (size_t)PROD_TOKENS * PROD_IN_DIM; i++) x[i] = rng_unit() * 0.02f;
    require_ok(ds4_gpu_tensor_write(x_t, 0, x,
                                    (uint64_t)PROD_TOKENS * PROD_IN_DIM * sizeof(float)),
               "production activation write");

    /* Every distinct pair the measured tables name, once. */
    uint32_t seen[16][2];
    uint32_t n_seen = 0;
    const expert_type_row *tables[2] = { UD_Q4_K_XL_EXPERT_TYPES,
                                         OTHER_RECIPE_EXPERT_TYPES };
    const uint32_t counts[2] = {
        (uint32_t)(sizeof(UD_Q4_K_XL_EXPERT_TYPES) / sizeof(expert_type_row)),
        (uint32_t)(sizeof(OTHER_RECIPE_EXPERT_TYPES) / sizeof(expert_type_row)),
    };
    for (uint32_t tbl = 0; tbl < 2; tbl++) {
        for (uint32_t r = 0; r < counts[tbl]; r++) {
            const expert_type_row row = tables[tbl][r];
            bool known = false;
            for (uint32_t i = 0; i < n_seen; i++)
                if (seen[i][0] == row.gate_up && seen[i][1] == row.down) known = true;
            if (known) continue;
            if (n_seen == 16) fail("more expert type pairs than the case list holds");
            seen[n_seen][0] = row.gate_up;
            seen[n_seen][1] = row.down;
            n_seen++;
        }
    }

    for (uint32_t c = 0; c < n_seen; c++) {
        const uint32_t gate_type = seen[c][0];
        const uint32_t down_type = seen[c][1];
        const uint32_t gi = prod_type_slot(PROD_GATE_UP_TYPES, n_gate_up, gate_type);
        const uint32_t dj = prod_type_slot(PROD_DOWN_TYPES, n_down, down_type);
        const uint64_t gate_row = type_row_bytes(gate_type, PROD_IN_DIM);
        const uint64_t down_row = type_row_bytes(down_type, PROD_MID_DIM);
        const ds4_gpu_qwen4exp_slab gate_slab = {
            image, image_bytes, gate_off[gi],
            (uint64_t)PROD_MID_DIM * gate_row, gate_row, gate_type };
        const ds4_gpu_qwen4exp_slab up_slab = {
            image, image_bytes, up_off[gi],
            (uint64_t)PROD_MID_DIM * gate_row, gate_row, gate_type };
        const ds4_gpu_qwen4exp_slab down_slab = {
            image, image_bytes, down_off[dj],
            (uint64_t)PROD_OUT_DIM * down_row, down_row, down_type };

        require_ok(ds4_gpu_qwen4exp_routed_moe_tensor(
                       out_t, mid_t, part_t, &gate_slab, &up_slab, &down_slab,
                       PROD_IN_DIM, PROD_MID_DIM, PROD_OUT_DIM,
                       selected_t, weights_t, PROD_EXPERTS, PROD_USED,
                       x_t, PROD_TOKENS, PROD_USED * PROD_MID_DIM),
                   "production routed MoE");
        require_ok(ds4_gpu_tensor_read(out_t, 0, got,
                                       (uint64_t)PROD_TOKENS * PROD_OUT_DIM * sizeof(float)),
                   "production output read");

        /* Compare every intermediate and output against the former joint
         * gate/up projection, including the Q8 and Q5_1 down consumers.
         * Covers both the two-token verify path and, below, the one-token
         * decode path that now shares the same R=2 split dispatch. */
        if (gate_type == TYPE_Q4_K) {
            const char *pin = getenv("DS4_QWEN4EXP_NO_SPLIT_GATEUP");
            char *saved = pin ? strdup(pin) : NULL;
            require_ok(!pin || saved, "split gate/up pin save");
            ds4_gpu_tensor *tensors[] = {out_t, mid_t, part_t};
            const size_t sizes[] = {
                (size_t)PROD_TOKENS * PROD_OUT_DIM * sizeof(float),
                (size_t)PROD_TOKENS * PROD_USED * PROD_MID_DIM * sizeof(float),
                (size_t)PROD_TOKENS * PROD_USED * PROD_OUT_DIM * sizeof(float)};
            void *reference[3], *candidate[3];
            for (unsigned j = 0; j < 3; j++) {
                reference[j] = malloc(sizes[j]); candidate[j] = malloc(sizes[j]);
                require_ok(reference[j] && candidate[j], "split gate/up buffers");
                require_ok(ds4_gpu_tensor_read(tensors[j], 0, candidate[j], sizes[j]),
                           "split gate/up candidate read");
            }
            require_ok(setenv("DS4_QWEN4EXP_NO_SPLIT_GATEUP", "1", 1) == 0,
                       "joint gate/up pin");
            require_ok(ds4_gpu_qwen4exp_routed_moe_tensor(
                           out_t, mid_t, part_t, &gate_slab, &up_slab, &down_slab,
                           PROD_IN_DIM, PROD_MID_DIM, PROD_OUT_DIM,
                           selected_t, weights_t, PROD_EXPERTS, PROD_USED,
                           x_t, PROD_TOKENS, PROD_USED * PROD_MID_DIM),
                       "joint gate/up oracle");
            for (unsigned j = 0; j < 3; j++) {
                require_ok(ds4_gpu_tensor_read(tensors[j], 0, reference[j], sizes[j]),
                           "joint gate/up read");
                require_ok(memcmp(reference[j], candidate[j], sizes[j]) == 0,
                           "split gate/up exact intermediates");
                free(reference[j]); free(candidate[j]);
            }
            /* One-token decode: same slabs, first token only. */
            {
                const size_t one_sizes[] = {
                    (size_t)1u * PROD_OUT_DIM * sizeof(float),
                    (size_t)1u * PROD_USED * PROD_MID_DIM * sizeof(float),
                    (size_t)1u * PROD_USED * PROD_OUT_DIM * sizeof(float)};
                void *one_ref[3], *one_cand[3];
                require_ok(unsetenv("DS4_QWEN4EXP_NO_SPLIT_GATEUP") == 0,
                           "split gate/up decode enable");
                require_ok(ds4_gpu_qwen4exp_routed_moe_tensor(
                               out_t, mid_t, part_t, &gate_slab, &up_slab, &down_slab,
                               PROD_IN_DIM, PROD_MID_DIM, PROD_OUT_DIM,
                               selected_t, weights_t, PROD_EXPERTS, PROD_USED,
                               x_t, 1u, PROD_USED * PROD_MID_DIM),
                           "split gate/up decode candidate");
                for (unsigned j = 0; j < 3; j++) {
                    one_cand[j] = malloc(one_sizes[j]);
                    require_ok(one_cand[j], "split decode buffers");
                    require_ok(ds4_gpu_tensor_read(tensors[j], 0, one_cand[j], one_sizes[j]),
                               "split decode candidate read");
                }
                require_ok(setenv("DS4_QWEN4EXP_NO_SPLIT_GATEUP", "1", 1) == 0,
                           "joint decode pin");
                require_ok(ds4_gpu_qwen4exp_routed_moe_tensor(
                               out_t, mid_t, part_t, &gate_slab, &up_slab, &down_slab,
                               PROD_IN_DIM, PROD_MID_DIM, PROD_OUT_DIM,
                               selected_t, weights_t, PROD_EXPERTS, PROD_USED,
                               x_t, 1u, PROD_USED * PROD_MID_DIM),
                           "joint decode oracle");
                for (unsigned j = 0; j < 3; j++) {
                    one_ref[j] = malloc(one_sizes[j]);
                    require_ok(one_ref[j], "joint decode buffers");
                    require_ok(ds4_gpu_tensor_read(tensors[j], 0, one_ref[j], one_sizes[j]),
                               "joint decode read");
                    require_ok(memcmp(one_ref[j], one_cand[j], one_sizes[j]) == 0,
                               "split decode exact intermediates");
                    free(one_ref[j]); free(one_cand[j]);
                }
            }
            require_ok((saved ? setenv("DS4_QWEN4EXP_NO_SPLIT_GATEUP", saved, 1) :
                                unsetenv("DS4_QWEN4EXP_NO_SPLIT_GATEUP")) == 0,
                       "split gate/up pin restore");
            free(saved);
            puts("split gate/up: complete out, mid and down partials bit-identical "
                 "(verify n=2 and decode n=1)");
        }

        prod_reference(image + gate_off[gi], image + up_off[gi], image + down_off[dj],
                       gate_type, down_type, x, selected, weights, expected);

        const double err = rel_frobenius(got, expected,
                                         (size_t)PROD_TOKENS * PROD_OUT_DIM);
        printf("production experts %s/%s gate-up, %s down: "
               "relative Frobenius error %.3e\n",
               type_name(gate_type), type_name(gate_type), type_name(down_type), err);
        /* Same reason as the routed case above: the kernels quantise the
         * activation to Q8_0 and the reference does not. */
        if (!(err <= 2e-2)) fail("production expert output outside the section 2 band");

        double magnitude = 0.0;
        for (size_t i = 0; i < (size_t)PROD_TOKENS * PROD_OUT_DIM; i++)
            magnitude += fabs((double)expected[i]);
        if (!(magnitude > 0.0)) fail("production reference is all zero");

        if (gate_type == TYPE_Q4_K && down_type == TYPE_Q5_1) {
            memcpy(single_shard_q51, got,
                   (size_t)PROD_TOKENS * PROD_OUT_DIM * sizeof(float));
        }
    }

    /* The split-shard case: gate and up in the first mapping, down in the
     * second, at the same offsets a real split set would give.  It must equal
     * the single-mapping run byte for byte. */
    {
        const uint64_t gate_row = type_row_bytes(TYPE_Q4_K, PROD_IN_DIM);
        const uint64_t down_row = type_row_bytes(TYPE_Q5_1, PROD_MID_DIM);
        const uint32_t gi = prod_type_slot(PROD_GATE_UP_TYPES, n_gate_up, TYPE_Q4_K);
        const ds4_gpu_qwen4exp_slab gate_slab = {
            image, image_bytes, gate_off[gi],
            (uint64_t)PROD_MID_DIM * gate_row, gate_row, TYPE_Q4_K };
        const ds4_gpu_qwen4exp_slab up_slab = {
            image, image_bytes, up_off[gi],
            (uint64_t)PROD_MID_DIM * gate_row, gate_row, TYPE_Q4_K };
        const ds4_gpu_qwen4exp_slab down_slab = {
            shard_b, shard_bytes, 0,
            (uint64_t)PROD_OUT_DIM * down_row, down_row, TYPE_Q5_1 };

        require_ok(ds4_gpu_qwen4exp_routed_moe_tensor(
                       out_t, mid_t, part_t, &gate_slab, &up_slab, &down_slab,
                       PROD_IN_DIM, PROD_MID_DIM, PROD_OUT_DIM,
                       selected_t, weights_t, PROD_EXPERTS, PROD_USED,
                       x_t, PROD_TOKENS, PROD_USED * PROD_MID_DIM),
                   "split-shard routed MoE");
        require_ok(ds4_gpu_tensor_read(out_t, 0, got,
                                       (uint64_t)PROD_TOKENS * PROD_OUT_DIM * sizeof(float)),
                   "split-shard output read");
        if (memcmp(got, single_shard_q51,
                   (size_t)PROD_TOKENS * PROD_OUT_DIM * sizeof(float)) != 0)
            fail("a down slab in a second mapping gave a different answer");
        puts("split-shard routed MoE: identical to the single-mapping run");
    }

    /* THE STAGED SHARED EXPERT AT THE CHECKPOINT'S SHAPE.  The small case in
     * run_shared_stage_cases has in_dim 256, so a row is eight groups and every
     * lane of the warp holds at most one of them -- which cannot see a change
     * in how the lanes chain their partials.  Here in_dim is 2560, so a row is
     * eighty groups and a lane sums two or three of them in a fixed order, and
     * mid_dim is 640, so the down row is twenty.  These are the widths the
     * tower actually runs. */
    {
        const ds4_gpu_qwen4exp_slab router = {
            image, image_bytes, sh_router_off, 0,
            (uint64_t)PROD_IN_DIM * sizeof(float), TYPE_F32 };
        float *sx = calloc((size_t)STAGE_AB_TOKENS * PROD_IN_DIM, sizeof(float));
        if (!sx) fail("production staged shared activation allocation");
        for (size_t i = 0; i < (size_t)STAGE_AB_TOKENS * PROD_IN_DIM; i++)
            sx[i] = rng_unit() * 0.5f;
        ds4_gpu_tensor *sx_t = ds4_gpu_tensor_alloc(
                (uint64_t)STAGE_AB_TOKENS * PROD_IN_DIM * sizeof(float));
        require_ok(sx_t != NULL, "production staged shared activation tensor");
        require_ok(ds4_gpu_tensor_write(sx_t, 0, sx,
                       (uint64_t)STAGE_AB_TOKENS * PROD_IN_DIM * sizeof(float)),
                   "production staged shared activation write");

        static const uint32_t pairs[][2] = {
            { TYPE_Q8_0, TYPE_Q8_0 },   /* what the checkpoint carries */
            { TYPE_Q4_K, TYPE_Q5_1 },
            { TYPE_Q6_K, TYPE_Q8_0 },   /* the two-half group fold */
        };
        for (uint32_t pi = 0; pi < 3u; pi++) {
            const uint32_t gt = pairs[pi][0], dt = pairs[pi][1];
            const uint32_t gi = prod_type_slot(PROD_GATE_UP_TYPES, n_gate_up, gt);
            const uint32_t dj = prod_type_slot(PROD_DOWN_TYPES, n_down, dt);
            const uint64_t grow = type_row_bytes(gt, PROD_IN_DIM);
            const uint64_t drow = type_row_bytes(dt, PROD_MID_DIM);
            const ds4_gpu_qwen4exp_slab g_slab = {
                image, image_bytes, gate_off[gi], 0, grow, gt };
            const ds4_gpu_qwen4exp_slab u_slab = {
                image, image_bytes, up_off[gi], 0, grow, gt };
            const ds4_gpu_qwen4exp_slab d_slab = {
                image, image_bytes, down_off[dj], 0, drow, dt };
            char label[64];
            snprintf(label, sizeof(label), "%s gate/up, %s down",
                     type_name(gt), type_name(dt));
            run_shared_stage_case(&router, &g_slab, &u_slab, &d_slab,
                                  PROD_IN_DIM, PROD_MID_DIM, PROD_OUT_DIM,
                                  sx_t, label);
            /* And the same pairs through the tile at the shape the tower
             * actually runs: in 2560, so a gate/up row is EIGHTY groups and a
             * class holds two or three of them chained in ascending order --
             * the case the small shape above cannot reach, because eight
             * groups leave every class with at most one and no chain to get
             * wrong.  Down is twenty groups over thirty-two classes, so
             * twelve of the tree's leaves are the exact zero the empty lanes
             * contributed. */
            run_shared_mma_case(&router, &g_slab, &u_slab, &d_slab,
                                PROD_IN_DIM, PROD_MID_DIM, PROD_OUT_DIM,
                                sx_t, label,
                                gt != TYPE_Q6_K, dt != TYPE_Q6_K);
        }
        ds4_gpu_tensor_free(sx_t);
        free(sx);
    }

    free(single_shard_q51);
    free(expected);
    free(got);
    free(x);
    ds4_gpu_tensor_free(out_t);
    ds4_gpu_tensor_free(mid_t);
    ds4_gpu_tensor_free(x_t);
    ds4_gpu_tensor_free(weights_t);
    ds4_gpu_tensor_free(selected_t);
    ds4_gpu_tensor_free(logits_t);
    munmap(shard_b, shard_bytes);
    munmap(image, image_bytes);
}

/* Shared-only comparison: the routed MMA path has different numerics and
 * must not be part of this reference.  Compare all three public tensors,
 * including the unused tail, with both zero and nonzero initial out.
 *
 * The reference is the per-row dp4a pair, pinned with
 * DS4_QWEN4EXP_SHARED_STAGE=0, and the candidate is the dispatch the tower
 * actually gets, so the widths below 64 check the staged path and the widths
 * from 64 up check the MMA pair, both against the same oracle. */
static void run_shared_exact_case(uint32_t in_dim, uint32_t mid_dim,
                                  uint32_t out_dim) {
    enum { MAX_TOKENS = 1024 };
    const uint32_t widths[] = {1, 7, 63, 64, 65, 96, MAX_TOKENS};
    const uint64_t grow = type_row_bytes(TYPE_Q8_0, in_dim);
    const uint64_t drow = type_row_bytes(TYPE_Q8_0, mid_dim);
    const uint64_t goff = ALIGN64((uint64_t)in_dim * sizeof(float));
    const uint64_t uoff = ALIGN64(goff + mid_dim * grow);
    const uint64_t doff = ALIGN64(uoff + mid_dim * grow);
    const uint64_t bytes = ALIGN64(doff + out_dim * drow);
    uint8_t *image = mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANON, -1, 0);
    if (image == MAP_FAILED) fail("shared exact model mmap");
    for (uint64_t i = 0; i < bytes; i++) image[i] = (uint8_t)rng_u32();
    for (uint32_t k = 0; k < in_dim; k++) {
        const float v = rng_unit() * 0.02f;
        memcpy(image + (uint64_t)k * sizeof(float), &v, sizeof(v));
    }
    for (uint32_t r = 0; r < mid_dim; r++) {
        prod_seed_row_scales(image + goff + r * grow, TYPE_Q8_0, in_dim);
        prod_seed_row_scales(image + uoff + r * grow, TYPE_Q8_0, in_dim);
    }
    for (uint32_t r = 0; r < out_dim; r++)
        prod_seed_row_scales(image + doff + r * drow, TYPE_Q8_0, mid_dim);
    /* Negative scales, signed zero and the smallest half subnormal. */
    const uint16_t scales[] = {0x0000, 0x8000, 0x0001, 0x8001, 0xa400};
    for (size_t i = 0; i < sizeof(scales) / sizeof(scales[0]); i++) {
        memcpy(image + goff + i * grow, &scales[i], sizeof(uint16_t));
        memcpy(image + uoff + i * grow, &scales[i], sizeof(uint16_t));
        memcpy(image + doff + i * drow, &scales[i], sizeof(uint16_t));
    }
#if defined(__APPLE__)
    require_ok(ds4_gpu_set_model_map_range(image, bytes, 0, bytes, 0),
               "shared exact model map");
#else
    require_ok(ds4_gpu_set_model_map(image, bytes), "shared exact model map");
#endif
    const ds4_gpu_qwen4exp_slab router = {
        image, bytes, 0, 0, (uint64_t)in_dim * sizeof(float), TYPE_F32 };
    const ds4_gpu_qwen4exp_slab gate = { image, bytes, goff, 0, grow, TYPE_Q8_0 };
    const ds4_gpu_qwen4exp_slab up = { image, bytes, uoff, 0, grow, TYPE_Q8_0 };
    const ds4_gpu_qwen4exp_slab down = { image, bytes, doff, 0, drow, TYPE_Q8_0 };
    const size_t nx = (size_t)MAX_TOKENS * in_dim;
    float *x = calloc(nx, sizeof(float));
    if (!x) fail("shared exact activation allocation");
    for (size_t i = in_dim; i < nx; i++) {
        const float scale = (i / in_dim) % 3 == 0 ? 0.0001f : 0.5f;
        x[i] = rng_unit() * scale;
    }
    ds4_gpu_tensor *xt = ds4_gpu_tensor_alloc(nx * sizeof(float));
    require_ok(xt != NULL, "shared exact activation tensor");
    require_ok(ds4_gpu_tensor_write(xt, 0, x, nx * sizeof(float)),
               "shared exact activation write");
    const char *names[] = {"out", "mid", "gate_scale"};
    const size_t sizes[] = {(size_t)MAX_TOKENS * out_dim * sizeof(float),
                            (size_t)MAX_TOKENS * mid_dim * sizeof(float),
                            (size_t)MAX_TOKENS * sizeof(float)};
    ds4_gpu_tensor *t[3];
    float *ref[3], *got[3];
    for (int j = 0; j < 3; j++) {
        t[j] = ds4_gpu_tensor_alloc(sizes[j]);
        ref[j] = malloc(sizes[j]);
        got[j] = malloc(sizes[j]);
        require_ok(t[j] && ref[j] && got[j], "shared exact output allocation");
    }
    const char *env = getenv("DS4_QWEN4EXP_SHARED_STAGE");
    char *saved = env ? strdup(env) : NULL;
    require_ok(!env || saved, "shared exact environment copy");
    for (size_t w = 0; w < sizeof(widths) / sizeof(widths[0]); w++) {
        for (int add = 0; add < 2; add++) {
            for (int pass = 0; pass < 2; pass++) {
                /* Pass 0 pins the per-row dp4a kernels at every width and is
                 * the reference; pass 1 takes whatever the launcher picks --
                 * per-row, staged, or the MMA pair from 64 tokens up. */
                require_ok((pass == 0
                    ? setenv("DS4_QWEN4EXP_SHARED_STAGE", "0", 1)
                    : unsetenv("DS4_QWEN4EXP_SHARED_STAGE")) == 0,
                    "shared exact dispatch switch");
                for (int j = 0; j < 3; j++) {
                    for (size_t i = 0; i < sizes[j] / sizeof(float); i++)
                        got[j][i] = j == 0 && !add ? 0.0f : (float)((int)(i % 17) - 8) * 0.125f;
                    require_ok(ds4_gpu_tensor_write(t[j], 0, got[j], sizes[j]),
                               "shared exact reset");
                }
                require_ok(ds4_gpu_qwen4exp_shared_expert_tensor(
                               t[0], t[1], t[2], &router, &gate, &up, &down,
                               in_dim, mid_dim, out_dim, xt, widths[w]),
                           "shared exact projection");
                for (int j = 0; j < 3; j++) {
                    require_ok(ds4_gpu_tensor_read(t[j], 0,
                                pass == 0 ? ref[j] : got[j], sizes[j]),
                               "shared exact read");
                    if (pass == 0) continue;
                    for (size_t i = 0; i < sizes[j] / sizeof(float); i++) {
                        if (memcmp(ref[j] + i, got[j] + i, sizeof(float)) != 0) {
                            uint32_t a, b;
                            memcpy(&a, ref[j] + i, sizeof(a));
                            memcpy(&b, got[j] + i, sizeof(b));
                            fprintf(stderr, "shared %u/%u/%u width %u add %d %s[%zu]: "
                                    "dp4a %08x, candidate %08x\n", in_dim, mid_dim,
                                    out_dim, widths[w], add, names[j], i, a, b);
                            fail("shared expert is not bit-identical to dp4a");
                        }
                    }
                }
            }
        }
        printf("shared %u/%u/%u width %u: out, mid, gate_scale bit-identical\n",
               in_dim, mid_dim, out_dim, widths[w]);
    }
    require_ok((saved ? setenv("DS4_QWEN4EXP_SHARED_STAGE", saved, 1)
                      : unsetenv("DS4_QWEN4EXP_SHARED_STAGE")) == 0,
               "shared exact environment restore");
    free(saved);
    for (int j = 0; j < 3; j++) {
        free(got[j]);
        free(ref[j]);
        ds4_gpu_tensor_free(t[j]);
    }
    ds4_gpu_tensor_free(xt);
    free(x);
    munmap(image, bytes);
}

int main(int argc, char **argv) {
    /* Fast mode for tests/qwen4exp_router_f32_mutants.sh: just the F32 router
     * projection's exactness sweep, so a mutant run costs one rebuild and a
     * few seconds instead of the whole MoE suite. */
    if (argc == 2 && strcmp(argv[1], "--router-f32") == 0) {
        require_ok(ds4_gpu_init(), "GPU init");
        /* The router (2560 -> 512), the GDN alpha/beta projections that share the
     * entry (2560 -> 48), and one shape whose chain length is NOT the 10 the
     * router's 2560 gives -- the tile's generic arm, which nothing in the
     * tower reaches today and which would otherwise go unchecked. */
    run_router_f32_prefill_exact_case(2560, 512);
    run_router_f32_prefill_exact_case(2560, 48);
    run_router_f32_prefill_exact_case(2048, 512);
        ds4_gpu_cleanup();
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--shared-exact") == 0) {
        require_ok(ds4_gpu_init(), "GPU init");
        run_shared_exact_case(PROD_IN_DIM, PROD_MID_DIM, PROD_OUT_DIM);
        run_shared_exact_case(1056, 1056, 19); /* 33 groups and an output tail */
        run_shared_exact_case(32, 32, 17);     /* one group and zero lane sums */
        ds4_gpu_cleanup();
        return 0;
    }
    const uint64_t gate_offset = 0;
    const uint64_t gate_bytes = (uint64_t)N_EXPERT * GATE_EXPERT_BYTES;
    const uint64_t up_offset = ALIGN64(gate_offset + gate_bytes);
    const uint64_t up_bytes = (uint64_t)N_EXPERT * UP_EXPERT_BYTES;
    const uint64_t down_offset = ALIGN64(up_offset + up_bytes);
    const uint64_t down_bytes = (uint64_t)N_EXPERT * DOWN_EXPERT_BYTES;
    const uint64_t sh_router_offset = ALIGN64(down_offset + down_bytes);
    const uint64_t sh_router_bytes = (uint64_t)IN_DIM * sizeof(float);
    const uint64_t sh_gate_offset = ALIGN64(sh_router_offset + sh_router_bytes);
    const uint64_t sh_gate_bytes = (uint64_t)SHARED_MID * Q80_IN_ROW;
    const uint64_t sh_up_offset = ALIGN64(sh_gate_offset + sh_gate_bytes);
    const uint64_t sh_up_bytes = sh_gate_bytes;
    const uint64_t sh_down_offset = ALIGN64(sh_up_offset + sh_up_bytes);
    const uint64_t sh_down_bytes = (uint64_t)OUT_DIM * Q80_MID_ROW;
    /* Two more shared-expert gate/up slabs, in Q5_K and in Q6_K.  The staged
     * shared kernels cache what the decoder returned, so the check that they
     * cache it faithfully has to see every shape the decoder returns -- and
     * Q6_K is the only type that returns two halves, which is the second arm
     * of the group fold.  A shared down slab cannot take these types here:
     * SHARED_MID is 64 and a K-quant superblock is 256 wide. */
    const uint64_t sh_q5k_row = (uint64_t)(IN_DIM / 256u) * 176u;
    const uint64_t sh_q6k_row = (uint64_t)(IN_DIM / 256u) * 210u;
    const uint64_t sh_gate_q5k_offset = ALIGN64(sh_down_offset + sh_down_bytes);
    const uint64_t sh_gu_q5k_bytes = (uint64_t)SHARED_MID * sh_q5k_row;
    const uint64_t sh_up_q5k_offset = ALIGN64(sh_gate_q5k_offset + sh_gu_q5k_bytes);
    const uint64_t sh_gate_q6k_offset = ALIGN64(sh_up_q5k_offset + sh_gu_q5k_bytes);
    const uint64_t sh_gu_q6k_bytes = (uint64_t)SHARED_MID * sh_q6k_row;
    const uint64_t sh_up_q6k_offset = ALIGN64(sh_gate_q6k_offset + sh_gu_q6k_bytes);
    const uint64_t model_bytes = ALIGN64(sh_up_q6k_offset + sh_gu_q6k_bytes);

    uint8_t *model = mmap(NULL, model_bytes, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANON, -1, 0);
    if (model == MAP_FAILED) fail("model mmap");
    for (uint64_t i = 0; i < model_bytes; i++) model[i] = (uint8_t)rng_u32();

    /* Overwrite every half field with a sane scale so the dot products stay in
     * range; the quantised payload bytes stay random. */
    for (uint32_t e = 0; e < N_EXPERT; e++) {
        for (uint32_t r = 0; r < MID_DIM; r++) {
            for (int which = 0; which < 2; which++) {
                const uint64_t base = (which ? up_offset : gate_offset) +
                    (uint64_t)e * GATE_EXPERT_BYTES + (uint64_t)r * Q4K_ROW_BYTES;
                for (uint32_t b = 0; b < IN_DIM / 256u; b++) {
                    const uint16_t d = rng_half_scale();
                    const uint16_t dmin = rng_half_scale();
                    model[base + b * 144 + 0] = (uint8_t)(d & 0xff);
                    model[base + b * 144 + 1] = (uint8_t)(d >> 8);
                    model[base + b * 144 + 2] = (uint8_t)(dmin & 0xff);
                    model[base + b * 144 + 3] = (uint8_t)(dmin >> 8);
                }
            }
        }
        for (uint32_t r = 0; r < OUT_DIM; r++) {
            const uint64_t base = down_offset + (uint64_t)e * DOWN_EXPERT_BYTES +
                                  (uint64_t)r * Q51_ROW_BYTES;
            for (uint32_t b = 0; b < MID_DIM / 32u; b++) {
                const uint16_t d = rng_half_scale();
                const uint16_t m = rng_half_scale();
                model[base + b * 24 + 0] = (uint8_t)(d & 0xff);
                model[base + b * 24 + 1] = (uint8_t)(d >> 8);
                model[base + b * 24 + 2] = (uint8_t)(m & 0xff);
                model[base + b * 24 + 3] = (uint8_t)(m >> 8);
            }
        }
    }
    for (uint32_t r = 0; r < SHARED_MID; r++) {
        for (int which = 0; which < 2; which++) {
            const uint64_t base = (which ? sh_up_offset : sh_gate_offset) +
                                  (uint64_t)r * Q80_IN_ROW;
            for (uint32_t b = 0; b < IN_DIM / 32u; b++) {
                const uint16_t d = rng_half_scale();
                model[base + b * 34 + 0] = (uint8_t)(d & 0xff);
                model[base + b * 34 + 1] = (uint8_t)(d >> 8);
            }
        }
    }
    for (uint32_t r = 0; r < OUT_DIM; r++) {
        const uint64_t base = sh_down_offset + (uint64_t)r * Q80_MID_ROW;
        for (uint32_t b = 0; b < MID_DIM / 32u; b++) {
            const uint16_t d = rng_half_scale();
            model[base + b * 34 + 0] = (uint8_t)(d & 0xff);
            model[base + b * 34 + 1] = (uint8_t)(d >> 8);
        }
    }
    for (uint32_t r = 0; r < SHARED_MID; r++) {
        prod_seed_row_scales(model + sh_gate_q5k_offset + (uint64_t)r * sh_q5k_row,
                             TYPE_Q5_K, IN_DIM);
        prod_seed_row_scales(model + sh_up_q5k_offset + (uint64_t)r * sh_q5k_row,
                             TYPE_Q5_K, IN_DIM);
        prod_seed_row_scales(model + sh_gate_q6k_offset + (uint64_t)r * sh_q6k_row,
                             TYPE_Q6_K, IN_DIM);
        prod_seed_row_scales(model + sh_up_q6k_offset + (uint64_t)r * sh_q6k_row,
                             TYPE_Q6_K, IN_DIM);
    }
    for (uint32_t k = 0; k < IN_DIM; k++) {
        const float v = rng_unit() * 0.05f;
        memcpy(model + sh_router_offset + (uint64_t)k * sizeof(float), &v, sizeof(v));
    }

    /* --- Q5_1 layout pin: a hand-built block with known element values. --- */
    {
        uint8_t blk[24];
        memset(blk, 0, sizeof(blk));
        blk[0] = 0x00; blk[1] = 0x3c;               /* d = 1.0 */
        blk[2] = 0x00; blk[3] = 0xc8;               /* m = -8.0 */
        blk[4] = 0x01;                              /* qh bit 0 set */
        blk[8] = 0x21;                              /* qs[0]: low 1, high 2 */
        if (ref_q5_1_value(blk, 0) != 9.0f) fail("Q5_1 element 0");
        if (ref_q5_1_value(blk, 16) != -6.0f) fail("Q5_1 element 16");
        if (ref_q5_1_value(blk, 1) != -8.0f) fail("Q5_1 element 1");
        if (ref_q5_1_value(blk, 31) != -8.0f) fail("Q5_1 element 31");
    }

    /* --- Q5_K and Q6_K layout pins: hand-built blocks with known values. --- */
    {
        uint8_t blk[176];
        memset(blk, 0, sizeof(blk));
        blk[0] = 0x00; blk[1] = 0x3c;               /* d = 1.0 */
        blk[2] = 0x00; blk[3] = 0x40;               /* dmin = 2.0 */
        blk[4 + 0] = 3; blk[4 + 4] = 1;             /* group 0: scale 3, min 1 */
        blk[4 + 1] = 2; blk[4 + 5] = 0;             /* group 1: scale 2, min 0 */
        blk[4 + 12 + 0] = 0x03;                     /* qh[0]: fifth bit for both */
        blk[4 + 12 + 32 + 0] = 0x45;                /* qs[0]: low 5, high 4 */
        if (ref_q5_K_value(blk, 0) != 3.0 * 21.0 - 2.0) fail("Q5_K element 0");
        if (ref_q5_K_value(blk, 32) != 2.0 * 20.0) fail("Q5_K element 32");
        if (ref_q5_K_value(blk, 1) != -2.0) fail("Q5_K element 1");
    }
    {
        uint8_t blk[210];
        memset(blk, 0, sizeof(blk));
        blk[0] = 0x27;                              /* ql[0]: low 7, high 2 */
        blk[32] = 0x09;                             /* ql[32]: low 9, high 0 */
        blk[128] = 0x1b;                            /* qh[0]: 3, 2, 1, 0 */
        blk[192 + 0] = 2;                           /* scales[0] */
        blk[192 + 2] = (uint8_t)(int8_t)-3;         /* scales[2] */
        blk[192 + 4] = 1;                           /* scales[4] */
        blk[192 + 6] = 5;                           /* scales[6] */
        blk[208] = 0x00; blk[209] = 0x3c;           /* d = 1.0 */
        if (ref_q6_K_value(blk, 0) != 2.0 * (55.0 - 32.0)) fail("Q6_K element 0");
        if (ref_q6_K_value(blk, 32) != -3.0 * (41.0 - 32.0)) fail("Q6_K element 32");
        if (ref_q6_K_value(blk, 64) != 1.0 * (18.0 - 32.0)) fail("Q6_K element 64");
        if (ref_q6_K_value(blk, 96) != 5.0 * (0.0 - 32.0)) fail("Q6_K element 96");
    }

    require_ok(ds4_gpu_init(), "GPU init");
    require_ok(ds4_gpu_set_model_map(model, model_bytes), "model map registration");

    /* ---------------- Router ---------------- */
    float *logits = calloc((size_t)ROUTER_TOKENS * N_EXPERT, sizeof(float));
    if (!logits) fail("logit allocation");
    for (int t = 0; t < 8; t++) {
        for (int e = 0; e < N_EXPERT; e++) logits[(size_t)t * N_EXPERT + e] = rng_unit() * 4.0f;
    }
    /* Token 8: twenty experts share one exact top score, so the tie rule
     * decides the whole selection. */
    for (int e = 0; e < N_EXPERT; e++) logits[(size_t)8 * N_EXPERT + e] = -1.0f;
    static const int tied[20] = { 3, 9, 17, 40, 63, 100, 128, 129, 200, 255,
                                  256, 257, 300, 333, 400, 401, 450, 500, 510, 511 };
    for (int i = 0; i < 20; i++) logits[(size_t)8 * N_EXPERT + tied[i]] = 2.5f;
    /* Token 9: every logit identical, so the answer is experts 0..9. */
    for (int e = 0; e < N_EXPERT; e++) logits[(size_t)9 * N_EXPERT + e] = 0.25f;
    /* Token 10: random logits with a six-way tie that lands entirely inside
     * the selection, so the tie rule fixes the order of the top six rather
     * than which experts are chosen.  Token 8 is the boundary straddle. */
    for (int e = 0; e < N_EXPERT; e++) logits[(size_t)10 * N_EXPERT + e] = rng_unit();
    for (int i = 0; i < 6; i++) logits[(size_t)10 * N_EXPERT + 20 + i * 7] = 7.0f;

    ds4_gpu_tensor *logits_t = ds4_gpu_tensor_alloc((uint64_t)ROUTER_TOKENS * N_EXPERT * sizeof(float));
    ds4_gpu_tensor *selected_t = ds4_gpu_tensor_alloc((uint64_t)ROUTER_TOKENS * N_EXPERT_USED * sizeof(int32_t));
    ds4_gpu_tensor *weights_t = ds4_gpu_tensor_alloc((uint64_t)ROUTER_TOKENS * N_EXPERT_USED * sizeof(float));
    require_ok(logits_t && selected_t && weights_t, "router tensor allocation");
    require_ok(ds4_gpu_tensor_write(logits_t, 0, logits,
                                    (uint64_t)ROUTER_TOKENS * N_EXPERT * sizeof(float)),
               "router logit write");
    require_ok(ds4_gpu_qwen4exp_router_select_tensor(selected_t, weights_t, logits_t,
                                                     N_EXPERT, N_EXPERT_USED, ROUTER_TOKENS),
               "router select");

    int32_t gpu_selected[ROUTER_TOKENS * N_EXPERT_USED];
    float gpu_weights[ROUTER_TOKENS * N_EXPERT_USED];
    require_ok(ds4_gpu_tensor_read(selected_t, 0, gpu_selected, sizeof(gpu_selected)),
               "router selection read");
    require_ok(ds4_gpu_tensor_read(weights_t, 0, gpu_weights, sizeof(gpu_weights)),
               "router weight read");

    for (int t = 0; t < ROUTER_TOKENS; t++) {
        int32_t ref_sel[N_EXPERT_USED];
        float ref_w[N_EXPERT_USED];
        ref_router(logits + (size_t)t * N_EXPERT, ref_sel, ref_w);
        for (int i = 0; i < N_EXPERT_USED; i++) {
            const int32_t got = gpu_selected[t * N_EXPERT_USED + i];
            if (got != ref_sel[i]) {
                fprintf(stderr, "router token %d slot %d: got %d, expected %d\n",
                        t, i, got, ref_sel[i]);
                exit(1);
            }
            const float dw = fabsf(gpu_weights[t * N_EXPERT_USED + i] - ref_w[i]);
            if (!(dw <= 1e-6f)) {
                fprintf(stderr, "router token %d slot %d weight: got %.9g, expected %.9g\n",
                        t, i, gpu_weights[t * N_EXPERT_USED + i], ref_w[i]);
                exit(1);
            }
        }
    }
    for (int i = 0; i < N_EXPERT_USED; i++) {
        if (gpu_selected[9 * N_EXPERT_USED + i] != i) fail("all-tied token must select 0..9");
    }
    for (int i = 0; i < 10; i++) {
        if (gpu_selected[8 * N_EXPERT_USED + i] != tied[i]) fail("tie rule must take the lowest indices");
    }

    /* ---------------- Routed experts + shared expert ---------------- */
    float *x = calloc((size_t)MOE_TOKENS * IN_DIM, sizeof(float));
    if (!x) fail("activation allocation");
    for (size_t i = 0; i < (size_t)MOE_TOKENS * IN_DIM; i++) x[i] = rng_unit() * 0.5f;

    ds4_gpu_tensor *x_t = ds4_gpu_tensor_alloc((uint64_t)MOE_TOKENS * IN_DIM * sizeof(float));
    ds4_gpu_tensor *mid_t = ds4_gpu_tensor_alloc((uint64_t)MOE_TOKENS * N_EXPERT_USED * MID_DIM * sizeof(float));
    ds4_gpu_tensor *out_t = ds4_gpu_tensor_alloc((uint64_t)MOE_TOKENS * OUT_DIM * sizeof(float));
    ds4_gpu_tensor *part_t = ds4_gpu_tensor_alloc(
        (uint64_t)MOE_TOKENS * N_EXPERT_USED * OUT_DIM * sizeof(float));
    ds4_gpu_tensor *out2_t = ds4_gpu_tensor_alloc((uint64_t)MOE_TOKENS * OUT_DIM * sizeof(float));
    ds4_gpu_tensor *sh_mid_t = ds4_gpu_tensor_alloc((uint64_t)MOE_TOKENS * SHARED_MID * sizeof(float));
    ds4_gpu_tensor *sh_gate_t = ds4_gpu_tensor_alloc((uint64_t)MOE_TOKENS * sizeof(float));
    require_ok(x_t && mid_t && out_t && out2_t && part_t && sh_mid_t && sh_gate_t,
               "MoE tensor allocation");
    require_ok(ds4_gpu_tensor_write(x_t, 0, x, (uint64_t)MOE_TOKENS * IN_DIM * sizeof(float)),
               "activation write");

    /* Reuse the first MOE_TOKENS router rows. */
    const ds4_gpu_qwen4exp_slab gate_slab = {
        model, model_bytes, gate_offset, GATE_EXPERT_BYTES, Q4K_ROW_BYTES, TYPE_Q4_K };
    const ds4_gpu_qwen4exp_slab up_slab = {
        model, model_bytes, up_offset, UP_EXPERT_BYTES, Q4K_ROW_BYTES, TYPE_Q4_K };
    const ds4_gpu_qwen4exp_slab down_slab = {
        model, model_bytes, down_offset, DOWN_EXPERT_BYTES, Q51_ROW_BYTES, TYPE_Q5_1 };
    require_ok(ds4_gpu_qwen4exp_routed_moe_tensor(
                   out_t, mid_t, part_t, &gate_slab, &up_slab, &down_slab,
                   IN_DIM, MID_DIM, OUT_DIM,
                   selected_t, weights_t, N_EXPERT, N_EXPERT_USED,
                   x_t, MOE_TOKENS, N_EXPERT_USED * MID_DIM),
               "routed MoE");

    float *routed = calloc((size_t)MOE_TOKENS * OUT_DIM, sizeof(float));
    float *combined = calloc((size_t)MOE_TOKENS * OUT_DIM, sizeof(float));
    float *ref_routed = calloc((size_t)MOE_TOKENS * OUT_DIM, sizeof(float));
    float *ref_combined = calloc((size_t)MOE_TOKENS * OUT_DIM, sizeof(float));
    if (!routed || !combined || !ref_routed || !ref_combined) fail("output allocation");
    require_ok(ds4_gpu_tensor_read(out_t, 0, routed,
                                   (uint64_t)MOE_TOKENS * OUT_DIM * sizeof(float)),
               "routed output read");

    float *xq_ref = calloc((size_t)MOE_TOKENS * IN_DIM, sizeof(float));
    if (!xq_ref) fail("reference activation allocation");
    for (int t = 0; t < MOE_TOKENS; t++) {
        ref_q8_0_roundtrip(x + (size_t)t * IN_DIM,
                           xq_ref + (size_t)t * IN_DIM, IN_DIM);
    }
    for (int t = 0; t < MOE_TOKENS; t++) {
        float mid[N_EXPERT_USED][MID_DIM];
        float midq[N_EXPERT_USED][MID_DIM];
        for (int slot = 0; slot < N_EXPERT_USED; slot++) {
            const int expert = gpu_selected[t * N_EXPERT_USED + slot];
            const float w = gpu_weights[t * N_EXPERT_USED + slot];
            for (int r = 0; r < MID_DIM; r++) {
                const uint8_t *grow = model + gate_offset +
                    (uint64_t)expert * GATE_EXPERT_BYTES + (uint64_t)r * Q4K_ROW_BYTES;
                const uint8_t *urow = model + up_offset +
                    (uint64_t)expert * UP_EXPERT_BYTES + (uint64_t)r * Q4K_ROW_BYTES;
                float g = 0.0f, u = 0.0f;
                for (int k = 0; k < IN_DIM; k++) {
                    const float xv = xq_ref[(size_t)t * IN_DIM + k];
                    g += ref_q4_K_value(grow, (uint32_t)k) * xv;
                    u += ref_q4_K_value(urow, (uint32_t)k) * xv;
                }
                mid[slot][r] = (g / (1.0f + expf(-g))) * u * w;
            }
            ref_q8_0_roundtrip(mid[slot], midq[slot], MID_DIM);
        }
        for (int r = 0; r < OUT_DIM; r++) {
            float acc = 0.0f;
            for (int slot = 0; slot < N_EXPERT_USED; slot++) {
                const int expert = gpu_selected[t * N_EXPERT_USED + slot];
                const uint8_t *drow = model + down_offset +
                    (uint64_t)expert * DOWN_EXPERT_BYTES + (uint64_t)r * Q51_ROW_BYTES;
                for (int k = 0; k < MID_DIM; k++)
                    acc += ref_q5_1_value(drow, (uint32_t)k) * midq[slot][k];
            }
            ref_routed[(size_t)t * OUT_DIM + r] = acc;
        }
    }

    const double routed_err = rel_frobenius(routed, ref_routed,
                                            (size_t)MOE_TOKENS * OUT_DIM);
    printf("routed expert relative Frobenius error: %.3e\n", routed_err);
    if (!(routed_err <= 2e-2)) fail("routed expert output outside the section 2 band");
    /* The kernels quantise the ACTIVATION to Q8_0 and factor its scale out of
     * an integer dot, which is what llama.cpp's quantised matmuls do and what
     * makes one decoded weight group serve a whole tile of rows.  The
     * reference above dequantises the weight in double precision and keeps the
     * activation exact, so the gap between them is the activation's
     * quantisation, not a kernel error.  One part in 127 per element over a
     * dot of IN_DIM terms measures 6.8e-3 on this fixture, whose quants and
     * activations are uniform random and so are harsher than a trained
     * checkpoint.  A wrong `wb` term would move the result by order one, not
     * by parts in a thousand, so this bound still convicts a formula error. */
    if (!(routed_err <= 1e-5)) fail("routed expert output outside the contract's rounding band");

    /* Determinism: the same inputs must give bit-identical bytes. */
    require_ok(ds4_gpu_qwen4exp_routed_moe_tensor(
                   out2_t, mid_t, part_t, &gate_slab, &up_slab, &down_slab,
                   IN_DIM, MID_DIM, OUT_DIM,
                   selected_t, weights_t, N_EXPERT, N_EXPERT_USED,
                   x_t, MOE_TOKENS, N_EXPERT_USED * MID_DIM),
               "routed MoE rerun");
    float *routed_again = calloc((size_t)MOE_TOKENS * OUT_DIM, sizeof(float));
    if (!routed_again) fail("determinism allocation");
    require_ok(ds4_gpu_tensor_read(out2_t, 0, routed_again,
                                   (uint64_t)MOE_TOKENS * OUT_DIM * sizeof(float)),
               "rerun output read");
    if (memcmp(routed, routed_again, (size_t)MOE_TOKENS * OUT_DIM * sizeof(float)) != 0)
        fail("routed MoE is not deterministic");

    /* Shared expert adds into the routed output: the combine. */
    const ds4_gpu_qwen4exp_slab sh_router_slab = {
        model, model_bytes, sh_router_offset, 0, IN_DIM * sizeof(float), TYPE_F32 };
    const ds4_gpu_qwen4exp_slab sh_gate_slab = {
        model, model_bytes, sh_gate_offset, 0, Q80_IN_ROW, TYPE_Q8_0 };
    const ds4_gpu_qwen4exp_slab sh_up_slab = {
        model, model_bytes, sh_up_offset, 0, Q80_IN_ROW, TYPE_Q8_0 };
    const ds4_gpu_qwen4exp_slab sh_down_slab = {
        model, model_bytes, sh_down_offset, 0, Q80_MID_ROW, TYPE_Q8_0 };
    require_ok(ds4_gpu_qwen4exp_shared_expert_tensor(
                   out_t, sh_mid_t, sh_gate_t,
                   &sh_router_slab, &sh_gate_slab, &sh_up_slab, &sh_down_slab,
                   IN_DIM, SHARED_MID, OUT_DIM, x_t, MOE_TOKENS),
               "shared expert");
    require_ok(ds4_gpu_tensor_read(out_t, 0, combined,
                                   (uint64_t)MOE_TOKENS * OUT_DIM * sizeof(float)),
               "combined output read");

    for (int t = 0; t < MOE_TOKENS; t++) {
        float gate_logit = 0.0f;
        for (int k = 0; k < IN_DIM; k++) {
            gate_logit += ref_value(TYPE_F32, model + sh_router_offset, (uint32_t)k) *
                          x[(size_t)t * IN_DIM + k];
        }
        const float g_scale = 1.0f / (1.0f + expf(-gate_logit));
        float mid[SHARED_MID];
        for (int r = 0; r < SHARED_MID; r++) {
            const uint8_t *grow = model + sh_gate_offset + (uint64_t)r * Q80_IN_ROW;
            const uint8_t *urow = model + sh_up_offset + (uint64_t)r * Q80_IN_ROW;
            float g = 0.0f, u = 0.0f;
            for (int k = 0; k < IN_DIM; k++) {
                const float xv = xq_ref[(size_t)t * IN_DIM + k];
                g += ref_q8_0_value(grow, (uint32_t)k) * xv;
                u += ref_q8_0_value(urow, (uint32_t)k) * xv;
            }
            mid[r] = (g / (1.0f + expf(-g))) * u;
        }
        {
            float midq[SHARED_MID];
            ref_q8_0_roundtrip(mid, midq, SHARED_MID);
            memcpy(mid, midq, sizeof(mid));
        }
        for (int r = 0; r < OUT_DIM; r++) {
            const uint8_t *drow = model + sh_down_offset + (uint64_t)r * Q80_MID_ROW;
            float acc = 0.0f;
            for (int k = 0; k < SHARED_MID; k++)
                acc += ref_q8_0_value(drow, (uint32_t)k) * mid[k];
            ref_combined[(size_t)t * OUT_DIM + r] =
                ref_routed[(size_t)t * OUT_DIM + r] + g_scale * acc;
        }
    }

    const double combined_err = rel_frobenius(combined, ref_combined,
                                              (size_t)MOE_TOKENS * OUT_DIM);
    printf("routed + shared relative Frobenius error: %.3e\n", combined_err);
    if (!(combined_err <= 2e-2)) fail("combined output outside the section 2 band");
    if (!(combined_err <= 1e-5)) fail("combined output outside the contract's rounding band");

    /* The shared expert must have contributed: a silent zero would still pass
     * the band if the routed term dominated. */
    {
        double delta = 0.0;
        for (size_t i = 0; i < (size_t)MOE_TOKENS * OUT_DIM; i++)
            delta += fabs((double)combined[i] - (double)routed[i]);
        if (!(delta > 0.0)) fail("shared expert contributed nothing");
    }

    /* The router (2560 -> 512), the GDN alpha/beta projections that share the
     * entry (2560 -> 48), and one shape whose chain length is NOT the 10 the
     * router's 2560 gives -- the tile's generic arm, which nothing in the
     * tower reaches today and which would otherwise go unchecked. */
    run_router_f32_prefill_exact_case(2560, 512);
    run_router_f32_prefill_exact_case(2560, 48);
    run_router_f32_prefill_exact_case(2048, 512);

    run_row_invariance_case(model, model_bytes, gate_offset, up_offset,
                            down_offset, sh_router_offset, sh_gate_offset,
                            sh_up_offset, sh_down_offset);

    run_shared_stage_cases(model, model_bytes, gate_offset, up_offset,
                           down_offset, sh_router_offset, sh_gate_offset,
                           sh_up_offset, sh_down_offset,
                           sh_gate_q5k_offset, sh_up_q5k_offset,
                           sh_gate_q6k_offset, sh_up_q6k_offset,
                           sh_q5k_row, sh_q6k_row);

    run_group_scan_boundary_cases();

    run_production_expert_cases();
    run_shared_exact_case(PROD_IN_DIM, PROD_MID_DIM, PROD_OUT_DIM);
    run_shared_exact_case(1056, 1056, 19);
    run_shared_exact_case(32, 32, 17);
    run_router_tail_fuse_case();

    free(xq_ref);
    free(routed_again);
    free(ref_combined);
    free(ref_routed);
    free(combined);
    free(routed);
    free(x);
    free(logits);
    ds4_gpu_tensor_free(sh_gate_t);
    ds4_gpu_tensor_free(sh_mid_t);
    ds4_gpu_tensor_free(part_t);
    ds4_gpu_tensor_free(out2_t);
    ds4_gpu_tensor_free(out_t);
    ds4_gpu_tensor_free(mid_t);
    ds4_gpu_tensor_free(x_t);
    ds4_gpu_tensor_free(weights_t);
    ds4_gpu_tensor_free(selected_t);
    ds4_gpu_tensor_free(logits_t);
    ds4_gpu_cleanup();
    munmap(model, model_bytes);
    puts("qwen4exp MoE GPU tests: PASS");
    return 0;
}
