/* Qwen4exp hyper-connection, norm, rope, embedding and head references.
 *
 * Plain f32 C, no GPU and no threading, written to be read next to
 * Qwen4ExpGatedResidual / Qwen4ExpRMSNorm / qwen4ExpRopePartial in
 * Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen4ExpText.swift.  The GPU
 * kernels in metal/qwen4exp_hc.metal and ds4_cuda.cu are checked against
 * these, so the summation order here is the order the kernels use:
 * hyper-connection streams low to high, embedding channels ascending.
 *
 * Header only: every entry point is static inline, so the file can be
 * included from a test or from a future CPU forward without a build rule.
 */
#ifndef DS4_QWEN4EXP_HC_REF_H
#define DS4_QWEN4EXP_HC_REF_H

#include <math.h>
#include <stdint.h>
#include <string.h>

/* Activation layout, shared with metal/dsv4_hc.metal:
 *
 *   hyper[token][hc][embd], embd contiguous
 *
 * which is also what MLX produces, because its stream is the hidden state
 * tiled `hc_count` times along the last axis. */

typedef struct {
    uint16_t d;
    int8_t   qs[32];
} ds4_qwen4exp_ref_block_q8_0;

static inline float ds4_qwen4exp_ref_half_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1fu;
    const uint32_t mant = h & 0x3ffu;
    union { uint32_t u; float f; } out;

    if (exp == 0u) {
        if (mant == 0u) {
            out.u = sign;
            return out.f;
        }
        /* Subnormal half: renormalize. */
        uint32_t e = 0u;
        uint32_t m = mant;
        while ((m & 0x400u) == 0u) {
            m <<= 1;
            e++;
        }
        m &= 0x3ffu;
        out.u = sign | ((127u - 15u - e + 1u) << 23) | (m << 13);
        return out.f;
    }
    if (exp == 31u) {
        out.u = sign | 0x7f800000u | (mant << 13);
        return out.f;
    }
    out.u = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
    return out.f;
}

/* Round to nearest even bf16, widened back to f32.  MLX runs this tower in
 * bf16 and its fused rmsNorm casts the normalized value to the activation
 * dtype BEFORE the weight multiply, so an f32 engine has to reproduce the
 * cast to land on the same product. */
static inline float ds4_qwen4exp_ref_round_bf16(float v) {
    union { float f; uint32_t u; } bits;
    bits.f = v;
    const uint32_t rounding = 0x7fffu + ((bits.u >> 16) & 1u);
    bits.u = (bits.u + rounding) & 0xffff0000u;
    return bits.f;
}

/* Longest rotary half this engine builds a frequency table for: 128 covers a
 * fully rotated 256-wide head, and the table stays small enough to travel as
 * an inline constant argument on Metal and as a by-value kernel parameter on
 * CUDA, so no rope call has to allocate. */
#define DS4_QWEN4EXP_ROPE_MAX_HALF 128u

/* inv_freq[j] = freq_base^(-2j / n_rot), the qwen4exp partial-rope
 * frequencies.
 *
 * THE one definition.  Every rope user calls it: the QSA block and its
 * indexer through ds4_gpu_qwen4exp_rope_inv_freq, which is this function under
 * an exported name, and the f32 reference below.  Nothing raises the base per
 * lane, because a one-ulp disagreement on a frequency reaches the rotation
 * multiplied by the token position -- at position 8192 a float log/exp pair
 * drifts by 2.4e-4 radians against this one, which is outside the rope band.
 *
 * The intermediate is double even though the result is float: log and exp in
 * float lose about 6.7e-7 relative, double about 4.9e-8, and the two lanes
 * disagreed bitwise on 23 of 32 entries when each raised its own. */
static inline void ds4_qwen4exp_rope_inv_freq(
        float *inv_freq, uint32_t n_rot, float freq_base) {
    if (!inv_freq || n_rot < 2u) return;
    const double scale = -log((double)freq_base) / (double)n_rot;
    for (uint32_t j = 0; j < n_rot / 2u; j++) {
        inv_freq[j] = (float)exp((double)(2u * j) * scale);
    }
}

static inline float ds4_qwen4exp_ref_sigmoid(float z) {
    return 1.0f / (1.0f + expf(-z));
}

/* Zero-centered RMS norm, optionally grouped.
 *
 * `weight_bias` is the offset the checkpoint does NOT bake: 1 for a
 * zero-centered checkpoint that stores `w` and wants `y * (1 + w)`, 0 for one
 * that stores `1 + w` already.  `group` equal to `n` is the ordinary
 * per-row norm; `group` equal to the hidden size is the hyper-connection
 * form, where each stream carries its own statistic and the weight still
 * indexes the flat row. */
static inline void ds4_qwen4exp_ref_rms_norm(
        float *out, const float *x, const float *w,
        uint32_t n, uint32_t group, uint32_t rows,
        float eps, float weight_bias, int round_bf16) {
    const uint32_t n_group = group ? n / group : 0u;
    for (uint32_t r = 0; r < rows; r++) {
        for (uint32_t g = 0; g < n_group; g++) {
            const uint64_t base = (uint64_t)r * n + (uint64_t)g * group;
            float sum = 0.0f;
            for (uint32_t i = 0; i < group; i++) {
                const float v = x[base + i];
                sum += v * v;
            }
            const float scale = 1.0f / sqrtf(sum / (float)group + eps);
            for (uint32_t i = 0; i < group; i++) {
                float normed = x[base + i] * scale;
                if (round_bf16) normed = ds4_qwen4exp_ref_round_bf16(normed);
                out[base + i] = normed * (weight_bias + w[(uint64_t)g * group + i]);
            }
        }
    }
}

/* Row-major Q8_0 matmul, llama.cpp block layout: `out_dim` rows of
 * `in_dim / 32` blocks each. */
static inline void ds4_qwen4exp_ref_matmul_q8_0(
        float *out, const void *weights,
        uint32_t in_dim, uint32_t out_dim,
        const float *x, uint32_t rows) {
    const uint32_t blocks = in_dim / 32u;
    const ds4_qwen4exp_ref_block_q8_0 *w =
        (const ds4_qwen4exp_ref_block_q8_0 *)weights;
    for (uint32_t r = 0; r < rows; r++) {
        for (uint32_t o = 0; o < out_dim; o++) {
            const ds4_qwen4exp_ref_block_q8_0 *row = w + (uint64_t)o * blocks;
            float acc = 0.0f;
            for (uint32_t b = 0; b < blocks; b++) {
                const float d = ds4_qwen4exp_ref_half_to_f32(row[b].d);
                float part = 0.0f;
                for (uint32_t i = 0; i < 32u; i++) {
                    part += (float)row[b].qs[i] * x[(uint64_t)r * in_dim + b * 32u + i];
                }
                acc += d * part;
            }
            out[(uint64_t)r * out_dim + o] = acc;
        }
    }
}

/* silu(x * scale), in place: the 1/n_hc divide plus the low-rank activation. */
static inline void ds4_qwen4exp_ref_scale_silu(
        float *x, uint32_t n, float scale) {
    for (uint32_t i = 0; i < n; i++) {
        const float z = x[i] * scale;
        x[i] = z * ds4_qwen4exp_ref_sigmoid(z);
    }
}

/* Block input: the mean over the streams of the gated normalized streams.
 * `wide` is the RAW low-rank up projection; the sigmoid is applied here. */
static inline void ds4_qwen4exp_ref_hc_mix(
        float *out, const float *normed, const float *wide,
        uint32_t n_embd, uint32_t n_hc, uint32_t rows) {
    for (uint32_t t = 0; t < rows; t++) {
        for (uint32_t d = 0; d < n_embd; d++) {
            const uint64_t row = (uint64_t)t * n_hc * n_embd + d;
            float acc = 0.0f;
            for (uint32_t h = 0; h < n_hc; h++) {
                const uint64_t idx = row + (uint64_t)h * n_embd;
                acc += ds4_qwen4exp_ref_sigmoid(wide[idx]) * normed[idx];
            }
            out[(uint64_t)t * n_embd + d] = acc * (1.0f / (float)n_hc);
        }
    }
}

/* inject[t][h] = 2 * sigmoid(dot(W[h], normed[t]) / n_hc), W dense F32. */
static inline void ds4_qwen4exp_ref_hc_inject_weights(
        float *out, const float *normed, const float *w,
        uint32_t n_embd, uint32_t n_hc, uint32_t rows) {
    const uint32_t wide = n_hc * n_embd;
    for (uint32_t t = 0; t < rows; t++) {
        for (uint32_t h = 0; h < n_hc; h++) {
            float acc = 0.0f;
            for (uint32_t i = 0; i < wide; i++) {
                acc += normed[(uint64_t)t * wide + i] * w[(uint64_t)h * wide + i];
            }
            out[(uint64_t)t * n_hc + h] =
                2.0f * ds4_qwen4exp_ref_sigmoid(acc * (1.0f / (float)n_hc));
        }
    }
}

/* out[t][h][d] = residual[t][h][d] + block[t][d] * inject[t][h] */
static inline void ds4_qwen4exp_ref_hc_inject(
        float *out, const float *residual, const float *block,
        const float *inject, uint32_t n_embd, uint32_t n_hc, uint32_t rows) {
    for (uint32_t t = 0; t < rows; t++) {
        for (uint32_t h = 0; h < n_hc; h++) {
            for (uint32_t d = 0; d < n_embd; d++) {
                const uint64_t idx = ((uint64_t)t * n_hc + h) * n_embd + d;
                /* fmaf, not `a + b * c`: both GPU kernels contract this into
                 * a single fused multiply-add, and so does clang on macOS,
                 * where the expression form happens to agree.  Under
                 * -std=c99 on Linux strict ISO turns contraction off, the
                 * reference rounds twice where the kernel rounds once, and the
                 * zero-tolerance comparison fails on a difference that is not
                 * an error in either.  Saying fmaf makes both sides round
                 * once on every platform. */
                out[idx] = fmaf(block[(uint64_t)t * n_embd + d],
                                inject[(uint64_t)t * n_hc + h],
                                residual[idx]);
            }
        }
    }
}

/* Partial rope over the LEADING n_rot dimensions, half split, in place.
 * The trailing head_dim - n_rot dimensions are never touched. */
static inline void ds4_qwen4exp_ref_rope_head(
        float *x, uint32_t n_tokens, uint32_t n_head, uint32_t head_dim,
        uint32_t n_rot, int32_t pos0, int32_t pos_stride, float freq_base) {
    const uint32_t half = n_rot / 2u;
    float inv_freq[DS4_QWEN4EXP_ROPE_MAX_HALF];
    ds4_qwen4exp_rope_inv_freq(inv_freq, n_rot, freq_base);
    for (uint32_t t = 0; t < n_tokens; t++) {
        const float pos = (float)(pos0 + (int32_t)t * pos_stride);
        for (uint32_t h = 0; h < n_head; h++) {
            float *head = x + ((uint64_t)t * n_head + h) * head_dim;
            for (uint32_t j = 0; j < half; j++) {
                const float theta = pos * inv_freq[j];
                const float c = cosf(theta);
                const float s = sinf(theta);
                const float x1 = head[j];
                const float x2 = head[j + half];
                head[j] = x1 * c - x2 * s;
                head[j + half] = x2 * c + x1 * s;
            }
        }
    }
}

/* Q8_0 token embedding gather, tiled into the hyper-connection streams.
 * This is the layer-0 seed: MLX tiles the hidden state hc_count times. */
static inline void ds4_qwen4exp_ref_embed_hc_q8_0(
        float *out_hc, const void *weights, const int32_t *tokens,
        uint32_t n_tokens, uint32_t n_embd, uint32_t n_hc) {
    const uint32_t blocks = n_embd / 32u;
    const ds4_qwen4exp_ref_block_q8_0 *w =
        (const ds4_qwen4exp_ref_block_q8_0 *)weights;
    for (uint32_t t = 0; t < n_tokens; t++) {
        const ds4_qwen4exp_ref_block_q8_0 *row =
            w + (uint64_t)tokens[t] * blocks;
        for (uint32_t b = 0; b < blocks; b++) {
            const float d = ds4_qwen4exp_ref_half_to_f32(row[b].d);
            for (uint32_t i = 0; i < 32u; i++) {
                const float v = d * (float)row[b].qs[i];
                for (uint32_t h = 0; h < n_hc; h++) {
                    out_hc[((uint64_t)t * n_hc + h) * n_embd + b * 32u + i] = v;
                }
            }
        }
    }
}

/* The whole gated residual mixer.
 *
 *   normed = hcNorm(hyper)                        grouped, one stat per stream
 *   w      = sigmoid(mixUp(silu(mixDown(normed) / n_hc)))
 *   mixed  = mean_h(w * normed)                   the block input
 *   inject = 2 * sigmoid(blockInject(normed) / n_hc)
 *
 * `inject` and `inject_w` are NULL for the tower's final mixer, which has no
 * inject head and stands in for the `model.norm` this checkpoint does not
 * carry.  `hyper` is left untouched: it IS the residual, and it is also the
 * pre-final-mixer stream the native MTP head reads. */
static inline void ds4_qwen4exp_ref_hc_mixer(
        float *mixed, float *inject,
        float *normed_scratch, float *lowrank_scratch, float *wide_scratch,
        const float *hyper,
        const float *norm_w, const void *down_w, const void *up_w,
        const float *inject_w,
        uint32_t n_embd, uint32_t n_hc, uint32_t n_lowrank, uint32_t rows,
        float eps, float weight_bias, int round_bf16) {
    const uint32_t wide = n_hc * n_embd;
    ds4_qwen4exp_ref_rms_norm(normed_scratch, hyper, norm_w, wide, n_embd,
                              rows, eps, weight_bias, round_bf16);
    ds4_qwen4exp_ref_matmul_q8_0(lowrank_scratch, down_w, wide, n_lowrank,
                                 normed_scratch, rows);
    ds4_qwen4exp_ref_scale_silu(lowrank_scratch, rows * n_lowrank,
                                1.0f / (float)n_hc);
    ds4_qwen4exp_ref_matmul_q8_0(wide_scratch, up_w, n_lowrank, wide,
                                 lowrank_scratch, rows);
    ds4_qwen4exp_ref_hc_mix(mixed, normed_scratch, wide_scratch,
                            n_embd, n_hc, rows);
    if (inject && inject_w) {
        ds4_qwen4exp_ref_hc_inject_weights(inject, normed_scratch, inject_w,
                                           n_embd, n_hc, rows);
    }
}

#endif /* DS4_QWEN4EXP_HC_REF_H */
