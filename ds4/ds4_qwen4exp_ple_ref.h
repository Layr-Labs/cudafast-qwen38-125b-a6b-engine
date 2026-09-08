/* Qwen4exp per-layer embedding (PLE) block reference.
 *
 * Plain C, no GPU and no threading, accumulated in double.  Written from the
 * MLX forward and NOT from the kernels: read it next to
 * `Qwen4ExpPLELayer.callAsFunction` and `Qwen4ExpPLELayer.shortConv` in
 * Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen4ExpNGram.swift, and the
 * `stream = stream + ple(stream, ...)` of `Qwen4ExpDecoderLayer` in
 * Qwen4Exp.swift.  Each entry point below names the reference lines it
 * mirrors.
 *
 * The band a test may assert is therefore the KERNEL's own error and not a
 * shared rounding order: nothing here shares an accumulation order with
 * metal/qwen4exp_ple.metal.
 *
 * Header only: every entry point is static inline, so the file can be
 * included from a test without a build rule, the way ds4_qwen4exp_hc_ref.h is.
 */
#ifndef DS4_QWEN4EXP_PLE_REF_H
#define DS4_QWEN4EXP_PLE_REF_H

#include <math.h>
#include <stdint.h>
#include <string.h>

/* Longest rolling convolution state this engine sizes for: the pinned
 * checkpoint needs (ple_conv_kernel - 1) * ngram_size = 9. */
#define DS4_QWEN4EXP_PLE_MAX_STATE 16u

/* The floor inside the signed square root.  A literal in the reference
 * (`MLXArray(Float(1e-6))`, Qwen4ExpNGram.swift:335), not the norm epsilon:
 * moving the norm epsilon must not move this. */
#define DS4_QWEN4EXP_PLE_GATE_FLOOR 1.0e-6

static inline double ds4_qwen4exp_ple_ref_sigmoid(double z) {
    return 1.0 / (1.0 + exp(-z));
}

/* `MLX.sqrt(maximum(MLX.abs(gate), 1e-6)) * MLX.sign(gate)`,
 * Qwen4ExpNGram.swift:335.  sign(0) is 0. */
static inline double ds4_qwen4exp_ple_ref_signed_sqrt(double v) {
    const double magnitude = sqrt(fabs(v) > DS4_QWEN4EXP_PLE_GATE_FLOOR
                                      ? fabs(v)
                                      : DS4_QWEN4EXP_PLE_GATE_FLOOR);
    if (v > 0.0) return magnitude;
    if (v < 0.0) return -magnitude;
    return 0.0;
}

/* Zero-centered RMS norm, grouped, the PLE block's three norms.
 *
 * `Qwen4ExpRMSNorm.callAsFunction` with `groupSize = hiddenSize`,
 * Qwen4ExpText.swift:305-319: each group carries its own statistic, the
 * weight indexes the flat row, and `weight_bias` is the offset the
 * checkpoint does not bake.  `round_bf16` reproduces MLX's cast of the
 * normalized value to the activation dtype before the weight multiply.
 *
 * This is the same function ds4_qwen4exp_hc_ref.h states in f32; it is
 * repeated in double here so the whole-block reference has one arithmetic
 * type from end to end. */
static inline void ds4_qwen4exp_ple_ref_rms_norm(
        double *out, const double *x, const float *w,
        uint32_t n, uint32_t group, uint32_t rows,
        double eps, double weight_bias, int round_bf16) {
    const uint32_t n_group = n / group;
    for (uint32_t r = 0; r < rows; r++) {
        for (uint32_t g = 0; g < n_group; g++) {
            const uint64_t base = (uint64_t)r * n + (uint64_t)g * group;
            double sum = 0.0;
            for (uint32_t i = 0; i < group; i++) {
                const double v = x[base + i];
                sum += v * v;
            }
            const double scale = 1.0 / sqrt(sum / (double)group + eps);
            for (uint32_t i = 0; i < group; i++) {
                double normed = x[base + i] * scale;
                if (round_bf16) {
                    union { float f; uint32_t u; } bits;
                    bits.f = (float)normed;
                    const uint32_t rounding = 0x7fffu + ((bits.u >> 16) & 1u);
                    bits.u = (bits.u + rounding) & 0xffff0000u;
                    normed = (double)bits.f;
                }
                out[base + i] = normed * (weight_bias + (double)w[(uint64_t)g * group + i]);
            }
        }
    }
}

/* Q8_0 matrix multiply, out[r][o] = sum_i dequant(w[o][i]) * x[r][i].
 *
 * The two PLE projections, `keyProj` and `valueProj` of Qwen4ExpPLELayer.
 * A Q8_0 block is an f16 scale then 32 int8 values, and dequantization is
 * `scale * q`; this accumulates in double so the whole-block reference has
 * one arithmetic type from end to end.  `half_to_f32` is passed in rather
 * than repeated: ds4_qwen4exp_hc_ref.h already states that conversion once. */
static inline void ds4_qwen4exp_ple_ref_matmul_q8_0(
        double *out, const void *weights, uint32_t in_dim, uint32_t out_dim,
        const double *x, uint32_t rows,
        float (*half_to_f32)(uint16_t)) {
    const uint32_t blocks = in_dim / 32u;
    const uint8_t *base = (const uint8_t *)weights;
    for (uint32_t r = 0; r < rows; r++) {
        for (uint32_t o = 0; o < out_dim; o++) {
            double acc = 0.0;
            for (uint32_t b = 0; b < blocks; b++) {
                const uint8_t *block = base + ((uint64_t)o * blocks + b) * 34u;
                uint16_t raw = 0;
                memcpy(&raw, block, sizeof(raw));
                const double d = (double)half_to_f32(raw);
                double part = 0.0;
                for (uint32_t i = 0; i < 32u; i++) {
                    part += (double)(int8_t)block[2u + i] *
                            x[(uint64_t)r * in_dim + b * 32u + i];
                }
                acc += d * part;
            }
            out[(uint64_t)r * out_dim + o] = acc;
        }
    }
}

/* The gate, Qwen4ExpNGram.swift:334-338.
 *
 *   gate = (key * query).sum(axis: -1, keepDims: true) / sqrt(hiddenSize)
 *   gate = signed_sqrt(gate)
 *   gated = sigmoid(gate) * value[.ellipsis, .newAxis, 0...]
 *
 * `key` and `query` are the [rows][n_hc][n_embd] reshapes of the two
 * 10240-wide normed rows; `value` is the 2560-wide value projection, shared
 * by every stream, which is what the `.newAxis` broadcasts. */
static inline void ds4_qwen4exp_ple_ref_gate(
        double *out, const double *key, const double *query,
        const double *value, uint32_t n_embd, uint32_t n_hc, uint32_t rows) {
    for (uint32_t t = 0; t < rows; t++) {
        for (uint32_t h = 0; h < n_hc; h++) {
            const uint64_t base = ((uint64_t)t * n_hc + h) * n_embd;
            double dot = 0.0;
            for (uint32_t d = 0; d < n_embd; d++) {
                dot += key[base + d] * query[base + d];
            }
            const double gate = ds4_qwen4exp_ple_ref_sigmoid(
                ds4_qwen4exp_ple_ref_signed_sqrt(dot / sqrt((double)n_embd)));
            for (uint32_t d = 0; d < n_embd; d++) {
                out[base + d] = gate * value[(uint64_t)t * n_embd + d];
            }
        }
    }
}

/* The short convolution, its state and the residual add.
 *
 * `Qwen4ExpPLELayer.shortConv`, Qwen4ExpNGram.swift:308-319:
 *
 *   full  = concatenated([state, x], axis: 1)
 *   state = full[..., -n:, ...]                       n = (kernel-1) * dilation
 *   out   = silu(conv1d(full[..., -(n + S):, ...]))
 *
 * with `conv1d` depthwise (groups = channels), stride 1, padding 0 and
 * dilation = ngram_size, so tap k of channel c reads row `t + dilation * k`
 * of `full` and tap `kernel - 1` reads the current row.
 *
 * Then Qwen4ExpNGram.swift:339 adds the un-convolved gated stream back,
 * `gated + shortConv(normConv(gated))`, and Qwen4Exp.swift:113-121 adds the
 * whole block into the layer's incoming stream, `stream = stream + ple(...)`.
 * Both adds are folded here, so `hyper` is accumulated into.
 *
 * `weight` is the checkpoint's [channels][conv_kernel] tensor, taps
 * contiguous.  `state` is [state_len][channels] and is advanced. */
static inline void ds4_qwen4exp_ple_ref_conv(
        double *hyper, double *state, const double *gated,
        const double *conv_in, const float *weight,
        uint32_t channels, uint32_t conv_kernel, uint32_t dilation,
        uint32_t rows) {
    const uint32_t state_len = (conv_kernel - 1u) * dilation;
    for (uint32_t c = 0; c < channels; c++) {
        for (uint32_t t = 0; t < rows; t++) {
            double acc = 0.0;
            for (uint32_t k = 0; k < conv_kernel; k++) {
                const uint32_t i = t + dilation * k;
                const double v = (i < state_len)
                    ? state[(uint64_t)i * channels + c]
                    : conv_in[(uint64_t)(i - state_len) * channels + c];
                acc += v * (double)weight[(uint64_t)c * conv_kernel + k];
            }
            const uint64_t index = (uint64_t)t * channels + c;
            hyper[index] += gated[index] + acc * ds4_qwen4exp_ple_ref_sigmoid(acc);
        }
        for (uint32_t j = 0; j < state_len; j++) {
            const uint32_t i = rows + j;
            state[(uint64_t)j * channels + c] =
                (i < state_len) ? state[(uint64_t)i * channels + c]
                                : conv_in[(uint64_t)(i - state_len) * channels + c];
        }
    }
}

#endif /* DS4_QWEN4EXP_PLE_REF_H */
