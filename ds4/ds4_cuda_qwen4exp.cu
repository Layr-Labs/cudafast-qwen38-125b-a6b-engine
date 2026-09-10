/*
 * Qwen4-Exp CUDA kernels.
 *
 * Every Qwen4-Exp device kernel lives here instead of in ds4_cuda.cu for one
 * reason: the Makefile compiles this unit WITHOUT --use_fast_math and with
 * -ftz=false -prec-div=true -prec-sqrt=true.  ds4_cuda.cu keeps the flags it
 * has always had, so nothing upstream of Qwen4-Exp changes.
 *
 * Qwen4-Exp needs the accurate library on three counts:
 *
 *   * The QSA rope takes cosf and sinf of theta = pos * inv_freq[d].  inv_freq
 *     starts at 1, so theta reaches the position index itself -- of the order
 *     of 1e5 at the context this model is served at.  --use_fast_math
 *     substitutes __cosf and __sinf, whose argument reduction is a single
 *     float multiply by 1/(2*pi); at that magnitude the reduced argument
 *     keeps no significant bits and the rope returns noise.  The Metal half
 *     computes the same angle in the Metal library's cos and sin, so the two
 *     backends only agree with the accurate versions here.
 *   * The GDN decay is exp(ssm_a * softplus(alpha + dt_bias)), ssm_a already
 *     being -exp(A_log) from the converter, and the
 *     QSA scores go through expf into a softmax.  -ftz=true flushes the small
 *     decays and the small softmax terms to zero.
 *   * The RMS norms divide by the row width and take rsqrtf and sqrtf, which
 *     -prec-div=false and -prec-sqrt=false relax.
 *
 * Fused multiply-add contraction is nvcc's default either way and stays on,
 * so the products the Metal shaders contract are contracted here too.
 *
 * The unit reaches ds4_cuda.cu's decode stream and weight resolver through
 * ds4_cuda_qwen4exp.cuh, and duplicates the leaf helpers it needs the way
 * ds4_rocm.cu duplicates them for the ROCm backend.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <stdint.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ds4_gpu_mgpu.h carries the complete ds4_gpu_tensor; ds4_gpu.h declares the
 * entry points this unit defines, so the compiler checks them here.  (ds4_cuda.cu
 * itself includes only the former, which is why its own prototypes go
 * unchecked.) */
#include "ds4_gpu_mgpu.h"
#include "ds4_gpu.h"
#include "ds4_cuda_qwen4exp.cuh"
#include "ds4_qwen4exp_moe_types.h"
#include "ds4_qwen4exp_hc_types.h"

#define CUDA_QK_K 256

/* ------------------------------------------------------------------
 * Leaf helpers, duplicated from ds4_cuda.cu.  Same names and same bodies:
 * the kernels below were written against them and read the same either way.
 * ------------------------------------------------------------------ */

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[12];
    uint8_t  qs[CUDA_QK_K / 2];
} cuda_block_q4_K;

static inline cudaStream_t cuda_decode_stream(void) {
    return ds4_cuda_qwen4exp_decode_stream();
}

static const char *cuda_resolve_weight_ptr(const void *model_map,
                                           uint64_t    offset,
                                           uint64_t    bytes,
                                           int         logical_tier,
                                           const char *label) {
    return ds4_cuda_qwen4exp_weight_ptr(model_map, offset, bytes,
                                        logical_tier, label);
}

static int cuda_ok(cudaError_t err, const char *what) {
    if (err == cudaSuccess) return 1;
    fprintf(stderr, "ds4: CUDA %s failed: %s\n", what, cudaGetErrorString(err));
    return 0;
}

static int cuda_current_tier(void) {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return 0;
    return dev;
}

static inline int ds4_tensor_device_idx(const ds4_gpu_tensor *t) {
    if (!t) return 0;
    int d = t->device_id;
    if (d < 0) return 0;
    return d;
}

/* Scratch for the routed MoE's pair list: three int32 arrays of n_expert and
 * one of n_tokens * n_expert_used, so tens of kilobytes at the widest prefill.
 * It is kept and grown rather than allocated per call, the way the backend
 * keeps its other per-device scratch.  Decode graphs retain these addresses;
 * growing the buffer must retire them before releasing the old allocation. */
static void *g_qwen4exp_group_scratch[16];
static uint64_t g_qwen4exp_group_bytes[16];

static void *qwen4exp_group_scratch(int tier, uint64_t bytes) {
    if (tier < 0 || tier >= 16) return NULL;
    if (g_qwen4exp_group_scratch[tier] && g_qwen4exp_group_bytes[tier] >= bytes) {
        return g_qwen4exp_group_scratch[tier];
    }
    void *next = NULL;
    if (!cuda_ok(cudaMalloc(&next, (size_t)bytes),
                 "qwen4exp MoE group scratch")) {
        return NULL;
    }
    if (g_qwen4exp_group_scratch[tier]) {
        ds4_gpu_decode_graphs_invalidate();
        cudaFree(g_qwen4exp_group_scratch[tier]);
    }
    g_qwen4exp_group_scratch[tier] = next;
    g_qwen4exp_group_bytes[tier] = bytes;
    return next;
}

static bool glm53_cuda_mul_u64(uint64_t a, uint64_t b, uint64_t *out) {
    if (!out || (a != 0u && b > UINT64_MAX / a)) return false;
    *out = a * b;
    return true;
}

static bool glm53_cuda_tensor_has(
        const ds4_gpu_tensor *tensor, uint64_t elements, uint64_t elem_size) {
    return tensor && tensor->ptr &&
        (elements == 0u || elem_size <= UINT64_MAX / elements) &&
        tensor->bytes >= elements * elem_size;
}

__device__ static float dev_f16_to_f32(uint16_t v) {
    return __half2float(*reinterpret_cast<const __half *>(&v));
}

__device__ static void dev_q4_K_get_scale_min(
        uint32_t j,
        const uint8_t *scales,
        uint8_t *d_out,
        uint8_t *m_out) {
    if (j < 4u) {
        *d_out = scales[j] & 63u;
        *m_out = scales[j + 4u] & 63u;
    } else {
        *d_out = (scales[j + 4u] & 0x0fu) | ((scales[j - 4u] >> 6u) << 4u);
        *m_out = (scales[j + 4u] >> 4u) | ((scales[j] >> 6u) << 4u);
    }
}

__device__ static float warp_sum_f32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

/* Metal's simd_sum, exactly: the butterfly leaves the total in EVERY lane, so
 * a reduction whose result the whole warp needs costs no broadcast after it.
 * Lane 0 adds the same operands in the same order as warp_sum_f32 -- at step
 * `offset` it takes lane `offset` either way -- and every other lane adds the
 * same pairs with the two sides swapped, which IEEE addition returns
 * unchanged.  So the value is the one warp_sum_f32 would have broadcast, bit
 * for bit, in one fewer shuffle. */
__device__ static float warp_sum_all_f32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, offset);
    }
    return v;
}

__device__ static float dot4_f32(float4 a, float4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

/* =========================================================================
 * Qwen4-Exp gated delta net (GDN), the CUDA twin of metal/qwen4exp_gdn.metal.
 *
 * Kernel for kernel, launch shape for launch shape and reduction for
 * reduction the same as the Metal half.  Metal's simd_sum and
 * warp_sum_all_f32 are the same five-step shuffle tree and both leave the
 * total in every lane, so the two halves reduce with the same instruction
 * count as well as the same operands.
 *
 * The convolution history and the recurrent state stay FP32 in both, so a
 * sequence split into chunks reproduces the single-chunk result bit for bit.
 * The one deliberate difference is softplus: CUDA has log1pf and Metal does
 * not, so this half sharpens the term that Metal has to spell as
 * log(1 + exp(-|x|)); the two differ only where max(x, 0) already carries
 * the value.
 * ========================================================================= */

enum {
    QWEN4EXP_GDN_DIM = 128,
    QWEN4EXP_GDN_HISTORY = 3
};

/*
 * PER-ROW STATE SNAPSHOTS, for the speculative cycle's rollback.  Twin of the
 * note on qwen4exp_gdn_args in metal/qwen4exp_gdn.metal, where the same pair
 * of buffers and the same count travel in the argument struct that backend
 * binds instead of in the kernel parameters below.
 *
 * When `n_snapshot_rows` is non-zero the two kernels mirror the carried state
 * into `conv_snapshot` and `state_snapshot` after each of the first
 * `n_snapshot_rows` tokens, so a round that accepts `a` of its drafts can
 * adopt the state as it stood after row `a` instead of rewinding and running a
 * shorter forward again.  The recurrence is token-serial in registers, so the
 * state after row k IS the state a (k + 1)-row feed leaves -- selecting it and
 * replaying it are the same value, which is what makes the shortcut exact
 * rather than close.
 *
 * Zero on every serial forward, where the guard never fires and both the
 * arithmetic and the memory traffic are what they were before the snapshots
 * existed.  Metal has to bind a stand-in buffer in that case because its
 * encoder requires every declared buffer bound; a CUDA kernel argument is an
 * ordinary pointer, so the host passes a null one and the guard keeps it from
 * ever being read.
 */

__device__ __forceinline__ static float qwen4exp_gdn_silu(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ static float qwen4exp_gdn_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

/* log(1 + exp(x)) through the branch MLX's `softplus` takes: the exponential
 * of the negative magnitude, so a large positive gate cannot overflow. */
__device__ __forceinline__ static float qwen4exp_gdn_softplus(float x) {
    return fmaxf(x, 0.0f) + log1pf(expf(-fabsf(x)));
}

/*
 * Depthwise 4-tap causal convolution, SiLU, and the query/key RMS norm.
 *
 * One block owns one (row, 128-channel block); each thread owns one channel
 * and carries that channel's 3-row history in registers, so the pass is
 * token-serial per channel and needs no scratch buffer: a thread reads
 * qkv[token][channel] before it overwrites the same element.  Blocks below
 * 2 * n_key_head are query and key heads and take the RMS norm; the rest are
 * value heads and only take the activation.
 */
__global__ static void qwen4exp_gdn_conv_kernel(
        float       *qkv,
        float       *conv_state,
        const float *conv_weight,
        float       *conv_snapshot,
        uint32_t     n_key_head,
        uint32_t     n_value_head,
        uint32_t     n_rows,
        uint32_t     n_tokens,
        uint32_t     n_snapshot_rows,
        float        qk_norm_eps) {
    const uint32_t block = blockIdx.x;
    const uint32_t row = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5u;
    const uint32_t key_blocks = 2u * n_key_head;
    const uint32_t blocks = key_blocks + n_value_head;
    if (block >= blocks || row >= n_rows) return;

    /* Two reduction slots, alternating by token.  One barrier a token then
     * bounds the skew between warps at one token, and the warp that has run
     * ahead writes the slot the warp behind is not reading. */
    __shared__ float reduce[2][4];
    const uint32_t conv_dim = blocks * QWEN4EXP_GDN_DIM;
    const uint32_t channel = block * QWEN4EXP_GDN_DIM + tid;
    const bool is_key = block < key_blocks;
    /* The reference l2-normalises the row -- `x * rsqrt(sum(x^2) + eps)`, the
     * epsilon on the SUM -- and then scales the query by `head_dim ** -0.5`.
     * So the key takes no post scale at all and the query takes 2^-3.5.
     * Twin of the Metal kernel; keep the two expressions identical. */
    const float post_scale = block < n_key_head
        ? 0x1.6a09e6p-4f
        : 1.0f;

    float *history = conv_state +
        (uint64_t)row * QWEN4EXP_GDN_HISTORY * conv_dim;
    float h0 = history[channel];
    float h1 = history[(uint64_t)conv_dim + channel];
    float h2 = history[(uint64_t)2u * conv_dim + channel];
    const float w0 = conv_weight[(uint64_t)channel * 4u + 0u];
    const float w1 = conv_weight[(uint64_t)channel * 4u + 1u];
    const float w2 = conv_weight[(uint64_t)channel * 4u + 2u];
    const float w3 = conv_weight[(uint64_t)channel * 4u + 3u];

    float raw = qkv[(uint64_t)row * n_tokens * conv_dim + channel];
    for (uint32_t token = 0; token < n_tokens; token++) {
        const uint64_t index =
            ((uint64_t)row * n_tokens + token) * conv_dim + channel;
        float acc = 0.0f;
        acc = fmaf(h0, w0, acc);
        acc = fmaf(h1, w1, acc);
        acc = fmaf(h2, w2, acc);
        acc = fmaf(raw, w3, acc);
        h0 = h1;
        h1 = h2;
        h2 = raw;

        /* The NEXT token's input, read before this token's output is stored.
         * A thread only ever reads and writes its own channel, so the two are
         * different elements -- but they travel through one pointer, so the
         * read has to come first in program order to be issued first, and the
         * memory latency then hides behind the reduction below.  The last
         * token re-reads its own element, which is still the raw value at
         * this point and is thrown away. */
        const uint64_t ahead =
            index + (token + 1u < n_tokens ? conv_dim : 0u);
        const float raw_next = qkv[ahead];

        /* The window as it stands AFTER this token, which is what a rollback
         * to length token + 1 needs.  Written before the key/value branch
         * because that branch closes the iteration with `continue` for a
         * value-head block, so a store placed after it would never run for
         * the value channels. */
        if (token < n_snapshot_rows) {
            float *slot = conv_snapshot +
                (uint64_t)token * QWEN4EXP_GDN_HISTORY * conv_dim;
            slot[channel] = h0;
            slot[(uint64_t)conv_dim + channel] = h1;
            slot[(uint64_t)2u * conv_dim + channel] = h2;
        }

        const float activated = qwen4exp_gdn_silu(acc);
        raw = raw_next;
        if (!is_key) {
            qkv[index] = activated;
            continue;
        }

        float *red = reduce[token & 1u];
        const float sumsq = warp_sum_f32(activated * activated);
        if (lane == 0u) red[warp] = sumsq;
        __syncthreads();
        float total = lane < 4u ? red[lane] : 0.0f;
        total = warp_sum_all_f32(total);
        qkv[index] = activated *
            rsqrtf(total + qk_norm_eps) * post_scale;
    }

    history[channel] = h0;
    history[(uint64_t)conv_dim + channel] = h1;
    history[(uint64_t)2u * conv_dim + channel] = h2;
}

/*
 * The delta rule itself, token-serial inside the kernel like KDA and like the
 * reference: one block owns one (row, value head, four value rows), one warp
 * owns one value row, and each lane owns four adjacent key columns.
 */
__global__ static void qwen4exp_gdn_recurrence_kernel(
        float       *__restrict__ out,
        float       *__restrict__ state,
        const float *__restrict__ qkv,
        const float *__restrict__ raw_alpha,
        const float *__restrict__ raw_beta,
        const float *__restrict__ a_log,
        const float *__restrict__ dt_bias,
        float       *state_snapshot,
        uint32_t     n_key_head,
        uint32_t     n_value_head,
        uint32_t     n_rows,
        uint32_t     n_tokens,
        uint32_t     head_layout,
        uint32_t     n_snapshot_rows) {
    const uint32_t head = blockIdx.x;
    const uint32_t value = blockIdx.y * 4u + (threadIdx.x >> 5u);
    const uint32_t row = blockIdx.z;
    const uint32_t lane = threadIdx.x & 31u;
    if (head >= n_value_head || value >= QWEN4EXP_GDN_DIM || row >= n_rows) {
        return;
    }

    const uint32_t key_dim = n_key_head * QWEN4EXP_GDN_DIM;
    const uint32_t value_dim = n_value_head * QWEN4EXP_GDN_DIM;
    const uint32_t conv_dim = 2u * key_dim + value_dim;
    /* Grouped order pairs value head hv with key head hv / repeats; the
     * converter's tiled order pairs it with hv % n_key_head. */
    const uint32_t key_head = head_layout != 0u
        ? head % n_key_head
        : head / (n_value_head / n_key_head);
    const uint32_t k0 = lane * 4u;

    float4 *state_ptr = (float4 *)(state +
        ((((uint64_t)row * n_value_head + head) * QWEN4EXP_GDN_DIM) + value) *
        QWEN4EXP_GDN_DIM + k0);
    float4 h = *state_ptr;
    /* ssm_a IS ALREADY -exp(A_log); see the note in metal/qwen4exp_gdn.metal.
     * Twin of that kernel -- keep the two expressions identical. */
    const float decay_coeff = a_log[head];
    const float bias = dt_bias[head];

    for (uint32_t token = 0; token < n_tokens; token++) {
        const uint64_t slot = (uint64_t)row * n_tokens + token;
        const uint64_t base = slot * conv_dim + key_head * QWEN4EXP_GDN_DIM;
        const float4 q4 = *(const float4 *)(qkv + base + k0);
        const float4 k4 = *(const float4 *)(qkv + base + key_dim + k0);
        const float v_row = qkv[slot * conv_dim + 2u * (uint64_t)key_dim +
            head * QWEN4EXP_GDN_DIM + value];
        const uint64_t gate = slot * n_value_head + head;
        const float g = expf(decay_coeff *
            qwen4exp_gdn_softplus(raw_alpha[gate] + bias));
        const float beta = qwen4exp_gdn_sigmoid(raw_beta[gate]);

        h.x *= g;
        h.y *= g;
        h.z *= g;
        h.w *= g;
        const float hk = warp_sum_all_f32(dot4_f32(h, k4));
        const float delta_v = (v_row - hk) * beta;
        h.x = fmaf(k4.x, delta_v, h.x);
        h.y = fmaf(k4.y, delta_v, h.y);
        h.z = fmaf(k4.z, delta_v, h.z);
        h.w = fmaf(k4.w, delta_v, h.w);
        const float result = warp_sum_all_f32(dot4_f32(h, q4));
        if (lane == 0u) {
            out[slot * value_dim + head * QWEN4EXP_GDN_DIM + value] = result;
        }

        /* The recurrent state AFTER this token.  Same element this thread owns
         * in the live state, one slot per row. */
        if (token < n_snapshot_rows) {
            const uint64_t stride = (uint64_t)n_rows * n_value_head *
                QWEN4EXP_GDN_DIM * QWEN4EXP_GDN_DIM;
            float4 *snap = (float4 *)(state_snapshot +
                (uint64_t)token * stride +
                ((((uint64_t)row * n_value_head + head) * QWEN4EXP_GDN_DIM) +
                 value) * QWEN4EXP_GDN_DIM + k0);
            *snap = h;
        }
    }
    *state_ptr = h;
}

/* Sigmoid-gated RMS output norm.  The weight is a plain scale, not an
 * offset-baked one, so it multiplies the normalised row directly. */
__global__ static void qwen4exp_gdn_output_kernel(
        float       *out,
        const float *output_gate,
        const float *output_norm,
        uint32_t     n_value_head,
        uint32_t     n_rows,
        uint32_t     n_tokens,
        float        norm_eps) {
    const uint32_t token = blockIdx.x;
    const uint32_t head = blockIdx.y;
    const uint32_t row = blockIdx.z;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5u;
    if (token >= n_tokens || head >= n_value_head || row >= n_rows) return;
    __shared__ float partial[4];
    const uint32_t value_dim = n_value_head * QWEN4EXP_GDN_DIM;
    const uint64_t base = ((uint64_t)row * n_tokens + token) * value_dim +
        head * QWEN4EXP_GDN_DIM;
    const float raw = out[base + tid];
    float total = warp_sum_f32(raw * raw);
    if (lane == 0u) partial[warp] = total;
    __syncthreads();
    total = lane < 4u ? partial[lane] : 0.0f;
    total = warp_sum_all_f32(total);
    const float scale = rsqrtf(total / (float)QWEN4EXP_GDN_DIM + norm_eps);
    out[base + tid] = raw * scale * output_norm[tid] *
        qwen4exp_gdn_sigmoid(output_gate[base + tid]);
}

static const float *qwen4exp_gdn_weight_f32(
        const void *model_map,
        uint64_t    model_size,
        uint64_t    offset,
        uint64_t    elements,
        int         logical_tier,
        const char *label) {
    uint64_t bytes = 0;
    if (!model_map || !glm53_cuda_mul_u64(elements, sizeof(float), &bytes) ||
        offset > model_size || bytes > model_size - offset) {
        fprintf(stderr, "ds4: qwen4exp %s range is outside the mapped model\n",
                label ? label : "weight");
        return NULL;
    }
    return (const float *)cuda_resolve_weight_ptr(
        model_map, offset, bytes, logical_tier, label);
}

/* The host half of qwen4exp_gpu_gdn_run in ds4_metal.m: the same validation,
 * the same four weight ranges, and the same three launches in stream order. */
static int qwen4exp_cuda_gdn_run(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *conv_state,
        ds4_gpu_tensor       *recurrent_state,
        ds4_gpu_tensor       *conv_snapshot,
        ds4_gpu_tensor       *state_snapshot,
        uint32_t              n_snapshot_rows,
        ds4_gpu_tensor       *qkv,
        const ds4_gpu_tensor *raw_alpha,
        const ds4_gpu_tensor *raw_beta,
        const ds4_gpu_tensor *output_gate,
        /* One slab per tensor: ssm_conv1d, ssm_a, ssm_dt.bias and ssm_norm are
         * four GGUF tensors and a shard boundary can fall between any two of
         * them.  Resolving them all through the convolution's mapping would
         * read the right offsets out of the wrong file on a split set. */
        const ds4_gpu_qwen4exp_slab *conv_weight_slab,
        const ds4_gpu_qwen4exp_slab *a_log_slab,
        const ds4_gpu_qwen4exp_slab *dt_bias_slab,
        const ds4_gpu_qwen4exp_slab *output_norm_slab,
        uint32_t              n_key_head,
        uint32_t              n_value_head,
        uint32_t              n_rows,
        uint32_t              n_tokens,
        uint32_t              head_layout,
        float                 qk_norm_eps,
        float                 norm_eps,
        const char           *label) {
    uint64_t key_dim = 0, value_dim = 0, conv_dim = 0, slots = 0;
    uint64_t qkv_elements = 0, out_elements = 0, gate_elements = 0;
    uint64_t conv_elements = 0, state_elements = 0, state_rows = 0;
    if (n_key_head == 0 || n_value_head == 0 || n_rows == 0 ||
        n_tokens == 0 || n_value_head % n_key_head != 0 ||
        head_layout > (uint32_t)DS4_QWEN4EXP_GDN_HEADS_TILED ||
        !glm53_cuda_mul_u64(n_key_head, QWEN4EXP_GDN_DIM, &key_dim) ||
        !glm53_cuda_mul_u64(n_value_head, QWEN4EXP_GDN_DIM, &value_dim) ||
        !glm53_cuda_mul_u64(n_rows, n_tokens, &slots) ||
        !glm53_cuda_mul_u64(2u, key_dim, &conv_dim)) {
        fprintf(stderr, "ds4: qwen4exp GDN %s received invalid shapes\n",
                label);
        return 0;
    }
    conv_dim += value_dim;
    if (!glm53_cuda_mul_u64(slots, conv_dim, &qkv_elements) ||
        !glm53_cuda_mul_u64(slots, value_dim, &out_elements) ||
        !glm53_cuda_mul_u64(slots, n_value_head, &gate_elements) ||
        !glm53_cuda_mul_u64((uint64_t)n_rows * QWEN4EXP_GDN_HISTORY,
                            conv_dim, &conv_elements) ||
        !glm53_cuda_mul_u64((uint64_t)n_rows * n_value_head,
                            QWEN4EXP_GDN_DIM, &state_rows) ||
        !glm53_cuda_mul_u64(state_rows, QWEN4EXP_GDN_DIM, &state_elements) ||
        !glm53_cuda_tensor_has(qkv, qkv_elements, sizeof(float)) ||
        !glm53_cuda_tensor_has(out, out_elements, sizeof(float)) ||
        !glm53_cuda_tensor_has(output_gate, out_elements, sizeof(float)) ||
        !glm53_cuda_tensor_has(raw_alpha, gate_elements, sizeof(float)) ||
        !glm53_cuda_tensor_has(raw_beta, gate_elements, sizeof(float)) ||
        !glm53_cuda_tensor_has(conv_state, conv_elements, sizeof(float)) ||
        !glm53_cuda_tensor_has(recurrent_state, state_elements,
                               sizeof(float))) {
        fprintf(stderr, "ds4: qwen4exp GDN %s received invalid buffers\n",
                label);
        return 0;
    }

    /* The snapshots are optional and travel together: a caller that asks for
     * rows must supply both buffers, big enough for that many slots.  A short
     * buffer is a refusal rather than a clamp -- a silently shortened snapshot
     * would roll back to the wrong row.  The convolution slot index carries no
     * `row` term, so more than one sequence row would have them collide; only
     * the speculative verify asks for slots and it is always one row wide, so
     * a second row here is a caller that has outgrown the layout. */
    if (n_snapshot_rows > 0) {
        uint64_t want_conv = 0, want_state = 0;
        if (n_rows != 1u ||
            n_snapshot_rows >= n_tokens ||
            !conv_snapshot || !state_snapshot ||
            !glm53_cuda_mul_u64(n_snapshot_rows, conv_elements, &want_conv) ||
            !glm53_cuda_mul_u64(n_snapshot_rows, state_elements, &want_state) ||
            !glm53_cuda_tensor_has(conv_snapshot, want_conv, sizeof(float)) ||
            !glm53_cuda_tensor_has(state_snapshot, want_state,
                                   sizeof(float))) {
            fprintf(stderr,
                    "ds4: qwen4exp GDN %s asked for %u snapshot rows over %u "
                    "tokens and %u sequence rows without buffers to hold "
                    "them\n",
                    label, n_snapshot_rows, n_tokens, n_rows);
            return 0;
        }
    }

    const int logical_tier = ds4_tensor_device_idx(out);
    const float *conv_weight = qwen4exp_gdn_weight_f32(
        conv_weight_slab->map, conv_weight_slab->map_size,
        conv_weight_slab->offset, conv_dim * 4u,
        logical_tier, "GDN convolution");
    const float *a_log = qwen4exp_gdn_weight_f32(
        a_log_slab->map, a_log_slab->map_size, a_log_slab->offset,
        n_value_head, logical_tier, "GDN A_log");
    const float *dt_bias = qwen4exp_gdn_weight_f32(
        dt_bias_slab->map, dt_bias_slab->map_size, dt_bias_slab->offset,
        n_value_head, logical_tier, "GDN dt bias");
    const float *output_norm = qwen4exp_gdn_weight_f32(
        output_norm_slab->map, output_norm_slab->map_size,
        output_norm_slab->offset, QWEN4EXP_GDN_DIM,
        logical_tier, "GDN output norm");
    if (!conv_weight || !a_log || !dt_bias || !output_norm) return 0;

    cudaStream_t stream = cuda_decode_stream();
    const uint32_t blocks = 2u * n_key_head + n_value_head;

    qwen4exp_gdn_conv_kernel<<<dim3(blocks, n_rows, 1u),
                               QWEN4EXP_GDN_DIM, 0, stream>>>(
            (float *)qkv->ptr, (float *)conv_state->ptr, conv_weight,
            conv_snapshot ? (float *)conv_snapshot->ptr : NULL,
            n_key_head, n_value_head, n_rows, n_tokens, n_snapshot_rows,
            qk_norm_eps);
    if (!cuda_ok(cudaGetLastError(), "qwen4exp GDN convolution launch")) {
        return 0;
    }

    qwen4exp_gdn_recurrence_kernel<<<
            dim3(n_value_head, QWEN4EXP_GDN_DIM / 4u, n_rows),
            QWEN4EXP_GDN_DIM, 0, stream>>>(
            (float *)out->ptr, (float *)recurrent_state->ptr,
            (const float *)qkv->ptr, (const float *)raw_alpha->ptr,
            (const float *)raw_beta->ptr, a_log, dt_bias,
            state_snapshot ? (float *)state_snapshot->ptr : NULL,
            n_key_head, n_value_head, n_rows, n_tokens, head_layout,
            n_snapshot_rows);
    if (!cuda_ok(cudaGetLastError(), "qwen4exp GDN recurrence launch")) {
        return 0;
    }

    qwen4exp_gdn_output_kernel<<<dim3(n_tokens, n_value_head, n_rows),
                                 QWEN4EXP_GDN_DIM, 0, stream>>>(
            (float *)out->ptr, (const float *)output_gate->ptr, output_norm,
            n_value_head, n_rows, n_tokens, norm_eps);
    return cuda_ok(cudaGetLastError(), "qwen4exp GDN output norm launch");
}

extern "C" int ds4_gpu_qwen4exp_gdn_prefill(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *conv_state,
        ds4_gpu_tensor       *recurrent_state,
        ds4_gpu_tensor       *conv_snapshot,
        ds4_gpu_tensor       *state_snapshot,
        uint32_t              n_snapshot_rows,
        ds4_gpu_tensor       *qkv,
        const ds4_gpu_tensor *raw_alpha,
        const ds4_gpu_tensor *raw_beta,
        const ds4_gpu_tensor *output_gate,
        /* One slab per tensor: ssm_conv1d, ssm_a, ssm_dt.bias and ssm_norm are
         * four GGUF tensors and a shard boundary can fall between any two of
         * them.  Resolving them all through the convolution's mapping would
         * read the right offsets out of the wrong file on a split set. */
        const ds4_gpu_qwen4exp_slab *conv_weight_slab,
        const ds4_gpu_qwen4exp_slab *a_log_slab,
        const ds4_gpu_qwen4exp_slab *dt_bias_slab,
        const ds4_gpu_qwen4exp_slab *output_norm_slab,
        uint32_t              n_key_head,
        uint32_t              n_value_head,
        uint32_t              n_tokens,
        uint32_t              head_layout,
        float                 qk_norm_eps,
        float                 norm_eps) {
    return qwen4exp_cuda_gdn_run(
        out, conv_state, recurrent_state, conv_snapshot, state_snapshot,
        n_snapshot_rows, qkv, raw_alpha, raw_beta,
        output_gate, conv_weight_slab, a_log_slab, dt_bias_slab,
        output_norm_slab,
        n_key_head, n_value_head, 1u, n_tokens, head_layout,
        qk_norm_eps, norm_eps, "prefill");
}

extern "C" int ds4_gpu_qwen4exp_gdn_decode(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *conv_state,
        ds4_gpu_tensor       *recurrent_state,
        ds4_gpu_tensor       *qkv,
        const ds4_gpu_tensor *raw_alpha,
        const ds4_gpu_tensor *raw_beta,
        const ds4_gpu_tensor *output_gate,
        /* One slab per tensor: ssm_conv1d, ssm_a, ssm_dt.bias and ssm_norm are
         * four GGUF tensors and a shard boundary can fall between any two of
         * them.  Resolving them all through the convolution's mapping would
         * read the right offsets out of the wrong file on a split set. */
        const ds4_gpu_qwen4exp_slab *conv_weight_slab,
        const ds4_gpu_qwen4exp_slab *a_log_slab,
        const ds4_gpu_qwen4exp_slab *dt_bias_slab,
        const ds4_gpu_qwen4exp_slab *output_norm_slab,
        uint32_t              n_key_head,
        uint32_t              n_value_head,
        uint32_t              n_rows,
        uint32_t              head_layout,
        float                 qk_norm_eps,
        float                 norm_eps) {
    /* A one-token forward has no row to roll back to but its own start, which
     * is what the round-start path already keeps. */
    return qwen4exp_cuda_gdn_run(
        out, conv_state, recurrent_state, NULL, NULL, 0u,
        qkv, raw_alpha, raw_beta,
        output_gate, conv_weight_slab, a_log_slab, dt_bias_slab,
        output_norm_slab,
        n_key_head, n_value_head, n_rows, 1u, head_layout,
        qk_norm_eps, norm_eps, "decode");
}

/* ------------------------------------------------------------------
 * qwen4exp routed MoE.  Twin of metal/qwen4exp_moe.metal: the router scores
 * raw float32 logits with no bias and softmaxes the selected logits only, the
 * SwiGLU has no clamp, and the expert types vary per block.  The K-quant
 * accessors reuse dev_q4_K_get_scale_min so gate/up follow the GLM expert path.
 * ------------------------------------------------------------------ */

typedef struct {
    uint16_t d;
    uint16_t m;
    uint8_t  qh[4];
    uint8_t  qs[16];
} cuda_block_q5_1;

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[12];
    uint8_t  qh[CUDA_QK_K / 8];
    uint8_t  qs[CUDA_QK_K / 2];
} cuda_block_q5_K;

typedef struct {
    uint8_t ql[CUDA_QK_K / 2];
    uint8_t qh[CUDA_QK_K / 4];
    int8_t  scales[CUDA_QK_K / 16];
    uint16_t d;
} cuda_block_q6_K;

__device__ __forceinline__ static float dev_qwen4exp_q4_K_value(
        const char *row, uint32_t k) {
    const cuda_block_q4_K *xb = (const cuda_block_q4_K *)row + (k / 256u);
    const uint32_t idx = k % 256u;
    const uint32_t group = idx / 32u;
    const uint32_t l = idx % 32u;
    uint8_t sc = 0, m = 0;
    dev_q4_K_get_scale_min(group, xb->scales, &sc, &m);
    const uint8_t byte = xb->qs[(group >> 1u) * 32u + l];
    const uint32_t q = (group & 1u) ? (uint32_t)(byte >> 4u)
                                    : (uint32_t)(byte & 0x0fu);
    return dev_f16_to_f32(xb->d) * (float)sc * (float)q -
           dev_f16_to_f32(xb->dmin) * (float)m;
}

/* ggml block_q5_1: 32 elements in 24 bytes.  See ggml-quants.c
 * dequantize_row_q5_1(): the low nibble of qs[j] is element j and the high
 * nibble is element j + 16, each taking its fifth bit from bit j respectively
 * bit j + 16 of the 32-bit qh. */
__device__ __forceinline__ static float dev_qwen4exp_q5_1_value(
        const char *row, uint32_t k) {
    const cuda_block_q5_1 *xb = (const cuda_block_q5_1 *)row + (k / 32u);
    const uint32_t idx = k % 32u;
    const uint32_t qh = (uint32_t)xb->qh[0] | ((uint32_t)xb->qh[1] << 8u) |
                        ((uint32_t)xb->qh[2] << 16u) | ((uint32_t)xb->qh[3] << 24u);
    const uint32_t j = idx & 15u;
    uint32_t q;
    if (idx < 16u) q = (uint32_t)(xb->qs[j] & 0x0fu) | (((qh >> j) & 1u) << 4u);
    else q = (uint32_t)(xb->qs[j] >> 4u) | (((qh >> (j + 16u)) & 1u) << 4u);
    return (float)q * dev_f16_to_f32(xb->d) + dev_f16_to_f32(xb->m);
}

__device__ __forceinline__ static float dev_qwen4exp_q8_0_value(
        const char *row, uint32_t k) {
    const char *blk = row + (uint64_t)(k / 32u) * 34u;
    const uint16_t d = (uint16_t)((uint8_t)blk[0]) |
                       (uint16_t)((uint16_t)(uint8_t)blk[1] << 8u);
    return dev_f16_to_f32(d) * (float)(int8_t)blk[2u + (k % 32u)];
}

__device__ __forceinline__ static float dev_qwen4exp_f32_value(
        const char *row, uint32_t k) {
    return ((const float *)row)[k];
}

/* Q5_K is Q4_K plus a high-bit plane: the same six-bit scale/min pair, the
 * same nibble, and bit `group` of qh[l] as the fifth bit.  See ggml-quants.c
 * dequantize_row_q5_K(). */
__device__ __forceinline__ static float dev_qwen4exp_q5_K_value(
        const char *row, uint32_t k) {
    const cuda_block_q5_K *xb = (const cuda_block_q5_K *)row + (k / 256u);
    const uint32_t idx = k % 256u;
    const uint32_t group = idx / 32u;
    const uint32_t l = idx % 32u;
    uint8_t sc = 0, m = 0;
    dev_q4_K_get_scale_min(group, xb->scales, &sc, &m);
    const uint32_t shift = (group & 1u) * 4u;
    uint32_t q = ((uint32_t)xb->qs[(group >> 1u) * 32u + l] >> shift) & 0x0fu;
    if (xb->qh[l] & (uint8_t)(1u << group)) q += 16u;
    return dev_f16_to_f32(xb->d) * (float)sc * (float)q -
           dev_f16_to_f32(xb->dmin) * (float)m;
}

/* Q6_K: one signed six-bit quant per element, four nibble/high-bit quarters
 * per 128 elements and an int8 scale per 16.  See ggml-quants.c
 * dequantize_row_q6_K(). */
__device__ __forceinline__ static float dev_qwen4exp_q6_K_value(
        const char *row, uint32_t k) {
    const cuda_block_q6_K *xb = (const cuda_block_q6_K *)row + (k / 256u);
    const uint32_t idx = k % 256u;
    const uint32_t n128 = idx >> 7u;
    const uint32_t r = idx & 127u;
    const uint32_t l = r & 31u;
    const uint32_t quarter = r >> 5u;
    const uint8_t *ql = xb->ql + n128 * 64u;
    const uint32_t hi = ((uint32_t)xb->qh[n128 * 32u + l] >> (quarter * 2u)) & 3u;
    const int8_t *sc = xb->scales + n128 * 8u + l / 16u + quarter * 2u;
    const uint32_t lo = (quarter & 1u) ? (uint32_t)ql[32u + l] : (uint32_t)ql[l];
    const uint32_t q = (quarter < 2u) ? ((lo & 0x0fu) | (hi << 4u))
                                      : ((lo >> 4u) | (hi << 4u));
    return dev_f16_to_f32(xb->d) * (float)(*sc) * (float)((int32_t)q - 32);
}

/* The hyper-connection inject weight's own set: F32 in the target, Q8_0 in the
 * MTP head.  Its cases come from ds4_qwen4exp_hc_types.h -- the table the
 * loader accepts from -- and reuse the accessors above. */
__device__ __forceinline__ static float dev_qwen4exp_inject_value(
        uint32_t type, const char *row, uint32_t k);

/* The cases come from ds4_qwen4exp_moe_types.h, the same table the loader
 * accepts from and metal/qwen4exp_moe.metal expands. */
__device__ __forceinline__ static float dev_qwen4exp_weight_value(
        uint32_t type, const char *row, uint32_t k) {
#define DS4_QWEN4EXP_VALUE_CASE(name, id) \
    case (uint32_t)(id): return dev_qwen4exp_ ## name ## _value(row, k);
    switch (type) {
    DS4_QWEN4EXP_MOE_TYPES(DS4_QWEN4EXP_VALUE_CASE)
    }
#undef DS4_QWEN4EXP_VALUE_CASE
    return 0.0f;
}

__device__ __forceinline__ static float dev_qwen4exp_inject_value(
        uint32_t type, const char *row, uint32_t k) {
#define DS4_QWEN4EXP_INJECT_CASE(name, id) \
    case (uint32_t)(id): return dev_qwen4exp_ ## name ## _value(row, k);
    switch (type) {
    DS4_QWEN4EXP_HC_INJECT_TYPES(DS4_QWEN4EXP_INJECT_CASE)
    }
#undef DS4_QWEN4EXP_INJECT_CASE
    return 0.0f;
}

/* ===========================================================================
 * ONE ARITHMETIC FOR THE EXPERT PROJECTIONS, AT EVERY WIDTH.
 *
 * The scalar accessors above decode one weight ELEMENT at a time and multiply
 * it by one float activation.  That costs a switch, a divide and two half
 * conversions for every element of every row, and it cannot share anything
 * across the rows of a prefill.
 *
 * The rule below is the one the dense Q8_0 projections already follow, applied
 * to the experts.  The activation row is quantised to Q8_0 groups of 32 -- per
 * row, so it never sees the batch.  A weight row is then walked in the same
 * groups of 32 in ascending order, and each group contributes
 *
 *     acc += (wa * xscale) * (float)dot        dot  = int32 dp4a of the group
 *     acc += (wb * xscale) * (float)xsum       xsum = int32 sum of the group
 *
 * with `wa` and `wb` decoded ONCE for the group.  Both integer terms are
 * exact, so the only floating-point operations per group are the two shown,
 * always in that order.  The pair falls out of the quantisation formats:
 *
 *   Q8_0   value = d*q                    wa = d,        wb = 0
 *   Q5_1   value = d*q + m                wa = d,        wb = m
 *   Q4_K   value = d*sc*q - dmin*m        wa = d*sc,     wb = -dmin*m
 *   Q5_K   as Q4_K with a fifth bit       wa = d*sc,     wb = -dmin*m
 *   Q6_K   value = d*sc*(q - 32)          wa = d*sc,     wb = 0, and the
 *                                         quant is stored as q - 32 so the
 *                                         offset needs no second term
 *
 * Q6_K is the one format whose scale changes inside a group of 32 -- it has
 * one per 16 -- so it reports two halves and contributes one term per half.
 * Every other format reports one.  `halves` depends only on the type, so a
 * warp never diverges on it.
 *
 * WHY THIS IS THE WIDTH-INVARIANT RULE.  Nothing in it mentions the batch: a
 * row's groups, their order, the two floats per group and the reduction that
 * follows are the same whether the call carries one row or five hundred.  The
 * kernels below are one template over the row tile R, and R = 1 is that same
 * template, so a one-row decode is not a second kernel that has to be kept in
 * agreement -- it IS the kernel.  DS4_QWEN4EXP_MOE_R forces a tile so a test
 * can run R = 1 and R = 4 over the same rows and compare the bits.
 *
 * This MOVES NUMBERS once, against the per-element path it replaces, for the
 * same reason llama.cpp's quantised matmuls differ from a scalar dequantise:
 * the activation is quantised and the scale is factored out of the sum.  The
 * goldens are re-authored once for it.
 * ======================================================================== */

/* The case labels come from the one type table, so a type added there without
 * a case here is a compile error rather than a silent zero. */
enum {
#define DS4_QWEN4EXP_TYPE_ENUM(name, id) DS4_QWEN4EXP_TY_ ## name = (id),
    DS4_QWEN4EXP_MOE_TYPES(DS4_QWEN4EXP_TYPE_ENUM)
#undef DS4_QWEN4EXP_TYPE_ENUM
};

__device__ __forceinline__ static int32_t qwen4exp_load_i8x4(const int8_t *p) {
    const uint8_t *u = (const uint8_t *)p;
    return (int32_t)((uint32_t)u[0] | ((uint32_t)u[1] << 8) |
                     ((uint32_t)u[2] << 16) | ((uint32_t)u[3] << 24));
}

template <int N>
__device__ __forceinline__ static int32_t qwen4exp_dp4a(const int8_t *a,
                                                        const int8_t *b) {
    /* b is an activation group in the quantizer's device scratch. Its
     * offset consists of four-byte index arrays and 32-byte quant groups;
     * the second Q6_K half adds 16 bytes, preserving word alignment.
     * Read its four signed bytes as the same little-endian word the byte
     * packer builds, without issuing four separate global byte loads.
     * a is decoded register-local weight data and keeps the byte packer.
     * The dp4a operands and integer accumulator order are unchanged. */
    const int32_t *activation_words = (const int32_t *)(const void *)b;
    int32_t d = 0;
#pragma unroll
    for (int i = 0; i < N; i += 4) {
        d = __dp4a(qwen4exp_load_i8x4(a + i), activation_words[i / 4], d);
    }
    return d;
}

/* Is this weight address a whole number of four-byte words from zero?
 *
 * The quantised block types below are declared with two-byte alignment, so the
 * compiler cannot assume more, and the payload loops read them one byte at a
 * time.  A GGUF tensor starts on a 32-byte boundary and every block and row
 * stride of the expert slabs is a multiple of four, so the answer is yes for
 * every lane of every warp on the pinned checkpoint -- and the branch below is
 * therefore warp uniform, not a divergence.  It is asked rather than assumed
 * because an unaligned word load on the device is a fault, not a slow path. */
__device__ __forceinline__ static bool qwen4exp_word_aligned(const void *p) {
    return (((uintptr_t)p) & 3u) == 0u;
}

/* Decode one 32-element group of a quantised weight row into int8 quants and
 * the one or two (wa, wb) pairs that turn an integer dot into the row's
 * contribution.  Called once per group per output row, not once per element.
 *
 * WHY THE WORD PATHS.  A group is thirty-two quants; read a byte at a time
 * that is thirty-two load instructions per group per row, and these kernels
 * decode two or three groups per thread per staging step, so the byte loads
 * outnumber the tensor-core instructions they feed by more than ten to one.
 * Reading the same payload as eight words and splitting the nibbles in
 * registers issues four times fewer loads for the same bytes.  The nibbles,
 * their order and the values they produce are unchanged, so every dot is bit
 * for bit what the byte loop produced. */
__device__ __forceinline__ static void dev_qwen4exp_group_decode(
        uint32_t type, const char *row, uint32_t g,
        int8_t *wq, float *wa, float *wb, int *halves) {
    *halves = 1;
    wa[1] = 0.0f;
    wb[0] = 0.0f;
    wb[1] = 0.0f;
    switch (type) {
    case (uint32_t)DS4_QWEN4EXP_TY_q8_0: {
        const char *blk = row + (uint64_t)g * 34u;
        const uint16_t d = (uint16_t)((uint8_t)blk[0]) |
                           (uint16_t)((uint16_t)(uint8_t)blk[1] << 8u);
        wa[0] = dev_f16_to_f32(d);
        const uint8_t *payload = (const uint8_t *)blk + 2u;
        const uintptr_t address = (uintptr_t)payload;
        const uint32_t shift = (uint32_t)(address & 3u) * 8u;
        const uint32_t *words =
            (const uint32_t *)(const void *)(address - (address & 3u));
        uint32_t previous = words[0];
#pragma unroll
        for (int i = 0; i < 7; i++) {
            const uint32_t next = words[i + 1];
            const uint32_t packed = __funnelshift_r(previous, next, shift);
            wq[i * 4 + 0] = (int8_t)(packed & 0xffu);
            wq[i * 4 + 1] = (int8_t)((packed >> 8u) & 0xffu);
            wq[i * 4 + 2] = (int8_t)((packed >> 16u) & 0xffu);
            wq[i * 4 + 3] = (int8_t)(packed >> 24u);
            previous = next;
        }
        /* The final four bytes are still inside this 34-byte block, but the
         * next aligned word can extend past it.  Keep that group on the byte
         * path rather than issue a speculative read into the next row. */
#pragma unroll
        for (int i = 28; i < 32; i++) wq[i] = (int8_t)payload[i];
        return;
    }
    case (uint32_t)DS4_QWEN4EXP_TY_q5_1: {
        const cuda_block_q5_1 *xb = (const cuda_block_q5_1 *)row + g;
        wa[0] = dev_f16_to_f32(xb->d);
        wb[0] = dev_f16_to_f32(xb->m);
        if (qwen4exp_word_aligned(xb)) {
            /* A q5_1 block is 24 bytes, so every block of a 4-byte-aligned row
             * is 4-byte aligned, and so are its qh at offset 4 and its qs at
             * offset 8.  Read both as words. */
            const uint32_t *qw = (const uint32_t *)(const void *)xb;
            const uint32_t qh = qw[1];
#pragma unroll
            for (int k = 0; k < 4; k++) {
                const uint32_t v = qw[2 + k];
                const uint32_t lo = v & 0x0f0f0f0fu;
                const uint32_t hi = (v >> 4u) & 0x0f0f0f0fu;
#pragma unroll
                for (int b = 0; b < 4; b++) {
                    const int j = k * 4 + b;
                    wq[j] = (int8_t)(((lo >> (b * 8)) & 0xffu) |
                                     (((qh >> j) & 1u) << 4u));
                    wq[16 + j] = (int8_t)(((hi >> (b * 8)) & 0xffu) |
                                          (((qh >> (j + 16u)) & 1u) << 4u));
                }
            }
            return;
        }
        const uint32_t qh = (uint32_t)xb->qh[0] | ((uint32_t)xb->qh[1] << 8u) |
                            ((uint32_t)xb->qh[2] << 16u) |
                            ((uint32_t)xb->qh[3] << 24u);
#pragma unroll
        for (int j = 0; j < 16; j++) {
            wq[j] = (int8_t)(((uint32_t)(xb->qs[j] & 0x0fu)) |
                             (((qh >> j) & 1u) << 4u));
            wq[16 + j] = (int8_t)(((uint32_t)(xb->qs[j] >> 4u)) |
                                  (((qh >> (j + 16u)) & 1u) << 4u));
        }
        return;
    }
    case (uint32_t)DS4_QWEN4EXP_TY_q4_K: {
        const cuda_block_q4_K *xb = (const cuda_block_q4_K *)row + (g / 8u);
        const uint32_t grp = g % 8u;
        uint8_t sc = 0, m = 0;
        dev_q4_K_get_scale_min(grp, xb->scales, &sc, &m);
        wa[0] = dev_f16_to_f32(xb->d) * (float)sc;
        wb[0] = -dev_f16_to_f32(xb->dmin) * (float)m;
        const uint8_t *qs = xb->qs + (grp >> 1u) * 32u;
        const uint32_t shift = (grp & 1u) ? 4u : 0u;
        if (qwen4exp_word_aligned(qs)) {
            const uint32_t *qw = (const uint32_t *)(const void *)qs;
#pragma unroll
            for (int i = 0; i < 8; i++) {
                /* Shifting the whole word moves each byte's wanted nibble to
                 * that byte's low four bits; the mask then takes exactly the
                 * four bits the byte-at-a-time loop took. */
                const uint32_t v = (qw[i] >> shift) & 0x0f0f0f0fu;
                wq[i * 4 + 0] = (int8_t)(v & 0xffu);
                wq[i * 4 + 1] = (int8_t)((v >> 8u) & 0xffu);
                wq[i * 4 + 2] = (int8_t)((v >> 16u) & 0xffu);
                wq[i * 4 + 3] = (int8_t)(v >> 24u);
            }
            return;
        }
#pragma unroll
        for (int i = 0; i < 32; i++) {
            wq[i] = (int8_t)(((uint32_t)qs[i] >> shift) & 0x0fu);
        }
        return;
    }
    case (uint32_t)DS4_QWEN4EXP_TY_q5_K: {
        const cuda_block_q5_K *xb = (const cuda_block_q5_K *)row + (g / 8u);
        const uint32_t grp = g % 8u;
        uint8_t sc = 0, m = 0;
        dev_q4_K_get_scale_min(grp, xb->scales, &sc, &m);
        wa[0] = dev_f16_to_f32(xb->d) * (float)sc;
        wb[0] = -dev_f16_to_f32(xb->dmin) * (float)m;
        const uint8_t *qs = xb->qs + (grp >> 1u) * 32u;
        const uint32_t shift = (grp & 1u) * 4u;
        if (qwen4exp_word_aligned(qs) && qwen4exp_word_aligned(xb->qh)) {
            const uint32_t *qw = (const uint32_t *)(const void *)qs;
            const uint32_t *hw = (const uint32_t *)(const void *)xb->qh;
            const uint32_t hbit = 0x01010101u << grp;
#pragma unroll
            for (int i = 0; i < 8; i++) {
                const uint32_t v = (qw[i] >> shift) & 0x0f0f0f0fu;
                /* The fifth bit is byte i's bit `grp`; the compare turns each
                 * byte that carries it into 0x10, which is the +16 the
                 * byte-at-a-time loop added. */
                const uint32_t h = hw[i] & hbit;
                const uint32_t add = ((h >> grp) & 0x01010101u) << 4u;
                const uint32_t q = v | add;
                wq[i * 4 + 0] = (int8_t)(q & 0xffu);
                wq[i * 4 + 1] = (int8_t)((q >> 8u) & 0xffu);
                wq[i * 4 + 2] = (int8_t)((q >> 16u) & 0xffu);
                wq[i * 4 + 3] = (int8_t)(q >> 24u);
            }
            return;
        }
#pragma unroll
        for (int i = 0; i < 32; i++) {
            uint32_t q = ((uint32_t)qs[i] >> shift) & 0x0fu;
            if (xb->qh[i] & (uint8_t)(1u << grp)) q += 16u;
            wq[i] = (int8_t)q;
        }
        return;
    }
    case (uint32_t)DS4_QWEN4EXP_TY_q6_K: {
        const cuda_block_q6_K *xb = (const cuda_block_q6_K *)row + (g / 8u);
        const uint32_t grp = g % 8u;
        const uint32_t n128 = grp >> 2u;
        const uint32_t quarter = grp & 3u;
        const uint8_t *ql = xb->ql + n128 * 64u;
        const uint8_t *qh = xb->qh + n128 * 32u;
        const int8_t *sc = xb->scales + n128 * 8u + quarter * 2u;
        const float d = dev_f16_to_f32(xb->d);
        *halves = 2;
        wa[0] = d * (float)sc[0];
        wa[1] = d * (float)sc[1];
#pragma unroll
        for (int i = 0; i < 32; i++) {
            const uint32_t hi = ((uint32_t)qh[i] >> (quarter * 2u)) & 3u;
            const uint32_t lo = (quarter & 1u) ? (uint32_t)ql[32 + i]
                                               : (uint32_t)ql[i];
            const uint32_t q = (quarter < 2u) ? ((lo & 0x0fu) | (hi << 4u))
                                              : ((lo >> 4u) | (hi << 4u));
            wq[i] = (int8_t)((int32_t)q - 32);
        }
        return;
    }
    default:
        wa[0] = 0.0f;
#pragma unroll
        for (int i = 0; i < 32; i++) wq[i] = 0;
        return;
    }
}

/* The two floating-point operations of the contract, for one group of one
 * row.  Everything above feeds this and nothing else adds to `acc`. */
__device__ __forceinline__ static void qwen4exp_group_accumulate(
        float *acc, const int8_t *wq, const float *wa, const float *wb,
        int halves, const int8_t *xqg, float xscale, int32_t xsum) {
    if (halves == 1) {
        const int32_t dot = qwen4exp_dp4a<32>(wq, xqg);
        *acc += (wa[0] * xscale) * (float)dot;
        *acc += (wb[0] * xscale) * (float)xsum;
    } else {
        const int32_t d0 = qwen4exp_dp4a<16>(wq, xqg);
        const int32_t d1 = qwen4exp_dp4a<16>(wq + 16, xqg + 16);
        *acc += (wa[0] * xscale) * (float)d0;
        *acc += (wa[1] * xscale) * (float)d1;
    }
}

/* Q8_0 quantisation of one activation row, plus the integer sum of each group
 * that the `wb` term needs.  One block per (row, group); the row is the only
 * thing it reads, so the result does not depend on how many rows the call
 * carries.  The scale and the rounding are ds4_cuda.cu's
 * quantize_q8_0_f32_kernel, unchanged. */
__global__ static void qwen4exp_quantize_rows_kernel(
        int8_t *xq, float *xscale, int32_t *xsum,
        const float *x, uint32_t width, uint32_t groups,
        uint64_t outer_stride, uint64_t inner_stride, uint32_t inner_count) {
    const uint32_t g = blockIdx.x;
    const uint32_t r = blockIdx.y;
    if (g >= groups) return;
    const uint32_t i0 = g * 32u;
    const uint32_t n = width - i0 < 32u ? width - i0 : 32u;
    const uint32_t outer = r / inner_count;
    const uint32_t inner = r - outer * inner_count;
    const float *xr = x + (uint64_t)outer * outer_stride +
                      (uint64_t)inner * inner_stride + i0;

    /* The only launch site is <<<dim3(groups, rows, 1), 32, 0, stream>>>, so a
     * block is exactly one warp and the two reductions below are
     * warp-synchronous.  A shuffle tree pairs lane i with lane i + stride for
     * the same strides, in the same order, with the same operand order, so the
     * max and the sum are the values the shared-memory trees returned; the
     * block barriers they spent on lanes that had already finished are what
     * goes away.  Every lane of the block reaches here (the early return above
     * is on blockIdx.x, uniform across the block), so the full mask is the
     * active set. */
    float a = 0.0f;
    if (threadIdx.x < n) a = fabsf(xr[threadIdx.x]);
    float m = a;
#pragma unroll
    for (uint32_t stride = 16u; stride > 0u; stride >>= 1u) {
        m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, stride));
    }
    m = __shfl_sync(0xffffffffu, m, 0);
    const float d = m / 127.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    const uint64_t at = (uint64_t)r * groups + g;
    if (threadIdx.x == 0u) xscale[at] = d;

    int8_t *dst = xq + at * 32u;
    int v = 0;
    if (threadIdx.x < n) {
        v = (int)lrintf(xr[threadIdx.x] * id);
        v = v > 127 ? 127 : (v < -128 ? -128 : v);
    }
    dst[threadIdx.x] = (int8_t)v;

    int sv = v;
#pragma unroll
    for (uint32_t stride = 16u; stride > 0u; stride >>= 1u) {
        sv += __shfl_down_sync(0xffffffffu, sv, stride);
    }
    if (threadIdx.x == 0u) xsum[at] = sv;
}

/* Block-wide sum over blockDim.x threads using a caller-supplied scratch of
 * blockDim.x floats.  The reduction tree matches the Metal kernels. */
__device__ __forceinline__ static float dev_qwen4exp_block_sum(
        float *scratch, float value) {
    const uint32_t tid = threadIdx.x;
    scratch[tid] = value;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        __syncthreads();
    }
    return scratch[0];
}

/* The checkpoint asks for ten of at most 512 experts.  One warp can read that
 * envelope as sixteen coalesced rows, keep them in registers, and select the
 * small top-k without sorting the 502 entries the model will discard. */
__global__ static void qwen4exp_router_select_topk_kernel(
        int32_t *selected,
        float *weights_out,
        const float *logits,
        uint32_t n_expert,
        uint32_t n_expert_used,
        uint32_t n_tokens) {
    const uint32_t tok = blockIdx.x;
    if (tok >= n_tokens) return;
    const uint32_t lane = threadIdx.x;
    const float *lg = logits + (uint64_t)tok * n_expert;
    int32_t *sel = selected + (uint64_t)tok * n_expert_used;
    float *w = weights_out + (uint64_t)tok * n_expert_used;

    float scores[16];
    uint32_t live = 0u;
#pragma unroll
    for (uint32_t j = 0; j < 16u; j++) {
        const uint32_t e = lane + j * 32u;
        scores[j] = e < n_expert ? lg[e] : -FLT_MAX;
        if (e < n_expert) live |= 1u << j;
    }

    for (uint32_t rank = 0; rank < n_expert_used; rank++) {
        float best_v = -FLT_MAX;
        int32_t best_i = INT32_MAX;
#pragma unroll
        for (uint32_t j = 0; j < 16u; j++) {
            if ((live & (1u << j)) == 0u) continue;
            const int32_t e = (int32_t)(lane + j * 32u);
            const float v = scores[j];
            if (v > best_v || (v == best_v && e < best_i)) {
                best_v = v;
                best_i = e;
            }
        }

#pragma unroll
        for (uint32_t off = 16u; off > 0u; off >>= 1u) {
            const float other_v =
                __shfl_down_sync(0xffffffffu, best_v, off);
            const int32_t other_i =
                __shfl_down_sync(0xffffffffu, best_i, off);
            if (other_v > best_v ||
                (other_v == best_v && other_i < best_i)) {
                best_v = other_v;
                best_i = other_i;
            }
        }
        const int32_t chosen =
            __shfl_sync(0xffffffffu, best_i, 0u);
        if (lane == 0u) sel[rank] = chosen;
        if (((uint32_t)chosen & 31u) == lane) {
            live &= ~(1u << ((uint32_t)chosen >> 5u));
        }
    }

    /* Same serial softmax and the same selected-logit order as the full-sort
     * path below. */
    if (lane == 0u) {
        float m = -FLT_MAX;
        for (uint32_t i = 0; i < n_expert_used; i++) {
            const float v = lg[(uint32_t)sel[i]];
            if (v > m) m = v;
        }
        float sum = 0.0f;
        for (uint32_t i = 0; i < n_expert_used; i++) {
            const float e = expf(lg[(uint32_t)sel[i]] - m);
            w[i] = e;
            sum += e;
        }
        const float inv = 1.0f / sum;
        for (uint32_t i = 0; i < n_expert_used; i++) w[i] *= inv;
    }
}

/* One block per token.  Bitonic sort over the raw logits with ties going to
 * the lower expert index, then a serial softmax over the selected logits. */
__global__ static void qwen4exp_router_select_kernel(
        int32_t *selected,
        float *weights_out,
        const float *logits,
        uint32_t n_expert,
        uint32_t n_expert_used,
        uint32_t n_tokens) {
    __shared__ float sh_v[512];
    __shared__ int32_t sh_i[512];
    const uint32_t tok = blockIdx.x;
    if (tok >= n_tokens) return;
    const uint32_t tid = threadIdx.x;
    const uint32_t width = blockDim.x;

    const float *lg = logits + (uint64_t)tok * n_expert;
    int32_t *sel = selected + (uint64_t)tok * n_expert_used;
    float *w = weights_out + (uint64_t)tok * n_expert_used;

    /* Padding lanes sit below every real logit and past every top-k slot.
     * The Metal twin pads with -INFINITY; nvcc builds with --use_fast_math, so
     * this side pads with -FLT_MAX, following the neighbouring CUDA kernels.
     * The two agree on every selection a real router makes: a logit at or
     * below -FLT_MAX would be needed to tell them apart, and the padding lanes
     * are past slot k_used either way. */
    sh_v[tid] = tid < n_expert ? lg[tid] : -FLT_MAX;
    sh_i[tid] = (int32_t)tid;
    __syncthreads();

    for (uint32_t k = 2u; k <= width; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            const uint32_t other = tid ^ j;
            if (other > tid) {
                const int32_t a = sh_i[tid];
                const int32_t b = sh_i[other];
                const float sa = sh_v[(uint32_t)a];
                const float sb = sh_v[(uint32_t)b];
                const bool b_better = sb > sa || (sb == sa && b < a);
                const bool a_better = sa > sb || (sa == sb && a < b);
                const bool swap = ((tid & k) == 0u) ? b_better : a_better;
                if (swap) {
                    sh_i[tid] = b;
                    sh_i[other] = a;
                }
            }
            __syncthreads();
        }
    }

    const uint32_t k_used = min(n_expert_used, n_expert);
    if (tid < k_used) sel[tid] = sh_i[tid];
    __syncthreads();

    if (tid == 0u) {
        float m = -FLT_MAX;
        for (uint32_t i = 0; i < k_used; i++) {
            const float v = sh_v[(uint32_t)sel[i]];
            if (v > m) m = v;
        }
        float sum = 0.0f;
        for (uint32_t i = 0; i < k_used; i++) {
            const float e = expf(sh_v[(uint32_t)sel[i]] - m);
            w[i] = e;
            sum += e;
        }
        const float inv = 1.0f / sum;
        for (uint32_t i = 0; i < k_used; i++) w[i] *= inv;
    }
}



/* ---------------------------------------------------------------------------
 * Weight reuse across the rows of a prefill.
 *
 * Every kernel above gives one threadgroup to one (output row, token) pair, so
 * a weight row is fetched and decoded once per TOKEN.  At a prefill width of
 * 512 that reads the whole active weight set 512 times, which is what held the
 * prompt phase to a fraction of the reference engine's throughput.
 *
 * The kernels below fetch and decode a weight element ONCE and use it for a
 * tile of rows.  Nothing else moves: each row keeps the same k walk
 * (k = tid, tid + ntg, ...), the same float accumulator, the same
 * `dev_qwen4exp_block_sum` halving tree, and therefore the same value it had
 * when it was computed alone.  The tile is a memory decision, not a numeric
 * one, so the one-row result and the tiled result are the same bits and the
 * router's discrete top-k downstream cannot see a difference.
 *
 * The routed experts need one more step: rows in a tile must share an expert
 * or there is no weight to share.  The pair list below sorts the
 * (token, slot) pairs by the expert they selected, and a block then owns one
 * expert and walks its own pairs.  Sorting changes which rows sit beside each
 * other, never what a row computes: `mid[token][slot][row]` depends on that
 * pair alone.
 * ------------------------------------------------------------------------ */

/* Pairs are packed as token * n_expert_used + slot, which is exactly the index
 * `selected` and `weights` are addressed by. */
__global__ static void qwen4exp_moe_group_count_kernel(
        int32_t *counts,
        const int32_t *selected,
        uint32_t n_total_expert,
        uint32_t n_pairs) {
    const uint32_t p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_pairs) return;
    const int32_t e = selected[p];
    if (e < 0 || (uint32_t)e >= n_total_expert) return;
    atomicAdd(&counts[e], 1);
}

/* One full warp scans 32 experts at a time.  Every lane participates in every
 * shuffle and ballot, including the tail, where out-of-range counts are zero.
 * `cursor` starts at the offset and the scatter bumps it. */
__global__ static void qwen4exp_moe_group_scan_kernel(
        int32_t *offsets,
        int32_t *cursor,
        int32_t *active,
        const int32_t *counts,
        uint32_t n_expert) {
    if (blockIdx.x != 0u) return;
    const uint32_t lane = threadIdx.x & 31u;
    int32_t run = 0;
    int32_t live = 0;
    for (uint64_t base = 0; base < (uint64_t)n_expert; base += 32u) {
        const uint64_t e = base + lane;
        const int32_t count = e < (uint64_t)n_expert ? counts[e] : 0;
        int32_t inclusive = count;
#pragma unroll
        for (uint32_t delta = 1u; delta < 32u; delta <<= 1u) {
            const int32_t prior = __shfl_up_sync(0xffffffffu, inclusive, delta);
            if (lane >= delta) inclusive += prior;
        }

        if (e < (uint64_t)n_expert) {
            const int32_t offset = run + inclusive - count;
            offsets[e] = offset;
            cursor[e] = offset;
        }

        /* The experts that have work, compacted in ascending id order.  The
         * 64-bit mask expression is defined for lane 31 as well as lane 0. */
        const uint32_t live_mask = __ballot_sync(
                0xffffffffu, e < (uint64_t)n_expert && count > 0);
        if (e < (uint64_t)n_expert && count > 0) {
            const uint32_t lower_lanes =
                (uint32_t)((1ull << lane) - 1ull);
            const int32_t rank = (int32_t)__popc(live_mask & lower_lanes);
            active[1 + live + rank] = (int32_t)e;
        }

        run += __shfl_sync(0xffffffffu, inclusive, 31);
        live += (int32_t)__popc(live_mask);
    }
    if (lane == 0u) active[0] = live;
}

/* The shipped router admits at most 512 experts.  Scan that fixed envelope in
 * parallel: both prefixes use integers, so this produces exactly the reference
 * offsets, cursor values, live count, and ascending active-expert list.  Slots
 * above n_expert carry zero and make non-power-of-two expert counts safe. */
enum { QWEN4EXP_MOE_SCAN_THREADS = 512 };

__global__ static void qwen4exp_moe_group_scan_parallel_kernel(
        int32_t *offsets,
        int32_t *cursor,
        int32_t *active,
        const int32_t *counts,
        uint32_t n_expert) {
    __shared__ int32_t warp_count_prefix[QWEN4EXP_MOE_SCAN_THREADS / 32];
    __shared__ int32_t warp_live_prefix[QWEN4EXP_MOE_SCAN_THREADS / 32];
    const uint32_t e = threadIdx.x;
    const uint32_t lane = e & 31u;
    const uint32_t warp = e >> 5u;
    const int32_t count = e < n_expert ? counts[e] : 0;
    int32_t count_prefix = count;
    int32_t live_prefix = count > 0 ? 1 : 0;

#pragma unroll
    for (uint32_t delta = 1u; delta < 32u; delta <<= 1u) {
        const int32_t prior_count =
            __shfl_up_sync(0xffffffffu, count_prefix, delta);
        const int32_t prior_live =
            __shfl_up_sync(0xffffffffu, live_prefix, delta);
        if (lane >= delta) {
            count_prefix += prior_count;
            live_prefix += prior_live;
        }
    }
    if (lane == 31u) {
        warp_count_prefix[warp] = count_prefix;
        warp_live_prefix[warp] = live_prefix;
    }
    __syncthreads();

    /* Warp zero scans the sixteen warp totals.  Lanes 16-31 carry zero but
     * participate in every shuffle, keeping the full-warp mask valid. */
    if (warp == 0u) {
        const uint32_t n_warps = QWEN4EXP_MOE_SCAN_THREADS / 32;
        int32_t warp_count = lane < n_warps ? warp_count_prefix[lane] : 0;
        int32_t warp_live = lane < n_warps ? warp_live_prefix[lane] : 0;
#pragma unroll
        for (uint32_t delta = 1u; delta < 32u; delta <<= 1u) {
            const int32_t prior_count =
                __shfl_up_sync(0xffffffffu, warp_count, delta);
            const int32_t prior_live =
                __shfl_up_sync(0xffffffffu, warp_live, delta);
            if (lane >= delta) {
                warp_count += prior_count;
                warp_live += prior_live;
            }
        }
        if (lane < n_warps) {
            warp_count_prefix[lane] = warp_count;
            warp_live_prefix[lane] = warp_live;
        }
    }
    __syncthreads();

    if (warp > 0u) {
        count_prefix += warp_count_prefix[warp - 1u];
        live_prefix += warp_live_prefix[warp - 1u];
    }

    if (e < n_expert) {
        const int32_t offset = count_prefix - count;
        offsets[e] = offset;
        cursor[e] = offset;
        if (count > 0) active[live_prefix] = (int32_t)e;
    }
    if (e == 0u) {
        active[0] = warp_live_prefix[QWEN4EXP_MOE_SCAN_THREADS / 32 - 1u];
    }
}

/* At decode and verify widths there are at most seventy pairs.  A single
 * 512-thread block can build the complete expert metadata without a memset,
 * count launch, scan launch, scatter launch, or inter-block atomics.  Each
 * expert thread scans the short pair list, the block performs the same integer
 * prefix scans as the wide path, and each expert writes its pairs in ascending
 * pair order. */
__global__ static void qwen4exp_moe_group_small_kernel(
        int32_t *counts,
        int32_t *offsets,
        int32_t *cursor,
        int32_t *active,
        int32_t *pairs,
        float *mid,
        const int32_t *selected,
        uint32_t n_expert,
        uint32_t n_pairs,
        uint32_t n_expert_used,
        uint32_t mid_dim,
        uint32_t mid_token_stride) {
    __shared__ int32_t warp_count_prefix[QWEN4EXP_MOE_SCAN_THREADS / 32];
    __shared__ int32_t warp_live_prefix[QWEN4EXP_MOE_SCAN_THREADS / 32];
    const uint32_t e = threadIdx.x;
    const uint32_t lane = e & 31u;
    const uint32_t warp = e >> 5u;

    int32_t count = 0;
    if (e < n_expert) {
        for (uint32_t p = 0; p < n_pairs; p++) {
            count += selected[p] == (int32_t)e;
        }
        counts[e] = count;
    }
    int32_t count_prefix = count;
    int32_t live_prefix = count > 0 ? 1 : 0;
#pragma unroll
    for (uint32_t delta = 1u; delta < 32u; delta <<= 1u) {
        const int32_t prior_count =
            __shfl_up_sync(0xffffffffu, count_prefix, delta);
        const int32_t prior_live =
            __shfl_up_sync(0xffffffffu, live_prefix, delta);
        if (lane >= delta) {
            count_prefix += prior_count;
            live_prefix += prior_live;
        }
    }
    if (lane == 31u) {
        warp_count_prefix[warp] = count_prefix;
        warp_live_prefix[warp] = live_prefix;
    }
    __syncthreads();

    if (warp == 0u) {
        const uint32_t n_warps = QWEN4EXP_MOE_SCAN_THREADS / 32;
        int32_t warp_count = lane < n_warps ? warp_count_prefix[lane] : 0;
        int32_t warp_live = lane < n_warps ? warp_live_prefix[lane] : 0;
#pragma unroll
        for (uint32_t delta = 1u; delta < 32u; delta <<= 1u) {
            const int32_t prior_count =
                __shfl_up_sync(0xffffffffu, warp_count, delta);
            const int32_t prior_live =
                __shfl_up_sync(0xffffffffu, warp_live, delta);
            if (lane >= delta) {
                warp_count += prior_count;
                warp_live += prior_live;
            }
        }
        if (lane < n_warps) {
            warp_count_prefix[lane] = warp_count;
            warp_live_prefix[lane] = warp_live;
        }
    }
    __syncthreads();

    if (warp > 0u) {
        count_prefix += warp_count_prefix[warp - 1u];
        live_prefix += warp_live_prefix[warp - 1u];
    }
    if (e < n_expert) {
        const int32_t offset = count_prefix - count;
        offsets[e] = offset;
        cursor[e] = offset + count;
        if (count > 0) active[live_prefix] = (int32_t)e;
        int32_t at = offset;
        for (uint32_t p = 0; p < n_pairs; p++) {
            if (selected[p] == (int32_t)e) pairs[at++] = (int32_t)p;
        }
    }
    if (e == 0u) {
        active[0] = warp_live_prefix[QWEN4EXP_MOE_SCAN_THREADS / 32 - 1u];
    }
    /* The normal router never emits an invalid id.  Preserve the public
     * tensor helper's defensive zero semantics without paying a second launch
     * in the normal case; an invalid pair's one thread writes its short
     * intermediate row here. */
    if (e < n_pairs) {
        const int32_t expert = selected[e];
        if (expert < 0 || (uint32_t)expert >= n_expert) {
            const uint32_t token = e / n_expert_used;
            const uint32_t slot = e - token * n_expert_used;
            float *dst = mid + (uint64_t)token * mid_token_stride +
                         (uint64_t)slot * mid_dim;
            for (uint32_t row = 0; row < mid_dim; row++) dst[row] = 0.0f;
        }
    }
}

__global__ static void qwen4exp_moe_group_scatter_kernel(
        int32_t *pairs,
        int32_t *cursor,
        const int32_t *selected,
        uint32_t n_total_expert,
        uint32_t n_pairs) {
    const uint32_t p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_pairs) return;
    const int32_t e = selected[p];
    if (e < 0 || (uint32_t)e >= n_total_expert) return;
    const int32_t at = atomicAdd(&cursor[e], 1);
    pairs[at] = (int32_t)p;
}

/* A pair whose expert id is out of range contributes nothing and its mid row
 * reads zero, which is what the per-token kernel wrote for it.  The grouped
 * kernel never visits such a pair, so the zero is written here.  The router
 * cannot produce one; this keeps the two paths equal anyway. */
__global__ static void qwen4exp_moe_zero_invalid_kernel(
        float *mid,
        const int32_t *selected,
        uint32_t n_total_expert,
        uint32_t n_expert_used,
        uint32_t mid_dim,
        uint32_t mid_token_stride,
        uint32_t n_pairs) {
    const uint32_t p = blockIdx.x;
    if (p >= n_pairs) return;
    const int32_t e = selected[p];
    if (e >= 0 && (uint32_t)e < n_total_expert) return;
    const uint32_t token = p / n_expert_used;
    const uint32_t slot = p - token * n_expert_used;
    float *dst = mid + (uint64_t)token * mid_token_stride +
                 (uint64_t)slot * mid_dim;
    for (uint32_t r = threadIdx.x; r < mid_dim; r += blockDim.x) {
        dst[r] = 0.0f;
    }
}


/* ---------------------------------------------------------------------------
 * The expert projections under the contract.
 *
 * One shape for all four: a warp owns one OUTPUT ROW, lane L owns the weight
 * groups L, L+32, L+64 ... in ascending order, and the warp carries R
 * activation rows at once so the group is decoded once for all of them.  The
 * per-row reduction is warp_sum_f32, the same butterfly the dense Q8_0
 * projections use.  R is a template parameter and R = 1 is the same code, so
 * the decode call and the prefill call are one kernel.
 * ------------------------------------------------------------------------ */

/* ===========================================================================
 * The expert gate/up projection on the int8 tensor core.
 *
 * Same contract, different instruction and a much larger tile.  A group of 32
 * is exactly the k of mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32, so one
 * MMA produces the int32 dot of one group for a 16x8 block of output elements
 * at once, and the int32 dot is exact however it is summed.  The scaling stays
 * what it was: per group, ascending, one f32 accumulator per output element,
 * two fmas (one when the type has no offset).  R = 1 is this kernel with the
 * token tile zero-padded, so row invariance is a property of the code rather
 * than of a routing rule.
 *
 * WHY A TILE AT ALL.  The dp4a kernel gave a warp one output row and eight
 * tokens, so its instruction stream was per-group dequant, address arithmetic
 * and a dependent load chain, with the multiply-accumulate a small part of it.
 * The measured cost was 73x what the arithmetic alone would take at peak
 * issue.  A tile deletes the overhead rather than the arithmetic: one MMA
 * replaces 1024 dp4a, the dequant is amortised over 32 tokens instead of 8,
 * and both operands are staged through shared memory so the global reads are
 * coalesced instead of eight scattered 16-byte segments per warp.
 *
 * BM 64 output rows, BN 32 tokens, four groups staged per barrier.  BN is 32
 * because an expert holds about ten tokens at a prefill width of 512 and about
 * twenty at 1024, so one pass covers a whole expert and the staged weight is
 * used by every token that chose it.  Shared: two 64x132 int8 weight tiles,
 * one 32x132 activation tile, and the per-group scales, about 26 KiB.
 *
 * Q6_K is NOT routed here.  Its scale changes every sixteen elements, so its
 * group needs two dots of sixteen and m16n8k32 cannot split k.  It keeps the
 * dp4a kernel, which is internally width invariant as before; the two types
 * simply do not share an instruction.
 * ======================================================================== */

#define QW_MMA_BM 64
#define QW_MMA_BN 32
#define QW_MMA_G  4
#define QW_MMA_KC (QW_MMA_G * 32)
#define QW_MMA_LD (QW_MMA_KC + 4)
#define QW_MMA_WARPS (QW_MMA_BM / 16)
#define QW_MMA_THREADS (QW_MMA_WARPS * 32)
#define QW_MMA_NT (QW_MMA_BN / 8)

__device__ __forceinline__ static uint32_t qw_pack4(const int8_t *p) {
    return ((uint32_t)(uint8_t)p[0]) | ((uint32_t)(uint8_t)p[1] << 8) |
           ((uint32_t)(uint8_t)p[2] << 16) | ((uint32_t)(uint8_t)p[3] << 24);
}

/* The same four bytes, read as the one word that holds them.
 *
 * Every tile below is declared sixteen-byte aligned and QW_MMA_LD is a
 * multiple of four, so every (row, k) offset a fragment names is a whole
 * number of words from the base.  The four bytes reach the register in the
 * same order qw_pack4 put them in, which is the order mma.sync reads them, so
 * the operand is the same value -- one shared load instead of four. */
__device__ __forceinline__ static uint32_t qw_tile_word(const int8_t *p) {
    return *(const uint32_t *)(const void *)p;
}

/* Write one decoded 32-quant group into a tile row as eight words.
 *
 * The byte-at-a-time store this replaces put element e at byte e of the row,
 * which is exactly what a little-endian word of elements 4i..4i+3 holds, so
 * the tile bytes are unchanged and only the instruction count moves. */
__device__ __forceinline__ static void qw_tile_store_group(int8_t *dst,
                                                           const int8_t *wq) {
    uint32_t *w = (uint32_t *)(void *)dst;
#pragma unroll
    for (int i = 0; i < 8; i++) w[i] = qw_pack4(wq + i * 4);
}

__device__ __forceinline__ static void qw_tile_store_zero(int8_t *dst) {
    uint32_t *w = (uint32_t *)(void *)dst;
#pragma unroll
    for (int i = 0; i < 8; i++) w[i] = 0u;
}

/* Copy one 32-quant activation group from the quantised scratch into a tile
 * row.  The scratch is a device allocation and every offset the caller cuts it
 * at -- the group index block, the pair list and the quantised rows -- is a
 * whole number of words, so a group start is always word aligned. */
__device__ __forceinline__ static void qw_tile_copy_group(int8_t *dst,
                                                          const int8_t *src) {
    uint32_t *w = (uint32_t *)(void *)dst;
    const uint32_t *s = (const uint32_t *)(const void *)src;
#pragma unroll
    for (int i = 0; i < 8; i++) w[i] = s[i];
}

__device__ __forceinline__ static void qw_mma_m16n8k32(
        int32_t *d, const uint32_t *a, const uint32_t *b) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

/* Grid (mid_dim / BM, the experts this call chose). */
template <int GateType = -1, int UpType = -1>
__global__ __launch_bounds__(QW_MMA_THREADS) static void
qwen4exp_moe_gateup_mma_kernel(
        float *mid,
        const char *gate,
        const char *up,
        const int8_t *xq,
        const float *xs,
        const int32_t *xsum,
        const int32_t *pairs,
        const int32_t *counts,
        const int32_t *offsets,
        const int32_t *active,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint64_t up_expert_bytes,
        uint64_t up_row_bytes,
        uint32_t gate_type,
        uint32_t up_type,
        uint32_t groups,
        uint32_t mid_dim,
        uint32_t mid_token_stride,
        uint32_t n_expert_used) {
    __shared__ __align__(16) int8_t sAg[QW_MMA_BM * QW_MMA_LD];
    __shared__ __align__(16) int8_t sAu[QW_MMA_BM * QW_MMA_LD];
    __shared__ __align__(16) int8_t sB [QW_MMA_BN * QW_MMA_LD];
    __shared__ float  sWAg[QW_MMA_BM * QW_MMA_G], sWBg[QW_MMA_BM * QW_MMA_G];
    __shared__ float  sWAu[QW_MMA_BM * QW_MMA_G], sWBu[QW_MMA_BM * QW_MMA_G];
    __shared__ float  sXS [QW_MMA_BN * QW_MMA_G];
    __shared__ float  sXSUM[QW_MMA_BN * QW_MMA_G];
    __shared__ uint32_t sTok[QW_MMA_BN];

    const uint32_t tid  = threadIdx.x;
    const uint32_t warp = tid >> 5;
    const uint32_t lane = tid & 31;
    const uint32_t row0 = blockIdx.x * QW_MMA_BM;
    if (row0 >= mid_dim) return;
    if (active) {
        if ((int32_t)blockIdx.y >= active[0]) return;
    }
    const uint32_t expert = active ? (uint32_t)active[1 + blockIdx.y]
                                   : blockIdx.y;
    const int32_t cnt = counts[expert];
    if (cnt <= 0) return;
    const int32_t base = offsets[expert];

    const char *gate_e = gate + (uint64_t)expert * gate_expert_bytes;
    const char *up_e   = up   + (uint64_t)expert * up_expert_bytes;

    for (int32_t nbase = 0; nbase < cnt; nbase += QW_MMA_BN) {
        const int32_t take = (cnt - nbase) < QW_MMA_BN ? (cnt - nbase)
                                                       : QW_MMA_BN;
        for (uint32_t i = tid; i < QW_MMA_BN; i += QW_MMA_THREADS) {
            sTok[i] = (int32_t)i < take
                ? (uint32_t)pairs[base + nbase + i] : 0xffffffffu;
        }
        __syncthreads();

        float accG[QW_MMA_NT * 4], accU[QW_MMA_NT * 4];
#pragma unroll
        for (int i = 0; i < QW_MMA_NT * 4; i++) { accG[i] = 0.0f; accU[i] = 0.0f; }

        for (uint32_t kc = 0; kc < groups; kc += QW_MMA_G) {
            __syncthreads();
            /* Weight tile: one thread decodes one (row, group) of 32. */
            for (uint32_t idx = tid; idx < QW_MMA_BM * QW_MMA_G;
                 idx += QW_MMA_THREADS) {
                const uint32_t r = idx / QW_MMA_G;
                const uint32_t gg = idx - r * QW_MMA_G;
                const uint32_t g = kc + gg;
                int8_t wq[32];
                float wa[2], wb[2];
                int halves = 1;
                const uint32_t mrow = row0 + r;
                if (mrow < mid_dim && g < groups) {
                    dev_qwen4exp_group_decode(
                            GateType < 0 ? gate_type : (uint32_t)GateType,
                            gate_e + (uint64_t)mrow * gate_row_bytes, g,
                            wq, wa, wb, &halves);
                    qw_tile_store_group(&sAg[r * QW_MMA_LD + gg * 32], wq);
                    sWAg[r * QW_MMA_G + gg] = wa[0];
                    sWBg[r * QW_MMA_G + gg] = wb[0];
                    dev_qwen4exp_group_decode(
                            UpType < 0 ? up_type : (uint32_t)UpType,
                            up_e + (uint64_t)mrow * up_row_bytes, g,
                            wq, wa, wb, &halves);
                    qw_tile_store_group(&sAu[r * QW_MMA_LD + gg * 32], wq);
                    sWAu[r * QW_MMA_G + gg] = wa[0];
                    sWBu[r * QW_MMA_G + gg] = wb[0];
                } else {
                    qw_tile_store_zero(&sAg[r * QW_MMA_LD + gg * 32]);
                    qw_tile_store_zero(&sAu[r * QW_MMA_LD + gg * 32]);
                    sWAg[r * QW_MMA_G + gg] = 0.0f; sWBg[r * QW_MMA_G + gg] = 0.0f;
                    sWAu[r * QW_MMA_G + gg] = 0.0f; sWBu[r * QW_MMA_G + gg] = 0.0f;
                }
            }
            /* Activation tile: a padded token row is zero, and zero contributes
             * nothing to an integer dot, so the pad is exact. */
            for (uint32_t idx = tid; idx < QW_MMA_BN * QW_MMA_G;
                 idx += QW_MMA_THREADS) {
                const uint32_t tk = idx / QW_MMA_G;
                const uint32_t gg = idx - tk * QW_MMA_G;
                const uint32_t g = kc + gg;
                const uint32_t p = sTok[tk];
                if (p != 0xffffffffu && g < groups) {
                    const uint32_t token = p / n_expert_used;
                    const uint64_t at = (uint64_t)token * groups + g;
                    qw_tile_copy_group(&sB[tk * QW_MMA_LD + gg * 32],
                                       xq + at * 32u);
                    sXS  [tk * QW_MMA_G + gg] = xs[at];
                    sXSUM[tk * QW_MMA_G + gg] = (float)xsum[at];
                } else {
                    qw_tile_store_zero(&sB[tk * QW_MMA_LD + gg * 32]);
                    sXS[tk * QW_MMA_G + gg] = 0.0f;
                    sXSUM[tk * QW_MMA_G + gg] = 0.0f;
                }
            }
            __syncthreads();

#pragma unroll
            for (int gg = 0; gg < QW_MMA_G; gg++) {
                if (kc + (uint32_t)gg >= groups) break;
                const uint32_t ar = warp * 16u + (lane >> 2);
                const uint32_t ak = (lane & 3u) * 4u;
                uint32_t ag[4], au[4], bf[2];
#pragma unroll
                for (int r = 0; r < 4; r++) {
                    const uint32_t rr = ar + ((r & 1) ? 8u : 0u);
                    const uint32_t kk = gg * 32u + ak + ((r & 2) ? 16u : 0u);
                    ag[r] = qw_tile_word(&sAg[rr * QW_MMA_LD + kk]);
                    au[r] = qw_tile_word(&sAu[rr * QW_MMA_LD + kk]);
                }
                const uint32_t m0 = warp * 16u + (lane >> 2);
                const uint32_t m1 = m0 + 8u;
#pragma unroll
                for (int nt = 0; nt < QW_MMA_NT; nt++) {
                    /* An MMA column covers eight pairs.  Expert tails often
                     * occupy only one or two columns; whole empty columns
                     * have no consumer.  take is block-uniform, so all lanes
                     * still participate in every live MMA instruction. */
                    if (nt * 8 >= take) break;
                    const uint32_t bn = nt * 8u + (lane >> 2);
#pragma unroll
                    for (int r = 0; r < 2; r++) {
                        bf[r] = qw_tile_word(&sB[bn * QW_MMA_LD + gg * 32u +
                                                 (lane & 3u) * 4u +
                                                 (r ? 16u : 0u)]);
                    }
                    int32_t dg[4] = {0, 0, 0, 0}, du[4] = {0, 0, 0, 0};
                    qw_mma_m16n8k32(dg, ag, bf);
                    qw_mma_m16n8k32(du, au, bf);
                    const uint32_t n0 = nt * 8u + (lane & 3u) * 2u;
#pragma unroll
                    for (int r = 0; r < 4; r++) {
                        const uint32_t mr = (r & 2) ? m1 : m0;
                        const uint32_t nn = n0 + (r & 1);
                        const float sc = sXS[nn * QW_MMA_G + gg];
                        const float sm = sXSUM[nn * QW_MMA_G + gg];
                        const int at = nt * 4 + r;
                        accG[at] = fmaf(sWAg[mr * QW_MMA_G + gg] * sc,
                                        (float)dg[r], accG[at]);
                        accG[at] = fmaf(sWBg[mr * QW_MMA_G + gg] * sc,
                                        sm, accG[at]);
                        accU[at] = fmaf(sWAu[mr * QW_MMA_G + gg] * sc,
                                        (float)du[r], accU[at]);
                        accU[at] = fmaf(sWBu[mr * QW_MMA_G + gg] * sc,
                                        sm, accU[at]);
                    }
                }
            }
        }

        const uint32_t m0 = warp * 16u + (lane >> 2);
#pragma unroll
        for (int nt = 0; nt < QW_MMA_NT; nt++) {
#pragma unroll
            for (int r = 0; r < 4; r++) {
                const uint32_t mr = m0 + ((r & 2) ? 8u : 0u);
                const uint32_t nn = nt * 8u + (lane & 3u) * 2u + (r & 1);
                if ((int32_t)nn >= take) continue;
                const uint32_t mrow = row0 + mr;
                if (mrow >= mid_dim) continue;
                const uint32_t p = sTok[nn];
                const uint32_t token = p / n_expert_used;
                const uint32_t slot = p - token * n_expert_used;
                const float g = accG[nt * 4 + r];
                mid[(uint64_t)token * mid_token_stride +
                    (uint64_t)slot * mid_dim + mrow] =
                    (g / (1.0f + expf(-g))) * accU[nt * 4 + r] * weights[p];
            }
        }
        __syncthreads();
    }
}

/* The down projection on the same tile.
 *
 * This is the operation the tile is worth the most on.  Its weight is the
 * largest thing the experts read -- ten experts times n_embd rows times
 * n_ff_exp elements, 590 MB a token over the tower against 177 MB for the gate
 * and up pair -- and the per-token kernel read it once per TOKEN, a reuse of
 * one, where gate and up already had eight.  Giving a block one expert and all
 * of that expert's pairs takes the reuse to about twenty at a prefill chunk of
 * 1024.
 *
 * The cost is that a token's ten slots no longer share an accumulator: the
 * pairs of one token sit in different experts' blocks.  Each pair's row dot is
 * written as a PARTIAL and the combine pass below sums a token's slots in
 * ascending slot order.  That sum is ten numbers in a fixed order and mentions
 * the batch nowhere, so it is the same at one row and at a thousand.
 *
 * The activation is the routed intermediate, quantised per (token, slot) pair,
 * so the pair index addresses it directly.
 */
template <int DownType = -1>
__global__ __launch_bounds__(QW_MMA_THREADS) static void
qwen4exp_moe_down_mma_kernel(
        float *partial,
        const char *down,
        const int8_t *mq,
        const float *ms,
        const int32_t *msum,
        const int32_t *pairs,
        const int32_t *counts,
        const int32_t *offsets,
        const int32_t *active,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t down_type,
        uint32_t groups,
        uint32_t out_dim) {
    __shared__ __align__(16) int8_t sA[QW_MMA_BM * QW_MMA_LD];
    __shared__ __align__(16) int8_t sB[QW_MMA_BN * QW_MMA_LD];
    __shared__ float  sWA[QW_MMA_BM * QW_MMA_G], sWB[QW_MMA_BM * QW_MMA_G];
    __shared__ float  sXS[QW_MMA_BN * QW_MMA_G], sXSUM[QW_MMA_BN * QW_MMA_G];
    __shared__ uint32_t sPair[QW_MMA_BN];

    const uint32_t tid  = threadIdx.x;
    const uint32_t warp = tid >> 5;
    const uint32_t lane = tid & 31;
    const uint32_t row0 = blockIdx.x * QW_MMA_BM;
    if (row0 >= out_dim) return;
    if (active && (int32_t)blockIdx.y >= active[0]) return;
    const uint32_t expert = active ? (uint32_t)active[1 + blockIdx.y]
                                   : blockIdx.y;
    const int32_t cnt = counts[expert];
    if (cnt <= 0) return;
    const int32_t base = offsets[expert];
    const char *down_e = down + (uint64_t)expert * down_expert_bytes;

    for (int32_t nbase = 0; nbase < cnt; nbase += QW_MMA_BN) {
        const int32_t take = (cnt - nbase) < QW_MMA_BN ? (cnt - nbase)
                                                       : QW_MMA_BN;
        for (uint32_t i = tid; i < QW_MMA_BN; i += QW_MMA_THREADS) {
            sPair[i] = (int32_t)i < take
                ? (uint32_t)pairs[base + nbase + i] : 0xffffffffu;
        }
        __syncthreads();

        float acc[QW_MMA_NT * 4];
#pragma unroll
        for (int i = 0; i < QW_MMA_NT * 4; i++) acc[i] = 0.0f;

        for (uint32_t kc = 0; kc < groups; kc += QW_MMA_G) {
            __syncthreads();
            for (uint32_t idx = tid; idx < QW_MMA_BM * QW_MMA_G;
                 idx += QW_MMA_THREADS) {
                const uint32_t r = idx / QW_MMA_G;
                const uint32_t gg = idx - r * QW_MMA_G;
                const uint32_t g = kc + gg;
                const uint32_t orow = row0 + r;
                int8_t wq[32];
                float wa[2], wb[2];
                int halves = 1;
                if (orow < out_dim && g < groups) {
                    dev_qwen4exp_group_decode(
                            DownType < 0 ? down_type : (uint32_t)DownType,
                            down_e + (uint64_t)orow * down_row_bytes, g,
                            wq, wa, wb, &halves);
                    qw_tile_store_group(&sA[r * QW_MMA_LD + gg * 32], wq);
                    sWA[r * QW_MMA_G + gg] = wa[0];
                    sWB[r * QW_MMA_G + gg] = wb[0];
                } else {
                    qw_tile_store_zero(&sA[r * QW_MMA_LD + gg * 32]);
                    sWA[r * QW_MMA_G + gg] = 0.0f;
                    sWB[r * QW_MMA_G + gg] = 0.0f;
                }
            }
            for (uint32_t idx = tid; idx < QW_MMA_BN * QW_MMA_G;
                 idx += QW_MMA_THREADS) {
                const uint32_t tk = idx / QW_MMA_G;
                const uint32_t gg = idx - tk * QW_MMA_G;
                const uint32_t g = kc + gg;
                const uint32_t p = sPair[tk];
                if (p != 0xffffffffu && g < groups) {
                    const uint64_t at = (uint64_t)p * groups + g;
                    qw_tile_copy_group(&sB[tk * QW_MMA_LD + gg * 32],
                                       mq + at * 32u);
                    sXS  [tk * QW_MMA_G + gg] = ms[at];
                    sXSUM[tk * QW_MMA_G + gg] = (float)msum[at];
                } else {
                    qw_tile_store_zero(&sB[tk * QW_MMA_LD + gg * 32]);
                    sXS[tk * QW_MMA_G + gg] = 0.0f;
                    sXSUM[tk * QW_MMA_G + gg] = 0.0f;
                }
            }
            __syncthreads();

#pragma unroll
            for (int gg = 0; gg < QW_MMA_G; gg++) {
                if (kc + (uint32_t)gg >= groups) break;
                const uint32_t ar = warp * 16u + (lane >> 2);
                const uint32_t ak = (lane & 3u) * 4u;
                uint32_t af[4], bf[2];
#pragma unroll
                for (int r = 0; r < 4; r++) {
                    const uint32_t rr = ar + ((r & 1) ? 8u : 0u);
                    const uint32_t kk = gg * 32u + ak + ((r & 2) ? 16u : 0u);
                    af[r] = qw_tile_word(&sA[rr * QW_MMA_LD + kk]);
                }
                const uint32_t m0 = warp * 16u + (lane >> 2);
#pragma unroll
                for (int nt = 0; nt < QW_MMA_NT; nt++) {
                    /* An MMA column covers eight pairs.  Expert tails often
                     * occupy only one or two columns; whole empty columns
                     * have no consumer.  take is block-uniform, so all lanes
                     * still participate in every live MMA instruction. */
                    if (nt * 8 >= take) break;
                    const uint32_t bn = nt * 8u + (lane >> 2);
#pragma unroll
                    for (int r = 0; r < 2; r++) {
                        bf[r] = qw_tile_word(&sB[bn * QW_MMA_LD + gg * 32u +
                                                 (lane & 3u) * 4u +
                                                 (r ? 16u : 0u)]);
                    }
                    int32_t d[4] = {0, 0, 0, 0};
                    qw_mma_m16n8k32(d, af, bf);
                    const uint32_t n0 = nt * 8u + (lane & 3u) * 2u;
#pragma unroll
                    for (int r = 0; r < 4; r++) {
                        const uint32_t mr = m0 + ((r & 2) ? 8u : 0u);
                        const uint32_t nn = n0 + (r & 1);
                        const float sc = sXS[nn * QW_MMA_G + gg];
                        const int at = nt * 4 + r;
                        acc[at] = fmaf(sWA[mr * QW_MMA_G + gg] * sc,
                                       (float)d[r], acc[at]);
                        acc[at] = fmaf(sWB[mr * QW_MMA_G + gg] * sc,
                                       sXSUM[nn * QW_MMA_G + gg], acc[at]);
                    }
                }
            }
        }

        const uint32_t m0 = warp * 16u + (lane >> 2);
#pragma unroll
        for (int nt = 0; nt < QW_MMA_NT; nt++) {
#pragma unroll
            for (int r = 0; r < 4; r++) {
                const uint32_t nn = nt * 8u + (lane & 3u) * 2u + (r & 1);
                if ((int32_t)nn >= take) continue;
                const uint32_t mr = m0 + ((r & 2) ? 8u : 0u);
                const uint32_t orow = row0 + mr;
                if (orow >= out_dim) continue;
                partial[(uint64_t)sPair[nn] * out_dim + orow] = acc[nt * 4 + r];
            }
        }
        __syncthreads();
    }
}

/* out[token][row] is the token's slots in ASCENDING order.  A slot whose
 * expert id is out of range contributes nothing, which is what the per-token
 * kernel did by skipping it, so its partial is never read. */
__global__ static void qwen4exp_moe_down_combine_kernel(
        float *out,
        const float *partial,
        const int32_t *selected,
        uint32_t out_dim,
        uint32_t n_tokens,
        uint32_t n_expert_used,
        uint32_t n_total_expert) {
    const uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (uint64_t)n_tokens * out_dim) return;
    const uint32_t token = (uint32_t)(idx / out_dim);
    const uint32_t row = (uint32_t)(idx - (uint64_t)token * out_dim);
    float acc = 0.0f;
    for (uint32_t slot = 0; slot < n_expert_used; slot++) {
        const uint64_t pair = (uint64_t)token * n_expert_used + slot;
        const int32_t e = selected[pair];
        if (e < 0 || (uint32_t)e >= n_total_expert) continue;
        acc += partial[pair * out_dim + row];
    }
    out[idx] = acc;
}


/* Grid (ceil(mid_dim / 8), n_expert).  The block owns one expert; the pair
 * list gives it the (token, slot) pairs that chose it, so a decoded group
 * serves R of them. */
/* Common slab formats get compile-time decoders below.  Keeping the
 * generic instantiation preserves every supported format combination. */
template <int R, int GateType = -1, int UpType = -1>
__global__ static void qwen4exp_moe_gateup_q_kernel(
        float *mid,
        const char *gate,
        const char *up,
        const int8_t *xq,
        const float *xs,
        const int32_t *xsum,
        const int32_t *pairs,
        const int32_t *counts,
        const int32_t *offsets,
        const int32_t *active,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint64_t up_expert_bytes,
        uint64_t up_row_bytes,
        uint32_t gate_type,
        uint32_t up_type,
        uint32_t groups,
        uint32_t mid_dim,
        uint32_t mid_token_stride,
        uint32_t n_expert_used) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t row = blockIdx.x * 8u + (threadIdx.x >> 5u);
    /* active[0] is how many experts this call actually chose; the launch
     * bounds the grid at the number of pairs, so the rest exit at once.  A
     * NULL list means the caller launched a row per expert instead, which is
     * how the two launch shapes are compared on one binary. */
    if (row >= mid_dim) return;
    uint32_t expert = blockIdx.y;
    if (active) {
        if ((int32_t)blockIdx.y >= active[0]) return;
        expert = (uint32_t)active[1 + blockIdx.y];
    }
    const int32_t cnt = counts[expert];
    if (cnt <= 0) return;
    const int32_t base = offsets[expert];

    const char *gate_row = gate + (uint64_t)expert * gate_expert_bytes +
                           (uint64_t)row * gate_row_bytes;
    const char *up_row = up + (uint64_t)expert * up_expert_bytes +
                         (uint64_t)row * up_row_bytes;

    for (int32_t at = 0; at < cnt; at += R) {
        const int32_t take = (cnt - at) < R ? (cnt - at) : R;
        uint32_t tok[R];
#pragma unroll
        for (int r = 0; r < R; r++) {
            const int32_t p = pairs[base + at + (r < take ? r : 0)];
            tok[r] = (uint32_t)p / n_expert_used;
        }
        float ag[R];
        float au[R];
#pragma unroll
        for (int r = 0; r < R; r++) { ag[r] = 0.0f; au[r] = 0.0f; }

        for (uint32_t g = lane; g < groups; g += 32u) {
            int8_t gw[32], uw[32];
            float ga[2], gb[2], ua[2], ub[2];
            int gh = 1, uh = 1;
            dev_qwen4exp_group_decode(
                    GateType < 0 ? gate_type : (uint32_t)GateType,
                    gate_row, g, gw, ga, gb, &gh);
            dev_qwen4exp_group_decode(
                    UpType < 0 ? up_type : (uint32_t)UpType,
                    up_row, g, uw, ua, ub, &uh);
#pragma unroll
            for (int r = 0; r < R; r++) {
                if (r < take) {
                    const uint64_t at_g = (uint64_t)tok[r] * groups + g;
                    const int8_t *xqg = xq + at_g * 32u;
                    const float sc = xs[at_g];
                    const int32_t sm = xsum[at_g];
                    qwen4exp_group_accumulate(&ag[r], gw, ga, gb, gh, xqg, sc, sm);
                    qwen4exp_group_accumulate(&au[r], uw, ua, ub, uh, xqg, sc, sm);
                }
            }
        }

#pragma unroll
        for (int r = 0; r < R; r++) {
            const float g = warp_sum_f32(ag[r]);
            const float u = warp_sum_f32(au[r]);
            if (lane == 0u && r < take) {
                const uint32_t p = (uint32_t)pairs[base + at + r];
                const uint32_t t = p / n_expert_used;
                const uint32_t slot = p - t * n_expert_used;
                mid[(uint64_t)t * mid_token_stride +
                    (uint64_t)slot * mid_dim + row] =
                    (g / (1.0f + expf(-g))) * u * weights[p];
            }
        }
    }
}

/* Grid (ceil(out_dim / 8), ceil(n_tokens / R)).  The slots of a token are
 * walked in ascending order into ONE accumulator, which is what the per-token
 * kernel did; the rows of the tile do not share a weight here, because each
 * one picked its own expert for the slot.  What the tile buys is that the
 * activation groups are read once for R rows and the decode is per group. */
template <int R, int DownType = -1>
__global__ static void qwen4exp_moe_down_q_kernel(
        float *out,
        const char *down,
        const int32_t *selected,
        const int8_t *mq,
        const float *ms,
        const int32_t *msum,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t down_type,
        uint32_t groups,
        uint32_t out_dim,
        uint32_t n_tokens,
        uint32_t n_total_expert,
        uint32_t n_expert_used) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t row = blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint32_t tok0 = blockIdx.y * (uint32_t)R;
    if (row >= out_dim || tok0 >= n_tokens) return;
    const uint32_t take = n_tokens - tok0 < (uint32_t)R ? n_tokens - tok0
                                                        : (uint32_t)R;

    float acc[R];
#pragma unroll
    for (int r = 0; r < R; r++) acc[r] = 0.0f;

    for (uint32_t slot = 0; slot < n_expert_used; slot++) {
#pragma unroll
        for (int r = 0; r < R; r++) {
            if ((uint32_t)r < take) {
                const uint32_t t = tok0 + (uint32_t)r;
                const int32_t e = selected[(uint64_t)t * n_expert_used + slot];
                if (e < 0 || (uint32_t)e >= n_total_expert) continue;
                const char *drow = down +
                    (uint64_t)(uint32_t)e * down_expert_bytes +
                    (uint64_t)row * down_row_bytes;
                const uint64_t mrow = (uint64_t)t * n_expert_used + slot;
                for (uint32_t g = lane; g < groups; g += 32u) {
                    int8_t wq[32];
                    float wa[2], wb[2];
                    int halves = 1;
                    dev_qwen4exp_group_decode(
                            DownType < 0 ? down_type : (uint32_t)DownType,
                            drow, g, wq, wa, wb, &halves);
                    const uint64_t at_g = mrow * groups + g;
                    qwen4exp_group_accumulate(&acc[r], wq, wa, wb, halves,
                                              mq + at_g * 32u, ms[at_g],
                                              msum[at_g]);
                }
            }
        }
    }

#pragma unroll
    for (int r = 0; r < R; r++) {
        const float tot = warp_sum_f32(acc[r]);
        if (lane == 0u && (uint32_t)r < take) {
            out[(uint64_t)(tok0 + (uint32_t)r) * out_dim + row] = tot;
        }
    }
}

/* The shared expert: no routing, so the tile is consecutive tokens and the
 * decoded group serves all of them. */
template <int R>
__global__ static void qwen4exp_shared_gateup_q_kernel(
        float *mid,
        const char *gate,
        const char *up,
        const int8_t *xq,
        const float *xs,
        const int32_t *xsum,
        uint64_t gate_row_bytes,
        uint64_t up_row_bytes,
        uint32_t gate_type,
        uint32_t up_type,
        uint32_t groups,
        uint32_t mid_dim,
        uint32_t n_tokens) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t row = blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint32_t tok0 = blockIdx.y * (uint32_t)R;
    if (row >= mid_dim || tok0 >= n_tokens) return;
    const uint32_t take = n_tokens - tok0 < (uint32_t)R ? n_tokens - tok0
                                                        : (uint32_t)R;
    const char *gate_row = gate + (uint64_t)row * gate_row_bytes;
    const char *up_row = up + (uint64_t)row * up_row_bytes;

    float ag[R];
    float au[R];
#pragma unroll
    for (int r = 0; r < R; r++) { ag[r] = 0.0f; au[r] = 0.0f; }

    for (uint32_t g = lane; g < groups; g += 32u) {
        int8_t gw[32], uw[32];
        float ga[2], gb[2], ua[2], ub[2];
        int gh = 1, uh = 1;
        dev_qwen4exp_group_decode(gate_type, gate_row, g, gw, ga, gb, &gh);
        dev_qwen4exp_group_decode(up_type, up_row, g, uw, ua, ub, &uh);
#pragma unroll
        for (int r = 0; r < R; r++) {
            if ((uint32_t)r < take) {
                const uint64_t at_g = (uint64_t)(tok0 + (uint32_t)r) * groups + g;
                const int8_t *xqg = xq + at_g * 32u;
                const float sc = xs[at_g];
                const int32_t sm = xsum[at_g];
                qwen4exp_group_accumulate(&ag[r], gw, ga, gb, gh, xqg, sc, sm);
                qwen4exp_group_accumulate(&au[r], uw, ua, ub, uh, xqg, sc, sm);
            }
        }
    }

#pragma unroll
    for (int r = 0; r < R; r++) {
        const float g = warp_sum_f32(ag[r]);
        const float u = warp_sum_f32(au[r]);
        if (lane == 0u && (uint32_t)r < take) {
            mid[(uint64_t)(tok0 + (uint32_t)r) * mid_dim + row] =
                (g / (1.0f + expf(-g))) * u;
        }
    }
}

template <int R>
__global__ static void qwen4exp_shared_down_q_kernel(
        float *out,
        const char *down,
        const int8_t *mq,
        const float *ms,
        const int32_t *msum,
        const float *gate_scale,
        uint64_t down_row_bytes,
        uint32_t down_type,
        uint32_t groups,
        uint32_t out_dim,
        uint32_t n_tokens) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t row = blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint32_t tok0 = blockIdx.y * (uint32_t)R;
    if (row >= out_dim || tok0 >= n_tokens) return;
    const uint32_t take = n_tokens - tok0 < (uint32_t)R ? n_tokens - tok0
                                                        : (uint32_t)R;
    const char *down_row = down + (uint64_t)row * down_row_bytes;

    float acc[R];
#pragma unroll
    for (int r = 0; r < R; r++) acc[r] = 0.0f;

    for (uint32_t g = lane; g < groups; g += 32u) {
        int8_t wq[32];
        float wa[2], wb[2];
        int halves = 1;
        dev_qwen4exp_group_decode(down_type, down_row, g, wq, wa, wb, &halves);
#pragma unroll
        for (int r = 0; r < R; r++) {
            if ((uint32_t)r < take) {
                const uint64_t at_g = (uint64_t)(tok0 + (uint32_t)r) * groups + g;
                qwen4exp_group_accumulate(&acc[r], wq, wa, wb, halves,
                                          mq + at_g * 32u, ms[at_g],
                                          msum[at_g]);
            }
        }
    }

#pragma unroll
    for (int r = 0; r < R; r++) {
        const float tot = warp_sum_f32(acc[r]);
        if (lane == 0u && (uint32_t)r < take) {
            const uint64_t off = (uint64_t)(tok0 + (uint32_t)r) * out_dim + row;
            out[off] += gate_scale[tok0 + (uint32_t)r] * tot;
        }
    }
}

/* =========================================================================
 * The shared expert, with the decoded row staged in shared memory.
 * =========================================================================
 *
 * WHAT THE PER-ROW KERNELS ABOVE SPEND.  `qwen4exp_shared_gateup_q_kernel`
 * gives one warp one output row and one tile of R tokens, so at a prefill of
 * n tokens the same weight row is decoded ceil(n / R) times -- 128 times at
 * 1024 tokens with R = 8 -- and every one of those decodes reads the same
 * bytes out of the same row.  Worse, `qwen4exp_dp4a` assembles each four-byte
 * operand out of four single-byte loads, so the activation side of the inner
 * loop issues 32 LDG.E.U8 per group per token: in the emitted SASS the R = 8
 * shared kernels carry more than five byte loads for every IDP.4A they feed.
 * The kernel is neither weight-bound nor MAC-bound; it is bound on
 * load-issue slots spent one byte at a time.
 *
 * WHAT THESE KERNELS DO INSTEAD.  A block owns ONE output row and all 256 of
 * its threads decode that row's groups once, cooperatively, into shared
 * memory -- as the eight four-byte words `qwen4exp_dp4a` would have assembled
 * from the decoded bytes, plus the (wa, wb) pair and the half count that group
 * produced.  After one barrier each of the eight warps takes its own tile of R
 * tokens and walks `for (g = lane; g < groups; g += 32)` exactly as the per-row
 * kernel does, reading the staged group instead of decoding it.  Eight warps
 * times R = 8 is 64 tokens served by one decode of the row, so the decode count
 * per row drops by eight, and the activation group is read as two sixteen-byte
 * loads instead of thirty-two one-byte loads.
 *
 * WHY THE NUMBERS DO NOT MOVE.  The staged value IS the decoder's output: the
 * words are `qwen4exp_load_i8x4` of the very bytes `dev_qwen4exp_group_decode`
 * wrote, and shared memory is a copy, not a re-representation.  The activation
 * words are the same four bytes the byte path packed, in the same little-endian
 * order, so every __dp4a takes the operands it took before.  Above all the
 * SCHEDULE is untouched: for a given (row, token) it is still lane `l` of one
 * warp that accumulates groups l, l + 32, l + 64, ... in ascending order into
 * one private float, and it is still the same `warp_sum_f32` butterfly that
 * folds the thirty-two lane partials.  Only which block that warp sits in, and
 * how many tokens share the decode with it, change -- and neither is an
 * operand.  Nothing above this comment is edited: the per-row kernels remain
 * available (DS4_QWEN4EXP_SHARED_STAGE=0, or any forced DS4_QWEN4EXP_MOE_R)
 * and are the bit-for-bit oracle the test compares this path against.
 */

enum {
    QWEN4EXP_STAGE_WARPS = 8,
    QWEN4EXP_STAGE_THREADS = 32 * QWEN4EXP_STAGE_WARPS,
    /* The token tile one warp carries.  It is the tile the per-row kernel
     * runs at a prefill, and it is only a schedule: `take` clamps it, so a
     * token's arithmetic does not depend on how many neighbours share its
     * warp.  Raising it alone is what the register file refuses; raising the
     * warps per decode is what this path does instead. */
    QWEN4EXP_STAGE_R = 8,
    /* Below this many tokens the per-row kernels are still the faster pair,
     * and they stay the ones that run.  Measured on the GB10 at the
     * checkpoint's shared-expert shape (in 2560, mid 640, out 2560, Q8_0),
     * one call, milliseconds:
     *
     *     tokens      1      4      8     12     16     32    128   1024
     *     per-row  0.029  0.049  0.086  0.111  0.152  0.263  0.987  7.755
     *     staged   0.068  0.085  0.113  0.120  0.127  0.137  0.299  3.113
     *
     * A staged block spends its whole decode phase on one row whatever the
     * token count, so at a decode width most of its work and seven of its
     * eight warps are wasted; the per-row kernel is DRAM-bound there and
     * cannot be beaten by moving the decode.  They cross near twelve tokens.
     * Both paths return the same bits, so this is only ever a speed choice. */
    QWEN4EXP_STAGE_MIN_TOKENS = 16,
};

/* `qwen4exp_dp4a` over operands already packed into four-byte words.  Same
 * number of __dp4a, same order, same accumulator chain; a word here is the
 * little-endian packing `qwen4exp_load_i8x4` returns for the four bytes it
 * replaces, so each __dp4a takes the identical pair of operands. */
template <int N>
__device__ __forceinline__ static int32_t qwen4exp_dp4a_w(const int32_t *a,
                                                          const int32_t *b) {
    int32_t d = 0;
#pragma unroll
    for (int i = 0; i < N / 4; i++) {
        d = __dp4a(a[i], b[i], d);
    }
    return d;
}

/* `qwen4exp_group_accumulate` over word operands.  The two floating-point
 * statements are the ones above, character for character, so the compiler
 * builds the same expression tree and contracts it the same way; only the
 * integer dot's operand form differs, and that form holds the same bits. */
__device__ __forceinline__ static void qwen4exp_group_accumulate_w(
        float *acc, const int32_t *wq, const float *wa, const float *wb,
        int halves, const int32_t *xqg, float xscale, int32_t xsum) {
    if (halves == 1) {
        const int32_t dot = qwen4exp_dp4a_w<32>(wq, xqg);
        *acc += (wa[0] * xscale) * (float)dot;
        *acc += (wb[0] * xscale) * (float)xsum;
    } else {
        const int32_t d0 = qwen4exp_dp4a_w<16>(wq, xqg);
        const int32_t d1 = qwen4exp_dp4a_w<16>(wq + 4, xqg + 4);
        *acc += (wa[0] * xscale) * (float)d0;
        *acc += (wa[1] * xscale) * (float)d1;
    }
}

/* The thirty-two decoded bytes of one group as the eight words the dot wants.
 * This is `qwen4exp_load_i8x4` at the eight offsets the dot reads, run once at
 * staging time instead of once per token. */
__device__ __forceinline__ static void qwen4exp_pack_group(
        int4 *lo, int4 *hi, const int8_t *wq) {
    lo->x = qwen4exp_load_i8x4(wq + 0);
    lo->y = qwen4exp_load_i8x4(wq + 4);
    lo->z = qwen4exp_load_i8x4(wq + 8);
    lo->w = qwen4exp_load_i8x4(wq + 12);
    hi->x = qwen4exp_load_i8x4(wq + 16);
    hi->y = qwen4exp_load_i8x4(wq + 20);
    hi->z = qwen4exp_load_i8x4(wq + 24);
    hi->w = qwen4exp_load_i8x4(wq + 28);
}

/* One quantised activation group, read as two sixteen-byte loads.  The
 * scratch base is a cudaMalloc return and a group starts at a multiple of
 * thirty-two bytes from it, so the address is sixteen-byte aligned; the host
 * checks that before it picks this kernel and keeps the per-row path when it
 * does not hold. */
__device__ __forceinline__ static void qwen4exp_load_group_w(
        int32_t *out, const int8_t *xqg) {
    const int4 *v = (const int4 *)(const void *)xqg;
    const int4 a = v[0];
    const int4 b = v[1];
    out[0] = a.x; out[1] = a.y; out[2] = a.z; out[3] = a.w;
    out[4] = b.x; out[5] = b.y; out[6] = b.z; out[7] = b.w;
}

/* Shared-memory bytes one staged row needs.  Two int4 arrays hold the eight
 * words of a group; the scalar arrays hold what the decoder returned in
 * (wa[0], wa[1], wb[0]) and the half count. */
static uint64_t qwen4exp_stage_bytes(uint64_t groups, int matrices) {
    return (uint64_t)matrices * groups * (2u * sizeof(int4) +
                                          3u * sizeof(float) + sizeof(int));
}

/* Is the staged path the right one for this call?
 *
 * It is declined when a test pins the tile (DS4_QWEN4EXP_MOE_R selects among
 * the per-row kernels and must keep selecting among them), when
 * DS4_QWEN4EXP_SHARED_STAGE is set to 0 -- which is how the test runs the
 * per-row kernels as this path's oracle -- when the activation scratch is not
 * sixteen-byte aligned, so the wide group load would fault, when one row's
 * staged form does not fit the default 48 KiB dynamic shared-memory budget,
 * and when the call is narrower than QWEN4EXP_STAGE_MIN_TOKENS, where the
 * per-row kernels are still faster.  Setting the variable to 1 asks for the
 * staged path at every width, which is how the test sweeps its tails.
 * Every decline lands on the untouched per-row kernels, which produce the same
 * bits, so a decline costs speed and never accuracy. */
static int qwen4exp_shared_stage_ok(const void *quant, uint32_t groups,
                                    int matrices, uint32_t n_tokens) {
    const char *sel = getenv("DS4_QWEN4EXP_SHARED_STAGE");
    const int forced = sel && sel[0] == '1' && sel[1] == '\0';
    if (sel && sel[0] == '0' && sel[1] == '\0') return 0;
    if (getenv("DS4_QWEN4EXP_MOE_R")) return 0;
    if (groups == 0u) return 0;
    if (((uintptr_t)quant & 15u) != 0u) return 0;
    if (qwen4exp_stage_bytes(groups, matrices) > 48u * 1024u) return 0;
    if (!forced && n_tokens < (uint32_t)QWEN4EXP_STAGE_MIN_TOKENS) return 0;
    return 1;
}

template <int R>
__global__ static void qwen4exp_shared_gateup_stage_kernel(
        float *mid,
        const char *gate,
        const char *up,
        const int8_t *xq,
        const float *xs,
        const int32_t *xsum,
        uint64_t gate_row_bytes,
        uint64_t up_row_bytes,
        uint32_t gate_type,
        uint32_t up_type,
        uint32_t groups,
        uint32_t mid_dim,
        uint32_t n_tokens) {
    extern __shared__ __align__(16) char qwen4exp_stage_smem[];
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t row = blockIdx.x;
    const char *gate_row = gate + (uint64_t)row * gate_row_bytes;
    const char *up_row = up + (uint64_t)row * up_row_bytes;

    int4 *s_lo = (int4 *)qwen4exp_stage_smem;            /* [2][groups] */
    int4 *s_hi = s_lo + 2u * groups;                     /* [2][groups] */
    float *s_a0 = (float *)(s_hi + 2u * groups);         /* [2][groups] */
    float *s_a1 = s_a0 + 2u * groups;
    float *s_b0 = s_a1 + 2u * groups;
    int *s_h = (int *)(s_b0 + 2u * groups);

    /* Decode the row once for the whole block.  Item `g` is the gate group,
     * item `groups + g` the up group, so consecutive threads read consecutive
     * blocks of one row exactly as consecutive lanes did before. */
    for (uint32_t item = threadIdx.x; item < groups * 2u; item += blockDim.x) {
        const bool second = item >= groups;
        const uint32_t g = second ? item - groups : item;
        int8_t wq[32];
        float wa[2], wb[2];
        int halves = 1;
        dev_qwen4exp_group_decode(second ? up_type : gate_type,
                                  second ? up_row : gate_row, g,
                                  wq, wa, wb, &halves);
        qwen4exp_pack_group(&s_lo[item], &s_hi[item], wq);
        s_a0[item] = wa[0];
        s_a1[item] = wa[1];
        s_b0[item] = wb[0];
        s_h[item] = halves;
    }
    __syncthreads();

    const uint32_t tok0 = blockIdx.y * (uint32_t)(R * QWEN4EXP_STAGE_WARPS) +
                          warp * (uint32_t)R;
    if (tok0 >= n_tokens) return;
    const uint32_t take = n_tokens - tok0 < (uint32_t)R ? n_tokens - tok0
                                                        : (uint32_t)R;

    float ag[R];
    float au[R];
#pragma unroll
    for (int r = 0; r < R; r++) { ag[r] = 0.0f; au[r] = 0.0f; }

    for (uint32_t g = lane; g < groups; g += 32u) {
        const int4 glo = s_lo[g], ghi = s_hi[g];
        const int4 ulo = s_lo[groups + g], uhi = s_hi[groups + g];
        const int32_t gw[8] = { glo.x, glo.y, glo.z, glo.w,
                                ghi.x, ghi.y, ghi.z, ghi.w };
        const int32_t uw[8] = { ulo.x, ulo.y, ulo.z, ulo.w,
                                uhi.x, uhi.y, uhi.z, uhi.w };
        const float ga[2] = { s_a0[g], s_a1[g] };
        const float gb[2] = { s_b0[g], 0.0f };
        const int gh = s_h[g];
        const float ua[2] = { s_a0[groups + g], s_a1[groups + g] };
        const float ub[2] = { s_b0[groups + g], 0.0f };
        const int uh = s_h[groups + g];
#pragma unroll
        for (int r = 0; r < R; r++) {
            if ((uint32_t)r < take) {
                const uint64_t at_g = (uint64_t)(tok0 + (uint32_t)r) * groups + g;
                int32_t xqg[8];
                qwen4exp_load_group_w(xqg, xq + at_g * 32u);
                const float sc = xs[at_g];
                const int32_t sm = xsum[at_g];
                qwen4exp_group_accumulate_w(&ag[r], gw, ga, gb, gh, xqg, sc, sm);
                qwen4exp_group_accumulate_w(&au[r], uw, ua, ub, uh, xqg, sc, sm);
            }
        }
    }

#pragma unroll
    for (int r = 0; r < R; r++) {
        const float g = warp_sum_f32(ag[r]);
        const float u = warp_sum_f32(au[r]);
        if (lane == 0u && (uint32_t)r < take) {
            mid[(uint64_t)(tok0 + (uint32_t)r) * mid_dim + row] =
                (g / (1.0f + expf(-g))) * u;
        }
    }
}

template <int R>
__global__ static void qwen4exp_shared_down_stage_kernel(
        float *out,
        const char *down,
        const int8_t *mq,
        const float *ms,
        const int32_t *msum,
        const float *gate_scale,
        uint64_t down_row_bytes,
        uint32_t down_type,
        uint32_t groups,
        uint32_t out_dim,
        uint32_t n_tokens) {
    extern __shared__ __align__(16) char qwen4exp_stage_smem[];
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t row = blockIdx.x;
    const char *down_row = down + (uint64_t)row * down_row_bytes;

    int4 *s_lo = (int4 *)qwen4exp_stage_smem;
    int4 *s_hi = s_lo + groups;
    float *s_a0 = (float *)(s_hi + groups);
    float *s_a1 = s_a0 + groups;
    float *s_b0 = s_a1 + groups;
    int *s_h = (int *)(s_b0 + groups);

    for (uint32_t g = threadIdx.x; g < groups; g += blockDim.x) {
        int8_t wq[32];
        float wa[2], wb[2];
        int halves = 1;
        dev_qwen4exp_group_decode(down_type, down_row, g, wq, wa, wb, &halves);
        qwen4exp_pack_group(&s_lo[g], &s_hi[g], wq);
        s_a0[g] = wa[0];
        s_a1[g] = wa[1];
        s_b0[g] = wb[0];
        s_h[g] = halves;
    }
    __syncthreads();

    const uint32_t tok0 = blockIdx.y * (uint32_t)(R * QWEN4EXP_STAGE_WARPS) +
                          warp * (uint32_t)R;
    if (tok0 >= n_tokens) return;
    const uint32_t take = n_tokens - tok0 < (uint32_t)R ? n_tokens - tok0
                                                        : (uint32_t)R;

    float acc[R];
#pragma unroll
    for (int r = 0; r < R; r++) acc[r] = 0.0f;

    for (uint32_t g = lane; g < groups; g += 32u) {
        const int4 wlo = s_lo[g], whi = s_hi[g];
        const int32_t wq[8] = { wlo.x, wlo.y, wlo.z, wlo.w,
                                whi.x, whi.y, whi.z, whi.w };
        const float wa[2] = { s_a0[g], s_a1[g] };
        const float wb[2] = { s_b0[g], 0.0f };
        const int halves = s_h[g];
#pragma unroll
        for (int r = 0; r < R; r++) {
            if ((uint32_t)r < take) {
                const uint64_t at_g = (uint64_t)(tok0 + (uint32_t)r) * groups + g;
                int32_t mqg[8];
                qwen4exp_load_group_w(mqg, mq + at_g * 32u);
                qwen4exp_group_accumulate_w(&acc[r], wq, wa, wb, halves,
                                            mqg, ms[at_g], msum[at_g]);
            }
        }
    }

#pragma unroll
    for (int r = 0; r < R; r++) {
        const float tot = warp_sum_f32(acc[r]);
        if (lane == 0u && (uint32_t)r < take) {
            const uint64_t off = (uint64_t)(tok0 + (uint32_t)r) * out_dim + row;
            out[off] += gate_scale[tok0 + (uint32_t)r] * tot;
        }
    }
}


/* Q8_0 blocks have a two-byte header and a 34-byte stride.  Read the same
 * packed quants without an unaligned word load or a read past the block. */
__device__ __forceinline__ static uint32_t qw_shared_q8_word(const char *p) {
    if (qwen4exp_word_aligned(p)) return qw_tile_word((const int8_t *)p);
    if (((uintptr_t)p & 1u) == 0u) {
        const uint16_t *h = (const uint16_t *)(const void *)p;
        return (uint32_t)h[0] | ((uint32_t)h[1] << 16u);
    }
    return qw_pack4((const int8_t *)p);
}

enum { QW_SH_BM = 16, QW_SH_BN = 32, QW_SH_NT = QW_SH_BN / 8,
       QW_SH_WARPS = 16, QW_SH_THREADS = QW_SH_WARPS * 32 };

/* Each warp computes the group sums of dp4a lanes W and W+16 separately:
 * g = L, L+32, ... .  Adding these two sums is reduction step 16.  Shared
 * memory then performs steps 8, 4, 2, 1 with the same left/right operands as
 * warp_sum_f32.  Each MMA starts at int32 zero and covers ONE group only.
 * Weights go straight from the original Q8_0 bytes to operand registers.
 *
 * GateUp fuses gate/up/SiLU; the other instance adds gate_scale * down to out.
 * wb is passed as +0.0f: keep the dp4a offset expression as a runtime operand,
 * including its signed-zero behavior.  Do not fold it away or reassociate.
 */
template <bool GateUp>
__global__ __launch_bounds__(QW_SH_THREADS) static void
qwen4exp_shared_q8_mma_kernel(
        float *out, const char *w0, const char *w1,
        const int8_t *xq, const float *xs, const int32_t *xsum,
        const float *gate_scale, uint64_t row_bytes0, uint64_t row_bytes1,
        uint32_t groups, uint32_t out_dim, uint32_t n_tokens, float wb) {
    enum { NP = GateUp ? 2 : 1, NF = QW_SH_NT * 4 };
    __shared__ float partial[NP][QW_SH_WARPS / 2][NF][32];
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t row0 = blockIdx.x * QW_SH_BM;
    const uint32_t tok0 = blockIdx.y * QW_SH_BN;
    const uint32_t mr0 = row0 + (lane >> 2u);
    const uint32_t ak = (lane & 3u) * 4u;
    float total[NP][NF];

#pragma unroll
    for (int half = 0; half < 2; half++) {
        float acc[NP][NF];
#pragma unroll
        for (int p = 0; p < NP; p++) {
#pragma unroll
            for (int i = 0; i < NF; i++) acc[p][i] = 0.0f;
        }
        for (uint32_t g = warp + (uint32_t)half * 16u; g < groups; g += 32u) {
            uint32_t a[NP][4];
            float wa[NP][2];
#pragma unroll
            for (int p = 0; p < NP; p++) {
                const char *weight = p == 0 ? w0 : w1;
                const uint64_t row_bytes = p == 0 ? row_bytes0 : row_bytes1;
#pragma unroll
                for (int r = 0; r < 2; r++) {
                    const uint32_t mr = mr0 + (uint32_t)r * 8u;
                    a[p][r] = a[p][r + 2] = 0u;
                    wa[p][r] = 0.0f;
                    if (mr < out_dim) {
                        const char *blk = weight + (uint64_t)mr * row_bytes +
                                          (uint64_t)g * 34u;
                        const uint16_t d = (uint16_t)(uint8_t)blk[0] |
                                          ((uint16_t)(uint8_t)blk[1] << 8u);
                        wa[p][r] = dev_f16_to_f32(d);
                        a[p][r] = qw_shared_q8_word(blk + 2u + ak);
                        a[p][r + 2] = qw_shared_q8_word(blk + 18u + ak);
                    }
                }
            }
#pragma unroll
            for (int nt = 0; nt < QW_SH_NT; nt++) {
                const uint32_t bt = tok0 + (uint32_t)nt * 8u + (lane >> 2u);
                uint32_t b[2] = {0u, 0u};
                if (bt < n_tokens) {
                    const int8_t *q = xq + ((uint64_t)bt * groups + g) * 32u;
                    b[0] = qw_tile_word(q + ak);
                    b[1] = qw_tile_word(q + ak + 16u);
                }
#pragma unroll
                for (int p = 0; p < NP; p++) {
                    int32_t dot[4] = {0, 0, 0, 0};
                    qw_mma_m16n8k32(dot, a[p], b);
#pragma unroll
                    for (int r = 0; r < 4; r++) {
                        const uint32_t tok = tok0 + (uint32_t)nt * 8u +
                                             (lane & 3u) * 2u + (r & 1);
                        if (tok < n_tokens) {
                            const uint64_t at = (uint64_t)tok * groups + g;
                            const float sc = xs[at];
                            const int32_t sm = xsum[at];
                            /* qwen4exp_group_accumulate, halves == 1. */
                            acc[p][nt * 4 + r] += (wa[p][r >> 1] * sc) * (float)dot[r];
                            acc[p][nt * 4 + r] += (wb * sc) * (float)sm;
                        }
                    }
                }
            }
        }
#pragma unroll
        for (int p = 0; p < NP; p++) {
#pragma unroll
            for (int i = 0; i < NF; i++) {
                if (half == 0) total[p][i] = acc[p][i];
                else total[p][i] += acc[p][i];
            }
        }
    }

#pragma unroll
    for (int offset = 8; offset > 0; offset >>= 1) {
        if (warp >= (uint32_t)offset && warp < (uint32_t)offset * 2u) {
#pragma unroll
            for (int p = 0; p < NP; p++) {
#pragma unroll
                for (int i = 0; i < NF; i++)
                    partial[p][warp - offset][i][lane] = total[p][i];
            }
        }
        __syncthreads();
        if (warp < (uint32_t)offset) {
#pragma unroll
            for (int p = 0; p < NP; p++) {
#pragma unroll
                for (int i = 0; i < NF; i++)
                    total[p][i] += partial[p][warp][i][lane];
            }
        }
        __syncthreads();
    }

    if (warp == 0u) {
#pragma unroll
        for (int nt = 0; nt < QW_SH_NT; nt++) {
#pragma unroll
            for (int r = 0; r < 4; r++) {
                const uint32_t row = mr0 + ((r & 2) ? 8u : 0u);
                const uint32_t tok = tok0 + (uint32_t)nt * 8u +
                                     (lane & 3u) * 2u + (r & 1);
                if (row < out_dim && tok < n_tokens) {
                    const uint64_t at = (uint64_t)tok * out_dim + row;
                    const float v = total[0][nt * 4 + r];
                    if (GateUp) {
                        out[at] = (v / (1.0f + expf(-v))) * total[1][nt * 4 + r];
                    } else {
                        out[at] += gate_scale[tok] * v;
                    }
                }
            }
        }
    }
}

__global__ static void qwen4exp_shared_gate_kernel(
        float *gate_out,
        const char *router,
        const float *x,
        uint32_t router_type,
        uint32_t in_dim,
        uint32_t n_tokens) {
    extern __shared__ float ds4_qwen4exp_smem[];
    const uint32_t token = blockIdx.x;
    if (token >= n_tokens) return;
    const float *token_x = x + (uint64_t)token * in_dim;
    float acc = 0.0f;
    for (uint32_t k = threadIdx.x; k < in_dim; k += blockDim.x) {
        acc += dev_qwen4exp_weight_value(router_type, router, k) * token_x[k];
    }
    const float total = dev_qwen4exp_block_sum(ds4_qwen4exp_smem, acc);
    if (threadIdx.x == 0u) gate_out[token] = 1.0f / (1.0f + expf(-total));
}



/* Twin of ds4_gpu_qwen4exp_moe_type_supported in ds4_metal.m. */
static bool cuda_qwen4exp_moe_type_supported(uint32_t type) {
#define DS4_QWEN4EXP_TYPE_MATCH(name, id) if (type == (uint32_t)(id)) return true;
    DS4_QWEN4EXP_MOE_TYPES(DS4_QWEN4EXP_TYPE_MATCH)
#undef DS4_QWEN4EXP_TYPE_MATCH
    return false;
}

extern "C" int ds4_gpu_qwen4exp_router_select_tensor(
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *weights,
        const ds4_gpu_tensor *logits,
        uint32_t              n_expert,
        uint32_t              n_expert_used,
        uint32_t              n_tokens) {
    if (!selected || !weights || !logits || n_tokens == 0 ||
        n_expert == 0 || n_expert > 512u ||
        n_expert_used == 0 || n_expert_used > n_expert) {
        return 0;
    }
    if (logits->bytes < (uint64_t)n_tokens * n_expert * sizeof(float) ||
        selected->bytes < (uint64_t)n_tokens * n_expert_used * sizeof(int32_t) ||
        weights->bytes < (uint64_t)n_tokens * n_expert_used * sizeof(float)) {
        fprintf(stderr, "ds4: CUDA qwen4exp router received undersized buffers\n");
        return 0;
    }
    if (n_expert_used <= 32u) {
        qwen4exp_router_select_topk_kernel<<<
                n_tokens, 32u, 0, cuda_decode_stream()>>>(
                (int32_t *)selected->ptr,
                (float *)weights->ptr,
                (const float *)logits->ptr,
                n_expert, n_expert_used, n_tokens);
    } else {
        const unsigned threads = n_expert > 256u ? 512u : 256u;
        qwen4exp_router_select_kernel<<<
                n_tokens, threads, 0, cuda_decode_stream()>>>(
                (int32_t *)selected->ptr,
                (float *)weights->ptr,
                (const float *)logits->ptr,
                n_expert, n_expert_used, n_tokens);
    }
    return cuda_ok(cudaGetLastError(), "qwen4exp router select launch");
}
/* The MTP head forms n_hc rows [e_normed(t) | h_normed(t,s)] before its
 * eh_proj.  At the production shape the portable implementation records eight
 * short device copies for every one-row head step.  The rows are pure copies,
 * so one flat kernel produces the identical layout while paying one launch. */
__global__ static void qwen4exp_ehx_pack_kernel(
        float *out, const float *embedding, const float *hidden,
        uint32_t n_hc, uint32_t n_embd) {
    const uint64_t pair = blockIdx.x;
    const uint32_t t = (uint32_t)(pair / n_hc);
    const uint64_t dst = pair * 2ull * n_embd;
    const uint64_t e_src = (uint64_t)t * n_embd;
    const uint64_t h_src = pair * n_embd;
    for (uint32_t k = threadIdx.x; k < n_embd; k += blockDim.x) {
        out[dst + k] = embedding[e_src + k];
        out[dst + n_embd + k] = hidden[h_src + k];
    }
}

extern "C" int ds4_gpu_qwen4exp_ehx_pack_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *embedding,
        const ds4_gpu_tensor *hidden,
        uint32_t              n_tokens,
        uint32_t              n_hc,
        uint32_t              n_embd) {
    if (!out || !embedding || !hidden || n_tokens == 0u || n_hc == 0u ||
        n_embd == 0u) {
        return 0;
    }
    const uint64_t pairs = (uint64_t)n_tokens * n_hc;
    const uint64_t total = pairs * 2ull * n_embd;
    if (out->bytes < total * sizeof(float) ||
        embedding->bytes < (uint64_t)n_tokens * n_embd * sizeof(float) ||
        hidden->bytes < pairs * n_embd * sizeof(float)) {
        fprintf(stderr, "ds4: CUDA qwen4exp ehx pack received undersized buffers\n");
        return 0;
    }
    const unsigned threads = 256u;
    qwen4exp_ehx_pack_kernel<<<
            (unsigned)pairs, threads, 0,
            cuda_decode_stream()>>>(
            (float *)out->ptr, (const float *)embedding->ptr,
            (const float *)hidden->ptr, n_hc, n_embd);
    return cuda_ok(cudaGetLastError(), "qwen4exp ehx pack launch");
}
/* Scratch for one expert call: the pair list, then the Q8_0 form of the
 * activation the projections consume.  Laid out here so the sizes are visible
 * beside the kernels that read them.
 *
 * At a prefill width of 512 with this model's shape the whole block is about
 * 6 MiB: 20 KiB of pair list, 1.6 MiB for the tower activation and 4.1 MiB for
 * the routed intermediate.  It is kept and grown, never allocated per call. */
typedef struct {
    int32_t *counts;
    int32_t *offsets;
    int32_t *cursor;
    int32_t *active;
    int32_t *pairs;
    int8_t  *xq;
    float   *xs;
    int32_t *xsum;
    int8_t  *mq;
    float   *ms;
    int32_t *msum;
} qwen4exp_moe_scratch;

static uint64_t qwen4exp_quant_bytes(uint64_t rows, uint64_t groups) {
    return rows * groups * (32u + sizeof(float) + sizeof(int32_t));
}

/* Quantise `rows` rows of `width` floats, where row r starts at
 * outer_stride * (r / inner_count) + inner_stride * (r % inner_count).  The
 * routed intermediate is addressed that way; a plain matrix passes
 * inner_count 1. */
static int qwen4exp_quantize_rows(
        int8_t *xq, float *xs, int32_t *xsum, const float *src,
        uint32_t rows, uint32_t width, uint32_t groups,
        uint64_t outer_stride, uint64_t inner_stride, uint32_t inner_count,
        cudaStream_t stream) {
    qwen4exp_quantize_rows_kernel<<<dim3(groups, rows, 1), 32, 0, stream>>>(
            xq, xs, xsum, src, width, groups,
            outer_stride, inner_stride, inner_count);
    return cuda_ok(cudaGetLastError(), "qwen4exp activation quantise");
}

/* The row tile.  One row is the decode call and takes the same kernel at
 * R = 1, which is what makes the tower row invariant by construction rather
 * than by comparison.  DS4_QWEN4EXP_MOE_R pins the tile so a test can run two
 * of them over the same rows. */
static int qwen4exp_moe_tile(uint32_t n_rows) {
    const char *forced = getenv("DS4_QWEN4EXP_MOE_R");
    if (forced) {
        const int r = atoi(forced);
        if (r == 1 || r == 4 || r == 8) return r;
    }
    if (n_rows >= 8u) return 8;
    if (n_rows >= 4u) return 4;
    /* Decode/verify widths (1-7 rows): the 8-wide padded dp4a tile is
     * measurably faster on GB10 than the per-row kernel and emits identical
     * output (row-invariant by construction; stream-diffed over three seeds). */
    return 8;
}

extern "C" int ds4_gpu_qwen4exp_routed_moe_tensor(
        ds4_gpu_tensor              *out,
        ds4_gpu_tensor              *mid,
        ds4_gpu_tensor              *down_partial,
        const ds4_gpu_qwen4exp_slab *gate_slab,
        const ds4_gpu_qwen4exp_slab *up_slab,
        const ds4_gpu_qwen4exp_slab *down_slab,
        uint32_t                     in_dim,
        uint32_t                     mid_dim,
        uint32_t                     out_dim,
        const ds4_gpu_tensor        *selected,
        const ds4_gpu_tensor        *weights,
        uint32_t                     n_total_expert,
        uint32_t                     n_expert_used,
        const ds4_gpu_tensor        *x,
        uint32_t                     n_tokens,
        uint32_t                     mid_token_stride) {
    if (!out || !mid || !gate_slab || !up_slab || !down_slab ||
        !gate_slab->map || !up_slab->map || !down_slab->map ||
        !selected || !weights || !x ||
        in_dim == 0 || mid_dim == 0 || out_dim == 0 || n_tokens == 0 ||
        n_total_expert == 0 || n_expert_used == 0 ||
        n_expert_used > n_total_expert ||
        (uint64_t)mid_token_stride < (uint64_t)n_expert_used * mid_dim) {
        return 0;
    }
    if (!cuda_qwen4exp_moe_type_supported(gate_slab->type) ||
        !cuda_qwen4exp_moe_type_supported(up_slab->type) ||
        !cuda_qwen4exp_moe_type_supported(down_slab->type)) {
        fprintf(stderr, "ds4: CUDA qwen4exp MoE unsupported expert types "
                        "(gate %u up %u down %u)\n",
                gate_slab->type, up_slab->type, down_slab->type);
        return 0;
    }
    /* The contract walks groups of 32 and the K-quants carry a 256-element
     * super-block, so a width that does not divide would silently read past a
     * block.  Refuse by name instead. */
    if ((in_dim & 31u) != 0 || (mid_dim & 31u) != 0) {
        fprintf(stderr, "ds4: CUDA qwen4exp MoE needs in_dim and mid_dim in "
                        "whole groups of 32 (got %u and %u)\n", in_dim, mid_dim);
        return 0;
    }
    if (!down_partial ||
        down_partial->bytes <
            (uint64_t)n_tokens * n_expert_used * out_dim * sizeof(float)) {
        fprintf(stderr, "ds4: CUDA qwen4exp MoE down partial buffer too small "
                        "for %u rows\n", n_tokens);
        return 0;
    }
    if (out->bytes < (uint64_t)n_tokens * out_dim * sizeof(float) ||
        mid->bytes < (uint64_t)n_tokens * mid_token_stride * sizeof(float) ||
        x->bytes < (uint64_t)n_tokens * in_dim * sizeof(float) ||
        selected->bytes < (uint64_t)n_tokens * n_expert_used * sizeof(int32_t) ||
        weights->bytes < (uint64_t)n_tokens * n_expert_used * sizeof(float)) {
        fprintf(stderr, "ds4: CUDA qwen4exp MoE received undersized buffers\n");
        return 0;
    }

    /* Each slab resolves through its OWN mapping: one block's expert tensors
     * can live in different shards of a split GGUF. */
    const int logical_tier = cuda_current_tier();
    const char *gate = cuda_resolve_weight_ptr(gate_slab->map, gate_slab->offset,
            (uint64_t)n_total_expert * gate_slab->expert_bytes, logical_tier,
            "qwen4exp_ffn_gate_exps");
    const char *up = cuda_resolve_weight_ptr(up_slab->map, up_slab->offset,
            (uint64_t)n_total_expert * up_slab->expert_bytes, logical_tier,
            "qwen4exp_ffn_up_exps");
    const char *down = cuda_resolve_weight_ptr(down_slab->map, down_slab->offset,
            (uint64_t)n_total_expert * down_slab->expert_bytes, logical_tier,
            "qwen4exp_ffn_down_exps");
    if (!gate || !up || !down) return 0;

    const uint32_t n_pairs = n_tokens * n_expert_used;
    const uint32_t xgroups = in_dim / 32u;
    const uint32_t mgroups = mid_dim / 32u;
    const uint64_t idx_bytes =
        ((uint64_t)n_total_expert * 4u + 1u) * sizeof(int32_t);
    const uint64_t pair_bytes = (uint64_t)n_pairs * sizeof(int32_t);
    const uint64_t xq_bytes = qwen4exp_quant_bytes(n_tokens, xgroups);
    const uint64_t mq_bytes = qwen4exp_quant_bytes(n_pairs, mgroups);

    char *base = (char *)qwen4exp_group_scratch(
            logical_tier, idx_bytes + pair_bytes + xq_bytes + mq_bytes);
    if (!base) return 0;
    qwen4exp_moe_scratch sc;
    sc.counts = (int32_t *)base;
    sc.offsets = sc.counts + n_total_expert;
    sc.cursor = sc.offsets + n_total_expert;
    sc.active = sc.cursor + n_total_expert;
    sc.pairs = sc.active + n_total_expert + 1u;
    char *at = base + idx_bytes + pair_bytes;
    sc.xq = (int8_t *)at;
    sc.xs = (float *)(at + (uint64_t)n_tokens * xgroups * 32u);
    sc.xsum = (int32_t *)(sc.xs + (uint64_t)n_tokens * xgroups);
    at += xq_bytes;
    sc.mq = (int8_t *)at;
    sc.ms = (float *)(at + (uint64_t)n_pairs * mgroups * 32u);
    sc.msum = (int32_t *)(sc.ms + (uint64_t)n_pairs * mgroups);

    cudaStream_t stream = cuda_decode_stream();
    const unsigned threads = 256u;
    const unsigned pair_blocks = (n_pairs + threads - 1u) / threads;

    const int small_group =
        n_tokens < 8u && n_total_expert <= QWEN4EXP_MOE_SCAN_THREADS &&
        getenv("DS4_QWEN4EXP_SERIAL_GROUP_SCAN") == NULL;
    if (small_group) {
        qwen4exp_moe_group_small_kernel<<<
                1, QWEN4EXP_MOE_SCAN_THREADS, 0, stream>>>(
                sc.counts, sc.offsets, sc.cursor, sc.active, sc.pairs,
                (float *)mid->ptr, (const int32_t *)selected->ptr,
                n_total_expert, n_pairs, n_expert_used, mid_dim,
                mid_token_stride);
    } else {
        if (!cuda_ok(cudaMemsetAsync(sc.counts, 0,
                                     (size_t)n_total_expert * sizeof(int32_t),
                                     stream),
                     "qwen4exp MoE group counts reset")) {
            return 0;
        }
        qwen4exp_moe_group_count_kernel<<<pair_blocks, threads, 0, stream>>>(
                sc.counts, (const int32_t *)selected->ptr,
                n_total_expert, n_pairs);
        if (n_total_expert <= QWEN4EXP_MOE_SCAN_THREADS &&
            getenv("DS4_QWEN4EXP_SERIAL_GROUP_SCAN") == NULL) {
            qwen4exp_moe_group_scan_parallel_kernel<<<
                    1, QWEN4EXP_MOE_SCAN_THREADS, 0, stream>>>(
                    sc.offsets, sc.cursor, sc.active, sc.counts,
                    n_total_expert);
        } else {
            qwen4exp_moe_group_scan_kernel<<<1, 32, 0, stream>>>(
                    sc.offsets, sc.cursor, sc.active, sc.counts,
                    n_total_expert);
        }
        qwen4exp_moe_group_scatter_kernel<<<pair_blocks, threads, 0, stream>>>(
                sc.pairs, sc.cursor, (const int32_t *)selected->ptr,
                n_total_expert, n_pairs);
    }
    if (!small_group) {
        qwen4exp_moe_zero_invalid_kernel<<<n_pairs, threads, 0, stream>>>(
                (float *)mid->ptr, (const int32_t *)selected->ptr,
                n_total_expert, n_expert_used, mid_dim, mid_token_stride,
                n_pairs);
    }
    if (!cuda_ok(cudaGetLastError(), "qwen4exp MoE pair list")) return 0;

    if (!qwen4exp_quantize_rows(sc.xq, sc.xs, sc.xsum, (const float *)x->ptr,
                                n_tokens, in_dim, xgroups, in_dim, 0, 1,
                                stream)) {
        return 0;
    }

    /* The tensor-core tile takes the gate and up projections when the shapes
     * divide it and neither type is Q6_K, whose scale changes inside a group.
     * DS4_QWEN4EXP_NO_MMA keeps the dp4a kernel for the comparison. */
    /* WIDTH DISPATCH.  The speculative cycle only ever runs one row (decode)
     * and two to four (verify at depths one to three), and those widths carry
     * the exactness requirement: a batched verify has to equal a serial
     * decode.  A prefill is identical across depths by construction, so it is
     * free to use different numerics as long as they are deterministic.
     *
     * So the tile takes width eight and above and the pre-existing per-row
     * kernels keep everything below it, unchanged.  The tile at one row would
     * pad thirty-one of its thirty-two token rows and cost a quarter of the
     * decode rate; this is what that buys back. */
    const int use_mma =
        n_tokens >= 8u &&
        (mid_dim % QW_MMA_BM) == 0 && (xgroups % QW_MMA_G) == 0 &&
        gate_slab->type != (uint32_t)DS4_QWEN4EXP_TY_q6_K &&
        up_slab->type != (uint32_t)DS4_QWEN4EXP_TY_q6_K &&
        getenv("DS4_QWEN4EXP_NO_MMA") == NULL;

    const int tile = qwen4exp_moe_tile(n_tokens);
    /* One block row per expert the call CHOSE, not per expert that exists.
     * n_pairs bounds the number of distinct experts, and the kernel exits the
     * rows past active[0]. */
    const int compact = getenv("DS4_QWEN4EXP_NO_EXPERT_COMPACT") == NULL;
    const uint32_t gu_rows = !compact ? n_total_expert
        : (n_pairs < n_total_expert ? n_pairs : n_total_expert);
    const int32_t *gu_active = compact ? sc.active : NULL;
    const dim3 gu_grid((mid_dim + 7u) / 8u, gu_rows, 1);
#define QWEN4EXP_GATEUP_IMPL(R, GT, UT) \
    qwen4exp_moe_gateup_q_kernel<R, GT, UT><<<gu_grid, threads, 0, stream>>>( \
            (float *)mid->ptr, gate, up, sc.xq, sc.xs, sc.xsum, \
            sc.pairs, sc.counts, sc.offsets, gu_active, \
            (const float *)weights->ptr, \
            gate_slab->expert_bytes, gate_slab->row_bytes, \
            up_slab->expert_bytes, up_slab->row_bytes, \
            gate_slab->type, up_slab->type, xgroups, mid_dim, \
            mid_token_stride, n_expert_used)
    /* Resolve the format once on the host, where tensor metadata already
     * lives.  This exposes fixed nibble decoding and a fixed one-half
     * accumulation to nvcc, without converting or copying any weight. */
    const bool specialize = getenv("DS4_QWEN4EXP_GENERIC_EXPERTS") == NULL;
#define QWEN4EXP_GATEUP(R) do { \
    if (specialize && gate_slab->type == DS4_QWEN4EXP_TY_q4_K && \
                      up_slab->type == DS4_QWEN4EXP_TY_q4_K) { \
        QWEN4EXP_GATEUP_IMPL(R, DS4_QWEN4EXP_TY_q4_K, DS4_QWEN4EXP_TY_q4_K); \
    } else if (specialize && gate_slab->type == DS4_QWEN4EXP_TY_q8_0 && \
                             up_slab->type == DS4_QWEN4EXP_TY_q8_0) { \
        QWEN4EXP_GATEUP_IMPL(R, DS4_QWEN4EXP_TY_q8_0, DS4_QWEN4EXP_TY_q8_0); \
    } else { \
        QWEN4EXP_GATEUP_IMPL(R, -1, -1); \
    } \
} while (0)
    if (use_mma) {
#define QWEN4EXP_GATEUP_MMA(GT, UT) \
        qwen4exp_moe_gateup_mma_kernel<GT, UT><<< \
                dim3(mid_dim / QW_MMA_BM, gu_rows, 1), \
                QW_MMA_THREADS, 0, stream>>>( \
                (float *)mid->ptr, gate, up, sc.xq, sc.xs, sc.xsum, \
                sc.pairs, sc.counts, sc.offsets, gu_active, \
                (const float *)weights->ptr, \
                gate_slab->expert_bytes, gate_slab->row_bytes, \
                up_slab->expert_bytes, up_slab->row_bytes, \
                gate_slab->type, up_slab->type, xgroups, mid_dim, \
                mid_token_stride, n_expert_used)
        if (specialize && gate_slab->type == DS4_QWEN4EXP_TY_q4_K &&
                          up_slab->type == DS4_QWEN4EXP_TY_q4_K) {
            QWEN4EXP_GATEUP_MMA(DS4_QWEN4EXP_TY_q4_K, DS4_QWEN4EXP_TY_q4_K);
        } else if (specialize && gate_slab->type == DS4_QWEN4EXP_TY_q8_0 &&
                                 up_slab->type == DS4_QWEN4EXP_TY_q8_0) {
            QWEN4EXP_GATEUP_MMA(DS4_QWEN4EXP_TY_q8_0, DS4_QWEN4EXP_TY_q8_0);
        } else {
            QWEN4EXP_GATEUP_MMA(-1, -1);
        }
#undef QWEN4EXP_GATEUP_MMA
    }
    else if (tile == 8) { QWEN4EXP_GATEUP(8); }
    else if (tile == 4) { QWEN4EXP_GATEUP(4); }
    else { QWEN4EXP_GATEUP(1); }
#undef QWEN4EXP_GATEUP
#undef QWEN4EXP_GATEUP_IMPL
    if (!cuda_ok(cudaGetLastError(), "qwen4exp MoE gate/up launch")) return 0;

    if (!qwen4exp_quantize_rows(sc.mq, sc.ms, sc.msum, (const float *)mid->ptr,
                                n_pairs, mid_dim, mgroups, mid_token_stride,
                                mid_dim, n_expert_used, stream)) {
        return 0;
    }

    const dim3 dn_grid((out_dim + 7u) / 8u,
                       (n_tokens + (uint32_t)tile - 1u) / (uint32_t)tile, 1);
#define QWEN4EXP_DOWN_IMPL(R, DT) \
    qwen4exp_moe_down_q_kernel<R, DT><<<dn_grid, threads, 0, stream>>>( \
            (float *)out->ptr, down, (const int32_t *)selected->ptr, \
            sc.mq, sc.ms, sc.msum, \
            down_slab->expert_bytes, down_slab->row_bytes, down_slab->type, \
            mgroups, out_dim, n_tokens, n_total_expert, n_expert_used)
#define QWEN4EXP_DOWN(R) do { \
    if (specialize && down_slab->type == DS4_QWEN4EXP_TY_q5_1) { \
        QWEN4EXP_DOWN_IMPL(R, DS4_QWEN4EXP_TY_q5_1); \
    } else if (specialize && down_slab->type == DS4_QWEN4EXP_TY_q8_0) { \
        QWEN4EXP_DOWN_IMPL(R, DS4_QWEN4EXP_TY_q8_0); \
    } else { \
        QWEN4EXP_DOWN_IMPL(R, -1); \
    } \
} while (0)
    const int down_mma = use_mma && (out_dim % QW_MMA_BM) == 0 &&
                         down_slab->type != (uint32_t)DS4_QWEN4EXP_TY_q6_K;
    if (down_mma) {
#define QWEN4EXP_DOWN_MMA(DT) \
        qwen4exp_moe_down_mma_kernel<DT><<< \
                dim3(out_dim / QW_MMA_BM, gu_rows, 1), \
                QW_MMA_THREADS, 0, stream>>>( \
                (float *)down_partial->ptr, down, sc.mq, sc.ms, sc.msum, \
                sc.pairs, sc.counts, sc.offsets, gu_active, \
                down_slab->expert_bytes, down_slab->row_bytes, down_slab->type, \
                mgroups, out_dim)
        if (specialize && down_slab->type == DS4_QWEN4EXP_TY_q5_1) {
            QWEN4EXP_DOWN_MMA(DS4_QWEN4EXP_TY_q5_1);
        } else if (specialize && down_slab->type == DS4_QWEN4EXP_TY_q8_0) {
            QWEN4EXP_DOWN_MMA(DS4_QWEN4EXP_TY_q8_0);
        } else {
            QWEN4EXP_DOWN_MMA(-1);
        }
#undef QWEN4EXP_DOWN_MMA
        if (!cuda_ok(cudaGetLastError(), "qwen4exp MoE down tile launch")) return 0;
        const uint64_t combine_n = (uint64_t)n_tokens * out_dim;
        qwen4exp_moe_down_combine_kernel<<<
                (unsigned)((combine_n + threads - 1u) / threads), threads, 0,
                stream>>>(
                (float *)out->ptr, (const float *)down_partial->ptr,
                (const int32_t *)selected->ptr, out_dim, n_tokens,
                n_expert_used, n_total_expert);
        return cuda_ok(cudaGetLastError(), "qwen4exp MoE down combine launch");
    }
    if (tile == 8) { QWEN4EXP_DOWN(8); }
    else if (tile == 4) { QWEN4EXP_DOWN(4); }
    else { QWEN4EXP_DOWN(1); }
#undef QWEN4EXP_DOWN
#undef QWEN4EXP_DOWN_IMPL
    return cuda_ok(cudaGetLastError(), "qwen4exp MoE down launch");
}

extern "C" int ds4_gpu_qwen4exp_shared_expert_tensor(
        ds4_gpu_tensor              *out,
        ds4_gpu_tensor              *mid,
        ds4_gpu_tensor              *gate_scale,
        const ds4_gpu_qwen4exp_slab *router_slab,
        const ds4_gpu_qwen4exp_slab *gate_slab,
        const ds4_gpu_qwen4exp_slab *up_slab,
        const ds4_gpu_qwen4exp_slab *down_slab,
        uint32_t                     in_dim,
        uint32_t                     mid_dim,
        uint32_t                     out_dim,
        const ds4_gpu_tensor        *x,
        uint32_t                     n_tokens) {
    if (!out || !mid || !gate_scale || !x ||
        !router_slab || !gate_slab || !up_slab || !down_slab ||
        !router_slab->map || !gate_slab->map || !up_slab->map || !down_slab->map ||
        in_dim == 0 || mid_dim == 0 || out_dim == 0 || n_tokens == 0) {
        return 0;
    }
    if (!cuda_qwen4exp_moe_type_supported(gate_slab->type) ||
        !cuda_qwen4exp_moe_type_supported(up_slab->type) ||
        !cuda_qwen4exp_moe_type_supported(down_slab->type) ||
        !cuda_qwen4exp_moe_type_supported(router_slab->type)) {
        fprintf(stderr, "ds4: CUDA qwen4exp shared expert unsupported types\n");
        return 0;
    }
    if ((in_dim & 31u) != 0 || (mid_dim & 31u) != 0) {
        fprintf(stderr, "ds4: CUDA qwen4exp shared expert needs in_dim and "
                        "mid_dim in whole groups of 32 (got %u and %u)\n",
                in_dim, mid_dim);
        return 0;
    }
    if (out->bytes < (uint64_t)n_tokens * out_dim * sizeof(float) ||
        mid->bytes < (uint64_t)n_tokens * mid_dim * sizeof(float) ||
        gate_scale->bytes < (uint64_t)n_tokens * sizeof(float) ||
        x->bytes < (uint64_t)n_tokens * in_dim * sizeof(float)) {
        fprintf(stderr, "ds4: CUDA qwen4exp shared expert received undersized buffers\n");
        return 0;
    }

    const int logical_tier = cuda_current_tier();
    const char *router = cuda_resolve_weight_ptr(router_slab->map,
            router_slab->offset, (uint64_t)in_dim * sizeof(float), logical_tier,
            "qwen4exp_ffn_gate_inp_shexp");
    const char *gate = cuda_resolve_weight_ptr(gate_slab->map, gate_slab->offset,
            (uint64_t)mid_dim * gate_slab->row_bytes, logical_tier,
            "qwen4exp_ffn_gate_shexp");
    const char *up = cuda_resolve_weight_ptr(up_slab->map, up_slab->offset,
            (uint64_t)mid_dim * up_slab->row_bytes, logical_tier,
            "qwen4exp_ffn_up_shexp");
    const char *down = cuda_resolve_weight_ptr(down_slab->map, down_slab->offset,
            (uint64_t)out_dim * down_slab->row_bytes, logical_tier,
            "qwen4exp_ffn_down_shexp");
    if (!router || !gate || !up || !down) return 0;

    cudaStream_t stream = cuda_decode_stream();
    const unsigned threads = 256u;
    const size_t shared = (size_t)threads * sizeof(float);

    /* The sigmoid gate is one dot against ONE F32 row per token.  It reads
     * kilobytes, not megabytes, so it keeps the scalar reduction. */
    qwen4exp_shared_gate_kernel<<<n_tokens, threads, shared, stream>>>(
            (float *)gate_scale->ptr, router, (const float *)x->ptr,
            router_slab->type, in_dim, n_tokens);
    if (!cuda_ok(cudaGetLastError(), "qwen4exp shared gate launch")) return 0;

    const uint32_t xgroups = in_dim / 32u;
    const uint32_t mgroups = mid_dim / 32u;
    char *base = (char *)qwen4exp_group_scratch(
            logical_tier,
            qwen4exp_quant_bytes(n_tokens, xgroups) +
            qwen4exp_quant_bytes(n_tokens, mgroups));
    if (!base) return 0;
    int8_t *xq = (int8_t *)base;
    float *xs = (float *)(base + (uint64_t)n_tokens * xgroups * 32u);
    int32_t *xsum = (int32_t *)(xs + (uint64_t)n_tokens * xgroups);
    char *at = base + qwen4exp_quant_bytes(n_tokens, xgroups);
    int8_t *mq = (int8_t *)at;
    float *ms = (float *)(at + (uint64_t)n_tokens * mgroups * 32u);
    int32_t *msum = (int32_t *)(ms + (uint64_t)n_tokens * mgroups);

    if (!qwen4exp_quantize_rows(xq, xs, xsum, (const float *)x->ptr,
                                n_tokens, in_dim, xgroups, in_dim, 0, 1,
                                stream)) {
        return 0;
    }

    const int tile = qwen4exp_moe_tile(n_tokens);
    const uint32_t tiles = (n_tokens + (uint32_t)tile - 1u) / (uint32_t)tile;

    /* Tokens one staged block serves, and the tiles that many needs. */
    const uint32_t stage_span =
        (uint32_t)QWEN4EXP_STAGE_R * (uint32_t)QWEN4EXP_STAGE_WARPS;
    const uint32_t stage_tiles = (n_tokens + stage_span - 1u) / stage_span;
    const int stage_gateup = qwen4exp_shared_stage_ok(xq, xgroups, 2, n_tokens);
    const int stage_down = qwen4exp_shared_stage_ok(mq, mgroups, 1, n_tokens);

    /* The m16n8k32 pair, taken ahead of both paths above when the whole shared
     * expert is Q8_0 and the call is a prefill.  It returns the per-row
     * kernels' bits and is the faster of the three there (GB10, prefill 528 ->
     * 638 tok/s); below 64 tokens it never runs, so a decode and a verify
     * dispatch exactly as they did before it existed.  It declines to the same
     * two pins the staged path declines to, so a test that fixes the shared
     * expert on one path still gets that path -- DS4_QWEN4EXP_SHARED_STAGE=0
     * is the per-row oracle for all three, and =1 the staged path at every
     * width.  Only the shared expert is affected; routed MMA is untouched. */
    const int use_mma = n_tokens >= 64u &&
        gate_slab->type == (uint32_t)DS4_QWEN4EXP_TY_q8_0 &&
        up_slab->type == (uint32_t)DS4_QWEN4EXP_TY_q8_0 &&
        down_slab->type == (uint32_t)DS4_QWEN4EXP_TY_q8_0 &&
        getenv("DS4_QWEN4EXP_SHARED_STAGE") == NULL &&
        getenv("DS4_QWEN4EXP_MOE_R") == NULL;
    const uint32_t mma_tiles = (n_tokens + QW_SH_BN - 1u) / QW_SH_BN;

    if (use_mma) {
        qwen4exp_shared_q8_mma_kernel<true><<<
                dim3((mid_dim + QW_SH_BM - 1u) / QW_SH_BM, mma_tiles, 1),
                QW_SH_THREADS, 0, stream>>>(
                (float *)mid->ptr, gate, up, xq, xs, xsum, NULL,
                gate_slab->row_bytes, up_slab->row_bytes,
                xgroups, mid_dim, n_tokens, 0.0f);
    } else if (stage_gateup) {
        qwen4exp_shared_gateup_stage_kernel<QWEN4EXP_STAGE_R>
            <<<dim3(mid_dim, stage_tiles, 1), QWEN4EXP_STAGE_THREADS,
               (size_t)qwen4exp_stage_bytes(xgroups, 2), stream>>>(
                (float *)mid->ptr, gate, up, xq, xs, xsum,
                gate_slab->row_bytes, up_slab->row_bytes,
                gate_slab->type, up_slab->type, xgroups, mid_dim, n_tokens);
    } else {
#define QWEN4EXP_SH_GATEUP(R) \
    qwen4exp_shared_gateup_q_kernel<R> \
        <<<dim3((mid_dim + 7u) / 8u, tiles, 1), threads, 0, stream>>>( \
            (float *)mid->ptr, gate, up, xq, xs, xsum, \
            gate_slab->row_bytes, up_slab->row_bytes, \
            gate_slab->type, up_slab->type, xgroups, mid_dim, n_tokens)
    if (tile == 8) { QWEN4EXP_SH_GATEUP(8); }
    else if (tile == 4) { QWEN4EXP_SH_GATEUP(4); }
    else { QWEN4EXP_SH_GATEUP(1); }
#undef QWEN4EXP_SH_GATEUP
    }
    if (!cuda_ok(cudaGetLastError(), "qwen4exp shared gate/up launch")) return 0;

    if (!qwen4exp_quantize_rows(mq, ms, msum, (const float *)mid->ptr,
                                n_tokens, mid_dim, mgroups, mid_dim, 0, 1,
                                stream)) {
        return 0;
    }

    if (use_mma) {
        qwen4exp_shared_q8_mma_kernel<false><<<
                dim3((out_dim + QW_SH_BM - 1u) / QW_SH_BM, mma_tiles, 1),
                QW_SH_THREADS, 0, stream>>>(
                (float *)out->ptr, down, NULL, mq, ms, msum,
                (const float *)gate_scale->ptr, down_slab->row_bytes, 0,
                mgroups, out_dim, n_tokens, 0.0f);
    } else if (stage_down) {
        qwen4exp_shared_down_stage_kernel<QWEN4EXP_STAGE_R>
            <<<dim3(out_dim, stage_tiles, 1), QWEN4EXP_STAGE_THREADS,
               (size_t)qwen4exp_stage_bytes(mgroups, 1), stream>>>(
                (float *)out->ptr, down, mq, ms, msum,
                (const float *)gate_scale->ptr, down_slab->row_bytes,
                down_slab->type, mgroups, out_dim, n_tokens);
    } else {
#define QWEN4EXP_SH_DOWN(R) \
    qwen4exp_shared_down_q_kernel<R> \
        <<<dim3((out_dim + 7u) / 8u, tiles, 1), threads, 0, stream>>>( \
            (float *)out->ptr, down, mq, ms, msum, \
            (const float *)gate_scale->ptr, down_slab->row_bytes, \
            down_slab->type, mgroups, out_dim, n_tokens)
    if (tile == 8) { QWEN4EXP_SH_DOWN(8); }
    else if (tile == 4) { QWEN4EXP_SH_DOWN(4); }
    else { QWEN4EXP_SH_DOWN(1); }
#undef QWEN4EXP_SH_DOWN
    }
    return cuda_ok(cudaGetLastError(), "qwen4exp shared down launch");
}

#include "ds4_qwen4exp_hc_ref.h"

/* =========================================================================
 * Qwen4exp hyper-connections, norms, rope, embedding.
 * =========================================================================
 *
 * CUDA twin of metal/qwen4exp_hc.metal.  Same activation layout as the
 * dsv4_hc path ([token][hc][embd], embd contiguous) and the same summation
 * order as ds4_qwen4exp_hc_ref.h: hyper-connection streams low to high.
 * The reduction shape follows rms_norm_weight_kernel above, so the norm and
 * the inject dot reduce the way every other row reduction in this file does.
 */

/* Round to nearest even bf16, widened back to f32: MLX's fused rmsNorm casts
 * the normalized value to the activation dtype before the weight multiply. */
__device__ __forceinline__ static float qwen4exp_round_bf16(float v) {
    const uint32_t bits = __float_as_uint(v);
    const uint32_t rounding = 0x7fffu + ((bits >> 16) & 1u);
    return __uint_as_float((bits + rounding) & 0xffff0000u);
}

__device__ __forceinline__ static float qwen4exp_sigmoid(float z) {
    return 1.0f / (1.0f + expf(-z));
}

/* The stride-halving tree the row reductions below have always used, with its
 * last five levels -- the ones whose pairs both sit in warp 0 -- taken in
 * registers instead of through shared memory.  Same pairs in the same order,
 * so the sum is the one the all-shared tree produced bit for bit; five shared
 * round trips and four barriers less to reach it.  Wants a power-of-two
 * blockDim.x of at least a warp, which is what every launch here uses.
 * `partial` must hold blockDim.x floats. */
__device__ __forceinline__ static float qwen4exp_block_sum_f32(
        float v, float *partial) {
    const uint32_t tid = threadIdx.x;
    partial[tid] = v;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride >= 32u; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    if (tid < 32u) {
        const float total = warp_sum_all_f32(partial[tid]);
        if (tid == 0u) partial[0] = total;
    }
    __syncthreads();
    return partial[0];
}

__global__ static void qwen4exp_rms_norm_kernel(
        float *out, const float *x, const float *w,
        uint32_t n, uint32_t group, uint32_t rows,
        float eps, float weight_bias, int round_bf16) {
    const uint32_t g = blockIdx.x;
    const uint32_t row = blockIdx.y;
    if (row >= rows) return;

    const uint64_t base = (uint64_t)row * n + (uint64_t)g * group;
    const float *xg = x + base;
    float *yg = out + base;
    const float *wg = w + (uint64_t)g * group;

    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < group; i += blockDim.x) {
        const float v = xg[i];
        sum += v * v;
    }
    __shared__ float partial[256];
    const float total = qwen4exp_block_sum_f32(sum, partial);
    /* 1/sqrt rather than rsqrtf: the exactness the qwen4exp op tests assert
     * needs the correctly rounded reciprocal square root. */
    const float scale = 1.0f / sqrtf(total / (float)group + eps);

    for (uint32_t i = threadIdx.x; i < group; i += blockDim.x) {
        float normed = xg[i] * scale;
        if (round_bf16) normed = qwen4exp_round_bf16(normed);
        yg[i] = normed * (weight_bias + wg[i]);
    }
}

__global__ static void qwen4exp_scale_silu_kernel(
        float *x, uint32_t n, float scale) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float z = x[i] * scale;
    x[i] = z * qwen4exp_sigmoid(z);
}

/* The grid carries the token, so the element index never has to be divided
 * back into one: a flat launch cost every thread a 64-bit division and a
 * modulo, which is a long instruction sequence beside the four multiply-adds
 * it guarded.  Twin of kernel_qwen4exp_hc_mix. */
__global__ static void qwen4exp_hc_mix_kernel(
        float *out, const float *normed, const float *wide,
        uint32_t n_embd, uint32_t n_hc, uint32_t n_tokens) {
    const uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t t = blockIdx.y;
    if (d >= n_embd || t >= n_tokens) return;

    const uint64_t row = ((uint64_t)t * n_hc) * n_embd + d;

    float acc = 0.0f;
    for (uint32_t h = 0; h < n_hc; h++) {
        const uint64_t idx = row + (uint64_t)h * n_embd;
        acc += qwen4exp_sigmoid(wide[idx]) * normed[idx];
    }
    out[(uint64_t)t * n_embd + d] = acc * (1.0f / (float)n_hc);
}

__global__ static void qwen4exp_hc_inject_weights_kernel(
        float *out, const float *normed, const char *w,
        uint32_t n_embd, uint32_t n_hc, uint32_t rows,
        uint32_t weight_type, uint32_t weight_row_bytes) {
    const uint32_t h = blockIdx.x;
    const uint32_t t = blockIdx.y;
    if (t >= rows || h >= n_hc) return;

    const uint32_t wide = n_hc * n_embd;
    const float *xr = normed + (uint64_t)t * wide;
    const char *wr = w + (uint64_t)h * weight_row_bytes;

    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < wide; i += blockDim.x) {
        sum += xr[i] * dev_qwen4exp_inject_value(weight_type, wr, i);
    }
    __shared__ float partial[256];
    const float total = qwen4exp_block_sum_f32(sum, partial);
    if (threadIdx.x == 0) {
        out[(uint64_t)t * n_hc + h] =
            2.0f * qwen4exp_sigmoid(total * (1.0f / (float)n_hc));
    }
}

/* Stream and token off the grid for the same reason as the mixer above: a flat
 * launch spent two 64-bit divisions and two modulos per element on an
 * expression that is one multiply and one add.  Twin of
 * kernel_qwen4exp_hc_inject. */
__global__ static void qwen4exp_hc_inject_kernel(
        float *out, const float *residual, const float *block,
        const float *inject, uint32_t n_embd, uint32_t n_hc,
        uint32_t n_tokens) {
    const uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t h = blockIdx.y;
    const uint32_t t = blockIdx.z;
    if (d >= n_embd || h >= n_hc || t >= n_tokens) return;

    const uint64_t i = ((uint64_t)t * n_hc + h) * n_embd + d;
    out[i] = residual[i] + block[(uint64_t)t * n_embd + d] *
        inject[(uint64_t)t * n_hc + h];
}


extern "C" int ds4_gpu_qwen4exp_rms_norm_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              n,
        uint32_t              group,
        uint32_t              rows,
        float                 eps,
        float                 weight_bias,
        int                   round_bf16) {
    if (!out || !x || !model_map || n == 0 || group == 0 || rows == 0 ||
        n % group != 0 || weight_offset > model_size ||
        model_size - weight_offset < (uint64_t)n * sizeof(float) ||
        out->bytes < (uint64_t)n * rows * sizeof(float) ||
        x->bytes < (uint64_t)n * rows * sizeof(float)) {
        return 0;
    }
    const int logical_tier = ds4_tensor_device_idx(out);
    const float *w = (const float *)cuda_resolve_weight_ptr(
            model_map, weight_offset, (uint64_t)n * sizeof(float),
            logical_tier, "qwen4exp_norm_weight");
    if (!w) return 0;
    dim3 grid(n / group, rows, 1u);
    qwen4exp_rms_norm_kernel<<<grid, 256, 0, cuda_decode_stream()>>>(
            (float *)out->ptr, (const float *)x->ptr, w,
            n, group, rows, eps, weight_bias, round_bf16);
    return cuda_ok(cudaGetLastError(), "qwen4exp_rms_norm launch");
}

extern "C" int ds4_gpu_qwen4exp_scale_silu_tensor(
        ds4_gpu_tensor *x,
        uint32_t        n,
        float           scale) {
    if (!x || n == 0 || x->bytes < (uint64_t)n * sizeof(float)) return 0;
    qwen4exp_scale_silu_kernel<<<(n + 255u) / 256u, 256, 0, cuda_decode_stream()>>>(
            (float *)x->ptr, n, scale);
    return cuda_ok(cudaGetLastError(), "qwen4exp_scale_silu launch");
}

extern "C" int ds4_gpu_qwen4exp_hc_mix_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *normed,
        const ds4_gpu_tensor *wide,
        uint32_t              n_embd,
        uint32_t              n_hc,
        uint32_t              rows) {
    if (!out || !normed || !wide || n_embd == 0 || n_hc == 0 || rows == 0) return 0;
    const uint64_t out_bytes = (uint64_t)rows * n_embd * sizeof(float);
    const uint64_t hc_bytes = out_bytes * n_hc;
    if (out->bytes < out_bytes || normed->bytes < hc_bytes ||
        wide->bytes < hc_bytes) {
        return 0;
    }
    qwen4exp_hc_mix_kernel<<<dim3((n_embd + 255u) / 256u, rows, 1u), 256, 0,
                             cuda_decode_stream()>>>(
            (float *)out->ptr, (const float *)normed->ptr,
            (const float *)wide->ptr, n_embd, n_hc, rows);
    return cuda_ok(cudaGetLastError(), "qwen4exp_hc_mix launch");
}

extern "C" int ds4_gpu_qwen4exp_hc_inject_weights_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *normed,
        const ds4_gpu_qwen4exp_slab *weight,
        uint32_t              n_embd,
        uint32_t              n_hc,
        uint32_t              rows) {
    if (!out || !normed || !weight || !weight->map ||
        n_embd == 0 || n_hc == 0 || rows == 0) {
        return 0;
    }
    const uint64_t wide = (uint64_t)n_hc * n_embd;
    /* The row stride is the slab's, so a quantised weight is addressed by its
     * own block layout rather than by an assumed float row. */
    const uint64_t row_bytes = weight->row_bytes ? weight->row_bytes
                                                 : wide * sizeof(float);
    const uint64_t weight_bytes = (uint64_t)n_hc * row_bytes;
    if (weight->offset > weight->map_size ||
        weight->map_size - weight->offset < weight_bytes ||
        normed->bytes < (uint64_t)rows * wide * sizeof(float) ||
        out->bytes < (uint64_t)rows * n_hc * sizeof(float)) {
        return 0;
    }
    const int logical_tier = ds4_tensor_device_idx(out);
    const char *w = cuda_resolve_weight_ptr(
            weight->map, weight->offset, weight_bytes, logical_tier,
            "qwen4exp_inject_weight");
    if (!w) return 0;
    dim3 grid(n_hc, rows, 1u);
    qwen4exp_hc_inject_weights_kernel<<<grid, 256, 0, cuda_decode_stream()>>>(
            (float *)out->ptr, (const float *)normed->ptr, w,
            n_embd, n_hc, rows, weight->type, (uint32_t)row_bytes);
    return cuda_ok(cudaGetLastError(), "qwen4exp_hc_inject_weights launch");
}

extern "C" int ds4_gpu_qwen4exp_hc_inject_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *block_out,
        const ds4_gpu_tensor *inject,
        uint32_t              n_embd,
        uint32_t              n_hc,
        uint32_t              rows) {
    if (!out_hc || !residual_hc || !block_out || !inject ||
        n_embd == 0 || n_hc == 0 || rows == 0) {
        return 0;
    }
    const uint64_t hc_bytes = (uint64_t)rows * n_hc * n_embd * sizeof(float);
    if (out_hc->bytes < hc_bytes || residual_hc->bytes < hc_bytes ||
        block_out->bytes < (uint64_t)rows * n_embd * sizeof(float) ||
        inject->bytes < (uint64_t)rows * n_hc * sizeof(float)) {
        return 0;
    }
    qwen4exp_hc_inject_kernel<<<dim3((n_embd + 255u) / 256u, n_hc, rows), 256,
                                0, cuda_decode_stream()>>>(
            (float *)out_hc->ptr, (const float *)residual_hc->ptr,
            (const float *)block_out->ptr, (const float *)inject->ptr,
            n_embd, n_hc, rows);
    return cuda_ok(cudaGetLastError(), "qwen4exp_hc_inject launch");
}

/* =========================================================================
 * Qwen4-Exp QSA block, the CUDA twin of metal/qwen4exp_qsa.metal.
 *
 * Kernel for kernel and argument for argument the same as the Metal half,
 * including the finite masked-score sentinel and the inverse-frequency table:
 * both engines have to agree bit for bit on which blocks a query keeps, and
 * both are compiled with fast math.  See ds4_gpu.h for the call order and
 * the Metal shader for why each step is shaped the way it is.
 * =========================================================================
 */

#define QWEN4EXP_QSA_MASKED_SCORE (-3.0e38f)
#define QWEN4EXP_QSA_MASKED_LIMIT (-1.0e30f)

/* Sum sdata[0 .. nth) into sdata[0]; `nth` must be a power of two.
 *
 * The last five steps of the tree pair lanes of the first warp with each
 * other, so they are that warp's own shuffle tree: the same operands added in
 * the same order, without the five block barriers the shared-memory form
 * spends on threads that have already finished.  A block narrower than a warp
 * keeps the shared-memory form, which has no lane mask to get wrong.  The
 * maximum below takes the same shape. */
__device__ __forceinline__ static float qwen4exp_blk_sum(
        float *sdata, uint32_t tid, uint32_t nth) {
    if (nth < 32u) {
        for (uint32_t step = nth >> 1; step > 0u; step >>= 1) {
            __syncthreads();
            if (tid < step) sdata[tid] += sdata[tid + step];
        }
        __syncthreads();
        return sdata[0];
    }
    for (uint32_t step = nth >> 1; step >= 32u; step >>= 1) {
        __syncthreads();
        if (tid < step) sdata[tid] += sdata[tid + step];
    }
    __syncthreads();
    if (tid < 32u) {
        float v = sdata[tid];
#pragma unroll
        for (uint32_t step = 16u; step > 0u; step >>= 1) {
            v += __shfl_down_sync(0xffffffffu, v, step);
        }
        if (tid == 0u) sdata[0] = v;
    }
    __syncthreads();
    return sdata[0];
}

__device__ __forceinline__ static float qwen4exp_blk_max(
        float *sdata, uint32_t tid, uint32_t nth) {
    if (nth < 32u) {
        for (uint32_t step = nth >> 1; step > 0u; step >>= 1) {
            __syncthreads();
            if (tid < step) sdata[tid] = fmaxf(sdata[tid], sdata[tid + step]);
        }
        __syncthreads();
        return sdata[0];
    }
    for (uint32_t step = nth >> 1; step >= 32u; step >>= 1) {
        __syncthreads();
        if (tid < step) sdata[tid] = fmaxf(sdata[tid], sdata[tid + step]);
    }
    __syncthreads();
    if (tid < 32u) {
        float v = sdata[tid];
#pragma unroll
        for (uint32_t step = 16u; step > 0u; step >>= 1) {
            v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, step));
        }
        if (tid == 0u) sdata[0] = v;
    }
    __syncthreads();
    return sdata[0];
}

__device__ __forceinline__ static void qwen4exp_rope_head_vec(
        float *vec,
        const float *inv_freq,
        uint32_t tid,
        uint32_t nth,
        uint32_t rot_dim,
        uint32_t pos) {
    const uint32_t rot_half = rot_dim / 2u;
    for (uint32_t d = tid; d < rot_half; d += nth) {
        const float theta = (float)pos * inv_freq[d];
        const float c = cosf(theta);
        const float s = sinf(theta);
        const float x1 = vec[d];
        const float x2 = vec[d + rot_half];
        vec[d] = x1 * c - x2 * s;
        vec[d + rot_half] = x2 * c + x1 * s;
    }
}

__global__ static void qwen4exp_qsa_split_qkv_kernel(
        const float *fused,
        float *q,
        float *gate,
        float *k,
        float *v,
        uint32_t n_tokens,
        uint32_t n_head,
        uint32_t n_kv_head,
        uint32_t head_dim) {
    const uint32_t q_width = n_head * head_dim;
    const uint32_t kv_width = n_kv_head * head_dim;
    const uint32_t width = q_width + 2u * kv_width;
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (uint64_t)n_tokens * width) return;

    const uint32_t token = (uint32_t)(gid / width);
    const uint32_t lane = (uint32_t)(gid - (uint64_t)token * width);
    const uint32_t fused_stride = 2u * q_width + 2u * kv_width;
    const float *row = fused + (uint64_t)token * fused_stride;

    if (lane < q_width) {
        const uint32_t head = lane / head_dim;
        const uint32_t d = lane % head_dim;
        const uint32_t base = head * 2u * head_dim;
        q[(uint64_t)token * q_width + lane] = row[base + d];
        gate[(uint64_t)token * q_width + lane] = row[base + head_dim + d];
        return;
    }

    const uint32_t kv_lane = lane - q_width;
    if (kv_lane < kv_width) {
        k[(uint64_t)token * kv_width + kv_lane] = row[2u * q_width + kv_lane];
    } else {
        const uint32_t vl = kv_lane - kv_width;
        v[(uint64_t)token * kv_width + vl] = row[2u * q_width + kv_width + vl];
    }
}

/* Split the DOUBLED query projection when q, k and v are separate tensors.
 * The loader binds attn_q / attn_k / attn_v apart, so k and v already land in
 * the layout the attention kernel wants and only the query, whose row is
 * head-major with the gate interleaved, needs splitting. */
__global__ static void qwen4exp_qsa_split_doubled_q_kernel(
        const float *doubled, float *q, float *gate,
        uint32_t n_tokens, uint32_t n_head, uint32_t head_dim) {
    const uint32_t q_width = n_head * head_dim;
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (uint64_t)n_tokens * q_width) return;

    const uint32_t token = (uint32_t)(gid / q_width);
    const uint32_t lane = (uint32_t)(gid % q_width);
    const uint32_t head = lane / head_dim;
    const uint32_t d = lane % head_dim;

    const float *row = doubled + (uint64_t)token * 2u * q_width;
    const uint32_t base = head * 2u * head_dim;
    q[(uint64_t)token * q_width + lane] = row[base + d];
    gate[(uint64_t)token * q_width + lane] = row[base + head_dim + d];
}

__global__ static void qwen4exp_head_rms_norm_kernel(
        const float *x,
        const float *weight,
        float *out,
        uint32_t n_rows,
        uint32_t head_dim,
        float eps,
        float weight_offset) {
    extern __shared__ float qwen4exp_norm_shared[];
    const uint32_t row = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t nth = blockDim.x;
    if (row >= n_rows) return;

    const float *src = x + (uint64_t)row * head_dim;
    float *dst = out + (uint64_t)row * head_dim;
    float partial = 0.0f;
    for (uint32_t d = tid; d < head_dim; d += nth) {
        const float value = src[d];
        partial += value * value;
    }
    qwen4exp_norm_shared[tid] = partial;
    const float sum = qwen4exp_blk_sum(qwen4exp_norm_shared, tid, nth);
    const float inv = rsqrtf(sum / (float)head_dim + eps);
    for (uint32_t d = tid; d < head_dim; d += nth) {
        dst[d] = src[d] * inv * (weight_offset + weight[d]);
    }
}

__global__ static void qwen4exp_rope_head_kernel(
        float *x,
        const float *inv_freq,
        uint32_t n_tokens,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t rot_dim,
        uint32_t pos0) {
    const uint32_t rot_half = rot_dim / 2u;
    const uint32_t per_token = n_head * rot_half;
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (uint64_t)n_tokens * per_token) return;

    const uint32_t token = (uint32_t)(gid / per_token);
    const uint32_t lane = (uint32_t)(gid - (uint64_t)token * per_token);
    const uint32_t head = lane / rot_half;
    const uint32_t d = lane % rot_half;

    float *vec = x + ((uint64_t)token * n_head + head) * head_dim;
    const float theta = (float)(pos0 + token) * inv_freq[d];
    const float c = cosf(theta);
    const float s = sinf(theta);
    const float x1 = vec[d];
    const float x2 = vec[d + rot_half];
    vec[d] = x1 * c - x2 * s;
    vec[d + rot_half] = x2 * c + x1 * s;
}

__global__ static void qwen4exp_qsa_tape_append_kernel(
        const float *raw_k,
        float *tape,
        uint32_t n_tokens,
        uint32_t head_dim,
        uint32_t pos0,
        uint32_t cache_cap) {
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (uint64_t)n_tokens * head_dim) return;
    const uint32_t token = (uint32_t)(gid / head_dim);
    const uint32_t d = (uint32_t)(gid - (uint64_t)token * head_dim);
    const uint32_t pos = pos0 + token;
    if (pos >= cache_cap) return;
    tape[(uint64_t)pos * head_dim + d] = raw_k[gid];
}

__global__ static void qwen4exp_qsa_pool_update_kernel(
        const float *tape,
        const float *weight,
        const float *inv_freq,
        float *pool,
        uint32_t block0,
        uint32_t n_blocks,
        uint32_t head_dim,
        uint32_t pool_size,
        uint32_t rot_dim,
        uint32_t cache_cap,
        float eps,
        float weight_offset) {
    extern __shared__ float qwen4exp_pool_shared[];
    const uint32_t slot = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t nth = blockDim.x;
    if (slot >= n_blocks) return;
    const uint32_t block = block0 + slot;
    if ((block + 1u) * pool_size > cache_cap) return;

    float *vec = qwen4exp_pool_shared;
    float *scratch = qwen4exp_pool_shared + head_dim;

    for (uint32_t d = tid; d < head_dim; d += nth) {
        float acc = 0.0f;
        for (uint32_t j = 0; j < pool_size; j++) {
            acc += tape[(uint64_t)(block * pool_size + j) * head_dim + d];
        }
        vec[d] = acc / (float)pool_size;
    }
    __syncthreads();

    float partial = 0.0f;
    for (uint32_t d = tid; d < head_dim; d += nth) partial += vec[d] * vec[d];
    scratch[tid] = partial;
    const float sum = qwen4exp_blk_sum(scratch, tid, nth);
    const float inv = rsqrtf(sum / (float)head_dim + eps);

    for (uint32_t d = tid; d < head_dim; d += nth) {
        vec[d] = vec[d] * inv * (weight_offset + weight[d]);
    }
    __syncthreads();

    qwen4exp_rope_head_vec(vec, inv_freq, tid, nth, rot_dim, block * pool_size);
    __syncthreads();

    float *dst = pool + (uint64_t)block * head_dim;
    for (uint32_t d = tid; d < head_dim; d += nth) dst[d] = vec[d];
}

__global__ static void qwen4exp_qsa_indexer_scores_kernel(
        const float *q,
        const float *pool,
        float *scores,
        uint32_t n_tokens,
        uint32_t n_blocks,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t pos0,
        uint32_t pool_size,
        float norm_divisor) {
    extern __shared__ float qwen4exp_score_shared[];
    const uint32_t block = blockIdx.x;
    const uint32_t token = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    const uint32_t nth = blockDim.x;
    if (block >= n_blocks || token >= n_tokens) return;

    float *dst = scores + (uint64_t)token * n_blocks + block;
    uint32_t visible = (pos0 + token + 1u) / pool_size;
    if (visible > n_blocks) visible = n_blocks;
    if (block >= visible) {
        if (tid == 0u) *dst = QWEN4EXP_QSA_MASKED_SCORE;
        return;
    }

    const float *k = pool + (uint64_t)block * head_dim;
    float total = 0.0f;
    for (uint32_t h = 0; h < n_head; h++) {
        const float *qh = q + ((uint64_t)token * n_head + h) * head_dim;
        float partial = 0.0f;
        for (uint32_t d = tid; d < head_dim; d += nth) partial += qh[d] * k[d];
        qwen4exp_score_shared[tid] = partial;
        const float dot = qwen4exp_blk_sum(qwen4exp_score_shared, tid, nth);
        total += fmaxf(dot, 0.0f);
    }
    if (tid == 0u) *dst = total / norm_divisor;
}

__global__ static void qwen4exp_qsa_indexer_select_kernel(
        const float *scores,
        const int32_t *topk,
        int32_t *selected,
        int32_t *counts,
        uint32_t n_tokens,
        uint32_t n_blocks,
        uint32_t top_k,
        uint32_t sort_width,
        uint32_t pos0,
        uint32_t pool_size,
        uint32_t max_selected) {
    extern __shared__ int32_t qwen4exp_select_shared[];
    __shared__ uint32_t n_valid;
    const uint32_t token = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t nth = blockDim.x;
    if (token >= n_tokens) return;

    int32_t *ids = qwen4exp_select_shared;
    const int32_t sentinel = 0x7fffffff;
    for (uint32_t i = tid; i < sort_width; i += nth) {
        int32_t block = sentinel;
        if (i < top_k) {
            const int32_t candidate = topk[(uint64_t)token * top_k + i];
            if (candidate >= 0 && (uint32_t)candidate < n_blocks) {
                const float score =
                    scores[(uint64_t)token * n_blocks + (uint32_t)candidate];
                if (score > QWEN4EXP_QSA_MASKED_LIMIT) block = candidate;
            }
        }
        ids[i] = block;
    }
    __syncthreads();

    for (uint32_t k = 2u; k <= sort_width; k <<= 1) {
        for (uint32_t j = k >> 1; j > 0u; j >>= 1) {
            for (uint32_t i = tid; i < sort_width; i += nth) {
                const uint32_t ixj = i ^ j;
                if (ixj > i) {
                    const bool ascending = (i & k) == 0u;
                    if ((ascending && ids[i] > ids[ixj]) ||
                        (!ascending && ids[i] < ids[ixj])) {
                        const int32_t tmp = ids[i];
                        ids[i] = ids[ixj];
                        ids[ixj] = tmp;
                    }
                }
            }
            __syncthreads();
        }
    }

    if (tid == 0u) {
        uint32_t valid = 0;
        while (valid < top_k && ids[valid] != sentinel) valid++;
        n_valid = valid;
    }
    __syncthreads();

    const uint32_t m = n_valid;
    const uint32_t block_tokens = m * pool_size;
    const uint32_t pos = pos0 + token;
    const uint32_t complete = (pos + 1u) / pool_size;
    const uint32_t own_start = complete * pool_size;
    const uint32_t own_count = pos + 1u - own_start;
    const uint32_t total = block_tokens + own_count;

    int32_t *dst = selected + (uint64_t)token * max_selected;
    for (uint32_t i = tid; i < max_selected; i += nth) {
        if (i < block_tokens) {
            dst[i] = ids[i / pool_size] * (int32_t)pool_size +
                     (int32_t)(i % pool_size);
        } else if (i < total) {
            dst[i] = (int32_t)(own_start + (i - block_tokens));
        } else {
            dst[i] = -1;
        }
    }
    if (tid == 0u) counts[token] = (int32_t)total;
}

__global__ static void qwen4exp_qsa_attention_kernel(
        const float *q,
        const float *k_cache,
        const float *v_cache,
        const int32_t *selected,
        const int32_t *counts,
        float *out,
        uint32_t n_tokens,
        uint32_t n_head,
        uint32_t n_kv_head,
        uint32_t head_dim,
        uint32_t pos0,
        uint32_t cache_cap,
        uint32_t max_selected,
        uint32_t sparse,
        float scale) {
    extern __shared__ __align__(16) float qwen4exp_attn_shared[];
    const uint32_t head = blockIdx.x;
    const uint32_t token = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    const uint32_t nth = blockDim.x;
    if (head >= n_head || token >= n_tokens) return;

    float *qvec = qwen4exp_attn_shared;
    float *tile = qvec + head_dim;
    float *probs = tile + nth;
    int32_t *keys = (int32_t *)(probs + nth);

    const uint32_t pos = pos0 + token;
    const uint32_t count = sparse ? (uint32_t)counts[token] : pos + 1u;
    const uint32_t kv_head = head / (n_head / n_kv_head);
    const uint32_t kv_stride = n_kv_head * head_dim;

    const float *qsrc = q + ((uint64_t)token * n_head + head) * head_dim;
    for (uint32_t d = tid; d < head_dim; d += nth) qvec[d] = qsrc[d];
    __syncthreads();

    float *dst = out + ((uint64_t)token * n_head + head) * head_dim;
    if (count == 0u) {
        for (uint32_t d = tid; d < head_dim; d += nth) dst[d] = 0.0f;
        return;
    }

    float run_max = QWEN4EXP_QSA_MASKED_SCORE;
    float run_sum = 0.0f;
    float acc = 0.0f;

    for (uint32_t base = 0; base < count; base += nth) {
        const uint32_t n_in_tile = min(nth, count - base);
        int32_t key = -1;
        float score = QWEN4EXP_QSA_MASKED_SCORE;
        if (tid < n_in_tile) {
            key = sparse ? selected[(uint64_t)token * max_selected + base + tid]
                         : (int32_t)(base + tid);
            if (key >= 0 && (uint32_t)key < cache_cap) {
                const float *kv = k_cache +
                    (uint64_t)key * kv_stride + (uint64_t)kv_head * head_dim;
                float dot = 0.0f;
                /* One 16-byte load per four channels instead of four 4-byte
                 * ones.  Consecutive lanes hold DIFFERENT keys, so the row a
                 * lane walks is its own: a scalar walk asks the cache for a
                 * 32-byte sector per lane per channel and uses four bytes of
                 * it, and this unit's key rows are far too many to keep those
                 * sectors in L1 between channels.  A word load spends one
                 * sector on sixteen bytes and issues a quarter of the
                 * instructions.  The products are the same four, added to
                 * `dot` in the same order, so the score is bit for bit the
                 * one the scalar walk returned.
                 *
                 * head_dim divides four here, and both `kv_stride` and the
                 * head offset are whole multiples of head_dim, so the row
                 * base is 16-byte aligned; the shared query block is aligned
                 * by its declaration.  The scalar walk stays for a head_dim
                 * that is not a multiple of four. */
                if ((head_dim & 3u) == 0u) {
                    const float4 *kv4 = (const float4 *)kv;
                    const float4 *qv4 = (const float4 *)qvec;
                    const uint32_t words = head_dim >> 2u;
                    for (uint32_t w = 0; w < words; w++) {
                        const float4 kk = kv4[w];
                        const float4 qq = qv4[w];
                        dot += qq.x * kk.x;
                        dot += qq.y * kk.y;
                        dot += qq.z * kk.z;
                        dot += qq.w * kk.w;
                    }
                } else {
                    for (uint32_t d = 0; d < head_dim; d++) {
                        dot += qvec[d] * kv[d];
                    }
                }
                score = dot * scale;
            } else {
                key = -1;
            }
        }
        keys[tid] = key;
        tile[tid] = score;
        const float tile_max = qwen4exp_blk_max(tile, tid, nth);
        const float new_max = fmaxf(run_max, tile_max);
        __syncthreads();

        probs[tid] = (key >= 0) ? expf(score - new_max) : 0.0f;
        tile[tid] = probs[tid];
        const float tile_sum = qwen4exp_blk_sum(tile, tid, nth);
        const float rescale = (run_max > QWEN4EXP_QSA_MASKED_LIMIT)
            ? expf(run_max - new_max) : 0.0f;
        run_sum = run_sum * rescale + tile_sum;

        if (tid < head_dim) {
            float contrib = 0.0f;
            for (uint32_t j = 0; j < n_in_tile; j++) {
                const int32_t kj = keys[j];
                if (kj < 0) continue;
                const float *vv = v_cache +
                    (uint64_t)kj * kv_stride + (uint64_t)kv_head * head_dim;
                contrib += probs[j] * vv[tid];
            }
            acc = acc * rescale + contrib;
        }
        run_max = new_max;
        __syncthreads();
    }

    if (tid < head_dim) {
        dst[tid] = (run_sum > 0.0f) ? acc / run_sum : 0.0f;
    }
}

/* The same attention, one block per HEAD GROUP instead of one per head.
 *
 * This model is grouped-query: 24 query heads share 2 KV heads, so twelve
 * heads read the identical key and value rows.  One block per head means the
 * unit fetches every K row and every V row twelve times over.  A block that
 * owns GROUP heads of one KV head fetches each row ONCE and hands it to all
 * GROUP heads out of a register (K) or a register broadcast (V).
 *
 * The reuse is a change of WHERE the bytes come from, and of nothing else.
 * Every float this kernel adds, it adds to the same running sum, in the same
 * position of the same sequence, as the per-head kernel above:
 *
 *   - `nth` is the same block width, so the tile partition `base += nth` cuts
 *     the key list at the same places and the online softmax takes the same
 *     number of rescales in the same order.
 *   - qwen4exp_blk_max / qwen4exp_blk_sum are called with the same `nth` and
 *     the same one scratch row, so the reduction tree and its warp-shuffle
 *     tail are the identical shape over the identical lane values.
 *   - the score dot keeps its accumulator per head and walks `w` upward,
 *     x then y then z then w, exactly as the per-head float4 walk does; the
 *     only difference is that the float4 `kk` it multiplies was loaded once
 *     for all GROUP heads instead of once per head.
 *   - the value accumulation keeps its accumulator per head and walks `j`
 *     upward skipping the same masked slots; `vv` is read once per j and
 *     broadcast, where the per-head kernel read it once per (head, j).
 *   - every product-accumulate this kernel performs is written as an
 *     explicit __fmaf_rn.  The per-head kernel writes the same four places as
 *     `a += b * c` and `a * b + c` and nvcc contracts all four into FFMA, and
 *     a fused multiply-add is one IEEE operation with one rounding, so the
 *     two agree bit for bit today.  Spelling the fusion out here is what
 *     keeps them agreeing: contraction is a decision the compiler makes per
 *     expression, and the array-of-accumulators shape this kernel needs is
 *     exactly the sort of rewrite that could talk it into an FMUL and an
 *     FADD instead -- two roundings, a different number.  The intrinsic
 *     takes that decision away from it.
 *
 * A line-by-line read of the two kernels' SASS says the same thing: every
 * floating-point opcode this one issues per head is the one the per-head
 * kernel issues, in the same count, down to the six instructions expf expands
 * to and the six the divide expands to.  Three of a head's fused multiply-adds
 * come out as UFFMA rather than FFMA, because the three operands are block-
 * uniform and ptxas puts uniform arithmetic on the uniform datapath.  Both are
 * the same PTX fma.rn.f32 -- one IEEE fused multiply-add, one rounding, no
 * flush, since this unit is built -ftz=false -- so the choice of datapath is
 * ptxas's and is not ours to see.  It is the one difference between the two
 * that this file cannot settle by reading; tests/test_qwen4exp_qsa.c settles
 * it on the device, by running the pipeline both ways and requiring the bytes
 * to match.
 *
 * The block-wide helpers are re-entered GROUP times per tile.  Every thread
 * of the block runs the same `for (h < GROUP)` over a compile-time bound with
 * no head-dependent branch in it, and the three early returns above are all
 * block-uniform (`head0`, `token`, `count`), so all threads enter all GROUP
 * calls in the same order and each internal __syncthreads is met by the whole
 * block.  `tile` is the one scratch row shared by those calls, so each head's
 * pass ends on a barrier before the next head overwrites it -- the barrier
 * the per-head kernel spends at the bottom of its tile loop, no more and no
 * fewer per reduction.
 *
 * `keys`, `count` and the selection do not depend on the head at all, so the
 * group shares them unchanged.  GROUP must divide n_head / n_kv_head, which
 * puts all GROUP heads under one `kv_head`.
 */
/* Key words a lane asks for before it consumes any of them, so several key
 * loads are outstanding at once.  A scheduling number: it changes how many
 * loads are in flight, not which products land in which accumulator. */
#define QWEN4EXP_QSA_KSTEP 4u

template <uint32_t GROUP>
__global__ static void qwen4exp_qsa_attention_group_kernel(
        const float *q,
        const float *k_cache,
        const float *v_cache,
        const int32_t *selected,
        const int32_t *counts,
        float *out,
        uint32_t n_tokens,
        uint32_t n_head,
        uint32_t n_kv_head,
        uint32_t head_dim,
        uint32_t pos0,
        uint32_t cache_cap,
        uint32_t max_selected,
        uint32_t sparse,
        float scale) {
    extern __shared__ __align__(16) float qwen4exp_attn_grp_shared[];
    const uint32_t group = blockIdx.x;
    const uint32_t token = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    const uint32_t nth = blockDim.x;
    const uint32_t head0 = group * GROUP;
    if (head0 + GROUP > n_head || token >= n_tokens) return;

    float *qvec = qwen4exp_attn_grp_shared;          /* GROUP * head_dim */
    float *tile = qvec + GROUP * head_dim;           /* nth              */
    float *probs = tile + nth;                       /* GROUP * nth      */
    int32_t *keys = (int32_t *)(probs + GROUP * nth);/* nth              */

    const uint32_t pos = pos0 + token;
    const uint32_t count = sparse ? (uint32_t)counts[token] : pos + 1u;
    const uint32_t kv_head = head0 / (n_head / n_kv_head);
    const uint32_t kv_stride = n_kv_head * head_dim;

    /* The GROUP query rows are adjacent in `q`, so one flat copy stages them
     * all; each row lands at qvec + h * head_dim, the layout the per-head
     * kernel's single row had. */
    const float *qsrc = q + ((uint64_t)token * n_head + head0) * head_dim;
    const uint32_t qspan = GROUP * head_dim;
    for (uint32_t d = tid; d < qspan; d += nth) qvec[d] = qsrc[d];
    __syncthreads();

    float *dst = out + ((uint64_t)token * n_head + head0) * head_dim;
    if (count == 0u) {
        for (uint32_t d = tid; d < qspan; d += nth) dst[d] = 0.0f;
        return;
    }

    float run_max[GROUP];
    float run_sum[GROUP];
    float acc[GROUP];
#pragma unroll
    for (uint32_t h = 0; h < GROUP; h++) {
        run_max[h] = QWEN4EXP_QSA_MASKED_SCORE;
        run_sum[h] = 0.0f;
        acc[h] = 0.0f;
    }

    for (uint32_t base = 0; base < count; base += nth) {
        const uint32_t n_in_tile = min(nth, count - base);
        int32_t key = -1;
        float score[GROUP];
#pragma unroll
        for (uint32_t h = 0; h < GROUP; h++) score[h] = QWEN4EXP_QSA_MASKED_SCORE;
        if (tid < n_in_tile) {
            key = sparse ? selected[(uint64_t)token * max_selected + base + tid]
                         : (int32_t)(base + tid);
            if (key >= 0 && (uint32_t)key < cache_cap) {
                const float *kv = k_cache +
                    (uint64_t)key * kv_stride + (uint64_t)kv_head * head_dim;
                float dot[GROUP];
#pragma unroll
                for (uint32_t h = 0; h < GROUP; h++) dot[h] = 0.0f;
                /* One 16-byte K load per four channels FOR THE WHOLE GROUP.
                 * `kk` is the same word the per-head kernel loaded; it is
                 * simply held in a register across the head loop instead of
                 * being asked of the memory system again for every head.
                 * Each dot[h] therefore sees q.x*k.x, q.y*k.y, q.z*k.z,
                 * q.w*k.w for w = 0, 1, 2 ... in that order, which is the
                 * per-head walk verbatim. */
                if ((head_dim & 3u) == 0u) {
                    const float4 *kv4 = (const float4 *)kv;
                    const uint32_t words = head_dim >> 2u;
                    uint32_t w = 0;
                    /* QWEN4EXP_QSA_KSTEP words are asked for before any of
                     * them is used, so the lane keeps that many key loads in
                     * flight the way the per-head kernel's unrolled walk does.
                     * One word at a time would leave exactly one load
                     * outstanding per lane and spend the whole loop waiting on
                     * it.  The heads still consume the words in the order the
                     * words were loaded, and each head still consumes all four
                     * channels of a word before moving to the next word, so
                     * every dot[h] sees w ascending and x, y, z, w within each
                     * -- the per-head sequence exactly. */
                    for (; w + QWEN4EXP_QSA_KSTEP <= words;
                           w += QWEN4EXP_QSA_KSTEP) {
                        float4 kk[QWEN4EXP_QSA_KSTEP];
#pragma unroll
                        for (uint32_t i = 0; i < QWEN4EXP_QSA_KSTEP; i++) {
                            kk[i] = kv4[w + i];
                        }
#pragma unroll
                        for (uint32_t h = 0; h < GROUP; h++) {
                            const float4 *qh =
                                (const float4 *)(qvec + h * head_dim);
#pragma unroll
                            for (uint32_t i = 0; i < QWEN4EXP_QSA_KSTEP; i++) {
                                const float4 qq = qh[w + i];
                                dot[h] = __fmaf_rn(qq.x, kk[i].x, dot[h]);
                                dot[h] = __fmaf_rn(qq.y, kk[i].y, dot[h]);
                                dot[h] = __fmaf_rn(qq.z, kk[i].z, dot[h]);
                                dot[h] = __fmaf_rn(qq.w, kk[i].w, dot[h]);
                            }
                        }
                    }
                    for (; w < words; w++) {
                        const float4 kk = kv4[w];
#pragma unroll
                        for (uint32_t h = 0; h < GROUP; h++) {
                            const float4 qq =
                                ((const float4 *)(qvec + h * head_dim))[w];
                            dot[h] = __fmaf_rn(qq.x, kk.x, dot[h]);
                            dot[h] = __fmaf_rn(qq.y, kk.y, dot[h]);
                            dot[h] = __fmaf_rn(qq.z, kk.z, dot[h]);
                            dot[h] = __fmaf_rn(qq.w, kk.w, dot[h]);
                        }
                    }
                } else {
                    for (uint32_t d = 0; d < head_dim; d++) {
                        const float kd = kv[d];
#pragma unroll
                        for (uint32_t h = 0; h < GROUP; h++) {
                            dot[h] = __fmaf_rn(qvec[h * head_dim + d], kd, dot[h]);
                        }
                    }
                }
#pragma unroll
                for (uint32_t h = 0; h < GROUP; h++) score[h] = dot[h] * scale;
            } else {
                key = -1;
            }
        }
        keys[tid] = key;

        /* One softmax pass per head, each over the same single `tile` row and
         * the same `nth`, so each is the per-head kernel's pass unchanged.
         * The trailing barrier is the one the per-head kernel spends at the
         * bottom of its tile loop, moved to the bottom of each head's pass
         * because `tile` is now reused GROUP times inside one tile. */
        float new_max[GROUP];
        float rescale[GROUP];
#pragma unroll
        for (uint32_t h = 0; h < GROUP; h++) {
            tile[tid] = score[h];
            const float tile_max = qwen4exp_blk_max(tile, tid, nth);
            new_max[h] = fmaxf(run_max[h], tile_max);
            __syncthreads();

            float *ph = probs + h * nth;
            ph[tid] = (key >= 0) ? expf(score[h] - new_max[h]) : 0.0f;
            tile[tid] = ph[tid];
            const float tile_sum = qwen4exp_blk_sum(tile, tid, nth);
            rescale[h] = (run_max[h] > QWEN4EXP_QSA_MASKED_LIMIT)
                ? expf(run_max[h] - new_max[h]) : 0.0f;
            run_sum[h] = __fmaf_rn(run_sum[h], rescale[h], tile_sum);
            __syncthreads();
        }

        if (tid < head_dim) {
            float contrib[GROUP];
#pragma unroll
            for (uint32_t h = 0; h < GROUP; h++) contrib[h] = 0.0f;
            for (uint32_t j = 0; j < n_in_tile; j++) {
                const int32_t kj = keys[j];
                if (kj < 0) continue;
                /* One V channel read for the whole group, where the per-head
                 * kernel read the same address once per head. */
                const float vvj = v_cache[
                    (uint64_t)kj * kv_stride + (uint64_t)kv_head * head_dim + tid];
#pragma unroll
                for (uint32_t h = 0; h < GROUP; h++) {
                    contrib[h] = __fmaf_rn(probs[h * nth + j], vvj, contrib[h]);
                }
            }
#pragma unroll
            for (uint32_t h = 0; h < GROUP; h++) {
                acc[h] = __fmaf_rn(acc[h], rescale[h], contrib[h]);
            }
        }
#pragma unroll
        for (uint32_t h = 0; h < GROUP; h++) run_max[h] = new_max[h];
        __syncthreads();
    }

    if (tid < head_dim) {
#pragma unroll
        for (uint32_t h = 0; h < GROUP; h++) {
            dst[h * head_dim + tid] =
                (run_sum[h] > 0.0f) ? acc[h] / run_sum[h] : 0.0f;
        }
    }
}

__global__ static void qwen4exp_qsa_output_gate_kernel(
        const float *gate,
        float *out,
        uint32_t n_values) {
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n_values) return;
    out[gid] = out[gid] * (1.0f / (1.0f + expf(-gate[gid])));
}

/* Largest power of two that is <= `value` and <= 1024, the CUDA block cap. */
static uint32_t qwen4exp_cuda_threads(uint32_t value) {
    uint32_t nth = 1;
    while (nth * 2u <= value && nth * 2u <= 1024u) nth *= 2;
    return nth;
}

/* How wide a head group qwen4exp_qsa_attention_group_kernel may take, and the
 * two limits the choice lives inside.
 *
 * The group kernel's block is the per-head block plus GROUP query rows and
 * GROUP probability rows of shared memory, so the width is bounded by the
 * 48 KiB a block gets without an opt-in carveout.  It also divides the grid
 * by GROUP, so it is only offered to a call wide enough that the smaller grid
 * still fills the device -- a prefill chunk, not a decode row.
 *
 * Both bounds are scheduling bounds.  The kernel's arithmetic does not depend
 * on GROUP at all (see its note), so moving these numbers cannot move a
 * single output bit. */
#define QWEN4EXP_QSA_GROUP_MIN_ROWS   64u
#define QWEN4EXP_QSA_GROUP_SHARED_CAP (48u * 1024u)

static size_t qwen4exp_qsa_group_shared(uint32_t group, uint32_t head_dim,
                                        uint32_t nth) {
    return ((size_t)group * head_dim + (size_t)group * nth + nth) *
               sizeof(float) +
           (size_t)nth * sizeof(int32_t);
}

/* Group width.  DS4_QWEN4EXP_NO_QSA_GROUP turns the group kernel off outright;
 * DS4_QWEN4EXP_QSA_GROUP sets the width (0 or 1 also turns it off).  The
 * default is the model's full 24/2 group.
 *
 * Read fresh rather than cached, so a test can put the two kernels side by
 * side in one process and diff their bytes -- tests/test_qwen4exp_qsa.c does
 * exactly that.  The caller asks only after it has already decided the row is
 * wide enough for the group kernel, so a decode step never reaches this and a
 * prefill chunk pays one getenv per layer against a kernel that runs for
 * milliseconds. */
static uint32_t qwen4exp_qsa_group_width(void) {
    if (getenv("DS4_QWEN4EXP_NO_QSA_GROUP") != NULL) return 1u;
    const char *forced = getenv("DS4_QWEN4EXP_QSA_GROUP");
    if (forced != NULL) {
        const long v = strtol(forced, NULL, 10);
        return (v > 0 && v <= 32) ? (uint32_t)v : 1u;
    }
    return 12u;
}

extern "C" void ds4_gpu_qwen4exp_rope_inv_freq(
        float *dst, uint32_t rot_dim, float freq_base) {
    /* One table for every rope user: ds4_qwen4exp_rope_inv_freq in
     * ds4_qwen4exp_hc_ref.h, which the f32 reference calls too. */
    ds4_qwen4exp_rope_inv_freq(dst, rot_dim, freq_base);
}

extern "C" int ds4_gpu_qwen4exp_qsa_split_qkv_tensor(
        ds4_gpu_tensor       *q,
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *k,
        ds4_gpu_tensor       *v,
        const ds4_gpu_tensor *fused,
        uint32_t              n_tokens,
        uint32_t              n_head,
        uint32_t              n_kv_head,
        uint32_t              head_dim) {
    if (n_tokens == 0u || n_head == 0u || n_kv_head == 0u || head_dim == 0u) return 0;
    const uint64_t q_elems = (uint64_t)n_tokens * n_head * head_dim;
    const uint64_t kv_elems = (uint64_t)n_tokens * n_kv_head * head_dim;
    if (!glm53_cuda_tensor_has(q, q_elems, sizeof(float)) ||
        !glm53_cuda_tensor_has(gate, q_elems, sizeof(float)) ||
        !glm53_cuda_tensor_has(k, kv_elems, sizeof(float)) ||
        !glm53_cuda_tensor_has(v, kv_elems, sizeof(float)) ||
        !glm53_cuda_tensor_has(fused, 2u * q_elems + 2u * kv_elems, sizeof(float))) {
        return 0;
    }
    const uint64_t total = q_elems + 2u * kv_elems;
    qwen4exp_qsa_split_qkv_kernel<<<
        (unsigned)((total + 255u) / 256u), 256u, 0, cuda_decode_stream()>>>(
            (const float *)fused->ptr, (float *)q->ptr, (float *)gate->ptr,
            (float *)k->ptr, (float *)v->ptr, n_tokens, n_head, n_kv_head,
            head_dim);
    return cuda_ok(cudaGetLastError(), "Qwen4-Exp qkv split launch");
}

extern "C" int ds4_gpu_qwen4exp_qsa_split_doubled_q_tensor(
        ds4_gpu_tensor       *q,
        ds4_gpu_tensor       *gate,
        const ds4_gpu_tensor *doubled,
        uint32_t              n_tokens,
        uint32_t              n_head,
        uint32_t              head_dim) {
    if (n_tokens == 0u || n_head == 0u || head_dim == 0u) return 0;
    const uint64_t q_elems = (uint64_t)n_tokens * n_head * head_dim;
    if (!glm53_cuda_tensor_has(q, q_elems, sizeof(float)) ||
        !glm53_cuda_tensor_has(gate, q_elems, sizeof(float)) ||
        !glm53_cuda_tensor_has(doubled, 2u * q_elems, sizeof(float))) {
        return 0;
    }
    qwen4exp_qsa_split_doubled_q_kernel<<<
        (unsigned)((q_elems + 255u) / 256u), 256u, 0, cuda_decode_stream()>>>(
            (const float *)doubled->ptr, (float *)q->ptr, (float *)gate->ptr,
            n_tokens, n_head, head_dim);
    return cuda_ok(cudaGetLastError(), "Qwen4-Exp doubled-query split launch");
}

extern "C" int ds4_gpu_qwen4exp_head_rms_norm_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *weight,
        uint32_t              n_rows,
        uint32_t              head_dim,
        float                 eps,
        float                 weight_offset) {
    if (n_rows == 0u || head_dim == 0u) return 0;
    const uint64_t elems = (uint64_t)n_rows * head_dim;
    if (!glm53_cuda_tensor_has(out, elems, sizeof(float)) ||
        !glm53_cuda_tensor_has(x, elems, sizeof(float)) ||
        !glm53_cuda_tensor_has(weight, head_dim, sizeof(float))) {
        return 0;
    }
    const uint32_t nth = qwen4exp_cuda_threads(head_dim);
    qwen4exp_head_rms_norm_kernel<<<n_rows, nth, nth * sizeof(float),
        cuda_decode_stream()>>>(
            (const float *)x->ptr, (const float *)weight->ptr,
            (float *)out->ptr, n_rows, head_dim, eps, weight_offset);
    return cuda_ok(cudaGetLastError(), "Qwen4-Exp head RMS norm launch");
}

extern "C" int ds4_gpu_qwen4exp_rope_head_tensor(
        ds4_gpu_tensor       *x,
        const ds4_gpu_tensor *inv_freq,
        uint32_t              n_tokens,
        uint32_t              n_head,
        uint32_t              head_dim,
        uint32_t              rot_dim,
        uint32_t              pos0) {
    if (n_tokens == 0u || n_head == 0u || head_dim == 0u || rot_dim == 0u ||
        rot_dim > head_dim || (rot_dim % 2u) != 0u ||
        !glm53_cuda_tensor_has(x, (uint64_t)n_tokens * n_head * head_dim, sizeof(float)) ||
        !glm53_cuda_tensor_has(inv_freq, rot_dim / 2u, sizeof(float))) {
        return 0;
    }
    const uint64_t total = (uint64_t)n_tokens * n_head * (rot_dim / 2u);
    qwen4exp_rope_head_kernel<<<
        (unsigned)((total + 255u) / 256u), 256u, 0, cuda_decode_stream()>>>(
            (float *)x->ptr, (const float *)inv_freq->ptr, n_tokens, n_head,
            head_dim, rot_dim, pos0);
    return cuda_ok(cudaGetLastError(), "Qwen4-Exp partial rope launch");
}

extern "C" int ds4_gpu_qwen4exp_qsa_indexer_pool_update_tensor(
        ds4_gpu_tensor       *pool,
        ds4_gpu_tensor       *tape,
        const ds4_gpu_tensor *raw_k,
        const ds4_gpu_tensor *k_norm_weight,
        const ds4_gpu_tensor *inv_freq,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_cap,
        uint32_t              head_dim,
        uint32_t              pool_size,
        uint32_t              rot_dim,
        float                 eps,
        float                 weight_offset) {
    if (n_tokens == 0u || head_dim == 0u || pool_size == 0u || rot_dim == 0u ||
        rot_dim > head_dim || (uint64_t)pos0 + n_tokens > cache_cap ||
        !glm53_cuda_tensor_has(tape, (uint64_t)cache_cap * head_dim, sizeof(float)) ||
        !glm53_cuda_tensor_has(pool, (uint64_t)(cache_cap / pool_size) * head_dim,
                               sizeof(float)) ||
        !glm53_cuda_tensor_has(raw_k, (uint64_t)n_tokens * head_dim, sizeof(float)) ||
        !glm53_cuda_tensor_has(k_norm_weight, head_dim, sizeof(float)) ||
        !glm53_cuda_tensor_has(inv_freq, rot_dim / 2u, sizeof(float))) {
        return 0;
    }
    const uint64_t append = (uint64_t)n_tokens * head_dim;
    qwen4exp_qsa_tape_append_kernel<<<
        (unsigned)((append + 255u) / 256u), 256u, 0, cuda_decode_stream()>>>(
            (const float *)raw_k->ptr, (float *)tape->ptr, n_tokens, head_dim,
            pos0, cache_cap);
    if (!cuda_ok(cudaGetLastError(), "Qwen4-Exp indexer tape append launch")) {
        return 0;
    }

    const uint32_t block0 = pos0 / pool_size;
    const uint32_t block1 = (pos0 + n_tokens) / pool_size;
    if (block1 > block0) {
        const uint32_t nth = qwen4exp_cuda_threads(head_dim);
        const size_t shared = ((size_t)head_dim + nth) * sizeof(float);
        qwen4exp_qsa_pool_update_kernel<<<block1 - block0, nth, shared,
            cuda_decode_stream()>>>(
                (const float *)tape->ptr, (const float *)k_norm_weight->ptr,
                (const float *)inv_freq->ptr, (float *)pool->ptr, block0,
                block1 - block0, head_dim, pool_size, rot_dim, cache_cap, eps,
                weight_offset);
        if (!cuda_ok(cudaGetLastError(), "Qwen4-Exp indexer pool update launch")) {
            return 0;
        }
    }
    return 1;
}

extern "C" int ds4_gpu_qwen4exp_qsa_indexer_scores_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *pool,
        uint32_t              n_tokens,
        uint32_t              n_blocks,
        uint32_t              n_head,
        uint32_t              head_dim,
        uint32_t              pos0,
        uint32_t              pool_size) {
    if (n_tokens == 0u || n_blocks == 0u || n_head == 0u || head_dim == 0u ||
        pool_size == 0u ||
        !glm53_cuda_tensor_has(scores, (uint64_t)n_tokens * n_blocks, sizeof(float)) ||
        !glm53_cuda_tensor_has(q, (uint64_t)n_tokens * n_head * head_dim, sizeof(float)) ||
        !glm53_cuda_tensor_has(pool, (uint64_t)n_blocks * head_dim, sizeof(float))) {
        return 0;
    }
    const uint32_t nth = qwen4exp_cuda_threads(head_dim);
    qwen4exp_qsa_indexer_scores_kernel<<<dim3(n_blocks, n_tokens), nth,
        nth * sizeof(float), cuda_decode_stream()>>>(
            (const float *)q->ptr, (const float *)pool->ptr,
            (float *)scores->ptr, n_tokens, n_blocks, n_head, head_dim, pos0,
            pool_size, sqrtf((float)head_dim));
    return cuda_ok(cudaGetLastError(), "Qwen4-Exp indexer scores launch");
}

extern "C" int ds4_gpu_qwen4exp_qsa_indexer_select_tensor(
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *counts,
        const ds4_gpu_tensor *scores,
        const ds4_gpu_tensor *topk,
        uint32_t              n_tokens,
        uint32_t              n_blocks,
        uint32_t              top_k,
        uint32_t              pos0,
        uint32_t              pool_size,
        uint32_t              max_selected) {
    if (n_tokens == 0u || n_blocks == 0u || top_k == 0u || top_k > n_blocks ||
        pool_size == 0u ||
        max_selected < (uint64_t)top_k * pool_size + pool_size - 1u ||
        !glm53_cuda_tensor_has(selected, (uint64_t)n_tokens * max_selected,
                               sizeof(int32_t)) ||
        !glm53_cuda_tensor_has(counts, n_tokens, sizeof(int32_t)) ||
        !glm53_cuda_tensor_has(scores, (uint64_t)n_tokens * n_blocks, sizeof(float)) ||
        !glm53_cuda_tensor_has(topk, (uint64_t)n_tokens * top_k, sizeof(int32_t))) {
        return 0;
    }
    uint32_t sort_width = 1;
    while (sort_width < top_k) sort_width *= 2;
    const uint32_t nth = qwen4exp_cuda_threads(sort_width);
    qwen4exp_qsa_indexer_select_kernel<<<n_tokens, nth,
        (size_t)sort_width * sizeof(int32_t), cuda_decode_stream()>>>(
            (const float *)scores->ptr, (const int32_t *)topk->ptr,
            (int32_t *)selected->ptr, (int32_t *)counts->ptr, n_tokens,
            n_blocks, top_k, sort_width, pos0, pool_size, max_selected);
    return cuda_ok(cudaGetLastError(), "Qwen4-Exp indexer selection launch");
}

extern "C" int ds4_gpu_qwen4exp_qsa_attention_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *k_cache,
        const ds4_gpu_tensor *v_cache,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *counts,
        uint32_t              n_tokens,
        uint32_t              n_head,
        uint32_t              n_kv_head,
        uint32_t              head_dim,
        uint32_t              pos0,
        uint32_t              cache_cap,
        uint32_t              max_selected,
        float                 scale) {
    if (n_tokens == 0u || n_head == 0u || n_kv_head == 0u || head_dim == 0u ||
        (n_head % n_kv_head) != 0u || (uint64_t)pos0 + n_tokens > cache_cap) {
        return 0;
    }
    const bool sparse = selected != NULL;
    if (sparse && (!counts || max_selected == 0u)) return 0;
    const uint64_t q_elems = (uint64_t)n_tokens * n_head * head_dim;
    const uint64_t cache_elems = (uint64_t)cache_cap * n_kv_head * head_dim;
    if (!glm53_cuda_tensor_has(out, q_elems, sizeof(float)) ||
        !glm53_cuda_tensor_has(q, q_elems, sizeof(float)) ||
        !glm53_cuda_tensor_has(k_cache, cache_elems, sizeof(float)) ||
        !glm53_cuda_tensor_has(v_cache, cache_elems, sizeof(float)) ||
        (sparse &&
         (!glm53_cuda_tensor_has(selected, (uint64_t)n_tokens * max_selected,
                                 sizeof(int32_t)) ||
          !glm53_cuda_tensor_has(counts, n_tokens, sizeof(int32_t))))) {
        return 0;
    }
    const uint32_t nth = qwen4exp_cuda_threads(head_dim);
    /* The kernel accumulates the value vector under `tid < head_dim`, so a
     * block narrower than head_dim would leave the tail channels unwritten
     * with no error anywhere.  Refuse instead. */
    if (nth < head_dim) return 0;

    /* Wide rows go to the head-group kernel, which reads each K and each V
     * row once for the whole group instead of once per head.  It is the same
     * arithmetic in the same order (see the kernel's own note), so the choice
     * is a scheduling one only; a narrow call keeps the per-head kernel,
     * whose grid is n_head times wider and which is what a one-row decode
     * needs to fill the device at all. */
    const uint32_t gqa = n_head / n_kv_head;
    const uint32_t want = (n_tokens >= QWEN4EXP_QSA_GROUP_MIN_ROWS)
        ? qwen4exp_qsa_group_width() : 1u;
    if (want > 1u) {
        uint32_t g = want < gqa ? want : gqa;
        while (g > 1u && (gqa % g) != 0u) g--;
        const size_t gshared = qwen4exp_qsa_group_shared(g, head_dim, nth);
        if (g > 1u && gshared <= QWEN4EXP_QSA_GROUP_SHARED_CAP) {
            const dim3 grid(n_head / g, n_tokens);
#define QWEN4EXP_QSA_GROUP_LAUNCH(G)                                          \
            qwen4exp_qsa_attention_group_kernel<G><<<grid, nth, gshared,      \
                cuda_decode_stream()>>>(                                      \
                    (const float *)q->ptr, (const float *)k_cache->ptr,       \
                    (const float *)v_cache->ptr,                              \
                    sparse ? (const int32_t *)selected->ptr : NULL,           \
                    sparse ? (const int32_t *)counts->ptr : NULL,             \
                    (float *)out->ptr, n_tokens, n_head, n_kv_head, head_dim, \
                    pos0, cache_cap, max_selected, sparse ? 1u : 0u, scale)
            switch (g) {
                case 12u: QWEN4EXP_QSA_GROUP_LAUNCH(12u); break;
                case 8u:  QWEN4EXP_QSA_GROUP_LAUNCH(8u);  break;
                case 6u:  QWEN4EXP_QSA_GROUP_LAUNCH(6u);  break;
                case 4u:  QWEN4EXP_QSA_GROUP_LAUNCH(4u);  break;
                case 3u:  QWEN4EXP_QSA_GROUP_LAUNCH(3u);  break;
                case 2u:  QWEN4EXP_QSA_GROUP_LAUNCH(2u);  break;
                default:  g = 1u; break;
            }
#undef QWEN4EXP_QSA_GROUP_LAUNCH
            if (g > 1u) {
                return cuda_ok(cudaGetLastError(),
                               "Qwen4-Exp QSA grouped attention launch");
            }
        }
    }

    const size_t shared = ((size_t)head_dim + 2u * nth) * sizeof(float) +
                          (size_t)nth * sizeof(int32_t);
    qwen4exp_qsa_attention_kernel<<<dim3(n_head, n_tokens), nth, shared,
        cuda_decode_stream()>>>(
            (const float *)q->ptr, (const float *)k_cache->ptr,
            (const float *)v_cache->ptr,
            sparse ? (const int32_t *)selected->ptr : NULL,
            sparse ? (const int32_t *)counts->ptr : NULL,
            (float *)out->ptr, n_tokens, n_head, n_kv_head, head_dim, pos0,
            cache_cap, max_selected, sparse ? 1u : 0u, scale);
    return cuda_ok(cudaGetLastError(), "Qwen4-Exp QSA attention launch");
}

extern "C" int ds4_gpu_qwen4exp_qsa_output_gate_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *gate,
        uint32_t              n_values) {
    if (n_values == 0u ||
        !glm53_cuda_tensor_has(out, n_values, sizeof(float)) ||
        !glm53_cuda_tensor_has(gate, n_values, sizeof(float))) {
        return 0;
    }
    qwen4exp_qsa_output_gate_kernel<<<
        (unsigned)((n_values + 255u) / 256u), 256u, 0, cuda_decode_stream()>>>(
            (const float *)gate->ptr, (float *)out->ptr, n_values);
    return cuda_ok(cudaGetLastError(), "Qwen4-Exp QSA output gate launch");
}

/* =========================================================================
 * Qwen4-Exp per-layer embedding (PLE) block, the CUDA twin of
 * metal/qwen4exp_ple.metal.
 * =========================================================================
 *
 * Kernel for kernel and argument for argument the same as the Metal half; a
 * normalized diff of the two bodies leaves the launch plumbing, the
 * shared-memory reduction this unit uses in place of Metal's simdgroup one,
 * and the float suffixes on exp, sqrt and fabs.  The double reference both
 * are checked against is ds4_qwen4exp_ple_ref.h, and the gate's signed square
 * root is one of the reasons this unit is built without --use_fast_math: it
 * takes sqrtf of a magnitude floored at 1e-6, which -ftz=true and
 * -prec-sqrt=false both disturb.
 *
 * The n-gram table never reaches the device: the host gathers the rows out of
 * the mapping and uploads a [rows][ple_embd] block, so the only PLE state
 * here is that block, the scratch and the rolling convolution window.
 */

__device__ __forceinline__ static float qwen4exp_ple_signed_sqrt(float v) {
    const float magnitude = sqrtf(fmaxf(fabsf(v), 1.0e-6f));
    return v > 0.0f ? magnitude : (v < 0.0f ? -magnitude : 0.0f);
}

__global__ static void qwen4exp_ple_gate_kernel(
        float *dst, const float *key, const float *query, const float *value,
        uint32_t n_embd, uint32_t n_hc, uint32_t n_tokens,
        float inv_sqrt_embd) {
    const uint32_t h = blockIdx.x;
    const uint32_t t = blockIdx.y;
    if (h >= n_hc || t >= n_tokens) return;

    const uint64_t base = ((uint64_t)t * n_hc + h) * n_embd;
    const float *kr = key + base;
    const float *qr = query + base;
    const float *vr = value + (uint64_t)t * n_embd;
    float *dr = dst + base;

    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < n_embd; i += blockDim.x) {
        sum += kr[i] * qr[i];
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }

    const float gate =
        qwen4exp_sigmoid(qwen4exp_ple_signed_sqrt(partial[0] * inv_sqrt_embd));

    /* Every read of `key` is complete at this barrier, so the write loop is
     * safe with dst == key. */
    __syncthreads();

    for (uint32_t i = threadIdx.x; i < n_embd; i += blockDim.x) {
        dr[i] = gate * vr[i];
    }
}

__global__ static void qwen4exp_ple_conv_kernel(
        float *hyper, float *state, const float *gated, const float *conv_in,
        const float *weight, float *snapshot, uint32_t channels,
        uint32_t conv_kernel, uint32_t dilation, uint32_t state_len,
        uint32_t n_tokens, uint32_t n_snapshot_rows) {
    const uint32_t c = (uint32_t)(blockIdx.x * blockDim.x + threadIdx.x);
    if (c >= channels) return;

    const uint32_t C = channels;
    const uint32_t S = state_len;

    for (uint32_t t = 0; t < n_tokens; t++) {
        float acc = 0.0f;
        for (uint32_t k = 0; k < conv_kernel; k++) {
            const uint32_t i = t + dilation * k;
            const float v = (i < S) ? state[(uint64_t)i * C + c]
                                    : conv_in[(uint64_t)(i - S) * C + c];
            acc = fmaf(v, weight[(uint64_t)c * conv_kernel + k], acc);
        }
        const uint64_t index = (uint64_t)t * C + c;
        hyper[index] += gated[index] + acc * qwen4exp_sigmoid(acc);
    }

    /* The rolling window as it stands after each of the first
     * `n_snapshot_rows` tokens.  This runs BEFORE the state write below,
     * because `full()` reads the incoming state and the write below clobbers
     * it -- the same read-before-write ordering the state write itself
     * depends on, one step earlier.  Per-row state snapshots for the
     * speculative cycle; see the note above the GDN kernels.  Zero on every
     * serial forward, where the loop runs no iterations. */
    for (uint32_t t = 0; t < n_snapshot_rows; t++) {
        float *slot = snapshot + (uint64_t)t * S * C;
        for (uint32_t j = 0; j < S; j++) {
            const uint32_t i = t + 1u + j;
            slot[(uint64_t)j * C + c] =
                (i < S) ? state[(uint64_t)i * C + c]
                        : conv_in[(uint64_t)(i - S) * C + c];
        }
    }

    /* Ascending, and the read index is n_tokens ahead of the write index, so
     * no slot is read after it has been overwritten. */
    for (uint32_t j = 0; j < S; j++) {
        const uint32_t i = n_tokens + j;
        state[(uint64_t)j * C + c] = (i < S) ? state[(uint64_t)i * C + c]
                                             : conv_in[(uint64_t)(i - S) * C + c];
    }
}

extern "C" int ds4_gpu_qwen4exp_ple_gate_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *key_hc,
        const ds4_gpu_tensor *query_hc,
        const ds4_gpu_tensor *value,
        uint32_t              n_embd,
        uint32_t              n_hc,
        uint32_t              rows) {
    if (!out_hc || !key_hc || !query_hc || !value ||
        n_embd == 0 || n_hc == 0 || rows == 0) {
        return 0;
    }
    const uint64_t wide_bytes = (uint64_t)rows * n_hc * n_embd * sizeof(float);
    const uint64_t value_bytes = (uint64_t)rows * n_embd * sizeof(float);
    if (out_hc->bytes < wide_bytes || key_hc->bytes < wide_bytes ||
        query_hc->bytes < wide_bytes || value->bytes < value_bytes) {
        return 0;
    }
    dim3 grid(n_hc, rows, 1u);
    qwen4exp_ple_gate_kernel<<<grid, 256, 0, cuda_decode_stream()>>>(
            (float *)out_hc->ptr, (const float *)key_hc->ptr,
            (const float *)query_hc->ptr, (const float *)value->ptr,
            n_embd, n_hc, rows, 1.0f / sqrtf((float)n_embd));
    return cuda_ok(cudaGetLastError(), "qwen4exp_ple_gate launch");
}

extern "C" int ds4_gpu_qwen4exp_ple_conv_tensor(
        ds4_gpu_tensor       *hyper,
        ds4_gpu_tensor       *conv_state,
        ds4_gpu_tensor       *conv_snapshot,
        uint32_t              n_snapshot_rows,
        const ds4_gpu_tensor *gated,
        const ds4_gpu_tensor *conv_in,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              channels,
        uint32_t              conv_kernel,
        uint32_t              dilation,
        uint32_t              rows) {
    if (!hyper || !conv_state || !gated || !conv_in || !model_map ||
        channels == 0 || conv_kernel < 2u || dilation == 0 || rows == 0) {
        return 0;
    }
    /* The last row's window IS the live state and needs no slot, so a request
     * that reaches the row count is a caller asking for a window this call
     * never leaves behind.  A missing buffer refuses rather than clamps. */
    if (n_snapshot_rows > 0 && (!conv_snapshot || n_snapshot_rows >= rows)) {
        fprintf(stderr,
                "ds4: CUDA qwen4exp PLE convolution asked for %u snapshot "
                "rows over %u\n", n_snapshot_rows, rows);
        return 0;
    }
    const uint32_t state_len = (conv_kernel - 1u) * dilation;
    const uint64_t row_bytes = (uint64_t)channels * sizeof(float);
    const uint64_t stream_bytes = row_bytes * rows;
    const uint64_t weight_bytes =
        (uint64_t)channels * conv_kernel * sizeof(float);
    if (weight_offset > model_size ||
        model_size - weight_offset < weight_bytes ||
        hyper->bytes < stream_bytes || gated->bytes < stream_bytes ||
        conv_in->bytes < stream_bytes ||
        conv_state->bytes < row_bytes * state_len ||
        (n_snapshot_rows > 0 &&
         conv_snapshot->bytes < row_bytes * state_len * n_snapshot_rows)) {
        return 0;
    }
    const int logical_tier = ds4_tensor_device_idx(hyper);
    const float *w = (const float *)cuda_resolve_weight_ptr(
            model_map, weight_offset, weight_bytes, logical_tier,
            "qwen4exp_ple_conv_weight");
    if (!w) return 0;
    qwen4exp_ple_conv_kernel<<<
        (unsigned)((channels + 255u) / 256u), 256, 0, cuda_decode_stream()>>>(
            (float *)hyper->ptr, (float *)conv_state->ptr,
            (const float *)gated->ptr, (const float *)conv_in->ptr, w,
            conv_snapshot ? (float *)conv_snapshot->ptr : NULL,
            channels, conv_kernel, dilation, state_len, rows,
            n_snapshot_rows);
    return cuda_ok(cudaGetLastError(), "qwen4exp_ple_conv launch");
}

#include "ds4_qwen4exp_hc_host.inc"
#include "ds4_qwen4exp_ple_host.inc"
