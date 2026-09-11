// Qwen4exp hyper-connection, norm and rope kernels.
//
// The qwen4exp gated residual is NOT DS4 Flash's Sinkhorn hyper-connection.
// Both carry `n_hc` residual streams side by side in the same
// [token][hc][embd] layout that metal/dsv4_hc.metal uses, and this file keeps
// that layout and the stride convention of kernel_dsv4_hc_expand so the two
// families can share activation buffers.  Everything else differs:
//
//   * DS4 splits one mixer row into pre weights, post gates and an
//     `n_hc * n_hc` Sinkhorn combination matrix.  qwen4exp has no combination
//     matrix at all: its inject is diagonal, one scalar per stream.
//   * DS4's pre-reduction is a weighted SUM with one scalar per stream.
//     qwen4exp's is a MEAN over the streams of a full-width gate, one value
//     per (stream, embedding channel), produced by a low-rank
//     10240 -> 320 -> 10240 projection of the normalized streams.
//   * The two `/ n_hc` divides (before the low-rank silu, and before the
//     inject sigmoid) have no DS4 counterpart and are easy to lose.
//
// Reference: Qwen4ExpGatedResidual in
// Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen4ExpText.swift.

struct ds4_metal_args_qwen4exp_norm {
    uint32_t n;            // full row width
    uint32_t group;        // group size; equal to n for an ungrouped norm
    uint32_t n_group;      // n / group
    float    eps;
    float    weight_bias;  // 0 when the checkpoint bakes the offset, else 1
    int32_t  round_bf16;   // round the normalized value to bf16 before scaling
};

struct ds4_metal_args_qwen4exp_hc {
    uint32_t n_embd;
    uint32_t n_hc;
    uint32_t n_tokens;
    float    inv_hc;
};

// The inject kernel alone: it carries the weight's TYPE and row stride, which
// the mixer's other kernels have no use for.
struct ds4_metal_args_qwen4exp_hc_inject {
    uint32_t n_embd;
    uint32_t n_hc;
    uint32_t n_tokens;
    float    inv_hc;
    uint32_t weight_type;        // ggml type id; see ds4_qwen4exp_hc_types.h
    uint32_t weight_row_bytes;   // bytes per [wide] row at that type
};

struct ds4_metal_args_qwen4exp_unary {
    uint32_t n;
    float    scale;
};

// Round to nearest even bf16 and widen back to f32.  MLX runs this tower in
// bf16, and MLX's fused rmsNorm casts the normalized value to the activation
// dtype BEFORE multiplying by the weight; the cast is reproduced here so an
// f32 engine can still match the reference bit pattern of the product.
// Matches f32_to_bf16 in tests/test_qwen4exp_hc_norm.c.
static inline float ds4_qwen4exp_round_bf16(float v) {
    const uint32_t bits = as_type<uint32_t>(v);
    const uint32_t rounding = 0x7fffu + ((bits >> 16) & 1u);
    return as_type<float>((bits + rounding) & 0xffff0000u);
}

static inline float ds4_qwen4exp_sigmoid(float z) {
    return 1.0f / (1.0f + exp(-z));
}

// Zero-centered RMS norm with optional grouping.
//
// Each group is normalized on its own statistic; the weight still indexes the
// FLAT row, so one weight tensor covers every group.  `n_group == 1` is the
// ordinary per-row norm, which is what the MTP head's ungrouped 10240-wide
// pre-norm needs; `group == n_embd` is the hyper-connection form, where each
// of the four streams carries its own statistic.
//
// One threadgroup per (group, row).  The reduction tree copies
// kernel_rms_norm_fuse_impl in metal/norm.metal.
kernel void kernel_qwen4exp_rms_norm(
        constant ds4_metal_args_qwen4exp_norm & args,
        device  const float * x,
        device  const float * weight,
        device        float * dst,
        threadgroup   float * shmem [[threadgroup(0)]],
        uint3   tgpig [[threadgroup_position_in_grid]],
        uint3   tpitg [[thread_position_in_threadgroup]],
        ushort  sgitg [[simdgroup_index_in_threadgroup]],
        ushort  tiisg [[thread_index_in_simdgroup]],
        uint3   ntg   [[threads_per_threadgroup]]) {
    if (sgitg == 0) {
        shmem[tiisg] = 0.0f;
    }

    const uint32_t group = tgpig.x;
    const uint32_t row = tgpig.y;
    const uint64_t base = (uint64_t)row * args.n + (uint64_t)group * args.group;

    device const float * xg = x + base;
    device       float * yg = dst + base;
    device const float * wg = weight + (uint64_t)group * args.group;

    float sumf = 0.0f;
    for (uint32_t i = tpitg.x; i < args.group; i += ntg.x) {
        const float v = xg[i];
        sumf += v * v;
    }
    sumf = simd_sum(sumf);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tiisg == 0) {
        shmem[sgitg] = sumf;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumf = shmem[tiisg];
    sumf = simd_sum(sumf);

    const float mean = sumf / (float)args.group;
    // 1/sqrt, not rsqrt: division and sqrt are correctly rounded, so the
    // scale is reproducible against the host reference.
    const float scale = 1.0f / sqrt(mean + args.eps);

    for (uint32_t i = tpitg.x; i < args.group; i += ntg.x) {
        float normed = xg[i] * scale;
        if (args.round_bf16) {
            normed = ds4_qwen4exp_round_bf16(normed);
        }
        yg[i] = normed * (args.weight_bias + wg[i]);
    }
}

// silu(x * scale), in place.  `scale` carries the 1/n_hc divide that sits
// between the low-rank down projection and the activation.
kernel void kernel_qwen4exp_scale_silu(
        constant ds4_metal_args_qwen4exp_unary & args,
        device        float * x,
        uint gid [[thread_position_in_grid]]) {
    if (gid >= args.n) return;
    const float z = x[gid] * args.scale;
    x[gid] = z * ds4_qwen4exp_sigmoid(z);
}

// Block input: mean over the hyper-connection streams of the gated normalized
// streams.  `wide` holds the RAW low-rank up projection; its sigmoid is fused
// here so the 10240-wide gate never has to be materialized twice.
//
//     out[t][d] = (1/n_hc) * sum_h sigmoid(wide[t][h][d]) * normed[t][h][d]
//
// The stream loop runs low to high so the summation order matches the host
// reference exactly.
// The grid carries the token, so the element index never has to be divided
// back into one: a flat launch cost every thread a division and a modulo,
// which is a long instruction sequence beside the four multiply-adds it
// guarded.
kernel void kernel_qwen4exp_hc_mix(
        constant ds4_metal_args_qwen4exp_hc & args,
        device  const float * normed,
        device  const float * wide,
        device        float * dst,
        uint2 gid [[thread_position_in_grid]]) {
    const uint32_t d = gid.x;
    const uint32_t t = gid.y;
    if (d >= args.n_embd || t >= args.n_tokens) return;

    const uint64_t row = (uint64_t)t * args.n_hc * args.n_embd + d;

    float acc = 0.0f;
    for (uint32_t h = 0; h < args.n_hc; ++h) {
        const uint64_t idx = row + (uint64_t)h * args.n_embd;
        acc += ds4_qwen4exp_sigmoid(wide[idx]) * normed[idx];
    }

    dst[(uint64_t)t * args.n_embd + d] = acc * args.inv_hc;
}

// Inject weights: one scalar per (token, stream).
//
//     inject[t][h] = 2 * sigmoid(dot(W[h], normed[t]) / n_hc)
//
// W is an [n_hc][n_hc*n_embd] tensor, small enough (160 KiB dense at the
// production shape) that a dedicated dot beats a general matvec dispatch.
// One threadgroup per (stream, token).
//
// Its type is not fixed: the TARGET stores it F32 and the MTP HEAD stores it
// Q8_0, and both must decode here.  The cases come from
// ds4_qwen4exp_hc_types.h, the same table the loader builds its accepted set
// from, and they reuse the accessors metal/qwen4exp_moe.metal defines -- which
// is why that file is concatenated ahead of this one.
static inline float ds4_qwen4exp_inject_value(
        uint type,
        device const char *row,
        uint k) {
#define DS4_QWEN4EXP_INJECT_CASE(name, id) \
    case (uint)(id): return ds4_qwen4exp_ ## name ## _value(row, k);
    switch (type) {
    DS4_QWEN4EXP_HC_INJECT_TYPES(DS4_QWEN4EXP_INJECT_CASE)
    }
#undef DS4_QWEN4EXP_INJECT_CASE
    return 0.0f;
}

kernel void kernel_qwen4exp_hc_inject_weights(
        constant ds4_metal_args_qwen4exp_hc_inject & args,
        device  const float * normed,
        device  const char * weight,
        device        float * dst,
        threadgroup   float * shmem [[threadgroup(0)]],
        uint3   tgpig [[threadgroup_position_in_grid]],
        uint3   tpitg [[thread_position_in_threadgroup]],
        ushort  sgitg [[simdgroup_index_in_threadgroup]],
        ushort  tiisg [[thread_index_in_simdgroup]],
        uint3   ntg   [[threads_per_threadgroup]]) {
    if (sgitg == 0) {
        shmem[tiisg] = 0.0f;
    }

    const uint32_t h = tgpig.x;
    const uint32_t t = tgpig.y;
    const uint32_t wide = args.n_hc * args.n_embd;

    device const float * xr = normed + (uint64_t)t * wide;
    device const char * wr = weight + (uint64_t)h * args.weight_row_bytes;

    float sumf = 0.0f;
    for (uint32_t i = tpitg.x; i < wide; i += ntg.x) {
        sumf += xr[i] * ds4_qwen4exp_inject_value(args.weight_type, wr, i);
    }
    sumf = simd_sum(sumf);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tiisg == 0) {
        shmem[sgitg] = sumf;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumf = shmem[tiisg];
    sumf = simd_sum(sumf);

    if (tpitg.x == 0) {
        dst[(uint64_t)t * args.n_hc + h] = 2.0f * ds4_qwen4exp_sigmoid(sumf * args.inv_hc);
    }
}

// Residual inject.  Structurally kernel_dsv4_hc_expand4's tail with the
// Sinkhorn combination matrix removed: qwen4exp's stream mixing is the
// identity, so each stream keeps its own residual and only picks up a scaled
// copy of the block output.
//
//     out[t][h][d] = residual[t][h][d] + block[t][d] * inject[t][h]
//
// Safe in place with out == residual.
// Stream and token off the grid for the same reason as the mixer above: a flat
// launch spent two divisions and two modulos per element on an expression that
// is one multiply and one add.
kernel void kernel_qwen4exp_hc_inject(
        constant ds4_metal_args_qwen4exp_hc & args,
        device  const float * residual,
        device  const float * block,
        device  const float * inject,
        device        float * dst,
        uint3 gid [[thread_position_in_grid]]) {
    const uint32_t d = gid.x;
    const uint32_t h = gid.y;
    const uint32_t t = gid.z;
    if (d >= args.n_embd || h >= args.n_hc || t >= args.n_tokens) return;

    const uint64_t i = ((uint64_t)t * args.n_hc + h) * args.n_embd + d;
    dst[i] = residual[i] +
             block[(uint64_t)t * args.n_embd + d] *
             inject[(uint64_t)t * args.n_hc + h];
}

// The partial rope kernel lives in metal/qwen4exp_qsa.metal: the QSA block and
// its indexer rope the same way, and one kernel means one theta.
