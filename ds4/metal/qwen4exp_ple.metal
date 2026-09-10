// Qwen4exp per-layer embedding (PLE) block kernels.
//
// The two ops the PLE block needs that no other qwen4exp block provides.
// Everything else it is built from already exists: the two projections are
// ds4_gpu_matmul_q8_0_tensor, the three norms are
// kernel_qwen4exp_rms_norm (grouped, one statistic per hyper-connection
// stream), and the row gather is host side, because the 26.8 GiB n-gram table
// stays on the solid-state disk and only the gathered rows reach the device.
//
// Reference: Qwen4ExpPLELayer in
// Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen4ExpNGram.swift, and the
// `stream = stream + ple(stream)` of Qwen4ExpDecoderLayer in Qwen4Exp.swift.
// The CUDA twin is the PLE section of ds4_cuda_qwen4exp.cu, and
// ds4_qwen4exp_ple_ref.h is the double reference both are checked against.
//
// Two things here have no counterpart anywhere else in this engine:
//
//   * The gate is a SIGNED SQUARE ROOT of the key/query inner product, with a
//     floor of 1e-6 inside the absolute value and the sign taken from the raw
//     product.  A plain sqrt of the absolute value flips every negative gate;
//     a sqrt without the floor is the same function here but its derivative is
//     not, and the floor is what the reference writes.
//   * The convolution is DILATED.  Its four taps sit `dilation` rows apart,
//     dilation is the n-gram size (3), so the rolling state is
//     (kernel - 1) * dilation = 9 rows and not 3.  A conv that walks adjacent
//     rows reads the right number of taps from the wrong rows and still
//     produces plausible values.

struct qwen4exp_ple_gate_args {
    uint  n_embd;
    uint  n_hc;
    uint  n_tokens;
    float inv_sqrt_embd;
};

struct qwen4exp_ple_conv_args {
    uint channels;
    uint conv_kernel;
    uint dilation;
    uint state_len;    // (conv_kernel - 1) * dilation
    uint n_tokens;
    // Per-row state snapshots for the speculative cycle's rollback; see the
    // note on qwen4exp_gdn_args.  Zero on every serial forward.
    uint n_snapshot_rows;
};

static inline float qwen4exp_ple_sigmoid(float z) {
    return 1.0f / (1.0f + exp(-z));
}

// sqrt(max(|v|, 1e-6)) * sign(v), the reference's `signed_sqrt`.
// sign(0) is 0, so a zero product gates to sigmoid(0) and not to
// sigmoid(1e-3).
static inline float qwen4exp_ple_signed_sqrt(float v) {
    const float magnitude = sqrt(max(fabs(v), 1.0e-6f));
    return v > 0.0f ? magnitude : (v < 0.0f ? -magnitude : 0.0f);
}

// The gate.
//
//     g[t][h]    = signed_sqrt(sum_d key[t][h][d] * query[t][h][d] / sqrt(E))
//     out[t][h][d] = sigmoid(g[t][h]) * value[t][d]
//
// The value row is shared by every stream: it is the 2560-wide value
// projection broadcast across the four hyper-connection streams, which is
// `value[.ellipsis, .newAxis, 0...]` in the reference.
//
// One threadgroup owns one (token, stream), so the inner product and the
// write live in the same threadgroup and the kernel is safe in place with
// `out == key`: every lane finishes reading its slice of `key` before the
// barrier that releases the write loop.  The reduction tree is
// kernel_qwen4exp_hc_inject_weights's.
kernel void kernel_qwen4exp_ple_gate(
        constant qwen4exp_ple_gate_args & args,
        device  const float * key,
        device  const float * query,
        device  const float * value,
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

    const uint h = tgpig.x;
    const uint t = tgpig.y;
    if (h >= args.n_hc || t >= args.n_tokens) return;

    const ulong base = ((ulong)t * args.n_hc + h) * args.n_embd;
    device const float * kr = key + base;
    device const float * qr = query + base;
    device const float * vr = value + (ulong)t * args.n_embd;
    device       float * dr = dst + base;

    float sumf = 0.0f;
    for (uint i = tpitg.x; i < args.n_embd; i += ntg.x) {
        sumf += kr[i] * qr[i];
    }
    sumf = simd_sum(sumf);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tiisg == 0) {
        shmem[sgitg] = sumf;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumf = shmem[tiisg];
    sumf = simd_sum(sumf);

    const float gate =
        qwen4exp_ple_sigmoid(qwen4exp_ple_signed_sqrt(sumf * args.inv_sqrt_embd));

    // Every read of `key` above is complete once this barrier retires, so the
    // in-place write below cannot race a lane that is still reducing.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tpitg.x; i < args.n_embd; i += ntg.x) {
        dr[i] = gate * vr[i];
    }
}

// The dilated depthwise short convolution, its rolling state, and the residual
// add that closes the block.
//
//     full(i)      = i < S ? state[i][c] : conv_in[i - S][c]
//     out[t][c]    = silu(sum_k weight[c][k] * full(t + dilation * k))
//     hyper[t][c] += gated[t][c] + out[t][c]
//     state[j][c]  = full(n_tokens + j)
//
// with S = (conv_kernel - 1) * dilation.  `weight` is the checkpoint's
// [channels][conv_kernel] tensor, taps contiguous, the layout
// kernel_qwen4exp_gdn_conv already reads.  Tap `conv_kernel - 1` multiplies
// the current row.
//
// One thread owns one channel and walks the tokens in order, so the state is
// carried in the buffer and never in a barrier.  The state write is last and
// ascends: it writes index j while it still has to read index n_tokens + j,
// and n_tokens >= 1 makes the read index strictly larger than every index
// already written, so no temporary is needed.
kernel void kernel_qwen4exp_ple_conv(
        constant qwen4exp_ple_conv_args & args,
        device        float * hyper,
        device        float * state,
        device  const float * gated,
        device  const float * conv_in,
        device  const float * weight,
        device        float * snapshot,
        uint gid [[thread_position_in_grid]]) {
    const uint c = gid;
    if (c >= args.channels) return;

    const uint C = args.channels;
    const uint S = args.state_len;

    for (uint t = 0; t < args.n_tokens; t++) {
        float acc = 0.0f;
        for (uint k = 0; k < args.conv_kernel; k++) {
            const uint i = t + args.dilation * k;
            const float v = (i < S) ? state[(ulong)i * C + c]
                                    : conv_in[(ulong)(i - S) * C + c];
            acc = fma(v, weight[(ulong)c * args.conv_kernel + k], acc);
        }
        const ulong index = (ulong)t * C + c;
        hyper[index] += gated[index] + acc * qwen4exp_ple_sigmoid(acc);
    }

    /* The rolling window as it stands after each of the first
     * `n_snapshot_rows` tokens.  This runs BEFORE the state write below,
     * because `full()` reads the incoming state and the write below clobbers
     * it -- the same read-before-write ordering the state write itself
     * depends on, one step earlier. */
    for (uint t = 0; t < args.n_snapshot_rows; t++) {
        device float *slot = snapshot + (ulong)t * S * C;
        for (uint j = 0; j < S; j++) {
            const uint i = t + 1u + j;
            slot[(ulong)j * C + c] = (i < S) ? state[(ulong)i * C + c]
                                             : conv_in[(ulong)(i - S) * C + c];
        }
    }

    for (uint j = 0; j < S; j++) {
        const uint i = args.n_tokens + j;
        state[(ulong)j * C + c] = (i < S) ? state[(ulong)i * C + c]
                                          : conv_in[(ulong)(i - S) * C + c];
    }
}
