// Qwen4exp gated delta net (GDN), forked from the GLM-5.3 KDA kernels above.
//
// Differences from Kimi Delta Attention, all of them semantic:
//
//   * Grouped heads.  KDA gives q, k and v the same head count.  GDN has
//     `n_key_head` query/key heads and `n_value_head` value heads, each of
//     width 128, and which key head a value head reads depends on the
//     checkpoint's value-head order: `hv / (n_value_head / n_key_head)` in
//     the grouped order the reference model stores, `hv % n_key_head` in the
//     tiled order llama.cpp's converter writes.  `head_layout` selects it.
//   * One convolution.  KDA convolves q, k and v with three weights over
//     three histories.  GDN runs a single depthwise 4-tap convolution over
//     the fused projection (`q | k | v`, `conv_dim` channels) with one
//     weight and one 3-row history, exactly as the checkpoint stores it.
//   * Scalar decay.  KDA's decay is per channel and floored:
//     `exp(lower_bound * sigmoid(exp(a_log) * (gate + dt_bias)))`.  GDN's is
//     one value per token and value head with no floor:
//     `exp(ssm_a * softplus(alpha + dt_bias))`, with ssm_a ALREADY holding
//     -exp(A_log) from the converter -- KDA's tensor is a raw A_log and this
//     one is not, which is the trap this port fell into.
//   * Query and key normalisation.  Both engines end with unit-L2 keys and
//     queries scaled by 2^-3.5, and both put the epsilon on the SUM of
//     squares: `x * rsqrt(sum(x^2) + 1e-6)`.  This kernel used to divide the
//     sum by D first, which is the same expression with an epsilon 128 times
//     too large and moves a small key by more than a factor of two.
//
// Convolution, recurrent state and both reductions stay FP32, so a sequence
// split into chunks reproduces the single-chunk result bit for bit.

struct qwen4exp_gdn_args {
    uint  n_key_head;
    uint  n_value_head;
    uint  n_rows;
    uint  n_tokens;
    uint  head_layout;   /* 0 grouped, 1 tiled; see ds4_gpu.h */
    /* PER-ROW STATE SNAPSHOTS, for the speculative cycle's rollback.
     *
     * When this is non-zero the kernels mirror the carried state into
     * `snapshot` after each of the first `n_snapshot_rows` tokens, so a round
     * that accepts `a` of its drafts can adopt the state as it stood after row
     * `a` instead of rewinding and running a shorter forward again.  The
     * recurrence is token-serial in registers, so the state after row k IS the
     * state a (k + 1)-row feed leaves -- selecting it and replaying it are the
     * same value, which is what makes the shortcut exact rather than close.
     *
     * Zero on every serial forward, where the branch never fires and the
     * arithmetic is untouched. */
    uint  n_snapshot_rows;
    float qk_norm_eps;
    float norm_eps;
};

static inline float qwen4exp_gdn_silu(float x) {
    return x / (1.0f + exp(-x));
}

static inline float qwen4exp_gdn_sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

/* log(1 + exp(x)) through the branch MLX's `softplus` takes: `logaddexp(x, 0)`
 * evaluates the exponential of the negative magnitude, so a large positive
 * gate cannot overflow.  The Metal standard library has no `log1p`; the
 * argument it would sharpen is only tiny where the whole term is negligible
 * beside `max(x, 0)`. */
static inline float qwen4exp_gdn_softplus(float x) {
    return max(x, 0.0f) + log(1.0f + exp(-abs(x)));
}

/*
 * Depthwise 4-tap causal convolution, SiLU, and the query/key RMS norm.
 *
 * One threadgroup owns one (row, 128-channel block); each thread owns one
 * channel and carries that channel's 3-row history in registers, so the pass
 * is token-serial per channel and needs no scratch buffer: a thread reads
 * `qkv[token][channel]` before it overwrites the same element.  Blocks below
 * `2 * n_key_head` are query and key heads and take the RMS norm; the rest
 * are value heads and only take the activation.
 */
kernel void kernel_qwen4exp_gdn_conv(
        constant qwen4exp_gdn_args &args,
        device float         *qkv,
        device float         *conv_state,
        device const float   *conv_weight,
        device float         *conv_snapshot,
        threadgroup float    *scratch [[threadgroup(0)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint D = 128u;
    constexpr uint HISTORY = 3u;
    const uint block = tgpig.x;
    const uint row = tgpig.y;
    const uint key_blocks = 2u * args.n_key_head;
    const uint blocks = key_blocks + args.n_value_head;
    if (block >= blocks || row >= args.n_rows) return;

    /* Two reduction slots, alternating by token.  One barrier a token then
     * bounds the skew between simdgroups at one token, and the simdgroup that
     * has run ahead writes the slot the one behind is not reading. */
    threadgroup float *reduce = scratch;
    const uint conv_dim = blocks * D;
    const uint channel = block * D + tid;
    const bool is_key = block < key_blocks;
    /* The reference l2-normalises the row -- `x * rsqrt(sum(x^2) + eps)`, the
     * epsilon on the SUM -- and then scales the query by `head_dim ** -0.5`.
     * So the key takes no post scale at all and the query takes 2^-3.5. */
    const float post_scale = block < args.n_key_head
        ? 0x1.6a09e6p-4f
        : 1.0f;

    device float *history = conv_state + (ulong)row * HISTORY * conv_dim;
    float h0 = history[channel];
    float h1 = history[conv_dim + channel];
    float h2 = history[2ul * conv_dim + channel];
    const float w0 = conv_weight[(ulong)channel * 4u + 0u];
    const float w1 = conv_weight[(ulong)channel * 4u + 1u];
    const float w2 = conv_weight[(ulong)channel * 4u + 2u];
    const float w3 = conv_weight[(ulong)channel * 4u + 3u];

    float raw = qkv[(ulong)row * args.n_tokens * conv_dim + channel];
    for (uint token = 0; token < args.n_tokens; token++) {
        const ulong index =
            ((ulong)row * args.n_tokens + token) * conv_dim + channel;
        float acc = 0.0f;
        acc = fma(h0, w0, acc);
        acc = fma(h1, w1, acc);
        acc = fma(h2, w2, acc);
        acc = fma(raw, w3, acc);
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
        const ulong ahead =
            index + (token + 1u < args.n_tokens ? conv_dim : 0u);
        const float raw_next = qkv[ahead];

        /* The window as it stands AFTER this token, which is what a rollback
         * to length token + 1 needs.  Written before the key/value branch so
         * every thread in the threadgroup takes the same path. */
        if (token < args.n_snapshot_rows) {
            device float *slot = conv_snapshot +
                (ulong)token * HISTORY * conv_dim;
            slot[channel] = h0;
            slot[conv_dim + channel] = h1;
            slot[2ul * conv_dim + channel] = h2;
        }

        const float activated = qwen4exp_gdn_silu(acc);
        raw = raw_next;
        if (!is_key) {
            qkv[index] = activated;
            continue;
        }

        threadgroup float *red = reduce + 4u * (token & 1u);
        float sumsq = simd_sum(activated * activated);
        if (lane == 0u) red[sg] = sumsq;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float total = lane < 4u ? red[lane] : 0.0f;
        total = simd_sum(total);
        qkv[index] = activated *
            rsqrt(total + args.qk_norm_eps) * post_scale;
    }

    history[channel] = h0;
    history[conv_dim + channel] = h1;
    history[2ul * conv_dim + channel] = h2;
}

/*
 * The delta rule itself, token-serial inside the kernel like KDA and like the
 * reference: one threadgroup owns one (row, value head, four value rows), one
 * simdgroup owns one value row, and each lane owns four adjacent key columns.
 */
kernel void kernel_qwen4exp_gdn_recurrence(
        constant qwen4exp_gdn_args &args,
        device const float   *qkv,
        device const float   *raw_alpha,
        device const float   *raw_beta,
        device const float   *a_log,
        device const float   *dt_bias,
        device float         *state,
        device float         *out,
        device float         *state_snapshot,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint D = 128u;
    const uint head = tgpig.x;
    const uint value = tgpig.y * 4u + sg;
    const uint row = tgpig.z;
    if (head >= args.n_value_head || value >= D || row >= args.n_rows) return;

    const uint key_dim = args.n_key_head * D;
    const uint value_dim = args.n_value_head * D;
    const uint conv_dim = 2u * key_dim + value_dim;
    /* Grouped order pairs value head hv with key head hv / repeats; the
     * converter's tiled order pairs it with hv % n_key_head. */
    const uint key_head = args.head_layout != 0u
        ? head % args.n_key_head
        : head / (args.n_value_head / args.n_key_head);
    const uint k0 = lane * 4u;

    device float4 *state_ptr = (device float4 *)(
        state + ((((ulong)row * args.n_value_head + head) * D) + value) * D +
        k0);
    float4 h = *state_ptr;
    /* ssm_a IS ALREADY -exp(A_log).  llama.cpp's converter stores the tensor
     * that way (conversion/qwen.py), and every value in the artifact is
     * negative -- blk.0 runs -157.985 to -0.0279 -- so exponentiating it again
     * is wrong twice over: it makes the coefficient positive-small and, for the
     * large-magnitude entries, indistinguishable from zero.  exp(-157.985) is
     * ~1e-69, which left the decay at 1.0 and the recurrent state undecayed.
     * The reference is gate = softplus(alpha + dt_bias) * ssm_a, then
     * exp(gate).  This is NOT the GLM 5.3 KDA convention it was inherited
     * from, where the tensor really is a raw A_log. */
    const float decay_coeff = a_log[head];
    const float bias = dt_bias[head];

    for (uint token = 0; token < args.n_tokens; token++) {
        const ulong slot = (ulong)row * args.n_tokens + token;
        const ulong base = slot * conv_dim + key_head * D;
        const float4 q4 = *((device const float4 *)(qkv + base + k0));
        const float4 k4 =
            *((device const float4 *)(qkv + base + key_dim + k0));
        const float v_row =
            qkv[slot * conv_dim + 2ul * key_dim + head * D + value];
        const ulong gate = slot * args.n_value_head + head;
        const float g = exp(decay_coeff *
            qwen4exp_gdn_softplus(raw_alpha[gate] + bias));
        const float beta = qwen4exp_gdn_sigmoid(raw_beta[gate]);

        h *= g;
        const float hk = simd_sum(dot(h, k4));
        const float delta_v = (v_row - hk) * beta;
        h = fma(k4, float4(delta_v), h);
        const float result = simd_sum(dot(h, q4));
        if (lane == 0u) out[slot * value_dim + head * D + value] = result;

        /* The recurrent state AFTER this token.  Same element this thread owns
         * in the live state, one slot per row. */
        if (token < args.n_snapshot_rows) {
            const ulong stride =
                (ulong)args.n_rows * args.n_value_head * D * D;
            device float4 *snap = (device float4 *)(
                state_snapshot + (ulong)token * stride +
                ((((ulong)row * args.n_value_head + head) * D) + value) * D +
                k0);
            *snap = h;
        }
    }
    *state_ptr = h;
}

/* Sigmoid-gated RMS output norm.  The weight is a plain scale, not an
 * offset-baked one, so it multiplies the normalised row directly. */
kernel void kernel_qwen4exp_gdn_output(
        constant qwen4exp_gdn_args &args,
        device float         *out,
        device const float   *output_gate,
        device const float   *output_norm,
        threadgroup float    *partial [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint D = 128u;
    const uint token = tgpig.x;
    const uint head = tgpig.y;
    const uint row = tgpig.z;
    if (token >= args.n_tokens || head >= args.n_value_head ||
        row >= args.n_rows) {
        return;
    }
    const uint value_dim = args.n_value_head * D;
    const ulong base =
        ((ulong)row * args.n_tokens + token) * value_dim + head * D;
    const float raw = out[base + tid];
    float sumsq = simd_sum(raw * raw);
    if (lane == 0u) partial[sg] = sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = lane < 4u ? partial[lane] : 0.0f;
    total = simd_sum(total);
    const float scale = rsqrt(total / (float)D + args.norm_eps);
    out[base + tid] = raw * scale * output_norm[tid] *
        qwen4exp_gdn_sigmoid(output_gate[base + tid]);
}
