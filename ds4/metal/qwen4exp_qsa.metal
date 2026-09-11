// Qwen4-Exp (Qwen 3.8 Flash-Next) QSA block: fused qkv split, per-head RMS
// norm, partial rope, indexer pooling/scoring/selection, and sparse attention.
//
// Semantics follow the MLX runner (`Qwen4ExpText.swift`,
// `Qwen4ExpQSAIndexer` and `Qwen4ExpAttention`). The pieces that are easy to
// get wrong, all of them load-bearing:
//
//   * `q_proj` is DOUBLED. The fused row holds, per query head, the query
//     vector immediately followed by that head's output gate. Splitting the
//     row into two flat halves instead of per-head pairs silently permutes
//     both.
//   * Query and key are RMS-normalized per head BEFORE rope. Value is not
//     normalized.
//   * Rope rotates the LEADING `rot_dim` entries of a head (64 of 256 for
//     attention, 64 of 128 for the indexer), half-split NeoX, base 1e7.
//     Upstream's `rope_tail_*` rotates the trailing half and cannot be reused.
//   * The indexer pools its EXACT, UNROTATED key tape by an fp32 mean over
//     `pool_size` tokens, norms the pooled block, and ropes it at position
//     `pool_size * block`.
//   * Block visibility is an INTEGER block count `(pos + 1) / pool_size`.
//     True division would admit the query's own incomplete block, so the query
//     would attend to its own future.
//   * The keep set is `blocks OR own`: the selected blocks' tokens plus the
//     tail of the query's own incomplete block. Without `own` a query whose
//     past holds fewer than `pool_size` tokens gets an all-masked row.
//
// PRODUCTION ROUNDING SEAMS. These kernels compute in f32 end to end. MLX runs
// the same graph in bf16 and rounds at three points: the pooled mean cast back
// to the tape dtype, the normalized value before the norm weight multiply, and
// the rope cos/sin cast before the rotation. The differences sit inside the
// tolerance band the design gives these ops, but a bit-exactness claim against
// MLX has to reproduce them.

// Masked-score sentinel. The Metal library is compiled with the default
// options, which enable fast math, so infinities are not dependable inside a
// kernel: a finite sentinel keeps the argsort, the selection filter and the
// softmax rescale on defined ground. Mirrored by
// DS4_QWEN4EXP_QSA_MASKED_SCORE in ds4_gpu.h.
#define QWEN4EXP_QSA_MASKED_SCORE (-3.0e38f)
#define QWEN4EXP_QSA_MASKED_LIMIT (-1.0e30f)

struct ds4_metal_args_qwen4exp_qsa_split_qkv {
    uint n_tokens;
    uint n_head;
    uint n_kv_head;
    uint head_dim;
};

struct ds4_metal_args_qwen4exp_head_rms_norm {
    uint n_rows;
    uint head_dim;
    float eps;
    float weight_offset;
};

struct ds4_metal_args_qwen4exp_rope_head {
    uint n_tokens;
    uint n_head;
    uint head_dim;
    uint rot_dim;
    uint pos0;
};

struct ds4_metal_args_qwen4exp_qsa_tape_append {
    uint n_tokens;
    uint head_dim;
    uint pos0;
    uint cache_cap;
};

struct ds4_metal_args_qwen4exp_qsa_pool_update {
    uint block0;
    uint n_blocks;
    uint head_dim;
    uint pool_size;
    uint rot_dim;
    uint cache_cap;
    float eps;
    float weight_offset;
};

struct ds4_metal_args_qwen4exp_qsa_indexer_scores {
    uint n_tokens;
    uint n_blocks;
    uint n_head;
    uint head_dim;
    uint pos0;
    uint pool_size;
    float norm_divisor;
};

struct ds4_metal_args_qwen4exp_qsa_indexer_select {
    uint n_tokens;
    uint n_blocks;
    uint top_k;
    uint sort_width;
    uint pos0;
    uint pool_size;
    uint max_selected;
};

struct ds4_metal_args_qwen4exp_qsa_attention {
    uint n_tokens;
    uint n_head;
    uint n_kv_head;
    uint head_dim;
    uint pos0;
    uint cache_cap;
    uint max_selected;
    uint sparse;
    float scale;
};

// Rotate the leading `rot_dim` entries of `vec` in place, half-split NeoX.
//
// The inverse frequencies arrive as a table rather than an in-kernel
// `exp(2*pair * -log(base)/rot_dim)`. A one-ulp disagreement between the
// device's exp and the host's is multiplied by the position, so within a few
// thousand tokens the in-kernel form drifts past the rope band and past the
// indexer's score gaps. `ds4_gpu_qwen4exp_rope_inv_freq` builds the table
// once, in double precision, for every consumer.
static inline void qwen4exp_rope_head_vec(
        threadgroup float *vec,
        device const float *inv_freq,
        uint tid,
        uint nth,
        uint rot_dim,
        uint pos) {
    const uint rot_half = rot_dim / 2u;
    for (uint d = tid; d < rot_half; d += nth) {
        const float theta = (float)pos * inv_freq[d];
        const float c = metal::cos(theta);
        const float s = metal::sin(theta);
        const float x1 = vec[d];
        const float x2 = vec[d + rot_half];
        vec[d]        = x1 * c - x2 * s;
        vec[d + rot_half] = x2 * c + x1 * s;
    }
}

// Sum `scratch[0 .. nth)` into `scratch[0]`. `nth` must be a power of two.
static inline float qwen4exp_tg_sum(threadgroup float *scratch, uint tid, uint nth) {
    for (uint step = nth >> 1; step > 0u; step >>= 1) {
        metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        if (tid < step) scratch[tid] += scratch[tid + step];
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    return scratch[0];
}

static inline float qwen4exp_tg_max(threadgroup float *scratch, uint tid, uint nth) {
    for (uint step = nth >> 1; step > 0u; step >>= 1) {
        metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        if (tid < step) scratch[tid] = metal::max(scratch[tid], scratch[tid + step]);
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    return scratch[0];
}

/*
 * Split the fused `attn_qkv` projection.
 *
 * Input row layout, per token:
 *     [ (q_h | gate_h) x n_head ] [ k x n_kv_head ] [ v x n_kv_head ]
 *
 * The query half is head-major with the gate interleaved per head, which is
 * what `qProj(x).reshaped(B, S, heads, -1).split(parts: 2, axis: -1)` means.
 */
kernel void kernel_qwen4exp_qsa_split_qkv(
        constant ds4_metal_args_qwen4exp_qsa_split_qkv &args,
        device const float *fused,
        device float *q,
        device float *gate,
        device float *k,
        device float *v,
        uint gid [[thread_position_in_grid]]) {
    const uint q_width = args.n_head * args.head_dim;
    const uint kv_width = args.n_kv_head * args.head_dim;
    const uint width = q_width + 2u * kv_width;
    if (gid >= args.n_tokens * width) return;

    const uint token = gid / width;
    const uint lane = gid % width;
    const uint fused_stride = 2u * q_width + 2u * kv_width;
    device const float *row = fused + (ulong)token * fused_stride;

    if (lane < q_width) {
        const uint head = lane / args.head_dim;
        const uint d = lane % args.head_dim;
        const uint base = head * 2u * args.head_dim;
        q[(ulong)token * q_width + lane] = row[base + d];
        gate[(ulong)token * q_width + lane] = row[base + args.head_dim + d];
        return;
    }

    const uint kv_lane = lane - q_width;
    if (kv_lane < kv_width) {
        k[(ulong)token * kv_width + kv_lane] = row[2u * q_width + kv_lane];
    } else {
        const uint vl = kv_lane - kv_width;
        v[(ulong)token * kv_width + vl] = row[2u * q_width + kv_width + vl];
    }
}

/*
 * Per-head RMS norm, `y = x * rsqrt(mean(x^2) + eps) * (weight_offset + w)`.
 * One threadgroup per (row, head); `n_rows` counts head vectors, not tokens.
 */
kernel void kernel_qwen4exp_head_rms_norm(
        constant ds4_metal_args_qwen4exp_head_rms_norm &args,
        device const float *x,
        device const float *weight,
        device float *out,
        threadgroup float *scratch [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint3 ntg [[threads_per_threadgroup]]) {
    const uint nth = ntg.x;
    const uint row = tgpig.x;
    if (row >= args.n_rows) return;
    device const float *src = x + (ulong)row * args.head_dim;
    device float *dst = out + (ulong)row * args.head_dim;

    float partial = 0.0f;
    for (uint d = tid; d < args.head_dim; d += nth) {
        const float value = src[d];
        partial += value * value;
    }
    scratch[tid] = partial;
    const float sum = qwen4exp_tg_sum(scratch, tid, nth);
    const float inv = metal::rsqrt(sum / (float)args.head_dim + args.eps);

    for (uint d = tid; d < args.head_dim; d += nth) {
        dst[d] = src[d] * inv * (args.weight_offset + weight[d]);
    }
}

/*
 * Partial rope over the leading `rot_dim` entries of every head vector.
 * Position of row `t` is `pos0 + t`, shared by every head.
 */
kernel void kernel_qwen4exp_rope_head(
        constant ds4_metal_args_qwen4exp_rope_head &args,
        device float *x,
        device const float *inv_freq,
        uint gid [[thread_position_in_grid]]) {
    const uint rot_half = args.rot_dim / 2u;
    const uint per_token = args.n_head * rot_half;
    if (gid >= args.n_tokens * per_token) return;

    const uint token = gid / per_token;
    const uint lane = gid % per_token;
    const uint head = lane / rot_half;
    const uint d = lane % rot_half;

    device float *vec = x + ((ulong)token * args.n_head + head) * args.head_dim;
    const uint pos = args.pos0 + token;
    const float theta = (float)pos * inv_freq[d];
    const float c = metal::cos(theta);
    const float s = metal::sin(theta);
    const float x1 = vec[d];
    const float x2 = vec[d + rot_half];
    vec[d]        = x1 * c - x2 * s;
    vec[d + rot_half] = x2 * c + x1 * s;
}

// Append raw indexer keys to the exact tape. The tape is never pooled,
// normalized or rotated in place: the pooling pass reads it fresh.
kernel void kernel_qwen4exp_qsa_tape_append(
        constant ds4_metal_args_qwen4exp_qsa_tape_append &args,
        device const float *raw_k,
        device float *tape,
        uint gid [[thread_position_in_grid]]) {
    if (gid >= args.n_tokens * args.head_dim) return;
    const uint token = gid / args.head_dim;
    const uint d = gid % args.head_dim;
    const uint pos = args.pos0 + token;
    if (pos >= args.cache_cap) return;
    tape[(ulong)pos * args.head_dim + d] = raw_k[gid];
}

/*
 * Build the pooled block cache for blocks `[block0, block0 + n_blocks)`.
 *
 * A block is the fp32 mean of `pool_size` consecutive raw tape rows, then
 * `k_layernorm`, then partial rope at position `pool_size * block`. A block
 * only becomes poolable once all of its rows are on the tape, so completed
 * blocks never change and the host recomputes only the newly complete range.
 */
kernel void kernel_qwen4exp_qsa_pool_update(
        constant ds4_metal_args_qwen4exp_qsa_pool_update &args,
        device const float *tape,
        device const float *weight,
        device const float *inv_freq,
        device float *pool,
        threadgroup float *shared [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint3 ntg [[threads_per_threadgroup]]) {
    const uint nth = ntg.x;
    const uint slot = tgpig.x;
    if (slot >= args.n_blocks) return;
    const uint block = args.block0 + slot;
    if ((block + 1u) * args.pool_size > args.cache_cap) return;

    threadgroup float *vec = shared;
    threadgroup float *scratch = shared + args.head_dim;

    for (uint d = tid; d < args.head_dim; d += nth) {
        float acc = 0.0f;
        for (uint j = 0; j < args.pool_size; j++) {
            acc += tape[(ulong)(block * args.pool_size + j) * args.head_dim + d];
        }
        vec[d] = acc / (float)args.pool_size;
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    float partial = 0.0f;
    for (uint d = tid; d < args.head_dim; d += nth) {
        partial += vec[d] * vec[d];
    }
    scratch[tid] = partial;
    const float sum = qwen4exp_tg_sum(scratch, tid, nth);
    const float inv = metal::rsqrt(sum / (float)args.head_dim + args.eps);

    for (uint d = tid; d < args.head_dim; d += nth) {
        vec[d] = vec[d] * inv * (args.weight_offset + weight[d]);
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    qwen4exp_rope_head_vec(vec, inv_freq, tid, nth, args.rot_dim,
                           block * args.pool_size);
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    device float *dst = pool + (ulong)block * args.head_dim;
    for (uint d = tid; d < args.head_dim; d += nth) {
        dst[d] = vec[d];
    }
}

/*
 * Block scores: `sum over index heads of relu(q . k)`, divided once by
 * `sqrt(head_dim)`. Blocks the query cannot see score
 * QWEN4EXP_QSA_MASKED_SCORE, which is what the selection pass reads to drop
 * them.
 *
 * The visibility rule is upstream's `glm_indexer_batch_visible_rows`
 * (`metal/dsv4_misc.metal`): an INTEGER count of complete blocks.
 */
kernel void kernel_qwen4exp_qsa_indexer_scores(
        constant ds4_metal_args_qwen4exp_qsa_indexer_scores &args,
        device const float *q,
        device const float *pool,
        device float *scores,
        threadgroup float *scratch [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint3 ntg [[threads_per_threadgroup]]) {
    const uint nth = ntg.x;
    const uint block = tgpig.x;
    const uint token = tgpig.y;
    if (block >= args.n_blocks || token >= args.n_tokens) return;

    device float *dst = scores + (ulong)token * args.n_blocks + block;
    const uint visible = metal::min((args.pos0 + token + 1u) / args.pool_size,
                                    args.n_blocks);
    if (block >= visible) {
        if (tid == 0) *dst = QWEN4EXP_QSA_MASKED_SCORE;
        return;
    }

    device const float *k = pool + (ulong)block * args.head_dim;
    float total = 0.0f;
    for (uint h = 0; h < args.n_head; h++) {
        device const float *qh = q +
            ((ulong)token * args.n_head + h) * args.head_dim;
        float partial = 0.0f;
        for (uint d = tid; d < args.head_dim; d += nth) {
            partial += qh[d] * k[d];
        }
        scratch[tid] = partial;
        const float dot = qwen4exp_tg_sum(scratch, tid, nth);
        total += metal::max(dot, 0.0f);
    }
    if (tid == 0) *dst = total / args.norm_divisor;
}

/*
 * Turn the block top-k into an ascending token id list.
 *
 * TIE ORDER. `topk` comes from `ds4_gpu_indexer_topk_tensor`, a descending
 * bitonic argsort: blocks with equal scores come out in an order the sorting
 * network fixes, not in index order, and MLX's `argPartition` gives no order
 * at all. Neither is a set difference, so this pass sorts the surviving block
 * ids ASCENDING and emits their tokens in that order. The output is therefore
 * independent of the tie order among selected blocks; only WHICH of several
 * equal-scored blocks lands inside the budget can differ, and the design
 * treats that as a set-equality question.
 *
 * A masked score means the block was never visible; dropping it is MLX's
 * `takeAlong(visible, top)` filter.
 */
kernel void kernel_qwen4exp_qsa_indexer_select(
        constant ds4_metal_args_qwen4exp_qsa_indexer_select &args,
        device const float *scores,
        device const int *topk,
        device int *selected,
        device int *counts,
        threadgroup int *ids [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint3 ntg [[threads_per_threadgroup]]) {
    const uint nth = ntg.x;
    const uint token = tgpig.x;
    if (token >= args.n_tokens) return;

    const int sentinel = 0x7fffffff;
    for (uint i = tid; i < args.sort_width; i += nth) {
        int block = sentinel;
        if (i < args.top_k) {
            const int candidate = topk[(ulong)token * args.top_k + i];
            if (candidate >= 0 && (uint)candidate < args.n_blocks) {
                const float score = scores[(ulong)token * args.n_blocks + (uint)candidate];
                if (score > QWEN4EXP_QSA_MASKED_LIMIT) block = candidate;
            }
        }
        ids[i] = block;
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    // Ascending bitonic sort over `sort_width` (a power of two >= top_k).
    for (uint k = 2u; k <= args.sort_width; k <<= 1) {
        for (uint j = k >> 1; j > 0u; j >>= 1) {
            for (uint i = tid; i < args.sort_width; i += nth) {
                const uint ixj = i ^ j;
                if (ixj > i) {
                    const bool ascending = (i & k) == 0u;
                    if ((ascending && ids[i] > ids[ixj]) ||
                        (!ascending && ids[i] < ids[ixj])) {
                        const int tmp = ids[i];
                        ids[i] = ids[ixj];
                        ids[ixj] = tmp;
                    }
                }
            }
            metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        }
    }

    // Surviving ids now occupy a prefix. Count them.
    threadgroup uint n_valid;
    if (tid == 0) {
        uint valid = 0;
        while (valid < args.top_k && ids[valid] != sentinel) valid++;
        n_valid = valid;
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    const uint m = n_valid;
    const uint block_tokens = m * args.pool_size;
    const uint pos = args.pos0 + token;
    // Integer block count; see the file header.
    const uint complete = (pos + 1u) / args.pool_size;
    const uint own_start = complete * args.pool_size;
    const uint own_count = pos + 1u - own_start;
    const uint total = block_tokens + own_count;

    device int *dst = selected + (ulong)token * args.max_selected;
    for (uint i = tid; i < args.max_selected; i += nth) {
        if (i < block_tokens) {
            dst[i] = ids[i / args.pool_size] * (int)args.pool_size +
                     (int)(i % args.pool_size);
        } else if (i < total) {
            dst[i] = (int)(own_start + (i - block_tokens));
        } else {
            dst[i] = -1;
        }
    }
    if (tid == 0) counts[token] = (int)total;
}

/*
 * Attention over the selected key set, f32 softmax.
 *
 * One threadgroup owns one (token, query head) and streams the key set in
 * tiles of `nth`. Every thread owns one output channel, so the value
 * accumulation needs no reduction; only the per-tile max and sum reduce.
 * The summation order is fixed by the tiling, so repeated runs are bit-exact.
 *
 * `sparse == 0` runs the dense causal set `[0, pos]`, which is what the
 * indexer asks for while the visible context still fits the token budget.
 */
kernel void kernel_qwen4exp_qsa_attention(
        constant ds4_metal_args_qwen4exp_qsa_attention &args,
        device const float *q,
        device const float *k_cache,
        device const float *v_cache,
        device const int *selected,
        device const int *counts,
        device float *out,
        threadgroup float *shared [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint3 ntg [[threads_per_threadgroup]]) {
    const uint nth = ntg.x;
    const uint head = tgpig.x;
    const uint token = tgpig.y;
    if (head >= args.n_head || token >= args.n_tokens) return;

    threadgroup float *qvec = shared;
    threadgroup float *tile = qvec + args.head_dim;   // reduction scratch
    threadgroup float *probs = tile + nth;            // tile probabilities
    threadgroup int *keys = (threadgroup int *)(probs + nth);

    const uint pos = args.pos0 + token;
    const uint count = args.sparse ? (uint)counts[token] : pos + 1u;
    const uint kv_head = head / (args.n_head / args.n_kv_head);
    const uint kv_stride = args.n_kv_head * args.head_dim;

    device const float *qsrc = q + ((ulong)token * args.n_head + head) * args.head_dim;
    for (uint d = tid; d < args.head_dim; d += nth) qvec[d] = qsrc[d];
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    device float *dst = out + ((ulong)token * args.n_head + head) * args.head_dim;
    if (count == 0u) {
        for (uint d = tid; d < args.head_dim; d += nth) dst[d] = 0.0f;
        return;
    }

    float run_max = QWEN4EXP_QSA_MASKED_SCORE;
    float run_sum = 0.0f;
    float acc = 0.0f;   // output channel `tid`

    for (uint base = 0; base < count; base += nth) {
        const uint n_in_tile = metal::min(nth, count - base);
        int key = -1;
        float score = QWEN4EXP_QSA_MASKED_SCORE;
        if (tid < n_in_tile) {
            key = args.sparse ? selected[(ulong)token * args.max_selected + base + tid]
                              : (int)(base + tid);
            if (key >= 0 && (uint)key < args.cache_cap) {
                device const float *kv = k_cache +
                    (ulong)key * kv_stride + (ulong)kv_head * args.head_dim;
                float dot = 0.0f;
                for (uint d = 0; d < args.head_dim; d++) dot += qvec[d] * kv[d];
                score = dot * args.scale;
            } else {
                key = -1;
            }
        }
        keys[tid] = key;
        tile[tid] = score;
        const float tile_max = qwen4exp_tg_max(tile, tid, nth);
        const float new_max = metal::max(run_max, tile_max);
        metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);

        probs[tid] = (key >= 0) ? metal::exp(score - new_max) : 0.0f;
        tile[tid] = probs[tid];
        const float tile_sum = qwen4exp_tg_sum(tile, tid, nth);
        const float rescale = (run_max > QWEN4EXP_QSA_MASKED_LIMIT)
            ? metal::exp(run_max - new_max) : 0.0f;
        run_sum = run_sum * rescale + tile_sum;

        if (tid < args.head_dim) {
            float contrib = 0.0f;
            for (uint j = 0; j < n_in_tile; j++) {
                const int kj = keys[j];
                if (kj < 0) continue;
                device const float *vv = v_cache +
                    (ulong)kj * kv_stride + (ulong)kv_head * args.head_dim;
                contrib += probs[j] * vv[tid];
            }
            acc = acc * rescale + contrib;
        }
        run_max = new_max;
        metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    }

    if (tid < args.head_dim) {
        dst[tid] = (run_sum > 0.0f) ? acc / run_sum : 0.0f;
    }
}

// `out * sigmoid(gate)`, the QSA output gate that rides in the doubled
// `q_proj`. Kept separate from the attention kernel so the gate can be tested
// against the reference on its own.
kernel void kernel_qwen4exp_qsa_output_gate(
        constant uint &n_values,
        device const float *gate,
        device float *out,
        uint gid [[thread_position_in_grid]]) {
    if (gid >= n_values) return;
    out[gid] = out[gid] * (1.0f / (1.0f + metal::exp(-gate[gid])));
}

/*
 * Split the DOUBLED query projection when q, k and v are separate tensors.
 *
 * The loader binds `blk.N.attn_q.weight` (2560 -> 12288), `attn_k` and
 * `attn_v` separately, so one matmul per tensor already lays k and v out the
 * way the attention kernel wants them and only the query needs splitting.
 * The row is head-major with the gate interleaved per head, the same shape
 * kernel_qwen4exp_qsa_split_qkv reads out of a fused row:
 *
 *     [ (q_h | gate_h) x n_head ]
 */
kernel void kernel_qwen4exp_qsa_split_doubled_q(
        constant ds4_metal_args_qwen4exp_qsa_split_qkv &args,
        device const float *doubled,
        device float *q,
        device float *gate,
        uint gid [[thread_position_in_grid]]) {
    const uint q_width = args.n_head * args.head_dim;
    if (gid >= args.n_tokens * q_width) return;

    const uint token = gid / q_width;
    const uint lane = gid % q_width;
    const uint head = lane / args.head_dim;
    const uint d = lane % args.head_dim;

    device const float *row = doubled + (ulong)token * 2u * q_width;
    const uint base = head * 2u * args.head_dim;
    q[(ulong)token * q_width + lane] = row[base + d];
    gate[(ulong)token * q_width + lane] = row[base + args.head_dim + d];
}
