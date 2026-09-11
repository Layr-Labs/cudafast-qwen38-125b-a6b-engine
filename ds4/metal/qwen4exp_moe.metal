// qwen4exp routed MoE: router, routed experts and the shared expert.
//
// The GLM router in metal/dsv4_misc.metal cannot be reused as it stands: it
// scores sigmoid(logit) + bias, then normalises the *unbiased* sigmoid
// probabilities over the selected set.  qwen4exp scores the raw float32
// logits, has no bias and takes a softmax over the selected logits only, so
// the order is branched here.  Everything below the router is shared with the
// GLM path: the K-quant element accessors come from metal/moe.metal, which is
// concatenated into the same library ahead of this file.

struct ds4_metal_qwen4exp_router_args {
    uint32_t n_expert;
    uint32_t n_expert_used;
    uint32_t n_tokens;
    uint32_t pad0;
};

struct ds4_metal_qwen4exp_moe_args {
    uint32_t in_dim;
    uint32_t mid_dim;
    uint32_t out_dim;
    uint32_t n_total_expert;
    uint32_t n_expert_used;
    uint32_t n_tokens;
    uint32_t mid_token_stride;
    uint32_t gate_type;
    uint32_t up_type;
    uint32_t down_type;
    uint32_t pad0;
    uint32_t pad1;
    uint64_t gate_expert_bytes;
    uint64_t gate_row_bytes;
    uint64_t up_expert_bytes;
    uint64_t up_row_bytes;
    uint64_t down_expert_bytes;
    uint64_t down_row_bytes;
};

// ggml block_q5_1: 32 elements in 24 bytes, f16 scale, f16 minimum, a 32-bit
// high-bit plane and 16 nibble bytes.  See ggml-quants.c
// dequantize_row_q5_1(): the low nibble of byte j carries element j and the
// high nibble carries element j + 16, each taking its fifth bit from bit j
// respectively bit j + 16 of qh.
struct block_q5_1 {
    half  d;
    half  m;
    uchar qh[4];
    uchar qs[16];
};

static inline float ds4_qwen4exp_q5_1_value(device const char *row, uint k) {
    device const block_q5_1 *blocks = (device const block_q5_1 *)row;
    const uint block = k / 32u;
    const uint idx = k - block * 32u;
    device const block_q5_1 *xb = blocks + block;
    const uint qh = (uint)xb->qh[0] | ((uint)xb->qh[1] << 8u) |
                    ((uint)xb->qh[2] << 16u) | ((uint)xb->qh[3] << 24u);
    const uint j = idx & 15u;
    uint q;
    if (idx < 16u) {
        q = ((uint)xb->qs[j] & 0x0Fu) | (((qh >> j) & 1u) << 4u);
    } else {
        q = ((uint)xb->qs[j] >> 4u) | (((qh >> (j + 16u)) & 1u) << 4u);
    }
    return (float)q * (float)xb->d + (float)xb->m;
}

static inline float ds4_qwen4exp_q8_0_value(device const char *row, uint k) {
    device const block_q8_0 *blocks = (device const block_q8_0 *)row;
    const uint block = k / 32u;
    const uint idx = k - block * 32u;
    device const block_q8_0 *xb = blocks + block;
    return (float)xb->d * (float)xb->qs[idx];
}

static inline float ds4_qwen4exp_f32_value(device const char *row, uint k) {
    return ((device const float *)row)[k];
}

// The K-quant accessors are metal/moe.metal's, unchanged, so a routed expert
// decodes exactly as the GLM expert path decodes the same block.
static inline float ds4_qwen4exp_q4_K_value(device const char *row, uint k) {
    return ds4_glm_q4_K_value((device const block_q4_K *)row, k);
}

static inline float ds4_qwen4exp_q5_K_value(device const char *row, uint k) {
    return ds4_glm_q5_K_value((device const block_q5_K *)row, k);
}

static inline float ds4_qwen4exp_q6_K_value(device const char *row, uint k) {
    return ds4_glm_q6_K_value((device const block_q6_K *)row, k);
}

// One scalar element of a quantised weight row.  The cases come from
// ds4_qwen4exp_moe_types.h, which the host prepends to this library and the
// loader expands into its accepted-type list, so every type that reaches here
// has a case and every case is a type the loader admits.
static inline float ds4_qwen4exp_weight_value(
        uint type,
        device const char *row,
        uint k) {
#define DS4_QWEN4EXP_VALUE_CASE(name, id) \
    case (uint)(id): return ds4_qwen4exp_ ## name ## _value(row, k);
    switch (type) {
    DS4_QWEN4EXP_MOE_TYPES(DS4_QWEN4EXP_VALUE_CASE)
    }
#undef DS4_QWEN4EXP_VALUE_CASE
    return 0.0f;
}

static inline bool ds4_qwen4exp_router_better(
        threadgroup const float *scores,
        int32_t                  a,
        int32_t                  b) {
    const float sa = scores[(uint)a];
    const float sb = scores[(uint)b];
    // Ties go to the lower expert index, so the ordering is a strict total
    // order and the bitonic network has one fixed point.
    return sa > sb || (sa == sb && a < b);
}

// Router for one token per threadgroup.  Selection is on the raw float32
// logits; the weights are a softmax over the selected logits only.
kernel void kernel_qwen4exp_router_select(
        constant ds4_metal_qwen4exp_router_args & args,
        device const float *logits,
        device int32_t *selected,
        device float *weights,
        threadgroup float *scratch [[threadgroup(0)]],
        uint token [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]]) {
    const uint sort_width = args.n_expert > 256u ? 512u : 256u;
    threadgroup float *sel_scores = scratch;
    threadgroup int32_t *idx = (threadgroup int32_t *)(scratch + sort_width);
    if (token >= args.n_tokens) return;

    device const float *token_logits = logits + (uint64_t)token * args.n_expert;
    device int32_t *token_selected = selected + (uint64_t)token * args.n_expert_used;
    device float *token_weights = weights + (uint64_t)token * args.n_expert_used;

    const uint n_expert = min(args.n_expert, 512u);
    const bool active = tid < n_expert;
    sel_scores[tid] = active ? token_logits[tid] : -INFINITY;
    idx[tid] = (int32_t)tid;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k = 2; k <= sort_width; k <<= 1) {
        for (uint j = k >> 1; j > 0; j >>= 1) {
            const uint other = tid ^ j;
            if (other > tid) {
                const int32_t a = idx[tid];
                const int32_t b = idx[other];
                const bool descending = (tid & k) == 0;
                const bool swap = descending
                    ? ds4_qwen4exp_router_better(sel_scores, b, a)
                    : ds4_qwen4exp_router_better(sel_scores, a, b);
                if (swap) {
                    idx[tid] = b;
                    idx[other] = a;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    const uint k_used = min(args.n_expert_used, n_expert);
    if (tid < k_used) token_selected[tid] = idx[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // k is ten, so one lane runs the softmax.  A serial reduction keeps the
    // summation order fixed and matches the float32 reference exactly.
    if (tid == 0u) {
        float m = -INFINITY;
        for (uint i = 0; i < k_used; i++) {
            m = max(m, sel_scores[(uint)token_selected[i]]);
        }
        float sum = 0.0f;
        for (uint i = 0; i < k_used; i++) {
            const float e = exp(sel_scores[(uint)token_selected[i]] - m);
            token_weights[i] = e;
            sum += e;
        }
        const float inv = 1.0f / sum;
        for (uint i = 0; i < k_used; i++) token_weights[i] *= inv;
    }
}



// ---------------------------------------------------------------------------
// Weight reuse across the rows of a prefill.
//
// The kernels above give one threadgroup to one (output row, token) pair, so a
// weight row is fetched and decoded once per TOKEN.  The kernels below decode a
// weight element ONCE and use it for a tile of rows.
//
// Each row keeps the same k walk (k = tid, tid + ntg, ...), the same float
// accumulator, and the same halving tree, so it computes what it computed
// alone: the tile is a memory decision, not a numeric one.
//
// The routed experts need one more step, because rows in a tile must share an
// expert or there is no weight to share.  The pair list below sorts the
// (token, slot) pairs by the expert they selected, and a threadgroup then owns
// one expert and walks its own pairs.  Sorting changes which rows sit beside
// each other, never what a row computes: mid[token][slot][row] depends on that
// pair alone.  Pairs are packed as token * n_expert_used + slot, which is the
// index `selected` and `weights` are already addressed by.


#define DS4_QWEN4EXP_MOE_TILE 8

struct ds4_metal_qwen4exp_group_args {
    uint32_t n_total_expert;
    uint32_t n_expert_used;
    uint32_t n_pairs;
    uint32_t mid_dim;
    uint32_t mid_token_stride;
    uint32_t in_dim;
    uint32_t gate_type;
    uint32_t up_type;
    uint32_t groups;
    uint32_t out_dim;
    uint32_t n_tokens;
    uint32_t down_type;
    uint64_t gate_expert_bytes;
    uint64_t gate_row_bytes;
    uint64_t up_expert_bytes;
    uint64_t up_row_bytes;
    uint64_t down_expert_bytes;
    uint64_t down_row_bytes;
};

struct ds4_metal_qwen4exp_quant_args {
    uint32_t width;
    uint32_t groups;
    uint32_t inner_count;
    uint32_t pad0;
    uint64_t outer_stride;
    uint64_t inner_stride;
};

kernel void kernel_qwen4exp_moe_group_reset(
        constant ds4_metal_qwen4exp_group_args &args,
        device uint32_t *counts,
        uint e [[thread_position_in_grid]]) {
    if (e < args.n_total_expert) counts[e] = 0u;
}

kernel void kernel_qwen4exp_moe_group_count(
        constant ds4_metal_qwen4exp_group_args &args,
        device const int32_t *selected,
        device atomic_uint *counts,
        uint p [[thread_position_in_grid]]) {
    if (p >= args.n_pairs) return;
    const int32_t e = selected[p];
    if (e < 0 || (uint)e >= args.n_total_expert) return;
    atomic_fetch_add_explicit(&counts[(uint)e], 1u, memory_order_relaxed);
}

// One thread: n_expert is 512, so the serial scan is a few hundred adds
// against the cost of dispatching anything cleverer.
kernel void kernel_qwen4exp_moe_group_scan(
        constant ds4_metal_qwen4exp_group_args &args,
        device const uint32_t *counts,
        device uint32_t *offsets,
        device uint32_t *cursor,
        device uint32_t *active,
        uint tid [[thread_position_in_grid]]) {
    if (tid != 0u) return;
    uint run = 0u;
    uint live = 0u;
    for (uint e = 0; e < args.n_total_expert; e++) {
        offsets[e] = run;
        cursor[e] = run;
        run += counts[e];
        // The experts this call actually chose, compacted.  A threadgroup of
        // the expert kernel indexes this list rather than the expert id, so a
        // narrow call launches one row per chosen expert instead of one per
        // expert that exists.
        if (counts[e] > 0u) active[1u + live++] = e;
    }
    active[0] = live;
}

kernel void kernel_qwen4exp_moe_group_scatter(
        constant ds4_metal_qwen4exp_group_args &args,
        device const int32_t *selected,
        device atomic_uint *cursor,
        device uint32_t *pairs,
        uint p [[thread_position_in_grid]]) {
    if (p >= args.n_pairs) return;
    const int32_t e = selected[p];
    if (e < 0 || (uint)e >= args.n_total_expert) return;
    const uint at = atomic_fetch_add_explicit(&cursor[(uint)e], 1u,
                                              memory_order_relaxed);
    pairs[at] = p;
}

// A pair whose expert id is out of range contributes nothing and its mid row
// reads zero, which is what the per-token kernel wrote for it.  The grouped
// kernel never visits such a pair, so the zero is written here.  The router
// cannot produce one; this keeps the two paths equal anyway.
kernel void kernel_qwen4exp_moe_zero_invalid(
        constant ds4_metal_qwen4exp_group_args &args,
        device const int32_t *selected,
        device float *mid,
        uint p [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]]) {
    if (p >= args.n_pairs) return;
    const int32_t e = selected[p];
    if (e >= 0 && (uint)e < args.n_total_expert) return;
    const uint token = p / args.n_expert_used;
    const uint slot = p - token * args.n_expert_used;
    device float *dst = mid + (uint64_t)token * args.mid_token_stride +
                        (uint64_t)slot * args.mid_dim;
    for (uint r = tid; r < args.mid_dim; r += 256u) dst[r] = 0.0f;
}


// ---------------------------------------------------------------------------
// ONE ARITHMETIC FOR THE EXPERT PROJECTIONS, AT EVERY WIDTH.
//
// The twin of ds4_cuda_qwen4exp.cu's contract, formula for formula.  The
// activation row is quantised to Q8_0 groups of 32 -- per row, so it never
// sees the batch -- and a weight row is walked in the same groups in ascending
// order, each group contributing
//
//     acc += (wa * xscale) * float(dot)     dot  = integer dot of the group
//     acc += (wb * xscale) * float(xsum)    xsum = integer sum of the group
//
// with wa and wb decoded ONCE for the group.  Both integer terms are exact, so
// this backend may compute them any way it likes and still get the same
// integers CUDA gets; only the two floating-point operations per group have to
// be the same expression, and they are.
//
// Q8_0 gives (d, 0), Q5_1 gives (d, m), Q4_K and Q5_K give (d*sc, -dmin*m).
// Q6_K stores its quant as q - 32 so its offset needs no second term, and it
// is the one format whose scale changes inside a group of 32 -- it reports two
// halves and contributes one term per half.
//
// Nothing here mentions the batch, and the kernels are one template over the
// row tile R with R = 1 the same code, so a one-row decode is the kernel
// rather than a second one that has to be kept in agreement.

static inline void ds4_qwen4exp_group_decode(
        uint type, device const char *row, uint g,
        thread char *wq, thread float *wa, thread float *wb,
        thread int *halves) {
    *halves = 1;
    wa[1] = 0.0f;
    wb[0] = 0.0f;
    wb[1] = 0.0f;
    switch (type) {
    case 8u: {                                  // q8_0
        device const block_q8_0 *xb = (device const block_q8_0 *)row + g;
        wa[0] = (float)xb->d;
        for (int i = 0; i < 32; i++) wq[i] = (char)xb->qs[i];
        return;
    }
    case 7u: {                                  // q5_1
        device const block_q5_1 *xb = (device const block_q5_1 *)row + g;
        const uint qh = (uint)xb->qh[0] | ((uint)xb->qh[1] << 8u) |
                        ((uint)xb->qh[2] << 16u) | ((uint)xb->qh[3] << 24u);
        wa[0] = (float)xb->d;
        wb[0] = (float)xb->m;
        for (int j = 0; j < 16; j++) {
            wq[j] = (char)(((uint)(xb->qs[j] & 0x0Fu)) | (((qh >> j) & 1u) << 4u));
            wq[16 + j] = (char)(((uint)(xb->qs[j] >> 4u)) |
                                (((qh >> (uint)(j + 16)) & 1u) << 4u));
        }
        return;
    }
    case 12u: {                                 // q4_K
        device const block_q4_K *xb = (device const block_q4_K *)row + (g / 8u);
        const uint grp = g % 8u;
        const uchar2 sm = get_scale_min_k4_just2((int)grp, 0, xb->scales);
        wa[0] = (float)xb->d * (float)sm.x;
        wb[0] = -(float)xb->dmin * (float)sm.y;
        device const uchar *qs = xb->qs + (grp >> 1u) * 32u;
        const uint shift = (grp & 1u) ? 4u : 0u;
        for (int i = 0; i < 32; i++) wq[i] = (char)(((uint)qs[i] >> shift) & 0x0Fu);
        return;
    }
    case 13u: {                                 // q5_K
        device const block_q5_K *xb = (device const block_q5_K *)row + (g / 8u);
        const uint grp = g % 8u;
        const uchar2 sm = get_scale_min_k4_just2((int)grp, 0, xb->scales);
        wa[0] = (float)xb->d * (float)sm.x;
        wb[0] = -(float)xb->dmin * (float)sm.y;
        device const uchar *qs = xb->qs + (grp >> 1u) * 32u;
        const uint shift = (grp & 1u) * 4u;
        for (int i = 0; i < 32; i++) {
            uint q = ((uint)qs[i] >> shift) & 0x0Fu;
            if (xb->qh[i] & (uchar)(1u << grp)) q += 16u;
            wq[i] = (char)q;
        }
        return;
    }
    case 14u: {                                 // q6_K
        device const block_q6_K *xb = (device const block_q6_K *)row + (g / 8u);
        const uint grp = g % 8u;
        const uint n128 = grp >> 2u;
        const uint quarter = grp & 3u;
        device const uchar *ql = xb->ql + n128 * 64u;
        device const uchar *qh = xb->qh + n128 * 32u;
        device const char *sc = xb->scales + n128 * 8u + quarter * 2u;
        const float d = (float)xb->d;
        *halves = 2;
        wa[0] = d * (float)sc[0];
        wa[1] = d * (float)sc[1];
        for (int i = 0; i < 32; i++) {
            const uint hi = ((uint)qh[i] >> (quarter * 2u)) & 3u;
            const uint lo = (quarter & 1u) ? (uint)ql[32 + i] : (uint)ql[i];
            const uint q = (quarter < 2u) ? ((lo & 0x0Fu) | (hi << 4u))
                                          : ((lo >> 4u) | (hi << 4u));
            wq[i] = (char)((int)q - 32);
        }
        return;
    }
    default:
        wa[0] = 0.0f;
        for (int i = 0; i < 32; i++) wq[i] = 0;
        return;
    }
}

static inline void ds4_qwen4exp_group_accumulate(
        thread float &acc, thread const char *wq,
        thread const float *wa, thread const float *wb, int halves,
        device const char *xqg, float xscale, int xsum) {
    if (halves == 1) {
        int dot = 0;
        for (int i = 0; i < 32; i++) dot += (int)wq[i] * (int)xqg[i];
        acc += (wa[0] * xscale) * (float)dot;
        acc += (wb[0] * xscale) * (float)xsum;
    } else {
        int d0 = 0, d1 = 0;
        for (int i = 0; i < 16; i++) d0 += (int)wq[i] * (int)xqg[i];
        for (int i = 16; i < 32; i++) d1 += (int)wq[i] * (int)xqg[i];
        acc += (wa[0] * xscale) * (float)d0;
        acc += (wa[1] * xscale) * (float)d1;
    }
}

// Q8_0 quantisation of one activation row and the integer sum of each group.
// One threadgroup per (row, group); the row is all it reads.
kernel void kernel_qwen4exp_quantize_rows(
        constant ds4_metal_qwen4exp_quant_args &args,
        device const float *x,
        device char *xq,
        device float *xscale,
        device int *xsum,
        threadgroup float *scratch [[threadgroup(0)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]]) {
    const uint g = tgpig.x;
    const uint r = tgpig.y;
    if (g >= args.groups) return;
    const uint i0 = g * 32u;
    const uint n = args.width - i0 < 32u ? args.width - i0 : 32u;
    const uint outer = r / args.inner_count;
    const uint inner = r - outer * args.inner_count;
    device const float *xr = x + (uint64_t)outer * args.outer_stride +
                            (uint64_t)inner * args.inner_stride + i0;

    float a = tid < n ? fabs(xr[tid]) : 0.0f;
    scratch[tid] = a;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 16u; stride > 0u; stride >>= 1u) {
        if (tid < stride) scratch[tid] = fmax(scratch[tid], scratch[tid + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float d = scratch[0] / 127.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    const uint64_t at = (uint64_t)r * args.groups + g;
    if (tid == 0u) xscale[at] = d;

    int v = 0;
    if (tid < n) {
        v = (int)rint(xr[tid] * id);
        v = v > 127 ? 127 : (v < -128 ? -128 : v);
    }
    xq[at * 32u + tid] = (char)v;

    threadgroup_barrier(mem_flags::mem_threadgroup);
    scratch[tid] = (float)v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 16u; stride > 0u; stride >>= 1u) {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) xsum[at] = (int)scratch[0];
}

// Grid (ceil(mid_dim / 8), n_expert).  One simdgroup owns one output row; lane
// L owns groups L, L+32, ...; the simdgroup carries R of the expert's pairs.
template<short R>
void kernel_qwen4exp_moe_gateup_q_impl(
        constant ds4_metal_qwen4exp_group_args &args,
        device const char *gate,
        device const char *up,
        device const char *xq,
        device const float *xs,
        device const int *xsum,
        device const uint *pairs,
        device const uint *counts,
        device const uint *offsets,
        device const uint *active,
        device const float *weights,
        device float *mid,
        uint2 tgpig, ushort tiisg, ushort sgitg) {
    const uint row = tgpig.x * 8u + (uint)sgitg;
    if (row >= args.mid_dim || tgpig.y >= active[0]) return;
    const uint expert = active[1u + tgpig.y];
    const uint cnt = counts[expert];
    if (cnt == 0u) return;
    const uint base = offsets[expert];

    device const char *gate_row = gate +
        (uint64_t)expert * args.gate_expert_bytes +
        (uint64_t)row * args.gate_row_bytes;
    device const char *up_row = up +
        (uint64_t)expert * args.up_expert_bytes +
        (uint64_t)row * args.up_row_bytes;

    for (uint at = 0; at < cnt; at += (uint)R) {
        const uint take = min((uint)R, cnt - at);
        uint tok[R];
        for (short r = 0; r < R; r++) {
            const uint p = pairs[base + at + (uint)((uint)r < take ? r : 0)];
            tok[r] = p / args.n_expert_used;
        }
        float ag[R];
        float au[R];
        for (short r = 0; r < R; r++) { ag[r] = 0.0f; au[r] = 0.0f; }

        for (uint g = (uint)tiisg; g < args.groups; g += 32u) {
            char gw[32], uw[32];
            float ga[2], gb[2], ua[2], ub[2];
            int gh = 1, uh = 1;
            ds4_qwen4exp_group_decode(args.gate_type, gate_row, g, gw, ga, gb, &gh);
            ds4_qwen4exp_group_decode(args.up_type, up_row, g, uw, ua, ub, &uh);
            for (short r = 0; r < R; r++) {
                if ((uint)r < take) {
                    const uint64_t at_g = (uint64_t)tok[r] * args.groups + g;
                    device const char *xqg = xq + at_g * 32u;
                    const float sc = xs[at_g];
                    const int sm = xsum[at_g];
                    ds4_qwen4exp_group_accumulate(ag[r], gw, ga, gb, gh, xqg, sc, sm);
                    ds4_qwen4exp_group_accumulate(au[r], uw, ua, ub, uh, xqg, sc, sm);
                }
            }
        }

        for (short r = 0; r < R; r++) {
            const float g = simd_sum(ag[r]);
            const float u = simd_sum(au[r]);
            if (tiisg == 0 && (uint)r < take) {
                const uint p = pairs[base + at + (uint)r];
                const uint t = p / args.n_expert_used;
                const uint slot = p - t * args.n_expert_used;
                mid[(uint64_t)t * args.mid_token_stride +
                    (uint64_t)slot * args.mid_dim + row] =
                    (g / (1.0f + exp(-g))) * u * weights[p];
            }
        }
    }
}

// Grid (ceil(out_dim / 8), ceil(n_tokens / R)).  The slots of a token go into
// ONE accumulator in ascending order, which is what the per-token kernel did.
template<short R>
void kernel_qwen4exp_moe_down_q_impl(
        constant ds4_metal_qwen4exp_group_args &args,
        device const char *down,
        device const int *selected,
        device const char *mq,
        device const float *ms,
        device const int *msum,
        device float *out,
        uint2 tgpig, ushort tiisg, ushort sgitg) {
    const uint row = tgpig.x * 8u + (uint)sgitg;
    const uint tok0 = tgpig.y * (uint)R;
    if (row >= args.out_dim || tok0 >= args.n_tokens) return;
    const uint take = min((uint)R, args.n_tokens - tok0);

    float acc[R];
    for (short r = 0; r < R; r++) acc[r] = 0.0f;

    for (uint slot = 0; slot < args.n_expert_used; slot++) {
        for (short r = 0; r < R; r++) {
            if ((uint)r < take) {
                const uint t = tok0 + (uint)r;
                const int e = selected[(uint64_t)t * args.n_expert_used + slot];
                if (e < 0 || (uint)e >= args.n_total_expert) continue;
                device const char *drow = down +
                    (uint64_t)(uint)e * args.down_expert_bytes +
                    (uint64_t)row * args.down_row_bytes;
                const uint64_t mrow = (uint64_t)t * args.n_expert_used + slot;
                for (uint g = (uint)tiisg; g < args.groups; g += 32u) {
                    char wq[32];
                    float wa[2], wb[2];
                    int halves = 1;
                    ds4_qwen4exp_group_decode(args.down_type, drow, g, wq, wa, wb,
                                              &halves);
                    const uint64_t at_g = mrow * args.groups + g;
                    ds4_qwen4exp_group_accumulate(acc[r], wq, wa, wb, halves,
                                                  mq + at_g * 32u, ms[at_g],
                                                  msum[at_g]);
                }
            }
        }
    }

    for (short r = 0; r < R; r++) {
        const float tot = simd_sum(acc[r]);
        if (tiisg == 0 && (uint)r < take) {
            out[(uint64_t)(tok0 + (uint)r) * args.out_dim + row] = tot;
        }
    }
}

template<short R>
void kernel_qwen4exp_shared_gateup_q_impl(
        constant ds4_metal_qwen4exp_group_args &args,
        device const char *gate,
        device const char *up,
        device const char *xq,
        device const float *xs,
        device const int *xsum,
        device float *mid,
        uint2 tgpig, ushort tiisg, ushort sgitg) {
    const uint row = tgpig.x * 8u + (uint)sgitg;
    const uint tok0 = tgpig.y * (uint)R;
    if (row >= args.mid_dim || tok0 >= args.n_tokens) return;
    const uint take = min((uint)R, args.n_tokens - tok0);
    device const char *gate_row = gate + (uint64_t)row * args.gate_row_bytes;
    device const char *up_row = up + (uint64_t)row * args.up_row_bytes;

    float ag[R];
    float au[R];
    for (short r = 0; r < R; r++) { ag[r] = 0.0f; au[r] = 0.0f; }

    for (uint g = (uint)tiisg; g < args.groups; g += 32u) {
        char gw[32], uw[32];
        float ga[2], gb[2], ua[2], ub[2];
        int gh = 1, uh = 1;
        ds4_qwen4exp_group_decode(args.gate_type, gate_row, g, gw, ga, gb, &gh);
        ds4_qwen4exp_group_decode(args.up_type, up_row, g, uw, ua, ub, &uh);
        for (short r = 0; r < R; r++) {
            if ((uint)r < take) {
                const uint64_t at_g = (uint64_t)(tok0 + (uint)r) * args.groups + g;
                device const char *xqg = xq + at_g * 32u;
                const float sc = xs[at_g];
                const int sm = xsum[at_g];
                ds4_qwen4exp_group_accumulate(ag[r], gw, ga, gb, gh, xqg, sc, sm);
                ds4_qwen4exp_group_accumulate(au[r], uw, ua, ub, uh, xqg, sc, sm);
            }
        }
    }

    for (short r = 0; r < R; r++) {
        const float g = simd_sum(ag[r]);
        const float u = simd_sum(au[r]);
        if (tiisg == 0 && (uint)r < take) {
            mid[(uint64_t)(tok0 + (uint)r) * args.mid_dim + row] =
                (g / (1.0f + exp(-g))) * u;
        }
    }
}

template<short R>
void kernel_qwen4exp_shared_down_q_impl(
        constant ds4_metal_qwen4exp_group_args &args,
        device const char *down,
        device const char *mq,
        device const float *ms,
        device const int *msum,
        device const float *gate_scale,
        device float *out,
        uint2 tgpig, ushort tiisg, ushort sgitg) {
    const uint row = tgpig.x * 8u + (uint)sgitg;
    const uint tok0 = tgpig.y * (uint)R;
    if (row >= args.out_dim || tok0 >= args.n_tokens) return;
    const uint take = min((uint)R, args.n_tokens - tok0);
    device const char *down_row = down + (uint64_t)row * args.down_row_bytes;

    float acc[R];
    for (short r = 0; r < R; r++) acc[r] = 0.0f;

    for (uint g = (uint)tiisg; g < args.groups; g += 32u) {
        char wq[32];
        float wa[2], wb[2];
        int halves = 1;
        ds4_qwen4exp_group_decode(args.down_type, down_row, g, wq, wa, wb, &halves);
        for (short r = 0; r < R; r++) {
            if ((uint)r < take) {
                const uint64_t at_g = (uint64_t)(tok0 + (uint)r) * args.groups + g;
                ds4_qwen4exp_group_accumulate(acc[r], wq, wa, wb, halves,
                                              mq + at_g * 32u, ms[at_g],
                                              msum[at_g]);
            }
        }
    }

    for (short r = 0; r < R; r++) {
        const float tot = simd_sum(acc[r]);
        if (tiisg == 0 && (uint)r < take) {
            const uint64_t off = (uint64_t)(tok0 + (uint)r) * args.out_dim + row;
            out[off] += gate_scale[tok0 + (uint)r] * tot;
        }
    }
}

#define DS4_QWEN4EXP_MOE_Q_KERNEL(NAME, IMPL, R, SIG, CALL)                   \
    kernel void NAME(                                                          \
            constant ds4_metal_qwen4exp_group_args &args,                      \
            SIG,                                                               \
            uint2 tgpig [[threadgroup_position_in_grid]],                      \
            ushort tiisg [[thread_index_in_simdgroup]],                        \
            ushort sgitg [[simdgroup_index_in_threadgroup]]) {                 \
        IMPL<R>(args, CALL, tgpig, tiisg, sgitg);                              \
    }

#define DS4_QWEN4EXP_GATEUP_SIG                                                \
    device const char *gate [[buffer(1)]],                                     \
    device const char *up [[buffer(2)]],                                       \
    device const char *xq [[buffer(3)]],                                       \
    device const float *xs [[buffer(4)]],                                      \
    device const int *xsum [[buffer(5)]],                                      \
    device const uint *pairs [[buffer(6)]],                                    \
    device const uint *counts [[buffer(7)]],                                   \
    device const uint *offsets [[buffer(8)]],                                  \
    device const uint *active [[buffer(9)]],                                   \
    device const float *weights [[buffer(10)]],                                \
    device float *mid [[buffer(11)]]
#define DS4_QWEN4EXP_GATEUP_CALL gate, up, xq, xs, xsum, pairs, counts, offsets, active, weights, mid

DS4_QWEN4EXP_MOE_Q_KERNEL(kernel_qwen4exp_moe_gateup_q_r1, kernel_qwen4exp_moe_gateup_q_impl, 1, DS4_QWEN4EXP_GATEUP_SIG, DS4_QWEN4EXP_GATEUP_CALL)
DS4_QWEN4EXP_MOE_Q_KERNEL(kernel_qwen4exp_moe_gateup_q_r4, kernel_qwen4exp_moe_gateup_q_impl, 4, DS4_QWEN4EXP_GATEUP_SIG, DS4_QWEN4EXP_GATEUP_CALL)
DS4_QWEN4EXP_MOE_Q_KERNEL(kernel_qwen4exp_moe_gateup_q_r8, kernel_qwen4exp_moe_gateup_q_impl, 8, DS4_QWEN4EXP_GATEUP_SIG, DS4_QWEN4EXP_GATEUP_CALL)

#define DS4_QWEN4EXP_DOWN_SIG                                                  \
    device const char *down [[buffer(1)]],                                     \
    device const int *selected [[buffer(2)]],                                  \
    device const char *mq [[buffer(3)]],                                       \
    device const float *ms [[buffer(4)]],                                      \
    device const int *msum [[buffer(5)]],                                      \
    device float *out [[buffer(6)]]
#define DS4_QWEN4EXP_DOWN_CALL down, selected, mq, ms, msum, out

DS4_QWEN4EXP_MOE_Q_KERNEL(kernel_qwen4exp_moe_down_q_r1, kernel_qwen4exp_moe_down_q_impl, 1, DS4_QWEN4EXP_DOWN_SIG, DS4_QWEN4EXP_DOWN_CALL)
DS4_QWEN4EXP_MOE_Q_KERNEL(kernel_qwen4exp_moe_down_q_r4, kernel_qwen4exp_moe_down_q_impl, 4, DS4_QWEN4EXP_DOWN_SIG, DS4_QWEN4EXP_DOWN_CALL)
DS4_QWEN4EXP_MOE_Q_KERNEL(kernel_qwen4exp_moe_down_q_r8, kernel_qwen4exp_moe_down_q_impl, 8, DS4_QWEN4EXP_DOWN_SIG, DS4_QWEN4EXP_DOWN_CALL)

#define DS4_QWEN4EXP_SHGU_SIG                                                  \
    device const char *gate [[buffer(1)]],                                     \
    device const char *up [[buffer(2)]],                                       \
    device const char *xq [[buffer(3)]],                                       \
    device const float *xs [[buffer(4)]],                                      \
    device const int *xsum [[buffer(5)]],                                      \
    device float *mid [[buffer(6)]]
#define DS4_QWEN4EXP_SHGU_CALL gate, up, xq, xs, xsum, mid

DS4_QWEN4EXP_MOE_Q_KERNEL(kernel_qwen4exp_shared_gateup_q_r1, kernel_qwen4exp_shared_gateup_q_impl, 1, DS4_QWEN4EXP_SHGU_SIG, DS4_QWEN4EXP_SHGU_CALL)
DS4_QWEN4EXP_MOE_Q_KERNEL(kernel_qwen4exp_shared_gateup_q_r4, kernel_qwen4exp_shared_gateup_q_impl, 4, DS4_QWEN4EXP_SHGU_SIG, DS4_QWEN4EXP_SHGU_CALL)
DS4_QWEN4EXP_MOE_Q_KERNEL(kernel_qwen4exp_shared_gateup_q_r8, kernel_qwen4exp_shared_gateup_q_impl, 8, DS4_QWEN4EXP_SHGU_SIG, DS4_QWEN4EXP_SHGU_CALL)

#define DS4_QWEN4EXP_SHDN_SIG                                                  \
    device const char *down [[buffer(1)]],                                     \
    device const char *mq [[buffer(2)]],                                       \
    device const float *ms [[buffer(3)]],                                      \
    device const int *msum [[buffer(4)]],                                      \
    device const float *gate_scale [[buffer(5)]],                              \
    device float *out [[buffer(6)]]
#define DS4_QWEN4EXP_SHDN_CALL down, mq, ms, msum, gate_scale, out

DS4_QWEN4EXP_MOE_Q_KERNEL(kernel_qwen4exp_shared_down_q_r1, kernel_qwen4exp_shared_down_q_impl, 1, DS4_QWEN4EXP_SHDN_SIG, DS4_QWEN4EXP_SHDN_CALL)
DS4_QWEN4EXP_MOE_Q_KERNEL(kernel_qwen4exp_shared_down_q_r4, kernel_qwen4exp_shared_down_q_impl, 4, DS4_QWEN4EXP_SHDN_SIG, DS4_QWEN4EXP_SHDN_CALL)
DS4_QWEN4EXP_MOE_Q_KERNEL(kernel_qwen4exp_shared_down_q_r8, kernel_qwen4exp_shared_down_q_impl, 8, DS4_QWEN4EXP_SHDN_SIG, DS4_QWEN4EXP_SHDN_CALL)

struct ds4_metal_qwen4exp_shared_args {
    uint32_t in_dim;
    uint32_t mid_dim;
    uint32_t out_dim;
    uint32_t n_tokens;
    uint32_t gate_type;
    uint32_t up_type;
    uint32_t down_type;
    uint32_t router_type;
    uint64_t gate_row_bytes;
    uint64_t up_row_bytes;
    uint64_t down_row_bytes;
};

// sigmoid(ffn_gate_inp_shexp . x), one scalar per token.
kernel void kernel_qwen4exp_shared_gate_f32(
        constant ds4_metal_qwen4exp_shared_args &args,
        device const char *router,
        device const float *x,
        device float *gate_out,
        threadgroup float *scratch [[threadgroup(0)]],
        uint token [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]]) {
    const uint ntg = 256u;
    if (token >= args.n_tokens) return;
    device const float *token_x = x + (uint64_t)token * args.in_dim;
    float acc = 0.0f;
    for (uint k = tid; k < args.in_dim; k += ntg) {
        acc += ds4_qwen4exp_weight_value(args.router_type, router, k) * token_x[k];
    }
    scratch[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = ntg >> 1u; stride > 0u; stride >>= 1u) {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) gate_out[token] = 1.0f / (1.0f + exp(-scratch[0]));
}





