/* Decode gate/up: deliver the existing verbatim cooperative panel directly
 * into shared memory. The arithmetic and panel lifetime are unchanged. */
#pragma once
template <int R, int Type, bool Vector = false,
          unsigned OutputRows = 4, bool Coop = false>
__global__ static void QW_GU_MAXNREG
qwen4exp_moe_gateup_async_kernel(
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
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t row = blockIdx.x * OutputRows + (warp >> 1u);
    const bool live = row < mid_dim;
    const bool second = (warp & 1u) != 0u;
    uint32_t expert = blockIdx.y;
    /* The routed gate/up edge, opened on the kernel the COOP decode path runs.
     *
     * The dependent launch already exists in this file on
     * qwen4exp_moe_gateup_q_kernel, and its comment states the mechanism: the
     * blocks are already up and scheduled when the quantizer's last group
     * retires, instead of paying a launch behind it.  The coop schedule does
     * not use that kernel -- it uses this one, and this one was launched
     * plainly.
     *
     * The fence sits ahead of EVERY producer read -- the active list, the
     * counts/offsets/pairs tables, the quantized activation and the router
     * weights are all written by the routed quantizer -- so it is placed
     * unconditionally, not inside the `active` branch: `counts` is read even
     * when `active` is NULL.  .nc rule: no pointer in this signature carries
     * __restrict__, so no activation load can be hoisted above the fence as
     * ld.global.nc.  Deadlock rule: it constrains the PRODUCER, and the
     * quantizer bounds itself to one wave before it triggers, so a multi-wave
     * dependent is safe.  Launched plainly -- three rows, every prefill width
     * -- the fence is a no-op, exactly as it is for the kernel beside it. */
    QWEN4EXP_PDL_SYNC();
    if (active) {
        if ((int32_t)blockIdx.y >= active[0]) return;
        expert = (uint32_t)active[1 + blockIdx.y];
    }
    const int32_t cnt = counts[expert];
    if (cnt <= 0) return;
    const int32_t base = offsets[expert];
    const char *weight_row = (second ? up : gate) +
        (uint64_t)expert * (second ? up_expert_bytes : gate_expert_bytes) +
        (uint64_t)(live ? row : 0u) * (second ? up_row_bytes : gate_row_bytes);
    __shared__ float projected[R][OutputRows * 2u];
    /* The staged panel.  Both early returns above are uniform over the block
     * (blockIdx.y, the active list and counts[expert] are block-invariant), so
     * every thread that reaches the barrier below reaches it together. */
    __shared__ __align__(16) uint4 wcoop[Coop ? 2u * QW_GU_COOP_U4 : 1u];
    const uint4 *wsh = NULL;
    uint32_t wrow = 0u;
    if (Coop) {
        /* The panel is a fixed-size static allocation, so the instantiation is
         * only answerable at the tower's q4_K gate/up row shape.  The launcher
         * refuses every other shape; this is the belt to that brace, and it is
         * uniform over the block, outside the group loop, and free. */
        if (groups != QW_GU_COOP_GROUPS ||
            gate_row_bytes != (uint64_t)QW_GU_COOP_ROW_U4 * 16u ||
            up_row_bytes != (uint64_t)QW_GU_COOP_ROW_U4 * 16u) return;
        const uint32_t row0 = blockIdx.x * OutputRows;
        const uint32_t left = mid_dim > row0 ? mid_dim - row0 : 0u;
        const uint32_t rows_here = left < OutputRows ? left : OutputRows;
        const uint32_t words = rows_here * QW_GU_COOP_ROW_U4;
        const char *const gb = gate +
            (uint64_t)expert * gate_expert_bytes +
            (uint64_t)row0 * gate_row_bytes;
        const char *const ub = up +
            (uint64_t)expert * up_expert_bytes +
            (uint64_t)row0 * up_row_bytes;
        for (uint32_t i = threadIdx.x; i < words; i += blockDim.x) {
            qw_cpasync16((uint32_t)__cvta_generic_to_shared(&wcoop[i]),
                         gb + (uint64_t)i * 16u);
            qw_cpasync16((uint32_t)__cvta_generic_to_shared(&wcoop[QW_GU_COOP_U4 + i]),
                         ub + (uint64_t)i * 16u);
        }
        qw_cpasync_commit();
        qw_cpasync_wait0();
        __syncthreads();
        wsh = wcoop + (second ? QW_GU_COOP_U4 : 0u);
        wrow = warp >> 1u;
    }
    for (int32_t at = 0; at < cnt; at += R) {
        const int32_t take = (cnt - at) < R ? (cnt - at) : R;
        uint32_t tok[R];
#pragma unroll
        for (int r = 0; r < R; r++) {
            const int32_t p = pairs[base + at + (r < take ? r : 0)];
            tok[r] = (uint32_t)p / n_expert_used;
        }
        float acc[R];
#pragma unroll
        for (int r = 0; r < R; r++) acc[r] = 0.0f;
        if (live) {
            /* WORD DECODE, the same one the routed-MoE MMA kernels use, and
             * only ever instantiated for Q4_K, whose decode leaves `halves`
             * at one -- the value passed to the accumulate below.
             *
             * TWO GROUPS IN FLIGHT.  A lane walks g, g+32, g+64 and adds
             * them to acc[r] in that order.  The single-group body consumed
             * its payload immediately, so a lane held one 32-byte read
             * outstanding and the loop ran at memory latency.  A routed
             * expert row is read exactly once per call, so there is nothing
             * to hit in cache and the only lever is reads in flight.  The
             * body below stages the SECOND group's payload before the FIRST
             * group's decode consumes its own, keeping one decoded group
             * live at a time so occupancy is unchanged.
             *
             * Nothing is reassociated: same terms, same order, same tail
             * lanes, same warp_sum_f32 tree, identical bits. */
#define QWEN4EXP_SPLIT_GROUP(GG, RAWP) do { \
                const uint32_t g_ = (GG); \
                int8_t wq[32]; \
                float wa[2] = {0.0f, 0.0f}; \
                float wb[2] = {0.0f, 0.0f}; \
                if (Coop) \
                    dev_qwen4exp_group_decode_w((uint32_t)Type, \
                        (const char *)(const void *) \
                            &wsh[wrow * QW_GU_COOP_ROW_U4], g_, \
                        (RAWP), wq, wa, wb); \
                else \
                    dev_qwen4exp_group_decode_w((uint32_t)Type, weight_row, g_, \
                                                (RAWP), wq, wa, wb); \
                const int halves = 1; \
                _Pragma("unroll") \
                for (int r = 0; r < R; r++) { \
                    if (r < take) { \
                        const uint64_t at_g = (uint64_t)tok[r] * groups + g_; \
                        if (Vector) \
                            qwen4exp_shared_vector_accumulate(&acc[r], wq, wa[0], wb[0], \
                                xq + at_g * 32u, xs[at_g], xsum[at_g]); \
                        else \
                            qwen4exp_group_accumulate(&acc[r], wq, wa, wb, halves, \
                                xq + at_g * 32u, xs[at_g], xsum[at_g]); \
                    } \
                } \
            } while (0)
            uint32_t g = lane;
            for (; g + 32u < groups; g += 64u) {
                uint32_t raw0[8];
                uint32_t raw1[8];
                const uint32_t *p0, *p1;
                if (Coop) {
                    qw_gu_coop_raw_load(wsh, wrow, g, raw0);
                    qw_gu_coop_raw_load(wsh, wrow, g + 32u, raw1);
                    p0 = raw0; p1 = raw1;
                } else {
                    p0 = qw_raw_load((uint32_t)Type, weight_row, g, raw0)
                       ? raw0 : NULL;
                    p1 = qw_raw_load((uint32_t)Type, weight_row, g + 32u, raw1)
                       ? raw1 : NULL;
                }
                QWEN4EXP_SPLIT_GROUP(g, p0);
                QWEN4EXP_SPLIT_GROUP(g + 32u, p1);
            }
            for (; g < groups; g += 32u) {
                uint32_t raw[8];
                const uint32_t *rawp;
                if (Coop) {
                    qw_gu_coop_raw_load(wsh, wrow, g, raw);
                    rawp = raw;
                } else {
                    rawp = qw_raw_load((uint32_t)Type, weight_row, g, raw)
                         ? raw : NULL;
                }
                QWEN4EXP_SPLIT_GROUP(g, rawp);
            }
#undef QWEN4EXP_SPLIT_GROUP
        }
#pragma unroll
        for (int r = 0; r < R; r++) {
            const float v = warp_sum_f32(acc[r]);
            if (lane == 0u) projected[r][warp] = v;
        }
        __syncthreads();
        if (live && !second && lane == 0u) {
#pragma unroll
            for (int r = 0; r < R; r++) {
                if (r < take) {
                    const uint32_t p = (uint32_t)pairs[base + at + r];
                    const uint32_t t = p / n_expert_used;
                    const uint32_t slot = p - t * n_expert_used;
                    const float g = projected[r][warp];
                    const float u = projected[r][warp + 1u];
                    mid[(uint64_t)t * mid_token_stride +
                        (uint64_t)slot * mid_dim + row] =
                        (g / (1.0f + expf(-g))) * u * weights[p];
                }
            }
        }
        /* Readers finish before a fast projection warp reuses this tile --
         * a hazard only a SECOND iteration of this loop can create, so the
         * barrier is dead whenever there is no second iteration.
         *
         * CREDIT: 0xpg (`37816fd`).  At the decode width the body runs exactly
         * once: `cnt` is counts[expert], the number of (token, slot) pairs
         * that routed to this block's expert, and a decode round verifies two
         * rows each selecting ten of 512 experts, so any one expert collects
         * one or two of the twenty pairs.  R is 2 for every instantiation the
         * launcher builds, so cnt <= R and `at + R >= cnt` on the first pass.
         *
         * The predicate is BLOCK-UNIFORM and therefore cannot deadlock: cnt is
         * counts[expert] with expert block-invariant, and `at` is loop-uniform.
         * Prefill, where cnt genuinely exceeds R, takes the barrier exactly as
         * before, byte for byte. */
        if (at + R < cnt) __syncthreads();
    }
}
