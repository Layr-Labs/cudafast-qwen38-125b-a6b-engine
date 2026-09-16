/* Audited 64-channel raw pipeline; shares qw_down_raw_words6 with raw32. */
#pragma once
template <int DownType = -1, bool Wide6 = false>
__global__ __launch_bounds__(QW_DOWN_MMA_THREADS) static void
qwen4exp_moe_down_raw64_kernel(
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
        uint32_t out_dim,
        uint32_t dq_stage) {
    __shared__ __align__(16) int8_t sA[QW_DOWN_MMA_BM * QW_MMA_LD];
    __shared__ __align__(16) int8_t sB[QW_MMA_BN * QW_MMA_LD];
    __shared__ float  sWA[QW_DOWN_MMA_BM * QW_MMA_G];
    __shared__ float  sWB[QW_DOWN_MMA_BM * QW_MMA_G];
    __shared__ float  sXS[QW_MMA_BN * QW_MMA_G], sXSUM[QW_MMA_BN * QW_MMA_G];
    __shared__ uint32_t sPair[QW_MMA_BN];
    /* Original Q5_1 bytes for one four-group K chunk, 64 * 96 bytes.
     * A single buffer suffices: all readers retire at the pre-MMA barrier;
     * the next fill then overlaps only the current decoded-tile MMA reads. */
    __shared__ __align__(16) uint4 sRaw[QW_DOWN_MMA_BM * 6u];

    const uint32_t tid  = threadIdx.x;
    const uint32_t warp = tid >> 5;
    const uint32_t lane = tid & 31;
    const uint32_t row0 = blockIdx.x * QW_DOWN_MMA_BM;
    if (row0 >= out_dim) return;
    if (active && (int32_t)blockIdx.y >= active[0]) return;
    const uint32_t expert = active ? (uint32_t)active[1 + blockIdx.y]
                                   : blockIdx.y;
    const int32_t cnt = counts[expert];
    if (cnt <= 0) return;
    const int32_t base = offsets[expert];
    const char *down_e = down + (uint64_t)expert * down_expert_bytes;

    /* WORD-DIRECT q5_1 STAGING.  dq_stage is DS4_QWEN4EXP_NO_DOWN_DQ left
     * unset; it stages the 24-byte block's six words and decodes them
     * straight into the tile row, where the oracle path decodes into a
     * byte array and repacks it with qw_tile_store_group's shifts and ors.
     * A q5_1 block carries no super-block structure to share (R1/R2 do not
     * apply) and the down tile's 132-byte stride is not a multiple of
     * sixteen (STS.128 does not apply), so the cut is the repack
     * elimination plus the spread-form high bits.  Q8_0 and the generic
     * instantiation keep the oracle; a row whose blocks are not word
     * aligned stages nothing and decodes from the row, exactly as before. */
    const uint32_t dtype = DownType < 0 ? down_type : (uint32_t)DownType;
    const bool w_dq = dq_stage != 0u &&
                      dtype == (uint32_t)DS4_QWEN4EXP_TY_q5_1;

    const bool raw_pipe = Wide6 && w_dq && (groups & 3u) == 0u &&
        ((((uintptr_t)down_e) | (uintptr_t)down_row_bytes) & 15u) == 0u;
    /* The public dispatcher retains the original tile for unsupported input.
     * This kernel owns only the aligned raw pipeline, so no cold global-load
     * fallback forces its six words through an address-taken local array. */
    if (!raw_pipe) return;
    auto issue_raw = [&](uint32_t kc) {
        /* Six consecutive lanes cover one complete 96-byte row slice.
         * No padding or numerical conversion is stored in this buffer. */
        for (uint32_t i = tid; i < QW_DOWN_MMA_BM * 6u;
             i += QW_DOWN_MMA_THREADS) {
            const uint32_t r = i / 6u, piece = i % 6u;
            if (row0 + r < out_dim) {
                const char *src = down_e +
                    (uint64_t)(row0 + r) * down_row_bytes +
                    (uint64_t)kc * 24u + piece * 16u;
                qw_cpasync16((uint32_t)__cvta_generic_to_shared(&sRaw[i]), src);
            }
        }
        qw_cpasync_commit();
    };

    for (int32_t nbase = 0; nbase < cnt; nbase += QW_MMA_BN) {
        const int32_t take = (cnt - nbase) < QW_MMA_BN ? (cnt - nbase)
                                                       : QW_MMA_BN;
        for (uint32_t i = tid; i < QW_MMA_BN; i += QW_DOWN_MMA_THREADS) {
            sPair[i] = (int32_t)i < take
                ? (uint32_t)pairs[base + nbase + i] : 0xffffffffu;
        }
        __syncthreads();

        if (raw_pipe) issue_raw(0u);

        float acc[QW_DOWN_MMA_NT * 4];
#pragma unroll
        for (int i = 0; i < QW_DOWN_MMA_NT * 4; i++) acc[i] = 0.0f;

        for (uint32_t kc = 0; kc < groups; kc += QW_MMA_G) {
            if (raw_pipe) qw_cpasync_wait0();
            __syncthreads();
            for (uint32_t idx = tid; idx < QW_DOWN_MMA_BM * QW_MMA_G;
                 idx += QW_DOWN_MMA_THREADS) {
                const uint32_t r = idx / QW_MMA_G;
                const uint32_t gg = idx - r * QW_MMA_G;
                const uint32_t g = kc + gg;
                const uint32_t orow = row0 + r;
                float wa[2], wb[2];
                if (orow < out_dim && g < groups) {
                    const char *const drow =
                        down_e + (uint64_t)orow * down_row_bytes;
                    uint32_t raw[6];
                    qw_down_raw_words6(sRaw, r, gg, raw);
                    dev_qwen4exp_group_decode_w(dtype, drow, g, raw,
                            &sA[r * QW_MMA_LD + gg * 32], wa, wb);
                    sWA[r * QW_MMA_G + gg] = wa[0];
                    sWB[r * QW_MMA_G + gg] = wb[0];
                } else {
                    qw_tile_store_zero(&sA[r * QW_MMA_LD + gg * 32]);
                    sWA[r * QW_MMA_G + gg] = 0.0f;
                    sWB[r * QW_MMA_G + gg] = 0.0f;
                }
            }
            for (uint32_t idx = tid; idx < QW_MMA_BN * QW_MMA_G;
                 idx += QW_DOWN_MMA_THREADS) {
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
            if (raw_pipe && kc + QW_MMA_G < groups)
                issue_raw(kc + QW_MMA_G);

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
                for (int nt = 0; nt < QW_DOWN_MMA_NT; nt++) {
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
        for (int nt = 0; nt < QW_DOWN_MMA_NT; nt++) {
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
