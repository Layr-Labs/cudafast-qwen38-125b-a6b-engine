(float *out,
                                      const unsigned char *w,
                                      const int8_t *xq,
                                      const float *xscale,
                                      uint64_t out_dim,
                                      uint32_t n_rows,
                                      uint64_t blocks) {
    typedef q8_mma_pipe_cfg<WM, WN, MT, NT, G, STAGES> C;
    constexpr int BM = C::BM, BN = C::BN;

    extern __shared__ __align__(16) unsigned char q8_mma_smem[];
    unsigned char *sA_all = q8_mma_smem;
    unsigned char *sB_all = sA_all + STAGES * C::A_BYTES;
    float *sAs_all = (float *)(sB_all + STAGES * C::B_BYTES);
    float *sWs_all = sAs_all + STAGES * (C::AS_BYTES / 4);

    const int tid = (int)threadIdx.x;
    const uint32_t lane = threadIdx.x & 31u;
    const int warp = tid >> 5;

    const uint32_t m0 = (uint32_t)blockIdx.x * BM;
    const uint64_t n0 = (uint64_t)blockIdx.y * BN;
    if (m0 >= n_rows || n0 >= out_dim) return;

    const uint64_t nstage = (blocks + (uint64_t)G - 1u) / (uint64_t)G;
    const uint64_t w_row_bytes = blocks * 34u;
    /* Barrier ids: 1 + 2*b is FULL[b], 2 + 2*b is EMPTY[b]; 0 is __syncthreads. */
    constexpr int BAR_COUNT = C::THREADS;

    if (warp >= C::CWARPS) {
        const int pw = warp - C::CWARPS;
        /* ---- The producer warps, cp.async form.  The same chunk lists as
         * the register-staged producer (chunk idx = lane + 32*(k*PWARPS +
         * pw) over the same rows and columns), but every 16-byte chunk goes
         * global -> shared as one LDGSTS with no register in between, so a
         * lane holds nothing while a stage is in flight and the NEXT stage's
         * copies are issued before this stage's are waited for.  A row past
         * n_rows / out_dim and the last row's window past the tensor land as
         * zeros through the src-size operand -- the bytes the register form
         * stored as zeros (blocks % 8 == 0 makes every window offset a
         * multiple of 16, so a window is either whole or wholly outside).
         * The weight scales are converted from the landed window after
         * wait_group + the producers' barrier, exactly as before.  No value,
         * order or rounding point moves: the buffers hold the same bytes. */
        constexpr int PT = 32 * C::PWARPS;
        constexpr int KA = (BM * C::A_CHUNKS + PT - 1) / PT;
        constexpr int KS = (BM * (G / 4) + PT - 1) / PT;
        constexpr int KB = (BN * C::B_CHUNKS + PT - 1) / PT;
        static_assert(G % 4 == 0, "activation scales: 16-byte chunks");
        const int pl = (int)lane + 32 * pw;   /* producer lane, 0 .. PT-1 */

        auto issue = [&](uint64_t s) {
            const int buf = (int)(s % (uint64_t)STAGES);
            unsigned char *sA = sA_all + buf * C::A_BYTES;
            unsigned char *sB = sB_all + buf * C::B_BYTES;
            float *sAs = sAs_all + buf * (C::AS_BYTES / 4);
            const uint64_t g0 = s * (uint64_t)G;
            const uint64_t seg_off = s * (uint64_t)C::B_RAW;
            const uint64_t win_off = seg_off & ~(uint64_t)15u;
#pragma unroll
            for (int k = 0; k < KA; k++) {
                const int idx = pl + k * PT;
                if (idx < BM * C::A_CHUNKS) {
                    const int r = idx / C::A_CHUNKS;
                    const int c = idx - r * C::A_CHUNKS;
                    const uint64_t row = (uint64_t)m0 + (uint32_t)r;
                    const bool in = row < (uint64_t)n_rows;
                    const void *src = in
                        ? (const void *)(xq + (row * blocks + g0 + (uint32_t)(c >> 1)) * 32u + (c & 1) * 16)
                        : (const void *)xq;
                    q8_mma_cpasync_16_cg(sA + r * C::A_STRIDE + c * 16, src, in ? 16u : 0u);
                }
            }
#pragma unroll
            for (int k = 0; k < KS; k++) {
                const int idx = pl + k * PT;
                if (idx < BM * (G / 4)) {
                    const int r = idx / (G / 4);
                    const int c = idx - r * (G / 4);
                    const uint64_t row = (uint64_t)m0 + (uint32_t)r;
                    const bool in = row < (uint64_t)n_rows;
                    const void *src = in
                        ? (const void *)(xscale + row * blocks + g0 + (uint32_t)c * 4u)
                        : (const void *)xscale;
                    q8_mma_cpasync_16_cg(sAs + r * G + c * 4, src, in ? 16u : 0u);
                }
            }
#pragma unroll
            for (int k = 0; k < KB; k++) {
                const int idx = pl + k * PT;
                if (idx < BN * C::B_CHUNKS) {
                    const int r = idx / C::B_CHUNKS;
                    const int c = idx - r * C::B_CHUNKS;
                    const uint64_t row = n0 + (uint32_t)r;
                    uint32_t bytes = 0u;
                    const unsigned char *src = w;
                    if (row < out_dim) {
                        src = w + row * w_row_bytes + win_off + (uint32_t)c * 16u;
                        /* Only the tensor's last row's window can leave it. */
                        const int64_t in_row = (row + 1u == out_dim)
                            ? (int64_t)w_row_bytes - (int64_t)win_off - (int64_t)c * 16 : 16;
                        bytes = in_row >= 16 ? 16u : 0u;
                    }
                    q8_mma_cpasync_16_ca(sB + r * C::B_STRIDE + c * 16, src, bytes);
                }
            }
            q8_mma_cpasync_commit();
        };

        issue(0u);
        for (uint64_t s = 0; s < nstage; s++) {
            const int buf = (int)(s % (uint64_t)STAGES);
            if (s + 1u < nstage) {
                const int nbuf = (int)((s + 1u) % (uint64_t)STAGES);
                /* The next buffer must be free: consumers arrive on EMPTY
                 * when they finish the stage that last used it. */
                if (s + 1u >= (uint64_t)STAGES) q8_mma_bar_sync(2 + 2 * nbuf, BAR_COUNT);
                issue(s + 1u);
                q8_mma_cpasync_wait<1>();   /* stage s has landed (this lane's copies) */
            } else {
                q8_mma_cpasync_wait<0>();
            }
            /* Every producer's copies are visible to every producer: the
             * scales below lie in rows another one copied. */
            q8_mma_bar_sync(15, C::PWARPS * 32);
            unsigned char *sB = sB_all + buf * C::B_BYTES;
            float *sWs = sWs_all + buf * (C::WS_BYTES / 4);
            const int skew = (int)((s * (uint64_t)C::B_RAW) & 15u);
            /* Weight scales, half -> float, [gg][BN]. */
#pragma unroll
            for (int j = 0; j < (G * BN + PT - 1) / PT; j++) {
                const int i = pl + j * PT;
                if (i < G * BN) {
                    const int gg = i / BN;
                    const int rr = i - gg * BN;
                    uint16_t h;
                    memcpy(&h, sB + rr * C::B_STRIDE + skew + gg * 34, 2);
                    sWs[gg * BN + rr] = __half2float(__ushort_as_half(h));
                }
            }
            __syncwarp();
            q8_mma_bar_arrive(1 + 2 * buf, BAR_COUNT);
        }
        return;
    }

    /* ---- Consumers. */
    const int wm = warp / WN;
    const int wn = warp % WN;
    const uint32_t g4 = lane >> 2u;
    const uint32_t t4 = lane & 3u;

    float acc[MT][NT][4];
#pragma unroll
    for (int mi = 0; mi < MT; mi++)
#pragma unroll
        for (int ni = 0; ni < NT; ni++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc[mi][ni][e] = 0.0f;

    const int a_lrow = (int)(lane & 15u);
    const int a_lk = (int)(lane >> 4u) * 16;
    const int32_t magic = Q8_MMA_MAGIC_BITS;

    for (uint64_t s = 0; s < nstage; s++) {
        const int buf = (int)(s % (uint64_t)STAGES);
        q8_mma_bar_sync(1 + 2 * buf, BAR_COUNT);

        const unsigned char *sA = sA_all + buf * C::A_BYTES; /* consumers: stage s's activations */
        const unsigned char *sB = sB_all + buf * C::B_BYTES;
        const float *sAs = sAs_all + buf * (C::AS_BYTES / 4);
        const float *sWs = sWs_all + buf * (C::WS_BYTES / 4);
        const int skew = (int)((s * (uint64_t)C::B_RAW) & 15u);

        float xs[MT][2][G];
#pragma unroll
        for (int mi = 0; mi < MT; mi++) {
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const int r = wm * MT * 16 + mi * 16 + h * 8 + (int)g4;
                if (G == 2) {
                    const float2 v = *(const float2 *)(sAs + r * G);
                    xs[mi][h][0] = v.x; xs[mi][h][1 % G] = v.y;
                } else {
#pragma unroll
                    for (int q = 0; q < G / 4; q++) {
                        const float4 v = *(const float4 *)(sAs + r * G + q * 4);
                        xs[mi][h][(q * 4 + 0) % G] = v.x; xs[mi][h][(q * 4 + 1) % G] = v.y;
                        xs[mi][h][(q * 4 + 2) % G] = v.z; xs[mi][h][(q * 4 + 3) % G] = v.w;
                    }
                }
            }
        }

#pragma unroll
        for (int gg = 0; gg < G; gg++) { /* the stage's k32 steps, ascending */
            uint32_t af[MT][4];
#pragma unroll
            for (int mi = 0; mi < MT; mi++) {
                const int rbase = wm * MT * 16 + mi * 16;
                q8_mma_ldmatrix_x4(af[mi], sA + (rbase + a_lrow) * C::A_STRIDE + gg * 32 + a_lk);
            }
#pragma unroll
            for (int ni = 0; ni < NT; ni++) {
                const int c = wn * NT * 8 + ni * 8;
                const unsigned char *pb = sB + (c + (int)g4) * C::B_STRIDE + skew + gg * 34 + 2 + (int)t4 * 4;
                uint32_t bf[2];
                if ((gg & 1) == 0) {
                    const uint32_t *pw = (const uint32_t *)(pb - 2);
                    bf[0] = __funnelshift_r(pw[0], pw[1], 16u);
                    bf[1] = __funnelshift_r(pw[4], pw[5], 16u);
                } else {
                    const uint32_t *pw = (const uint32_t *)pb;
                    bf[0] = pw[0];
                    bf[1] = pw[4];
                }
                const float2 wsp = *(const float2 *)(sWs + gg * BN + c + (int)t4 * 2);
                int32_t d[MT][4];
#pragma unroll
                for (int mi = 0; mi < MT; mi++) q8_mma_m16n8k32_seeded(d[mi], af[mi], bf, magic);
#pragma unroll
                for (int mi = 0; mi < MT; mi++) {
                    acc[mi][ni][0] = q8_mma_fma_ftz(q8_mma_fmul_ftz(wsp.x, xs[mi][0][gg]), q8_mma_dot_to_f32(d[mi][0]), acc[mi][ni][0]);
                    acc[mi][ni][1] = q8_mma_fma_ftz(q8_mma_fmul_ftz(wsp.y, xs[mi][0][gg]), q8_mma_dot_to_f32(d[mi][1]), acc[mi][ni][1]);
                    acc[mi][ni][2] = q8_mma_fma_ftz(q8_mma_fmul_ftz(wsp.x, xs[mi][1][gg]), q8_mma_dot_to_f32(d[mi][2]), acc[mi][ni][2]);
                    acc[mi][ni][3] = q8_mma_fma_ftz(q8_mma_fmul_ftz(wsp.y, xs[mi][1][gg]), q8_mma_dot_to_f32(d[mi][3]), acc[mi][ni][3]);
                }
            }
        }
        /* Done with this buffer. */
        q8_mma_bar_arrive(2 + 2 * buf, BAR_COUNT);
    }

    /* The tile through shared memory, stored as whole rows.  The MMA layout
     * leaves every thread two adjacent floats of eight rows, a scattered
     * 8-byte store pattern; staged through the stage buffers the K loop has
     * finished with, a row goes out as 512-byte lines.  The values are the
     * accumulators, moved; no arithmetic.  Every production rung's tile fits
     * its stage buffers; a rung whose tile would not (the 256-wide opt-in)
     * keeps the direct store. */
    constexpr bool row_store = BM * (BN + 4) * 4 <= C::SMEM;
    if (!row_store) {
#pragma unroll
        for (int mi = 0; mi < MT; mi++) {
            const uint64_t r_lo = (uint64_t)m0 + wm * MT * 16 + mi * 16 + g4;
            const uint64_t r_hi = r_lo + 8u;
#pragma unroll
            for (int ni = 0; ni < NT; ni++) {
                const uint64_t c0 = n0 + wn * NT * 8 + ni * 8 + t4 * 2u;
                const uint64_t c1 = c0 + 1u;
                if (r_lo < (uint64_t)n_rows) {
                    if (c0 < out_dim) out[r_lo * out_dim + c0] = acc[mi][ni][0];
                    if (c1 < out_dim) out[r_lo * out_dim + c1] = acc[mi][ni][1];
                }
                if (r_hi < (uint64_t)n_rows) {
                    if (c0 < out_dim) out[r_hi * out_dim + c0] = acc[mi][ni][2];
                    if (c1 < out_dim) out[r_hi * out_dim + c1] = acc[mi][ni][3];
                }
            }
        }
        return;
    }
    q8_mma_bar_sync(13, C::CWARPS * 32);   /* every consumer has left the last stage */
    float *tile = (float *)q8_mma_smem;
    constexpr int TS = BN + 4;
#pragma unroll
    for (int mi = 0; mi < MT; mi++) {
        const int rl = wm * MT * 16 + mi * 16 + (int)g4, rh = rl + 8;
#pragma unroll
        for (int ni = 0; ni < NT; ni++) {
            const int c = wn * NT * 8 + ni * 8 + (int)t4 * 2;
            tile[rl * TS + c] = acc[mi][ni][0];
            tile[rl * TS + c + 1] = acc[mi][ni][1];
            tile[rh * TS + c] = acc[mi][ni][2];
            tile[rh * TS + c + 1] = acc[mi][ni][3];
        }
    }
    q8_mma_bar_sync(13, C::CWARPS * 32);   /* the tile is whole */
    const bool vec = (out_dim & 3u) == 0u && ((((uintptr_t)out) & 15u) == 0u);
    for (int r = warp; r < BM; r += C::CWARPS) {
        const uint64_t row = (uint64_t)m0 + (uint32_t)r;
        if (row >= (uint64_t)n_rows) continue;
        for (int c = (int)lane * 4; c < BN; c += 128) {
            if (vec && n0 + (uint32_t)c + 3u < out_dim) {
                *(float4 *)(out + row * out_dim + n0 + (uint32_t)c) = *(const float4 *)(tile + r * TS + c);
            } else {
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    if (n0 + (uint32_t)(c + e) < out_dim) out[row * out_dim + n0 + (uint32_t)(c + e)] = tile[r * TS + c + e];
                }
            }
        }
    }
}
