(
        float *out,
        const float *w,
        const float *x,
        uint64_t in_dim,
        uint64_t out_dim,
        uint32_t n_rows) {
    const int NW = WC * TN;            /* 16 weight columns per block  */
    const int NX = WR * TM;            /* 24 activation rows per block */
    const int SROW = 64;               /* two 32-chunks, ja then jb    */
    const int SSZ = (NW + NX) * SROW;  /* 2560 floats = 10,240 B       */
    __shared__ float st[STAGES * SSZ];

    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t wr_i = warp / (uint32_t)WC, wc_i = warp % (uint32_t)WC;
    const uint32_t tile = blockIdx.x * (uint32_t)WC + wc_i;
    const uint32_t ntn = (uint32_t)(out_dim / (uint64_t)TN);
    const bool live_col = tile < ntn;
    const uint32_t col0 = live_col ? tile * (uint32_t)TN : 0u;
    const uint32_t row0_raw = (blockIdx.y * (uint32_t)WR + wr_i) * (uint32_t)TM;
    const bool live_row = row0_raw < n_rows;
    const uint32_t row0 = live_row ? row0_raw : 0u;
    const uint32_t take = live_row
        ? (n_rows - row0 < (uint32_t)TM ? n_rows - row0 : (uint32_t)TM)
        : (uint32_t)TM;

    /* Fill assignment, fixed for the kernel.  NW*64/4 = 256 weight chunks:
     * thread tid takes chunk tid.  NX*64/4 = 384 activation chunks: thread tid
     * takes tid, and tid < 128 takes 256 + tid.  A chunk is 4 consecutive
     * floats inside one 32-element half, so it never straddles the ja|jb seam. */
    const uint32_t tid = threadIdx.x;
    const uint32_t wq_row = tid >> 4, wq_k = (tid & 15u) << 2;
    const uint32_t xq0_row = tid >> 4, xq0_k = (tid & 15u) << 2;
    const uint32_t xq1 = tid + 256u;
    const uint32_t xq1_row = xq1 >> 4, xq1_k = (xq1 & 15u) << 2;

    const uint32_t gcol = blockIdx.x * (uint32_t)NW + wq_row;
    const uint32_t gcol_c = (uint64_t)gcol < out_dim ? gcol : 0u;

    ds4_stage_fill_ctx f;
    f.wsrc = w + (uint64_t)gcol_c * in_dim + (wq_k & 31u);
    {   /* per-ROW-GROUP source row, with the shipping substitution */
        const uint32_t g0 = xq0_row / (uint32_t)TM, t0i = xq0_row % (uint32_t)TM;
        const uint32_t rb0 = (blockIdx.y * (uint32_t)WR + g0) * (uint32_t)TM;
        const uint32_t tk0 = rb0 < n_rows
            ? (n_rows - rb0 < (uint32_t)TM ? n_rows - rb0 : (uint32_t)TM) : 0u;
        const uint32_t sr0 = rb0 < n_rows ? (rb0 + (t0i < tk0 ? t0i : 0u)) : 0u;
        const uint32_t g1 = xq1_row / (uint32_t)TM, t1i = xq1_row % (uint32_t)TM;
        const uint32_t rb1 = (blockIdx.y * (uint32_t)WR + g1) * (uint32_t)TM;
        const uint32_t tk1 = rb1 < n_rows
            ? (n_rows - rb1 < (uint32_t)TM ? n_rows - rb1 : (uint32_t)TM) : 0u;
        const uint32_t sr1 = rb1 < n_rows ? (rb1 + (t1i < tk1 ? t1i : 0u)) : 0u;
        f.x0src = x + (uint64_t)sr0 * in_dim + (xq0_k & 31u);
        f.x1src = x + (uint64_t)sr1 * in_dim + (xq1_k & 31u);
    }
    f.wdst  = &st[wq_row * SROW + wq_k];
    f.x0dst = &st[(NW + xq0_row) * SROW + xq0_k];
    f.x1dst = &st[(NW + xq1_row) * SROW + xq1_k];
    f.wj_hi  = wq_k  >= 32u ? 1u : 0u;
    f.x0j_hi = xq0_k >= 32u ? 1u : 0u;
    f.x1j_hi = xq1_k >= 32u ? 1u : 0u;
    f.xq1_live = tid < 128u;

#pragma unroll
    for (int s = 0; s < STAGES - 1; s++)
        ds4_stage_fill<CG>(f, s / MC, s % MC, s % STAGES, SSZ);

    float A[TM][TN], B[TM][TN], t0[TM][TN], t1[TM][TN];
    const float *sw = &st[wc_i * (uint32_t)TN * SROW];
    const float *sx = &st[(NW + wr_i * (uint32_t)TM) * SROW];

    /* h(p,64) = (c_p + c_{p+128}) + (c_{p+64} + c_{p+192}) */
    ds4_stage_pair<TM, TN, MC, STAGES, 0, SROW, SSZ, CG>(t0, t1, sw, sx, lane, f);
#pragma unroll
    for (int t = 0; t < TM; t++)
#pragma unroll
        for (int c = 0; c < TN; c++) A[t][c] = t0[t][c] + t1[t][c];
    ds4_stage_pair<TM, TN, MC, STAGES, 1, SROW, SSZ, CG>(t0, t1, sw, sx, lane, f);
#pragma unroll
    for (int t = 0; t < TM; t++)
#pragma unroll
        for (int c = 0; c < TN; c++) A[t][c] = A[t][c] + (t0[t][c] + t1[t][c]);
    /* h(p+32,64) = (c_{p+32} + c_{p+160}) + (c_{p+96} + c_{p+224}) */
    ds4_stage_pair<TM, TN, MC, STAGES, 2, SROW, SSZ, CG>(t0, t1, sw, sx, lane, f);
#pragma unroll
    for (int t = 0; t < TM; t++)
#pragma unroll
        for (int c = 0; c < TN; c++) B[t][c] = t0[t][c] + t1[t][c];
    ds4_stage_pair<TM, TN, MC, STAGES, 3, SROW, SSZ, CG>(t0, t1, sw, sx, lane, f);
#pragma unroll
    for (int t = 0; t < TM; t++)
#pragma unroll
        for (int c = 0; c < TN; c++) B[t][c] = B[t][c] + (t0[t][c] + t1[t][c]);

    /* h(p,32) = h(p,64) + h(p+32,64), then strides 16, 8, 4, 2, 1. */
#pragma unroll
    for (int t = 0; t < TM; t++)
#pragma unroll
        for (int c = 0; c < TN; c++) {
            float s = A[t][c] + B[t][c];
#pragma unroll
            for (int d = 16; d > 0; d >>= 1) {
                s = s + __shfl_down_sync(0xffffffffu, s, d);
            }
            A[t][c] = s;
        }
    if (lane == 0u && live_col && live_row) {
#pragma unroll
        for (int t = 0; t < TM; t++) {
            if ((uint32_t)t < take) {
#pragma unroll
                for (int c = 0; c < TN; c++) {
                    out[(uint64_t)(row0 + (uint32_t)t) * out_dim +
                        (uint64_t)(col0 + (uint32_t)c)] = A[t][c];
                }
            }
        }
    }
}
