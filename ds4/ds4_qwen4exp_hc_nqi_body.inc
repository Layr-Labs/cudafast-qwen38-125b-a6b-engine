(
        int8_t *xq, float *xscale, float *nscale, float *inject,
        const float *x, const float *w, const char *iw,
        uint32_t group, uint32_t n_hc, uint32_t rows,
        float eps, float weight_bias, int round_bf16,
        uint32_t weight_type, uint32_t weight_row_bytes,
        float *xw, const float *pblock, const float *pinject) {
    /* PDL producer, as the per-stream norm above; this arm runs at the row
     * threshold only, where the projection behind it is the plain MMA tile.
     * The row gate makes that structural rather than a caller convention:
     * the threshold's widths never fire the trigger at all (the deadlock
     * rule, ds4_cuda_qwen4exp.cuh). */
    if (rows <= 2u) QWEN4EXP_PDL_TRIGGER();
    const uint32_t row = blockIdx.x;
    if (row >= rows) return;

    __shared__ float partial[QWEN4EXP_HC_THREADS];
    float iacc[QWEN4EXP_HC_MAX_STREAMS];
#pragma unroll
    for (int ho = 0; ho < QWEN4EXP_HC_MAX_STREAMS; ho++) iacc[ho] = 0.0f;

    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t warps = blockDim.x >> 5u;
    const uint32_t n = n_hc * group;
    const uint64_t row_blocks = n / 32u;
    const float *pb = Pending ? pblock + (uint64_t)row * group : NULL;

    for (uint32_t g = 0; g < n_hc; g++) {
        const float *xg = x + (uint64_t)row * n + (uint64_t)g * group;
        const float *wg = w + (uint64_t)g * group;
        const uint64_t blk0 = (uint64_t)row * row_blocks + (uint64_t)g * (group / 32u);
        if (Staged) {
            float xv[QWEN4EXP_HC_STAGED_STEPS];
            float wv[QWEN4EXP_HC_STAGED_STEPS];
#pragma unroll
            for (uint32_t s = 0; s < QWEN4EXP_HC_STAGED_STEPS; s++) {
                xv[s] = xg[s * QWEN4EXP_HC_THREADS + threadIdx.x];
                wv[s] = wg[s * QWEN4EXP_HC_THREADS + threadIdx.x];
            }
            if (Pending) {
                /* qwen4exp_hc_inject_kernel's FFMA: residual + block * inject. */
                const float pi = pinject[(uint64_t)row * n_hc + g];
                float *xo = xw + (uint64_t)row * n + (uint64_t)g * group;
#pragma unroll
                for (uint32_t s = 0; s < QWEN4EXP_HC_STAGED_STEPS; s++) {
                    const uint32_t d = s * QWEN4EXP_HC_THREADS + threadIdx.x;
                    xv[s] = __fmaf_rn(pb[d], pi, xv[s]);
                    xo[d] = xv[s];
                }
            }
            /* partial[0] is still being read by the previous stream's callers. */
            __syncthreads();
            const float scale = qwen4exp_hc_norm_scale_regs(xv, group, eps, partial);
            if (threadIdx.x == 0u) nscale[(uint64_t)row * n_hc + g] = scale;
#pragma unroll
            for (uint32_t k = 0; k < QWEN4EXP_HC_STAGED_STEPS; k++) {
                const uint32_t i = k * QWEN4EXP_HC_THREADS + threadIdx.x;
                const float v = qwen4exp_hc_normed_value(xv[k], scale, wv[k],
                                                         weight_bias, round_bf16);
                const float vz = qwen4exp_q8_ftz(v);
                float a = qwen4exp_q8_ftz(fabsf(v));
#pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, off));
                }
                const float d = qwen4exp_q8_ftz(a * QWEN4EXP_Q8_RCP127);
                const float id = d != 0.0f ? qwen4exp_q8_rcp_approx(d) : 0.0f;
                const uint64_t pair = blk0 + (uint64_t)(k * warps + warp);
                if (lane == 0u) xscale[pair] = d;
                int q = (int)lrintf(qwen4exp_q8_ftz(vz * id));
                q = q > 127 ? 127 : (q < -128 ? -128 : q);
                xq[pair * 32u + lane] = (int8_t)q;

                /* The same __fmaf_rn chain as the rolled arm, the inject value
                 * taken by the typed staged accessor (n_embd := group, hs := g,
                 * s := k resolves to flat index g*group + i). */
#pragma unroll
                for (int ho = 0; ho < QWEN4EXP_HC_MAX_STREAMS; ho++) {
                    if ((uint32_t)ho < n_hc) {
                        iacc[ho] = __fmaf_rn(v, qwen4exp_hc_inject_value_staged<InjectType>(
                                iw + (uint64_t)ho * weight_row_bytes, group, g, k),
                                iacc[ho]);
                    }
                }
                (void)i;
            }
            continue;
        }
        if (Pending) {
            /* qwen4exp_hc_inject_kernel's FFMA, applied in place before the
             * statistic reads the stream. */
            const float pi = pinject[(uint64_t)row * n_hc + g];
            float *xo = xw + (uint64_t)row * n + (uint64_t)g * group;
            for (uint32_t i = threadIdx.x; i < group; i += blockDim.x) {
                xo[i] = __fmaf_rn(pb[i], pi, xg[i]);
            }
            __syncthreads();
        }
        /* partial[0] is still being read by the previous stream's callers. */
        __syncthreads();
        const float scale = qwen4exp_hc_norm_scale(xg, group, eps, partial);
        if (threadIdx.x == 0u) nscale[(uint64_t)row * n_hc + g] = scale;

        uint32_t k = 0;
        for (uint32_t i = threadIdx.x; i < group; i += blockDim.x, k++) {
            const float v = qwen4exp_hc_normed_value(xg[i], scale, wg[i],
                                                     weight_bias, round_bf16);
            const float vz = qwen4exp_q8_ftz(v);
            float a = qwen4exp_q8_ftz(fabsf(v));
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, off));
            }
            const float d = qwen4exp_q8_ftz(a * QWEN4EXP_Q8_RCP127);
            const float id = d != 0.0f ? qwen4exp_q8_rcp_approx(d) : 0.0f;
            const uint64_t pair = blk0 + (uint64_t)(k * warps + warp);
            if (lane == 0u) xscale[pair] = d;
            int q = (int)lrintf(qwen4exp_q8_ftz(vz * id));
            q = q > 127 ? 127 : (q < -128 ? -128 : q);
            xq[pair * 32u + lane] = (int8_t)q;

            /* __fmaf_rn, not `+= v * w`: the inject kernels' `+=` contracts
             * to one FFMA, and here the compiler hoisted the add of the
             * always-live stream 0 out of the weight-type switch, leaving a
             * rounded FMUL behind it -- one ulp on one inject in 192. */
            const uint32_t fi = g * group + i;
#pragma unroll
            for (int ho = 0; ho < QWEN4EXP_HC_MAX_STREAMS; ho++) {
                if ((uint32_t)ho < n_hc) {
                    iacc[ho] = __fmaf_rn(v, dev_qwen4exp_inject_value(
                            weight_type, iw + (uint64_t)ho * weight_row_bytes, fi),
                            iacc[ho]);
                }
            }
        }
    }

#pragma unroll
    for (int ho = 0; ho < QWEN4EXP_HC_MAX_STREAMS; ho++) {
        if ((uint32_t)ho < n_hc) {
            __syncthreads();
            const float total = qwen4exp_block_sum_f32(iacc[ho], partial);
            if (threadIdx.x == 0) {
                inject[(uint64_t)row * n_hc + (uint32_t)ho] =
                    2.0f * qwen4exp_sigmoid(total * (1.0f / (float)n_hc));
            }
        }
    }
}
