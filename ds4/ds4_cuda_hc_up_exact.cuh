/* HC up, 320 -> 10240: integer tensor-core dots with the original
 * 32-chain floating-point reduction. This is NOT the prefill MMA order.
 * One warp computes eight output channels for one/two activation rows.
 * All ten group products remain separate until the original tree. */
__device__ __forceinline__ static float hc_up_add(float a, float b) {
    float r;
    asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}
__device__ __forceinline__ static float hc_up_product(float w, float x, int d) {
    float scale, r;
    asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(scale) : "f"(w), "f"(x));
    asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(r)
        : "f"(scale), "f"((float)d), "f"(0.0f));
    return r;
}
__device__ __forceinline__ static float hc_up_tree(float p[10]) {
    /* Stride 16 adds the original zero chains. Preserve those additions,
     * including their signed-zero/FTZ behavior, before strides 8,4,2,1. */
#pragma unroll
    for (int g = 0; g < 10; g++) p[g] = hc_up_add(p[g], 0.0f);
#pragma unroll
    for (int g = 0; g < 8; g++)
        p[g] = hc_up_add(p[g], g < 2 ? p[g + 8] : 0.0f);
#pragma unroll
    for (int g = 0; g < 4; g++) p[g] = hc_up_add(p[g], p[g + 4]);
#pragma unroll
    for (int g = 0; g < 2; g++) p[g] = hc_up_add(p[g], p[g + 2]);
    return hc_up_add(p[0], p[1]);
}

/* Every access stays in the original 34-byte Q8 block. In particular the
 * last shifted word takes its upper half from payload[30..31], never from
 * the following block or from beyond the final mapped tensor. */
__device__ __forceinline__ static uint32_t hc_up_weight_word(
        const unsigned char *payload, unsigned byte) {
    const uintptr_t address = (uintptr_t)(payload + byte);
    const unsigned shift = (unsigned)(address & 3u) * 8u;
    const uint32_t *words = (const uint32_t *)(address & ~(uintptr_t)3u);
    const uint32_t lo = __ldcs(words);
    if (!shift) return lo;
    const uint32_t hi = byte == 28u
        ? (uint32_t)__ldcs((const uint16_t *)(payload + 30u))
        : __ldcs(words + 1);
    return __funnelshift_r(lo, hi, shift);
}

__global__ static void matmul_q8_hc_up_exact_mma_kernel(
        float *out, const unsigned char *w, const int8_t *xq,
        const float *xs, uint32_t rows, uint32_t out_dim) {
    const unsigned lane = threadIdx.x & 31u;
    const unsigned m = lane >> 2u, k = (lane & 3u) * 4u;
    const unsigned n0 = (blockIdx.x * (blockDim.x >> 5u) +
                         (threadIdx.x >> 5u)) * 8u;
    if (n0 >= out_dim) return;  // uniform across each warp
    float products0[10], products1[10];
#pragma unroll
    for (unsigned g = 0; g < 10u; g++) {
        const unsigned char *blk = w + (uint64_t)(n0 + m) * 340u + g * 34u;
        const uint32_t b[2] = {hc_up_weight_word(blk + 2u, k),
                               hc_up_weight_word(blk + 2u, k + 16u)};
        const float ws = __half2float(__ushort_as_half(
                __ldcs((const uint16_t *)blk)));
        /* Preserve silu-quant -> up PDL: prefetch immutable first-group
         * weights before the fence; all activation reads stay after it. */
        if (g == 0u) QWEN4EXP_PDL_SYNC();
        uint32_t a[4] = {0u, 0u, 0u, 0u};
        float scale = 0.0f;
        if (m < rows) {
            const uint64_t at = (uint64_t)m * 10u + g;
            const uint32_t *xp = (const uint32_t *)(xq + at * 32u + k);
            a[0] = xp[0]; a[2] = xp[4];
            scale = xs[at];
        }
        int32_t d[4] = {0, 0, 0, 0};
        mma_m16n8k32_s8_q8(d, a, b);
        const unsigned col = (lane & 3u) * 2u;
        const float w0 = __shfl_sync(0xffffffffu, ws, col * 4u);
        const float w1 = __shfl_sync(0xffffffffu, ws, (col + 1u) * 4u);
        products0[g] = hc_up_product(w0, scale, d[0]);
        products1[g] = hc_up_product(w1, scale, d[1]);
    }
    if (m < rows) {
        const uint64_t at = (uint64_t)m * out_dim + n0 + (lane & 3u) * 2u;
        out[at] = hc_up_tree(products0);
        out[at + 1u] = hc_up_tree(products1);
    }
}

static int hc_up_tuned_device = -1;
static bool hc_up_prefer_mma = false;

static bool hc_up_exact_use(int tier, uint32_t rows) {
    if (!rows || rows > 2u || tier < 0 || tier >= g_n_gpus ||
        getenv("DS4_Q8_NO_HC_UP_MMA") || getenv("DS4_Q8_NO_HC_WARP_PAIR")) return false;
    return getenv("DS4_Q8_FORCE_HC_UP_MMA") ||
        (hc_up_prefer_mma && hc_up_tuned_device == g_gpu[tier].device_id);
}
