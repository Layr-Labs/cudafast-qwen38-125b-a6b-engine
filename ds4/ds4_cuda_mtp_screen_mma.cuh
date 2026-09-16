/* Exact alternative for the current 24-group native-MTP proposal screen.
 * The screen policy, mandatory IDs and subsequent full projections do not
 * change. A CTA copies the original 24 Q8 blocks for sixteen candidates into
 * a private shared slab, then integer MMA consumes those same bytes. The slab
 * lives only for this launch; every group enters the original float tree. */
__device__ __forceinline__ static float mtp_screen_add(float a, float b) {
    float r;
    asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}
__device__ __forceinline__ static float mtp_screen_product(float w, float x, int d) {
    float scale, r;
    asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(scale) : "f"(w), "f"(x));
    asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(r)
        : "f"(scale), "f"((float)d), "f"(0.0f));
    return r;
}

__device__ __forceinline__ static uint4 mtp_stage_load4(const void *p) {
    return *(const uint4 *)p;
}
__device__ __forceinline__ static uint16_t mtp_stage_load2(const void *p) {
    return *(const uint16_t *)p;
}
__device__ __forceinline__ static uint32_t mtp_screen_shared4(unsigned address) {
    uint32_t raw;
    asm("ld.shared.u32 %0, [%1];" : "=r"(raw) : "r"(address));
    return raw;
}
__device__ __forceinline__ static uint32_t mtp_screen_shared2(unsigned address) {
    uint32_t raw;
    asm("ld.shared.u16 %0, [%1];" : "=r"(raw) : "r"(address));
    return raw;
}

__device__ __forceinline__ static uint32_t mtp_screen_weight_word(
        const unsigned char *payload, unsigned byte) {
    const unsigned address = (unsigned)__cvta_generic_to_shared(payload + byte);
    const unsigned shift = (address & 3u) * 8u;
    const unsigned aligned = address & ~3u;
    const uint32_t lo = mtp_screen_shared4(aligned);
    if (!shift) return lo;
    const uint32_t hi = byte == 28u
        ? mtp_screen_shared2(address + 2u)
        : mtp_screen_shared4(aligned + 4u);
    return __funnelshift_r(lo, hi, shift);
}
/* Put weights on the 16-row MMA axis and the single activation on its
 * eight-column axis. A warp now produces sixteen candidates, with no scale
 * shuffles. Visit independent integer-dot leaves in the existing float tree's
 * depth-first order, retaining the exact same float addition nodes while
 * keeping only completed subtrees live instead of all 24 group products. */
struct mtp_tree_pair { float lo, hi; };

template<unsigned G>
__device__ __forceinline__ static mtp_tree_pair mtp_tree_leaf(
        const unsigned char *w0, const unsigned char *w1,
        bool valid0, bool valid1, const int8_t *xq, const float *xs,
        unsigned m, unsigned k) {
    if constexpr (G >= 24u) {
        return {0.0f, 0.0f};
    } else {
        uint32_t a[4] = {0, 0, 0, 0};
        float ws0 = 0.0f, ws1 = 0.0f;
        if (valid0) {
            const unsigned char *blk = w0 + G * 34u;
            a[0] = mtp_screen_weight_word(blk + 2u, k);
            a[2] = mtp_screen_weight_word(blk + 2u, k + 16u);
            ws0 = __half2float(*(const __half *)blk);
        }
        if (valid1) {
            const unsigned char *blk = w1 + G * 34u;
            a[1] = mtp_screen_weight_word(blk + 2u, k);
            a[3] = mtp_screen_weight_word(blk + 2u, k + 16u);
            ws1 = __half2float(*(const __half *)blk);
        }
        uint32_t b[2] = {0, 0};
        if (m == 0u) {
            const uint32_t *xp = (const uint32_t *)(xq + G * 32u + k);
            b[0] = xp[0]; b[1] = xp[4];
        }
        int32_t d[4] = {0, 0, 0, 0};
        mma_m16n8k32_s8_q8(d, a, b);
        return {mtp_screen_product(ws0, xs[G], d[0]),
                mtp_screen_product(ws1, xs[G], d[2])};
    }
}

template<unsigned BASE, unsigned STRIDE>
__device__ __forceinline__ static mtp_tree_pair mtp_tree_reduce(
        const unsigned char *w0, const unsigned char *w1,
        bool valid0, bool valid1, const int8_t *xq, const float *xs,
        unsigned m, unsigned k) {
    if constexpr (STRIDE == 32u) {
        return mtp_tree_leaf<BASE>(w0, w1, valid0, valid1, xq, xs, m, k);
    } else {
        const mtp_tree_pair a = mtp_tree_reduce<BASE, STRIDE * 2u>(
                w0, w1, valid0, valid1, xq, xs, m, k);
        const mtp_tree_pair b = mtp_tree_reduce<BASE + STRIDE, STRIDE * 2u>(
                w0, w1, valid0, valid1, xq, xs, m, k);
        return {mtp_screen_add(a.lo, b.lo), mtp_screen_add(a.hi, b.hi)};
    }
}

__global__ static void mtp_native_screen_mma_kernel(
        uint64_t *keys, uint32_t *invalid, const unsigned char *w,
        const int8_t *xq, const float *xs, uint32_t width,
        uint32_t vocab, uint32_t prefix, uint32_t tail) {
    const unsigned lane = threadIdx.x & 31u;
    const unsigned m = lane >> 2u, k = (lane & 3u) * 4u;
    const unsigned n0 = blockIdx.x * 16u;
    if (n0 >= width) return;
    /* Sixteen independent verbatim prefixes, each 24 * 34 bytes. Global
     * rows retain their original 80-group stride and original-ID mapping.
     * The vector path covers adjacent bytes; the halfword path preserves
     * the supported two-byte weight alignment without an overread. */
    __shared__ __align__(16) unsigned char staged[16u * 816u];
    if (((uintptr_t)w & 15u) == 0u) {
        for (unsigned v = threadIdx.x; v < 16u * 51u; v += blockDim.x) {
            const unsigned index = n0 + v / 51u;
            const unsigned word = v % 51u;
            const uint32_t id = index < prefix ? index
                : vocab - tail + (index - prefix);
            const uint4 raw = index < width && id < vocab
                ? mtp_stage_load4(w + (uint64_t)id * 2720u + word * 16u)
                : uint4{0u, 0u, 0u, 0u};
            ((uint4 *)staged)[v] = raw;
        }
    } else {
        for (unsigned v = threadIdx.x; v < 16u * 408u; v += blockDim.x) {
            const unsigned index = n0 + v / 408u;
            const unsigned half = v % 408u;
            const uint32_t id = index < prefix ? index
                : vocab - tail + (index - prefix);
            ((uint16_t *)staged)[v] = index < width && id < vocab
                ? mtp_stage_load2(w + (uint64_t)id * 2720u + half * 2u) : 0u;
        }
    }
    __syncthreads();
    const unsigned row0 = n0 + m, row1 = row0 + 8u;
    const uint32_t id0 = row0 < prefix ? row0 : vocab - tail + (row0 - prefix);
    const uint32_t id1 = row1 < prefix ? row1 : vocab - tail + (row1 - prefix);
    const bool valid0 = row0 < width && id0 < vocab;
    const bool valid1 = row1 < width && id1 < vocab;
    const unsigned char *w0 = staged + m * 816u;
    const unsigned char *w1 = staged + (m + 8u) * 816u;
    const mtp_tree_pair result = mtp_tree_reduce<0, 1>(
            w0, w1, valid0, valid1, xq, xs, m, k);
    if ((lane & 3u) == 0u) {
        const float totals[2] = {result.lo, result.hi};
#pragma unroll
        for (unsigned j = 0; j < 2u; j++) {
            const unsigned index = n0 + m + j * 8u;
            if (index < width) {
                const uint32_t original_id = index < prefix ? index
                    : vocab - tail + (index - prefix);
                const float value = totals[j];
                if (!isfinite(value)) atomicOr(invalid, 1u);
                if (!original_id || index >= prefix) keys[index] = UINT64_MAX - original_id;
                else keys[index] = q8_top1_pack_key(value == 0.0f ? 0.0f : value, original_id);
            }
        }
    }
}
