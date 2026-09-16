/* Exact native-screen alternative: coalesced raw staging, one complete Q8 group per lane.
 * A warp owns one candidate and preserves the parent's 32-chain float tree. */
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

__global__ static void mtp_native_screen_staged_warp_kernel(
        uint64_t *keys, uint32_t *invalid, const unsigned char *w,
        const int8_t *xq, const float *xs, uint32_t width,
        uint32_t vocab, uint32_t prefix, uint32_t tail) {
    const unsigned lane = threadIdx.x & 31u;
    const unsigned warp = threadIdx.x / 32u;
    const unsigned row = blockIdx.x * 4u + warp;
    if (row >= width) return;
    const uint32_t id = row < prefix ? row : vocab - tail + row - prefix;
    /* Each warp copies its own candidate's original 24-group byte prefix.
     * Adjacent lanes read adjacent vectors; no warp consumes another warp's
     * stores. The halfword path retains the supported two-byte alignment. */
    __shared__ __align__(16) unsigned char staged[4u * 816u];
    unsigned char *const raw_row = staged + warp * 816u;
    const unsigned char *const source = w + (uint64_t)id * 2720u;
    if (((uintptr_t)w & 15u) == 0u) {
        for (unsigned word = lane; word < 51u; word += 32u) {
            ((uint4 *)raw_row)[word] = id < vocab
                ? mtp_stage_load4(source + word * 16u) : uint4{0u,0u,0u,0u};
        }
    } else {
        for (unsigned half = lane; half < 408u; half += 32u) {
            ((uint16_t *)raw_row)[half] = id < vocab
                ? mtp_stage_load2(source + half * 2u) : 0u;
        }
    }
    __syncwarp();
    float value = 0.0f;
    if (lane < 24u && id < vocab) {
        const unsigned char *block = raw_row + lane * 34u;
        const unsigned char *payload = block + 2u;
        const unsigned address = (unsigned)__cvta_generic_to_shared(payload);
        const unsigned shift = (unsigned)(address & 3u) * 8u;
        const unsigned words = address & ~3u;
        uint32_t previous = mtp_screen_shared4(words);
        int dot = 0;
#pragma unroll
        for (unsigned j = 0; j < 8u; j++) {
            const uint32_t next = j == 7u
                ? (uint32_t)mtp_screen_shared2(address + 30u)
                : mtp_screen_shared4(words + 4u * (j + 1u));
            const int packed = (int)__funnelshift_r(previous, next, shift);
            const int x = ((const int32_t *)(xq + lane * 32u))[j];
            dot = __dp4a(packed, x, dot);
            previous = next;
        }
        value = mtp_screen_product(__half2float(*(const __half *)block), xs[lane], dot);
    }
#pragma unroll
    for (unsigned step = 16; step; step /= 2u)
        value = mtp_screen_add(value, __shfl_down_sync(0xffffffffu, value, step));
    if (lane == 0u) {
        if (!isfinite(value)) atomicOr(invalid, 1u);
        if (!id || row >= prefix) keys[row] = UINT64_MAX - id;
        else keys[row] = q8_top1_pack_key(value == 0.0f ? 0.0f : value, id);
    }
}
