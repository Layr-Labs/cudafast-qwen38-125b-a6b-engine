/* Exact native-screen alternative: one lane computes one complete original Q8 group.
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
__device__ __forceinline__ static uint32_t mtp_warp_read4(const void *p) {
    uint32_t raw;
    asm("ld.global.u32 %0, [%1];" : "=r"(raw) : "l"(__cvta_generic_to_global(p)));
    return raw;
}
__device__ __forceinline__ static uint16_t mtp_warp_read2(const void *p) {
    uint32_t raw;
    asm("ld.global.u16 %0, [%1];" : "=r"(raw) : "l"(__cvta_generic_to_global(p)));
    return (uint16_t)raw;
}
__global__ static void mtp_native_screen_warp_kernel(
        uint64_t *keys, uint32_t *invalid, const unsigned char *w,
        const int8_t *xq, const float *xs, uint32_t width,
        uint32_t vocab, uint32_t prefix, uint32_t tail) {
    const unsigned lane = threadIdx.x & 31u;
    const unsigned row = blockIdx.x * 4u + threadIdx.x / 32u;
    if (row >= width) return;
    const uint32_t id = row < prefix ? row : vocab - tail + row - prefix;
    float value = 0.0f;
    if (lane < 24u && id < vocab) {
        const unsigned char *block = w + (uint64_t)id * 2720u + lane * 34u;
        const unsigned char *payload = block + 2u;
        const uintptr_t address = (uintptr_t)payload;
        const unsigned shift = (unsigned)(address & 3u) * 8u;
        const uint32_t *words = (const uint32_t *)(address & ~(uintptr_t)3u);
        uint32_t previous = mtp_warp_read4(words);
        int dot = 0;
#pragma unroll
        for (unsigned j = 0; j < 8u; j++) {
            const uint32_t next = j == 7u
                ? (uint32_t)mtp_warp_read2(payload + 30u)
                : mtp_warp_read4(words + j + 1u);
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
