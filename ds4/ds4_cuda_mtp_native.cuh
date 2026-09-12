/* Current native-head partial screening, then independent exact row dots.
 * Uses the original Q8_0 mapping; no transformed weight storage. New kernels
 * use ordinary stream ordering, not PDL: refinement reads freshly selected IDs. */
static constexpr uint32_t MTP_NATIVE_CAP = 16384u;
static constexpr uint32_t MTP_NATIVE_DIM = 2560u;
static constexpr uint32_t MTP_NATIVE_MAX_WIDTH = 1u << 20;
template <bool Screen>
__global__ static void mtp_native_projection_kernel(
        float *out, const unsigned char *w,
        const int8_t *xq, const float *xscale,
        uint64_t out_dim, uint32_t n_rows, uint64_t blocks,
        const uint32_t *ids, uint64_t n_vocab, uint32_t prefix, uint32_t tail) {
    constexpr int R = 1;
    constexpr bool Streaming = false;
    const uint64_t work_blocks = Screen ? blocks / 2u : blocks;
    const uint32_t local_row = threadIdx.x >> 6u;
    const uint32_t local_lane = threadIdx.x & 63u;
    const uint32_t group = local_lane >> 1u;
    const uint32_t half = local_lane & 1u;
    const uint64_t row = (uint64_t)blockIdx.x * 4u + local_row;
    const uint32_t row0 = blockIdx.y * R;
    const uint32_t take = n_rows - row0 < R ? n_rows - row0 : R;
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; r++) acc[r] = 0.0f;

    const uint64_t weight_row = row >= out_dim ? n_vocab : Screen
        ? (row < prefix ? row : n_vocab - tail + row - prefix) : ids[row];
    const bool valid = row < out_dim && weight_row < n_vocab;
    if (valid) {
        const unsigned char *wr = w + weight_row * blocks * 34u;
        for (uint64_t b = group; b < work_blocks; b += 32u) {
            /* Name both lanes of every live pair even if independent
             * scheduling has temporarily separated their execution. */
            const uint64_t warp_base = b - (uint64_t)(group & 15u);
            const uint64_t remaining = work_blocks - warp_base;
            const uint32_t live_pairs = (uint32_t)(remaining < 16u ? remaining : 16u);
            const unsigned active = 0xffffffffu >> (32u - 2u * live_pairs);
            const int8_t *payload = (const int8_t *)(wr + b * 34u + 2u) + half * 16u;
            const uintptr_t address = (uintptr_t)payload;
            const uint32_t shift = (uint32_t)(address & 3u) * 8u;
            const uint32_t *words = (const uint32_t *)(address & ~(uintptr_t)3u);
            /* Weights stream through each projection once. Mark their reads
             * evict-first while leaving the reusable activation loads alone. */
            uint32_t previous = Streaming ? __ldcs(words) : words[0];
            int32_t wq[4];
#pragma unroll
            for (int j = 0; j < 3; j++) {
                const uint32_t next = Streaming ? __ldcs(words + j + 1) : words[j + 1];
                wq[j] = (int32_t)__funnelshift_r(previous, next, shift);
                previous = next;
            }
            const uint16_t *lastp = (const uint16_t *)(const void *)(payload + 14);
            const uint16_t last = Streaming ? __ldcs(lastp) : *lastp;
            wq[3] = (int32_t)__funnelshift_r(previous, (uint32_t)last, shift);
            const __half *scale = (const __half *)(wr + b * 34u);
            const float ws = Streaming
                ? __half2float(__ushort_as_half(__ldcs((const uint16_t *)scale)))
                : __half2float(*scale);
#pragma unroll
            for (int r = 0; r < R; r++) {
                if ((uint32_t)r < take) {
                    const uint64_t at = ((uint64_t)row0 + r) * blocks + b;
                    const int32_t *xw = (const int32_t *)(xq + at * 32u + half * 16u);
                    int dot = 0;
#pragma unroll
                    for (int j = 0; j < 4; j++) dot = __dp4a(wq[j], xw[j], dot);
                    dot += __shfl_xor_sync(active, dot, 1);
                    if (half == 0u) acc[r] += ws * xscale[at] * (float)dot;
                }
            }
        }
    }

    __shared__ float partial[R][4][32];
    if (half == 0u) {
#pragma unroll
        for (int r = 0; r < R; r++) partial[r][local_row][group] = acc[r];
    }
    __syncthreads();
    if (local_lane < 32u) {
#pragma unroll
        for (int r = 0; r < R; r++) {
            const float total = warp_sum_f32(partial[r][local_row][local_lane]);
            if (local_lane == 0u && row < out_dim && (uint32_t)r < take)
                out[((uint64_t)row0 + r) * out_dim + row] = valid ? total : -INFINITY;
        }
    }
}

struct mtp_native_layout {
    uint64_t scores, key_in, key_out, flag, temporary;
};
static uint64_t mtp_native_align(uint64_t n) { return (n + 255u) & ~255ull; }
static mtp_native_layout mtp_native_offsets(uint32_t width) {
    mtp_native_layout l;
    l.scores = mtp_native_align(MTP_NATIVE_DIM + (MTP_NATIVE_DIM / 32u) * 4u);
    l.key_in = mtp_native_align(l.scores + (uint64_t)width * 4u);
    l.key_out = mtp_native_align(l.key_in + (uint64_t)width * 8u);
    l.flag = mtp_native_align(l.key_out + (uint64_t)width * 8u);
    l.temporary = mtp_native_align(l.flag + 4u);
    return l;
}
extern "C" int ds4_gpu_mtp_native_screen_init(uint32_t width,
        uint64_t *bytes, uint32_t *capacity) {
    if (!bytes || !capacity) return -1;
    *bytes = 0; *capacity = 0;
    if (width <= MTP_NATIVE_CAP || width > MTP_NATIVE_MAX_WIDTH) return 0;
    size_t a = 0;
    if (cub::DeviceRadixSort::SortKeysDescending(nullptr, a,
            (const uint64_t *)nullptr, (uint64_t *)nullptr, width, 0, 64,
            cuda_decode_stream()) != cudaSuccess) return -1;
    *bytes = mtp_native_offsets(width).temporary + a;
    *capacity = MTP_NATIVE_CAP;
    return 1;
}
__global__ static void mtp_native_keys(uint64_t *keys, uint32_t *invalid,
        const float *scores, uint32_t width, uint32_t prefix,
        uint32_t tail, uint32_t vocab) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width) return;
    const uint32_t id = i < prefix ? i : vocab - tail + i - prefix;
    const float value = scores[i];
    if (!isfinite(value)) atomicOr(invalid, 1u);
    /* Mandatory zero/tail reserve their own slots, independently of scores.
     * Finite keys never overlap that top range. Canonicalize zero for ties. */
    if (!id || i >= prefix) keys[i] = UINT64_MAX - id;
    else keys[i] = q8_top1_pack_key(value == 0.0f ? 0.0f : value, id);
}
__global__ static void mtp_native_unpack_ids(uint32_t *ids, const uint64_t *keys) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < MTP_NATIVE_CAP) ids[i] = UINT32_MAX - (uint32_t)keys[i];
}
/* -1: backend error, 0: ordinary full-static fallback, positive: exact number
 * of score-ordered candidates whose FULL refined logits now occupy out. */
extern "C" int ds4_gpu_mtp_native_screen(ds4_gpu_tensor *out,
        ds4_gpu_tensor *ids, ds4_gpu_tensor *scratch, const void *map,
        uint64_t map_bytes, uint64_t offset, uint32_t in_dim, uint32_t vocab,
        uint32_t prefix, uint32_t tail, const ds4_gpu_tensor *x) {
    const uint64_t wide = (uint64_t)prefix + tail;
    if (in_dim != MTP_NATIVE_DIM || !prefix || !tail || tail >= MTP_NATIVE_CAP ||
        prefix > vocab || tail > vocab - prefix || wide <= MTP_NATIVE_CAP ||
        wide > MTP_NATIVE_MAX_WIDTH || !cuda_q8_use_dp4a() ||
        getenv("DS4_QWEN4EXP_NO_ROW_TILE") != nullptr ||
        getenv("DS4_QWEN4EXP_PAIR_LANES_R2") != nullptr) return 0;
    const uint32_t width = (uint32_t)wide;
    const mtp_native_layout l = mtp_native_offsets(width);
    if (!out || !ids || !scratch || !x || !map ||
        out->bytes < MTP_NATIVE_CAP * 4ull || ids->bytes < MTP_NATIVE_CAP * 4ull ||
        scratch->bytes <= l.temporary || x->bytes < in_dim * 4ull ||
        offset > map_bytes || (uint64_t)vocab > (map_bytes-offset) / (80u*34u)) return -1;
    const int tier = ds4_tensor_device_idx(out);
    int current = -1;
    cudaStreamCaptureStatus capture;
    if (g_n_gpus != 1 || tier != 0 || ds4_tensor_device_idx(ids) != tier ||
        ds4_tensor_device_idx(scratch) != tier || ds4_tensor_device_idx(x) != tier) return 0;
    if (cudaGetDevice(&current) != cudaSuccess ||
        cudaStreamIsCapturing(cuda_decode_stream(), &capture) != cudaSuccess) return -1;
    if (current != g_gpu[0].device_id || capture != cudaStreamCaptureStatusNone) return 0;
    const char *w = cuda_resolve_weight_ptr(map, offset, (uint64_t)vocab*80u*34u,
                                          tier, "native MTP output");
    if (!w) return -1;
    if ((uintptr_t)w & 1u) return 0;
    char *base = (char *)scratch->ptr;
    int8_t *xq = (int8_t *)base;
    float *xs = (float *)(base + MTP_NATIVE_DIM);
    float *scores = (float *)(base + l.scores);
    uint64_t *key_in = (uint64_t *)(base + l.key_in);
    uint64_t *key_out = (uint64_t *)(base + l.key_out);
    uint32_t *flag = (uint32_t *)(base + l.flag);
    if (!cuda_ok(cudaMemsetAsync(flag,0,4,cuda_decode_stream()),"native screen flag")) return -1;
    quantize_q8_0_f32_rows_warp_kernel<<<10,256,0,cuda_decode_stream()>>>(
        xq,xs,(const float *)x->ptr,in_dim,80,1);
    if (!cuda_ok(cudaGetLastError(),"native screen quantize")) return -1;
    mtp_native_projection_kernel<true><<<(width+3u)/4u,256,0,cuda_decode_stream()>>>(
        scores,(const unsigned char *)w,xq,xs,width,1,80,nullptr,vocab,prefix,tail);
    if (!cuda_ok(cudaGetLastError(),"native half-column screen")) return -1;
    mtp_native_keys<<<(width+255u)/256u,256,0,cuda_decode_stream()>>>(
        key_in,flag,scores,width,prefix,tail,vocab);
    if (!cuda_ok(cudaGetLastError(),"native screen keys")) return -1;
    uint32_t invalid = 0;
    if (!ds4_gpu_tensor_read(scratch,l.flag,&invalid,4)) return -1;
    if (invalid) return 0;
    size_t temporary = (size_t)(scratch->bytes-l.temporary);
    if (!cuda_ok(cub::DeviceRadixSort::SortKeysDescending(base+l.temporary,temporary,
            key_in,key_out,width,0,64,cuda_decode_stream()),"native score sort")) return -1;
    mtp_native_unpack_ids<<<(MTP_NATIVE_CAP+255u)/256u,256,0,cuda_decode_stream()>>>((uint32_t *)ids->ptr,key_out);
    if (!cuda_ok(cudaGetLastError(),"native candidate unpack")) return -1;
    mtp_native_projection_kernel<false><<<(MTP_NATIVE_CAP+3u)/4u,256,0,cuda_decode_stream()>>>(
        (float *)out->ptr,(const unsigned char *)w,xq,xs,MTP_NATIVE_CAP,1,80,
        (const uint32_t *)ids->ptr,vocab,prefix,tail);
    return cuda_ok(cudaGetLastError(),"native exact refinement") ? (int)MTP_NATIVE_CAP : -1;
}
/* Original IDs provide the same tie order as sorting IDs before the old
 * compact top1. NaNs other than row0 never beat (-infinity,0); row0 keeps
 * the inherited NaN pin. Signed zeros compare equal. */
__global__ static void mtp_native_map(uint32_t *winner, const float *logits,
                                      const uint32_t *ids, uint32_t count, uint32_t vocab) {
    const uint32_t tid = threadIdx.x;
    float best = -INFINITY; uint32_t id = 0;
    for (uint32_t i = tid; i < count; i += 1024u) {
        const float value = logits[i]; const uint32_t original = ids[i];
        if (topk_score_better(value, original, best, id)) { best = value; id = original; }
    }
    __shared__ float values[1024];
    __shared__ uint32_t indices[1024];
    values[tid] = best; indices[tid] = id;
    __syncthreads();
    for (uint32_t stride = 512u; stride; stride >>= 1u) {
        if (tid < stride && topk_score_better(values[tid+stride], indices[tid+stride], values[tid], indices[tid])) {
            values[tid] = values[tid+stride]; indices[tid] = indices[tid+stride];
        }
        __syncthreads();
    }
    if (!tid) {
        const uint32_t bits = __float_as_uint(logits[0]);
        const uint32_t original = (bits & 0x7fffffffu) > 0x7f800000u ? 0u : indices[0];
        winner[0] = original < vocab ? original : UINT32_MAX;
    }
}
extern "C" int ds4_gpu_mtp_native_map(ds4_gpu_tensor *winner,
        const ds4_gpu_tensor *logits, const ds4_gpu_tensor *ids,
        uint32_t count, uint32_t vocab) {
    if (!winner || !logits || !ids || count != MTP_NATIVE_CAP || !vocab ||
        winner->bytes < 4 || logits->bytes < (uint64_t)count*4 || ids->bytes < (uint64_t)count*4) return 0;
    const int tier=ds4_tensor_device_idx(winner); int current=-1;
    if (tier<0 || tier>=g_n_gpus || ds4_tensor_device_idx(logits)!=tier ||
        ds4_tensor_device_idx(ids)!=tier || cudaGetDevice(&current)!=cudaSuccess ||
        current!=g_gpu[tier].device_id) return 0;
    mtp_native_map<<<1,1024,0,cuda_decode_stream()>>>((uint32_t *)winner->ptr,
        (const float *)logits->ptr,(const uint32_t *)ids->ptr,count,vocab);
    return cuda_ok(cudaGetLastError(),"native original winner map");
}
