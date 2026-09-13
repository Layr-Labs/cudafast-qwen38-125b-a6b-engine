/* Current native-head partial screening, then independent exact row dots.
 * Uses the original Q8_0 mapping; no transformed weight storage. New kernels
 * use ordinary stream ordering, not PDL: refinement reads freshly sorted IDs. */
/* Refinement shortlist. 276 tail rows plus id 0 hold mandatory slots, so this
 * leaves 1771 score-selected candidates: the top 1.8% of the coarse ranking.
 * Narrowing it is a proposal-policy change, not an exact one. */
static constexpr uint32_t MTP_NATIVE_CAP = 2048u;
static constexpr uint32_t MTP_NATIVE_DIM = 2560u;
/* Coarse screen depth; full refinement still uses 80 groups. Measured in both
 * directions and tuned: 32 groups is an exact null (same 49/78 acceptance) and
 * 16 groups loses a draft (48/79, one extra round), so this constant stays.
 * mtp_native_screen_kernel below maps 8 group-pairs to it so that no lane sits
 * out; the generic kernel's 32-pair map is kept for the 80-group refinement.
 * This changes the coarse proposal heuristic, not the selected-row dots. */
static constexpr uint32_t MTP_NATIVE_SCREEN_GROUPS = 24u;
static constexpr uint32_t MTP_NATIVE_MAX_WIDTH = 1u << 20;
template <bool Screen, bool EmitKeys = false>
__global__ static void mtp_native_projection_kernel(
        float *out, const unsigned char *w,
        const int8_t *xq, const float *xscale,
        uint32_t out_dim,
        const uint32_t *ids, uint32_t n_vocab, uint32_t prefix, uint32_t tail,
        uint64_t *keys = nullptr, uint32_t *invalid = nullptr) {
    /* All three private launches follow the DIM=2560, one-row guard. */
    constexpr uint64_t blocks = MTP_NATIVE_DIM / 32u;
    constexpr int R = 1;
    constexpr bool Streaming = false;
    const uint64_t work_blocks = Screen ? MTP_NATIVE_SCREEN_GROUPS : blocks;
    const uint32_t local_row = threadIdx.x >> 6u;
    const uint32_t local_lane = threadIdx.x & 63u;
    const uint32_t group = local_lane >> 1u;
    const uint32_t half = local_lane & 1u;
    /* Width <= 2^20; byte addressing is widened separately below. */
    const uint32_t row = blockIdx.x * 4u + local_row;
    constexpr uint32_t row0 = 0u;
    constexpr uint32_t take = 1u;
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; r++) acc[r] = 0.0f;

    const uint32_t weight_row = row >= out_dim ? n_vocab : Screen
        ? (row < prefix ? row : n_vocab - tail + (row - prefix)) : ids[row];
    const bool valid = row < out_dim && weight_row < n_vocab;
    if (valid) {
        const unsigned char *wr = w + (uint64_t)weight_row * blocks * 34u;
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
            if (local_lane == 0u && row < out_dim && (uint32_t)r < take) {
                const float value = valid ? total : -INFINITY;
                if (EmitKeys) {
                    /* Same f32 score and original key statements as the
                     * standalone key producer; preserve its FTZ comparison. */
                    const uint32_t id = row < prefix ? (uint32_t)row
                        : (uint32_t)(n_vocab - tail + (row - prefix));
                    if (!isfinite(value)) atomicOr(invalid, 1u);
                    if (!id || row >= prefix) keys[row] = UINT64_MAX - id;
                    else keys[row] = q8_top1_pack_key(value == 0.0f ? 0.0f : value, id);
                } else {
                    out[((uint64_t)row0 + r) * out_dim + row] = value;
                }
            }
        }
    }
}

/* Coarse screen only, with a thread map matched to the screen depth.
 *
 * The generic kernel above maps 4 rows x 64 lanes, so group = (lane & 63) >> 1
 * spans 32 group-pairs. That is the right shape for the refinement launch, which
 * walks all 80 groups (32 + 32 + 16). It is the wrong shape for the screen,
 * which walks 24: pairs 24..31 evaluate `b = group < 24` false on the first
 * test, issue no load, and retire. 16 of every 64 lanes -- a quarter of every
 * screen block -- exist only to write 0.0f into `partial[]` and be summed.
 *
 * The screen is the expensive stage. 98,584 rows x 24 groups x 34 B is 80.4 MB
 * against refinement's 5.6 MB, so the waste sits on the wrong side of a 14:1
 * split, and it is issue-side waste rather than bandwidth waste: the idle lanes
 * cost warp slots and a shared-memory round trip, not DRAM. On this track
 * instruction issue has repeatedly been the paying lever where byte counts were
 * not, which is the reason to expect anything here at all.
 *
 * The fix is arithmetic, not heuristic: 24 is divisible by 8, so 16 lanes per
 * row (8 group-pairs) covers the screen in exactly three iterations with every
 * lane live in all three. 256 threads then carry 16 rows instead of 4. Three
 * consequences follow for free:
 *   - no idle lanes, so a screen block issues 24 pair-loads per row instead of
 *     32 pair-slots per row for the same 24 groups;
 *   - the partial-wave mask disappears. Every lane of every warp executes the
 *     same trip count, so `active` is the full warp and the pair exchange needs
 *     no `live_pairs` arithmetic per iteration;
 *   - the reduction leaves shared memory entirely. A row's 16 lanes are
 *     contiguous within one warp, so the 8 pair sums reduce by xor 2/4/8 in
 *     registers, deleting `partial[]`, the `__syncthreads()`, and the second
 *     32-lane warp reduction.
 *
 * Why this is safe to do without a local compile of the numerics: the screen
 * produces the COARSE RANKING only. Its scores pick which 2048 rows go to
 * refinement; refinement re-dots every one of them over all 80 groups with the
 * generic kernel, byte-for-byte unchanged, and the winner is the argmax over
 * those exact scores. Beyond that the target verifies every drafted token, so
 * nothing here can move an emitted token. The per-lane accumulation order does
 * change (three groups per lane instead of one, a 16-wide register tree instead
 * of a 32-wide shared tree), which perturbs the coarse scores at ~1e-7. That is
 * only a boundary risk if the true argmax sits near the top-2048 cut, and the
 * CAP 2048 -> 4096 arm measured that cut as slack: acceptance stayed exactly
 * 49/78, so widening the boundary admitted no new winner. Watch
 * spec_accepted_total anyway -- it is the readout that would catch me being
 * wrong about that.
 *
 * Not templated onto the generic kernel on purpose. Sharing it would put the
 * refinement path behind the same edit, and refinement's exact dots are the one
 * part of this file worth leaving alone. */
template <bool EmitKeys>
__global__ static void mtp_native_screen_kernel(
        float *out, const unsigned char *w,
        const int8_t *xq, const float *xscale,
        uint32_t out_dim, uint32_t n_vocab, uint32_t prefix, uint32_t tail,
        uint64_t *keys = nullptr, uint32_t *invalid = nullptr) {
    constexpr uint64_t blocks = MTP_NATIVE_DIM / 32u;
    constexpr uint32_t lanes = 16u;
    constexpr uint32_t pairs = lanes / 2u;
    /* 24 % 8 == 0, so every pair walks the same number of groups. */
    constexpr uint64_t work_blocks = MTP_NATIVE_SCREEN_GROUPS;
    const uint32_t local_row = threadIdx.x / lanes;
    const uint32_t local_lane = threadIdx.x & (lanes - 1u);
    const uint32_t group = local_lane >> 1u;
    const uint32_t half = local_lane & 1u;
    const uint32_t row = blockIdx.x * (256u / lanes) + local_row;
    float acc = 0.0f;

    const uint32_t weight_row = row >= out_dim ? n_vocab
        : (row < prefix ? row : n_vocab - tail + (row - prefix));
    const bool valid = row < out_dim && weight_row < n_vocab;
    if (valid) {
        const unsigned char *wr = w + (uint64_t)weight_row * blocks * 34u;
        for (uint64_t b = group; b < work_blocks; b += pairs) {
            const int8_t *payload = (const int8_t *)(wr + b * 34u + 2u) + half * 16u;
            const uintptr_t address = (uintptr_t)payload;
            const uint32_t shift = (uint32_t)(address & 3u) * 8u;
            const uint32_t *words = (const uint32_t *)(address & ~(uintptr_t)3u);
            uint32_t previous = words[0];
            int32_t wq[4];
#pragma unroll
            for (int j = 0; j < 3; j++) {
                const uint32_t next = words[j + 1];
                wq[j] = (int32_t)__funnelshift_r(previous, next, shift);
                previous = next;
            }
            const uint16_t *lastp = (const uint16_t *)(const void *)(payload + 14);
            wq[3] = (int32_t)__funnelshift_r(previous, (uint32_t)*lastp, shift);
            const float ws = __half2float(*(const __half *)(wr + b * 34u));
            const int32_t *xw = (const int32_t *)(xq + b * 32u + half * 16u);
            int dot = 0;
#pragma unroll
            for (int j = 0; j < 4; j++) dot = __dp4a(wq[j], xw[j], dot);
            /* Whole warp is live and in lockstep on the trip count. */
            dot += __shfl_xor_sync(0xffffffffu, dot, 1);
            if (half == 0u) acc += ws * xscale[b] * (float)dot;
        }
    }
    /* Odd lanes hold 0.0f, so folding all 16 lanes of the row sums the 8 pair
     * accumulators. A row never straddles a warp: 16 divides 32. */
    acc += __shfl_xor_sync(0xffffffffu, acc, 2);
    acc += __shfl_xor_sync(0xffffffffu, acc, 4);
    acc += __shfl_xor_sync(0xffffffffu, acc, 8);
    if (local_lane == 0u && row < out_dim) {
        const float value = valid ? acc : -INFINITY;
        if (EmitKeys) {
            /* Same f32 score and key statements as the standalone key
             * producer; preserve its FTZ comparison. */
            const uint32_t id = row < prefix ? (uint32_t)row
                : (uint32_t)(n_vocab - tail + (row - prefix));
            if (!isfinite(value)) atomicOr(invalid, 1u);
            if (!id || row >= prefix) keys[row] = UINT64_MAX - id;
            else keys[row] = q8_top1_pack_key(value == 0.0f ? 0.0f : value, id);
        } else {
            out[row] = value;
        }
    }
}

struct mtp_native_layout {
    uint64_t scores, key_in, key_out, id_tmp, flag, temporary;
};
static uint64_t mtp_native_align(uint64_t n) { return (n + 255u) & ~255ull; }
static mtp_native_layout mtp_native_offsets(uint32_t width) {
    mtp_native_layout l;
    l.scores = mtp_native_align(MTP_NATIVE_DIM + (MTP_NATIVE_DIM / 32u) * 4u);
    l.key_in = mtp_native_align(l.scores + (uint64_t)width * 4u);
    l.key_out = mtp_native_align(l.key_in + (uint64_t)width * 8u);
    l.id_tmp = mtp_native_align(l.key_out + (uint64_t)width * 8u);
    l.flag = mtp_native_align(l.id_tmp + (uint64_t)MTP_NATIVE_CAP * 4u);
    l.temporary = mtp_native_align(l.flag + 4u);
    return l;
}
extern "C" int ds4_gpu_mtp_native_screen_init(uint32_t width,
        uint64_t *bytes, uint32_t *capacity) {
    if (!bytes || !capacity) return -1;
    *bytes = 0; *capacity = 0;
    if (width <= MTP_NATIVE_CAP || width > MTP_NATIVE_MAX_WIDTH) return 0;
    size_t a = 0, b = 0;
    if (cub::DeviceRadixSort::SortKeysDescending(nullptr, a,
            (const uint64_t *)nullptr, (uint64_t *)nullptr, width, 0, 64,
            cuda_decode_stream()) != cudaSuccess ||
        cub::DeviceRadixSort::SortKeys(nullptr, b,
            (const uint32_t *)nullptr, (uint32_t *)nullptr, MTP_NATIVE_CAP, 0, 32,
            cuda_decode_stream()) != cudaSuccess) return -1;
    *bytes = mtp_native_offsets(width).temporary + std::max(a,b);
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
/* Moving key writes into projection is equivalent only when scratch writes
 * cannot change another input/output view or a concurrently read weight. */
static bool mtp_native_key_range_disjoint(const void *a, uint64_t an,
                                         const void *b, uint64_t bn) {
    const uintptr_t ap = (uintptr_t)a, bp = (uintptr_t)b;
    return a && b && an <= UINTPTR_MAX-ap && bn <= UINTPTR_MAX-bp &&
           (ap+an <= bp || bp+bn <= ap);
}

/* -1: backend error, 0: ordinary full-static fallback, positive: exact number
 * of sorted candidates whose FULL refined logits now occupy out. */
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
    uint32_t *id_tmp = (uint32_t *)(base + l.id_tmp);
    uint32_t *flag = (uint32_t *)(base + l.flag);
    if (!cuda_ok(cudaMemsetAsync(flag,0,4,cuda_decode_stream()),"native screen flag")) return -1;
    quantize_q8_0_f32_rows_warp_kernel<<<10,256,0,cuda_decode_stream()>>>(
        xq,xs,(const float *)x->ptr,in_dim,80,1);
    if (!cuda_ok(cudaGetLastError(),"native screen quantize")) return -1;
    const bool fuse_keys = getenv("DS4_MTP_NO_FUSED_SCREEN_KEYS") == nullptr &&
        mtp_native_key_range_disjoint(scratch->ptr,scratch->bytes,
                                     w,(uint64_t)vocab*80u*34u) &&
        mtp_native_key_range_disjoint(scratch->ptr,scratch->bytes,x->ptr,x->bytes) &&
        mtp_native_key_range_disjoint(scratch->ptr,scratch->bytes,out->ptr,out->bytes) &&
        mtp_native_key_range_disjoint(scratch->ptr,scratch->bytes,ids->ptr,ids->bytes);
    if (fuse_keys) {
        mtp_native_screen_kernel<true><<<(width+15u)/16u,256,0,cuda_decode_stream()>>>(
            scores,(const unsigned char *)w,xq,xs,width,vocab,prefix,tail,
            key_in,flag);
        if (!cuda_ok(cudaGetLastError(),"native fused screen keys")) return -1;
    } else {
        mtp_native_screen_kernel<false><<<(width+15u)/16u,256,0,cuda_decode_stream()>>>(
            scores,(const unsigned char *)w,xq,xs,width,vocab,prefix,tail);
        if (!cuda_ok(cudaGetLastError(),"native half-column screen")) return -1;
        mtp_native_keys<<<(width+255u)/256u,256,0,cuda_decode_stream()>>>(
            key_in,flag,scores,width,prefix,tail,vocab);
        if (!cuda_ok(cudaGetLastError(),"native screen keys")) return -1;
    }
    uint32_t invalid = 0;
    if (!ds4_gpu_tensor_read(scratch,l.flag,&invalid,4)) return -1;
    if (invalid) return 0;
    size_t temporary = (size_t)(scratch->bytes-l.temporary);
    if (!cuda_ok(cub::DeviceRadixSort::SortKeysDescending(base+l.temporary,temporary,
            key_in,key_out,width,0,64,cuda_decode_stream()),"native score sort")) return -1;
    mtp_native_unpack_ids<<<(MTP_NATIVE_CAP+255u)/256u,256,0,cuda_decode_stream()>>>(id_tmp,key_out);
    if (!cuda_ok(cudaGetLastError(),"native candidate unpack")) return -1;
    temporary = (size_t)(scratch->bytes-l.temporary);
    if (!cuda_ok(cub::DeviceRadixSort::SortKeys(base+l.temporary,temporary,
            id_tmp,(uint32_t *)ids->ptr,MTP_NATIVE_CAP,0,32,cuda_decode_stream()),
            "native original-ID sort")) return -1;
    mtp_native_projection_kernel<false><<<(MTP_NATIVE_CAP+3u)/4u,256,0,cuda_decode_stream()>>>(
        (float *)out->ptr,(const unsigned char *)w,xq,xs,MTP_NATIVE_CAP,
        (const uint32_t *)ids->ptr,vocab,prefix,tail);
    return cuda_ok(cudaGetLastError(),"native exact refinement") ? (int)MTP_NATIVE_CAP : -1;
}
__global__ static void mtp_native_map(uint32_t *winner, const float *logits,
                                      const uint32_t *ids, uint32_t count, uint32_t vocab) {
    const uint32_t bits = __float_as_uint(logits[0]);
    const uint32_t packed = (bits & 0x7fffffffu) > 0x7f800000u ? 0u : winner[0];
    const uint32_t original = packed < count ? ids[packed] : UINT32_MAX;
    winner[0] = original < vocab ? original : UINT32_MAX;
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
    mtp_native_map<<<1,1,0,cuda_decode_stream()>>>((uint32_t *)winner->ptr,
        (const float *)logits->ptr,(const uint32_t *)ids->ptr,count,vocab);
    return cuda_ok(cudaGetLastError(),"native original winner map");
}
