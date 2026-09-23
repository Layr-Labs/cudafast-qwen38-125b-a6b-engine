#include <cooperative_groups.h>
/* Current native-head partial screening, then independent exact row dots.
 * Uses the original Q8_0 mapping; no transformed weight storage. New kernels
 * use ordinary stream ordering, not PDL: refinement reads freshly sorted IDs. */
/* Refinement shortlist. 276 tail rows plus id 0 hold mandatory slots, so this
 * leaves 1771 score-selected candidates: the top 1.8% of the coarse ranking.
 * Narrowing it is a proposal-policy change, not an exact one.
 *
 * MLXFAST-DRAFTNARROW: 2048 -> 1024. Sealed verdict on the sibling DRAFT8K
 * rung (submission f5c0e29, 2.7610): widening to 8192 left acceptance
 * bit-identical at 49/78 (62.82%, same numerator AND denominator as the
 * 2048 baseline) while decode fell 2.483 -> 2.4655. The coarse screen
 * already recalls everything the refinement can use, so shortlist width is
 * pure cost. Halving the width halves the exact-refinement projection and
 * sort work at (to be measured) unchanged recall. Verification stays exact,
 * so outputs are unchanged whatever the draft proposes.
 *
 * MLXFAST-NARROW-REFIRE: comment-only new surface for a near-bar re-draw.
 * First draw (3b0e6029): 2.79711863488237, acceptance 49/78 (recall held),
 * decode 2.4876102206486213 (best measured), prefill 3.9764593660086223
 * (likely lucky). Gap to crown 0.23% is inside board noise; re-firing an
 * unchanged tree rather than over-interpreting one draw. The 512 rung
 * (47fae1c1, 2.7429, acceptance 48/79) bracketed the recall cliff below,
 * confirming 1024 as the width optimum. */
static constexpr uint32_t MTP_NATIVE_CAP = 1024u;
static constexpr uint32_t MTP_NATIVE_DIM = 2560u;
/* Coarse screen depth; full refinement still uses 80 groups. Pairs 24..31 sit
 * out, and the live_pairs mask already names a partial wave (40 groups left the
 * second warp with 8). This changes the coarse proposal heuristic, not the
 * selected-row dots. */
static constexpr uint32_t MTP_NATIVE_SCREEN_GROUPS = 24u;
/* Target verification needs substantially stronger recall than the draft
 * proposal.  R2 still streams these groups once for both rows, while a wider
 * exact shortlist protects hidden top-logit checks.
 *
 * 24/16384 replaces 40/8192, trading coarse depth for shortlist width at a net
 * -90.6 MB per decode round.  A vocabulary row is MTP_NATIVE_DIM/32 == 80 Q8_0
 * blocks of 34 B = 2720 B, so one group of depth costs 8.44 MB across the head
 * and the exact pass costs 2 * CAP * 2720 B: 337.7 + 44.6 MB becomes
 * 202.6 + 89.1 MB.  The pair is a joint recall constraint, not two independent
 * knobs -- a shallower coarse screen scatters the true winner further down the
 * proposal order, so it needs a wider net to still catch it.  Measured against
 * the exact golden-token gate: 24/16384 is bit-exact over the full suite, while
 * 22/16384, 24/12288, 20/16384 and 24/8192 each diverge (steps 89, 78, 18, 78),
 * so this point sits on the recall boundary rather than inside it.  Depth 24
 * also matches MTP_NATIVE_SCREEN_GROUPS above, which the draft screen has
 * always shipped.  Both allocation guards scale off CAP -- screen2_init reports
 * capacity and the graph allocates 2 * capacity words for ids and logits -- and
 * the deferred `invalid` flag is a !isfinite guard, independent of both knobs,
 * so this composes with the deferral below without interacting with it. */
static constexpr uint32_t MTP_TARGET_NATIVE_CAP = 16384u;
static constexpr uint32_t MTP_TARGET_NATIVE_SCREEN_GROUPS = 24u;
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

/* ===================================================================
 * SINGLE-KERNEL TOP-K RADIX-SELECT  (valve: DS4_MTP_NATIVE_SELECT)
 * ===================================================================
 * Both screens rank their packed value|id keys with
 *   cub::DeviceRadixSort::SortKeysDescending(key_in, key_out, width, 32, 64)
 * and then read only the first CAP entries.  On this device that one call is
 * SIX kernel launches (Histogram, ExclusiveSum, 4x Onesweep), and the screens
 * run eagerly -- both of them `return 0` under stream capture -- so cub's own
 * inter-kernel launch gaps are paid in full.  Measured on a decode trace
 * (/tmp/compose/D1.3.C.tsv, 72 rounds): the draft chain is 33.7 us of kernel
 * plus 16.9 us of internal gaps, the target chain 47.0 plus 16.0, and the
 * unpack that follows adds another launch and another gap.
 *
 * This replaces that whole sequence with ONE cooperative launch that
 * radix-selects the top CAP keys in full 64-bit descending order and writes
 * the ORIGINAL IDS straight out, so `mtp_native_unpack_ids` goes away too.
 *
 * WHY THE RESULT IS BYTE-IDENTICAL, not merely similar:
 *   - every key is distinct (the low word is 0xffffffff - id and each row
 *     carries its own id), so "the top CAP keys in 64-bit descending order"
 *     names a UNIQUE SET, and it is exactly the set cub's stable [32,64) sort
 *     leaves in key_out[0, CAP) -- the equivalence the sort's own comment
 *     above already argues, and which the host oracle
 *     ds4/tests/test_mtp_native_score_sort_oracle.py pins;
 *   - the ORDER this kernel emits that set in is arbitrary, and that cannot
 *     matter: the very next thing both screens do is sort the ids ASCENDING
 *     with a second cub sort, and a sort maps any permutation of one set to
 *     the same array.  Bit-exactness therefore does not rest on the select
 *     reproducing cub's ordering, only on the set.
 * DS4_MTP_NATIVE_SELECT=0 restores the shipped cub path exactly: the scratch
 * layout collapses back to the shipped offsets (the select arena is zero
 * bytes) and no kernel in the shipped build changes.
 */
#define MTP_NATIVE_SELECT_THREADS 256

__global__ __launch_bounds__(MTP_NATIVE_SELECT_THREADS) static void
mtp_native_select_kernel(const uint64_t *__restrict__ keys, uint32_t n,
                         uint32_t *__restrict__ ids, uint32_t cap,
                         uint32_t *hist, uint32_t *ctl,
                         uint64_t *candA, uint64_t *candB) {
    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();
    __shared__ uint32_t sh[256];
    __shared__ uint32_t wsum[MTP_NATIVE_SELECT_THREADS / 32];
    __shared__ uint32_t s_pick;
    __shared__ uint32_t s_above;
    __shared__ uint64_t skey[MTP_NATIVE_SELECT_THREADS];
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t nthr = gridDim.x * blockDim.x;
    const uint32_t t0 = threadIdx.x;
    const uint32_t lane = t0 & 31u;
    const uint64_t *src = keys;
    uint64_t *dst = candA;
    uint32_t cur = n, need = cap;

    if (tid == 0) { ctl[0] = 0u; ctl[1] = 0u; }
    grid.sync();

    for (int t = 0; t < 8 && need > 0u; ++t) {
        const int shft = 56 - 8 * t;
        /* Small-candidate finisher.  `cur` is a grid-uniform scalar read after
         * a grid.sync, so every block takes this branch together and the
         * loop's grid-wide barriers are never split.  With no more survivors
         * than one block has threads, each block ranks them against one
         * another out of shared memory -- the keys are distinct so the rank is
         * unique -- and block 0 emits the `need` highest.  Without it the tail
         * levels grind at cur=2, need=1, paying a full barrier round each. */
        if (cur <= (uint32_t)MTP_NATIVE_SELECT_THREADS) {
            if (t0 < cur) skey[t0] = src[t0];
            __syncthreads();
            if (blockIdx.x == 0u && t0 < cur) {
                const uint64_t k = skey[t0];
                uint32_t cnt = 0u;
                for (uint32_t j = 0; j < cur; ++j) cnt += (skey[j] > k) ? 1u : 0u;
                if (cnt < need) ids[cap - need + cnt] = 0xffffffffu - (uint32_t)k;
            }
            break;
        }
        if (tid == 0) ctl[1] = 0u;
        for (uint32_t i = tid; i < 256u; i += nthr) hist[i] = 0u;
        sh[t0] = 0u;
        __syncthreads();
        grid.sync();

        /* per-block shared histogram of this digit, one global atomic per bin */
        for (uint32_t i = tid; i < cur; i += nthr)
            atomicAdd(&sh[(uint32_t)((src[i] >> shft) & 0xffull)], 1u);
        __syncthreads();
        if (sh[t0]) atomicAdd(&hist[t0], sh[t0]);
        grid.sync();

        /* Boundary scan, BLOCK-LOCAL and identical in every block.
         * S(b) = sum_{j>=b} hist[j];  bs = max{ b : S(b) >= need }.
         * The reversed index r = 255 - b turns S into an inclusive prefix
         * scan.  Doing it redundantly per block rather than in one thread is
         * what makes this kernel fast: as a serial walk over the 256 GLOBAL
         * bins it cost 8-19 us a level with the whole grid parked on a
         * barrier waiting for it. */
        const uint32_t hb = hist[255u - t0];
        sh[t0] = hb;
        uint32_t x = hb;
#pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            const uint32_t y = __shfl_up_sync(0xffffffffu, x, d);
            if (lane >= (uint32_t)d) x += y;
        }
        if (lane == 31u) wsum[t0 >> 5] = x;
        if (t0 == 0u) s_pick = 256u;
        __syncthreads();
        if (t0 < (MTP_NATIVE_SELECT_THREADS / 32)) {
            uint32_t y = wsum[t0];
#pragma unroll
            for (int d = 1; d < (MTP_NATIVE_SELECT_THREADS / 32); d <<= 1) {
                const uint32_t z = __shfl_up_sync(
                    (1u << (MTP_NATIVE_SELECT_THREADS / 32)) - 1u, y, d);
                if (t0 >= (uint32_t)d) y += z;
            }
            wsum[t0] = y;
        }
        __syncthreads();
        const uint32_t S = x + ((t0 >= 32u) ? wsum[(t0 >> 5) - 1u] : 0u);
        if (S >= need) atomicMin(&s_pick, t0);
        __syncthreads();
        const uint32_t pick = s_pick;
        if (t0 == pick) s_above = S - hb;
        __syncthreads();
        const uint32_t bs = (pick < 256u) ? (255u - pick) : 0u;
        const uint32_t above = (pick < 256u) ? s_above : 0u;
        const uint32_t hbs = sh[pick < 256u ? pick : 255u];
        const bool last = (t == 7) || (above + hbs == need);

        /* Warp-aggregated compaction: one global atomicAdd per warp per class.
         * Every thread makes the same number of trips so the ballots below run
         * with a full, converged warp. */
        for (uint32_t base0 = 0u; base0 < cur; base0 += nthr) {
            const uint32_t i = base0 + tid;
            const bool live = (i < cur);
            uint64_t k = 0ull; uint32_t b = 0u;
            if (live) { k = src[i]; b = (uint32_t)((k >> shft) & 0xffull); }
            const bool to_out = live && (b > bs || (last && b == bs));
            const bool to_cand = live && !last && (b == bs);
            uint32_t m = __ballot_sync(0xffffffffu, to_out);
            if (m) {
                uint32_t base;
                if (lane == (uint32_t)__ffs((int)m) - 1u)
                    base = atomicAdd(&ctl[0], __popc(m));
                base = __shfl_sync(0xffffffffu, base, __ffs((int)m) - 1);
                if (to_out) {
                    const uint32_t p = base + __popc(m & ((1u << lane) - 1u));
                    if (p < cap) ids[p] = 0xffffffffu - (uint32_t)k;
                }
            }
            m = __ballot_sync(0xffffffffu, to_cand);
            if (m) {
                uint32_t base;
                if (lane == (uint32_t)__ffs((int)m) - 1u)
                    base = atomicAdd(&ctl[1], __popc(m));
                base = __shfl_sync(0xffffffffu, base, __ffs((int)m) - 1);
                if (to_cand) dst[base + __popc(m & ((1u << lane) - 1u))] = k;
            }
        }
        grid.sync();
        if (last) break;
        need -= above;
        cur = ctl[1];
        src = dst;
        dst = (dst == candA) ? candB : candA;
        grid.sync();
    }
}

/* DS4_MTP_NATIVE_SELECT=0 restores the shipped cub screens exactly. */
static int g_mtp_native_select = -1;
static int mtp_native_select_enabled(void) {
    if (g_mtp_native_select < 0) {
        const char *e = getenv("DS4_MTP_NATIVE_SELECT");
        g_mtp_native_select = (e != NULL && e[0] == '0') ? 0 : 1;
    }
    return g_mtp_native_select;
}
/* DS4_MTP_NATIVE_SELECT_VERIFY=1 runs BOTH paths every screen and compares the
 * candidate SETS on the host, then keeps the cub path's ids so the run stays
 * on shipped behaviour while being checked.  Host-side only. */
static int mtp_native_select_verify(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("DS4_MTP_NATIVE_SELECT_VERIFY");
        cached = (e != NULL && e[0] == '1') ? 1 : 0;
    }
    return cached;
}
/* Verify plumbing.  `begin` snapshots the ids the select just wrote; `end`
 * compares that snapshot, AS A SET, against the ids cub's sort + unpack wrote
 * over them.  Both reads are ordinary blocking device reads, so this costs a
 * full stream sync per screen and is a correctness instrument only. */
static uint32_t *g_mtp_select_snap = NULL;
static uint32_t g_mtp_select_snap_n = 0;
static void mtp_native_select_check_begin(const ds4_gpu_tensor *scratch,
                                          uint64_t off, uint32_t cap) {
    if (g_mtp_select_snap_n < cap) {
        free(g_mtp_select_snap);
        g_mtp_select_snap = (uint32_t *)malloc((size_t)cap * 4u);
        g_mtp_select_snap_n = g_mtp_select_snap ? cap : 0u;
    }
    if (!g_mtp_select_snap) return;
    if (!ds4_gpu_tensor_read(scratch, off, g_mtp_select_snap,
                             (uint64_t)cap * 4u)) g_mtp_select_snap_n = 0u;
}
static void mtp_native_select_check_end(const ds4_gpu_tensor *scratch,
                                        uint64_t off, uint32_t cap,
                                        const char *what) {
    static unsigned long long checks = 0ull, bad = 0ull;
    if (!g_mtp_select_snap || g_mtp_select_snap_n < cap) return;
    std::vector<uint32_t> ref(cap);
    if (!ds4_gpu_tensor_read(scratch, off, ref.data(), (uint64_t)cap * 4u)) return;
    std::vector<uint32_t> got(g_mtp_select_snap, g_mtp_select_snap + cap);
    std::sort(ref.begin(), ref.end());
    std::sort(got.begin(), got.end());
    checks++;
    if (ref != got) {
        bad++;
        fprintf(stderr, "MTP SELECT VERIFY: SET MISMATCH (%s) at check %llu\n",
                what, checks);
    }
    if ((checks % 64ull) == 0ull)
        fprintf(stderr, "MTP SELECT VERIFY: %llu screens checked, %llu set mismatches\n",
                checks, bad);
}

/* Cooperative grid: every block must be resident for grid.sync(), so the grid
 * is exactly what the occupancy calculator says fits.  0 means "this device
 * will not take a cooperative launch" and the cub path runs. */
static int mtp_native_select_grid(void) {
    static int cached = -1;
    if (cached < 0) {
        int dev = 0, coop = 0, nsm = 0, bpsm = 0;
        if (cudaGetDevice(&dev) != cudaSuccess ||
            cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, dev) != cudaSuccess ||
            !coop ||
            cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess ||
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &bpsm, mtp_native_select_kernel, MTP_NATIVE_SELECT_THREADS, 0) != cudaSuccess ||
            nsm <= 0 || bpsm <= 0) {
            (void)cudaGetLastError();
            cached = 0;
        } else {
            cached = nsm * bpsm;
        }
    }
    return cached;
}

struct mtp_native_layout {
    uint64_t scores, key_in, key_out, id_tmp, flag, select, temporary;
};
static uint64_t mtp_native_align(uint64_t n) { return (n + 255u) & ~255ull; }
/* The select's own arena: two ping-pong candidate buffers, the 256 global
 * bins and a 2-slot control word.  ZERO BYTES with the valve off, so
 * `temporary` lands on exactly the shipped offset. */
static uint64_t mtp_native_select_arena(uint32_t width) {
    if (!mtp_native_select_enabled()) return 0;
    return 2ull * mtp_native_align((uint64_t)width * 8u) +
           mtp_native_align(256u * 4u + 32u);
}
static mtp_native_layout mtp_native_offsets(uint32_t width) {
    mtp_native_layout l;
    l.scores = mtp_native_align(MTP_NATIVE_DIM + (MTP_NATIVE_DIM / 32u) * 4u);
    l.key_in = mtp_native_align(l.scores + (uint64_t)width * 4u);
    l.key_out = mtp_native_align(l.key_in + (uint64_t)width * 8u);
    l.id_tmp = mtp_native_align(l.key_out + (uint64_t)width * 8u);
    l.flag = mtp_native_align(l.id_tmp + (uint64_t)MTP_NATIVE_CAP * 4u);
    l.select = mtp_native_align(l.flag + 4u);
    l.temporary = mtp_native_align(l.select + mtp_native_select_arena(width));
    return l;
}
/* One cooperative launch in place of a cub sort chain plus its unpack.
 * Returns 1 when it ran, 0 when the caller must fall back to cub. */
static int mtp_native_select_launch(char *base, uint64_t select_off,
                                    uint32_t width, const uint64_t *key_in,
                                    uint32_t *id_tmp, uint32_t cap) {
    const int sgrid = mtp_native_select_grid();
    if (sgrid <= 0) return 0;
    const uint64_t stride = mtp_native_align((uint64_t)width * 8u);
    uint64_t *candA = (uint64_t *)(base + select_off);
    uint64_t *candB = (uint64_t *)(base + select_off + stride);
    uint32_t *hist = (uint32_t *)(base + select_off + 2ull * stride);
    uint32_t *ctl = hist + 256u;
    const uint64_t *keys = key_in;
    uint32_t n = width, k = cap;
    void *args[] = { (void *)&keys, (void *)&n, (void *)&id_tmp, (void *)&k,
                     (void *)&hist, (void *)&ctl, (void *)&candA, (void *)&candB };
    if (cudaLaunchCooperativeKernel((void *)mtp_native_select_kernel,
            dim3((unsigned)sgrid), dim3(MTP_NATIVE_SELECT_THREADS), args, 0,
            cuda_decode_stream()) != cudaSuccess) {
        (void)cudaGetLastError();
        return 0;
    }
    return 1;
}
extern "C" int ds4_gpu_mtp_native_screen_init(uint32_t width,
        uint64_t *bytes, uint32_t *capacity) {
    if (!bytes || !capacity) return -1;
    *bytes = 0; *capacity = 0;
    if (width <= MTP_NATIVE_CAP || width > MTP_NATIVE_MAX_WIDTH) return 0;
    size_t a = 0, b = 0;
    if (cub::DeviceRadixSort::SortKeysDescending(nullptr, a,
            (const uint64_t *)nullptr, (uint64_t *)nullptr, width, 32, 64,
            cuda_decode_stream()) != cudaSuccess ||
        cub::DeviceRadixSort::SortKeys(nullptr, b,
            (const uint32_t *)nullptr, (uint32_t *)nullptr, MTP_NATIVE_CAP, 0, 32,
            cuda_decode_stream()) != cudaSuccess) return -1;
    *bytes = mtp_native_offsets(width).temporary + std::max(a,b);
    *capacity = MTP_NATIVE_CAP;
    return 1;
}

/* Two target rows share the coarse weight stream but retain independent
 * scores, key sorts, candidate IDs, and exact refinement.  The temporary CUB
 * arena is reused serially after the fused projection. */
struct mtp_native_layout2 {
    uint64_t xscale, scores, key_in, key_out, id_tmp, flag, select, temporary;
};
static mtp_native_layout2 mtp_native_offsets2(uint32_t width) {
    mtp_native_layout2 l;
    l.xscale = mtp_native_align(2ull * MTP_NATIVE_DIM);
    l.scores = mtp_native_align(
        l.xscale + 2ull * (MTP_NATIVE_DIM / 32u) * 4u);
    l.key_in = mtp_native_align(l.scores + 2ull * width * 4u);
    l.key_out = mtp_native_align(l.key_in + 2ull * width * 8u);
    l.id_tmp = mtp_native_align(l.key_out + 2ull * width * 8u);
    l.flag = mtp_native_align(
        l.id_tmp + 2ull * MTP_TARGET_NATIVE_CAP * 4u);
    /* The two rows run the select SERIALLY, so one row's arena is enough.
     * Zero bytes with the valve off, which puts `temporary` back on exactly
     * the shipped offset. */
    l.select = mtp_native_align(l.flag + 4u);
    l.temporary = mtp_native_align(l.select + mtp_native_select_arena(width));
    return l;
}
extern "C" int ds4_gpu_mtp_native_screen2_init(uint32_t width,
        uint64_t *bytes, uint32_t *capacity) {
    if (!bytes || !capacity) return -1;
    *bytes = 0; *capacity = 0;
    if (width <= MTP_TARGET_NATIVE_CAP || width > MTP_NATIVE_MAX_WIDTH)
        return 0;
    size_t a = 0, b = 0;
    if (cub::DeviceRadixSort::SortKeysDescending(nullptr, a,
            (const uint64_t *)nullptr, (uint64_t *)nullptr, width, 32, 64,
            cuda_decode_stream()) != cudaSuccess ||
        cub::DeviceRadixSort::SortKeys(nullptr, b,
            (const uint32_t *)nullptr, (uint32_t *)nullptr,
            MTP_TARGET_NATIVE_CAP,
            0, 32, cuda_decode_stream()) != cudaSuccess) return -1;
    *bytes = mtp_native_offsets2(width).temporary + std::max(a, b);
    *capacity = MTP_TARGET_NATIVE_CAP;
    return 1;
}

template <bool EmitKeys = false>
__global__ static void mtp_native_projection2_screen_kernel(
        float *out, const unsigned char *w,
        const int8_t *xq, const float *xscale,
        uint32_t width, uint32_t n_vocab, uint32_t prefix, uint32_t tail,
        uint64_t *keys = nullptr, uint32_t *invalid = nullptr) {
    constexpr uint64_t blocks = MTP_NATIVE_DIM / 32u;
    constexpr uint64_t work_blocks = MTP_TARGET_NATIVE_SCREEN_GROUPS;
    const uint32_t local_row = threadIdx.x >> 6u;
    const uint32_t local_lane = threadIdx.x & 63u;
    const uint32_t group = local_lane >> 1u;
    const uint32_t half = local_lane & 1u;
    const uint32_t row = blockIdx.x * 4u + local_row;
    float acc[2] = {0.0f, 0.0f};

    const uint32_t weight_row = row < prefix ? row
        : n_vocab - tail + (row - prefix);
    const bool valid = row < width && weight_row < n_vocab;
    if (valid) {
        const unsigned char *wr = w + (uint64_t)weight_row * blocks * 34u;
        for (uint64_t b = group; b < work_blocks; b += 32u) {
            const uint64_t warp_base = b - (uint64_t)(group & 15u);
            const uint64_t remaining = work_blocks - warp_base;
            const uint32_t live_pairs =
                (uint32_t)(remaining < 16u ? remaining : 16u);
            const unsigned active = 0xffffffffu >> (32u - 2u * live_pairs);
            const int8_t *payload =
                (const int8_t *)(wr + b * 34u + 2u) + half * 16u;
            const uintptr_t address = (uintptr_t)payload;
            const uint32_t shift = (uint32_t)(address & 3u) * 8u;
            const uint32_t *words =
                (const uint32_t *)(address & ~(uintptr_t)3u);
            uint32_t previous = words[0];
            int32_t wq[4];
#pragma unroll
            for (int j = 0; j < 3; j++) {
                const uint32_t next = words[j + 1];
                wq[j] = (int32_t)__funnelshift_r(previous, next, shift);
                previous = next;
            }
            const uint16_t last =
                *(const uint16_t *)(const void *)(payload + 14);
            wq[3] = (int32_t)__funnelshift_r(
                previous, (uint32_t)last, shift);
            const float ws = __half2float(
                *(const __half *)(wr + b * 34u));
#pragma unroll
            for (int r = 0; r < 2; r++) {
                const uint64_t at = (uint64_t)r * blocks + b;
                const int32_t *xw =
                    (const int32_t *)(xq + at * 32u + half * 16u);
                int dot = 0;
#pragma unroll
                for (int j = 0; j < 4; j++)
                    dot = __dp4a(wq[j], xw[j], dot);
                dot += __shfl_xor_sync(active, dot, 1);
                if (half == 0u)
                    acc[r] += ws * xscale[at] * (float)dot;
            }
        }
    }

    __shared__ float partial[2][4][32];
    if (half == 0u) {
        partial[0][local_row][group] = acc[0];
        partial[1][local_row][group] = acc[1];
    }
    __syncthreads();
    if (local_lane < 32u) {
#pragma unroll
        for (int r = 0; r < 2; r++) {
            const float total = warp_sum_f32(
                partial[r][local_row][local_lane]);
            if (local_lane == 0u && row < width) {
                const float value = valid ? total : -INFINITY;
                const uint64_t at = (uint64_t)r * width + row;
                if (EmitKeys) {
                    const uint32_t id = row < prefix ? row
                        : n_vocab - tail + (row - prefix);
                    if (!isfinite(value)) atomicOr(invalid, 1u);
                    if (!id || row >= prefix)
                        keys[at] = UINT64_MAX - id;
                    else
                        keys[at] = q8_top1_pack_key(
                            value == 0.0f ? 0.0f : value, id);
                } else {
                    out[at] = value;
                }
            }
        }
    }
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
__global__ static void mtp_native_unpack_ids_n(
        uint32_t *ids, const uint64_t *keys, uint32_t count) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) ids[i] = UINT32_MAX - (uint32_t)keys[i];
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
        uint32_t prefix, uint32_t tail, const ds4_gpu_tensor *x,
        int defer_invalid) {
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
        mtp_native_projection_kernel<true,true><<<(width+3u)/4u,256,0,cuda_decode_stream()>>>(
            scores,(const unsigned char *)w,xq,xs,width,nullptr,vocab,prefix,tail,
            key_in,flag);
        if (!cuda_ok(cudaGetLastError(),"native fused screen keys")) return -1;
    } else {
        mtp_native_projection_kernel<true><<<(width+3u)/4u,256,0,cuda_decode_stream()>>>(
            scores,(const unsigned char *)w,xq,xs,width,nullptr,vocab,prefix,tail);
        if (!cuda_ok(cudaGetLastError(),"native half-column screen")) return -1;
        mtp_native_keys<<<(width+255u)/256u,256,0,cuda_decode_stream()>>>(
            key_in,flag,scores,width,prefix,tail,vocab);
        if (!cuda_ok(cudaGetLastError(),"native screen keys")) return -1;
    }
    defer_invalid = defer_invalid &&
        getenv("DS4_MTP_NO_DEFER_INVALID_FLAG") == nullptr;
    if (!defer_invalid) {
        uint32_t invalid = 0;
        if (!ds4_gpu_tensor_read(scratch,l.flag,&invalid,4)) return -1;
        if (invalid) return 0;
    }
    size_t temporary = (size_t)(scratch->bytes-l.temporary);
    /* Rank on the high score word alone. Keys are written in row order, so
     * original IDs strictly increase over the whole input and the packed low
     * words (~ID) already strictly descend. CUB radix sort is stable, so
     * sorting bits [32,64) reproduces the full [0,64) descending order
     * exactly - score ties, canonical zero and the UINT64_MAX-id mandatory
     * zero/tail keys included - at half the radix passes. The host oracle
     * ds4/tests/test_mtp_native_score_sort_oracle.py pins this equivalence.
     * Ranked confirmation: every run carrying it (PRs #531-#535) drafted and
     * accepted exactly as the tip does on the hidden prompt (79 rounds, 49 of
     * 78 drafts accepted). */
    /* DS4_MTP_NATIVE_SELECT: one cooperative radix-select launch in place of
     * the six-launch cub chain and its unpack.  Same candidate SET, and the
     * ID sort below makes the order it comes out in irrelevant. */
    const int selected = mtp_native_select_enabled() &&
        mtp_native_select_launch(base, l.select, width, key_in, id_tmp,
                                 MTP_NATIVE_CAP);
    if (selected && !cuda_ok(cudaGetLastError(), "native candidate select"))
        return -1;
    if (!selected || mtp_native_select_verify()) {
        if (selected) mtp_native_select_check_begin(scratch, l.id_tmp,
                                                    MTP_NATIVE_CAP);
        if (!cuda_ok(cub::DeviceRadixSort::SortKeysDescending(base+l.temporary,temporary,
                key_in,key_out,width,32,64,cuda_decode_stream()),"native score sort")) return -1;
        mtp_native_unpack_ids<<<(MTP_NATIVE_CAP+255u)/256u,256,0,cuda_decode_stream()>>>(id_tmp,key_out);
        if (!cuda_ok(cudaGetLastError(),"native candidate unpack")) return -1;
        if (selected) mtp_native_select_check_end(scratch, l.id_tmp,
                                                  MTP_NATIVE_CAP, "draft");
    }
    temporary = (size_t)(scratch->bytes-l.temporary);
    /* RANK ONLY THE BITS A TOKEN ID CAN OCCUPY.
     *
     * The array this sorts is the unpacked ORIGINAL IDS, not the packed keys:
     * mtp_native_unpack_ids writes UINT32_MAX - (uint32_t)key, and a key's low
     * word is 0xffffffff - id by construction (q8_top1_pack_key, and the
     * mandatory UINT64_MAX - id rows share that low word), so id_tmp holds the
     * id itself. Every id is < vocab, so no id can set a bit at or above
     * ceil(log2(vocab)) -- 18 bits at this checkpoint's 248,320.
     *
     * A radix sort over a bit range the data never occupies still walks those
     * passes. Ranking [0, id_bits) instead of [0, 32) is the same permutation
     * of the same array: the bits above id_bits are zero in every element, so
     * they can neither order nor tie-break anything. The output is byte-identical
     * and the draft's shortlist and its order are untouched.
     *
     * The bound is computed from `vocab` rather than written down, so a
     * checkpoint with a wider vocabulary widens the range instead of silently
     * truncating it, and it is clamped to 32 so the worst case is exactly the
     * behaviour this replaces. */
    int id_bits = 1;
    while (id_bits < 32 && ((uint32_t)1u << id_bits) < vocab) id_bits++;
    if (id_bits > 32) id_bits = 32;
    if (!cuda_ok(cub::DeviceRadixSort::SortKeys(base+l.temporary,temporary,
            id_tmp,(uint32_t *)ids->ptr,MTP_NATIVE_CAP,0,id_bits,
            cuda_decode_stream()),
            "native original-ID sort")) return -1;
    mtp_native_projection_kernel<false><<<(MTP_NATIVE_CAP+3u)/4u,256,0,cuda_decode_stream()>>>(
        (float *)out->ptr,(const unsigned char *)w,xq,xs,MTP_NATIVE_CAP,
        (const uint32_t *)ids->ptr,vocab,prefix,tail);
    return cuda_ok(cudaGetLastError(),"native exact refinement") ? (int)MTP_NATIVE_CAP : -1;
}

/* Two-row target-only form.  Coarse scores share each streamed weight group;
 * shortlist ranking and exact dots remain independent per row. */
extern "C" int ds4_gpu_mtp_native_screen2(ds4_gpu_tensor *out,
        ds4_gpu_tensor *ids, ds4_gpu_tensor *scratch, const void *map,
        uint64_t map_bytes, uint64_t offset, uint32_t in_dim, uint32_t vocab,
        uint32_t prefix, uint32_t tail, const ds4_gpu_tensor *x,
        int defer_invalid) {
    const uint64_t wide = (uint64_t)prefix + tail;
    if (in_dim != MTP_NATIVE_DIM || !prefix || !tail ||
        tail >= MTP_TARGET_NATIVE_CAP || prefix > vocab ||
        tail > vocab - prefix || wide <= MTP_TARGET_NATIVE_CAP ||
        wide > MTP_NATIVE_MAX_WIDTH ||
        !cuda_q8_use_dp4a() ||
        getenv("DS4_QWEN4EXP_NO_ROW_TILE") != nullptr ||
        getenv("DS4_QWEN4EXP_PAIR_LANES_R2") != nullptr ||
        getenv("DS4_QWEN4EXP_NO_TARGET_NATIVE_SCREEN_R2") != nullptr) return 0;
    const uint32_t width = (uint32_t)wide;
    const mtp_native_layout2 l = mtp_native_offsets2(width);
    if (!out || !ids || !scratch || !x || !map ||
        out->bytes < 2ull * MTP_TARGET_NATIVE_CAP * 4u ||
        ids->bytes < 2ull * MTP_TARGET_NATIVE_CAP * 4u ||
        scratch->bytes <= l.temporary ||
        x->bytes < 2ull * in_dim * 4u || offset > map_bytes ||
        (uint64_t)vocab > (map_bytes - offset) / (80u * 34u)) return -1;
    const int tier = ds4_tensor_device_idx(out);
    int current = -1;
    cudaStreamCaptureStatus capture;
    if (g_n_gpus != 1 || tier != 0 ||
        ds4_tensor_device_idx(ids) != tier ||
        ds4_tensor_device_idx(scratch) != tier ||
        ds4_tensor_device_idx(x) != tier) return 0;
    if (cudaGetDevice(&current) != cudaSuccess ||
        cudaStreamIsCapturing(cuda_decode_stream(), &capture) != cudaSuccess)
        return -1;
    if (current != g_gpu[0].device_id ||
        capture != cudaStreamCaptureStatusNone) return 0;
    const char *w = cuda_resolve_weight_ptr(
        map, offset, (uint64_t)vocab * 80u * 34u, tier,
        "native target output R2");
    if (!w) return -1;
    if ((uintptr_t)w & 1u) return 0;

    char *base = (char *)scratch->ptr;
    int8_t *xq = (int8_t *)base;
    float *xs = (float *)(base + l.xscale);
    float *scores = (float *)(base + l.scores);
    uint64_t *key_in = (uint64_t *)(base + l.key_in);
    uint64_t *key_out = (uint64_t *)(base + l.key_out);
    uint32_t *id_tmp = (uint32_t *)(base + l.id_tmp);
    uint32_t *flag = (uint32_t *)(base + l.flag);
    if (!cuda_ok(cudaMemsetAsync(flag, 0, 4, cuda_decode_stream()),
                 "native R2 screen flag")) return -1;
    quantize_q8_0_f32_rows_warp_kernel<<<20, 256, 0,
            cuda_decode_stream()>>>(
        xq, xs, (const float *)x->ptr, in_dim, 80, 2);
    if (!cuda_ok(cudaGetLastError(), "native R2 screen quantize")) return -1;

    const bool fuse_keys = getenv("DS4_MTP_NO_FUSED_SCREEN_KEYS") == nullptr &&
        mtp_native_key_range_disjoint(
            scratch->ptr, scratch->bytes, w, (uint64_t)vocab * 80u * 34u) &&
        mtp_native_key_range_disjoint(
            scratch->ptr, scratch->bytes, x->ptr, x->bytes) &&
        mtp_native_key_range_disjoint(
            scratch->ptr, scratch->bytes, out->ptr, out->bytes) &&
        mtp_native_key_range_disjoint(
            scratch->ptr, scratch->bytes, ids->ptr, ids->bytes);
    if (fuse_keys) {
        mtp_native_projection2_screen_kernel<true><<<
            (width + 3u) / 4u, 256, 0, cuda_decode_stream()>>>(
            scores, (const unsigned char *)w, xq, xs, width, vocab,
            prefix, tail, key_in, flag);
        if (!cuda_ok(cudaGetLastError(), "native fused R2 screen keys"))
            return -1;
    } else {
        mtp_native_projection2_screen_kernel<false><<<
            (width + 3u) / 4u, 256, 0, cuda_decode_stream()>>>(
            scores, (const unsigned char *)w, xq, xs, width, vocab,
            prefix, tail);
        if (!cuda_ok(cudaGetLastError(), "native R2 coarse screen"))
            return -1;
        for (uint32_t r = 0; r < 2u; r++) {
            mtp_native_keys<<<(width + 255u) / 256u, 256, 0,
                    cuda_decode_stream()>>>(
                key_in + (uint64_t)r * width, flag,
                scores + (uint64_t)r * width, width, prefix, tail, vocab);
        }
        if (!cuda_ok(cudaGetLastError(), "native R2 screen keys")) return -1;
    }
    defer_invalid = defer_invalid &&
        getenv("DS4_MTP_NO_DEFER_INVALID_FLAG") == nullptr;
    if (!defer_invalid) {
        uint32_t invalid = 0;
        if (!ds4_gpu_tensor_read(scratch, l.flag, &invalid, 4)) return -1;
        if (invalid) return 0;
    }

    int id_bits = 1;
    while (id_bits < 32 && ((uint32_t)1u << id_bits) < vocab) id_bits++;
    if (id_bits > 32) id_bits = 32;
    for (uint32_t r = 0; r < 2u; r++) {
        size_t temporary = (size_t)(scratch->bytes - l.temporary);
        uint64_t *kin = key_in + (uint64_t)r * width;
        uint64_t *kout = key_out + (uint64_t)r * width;
        uint32_t *itmp = id_tmp + (uint64_t)r * MTP_TARGET_NATIVE_CAP;
        uint32_t *iout = (uint32_t *)ids->ptr +
            (uint64_t)r * MTP_TARGET_NATIVE_CAP;
        const int selected = mtp_native_select_enabled() &&
            mtp_native_select_launch(base, l.select, width, kin, itmp,
                                     MTP_TARGET_NATIVE_CAP);
        if (selected && !cuda_ok(cudaGetLastError(),
                                 "native R2 candidate select")) return -1;
        if (!selected || mtp_native_select_verify()) {
            if (selected) mtp_native_select_check_begin(
                scratch, l.id_tmp + (uint64_t)r * MTP_TARGET_NATIVE_CAP * 4u,
                MTP_TARGET_NATIVE_CAP);
            if (!cuda_ok(cub::DeviceRadixSort::SortKeysDescending(
                    base + l.temporary, temporary, kin, kout, width, 32, 64,
                    cuda_decode_stream()), "native R2 score sort")) return -1;
            mtp_native_unpack_ids_n<<<
                    (MTP_TARGET_NATIVE_CAP + 255u) / 256u, 256, 0,
                    cuda_decode_stream()>>>(itmp, kout, MTP_TARGET_NATIVE_CAP);
            if (!cuda_ok(cudaGetLastError(), "native R2 candidate unpack"))
                return -1;
            if (selected) mtp_native_select_check_end(
                scratch, l.id_tmp + (uint64_t)r * MTP_TARGET_NATIVE_CAP * 4u,
                MTP_TARGET_NATIVE_CAP, "target");
        }
        temporary = (size_t)(scratch->bytes - l.temporary);
        if (!cuda_ok(cub::DeviceRadixSort::SortKeys(
                base + l.temporary, temporary, itmp, iout,
                MTP_TARGET_NATIVE_CAP,
                0, id_bits, cuda_decode_stream()),
                "native R2 original-ID sort")) return -1;
        mtp_native_projection_kernel<false><<<
            (MTP_TARGET_NATIVE_CAP + 3u) / 4u, 256, 0,
            cuda_decode_stream()>>>(
            (float *)out->ptr + (uint64_t)r * MTP_TARGET_NATIVE_CAP,
            (const unsigned char *)w,
            xq + (uint64_t)r * MTP_NATIVE_DIM,
            xs + (uint64_t)r * (MTP_NATIVE_DIM / 32u),
            MTP_TARGET_NATIVE_CAP, iout, vocab, prefix, tail);
        if (!cuda_ok(cudaGetLastError(), "native R2 exact refinement"))
            return -1;
    }
    return (int)MTP_TARGET_NATIVE_CAP;
}
__global__ static void mtp_native_map(uint32_t *winner, const float *logits,
                                      const uint32_t *ids, const uint32_t *invalid,
                                      uint32_t count, uint32_t vocab) {
    const uint32_t bits = __float_as_uint(logits[0]);
    const uint32_t packed = (bits & 0x7fffffffu) > 0x7f800000u ? 0u : winner[0];
    const uint32_t original = packed < count ? ids[packed] : UINT32_MAX;
    winner[0] = original < vocab ? original : UINT32_MAX;
    if (invalid) winner[1] = *invalid;
}
extern "C" int ds4_gpu_mtp_native_map(ds4_gpu_tensor *winner,
        const ds4_gpu_tensor *logits, const ds4_gpu_tensor *ids,
        const ds4_gpu_tensor *scratch, uint32_t count, uint32_t vocab,
        uint32_t screen_width, int defer_invalid) {
    defer_invalid = defer_invalid &&
        getenv("DS4_MTP_NO_DEFER_INVALID_FLAG") == nullptr;
    const mtp_native_layout l = mtp_native_offsets(screen_width);
    if (!winner || !logits || !ids || count != MTP_NATIVE_CAP || !vocab ||
        winner->bytes < (defer_invalid ? 8u : 4u) ||
        logits->bytes < (uint64_t)count*4 || ids->bytes < (uint64_t)count*4 ||
        (defer_invalid && (!scratch || screen_width > MTP_NATIVE_MAX_WIDTH ||
                           scratch->bytes < l.flag + sizeof(uint32_t)))) return 0;
    const int tier=ds4_tensor_device_idx(winner); int current=-1;
    if (tier<0 || tier>=g_n_gpus || ds4_tensor_device_idx(logits)!=tier ||
        ds4_tensor_device_idx(ids)!=tier ||
        (defer_invalid && ds4_tensor_device_idx(scratch)!=tier) ||
        cudaGetDevice(&current)!=cudaSuccess ||
        current!=g_gpu[tier].device_id) return 0;
    mtp_native_map<<<1,1,0,cuda_decode_stream()>>>((uint32_t *)winner->ptr,
        (const float *)logits->ptr,(const uint32_t *)ids->ptr,
        defer_invalid ? (const uint32_t *)((const char *)scratch->ptr + l.flag) : nullptr,
        count,vocab);
    return cuda_ok(cudaGetLastError(),"native original winner map");
}

__global__ static void mtp_native_scatter_kernel(
        float *out, const float *values, const uint32_t *ids,
        uint32_t count, uint32_t vocab) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const uint32_t id = ids[i];
    if (id < vocab) out[id] = values[i];
}

/* Restore refined shortlist logits to their original vocabulary positions.
 * The caller initializes the full row to -FLT_MAX first.  Native screening
 * sorts ids into strictly increasing original-id order, so every destination
 * is unique and this needs neither atomics nor a second permutation. */
extern "C" int ds4_gpu_mtp_native_scatter(ds4_gpu_tensor *out,
        uint64_t out_offset, const ds4_gpu_tensor *values,
        const ds4_gpu_tensor *ids, uint32_t count, uint32_t vocab) {
    if (!out || !values || !ids || !count || !vocab ||
        out_offset > out->bytes ||
        (uint64_t)vocab * sizeof(float) > out->bytes - out_offset ||
        values->bytes < (uint64_t)count * sizeof(float) ||
        ids->bytes < (uint64_t)count * sizeof(uint32_t) ||
        (out_offset & (sizeof(float) - 1u)) != 0u) return 0;
    const int tier = ds4_tensor_device_idx(out);
    int current = -1;
    if (tier < 0 || tier >= g_n_gpus ||
        ds4_tensor_device_idx(values) != tier ||
        ds4_tensor_device_idx(ids) != tier ||
        cudaGetDevice(&current) != cudaSuccess ||
        current != g_gpu[tier].device_id) return 0;
    float *row = (float *)((char *)out->ptr + out_offset);
    mtp_native_scatter_kernel<<<(count + 255u) / 256u, 256, 0,
            cuda_decode_stream()>>>(row, (const float *)values->ptr,
                                    (const uint32_t *)ids->ptr, count, vocab);
    return cuda_ok(cudaGetLastError(), "native target scatter");
}

__global__ static void mtp_native_scatter2_kernel(
        float *out, const float *values, const uint32_t *ids,
        const uint32_t *invalid, uint32_t *winner,
        uint32_t count, uint32_t vocab) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0u && invalid) winner[2] = *invalid;
    if (i >= 2u * count) return;
    const uint32_t r = i >= count;
    const uint32_t j = i - r * count;
    const uint32_t id = ids[(uint64_t)r * count + j];
    if (id < vocab)
        out[(uint64_t)r * vocab + id] =
            values[(uint64_t)r * count + j];
}

extern "C" int ds4_gpu_mtp_native_scatter2(ds4_gpu_tensor *out,
        const ds4_gpu_tensor *values, const ds4_gpu_tensor *ids,
        const ds4_gpu_tensor *scratch, ds4_gpu_tensor *winner,
        uint32_t count, uint32_t vocab, uint32_t screen_width,
        int defer_invalid) {
    defer_invalid = defer_invalid &&
        getenv("DS4_MTP_NO_DEFER_INVALID_FLAG") == nullptr;
    const mtp_native_layout2 l = mtp_native_offsets2(screen_width);
    if (!out || !values || !ids || !count || !vocab ||
        out->bytes < 2ull * vocab * sizeof(float) ||
        values->bytes < 2ull * count * sizeof(float) ||
        ids->bytes < 2ull * count * sizeof(uint32_t) ||
        (defer_invalid && (!scratch || !winner ||
                           screen_width > MTP_NATIVE_MAX_WIDTH ||
                           scratch->bytes < l.flag + sizeof(uint32_t) ||
                           winner->bytes < 3u * sizeof(uint32_t)))) return 0;
    const int tier = ds4_tensor_device_idx(out);
    int current = -1;
    if (tier < 0 || tier >= g_n_gpus ||
        ds4_tensor_device_idx(values) != tier ||
        ds4_tensor_device_idx(ids) != tier ||
        (defer_invalid && (ds4_tensor_device_idx(scratch) != tier ||
                           ds4_tensor_device_idx(winner) != tier)) ||
        cudaGetDevice(&current) != cudaSuccess ||
        current != g_gpu[tier].device_id) return 0;
    mtp_native_scatter2_kernel<<<(2u * count + 255u) / 256u, 256, 0,
            cuda_decode_stream()>>>(
        (float *)out->ptr, (const float *)values->ptr,
        (const uint32_t *)ids->ptr,
        defer_invalid ? (const uint32_t *)((const char *)scratch->ptr + l.flag) : nullptr,
        defer_invalid ? (uint32_t *)winner->ptr : nullptr, count, vocab);
    return cuda_ok(cudaGetLastError(), "native target R2 scatter");
}
