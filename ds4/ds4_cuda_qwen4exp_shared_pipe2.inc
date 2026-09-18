/* ds4_cuda_qwen4exp_shared_pipe2.inc -- SWEEP 2026-09-18 (F3): the shared expert's
 * prefill GEMMs (qwen4exp_shared_pipe_mma_kernel above) on a different STAGING
 * schedule, with that kernel's per-element arithmetic character for character.
 *
 * THE CONTRACT (unchanged).  For one output element (token t, weight row n):
 *   P(c) = the ascending chain fma.rn(mul.rn(ws[n][g], xs[t][g]), (float)dot(t,n,g), P)
 *          over the groups g = c, c+32, c+64, ... of class c, from +0.0;
 *   the 32 class values folded by warp_sum_f32's tree, walked as the
 *   streaming pairwise sum over the flat class index i (class rev5(i)), the
 *   older subtree on the left -- the shipped W/S pending-partial logic below is
 *   that walk, and it is invariant to how the 32 indices are cut into stages;
 *   the epilogue expressions of the shipped kernel.
 * The integer dot comes from the same seeded m16n8k32 MMA and the same
 * magic subtraction.  Nothing here changes an operand, an order or a rounding
 * point; what changes is only how the bytes reach shared memory:
 *
 *   gate+up (K = 2560, 80 groups): the shipped pipe is one block per SM (168
 *   registers, 59 KB) with two stages, and its producers issue a stage's loads
 *   into registers and wait for them before storing -- one L2 round trip per
 *   stage, exposed.  Its per-stage L2 SECTOR traffic is ~49 KB for 12 KB of
 *   activations: the 64 x 6 activation scales are 4-byte loads at a 320-byte
 *   stride (a full 32-byte sector each, 12 KB of sectors for 1.5 KB of data)
 *   and every 34-byte weight group is a 48-byte window over two sectors.
 *   Here: the activation SCALES are staged once per block (64 tokens x groups
 *   floats, contiguous rows, 16-byte copies), every other byte moves by
 *   cp.async into a ring of up to four stages -- activation chunks straight
 *   into their slots, weight groups as their raw 48-byte windows into 48-byte
 *   slots, funnel-shifted onto word boundaries IN PLACE by the thread that
 *   copied them once they land (the shipped producer's realignment, on the
 *   same words) -- so three stages of copies are in flight while the
 *   consumers run, and no producer register round trip sits in the loop.
 *
 *   down (K = 640, 20 groups): the shipped pipe runs 4 stages per block over
 *   1280 blocks and never reaches steady state.  Here the block holds ALL of
 *   K: activations and scales by cp.async (the token's whole 640 + 80 bytes,
 *   contiguous), the 32 weight rows as register windows realigned exactly as
 *   the shipped producer does, one fill phase by every thread of the block,
 *   one __syncthreads, then the consumers walk the 32 classes (KMAX 1, one
 *   stage, the fold entirely in W).  At 32 tokens per block it is 47 KB and
 *   two blocks per SM, so one block's fill overlaps the other's MMAs.
 *
 * Off unless DS4_SHARED_PIPE2 is set: 1 = both, 2 = gate+up only, 3 = down
 * only.  DS4_SHARED_PIPE2_GU=S:L picks the gate+up ring (stages S in 2..4,
 * classes-per-stage log L; default 4:0 = four stages of one class, three
 * slots), DS4_SHARED_PIPE2_DN_BM=32|64 the down tile height (default 32).
 * Everything the shipped picker refuses (non-Q8_0, unaligned, under 64
 * tokens, > 4 groups per class) this refuses too and the shipped kernel runs;
 * the decode and every speculative verify are untouched. */
#ifndef DS4_QSP2_INC
#define DS4_QSP2_INC

__device__ __forceinline__ static void qsp2_cp16_cg(void *smem, const void *g) {
    const uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(g) : "memory");
}
__device__ __forceinline__ static void qsp2_cp16_ca(void *smem, const void *g) {
    const uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(g) : "memory");
}
__device__ __forceinline__ static void qsp2_cp4_ca(void *smem, const void *g) {
    const uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(s), "l"(g) : "memory");
}
__device__ __forceinline__ static void qsp2_commit(void) {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}
template <int N>
__device__ __forceinline__ static void qsp2_wait(void) {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}
__device__ __forceinline__ static void qsp2_zero16(void *smem) {
    qsp_sts_16(smem, make_uint4(0u, 0u, 0u, 0u));
}
__device__ __forceinline__ static void qsp2_sts4(void *smem, uint32_t v) {
    const uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("st.shared.u32 [%0], %1;\n" :: "r"(s), "r"(v) : "memory");
}

/* The shipped producer's realignment of one Q8_0 group: the 12 words of the
 * 16-byte-aligned 48-byte window around the group, the group's start at
 * `soff` bytes into it (soff = block address mod 16), the quants two bytes
 * past that.  Same selects, same funnel shifts, same half. */
struct qsp2_aligned { uint4 lo, hi; uint16_t h; };
__device__ __forceinline__ static qsp2_aligned qsp2_realign(const uint4 r0, const uint4 r1,
                                                            const uint4 r2, uint32_t soff) {
    const uint32_t raw[12] = { r0.x, r0.y, r0.z, r0.w, r1.x, r1.y, r1.z, r1.w,
                               r2.x, r2.y, r2.z, r2.w };
    const uint32_t qoff = soff + 2u;
    const uint32_t wq = qoff >> 2;
    const uint32_t sh = (qoff & 2u) ? 16u : 0u;
    uint32_t q[8];
#pragma unroll
    for (int j = 0; j < 8; j++) {
        uint32_t lo = 0u, hi = 0u;
#pragma unroll
        for (int w = 0; w <= 4; w++) {
            if (wq == (uint32_t)w) { lo = raw[w + j]; hi = (w + j + 1 < 12) ? raw[w + j + 1] : 0u; }
        }
        q[j] = __funnelshift_r(lo, hi, sh);
    }
    uint32_t sw = 0u;
#pragma unroll
    for (int w = 0; w < 4; w++) if ((soff >> 2) == (uint32_t)w) sw = raw[w];
    qsp2_aligned a;
    a.lo = make_uint4(q[0], q[1], q[2], q[3]);
    a.hi = make_uint4(q[4], q[5], q[6], q[7]);
    a.h = (soff & 2u) ? (uint16_t)(sw >> 16) : (uint16_t)(sw & 0xffffu);
    return a;
}

template <int MATRICES, int CWM, int STAGES, int KMAX, int LOGCH, int WHOLE>
struct qsp2_cfg {
    static constexpr int NT = 2;
    static constexpr int BM = 16 * CWM;                 /* tokens per block */
    static constexpr int CWARPS = 2 * CWM;              /* (wm, wn) consumer warps */
    static constexpr int PWARPS = CWM;                  /* producer warps */
    static constexpr int THREADS = (CWARPS + PWARPS) * 32;
    static constexpr int BN = 2 * NT * 8;               /* 32 weight rows per matrix */
    static constexpr int CH = 1 << LOGCH;
    static constexpr int NLEV = 5 - LOGCH;
    static constexpr int NSTAGE = 1 << NLEV;
    static constexpr int SLOTS = CH * KMAX;             /* ring: slots per stage */
    static constexpr int MINB = THREADS <= 192 ? 2 : 1;
    static_assert(!WHOLE || (KMAX == 1 && LOGCH == 5 && STAGES == 1), "whole-K: one stage of every class");
    static_assert(STAGES >= 1 && STAGES <= 4, "ring depth");
    static_assert(LOGCH >= 0 && LOGCH <= 5, "classes per stage");
};

/* Shared memory of one launch: the scale panel, then STAGES stage buffers of
 * [A: BM x LDA][B: MATRICES x BN x LDB][Ws: MATRICES x S x BN floats].
 * LDXS is groups rounded so that LDXS/4 is odd (eight tokens of a fragment
 * on eight distinct bank quads); LDA and LDB keep the shipped ldmatrix rule
 * (word stride 4 mod 8). */
__host__ __device__ __forceinline__ static int qsp2_ldxs(uint32_t groups) {
    return (int)groups + (int)((4u - (groups & 7u) + 8u) & 7u);
}
__host__ __device__ __forceinline__ static int qsp2_ldb(int slots, int whole) {
    if (whole) return slots * 32 + 16;
    return slots * 48 + ((((slots * 12) & 7) == 4) ? 0 : 16);
}
__host__ __device__ __forceinline__ static int qsp2_smem_bytes(int matrices, int bm, int stages,
                                                                int slots, int whole, uint32_t groups) {
    const int lda = slots * 32 + 16;
    const int ldb = qsp2_ldb(slots, whole);
    const int stage = bm * lda + matrices * 32 * ldb + matrices * slots * 32 * 4;
    return bm * qsp2_ldxs(groups) * 4 + stages * stage;
}

template <int MATRICES, int CWM, int STAGES, int KMAX, int LOGCH, int WHOLE>
__global__ __launch_bounds__(qsp2_cfg<MATRICES, CWM, STAGES, KMAX, LOGCH, WHOLE>::THREADS,
                             qsp2_cfg<MATRICES, CWM, STAGES, KMAX, LOGCH, WHOLE>::MINB) static void
qwen4exp_shared_pipe2_mma_kernel(
        float *out,
        const char *w0,
        const char *w1,
        const int8_t *xq,
        const float *xs,
        const float *gate_scale,
        uint64_t w0_row_bytes,
        uint64_t w1_row_bytes,
        uint32_t groups,
        uint32_t n_dim,
        uint32_t n_tokens,
        uint32_t xs_vec16) {
    typedef qsp2_cfg<MATRICES, CWM, STAGES, KMAX, LOGCH, WHOLE> C;
    constexpr int BM = C::BM, CWARPS = C::CWARPS, PWARPS = C::PWARPS, THREADS = C::THREADS;
    constexpr int NT = C::NT, BN = C::BN, CH = C::CH, NLEV = C::NLEV, NSTAGE = C::NSTAGE;
    constexpr int BS = WHOLE ? 32 : 48;                  /* bytes per B slot */
    const int S = WHOLE ? (int)groups : C::SLOTS;        /* slots per stage */
    const int LDA = S * 32 + 16;
    const int LDB = qsp2_ldb(S, WHOLE);
    const int LDXS = qsp2_ldxs(groups);
    const int A_BYTES = BM * LDA;
    const int B_BYTES = MATRICES * BN * LDB;
    const int STAGE_BYTES = A_BYTES + B_BYTES + MATRICES * S * BN * 4;
    extern __shared__ __align__(16) unsigned char qsp2_smem[];
    float *sXs = (float *)qsp2_smem;
    unsigned char *stage0 = qsp2_smem + (size_t)BM * (size_t)LDXS * 4u;

    const int tid = (int)threadIdx.x;
    const uint32_t lane = threadIdx.x & 31u;
    const int warp = tid >> 5;
    const uint32_t tok0 = blockIdx.x * (uint32_t)BM;
    const uint32_t row0 = blockIdx.y * (uint32_t)BN;
    if (tok0 >= n_tokens || row0 >= n_dim) return;
    (void)PWARPS;

    /* ---- The producer thread set: the producer warps (ring), every thread
     * (whole-K).  Items of a stage: activation chunks (token, slot, half),
     * weight groups (matrix, row, slot); item i belongs to producer lane
     * i % PT, and the thread that copies a group is the thread that realigns
     * it, so its own cp.async wait is all the ordering the realignment needs. */
    const bool producer = WHOLE ? true : (warp >= CWARPS);
    const int pl = WHOLE ? tid : (tid - CWARPS * 32);
    constexpr int PT = WHOLE ? THREADS : PWARPS * 32;

    if (producer) {
        /* The scale panel, once: token rows of `groups` floats. */
        if (xs_vec16) {
            const int cpt = (int)groups >> 2;                   /* 16-byte chunks per token */
            const int NX = BM * cpt;
            for (int i = pl; i < NX; i += PT) {
                const int t = i / cpt, c = i - t * cpt;
                const uint32_t tok = tok0 + (uint32_t)t;
                unsigned char *dst = (unsigned char *)(sXs + t * LDXS + c * 4);
                if (tok < n_tokens) qsp2_cp16_ca(dst, xs + (uint64_t)tok * groups + (uint64_t)c * 4u);
                else qsp2_zero16(dst);
            }
        } else {
            const int NX = BM * (int)groups;
            for (int i = pl; i < NX; i += PT) {
                const int t = i / (int)groups, g = i - t * (int)groups;
                const uint32_t tok = tok0 + (uint32_t)t;
                unsigned char *dst = (unsigned char *)(sXs + t * LDXS + g);
                if (tok < n_tokens) qsp2_cp4_ca(dst, xs + (uint64_t)tok * groups + (uint64_t)g);
                else qsp2_sts4(dst, 0u);
            }
        }

        /* One stage's copies: the activation chunks, and (ring) the raw
         * weight windows. */
        auto issue = [&](int s, int buf) {
            unsigned char *sA = stage0 + buf * STAGE_BYTES;
            unsigned char *sB = sA + A_BYTES;
            const int NA = BM * S * 2;
            for (int i = pl; i < NA; i += PT) {
                const int t = i / (2 * S);
                const int rem = i - t * (2 * S);
                const int slot = rem >> 1, half = rem & 1;
                const uint32_t g = WHOLE ? (uint32_t)slot
                    : qs_rev5((uint32_t)(s * CH + slot / KMAX)) + 32u * (uint32_t)(slot % KMAX);
                const uint32_t tok = tok0 + (uint32_t)t;
                unsigned char *dst = sA + t * LDA + slot * 32 + half * 16;
                if (tok < n_tokens && g < groups)
                    qsp2_cp16_cg(dst, xq + ((uint64_t)tok * groups + g) * 32u + (uint64_t)half * 16u);
                else
                    qsp2_zero16(dst);
            }
            if (!WHOLE) {
                const int NB = MATRICES * BN * S;
                for (int i = pl; i < NB; i += PT) {
                    const int m = i / (BN * S);
                    const int rem = i - m * (BN * S);
                    const int r = rem / S;
                    const int slot = rem - r * S;
                    const uint32_t g = qs_rev5((uint32_t)(s * CH + slot / KMAX)) + 32u * (uint32_t)(slot % KMAX);
                    const uint32_t row = row0 + (uint32_t)r;
                    unsigned char *dst = sB + (m * BN + r) * LDB + slot * BS;
                    if (row < n_dim && g < groups) {
                        const char *wbase = (m == 0 ? w0 : w1);
                        const uint64_t rbytes = (m == 0 ? w0_row_bytes : w1_row_bytes);
                        const char *p = wbase + (uint64_t)row * rbytes + (uint64_t)g * 34u;
                        const char *win = (const char *)((uintptr_t)p & ~(uintptr_t)15u);
                        const char *mend = wbase + (uint64_t)n_dim * rbytes;
                        qsp2_cp16_ca(dst, win);
                        qsp2_cp16_ca(dst + 16, win + 16);
                        if (win + 48 <= mend) {
                            qsp2_cp16_ca(dst + 32, win + 32);
                        } else {
                            const int inside = (int)(mend - (win + 32));
#pragma unroll
                            for (int j = 0; j < 4; j++) {
                                if (j * 4 < inside) qsp2_cp4_ca(dst + 32 + j * 4, win + 32 + j * 4);
                                else qsp2_sts4(dst + 32 + j * 4, 0u);
                            }
                        }
                    } else {
                        qsp2_zero16(dst); qsp2_zero16(dst + 16); qsp2_zero16(dst + 32);
                    }
                }
            }
        };
        /* Ring: realign the groups this thread copied, in place, after they
         * landed. */
        auto realign_stage = [&](int s, int buf) {
            unsigned char *sB = stage0 + buf * STAGE_BYTES + A_BYTES;
            float *sWs = (float *)(sB + B_BYTES);
            const int NB = MATRICES * BN * S;
            for (int i = pl; i < NB; i += PT) {
                const int m = i / (BN * S);
                const int rem = i - m * (BN * S);
                const int r = rem / S;
                const int slot = rem - r * S;
                const uint32_t g = qs_rev5((uint32_t)(s * CH + slot / KMAX)) + 32u * (uint32_t)(slot % KMAX);
                const uint32_t row = row0 + (uint32_t)r;
                unsigned char *slotp = sB + (m * BN + r) * LDB + slot * BS;
                const bool valid = row < n_dim && g < groups;
                const uint4 r0 = *(const uint4 *)(slotp);
                const uint4 r1 = *(const uint4 *)(slotp + 16);
                const uint4 r2 = *(const uint4 *)(slotp + 32);
                const char *pblk = (m == 0 ? w0 : w1) +
                    (uint64_t)row * (m == 0 ? w0_row_bytes : w1_row_bytes) + (uint64_t)g * 34u;
                qsp2_aligned a = qsp2_realign(r0, r1, r2, (uint32_t)((uintptr_t)pblk & 15u));
                if (!valid) { a.lo = make_uint4(0u, 0u, 0u, 0u); a.hi = a.lo; }
                qsp_sts_16(slotp, a.lo);
                qsp_sts_16(slotp + 16, a.hi);
                sWs[(m * S + slot) * BN + r] = valid ? __half2float(__ushort_as_half(a.h)) : 0.0f;
            }
        };

        if (WHOLE) {
            /* Whole K, one stage: activations by cp.async (above), the weight
             * rows as register windows realigned the shipped way, four groups
             * a thread in flight. */
            issue(0, 0);
            qsp2_commit();
            unsigned char *sB = stage0 + A_BYTES;
            float *sWs = (float *)(sB + B_BYTES);
            const int NB = MATRICES * BN * S;
            constexpr int KBW = 4;
            for (int i0 = pl; i0 < NB; i0 += PT * KBW) {
                uint4 rb[KBW][3];
#pragma unroll
                for (int k = 0; k < KBW; k++) {
                    const int i = i0 + k * PT;
                    rb[k][0] = make_uint4(0u, 0u, 0u, 0u); rb[k][1] = rb[k][0]; rb[k][2] = rb[k][0];
                    if (i < NB) {
                        const int m = i / (BN * S);
                        const int rem = i - m * (BN * S);
                        const int r = rem / S;
                        const int slot = rem - r * S;
                        const uint32_t g = (uint32_t)slot;
                        const uint32_t row = row0 + (uint32_t)r;
                        if (row < n_dim && g < groups) {
                            const char *wbase = (m == 0 ? w0 : w1);
                            const uint64_t rbytes = (m == 0 ? w0_row_bytes : w1_row_bytes);
                            const char *p = wbase + (uint64_t)row * rbytes + (uint64_t)g * 34u;
                            const char *win = (const char *)((uintptr_t)p & ~(uintptr_t)15u);
                            const char *mend = wbase + (uint64_t)n_dim * rbytes;
                            rb[k][0] = qsp_ldg_16(win);
                            rb[k][1] = qsp_ldg_16(win + 16);
                            if (win + 48 <= mend) {
                                rb[k][2] = qsp_ldg_16(win + 32);
                            } else {
                                const int inside = (int)(mend - (win + 32));
                                uint32_t qv[4] = {0u, 0u, 0u, 0u};
#pragma unroll
                                for (int j = 0; j < 4; j++) {
                                    if (j * 4 < inside) {
                                        uint32_t v;
                                        asm volatile("ld.global.u32 %0, [%1];" : "=r"(v) : "l"(win + 32 + j * 4));
                                        qv[j] = v;
                                    }
                                }
                                rb[k][2] = make_uint4(qv[0], qv[1], qv[2], qv[3]);
                            }
                        }
                    }
                }
#pragma unroll
                for (int k = 0; k < KBW; k++) {
                    const int i = i0 + k * PT;
                    if (i < NB) {
                        const int m = i / (BN * S);
                        const int rem = i - m * (BN * S);
                        const int r = rem / S;
                        const int slot = rem - r * S;
                        const uint32_t g = (uint32_t)slot;
                        const uint32_t row = row0 + (uint32_t)r;
                        const bool valid = row < n_dim && g < groups;
                        const char *pblk = (m == 0 ? w0 : w1) +
                            (uint64_t)row * (m == 0 ? w0_row_bytes : w1_row_bytes) + (uint64_t)g * 34u;
                        const qsp2_aligned a = qsp2_realign(rb[k][0], rb[k][1], rb[k][2],
                                                            (uint32_t)((uintptr_t)pblk & 15u));
                        unsigned char *dst = sB + (m * BN + r) * LDB + slot * BS;
                        qsp_sts_16(dst, a.lo);
                        qsp_sts_16(dst + 16, a.hi);
                        sWs[(m * S + slot) * BN + r] = valid ? __half2float(__ushort_as_half(a.h)) : 0.0f;
                    }
                }
            }
            qsp2_wait<0>();
            __syncthreads();
            if (warp >= CWARPS) return;
        } else {
            /* The ring.  Stages 0 .. STAGES-2 go out before the loop; at
             * iteration s the stage s+STAGES-1 is issued into the buffer the
             * consumers released at stage s-1, then stage s is waited for,
             * realigned and published.  Barriers as the shipped pipe: 1+2b
             * "stage in buffer b ready" (producers arrive, consumers sync),
             * 2+2b "buffer b released" (consumers arrive, producers sync). */
            constexpr int PRO = STAGES - 1 < NSTAGE ? STAGES - 1 : NSTAGE;
            for (int p = 0; p < PRO; p++) { issue(p, p % STAGES); qsp2_commit(); }
            for (int s = 0; s < NSTAGE; s++) {
                const int nxt = s + STAGES - 1;
                if (nxt < NSTAGE) {
                    if (s >= 1) qsp_bar_sync(2 + 2 * ((s - 1) % STAGES), THREADS);
                    issue(nxt, nxt % STAGES);
                    qsp2_commit();
                }
                const int allow = (STAGES - 1) < (NSTAGE - 1 - s) ? (STAGES - 1) : (NSTAGE - 1 - s);
                if (allow >= 3) qsp2_wait<(STAGES > 3 ? 3 : 0)>();
                else if (allow == 2) qsp2_wait<(STAGES > 2 ? 2 : 0)>();
                else if (allow == 1) qsp2_wait<(STAGES > 1 ? 1 : 0)>();
                else qsp2_wait<0>();
                realign_stage(s, s % STAGES);
                __syncwarp();
                qsp_bar_arrive(1 + 2 * (s % STAGES), THREADS);
            }
            return;
        }
    }

    /* ---- Consumers: warp (wm, wn) owns tokens wm*16.. and rows wn*16.. --
     * the shipped consumer, reading the scale panel and the twin's strides. */
    const int wm = warp >> 1;
    const int wn = warp & 1;
    const uint32_t g4 = lane >> 2u;
    const uint32_t t4 = lane & 3u;
    const int a_lrow = (int)(lane & 15u);
    const int a_lk = (int)(lane >> 4u) * 16;
    const int b_lrow = (int)(lane & 7u) + (int)((lane >> 4u) & 1u) * 8;
    const int b_lk = (int)((lane >> 3u) & 1u) * 16;
    const int32_t magic = QSP_MAGIC_BITS;

    float P[MATRICES][NT][4];
    float W[MATRICES][LOGCH > 0 ? LOGCH : 1][NT][4];
    float Sp[MATRICES][NLEV > 0 ? NLEV : 1][NT][4];
#pragma unroll
    for (int m = 0; m < MATRICES; m++)
#pragma unroll
        for (int ni = 0; ni < NT; ni++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                P[m][ni][e] = 0.0f;
#pragma unroll
                for (int b = 0; b < (LOGCH > 0 ? LOGCH : 1); b++) W[m][b][ni][e] = 0.0f;
#pragma unroll
                for (int b = 0; b < (NLEV > 0 ? NLEV : 1); b++) Sp[m][b][ni][e] = 0.0f;
            }

#pragma unroll 1
    for (int s = 0; s < NSTAGE; s++) {
        const int buf = s % STAGES;
        if (!WHOLE) qsp_bar_sync(1 + 2 * buf, THREADS);
        const unsigned char *sA = stage0 + buf * STAGE_BYTES;
        const unsigned char *sB = sA + A_BYTES;
        const float *sWs = (const float *)(sB + B_BYTES);

#pragma unroll
        for (int cls = 0; cls < CH; cls++) {
#pragma unroll
            for (int m = 0; m < MATRICES; m++)
#pragma unroll
                for (int ni = 0; ni < NT; ni++)
#pragma unroll
                    for (int e = 0; e < 4; e++) P[m][ni][e] = 0.0f;
            const uint32_t cval = qs_rev5((uint32_t)(s * CH + cls));
            if (cval < groups)
#pragma unroll
            for (int kk = 0; kk < KMAX; kk++) {
                const int slot = WHOLE ? (int)cval : (cls * KMAX + kk);
                const uint32_t g = cval + 32u * (uint32_t)kk;
                uint32_t af[4];
                qsp_ldmatrix_x4(af, sA + (wm * 16 + a_lrow) * LDA + slot * 32 + a_lk);
                const float xs0 = g < groups ? sXs[(wm * 16 + (int)g4) * LDXS + (int)g] : 0.0f;
                const float xs1 = g < groups ? sXs[(wm * 16 + 8 + (int)g4) * LDXS + (int)g] : 0.0f;
#pragma unroll
                for (int m = 0; m < MATRICES; m++) {
                    uint32_t bq[NT / 2][4];
#pragma unroll
                    for (int np = 0; np < NT / 2; np++) {
                        qsp_ldmatrix_x4(bq[np], sB + (m * BN + wn * NT * 8 + np * 16 + b_lrow) * LDB + slot * BS + b_lk);
                    }
#pragma unroll
                    for (int ni = 0; ni < NT; ni++) {
                        const uint32_t bf[2] = { bq[ni / 2][(ni & 1) * 2], bq[ni / 2][(ni & 1) * 2 + 1] };
                        const float2 wsp = *(const float2 *)(sWs + (m * S + slot) * BN + wn * NT * 8 + ni * 8 + (int)t4 * 2);
                        int32_t d[4];
                        qsp_mma_seeded(d, af, bf, magic);
                        P[m][ni][0] = qsp_fma(qsp_fmul(wsp.x, xs0), qsp_dot_to_f32(d[0]), P[m][ni][0]);
                        P[m][ni][1] = qsp_fma(qsp_fmul(wsp.y, xs0), qsp_dot_to_f32(d[1]), P[m][ni][1]);
                        P[m][ni][2] = qsp_fma(qsp_fmul(wsp.x, xs1), qsp_dot_to_f32(d[2]), P[m][ni][2]);
                        P[m][ni][3] = qsp_fma(qsp_fmul(wsp.y, xs1), qsp_dot_to_f32(d[3]), P[m][ni][3]);
                    }
                }
            }
#pragma unroll
            for (int b = 0; b < LOGCH; b++) {
                const int mk = (1 << (b + 1)) - 1;
                if ((cls & mk) == mk) {
#pragma unroll
                    for (int m = 0; m < MATRICES; m++)
#pragma unroll
                        for (int ni = 0; ni < NT; ni++)
#pragma unroll
                            for (int e = 0; e < 4; e++) P[m][ni][e] = qsp_fadd(W[m][b][ni][e], P[m][ni][e]);
                }
            }
#pragma unroll
            for (int b = 0; b < LOGCH; b++) {
                const int lm = (1 << b) - 1;
                if ((cls & lm) == lm && ((cls >> b) & 1) == 0) {
#pragma unroll
                    for (int m = 0; m < MATRICES; m++)
#pragma unroll
                        for (int ni = 0; ni < NT; ni++)
#pragma unroll
                            for (int e = 0; e < 4; e++) W[m][b][ni][e] = P[m][ni][e];
                }
            }
        }
        if (!WHOLE) qsp_bar_arrive(2 + 2 * buf, THREADS);

#pragma unroll
        for (int b = 0; b < NLEV; b++) {
            const int mk = (1 << (b + 1)) - 1;
            if ((s & mk) == mk) {
#pragma unroll
                for (int m = 0; m < MATRICES; m++)
#pragma unroll
                    for (int ni = 0; ni < NT; ni++)
#pragma unroll
                        for (int e = 0; e < 4; e++) P[m][ni][e] = qsp_fadd(Sp[m][b][ni][e], P[m][ni][e]);
            }
        }
#pragma unroll
        for (int b = 0; b < NLEV; b++) {
            const int lm = (1 << b) - 1;
            if ((s & lm) == lm && ((s >> b) & 1) == 0) {
#pragma unroll
                for (int m = 0; m < MATRICES; m++)
#pragma unroll
                    for (int ni = 0; ni < NT; ni++)
#pragma unroll
                        for (int e = 0; e < 4; e++) Sp[m][b][ni][e] = P[m][ni][e];
            }
        }
    }

#pragma unroll
    for (int ni = 0; ni < NT; ni++) {
#pragma unroll
        for (int e = 0; e < 4; e++) {
            const uint32_t tok = tok0 + (uint32_t)(wm * 16 + (int)g4 + (e >> 1) * 8);
            const uint32_t row = row0 + (uint32_t)(wn * NT * 8 + ni * 8 + (int)t4 * 2 + (e & 1));
            if (tok < n_tokens && row < n_dim) {
                const uint64_t off = (uint64_t)tok * n_dim + row;
                if (MATRICES == 2) {
                    const float g = P[0][ni][e];
                    const float u = P[MATRICES - 1][ni][e];
                    out[off] = (g / (1.0f + expf(-g))) * u;
                } else {
                    out[off] += gate_scale[tok] * P[0][ni][e];
                }
            }
        }
    }
}

/* ---- Valve and launchers -------------------------------------------------- */
static int qsp2_mode(void) {
    static int m = -1;
    if (m < 0) { const char *e = getenv("DS4_SHARED_PIPE2"); m = e ? atoi(e) : 3;  /* default: the down twin only */ if (m < 0) m = 0; }
    return m;
}
static void qsp2_gu_cfg(int *stages, int *logch) {
    static int s = -1, l = -1;
    if (s < 0) {
        const char *e = getenv("DS4_SHARED_PIPE2_GU");
        int ss = 4, ll = 0;
        if (e && e[0]) { ss = atoi(e); const char *c = strchr(e, ':'); if (c) ll = atoi(c + 1); }
        if (ss < 2 || ss > 4) ss = 4;
        if (ll < 0 || ll > 1) ll = 0;
        s = ss; l = ll;
    }
    *stages = s; *logch = l;
}
static int qsp2_dn_bm(void) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("DS4_SHARED_PIPE2_DN_BM"); v = e ? atoi(e) : 32; if (v != 64) v = 32; }
    return v;
}

template <int MATRICES, int CWM, int STAGES, int KMAX, int LOGCH, int WHOLE>
static int qwen4exp_shared_pipe2_launch(
        float *out, const char *w0, const char *w1, const int8_t *xq,
        const float *xs, const float *gate_scale, uint64_t w0_row_bytes,
        uint64_t w1_row_bytes, uint32_t groups, uint32_t n_dim,
        uint32_t n_tokens, cudaStream_t stream) {
    typedef qsp2_cfg<MATRICES, CWM, STAGES, KMAX, LOGCH, WHOLE> C;
    static int attr_bytes = 0;   /* the largest dynamic size granted; -1 refused */
    const int smem = qsp2_smem_bytes(MATRICES, C::BM, STAGES, WHOLE ? (int)groups : C::SLOTS, WHOLE, groups);
    if (attr_bytes < 0) return 0;
    if (smem > attr_bytes) {
        if (cudaFuncSetAttribute(qwen4exp_shared_pipe2_mma_kernel<MATRICES, CWM, STAGES, KMAX, LOGCH, WHOLE>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, smem) != cudaSuccess) {
            (void)cudaGetLastError();
            attr_bytes = -1;
            return 0;
        }
        attr_bytes = smem;
    }
    const uint32_t xs_vec16 = ((((uintptr_t)xs) & 15u) == 0u && (groups & 3u) == 0u) ? 1u : 0u;
    const dim3 grid((n_tokens + (uint32_t)C::BM - 1u) / (uint32_t)C::BM,
                    (n_dim + (uint32_t)C::BN - 1u) / (uint32_t)C::BN, 1);
    qwen4exp_shared_pipe2_mma_kernel<MATRICES, CWM, STAGES, KMAX, LOGCH, WHOLE>
        <<<grid, C::THREADS, (size_t)smem, stream>>>(
            out, w0, w1, xq, xs, gate_scale, w0_row_bytes, w1_row_bytes,
            groups, n_dim, n_tokens, xs_vec16);
    return 1;
}

/* gate+up: only where the shipped picker (Q8_0, aligned, >= 64 tokens, <= 4
 * groups a class) already said yes -- the caller passes that verdict. */
static int qwen4exp_shared_pipe2_gateup(
        float *mid, const char *gate, const char *up, const int8_t *xq, const float *xs,
        uint64_t gate_row_bytes, uint64_t up_row_bytes, uint32_t groups, uint32_t n_dim,
        uint32_t n_tokens, cudaStream_t stream) {
    const int mode = qsp2_mode();
    if (mode != 1 && mode != 2) return 0;
    if (((uintptr_t)xs & 3u) != 0u) return 0;
    const int kmax = (int)((groups + 31u) / 32u);
    int stages = 4, logch = 0;
    qsp2_gu_cfg(&stages, &logch);
#define QSP2_GU(S, K, L) \
    if (stages == (S) && kmax == (K) && logch == (L)) \
        return qwen4exp_shared_pipe2_launch<2, 4, S, K, L, 0>(mid, gate, up, xq, xs, NULL, \
                gate_row_bytes, up_row_bytes, groups, n_dim, n_tokens, stream)
    QSP2_GU(4, 3, 0); QSP2_GU(3, 3, 0); QSP2_GU(2, 3, 1); QSP2_GU(2, 3, 0);
    QSP2_GU(4, 1, 0); QSP2_GU(4, 2, 0); QSP2_GU(4, 4, 0);
#undef QSP2_GU
    return 0;
}

/* down: whole K resident, so at most 32 groups (K <= 1024). */
static int qwen4exp_shared_pipe2_down(
        float *out, const char *down, const int8_t *mq, const float *ms, const float *gate_scale,
        uint64_t down_row_bytes, uint32_t groups, uint32_t n_dim, uint32_t n_tokens,
        cudaStream_t stream) {
    const int mode = qsp2_mode();
    if (mode != 1 && mode != 3) return 0;
    if (groups > 32u || ((uintptr_t)ms & 3u) != 0u) return 0;
    if (qsp2_dn_bm() == 64)
        return qwen4exp_shared_pipe2_launch<1, 4, 1, 1, 5, 1>(out, down, NULL, mq, ms, gate_scale,
                down_row_bytes, 0u, groups, n_dim, n_tokens, stream);
    return qwen4exp_shared_pipe2_launch<1, 2, 1, 1, 5, 1>(out, down, NULL, mq, ms, gate_scale,
            down_row_bytes, 0u, groups, n_dim, n_tokens, stream);
}
#endif /* DS4_QSP2_INC */
