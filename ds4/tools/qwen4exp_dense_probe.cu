/* Prices the qwen4exp dense Q8_0 projection at prefill width.
 *
 * The tower's dense tier -- the GDN in-projections, the QSA in and out
 * projections, the hyper-connection mixers and the LM head -- runs today on
 * matmul_q8_0_preq_rows_exact_tile_kernel (ds4_cuda.cu:5377), which reads each
 * weight block once for up to EIGHT activation rows.  At a prefill width of
 * 512 that is 64 passes over the whole dense weight set per chunk, and the
 * measured rate is 1.63 TFLOPS on a box whose int8 tensor cores do tens of
 * times that.  This probe asks, separately so that no answer hides another:
 *
 *   PLACEMENT.  The weights are not in device memory: on this box
 *   (cudaDevAttrIntegrated = 1, cudaDevAttrPageableMemoryAccess = 1) the host
 *   mmap pointer goes in as the device pointer, so every weight byte is a
 *   file-backed host page reached over the coherent fabric.  Kernels run
 *   against BOTH a cudaMalloc'd copy and a pre-faulted file-backed mmap of the
 *   same bytes, so the price of the placement is one subtraction.
 *
 *   TILING.  Eight rows per weight read, against a large-M tile.
 *
 *   ARITHMETIC.  dp4a against mma.sync.m16n8k32.s8.s8.s32.  k = 32 of that MMA
 *   is exactly one Q8_0 group, so the int32 dot is the SAME number either way
 *   and the comparison is purely one of throughput.
 *
 *   CEILING.  A pure MMA loop with no memory traffic, so the tuned kernel can
 *   be read as a fraction of what the hardware will do rather than against a
 *   number from a data sheet.
 *
 * It also checks what a replacement must satisfy to ship: output bit-for-bit
 * equal at width 1 with a zero-padded tile and at width 512 -- the property
 * tests/test_qwen4exp_graph asserts as exactly zero -- and agreement with a
 * double-precision host reference.
 *
 * Build (on the box):
 *   nvcc -O3 -arch=sm_121 -o qwen4exp_dense_probe tools/qwen4exp_dense_probe.cu
 * Run under the GPU lock.  It allocates well under 2 GiB.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define CHECK(call)                                                            \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #call,     \
                    cudaGetErrorString(_e));                                   \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

/* ---------------------------------------------------------------- helpers */

__device__ __forceinline__ static int32_t dot_i8_block(const int8_t *a,
                                                       const int8_t *b,
                                                       uint64_t n,
                                                       int use_dp4a) {
    int32_t s = 0;
    if (use_dp4a && n == 32) {
#pragma unroll
        for (int i = 0; i < 8; i++) {
            int va, vb;
            memcpy(&va, a + i * 4, 4);
            memcpy(&vb, b + i * 4, 4);
            s = __dp4a(va, vb, s);
        }
        return s;
    }
    for (uint64_t i = 0; i < n; i++) s += (int32_t)a[i] * (int32_t)b[i];
    return s;
}

__device__ static float warp_sum_f32(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, off);
    }
    return __shfl_sync(0xffffffffu, v, 0);
}

__device__ __forceinline__ static void mma_m16n8k32_s8(int32_t d[4],
                                                       const uint32_t a[4],
                                                       const uint32_t b[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

/* ------------------------------------------------- activation quantisation */

/* Byte for byte the arithmetic of quantize_q8_0_f32_kernel (ds4_cuda.cu). */
__global__ static void quantize_q8_0_f32_kernel(int8_t *xq, float *xscale,
                                                const float *x, uint64_t in_dim,
                                                uint64_t blocks) {
    const uint64_t b = blockIdx.x;
    const uint64_t row = blockIdx.y;
    if (b >= blocks) return;
    const uint64_t i0 = b * 32u;
    const uint64_t bn = in_dim - i0 < 32u ? in_dim - i0 : 32u;
    const float *xr = x + row * in_dim + i0;

    float a = 0.0f;
    if (threadIdx.x < bn) a = fabsf(xr[threadIdx.x]);
    __shared__ float vals[32];
    vals[threadIdx.x] = a;
    __syncthreads();
    for (uint32_t stride = 16u; stride > 0u; stride >>= 1u) {
        if (threadIdx.x < stride) {
            vals[threadIdx.x] = fmaxf(vals[threadIdx.x], vals[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float d = vals[0] / 127.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    if (threadIdx.x == 0u) xscale[row * blocks + b] = d;
    int8_t *dst = xq + (row * blocks + b) * 32u;
    if (threadIdx.x < bn) {
        int v = (int)lrintf(xr[threadIdx.x] * id);
        v = v > 127 ? 127 : (v < -128 ? -128 : v);
        dst[threadIdx.x] = (int8_t)v;
    } else {
        dst[threadIdx.x] = 0;
    }
}

/* The same quantisation with one warp per (row, group) instead of one block:
 * a 512x2560 activation is 40960 one-warp blocks on the kernel above and 5120
 * eight-warp blocks on this one.  fmaxf is associative and exact, so taking the
 * group maximum across a warp instead of through a shared-memory tree is the
 * same number. */
__global__ static void quantize_q8_0_f32_rows_warp_kernel(int8_t *xq,
                                                          float *xscale,
                                                          const float *x,
                                                          uint64_t in_dim,
                                                          uint64_t blocks,
                                                          uint32_t n_rows) {
    const uint64_t pair =
        (uint64_t)blockIdx.x * (blockDim.x >> 5u) + (threadIdx.x >> 5u);
    if (pair >= (uint64_t)n_rows * blocks) return;
    const uint64_t row = pair / blocks;
    const uint64_t b = pair - row * blocks;
    const uint32_t lane = threadIdx.x & 31u;
    const uint64_t i0 = b * 32u;
    const uint64_t bn = in_dim - i0 < 32u ? in_dim - i0 : 32u;
    const float *xr = x + row * in_dim + i0;

    float a = (uint64_t)lane < bn ? fabsf(xr[lane]) : 0.0f;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, off));
    }
    const float d = a / 127.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    if (lane == 0u) xscale[pair] = d;
    int8_t *dst = xq + pair * 32u;
    if ((uint64_t)lane < bn) {
        int v = (int)lrintf(xr[lane] * id);
        v = v > 127 ? 127 : (v < -128 ? -128 : v);
        dst[lane] = (int8_t)v;
    } else {
        dst[lane] = 0;
    }
}

/* ------------------------------------------------------ A: today's kernel */

template <int R>
__global__ static void tile_kernel(float *out, const unsigned char *w,
                                   const int8_t *xq, const float *xscale,
                                   uint64_t in_dim, uint64_t out_dim,
                                   uint32_t n_rows, uint64_t blocks,
                                   int use_dp4a) {
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint32_t row0 = (uint32_t)blockIdx.y * (uint32_t)R;
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim || row0 >= n_rows) return;
    const uint32_t take = n_rows - row0 < (uint32_t)R ? n_rows - row0 : (uint32_t)R;

    const unsigned char *wr = w + row * blocks * 34u;
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; r++) acc[r] = 0.0f;

    for (uint64_t b = lane; b < blocks; b += 32u) {
        const uint64_t i0 = b * 32u;
        const uint64_t bn = in_dim - i0 < 32u ? in_dim - i0 : 32u;
        const __half *scale_h = (const __half *)(wr + b * 34u);
        const int8_t *qs = (const int8_t *)(wr + b * 34u + 2u);
        const float ws = __half2float(*scale_h);
#pragma unroll
        for (int r = 0; r < R; r++) {
            if ((uint32_t)r < take) {
                const uint64_t at = ((uint64_t)row0 + (uint64_t)r) * blocks + b;
                const int dot = dot_i8_block(qs, xq + at * 32u, bn, use_dp4a);
                acc[r] += ws * xscale[at] * (float)dot;
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; r++) {
        const float tot = warp_sum_f32(acc[r]);
        if (lane == 0u && (uint32_t)r < take) {
            out[((uint64_t)row0 + (uint64_t)r) * out_dim + row] = tot;
        }
    }
}

/* --------------------------------------------------- B: the MMA GEMM tile */

/* WM x WN warps; each warp owns MT m16 tiles and NT n8 tiles, so the block
 * covers BM = WM*MT*16 activation rows and BN = WN*NT*8 output rows.  G Q8_0
 * groups (G*32 of the input dimension) are staged into shared memory per step.
 *
 * grid.x indexes the M tile and grid.y the output-row slab, so blocks sharing a
 * weight slab are ADJACENT in launch order and the slab is read from DRAM once.
 *
 * Per output element the arithmetic is, for g ascending:
 *     acc = fmaf(ws[n][g] * xs[m][g], (float)dot[m][n][g], acc)
 * one f32 accumulator, no split-K, no butterfly.  Nothing in that depends on
 * the tile shape or on the row count, which is why the result is bit-for-bit
 * the same at width 1 with a zero-padded tile as at width 512, and why the
 * tile may be chosen per projection shape without touching the numbers.
 *
 * Static shared memory is capped at 48 KiB per block, which bounds the tile:
 * BM*(G*32+16) + BN*(G*32+16) + 4*G*(BM+BN) must fit.
 *
 * The shared row stride is padded by 16 bytes so the 32 lanes of a fragment
 * load land on 32 distinct banks: the word index carries 36*row and
 * 36 mod 32 = 4, so consecutive fragment rows start four banks apart.
 */
/* The 32 quants of a Q8_0 block begin two bytes into its 34, so they are only
 * 2-byte aligned in the file and cannot be read as uint32 where they lie.
 * These read the aligned words that cover them and funnel-shift -- nine loads
 * and eight shifts for a whole block, against thirty-two byte loads.
 *
 * The block stride is even and a GGUF tensor starts aligned, so the shift is
 * always 0 or 16 and never a byte's worth.  The last word of a shifted block
 * would reach two bytes PAST the block, which for a tensor's final block is
 * past the tensor and, for a tensor that ends a page-sized mapping, past the
 * mapping; those bytes are taken one at a time instead.  Nothing here reads
 * outside the block it was given.
 */
__device__ __forceinline__ static void q8_0_block_quant_words(
        uint32_t out[8], const unsigned char *blk) {
    const unsigned char *q = blk + 2;
    const uint32_t off = (uint32_t)((uintptr_t)q & 3u);
    const uint32_t *base = (const uint32_t *)(q - off);
    if (off == 0u) {
#pragma unroll
        for (int i = 0; i < 8; i++) out[i] = base[i];
        return;
    }
    /* off is 2 here for every block this model has: a 34-byte stride from an
     * aligned tensor leaves the quants on one of two alignments, and a 2-byte
     * load reads exactly the bytes the funnel keeps.  The loop is the honest
     * fallback for an alignment that cannot arise. */
    uint32_t last;
    const unsigned char *tail = (const unsigned char *)(base + 8);
    if (off == 2u) {
        uint16_t t;
        memcpy(&t, tail, 2);
        last = (uint32_t)t;
    } else {
        last = 0u;
        for (uint32_t i = 0; i < off; i++) {
            last |= ((uint32_t)tail[i]) << (8u * i);
        }
    }
    uint32_t prev = base[0];
#pragma unroll
    for (int i = 0; i < 8; i++) {
        const uint32_t next = (i == 7) ? last : base[i + 1];
        out[i] = __funnelshift_r(prev, next, off * 8u);
        prev = next;
    }
}

/* One of those eight words, for a caller that wants four quants and not all
 * thirty-two.  `idx` is 0 to 7. */
__device__ __forceinline__ static uint32_t q8_0_block_quant_word(
        const unsigned char *blk, uint32_t idx) {
    const unsigned char *q = blk + 2u + idx * 4u;
    const uint32_t off = (uint32_t)((uintptr_t)q & 3u);
    const uint32_t *base = (const uint32_t *)(q - off);
    if (off == 0u) return base[0];
    uint32_t hi;
    if (idx != 7u) {
        hi = base[1];
    } else if (off == 2u) {
        uint16_t t;
        memcpy(&t, (const unsigned char *)(base + 1), 2);
        hi = (uint32_t)t;
    } else {
        hi = 0u;
        const unsigned char *tail = (const unsigned char *)(base + 1);
        for (uint32_t i = 0; i < off; i++) hi |= ((uint32_t)tail[i]) << (8u * i);
    }
    return __funnelshift_r(base[0], hi, off * 8u);
}

template <int WM, int WN, int MT, int NT, int G>
__global__ __launch_bounds__(WM * WN * 32) static void mma_kernel(
        float *out, const unsigned char *w, const int8_t *xq,
        const float *xscale, uint64_t in_dim, uint64_t out_dim, uint32_t n_rows,
        uint64_t blocks) {
    constexpr int BM = WM * MT * 16;
    constexpr int BN = WN * NT * 8;
    constexpr int KS = G * 32;
    constexpr int SPAD = KS + 16;
    constexpr int THREADS = WM * WN * 32;

    __shared__ int8_t sA[BM][SPAD];
    __shared__ int8_t sB[BN][SPAD];
    __shared__ float sAs[BM][G];
    __shared__ float sBs[BN][G];

    const int tid = (int)threadIdx.x;
    const uint32_t lane = threadIdx.x & 31u;
    const int warp = tid >> 5;
    const int wm = warp / WN;
    const int wn = warp % WN;

    const uint32_t m0 = (uint32_t)blockIdx.x * BM;
    const uint64_t n0 = (uint64_t)blockIdx.y * BN;
    if (m0 >= n_rows || n0 >= out_dim) return;

    const uint32_t g4 = lane >> 2u;  /* 0..7 */
    const uint32_t t4 = lane & 3u;   /* 0..3 */
    const int a_k = (int)t4 * 4;

    float acc[MT][NT][4];
#pragma unroll
    for (int mi = 0; mi < MT; mi++)
#pragma unroll
        for (int ni = 0; ni < NT; ni++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc[mi][ni][e] = 0.0f;

    const uint64_t nstage = (blocks + (uint64_t)G - 1u) / (uint64_t)G;
    for (uint64_t s = 0; s < nstage; s++) {
        const uint64_t g0 = s * (uint64_t)G;
        __syncthreads();

        /* Stage the activation tile.  xq groups are 32-byte aligned, so this is
         * two uint4 loads and two uint4 stores per (row, group). */
        for (int p = tid; p < BM * G; p += THREADS) {
            const int r = p / G;
            const int gg = p - r * G;
            const uint64_t row = (uint64_t)m0 + (uint32_t)r;
            const uint64_t b = g0 + (uint64_t)gg;
            uint4 *dst = (uint4 *)&sA[r][gg * 32];
            if (row < (uint64_t)n_rows && b < blocks) {
                const uint4 *src = (const uint4 *)(xq + (row * blocks + b) * 32u);
                dst[0] = src[0];
                dst[1] = src[1];
                sAs[r][gg] = xscale[row * blocks + b];
            } else {
                /* The zero rows that make width 1 and width 512 the same
                 * kernel: a zero activation group contributes a zero int32 dot
                 * and a zero scale, so the live rows are untouched. */
                const uint4 z = make_uint4(0u, 0u, 0u, 0u);
                dst[0] = z;
                dst[1] = z;
                sAs[r][gg] = 0.0f;
            }
        }

        /* Stage the weight tile.  q8_0_block_quant_words lands the block's
         * quants 4-byte aligned in shared memory, where the fragment loads
         * want them. */
        for (int p = tid; p < BN * G; p += THREADS) {
            const int r = p / G;
            const int gg = p - r * G;
            const uint64_t row = n0 + (uint32_t)r;
            const uint64_t b = g0 + (uint64_t)gg;
            uint32_t *dst = (uint32_t *)&sB[r][gg * 32];
            if (row < out_dim && b < blocks) {
                const unsigned char *blk = w + (row * blocks + b) * 34u;
                __half h;
                memcpy(&h, blk, 2);
                sBs[r][gg] = __half2float(h);
                q8_0_block_quant_words(dst, blk);
            } else {
                sBs[r][gg] = 0.0f;
#pragma unroll
                for (int i = 0; i < 8; i++) dst[i] = 0u;
            }
        }
        __syncthreads();

#pragma unroll 1
        for (int gg = 0; gg < G; gg++) {
            uint32_t af[MT][4];
            float xs[MT][2];
#pragma unroll
            for (int mi = 0; mi < MT; mi++) {
                const int r0 = wm * MT * 16 + mi * 16 + (int)g4;
                const int r1 = r0 + 8;
                const int8_t *p0 = &sA[r0][gg * 32 + a_k];
                const int8_t *p1 = &sA[r1][gg * 32 + a_k];
                af[mi][0] = *(const uint32_t *)p0;
                af[mi][2] = *(const uint32_t *)(p0 + 16);
                af[mi][1] = *(const uint32_t *)p1;
                af[mi][3] = *(const uint32_t *)(p1 + 16);
                xs[mi][0] = sAs[r0][gg];
                xs[mi][1] = sAs[r1][gg];
            }
#pragma unroll
            for (int ni = 0; ni < NT; ni++) {
                const int c = wn * NT * 8 + ni * 8;
                const int8_t *pb = &sB[c + (int)g4][gg * 32 + a_k];
                uint32_t bf[2];
                bf[0] = *(const uint32_t *)pb;
                bf[1] = *(const uint32_t *)(pb + 16);
                const float w0 = sBs[c + (int)t4 * 2][gg];
                const float w1 = sBs[c + (int)t4 * 2 + 1][gg];
#pragma unroll
                for (int mi = 0; mi < MT; mi++) {
                    int32_t d[4] = {0, 0, 0, 0};
                    mma_m16n8k32_s8(d, af[mi], bf);
                    acc[mi][ni][0] = fmaf(w0 * xs[mi][0], (float)d[0], acc[mi][ni][0]);
                    acc[mi][ni][1] = fmaf(w1 * xs[mi][0], (float)d[1], acc[mi][ni][1]);
                    acc[mi][ni][2] = fmaf(w0 * xs[mi][1], (float)d[2], acc[mi][ni][2]);
                    acc[mi][ni][3] = fmaf(w1 * xs[mi][1], (float)d[3], acc[mi][ni][3]);
                }
            }
        }
    }

#pragma unroll
    for (int mi = 0; mi < MT; mi++) {
        const uint64_t r_lo = (uint64_t)m0 + wm * MT * 16 + mi * 16 + g4;
        const uint64_t r_hi = r_lo + 8u;
#pragma unroll
        for (int ni = 0; ni < NT; ni++) {
            const uint64_t c0 = n0 + wn * NT * 8 + ni * 8 + t4 * 2u;
            const uint64_t c1 = c0 + 1u;
            if (r_lo < (uint64_t)n_rows) {
                if (c0 < out_dim) out[r_lo * out_dim + c0] = acc[mi][ni][0];
                if (c1 < out_dim) out[r_lo * out_dim + c1] = acc[mi][ni][1];
            }
            if (r_hi < (uint64_t)n_rows) {
                if (c0 < out_dim) out[r_hi * out_dim + c0] = acc[mi][ni][2];
                if (c1 < out_dim) out[r_hi * out_dim + c1] = acc[mi][ni][3];
            }
        }
    }
}


/* The SAME arithmetic as the MMA kernel above, for the shapes the MMA cannot
 * fill the device with.
 *
 * The MMA's unit of parallelism is a 16x8 output tile, so a call with one
 * activation row and 320 output rows -- the hyper-connection down projection at
 * decode -- offers it forty tiles for forty-eight multiprocessors, and it comes
 * out four times slower than the eight-row kernel it replaced.  That is the
 * whole of the decode regression: every other projection is within a few per
 * cent either way at one row.
 *
 * This kernel's unit is one output row per WARP, so the same call offers 320
 * warps, and the K dimension is walked cooperatively inside the warp instead of
 * across it: eight lanes take one Q8_0 group, four groups at a time.  The eight
 * lanes' partial products are summed as int32, which is exact and therefore
 * order-free, and the four group dots are then folded into a single f32
 * accumulator in ASCENDING group order -- which is the contract, and which is
 * why this kernel and the MMA kernel produce the same bits and not merely the
 * same value to a tolerance.  The tests assert that rather than assume it.
 *
 * The old eight-row kernel could not be used here for exactly this reason: it
 * gives lane L the groups L, L+32, L+64 ... and closes with a warp butterfly,
 * so its sum is over a different order.
 */
__global__ static void matmul_q8_0_preq_rows_ascending_kernel(
        float *out,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t out_dim,
        uint32_t n_rows,
        uint64_t blocks) {
    const uint64_t row = (uint64_t)blockIdx.x * (blockDim.x >> 5u) +
                         (threadIdx.x >> 5u);
    const uint64_t arow = blockIdx.y;
    if (row >= out_dim || arow >= (uint64_t)n_rows) return;

    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t sub = lane >> 3u;  /* which of the four groups   */
    const uint32_t inl = lane & 7u;   /* which four bytes within it */
    const unsigned char *wr = w + row * blocks * 34u;
    const int8_t *xr = xq + arow * blocks * 32u;
    const float *xsr = xscale + arow * blocks;

    float acc = 0.0f;
    for (uint64_t b0 = 0; b0 < blocks; b0 += 4u) {
        const uint64_t b = b0 + sub;
        int32_t dot = 0;
        float s = 0.0f;
        if (b < blocks) {
            const unsigned char *blk = wr + b * 34u;
            const uint32_t wv = q8_0_block_quant_word(blk, inl);
            uint32_t av;
            memcpy(&av, xr + b * 32u + inl * 4u, 4);
            dot = __dp4a((int)wv, (int)av, 0);
            __half h;
            memcpy(&h, blk, 2);
            s = __half2float(h) * xsr[b];
        }
        /* Eight lanes, int32: exact whatever order the butterfly takes. */
#pragma unroll
        for (int o = 4; o > 0; o >>= 1) {
            dot += __shfl_xor_sync(0xffffffffu, dot, o);
        }
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int32_t dj = __shfl_sync(0xffffffffu, dot, j * 8);
            const float sj = __shfl_sync(0xffffffffu, s, j * 8);
            if (b0 + (uint64_t)j < blocks) acc = fmaf(sj, (float)dj, acc);
        }
    }
    if (lane == 0u) out[arow * out_dim + row] = acc;
}

/* ------------------------------------------------------------- C: ceiling */

/* What the int8 tensor cores will do with no memory system in the way: the
 * same mma the GEMM issues, from registers, with four independent chains so
 * the pipeline is not latency-bound.  Reported so the GEMM can be read as a
 * fraction of the hardware rather than of a data sheet. */
__global__ __launch_bounds__(256) static void mma_ceiling_kernel(int32_t *sink,
                                                                 int iters) {
    uint32_t a[4] = {0x01020304u, 0x05060708u, 0x090a0b0cu, 0x0d0e0f10u};
    uint32_t b[2] = {0x11121314u, 0x15161718u};
    int32_t d0[4] = {0, 0, 0, 0}, d1[4] = {0, 0, 0, 0};
    int32_t d2[4] = {0, 0, 0, 0}, d3[4] = {0, 0, 0, 0};
    a[0] ^= threadIdx.x;
    for (int i = 0; i < iters; i++) {
        mma_m16n8k32_s8(d0, a, b);
        mma_m16n8k32_s8(d1, a, b);
        mma_m16n8k32_s8(d2, a, b);
        mma_m16n8k32_s8(d3, a, b);
    }
    sink[blockIdx.x] = d0[0] + d1[1] + d2[2] + d3[3];
}

/* ------------------------------------------------------------------ host */

static void fill_weights(unsigned char *w, uint64_t out_dim, uint64_t blocks,
                         unsigned seed) {
    uint64_t s = 0x9e3779b97f4a7c15ull ^ seed;
    for (uint64_t r = 0; r < out_dim; r++) {
        for (uint64_t b = 0; b < blocks; b++) {
            unsigned char *p = w + (r * blocks + b) * 34u;
            s = s * 6364136223846793005ull + 1442695040888963407ull;
            const float d = 0.002f + (float)((s >> 33) & 0xffffu) * 1.0e-7f;
            __half h = __float2half(d);
            memcpy(p, &h, 2);
            for (int i = 0; i < 32; i++) {
                s = s * 6364136223846793005ull + 1442695040888963407ull;
                p[2 + i] = (unsigned char)(int8_t)((int)((s >> 33) % 255u) - 127);
            }
        }
    }
}

struct Shape {
    const char *name;
    uint64_t in_dim;
    uint64_t out_dim;
    uint32_t calls; /* per prefill chunk, over all 48 blocks */
};

/* The dense Q8_0 projections of Qwen3.8-Flash-Next 125B-A6B: 36 GDN blocks,
 * 12 QSA blocks, two hyper-connection mixers on every block. */
static const Shape kShapes[] = {
    {"gdn attn_qkv   2560->10240", 2560, 10240, 36},
    {"gdn attn_gate  2560->6144", 2560, 6144, 36},
    {"gdn ssm_out    6144->2560", 6144, 2560, 36},
    {"qsa attn_q     2560->12288", 2560, 12288, 12},
    {"qsa attn_k     2560->512", 2560, 512, 12},
    {"qsa attn_v     2560->512", 2560, 512, 12},
    {"qsa attn_out   6144->2560", 6144, 2560, 12},
    {"hc down       10240->320", 10240, 320, 96},
    {"hc up           320->10240", 320, 10240, 96},
};
static const int kNShapes = (int)(sizeof(kShapes) / sizeof(kShapes[0]));

struct Ctx {
    const unsigned char *w;
    const int8_t *xq;
    const float *xscale;
    float *out;
    uint64_t in_dim, out_dim, blocks;
    uint32_t n_rows;
    int use_dp4a;
};

typedef void (*LaunchFn)(const Ctx &);

static void launch_tile(const Ctx &c) {
    const unsigned wgrid = ((unsigned)c.out_dim + 7u) / 8u;
    dim3 grid(wgrid, (c.n_rows + 7u) / 8u, 1u);
    tile_kernel<8><<<grid, 256>>>(c.out, c.w, c.xq, c.xscale, c.in_dim,
                                  c.out_dim, c.n_rows, c.blocks, c.use_dp4a);
}

template <int WM, int WN, int MT, int NT, int G>
static void launch_mma(const Ctx &c) {
    constexpr int BM = WM * MT * 16;
    constexpr int BN = WN * NT * 8;
    dim3 grid((c.n_rows + BM - 1u) / BM,
              (unsigned)((c.out_dim + BN - 1u) / BN), 1u);
    mma_kernel<WM, WN, MT, NT, G><<<grid, WM * WN * 32>>>(
            c.out, c.w, c.xq, c.xscale, c.in_dim, c.out_dim, c.n_rows, c.blocks);
}

static void launch_ascending(const Ctx &c) {
    const unsigned warps = 8u;
    dim3 grid((unsigned)((c.out_dim + warps - 1u) / warps), c.n_rows, 1u);
    matmul_q8_0_preq_rows_ascending_kernel<<<grid, warps * 32u>>>(
            c.out, c.w, c.xq, c.xscale, c.out_dim, c.n_rows, c.blocks);
}

static double bench(LaunchFn fn, const Ctx &c, int iters) {
    cudaEvent_t a, b;
    CHECK(cudaEventCreate(&a));
    CHECK(cudaEventCreate(&b));
    fn(c);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventRecord(a));
    for (int i = 0; i < iters; i++) fn(c);
    CHECK(cudaEventRecord(b));
    CHECK(cudaEventSynchronize(b));
    float ms = 0.0f;
    CHECK(cudaEventElapsedTime(&ms, a, b));
    CHECK(cudaEventDestroy(a));
    CHECK(cudaEventDestroy(b));
    return (double)ms / iters;
}

struct Variant {
    const char *name;
    LaunchFn fn;
};

static const Variant kVariants[] = {
    {"tile R8", launch_tile},
    {"mma128x128G4", launch_mma<2, 4, 4, 4, 4>},
    {"mma128x64G4", launch_mma<2, 4, 4, 2, 4>},
    {"mma64x128G4", launch_mma<2, 4, 2, 4, 4>},
    {"mma64x64G4", launch_mma<2, 2, 2, 4, 4>},
    {"mma32x64G8", launch_mma<2, 2, 1, 4, 8>},
    {"mma64x64G8", launch_mma<2, 2, 2, 4, 8>},
    {"mma256x32G4", launch_mma<4, 2, 4, 2, 4>},
    {"warp-per-row", launch_ascending},
};
static const int kNVariants = (int)(sizeof(kVariants) / sizeof(kVariants[0]));

int main(int argc, char **argv) {
    const uint32_t width = (argc > 1) ? (uint32_t)atoi(argv[1]) : 512u;
    int dev = 0;
    CHECK(cudaSetDevice(dev));
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, dev));
    int integrated = 0, pageable = 0;
    CHECK(cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, dev));
    CHECK(cudaDeviceGetAttribute(&pageable, cudaDevAttrPageableMemoryAccess, dev));
    size_t freeb = 0, totalb = 0;
    CHECK(cudaMemGetInfo(&freeb, &totalb));
    printf("device      %s  sm_%d%d  SMs %d  L2 %.1f MiB  shared/block %d KiB\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount,
           (double)prop.l2CacheSize / 1048576.0,
           (int)(prop.sharedMemPerBlockOptin / 1024));
    printf("integrated  %d   pageableMemoryAccess %d   mem free %.1f / %.1f GiB\n",
           integrated, pageable, (double)freeb / 1073741824.0,
           (double)totalb / 1073741824.0);
    printf("width       %u\n\n", width);

    /* ---- the int8 tensor-core ceiling, no memory in the way */
    {
        int32_t *sink = NULL;
        const int nblocks = prop.multiProcessorCount * 4;
        CHECK(cudaMalloc(&sink, (size_t)nblocks * sizeof(int32_t)));
        const int iters = 20000;
        mma_ceiling_kernel<<<nblocks, 256>>>(sink, 100);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
        cudaEvent_t a, b;
        CHECK(cudaEventCreate(&a));
        CHECK(cudaEventCreate(&b));
        CHECK(cudaEventRecord(a));
        mma_ceiling_kernel<<<nblocks, 256>>>(sink, iters);
        CHECK(cudaEventRecord(b));
        CHECK(cudaEventSynchronize(b));
        float ms = 0.0f;
        CHECK(cudaEventElapsedTime(&ms, a, b));
        const double macs = (double)nblocks * 8.0 * (double)iters * 4.0 * 4096.0;
        printf("int8 mma.sync ceiling: %.1f TMAC/s (%.1f TOPS) over %.2f ms\n\n",
               macs / (ms * 1.0e-3) / 1.0e12,
               2.0 * macs / (ms * 1.0e-3) / 1.0e12, ms);
        CHECK(cudaEventDestroy(a));
        CHECK(cudaEventDestroy(b));
        CHECK(cudaFree(sink));
    }

    uint64_t max_in = 0, max_out = 0;
    for (int i = 0; i < kNShapes; i++) {
        if (kShapes[i].in_dim > max_in) max_in = kShapes[i].in_dim;
        if (kShapes[i].out_dim > max_out) max_out = kShapes[i].out_dim;
    }
    const uint64_t max_blocks = (max_in + 31u) / 32u;
    const uint64_t wbytes = max_out * max_blocks * 34u;

    unsigned char *w_host = (unsigned char *)malloc(wbytes);
    if (!w_host) { fprintf(stderr, "host weight alloc failed\n"); return 1; }
    fill_weights(w_host, max_out, max_blocks, 1u);

    unsigned char *w_dev = NULL;
    CHECK(cudaMalloc(&w_dev, wbytes));
    CHECK(cudaMemcpy(w_dev, w_host, wbytes, cudaMemcpyHostToDevice));

    /* The placement the model actually uses: a file-backed mapping handed to
     * the kernel as a device pointer.  Pre-faulted, so this prices the fabric
     * and not a page-fault storm. */
    const char *path = "./probe_weights.bin";
    int fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0600);
    if (fd < 0) { perror("open"); return 1; }
    if (ftruncate(fd, (off_t)wbytes) != 0) { perror("ftruncate"); return 1; }
    if (write(fd, w_host, wbytes) != (ssize_t)wbytes) { perror("write"); return 1; }
    unsigned char *w_map = (unsigned char *)mmap(NULL, wbytes, PROT_READ,
                                                 MAP_SHARED, fd, 0);
    if (w_map == MAP_FAILED) { perror("mmap"); return 1; }
    volatile unsigned char sink = 0;
    for (uint64_t i = 0; i < wbytes; i += 4096) sink ^= w_map[i];
    (void)sink;

    float *x_dev = NULL;
    int8_t *xq = NULL;
    float *xscale = NULL;
    float *out_a = NULL, *out_b = NULL;
    CHECK(cudaMalloc(&x_dev, (size_t)width * max_in * sizeof(float)));
    CHECK(cudaMalloc(&xq, (size_t)width * max_blocks * 32u));
    CHECK(cudaMalloc(&xscale, (size_t)width * max_blocks * sizeof(float)));
    CHECK(cudaMalloc(&out_a, (size_t)width * max_out * sizeof(float)));
    CHECK(cudaMalloc(&out_b, (size_t)width * max_out * sizeof(float)));

    float *x_host = (float *)malloc((size_t)width * max_in * sizeof(float));
    for (uint64_t i = 0; i < (uint64_t)width * max_in; i++) {
        x_host[i] = sinf((float)i * 0.001f) * 0.7f;
    }
    CHECK(cudaMemcpy(x_dev, x_host, (size_t)width * max_in * sizeof(float),
                     cudaMemcpyHostToDevice));

    /* ---- the activation quantiser, which the dense entry pays per projection */
    {
        printf("activation quantise, ms per call at width %u:\n", width);
        double tot_block = 0.0, tot_warp = 0.0;
        int mismatch = 0;
        int8_t *xq2 = NULL;
        float *xs2 = NULL;
        CHECK(cudaMalloc(&xq2, (size_t)width * max_blocks * 32u));
        CHECK(cudaMalloc(&xs2, (size_t)width * max_blocks * sizeof(float)));
        for (int si = 0; si < kNShapes; si++) {
            const Shape &s = kShapes[si];
            const uint64_t blocks = (s.in_dim + 31u) / 32u;
            const uint64_t pairs = (uint64_t)width * blocks;
            dim3 qgrid((unsigned)blocks, width, 1u);
            const unsigned wgrid2 = (unsigned)((pairs + 7u) / 8u);
            cudaEvent_t a, b;
            CHECK(cudaEventCreate(&a));
            CHECK(cudaEventCreate(&b));
            quantize_q8_0_f32_kernel<<<qgrid, 32>>>(xq, xscale, x_dev, s.in_dim, blocks);
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaEventRecord(a));
            for (int i = 0; i < 20; i++) {
                quantize_q8_0_f32_kernel<<<qgrid, 32>>>(xq, xscale, x_dev, s.in_dim, blocks);
            }
            CHECK(cudaEventRecord(b));
            CHECK(cudaEventSynchronize(b));
            float m1 = 0.0f;
            CHECK(cudaEventElapsedTime(&m1, a, b));
            CHECK(cudaEventRecord(a));
            for (int i = 0; i < 20; i++) {
                quantize_q8_0_f32_rows_warp_kernel<<<wgrid2, 256>>>(
                        xq2, xs2, x_dev, s.in_dim, blocks, width);
            }
            CHECK(cudaEventRecord(b));
            CHECK(cudaEventSynchronize(b));
            float m2 = 0.0f;
            CHECK(cudaEventElapsedTime(&m2, a, b));
            CHECK(cudaEventDestroy(a));
            CHECK(cudaEventDestroy(b));

            int8_t *h1 = (int8_t *)malloc((size_t)pairs * 32u);
            int8_t *h2 = (int8_t *)malloc((size_t)pairs * 32u);
            float *s1 = (float *)malloc((size_t)pairs * sizeof(float));
            float *s2 = (float *)malloc((size_t)pairs * sizeof(float));
            CHECK(cudaMemcpy(h1, xq, (size_t)pairs * 32u, cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(h2, xq2, (size_t)pairs * 32u, cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(s1, xscale, (size_t)pairs * sizeof(float), cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(s2, xs2, (size_t)pairs * sizeof(float), cudaMemcpyDeviceToHost));
            if (memcmp(h1, h2, (size_t)pairs * 32u) != 0 ||
                memcmp(s1, s2, (size_t)pairs * sizeof(float)) != 0) {
                mismatch = 1;
            }
            free(h1); free(h2); free(s1); free(s2);

            printf("  %-28s block-per-group %7.4f  warp-per-group %7.4f\n",
                   s.name, m1 / 20.0, m2 / 20.0);
            tot_block += (m1 / 20.0) * s.calls;
            tot_warp += (m2 / 20.0) * s.calls;
        }
        printf("  whole chunk: block-per-group %.1f ms (%.4f ms/token), "
               "warp-per-group %.1f ms (%.4f ms/token); same bytes: %s\n\n",
               tot_block, tot_block / width, tot_warp, tot_warp / width,
               mismatch ? "NO -- DIFFERS" : "yes");
        CHECK(cudaFree(xq2));
        CHECK(cudaFree(xs2));
    }

    /* ---- per shape, every variant, weights in device memory */
    printf("%-28s", "projection (ms / TFLOPs)");
    for (int v = 0; v < kNVariants; v++) printf(" %14s", kVariants[v].name);
    printf("\n");

    double tot[16];
    double best_tot = 0.0;
    for (int v = 0; v < kNVariants; v++) tot[v] = 0.0;
    for (int si = 0; si < kNShapes; si++) {
        const Shape &s = kShapes[si];
        const uint64_t blocks = (s.in_dim + 31u) / 32u;
        dim3 qgrid((unsigned)blocks, width, 1u);
        quantize_q8_0_f32_kernel<<<qgrid, 32>>>(xq, xscale, x_dev, s.in_dim, blocks);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());

        Ctx c;
        c.xq = xq; c.xscale = xscale; c.in_dim = s.in_dim; c.out_dim = s.out_dim;
        c.blocks = blocks; c.n_rows = width; c.use_dp4a = 1; c.w = w_dev;
        c.out = out_a;

        printf("%-28s", s.name);
        double best = 1e30;
        const char *bestname = "";
        for (int v = 0; v < kNVariants; v++) {
            const double ms = bench(kVariants[v].fn, c, 20);
            tot[v] += ms * s.calls;
            if (v > 0 && ms < best) { best = ms; bestname = kVariants[v].name; }
            const double flop = 2.0 * (double)s.in_dim * (double)s.out_dim * width;
            printf(" %7.3f/%6.1f", ms, flop / (ms * 1.0e-3) / 1.0e12);
        }
        printf("   best %s\n", bestname);
        best_tot += best * s.calls;
    }
    printf("\n%-28s", "whole chunk (ms)");
    for (int v = 0; v < kNVariants; v++) printf(" %14.1f", tot[v]);
    printf("\n%-28s", "  ms/token");
    for (int v = 0; v < kNVariants; v++) printf(" %14.4f", tot[v] / width);
    printf("\nbest tile per shape: %.1f ms = %.4f ms/token\n",
           best_tot, best_tot / width);

    /* ---- placement, on one mma variant */
    {
        printf("\nplacement (mma128x128G4): device vs mapped, ms\n");
        double d = 0.0, m = 0.0;
        for (int si = 0; si < kNShapes; si++) {
            const Shape &s = kShapes[si];
            const uint64_t blocks = (s.in_dim + 31u) / 32u;
            dim3 qgrid((unsigned)blocks, width, 1u);
            quantize_q8_0_f32_kernel<<<qgrid, 32>>>(xq, xscale, x_dev, s.in_dim, blocks);
            CHECK(cudaDeviceSynchronize());
            Ctx c;
            c.xq = xq; c.xscale = xscale; c.in_dim = s.in_dim;
            c.out_dim = s.out_dim; c.blocks = blocks; c.n_rows = width;
            c.use_dp4a = 1; c.out = out_a;
            c.w = w_dev; const double dd = bench(kVariants[1].fn, c, 20);
            c.w = w_map; const double mm = bench(kVariants[1].fn, c, 20);
            printf("  %-28s %8.3f %8.3f\n", s.name, dd, mm);
            d += dd * s.calls;
            m += mm * s.calls;
        }
        printf("  %-28s %8.1f %8.1f\n", "whole chunk", d, m);
    }

    /* ---- row invariance: the same kernel at width 1 with a padded tile */
    {
        const uint64_t in_dim = 2560, out_dim = 1024;
        const uint64_t blocks = in_dim / 32u;
        dim3 qgrid((unsigned)blocks, width, 1u);

        Ctx c;
        c.w = w_dev; c.xq = xq; c.xscale = xscale; c.in_dim = in_dim;
        c.out_dim = out_dim; c.blocks = blocks; c.use_dp4a = 1;

        int failures = 0;
        printf("\n");
        for (int v = 1; v < kNVariants; v++) {
            quantize_q8_0_f32_kernel<<<qgrid, 32>>>(xq, xscale, x_dev, in_dim, blocks);
            CHECK(cudaDeviceSynchronize());
            c.n_rows = width; c.out = out_b; kVariants[v].fn(c);
            CHECK(cudaDeviceSynchronize());
            float *wide = (float *)malloc((size_t)width * out_dim * sizeof(float));
            CHECK(cudaMemcpy(wide, out_b, (size_t)width * out_dim * sizeof(float),
                             cudaMemcpyDeviceToHost));
            int bad = 0;
            for (uint32_t r = 0; r < width && !bad; r += (width > 8 ? width / 8 : 1)) {
                dim3 q1((unsigned)blocks, 1u, 1u);
                quantize_q8_0_f32_kernel<<<q1, 32>>>(
                        xq, xscale, x_dev + (uint64_t)r * in_dim, in_dim, blocks);
                c.n_rows = 1u; c.out = out_a; kVariants[v].fn(c);
                CHECK(cudaDeviceSynchronize());
                float *one = (float *)malloc((size_t)out_dim * sizeof(float));
                CHECK(cudaMemcpy(one, out_a, (size_t)out_dim * sizeof(float),
                                 cudaMemcpyDeviceToHost));
                for (uint64_t j = 0; j < out_dim; j++) {
                    if (memcmp(&one[j], &wide[(uint64_t)r * out_dim + j], 4) != 0) {
                        printf("  %-14s ROW INVARIANCE FAILED row %u col %llu: "
                               "%.9g vs %.9g\n", kVariants[v].name, r,
                               (unsigned long long)j, one[j],
                               wide[(uint64_t)r * out_dim + j]);
                        bad = 1;
                        break;
                    }
                }
                free(one);
            }
            if (bad) failures++;
            free(wide);
        }
        printf("row invariance (width 1 padded vs width %u), all mma tiles: %s\n",
               width, failures ? "FAILED" : "exactly 0");
    }

    /* ---- every mma tile agrees with every other, bit for bit */
    {
        const uint64_t in_dim = 2560, out_dim = 1024;
        const uint64_t blocks = in_dim / 32u;
        /* Deliberately not a multiple of any tile, and never wider than the
         * buffers this run allocated. */
        const uint32_t rows = width < 300u ? width : 300u;
        dim3 qgrid((unsigned)blocks, rows, 1u);
        quantize_q8_0_f32_kernel<<<qgrid, 32>>>(xq, xscale, x_dev, in_dim, blocks);
        CHECK(cudaDeviceSynchronize());
        Ctx c;
        c.w = w_dev; c.xq = xq; c.xscale = xscale; c.in_dim = in_dim;
        c.out_dim = out_dim; c.blocks = blocks; c.n_rows = rows; c.use_dp4a = 1;
        c.out = out_a; kVariants[1].fn(c);
        CHECK(cudaDeviceSynchronize());
        float *ref = (float *)malloc((size_t)rows * out_dim * sizeof(float));
        CHECK(cudaMemcpy(ref, out_a, (size_t)rows * out_dim * sizeof(float),
                         cudaMemcpyDeviceToHost));
        int bad = 0;
        for (int v = 2; v < kNVariants; v++) {
            c.out = out_b; kVariants[v].fn(c);
            CHECK(cudaDeviceSynchronize());
            float *got = (float *)malloc((size_t)rows * out_dim * sizeof(float));
            CHECK(cudaMemcpy(got, out_b, (size_t)rows * out_dim * sizeof(float),
                             cudaMemcpyDeviceToHost));
            if (memcmp(got, ref, (size_t)rows * out_dim * sizeof(float)) != 0) {
                printf("  %-14s DISAGREES with mma128x128G4\n", kVariants[v].name);
                bad = 1;
            }
            free(got);
        }
        printf("kernel independence (%u rows, every tile AND the warp-per-row "
               "kernel vs mma128x128G4): %s\n",
               rows, bad ? "FAILED" : "exactly 0");
        free(ref);
    }

    /* ---- against a double-precision host reference */
    {
        const uint64_t in_dim = 2560, out_dim = 256;
        const uint64_t blocks = in_dim / 32u;
        const uint32_t rows = width < 64u ? width : 64u;
        dim3 qgrid((unsigned)blocks, rows, 1u);
        quantize_q8_0_f32_kernel<<<qgrid, 32>>>(xq, xscale, x_dev, in_dim, blocks);
        CHECK(cudaDeviceSynchronize());

        Ctx c;
        c.w = w_dev; c.xq = xq; c.xscale = xscale; c.in_dim = in_dim;
        c.out_dim = out_dim; c.blocks = blocks; c.n_rows = rows; c.use_dp4a = 1;
        c.out = out_a; launch_tile(c);
        c.out = out_b; kVariants[1].fn(c);
        CHECK(cudaDeviceSynchronize());

        float *ha = (float *)malloc((size_t)rows * out_dim * sizeof(float));
        float *hb = (float *)malloc((size_t)rows * out_dim * sizeof(float));
        CHECK(cudaMemcpy(ha, out_a, (size_t)rows * out_dim * sizeof(float),
                         cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(hb, out_b, (size_t)rows * out_dim * sizeof(float),
                         cudaMemcpyDeviceToHost));

        int8_t *hxq = (int8_t *)malloc((size_t)rows * blocks * 32u);
        float *hxs = (float *)malloc((size_t)rows * blocks * sizeof(float));
        CHECK(cudaMemcpy(hxq, xq, (size_t)rows * blocks * 32u, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(hxs, xscale, (size_t)rows * blocks * sizeof(float),
                         cudaMemcpyDeviceToHost));

        double worst_a = 0.0, worst_b = 0.0;
        for (uint32_t r = 0; r < rows; r++) {
            for (uint64_t o = 0; o < out_dim; o++) {
                double ref = 0.0;
                for (uint64_t b = 0; b < blocks; b++) {
                    const unsigned char *p = w_host + (o * blocks + b) * 34u;
                    __half h; memcpy(&h, p, 2);
                    int dot = 0;
                    for (int i = 0; i < 32; i++) {
                        dot += (int)(int8_t)p[2 + i] * (int)hxq[(r * blocks + b) * 32 + i];
                    }
                    ref += (double)__half2float(h) * (double)hxs[r * blocks + b] * (double)dot;
                }
                const double den = fabs(ref) > 1e-6 ? fabs(ref) : 1e-6;
                const double ea = fabs((double)ha[r * out_dim + o] - ref) / den;
                const double eb = fabs((double)hb[r * out_dim + o] - ref) / den;
                if (ea > worst_a) worst_a = ea;
                if (eb > worst_b) worst_b = eb;
            }
        }
        printf("vs double host reference: tile %.3e   mma %.3e\n", worst_a, worst_b);
        free(ha); free(hb); free(hxq); free(hxs);
    }

    munmap(w_map, wbytes);
    close(fd);
    unlink(path);
    free(w_host);
    free(x_host);
    return 0;
}
