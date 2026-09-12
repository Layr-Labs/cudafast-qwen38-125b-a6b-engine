/* Qwen4exp hyper-connection, norm, rope, embedding and head kernel tests.
 *
 * Shapes are the production ones: hidden 2560, hyper-connection width 4
 * (10240 wide), low rank 320, head dim 256 with 64 rotated dimensions at base
 * 1e7, at both sequence lengths that matter -- 1 (decode) and 1024 (prefill).
 * The vocabulary is the one reduced dimension: it is a pure GEMM extent with
 * no numerics coupling, and a Q8_0 248320x2560 table would be 675 MiB of test
 * fixture.
 *
 * Every kernel is checked against ds4_qwen4exp_hc_ref.h, the f32 reference,
 * inside the DESIGN.md section 2 bands:
 *
 *   norms, rope, elementwise, gates   max abs error <= 1e-5
 *   HC mix                            max abs error <= 2e-3, cosine >= 0.9999
 *   dense GEMM at real quant          relative Frobenius error <= 2e-2
 *
 * The bf16 rounding of the normalized value cannot be held to 1e-5 against a
 * reference that reduces the sum of squares in a different order: a scale that
 * differs by a few f32 ulps flips the occasional element to the neighbouring
 * bf16 value.  Those forms are held to one bf16 ulp with a cap on how many
 * elements may disagree, and are pinned exactly by the unit-stream case below,
 * where the statistic is exact and the two sides must agree bit for bit.
 *
 * Where the arithmetic is exact in f32 the tolerance is zero instead: the
 * rope tail, rope at position 0, the quantized embedding gather, the residual
 * inject, and a synthetic-weights run of the whole mixer whose every
 * intermediate is representable.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#include "ds4.h"
#include "ds4_gpu.h"
#include "ds4_qwen4exp_hc_ref.h"

bool ds4_log_is_tty(FILE *fp) {
    (void)fp;
    return false;
}

enum {
    N_EMBD = 2560,
    N_HC = 4,
    WIDE = N_EMBD * N_HC,
    N_LOWRANK = 320,
    HEAD_DIM = 256,
    N_ROT = 64,
    N_HEAD = 4,
    N_VOCAB = 4096,
    ROWS_LONG = 1024,
    /* GGUF quantized tensor type id for Q8_0; ds4_metal.m maps the same id. */
    TENSOR_Q8_0 = 8,
};

/* Q8_0 bytes for one row of `in_dim` values. */
enum {
    NORM_WIDE_OFF = 0,                                      /* f32[10240] */
    NORM_EMBD_OFF = NORM_WIDE_OFF + WIDE * 4,               /* f32[2560]  */
    NORM_ONES_OFF = NORM_EMBD_OFF + N_EMBD * 4,             /* f32[10240] */
    DOWN_OFF = NORM_ONES_OFF + WIDE * 4,                    /* Q8_0 320x10240 */
    UP_OFF = DOWN_OFF + N_LOWRANK * ((WIDE / 32) * 34),     /* Q8_0 10240x320 */
    INJECT_OFF = UP_OFF + WIDE * ((N_LOWRANK / 32) * 34),   /* f32 4x10240 */
    EMBD_OFF = INJECT_OFF + N_HC * WIDE * 4,                /* Q8_0 4096x2560 */
    HEAD_OFF = EMBD_OFF + N_VOCAB * ((N_EMBD / 32) * 34),   /* Q8_0 4096x2560 */
    ZERO_OFF = HEAD_OFF + N_VOCAB * ((N_EMBD / 32) * 34),   /* zeros, both quant and f32 */
    /* The SAME inject weights as INJECT_OFF, encoded Q8_0: the MTP head stores
     * this tensor quantised where the target stores it dense, and both must
     * decode.  Its f32 twin is rounded onto this grid, so the two regions hold
     * bit-identical values and the two paths are directly comparable. */
    INJECT_Q8_OFF = ZERO_OFF + N_LOWRANK * ((WIDE / 32) * 34),
    MODEL_BYTES = INJECT_Q8_OFF + N_HC * ((WIDE / 32) * 34),
};

/* Exact half constants: every scale is a power of two, so dequantization is
 * exact in f32 on the GPU and in the reference alike. */
enum {
    HALF_2_M13 = 0x0800, /* 2^-13 */
    HALF_2_M8 = 0x1c00,  /* 2^-8  */
    HALF_2_M6 = 0x2400,  /* 2^-6  */
    HALF_2_M12 = 0x0c00, /* 2^-12 */
    HALF_2_M14 = 0x0400, /* 2^-14, the inject grid */
};

static uint32_t g_rng = 0x12345678u;

static uint32_t next_u32(void) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 17;
    g_rng ^= g_rng << 5;
    return g_rng;
}

/* Uniform in [-1, 1). */
static float next_unit(void) {
    return (float)((int32_t)(next_u32() >> 8) - 8388608) * (1.0f / 8388608.0f);
}

/* The mixer takes one slab per weight, so the test builds them the way the
 * graph does.  Every slab here names the same single test mapping: what is
 * under test is the arithmetic, not the split-file resolution, which
 * tests/test_qwen4exp_graph.c covers with a two-shard model. */
static ds4_gpu_qwen4exp_slab hc_slab(const void *map, uint64_t size,
                                     uint64_t offset) {
    ds4_gpu_qwen4exp_slab slab;
    memset(&slab, 0, sizeof(slab));
    slab.map = map;
    slab.map_size = size;
    slab.offset = offset;
    return slab;
}

static void require_ok(int ok, const char *what) {
    if (!ok) {
        fprintf(stderr, "%s failed\n", what);
        exit(1);
    }
}

static void *alloc_floats(uint64_t count) {
    void *p = malloc((size_t)count * sizeof(float));
    require_ok(p != NULL, "host allocation");
    return p;
}

/* Max absolute difference, with a finiteness check on both sides. */
static void require_band(const char *what, const float *actual,
                         const float *expected, uint64_t count,
                         float tolerance) {
    double worst = 0.0;
    uint64_t worst_at = 0;
    for (uint64_t i = 0; i < count; i++) {
        if (!isfinite(actual[i]) || !isfinite(expected[i])) {
            fprintf(stderr, "%s: non-finite at %llu (%g vs %g)\n", what,
                    (unsigned long long)i, (double)actual[i],
                    (double)expected[i]);
            exit(1);
        }
        const double diff = fabs((double)actual[i] - (double)expected[i]);
        if (diff > worst) {
            worst = diff;
            worst_at = i;
        }
    }
    if (worst > (double)tolerance) {
        fprintf(stderr, "%s: max abs error %.6g at %llu (%.9g vs %.9g), band %.6g\n",
                what, worst, (unsigned long long)worst_at,
                (double)actual[worst_at], (double)expected[worst_at],
                (double)tolerance);
        exit(1);
    }
    printf("  %-56s max abs %.3g (band %.3g)\n", what, worst, (double)tolerance);
}

static void require_cosine(const char *what, const float *actual,
                           const float *expected, uint64_t count,
                           double floor_value) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (uint64_t i = 0; i < count; i++) {
        dot += (double)actual[i] * (double)expected[i];
        na += (double)actual[i] * (double)actual[i];
        nb += (double)expected[i] * (double)expected[i];
    }
    const double cosine = (na > 0.0 && nb > 0.0) ? dot / sqrt(na * nb) : 1.0;
    if (!(cosine >= floor_value)) {
        fprintf(stderr, "%s: cosine %.9g below %.9g\n", what, cosine, floor_value);
        exit(1);
    }
    printf("  %-56s cosine %.9g (floor %.4f)\n", what, cosine, floor_value);
}

static void require_relative_frobenius(const char *what, const float *actual,
                                       const float *expected, uint64_t count,
                                       double tolerance) {
    double num = 0.0, den = 0.0;
    for (uint64_t i = 0; i < count; i++) {
        const double diff = (double)actual[i] - (double)expected[i];
        num += diff * diff;
        den += (double)expected[i] * (double)expected[i];
    }
    const double rel = den > 0.0 ? sqrt(num / den) : sqrt(num);
    if (!(rel <= tolerance)) {
        fprintf(stderr, "%s: relative Frobenius error %.6g above %.6g\n",
                what, rel, tolerance);
        exit(1);
    }
    printf("  %-56s rel Frobenius %.3g (band %.3g)\n", what, rel, tolerance);
}

/* The bf16 rounding of the normalized value quantizes to 7 explicit mantissa
 * bits, so a reference that reduces the sum of squares in a different order
 * from the GPU will occasionally land on the other side of a rounding
 * boundary.  Such an element is off by exactly one bf16 ulp and no more, and
 * only a handful of elements can be affected; anything systematic shows up as
 * either a wider error or a high disagreement count. */
static void require_bf16_rounding_band(const char *what, const float *actual,
                                       const float *expected, uint64_t count) {
    uint64_t disagreements = 0;
    for (uint64_t i = 0; i < count; i++) {
        const double diff = fabs((double)actual[i] - (double)expected[i]);
        if (diff <= 1e-5) continue;
        disagreements++;
        const double one_ulp = fabs((double)expected[i]) * 0.0078125 + 1e-5;
        if (diff > one_ulp) {
            fprintf(stderr,
                    "%s: error %.6g at %llu exceeds one bf16 ulp %.6g\n",
                    what, diff, (unsigned long long)i, one_ulp);
            exit(1);
        }
    }
    /* One in a thousand is already two orders above what a reduction-order
     * difference of a few ulps can produce. */
    if (disagreements * 1000u > count) {
        fprintf(stderr, "%s: %llu of %llu elements disagree beyond 1e-5\n",
                what, (unsigned long long)disagreements,
                (unsigned long long)count);
        exit(1);
    }
    printf("  %-56s %llu of %llu at one bf16 ulp\n", what,
           (unsigned long long)disagreements, (unsigned long long)count);
}

static void require_identical(const char *what, const void *a, const void *b,
                              uint64_t bytes) {
    if (memcmp(a, b, (size_t)bytes) != 0) {
        /* Where and by how much, for a float stream: an ulp on one element
         * points at a rounding point, a whole stream at an index. */
        const float *fa = (const float *)a, *fb = (const float *)b;
        uint64_t nd = 0, first = 0;
        for (uint64_t i = 0; i < bytes / 4; i++) {
            if (fa[i] != fb[i] && nd++ == 0) first = i;
        }
        fprintf(stderr, "%s: byte streams differ (%llu of %llu floats, first "
                "[%llu] %.9g vs %.9g)\n", what, (unsigned long long)nd,
                (unsigned long long)(bytes / 4), (unsigned long long)first,
                fa[first], fb[first]);
        exit(1);
    }
}

static void require_identical_reported(const char *what, const void *a,
                                       const void *b, uint64_t bytes) {
    require_identical(what, a, b, bytes);
    printf("  %-56s exact\n", what);
}

/* Fill a Q8_0 tensor of `out_dim` rows over `in_dim` inputs. */
static void fill_q8_0(uint8_t *base, uint32_t in_dim, uint32_t out_dim,
                      uint16_t scale) {
    const uint32_t blocks = in_dim / 32u;
    for (uint32_t o = 0; o < out_dim; o++) {
        for (uint32_t b = 0; b < blocks; b++) {
            uint8_t *block = base + ((uint64_t)o * blocks + b) * 34u;
            memcpy(block, &scale, sizeof(scale));
            for (uint32_t i = 0; i < 32u; i++) {
                block[2 + i] = (uint8_t)(int8_t)((int32_t)(next_u32() % 255u) - 127);
            }
        }
    }
}

/* Encode `src` [out_dim][in_dim] as Q8_0 at a power-of-two `scale`, and round
 * `src` in place onto that grid.  Both sides then hold the same values exactly:
 * the scale is a power of two and every code is an integer, so dequantisation
 * is exact in f32 on the GPU and in the double reference alike. */
static void encode_q8_0_from_f32(uint8_t *base, float *src, uint32_t in_dim,
                                 uint32_t out_dim, uint16_t scale_half,
                                 float scale) {
    const uint32_t blocks = in_dim / 32u;
    for (uint32_t o = 0; o < out_dim; o++) {
        for (uint32_t b = 0; b < blocks; b++) {
            uint8_t *block = base + ((uint64_t)o * blocks + b) * 34u;
            memcpy(block, &scale_half, sizeof(scale_half));
            for (uint32_t i = 0; i < 32u; i++) {
                const uint64_t at = (uint64_t)o * in_dim + b * 32u + i;
                int32_t q = (int32_t)lrintf(src[at] / scale);
                if (q > 127) q = 127;
                if (q < -127) q = -127;
                block[2 + i] = (uint8_t)(int8_t)q;
                src[at] = (float)q * scale;
            }
        }
    }
}

static ds4_gpu_tensor *upload(const float *host, uint64_t count) {
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(count * sizeof(float));
    require_ok(t != NULL, "tensor allocation");
    require_ok(ds4_gpu_tensor_write(t, 0, host, count * sizeof(float)),
               "tensor upload");
    return t;
}

static void download(const ds4_gpu_tensor *t, float *host, uint64_t count) {
    require_ok(ds4_gpu_tensor_read(t, 0, host, count * sizeof(float)),
               "tensor download");
}

/* ---- norms --------------------------------------------------------- */

static void check_norm(uint8_t *model, const float *hyper_host,
                       float *gpu_out, float *ref_out,
                       ds4_gpu_tensor *x, ds4_gpu_tensor *out,
                       uint32_t n, uint32_t group, uint32_t rows,
                       uint64_t weight_offset, float weight_bias,
                       int round_bf16, const char *label) {
    const uint64_t count = (uint64_t)n * rows;
    require_ok(ds4_gpu_qwen4exp_rms_norm_tensor(
                   out, x, model, MODEL_BYTES, weight_offset, n, group, rows,
                   1e-6f, weight_bias, round_bf16),
               label);
    download(out, gpu_out, count);
    ds4_qwen4exp_ref_rms_norm(ref_out, hyper_host,
                              (const float *)(model + weight_offset), n, group,
                              rows, 1e-6f, weight_bias, round_bf16);
    if (round_bf16) {
        require_bf16_rounding_band(label, gpu_out, ref_out, count);
    } else {
        require_band(label, gpu_out, ref_out, count, 1e-5f);
    }
}

/* ---- the dense Q8_0 decode entry: the row tile against one row ------ */

/* THE INVARIANT THE SPECULATIVE CYCLE STANDS ON.
 *
 * ds4_gpu_matmul_q8_0_decode_rows_exact_tensor() serves the serial decode at
 * width one and the MTP verify at width two out of two different kernels:
 * width one takes matmul_q8_0_preq_warp8_kernel, and every width above it
 * takes matmul_q8_0_preq_rows_exact_tile_kernel<R>, which reads a weight
 * block ONCE for R activation rows.  Row j of an n-row call has to be the
 * same BITS as that row computed alone, or a batched verify and a one-row
 * decode of the same row disagree and the accept loop commits a token the
 * serial leg would never have emitted.  The gate on this track is exact token
 * equality, so a band here would hide the only failure that matters.
 *
 * The one-row kernel is the oracle: it is not part of the tile and nothing in
 * the tile's weight sharing touches it.  Every row of every width below is
 * compared against it byte for byte, at the checkpoint's real hidden width,
 * over BOTH group shapes the kernel has -- an in_dim that is a whole number
 * of 32-element groups, and one whose last group is a 16-wide tail, which is
 * the case that falls off the dp4a form onto the scalar loop -- and with the
 * dp4a form both enabled and disabled.  Widths 2 and 3 exercise the R = 2
 * tile, 4 and 5 the R = 4 tile, 8 and 9 the R = 8 tile, and 9 also exercises
 * a tile whose last slot is padded.
 */
enum { TILE_MAX_ROWS = 9 };

static void check_q8_row_tile_case(const uint8_t *model, uint64_t in_dim,
                                   const char *what) {
    static const uint32_t widths[] = { 1u, 2u, 3u, 4u, 5u, 8u, 9u };
    const uint64_t n_widths = sizeof(widths) / sizeof(widths[0]);
    const uint64_t x_count = (uint64_t)TILE_MAX_ROWS * in_dim;

    float *x = alloc_floats(x_count);
    for (uint64_t i = 0; i < x_count; i++) x[i] = next_unit();

    float *tiled = alloc_floats((uint64_t)TILE_MAX_ROWS * N_VOCAB);
    float *alone = alloc_floats(N_VOCAB);

    ds4_gpu_tensor *x_all = upload(x, x_count);
    ds4_gpu_tensor *out_all = ds4_gpu_tensor_alloc(
            (uint64_t)TILE_MAX_ROWS * N_VOCAB * sizeof(float));
    ds4_gpu_tensor *out_one =
        ds4_gpu_tensor_alloc((uint64_t)N_VOCAB * sizeof(float));
    require_ok(out_all != NULL && out_one != NULL,
               "q8_0 row tile output allocation");

    for (uint64_t w = 0; w < n_widths; w++) {
        const uint32_t rows = widths[w];
        require_ok(ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(
                       out_all, model, MODEL_BYTES, HEAD_OFF, in_dim, N_VOCAB,
                       x_all, rows),
                   "q8_0 decode rows exact, tiled call");
        download(out_all, tiled, (uint64_t)rows * N_VOCAB);
        for (uint32_t r = 0; r < rows; r++) {
            ds4_gpu_tensor *x_row = upload(x + (uint64_t)r * in_dim, in_dim);
            require_ok(ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(
                           out_one, model, MODEL_BYTES, HEAD_OFF, in_dim,
                           N_VOCAB, x_row, 1u),
                       "q8_0 decode rows exact, one-row call");
            download(out_one, alone, N_VOCAB);
            require_identical(what, tiled + (uint64_t)r * N_VOCAB, alone,
                              (uint64_t)N_VOCAB * sizeof(float));
            ds4_gpu_tensor_free(x_row);
        }
    }

    printf("  %-56s exact\n", what);
    ds4_gpu_tensor_free(out_one);
    ds4_gpu_tensor_free(out_all);
    ds4_gpu_tensor_free(x_all);
    free(alone);
    free(tiled);
    free(x);
}

static void check_q8_row_tile(const uint8_t *model) {
    const char *saved = getenv("DS4_CUDA_NO_Q8_DP4A");
    char *keep = saved ? strdup(saved) : NULL;
    require_ok(saved == NULL || keep != NULL, "environment save");

    if (saved != NULL) unsetenv("DS4_CUDA_NO_Q8_DP4A");
    check_q8_row_tile_case(model, N_EMBD,
                           "q8_0 row tile == one row, 2560 in, dp4a");
    check_q8_row_tile_case(model, N_EMBD - 16u,
                           "q8_0 row tile == one row, 2544 in, dp4a");

    /* The scalar form of the block dot, which a 16-wide tail group takes on
     * every device and which a device without dp4a takes for every group. */
    setenv("DS4_CUDA_NO_Q8_DP4A", "1", 1);
    check_q8_row_tile_case(model, N_EMBD,
                           "q8_0 row tile == one row, 2560 in, scalar");
    check_q8_row_tile_case(model, N_EMBD - 16u,
                           "q8_0 row tile == one row, 2544 in, scalar");

    if (keep != NULL) {
        setenv("DS4_CUDA_NO_Q8_DP4A", keep, 1);
        free(keep);
    } else {
        unsetenv("DS4_CUDA_NO_Q8_DP4A");
    }
}

/* ---- the pipelined MMA tile against the MMA tile ---------------------- */

/* matmul_q8_0_preq_rows_mma_pipe_kernel serves the prefill widths of every
 * dense Q8_0 projection; matmul_q8_0_preq_rows_mma_kernel is what it
 * replaces and stays as the fallback.  The two must agree BIT FOR BIT: the
 * pipelined tile changes how operands reach the tensor core and how the
 * int32 dot becomes a float, and nothing else, and a moved rounding point
 * in a 48-layer prefill diverges the carried state.  So: every production
 * (K, N) the entry sees at prefill, at the widths the tile takes -- 8, 16
 * and 64 through the padded first M tile, 1024 the prefill chunk, and 1017
 * so the last M tile is ragged -- with weights whose scales are NOT powers
 * of two (a rounding point that moved would show), quants covering the
 * int8 extremes, and the dp4a form both ways.
 *
 * ds4_gpu_set_q8_mma_pipe(2) forces the pipelined tile at every width the
 * MMA tier takes; 0 forces the tile it replaces; 1 is production. */
void ds4_gpu_set_q8_mma_pipe(int mode);

static uint16_t f32_to_half_bits(float f) {
    /* Round-to-nearest-even f32 -> f16 for normal values, which the random
     * scales below all are. */
    uint32_t u;
    memcpy(&u, &f, 4);
    const uint32_t sign = (u >> 16) & 0x8000u;
    const int32_t exp = (int32_t)((u >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = u & 0x7fffffu;
    if (exp <= 0 || exp >= 31) return (uint16_t)sign; /* out of range: zero */
    uint32_t h = sign | ((uint32_t)exp << 10) | (mant >> 13);
    const uint32_t rem = mant & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) h++;
    return (uint16_t)h;
}

static void fill_q8_0_random_scales(uint8_t *base, uint32_t in_dim,
                                    uint32_t out_dim) {
    const uint32_t blocks = in_dim / 32u;
    for (uint32_t o = 0; o < out_dim; o++) {
        for (uint32_t b = 0; b < blocks; b++) {
            uint8_t *block = base + ((uint64_t)o * blocks + b) * 34u;
            const uint32_t r = next_u32();
            float sc;
            switch (r & 3u) {
            case 0: sc = ldexpf(1.0f, -(int)((r >> 2) & 15u)); break;
            case 1: sc = ((float)(int32_t)((r >> 2) & 0xffffu) - 32768.0f) * (1.0f / 4096.0f); break;
            case 2: sc = (float)((r >> 2) & 0xfffu) * (1.0f / 512.0f); break;
            default: sc = 0.03125f * (float)((r >> 2) & 7u); break;
            }
            const uint16_t h = f32_to_half_bits(sc);
            memcpy(block, &h, sizeof(h));
            const uint32_t mode = next_u32() & 15u;
            for (uint32_t i = 0; i < 32u; i++) {
                int32_t q;
                if (mode == 0u) q = -128;
                else if (mode == 1u) q = 127;
                else if (mode == 2u) q = (i & 1u) ? 127 : -128;
                else q = (int32_t)(next_u32() & 255u) - 128;
                block[2 + i] = (uint8_t)(int8_t)q;
            }
        }
    }
}

static void check_q8_mma_pipe_shape(uint32_t in_dim, uint32_t out_dim,
                                    const char *what) {
    static const uint32_t widths[] = { 8u, 16u, 64u, 1024u, 1017u };
    const uint64_t n_widths = sizeof(widths) / sizeof(widths[0]);
    const uint32_t max_rows = 1024u;
    const uint64_t wbytes = (uint64_t)out_dim * (in_dim / 32u) * 34u;
    /* A page-aligned mapping of its own, so the entry registers it the way
     * it registers a weight range of the model file. */
    uint8_t *w = mmap(NULL, wbytes, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANON, -1, 0);
    require_ok(w != MAP_FAILED, "q8_0 mma pipe weight mapping");
    fill_q8_0_random_scales(w, in_dim, out_dim);

    const uint64_t x_count = (uint64_t)max_rows * in_dim;
    float *x = alloc_floats(x_count);
    for (uint64_t i = 0; i < x_count; i++) x[i] = next_unit();
    ds4_gpu_tensor *x_all = upload(x, x_count);
    ds4_gpu_tensor *out_t = ds4_gpu_tensor_alloc((uint64_t)max_rows * out_dim * sizeof(float));
    require_ok(out_t != NULL, "q8_0 mma pipe output allocation");
    float *ref = alloc_floats((uint64_t)max_rows * out_dim);
    float *got = alloc_floats((uint64_t)max_rows * out_dim);

    for (uint64_t wi = 0; wi < n_widths; wi++) {
        const uint32_t rows = widths[wi];
        ds4_gpu_set_q8_mma_pipe(0);
        require_ok(ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(
                       out_t, w, wbytes, 0u, in_dim, out_dim, x_all, rows),
                   "q8_0 mma tile call");
        download(out_t, ref, (uint64_t)rows * out_dim);
        ds4_gpu_set_q8_mma_pipe(2);
        require_ok(ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(
                       out_t, w, wbytes, 0u, in_dim, out_dim, x_all, rows),
                   "q8_0 mma pipe tile call");
        download(out_t, got, (uint64_t)rows * out_dim);
        char label[128];
        snprintf(label, sizeof(label), "%s, %u rows", what, rows);
        require_identical(label, got, ref, (uint64_t)rows * out_dim * sizeof(float));
    }
    ds4_gpu_set_q8_mma_pipe(1);
    printf("  %-56s exact\n", what);
    free(got);
    free(ref);
    ds4_gpu_tensor_free(out_t);
    ds4_gpu_tensor_free(x_all);
    free(x);
    munmap(w, wbytes);
}

static void check_q8_mma_pipe(void) {
    /* The dense projections of the tower at prefill, by (K, N): the GDN
     * qkv (2560 -> 10240), the attention q (2560 -> 12288), the gate
     * (2560 -> 6144), k/v (2560 -> 512), the output (6144 -> 2560), and the
     * hyper-connection down (10240 -> 320). */
    static const struct { uint32_t k, n; const char *what; } shapes[] = {
        { 2560u, 10240u, "q8_0 mma pipe == mma tile, 2560 -> 10240" },
        { 2560u, 12288u, "q8_0 mma pipe == mma tile, 2560 -> 12288" },
        { 2560u, 6144u,  "q8_0 mma pipe == mma tile, 2560 -> 6144" },
        { 2560u, 512u,   "q8_0 mma pipe == mma tile, 2560 -> 512" },
        { 6144u, 2560u,  "q8_0 mma pipe == mma tile, 6144 -> 2560" },
        { 10240u, 320u,  "q8_0 mma pipe == mma tile, 10240 -> 320" },
    };
    const char *saved = getenv("DS4_CUDA_NO_Q8_DP4A");
    char *keep = saved ? strdup(saved) : NULL;
    require_ok(saved == NULL || keep != NULL, "environment save");
    for (int dp4a_off = 0; dp4a_off < 2; dp4a_off++) {
        if (dp4a_off) setenv("DS4_CUDA_NO_Q8_DP4A", "1", 1);
        else unsetenv("DS4_CUDA_NO_Q8_DP4A");
        for (uint64_t i = 0; i < sizeof(shapes) / sizeof(shapes[0]); i++) {
            char what[96];
            snprintf(what, sizeof(what), "%s, %s", shapes[i].what,
                     dp4a_off ? "scalar" : "dp4a");
            check_q8_mma_pipe_shape(shapes[i].k, shapes[i].n, what);
        }
    }
    if (keep != NULL) {
        setenv("DS4_CUDA_NO_Q8_DP4A", keep, 1);
        free(keep);
    } else {
        unsetenv("DS4_CUDA_NO_Q8_DP4A");
    }
}

/* ---- the fused mixer against the op-by-op one ----------------------- */

/* ds4_gpu_qwen4exp_hc_mixer_tensor may fuse the chain inside the backend.  The
 * fusion is only allowed to be a faster way to compute the SAME numbers, so
 * this holds it against ds4_gpu_qwen4exp_hc_mixer_unfused_tensor -- the same
 * per-op wrappers, in the same order, with nothing fused -- and requires the
 * two to agree BIT FOR BIT.  No band, no cosine: a tolerance here would let a
 * moved rounding point through, and a moved rounding point in a 48-layer tower
 * flips an argmax and diverges an autoregressive stream.
 *
 * Every axis the fused path branches on is swept:
 *
 *   rows        1 and 2 (decode and the speculative verify width), 4, then
 *               either side of the row threshold at which the fused mix and
 *               inject collapse into one kernel, then a full prefill chunk;
 *   inject      present (F32 weights and Q8_0 weights, the target's encoding
 *               and the MTP head's) and absent, which is the tower's final
 *               mixer;
 *   weight_bias 0, the baked-offset checkpoint, and 1, the zero-centered one;
 *   round_bf16  on, which is production, and off.
 */
void ds4_gpu_enable_q8_dense_mma(void);

static void check_mixer_equivalence(uint8_t *model, const char *up_path) {
    static const uint32_t row_set[] = { 1u, 2u, 3u, 4u, 7u, 47u, 48u, 64u, 1017u, ROWS_LONG };
    const ds4_gpu_qwen4exp_slab norm_slab =
        hc_slab(model, MODEL_BYTES, NORM_WIDE_OFF);
    const ds4_gpu_qwen4exp_slab down_slab =
        hc_slab(model, MODEL_BYTES, DOWN_OFF);
    const ds4_gpu_qwen4exp_slab up_slab =
        hc_slab(model, MODEL_BYTES, UP_OFF);
    ds4_gpu_qwen4exp_slab inject_f32 =
        hc_slab(model, MODEL_BYTES, INJECT_OFF);
    ds4_gpu_qwen4exp_slab inject_q8 =
        hc_slab(model, MODEL_BYTES, INJECT_Q8_OFF);
    inject_q8.row_bytes = (uint64_t)(WIDE / 32) * 34u;
    inject_q8.type = TENSOR_Q8_0;

    uint64_t cases = 0;
#if !defined(__APPLE__) && !defined(__HIP_PLATFORM_AMD__)
    uint64_t graph_cases = 0;
    setenv("DS4_CUDA_DECODE_GRAPHS", "1", 1);
#endif
    for (uint32_t ri = 0; ri < sizeof(row_set) / sizeof(row_set[0]); ri++) {
        const uint32_t rows = row_set[ri];
        const uint64_t hc_count = (uint64_t)rows * WIDE;
        const uint64_t embd_count = (uint64_t)rows * N_EMBD;
        const uint64_t inj_count = (uint64_t)rows * N_HC;
        const uint64_t mix_slots = embd_count + 16u;
        const uint64_t inj_slots = inj_count + 16u;

        float *hyper = alloc_floats(hc_count);
        for (uint64_t i = 0; i < hc_count; i++) hyper[i] = next_unit();
        ds4_gpu_tensor *hyper_t = upload(hyper, hc_count);

        ds4_gpu_tensor *normed_t = ds4_gpu_tensor_alloc(hc_count * sizeof(float));
        ds4_gpu_tensor *wide_t = ds4_gpu_tensor_alloc(hc_count * sizeof(float));
        ds4_gpu_tensor *lowrank_t =
            ds4_gpu_tensor_alloc((uint64_t)rows * N_LOWRANK * sizeof(float));
        ds4_gpu_tensor *mixed_a = ds4_gpu_tensor_alloc(mix_slots * sizeof(float));
        ds4_gpu_tensor *mixed_b = ds4_gpu_tensor_alloc(mix_slots * sizeof(float));
        ds4_gpu_tensor *inject_a = ds4_gpu_tensor_alloc(inj_slots * sizeof(float));
        ds4_gpu_tensor *inject_b = ds4_gpu_tensor_alloc(inj_slots * sizeof(float));
        require_ok(normed_t && wide_t && lowrank_t && mixed_a && mixed_b &&
                   inject_a && inject_b, "equivalence tensor allocation");

        float *mixed_ref = alloc_floats(mix_slots);
        float *mixed_got = alloc_floats(mix_slots);
        float *inject_ref = alloc_floats(inj_slots);
        float *inject_got = alloc_floats(inj_slots);
        float *hyper_after = alloc_floats(hc_count);

        for (int head = 0; head < 3; head++) {
            const ds4_gpu_qwen4exp_slab *iw =
                head == 0 ? &inject_f32 : (head == 1 ? &inject_q8 : NULL);
            ds4_gpu_tensor *ia = head == 2 ? NULL : inject_a;
            ds4_gpu_tensor *ib = head == 2 ? NULL : inject_b;
            for (int bias = 0; bias < 2; bias++) {
                for (int bf16 = 0; bf16 < 2; bf16++) {
                    const float weight_bias = bias ? 1.0f : 0.0f;
                    memset(mixed_ref, 0xa5, mix_slots * sizeof(float));
                    memset(inject_ref, 0xa5, inj_slots * sizeof(float));
                    require_ok(ds4_gpu_tensor_write(mixed_a, 0, mixed_ref,
                                   mix_slots * sizeof(float)) &&
                               ds4_gpu_tensor_write(mixed_b, 0, mixed_ref,
                                   mix_slots * sizeof(float)) &&
                               ds4_gpu_tensor_write(inject_a, 0, inject_ref,
                                   inj_slots * sizeof(float)) &&
                               ds4_gpu_tensor_write(inject_b, 0, inject_ref,
                                   inj_slots * sizeof(float)), "HC output canaries");
                    /* Unfused first: it writes normed_scratch as a norm, the
                     * fused path writes the same buffer as quantized bytes,
                     * so running it second proves it does not depend on what
                     * the chain happened to leave there. */
                    require_ok(ds4_gpu_qwen4exp_hc_mixer_unfused_tensor(
                                   mixed_a, ia, normed_t, lowrank_t, wide_t,
                                   hyper_t, &norm_slab, &down_slab, &up_slab,
                                   iw, N_EMBD, N_HC, N_LOWRANK, rows, 1e-6f,
                                   weight_bias, bf16),
                               "unfused HC mixer");
                    download(mixed_a, mixed_ref, mix_slots);
                    if (ia) download(inject_a, inject_ref, inj_slots);

                    require_ok(ds4_gpu_qwen4exp_hc_mixer_tensor(
                                   mixed_b, ib, normed_t, lowrank_t, wide_t,
                                   hyper_t, &norm_slab, &down_slab, &up_slab,
                                   iw, N_EMBD, N_HC, N_LOWRANK, rows, 1e-6f,
                                   weight_bias, bf16),
                               "fused HC mixer");
                    download(mixed_b, mixed_got, mix_slots);
                    if (ib) download(inject_b, inject_got, inj_slots);

                    require_identical("fused HC mixer block input", mixed_got,
                                      mixed_ref, mix_slots * sizeof(float));
                    if (ia) {
                        require_identical("fused HC inject weights", inject_got,
                                          inject_ref, inj_slots * sizeof(float));
                    }
                    /* The residual is an input to both, and the fused path
                     * reads it three times instead of once; it must still not
                     * write it. */
                    download(hyper_t, hyper_after, hc_count);
                    require_identical("fused HC mixer leaves the residual alone",
                                      hyper_after, hyper,
                                      hc_count * sizeof(float));
                    cases++;
#if !defined(__APPLE__) && !defined(__HIP_PLATFORM_AMD__)
                    if (rows <= 7u) {
                        const ds4_decode_graph_key key = {
                            .il = 1u, .island = 0u, .variant = rows
                        };
                        ds4_gpu_decode_graphs_invalidate();
                        require_ok(ds4_gpu_decode_graph_begin(&key) == -1,
                                   "HC graph warm state");
                        require_ok(ds4_gpu_qwen4exp_hc_mixer_tensor(
                                       mixed_b, ib, normed_t, lowrank_t, wide_t,
                                       hyper_t, &norm_slab, &down_slab, &up_slab,
                                       iw, N_EMBD, N_HC, N_LOWRANK, rows, 1e-6f,
                                       weight_bias, bf16), "HC graph warm operation");
                        require_ok(ds4_gpu_decode_graph_begin(&key) == 0,
                                   "HC graph capture state");
                        require_ok(ds4_gpu_qwen4exp_hc_mixer_tensor(
                                       mixed_b, ib, normed_t, lowrank_t, wide_t,
                                       hyper_t, &norm_slab, &down_slab, &up_slab,
                                       iw, N_EMBD, N_HC, N_LOWRANK, rows, 1e-6f,
                                       weight_bias, bf16), "HC graph capture operation");
                        require_ok(ds4_gpu_decode_graph_end(&key) == 0,
                                   "HC graph capture complete");
                        const float magnitudes[] = {1.0f, 1e-30f, 1e10f};
                        for (unsigned change = 0; change < 3u; change++) {
                            for (uint64_t i = 0; i < hc_count; i++)
                                hyper[i] = next_unit() * magnitudes[change];
                            require_ok(ds4_gpu_tensor_write(hyper_t, 0, hyper,
                                           hc_count * sizeof(float)), "HC changed input");
                            require_ok(ds4_gpu_qwen4exp_hc_mixer_unfused_tensor(
                                           mixed_a, ia, normed_t, lowrank_t, wide_t,
                                           hyper_t, &norm_slab, &down_slab, &up_slab,
                                           iw, N_EMBD, N_HC, N_LOWRANK, rows, 1e-6f,
                                           weight_bias, bf16), "HC replay reference");
                            require_ok(ds4_gpu_decode_graph_begin(&key) == 1,
                                       "HC replay executes captured operation");
                            download(mixed_a, mixed_ref, mix_slots);
                            download(mixed_b, mixed_got, mix_slots);
                            require_identical("HC replay mixed bytes and canaries",
                                              mixed_got, mixed_ref, mix_slots * sizeof(float));
                            if (ia) {
                                download(inject_a, inject_ref, inj_slots);
                                download(inject_b, inject_got, inj_slots);
                                require_identical("HC replay inject bytes and canaries",
                                                  inject_got, inject_ref, inj_slots * sizeof(float));
                            }
                            download(hyper_t, hyper_after, hc_count);
                            require_identical("HC replay input immutability",
                                              hyper_after, hyper, hc_count * sizeof(float));
                            graph_cases++;
                        }
                        ds4_gpu_decode_graphs_invalidate();
                    }
#endif
                }
            }
        }

        free(hyper_after);
        free(inject_got);
        free(inject_ref);
        free(mixed_got);
        free(mixed_ref);
        ds4_gpu_tensor_free(inject_b);
        ds4_gpu_tensor_free(inject_a);
        ds4_gpu_tensor_free(mixed_b);
        ds4_gpu_tensor_free(mixed_a);
        ds4_gpu_tensor_free(lowrank_t);
        ds4_gpu_tensor_free(wide_t);
        ds4_gpu_tensor_free(normed_t);
        ds4_gpu_tensor_free(hyper_t);
        free(hyper);
    }
    printf("  %-56s exact over %llu cases (%s)\n",
           "fused HC mixer equals the op-by-op chain",
           (unsigned long long)cases, up_path);
#if !defined(__APPLE__) && !defined(__HIP_PLATFORM_AMD__)
    printf("  HC changed-input graph replay: %llu complete comparisons passed\n",
           (unsigned long long)graph_cases);
#endif
}

/* ---- the mixer with an owed inject against the two-call pair ---------- */

/* ds4_gpu_qwen4exp_hc_mixer_pending_tensor is DEFINED as the pair
 *
 *     ds4_gpu_qwen4exp_hc_inject_tensor(hyper, hyper, block, inject)
 *     ds4_gpu_qwen4exp_hc_mixer_tensor(..., hyper, ...)
 *
 * and the CUDA backend folds the apply into the norm pass's first read of the
 * residual at prefill widths.  This holds the one call against the pair at
 * zero tolerance on all three things it touches: the residual it leaves
 * behind, the mixed block input, and the new inject head.  The pending inject
 * is passed AS the mixer's own inject output tensor, which is how the engine
 * calls it (the same session slot holds last block's head and this block's),
 * so the read-before-write inside the kernel is exercised, not assumed.
 *
 * Widths: below the fold threshold (the pair runs verbatim), at it, a full
 * chunk, and a ragged one (1017: the last token block is no different, but a
 * width that is not a multiple of anything is the one a stride error shows
 * on).  Inject encodings f32 and Q8_0, both flags, and the final mixer
 * (inject head absent, the pending apply still owed). */
static void check_mixer_pending(uint8_t *model, const char *up_path) {
    static const uint32_t row_set[] = { 1u, 7u, 47u, 48u, 64u, 1017u, ROWS_LONG };
    const ds4_gpu_qwen4exp_slab norm_slab =
        hc_slab(model, MODEL_BYTES, NORM_WIDE_OFF);
    const ds4_gpu_qwen4exp_slab down_slab =
        hc_slab(model, MODEL_BYTES, DOWN_OFF);
    const ds4_gpu_qwen4exp_slab up_slab =
        hc_slab(model, MODEL_BYTES, UP_OFF);
    ds4_gpu_qwen4exp_slab inject_f32 =
        hc_slab(model, MODEL_BYTES, INJECT_OFF);
    ds4_gpu_qwen4exp_slab inject_q8 =
        hc_slab(model, MODEL_BYTES, INJECT_Q8_OFF);
    inject_q8.row_bytes = (uint64_t)(WIDE / 32) * 34u;
    inject_q8.type = TENSOR_Q8_0;

    uint64_t cases = 0;
    for (uint32_t ri = 0; ri < sizeof(row_set) / sizeof(row_set[0]); ri++) {
        const uint32_t rows = row_set[ri];
        const uint64_t hc_count = (uint64_t)rows * WIDE;
        const uint64_t embd_count = (uint64_t)rows * N_EMBD;
        const uint64_t inj_count = (uint64_t)rows * N_HC;
        const uint64_t mix_slots = embd_count + 16u;

        float *hyper = alloc_floats(hc_count);
        float *block = alloc_floats(embd_count);
        float *inj0 = alloc_floats(inj_count);
        for (uint64_t i = 0; i < hc_count; i++) hyper[i] = next_unit();
        for (uint64_t i = 0; i < embd_count; i++) block[i] = 0.5f * next_unit();
        for (uint64_t i = 0; i < inj_count; i++) inj0[i] = 1.0f + 0.7f * next_unit();

        ds4_gpu_tensor *hyper_a = upload(hyper, hc_count);
        ds4_gpu_tensor *hyper_b = upload(hyper, hc_count);
        ds4_gpu_tensor *block_t = upload(block, embd_count);
        ds4_gpu_tensor *inject_a = upload(inj0, inj_count);
        ds4_gpu_tensor *inject_b = upload(inj0, inj_count);
        ds4_gpu_tensor *normed_t = ds4_gpu_tensor_alloc(hc_count * sizeof(float));
        ds4_gpu_tensor *wide_t = ds4_gpu_tensor_alloc(hc_count * sizeof(float));
        ds4_gpu_tensor *lowrank_t =
            ds4_gpu_tensor_alloc((uint64_t)rows * N_LOWRANK * sizeof(float));
        ds4_gpu_tensor *mixed_a = ds4_gpu_tensor_alloc(mix_slots * sizeof(float));
        ds4_gpu_tensor *mixed_b = ds4_gpu_tensor_alloc(mix_slots * sizeof(float));
        require_ok(hyper_a && hyper_b && block_t && inject_a && inject_b &&
                   normed_t && wide_t && lowrank_t && mixed_a && mixed_b,
                   "pending tensor allocation");

        float *hyper_ref = alloc_floats(hc_count);
        float *hyper_got = alloc_floats(hc_count);
        float *mixed_ref = alloc_floats(mix_slots);
        float *mixed_got = alloc_floats(mix_slots);
        float *inject_ref = alloc_floats(inj_count);
        float *inject_got = alloc_floats(inj_count);

        for (int head = 0; head < 3; head++) {
            const ds4_gpu_qwen4exp_slab *iw =
                head == 0 ? &inject_f32 : (head == 1 ? &inject_q8 : NULL);
            for (int bias = 0; bias < 2; bias++) {
                for (int bf16 = 0; bf16 < 2; bf16++) {
                    const float weight_bias = bias ? 1.0f : 0.0f;
                    /* Fresh residual and pending head on both sides. */
                    require_ok(ds4_gpu_tensor_write(hyper_a, 0, hyper, hc_count * sizeof(float)) &&
                               ds4_gpu_tensor_write(hyper_b, 0, hyper, hc_count * sizeof(float)) &&
                               ds4_gpu_tensor_write(inject_a, 0, inj0, inj_count * sizeof(float)) &&
                               ds4_gpu_tensor_write(inject_b, 0, inj0, inj_count * sizeof(float)),
                               "pending inputs");
                    memset(mixed_ref, 0xa5, mix_slots * sizeof(float));
                    require_ok(ds4_gpu_tensor_write(mixed_a, 0, mixed_ref, mix_slots * sizeof(float)) &&
                               ds4_gpu_tensor_write(mixed_b, 0, mixed_ref, mix_slots * sizeof(float)),
                               "pending canaries");

                    /* The pair. */
                    require_ok(ds4_gpu_qwen4exp_hc_inject_tensor(
                                   hyper_a, hyper_a, block_t, inject_a,
                                   N_EMBD, N_HC, rows), "pending reference inject");
                    require_ok(ds4_gpu_qwen4exp_hc_mixer_tensor(
                                   mixed_a, iw ? inject_a : NULL, normed_t, lowrank_t,
                                   wide_t, hyper_a, &norm_slab, &down_slab, &up_slab,
                                   iw, N_EMBD, N_HC, N_LOWRANK, rows, 1e-6f,
                                   weight_bias, bf16), "pending reference mixer");
                    download(hyper_a, hyper_ref, hc_count);
                    download(mixed_a, mixed_ref, mix_slots);
                    download(inject_a, inject_ref, inj_count);

                    /* The one call, the pending head in the mixer's own slot. */
                    require_ok(ds4_gpu_qwen4exp_hc_mixer_pending_tensor(
                                   mixed_b, iw ? inject_b : NULL, normed_t, lowrank_t,
                                   wide_t, hyper_b, &norm_slab, &down_slab, &up_slab,
                                   iw, N_EMBD, N_HC, N_LOWRANK, rows, 1e-6f,
                                   weight_bias, bf16, block_t, inject_b),
                               "pending mixer");
                    download(hyper_b, hyper_got, hc_count);
                    download(mixed_b, mixed_got, mix_slots);
                    download(inject_b, inject_got, inj_count);

                    require_identical("pending mixer residual", hyper_got, hyper_ref,
                                      hc_count * sizeof(float));
                    require_identical("pending mixer block input", mixed_got, mixed_ref,
                                      mix_slots * sizeof(float));
                    require_identical("pending mixer inject head", inject_got, inject_ref,
                                      inj_count * sizeof(float));
                    /* And it did apply: the residual moved unless the pending head
                     * was exactly zero, which it is not. */
                    require_ok(memcmp(hyper_got, hyper, hc_count * sizeof(float)) != 0,
                               "pending mixer applied the inject");
                    cases++;
                }
            }
        }

        free(inject_got); free(inject_ref);
        free(mixed_got); free(mixed_ref);
        free(hyper_got); free(hyper_ref);
        ds4_gpu_tensor_free(mixed_b); ds4_gpu_tensor_free(mixed_a);
        ds4_gpu_tensor_free(lowrank_t); ds4_gpu_tensor_free(wide_t);
        ds4_gpu_tensor_free(normed_t);
        ds4_gpu_tensor_free(inject_b); ds4_gpu_tensor_free(inject_a);
        ds4_gpu_tensor_free(block_t);
        ds4_gpu_tensor_free(hyper_b); ds4_gpu_tensor_free(hyper_a);
        free(inj0); free(block); free(hyper);
    }
    printf("  %-56s exact over %llu cases (%s)\n",
           "mixer with owed inject equals inject then mixer",
           (unsigned long long)cases, up_path);
}

int main(void) {
    uint8_t *model = mmap(NULL, MODEL_BYTES, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANON, -1, 0);
    if (model == MAP_FAILED) {
        perror("mmap");
        return 1;
    }

    float *norm_wide = (float *)(model + NORM_WIDE_OFF);
    float *norm_embd = (float *)(model + NORM_EMBD_OFF);
    float *norm_ones = (float *)(model + NORM_ONES_OFF);
    float *inject_w = (float *)(model + INJECT_OFF);
    for (uint32_t i = 0; i < WIDE; i++) {
        norm_wide[i] = 0.25f * next_unit();
        norm_ones[i] = 1.0f;
    }
    for (uint32_t i = 0; i < N_EMBD; i++) norm_embd[i] = 0.25f * next_unit();
    for (uint32_t i = 0; i < N_HC * WIDE; i++) inject_w[i] = 0.002f * next_unit();
    /* Round the dense inject weights onto the Q8_0 grid and write the same
     * values into the quantised region.  Both are then exact and equal, so a
     * difference between the two paths is the KERNEL's, not the encoding's. */
    encode_q8_0_from_f32(model + INJECT_Q8_OFF, inject_w, WIDE, N_HC,
                         HALF_2_M14, ldexpf(1.0f, -14));
    fill_q8_0(model + DOWN_OFF, WIDE, N_LOWRANK, HALF_2_M13);
    fill_q8_0(model + UP_OFF, N_LOWRANK, WIDE, HALF_2_M6);
    fill_q8_0(model + EMBD_OFF, N_EMBD, N_VOCAB, HALF_2_M8);
    fill_q8_0(model + HEAD_OFF, N_EMBD, N_VOCAB, HALF_2_M12);

    require_ok(ds4_gpu_init(), "GPU initialization");
    require_ok(ds4_gpu_set_model_map(model, MODEL_BYTES), "model map registration");

    check_q8_row_tile(model);

    const uint32_t row_counts[2] = { 1u, ROWS_LONG };
    for (uint32_t which = 0; which < 2u; which++) {
        const uint32_t rows = row_counts[which];
        const uint64_t hc_count = (uint64_t)rows * WIDE;
        const uint64_t embd_count = (uint64_t)rows * N_EMBD;

        float *hyper = alloc_floats(hc_count);
        float *gpu_wide = alloc_floats(hc_count);
        float *ref_wide = alloc_floats(hc_count);
        for (uint64_t i = 0; i < hc_count; i++) hyper[i] = next_unit();

        ds4_gpu_tensor *hyper_t = upload(hyper, hc_count);
        ds4_gpu_tensor *normed_t = ds4_gpu_tensor_alloc(hc_count * sizeof(float));
        ds4_gpu_tensor *wide_t = ds4_gpu_tensor_alloc(hc_count * sizeof(float));
        require_ok(normed_t && wide_t, "wide tensor allocation");

        /* Grouped hc_norm: each of the four streams on its own statistic,
         * both checkpoint conventions, with and without the bf16 rounding of
         * the normalized value that MLX's activation dtype imposes. */
        check_norm(model, hyper, gpu_wide, ref_wide, hyper_t, normed_t,
                   WIDE, N_EMBD, rows, NORM_WIDE_OFF, 0.0f, 1,
                   "qwen4exp grouped RMS norm, offset baked, bf16 rounded");
        check_norm(model, hyper, gpu_wide, ref_wide, hyper_t, normed_t,
                   WIDE, N_EMBD, rows, NORM_WIDE_OFF, 0.0f, 0,
                   "qwen4exp grouped RMS norm, offset baked, f32");
        check_norm(model, hyper, gpu_wide, ref_wide, hyper_t, normed_t,
                   WIDE, N_EMBD, rows, NORM_WIDE_OFF, 1.0f, 0,
                   "qwen4exp grouped RMS norm, zero centered");
        check_norm(model, hyper, gpu_wide, ref_wide, hyper_t, normed_t,
                   WIDE, N_EMBD, rows, NORM_WIDE_OFF, 1.0f, 1,
                   "qwen4exp grouped RMS norm, zero centered, bf16 rounded");
        /* Ungrouped over the whole 10240: the MTP head's hidden pre-norm. */
        check_norm(model, hyper, gpu_wide, ref_wide, hyper_t, normed_t,
                   WIDE, WIDE, rows, NORM_WIDE_OFF, 0.0f, 0,
                   "qwen4exp ungrouped RMS norm over the hyper stream");

        /* Ordinary 2560-wide norm. */
        {
            float *narrow = alloc_floats(embd_count);
            float *gpu_narrow = alloc_floats(embd_count);
            float *ref_narrow = alloc_floats(embd_count);
            for (uint64_t i = 0; i < embd_count; i++) narrow[i] = next_unit();
            ds4_gpu_tensor *nx = upload(narrow, embd_count);
            ds4_gpu_tensor *no = ds4_gpu_tensor_alloc(embd_count * sizeof(float));
            require_ok(no != NULL, "narrow norm output allocation");
            check_norm(model, narrow, gpu_narrow, ref_narrow, nx, no,
                       N_EMBD, N_EMBD, rows, NORM_EMBD_OFF, 1.0f, 0,
                       "qwen4exp hidden RMS norm");
            ds4_gpu_tensor_free(no);
            ds4_gpu_tensor_free(nx);
            free(ref_narrow);
            free(gpu_narrow);
            free(narrow);
        }

        /* Exact in f32: unit-magnitude streams, unit weights, no epsilon, so
         * the statistic is exactly 1 and the norm is the identity. */
        {
            float *unit = alloc_floats(hc_count);
            for (uint64_t i = 0; i < hc_count; i++)
                unit[i] = (next_u32() & 1u) ? 1.0f : -1.0f;
            ds4_gpu_tensor *ux = upload(unit, hc_count);
            require_ok(ds4_gpu_qwen4exp_rms_norm_tensor(
                           normed_t, ux, model, MODEL_BYTES, NORM_ONES_OFF,
                           WIDE, N_EMBD, rows, 0.0f, 0.0f, 1),
                       "qwen4exp exact RMS norm");
            download(normed_t, gpu_wide, hc_count);
            require_identical_reported("qwen4exp exact RMS norm is the identity", gpu_wide,
                              unit, hc_count * sizeof(float));
            ds4_gpu_tensor_free(ux);
            free(unit);
        }

        /* ---- the gated residual mixer ------------------------------- */
        {
            float *lowrank_ref = alloc_floats((uint64_t)rows * N_LOWRANK);
            float *mixed_ref = alloc_floats(embd_count);
            float *mixed_gpu = alloc_floats(embd_count);
            float *inject_ref = alloc_floats((uint64_t)rows * N_HC);
            float *inject_gpu = alloc_floats((uint64_t)rows * N_HC);

            ds4_gpu_tensor *lowrank_t =
                ds4_gpu_tensor_alloc((uint64_t)rows * N_LOWRANK * sizeof(float));
            ds4_gpu_tensor *mixed_t =
                ds4_gpu_tensor_alloc(embd_count * sizeof(float));
            ds4_gpu_tensor *inject_t =
                ds4_gpu_tensor_alloc((uint64_t)rows * N_HC * sizeof(float));
            require_ok(lowrank_t && mixed_t && inject_t, "mixer tensor allocation");

            const ds4_gpu_qwen4exp_slab norm_wide_slab =
                hc_slab(model, MODEL_BYTES, NORM_WIDE_OFF);
            const ds4_gpu_qwen4exp_slab down_slab =
                hc_slab(model, MODEL_BYTES, DOWN_OFF);
            const ds4_gpu_qwen4exp_slab up_slab =
                hc_slab(model, MODEL_BYTES, UP_OFF);
            const ds4_gpu_qwen4exp_slab inject_slab =
                hc_slab(model, MODEL_BYTES, INJECT_OFF);
            const ds4_gpu_qwen4exp_slab norm_ones_slab =
                hc_slab(model, MODEL_BYTES, NORM_ONES_OFF);
            const ds4_gpu_qwen4exp_slab zero_slab =
                hc_slab(model, MODEL_BYTES, ZERO_OFF);

            require_ok(ds4_gpu_qwen4exp_hc_mixer_tensor(
                           mixed_t, inject_t, normed_t, lowrank_t, wide_t,
                           hyper_t, &norm_wide_slab, &down_slab, &up_slab,
                           &inject_slab, N_EMBD, N_HC,
                           N_LOWRANK, rows, 1e-6f, 0.0f, 1),
                       "qwen4exp HC mixer");
            download(mixed_t, mixed_gpu, embd_count);
            download(inject_t, inject_gpu, (uint64_t)rows * N_HC);

            ds4_qwen4exp_ref_hc_mixer(mixed_ref, inject_ref, ref_wide,
                                      lowrank_ref, gpu_wide, hyper,
                                      norm_wide, model + DOWN_OFF,
                                      model + UP_OFF, inject_w, N_EMBD, N_HC,
                                      N_LOWRANK, rows, 1e-6f, 0.0f, 1);
            require_band("qwen4exp HC mixer block input", mixed_gpu, mixed_ref,
                         embd_count, 2e-3f);
            require_cosine("qwen4exp HC mixer block input", mixed_gpu, mixed_ref,
                           embd_count, 0.9999);
            require_band("qwen4exp HC inject weights", inject_gpu, inject_ref,
                         (uint64_t)rows * N_HC, 2e-3f);

            /* The SAME inject weights, stored Q8_0 -- which is how the MTP head
             * ships this tensor while the target ships it F32.  The loader
             * accepted both from the start; the kernel read dense F32 only, so
             * a head block reached it and got zero back.  Both decode now, from
             * the one table in ds4_qwen4exp_hc_types.h.
             *
             * The dense region was rounded onto this Q8_0 grid at setup, so the
             * two hold identical values and the outputs must be BIT-identical:
             * a band would hide a decoder that read the right magnitude from
             * the wrong bytes. */
            {
                ds4_gpu_qwen4exp_slab q8_slab =
                    hc_slab(model, MODEL_BYTES, INJECT_Q8_OFF);
                q8_slab.type = TENSOR_Q8_0;
                q8_slab.row_bytes = (uint64_t)(WIDE / 32u) * 34u;

                /* `normed_scratch` is scratch: a backend that fuses the
                 * chain may leave the down projection's quantized bytes there
                 * rather than the normalized stream, and nothing in the engine
                 * reads it after the mixer returns.  Rebuild it here, with the
                 * mixer's own norm arguments, so this check reads exactly the
                 * values the mixer's inject head read. */
                require_ok(ds4_gpu_qwen4exp_rms_norm_tensor(
                               normed_t, hyper_t, model, MODEL_BYTES,
                               NORM_WIDE_OFF, WIDE, N_EMBD, rows, 1e-6f, 0.0f,
                               1),
                           "qwen4exp normed rebuild for the Q8_0 inject check");

                float *inject_q8 = alloc_floats((uint64_t)rows * N_HC);
                ds4_gpu_tensor *inject_q8_t =
                    ds4_gpu_tensor_alloc((uint64_t)rows * N_HC * sizeof(float));
                require_ok(inject_q8_t != NULL, "Q8_0 inject tensor allocation");
                require_ok(ds4_gpu_qwen4exp_hc_inject_weights_tensor(
                               inject_q8_t, normed_t, &q8_slab, N_EMBD, N_HC,
                               rows),
                           "qwen4exp Q8_0 inject weights");
                download(inject_q8_t, inject_q8, (uint64_t)rows * N_HC);
                require_band("qwen4exp Q8_0 inject weights", inject_q8,
                             inject_ref, (uint64_t)rows * N_HC, 2e-3f);
                require_identical_reported(
                        "qwen4exp Q8_0 inject equals the F32 inject exactly",
                        inject_q8, inject_gpu,
                        (uint64_t)rows * N_HC * sizeof(float));
                ds4_gpu_tensor_free(inject_q8_t);
                free(inject_q8);
            }

            /* LM head over the mixer output, at the real hidden width. */
            {
                float *logits_ref = alloc_floats((uint64_t)rows * N_VOCAB);
                float *logits_gpu = alloc_floats((uint64_t)rows * N_VOCAB);
                ds4_gpu_tensor *logits_t =
                    ds4_gpu_tensor_alloc((uint64_t)rows * N_VOCAB * sizeof(float));
                require_ok(logits_t != NULL, "logits allocation");
                require_ok(ds4_gpu_matmul_q8_0_tensor(
                               logits_t, model, MODEL_BYTES, HEAD_OFF, N_EMBD,
                               N_VOCAB, mixed_t, rows),
                           "qwen4exp LM head");
                download(logits_t, logits_gpu, (uint64_t)rows * N_VOCAB);
                ds4_qwen4exp_ref_matmul_q8_0(logits_ref, model + HEAD_OFF,
                                             N_EMBD, N_VOCAB, mixed_gpu, rows);
                require_relative_frobenius("qwen4exp LM head", logits_gpu,
                                           logits_ref,
                                           (uint64_t)rows * N_VOCAB, 2e-2);
                ds4_gpu_tensor_free(logits_t);
                free(logits_gpu);
                free(logits_ref);
            }

            /* The mixer must not touch the stream it reads: that stream is
             * the residual, and at the tower's last mixer it is also what the
             * native MTP head consumes. */
            download(hyper_t, gpu_wide, hc_count);
            require_identical_reported("qwen4exp pre-mixer stream survives the mixer",
                              gpu_wide, hyper, hc_count * sizeof(float));

            /* Determinism: three runs, byte for byte. */
            {
                float *again = alloc_floats(embd_count);
                for (int run = 0; run < 2; run++) {
                    require_ok(ds4_gpu_qwen4exp_hc_mixer_tensor(
                                   mixed_t, inject_t, normed_t, lowrank_t,
                                   wide_t, hyper_t, &norm_wide_slab,
                                   &down_slab, &up_slab, &inject_slab,
                                   N_EMBD, N_HC, N_LOWRANK, rows, 1e-6f, 0.0f, 1),
                               "qwen4exp HC mixer replay");
                    download(mixed_t, again, embd_count);
                    require_identical("qwen4exp HC mixer is deterministic",
                                      again, mixed_gpu, embd_count * sizeof(float));
                }
                free(again);
            }

            /* Residual inject: identical operands on both sides, exact in
             * f32, so it has to match to the bit. */
            {
                float *out_ref = alloc_floats(hc_count);
                float *out_gpu = alloc_floats(hc_count);
                ds4_gpu_tensor *out_t =
                    ds4_gpu_tensor_alloc(hc_count * sizeof(float));
                ds4_gpu_tensor *inject_exact_t =
                    upload(inject_gpu, (uint64_t)rows * N_HC);
                ds4_gpu_tensor *block_t = upload(mixed_gpu, embd_count);
                require_ok(out_t != NULL, "inject output allocation");
                require_ok(ds4_gpu_qwen4exp_hc_inject_tensor(
                               out_t, hyper_t, block_t, inject_exact_t,
                               N_EMBD, N_HC, rows),
                           "qwen4exp HC inject");
                download(out_t, out_gpu, hc_count);
                ds4_qwen4exp_ref_hc_inject(out_ref, hyper, mixed_gpu, inject_gpu,
                                           N_EMBD, N_HC, rows);
                require_identical_reported("qwen4exp HC inject is exact", out_gpu, out_ref,
                                  hc_count * sizeof(float));
                ds4_gpu_tensor_free(block_t);
                ds4_gpu_tensor_free(inject_exact_t);
                ds4_gpu_tensor_free(out_t);
                free(out_gpu);
                free(out_ref);
            }

            /* Synthetic weights that make the whole mixer exact: unit
             * streams, unit norm weights, no epsilon, and a zeroed low-rank
             * gate, so every gate is sigmoid(0) = 0.5 and every inject is
             * 2*sigmoid(0) = 1.  Nothing in the chain rounds. */
            {
                float *unit = alloc_floats(hc_count);
                for (uint64_t i = 0; i < hc_count; i++)
                    unit[i] = (next_u32() & 1u) ? 1.0f : -1.0f;
                ds4_gpu_tensor *ux = upload(unit, hc_count);
                require_ok(ds4_gpu_qwen4exp_hc_mixer_tensor(
                               mixed_t, inject_t, normed_t, lowrank_t, wide_t,
                               ux, &norm_ones_slab, &zero_slab, &zero_slab,
                               &zero_slab, N_EMBD, N_HC, N_LOWRANK,
                               rows, 0.0f, 0.0f, 1),
                           "qwen4exp synthetic HC mixer");
                download(mixed_t, mixed_gpu, embd_count);
                download(inject_t, inject_gpu, (uint64_t)rows * N_HC);
                ds4_qwen4exp_ref_hc_mixer(mixed_ref, inject_ref, ref_wide,
                                          lowrank_ref, gpu_wide, unit,
                                          norm_ones, model + ZERO_OFF,
                                          model + ZERO_OFF,
                                          (const float *)(model + ZERO_OFF),
                                          N_EMBD, N_HC, N_LOWRANK, rows,
                                          0.0f, 0.0f, 1);
                require_identical_reported("qwen4exp synthetic HC mixer is exact",
                                  mixed_gpu, mixed_ref, embd_count * sizeof(float));
                require_identical("qwen4exp synthetic inject weights are exact",
                                  inject_gpu, inject_ref,
                                  (uint64_t)rows * N_HC * sizeof(float));
                for (uint64_t i = 0; i < (uint64_t)rows * N_HC; i++) {
                    if (inject_gpu[i] != 1.0f) {
                        fprintf(stderr,
                                "qwen4exp synthetic inject weight %llu is %.9g, not 1\n",
                                (unsigned long long)i, (double)inject_gpu[i]);
                        return 1;
                    }
                }
                ds4_gpu_tensor_free(ux);
                free(unit);
            }

            ds4_gpu_tensor_free(inject_t);
            ds4_gpu_tensor_free(mixed_t);
            ds4_gpu_tensor_free(lowrank_t);
            free(inject_gpu);
            free(inject_ref);
            free(mixed_gpu);
            free(mixed_ref);
            free(lowrank_ref);
        }

        /* ---- embedding gather, tiled into the streams ---------------- */
        {
            int32_t *tokens = malloc((size_t)rows * sizeof(int32_t));
            require_ok(tokens != NULL, "token allocation");
            for (uint32_t t = 0; t < rows; t++)
                tokens[t] = (int32_t)(next_u32() % (uint32_t)N_VOCAB);
            ds4_gpu_tensor *tok_t =
                ds4_gpu_tensor_alloc((uint64_t)rows * sizeof(int32_t));
            ds4_gpu_tensor *rows_t =
                ds4_gpu_tensor_alloc(embd_count * sizeof(float));
            require_ok(tok_t && rows_t, "embedding tensor allocation");
            require_ok(ds4_gpu_tensor_write(tok_t, 0, tokens,
                                            (uint64_t)rows * sizeof(int32_t)),
                       "token upload");
            require_ok(ds4_gpu_qwen4exp_embed_tokens_hc_tensor(
                           normed_t, rows_t, tok_t, model, MODEL_BYTES,
                           EMBD_OFF, TENSOR_Q8_0, N_VOCAB, rows, N_EMBD, N_HC),
                       "qwen4exp embedding gather");
            download(normed_t, gpu_wide, hc_count);
            ds4_qwen4exp_ref_embed_hc_q8_0(ref_wide, model + EMBD_OFF, tokens,
                                           rows, N_EMBD, N_HC);
            require_identical_reported("qwen4exp embedding gather is exact", gpu_wide,
                              ref_wide, hc_count * sizeof(float));
            ds4_gpu_tensor_free(rows_t);
            ds4_gpu_tensor_free(tok_t);
            free(tokens);
        }

        /* ---- partial rope -------------------------------------------- */
        {
            /* The inverse-frequency table is shared with the QSA lane through
             * ds4_gpu_qwen4exp_rope_inv_freq, which is ds4_qwen4exp_rope_inv_freq
             * under an exported name.  Pin it against a double-precision
             * reference computed offline in Python:
             *   [math.exp(2*j*(-math.log(1e7)/64)) for j in range(32)]
             * rounded to float32.  A float log/exp pair drifts here, and the
             * drift only shows up multiplied by the token position. */
            static const uint32_t expected_bits[N_ROT / 2] = {
                0x3f800000u, 0x3f1ab32bu, 0x3ebaf81au, 0x3e61f836u,
                0x3e088d77u, 0x3da50957u, 0x3d47763fu, 0x3cf11176u,
                0x3c91ad39u, 0x3c301052u, 0x3bd4ca14u, 0x3b80967du,
                0x3b1b690du, 0x3abbd3ecu, 0x3a6301e2u, 0x3a092e02u,
                0x39a5cb5fu, 0x394860c1u, 0x38f22ce3u, 0x3892587fu,
                0x3830df51u, 0x37d5c442u, 0x37812dacu, 0x371c1fc4u,
                0x36bcb0c1u, 0x36640cc6u, 0x3609cf4bu, 0x35a68e4cu,
                0x35494c56u, 0x34f3499cu, 0x3493048eu, 0x3431af44u,
            };
            float inv_freq[N_ROT / 2];
            ds4_gpu_qwen4exp_rope_inv_freq(inv_freq, N_ROT, 1e7f);
            require_identical_reported(
                "qwen4exp rope inverse frequencies match the double reference",
                inv_freq, expected_bits, sizeof(expected_bits));

            const uint64_t rope_count = (uint64_t)rows * N_HEAD * HEAD_DIM;
            float *rope_host = alloc_floats(rope_count);
            float *rope_ref = alloc_floats(rope_count);
            float *rope_gpu = alloc_floats(rope_count);
            for (uint64_t i = 0; i < rope_count; i++) rope_host[i] = next_unit();

            ds4_gpu_tensor *freq_t = upload(inv_freq, N_ROT / 2);
            ds4_gpu_tensor *rope_t = upload(rope_host, rope_count);
            require_ok(ds4_gpu_qwen4exp_rope_head_tensor(
                           rope_t, freq_t, rows, N_HEAD, HEAD_DIM, N_ROT, 0),
                       "qwen4exp partial rope");
            download(rope_t, rope_gpu, rope_count);
            memcpy(rope_ref, rope_host, (size_t)rope_count * sizeof(float));
            ds4_qwen4exp_ref_rope_head(rope_ref, rows, N_HEAD, HEAD_DIM, N_ROT,
                                       0, 1, 1e7f);
            require_band("qwen4exp partial rope", rope_gpu, rope_ref,
                         rope_count, 1e-5f);
            /* The unrotated dimensions of every head are untouched. */
            for (uint64_t head = 0; head < (uint64_t)rows * N_HEAD; head++) {
                require_identical("qwen4exp rope leaves the head tail alone",
                                  rope_gpu + head * HEAD_DIM + N_ROT,
                                  rope_host + head * HEAD_DIM + N_ROT,
                                  (HEAD_DIM - N_ROT) * sizeof(float));
            }

            /* Position 0 is the identity, exactly. */
            require_ok(ds4_gpu_tensor_write(rope_t, 0, rope_host,
                                            rope_count * sizeof(float)),
                       "rope reset");
            require_ok(ds4_gpu_qwen4exp_rope_head_tensor(
                           rope_t, freq_t, 1, N_HEAD, HEAD_DIM, N_ROT, 0),
                       "qwen4exp partial rope at position zero");
            download(rope_t, rope_gpu, (uint64_t)N_HEAD * HEAD_DIM);
            require_identical_reported("qwen4exp rope at position zero is the identity",
                              rope_gpu, rope_host,
                              (uint64_t)N_HEAD * HEAD_DIM * sizeof(float));

            ds4_gpu_tensor_free(rope_t);
            ds4_gpu_tensor_free(freq_t);
            free(rope_gpu);
            free(rope_ref);
            free(rope_host);
        }

        ds4_gpu_tensor_free(wide_t);
        ds4_gpu_tensor_free(normed_t);
        ds4_gpu_tensor_free(hyper_t);
        free(ref_wide);
        free(gpu_wide);
        free(hyper);
        printf("qwen4exp HC/norm/rope/embedding/head: %u row%s ok\n", rows,
               rows == 1u ? "" : "s");
    }

    check_mixer_equivalence(model, "eight-row tile");
    check_mixer_pending(model, "eight-row tile");

    /* The qwen4exp tower switches the dense Q8_0 projections onto the int8
     * MMA tile (ds4_qwen4exp.inc, qwen4exp_finish_derived), and the fused
     * mixer's up+mix epilogue reproduces THAT tile's arithmetic, so the same
     * 84 cases run again with the switch thrown: the unfused chain now takes
     * the MMA at eight rows and up, and the fused path takes the epilogue.
     * The switch is one-way, which is why this pass is last. */
    ds4_gpu_enable_q8_dense_mma();
    check_q8_mma_pipe();
    check_mixer_equivalence(model, "MMA tile");
    check_mixer_pending(model, "MMA tile");

    munmap(model, MODEL_BYTES);
    printf("test_qwen4exp_hc_norm: ok\n");
    return 0;
}
