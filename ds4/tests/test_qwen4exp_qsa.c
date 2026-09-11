/*
 * Qwen4-Exp QSA block: GPU kernels against f32 CPU references.
 *
 * Production shape, from the MLX runner's `Qwen4ExpTextConfiguration`:
 * 24 query heads and 2 KV heads at head_dim 256, GQA 12; partial rope over the
 * leading 64 dims at base 1e7; an indexer of 4 heads at head_dim 128 with one
 * KV head, compress ratio 4 and a 2048-token budget, so 512 blocks.
 *
 * The run walks a single sequence through five segments so that both indexer
 * states are exercised:
 *
 *   pos    0 + 1024 tokens   kv 1024   dense, the budget still covers the tape
 *   pos 1024 + 1024 tokens   kv 2048   dense, exactly at the budget
 *   pos 2048 + 1024 tokens   kv 3072   sparse, 512 blocks selected
 *   pos 3072 +   64 tokens   kv 3136   sparse
 *   pos 3136 +    1 token    kv 3137   sparse, the decode shape
 *
 * Checks: the fused split is bit-exact, norm/rope/pool/gate are within 1e-5,
 * indexer scores within a small absolute band, the selected token id list is
 * SET-equal to the reference (see `compare_selection` for the tie rule),
 * attention is inside the design's 2e-3 / cosine 0.9999 band, and the whole
 * pipeline is bit-exact across two runs.
 */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ds4_gpu.h"

enum {
    N_HEAD = 24,
    N_KV_HEAD = 2,
    HEAD_DIM = 256,
    ROT_DIM = 64,
    IDX_HEAD = 4,
    IDX_HEAD_DIM = 128,
    POOL_SIZE = 4,
    TOKEN_BUDGET = 2048,
    BLOCK_TOP_K = TOKEN_BUDGET / POOL_SIZE,
    CACHE_CAP = 8192,
    MAX_BLOCKS = CACHE_CAP / POOL_SIZE,
    MAX_SELECTED = BLOCK_TOP_K * POOL_SIZE + POOL_SIZE,
    MAX_TOKENS = 1024,
    /* Tokens whose attention output is checked against the CPU reference.
     * The GPU computes every token; the reference is quadratic, so it runs on
     * a spread sample. */
    ATTN_SAMPLES = 6,
};

static const float RMS_EPS = 1e-6f;
static const float WEIGHT_OFFSET = 1.0f;
static const float ROPE_THETA = 10000000.0f;

bool ds4_log_is_tty(FILE *fp) {
    (void)fp;
    return false;
}

static void fail(const char *what) {
    fprintf(stderr, "test_qwen4exp_qsa: %s\n", what);
    exit(1);
}

static void require(int ok, const char *what) {
    if (!ok) fail(what);
}

static void *xcalloc(size_t count, size_t size) {
    void *p = calloc(count, size);
    if (!p) fail("out of host memory");
    return p;
}

/* ------------------------------------------------------------------ RNG */

static uint64_t g_rng_state;

static void rng_seed(uint64_t seed) {
    g_rng_state = seed;
}

static uint64_t rng_next(void) {
    uint64_t z = (g_rng_state += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

/* Uniform in [-1, 1), exactly representable steps so the two runs of the
 * determinism check see identical bytes. */
static float rng_uniform(void) {
    const uint32_t bits = (uint32_t)(rng_next() >> 40);   /* 24 bits */
    return (float)bits / 8388608.0f - 1.0f;
}

static void rng_fill(float *dst, size_t count) {
    for (size_t i = 0; i < count; i++) dst[i] = rng_uniform();
}

/* --------------------------------------------------------- CPU reference */

/* The inverse frequency table the GPU is given; both sides read it, so the
 * comparison measures the kernels and not two libm implementations. */
static float g_inv_freq[ROT_DIM / 2];

static void ref_split_qkv(float *q, float *gate, float *k, float *v,
                          const float *fused, uint32_t n_tokens) {
    const uint32_t q_width = N_HEAD * HEAD_DIM;
    const uint32_t kv_width = N_KV_HEAD * HEAD_DIM;
    const uint32_t stride = 2u * q_width + 2u * kv_width;
    for (uint32_t t = 0; t < n_tokens; t++) {
        const float *row = fused + (size_t)t * stride;
        for (uint32_t h = 0; h < N_HEAD; h++) {
            for (uint32_t d = 0; d < HEAD_DIM; d++) {
                const size_t dst = (size_t)t * q_width + h * HEAD_DIM + d;
                q[dst] = row[h * 2u * HEAD_DIM + d];
                gate[dst] = row[h * 2u * HEAD_DIM + HEAD_DIM + d];
            }
        }
        for (uint32_t i = 0; i < kv_width; i++) {
            k[(size_t)t * kv_width + i] = row[2u * q_width + i];
            v[(size_t)t * kv_width + i] = row[2u * q_width + kv_width + i];
        }
    }
}

static void ref_head_rms_norm(float *out, const float *x, const float *weight,
                              uint32_t n_rows, uint32_t head_dim,
                              float eps, float weight_offset) {
    for (uint32_t row = 0; row < n_rows; row++) {
        const float *src = x + (size_t)row * head_dim;
        float *dst = out + (size_t)row * head_dim;
        float sum = 0.0f;
        for (uint32_t d = 0; d < head_dim; d++) sum += src[d] * src[d];
        const float inv = 1.0f / sqrtf(sum / (float)head_dim + eps);
        for (uint32_t d = 0; d < head_dim; d++) {
            dst[d] = src[d] * inv * (weight_offset + weight[d]);
        }
    }
}

static void ref_rope_head_vec(float *vec, uint32_t rot_dim, uint32_t pos) {
    const uint32_t half = rot_dim / 2u;
    for (uint32_t d = 0; d < half; d++) {
        const float theta = (float)pos * g_inv_freq[d];
        const float c = cosf(theta);
        const float s = sinf(theta);
        const float x1 = vec[d];
        const float x2 = vec[d + half];
        vec[d] = x1 * c - x2 * s;
        vec[d + half] = x2 * c + x1 * s;
    }
}

static void ref_rope_head(float *x, uint32_t n_tokens, uint32_t n_head,
                          uint32_t head_dim, uint32_t rot_dim, uint32_t pos0) {
    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t h = 0; h < n_head; h++) {
            ref_rope_head_vec(x + ((size_t)t * n_head + h) * head_dim,
                              rot_dim, pos0 + t);
        }
    }
}

/* Pool blocks [block0, block1) out of the raw tape: fp32 mean over
 * POOL_SIZE rows, k_layernorm, then partial rope at POOL_SIZE * block. */
static void ref_pool_update(float *pool, const float *tape, const float *weight,
                            uint32_t block0, uint32_t block1) {
    for (uint32_t block = block0; block < block1; block++) {
        float *dst = pool + (size_t)block * IDX_HEAD_DIM;
        for (uint32_t d = 0; d < IDX_HEAD_DIM; d++) {
            float acc = 0.0f;
            for (uint32_t j = 0; j < POOL_SIZE; j++) {
                acc += tape[(size_t)(block * POOL_SIZE + j) * IDX_HEAD_DIM + d];
            }
            dst[d] = acc / (float)POOL_SIZE;
        }
        ref_head_rms_norm(dst, dst, weight, 1, IDX_HEAD_DIM, RMS_EPS, WEIGHT_OFFSET);
        ref_rope_head_vec(dst, ROT_DIM, block * POOL_SIZE);
    }
}

static void ref_indexer_scores(float *scores, const float *q, const float *pool,
                               uint32_t n_tokens, uint32_t n_blocks,
                               uint32_t pos0) {
    const float divisor = sqrtf((float)IDX_HEAD_DIM);
    for (uint32_t t = 0; t < n_tokens; t++) {
        uint32_t visible = (pos0 + t + 1u) / POOL_SIZE;
        if (visible > n_blocks) visible = n_blocks;
        for (uint32_t b = 0; b < n_blocks; b++) {
            if (b >= visible) {
                scores[(size_t)t * n_blocks + b] = DS4_QWEN4EXP_QSA_MASKED_SCORE;
                continue;
            }
            const float *k = pool + (size_t)b * IDX_HEAD_DIM;
            float total = 0.0f;
            for (uint32_t h = 0; h < IDX_HEAD; h++) {
                const float *qh = q + ((size_t)t * IDX_HEAD + h) * IDX_HEAD_DIM;
                float dot = 0.0f;
                for (uint32_t d = 0; d < IDX_HEAD_DIM; d++) dot += qh[d] * k[d];
                if (dot > 0.0f) total += dot;
            }
            scores[(size_t)t * n_blocks + b] = total / divisor;
        }
    }
}

/*
 * Reference block selection for one query.
 *
 * TIE RULE. MLX picks with `argPartition`, which gives a SET and no order;
 * ds4 picks with a descending bitonic argsort, whose order among equal scores
 * is whatever the sorting network produces. Neither is index order, so this
 * reference defines its own total order -- score descending, then block index
 * ascending -- and `compare_selection` treats a disagreement as acceptable
 * only when it sits inside the score gap at the budget boundary, which is the
 * design's indexer criterion.
 *
 * Writes the selected block ids ascending; returns how many there are.
 */
static uint32_t ref_select_blocks(uint32_t *blocks, const float *scores,
                                  uint32_t n_blocks, uint32_t top_k) {
    uint32_t *order = xcalloc(n_blocks, sizeof(uint32_t));
    uint32_t visible = 0;
    for (uint32_t b = 0; b < n_blocks; b++) {
        if (scores[b] > DS4_QWEN4EXP_QSA_MASKED_LIMIT) order[visible++] = b;
    }
    /* Insertion sort on (score desc, index asc); `visible` is at most the
     * block count of the segment under test. */
    for (uint32_t i = 1; i < visible; i++) {
        const uint32_t cur = order[i];
        uint32_t j = i;
        while (j > 0) {
            const uint32_t prev = order[j - 1];
            const bool swap = scores[prev] < scores[cur] ||
                (scores[prev] == scores[cur] && prev > cur);
            if (!swap) break;
            order[j] = prev;
            j--;
        }
        order[j] = cur;
    }
    const uint32_t kept = visible < top_k ? visible : top_k;
    for (uint32_t i = 0; i < kept; i++) blocks[i] = order[i];
    free(order);
    /* Ascending, matching the kernel's output order. */
    for (uint32_t i = 1; i < kept; i++) {
        const uint32_t cur = blocks[i];
        uint32_t j = i;
        while (j > 0 && blocks[j - 1] > cur) {
            blocks[j] = blocks[j - 1];
            j--;
        }
        blocks[j] = cur;
    }
    return kept;
}

/* Expand selected blocks into the ascending token list, then append the tail
 * of the query's own incomplete block ("keep OR own"). */
static uint32_t ref_expand_selection(int32_t *selected, const uint32_t *blocks,
                                     uint32_t n_selected_blocks, uint32_t pos) {
    uint32_t n = 0;
    for (uint32_t i = 0; i < n_selected_blocks; i++) {
        for (uint32_t r = 0; r < POOL_SIZE; r++) {
            selected[n++] = (int32_t)(blocks[i] * POOL_SIZE + r);
        }
    }
    const uint32_t own_start = ((pos + 1u) / POOL_SIZE) * POOL_SIZE;
    for (uint32_t token = own_start; token <= pos; token++) {
        selected[n++] = (int32_t)token;
    }
    return n;
}

static void ref_attention_row(float *out, const float *q, const float *k_cache,
                              const float *v_cache, const int32_t *selected,
                              uint32_t count, uint32_t token, uint32_t head) {
    const uint32_t kv_head = head / (N_HEAD / N_KV_HEAD);
    const uint32_t kv_stride = N_KV_HEAD * HEAD_DIM;
    const float scale = 1.0f / sqrtf((float)HEAD_DIM);
    const float *qv = q + ((size_t)token * N_HEAD + head) * HEAD_DIM;

    float *probs = xcalloc(count, sizeof(float));
    float max_score = -3.0e38f;
    for (uint32_t j = 0; j < count; j++) {
        const int32_t key = selected[j];
        float dot = 0.0f;
        const float *kv = k_cache + (size_t)key * kv_stride + kv_head * HEAD_DIM;
        for (uint32_t d = 0; d < HEAD_DIM; d++) dot += qv[d] * kv[d];
        probs[j] = dot * scale;
        if (probs[j] > max_score) max_score = probs[j];
    }
    float sum = 0.0f;
    for (uint32_t j = 0; j < count; j++) {
        probs[j] = expf(probs[j] - max_score);
        sum += probs[j];
    }
    for (uint32_t d = 0; d < HEAD_DIM; d++) out[d] = 0.0f;
    for (uint32_t j = 0; j < count; j++) {
        const float *vv = v_cache + (size_t)selected[j] * kv_stride + kv_head * HEAD_DIM;
        for (uint32_t d = 0; d < HEAD_DIM; d++) out[d] += probs[j] * vv[d];
    }
    for (uint32_t d = 0; d < HEAD_DIM; d++) out[d] /= sum;
    free(probs);
}

/* ------------------------------------------------------------- comparing */

static void compare_band(const char *what, const float *got, const float *want,
                         size_t count, float tolerance) {
    float worst = 0.0f;
    size_t worst_at = 0;
    for (size_t i = 0; i < count; i++) {
        const float diff = fabsf(got[i] - want[i]);
        if (!(diff <= worst)) {
            worst = diff;
            worst_at = i;
        }
    }
    if (!(worst <= tolerance)) {
        fprintf(stderr,
                "test_qwen4exp_qsa: %s max abs error %.6g at %zu (got %.9g, want %.9g), "
                "tolerance %.6g\n",
                what, (double)worst, worst_at, (double)got[worst_at],
                (double)want[worst_at], (double)tolerance);
        exit(1);
    }
    printf("  %-38s max abs error %.3g (tolerance %.3g)\n", what,
           (double)worst, (double)tolerance);
}

static double cosine(const float *a, const float *b, size_t count) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < count; i++) {
        dot += (double)a[i] * (double)b[i];
        na += (double)a[i] * (double)a[i];
        nb += (double)b[i] * (double)b[i];
    }
    if (na == 0.0 || nb == 0.0) return 1.0;
    return dot / (sqrt(na) * sqrt(nb));
}

/*
 * Selected-id SET equality with the design's indexer criterion: identical
 * lists, or at least 99.5% overlap with every disagreement inside the score
 * gap at the budget boundary. Both lists are ascending, so list equality and
 * set equality are the same test.
 */
static void compare_selection(uint32_t token, uint32_t pos,
                              const int32_t *got, uint32_t got_count,
                              const int32_t *want, uint32_t want_count,
                              const float *scores, uint32_t n_blocks,
                              uint32_t top_k, uint32_t *worst_shared,
                              uint32_t *worst_total, uint32_t *worst_ties) {
    if (got_count != want_count) {
        fprintf(stderr,
                "test_qwen4exp_qsa: token %u (pos %u) selected %u ids, reference %u\n",
                token, pos, got_count, want_count);
        exit(1);
    }
    for (uint32_t i = 1; i < got_count; i++) {
        if (got[i] <= got[i - 1]) {
            fprintf(stderr,
                    "test_qwen4exp_qsa: token %u selection is not ascending at %u\n",
                    token, i);
            exit(1);
        }
    }
    if (got_count > 0 && (uint32_t)got[got_count - 1] > pos) {
        fprintf(stderr,
                "test_qwen4exp_qsa: token %u (pos %u) selected future key %d\n",
                token, pos, got[got_count - 1]);
        exit(1);
    }

    uint32_t shared = 0;
    for (uint32_t i = 0, j = 0; i < got_count && j < want_count;) {
        if (got[i] == want[j]) { shared++; i++; j++; }
        else if (got[i] < want[j]) i++;
        else j++;
    }
    if (shared < *worst_shared) {
        *worst_shared = shared;
        *worst_total = got_count;
    }
    if (shared == got_count) return;

    /*
     * The lists disagree. That is only acceptable when the kernel still
     * returned A valid top-k: its weakest selected block must score no worse
     * than the reference's weakest, up to the score band. Exact ties are
     * common here -- a block scores exactly zero whenever relu kills every
     * index head -- and neither MLX's argPartition nor ds4's bitonic argsort
     * defines an order among them.
     *
     * The own-block tail is not block-aligned, so only the leading
     * `count - own` entries name blocks.
     */
    const uint32_t own = (pos + 1u) % POOL_SIZE;
    const float tolerance = 1e-4f;
    float boundary = 3.0e38f;
    for (uint32_t i = 0; i + own < want_count; i += POOL_SIZE) {
        const uint32_t block = (uint32_t)want[i] / POOL_SIZE;
        if (block < n_blocks && scores[block] < boundary) boundary = scores[block];
    }
    float weakest = 3.0e38f;
    uint32_t weakest_block = 0;
    for (uint32_t i = 0; i + own < got_count; i += POOL_SIZE) {
        const uint32_t block = (uint32_t)got[i] / POOL_SIZE;
        if (block < n_blocks && scores[block] < weakest) {
            weakest = scores[block];
            weakest_block = block;
        }
    }
    if (weakest < boundary - tolerance) {
        fprintf(stderr,
                "test_qwen4exp_qsa: token %u kept block %u at score %.9g, but the "
                "reference's %u-th best scores %.9g\n",
                token, weakest_block, (double)weakest, top_k, (double)boundary);
        exit(1);
    }

    /* Every disagreement has to be a tie at the boundary, so there cannot be
     * more differing blocks than there are blocks sitting on it. */
    uint32_t tied = 0;
    for (uint32_t b = 0; b < n_blocks; b++) {
        if (scores[b] > DS4_QWEN4EXP_QSA_MASKED_LIMIT &&
            fabsf(scores[b] - boundary) <= tolerance) {
            tied++;
        }
    }
    const uint32_t differing = (got_count - shared) / POOL_SIZE;
    if (differing > tied) {
        fprintf(stderr,
                "test_qwen4exp_qsa: token %u differs in %u blocks but only %u tie "
                "the boundary score %.9g\n",
                token, differing, tied, (double)boundary);
        exit(1);
    }
    if (tied > *worst_ties) *worst_ties = tied;
}

/* ------------------------------------------------------------ GPU helpers */

static ds4_gpu_tensor *tensor_new(size_t count, size_t elem) {
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc((uint64_t)count * elem);
    if (!t) fail("ds4_gpu_tensor_alloc returned NULL");
    return t;
}

static void tensor_put(ds4_gpu_tensor *t, const void *data, size_t bytes) {
    require(ds4_gpu_tensor_write(t, 0, data, (uint64_t)bytes), "tensor write");
}

static void tensor_get(const ds4_gpu_tensor *t, void *data, size_t bytes) {
    require(ds4_gpu_tensor_read(t, 0, data, (uint64_t)bytes), "tensor read");
}

/* ------------------------------------------------------------------ state */

typedef struct {
    uint32_t pos0;
    uint32_t n_tokens;
} segment;

static const segment SEGMENTS[] = {
    { 0,    1024 },
    { 1024, 1024 },
    { 2048, 1024 },
    { 3072, 64 },
    { 3136, 1 },
};
static const uint32_t N_SEGMENTS = (uint32_t)(sizeof(SEGMENTS) / sizeof(SEGMENTS[0]));

typedef struct {
    /* Weights, shared by both runs. */
    float *q_norm_w;
    float *k_norm_w;
    float *idx_q_norm_w;
    float *idx_k_norm_w;
    /* Per-segment fused projections and indexer projections. */
    float *fused[5];
    float *idx_q[5];
    float *idx_k[5];
} inputs;

static void inputs_build(inputs *in) {
    in->q_norm_w = xcalloc(HEAD_DIM, sizeof(float));
    in->k_norm_w = xcalloc(HEAD_DIM, sizeof(float));
    in->idx_q_norm_w = xcalloc(IDX_HEAD_DIM, sizeof(float));
    in->idx_k_norm_w = xcalloc(IDX_HEAD_DIM, sizeof(float));
    rng_seed(0x5115ec7edull);
    rng_fill(in->q_norm_w, HEAD_DIM);
    rng_fill(in->k_norm_w, HEAD_DIM);
    rng_fill(in->idx_q_norm_w, IDX_HEAD_DIM);
    rng_fill(in->idx_k_norm_w, IDX_HEAD_DIM);
    for (uint32_t s = 0; s < N_SEGMENTS; s++) {
        const uint32_t n = SEGMENTS[s].n_tokens;
        const size_t fused_len = (size_t)n * (2u * N_HEAD * HEAD_DIM + 2u * N_KV_HEAD * HEAD_DIM);
        in->fused[s] = xcalloc(fused_len, sizeof(float));
        in->idx_q[s] = xcalloc((size_t)n * IDX_HEAD * IDX_HEAD_DIM, sizeof(float));
        in->idx_k[s] = xcalloc((size_t)n * IDX_HEAD_DIM, sizeof(float));
        rng_fill(in->fused[s], fused_len);
        rng_fill(in->idx_q[s], (size_t)n * IDX_HEAD * IDX_HEAD_DIM);
        rng_fill(in->idx_k[s], (size_t)n * IDX_HEAD_DIM);
    }
}

static void inputs_free(inputs *in) {
    free(in->q_norm_w);
    free(in->k_norm_w);
    free(in->idx_q_norm_w);
    free(in->idx_k_norm_w);
    for (uint32_t s = 0; s < N_SEGMENTS; s++) {
        free(in->fused[s]);
        free(in->idx_q[s]);
        free(in->idx_k[s]);
    }
}

/*
 * Run the whole schedule on the GPU. `verify` compares against the CPU
 * reference; the second pass runs with it off and only the outputs are
 * compared, which is the determinism check.
 *
 * `attn_out` receives the attention output of every segment, concatenated.
 */
static void run_pipeline(const inputs *in, bool verify, float *attn_out) {
    const uint32_t q_width = N_HEAD * HEAD_DIM;
    const uint32_t kv_width = N_KV_HEAD * HEAD_DIM;

    ds4_gpu_tensor *t_fused = tensor_new((size_t)MAX_TOKENS * (2u * q_width + 2u * kv_width), sizeof(float));
    ds4_gpu_tensor *t_q = tensor_new((size_t)MAX_TOKENS * q_width, sizeof(float));
    ds4_gpu_tensor *t_gate = tensor_new((size_t)MAX_TOKENS * q_width, sizeof(float));
    ds4_gpu_tensor *t_k = tensor_new((size_t)MAX_TOKENS * kv_width, sizeof(float));
    ds4_gpu_tensor *t_v = tensor_new((size_t)MAX_TOKENS * kv_width, sizeof(float));
    ds4_gpu_tensor *t_out = tensor_new((size_t)MAX_TOKENS * q_width, sizeof(float));
    ds4_gpu_tensor *t_kcache = tensor_new((size_t)CACHE_CAP * kv_width, sizeof(float));
    ds4_gpu_tensor *t_vcache = tensor_new((size_t)CACHE_CAP * kv_width, sizeof(float));
    ds4_gpu_tensor *t_qnorm = tensor_new(HEAD_DIM, sizeof(float));
    ds4_gpu_tensor *t_knorm = tensor_new(HEAD_DIM, sizeof(float));
    ds4_gpu_tensor *t_iqnorm = tensor_new(IDX_HEAD_DIM, sizeof(float));
    ds4_gpu_tensor *t_iknorm = tensor_new(IDX_HEAD_DIM, sizeof(float));
    ds4_gpu_tensor *t_idx_q = tensor_new((size_t)MAX_TOKENS * IDX_HEAD * IDX_HEAD_DIM, sizeof(float));
    ds4_gpu_tensor *t_idx_k = tensor_new((size_t)MAX_TOKENS * IDX_HEAD_DIM, sizeof(float));
    ds4_gpu_tensor *t_tape = tensor_new((size_t)CACHE_CAP * IDX_HEAD_DIM, sizeof(float));
    ds4_gpu_tensor *t_pool = tensor_new((size_t)MAX_BLOCKS * IDX_HEAD_DIM, sizeof(float));
    ds4_gpu_tensor *t_scores = tensor_new((size_t)MAX_TOKENS * MAX_BLOCKS, sizeof(float));
    ds4_gpu_tensor *t_topk = tensor_new((size_t)MAX_TOKENS * BLOCK_TOP_K, sizeof(int32_t));
    ds4_gpu_tensor *t_selected = tensor_new((size_t)MAX_TOKENS * MAX_SELECTED, sizeof(int32_t));
    ds4_gpu_tensor *t_counts = tensor_new(MAX_TOKENS, sizeof(int32_t));
    ds4_gpu_tensor *t_inv_freq = tensor_new(ROT_DIM / 2, sizeof(float));
    tensor_put(t_inv_freq, g_inv_freq, sizeof(g_inv_freq));

    tensor_put(t_qnorm, in->q_norm_w, HEAD_DIM * sizeof(float));
    tensor_put(t_knorm, in->k_norm_w, HEAD_DIM * sizeof(float));
    tensor_put(t_iqnorm, in->idx_q_norm_w, IDX_HEAD_DIM * sizeof(float));
    tensor_put(t_iknorm, in->idx_k_norm_w, IDX_HEAD_DIM * sizeof(float));

    /* Host mirrors of the caches, so the reference sees what the GPU sees. */
    float *host_kcache = xcalloc((size_t)CACHE_CAP * kv_width, sizeof(float));
    float *host_vcache = xcalloc((size_t)CACHE_CAP * kv_width, sizeof(float));
    float *host_tape = xcalloc((size_t)CACHE_CAP * IDX_HEAD_DIM, sizeof(float));
    float *host_pool = xcalloc((size_t)MAX_BLOCKS * IDX_HEAD_DIM, sizeof(float));

    float *ref_q = xcalloc((size_t)MAX_TOKENS * q_width, sizeof(float));
    float *ref_gate = xcalloc((size_t)MAX_TOKENS * q_width, sizeof(float));
    float *ref_k = xcalloc((size_t)MAX_TOKENS * kv_width, sizeof(float));
    float *ref_v = xcalloc((size_t)MAX_TOKENS * kv_width, sizeof(float));
    float *ref_idx_q = xcalloc((size_t)MAX_TOKENS * IDX_HEAD * IDX_HEAD_DIM, sizeof(float));
    float *got = xcalloc((size_t)MAX_TOKENS * q_width, sizeof(float));
    float *ref_scores = NULL;
    float *got_scores = NULL;
    int32_t *got_selected = xcalloc((size_t)MAX_TOKENS * MAX_SELECTED, sizeof(int32_t));
    int32_t *got_counts = xcalloc(MAX_TOKENS, sizeof(int32_t));
    int32_t *ref_selected = xcalloc(MAX_SELECTED, sizeof(int32_t));
    uint32_t *ref_blocks = xcalloc(MAX_BLOCKS, sizeof(uint32_t));
    float *ref_row = xcalloc(HEAD_DIM, sizeof(float));
    float *kv_stage = xcalloc((size_t)MAX_TOKENS * kv_width, sizeof(float));
    size_t attn_written = 0;

    for (uint32_t s = 0; s < N_SEGMENTS; s++) {
        const uint32_t pos0 = SEGMENTS[s].pos0;
        const uint32_t n = SEGMENTS[s].n_tokens;
        const uint32_t kv_length = pos0 + n;
        const uint32_t n_blocks = kv_length / POOL_SIZE;
        const bool sparse = kv_length > TOKEN_BUDGET;
        const uint32_t top_k = sparse
            ? (n_blocks < (uint32_t)BLOCK_TOP_K ? n_blocks : (uint32_t)BLOCK_TOP_K)
            : 0u;

        /* --- the attention half --------------------------------------- */
        const size_t fused_len = (size_t)n * (2u * q_width + 2u * kv_width);
        tensor_put(t_fused, in->fused[s], fused_len * sizeof(float));
        require(ds4_gpu_qwen4exp_qsa_split_qkv_tensor(t_q, t_gate, t_k, t_v, t_fused,
                                                     n, N_HEAD, N_KV_HEAD, HEAD_DIM),
                "split_qkv");
        if (verify) {
            ref_split_qkv(ref_q, ref_gate, ref_k, ref_v, in->fused[s], n);
            tensor_get(t_q, got, (size_t)n * q_width * sizeof(float));
            compare_band("split_qkv q", got, ref_q, (size_t)n * q_width, 0.0f);
            tensor_get(t_gate, got, (size_t)n * q_width * sizeof(float));
            compare_band("split_qkv gate", got, ref_gate, (size_t)n * q_width, 0.0f);
            tensor_get(t_k, got, (size_t)n * kv_width * sizeof(float));
            compare_band("split_qkv k", got, ref_k, (size_t)n * kv_width, 0.0f);
            tensor_get(t_v, got, (size_t)n * kv_width * sizeof(float));
            compare_band("split_qkv v", got, ref_v, (size_t)n * kv_width, 0.0f);
        } else {
            ref_split_qkv(ref_q, ref_gate, ref_k, ref_v, in->fused[s], n);
        }

        require(ds4_gpu_qwen4exp_head_rms_norm_tensor(t_q, t_q, t_qnorm,
                                                     n * N_HEAD, HEAD_DIM,
                                                     RMS_EPS, WEIGHT_OFFSET),
                "q norm");
        require(ds4_gpu_qwen4exp_head_rms_norm_tensor(t_k, t_k, t_knorm,
                                                     n * N_KV_HEAD, HEAD_DIM,
                                                     RMS_EPS, WEIGHT_OFFSET),
                "k norm");
        ref_head_rms_norm(ref_q, ref_q, in->q_norm_w, n * N_HEAD, HEAD_DIM,
                          RMS_EPS, WEIGHT_OFFSET);
        ref_head_rms_norm(ref_k, ref_k, in->k_norm_w, n * N_KV_HEAD, HEAD_DIM,
                          RMS_EPS, WEIGHT_OFFSET);
        if (verify) {
            tensor_get(t_q, got, (size_t)n * q_width * sizeof(float));
            compare_band("q head RMS norm", got, ref_q, (size_t)n * q_width, 1e-5f);
        }

        require(ds4_gpu_qwen4exp_rope_head_tensor(t_q, t_inv_freq, n, N_HEAD,
                                                 HEAD_DIM, ROT_DIM, pos0),
                "q rope");
        require(ds4_gpu_qwen4exp_rope_head_tensor(t_k, t_inv_freq, n, N_KV_HEAD,
                                                 HEAD_DIM, ROT_DIM, pos0),
                "k rope");
        ref_rope_head(ref_q, n, N_HEAD, HEAD_DIM, ROT_DIM, pos0);
        ref_rope_head(ref_k, n, N_KV_HEAD, HEAD_DIM, ROT_DIM, pos0);
        if (verify) {
            tensor_get(t_q, got, (size_t)n * q_width * sizeof(float));
            compare_band("q partial rope", got, ref_q, (size_t)n * q_width, 1e-5f);
        }

        /* Append the KV cache rows.  ds4_gpu_tensor_copy needs an open batch
         * command buffer, which this test does not run, so the rows go through
         * the host -- what lands in the cache is still the GPU's own output. */
        tensor_get(t_k, kv_stage, (size_t)n * kv_width * sizeof(float));
        require(ds4_gpu_tensor_write(t_kcache,
                                     (uint64_t)pos0 * kv_width * sizeof(float),
                                     kv_stage, (uint64_t)n * kv_width * sizeof(float)),
                "k cache append");
        tensor_get(t_v, kv_stage, (size_t)n * kv_width * sizeof(float));
        require(ds4_gpu_tensor_write(t_vcache,
                                     (uint64_t)pos0 * kv_width * sizeof(float),
                                     kv_stage, (uint64_t)n * kv_width * sizeof(float)),
                "v cache append");
        memcpy(host_kcache + (size_t)pos0 * kv_width, ref_k,
               (size_t)n * kv_width * sizeof(float));
        memcpy(host_vcache + (size_t)pos0 * kv_width, ref_v,
               (size_t)n * kv_width * sizeof(float));

        /* --- the indexer ---------------------------------------------- */
        tensor_put(t_idx_q, in->idx_q[s],
                   (size_t)n * IDX_HEAD * IDX_HEAD_DIM * sizeof(float));
        tensor_put(t_idx_k, in->idx_k[s], (size_t)n * IDX_HEAD_DIM * sizeof(float));

        require(ds4_gpu_qwen4exp_qsa_indexer_pool_update_tensor(
                    t_pool, t_tape, t_idx_k, t_iknorm, t_inv_freq, pos0, n,
                    CACHE_CAP, IDX_HEAD_DIM, POOL_SIZE, ROT_DIM, RMS_EPS,
                    WEIGHT_OFFSET),
                "indexer pool update");
        memcpy(host_tape + (size_t)pos0 * IDX_HEAD_DIM, in->idx_k[s],
               (size_t)n * IDX_HEAD_DIM * sizeof(float));
        ref_pool_update(host_pool, host_tape, in->idx_k_norm_w,
                        pos0 / POOL_SIZE, (pos0 + n) / POOL_SIZE);
        if (verify) {
            float *pool_got = xcalloc((size_t)n_blocks * IDX_HEAD_DIM, sizeof(float));
            tensor_get(t_pool, pool_got, (size_t)n_blocks * IDX_HEAD_DIM * sizeof(float));
            compare_band("indexer pooled blocks", pool_got, host_pool,
                         (size_t)n_blocks * IDX_HEAD_DIM, 1e-5f);
            free(pool_got);
        }

        require(ds4_gpu_qwen4exp_head_rms_norm_tensor(t_idx_q, t_idx_q, t_iqnorm,
                                                     n * IDX_HEAD, IDX_HEAD_DIM,
                                                     RMS_EPS, WEIGHT_OFFSET),
                "indexer q norm");
        require(ds4_gpu_qwen4exp_rope_head_tensor(t_idx_q, t_inv_freq, n, IDX_HEAD,
                                                 IDX_HEAD_DIM, ROT_DIM, pos0),
                "indexer q rope");
        memcpy(ref_idx_q, in->idx_q[s],
               (size_t)n * IDX_HEAD * IDX_HEAD_DIM * sizeof(float));
        ref_head_rms_norm(ref_idx_q, ref_idx_q, in->idx_q_norm_w, n * IDX_HEAD,
                          IDX_HEAD_DIM, RMS_EPS, WEIGHT_OFFSET);
        ref_rope_head(ref_idx_q, n, IDX_HEAD, IDX_HEAD_DIM, ROT_DIM, pos0);

        if (sparse) {
            require(ds4_gpu_qwen4exp_qsa_indexer_scores_tensor(
                        t_scores, t_idx_q, t_pool, n, n_blocks, IDX_HEAD,
                        IDX_HEAD_DIM, pos0, POOL_SIZE),
                    "indexer scores");
            require(ds4_gpu_indexer_topk_tensor(t_topk, t_scores, n_blocks, n, top_k),
                    "indexer top-k");
            require(ds4_gpu_qwen4exp_qsa_indexer_select_tensor(
                        t_selected, t_counts, t_scores, t_topk, n, n_blocks,
                        top_k, pos0, POOL_SIZE, MAX_SELECTED),
                    "indexer select");
            tensor_get(t_selected, got_selected,
                       (size_t)n * MAX_SELECTED * sizeof(int32_t));
            tensor_get(t_counts, got_counts, (size_t)n * sizeof(int32_t));

            if (verify) {
                uint32_t worst_shared = UINT32_MAX;
                uint32_t worst_total = 1u;
                uint32_t worst_ties = 0;
                ref_scores = xcalloc((size_t)n * n_blocks, sizeof(float));
                got_scores = xcalloc((size_t)n * n_blocks, sizeof(float));
                ref_indexer_scores(ref_scores, ref_idx_q, host_pool, n, n_blocks, pos0);
                tensor_get(t_scores, got_scores, (size_t)n * n_blocks * sizeof(float));
                /* Compare only the visible entries; masked ones are the
                 * sentinel on both sides and would swamp the band. */
                float worst = 0.0f;
                for (size_t i = 0; i < (size_t)n * n_blocks; i++) {
                    if (ref_scores[i] <= DS4_QWEN4EXP_QSA_MASKED_LIMIT) {
                        require(got_scores[i] <= DS4_QWEN4EXP_QSA_MASKED_LIMIT,
                                "indexer masked a block the GPU left visible");
                        continue;
                    }
                    const float diff = fabsf(got_scores[i] - ref_scores[i]);
                    if (!(diff <= worst)) worst = diff;
                }
                printf("  %-38s max abs error %.3g (tolerance %.3g)\n",
                       "indexer block scores", (double)worst, 1e-4);
                require(worst <= 1e-4f, "indexer block scores outside the band");

                for (uint32_t t = 0; t < n; t++) {
                    const uint32_t kept = ref_select_blocks(
                        ref_blocks, ref_scores + (size_t)t * n_blocks, n_blocks, top_k);
                    const uint32_t want = ref_expand_selection(ref_selected, ref_blocks,
                                                               kept, pos0 + t);
                    compare_selection(t, pos0 + t,
                                      got_selected + (size_t)t * MAX_SELECTED,
                                      (uint32_t)got_counts[t], ref_selected, want,
                                      ref_scores + (size_t)t * n_blocks, n_blocks,
                                      top_k, &worst_shared, &worst_total,
                                      &worst_ties);
                }
                printf("  %-38s %u tokens, worst id overlap %.2f%% (%u of %u), "
                       "up to %u blocks tie the budget boundary\n",
                       "indexer selection", n,
                       100.0 * (double)worst_shared / (double)worst_total,
                       worst_shared, worst_total, worst_ties);
                free(ref_scores);
                free(got_scores);
                ref_scores = NULL;
                got_scores = NULL;
            }
        }

        /* --- attention ------------------------------------------------ */
        require(ds4_gpu_qwen4exp_qsa_attention_tensor(
                    t_out, t_q, t_kcache, t_vcache,
                    sparse ? t_selected : NULL, sparse ? t_counts : NULL,
                    n, N_HEAD, N_KV_HEAD, HEAD_DIM, pos0, CACHE_CAP,
                    MAX_SELECTED, 1.0f / sqrtf((float)HEAD_DIM)),
                "qsa attention");
        require(ds4_gpu_qwen4exp_qsa_output_gate_tensor(t_out, t_gate, n * q_width),
                "qsa output gate");
        tensor_get(t_out, got, (size_t)n * q_width * sizeof(float));
        memcpy(attn_out + attn_written, got, (size_t)n * q_width * sizeof(float));
        attn_written += (size_t)n * q_width;

        if (verify) {
            const uint32_t heads[3] = { 0, N_HEAD / 2, N_HEAD - 1 };
            double worst_cos = 1.0;
            float worst_abs = 0.0f;
            for (uint32_t si = 0; si < ATTN_SAMPLES; si++) {
                const uint32_t t = (n == 1) ? 0 : (uint32_t)((uint64_t)si * (n - 1) / (ATTN_SAMPLES - 1));
                uint32_t count;
                const int32_t *keys;
                int32_t *dense = NULL;
                if (sparse) {
                    count = (uint32_t)got_counts[t];
                    keys = got_selected + (size_t)t * MAX_SELECTED;
                } else {
                    count = pos0 + t + 1u;
                    dense = xcalloc(count, sizeof(int32_t));
                    for (uint32_t i = 0; i < count; i++) dense[i] = (int32_t)i;
                    keys = dense;
                }
                for (uint32_t hi = 0; hi < 3; hi++) {
                    const uint32_t h = heads[hi];
                    ref_attention_row(ref_row, ref_q, host_kcache, host_vcache,
                                      keys, count, t, h);
                    const float *gate_row = ref_gate + ((size_t)t * N_HEAD + h) * HEAD_DIM;
                    for (uint32_t d = 0; d < HEAD_DIM; d++) {
                        ref_row[d] *= 1.0f / (1.0f + expf(-gate_row[d]));
                    }
                    const float *got_row = got + ((size_t)t * N_HEAD + h) * HEAD_DIM;
                    for (uint32_t d = 0; d < HEAD_DIM; d++) {
                        const float diff = fabsf(got_row[d] - ref_row[d]);
                        if (!(diff <= worst_abs)) worst_abs = diff;
                    }
                    const double c = cosine(got_row, ref_row, HEAD_DIM);
                    if (c < worst_cos) worst_cos = c;
                }
                free(dense);
            }
            printf("  %-38s max abs error %.3g, cosine %.7f\n",
                   sparse ? "qsa attention (sparse) + gate"
                          : "qsa attention (dense) + gate",
                   (double)worst_abs, worst_cos);
            require(worst_abs <= 2e-3f, "attention outside the 2e-3 band");
            require(worst_cos >= 0.9999, "attention cosine below 0.9999");
        }
    }

    free(host_kcache);
    free(host_vcache);
    free(host_tape);
    free(host_pool);
    free(ref_q);
    free(ref_gate);
    free(ref_k);
    free(ref_v);
    free(ref_idx_q);
    free(got);
    free(got_selected);
    free(got_counts);
    free(ref_selected);
    free(ref_blocks);
    free(ref_row);
    free(kv_stage);

    ds4_gpu_tensor_free(t_fused);
    ds4_gpu_tensor_free(t_q);
    ds4_gpu_tensor_free(t_gate);
    ds4_gpu_tensor_free(t_k);
    ds4_gpu_tensor_free(t_v);
    ds4_gpu_tensor_free(t_out);
    ds4_gpu_tensor_free(t_kcache);
    ds4_gpu_tensor_free(t_vcache);
    ds4_gpu_tensor_free(t_qnorm);
    ds4_gpu_tensor_free(t_knorm);
    ds4_gpu_tensor_free(t_iqnorm);
    ds4_gpu_tensor_free(t_iknorm);
    ds4_gpu_tensor_free(t_idx_q);
    ds4_gpu_tensor_free(t_idx_k);
    ds4_gpu_tensor_free(t_tape);
    ds4_gpu_tensor_free(t_pool);
    ds4_gpu_tensor_free(t_scores);
    ds4_gpu_tensor_free(t_topk);
    ds4_gpu_tensor_free(t_selected);
    ds4_gpu_tensor_free(t_counts);
    ds4_gpu_tensor_free(t_inv_freq);
}

/* ------------------------------------------------ split attention path */

/* The decode-width split path against the per-head kernel, byte for byte.
 *
 * The split path (ds4_cuda_qwen4exp.cu, qwen4exp_qsa_split_*) computes the
 * per-head kernel's recurrence over a (head group, tile) grid and folds the
 * tiles at the end.  Its note argues that every expf sees the same argument
 * and every fused multiply-add nests the same way; this is the check that
 * the argument holds on the device.  The same random cache serves every
 * width 1..7 at positions on both sides of every tile edge, up to the
 * indexer budget, in the dense form and in a sparse form with a masked tail
 * and holes; half the calls take the position through `d_pos`, the form the
 * captured decode graph uses. */
static void check_split_path(void) {
    const uint32_t kv_width = N_KV_HEAD * HEAD_DIM;
    const uint32_t q_width = N_HEAD * HEAD_DIM;
    const uint32_t cap = TOKEN_BUDGET + 64u;
    const uint32_t max_count = MAX_SELECTED > TOKEN_BUDGET ? MAX_SELECTED : TOKEN_BUDGET;
    const float scale = 1.0f / sqrtf((float)HEAD_DIM);
    const size_t rows = (size_t)7 * q_width;

    float *host = xcalloc((size_t)cap * kv_width, sizeof(float));
    rng_seed(0x5157u);
    ds4_gpu_tensor *t_k = tensor_new((size_t)cap * kv_width, sizeof(float));
    rng_fill(host, (size_t)cap * kv_width);
    tensor_put(t_k, host, (size_t)cap * kv_width * sizeof(float));
    ds4_gpu_tensor *t_v = tensor_new((size_t)cap * kv_width, sizeof(float));
    rng_fill(host, (size_t)cap * kv_width);
    tensor_put(t_v, host, (size_t)cap * kv_width * sizeof(float));
    free(host);
    float *hq = xcalloc(rows, sizeof(float));
    rng_fill(hq, rows);
    ds4_gpu_tensor *t_q = tensor_new(rows, sizeof(float));
    tensor_put(t_q, hq, rows * sizeof(float));
    free(hq);
    ds4_gpu_tensor *t_a = tensor_new(rows, sizeof(float));
    ds4_gpu_tensor *t_b = tensor_new(rows, sizeof(float));
    ds4_gpu_tensor *t_dpos = tensor_new(1, sizeof(uint32_t));
    const uint64_t scratch_bytes = ds4_gpu_qwen4exp_qsa_split_scratch_bytes(
        7u, N_HEAD, HEAD_DIM, max_count);
    require(scratch_bytes > 0u, "split scratch size");
    ds4_gpu_tensor *t_scratch = ds4_gpu_tensor_alloc(scratch_bytes);
    require(t_scratch != NULL, "split scratch alloc");
    int32_t *hsel = xcalloc((size_t)7 * MAX_SELECTED, sizeof(int32_t));
    int32_t hcnt[7];
    ds4_gpu_tensor *t_sel = tensor_new((size_t)7 * MAX_SELECTED, sizeof(int32_t));
    ds4_gpu_tensor *t_cnt = tensor_new(7, sizeof(int32_t));
    float *got_a = xcalloc(rows, sizeof(float));
    float *got_b = xcalloc(rows, sizeof(float));

    static const uint32_t positions[] = {
        0u, 1u, 200u, 254u, 255u, 256u, 257u, 511u, 512u, 513u, 1000u, 1145u,
        1152u, 1535u, 1600u, 2000u, 2040u, 2041u
    };
    size_t checked = 0;
    size_t calls = 0;
    for (uint32_t form = 0; form < 2u; form++) {
        const bool sparse = form == 1u;
        for (uint32_t w = 1; w <= 7u; w++) {
            for (size_t pi = 0; pi < sizeof(positions) / sizeof(positions[0]); pi++) {
                const uint32_t pos0 = positions[pi];
                if ((uint64_t)pos0 + w > TOKEN_BUDGET) continue;
                const bool via_dpos = ((pi + w) & 1u) != 0u;
                if (via_dpos) tensor_put(t_dpos, &pos0, sizeof(pos0));
                if (sparse) {
                    /* Per row: a count near pos + 1 with a few keys dropped
                     * (never the first, so every row has a key to score),
                     * one out-of-cache key and a -1 hole, ascending as the
                     * selection kernel emits them, then a -1 tail. */
                    for (uint32_t t = 0; t < w; t++) {
                        int32_t *sel = hsel + (size_t)t * MAX_SELECTED;
                        uint32_t n = 0;
                        for (uint32_t k = 0; k <= pos0 + t && n < (uint32_t)MAX_SELECTED; k++) {
                            if (k > 0u && (rng_next() & 15u) == 0u) continue;
                            sel[n++] = (int32_t)k;
                        }
                        if (n > 3u) sel[n / 2u] = -1;
                        if (n > 5u) sel[n / 3u] = (int32_t)cap + 7;
                        for (uint32_t k = n; k < (uint32_t)MAX_SELECTED; k++) sel[k] = -1;
                        hcnt[t] = (int32_t)n;
                    }
                    tensor_put(t_sel, hsel, (size_t)w * MAX_SELECTED * sizeof(int32_t));
                    tensor_put(t_cnt, hcnt, (size_t)w * sizeof(int32_t));
                }
                require(ds4_gpu_qwen4exp_qsa_attention_dpos_tensor(
                            t_a, t_q, t_k, t_v, sparse ? t_sel : NULL,
                            sparse ? t_cnt : NULL, w, N_HEAD, N_KV_HEAD,
                            HEAD_DIM, pos0, cap, MAX_SELECTED, scale,
                            via_dpos ? t_dpos : NULL, NULL, 0u),
                        "per-head attention");
                /* The split path is what is under test, so the call must
                 * not have quietly fallen through to the per-head kernel:
                 * the scratch's first tile row of scores is zeroed before
                 * and must be written after. */
                memset(got_b, 0, HEAD_DIM * sizeof(float));
                require(ds4_gpu_tensor_write(t_scratch, 0, got_b, HEAD_DIM * sizeof(float)),
                        "scratch clear");
                require(ds4_gpu_qwen4exp_qsa_attention_dpos_tensor(
                            t_b, t_q, t_k, t_v, sparse ? t_sel : NULL,
                            sparse ? t_cnt : NULL, w, N_HEAD, N_KV_HEAD,
                            HEAD_DIM, pos0, cap, MAX_SELECTED, scale,
                            via_dpos ? t_dpos : NULL, t_scratch, max_count),
                        "split attention");
                const size_t n = (size_t)w * q_width;
                tensor_get(t_scratch, got_b, HEAD_DIM * sizeof(float));
                {
                    float touched = 0.0f;
                    for (uint32_t d = 0; d < HEAD_DIM; d++) touched += fabsf(got_b[d]);
                    require(touched > 0.0f, "split path did not run (scratch untouched)");
                }
                tensor_get(t_a, got_a, n * sizeof(float));
                tensor_get(t_b, got_b, n * sizeof(float));
                if (memcmp(got_a, got_b, n * sizeof(float)) != 0) {
                    size_t at = 0;
                    size_t differing = 0;
                    for (size_t i = 0; i < n; i++) {
                        if (got_a[i] != got_b[i]) {
                            if (differing == 0) at = i;
                            differing++;
                        }
                    }
                    fprintf(stderr,
                            "test_qwen4exp_qsa: split attention is not bit-exact "
                            "(%s, width %u, pos %u%s): %zu of %zu values differ, "
                            "first at %zu (head %zu, channel %zu): %.9g vs %.9g\n",
                            sparse ? "sparse" : "dense", w, pos0,
                            via_dpos ? ", d_pos" : "", differing, n, at,
                            (at / HEAD_DIM) % N_HEAD, at % HEAD_DIM,
                            (double)got_a[at], (double)got_b[at]);
                    exit(1);
                }
                /* A zero row would agree trivially; the cache is random, so
                 * it means a kernel did not run. */
                float any = 0.0f;
                for (size_t i = 0; i < n; i++) any += fabsf(got_b[i]);
                require(any > 0.0f, "split attention produced an all-zero row");
                checked += n;
                calls++;
            }
        }
    }
    printf("  %-38s %zu values bit-exact against the per-head kernel over %zu calls\n",
           "split attention (widths 1..7)", checked, calls);

    free(got_a);
    free(got_b);
    free(hsel);
    ds4_gpu_tensor_free(t_k);
    ds4_gpu_tensor_free(t_v);
    ds4_gpu_tensor_free(t_q);
    ds4_gpu_tensor_free(t_a);
    ds4_gpu_tensor_free(t_b);
    ds4_gpu_tensor_free(t_dpos);
    ds4_gpu_tensor_free(t_scratch);
    ds4_gpu_tensor_free(t_sel);
    ds4_gpu_tensor_free(t_cnt);
}

int main(void) {
    if (!ds4_gpu_init()) {
        printf("test_qwen4exp_qsa: no GPU backend, skipping\n");
        return 0;
    }

    ds4_gpu_qwen4exp_rope_inv_freq(g_inv_freq, ROT_DIM, ROPE_THETA);

    inputs in;
    memset(&in, 0, sizeof(in));
    inputs_build(&in);

    size_t attn_len = 0;
    for (uint32_t s = 0; s < N_SEGMENTS; s++) {
        attn_len += (size_t)SEGMENTS[s].n_tokens * N_HEAD * HEAD_DIM;
    }
    float *first = xcalloc(attn_len, sizeof(float));
    float *second = xcalloc(attn_len, sizeof(float));

    printf("test_qwen4exp_qsa: 24x2 heads @%d, indexer %dx%d ratio %d budget %d\n",
           HEAD_DIM, IDX_HEAD, IDX_HEAD_DIM, POOL_SIZE, TOKEN_BUDGET);
    run_pipeline(&in, true, first);
    run_pipeline(&in, false, second);

    if (memcmp(first, second, attn_len * sizeof(float)) != 0) {
        for (size_t i = 0; i < attn_len; i++) {
            if (first[i] != second[i]) {
                fprintf(stderr,
                        "test_qwen4exp_qsa: not deterministic at %zu: %.9g vs %.9g\n",
                        i, (double)first[i], (double)second[i]);
                break;
            }
        }
        return 1;
    }
    printf("  %-38s %zu values bit-exact across two runs\n", "determinism", attn_len);

    /* The same pipeline again with the head-group attention kernel switched
     * off, so the two kernels stand side by side in one process.  They read
     * the same K and V rows out of different places -- the group kernel reads
     * each row once for all twelve heads of its KV head, the per-head kernel
     * once per head -- but every product they add, they add to the same
     * running sum in the same position of the same sequence.  So this is not
     * a tolerance band: the bytes must be equal.  A single differing float
     * here is the whole reason to leave the group kernel switched off.
     *
     * The segment list covers both sides of the width gate: the 1024-row and
     * 64-row segments take the group kernel, the 1-row one does not. */
    float *ungrouped = xcalloc(attn_len, sizeof(float));
    setenv("DS4_QWEN4EXP_NO_QSA_GROUP", "1", 1);
    run_pipeline(&in, false, ungrouped);
    unsetenv("DS4_QWEN4EXP_NO_QSA_GROUP");

    if (memcmp(first, ungrouped, attn_len * sizeof(float)) != 0) {
        size_t differing = 0;
        size_t at = 0;
        for (size_t i = 0; i < attn_len; i++) {
            if (first[i] != ungrouped[i]) {
                if (differing == 0) at = i;
                differing++;
            }
        }
        fprintf(stderr,
                "test_qwen4exp_qsa: head-group attention is not bit-exact: "
                "%zu of %zu values differ, first at %zu: %.9g vs %.9g\n",
                differing, attn_len, at, (double)first[at],
                (double)ungrouped[at]);
        return 1;
    }
    printf("  %-38s %zu values bit-exact against the per-head kernel\n",
           "head-group attention", attn_len);

    check_split_path();

    free(ungrouped);
    free(first);
    free(second);
    inputs_free(&in);
    ds4_gpu_cleanup();
    printf("test_qwen4exp_qsa: OK\n");
    return 0;
}
