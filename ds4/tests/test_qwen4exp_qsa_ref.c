/*
 * Qwen4-Exp dense QSA against a plain double-precision reference.
 *
 * tests/test_qwen4exp_qsa.c compares the kernels against an f32 reference that
 * is HANDED the GPU's own inverse-frequency table, so the rope base and the
 * table derivation are never checked, and its attention case is a spread
 * sample of six tokens inside long segments.  This file re-states the whole
 * dense path in double from the model semantics, derives its own inverse
 * frequencies from the rope base, and walks the lengths a collapsing prompt
 * passes through: 1, 6, 8, 16, 33, 64, 65, 128, 200.
 *
 * The path, exactly as ds4_qwen4exp_graph.inc runs it:
 *
 *   split the DOUBLED attn_q row              [ (q_h | gate_h) x n_head ]
 *   q = rms(q) over head_dim, eps on the MEAN, weight (offset + w)
 *   k = rms(k) the same way; v untouched
 *   rope q and k over the LEADING rot_dim entries, half-split NeoX,
 *     position pos0 + t, inv_freq[j] = base^(-2j/rot_dim)
 *   append k and v to the caches at row pos0
 *   softmax((q . k) / sqrt(head_dim)) over keys [0, pos0 + t], GQA head
 *     mapping head / (n_head / n_kv_head)
 *   out *= sigmoid(gate)
 *
 * Cases: a fresh prefill at each length from position 0, then two
 * cached-key cases -- prefill 33 then one decode row, prefill 64 then three
 * rows -- where the second call's rope positions and causal window both have
 * to continue from the cache length.
 */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ds4_gpu.h"

enum {
    N_HEAD = 24,       /* DS4_SHAPE_QWEN4EXP.n_head        */
    N_KV_HEAD = 2,     /* .n_head_kv                       */
    HEAD_DIM = 256,    /* .n_head_dim                      */
    ROT_DIM = 64,      /* .n_rot                           */
    GQA = N_HEAD / N_KV_HEAD,
    Q_WIDTH = N_HEAD * HEAD_DIM,
    KV_WIDTH = N_KV_HEAD * HEAD_DIM,
    CACHE_CAP = 512,
    MAX_TOKENS = 200,
};

static const float ROPE_THETA = 10000000.0f;   /* .rope_freq_base */
static const float RMS_EPS = 1e-6f;            /* .rms_eps        */
static const float WEIGHT_OFFSET = 1.0f;       /* zero-centered checkpoint */

/* Relative band, |kernel - reference|_max / rms(reference). */
static const double BAND = 2.0e-4;

bool ds4_log_is_tty(FILE *fp) { (void)fp; return false; }

static void require(int ok, const char *what) {
    if (!ok) { fprintf(stderr, "test_qwen4exp_qsa_ref: %s failed\n", what); exit(1); }
}

static void *xalloc(size_t bytes) {
    void *p = calloc(1, bytes);
    require(p != NULL, "allocation");
    return p;
}

static double sample(uint64_t seed) {
    uint64_t z = seed + 0x9e3779b97f4a7c15ull;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    z ^= z >> 31;
    return (double)(z >> 11) * (2.0 / 9007199254740992.0) - 1.0;
}

/* ------------------------------------------------------------- reference */

/* Own derivation, in double, from the rope base: base^(-2j/rot_dim). */
static double g_ref_inv_freq[ROT_DIM / 2];

static void ref_build_inv_freq(void) {
    for (uint32_t j = 0; j < ROT_DIM / 2u; j++) {
        g_ref_inv_freq[j] =
            pow((double)ROPE_THETA, -(double)(2u * j) / (double)ROT_DIM);
    }
}

static void ref_head_rms_norm(double *vec, const float *weight) {
    double sumsq = 0.0;
    for (uint32_t d = 0; d < HEAD_DIM; d++) sumsq += vec[d] * vec[d];
    const double inv = 1.0 / sqrt(sumsq / (double)HEAD_DIM + (double)RMS_EPS);
    for (uint32_t d = 0; d < HEAD_DIM; d++) {
        vec[d] = vec[d] * inv * ((double)WEIGHT_OFFSET + (double)weight[d]);
    }
}

static void ref_rope(double *vec, uint32_t pos) {
    const uint32_t half = ROT_DIM / 2u;
    for (uint32_t j = 0; j < half; j++) {
        const double theta = (double)pos * g_ref_inv_freq[j];
        const double c = cos(theta), s = sin(theta);
        const double x1 = vec[j], x2 = vec[j + half];
        vec[j] = x1 * c - x2 * s;
        vec[j + half] = x2 * c + x1 * s;
    }
}

/* The reference's own caches, [CACHE_CAP][N_KV_HEAD][HEAD_DIM] in double. */
typedef struct { double *k; double *v; } ref_cache;

static void ref_cache_open(ref_cache *c) {
    c->k = xalloc((size_t)CACHE_CAP * KV_WIDTH * sizeof(double));
    c->v = xalloc((size_t)CACHE_CAP * KV_WIDTH * sizeof(double));
}

static void ref_cache_close(ref_cache *c) { free(c->k); free(c->v); }

/*
 * One segment: `n_tokens` rows starting at cache row `pos0`.  Writes the
 * gated attention output for those rows into `out` ([n_tokens][Q_WIDTH]).
 */
static void ref_segment(ref_cache *cache, double *out,
                        const float *doubled, const float *k_in,
                        const float *v_in, const float *q_norm_w,
                        const float *k_norm_w,
                        uint32_t pos0, uint32_t n_tokens) {
    double *q = xalloc((size_t)n_tokens * Q_WIDTH * sizeof(double));
    double *gate = xalloc((size_t)n_tokens * Q_WIDTH * sizeof(double));

    for (uint32_t t = 0; t < n_tokens; t++) {
        /* Split the doubled query row: head-major, gate interleaved. */
        const float *row = doubled + (size_t)t * 2u * Q_WIDTH;
        for (uint32_t h = 0; h < N_HEAD; h++) {
            double *qh = q + (size_t)t * Q_WIDTH + (size_t)h * HEAD_DIM;
            double *gh = gate + (size_t)t * Q_WIDTH + (size_t)h * HEAD_DIM;
            for (uint32_t d = 0; d < HEAD_DIM; d++) {
                qh[d] = (double)row[(size_t)h * 2u * HEAD_DIM + d];
                gh[d] = (double)row[(size_t)h * 2u * HEAD_DIM + HEAD_DIM + d];
            }
            ref_head_rms_norm(qh, q_norm_w);
            ref_rope(qh, pos0 + t);
        }
        /* Keys and values into the cache at row pos0 + t. */
        for (uint32_t h = 0; h < N_KV_HEAD; h++) {
            double *kc = cache->k +
                ((size_t)(pos0 + t) * N_KV_HEAD + h) * HEAD_DIM;
            double *vc = cache->v +
                ((size_t)(pos0 + t) * N_KV_HEAD + h) * HEAD_DIM;
            for (uint32_t d = 0; d < HEAD_DIM; d++) {
                kc[d] = (double)k_in[(size_t)t * KV_WIDTH + h * HEAD_DIM + d];
                vc[d] = (double)v_in[(size_t)t * KV_WIDTH + h * HEAD_DIM + d];
            }
            ref_head_rms_norm(kc, k_norm_w);
            ref_rope(kc, pos0 + t);
        }
    }

    const double scale = 1.0 / sqrt((double)HEAD_DIM);
    for (uint32_t t = 0; t < n_tokens; t++) {
        const uint32_t pos = pos0 + t;
        const uint32_t count = pos + 1u;      /* causal over cached + new */
        double *probs = xalloc((size_t)count * sizeof(double));
        for (uint32_t h = 0; h < N_HEAD; h++) {
            const uint32_t kv = h / GQA;
            const double *qh = q + (size_t)t * Q_WIDTH + (size_t)h * HEAD_DIM;
            double best = -HUGE_VAL;
            for (uint32_t j = 0; j < count; j++) {
                const double *kc =
                    cache->k + ((size_t)j * N_KV_HEAD + kv) * HEAD_DIM;
                double dot = 0.0;
                for (uint32_t d = 0; d < HEAD_DIM; d++) dot += qh[d] * kc[d];
                probs[j] = dot * scale;
                if (probs[j] > best) best = probs[j];
            }
            double sum = 0.0;
            for (uint32_t j = 0; j < count; j++) {
                probs[j] = exp(probs[j] - best);
                sum += probs[j];
            }
            double *dst = out + (size_t)t * Q_WIDTH + (size_t)h * HEAD_DIM;
            for (uint32_t d = 0; d < HEAD_DIM; d++) dst[d] = 0.0;
            for (uint32_t j = 0; j < count; j++) {
                const double *vc =
                    cache->v + ((size_t)j * N_KV_HEAD + kv) * HEAD_DIM;
                for (uint32_t d = 0; d < HEAD_DIM; d++)
                    dst[d] += probs[j] * vc[d];
            }
            const double *gh = gate + (size_t)t * Q_WIDTH + (size_t)h * HEAD_DIM;
            for (uint32_t d = 0; d < HEAD_DIM; d++) {
                dst[d] = (dst[d] / sum) * (1.0 / (1.0 + exp(-gh[d])));
            }
        }
        free(probs);
    }
    free(gate);
    free(q);
}

/* ------------------------------------------------------------------ GPU */

typedef struct {
    ds4_gpu_tensor *doubled, *q, *gate, *k_in, *v_in;
    ds4_gpu_tensor *q_norm, *k_norm, *inv_freq;
    ds4_gpu_tensor *k_cache, *v_cache, *out;
} gpu_set;

static void gpu_open(gpu_set *g, const float *q_norm_w, const float *k_norm_w) {
    g->doubled = ds4_gpu_tensor_alloc((size_t)MAX_TOKENS * 2u * Q_WIDTH * sizeof(float));
    g->q = ds4_gpu_tensor_alloc((size_t)MAX_TOKENS * Q_WIDTH * sizeof(float));
    g->gate = ds4_gpu_tensor_alloc((size_t)MAX_TOKENS * Q_WIDTH * sizeof(float));
    g->k_in = ds4_gpu_tensor_alloc((size_t)MAX_TOKENS * KV_WIDTH * sizeof(float));
    g->v_in = ds4_gpu_tensor_alloc((size_t)MAX_TOKENS * KV_WIDTH * sizeof(float));
    g->q_norm = ds4_gpu_tensor_alloc((size_t)HEAD_DIM * sizeof(float));
    g->k_norm = ds4_gpu_tensor_alloc((size_t)HEAD_DIM * sizeof(float));
    g->inv_freq = ds4_gpu_tensor_alloc((size_t)(ROT_DIM / 2) * sizeof(float));
    g->k_cache = ds4_gpu_tensor_alloc((size_t)CACHE_CAP * KV_WIDTH * sizeof(float));
    g->v_cache = ds4_gpu_tensor_alloc((size_t)CACHE_CAP * KV_WIDTH * sizeof(float));
    g->out = ds4_gpu_tensor_alloc((size_t)MAX_TOKENS * Q_WIDTH * sizeof(float));
    require(g->doubled && g->q && g->gate && g->k_in && g->v_in && g->q_norm &&
            g->k_norm && g->inv_freq && g->k_cache && g->v_cache && g->out,
            "tensor allocation");

    float inv_freq[ROT_DIM / 2];
    ds4_gpu_qwen4exp_rope_inv_freq(inv_freq, ROT_DIM, ROPE_THETA);
    require(ds4_gpu_tensor_write(g->inv_freq, 0, inv_freq, sizeof(inv_freq)),
            "inverse frequency upload");
    require(ds4_gpu_tensor_write(g->q_norm, 0, q_norm_w,
            (size_t)HEAD_DIM * sizeof(float)), "q norm upload");
    require(ds4_gpu_tensor_write(g->k_norm, 0, k_norm_w,
            (size_t)HEAD_DIM * sizeof(float)), "k norm upload");
}

static void gpu_close(gpu_set *g) {
    ds4_gpu_tensor_free(g->out);
    ds4_gpu_tensor_free(g->v_cache);
    ds4_gpu_tensor_free(g->k_cache);
    ds4_gpu_tensor_free(g->inv_freq);
    ds4_gpu_tensor_free(g->k_norm);
    ds4_gpu_tensor_free(g->q_norm);
    ds4_gpu_tensor_free(g->v_in);
    ds4_gpu_tensor_free(g->k_in);
    ds4_gpu_tensor_free(g->gate);
    ds4_gpu_tensor_free(g->q);
    ds4_gpu_tensor_free(g->doubled);
}

/* One segment through the kernels, the same call order as the graph. */
static void gpu_segment(gpu_set *g, float *out,
                        const float *doubled, const float *k_in,
                        const float *v_in, uint32_t pos0, uint32_t n_tokens) {
    require(ds4_gpu_tensor_write(g->doubled, 0, doubled,
            (size_t)n_tokens * 2u * Q_WIDTH * sizeof(float)), "doubled write");
    require(ds4_gpu_tensor_write(g->k_in, 0, k_in,
            (size_t)n_tokens * KV_WIDTH * sizeof(float)), "k write");
    require(ds4_gpu_tensor_write(g->v_in, 0, v_in,
            (size_t)n_tokens * KV_WIDTH * sizeof(float)), "v write");

    /* The cache append is a blit, which only exists inside a command batch;
     * the graph runs the whole layer inside one, so this does too. */
    require(ds4_gpu_begin_commands(), "command batch");
    require(ds4_gpu_qwen4exp_qsa_split_doubled_q_tensor(
                g->q, g->gate, g->doubled, n_tokens, N_HEAD, HEAD_DIM),
            "doubled query split");
    require(ds4_gpu_qwen4exp_head_rms_norm_tensor(
                g->q, g->q, g->q_norm, n_tokens * N_HEAD, HEAD_DIM,
                RMS_EPS, WEIGHT_OFFSET), "query head norm");
    require(ds4_gpu_qwen4exp_head_rms_norm_tensor(
                g->k_in, g->k_in, g->k_norm, n_tokens * N_KV_HEAD, HEAD_DIM,
                RMS_EPS, WEIGHT_OFFSET), "key head norm");
    require(ds4_gpu_qwen4exp_rope_head_tensor(
                g->q, g->inv_freq, n_tokens, N_HEAD, HEAD_DIM, ROT_DIM, pos0),
            "query rope");
    require(ds4_gpu_qwen4exp_rope_head_tensor(
                g->k_in, g->inv_freq, n_tokens, N_KV_HEAD, HEAD_DIM, ROT_DIM,
                pos0), "key rope");

    const uint64_t kv_row = (uint64_t)KV_WIDTH * sizeof(float);
    require(ds4_gpu_tensor_copy(g->k_cache, (uint64_t)pos0 * kv_row,
            g->k_in, 0, (uint64_t)n_tokens * kv_row), "key cache append");
    require(ds4_gpu_tensor_copy(g->v_cache, (uint64_t)pos0 * kv_row,
            g->v_in, 0, (uint64_t)n_tokens * kv_row), "value cache append");

    const float scale = 1.0f / sqrtf((float)HEAD_DIM);
    require(ds4_gpu_qwen4exp_qsa_attention_tensor(
                g->out, g->q, g->k_cache, g->v_cache, NULL, NULL, n_tokens,
                N_HEAD, N_KV_HEAD, HEAD_DIM, pos0, CACHE_CAP, 0, scale),
            "dense attention");
    require(ds4_gpu_qwen4exp_qsa_output_gate_tensor(
                g->out, g->gate, n_tokens * Q_WIDTH), "output gate");
    require(ds4_gpu_end_commands(), "command batch completion");
    require(ds4_gpu_tensor_read(g->out, 0, out,
            (size_t)n_tokens * Q_WIDTH * sizeof(float)), "output read");
}

/* ------------------------------------------------------------ comparison */

static int g_failures;

static void report(const char *what, const float *got, const double *want,
                   size_t count) {
    double max_abs = 0.0, sumsq = 0.0;
    size_t worst = 0;
    for (size_t i = 0; i < count; i++) {
        if (!isfinite(got[i])) {
            fprintf(stderr, "test_qwen4exp_qsa_ref: %s element %zu is %g\n",
                    what, i, got[i]);
            exit(1);
        }
        const double diff = fabs((double)got[i] - want[i]);
        if (diff > max_abs) { max_abs = diff; worst = i; }
        sumsq += want[i] * want[i];
    }
    const double rms = sqrt(sumsq / (double)count);
    if (rms < 1e-6) {
        fprintf(stderr, "  %-40s REFERENCE DEGENERATE (rms %.3g)\n", what, rms);
        g_failures++;
        return;
    }
    const double rel = max_abs / rms;
    const bool ok = rel <= BAND;
    printf("  %-40s rms %.4g  max abs %.4g  max rel %.4g  %s\n",
           what, rms, max_abs, rel, ok ? "PASS" : "FAIL");
    if (!ok) {
        fprintf(stderr, "    worst element %zu: got %.9g want %.9g\n",
                worst, (double)got[worst], want[worst]);
        g_failures++;
    }
}

int main(void) {
    ref_build_inv_freq();
    require(ds4_gpu_init(), "GPU initialisation");

    float *q_norm_w = xalloc((size_t)HEAD_DIM * sizeof(float));
    float *k_norm_w = xalloc((size_t)HEAD_DIM * sizeof(float));
    for (uint32_t d = 0; d < HEAD_DIM; d++) {
        q_norm_w[d] = (float)(0.1 * sample(0xC1000000ull + d));
        k_norm_w[d] = (float)(0.1 * sample(0xC2000000ull + d));
    }

    float *doubled = xalloc((size_t)MAX_TOKENS * 2u * Q_WIDTH * sizeof(float));
    float *k_in = xalloc((size_t)MAX_TOKENS * KV_WIDTH * sizeof(float));
    float *v_in = xalloc((size_t)MAX_TOKENS * KV_WIDTH * sizeof(float));
    for (size_t i = 0; i < (size_t)MAX_TOKENS * 2u * Q_WIDTH; i++)
        doubled[i] = (float)(0.8 * sample(0xD1000000ull + i));
    for (size_t i = 0; i < (size_t)MAX_TOKENS * KV_WIDTH; i++) {
        k_in[i] = (float)(0.8 * sample(0xD2000000ull + i));
        v_in[i] = (float)(0.8 * sample(0xD3000000ull + i));
    }

    float *got = xalloc((size_t)MAX_TOKENS * Q_WIDTH * sizeof(float));
    double *want = xalloc((size_t)MAX_TOKENS * Q_WIDTH * sizeof(double));

    gpu_set g;
    gpu_open(&g, q_norm_w, k_norm_w);

    printf("Qwen4-Exp dense QSA vs double reference (band: max rel %.1g)\n",
           BAND);

    static const uint32_t lengths[] = {1u, 6u, 8u, 16u, 33u, 64u, 65u, 128u, 200u};
    for (size_t li = 0; li < sizeof(lengths) / sizeof(lengths[0]); li++) {
        const uint32_t tokens = lengths[li];
        char label[96];
        ref_cache cache;
        ref_cache_open(&cache);
        ref_segment(&cache, want, doubled, k_in, v_in, q_norm_w, k_norm_w,
                    0u, tokens);
        require(ds4_gpu_tensor_fill_f32(g.k_cache, 0.0f,
                (uint64_t)CACHE_CAP * KV_WIDTH), "key cache clear");
        require(ds4_gpu_tensor_fill_f32(g.v_cache, 0.0f,
                (uint64_t)CACHE_CAP * KV_WIDTH), "value cache clear");
        gpu_segment(&g, got, doubled, k_in, v_in, 0u, tokens);
        snprintf(label, sizeof(label), "prefill %u from position 0", tokens);
        report(label, got, want, (size_t)tokens * Q_WIDTH);
        ref_cache_close(&cache);
    }

    /* Cached keys: the second call has to rope at pos0 and attend over the
     * rows the first call left in the cache. */
    static const uint32_t cached[][2] = { {33u, 1u}, {64u, 3u} };
    for (size_t ci = 0; ci < sizeof(cached) / sizeof(cached[0]); ci++) {
        const uint32_t prefill = cached[ci][0];
        const uint32_t rows = cached[ci][1];
        char label[96];
        ref_cache cache;
        ref_cache_open(&cache);
        double *scratch = xalloc((size_t)prefill * Q_WIDTH * sizeof(double));
        ref_segment(&cache, scratch, doubled, k_in, v_in, q_norm_w, k_norm_w,
                    0u, prefill);
        free(scratch);
        ref_segment(&cache, want,
                    doubled + (size_t)prefill * 2u * Q_WIDTH,
                    k_in + (size_t)prefill * KV_WIDTH,
                    v_in + (size_t)prefill * KV_WIDTH,
                    q_norm_w, k_norm_w, prefill, rows);

        require(ds4_gpu_tensor_fill_f32(g.k_cache, 0.0f,
                (uint64_t)CACHE_CAP * KV_WIDTH), "key cache clear");
        require(ds4_gpu_tensor_fill_f32(g.v_cache, 0.0f,
                (uint64_t)CACHE_CAP * KV_WIDTH), "value cache clear");
        gpu_segment(&g, got, doubled, k_in, v_in, 0u, prefill);
        gpu_segment(&g, got,
                    doubled + (size_t)prefill * 2u * Q_WIDTH,
                    k_in + (size_t)prefill * KV_WIDTH,
                    v_in + (size_t)prefill * KV_WIDTH,
                    prefill, rows);
        snprintf(label, sizeof(label), "prefill %u then %u cached row%s",
                 prefill, rows, rows == 1u ? "" : "s");
        report(label, got, want, (size_t)rows * Q_WIDTH);
        ref_cache_close(&cache);
    }

    gpu_close(&g);
    free(want); free(got); free(v_in); free(k_in); free(doubled);
    free(k_norm_w); free(q_norm_w);
    ds4_gpu_cleanup();

    if (g_failures) {
        printf("Qwen4-Exp dense QSA reference tests: FAIL (%d)\n", g_failures);
        return 1;
    }
    puts("Qwen4-Exp dense QSA reference tests: PASS");
    return 0;
}
