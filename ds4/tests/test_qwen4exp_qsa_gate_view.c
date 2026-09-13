/*
 * Bit-exact check: packed qsa_gate materialization vs reading the gate half
 * from the interleaved qsa_doubled row.  Production 24 x 256, prefill B=1024,
 * decode/verify widths, and adversarial gate values.
 *
 * The sigmoid/multiply must match lane for lane, including signed zeros.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ds4_gpu.h"

enum {
    N_HEAD = 24,
    HEAD_DIM = 256,
    ROT_DIM = 64,
    MAX_TOKENS = 1024
};

static const float RMS_EPS = 1e-6f;
static const float WEIGHT_OFFSET = 1.0f;
static const float ROPE_THETA = 10000000.0f;

bool ds4_log_is_tty(FILE *fp) {
    (void)fp;
    return false;
}

static void fail(const char *what) {
    fprintf(stderr, "test_qwen4exp_qsa_gate_view: %s\n", what);
    exit(1);
}

static void require(int ok, const char *what) {
    if (!ok) fail(what);
}

static void *xcalloc(size_t count, size_t size) {
    void *p = calloc(count, size);
    if (!p) fail("host allocation");
    return p;
}

static uint64_t g_rng;

static uint64_t rng_next(void) {
    uint64_t z = (g_rng += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

static float rng_uniform(void) {
    const uint32_t bits = (uint32_t)(rng_next() >> 40);
    return (float)bits / 8388608.0f - 1.0f;
}

static ds4_gpu_tensor *tensor_new(size_t count, size_t elem) {
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc((uint64_t)count * elem);
    if (!t) fail("ds4_gpu_tensor_alloc");
    return t;
}

static void tensor_put(ds4_gpu_tensor *t, const void *data, size_t bytes) {
    require(ds4_gpu_tensor_write(t, 0, data, (uint64_t)bytes), "tensor write");
}

static void tensor_get(const ds4_gpu_tensor *t, void *data, size_t bytes) {
    require(ds4_gpu_tensor_read(t, 0, data, (uint64_t)bytes), "tensor read");
}

static float adversarial_at(size_t i) {
    switch (i % 16u) {
    case 0:  return 0.0f;
    case 1:  return -0.0f;
    case 2:  return 1.401298464e-45f;
    case 3:  return -1.401298464e-45f;
    case 4:  return 88.0f;
    case 5:  return -88.0f;
    case 6:  return 1.0e8f;
    case 7:  return -1.0e8f;
    case 8:  return INFINITY;
    case 9:  return -INFINITY;
    case 10: return 1.0f;
    case 11: return -1.0f;
    case 12: return 16.0f;
    case 13: return nextafterf(0.0f, 1.0f);
    case 14: return rng_uniform() * 40.0f;
    default: return rng_uniform();
    }
}

static void require_bits(const char *what, const float *a, const float *b,
                         size_t n) {
    for (size_t i = 0; i < n; i++) {
        uint32_t ua, ub;
        memcpy(&ua, a + i, 4);
        memcpy(&ub, b + i, 4);
        if (ua == ub) continue;
        fprintf(stderr,
                "test_qwen4exp_qsa_gate_view: %s mismatch at %zu: "
                "got %.9g (%08x) want %.9g (%08x)\n",
                what, i, (double)a[i], ua, (double)b[i], ub);
        exit(1);
    }
}

static void fill_doubled(float *doubled, uint32_t n_tokens) {
    const size_t n = (size_t)n_tokens * 2u * N_HEAD * HEAD_DIM;
    for (size_t i = 0; i < n; i++) doubled[i] = adversarial_at(i);
}

static void fill_out(float *out, uint32_t n_tokens) {
    const size_t n = (size_t)n_tokens * N_HEAD * HEAD_DIM;
    for (size_t i = 0; i < n; i++) out[i] = rng_uniform() * 3.0f;
}

static void check_gate_view(uint32_t n_tokens) {
    const uint32_t q_width = N_HEAD * HEAD_DIM;
    const size_t packed = (size_t)n_tokens * q_width;
    const size_t doubled_n = packed * 2u;

    float *h_doubled = xcalloc(doubled_n, sizeof(float));
    float *h_out = xcalloc(packed, sizeof(float));
    float *got_pack = xcalloc(packed, sizeof(float));
    float *got_view = xcalloc(packed, sizeof(float));
    fill_doubled(h_doubled, n_tokens);
    fill_out(h_out, n_tokens);

    ds4_gpu_tensor *t_doubled = tensor_new(doubled_n, sizeof(float));
    ds4_gpu_tensor *t_q = tensor_new(packed, sizeof(float));
    ds4_gpu_tensor *t_gate = tensor_new(packed, sizeof(float));
    ds4_gpu_tensor *t_pack = tensor_new(packed, sizeof(float));
    ds4_gpu_tensor *t_view = tensor_new(packed, sizeof(float));
    tensor_put(t_doubled, h_doubled, doubled_n * sizeof(float));
    tensor_put(t_pack, h_out, packed * sizeof(float));
    tensor_put(t_view, h_out, packed * sizeof(float));

    require(ds4_gpu_qwen4exp_qsa_split_doubled_q_tensor(
                t_q, t_gate, t_doubled, n_tokens, N_HEAD, HEAD_DIM),
            "split doubled");
    require(ds4_gpu_qwen4exp_qsa_output_gate_tensor(t_pack, t_gate,
                                                    (uint32_t)packed),
            "packed gate");
    require(ds4_gpu_qwen4exp_qsa_output_gate_interleaved_tensor(
                t_view, t_doubled, (uint32_t)packed, HEAD_DIM),
            "interleaved gate");
    tensor_get(t_pack, got_pack, packed * sizeof(float));
    tensor_get(t_view, got_view, packed * sizeof(float));
    require_bits("output gate view", got_view, got_pack, packed);

    ds4_gpu_tensor_free(t_doubled);
    ds4_gpu_tensor_free(t_q);
    ds4_gpu_tensor_free(t_gate);
    ds4_gpu_tensor_free(t_pack);
    ds4_gpu_tensor_free(t_view);
    free(h_doubled);
    free(h_out);
    free(got_pack);
    free(got_view);
    printf("  gate view n_tokens=%u  %zu values bit-exact\n",
           n_tokens, packed);
}

static void check_prep_skip(uint32_t n_tokens) {
    const uint32_t q_width = N_HEAD * HEAD_DIM;
    const size_t packed = (size_t)n_tokens * q_width;
    const size_t doubled_n = packed * 2u;
    const size_t freq_n = (size_t)ROT_DIM / 2u;

    float *h_doubled = xcalloc(doubled_n, sizeof(float));
    float *h_weight = xcalloc(HEAD_DIM, sizeof(float));
    float *h_freq = xcalloc(freq_n, sizeof(float));
    float *q_with = xcalloc(packed, sizeof(float));
    float *q_skip = xcalloc(packed, sizeof(float));
    fill_doubled(h_doubled, n_tokens);
    for (uint32_t d = 0; d < HEAD_DIM; d++) h_weight[d] = rng_uniform() * 0.25f;
    ds4_gpu_qwen4exp_rope_inv_freq(h_freq, ROT_DIM, ROPE_THETA);

    ds4_gpu_tensor *t_doubled = tensor_new(doubled_n, sizeof(float));
    ds4_gpu_tensor *t_weight = tensor_new(HEAD_DIM, sizeof(float));
    ds4_gpu_tensor *t_freq = tensor_new(freq_n, sizeof(float));
    ds4_gpu_tensor *t_q_with = tensor_new(packed, sizeof(float));
    ds4_gpu_tensor *t_q_skip = tensor_new(packed, sizeof(float));
    ds4_gpu_tensor *t_gate = tensor_new(packed, sizeof(float));
    tensor_put(t_doubled, h_doubled, doubled_n * sizeof(float));
    tensor_put(t_weight, h_weight, HEAD_DIM * sizeof(float));
    tensor_put(t_freq, h_freq, freq_n * sizeof(float));

    require(ds4_gpu_qwen4exp_qsa_prep_q_fused_dpos_tensor(
                t_q_with, t_gate, t_doubled, t_weight, t_freq, n_tokens,
                N_HEAD, HEAD_DIM, ROT_DIM, 17u, RMS_EPS, WEIGHT_OFFSET, NULL),
            "fused Q-prep with gate");
    require(ds4_gpu_qwen4exp_qsa_prep_q_fused_dpos_tensor(
                t_q_skip, NULL, t_doubled, t_weight, t_freq, n_tokens,
                N_HEAD, HEAD_DIM, ROT_DIM, 17u, RMS_EPS, WEIGHT_OFFSET, NULL),
            "fused Q-prep skip gate");
    tensor_get(t_q_with, q_with, packed * sizeof(float));
    tensor_get(t_q_skip, q_skip, packed * sizeof(float));
    require_bits("fused Q-prep skip-gate Q", q_skip, q_with, packed);

    ds4_gpu_tensor_free(t_doubled);
    ds4_gpu_tensor_free(t_weight);
    ds4_gpu_tensor_free(t_freq);
    ds4_gpu_tensor_free(t_q_with);
    ds4_gpu_tensor_free(t_q_skip);
    ds4_gpu_tensor_free(t_gate);
    free(h_doubled);
    free(h_weight);
    free(h_freq);
    free(q_with);
    free(q_skip);
    printf("  fused Q-prep skip-gate n_tokens=%u  Q bit-exact\n", n_tokens);
}

int main(void) {
    if (!ds4_gpu_init()) {
        printf("test_qwen4exp_qsa_gate_view: no GPU backend, skipping\n");
        return 0;
    }
    g_rng = 0x5115ec7edull;
    static const uint32_t widths[] = {1u, 2u, 7u, 8u, 64u, 1023u, 1024u};
    printf("test_qwen4exp_qsa_gate_view: 24x256, widths 1..1024\n");
    for (size_t i = 0; i < sizeof(widths) / sizeof(widths[0]); i++) {
        check_gate_view(widths[i]);
        if (widths[i] >= 8u) check_prep_skip(widths[i]);
    }
    ds4_gpu_cleanup();
    printf("test_qwen4exp_qsa_gate_view: OK\n");
    return 0;
}
