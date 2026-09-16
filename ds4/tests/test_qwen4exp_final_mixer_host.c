/* Execute the production final-mixer host fragment without a GPU. The mixer
 * stub records which rows would run; CUDA numerical equivalence is covered
 * separately by test_qwen4exp_hc_norm and still needs real hardware. */
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DS4_TEST_HOOKS 1
#define DS4_N_HC 4u
#define DS4_N_EMBD 3u
enum { WIDTH = DS4_N_HC * DS4_N_EMBD, MAX_ROWS = 4096 };

typedef struct { float *data; uint64_t bytes; } ds4_gpu_tensor;
typedef struct { int unused; } ds4_tensor;
typedef struct { int unused; } ds4_model;
typedef struct {
    const ds4_tensor *output_hc_norm, *output_hc_down, *output_hc_up;
    float output_hc_norm_offset;
} ds4_qwen4exp_weights;
typedef struct {
    ds4_gpu_tensor *hyper, *final_mixer_snapshot;
    bool spec_logit_rows, hc_pending;
} ds4_qwen4exp_session;

static unsigned views_live, mixed_rows, injected_rows;
static bool fail_view, fail_mix, fail_copy, fail_inject, keep_full;
static float first_mixed, last_mixed;

static ds4_gpu_tensor *ds4_gpu_tensor_view(const ds4_gpu_tensor *base,
        uint64_t offset, uint64_t bytes) {
    if (fail_view || offset > base->bytes || bytes > base->bytes - offset) return NULL;
    ds4_gpu_tensor *v = malloc(sizeof(*v));
    assert(v);
    *v = (ds4_gpu_tensor){base->data + offset / sizeof(float), bytes};
    views_live++;
    return v;
}
static void ds4_gpu_tensor_free(ds4_gpu_tensor *v) {
    assert(views_live);
    views_live--;
    free(v);
}
static int ds4_gpu_tensor_copy(ds4_gpu_tensor *out, uint64_t to,
        const ds4_gpu_tensor *in, uint64_t from, uint64_t bytes) {
    if (fail_copy) return 0;
    assert(to <= out->bytes && bytes <= out->bytes - to);
    assert(from <= in->bytes && bytes <= in->bytes - from);
    memcpy((char *)out->data + to, (const char *)in->data + from, bytes);
    return 1;
}
static bool qwen4exp_hc_flush_pending(ds4_qwen4exp_session *s, uint32_t rows) {
    if (!s->hc_pending) return true;
    if (fail_inject) return false;
    for (uint32_t i = 0; i < rows * WIDTH; i++) s->hyper->data[i] += 1000.0f;
    injected_rows = rows;
    s->hc_pending = false;
    return true;
}
static bool qwen4exp_graph_residual(ds4_qwen4exp_session *s,
        const ds4_model *m, const ds4_tensor *norm, const ds4_tensor *down,
        const ds4_tensor *up, const ds4_tensor *inject, float bias, uint32_t rows) {
    (void)m; (void)norm; (void)down; (void)up; (void)bias;
    assert(!inject);
    if (fail_mix) return false;
    if (!qwen4exp_hc_flush_pending(s, rows)) return false;
    assert(s->hyper->bytes >= (uint64_t)rows * WIDTH * sizeof(float));
    mixed_rows = rows;
    first_mixed = s->hyper->data[0];
    last_mixed = s->hyper->data[(rows - 1u) * WIDTH];
    return true;
}

#include "ds4_qwen4exp_final_mixer.inc"

static void run_case(uint32_t rows, bool all_logits, bool pending, unsigned failure) {
    float data[MAX_ROWS * WIDTH], snapshot[MAX_ROWS * WIDTH];
    for (uint32_t r = 0; r < rows; r++) {
        for (unsigned c = 0; c < WIDTH; c++) data[r * WIDTH + c] = (float)r;
    }
    memset(snapshot, 0, sizeof(snapshot));
    ds4_gpu_tensor hyper = {data, (uint64_t)rows * WIDTH * sizeof(float)};
    ds4_gpu_tensor snap = {snapshot, sizeof(snapshot)};
    ds4_qwen4exp_session s = {&hyper, &snap, all_logits, pending};
    const ds4_qwen4exp_weights w = {0};
    const ds4_model m = {0};
    mixed_rows = injected_rows = 0;
    fail_view = failure == 1;
    fail_mix = failure == 2;
    fail_copy = failure == 3;
    fail_inject = failure == 4;
    const uint32_t got = qwen4exp_graph_final_mixer(&s, &w, &m, rows);
    assert(s.hyper == &hyper && s.final_mixer_snapshot == &snap && views_live == 0);
    if (failure) {
        assert(got == 0);
        return;
    }
#if defined(__APPLE__) || defined(DS4_ROCM_BUILD)
    const uint32_t want = rows;
#else
    const uint32_t want = !keep_full && !all_logits && rows > 8u ? 8u : rows;
#endif
    assert(got == want && mixed_rows == want);
    assert(injected_rows == (pending ? rows : 0u));
    const float delta = pending ? 1000.0f : 0.0f;
    assert(first_mixed == (float)(rows - want) + delta);
    assert(last_mixed == (float)(rows - 1u) + delta);
    /* MTP's next reader must still see every row, including the skipped prefix. */
    for (uint32_t r = 0; r < rows; r++) {
        for (unsigned c = 0; c < WIDTH; c++) assert(data[r * WIDTH + c] == (float)r + delta);
    }
    if (want < rows) assert(memcmp(data, snapshot, (size_t)hyper.bytes) == 0);
}

int main(void) {
    unsetenv("DS4_QWEN4EXP_NO_FINAL_MIXER_TAIL");
    const uint32_t rows[] = {1, 2, 7, 8, 9, 17, 47, 48, 64, 65, 513, 1024, 4096};
    unsigned cases = 0;
    for (unsigned i = 0; i < sizeof(rows) / sizeof(rows[0]); i++) {
        for (unsigned all = 0; all < 2; all++) {
            for (unsigned pending = 0; pending < 2; pending++) {
                run_case(rows[i], all != 0, pending != 0, 0);
                cases++;
            }
        }
    }
    for (unsigned failure = 1; failure <= 4; failure++) {
#if defined(__APPLE__) || defined(DS4_ROCM_BUILD)
        /* These backends never allocate the narrowed view or take its copy. */
        if (failure == 1 || failure == 3) continue;
#endif
        run_case(65, false, true, failure);
        cases++;
    }
    keep_full = true;
    assert(setenv("DS4_QWEN4EXP_NO_FINAL_MIXER_TAIL", "1", 1) == 0);
    for (unsigned i = 0; i < sizeof(rows) / sizeof(rows[0]); i++)
        for (unsigned pending = 0; pending < 2; pending++) {
            run_case(rows[i], false, pending != 0, 0);
            cases++;
        }
    unsetenv("DS4_QWEN4EXP_NO_FINAL_MIXER_TAIL");
    printf("final mixer host: %u state/row/failure cases passed; GPU numerics untested\n", cases);
    return 0;
}
