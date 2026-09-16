/* Exercise the production cache dispatcher, not GPU arithmetic. Recorded
 * work runs only at graph commit/replay and reads the live device position
 * and input buffer, making stale captures and duplicate publication visible. */
#include "../ds4_gpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { DS4_MAX_LAYER = 64, DS4_N_INDEXER_TOP_K = 2048,
       DS4_QWEN4EXP_MTP_MAX_COMMIT = 7, CACHE_CAP = 4096 };
struct ds4_gpu_tensor { uint32_t position; float input[16]; };
typedef struct { bool is_full_attention; } ds4_qwen4exp_layer_weights;
typedef struct { int identity; } ds4_model;
typedef struct {
    struct { uint32_t n_batch, n_ctx; } plan;
    ds4_gpu_tensor *hyper, *d_pos, *qsa_k[DS4_MAX_LAYER];
    uint32_t pos, head_cache_pos;
    bool hc_pending, state_dirty;
    float cache[CACHE_CAP];
    uint32_t writes[CACHE_CAP];
} ds4_qwen4exp_session;
typedef struct {
    const ds4_qwen4exp_layer_weights *l;
    const ds4_model *m;
} ds4_qwen4exp_head_block_ctx;
static const struct { uint32_t n_layer; } g_ds4_qwen4exp = {48u};

static int failures, checks;
#define CHECK(c) do { checks++; if (!(c)) { failures++; \
    fprintf(stderr, "line %d: %s\n", __LINE__, #c); } } while (0)

typedef struct {
    ds4_qwen4exp_session *session;
    ds4_gpu_tensor *input, *position;
    uint32_t literal_pos, rows;
} recorded_work;
static struct {
    int begin_result, fail_encode, fail_update_at;
    bool supported, timing, fail_end, capturing, have_exec;
    unsigned begins, encodes, ends, aborts, updates, executions;
    recorded_work pending, exec;
    ds4_decode_graph_key key, exec_key;
} g;

static void reset_backend(void) {
    memset(&g, 0, sizeof(g));
    g.supported = true;
    g.begin_result = -1;
    unsetenv("DS4_QWEN4EXP_TIME_SLICES");
}
static void execute(recorded_work w) {
    const uint32_t pos = w.position ? w.position->position : w.literal_pos;
    CHECK((uint64_t)pos + w.rows <= w.session->plan.n_ctx);
    if ((uint64_t)pos + w.rows > w.session->plan.n_ctx) return;
    g.executions++;
    for (uint32_t t = 0; t < w.rows; t++) {
        w.session->writes[pos + t]++;
        w.session->cache[pos + t] = w.input->input[t] + (float)(pos + t);
    }
}
int ds4_gpu_decode_graphs_supported(void) { return g.supported; }
static int qw_hb_time_on(void) { return g.timing; }
int ds4_gpu_qwen4exp_update_dpos(ds4_gpu_tensor *d_pos, uint32_t pos) {
    g.updates++;
    if ((int)g.updates == g.fail_update_at) return 0;
    d_pos->position = pos;
    return 1;
}
int ds4_gpu_decode_graph_begin(const ds4_decode_graph_key *key) {
    g.begins++;
    g.key = *key;
    if (g.begin_result == 0) g.capturing = true;
    if (g.begin_result == 1) {
        CHECK(g.have_exec && memcmp(key, &g.exec_key, sizeof(*key)) == 0);
        if (!g.have_exec) return -1;
        execute(g.exec);
    }
    return g.begin_result;
}
int ds4_gpu_decode_graph_end(const ds4_decode_graph_key *key) {
    CHECK(g.capturing);
    g.ends++;
    g.capturing = false;
    if (g.fail_end) return -1;
    g.exec = g.pending;
    g.exec_key = *key;
    g.have_exec = true;
    execute(g.exec);
    return 0;
}
void ds4_gpu_decode_graph_abort(const ds4_decode_graph_key *key) {
    (void)key;
    CHECK(g.capturing);
    g.aborts++;
    g.capturing = false;
}
static bool qwen4exp_graph_head_cache_encode(ds4_qwen4exp_session *s,
        const ds4_qwen4exp_layer_weights *l, const ds4_model *m,
        uint32_t il, uint32_t rows) {
    (void)l; (void)m;
    CHECK(il >= g_ds4_qwen4exp.n_layer);
    g.encodes++;
    if (g.fail_encode) {
        if (g.fail_encode > 0) g.fail_encode--;
        return false;
    }
    recorded_work w = {s, s->hyper, s->d_pos, s->pos, rows};
    if (g.capturing) g.pending = w;
    else execute(w);
    return true;
}

#include "../ds4_qwen4exp_head_cache.inc"

typedef struct {
    ds4_qwen4exp_session s;
    ds4_gpu_tensor target, head, position, kv;
    ds4_qwen4exp_layer_weights weights;
    ds4_model model;
    ds4_qwen4exp_head_block_ctx ctx;
} fixture;
static void init_fixture(fixture *f) {
    memset(f, 0, sizeof(*f));
    f->s.plan.n_batch = 16;
    f->s.plan.n_ctx = CACHE_CAP;
    f->s.pos = f->position.position = 37u;
    f->s.head_cache_pos = 9u;
    f->s.hyper = &f->target;
    f->s.d_pos = &f->position;
    f->s.qsa_k[48] = f->s.qsa_k[49] = &f->kv;
    f->weights.is_full_attention = true;
    f->ctx.l = &f->weights;
    f->ctx.m = &f->model;
    for (unsigned i = 0; i < 16; i++) f->head.input[i] = (float)i + 0.5f;
}
static int forward(fixture *f, uint32_t pos, uint32_t rows) {
    return ds4_qwen4exp_graph_head_cache(&f->ctx, &f->s, &f->head, 48u, pos, rows);
}
static void check_restored(fixture *f) {
    CHECK(f->s.hyper == &f->target && f->s.pos == 37u);
    CHECK(f->s.state_dirty);
    if (f->s.d_pos) CHECK(f->position.position == 37u);
}
static void check_published(fixture *f, uint32_t pos, uint32_t rows) {
    for (uint32_t t = 0; t < rows; t++) {
        CHECK(f->s.writes[pos + t] == 1u);
        CHECK(f->s.cache[pos + t] == f->head.input[t] + (float)(pos + t));
    }
    CHECK(f->s.head_cache_pos == pos + rows);
    check_restored(f);
}

static void test_replay(void) {
    fixture f;
    init_fixture(&f); reset_backend();
    CHECK(forward(&f, 3u, 2u)); /* eager warm pass */
    CHECK(g.begins == 1u && g.encodes == 1u && g.executions == 1u);
    CHECK(g.key.il == 48u && g.key.island == 2u && g.key.variant == 2u &&
          g.key._pad == 0x4d545043u);
    const ds4_decode_graph_key key = g.key;
    check_published(&f, 3u, 2u);
    g.begin_result = 0;
    CHECK(forward(&f, 7u, 2u)); /* capture commits this call exactly once */
    CHECK(g.encodes == 2u && g.ends == 1u && g.executions == 2u);
    check_published(&f, 7u, 2u);
    g.begin_result = 1;
    f.head.input[0] = 100.0f;
    f.head.input[1] = -200.0f;
    CHECK(forward(&f, 11u, 2u)); /* same graph, new position and input */
    CHECK(g.encodes == 2u && g.executions == 3u && g.updates == 6u);
    CHECK(memcmp(&key, &g.key, sizeof(key)) == 0);
    check_published(&f, 11u, 2u);
    unsigned written = 0;
    for (unsigned p = 0; p < CACHE_CAP; p++) written += f.s.writes[p];
    CHECK(written == 6u);
}

static void test_capture_failures(void) {
    for (unsigned mode = 0; mode < 5u; mode++) {
        fixture f;
        init_fixture(&f); reset_backend();
        g.begin_result = mode == 0 ? -1 : 0;
        g.fail_end = mode == 2;
        g.fail_encode = mode == 1 ? 1 : (mode >= 3 ? -1 : 0);
        if (mode == 4) g.begin_result = -1;
        const int rc = forward(&f, 3u, 2u);
        CHECK((rc != 0) == (mode < 3));
        CHECK(!g.capturing);
        CHECK(g.executions == (mode < 3 ? 1u : 0u));
        CHECK(g.encodes == ((mode == 0 || mode == 4) ? 1u : 2u));
        CHECK(g.aborts == ((mode == 1 || mode == 3) ? 1u : 0u));
        CHECK(g.ends == (mode == 2 ? 1u : 0u));
        if (rc) check_published(&f, 3u, 2u);
        else { CHECK(f.s.head_cache_pos == 9u); check_restored(&f); }
    }
    for (int update = 1; update <= 2; update++) {
        fixture f;
        init_fixture(&f); reset_backend();
        g.fail_update_at = update;
        CHECK(!forward(&f, 3u, 1u));
        CHECK(f.s.hyper == &f.target && f.s.pos == 37u);
        CHECK(f.s.head_cache_pos == 9u && g.updates == 2u);
        CHECK(g.executions == (update == 1 ? 0u : 1u));
        CHECK(g.encodes == g.executions);
    }
}

static void test_eager_envelope(void) {
    for (unsigned mode = 0; mode < 6u; mode++) {
        fixture f;
        init_fixture(&f); reset_backend();
        uint32_t pos = 3u, rows = 1u;
        switch (mode) {
        case 0: f.s.d_pos = NULL; break;
        case 1: rows = DS4_QWEN4EXP_MTP_MAX_COMMIT + 1u; break;
        case 2: pos = DS4_N_INDEXER_TOP_K; break;
        case 3: g.timing = true; break;
        case 4: setenv("DS4_QWEN4EXP_TIME_SLICES", "1", 1); break;
        case 5: g.supported = false; break;
        }
        CHECK(forward(&f, pos, rows));
        CHECK(g.begins == 0u && g.encodes == 1u && g.executions == 1u);
        check_published(&f, pos, rows);
    }
    for (uint32_t rows = 1; rows <= DS4_QWEN4EXP_MTP_MAX_COMMIT; rows++) {
        fixture f;
        init_fixture(&f); reset_backend();
        CHECK(forward(&f, DS4_N_INDEXER_TOP_K - rows, rows));
        CHECK(g.begins == 1u && g.key.variant == rows);
        check_published(&f, DS4_N_INDEXER_TOP_K - rows, rows);
    }
}

static void test_keys_and_validation(void) {
    fixture f, other;
    init_fixture(&f); init_fixture(&other); reset_backend();
    CHECK(forward(&f, 3u, 1u));
    const ds4_decode_graph_key original = g.key;
    CHECK(original.cur_hc == &f.head && original.after_attn_hc == &f.s &&
          original.after_ffn_hc == &f.weights && original.attn_norm == &f.model);
    CHECK(forward(&f, 5u, 2u)); CHECK(memcmp(&original, &g.key, sizeof(g.key)) != 0);
    CHECK(ds4_qwen4exp_graph_head_cache(&f.ctx, &f.s, &other.head, 48u, 8u, 1u));
    CHECK(g.key.cur_hc != original.cur_hc);
    CHECK(forward(&other, 3u, 1u)); CHECK(g.key.after_attn_hc != original.after_attn_hc);
    other.ctx.l = &f.weights;
    CHECK(forward(&other, 5u, 1u)); CHECK(g.key.after_ffn_hc == original.after_ffn_hc);
    other.ctx.m = &f.model;
    CHECK(forward(&other, 7u, 1u)); CHECK(g.key.attn_norm == original.attn_norm);
    CHECK(ds4_qwen4exp_graph_head_cache(&f.ctx, &f.s, &f.head, 49u, 9u, 1u));
    CHECK(g.key.il == 49u);

    for (unsigned mode = 0; mode < 10u; mode++) {
        init_fixture(&f); reset_backend();
        uint32_t il = 48u, pos = 3u, rows = 1u;
        void *ctx = &f.ctx, *cache = &f.s;
        ds4_gpu_tensor *hyper = &f.head;
        switch (mode) {
        case 0: ctx = NULL; break;
        case 1: cache = NULL; break;
        case 2: hyper = NULL; break;
        case 3: rows = 0u; break;
        case 4: rows = 17u; break;
        case 5: il = 47u; f.s.qsa_k[il] = &f.kv; break;
        case 6: f.s.qsa_k[il] = NULL; break;
        case 7: f.weights.is_full_attention = false; break;
        case 8: f.s.hc_pending = true; break;
        case 9: pos = UINT32_MAX; break;
        }
        CHECK(!ds4_qwen4exp_graph_head_cache(ctx, cache, hyper, il, pos, rows));
        CHECK(!f.s.state_dirty && g.begins == 0u && g.encodes == 0u && g.updates == 0u);
        CHECK(f.s.hyper == &f.target && f.s.pos == 37u && f.s.head_cache_pos == 9u);
    }
}

int main(void) {
    test_replay();
    test_capture_failures();
    test_eager_envelope();
    test_keys_and_validation();
    unsetenv("DS4_QWEN4EXP_TIME_SLICES");
    printf("head-cache host: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
