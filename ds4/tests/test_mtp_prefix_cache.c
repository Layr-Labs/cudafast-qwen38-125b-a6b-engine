/* Portable API contracts, using the established independent head algebra.
 * cc -O2 -std=c11 -D_GNU_SOURCE -Ids4 ds4/tests/test_mtp_prefix_cache.c
 *    ds4/ds4_qwen4exp_mtp.c -lm -o <local-output>
 */
#define main original_mtp_main
#include "test_qwen4exp_mtp.c"
#undef main

static unsigned seed_calls;
static unsigned seed_rows, seed_pos;
static int seed_fail;
static float seed_seen[DS4_QWEN4EXP_MTP_CACHE_SEED_MAX_ROWS * HEAD_HC_DIM];

static int seed_mock(void *graph, void *cache, ds4_gpu_tensor *hyper,
                     uint32_t il, uint32_t pos, uint32_t rows) {
    (void)graph; (void)cache;
    CHECK(il == HEAD_BLOCK_IL, "cache seed uses the native head slot");
    CHECK(rows <= DS4_QWEN4EXP_MTP_CACHE_SEED_MAX_ROWS, "bounded cache width");
    seed_calls++;
    seed_rows = rows;
    seed_pos = pos;
    if (seed_fail) return 0;
    memcpy(seed_seen, hyper->data, (size_t)rows * HEAD_HC_DIM * sizeof(float));
    return 1;
}

/* Independent input algebra: the test's enorm/hnorm are identities. Avoid
 * reading candidate scratch when checking token shift or device row offset. */
static void expected_eh(int token, const float *hc, float *want) {
    const float *eh = map_at(g_head_map, OFF_EH_PROJ);
    const float *emb = map_at(g_target_map, OFF_TOKEN_EMBD);
    for (uint32_t stream = 0; stream < HEAD_N_HC; stream++) {
        for (uint32_t out = 0; out < HEAD_N_EMBD; out++) {
            float acc = 0.0f;
            for (uint32_t i = 0; i < HEAD_N_EMBD; i++)
                acc += emb[(size_t)token * HEAD_N_EMBD + i] * eh[i * HEAD_N_EMBD + out];
            for (uint32_t i = 0; i < HEAD_N_EMBD; i++)
                acc += hc[stream * HEAD_N_EMBD + i] * eh[(HEAD_N_EMBD + i) * HEAD_N_EMBD + out];
            want[stream * HEAD_N_EMBD + out] = acc;
        }
    }
}

static void clear_calls(void) {
    seed_calls = 0;
    seed_rows = seed_pos = 0;
    seed_fail = 0;
    memset(&g_log, 0, sizeof(g_log));
}

static unsigned prime_block_calls;
static int prime_block_fail;
static float prime_seen[HEAD_HC_DIM];

/* The host block stands in for cache publication, as in the established
 * wiring test. Actual K/V and indexer bytes require the GPU oracle. */
static int prime_block(void *graph, void *cache, ds4_gpu_tensor *hyper,
                       uint32_t il, uint32_t pos, uint32_t rows) {
    prime_block_calls++;
    CHECK(rows == 1u, "tail priming stays on the one-row native path");
    memcpy(prime_seen, hyper->data, sizeof prime_seen);
    if (prime_block_fail) return 0;
    return stub_block(graph, cache, hyper, il, pos, rows);
}

static void prime_skip(ds4_qwen4exp_mtp_state *st, ds4_qwen4exp_mtp_head *h,
        int token, uint32_t pos, int budget, int cap,
        uint32_t ctx, uint32_t batch, const char *why) {
    ds4_qwen4exp_mtp_state saved_st;
    ds4_qwen4exp_mtp_head saved_h;
    memcpy(&saved_st, st, sizeof saved_st);
    memcpy(&saved_h, h, sizeof saved_h);
    clear_calls();
    prime_block_calls = 0;
    CHECK(ds4_qwen4exp_mtp_prime_cache_tail(st, h, token, pos, budget, cap,
                ctx, batch, g_err, sizeof(g_err)) == 0, "%s must fall back", why);
    CHECK(!prime_block_calls && !seed_calls && !g_log.n_log,
          "%s performs no head arithmetic", why);
    CHECK(memcmp(&saved_st, st, sizeof saved_st) == 0 &&
          memcmp(&saved_h, h, sizeof saved_h) == 0,
          "%s preserves pending, frontier, cache policy and all counters", why);
}

static void test_tail_prime(ds4_qwen4exp_mtp_head *h,
                           ds4_gpu_tensor *source, const float *inputs) {
    refmodel target;
    ds4_qwen4exp_mtp_model model;
    ds4_qwen4exp_rollback_set rollback;
    ds4_qwen4exp_mtp_state st;
    ref_reset(&target, BREAK_NONE, 0);
    CHECK(ref_build(&target, &model, &rollback) == 0, "build rollback contract");
    CHECK(ds4_qwen4exp_mtp_state_init(&st, 1, &rollback, HEAD_HC_DIM,
                HEAD_N_VOCAB, g_err, sizeof(g_err)) == 0, "init prime state");
    const uint32_t pos = 55u;
    const int actual = 7;
    st.head_rows = pos - 1u;
    memset(st.hc_scratch, 0xa5,
           (size_t)DS4_QWEN4EXP_MTP_HC_ROWS * HEAD_HC_DIM * sizeof(float));
    memset(st.logits_rows, 0,
           (size_t)DS4_QWEN4EXP_MTP_MAX_COMMIT * HEAD_N_VOCAB * sizeof(float));
    st.logits_rows[2] = 1.0f;
    CHECK(actual != ds4_qwen4exp_mtp_argmax(st.logits_rows, HEAD_N_VOCAB),
          "actual input differs from the available target argmax");
    float hc_before[DS4_QWEN4EXP_MTP_HC_ROWS * HEAD_HC_DIM];
    float logits_before[DS4_QWEN4EXP_MTP_MAX_COMMIT * HEAD_N_VOCAB];
    memcpy(hc_before, st.hc_scratch, sizeof hc_before);
    memcpy(logits_before, st.logits_rows, sizeof logits_before);
    CHECK(ds4_qwen4exp_mtp_head_retain_cache_tail(h, source, 3u, pos - 1u,
                -1, g_err, sizeof(g_err)) == 0, "retain nonzero unknown tail");
    h->hooks.block = prime_block;
    prime_block_fail = 0;
    CHECK(!h->cache_tail_prime_disabled, "default initialization enables priming");

    prime_skip(&st, h, actual, pos, 1, 2, 128u, 2u, "budget one");
    prime_skip(&st, h, actual, pos, 2, 1, 128u, 2u, "capacity one");
    prime_skip(&st, h, actual, pos, 2, 2, pos + 1u, 2u, "context has one row");
    prime_skip(&st, h, actual, pos, 2, 2, 128u, 1u, "batch has one row");
    prime_skip(&st, h, actual, UINT32_MAX, 2, 2, UINT32_MAX, 2u, "context overflow");
    prime_skip(&st, h, actual, 0u, 2, 2, 128u, 2u, "no previous token");
    h->cache_tail_valid = false;
    prime_skip(&st, h, actual, pos, 2, 2, 128u, 2u, "absent retained tail");
    h->cache_tail_valid = true;
    h->cache_seed_capacity = 0u;
    prime_skip(&st, h, actual, pos, 2, 2, 128u, 2u, "cache disabled");
    h->cache_seed_capacity = 5u;
    h->cache_tail_next_token = actual;
    prime_skip(&st, h, actual, pos, 2, 2, 128u, 2u, "already known tail");
    h->cache_tail_next_token = -1;
    st.head_rows--;
    prime_skip(&st, h, actual, pos, 2, 2, 128u, 2u, "incomplete head prefix");
    st.head_rows++;
    h->cache_tail_pos--;
    prime_skip(&st, h, actual, pos, 2, 2, 128u, 2u, "stale tail position");
    h->cache_tail_pos++;
    st.depth = 2;
    prime_skip(&st, h, actual, pos, 2, 2, 128u, 2u, "deeper chain");
    st.depth = 1;
    st.pending[0] = 4; st.n_pending = 1; st.pending_parent = 3;
    prime_skip(&st, h, actual, pos, 2, 2, 128u, 2u, "changed parent with carried chain");
    ds4_qwen4exp_mtp_invalidate(&st);
    CHECK(ds4_qwen4exp_mtp_prime_cache_tail(&st, h, -1, pos, 2, 2, 128u, 2u,
                g_err, sizeof(g_err)) < 0, "invalid actual token refuses");

    /* Separate ordinary full-head reference, with the same one-row inputs.
     * The independent algebra additionally checks the input token and HC. */
    ds4_qwen4exp_mtp_head reference;
    CHECK(build_shortlist_head(&reference) == 0, "build full-head reference");
    reference.hooks.block = prime_block;
    int expected = -1;
    clear_calls(); prime_block_calls = 0;
    CHECK(ds4_qwen4exp_mtp_head_forward(&reference, &actual,
                inputs + 3u * HEAD_HC_DIM, pos - 1u, 1u, &expected, NULL,
                g_err, sizeof(g_err)) == 0, "run ordinary one-row reference");
    float reference_eh[HEAD_HC_DIM], want[HEAD_HC_DIM], full_logits[HEAD_N_VOCAB];
    memcpy(reference_eh, prime_seen, sizeof reference_eh);
    expected_eh(actual, inputs + 3u * HEAD_HC_DIM, want);
    oracle_logits(&actual, inputs + 3u * HEAD_HC_DIM, 0u, full_logits);
    CHECK(memcmp(want, reference_eh, sizeof want) == 0 &&
          expected == shortlist_argmax(full_logits, h->draft_vocab_prefix,
                                       h->draft_vocab_tail),
          "independent head algebra establishes the one-row reference");
    ds4_qwen4exp_mtp_head_free(&reference);

    const ds4_qwen4exp_mtp_counters before = st.counters;
    clear_calls(); prime_block_calls = 0;
    CHECK(ds4_qwen4exp_mtp_prime_cache_tail(&st, h, actual, pos, 2, 2,
                pos + 2u, 2u, g_err, sizeof(g_err)) == 1,
          "exact context boundary primes from the actual caller token");
    CHECK(prime_block_calls == 1 && !seed_calls && g_log.block_pos0 == pos - 1u &&
          g_log.block_tokens == 1u && g_log.mixer_rows == 1u,
          "one complete head runs at P-1, with no duplicate cache-only call");
    CHECK(st.pending[0] == expected && st.n_pending == 1 &&
          st.pending_parent == actual && st.head_rows == pos &&
          h->cache_tail_next_token == actual,
          "proposal and parent/cache ownership publish only after the real head");
    CHECK(memcmp(prime_seen, reference_eh, sizeof reference_eh) == 0 &&
          memcmp(h->t_cache_tail->data, inputs + 3u * HEAD_HC_DIM, sizeof want) == 0,
          "native one-row input/cache stub equals reference and retained target stays immutable");
    CHECK(memcmp(hc_before, st.hc_scratch, sizeof hc_before) == 0 &&
          memcmp(logits_before, st.logits_rows, sizeof logits_before) == 0,
          "priming never uses or overwrites target HC/logit scratch");
    ds4_qwen4exp_mtp_counters after = st.counters;
    CHECK(after.draft_ns >= before.draft_ns, "head elapsed work is accounted");
    after.draft_ns = before.draft_ns;
    CHECK(memcmp(&after, &before, sizeof before) == 0,
          "priming changes no rounds, drafts, hits, commits or other phase counters");
    prime_skip(&st, h, actual, pos, 2, 2, 128u, 2u, "repeat cannot prime twice");
    bool changed = true;
    clear_calls();
    CHECK(ds4_qwen4exp_mtp_head_feed_cache_tail(h, actual, pos, &changed,
                g_err, sizeof(g_err)) == 0 && !changed && !seed_calls,
          "ordinary tail feed preserves the primed row");

    ds4_qwen4exp_mtp_invalidate(&st);
    prime_skip(&st, h, 3, pos, 2, 2, 128u, 2u, "known tail with changed parent");
    CHECK(ds4_qwen4exp_mtp_head_feed_cache_tail(h, 3, pos, &changed,
                g_err, sizeof(g_err)) == 0 && changed,
          "changed parent still takes the existing cache repair path");
    expected_eh(3, inputs + 3u * HEAD_HC_DIM, want);
    CHECK(memcmp(seed_seen, want, sizeof want) == 0,
          "changed-parent repair uses the selected target tail");

    ds4_qwen4exp_mtp_head_reset_cache(h);
    st.head_rows = 0u;
    prime_skip(&st, h, actual, 1u, 2, 2, 128u, 2u, "reset has no inherited tail");
    CHECK(ds4_qwen4exp_mtp_head_retain_cache_tail(h, source, 4u, 0u, -1,
                g_err, sizeof(g_err)) == 0, "retain a different request tail");
    prime_block_fail = 1;
    CHECK(ds4_qwen4exp_mtp_prime_cache_tail(&st, h, 3, 1u, 2, 2, 128u, 2u,
                g_err, sizeof(g_err)) < 0, "full-head failure propagates");
    CHECK(st.n_pending == 0 && st.pending_parent == -1 && st.head_rows == 0u &&
          h->cache_tail_next_token == -1, "failure publishes no pending proposal");
    prime_block_fail = 0;
    ds4_qwen4exp_mtp_head_reset_cache(h);
    CHECK(ds4_qwen4exp_mtp_head_retain_cache_tail(h, source, 4u, 0u, -1,
                g_err, sizeof(g_err)) == 0, "reset and retain after failed request");
    clear_calls();
    CHECK(ds4_qwen4exp_mtp_prime_cache_tail(&st, h, 3, 1u, 2, 2, 128u, 2u,
                g_err, sizeof(g_err)) == 1 && st.pending_parent == 3,
          "new request primes from its own retained tail");

    /* The diagnostic control is resolved at initialization, not per call. */
    setenv("DS4_MTP_NO_TAIL_PRIME", "1", 1);
    ds4_qwen4exp_mtp_head control;
    CHECK(build_shortlist_head(&control) == 0, "build disabled control");
    ds4_qwen4exp_mtp_head_free(&control);
    control.cache_seed_capacity = 1u;
    control.hooks.cache_seed = seed_mock;
    CHECK(ds4_qwen4exp_mtp_head_init(&control, g_err, sizeof(g_err)) == 0,
          "initialize disabled control cache");
    CHECK(control.cache_tail_prime_disabled, "control flag captured at init");
    unsetenv("DS4_MTP_NO_TAIL_PRIME");
    ds4_qwen4exp_mtp_invalidate(&st); st.head_rows = 0u;
    CHECK(ds4_qwen4exp_mtp_head_retain_cache_tail(&control, source, 0u, 0u, -1,
                g_err, sizeof(g_err)) == 0, "retain disabled control tail");
    prime_skip(&st, &control, 3, 1u, 2, 2, 128u, 2u, "latched diagnostic control");
    ds4_qwen4exp_mtp_head_free(&control);
    h->hooks.block = stub_block;
    ds4_qwen4exp_mtp_head_reset_cache(h);
    ds4_qwen4exp_mtp_state_free(&st);
}

/* Keep real row capacities while the independent algebra stays small. The
 * native-shape allocation branch is covered by test_mtp_native_hooks.c; this
 * fixture supplies its reserved buffers explicitly to test state ownership. */
static void adaptive_cache_lifetime_contract(void) {
    enum { OUTPUT_ROWS = DS4_QWEN4EXP_MTP_MAX_COMMIT,
           CACHE_ROWS = DS4_QWEN4EXP_MTP_CACHE_SEED_MAX_ROWS,
           SOURCE_ROWS = CACHE_ROWS + 2u };
    ds4_qwen4exp_mtp_head h;
    if (build_shortlist_head(&h) != 0) {
        CHECK(0, "adaptive/cache head template: %s", g_err);
        return;
    }
    ds4_qwen4exp_mtp_head_free(&h);
    h.max_tokens = OUTPUT_ROWS;
    h.cache_seed_capacity = CACHE_ROWS;
    h.hooks.cache_seed = seed_mock;
    if (ds4_qwen4exp_mtp_head_init(&h, g_err, sizeof(g_err)) != 0) {
        CHECK(0, "adaptive/cache 7-output/128-input head: %s", g_err);
        return;
    }
    CHECK(h.max_tokens == 7u && h.cache_seed_capacity == 128u,
          "production row capacities stay independent");
    CHECK(h.t_hyper->bytes == CACHE_ROWS * HEAD_HC_DIM * sizeof(float) &&
          h.t_logits->bytes == OUTPUT_ROWS * HEAD_N_VOCAB * sizeof(float) &&
          h.t_sample->bytes == OUTPUT_ROWS * HEAD_N_EMBD * sizeof(float),
          "128-row input allocation leaves proposal output at seven rows");
    ds4_gpu_tensor_free(h.t_logits_prefix);
    h.t_logits_prefix = ds4_gpu_tensor_alloc(OUTPUT_ROWS * 6u * sizeof(float));
    h.t_native_scratch = ds4_gpu_tensor_alloc(32u);
    h.t_native_ids = ds4_gpu_tensor_alloc(16u);
    h.native_capacity = 4u;
    h.draft_vocab_prefix_capacity = 6u;
    if (!h.t_logits_prefix || !h.t_native_scratch || !h.t_native_ids) {
        CHECK(0, "adaptive/cache reserved fixture allocations");
        ds4_qwen4exp_mtp_head_free(&h);
        return;
    }

    float inputs[SOURCE_ROWS * HEAD_HC_DIM], original[SOURCE_ROWS * HEAD_HC_DIM];
    int next[CACHE_ROWS];
    for (unsigned i = 0; i < SOURCE_ROWS * HEAD_HC_DIM; i++)
        inputs[i] = (int)(i % 23u) * 0.03125f - 0.375f;
    for (unsigned i = 0; i < CACHE_ROWS; i++) next[i] = (int)(i % HEAD_N_VOCAB);
    memcpy(original, inputs, sizeof inputs);
    ds4_gpu_tensor source = {sizeof inputs, (unsigned char *)inputs, 1};
    const int excluded = 5;
    int draft = -1;
    clear_calls();
    CHECK(ds4_qwen4exp_mtp_head_forward(&h, &excluded, inputs, 24u, 1u,
            &draft, NULL, g_err, sizeof(g_err)) == 0,
          "ordinary proposal forward activates the reserved prefix");
    CHECK(h.draft_vocab_prefix == 6u && h.draft_vocab_prefix_initial == 5u,
          "ordinary proposal expands while preserving initial policy");
    CHECK(g_log.block_tokens == 1u && g_log.mixer_rows == 1u && seed_calls == 0u,
          "expansion was exercised by full head arithmetic, not a seed call");
    float logits[HEAD_N_VOCAB];
    oracle_logits(&excluded, inputs, 0u, logits);
    CHECK(memcmp(h.t_logits->data, logits, sizeof logits) == 0,
          "expanded ordinary proposal preserves every independent full-vocabulary logit");
    CHECK(draft == shortlist_argmax(logits, 6u, 2u),
          "expanded ordinary proposal agrees with independent full algebra");

    CHECK(ds4_qwen4exp_mtp_head_retain_cache_tail(&h, &source, CACHE_ROWS,
            511u, 2, g_err, sizeof(g_err)) == 0,
          "retain target tail before wide cache publication");
    float saved_tail[HEAD_HC_DIM];
    memcpy(saved_tail, inputs + CACHE_ROWS * HEAD_HC_DIM, sizeof saved_tail);
    unsigned char saved_logits[OUTPUT_ROWS * HEAD_N_VOCAB * sizeof(float)];
    memcpy(saved_logits, h.t_logits->data, sizeof saved_logits);
    ds4_gpu_tensor *tail_ptr = h.t_cache_tail, *input_ptr = h.t_hyper;
    ds4_gpu_tensor *prefix_ptr = h.t_logits_prefix, *scratch_ptr = h.t_native_scratch;
    ds4_gpu_tensor *ids_ptr = h.t_native_ids;

    clear_calls();
    CHECK(ds4_qwen4exp_mtp_head_seed_cache(&h, next, &source, 1u, 256u,
            CACHE_ROWS, g_err, sizeof(g_err)) == 0,
          "128-row cache seed runs after vocabulary expansion");
    CHECK(seed_calls == 1u && seed_rows == CACHE_ROWS && seed_pos == 256u &&
          g_log.n_mm == 1 && g_log.mm_offset[0] == OFF_EH_PROJ &&
          g_log.mm_ntok[0] == (uint64_t)CACHE_ROWS * HEAD_N_HC &&
          g_log.mixer_rows == 0u,
          "wide seed reaches cache hook after one native-width EH and no output tail");
    for (unsigned row = 0; row < CACHE_ROWS; row++) {
        float want[HEAD_HC_DIM];
        expected_eh(next[row], inputs + (row + 1u) * HEAD_HC_DIM, want);
        CHECK(memcmp(want, seed_seen + row * HEAD_HC_DIM, sizeof want) == 0,
              "128-row seed preserves independent shifted input algebra at row %u", row);
    }
    CHECK(h.draft_vocab_prefix == 6u && h.draft_vocab_prefix_initial == 5u &&
          h.draft_vocab_prefix_capacity == 6u,
          "wide cache seed preserves expanded policy");
    CHECK(h.cache_tail_valid && h.cache_tail_pos == 511u && h.cache_tail_next_token == 2 &&
          memcmp(h.t_cache_tail->data, saved_tail, sizeof saved_tail) == 0,
          "wide cache seed preserves retained-tail metadata and content");
    CHECK(memcmp(inputs, original, sizeof inputs) == 0 &&
          memcmp(saved_logits, h.t_logits->data, sizeof saved_logits) == 0,
          "wide cache seed leaves target input and seven-row logits untouched");

    clear_calls();
    ds4_qwen4exp_mtp_head_reset_cache(&h);
    ds4_qwen4exp_mtp_head_reset_cache(&h);
    CHECK(!h.cache_tail_valid && h.cache_tail_pos == 0u && h.cache_tail_next_token == -1 &&
          h.draft_vocab_prefix == 6u,
          "cache reset clears retained validity without narrowing an expanded vocabulary");
    CHECK(h.t_cache_tail == tail_ptr && h.t_hyper == input_ptr &&
          h.t_logits_prefix == prefix_ptr && h.t_native_scratch == scratch_ptr &&
          h.t_native_ids == ids_ptr && h.cache_seed_capacity == CACHE_ROWS &&
          memcmp(h.t_cache_tail->data, saved_tail, sizeof saved_tail) == 0,
          "cache reset preserves input, prefix, native reserve and tail allocations and tail bytes");

    CHECK(ds4_qwen4exp_mtp_head_retain_cache_tail(&h, &source, CACHE_ROWS,
            767u, 3, g_err, sizeof(g_err)) == 0,
          "retain a valid tail before vocabulary reset");
    ds4_qwen4exp_mtp_head_reset_vocab(&h);
    ds4_qwen4exp_mtp_head_reset_vocab(&h);
    CHECK(h.draft_vocab_prefix == 5u && h.draft_vocab_prefix_initial == 5u &&
          h.draft_vocab_prefix_capacity == 6u,
          "vocabulary reset restores initial prefix while keeping the reserve");
    CHECK(h.cache_tail_valid && h.cache_tail_pos == 767u && h.cache_tail_next_token == 3 &&
          h.t_cache_tail == tail_ptr && h.t_hyper == input_ptr &&
          h.t_logits_prefix == prefix_ptr && h.t_native_scratch == scratch_ptr &&
          h.t_native_ids == ids_ptr && h.cache_seed_capacity == CACHE_ROWS &&
          memcmp(h.t_cache_tail->data, saved_tail, sizeof saved_tail) == 0,
          "vocabulary reset preserves valid retained tail, metadata, content and pointers");
    CHECK(seed_calls == 0u && g_log.n_mm == 0 && g_log.mixer_rows == 0u,
          "neither reset launches head arithmetic");

    clear_calls();
    CHECK(ds4_qwen4exp_mtp_head_seed_cache(&h, next, &source, 1u, 256u,
            CACHE_ROWS, g_err, sizeof(g_err)) == 0 && h.draft_vocab_prefix == 5u,
          "128-row seed with excluded tokens does not re-expand a reset vocabulary");
    clear_calls();
    CHECK(ds4_qwen4exp_mtp_head_forward(&h, &excluded, inputs, 768u, 1u,
            &draft, NULL, g_err, sizeof(g_err)) == 0 && h.draft_vocab_prefix == 6u,
          "a later ordinary proposal re-expands the same head");
    ds4_qwen4exp_mtp_head_free(&h);
    CHECK(!h.cache_seed_capacity && !h.t_cache_tail && !h.cache_tail_valid &&
          !h.draft_vocab_prefix_capacity && !h.draft_vocab_prefix_initial &&
          !h.native_capacity && !h.t_native_scratch && !h.t_native_ids,
          "free clears both independent ownership sets");
}

int main(void) {
    unsetenv("DS4_MTP_NO_TAIL_PRIME");
    setenv("DS4_QWEN4EXP_DRAFT_VOCAB_PREFIX", "5", 1);
    setenv("DS4_QWEN4EXP_DRAFT_VOCAB_TAIL", "2", 1);
    ds4_qwen4exp_mtp_head h;
    CHECK(build_shortlist_head(&h) == 0, "build head template");
    ds4_qwen4exp_mtp_head_free(&h);
    h.cache_seed_capacity = 5u;
    h.hooks.cache_seed = seed_mock;
    clear_calls();
    CHECK(ds4_qwen4exp_mtp_head_init(&h, g_err, sizeof(g_err)) == 0, "init bounded cache workspace");
    CHECK(seed_calls == 0, "metadata initialization runs no head arithmetic");
    CHECK(h.max_tokens == HEAD_ROWS && h.cache_seed_capacity == 5u, "separate cache and output capacities");
    CHECK(h.t_hyper->bytes == 5u * HEAD_HC_DIM * sizeof(float), "EH input workspace grows");
    CHECK(h.t_ehx->bytes == 5u * HEAD_N_HC * 2u * HEAD_N_EMBD * sizeof(float), "EH pack workspace grows");
    CHECK(h.t_logits->bytes == HEAD_ROWS * HEAD_N_VOCAB * sizeof(float), "vocabulary workspace remains narrow");
    CHECK(h.t_logits_prefix->bytes == HEAD_ROWS * 5u * sizeof(float), "shortlist workspace remains narrow");
    CHECK(h.t_mix_normed->bytes == HEAD_ROWS * HEAD_HC_DIM * sizeof(float), "final mixer workspace remains narrow");
    CHECK(h.t_cache_tail->bytes == HEAD_HC_DIM * sizeof(float), "only one HC tail retained");

    /* Simulate the adaptive capacity eligibility branch without changing this
     * reduced fixture's dimensions. A seed token outside prefix must not
     * trigger the proposal-selection side effect. */
    h.draft_vocab_prefix_capacity = 6u;
    h.t_native_scratch = ds4_gpu_tensor_alloc(8);
    h.t_native_ids = ds4_gpu_tensor_alloc(8);
    float inputs[6u * HEAD_HC_DIM], original[6u * HEAD_HC_DIM];
    for (unsigned i = 0; i < 6u * HEAD_HC_DIM; i++) inputs[i] = (int)(i % 13u) * 0.0625f - 0.25f;
    memcpy(original, inputs, sizeof inputs);
    ds4_gpu_tensor source = {sizeof inputs, (unsigned char *)inputs, 1};
    float head_weights[HEAD_MAP_FLOATS], target_weights[TARGET_MAP_FLOATS];
    memcpy(head_weights, h.head_map, sizeof head_weights);
    memcpy(target_weights, h.target_map, sizeof target_weights);
    unsigned char saved_logits[HEAD_ROWS * HEAD_N_VOCAB * sizeof(float)];
    memset(h.t_logits->data, 0xa5, sizeof saved_logits);
    memcpy(saved_logits, h.t_logits->data, sizeof saved_logits);
    const int next[5] = {2, 5, 4, 7, 0};
    clear_calls();
    CHECK(ds4_qwen4exp_mtp_head_seed_cache(&h, next, &source, 1u, 40u, 5u,
                                           g_err, sizeof(g_err)) == 0, "seed nonzero prefix and source offset");
    CHECK(seed_calls == 1 && seed_rows == 5u && seed_pos == 40u, "cache-only hook receives every seed row");
    CHECK(g_log.n_mm == 1 && g_log.mm_offset[0] == OFF_EH_PROJ,
          "only EH projection runs, no vocabulary projection");
    CHECK(g_log.n_norm == 2 && g_log.mixer_rows == 0u, "enorm/hnorm run, final head mixer does not");
    for (unsigned row = 0; row < 5u; row++) {
        float want[HEAD_HC_DIM];
        expected_eh(next[row], inputs + (row + 1u) * HEAD_HC_DIM, want);
        CHECK(memcmp(want, seed_seen + row * HEAD_HC_DIM, sizeof want) == 0,
              "independent EH algebra preserves shifted token and HC offset at row %u", row);
    }
    CHECK(memcmp(inputs, original, sizeof inputs) == 0, "target HC source unchanged");
    CHECK(memcmp(head_weights, h.head_map, sizeof head_weights) == 0 &&
          memcmp(target_weights, h.target_map, sizeof target_weights) == 0, "both weight mappings unchanged");
    CHECK(memcmp(saved_logits, h.t_logits->data, sizeof saved_logits) == 0, "cache-only call leaves logits untouched");
    CHECK(h.draft_vocab_prefix == 5u && h.draft_vocab_prefix_initial == 5u &&
          h.draft_vocab_prefix_capacity == 6u, "cache seeding does not expand adaptive vocabulary");

    clear_calls();
    CHECK(ds4_qwen4exp_mtp_head_seed_cache(&h, next, &source, 2u, 40u, 5u,
                                           g_err, sizeof(g_err)) != 0 && seed_calls == 0,
          "out-of-bounds source fails before cache publication");
    clear_calls();
    CHECK(ds4_qwen4exp_mtp_head_seed_cache(&h, next, &source, 0u, 40u, 6u,
                                           g_err, sizeof(g_err)) != 0, "capacity overflow refused");
    CHECK(ds4_qwen4exp_mtp_head_seed_cache(&h, next, &source, 0u, UINT32_MAX, 1u,
                                           g_err, sizeof(g_err)) != 0, "position overflow refused");
    const int bad[] = {-1, HEAD_N_VOCAB};
    for (unsigned i = 0; i < 2u; i++)
        CHECK(ds4_qwen4exp_mtp_head_seed_cache(&h, bad + i, &source, 0u, 0u, 1u,
                                               g_err, sizeof(g_err)) != 0, "bad token refused");
    h.hooks.cache_seed = NULL;
    CHECK(ds4_qwen4exp_mtp_head_seed_cache(&h, next, &source, 0u, 0u, 1u,
                                           g_err, sizeof(g_err)) != 0, "absent optional cache hook refuses");
    h.hooks.cache_seed = seed_mock;
    clear_calls(); seed_fail = 1;
    CHECK(ds4_qwen4exp_mtp_head_seed_cache(&h, next, &source, 0u, 40u, 1u,
                                           g_err, sizeof(g_err)) != 0 && seed_calls == 1,
          "cache hook failure propagates");

    /* The last prompt HC row waits for its actual next token. Repeating the
     * same parent preserves an already written row; changing it recomputes
     * exactly that row and leaves the retained target input intact. */
    clear_calls();
    CHECK(ds4_qwen4exp_mtp_head_retain_cache_tail(&h, &source, 3u, 54u, -1,
                                                 g_err, sizeof(g_err)) == 0, "retain unknown-next prompt tail");
    bool changed = false;
    CHECK(ds4_qwen4exp_mtp_head_feed_cache_tail(&h, 3, 55u, &changed,
                                               g_err, sizeof(g_err)) == 0 && changed && seed_calls == 1,
          "actual first fed token completes tail");
    float want[HEAD_HC_DIM];
    expected_eh(3, inputs + 3u * HEAD_HC_DIM, want);
    CHECK(seed_pos == 54u && memcmp(want, seed_seen, sizeof want) == 0,
          "P-1 pairs with actual first token");
    clear_calls();
    CHECK(ds4_qwen4exp_mtp_head_feed_cache_tail(&h, 3, 55u, &changed,
                                               g_err, sizeof(g_err)) == 0 && !changed && seed_calls == 0,
          "same-parent continuation preserves cache row");
    CHECK(ds4_qwen4exp_mtp_head_feed_cache_tail(&h, 7, 55u, &changed,
                                               g_err, sizeof(g_err)) == 0 && changed && seed_calls == 1,
          "different actual token repairs only frontier cache row");
    expected_eh(7, inputs + 3u * HEAD_HC_DIM, want);
    CHECK(memcmp(want, seed_seen, sizeof want) == 0, "repair uses target HC, not previous head output");
    CHECK(memcmp(h.t_cache_tail->data, inputs + 3u * HEAD_HC_DIM, sizeof want) == 0,
          "retained target tail is immutable across completions");
    CHECK(ds4_qwen4exp_mtp_head_feed_cache_tail(&h, 7, 56u, &changed,
                                               g_err, sizeof(g_err)) != 0, "stale tail position refuses");

    clear_calls();
    CHECK(ds4_qwen4exp_mtp_head_retain_cache_tail(&h, &source, 4u, 70u, 2,
                                                 g_err, sizeof(g_err)) == 0, "record speculative selected target row");
    CHECK(ds4_qwen4exp_mtp_head_feed_cache_tail(&h, 2, 71u, &changed,
                                               g_err, sizeof(g_err)) == 0 && !changed && seed_calls == 0,
          "known speculative next token needs no extra cache forward");
    ds4_gpu_tensor *tail_ptr = h.t_cache_tail, *input_ptr = h.t_hyper;
    ds4_qwen4exp_mtp_head_reset_cache(&h);
    ds4_qwen4exp_mtp_head_reset_cache(&h);
    CHECK(!h.cache_tail_valid && h.cache_tail_next_token == -1 && h.cache_tail_pos == 0u,
          "repeated request reset clears all retained state");
    CHECK(h.t_cache_tail == tail_ptr && h.t_hyper == input_ptr && h.cache_seed_capacity == 5u,
          "reset preserves allocated workspace");
    CHECK(h.draft_vocab_prefix == 5u && h.draft_vocab_prefix_initial == 5u &&
          h.draft_vocab_prefix_capacity == 6u, "cache reset leaves vocabulary policy separate");
    CHECK(ds4_qwen4exp_mtp_head_feed_cache_tail(&h, 2, 1u, &changed,
                                               g_err, sizeof(g_err)) == 0 && !changed,
          "new request cannot inherit old tail");
    CHECK(ds4_qwen4exp_mtp_head_retain_cache_tail(&h, &source, 0u, 0u, -1,
                                                 g_err, sizeof(g_err)) == 0, "retain next request tail");
    clear_calls();
    CHECK(ds4_qwen4exp_mtp_head_feed_cache_tail(&h, 5, 1u, &changed,
                                               g_err, sizeof(g_err)) == 0 && changed && seed_calls == 1,
          "same head can seed a second request");

    int draft[HEAD_ROWS];
    clear_calls();
    CHECK(ds4_qwen4exp_mtp_head_forward(&h, next, inputs, 0u, HEAD_ROWS + 1u,
                draft, NULL, g_err, sizeof(g_err)) != 0,
          "wider input scratch does not widen ordinary output API");
    test_tail_prime(&h, &source, inputs);
    ds4_qwen4exp_mtp_head_free(&h);
    CHECK(!h.cache_seed_capacity && !h.t_cache_tail && !h.cache_tail_valid,
          "free clears prefix ownership");
    h.cache_seed_capacity = DS4_QWEN4EXP_MTP_CACHE_SEED_MAX_ROWS + 1u;
    CHECK(ds4_qwen4exp_mtp_head_init(&h, g_err, sizeof(g_err)) != 0,
          "unbounded seed workspace refused at init");
    h.cache_seed_capacity = 1u;
    h.hooks.cache_seed = NULL;
    CHECK(ds4_qwen4exp_mtp_head_init(&h, g_err, sizeof(g_err)) != 0,
          "requested seed workspace needs an actual hook");
    adaptive_cache_lifetime_contract();
    unsetenv("DS4_QWEN4EXP_DRAFT_VOCAB_PREFIX");
    unsetenv("DS4_QWEN4EXP_DRAFT_VOCAB_TAIL");
    if (g_failures) { printf("FAILED: %d cache-prefix checks\n", g_failures); return 1; }
    printf("all MTP prefix-cache host contracts passed\n");
    return 0;
}
