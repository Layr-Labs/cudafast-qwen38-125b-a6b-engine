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

int main(void) {
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

    /* Seed tokens may lie outside the proposal shortlist. Cache publication
     * must preserve the prefix and tail selected when the head was built. */
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
    CHECK(h.draft_vocab_prefix == 5u && h.draft_vocab_tail == 2u,
          "cache seeding leaves the proposal vocabulary unchanged");

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
    CHECK(h.draft_vocab_prefix == 5u && h.draft_vocab_tail == 2u,
          "cache reset leaves the proposal vocabulary unchanged");
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
    unsetenv("DS4_QWEN4EXP_DRAFT_VOCAB_PREFIX");
    unsetenv("DS4_QWEN4EXP_DRAFT_VOCAB_TAIL");
    if (g_failures) { printf("FAILED: %d cache-prefix checks\n", g_failures); return 1; }
    printf("all MTP prefix-cache host contracts passed\n");
    return 0;
}
