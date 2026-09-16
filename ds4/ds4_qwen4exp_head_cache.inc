/* Native head-cache dispatch. Included after the unchanged cache encoder in
 * ds4_qwen4exp_graph.inc. The host test includes this same production wrapper
 * with a mock graph backend to check capture/replay and position lifetimes. */
int ds4_qwen4exp_graph_head_cache(void *graph, void *cache,
        ds4_gpu_tensor *hyper, uint32_t il, uint32_t pos0, uint32_t n_tokens) {
    const ds4_qwen4exp_head_block_ctx *c = graph;
    ds4_qwen4exp_session *s = cache;
    if (!c || !s || !c->l || !c->m || !hyper || !n_tokens ||
        n_tokens > s->plan.n_batch || il >= DS4_MAX_LAYER ||
        il < g_ds4_qwen4exp.n_layer || !s->qsa_k[il] ||
        !c->l->is_full_attention || s->hc_pending ||
        (uint64_t)pos0 + n_tokens > s->plan.n_ctx) return 0;
    ds4_gpu_tensor *saved_hyper = s->hyper;
    const uint32_t saved_pos = s->pos;
    s->state_dirty = true;
    s->hyper = hyper;
    s->pos = pos0;
    bool ok = !s->d_pos || ds4_gpu_qwen4exp_update_dpos(s->d_pos, s->pos);
    /* This is the full head block's existing decode capture envelope.
     * Require device positions so every replay follows the current call.
     * Wide prompt-cache publication keeps its eager path. */
    const bool capture_ok = s->d_pos &&
        n_tokens <= (uint32_t)DS4_QWEN4EXP_MTP_MAX_COMMIT &&
        (uint64_t)pos0 + n_tokens <= DS4_N_INDEXER_TOP_K &&
        !qw_hb_time_on() && getenv("DS4_QWEN4EXP_TIME_SLICES") == NULL &&
        ds4_gpu_decode_graphs_supported();
    ds4_decode_graph_key key;
    int graph_state = -1;
    if (ok && capture_ok) {
        memset(&key, 0, sizeof(key));
        key.il = il;
        /* Head slots are above the target layers. Their island 2 is unused
         * by target QSA; island 3 remains the complete proposal block. */
        key.island = 2u;
        key.variant = n_tokens;
        key._pad = 0x4d545043u; /* MTPC: native head-cache publication */
        key.cur_hc = hyper;
        key.after_attn_hc = s;
        key.after_ffn_hc = (void *)c->l;
        key.attn_norm = (void *)c->m;
        graph_state = ds4_gpu_decode_graph_begin(&key);
    }
    if (ok && graph_state != 1) {
        ok = qwen4exp_graph_head_cache_encode(s, c->l, c->m, il, n_tokens);
        if (graph_state == 0) {
            if (ok && ds4_gpu_decode_graph_end(&key) == 0) {
                ok = true;
            } else {
                if (!ok) ds4_gpu_decode_graph_abort(&key);
                /* Capture records but executes nothing until commit. As in
                 * the complete head block, failed capture falls back once. */
                ok = qwen4exp_graph_head_cache_encode(s, c->l, c->m, il, n_tokens);
            }
        }
    }
    s->hyper = saved_hyper;
    s->pos = saved_pos;
    if (s->d_pos && !ds4_gpu_qwen4exp_update_dpos(s->d_pos, s->pos)) ok = false;
    if (ok) s->head_cache_pos = pos0 + n_tokens;
    if (!ok) fprintf(stderr, "ds4: qwen4exp native head cache publication failed\n");
    return ok ? 1 : 0;
}
