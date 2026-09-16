/* Encode the MTP block once, either eagerly or into the decode graph stream.
 * All arguments that become kernel parameters are already stable here. */
static bool qwen4exp_graph_head_block_encode(
        ds4_qwen4exp_session *s,
        const ds4_qwen4exp_layer_weights *l,
        const ds4_model *m, uint32_t il, uint32_t n_tokens,
        ds4_gpu_tensor *last_hyper, const char **failed_stage) {
    const int timing = qw_hb_time_on();
    uint64_t tmark = timing ? qw_hb_now_ns() : 0;
    const char *stage = "attention mixer";
    bool ok = qwen4exp_graph_residual(s, m, l->hc_attn_norm, l->hc_attn_down,
                                      l->hc_attn_up, l->hc_attn_inject,
                                      l->hc_attn_norm_offset, n_tokens);
    QW_HB_TICK(QW_HB_T_ATTN_MIX);
    if (ok) { stage = "QSA"; ok = qwen4exp_graph_qsa_block(s, l, m, il, n_tokens); }
    QW_HB_TICK(QW_HB_T_QSA);
    if (ok) {
        stage = "attention inject";
        ok = ds4_gpu_qwen4exp_hc_inject_tensor(s->hyper, s->hyper, s->block_out,
                                               s->inject, DS4_N_EMBD, DS4_N_HC,
                                               n_tokens) != 0;
    }
    QW_HB_TICK(QW_HB_T_ATTN_INJECT);
    /* Attention and its caches keep the original batch and dispatch. Only
     * the independent FFN tail is narrowed, after the final row has consumed
     * the complete attention result. This moves the caller's existing copy. */
    ds4_gpu_tensor *batch_hyper = s->hyper;
    const uint32_t tail_rows = last_hyper ? 1u : n_tokens;
    if (ok && last_hyper) {
        stage = "last attention row";
        const uint64_t row_bytes =
            (uint64_t)DS4_N_HC * DS4_N_EMBD * sizeof(float);
        ok = ds4_gpu_tensor_copy(last_hyper, 0, batch_hyper,
                                  (uint64_t)(n_tokens - 1u) * row_bytes,
                                  row_bytes) != 0;
        if (ok) s->hyper = last_hyper;
    }
    if (ok) {
        stage = "FFN mixer";
        ok = qwen4exp_graph_residual(s, m, l->hc_ffn_norm, l->hc_ffn_down,
                                     l->hc_ffn_up, l->hc_ffn_inject,
                                     l->hc_ffn_norm_offset, tail_rows);
    }
    QW_HB_TICK(QW_HB_T_FFN_MIX);
    if (ok) {
        stage = "MoE";
        qw_moe_in_head_block = 1;
        ok = qwen4exp_graph_moe_block(s, l, m, il, tail_rows);
        qw_moe_in_head_block = 0;
    }
    QW_HB_TICK(QW_HB_T_MOE);
    if (ok) {
        stage = "FFN inject";
        ok = ds4_gpu_qwen4exp_hc_inject_tensor(s->hyper, s->hyper, s->block_out,
                                               s->inject, DS4_N_EMBD, DS4_N_HC,
                                               tail_rows) != 0;
    }
    QW_HB_TICK(QW_HB_T_FFN_INJECT);
    if (timing && ok) qw_hb_stage_calls++;

    s->hyper = batch_hyper;
    if (failed_stage) *failed_stage = stage;
    return ok;
}

/*
 * ds4_qwen4exp_block_forward_fn.  `graph` is the head's weight context and
 * `cache` is the session that owns the slot; `hyper` is the head's own stream,
 * rewritten in place.  Returns NONZERO on success -- the ds4_gpu convention,
 * not ds4.c's.
 *
 * Device-positioned QSA makes the complete head block position-independent.
 * Capturing it as island 3 removes the attention, mixer and MoE launch train
 * from every MTP draft round.  The session's hyper pointer and position are
 * lent to the head for the call and restored afterwards.
 */
static int qwen4exp_graph_head_block_impl(void *graph, void *cache,
        ds4_gpu_tensor *hyper, ds4_gpu_tensor *last_hyper, uint32_t il,
        uint32_t pos0, uint32_t n_tokens) {
    const ds4_qwen4exp_head_block_ctx *c = graph;
    ds4_qwen4exp_session *s = cache;
    if (!c || !s || !c->l || !c->m || !hyper || n_tokens == 0) return 0;
    if (n_tokens > s->plan.n_batch) return 0;
    if (last_hyper && (last_hyper == hyper ||
        n_tokens > (uint32_t)DS4_QWEN4EXP_MTP_MAX_COMMIT ||
        (uint64_t)pos0 + n_tokens > s->plan.n_ctx)) return 0;
    if (il >= DS4_MAX_LAYER || !s->qsa_k[il]) {
        fprintf(stderr,
                "ds4: qwen4exp head block %u has no cache slot; "
                "ds4_qwen4exp_session_add_head_block() was not called\n", il);
        return 0;
    }
    if (!c->l->is_full_attention) return 0;

    /* The head's QSA block appends to the slot's key/value caches and its
     * indexer tape; see `state_dirty`. */
    s->state_dirty = true;

    ds4_gpu_tensor *saved_hyper = s->hyper;
    const uint32_t saved_pos = s->pos;
    s->hyper = hyper;
    s->pos = pos0;
    if (s->d_pos && !ds4_gpu_qwen4exp_update_dpos(s->d_pos, s->pos)) {
        s->hyper = saved_hyper;
        s->pos = saved_pos;
        fprintf(stderr, "ds4: qwen4exp head position update failed\n");
        return 0;
    }

    const bool capture_ok =
        (!last_hyper || s->d_pos) &&
        n_tokens <= (uint32_t)DS4_QWEN4EXP_MTP_MAX_COMMIT &&
        (uint64_t)s->pos + n_tokens <= DS4_N_INDEXER_TOP_K &&
        !qw_hb_time_on() && getenv("DS4_QWEN4EXP_TIME_SLICES") == NULL &&
        ds4_gpu_decode_graphs_supported();
    ds4_decode_graph_key key;
    int graph_state = -1;
    if (capture_ok) {
        memset(&key, 0, sizeof(key));
        key.il = il;
        key.island = 3u;
        key.variant = n_tokens | (s->spec_snapshot_rows << 8u);
        key._pad = 0x4d545048u;
        key.cur_hc = hyper;
        key.after_attn_hc = s;
        key.after_ffn_hc = (void *)c->l;
        key.attn_norm = (void *)c->m;
        if (last_hyper) {
            /* The bound head context is immutable for this session. Include
             * both tensor identities: a reused output must never overwrite
             * a different caller's final row. Full-output keys stay intact. */
            key._pad = 0x4d54504cu; /* MTPL: batched attention, last FFN */
            key.after_ffn_hc = last_hyper;
            key.attn_norm = (void *)c;
        }
        graph_state = ds4_gpu_decode_graph_begin(&key);
    }

    const char *stage = "attention mixer";
    bool ok = graph_state == 1;
    if (graph_state != 1) {
        ok = qwen4exp_graph_head_block_encode(
                s, c->l, c->m, il, n_tokens, last_hyper, &stage);
        if (graph_state == 0) {
            if (ok && ds4_gpu_decode_graph_end(&key) == 0) {
                ok = true;
            } else {
                if (!ok) ds4_gpu_decode_graph_abort(&key);
                /* A failed capture executed no kernels.  Re-encode the block
                 * eagerly once so this proposal still advances its caches. */
                ok = qwen4exp_graph_head_block_encode(
                        s, c->l, c->m, il, n_tokens, last_hyper, &stage);
            }
        }
    }

    s->hyper = saved_hyper;
    s->pos = saved_pos;
    if (s->d_pos && !ds4_gpu_qwen4exp_update_dpos(s->d_pos, s->pos)) {
        stage = "position restore";
        ok = false;
    }
    if (!ok) {
        fprintf(stderr, "ds4: qwen4exp head block %u failed at the %s\n",
                il, stage);
    }
    return ok ? 1 : 0;
}


int ds4_qwen4exp_graph_head_block(void *graph, void *cache,
        ds4_gpu_tensor *hyper, uint32_t il, uint32_t pos0, uint32_t n_tokens) {
    return qwen4exp_graph_head_block_impl(graph, cache, hyper, NULL,
                                         il, pos0, n_tokens);
}

int ds4_qwen4exp_graph_head_block_last(void *graph, void *cache,
        ds4_gpu_tensor *hyper, ds4_gpu_tensor *last_hyper, uint32_t il,
        uint32_t pos0, uint32_t n_tokens) {
    if (!last_hyper) return 0;
    return qwen4exp_graph_head_block_impl(graph, cache, hyper, last_hyper,
                                         il, pos0, n_tokens);
}
