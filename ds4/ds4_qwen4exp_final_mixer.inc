/* The final mixer has no carried state. A last-logit caller consumes only
 * its final row, but CUDA must retain the prefill arithmetic: Q8 MMA starts
 * at eight rows, and narrowing to one would select the decode reduction.
 * Keep the last eight real rows, including when the input is not tile-aligned.
 * Full-logit verifies and other backends retain their existing widths.
 *
 * Settle the previous FFN inject over ALL rows first. MTP cache seeding reads
 * that entire hyper stream after the forward; narrowing the inject as well
 * would silently leave its earlier rows stale. The view owns no GPU memory.
 * DS4_QWEN4EXP_NO_FINAL_MIXER_TAIL restores the full-width path.
 * Returns the number of mixed rows, or zero on failure. */
static uint32_t qwen4exp_graph_final_mixer(ds4_qwen4exp_session *s,
        const ds4_qwen4exp_weights *w, const ds4_model *m, uint32_t n_tokens) {
    uint32_t rows = n_tokens;
#if !defined(__APPLE__) && !defined(DS4_ROCM_BUILD)
    if (!s->spec_logit_rows && n_tokens > 8u &&
        getenv("DS4_QWEN4EXP_NO_FINAL_MIXER_TAIL") == NULL) rows = 8u;
#endif
    if (rows == n_tokens) {
        return qwen4exp_graph_residual(s, m, w->output_hc_norm,
                   w->output_hc_down, w->output_hc_up, NULL,
                   w->output_hc_norm_offset, n_tokens) ? n_tokens : 0u;
    }
    if (!qwen4exp_hc_flush_pending(s, n_tokens)) return 0u;
    const uint64_t row_bytes = (uint64_t)DS4_N_HC * DS4_N_EMBD * sizeof(float);
    ds4_gpu_tensor *tail = ds4_gpu_tensor_view(s->hyper,
        (uint64_t)(n_tokens - rows) * row_bytes, (uint64_t)rows * row_bytes);
    if (!tail) return 0u;

#ifdef DS4_TEST_HOOKS
    /* The graph test compares the whole retained HC stream with this copy.
     * Preserve that diagnostic before the temporary view narrows the reader. */
    ds4_gpu_tensor *saved_snapshot = s->final_mixer_snapshot;
    if (saved_snapshot && !ds4_gpu_tensor_copy(saved_snapshot, 0, s->hyper, 0,
                                              (uint64_t)n_tokens * row_bytes)) {
        ds4_gpu_tensor_free(tail);
        return 0u;
    }
    s->final_mixer_snapshot = NULL;
#endif
    ds4_gpu_tensor *saved_hyper = s->hyper;
    s->hyper = tail;
    const bool ok = qwen4exp_graph_residual(s, m, w->output_hc_norm,
        w->output_hc_down, w->output_hc_up, NULL,
        w->output_hc_norm_offset, rows);
    s->hyper = saved_hyper;
#ifdef DS4_TEST_HOOKS
    s->final_mixer_snapshot = saved_snapshot;
#endif
    ds4_gpu_tensor_free(tail);
    return ok ? rows : 0u;
}
