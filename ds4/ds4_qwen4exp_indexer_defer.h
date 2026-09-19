#ifndef DS4_QWEN4EXP_INDEXER_DEFER_H
#define DS4_QWEN4EXP_INDEXER_DEFER_H

/* Yukon redraw identity: 2026-09-19T19:56Z. */

#include <stdbool.h>
#include <stdint.h>

/* Host policy for the CUDA indexer-K staging buffer.  Before the indexer
 * budget no reader observes the projected keys, so the graph stores the exact
 * f32 mixer input and delays the BF16 projection and pool construction. */
typedef struct {
    uint32_t rows;          /* contiguous staged prefix [0, rows) */
    bool materialized;      /* tape/pool have replaced the staged prefix */
} ds4_qwen4exp_indexer_defer_state;

typedef enum {
    DS4_QWEN4EXP_INDEXER_EAGER = 0,
    DS4_QWEN4EXP_INDEXER_DEFER = 1,
    DS4_QWEN4EXP_INDEXER_MATERIALIZE_EAGER = 2,
} ds4_qwen4exp_indexer_defer_action;

static inline ds4_qwen4exp_indexer_defer_action
ds4_qwen4exp_indexer_defer_plan(
        const ds4_qwen4exp_indexer_defer_state *state,
        bool available, uint32_t pos, uint32_t width, uint32_t budget) {
    if (!available || !state || state->materialized) {
        return DS4_QWEN4EXP_INDEXER_EAGER;
    }
    if (state->rows == pos && (uint64_t)pos + width <= budget) {
        return DS4_QWEN4EXP_INDEXER_DEFER;
    }
    /* Crossing the budget and a discontinuous seed both need the prefix that
     * really exists materialized before the current rows take the eager path.
     * A forward gap remains zero exactly as it did in the eager cache. */
    return DS4_QWEN4EXP_INDEXER_MATERIALIZE_EAGER;
}

static inline bool ds4_qwen4exp_indexer_defer_commit(
        ds4_qwen4exp_indexer_defer_state *state,
        uint32_t pos, uint32_t width) {
    if (!state || state->materialized || state->rows != pos ||
        width > UINT32_MAX - pos) return false;
    state->rows = pos + width;
    return true;
}

/* Commit the host shadow for an outer graph that covers several QSA layers.
 * Validate every selected layer first: a failed replay check must not advance
 * only the prefix of the range and make the next fallback discontinuous. */
static inline bool ds4_qwen4exp_indexer_defer_commit_layers(
        ds4_qwen4exp_indexer_defer_state *states, uint32_t n_states,
        const uint32_t *layers, uint32_t n_layers,
        uint32_t pos, uint32_t width) {
    if (!states || (!layers && n_layers != 0u) ||
        width > UINT32_MAX - pos) return false;
    for (uint32_t i = 0; i < n_layers; i++) {
        if (layers[i] >= n_states || states[layers[i]].materialized ||
            states[layers[i]].rows != pos) return false;
    }
    for (uint32_t i = 0; i < n_layers; i++) {
        states[layers[i]].rows = pos + width;
    }
    return true;
}

static inline void ds4_qwen4exp_indexer_defer_materialized(
        ds4_qwen4exp_indexer_defer_state *state) {
    if (!state) return;
    state->rows = 0u;
    state->materialized = true;
}

static inline void ds4_qwen4exp_indexer_defer_reset(
        ds4_qwen4exp_indexer_defer_state *state) {
    if (!state) return;
    state->rows = 0u;
    state->materialized = false;
}

static inline void ds4_qwen4exp_indexer_defer_rollback(
        ds4_qwen4exp_indexer_defer_state *state, uint32_t pos) {
    if (!state || state->materialized) return;
    if (pos < state->rows) state->rows = pos;
}

#endif /* DS4_QWEN4EXP_INDEXER_DEFER_H */
