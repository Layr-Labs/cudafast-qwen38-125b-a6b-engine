#ifndef DS4_QWEN4EXP_GDN_REPLAY_H
#define DS4_QWEN4EXP_GDN_REPLAY_H

#include <stdbool.h>
#include <stdint.h>

#define DS4_QWEN4EXP_GDN_REPLAY_ROWS 2u
#define DS4_QWEN4EXP_GDN_DEFER_TAPE_ROWS 4u

/* A four-slot input ring. At most three committed transitions follow the
 * checkpoint. Before appending at that length, row zero publishes a new
 * checkpoint and only row one's input is recorded in the one free slot. */
static inline uint32_t ds4_qwen4exp_gdn_deferred_control(
        uint32_t start, uint32_t rows) {
    return (start << 2u) | rows;
}

static inline uint32_t ds4_qwen4exp_gdn_deferred_after(
        uint32_t start, uint32_t prefix, uint32_t committed) {
    if (prefix >= DS4_QWEN4EXP_GDN_REPLAY_ROWS) {
        start = (start + prefix) & (DS4_QWEN4EXP_GDN_DEFER_TAPE_ROWS - 1u);
        return ds4_qwen4exp_gdn_deferred_control(start, committed - 1u);
    }
    return ds4_qwen4exp_gdn_deferred_control(start, prefix + committed);
}

/* Host transition policy, independent of CUDA. A replay verify leaves a
 * virtual row-zero snapshot: checkpoint + its bounded transition log. */
typedef struct {
    bool active;
    bool settle;
    bool swap;
    uint32_t prefix;
    uint32_t start;
    bool materialize;
} ds4_qwen4exp_gdn_replay_step;

static inline ds4_qwen4exp_gdn_replay_step ds4_qwen4exp_gdn_replay_plan(
        bool enabled, bool previous, uint32_t old_prefix, uint32_t capacity,
        uint32_t width, uint32_t snapshots, uint32_t max_adopt_width,
        uint32_t recurrent, uint32_t conv) {
    ds4_qwen4exp_gdn_replay_step p = {false, false, false, 0u, 0u, false};
    p.active = enabled && width == 2u && snapshots == 1u;
    const bool reuse = p.active && previous && recurrent == 1u && conv == 1u;
    p.settle = (recurrent != 0u || conv != 0u) &&
        (width > max_adopt_width || recurrent != conv ||
         (p.active && !reuse) || (!p.active && previous && recurrent != 0u));
    p.swap = p.active && !reuse;
    p.prefix = reuse ? (old_prefix == capacity ? 0u : old_prefix + 1u) : 0u;
    return p;
}

static inline ds4_qwen4exp_gdn_replay_step ds4_qwen4exp_gdn_deferred_plan(
        bool enabled, bool previous, uint32_t old_prefix, uint32_t old_start,
        uint32_t width, uint32_t snapshots, uint32_t max_adopt_width,
        uint32_t recurrent, uint32_t conv) {
    ds4_qwen4exp_gdn_replay_step p = {false, false, false, 0u, 0u, false};
    p.active = enabled && width == 2u && snapshots == 1u;
    const bool reuse = p.active && previous && recurrent == conv && recurrent <= 1u;
    p.settle = (recurrent != 0u || conv != 0u) &&
        (width > max_adopt_width || recurrent != conv ||
         (p.active && !reuse) || (!p.active && previous && recurrent != 0u));
    p.materialize = previous && !reuse;
    p.swap = p.active && !reuse;
    if (reuse) {
        const uint32_t control = ds4_qwen4exp_gdn_deferred_after(
            old_start, old_prefix, recurrent == 1u ? 1u : 2u);
        p.prefix = control & 3u;
        p.start = control >> 2u;
    }
    return p;
}

/* Physical recurrent-buffer addresses alternate even on an ordinary forward
 * after replay. Both parity and kernel choice must identify a graph. The log
 * length is device data, so it deliberately does not multiply graph entries. */
static inline uint32_t ds4_qwen4exp_gdn_graph_variant(
        uint32_t width, uint32_t snapshots, uint32_t phase, bool active,
        bool deferred) {
    return width | (snapshots << 8u) | (phase << 16u) |
           ((uint32_t)active << 17u) | ((uint32_t)deferred << 18u);
}

#endif
