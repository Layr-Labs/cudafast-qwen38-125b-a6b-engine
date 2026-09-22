#ifndef DS4_QWEN4EXP_GDN_REPLAY_H
#define DS4_QWEN4EXP_GDN_REPLAY_H

#include <stdbool.h>
#include <stdint.h>

#define DS4_QWEN4EXP_GDN_REPLAY_ROWS 2u

/* Host transition policy, independent of CUDA. A replay verify leaves a
 * virtual row-zero snapshot: checkpoint + its bounded transition log. */
typedef struct {
    bool active;
    bool settle;
    bool swap;
    uint32_t prefix;
} ds4_qwen4exp_gdn_replay_step;

static inline ds4_qwen4exp_gdn_replay_step ds4_qwen4exp_gdn_replay_plan(
        bool enabled, bool previous, uint32_t old_prefix, uint32_t capacity,
        uint32_t width, uint32_t snapshots, uint32_t max_adopt_width,
        uint32_t recurrent, uint32_t conv) {
    ds4_qwen4exp_gdn_replay_step p = {false, false, false, 0u};
    p.active = enabled && width == 2u && snapshots == 1u;
    const bool reuse = p.active && previous && recurrent == 1u && conv == 1u;
    p.settle = (recurrent != 0u || conv != 0u) &&
        (width > max_adopt_width || recurrent != conv ||
         (p.active && !reuse) || (!p.active && previous && recurrent != 0u));
    p.swap = p.active && !reuse;
    p.prefix = reuse ? (old_prefix == capacity ? 0u : old_prefix + 1u) : 0u;
    return p;
}

/* Physical recurrent-buffer addresses alternate even on an ordinary forward
 * after replay. Both parity and kernel choice must identify a graph. The log
 * length is device data, so it deliberately does not multiply graph entries. */
static inline uint32_t ds4_qwen4exp_gdn_graph_variant(
        uint32_t width, uint32_t snapshots, uint32_t phase, bool active) {
    return width | (snapshots << 8u) | (phase << 16u) |
           ((uint32_t)active << 17u);
}

#endif
