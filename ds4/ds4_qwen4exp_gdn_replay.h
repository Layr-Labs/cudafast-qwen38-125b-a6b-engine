#ifndef DS4_QWEN4EXP_GDN_REPLAY_H
#define DS4_QWEN4EXP_GDN_REPLAY_H

#include <stdbool.h>
#include <stdint.h>

/* THE TAPE DEPTH IS A PURE COST, NOT A CAPABILITY.
 *
 * `prefix` cycles 0, 1, ... ROWS and then folds into the checkpoint and resets,
 * and the recurrent kernel's loop runs `prefix + n_tokens` sequential steps --
 * so the replayed steps average ROWS/2 per forward, every one of them a
 * register-carried step with two warp reductions in it.  Nothing else in the
 * controller depends on the depth: `reuse` (and therefore `settle`, the host
 * rollback copy that drains the stream, and `swap`) is a function of
 * `previous`, `recurrent` and `conv` only, never of `prefix`.  So a deeper tape
 * buys no avoided settle -- it only lengthens the replay -- and the fold it
 * defers is a single float4 store per state element.
 *
 * At depth 1 the decode forward (n_tokens == 2) averages 0.5 + 2 = 2.5 steps
 * instead of 1.0 + 2 = 3.0: one sixth of the sequential work in a latency-bound
 * recurrence, for a smaller tape.
 *
 * BIT-EXACT.  The checkpoint is a float4 *copy* of the carried state at a step
 * boundary, not a recomputation, so folding after one transition instead of two
 * reproduces the identical `h` bit for bit: the same transitions are applied to
 * the same state in the same order, and only the point at which the register
 * value is spilled to the checkpoint moves.  The depth also has to satisfy
 * prefix <= ROWS for the virtual row-zero snapshot (rows = prefix + 1 entries
 * when prefix < ROWS, 0 when it equals ROWS), which holds at 1 exactly as it
 * does at 2. */
#define DS4_QWEN4EXP_GDN_REPLAY_ROWS 1u

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
