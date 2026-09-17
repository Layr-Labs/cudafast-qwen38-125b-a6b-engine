#ifndef DS4_QWEN4EXP_GDN_REPLAY_H
#define DS4_QWEN4EXP_GDN_REPLAY_H

#include <stdbool.h>
#include <stdint.h>

/* THE TAPE DEPTH TRADES A 108 MiB STORE AGAINST A 33 KiB READ, SO GO DEEPER.
 *
 * `prefix` cycles 0, 1, ... ROWS, folds the carried state into the checkpoint,
 * and resets.  Two costs move in opposite directions with the depth:
 *
 *   the replay  -- the kernel loop runs `prefix + n_tokens` sequential steps, so
 *                  the replayed steps average ROWS/2 per forward.  Each one
 *                  re-reads ONE tape row: (n_key + n_value) * head_dim + 2 *
 *                  n_value floats = (16 + 48) * 128 + 96 = 8,288 floats,
 *                  33 KiB per GDN layer.  It is a latency cost, not a traffic
 *                  cost -- two warp reductions inside the dependency chain.
 *
 *   the fold    -- `*(float4 *)(checkpoint + state_off) = h` writes the WHOLE
 *                  recurrent state: n_value * head_dim * n_state * 4 =
 *                  48 * 128 * 128 * 4 = 3 MiB per layer, and with
 *                  n_full_attn_interval == 4 there are 36 GDN layers of 48, so
 *                  108 MiB.  It fires once per ROWS + 1 forwards.
 *
 * The fold is the term that dominates, and it gets CHEAPER as the tape deepens.
 * Per decode forward the kernel already moves 108 MiB reading the checkpoint
 * plus 108 MiB storing the state unconditionally; the fold adds 108/(ROWS + 1):
 *
 *     ROWS  fold traffic   total   vs ROWS 2   replayed steps
 *        0     108 MiB     324 MiB   +28.6%        2.0
 *        1      54 MiB     270 MiB    +7.1%        2.5
 *        2      36 MiB     252 MiB       --        3.0
 *        4      21.6 MiB   237.6 MiB   -5.7%       4.0
 *
 * MEASURED, and it is why this moves UP rather than down.  Submission 37850f89
 * set the depth to 1 -- fewer steps, more folds -- and the decode leg came in at
 * 2.290473802801815 against the frontier's 2.3007788632224058, -0.448%.  The
 * depth provably cannot touch the prefill leg (width 1024 != 2 leaves `active`
 * false, so `prefix` is pinned at 0 at every depth), so that leg is a pure noise
 * draw and it came in -0.873%; correcting the decode leg for the +0.674 leg
 * correlation leaves about -0.34% for the arm itself.  Trading -16.7% of the
 * sequential steps for +18 MiB of stores LOST.  So run the trade backwards.
 *
 * Depth 4 keeps 80% of the reachable fold saving (-14.4 MiB of the -36 MiB
 * limit) while the replay stays short enough to stay off the critical path.
 * Beyond it the saving halves per rung -- 108/((d+1)(d+2)) -- while each rung
 * costs another half step, so the marginal trade turns over quickly.
 *
 * BIT-EXACT.  The checkpoint is a float4 *copy* of the carried state at a step
 * boundary, not a recomputation, so deferring the fold applies the same
 * transitions to the same state in the same order; only the point at which the
 * register value is spilled moves.  A host rig driving this exact plan function
 * confirms depths 1, 2, 3 and 4 emit bit-identical values while the step count
 * goes 2.5, 3.0, 3.5, 4.0.  Nothing else in the controller reads the capacity:
 * `active` -- and so `settle`, the host rollback copy that drains the stream,
 * and `swap` -- is a function of `previous`, `recurrent` and `conv` only.  The
 * graph variant key deliberately excludes the log length, so no new entry. */
#define DS4_QWEN4EXP_GDN_REPLAY_ROWS 4u

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
