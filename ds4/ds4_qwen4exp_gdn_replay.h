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
    /* Deferred mode only (ds4_qwen4exp_gdn_defer_plan); the shipped plan
     * leaves both zero.  `flush`: the live recurrent buffer is stale and must
     * be rebuilt from checkpoint + tape before anything else reads it.
     * `parity`: the tape buffer the next round replays from. */
    bool flush;
    uint32_t parity;
} ds4_qwen4exp_gdn_replay_step;

static inline ds4_qwen4exp_gdn_replay_step ds4_qwen4exp_gdn_replay_plan(
        bool enabled, bool previous, uint32_t old_prefix, uint32_t capacity,
        uint32_t width, uint32_t snapshots, uint32_t max_adopt_width,
        uint32_t recurrent, uint32_t conv) {
    ds4_qwen4exp_gdn_replay_step p = {false, false, false, 0u, false, 0u};
    p.active = enabled && width == 2u && snapshots == 1u;
    const bool reuse = p.active && previous && recurrent == 1u && conv == 1u;
    p.settle = (recurrent != 0u || conv != 0u) &&
        (width > max_adopt_width || recurrent != conv ||
         (p.active && !reuse) || (!p.active && previous && recurrent != 0u));
    p.swap = p.active && !reuse;
    p.prefix = reuse ? (old_prefix == capacity ? 0u : old_prefix + 1u) : 0u;
    return p;
}

/* DEFERRED STATE MATERIALISATION (DS4_QWEN4EXP_NO_GDN_DEFER restores the
 * plan above exactly).
 *
 * Row zero of a two-row verify is always committed; only row one can be
 * rejected.  So the committed recurrent state is always
 *
 *     checkpoint . tape[parity][0 .. prefix)
 *
 * and a verify round need not store the 3.1 MB state at all.  The tape is two
 * buffers of `capacity` rows each (K, V and the (decay, beta) pair of one
 * transition, recorded from the very operands the direct step used).  A round
 * replays tape[parity][0..prefix), then runs both live rows and
 *   - prefix + 2 <= capacity: records row 0 at [parity][prefix] and row 1 at
 *     [parity][prefix + 1]; writes no state;
 *   - otherwise (a FLUSH round): stores the post-row-0 state into the
 *     checkpoint in place (each cell by its owning thread, after that thread
 *     loaded it) and records row 1 at [parity ^ 1][0] -- never in the buffer
 *     other CTAs are still replaying.
 * The host then adopts: accept (both flags 0) -> prefix + 2 (flush: 1 in the
 * other buffer), reject (both flags 1) -> prefix + 1 (flush: 0, other buffer).
 * The device control word is prefix | parity << 8.
 *
 * Leaving the mode (a width-1 forward, unequal adoption flags) first
 * materialises the live state into the recurrent buffer (`flush`), or lets
 * the settle copy rebuild row zero when the recurrent flag is set; entry is
 * the shipped one-time pointer swap.  In steady state nothing swaps, so the
 * graph phase stays put and the variant set is a subset of the shipped one. */
#define DS4_QWEN4EXP_GDN_TAPE_ROWS 6u   /* default rows per tape buffer */
#define DS4_QWEN4EXP_GDN_TAPE_MAX 8u    /* env ceiling; the memory plan's bound */
#define DS4_QWEN4EXP_GDN_TAPE_MIN 2u

static inline uint32_t ds4_qwen4exp_gdn_defer_control(uint32_t prefix,
                                                      uint32_t parity) {
    return prefix | (parity << 8u);
}

/* Does a round replaying `prefix` rows publish a new checkpoint? */
static inline bool ds4_qwen4exp_gdn_defer_flushes(uint32_t prefix,
                                                  uint32_t capacity) {
    return prefix + 2u > capacity;
}

/* The tape rows that rebuild a state the last deferred round (prefix,
 * parity) left, from the checkpoint as that round left it: row zero (after its
 * committed first token) or, with `final_row`, the state after both tokens.
 * Returns the row count; *first is the absolute first row of the 2*capacity
 * row tape. */
static inline uint32_t ds4_qwen4exp_gdn_defer_rows(uint32_t prefix,
        uint32_t parity, uint32_t capacity, bool final_row, uint32_t *first) {
    if (ds4_qwen4exp_gdn_defer_flushes(prefix, capacity)) {
        *first = (parity ^ 1u) * capacity;
        return final_row ? 1u : 0u;
    }
    *first = parity * capacity;
    return prefix + (final_row ? 2u : 1u);
}

static inline ds4_qwen4exp_gdn_replay_step ds4_qwen4exp_gdn_defer_plan(
        bool enabled, bool previous, uint32_t old_prefix, uint32_t old_parity,
        uint32_t capacity, uint32_t width, uint32_t snapshots,
        uint32_t max_adopt_width, uint32_t recurrent, uint32_t conv) {
    ds4_qwen4exp_gdn_replay_step p = {false, false, false, 0u, false, 0u};
    p.active = enabled && width == 2u && snapshots == 1u;
    /* Both outcomes continue on the tape: accept (0, 0) and reject (1, 1). */
    const bool reuse = p.active && previous && recurrent == conv &&
        recurrent <= 1u;
    p.settle = (recurrent != 0u || conv != 0u) &&
        (width > max_adopt_width || recurrent != conv ||
         (p.active && !reuse) || (!p.active && previous && recurrent != 0u));
    /* A settle with the recurrent flag set rebuilds the buffer from the
     * virtual row zero itself; otherwise the final state is materialised. */
    p.flush = previous && !reuse && recurrent == 0u;
    p.swap = p.active && !reuse;
    if (reuse) {
        if (ds4_qwen4exp_gdn_defer_flushes(old_prefix, capacity)) {
            p.parity = old_parity ^ 1u;
            p.prefix = recurrent != 0u ? 0u : 1u;
        } else {
            p.parity = old_parity;
            p.prefix = old_prefix + (recurrent != 0u ? 1u : 2u);
        }
    }
    return p;
}

/* Physical recurrent-buffer addresses alternate even on an ordinary forward
 * after replay. Both parity and kernel choice must identify a graph. The log
 * length is device data, so it deliberately does not multiply graph entries.
 * Deferred mode: the tape parity is device data too (control bit 8); `phase`
 * flips only on the entry swap, never per accepted round. */
static inline uint32_t ds4_qwen4exp_gdn_graph_variant(
        uint32_t width, uint32_t snapshots, uint32_t phase, bool active) {
    return width | (snapshots << 8u) | (phase << 16u) |
           ((uint32_t)active << 17u);
}

#endif
