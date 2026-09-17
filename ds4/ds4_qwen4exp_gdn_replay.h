#ifndef DS4_QWEN4EXP_GDN_REPLAY_H
#define DS4_QWEN4EXP_GDN_REPLAY_H

#include <stdbool.h>
#include <stdint.h>

#define DS4_QWEN4EXP_GDN_REPLAY_ROWS 2u

/* Host transition policy, independent of CUDA. A replay verify leaves a
 * virtual row-zero snapshot: checkpoint + its bounded transition log. */

/* The four fields above are a contract, not just a record, and two shipped
 * decisions rest on it:
 *
 *   - The boot warm-up in the resident shim carries a static assertion that
 *     its round count can reach both recurrent-buffer parities. That argument
 *     needs `swap` to be exactly "the replay is active and the previous round
 *     was not reusable" -- a parity can only be captured on a round that
 *     settles and swaps -- and it needs `active` to mean one specific shape
 *     (a two-row verify with a single snapshot).
 *   - The decode-graph variant table's slot count was raised 4 -> 8 -> 16
 *     across three promoted submissions on the reasoning that the key grew.
 *     That reasoning needs ds4_qwen4exp_gdn_graph_variant() to be injective in
 *     every one of its four fields, or slots would alias rather than run out.
 *
 * Both functions are `static inline` here and this header includes only
 * <stdbool.h> and <stdint.h>, so the contract is drivable with no GPU, no
 * engine and no checkpoint: ds4/tests/test_qwen4exp_gdn_variant.c drives both
 * over their reachable domains and recomputes each predicate rather than
 * trusting it. Change a clause here and that test is expected to fail. */
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
/* The decode-graph identity: width, snapshots, phase and replay-active are
 * packed into disjoint bit fields (see above -- injectivity across all four is
 * tested, and the slot table's size depends on it). "Phase" is the recurrent
 * buffer parity, so a round that never swaps can never reveal the other
 * identity; that is why the warm-up must be long enough to settle and swap. */
static inline uint32_t ds4_qwen4exp_gdn_graph_variant(
        uint32_t width, uint32_t snapshots, uint32_t phase, bool active) {
    return width | (snapshots << 8u) | (phase << 16u) |
           ((uint32_t)active << 17u);
}

#endif
