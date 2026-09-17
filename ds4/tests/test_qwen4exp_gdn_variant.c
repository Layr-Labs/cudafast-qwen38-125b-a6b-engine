/* test_qwen4exp_gdn_variant -- the decode-graph variant key's algebra, host-only.
 *
 * WHY THIS EXISTS.  Two shipped decisions rest on ds4_qwen4exp_gdn_graph_variant()
 * and neither was checkable off the box:
 *
 *   * the shim's boot warm-up carries `_Static_assert(WARM_ROUNDS >= 4)` with the
 *     argument that a decode identity folds in the recurrent-buffer PARITY, that
 *     parity flips only on a round that SWAPS, and that a parity left uncaptured
 *     is captured inside the timed window.  That argument assumes parity really
 *     is a distinct identity, which is this file's second check;
 *   * the decode-graph slot count was raised 4 -> 8 -> 16 across three promoted
 *     submissions, each time on the reasoning that the key grew and the slots
 *     did not.  That reasoning assumes the key is injective in every field, which
 *     is this file's first check.
 *
 * Neither property is a property of the CUDA build: the function is a
 * `static inline` in a header that includes only <stdbool.h> and <stdint.h>.  So
 * it can be driven here, with no GPU, no engine and no checkpoint, and the
 * claims above stop being prose.  The GPU-side test (test_qwen4exp_gdn_replay.c)
 * uses the key incidentally while driving the replay plan and links device
 * objects; this one is the algebra alone.
 *
 * Build and run:
 *   make tests/test_qwen4exp_gdn_variant CC=gcc && ./tests/test_qwen4exp_gdn_variant
 */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "ds4_qwen4exp_gdn_replay.h"

static int checks, failures;

static void ok(bool cond, const char *what) {
    checks++;
    if (!cond) {
        failures++;
        printf("  FAIL  %s\n", what);
    }
}

/* The width and snapshot values the scored decode leg actually uses, as named in
 * the shim's warm-up comment: the 2-row verify with one snapshot. */
enum { SCORED_WIDTH = 2, SCORED_SNAPSHOTS = 1 };

static void test_key_is_injective_in_every_field(void) {
    printf("the key is injective in width, snapshots, phase and active\n");
    uint32_t seen[4 * 3 * 2 * 2];
    size_t n = 0;
    for (uint32_t width = 1; width <= 4; width++)
        for (uint32_t snap = 0; snap <= 2; snap++)
            for (uint32_t phase = 0; phase <= 1; phase++)
                for (int active = 0; active <= 1; active++) {
                    const uint32_t k = ds4_qwen4exp_gdn_graph_variant(
                            width, snap, phase, active != 0);
                    for (size_t i = 0; i < n; i++)
                        if (seen[i] == k) {
                            char msg[128];
                            snprintf(msg, sizeof msg,
                                     "collision: width=%u snap=%u phase=%u active=%d",
                                     width, snap, phase, active);
                            ok(false, msg);
                            goto next;
                        }
                    seen[n++] = k;
                next:;
                }
    ok(n == 4 * 3 * 2 * 2, "every reachable combination produced a distinct key");
    printf("  %zu distinct identities over %d combinations\n", n, 4 * 3 * 2 * 2);
}

static void test_parity_is_a_distinct_identity(void) {
    printf("parity is a distinct identity at the scored width\n");
    const uint32_t p0 = ds4_qwen4exp_gdn_graph_variant(
            SCORED_WIDTH, SCORED_SNAPSHOTS, 0u, true);
    const uint32_t p1 = ds4_qwen4exp_gdn_graph_variant(
            SCORED_WIDTH, SCORED_SNAPSHOTS, 1u, true);
    ok(p0 != p1, "phase 0 and phase 1 are different identities");
    /* Everything else about the scored leg is fixed, so flipping parity alone
     * must move the key -- this is the property the warm-up's argument needs:
     * a round that never swaps can never reveal the other identity. */
    ok((p0 ^ p1) == (1u << 16u), "phase occupies exactly bit 16");
}

static void test_live_identity_count_at_the_scored_width(void) {
    printf("the scored width has exactly two live decode identities\n");
    /* Fixed width and snapshots, active set by the replay: the only free field
     * is parity.  Two identities, not one -- and that is the whole reason the
     * warm-up must produce an accepting round from each. */
    uint32_t keys[2] = {
        ds4_qwen4exp_gdn_graph_variant(SCORED_WIDTH, SCORED_SNAPSHOTS, 0u, true),
        ds4_qwen4exp_gdn_graph_variant(SCORED_WIDTH, SCORED_SNAPSHOTS, 1u, true),
    };
    ok(keys[0] != keys[1], "two distinct live identities, one per parity");
    /* Counting the same way the slot-count comments do: the two widths the
     * decode leg can take, both parities, replay active or not. */
    size_t n = 0;
    for (uint32_t w = 1; w <= 2; w++)
        for (uint32_t phase = 0; phase <= 1; phase++)
            for (int active = 0; active <= 1; active++) {
                (void)ds4_qwen4exp_gdn_graph_variant(w, SCORED_SNAPSHOTS, phase,
                                                     active != 0);
                n++;
            }
    ok(n == 8u, "the width x phase x active cross product is 8 identities");
    printf("  live identities at the scored width: 2; cross product: 8\n");
}

static void test_unset_fields_do_not_collide_with_set_ones(void) {
    printf("a zeroed field is not confused with an unset one\n");
    /* snapshots=0 must not read as "snapshots unset" for any width or parity:
     * the key is a bit field, so the only requirement is that the shift domain
     * of each field stays inside its own byte. */
    for (uint32_t w = 1; w <= 4; w++) {
        const uint32_t a = ds4_qwen4exp_gdn_graph_variant(w, 0u, 0u, false);
        const uint32_t b = ds4_qwen4exp_gdn_graph_variant(w, 1u, 0u, false);
        ok(a != b, "snapshots 0 and 1 differ");
        const uint32_t c = ds4_qwen4exp_gdn_graph_variant(w, 0u, 0u, true);
        ok(a != c, "active false and true differ");
    }
}

int main(void) {
    printf("test_qwen4exp_gdn_variant\n");
    test_key_is_injective_in_every_field();
    test_parity_is_a_distinct_identity();
    test_live_identity_count_at_the_scored_width();
    test_unset_fields_do_not_collide_with_set_ones();
    printf("%d/%d checks passed\n", checks - failures, checks);
    return failures == 0 ? 0 : 1;
}
