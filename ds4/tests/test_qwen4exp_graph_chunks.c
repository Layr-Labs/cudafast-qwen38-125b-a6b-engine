/* Unit tests for the decode-round chunk schedule,
 * ds4_qwen4exp_graph_chunks.h.
 *
 * The schedule decides which consecutive layer range each captured piece of
 * the target stack replays, and the chunk cache keys a piece by its INDEX,
 * never by its range -- so the shapes are a correctness contract: a wrong
 * table replays a wrong range under a valid key, silently.  What is pinned
 * here:
 *
 *   - the default (headstart) shapes, including the production 48/4
 *     [1,16,16,15] as exact BOUNDS and not just widths: the next-chunk
 *     prefetch walks chunk+1 and stops at n_layer, so where the boundaries
 *     fall is the contract, not merely how wide the pieces are;
 *   - the uniform shapes DS4_CUDA_DECODE_GRAPH_NO_HEADSTART=1 restores,
 *     asserted equal to the pre-headstart loop arithmetic piece by piece;
 *   - the fallbacks: a one-layer stack and a single piece take the uniform
 *     whole-stack schedule under BOTH flags, and an empty stack yields no
 *     pieces rather than dividing by zero;
 *   - the structural invariants over 1..64 layers x 1..8 chunks, both
 *     schedules: exact cover (first piece starts at 0, last ends at n_layer,
 *     no gap, no overlap), non-empty pieces, monotone starts, no more pieces
 *     than asked, and only the last piece reaching n_layer.
 *
 * Pure C99: no CUDA, no Metal, no engine state. Builds and runs on any host.
 */

#include "ds4_qwen4exp_graph_chunks.h"

#include <stdio.h>

static int g_failed = 0;
static int g_total  = 0;

static char g_case[192];        /* sweep failures name their case */
static char g_case_tag[48];

#define CHECK(cond, msg) do {                                                  \
    g_total++;                                                                 \
    if (!(cond)) {                                                             \
        fprintf(stderr, "  FAIL: %s (line %d)\n", (msg), __LINE__);            \
        g_failed++;                                                            \
    }                                                                          \
} while (0)

/* CHECK with the sweep case stamped into the message. */
#define CASE_CHECK(cond, what) do {                                            \
    snprintf(g_case, sizeof(g_case), "%s: %s", g_case_tag, what);              \
    CHECK(cond, g_case);                                                       \
} while (0)

#define RUN(fn) do {                                                           \
    fprintf(stderr, "RUN: %s\n", #fn);                                         \
    int _before = g_failed;                                                    \
    (fn)();                                                                    \
    fprintf(stderr, "  %s\n", (_before == g_failed) ? "ok" : "FAIL");          \
} while (0)

enum { MAX_PIECES = 16 };

/* The engine loop's walk (qwen4exp_graph_layers in ds4_qwen4exp_graph.inc):
 * clamp the piece count to the layer count, then stop at the first empty
 * piece the schedule reports. */
static int sched_pieces(uint32_t n_layer, uint32_t chunks, bool headstart,
                        uint32_t *first, uint32_t *last) {
    if (chunks > n_layer) chunks = n_layer;
    int n = 0;
    for (uint32_t c = 0u; c < chunks; c++) {
        uint32_t a = 0u, b = 0u;
        if (!qw_decode_graph_chunk_range(c, n_layer, chunks, headstart,
                                         &a, &b)) {
            break;
        }
        first[n] = a;
        last[n] = b;
        n++;
    }
    return n;
}

static void expect_widths(const char *what, uint32_t n_layer, uint32_t chunks,
                          bool headstart, const uint32_t *exp, int n_exp) {
    uint32_t first[MAX_PIECES], last[MAX_PIECES];
    const int n = sched_pieces(n_layer, chunks, headstart, first, last);
    int ok = (n == n_exp);
    for (int i = 0; ok && i < n; i++) {
        ok = (last[i] - first[i]) == exp[i];
    }
    if (!ok) {
        fprintf(stderr, "  %s: got", what);
        for (int i = 0; i < n; i++) fprintf(stderr, " %u", last[i] - first[i]);
        fprintf(stderr, ", expected");
        for (int i = 0; i < n_exp; i++) fprintf(stderr, " %u", exp[i]);
        fprintf(stderr, "\n");
    }
    CHECK(ok, what);
}

/* bounds = {first0, last0, first1, last1, ...}, half-open [first, last). */
static void expect_bounds(const char *what, uint32_t n_layer, uint32_t chunks,
                          bool headstart, const uint32_t *bounds,
                          int n_pieces) {
    uint32_t first[MAX_PIECES], last[MAX_PIECES];
    const int n = sched_pieces(n_layer, chunks, headstart, first, last);
    int ok = (n == n_pieces);
    for (int i = 0; ok && i < n; i++) {
        ok = first[i] == bounds[2 * i] && last[i] == bounds[2 * i + 1];
    }
    if (!ok) {
        fprintf(stderr, "  %s: got %d pieces", what, n);
        for (int i = 0; i < n; i++) {
            fprintf(stderr, " [%u,%u)", first[i], last[i]);
        }
        fprintf(stderr, "\n");
    }
    CHECK(ok, what);
}

static int same_sched(uint32_t n_layer, uint32_t chunks, bool hs_a, bool hs_b) {
    uint32_t fa[MAX_PIECES], la[MAX_PIECES], fb[MAX_PIECES], lb[MAX_PIECES];
    const int na = sched_pieces(n_layer, chunks, hs_a, fa, la);
    const int nb = sched_pieces(n_layer, chunks, hs_b, fb, lb);
    if (na != nb) return 0;
    for (int i = 0; i < na; i++) {
        if (fa[i] != fb[i] || la[i] != lb[i]) return 0;
    }
    return 1;
}

/* ----- the default (headstart) schedule ----- */

static void test_headstart_shapes(void) {
    /* The production geometry: 48 layers, the default 4 pieces. */
    static const uint32_t b48x4[] = { 0, 1, 1, 17, 17, 33, 33, 48 };
    expect_bounds("48L/4 headstart bounds", 48u, 4u, true, b48x4, 4);
    static const uint32_t w48x4[] = { 1, 16, 16, 15 };
    expect_widths("48L/4 headstart widths", 48u, 4u, true, w48x4, 4);

    static const uint32_t w48x2[] = { 1, 47 };
    expect_widths("48L/2 headstart", 48u, 2u, true, w48x2, 2);
    static const uint32_t w48x8[] = { 1, 7, 7, 7, 7, 7, 7, 5 };
    expect_widths("48L/8 headstart", 48u, 8u, true, w48x8, 8);

    /* The synthetic test fixture is 4 blocks, where the headstart split and
     * the uniform split coincide -- the GPU graph test cannot tell the
     * schedules apart there, which is why this host test exists. */
    static const uint32_t w4x4[] = { 1, 1, 1, 1 };
    expect_widths("4L/4 headstart", 4u, 4u, true, w4x4, 4);
    static const uint32_t w2x2[] = { 1, 1 };
    expect_widths("2L/2 headstart", 2u, 2u, true, w2x2, 2);

    /* rest_per can over-cover: 5/4 gives rest_per 2, so the fourth piece
     * would start at 5 and the walk stops after three pieces, the last
     * clamped to n_layer.  Coverage stays exact. */
    static const uint32_t w5x4[] = { 1, 2, 2 };
    expect_widths("5L/4 headstart (over-cover, walk stops)", 5u, 4u, true,
                  w5x4, 3);
    static const uint32_t w6x3[] = { 1, 3, 2 };
    expect_widths("6L/3 headstart", 6u, 3u, true, w6x3, 3);
    static const uint32_t w5x2[] = { 1, 4 };
    expect_widths("5L/2 headstart", 5u, 2u, true, w5x2, 2);
}

/* ----- the valve's uniform schedule ----- */

static void test_uniform_shapes(void) {
    /* What DS4_CUDA_DECODE_GRAPH_NO_HEADSTART=1 restores: the arithmetic the
     * loop had before the headstart schedule existed. */
    static const uint32_t b48x4[] = { 0, 12, 12, 24, 24, 36, 36, 48 };
    expect_bounds("48L/4 uniform bounds", 48u, 4u, false, b48x4, 4);
    static const uint32_t w48x4[] = { 12, 12, 12, 12 };
    expect_widths("48L/4 uniform widths", 48u, 4u, false, w48x4, 4);

    static const uint32_t w48x2[] = { 24, 24 };
    expect_widths("48L/2 uniform", 48u, 2u, false, w48x2, 2);
    static const uint32_t w48x8[] = { 6, 6, 6, 6, 6, 6, 6, 6 };
    expect_widths("48L/8 uniform", 48u, 8u, false, w48x8, 8);
    static const uint32_t w4x4[] = { 1, 1, 1, 1 };
    expect_widths("4L/4 uniform", 4u, 4u, false, w4x4, 4);
    static const uint32_t w2x2[] = { 1, 1 };
    expect_widths("2L/2 uniform", 2u, 2u, false, w2x2, 2);
    static const uint32_t w3x2[] = { 2, 1 };
    expect_widths("3L/2 uniform", 3u, 2u, false, w3x2, 2);
    static const uint32_t w47x4[] = { 12, 12, 12, 11 };
    expect_widths("47L/4 uniform", 47u, 4u, false, w47x4, 4);
}

/* ----- the fallbacks ----- */

static void test_whole_stack_fallbacks(void) {
    /* One layer: headstart cannot apply, both flags take the uniform
     * whole-stack schedule. */
    static const uint32_t one[] = { 1 };
    expect_widths("1L/4 headstart demoted", 1u, 4u, true, one, 1);
    CHECK(same_sched(1u, 4u, true, false), "1L/4: both flags agree");

    /* A single piece (CHUNKS=1) is the whole stack under both flags; the
     * engine reaches the whole-stack key directly, and the helper agrees. */
    static const uint32_t whole48[] = { 48 };
    expect_widths("48L/1 is the whole stack", 48u, 1u, true, whole48, 1);
    CHECK(same_sched(48u, 1u, true, false), "48L/1: both flags agree");
    static const uint32_t whole2[] = { 2 };
    expect_widths("2L/1 is the whole stack", 2u, 1u, true, whole2, 1);

    /* Direct demotion check: asking for headstart with one piece returns the
     * uniform bounds, not a division by zero. */
    uint32_t a0 = 99u, b0 = 99u, a1 = 99u, b1 = 99u;
    const bool r0 = qw_decode_graph_chunk_range(0u, 48u, 1u, true, &a0, &b0);
    const bool r1 = qw_decode_graph_chunk_range(0u, 48u, 1u, false, &a1, &b1);
    CHECK(r0 && r1 && a0 == a1 && b0 == b1,
          "chunks=1: headstart is demoted to uniform");

    /* An empty stack yields no pieces at all. */
    {
        uint32_t f[MAX_PIECES], l[MAX_PIECES];
        CHECK(sched_pieces(0u, 4u, true, f, l) == 0,
              "0 layers: no pieces, no division by zero");
    }
}

/* ----- uniform == the pre-headstart loop, piece by piece ----- */

static void test_uniform_is_preheadstart_arithmetic(void) {
    for (uint32_t n_layer = 1u; n_layer <= 64u; n_layer++) {
        for (uint32_t chunks = 1u; chunks <= 8u; chunks++) {
            uint32_t first[MAX_PIECES], last[MAX_PIECES];
            const int n = sched_pieces(n_layer, chunks, false, first, last);
            const uint32_t capped =
                chunks > n_layer ? n_layer : chunks;
            /* The loop as it stood before the headstart schedule:
             * per = ceil(n_layer/chunks) over the clamped count. */
            const uint32_t per = (n_layer + capped - 1u) / capped;
            int same = 1, count = 0;
            for (uint32_t c = 0u; c < capped; c++) {
                const uint32_t il_first = c * per;
                if (il_first >= n_layer) break;
                uint32_t il_last = il_first + per;
                if (il_last > n_layer) il_last = n_layer;
                if (count >= n || first[count] != il_first ||
                    last[count] != il_last) {
                    same = 0;
                    break;
                }
                count++;
            }
            if (same && count != n) same = 0;
            if (!same) {
                fprintf(stderr, "  uniform %uL/%u diverges from the "
                                "pre-headstart loop at piece %d\n",
                        n_layer, chunks, count);
            }
            CHECK(same, "uniform schedule is the pre-headstart arithmetic");
        }
    }
}

/* ----- structural invariants over the whole (layers, chunks) grid ----- */

static void test_invariants_sweep(void) {
    for (uint32_t n_layer = 1u; n_layer <= 64u; n_layer++) {
        for (uint32_t chunks = 1u; chunks <= 8u; chunks++) {
            for (int hs = 0; hs <= 1; hs++) {
                snprintf(g_case_tag, sizeof(g_case_tag), "%uL/%u hs=%d",
                         n_layer, chunks, hs);
                uint32_t first[MAX_PIECES], last[MAX_PIECES];
                const int n = sched_pieces(n_layer, chunks, hs != 0,
                                           first, last);

                CASE_CHECK(n >= 1, "at least one piece");
                if (n < 1) continue;

                int cover = first[0] == 0u && last[n - 1] == n_layer;
                int contiguous = 1, nonempty = 1, monotone = 1;
                int only_last_ends = 1;
                for (int i = 0; i < n; i++) {
                    if (last[i] <= first[i]) nonempty = 0;
                    if (i > 0) {
                        if (first[i] != last[i - 1]) contiguous = 0;
                        if (first[i] <= first[i - 1]) monotone = 0;
                    }
                    if (i + 1 < n && last[i] >= n_layer) only_last_ends = 0;
                }
                CASE_CHECK(cover, "exact cover: starts at 0, ends at "
                                  "n_layer");
                CASE_CHECK(contiguous, "consecutive pieces, no gap, no "
                                       "overlap");
                CASE_CHECK(nonempty && monotone, "every piece non-empty, "
                                                 "starts strictly increasing");
                CASE_CHECK(only_last_ends,
                           "only the last piece reaches n_layer (prefetch "
                           "guard)");

                const uint32_t capped =
                    chunks > n_layer ? n_layer : chunks;
                CASE_CHECK((uint32_t)n <= capped,
                           "no more pieces than asked");

                /* Where headstart applies, piece 0 is exactly one layer and
                 * never stands alone. */
                if (hs && n_layer >= 2u && capped >= 2u) {
                    CASE_CHECK(last[0] - first[0] == 1u && n >= 2,
                               "headstart: piece 0 is one layer");
                }

                /* A one-layer stack ignores the flag entirely. */
                if (n_layer == 1u) {
                    CASE_CHECK(same_sched(1u, chunks, true, false),
                               "1L: both flags agree");
                }
            }
        }
    }
}

int main(void) {
    RUN(test_headstart_shapes);
    RUN(test_uniform_shapes);
    RUN(test_whole_stack_fallbacks);
    RUN(test_uniform_is_preheadstart_arithmetic);
    RUN(test_invariants_sweep);

    fprintf(stderr, "\ntest_qwen4exp_graph_chunks: %d/%d checks passed "
                    "(%d failed)\n", g_total - g_failed, g_total, g_failed);
    return g_failed == 0 ? 0 : 1;
}
