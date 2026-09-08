/* Tests and probe for the host side of the Qwen4-Exp PLE n-gram table.
 *
 * Pure C99: no CUDA, no Metal, no model.  With no arguments it runs the
 * self-contained checks.  With a mode argument it prints machine-readable
 * results for tests/test_qwen4exp_ple.py, which compares them against an
 * independent Python implementation of the same rules. */

#include "ds4_qwen4exp_ple.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_failed = 0;
static int g_total  = 0;

#define CHECK(cond, msg) do {                                                  \
    g_total++;                                                                 \
    if (!(cond)) {                                                             \
        fprintf(stderr, "  FAIL: %s (line %d)\n", (msg), __LINE__);            \
        g_failed++;                                                            \
    }                                                                          \
} while (0)

#define RUN(fn) do {                                                           \
    fprintf(stderr, "RUN: %s\n", #fn);                                         \
    int _before = g_failed;                                                    \
    (fn)();                                                                    \
    fprintf(stderr, "  %s\n", (_before == g_failed) ? "ok" : "FAIL");          \
} while (0)

/* ===================================================================== *
 * Constants of the pinned checkpoint, for the file-free checks.
 * unsloth/Qwen3.8-Flash-Next-GGUF UD-Q4_K_XL, qwen4exp.ple.*
 * ===================================================================== */

static void pinned_constants(ds4_ple_constants *c) {
    static const uint64_t multipliers[3] = {
        23703573157769ull, 20109073645365ull, 8052911324071ull
    };
    static const uint64_t sizes[16] = {
        20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069,
        20000077, 20000081, 20000093, 20000107, 20000147, 20000153, 20000159,
        20000161, 20000171
    };
    memset(c, 0, sizeof(*c));
    c->ngram_size      = 3;
    c->heads_per_ngram = 8;
    c->head_count      = 16;
    c->row_dim         = 160;
    c->conv_kernel     = 4;
    c->ple_layer_count = 1;
    c->ple_layers[0]   = 1;
    c->vocab_size      = 248320;
    c->eos_token_id    = 248044;
    memcpy(c->multipliers, multipliers, sizeof(multipliers));

    uint64_t running = 0;
    for (int h = 0; h < 16; h++) {
        c->head_vocab_sizes[h] = sizes[h];
        c->head_offsets[h]     = running;
        running += sizes[h];
    }
    c->row_total  = running;
    c->table_rows = 320001536ull;
}

/* ===================================================================== *
 * An independent statement of the end-of-sequence shift rule.
 *
 * The module carries previous_1/previous_2 forward as a recurrence.  This
 * scans back to the last end-of-sequence token instead, which is the shape
 * the mlx reference uses, so agreement is evidence and not a tautology.
 * ===================================================================== */

static int32_t shifted_by_scan(const int32_t *tokens, size_t t, size_t shift,
                               int32_t eos) {
    if (shift == 0) return tokens[t];
    if (shift > t) return eos;
    for (size_t back = 1; back <= shift; back++) {
        if (tokens[t - back] == eos) return eos;
    }
    return tokens[t - shift];
}

static void row_ids_by_scan(const ds4_ple_constants *c, const int32_t *tokens,
                            size_t count, uint64_t *out) {
    for (size_t t = 0; t < count; t++) {
        uint64_t mixed = (uint64_t)(uint32_t)tokens[t] * c->multipliers[0];
        for (uint32_t n = 2; n <= c->ngram_size; n++) {
            int32_t prev = shifted_by_scan(tokens, t, n - 1, c->eos_token_id);
            mixed ^= (uint64_t)(uint32_t)prev * c->multipliers[n - 1];
            for (uint32_t k = 0; k < c->heads_per_ngram; k++) {
                uint32_t head = (n - 2) * c->heads_per_ngram + k;
                out[t * c->head_count + head] =
                    mixed % c->head_vocab_sizes[head] + c->head_offsets[head];
            }
        }
    }
}

/* Deterministic token stream with end-of-sequence tokens sprinkled in. */
static uint64_t rng_state = 0x243F6A8885A308D3ull;

static uint64_t next_random(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}

/* ===================================================================== *
 * Checks.
 * ===================================================================== */

static void test_row_ids_match_the_scan_rule(void) {
    ds4_ple_constants c;
    pinned_constants(&c);

    enum { N = 512 };
    int32_t tokens[N];
    uint64_t mine[N * 16];
    uint64_t theirs[N * 16];

    for (int trial = 0; trial < 8; trial++) {
        for (int i = 0; i < N; i++) {
            uint64_t r = next_random();
            /* One token in sixteen is the end-of-sequence token, so runs of
             * two and three boundaries are exercised. */
            tokens[i] = ((r >> 40) % 16 == 0) ? c.eos_token_id
                                              : (int32_t)(r % c.vocab_size);
        }
        ds4_ple_history h;
        ds4_ple_history_reset(&c, &h);
        ds4_ple_row_ids(&c, &h, tokens, N, mine);
        row_ids_by_scan(&c, tokens, N, theirs);
        CHECK(memcmp(mine, theirs, sizeof(mine)) == 0,
              "carried history matches the scan rule");
    }
}

static void test_prefill_and_decode_agree(void) {
    ds4_ple_constants c;
    pinned_constants(&c);

    enum { N = 300 };
    int32_t tokens[N];
    uint64_t batched[N * 16];
    uint64_t stepped[N * 16];

    for (int i = 0; i < N; i++) {
        uint64_t r = next_random();
        tokens[i] = ((r >> 40) % 11 == 0) ? c.eos_token_id
                                          : (int32_t)(r % c.vocab_size);
    }

    ds4_ple_history h;
    ds4_ple_history_reset(&c, &h);
    ds4_ple_row_ids(&c, &h, tokens, N, batched);

    /* A prefill of 97 tokens, then single-token decode steps. */
    ds4_ple_history d;
    ds4_ple_history_reset(&c, &d);
    ds4_ple_row_ids(&c, &d, tokens, 97, stepped);
    for (int i = 97; i < N; i++) {
        ds4_ple_row_ids(&c, &d, &tokens[i], 1, &stepped[(size_t)i * 16]);
    }
    CHECK(memcmp(batched, stepped, sizeof(batched)) == 0,
          "one batch and prefill-then-decode produce the same ids");
}

static void test_eos_boundary_shape(void) {
    ds4_ple_constants c;
    pinned_constants(&c);
    const int32_t eos = c.eos_token_id;

    /* previous_1 is eos at the start of a sequence, and previous_2 is eos
     * whenever previous_1 is. */
    const int32_t tokens[] = { 7, 11, eos, 13, 17, 19 };
    const size_t  n = sizeof(tokens) / sizeof(tokens[0]);
    uint64_t ids[6 * 16];
    uint64_t want[6 * 16];

    ds4_ple_history h;
    ds4_ple_history_reset(&c, &h);
    ds4_ple_row_ids(&c, &h, tokens, n, ids);
    row_ids_by_scan(&c, tokens, n, want);
    CHECK(memcmp(ids, want, sizeof(ids)) == 0, "hand-written boundary sequence");

    /* Token 3 follows the end-of-sequence token: its bigram head must read
     * (13, eos) and its trigram head (13, eos, eos), so the two blocks of the
     * mixed value differ only by eos * m2. */
    uint64_t bigram  = (uint64_t)13 * c.multipliers[0] ^ (uint64_t)(uint32_t)eos * c.multipliers[1];
    uint64_t trigram = bigram ^ (uint64_t)(uint32_t)eos * c.multipliers[2];
    CHECK(ids[3 * 16 + 0] == bigram % c.head_vocab_sizes[0] + c.head_offsets[0],
          "bigram head after a boundary");
    CHECK(ids[3 * 16 + 8] == trigram % c.head_vocab_sizes[8] + c.head_offsets[8],
          "trigram head after a boundary");
}

static void test_row_ids_stay_inside_the_table(void) {
    ds4_ple_constants c;
    pinned_constants(&c);

    enum { N = 4096 };
    int32_t tokens[N];
    static uint64_t ids[N * 16];
    for (int i = 0; i < N; i++) {
        tokens[i] = (int32_t)(next_random() % c.vocab_size);
    }
    ds4_ple_history h;
    ds4_ple_history_reset(&c, &h);
    ds4_ple_row_ids(&c, &h, tokens, N, ids);

    int in_range = 1;
    int in_head  = 1;
    for (int i = 0; i < N * 16; i++) {
        uint32_t head = (uint32_t)(i % 16);
        if (ids[i] >= c.row_total) in_range = 0;
        if (ids[i] < c.head_offsets[head] ||
            ids[i] >= c.head_offsets[head] + c.head_vocab_sizes[head]) in_head = 0;
    }
    CHECK(in_range, "every id is inside the table");
    CHECK(in_head, "every id is inside its own head's slice");
    CHECK(c.row_total == 320001446ull, "the head table covers 320,001,446 rows");
    CHECK(c.row_total <= c.table_rows, "the table is at least as large as the head table");
}

/* IQ4_NL, restated from the ggml block layout. */
static void test_dequant_iq4_nl(void) {
    static const int8_t kvalues[16] = {
        -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
    };
    /* Scale 1.0 in fp16 is 0x3C00; the nibbles walk the whole code book. */
    uint8_t block[DS4_PLE_IQ4_NL_BLOCK_BYTES];
    block[0] = 0x00;
    block[1] = 0x3C;
    for (int j = 0; j < 16; j++) {
        block[2 + j] = (uint8_t)((j & 0x0F) | (((15 - j) & 0x0F) << 4));
    }

    float out[DS4_PLE_IQ4_NL_BLOCK_ELEMS];
    ds4_ple_dequant_iq4_nl(block, 1, out);

    int low_ok = 1, high_ok = 1;
    for (int j = 0; j < 16; j++) {
        if (out[j] != (float)kvalues[j]) low_ok = 0;
        if (out[j + 16] != (float)kvalues[15 - j]) high_ok = 0;
    }
    CHECK(low_ok, "low nibbles fill the first sixteen values");
    CHECK(high_ok, "high nibbles fill the last sixteen values");

    /* Scale 0.5 (0x3800) and a negative scale must scale exactly. */
    block[0] = 0x00; block[1] = 0x38;
    ds4_ple_dequant_iq4_nl(block, 1, out);
    CHECK(out[0] == 0.5f * (float)kvalues[0], "positive scale applies");
    block[0] = 0x00; block[1] = 0xB8;
    ds4_ple_dequant_iq4_nl(block, 1, out);
    CHECK(out[0] == -0.5f * (float)kvalues[0], "negative scale applies");

    /* Zero and a subnormal fp16 scale. */
    block[0] = 0x00; block[1] = 0x00;
    ds4_ple_dequant_iq4_nl(block, 1, out);
    CHECK(out[0] == 0.0f, "a zero scale gives zeros");
    block[0] = 0x01; block[1] = 0x00; /* smallest fp16 subnormal, 2^-24 */
    ds4_ple_dequant_iq4_nl(block, 1, out);
    CHECK(out[0] == (float)kvalues[0] / 16777216.0f, "a subnormal scale converts exactly");
}

static void test_refusal_without_a_checkpoint(void) {
    char err[DS4_PLE_ERROR_SIZE];
    ds4_ple_constants c;
    const char *nothing[1] = { "/nonexistent/ds4-ple-test.gguf" };

    CHECK(!ds4_ple_constants_read(NULL, 0, &c, err, sizeof(err)),
          "no shard is a refusal");
    CHECK(err[0] != '\0', "the refusal says why");
    CHECK(!ds4_ple_constants_read(nothing, 1, &c, err, sizeof(err)),
          "a missing file is a refusal");

    ds4_ple_table *t = NULL;
    CHECK(!ds4_ple_table_open(nothing, 1, DS4_PLE_DEFAULT_CACHE_BYTES, &t, err, sizeof(err)),
          "opening a missing file is a refusal");
    CHECK(t == NULL, "a refused open leaves no table");
}

/* ===================================================================== *
 * Probe modes.
 * ===================================================================== */

static uint32_t float_bits(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    return bits;
}

static size_t parse_list_u64(const char *csv, uint64_t *out, size_t cap) {
    size_t n = 0;
    const char *p = csv;
    while (*p && n < cap) {
        out[n++] = strtoull(p, (char **)&p, 10);
        if (*p == ',') p++;
    }
    return n;
}

static int probe_constants(int argc, char **argv) {
    char err[DS4_PLE_ERROR_SIZE];
    ds4_ple_constants c;
    if (!ds4_ple_constants_read((const char *const *)argv, (size_t)argc, &c,
                                err, sizeof(err))) {
        fprintf(stderr, "%s\n", err);
        return 1;
    }
    printf("ngram_size=%u\n", c.ngram_size);
    printf("heads_per_ngram=%u\n", c.heads_per_ngram);
    printf("head_count=%u\n", c.head_count);
    printf("row_dim=%u\n", c.row_dim);
    printf("vocab_size=%u\n", c.vocab_size);
    printf("eos_token_id=%d\n", c.eos_token_id);
    printf("conv_kernel=%u\n", c.conv_kernel);
    for (uint32_t i = 0; i < c.ple_layer_count; i++) {
        printf("ple_layer[%u]=%u\n", i, c.ple_layers[i]);
    }
    printf("row_total=%" PRIu64 "\n", c.row_total);
    for (uint32_t i = 0; i < c.ngram_size; i++) {
        printf("multiplier[%u]=%" PRIu64 "\n", i, c.multipliers[i]);
    }
    for (uint32_t h = 0; h < c.head_count; h++) {
        printf("head[%u]=%" PRIu64 ",%" PRIu64 "\n", h,
               c.head_vocab_sizes[h], c.head_offsets[h]);
    }
    return 0;
}

static int probe_ids(const char *csv, int argc, char **argv) {
    char err[DS4_PLE_ERROR_SIZE];
    ds4_ple_constants c;
    if (!ds4_ple_constants_read((const char *const *)argv, (size_t)argc, &c,
                                err, sizeof(err))) {
        fprintf(stderr, "%s\n", err);
        return 1;
    }
    uint64_t raw[8192];
    size_t n = parse_list_u64(csv, raw, sizeof(raw) / sizeof(raw[0]));

    int32_t *tokens = malloc(n * sizeof(int32_t) + 1);
    uint64_t *ids   = malloc(n * c.head_count * sizeof(uint64_t) + 1);
    if (!tokens || !ids) { free(tokens); free(ids); return 1; }
    for (size_t i = 0; i < n; i++) tokens[i] = (int32_t)raw[i];

    ds4_ple_history h;
    ds4_ple_history_reset(&c, &h);
    ds4_ple_row_ids(&c, &h, tokens, n, ids);

    for (size_t i = 0; i < n; i++) {
        for (uint32_t k = 0; k < c.head_count; k++) {
            printf("%s%" PRIu64, k ? " " : "", ids[i * c.head_count + k]);
        }
        printf("\n");
    }
    free(tokens);
    free(ids);
    return 0;
}

static int probe_rows(const char *csv, uint64_t cache_bytes, int argc, char **argv) {
    char err[DS4_PLE_ERROR_SIZE];
    ds4_ple_table *t = NULL;
    if (!ds4_ple_table_open((const char *const *)argv, (size_t)argc, cache_bytes,
                            &t, err, sizeof(err))) {
        fprintf(stderr, "%s\n", err);
        return 1;
    }
    const ds4_ple_constants *c = ds4_ple_table_constants(t);
    uint64_t ids[4096];
    size_t n = parse_list_u64(csv, ids, sizeof(ids) / sizeof(ids[0]));

    float *rows = malloc(n * c->row_dim * sizeof(float) + 1);
    if (!rows || !ds4_ple_table_rows(t, ids, n, rows)) {
        fprintf(stderr, "ds4_ple: row read refused\n");
        free(rows);
        ds4_ple_table_close(t);
        return 1;
    }
    for (size_t i = 0; i < n; i++) {
        for (uint32_t k = 0; k < c->row_dim; k++) {
            printf("%s%08x", k ? " " : "", float_bits(rows[i * c->row_dim + k]));
        }
        printf("\n");
    }
    free(rows);
    ds4_ple_table_close(t);
    return 0;
}

static int probe_raw(const char *csv, int argc, char **argv) {
    char err[DS4_PLE_ERROR_SIZE];
    ds4_ple_table *t = NULL;
    if (!ds4_ple_table_open((const char *const *)argv, (size_t)argc, 0, &t,
                            err, sizeof(err))) {
        fprintf(stderr, "%s\n", err);
        return 1;
    }
    size_t bytes = ds4_ple_table_quant_row_bytes(t);
    uint64_t ids[4096];
    size_t n = parse_list_u64(csv, ids, sizeof(ids) / sizeof(ids[0]));
    uint8_t *row = malloc(bytes);
    if (!row) { ds4_ple_table_close(t); return 1; }

    int rc = 0;
    for (size_t i = 0; i < n; i++) {
        if (!ds4_ple_table_quant_row(t, ids[i], row)) { rc = 1; break; }
        for (size_t k = 0; k < bytes; k++) printf("%02x", row[k]);
        printf("\n");
    }
    free(row);
    ds4_ple_table_close(t);
    return rc;
}

/* Random row workload: reports the counters, the ceiling and a checksum over
 * every value returned, so a run with a tiny hot set can be compared against a
 * run with none. */
static int probe_soak(uint64_t count, uint64_t seed, uint64_t distinct,
                      uint64_t cache_bytes, int argc, char **argv) {
    char err[DS4_PLE_ERROR_SIZE];
    ds4_ple_table *t = NULL;
    if (!ds4_ple_table_open((const char *const *)argv, (size_t)argc, cache_bytes,
                            &t, err, sizeof(err))) {
        fprintf(stderr, "%s\n", err);
        return 1;
    }
    const ds4_ple_constants *c = ds4_ple_table_constants(t);
    if (distinct == 0 || distinct > c->table_rows) distinct = c->table_rows;

    enum { BATCH = 16 };
    uint64_t ids[BATCH];
    float *rows = malloc((size_t)BATCH * c->row_dim * sizeof(float));
    if (!rows) { ds4_ple_table_close(t); return 1; }

    rng_state = seed | 1u;
    uint64_t checksum = 1469598103934665603ull; /* FNV-1a 64 */
    int rc = 0;

    for (uint64_t done = 0; done < count; done += BATCH) {
        uint64_t n = count - done < BATCH ? count - done : BATCH;
        for (uint64_t i = 0; i < n; i++) ids[i] = next_random() % distinct;
        ds4_ple_table_prefetch(t, ids, (size_t)n);
        if (!ds4_ple_table_rows(t, ids, (size_t)n, rows)) { rc = 1; break; }
        for (uint64_t v = 0; v < n * c->row_dim; v++) {
            uint32_t bits = float_bits(rows[v]);
            for (int b = 0; b < 4; b++) {
                checksum ^= (uint8_t)(bits >> (b * 8));
                checksum *= 1099511628211ull;
            }
        }
    }

    ds4_ple_table_stats stats;
    ds4_ple_table_stats_get(t, &stats);
    printf("checksum=%" PRIu64 "\n", checksum);
    printf("hits=%" PRIu64 "\n", stats.hits);
    printf("misses=%" PRIu64 "\n", stats.misses);
    printf("evictions=%" PRIu64 "\n", stats.evictions);
    printf("resident_rows=%" PRIu64 "\n", stats.resident_rows);
    printf("resident_bytes=%" PRIu64 "\n", stats.resident_bytes);
    printf("ceiling_bytes=%" PRIu64 "\n", stats.ceiling_bytes);
    printf("capacity_rows=%" PRIu64 "\n", stats.capacity_rows);

    free(rows);
    ds4_ple_table_close(t);
    return rc;
}

/* ===================================================================== *
 * The hash-constant tripwire.
 *
 * The engine consumes the artifact's key-values; this derivation exists only
 * to catch a file whose constants were produced by something other than the
 * model implementation's own rule.  The positive case is the pinned
 * checkpoint's real triple.  The negative control is the triple a seed of 0
 * produces -- the MLX runner's configuration default, a real defect on a real
 * port -- which must NOT be accepted, because every shape and byte count of
 * such a file still validates and nothing else would notice.
 * ===================================================================== */

static void test_hash_constants_derivation(void) {
    ds4_ple_constants pinned;
    pinned_constants(&pinned);

    /* Positive: seed 1234 at ple_layer_ids position 0 over the checkpoint's
     * 248,320-token vocabulary derives the artifact's own multipliers. */
    uint64_t derived[DS4_PLE_MAX_NGRAM];
    ds4_ple_derive_multipliers(pinned.ngram_size, pinned.vocab_size, 0,
                               DS4_PLE_HASH_SEED_DEFAULT, derived);
    for (uint32_t i = 0; i < pinned.ngram_size; i++) {
        CHECK(derived[i] == pinned.multipliers[i],
              "the reference derivation reproduces the artifact multipliers");
    }

    /* Negative control: the seed-0 triple is NOT what the rule derives, so an
     * artifact carrying it is refused. */
    static const uint64_t seed_zero[3] = {
        4788054244585ull, 5075510189727ull, 24189832309785ull
    };
    uint64_t derived_zero[DS4_PLE_MAX_NGRAM];
    ds4_ple_derive_multipliers(pinned.ngram_size, pinned.vocab_size, 0, 0,
                               derived_zero);
    int matches_rule = 1;
    for (uint32_t i = 0; i < pinned.ngram_size; i++) {
        if (seed_zero[i] != derived[i]) matches_rule = 0;
        CHECK(derived_zero[i] == seed_zero[i],
              "seed 0 is what produced the port's multipliers");
    }
    CHECK(matches_rule == 0,
          "the seed-0 multipliers are rejected by the derivation");

    /* The ordinal is the position in ple_layer_ids, not the layer number:
     * position 1 must derive a different set, or the tripwire would pass a
     * file that seeded on the wrong one. */
    uint64_t other[DS4_PLE_MAX_NGRAM];
    ds4_ple_derive_multipliers(pinned.ngram_size, pinned.vocab_size, 1,
                               DS4_PLE_HASH_SEED_DEFAULT, other);
    int ordinal_moves = 0;
    for (uint32_t i = 0; i < pinned.ngram_size; i++) {
        if (other[i] != derived[i]) ordinal_moves = 1;
    }
    CHECK(ordinal_moves == 1, "the layer ordinal changes the multipliers");

    /* Every multiplier is odd and inside the half bound, which is what keeps
     * `token * multiplier` non-negative in signed int64. */
    const uint64_t half =
        (0x7FFFFFFFFFFFFFFFull / (uint64_t)pinned.vocab_size) / 2ull;
    for (uint32_t i = 0; i < pinned.ngram_size; i++) {
        CHECK((derived[i] & 1ull) == 1ull, "multipliers are odd");
        CHECK(derived[i] < 2ull * half + 1ull,
              "multipliers stay inside the half bound");
    }
}

/* ===================================================================== *
 * Golden row ids from the parity reference.
 *
 * Source: ivanfioravanti/ds4-metal, branch qwen3.8-flash-next, MIT licence,
 * tests/test_qwen4_host.c:437-469 (`test_ngram_hash`).  Used with attribution
 * as an independent implementation of the same model.
 *
 * Their values are exact integers produced by a scalar host implementation,
 * not by a kernel, so this is a GROUND-TRUTH check and not a tolerance band.
 * Two implementations that were written separately agree on every one of the
 * 80 ids, so a silent divergence in the hash is now visible.
 *
 * Their end-of-sequence rule scans back to the last end-of-sequence token in
 * the stream (ds4_qwen4.c:783-825).  Ours carries the same rule forward as a
 * recurrence (ds4_qwen4exp_ple.c:644-673).  The vector below crosses an
 * end-of-sequence token in the middle of the stream, so it separates the two
 * shapes if they ever disagree.
 * ===================================================================== */

static void test_reference_golden_row_ids(void) {
    ds4_ple_constants c;
    pinned_constants(&c);

    /* Their published constants, checked against ours before the ids, so a
     * mismatch names the constant and not the hash. */
    CHECK(c.multipliers[0] == 23703573157769ull, "reference multiplier 0");
    CHECK(c.multipliers[1] == 20109073645365ull, "reference multiplier 1");
    CHECK(c.multipliers[2] == 8052911324071ull,  "reference multiplier 2");
    CHECK(c.head_vocab_sizes[0]  == 20000003ull, "reference head 0 vocabulary");
    CHECK(c.head_vocab_sizes[15] == 20000171ull, "reference head 15 vocabulary");
    CHECK(c.head_offsets[15] == 300001275ull,    "reference head 15 offset");
    CHECK(c.table_rows == 320001536ull,          "reference table row count");
    CHECK(c.eos_token_id == 248044,              "reference end-of-sequence id");

    /* Their inputs: a history of two end-of-sequence tokens, then five
     * tokens with an end-of-sequence token third. */
    static const int32_t tokens[5] = {5, 7, 248044, 9, 11};

    /* Their expected ids, transcribed from test_qwen4_host.c:459-465. */
    static const uint64_t expected[5][16] = {
        { 15389869u, 39778609u, 55713969u, 62213332u, 88817728u,118483999u,
         133731511u,155458159u,179763390u,197956758u,205378969u,220499474u,
         242466248u,265658744u,293662119u,315720898u},
        { 12441580u, 26378836u, 53347667u, 75104214u, 99467174u,114254887u,
         126436461u,156012011u,169119442u,187827161u,214803956u,239809754u,
         242938905u,266427765u,294337448u,314484167u},
        { 10204458u, 27984170u, 41283776u, 68842151u, 85621153u,118821647u,
         129504214u,158727320u,176298516u,181690702u,206665473u,238343128u,
         252151767u,267018740u,285543023u,319927855u},
        { 18043673u, 37626835u, 51159316u, 78294604u, 94015356u,106720349u,
         136526052u,144330141u,176817901u,186368539u,203707490u,230017629u,
         247662678u,266533413u,293096193u,307951937u},
        { 10041117u, 28960672u, 48420531u, 71664411u, 83016360u,106800418u,
         122476460u,150044571u,163654473u,184259024u,206781966u,224776026u,
         248853488u,273290488u,294849492u,303242927u},
    };

    uint64_t mine[5 * 16];
    ds4_ple_history h;
    ds4_ple_history_reset(&c, &h);
    ds4_ple_row_ids(&c, &h, tokens, 5, mine);
    CHECK(memcmp(mine, expected, sizeof(expected)) == 0,
          "our row ids equal the reference row ids exactly");

    /* Our own scan implementation of the rule must reach the same vector, so
     * agreement is three ways and not two. */
    uint64_t scanned[5 * 16];
    row_ids_by_scan(&c, tokens, 5, scanned);
    CHECK(memcmp(scanned, expected, sizeof(expected)) == 0,
          "the scan rule equals the reference row ids exactly");

    /* Their split check (test_qwen4_host.c:471-486): starting a pass at the
     * end-of-sequence token must give the same ids as one uninterrupted pass.
     * We restate it against our carried history. */
    uint64_t split[3 * 16];
    ds4_ple_history_reset(&c, &h);
    ds4_ple_row_ids(&c, &h, tokens, 2, split);
    ds4_ple_row_ids(&c, &h, tokens + 2, 3, split);
    CHECK(memcmp(split, expected[2], sizeof(split)) == 0,
          "a pass split at the end-of-sequence token matches");
}

static void test_head_vocab_rule(void) {
    ds4_ple_constants pinned;
    pinned_constants(&pinned);

    CHECK(ds4_ple_head_vocab_follows_rule(pinned.head_vocab_sizes,
                                          pinned.head_count),
          "the pinned head vocabularies are consecutive primes");

    /* Negative control: one head moved to the prime after its neighbour's
     * neighbour -- still prime, still ascending, still a self-consistent
     * partition, and still refused. */
    uint64_t skewed[DS4_PLE_MAX_HEADS];
    memcpy(skewed, pinned.head_vocab_sizes, sizeof(skewed));
    skewed[4] = 20000063ull;  /* the size head 5 carries, so head 4 skips one */
    CHECK(!ds4_ple_head_vocab_follows_rule(skewed, pinned.head_count),
          "a skipped prime is refused");

    /* Negative control: a composite in the table. */
    memcpy(skewed, pinned.head_vocab_sizes, sizeof(skewed));
    skewed[0] = 20000004ull;
    CHECK(!ds4_ple_head_vocab_follows_rule(skewed, pinned.head_count),
          "a composite head vocabulary is refused");
}

int main(int argc, char **argv) {
    if (argc >= 2 && strcmp(argv[1], "constants") == 0) {
        return probe_constants(argc - 2, argv + 2);
    }
    if (argc >= 3 && strcmp(argv[1], "ids") == 0) {
        return probe_ids(argv[2], argc - 3, argv + 3);
    }
    if (argc >= 5 && strcmp(argv[1], "rows") == 0) {
        return probe_rows(argv[2], strtoull(argv[3], NULL, 10), argc - 4, argv + 4);
    }
    if (argc >= 4 && strcmp(argv[1], "raw") == 0) {
        return probe_raw(argv[2], argc - 3, argv + 3);
    }
    if (argc >= 7 && strcmp(argv[1], "soak") == 0) {
        return probe_soak(strtoull(argv[2], NULL, 10), strtoull(argv[3], NULL, 10),
                          strtoull(argv[4], NULL, 10), strtoull(argv[5], NULL, 10),
                          argc - 6, argv + 6);
    }
    if (argc > 1) {
        fprintf(stderr,
                "usage: %s [constants GGUF... | ids TOKENS GGUF... | "
                "rows ROWS CACHE_BYTES GGUF... | raw ROWS GGUF... | "
                "soak COUNT SEED DISTINCT CACHE_BYTES GGUF...]\n", argv[0]);
        return 2;
    }

    RUN(test_row_ids_match_the_scan_rule);
    RUN(test_prefill_and_decode_agree);
    RUN(test_eos_boundary_shape);
    RUN(test_row_ids_stay_inside_the_table);
    RUN(test_dequant_iq4_nl);
    RUN(test_refusal_without_a_checkpoint);
    RUN(test_hash_constants_derivation);
    RUN(test_head_vocab_rule);
    RUN(test_reference_golden_row_ids);

    fprintf(stderr, "%d/%d checks passed\n", g_total - g_failed, g_total);
    return g_failed == 0 ? 0 : 1;
}
