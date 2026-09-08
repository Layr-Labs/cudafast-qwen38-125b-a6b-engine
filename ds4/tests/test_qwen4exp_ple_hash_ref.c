/*
 * The PLE n-gram row selection against the reference, on real prompt ids.
 *
 * tests/test_qwen4exp_ple.c checks the multiplier DERIVATION against the
 * published triple.  It does not check the row ids the hash then names, and
 * the derivation is only the first of four steps: multiply, XOR in n-gram
 * position order, modulo the per-head prime vocabulary, add the per-head
 * offset -- with an end-of-sequence rule shifting the history underneath.  A
 * port can get the triple right and still address the wrong rows.
 *
 * The golden table below is the reference's answer for one real prompt, the
 * PRODUCTION constants, and every position and head:
 *
 *   ids       11762 279 20438 1881 279 9212 6681 13, then the generated 271
 *   ngram 3, heads_per_ngram 8, so 16 heads; row_dim 160
 *   vocab_size 248320, eos_token_id 248044
 *   seed 1234, ple_layer_index 0 (the POSITION in ple_layer_ids, not the
 *   layer number: the checkpoint's ple_layer_ids is [2] one-based / [1]
 *   zero-based, and position 0 is what reproduces the artifact's multipliers)
 *
 * Provenance of the golden.  It was computed from
 * transformers.models.qwen4_exp.modeling_qwen4_exp.Qwen4ExpTextNGramEmbedding
 * -- _splitmix64, _build_layer_multipliers, _find_nth_prime_after,
 * _shift_right_ignore_eos and forward() -- transplanted term for term, and the
 * per-head vocabularies, offsets and multipliers it derives are BYTE EQUAL to
 * the ones the shipped artifact publishes (per_head_vocabulary_sizes,
 * per_head_offsets, layer_multipliers).  Three independent sources agree on
 * the constants, so the table below is the reference's, not this port's.
 *
 * The head partition is stated rather than derived: the loader reads
 * head_vocab_sizes and head_offsets off the GGUF, and this port has no second
 * derivation of them.  ds4_ple_head_vocab_follows_rule() is the only rule
 * check, and it holds for any base and any ordinal, so it cannot catch a file
 * seeded on the wrong ordinal -- the multiplier tripwire is what catches that.
 */

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ds4_qwen4exp_ple.h"

enum { NGRAM = 3, HEADS_PER = 8, HEADS = (NGRAM - 1) * HEADS_PER,
       VOCAB = 248320, EOS = 248044, ORDINAL = 0, SEED = 1234 };

/* tokens */
static const int32_t PROMPT[] = {11762, 279, 20438, 1881, 279, 9212, 6681, 13, 271};
static const uint64_t HEAD_VOCAB[16] = {
    20000003ull, 20000023ull, 20000033ull, 20000047ull, 20000059ull, 20000063ull, 20000069ull, 20000077ull,
    20000081ull, 20000093ull, 20000107ull, 20000147ull, 20000153ull, 20000159ull, 20000161ull, 20000171ull };
static const uint64_t HEAD_OFFSET[16] = {
    0ull, 20000003ull, 40000026ull, 60000059ull, 80000106ull, 100000165ull, 120000228ull, 140000297ull,
    160000374ull, 180000455ull, 200000548ull, 220000655ull, 240000802ull, 260000955ull, 280001114ull, 300001275ull };
static const uint64_t MULTIPLIERS[3] = { 23703573157769ull, 20109073645365ull, 8052911324071ull };
static const uint64_t GOLDEN[9][16] = {
    { 15277278ull, 32918184ull, 45570257ull, 79574599ull, 98420318ull, 118853009ull, 120268271ull, 143585820ull,
      175176060ull, 197288956ull, 202542870ull, 236741304ull, 240513213ull, 265496279ull, 287426521ull, 319095437ull },
    { 3506531ull, 35278321ull, 51346456ull, 74045951ull, 93692189ull, 100279811ull, 130197757ull, 143489633ull,
      168265205ull, 190492688ull, 208057679ull, 236504748ull, 241009979ull, 265838367ull, 280852996ull, 316464169ull },
    { 9707655ull, 26790501ull, 55692277ull, 60558268ull, 87961050ull, 103838869ull, 137727701ull, 149713890ull,
      168972593ull, 196388648ull, 218897137ull, 234882499ull, 259677119ull, 264575062ull, 299564304ull, 314681696ull },
    { 1089119ull, 32936352ull, 59139140ull, 68135645ull, 96137298ull, 118864111ull, 123010066ull, 148642291ull,
      166098091ull, 190763263ull, 219878895ull, 225081067ull, 258119112ull, 271224180ull, 295607623ull, 317635993ull },
    { 1925516ull, 33986868ull, 50048194ull, 76568385ull, 87903251ull, 118354825ull, 124038192ull, 144960834ull,
      169069492ull, 183132154ull, 219725600ull, 228244006ull, 255663672ull, 263120157ull, 285614027ull, 318144363ull },
    { 6670162ull, 35774284ull, 50492745ull, 75284965ull, 85279893ull, 115313781ull, 120397769ull, 140571919ull,
      164020807ull, 181496072ull, 205422666ull, 232146031ull, 251310412ull, 270515431ull, 296926295ull, 309047379ull },
    { 6218976ull, 27402156ull, 58027669ull, 72941309ull, 85759673ull, 110039752ull, 136466625ull, 145048353ull,
      170266358ull, 185839461ull, 207384340ull, 226338365ull, 244214111ull, 262098347ull, 288061806ull, 317892456ull },
    { 8606605ull, 33015020ull, 55319809ull, 62659104ull, 97626016ull, 102636423ull, 120172177ull, 150257424ull,
      164654057ull, 196421419ull, 200206175ull, 239925545ull, 255926232ull, 271938061ull, 283944533ull, 303995141ull },
    { 5129529ull, 22831471ull, 51687175ull, 68090399ull, 90726650ull, 118273134ull, 139593745ull, 154689607ull,
      170383671ull, 190827692ull, 211389031ull, 233249738ull, 253561819ull, 273882474ull, 293991371ull, 314549502ull },
};

static int g_failures;

static void fail(const char *what) {
    fprintf(stderr, "test_qwen4exp_ple_hash_ref: %s\n", what);
    g_failures++;
}

int main(void) {
    const uint32_t n_tokens = (uint32_t)(sizeof(PROMPT) / sizeof(PROMPT[0]));

    ds4_ple_constants c;
    memset(&c, 0, sizeof(c));
    c.ngram_size = NGRAM;
    c.heads_per_ngram = HEADS_PER;
    c.head_count = HEADS;
    c.vocab_size = VOCAB;
    c.eos_token_id = EOS;
    c.row_dim = 160;
    for (uint32_t i = 0; i < HEADS; i++) {
        c.head_vocab_sizes[i] = HEAD_VOCAB[i];
        c.head_offsets[i] = HEAD_OFFSET[i];
    }

    /* OUR derivation of the multipliers, against the artifact's published
     * triple.  A port that reads the triple out of the file and never derives
     * it would pass every row check below on a file that carries the wrong
     * one. */
    ds4_ple_derive_multipliers(NGRAM, VOCAB, ORDINAL, SEED, c.multipliers);
    for (uint32_t i = 0; i < NGRAM; i++) {
        if (c.multipliers[i] != MULTIPLIERS[i]) {
            fprintf(stderr,
                    "multiplier %u is %" PRIu64 ", the artifact publishes %"
                    PRIu64 "\n", i, c.multipliers[i], MULTIPLIERS[i]);
            g_failures++;
        }
    }

    uint64_t ids[9 * HEADS];
    ds4_ple_history h;
    ds4_ple_history_reset(&c, &h);
    ds4_ple_row_ids(&c, &h, PROMPT, n_tokens, ids);

    uint32_t bad = 0;
    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t k = 0; k < HEADS; k++) {
            const uint64_t got = ids[(size_t)t * HEADS + k];
            if (got == GOLDEN[t][k]) continue;
            if (bad < 8u) {
                fprintf(stderr,
                        "position %u (token %d) head %u: ours %" PRIu64
                        ", reference %" PRIu64 "\n",
                        t, PROMPT[t], k, got, GOLDEN[t][k]);
            }
            bad++;
        }
    }
    if (bad) {
        fprintf(stderr, "%u of %u row ids disagree with the reference\n",
                bad, n_tokens * HEADS);
        g_failures++;
    } else {
        printf("  %u row ids over %u positions and %u heads match the "
               "reference\n", n_tokens * HEADS, n_tokens, (uint32_t)HEADS);
    }

    /* CALL SHAPE.  A decode step feeds one token at a time through the carried
     * history; a prefill feeds the batch.  The ids must not know which. */
    {
        ds4_ple_history one;
        ds4_ple_history_reset(&c, &one);
        for (uint32_t t = 0; t < n_tokens; t++) {
            uint64_t row[HEADS];
            ds4_ple_row_ids(&c, &one, &PROMPT[t], 1, row);
            for (uint32_t k = 0; k < HEADS; k++) {
                if (row[k] != GOLDEN[t][k]) {
                    fail("a one-token call does not reproduce the batch");
                    t = n_tokens;
                    break;
                }
            }
        }
    }

    /* NEGATIVE CONTROLS.  Both of these are conventions a port has actually
     * shipped: the MLX Swift runner defaults the seed to 0, and reading
     * ple_layer_ids as the layer NUMBER rather than the position gives
     * ordinal 1 or 2.  Each must move the multipliers. */
    {
        uint64_t m[NGRAM];
        ds4_ple_derive_multipliers(NGRAM, VOCAB, ORDINAL, 0, m);
        if (memcmp(m, MULTIPLIERS, sizeof(m)) == 0) {
            fail("seed 0 derives the artifact's triple, so the seed is not "
                 "being used");
        }
        ds4_ple_derive_multipliers(NGRAM, VOCAB, 1u, SEED, m);
        if (memcmp(m, MULTIPLIERS, sizeof(m)) == 0) {
            fail("ordinal 1 derives the artifact's triple, so the ordinal is "
                 "not being used");
        }
    }

    if (g_failures) {
        printf("Qwen4-Exp PLE hash reference test: FAIL (%d)\n", g_failures);
        return 1;
    }
    puts("Qwen4-Exp PLE hash reference test: PASS");
    return 0;
}
