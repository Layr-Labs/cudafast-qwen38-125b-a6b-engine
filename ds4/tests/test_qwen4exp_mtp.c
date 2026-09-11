/*
 * qwen4exp MTP: the exactness property, the rollback contract, the counters
 * and the head wiring.
 *
 * The load-bearing test is the first one: a greedy MTP stream must equal the
 * greedy serial stream token for token.  It runs against a reduced reference
 * model that carries the same STATE CLASSES the real graph does -- an in-place
 * recurrent state, an in-place conv history, an in-place n-gram history, an
 * append-only KV cache with its own cursor, an append-only indexer tape with
 * pooled blocks derived from it, and the head's own cache -- so the property
 * is checked against the shape of the state, not against block internals.  The
 * QSA and MoE hooks the real block needs are pass-throughs here, documented at
 * the head test below and used exactly the way L7's integration test uses them
 * until L4 is merged.
 *
 * Every rollback object then gets a negative control: with that one object's
 * rollback disabled the property must STOP holding.  A rollback nobody can
 * break is a rollback nobody needs.
 */

#define _POSIX_C_SOURCE 200809L

#include "../ds4_qwen4exp_mtp.h"

#include <math.h>
#include <stdarg.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_failures = 0;
static char g_err[512];

#define CHECK(cond, ...) do { \
    if (!(cond)) { \
        printf("  FAIL %s:%d: ", __func__, __LINE__); \
        printf(__VA_ARGS__); \
        printf("\n"); \
        g_failures++; \
    } \
} while (0)

/* ========================================================================
 * A reduced reference model
 * ======================================================================== */

#define REF_CAP     512u
#define REF_VOCAB   61u
#define REF_N_EMBD  4u
#define REF_N_HC    4u
#define REF_HC_DIM  (REF_N_EMBD * REF_N_HC)
#define REF_POOL    4u    /* the indexer's compress ratio */

/* Which rollback object is sabotaged, for the negative controls. */
typedef enum {
    BREAK_NONE = -1,
    BREAK_GDN_RECURRENT = 0,
    BREAK_GDN_CONV,
    BREAK_QSA_KV,
    BREAK_INDEXER_POOL,
    BREAK_PLE,
    BREAK_PLE_CONV,
    BREAK_HEAD_CACHE,
    /* The state slot is there and holds the right value, but the wrong one is
     * adopted: one row too far, which is the state including a token the round
     * rejected.  Distinct from a select that does nothing. */
    BREAK_SELECT_ONE_ROW,
    /* The one-row rollback bugs: not a no-op, an OFF-BY-ONE.  A depth-N round
     * that rejects has to unwind every speculative row, and a rollback that
     * stops one row short is the failure that a no-op mutant cannot stand in
     * for -- it leaves the cursors self-consistent and only one stale row
     * behind. */
    BREAK_HEAD_CACHE_ONE_ROW,
    BREAK_QSA_KV_ONE_ROW,
    /* The verify leaves EVERY row's logits and the accept loop reads row a's.
     * This mutant gives every row the LAST row's -- the one row a
     * last-row-only verify would have left standing -- so a loop that reads
     * the wrong row, or a verify that fills only one, is caught here rather
     * than at whatever token it first mis-accepts on the real model. */
    BREAK_VERIFY_ROW_LOGITS,
    BREAK_COUNT,
} ref_break;

static const char *ref_break_name(ref_break b) {
    switch (b) {
    case BREAK_GDN_RECURRENT: return "gdn recurrent select is a no-op";
    case BREAK_GDN_CONV:      return "gdn conv select is a no-op";
    case BREAK_QSA_KV:        return "qsa kv truncate is a no-op";
    case BREAK_INDEXER_POOL:  return "indexer truncate drops the tape but "
                                     "keeps the pooled blocks";
    case BREAK_PLE:           return "ple n-gram history select is a no-op";
    case BREAK_PLE_CONV:      return "ple conv state select is a no-op";
    case BREAK_HEAD_CACHE:    return "mtp head cache truncate is a no-op";
    case BREAK_SELECT_ONE_ROW:
                              return "gdn recurrent adopts the slot one row "
                                     "past the accepted length";
    case BREAK_HEAD_CACHE_ONE_ROW:
                              return "mtp head cache truncate stops one row "
                                     "short";
    case BREAK_QSA_KV_ONE_ROW:
                              return "qsa kv truncate stops one row short";
    case BREAK_VERIFY_ROW_LOGITS:
                              return "every verify row carries the LAST row's "
                                     "logits";
    case BREAK_NONE:
    case BREAK_COUNT:
    default:                  return "nothing";
    }
}

typedef struct {
    /* SNAPSHOT class: overwritten in place, no position of their own. */
    uint64_t recurrent;
    uint64_t conv[3];
    uint64_t ple[9];        /* the 9-row dilation-3 convolution state      */
    uint64_t ple_hist[2];   /* the two previous token ids the hash folds in */
    /* The per-row slots the VERIFY writes: slot k is every running object as
     * it stood after row k.  The real graph writes these inside the GDN and
     * PLE kernels; here the reference model writes them from the same place
     * the kernels would, which is the row loop of the verify. */
    struct {
        uint64_t recurrent;
        uint64_t conv[3];
        uint64_t ple[9];
        uint64_t ple_hist[2];
        int      written;
    } slot[DS4_QWEN4EXP_IMPLEMENTED_DEPTH];
    uint64_t selects;

    /* TRUNCATE class: append-only, each with its own cursor. */
    uint64_t kv[REF_CAP];    uint32_t kv_len;
    uint64_t tape[REF_CAP];  uint32_t tape_len;
    uint64_t pool[REF_CAP / REF_POOL]; uint32_t pool_len;
    uint64_t head[REF_CAP];  uint32_t head_len;
    uint32_t head_stamp[REF_CAP];

    ref_break broken;
    int  batch_variant;  /* perturb multi-row rows: the batch-invariance control */
    int  faulted;
    char fault[256];

    uint64_t n_decode, n_verify, n_head, n_draft, n_read_logit;

    /* The two answers a rejecting round produces for the same position: the
     * batched verify's row-0 argmax, through the borrowed head, and the
     * one-row replay's.  Recorded here because the cycle does not report
     * them, and the ruling on which one the frontier must follow cannot be
     * asserted from outside without both. */
    int last_head_argmax;    /* the last row the one-row head was run over */
    int last_decode_argmax;  /* the one-row replay's           */
    /* The argmax of the LAST row of the last forward of ANY width.  The
     * frontier the cycle leaves must always be this: at depth 1 the replay was
     * always one row wide and last_decode_argmax said the same thing, but a
     * deeper round replays its accepted prefix at width a + 1. */
    int last_frontier_argmax;
    /* Every verify row's argmax, as the forward left it, and how many rows the
     * last verify wrote.  The frontier a round leaves must be the argmax of
     * the row it committed through, and that row is now one of these. */
    int      row_argmax[DS4_QWEN4EXP_MTP_MAX_COMMIT];
    uint32_t row_argmax_n;
    float compact_logits[DS4_QWEN4EXP_MTP_MAX_COMMIT][REF_VOCAB];
    /* The speculative chain the head is walking, so the draft oracle can look
     * one position past the frontier for each step.  Without this every step
     * after the first would draft from the frontier again and depth 2 and 3
     * would accept almost nothing. */
    int      chain_tok[DS4_QWEN4EXP_IMPLEMENTED_DEPTH + 1];
    uint32_t chain_len;
} refmodel;

static uint64_t mix64(uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

static void ref_reset(refmodel *m, ref_break broken, int batch_variant) {
    memset(m, 0, sizeof(*m));
    m->recurrent = 0x243f6a8885a308d3ULL;
    for (int i = 0; i < 3; i++) m->conv[i] = 0x13198a2e03707344ULL + (uint64_t)i;
    for (int i = 0; i < 9; i++) m->ple[i] = 0xa4093822299f31d0ULL + (uint64_t)i;
    m->broken = broken;
    m->batch_variant = batch_variant;
}

static int ref_fault(refmodel *m, const char *fmt, ...) {
    if (!m->faulted) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(m->fault, sizeof(m->fault), fmt, ap);
        va_end(ap);
        m->faulted = 1;
    }
    return -1;
}

/* Encode the 64-bit hidden state into an hc row and back.  Four 16-bit limbs
 * are exact in float32, so the borrowed head reads back exactly what the
 * target wrote -- which is what the real head does with the pre-final-mixer
 * stream, only there the reversal is the LM head's business. */
static void ref_hidden_to_hc(uint64_t h, float *hc) {
    for (uint32_t i = 0; i < 4; i++) {
        hc[i] = (float)((h >> (16u * i)) & 0xFFFFu);
    }
    for (uint32_t i = 4; i < REF_HC_DIM; i++) {
        hc[i] = (float)((mix64(h + i) >> 40) & 0xFFFFu);
    }
}

static uint64_t ref_hc_to_hidden(const float *hc) {
    uint64_t h = 0;
    for (uint32_t i = 0; i < 4; i++) {
        h |= (uint64_t)(uint32_t)hc[i] << (16u * i);
    }
    return h;
}

static void ref_logits(uint64_t h, float *logits) {
    for (uint32_t v = 0; v < REF_VOCAB; v++) {
        const uint64_t r = mix64(h ^ (0x9E3779B97F4A7C15ULL * (v + 1u)));
        logits[v] = (float)((r >> 40) & 0xFFFFu) * (1.0f / 65536.0f);
    }
}

/* One row.  Every state class is read, so a class that fails to roll back
 * changes the hidden state and therefore the token stream. */
static int ref_step(refmodel *m, int token, uint32_t pos, uint32_t batch_n,
                    uint64_t *hidden_out) {
    if (pos >= REF_CAP) return ref_fault(m, "position %u past the cap", pos);
    if (m->kv_len != pos) {
        return ref_fault(m, "kv cursor is %u at position %u", m->kv_len, pos);
    }
    if (m->tape_len != pos) {
        return ref_fault(m, "tape cursor is %u at position %u", m->tape_len, pos);
    }

    m->conv[2] = m->conv[1]; m->conv[1] = m->conv[0];
    m->conv[0] = mix64((uint64_t)(uint32_t)token * 31u + pos);
    m->recurrent = mix64(m->recurrent * 1000003ULL + (uint64_t)(uint32_t)token);
    for (int i = 8; i > 0; i--) m->ple[i] = m->ple[i - 1];
    m->ple[0] = mix64((uint64_t)(uint32_t)token ^ (uint64_t)pos << 13);
    m->ple_hist[1] = m->ple_hist[0];
    m->ple_hist[0] = (uint64_t)(uint32_t)token;

    m->kv[m->kv_len++] = mix64((uint64_t)(uint32_t)token ^ ((uint64_t)pos << 7));
    m->tape[m->tape_len++] = mix64((uint64_t)(uint32_t)token + 17ull * pos);
    while ((m->pool_len + 1u) * REF_POOL <= m->tape_len) {
        uint64_t acc = 0;
        for (uint32_t i = 0; i < REF_POOL; i++) {
            acc ^= m->tape[m->pool_len * REF_POOL + i];
        }
        m->pool[m->pool_len++] = mix64(acc);
    }

    uint64_t h = m->recurrent;
    for (int i = 0; i < 3; i++) h = mix64(h ^ m->conv[i]);
    for (int i = 0; i < 9; i++) h = mix64(h ^ m->ple[i]);
    for (int i = 0; i < 2; i++) h = mix64(h ^ m->ple_hist[i]);
    for (uint32_t i = 0; i < m->kv_len; i++) h ^= m->kv[i];
    for (uint32_t i = 0; i < m->pool_len; i++) h = mix64(h ^ m->pool[i]);
    h = mix64(h ^ ((uint64_t)m->pool_len << 32) ^ m->kv_len);
    if (m->batch_variant && batch_n > 1u) h = mix64(h ^ (0x5bf03635ULL * batch_n));
    *hidden_out = h;
    return 0;
}

/*
 * The verify, leaving EVERY row's logits.
 *
 * `row_logits` is [n][REF_VOCAB].  Row t's are written from the same hidden
 * state the row's hc goes out as, and checked against what the borrowed head
 * makes of that hc row: the cycle's accept loop reads these instead of calling
 * head_logits per row, so the two have to be the same numbers.  On the real
 * graph that is the LM head at n rows against the LM head at one, which is why
 * the head is routed through the decode-order entry; here it is exact by
 * construction and the check is what keeps it that way.
 */
static int ref_head_logits(void *ctx, const float *hc_row, float *logits);

static int ref_verify_rows(void *ctx, const int *tokens, uint32_t n,
                           uint32_t pos0, float *hc_rows, float *row_logits) {
    refmodel *m = ctx;
    m->n_verify++;
    for (int k = 0; k < DS4_QWEN4EXP_IMPLEMENTED_DEPTH; k++) {
        m->slot[k].written = 0;
    }
    if (n > (uint32_t)DS4_QWEN4EXP_MTP_MAX_COMMIT) {
        return ref_fault(m, "verify of %u rows is wider than the envelope", n);
    }
    for (uint32_t t = 0; t < n; t++) {
        uint64_t h = 0;
        if (ref_step(m, tokens[t], pos0 + t, n, &h) != 0) return -1;
        /* Mirror the running state after this row.  Only the DRAFTED rows get
         * a slot: the last row's state is the live state. */
        if (t + 1u < n && t < (uint32_t)DS4_QWEN4EXP_IMPLEMENTED_DEPTH) {
            m->slot[t].recurrent = m->recurrent;
            memcpy(m->slot[t].conv, m->conv, sizeof(m->conv));
            memcpy(m->slot[t].ple, m->ple, sizeof(m->ple));
            memcpy(m->slot[t].ple_hist, m->ple_hist, sizeof(m->ple_hist));
            m->slot[t].written = 1;
        }
        float *const hc = hc_rows + (size_t)t * REF_HC_DIM;
        float *const row = row_logits + (size_t)t * REF_VOCAB;
        ref_hidden_to_hc(h, hc);
        ref_logits(h, row);
        /* The wide head against the one-row head, per row, through the seam's
         * own one-row entry. */
        float one_row[REF_VOCAB];
        if (ref_head_logits(m, hc, one_row) != 0) return -1;
        if (memcmp(row, one_row, sizeof(one_row)) != 0) {
            return ref_fault(m, "verify row %u disagrees with the borrowed "
                             "head over the same hc row", t);
        }
        if (t + 1u == n) {
            m->last_frontier_argmax = ds4_qwen4exp_mtp_argmax(row, REF_VOCAB);
        }
    }
    /* THE MUTANT: every row carries the LAST row's logits, which is the one
     * row a last-row-only verify would have left standing. */
    if (m->broken == BREAK_VERIFY_ROW_LOGITS && n > 1u) {
        const float *last = row_logits + (size_t)(n - 1u) * REF_VOCAB;
        for (uint32_t t = 0; t + 1u < n; t++) {
            memcpy(row_logits + (size_t)t * REF_VOCAB, last,
                   REF_VOCAB * sizeof(float));
        }
    }
    for (uint32_t t = 0; t < n; t++) {
        m->row_argmax[t] = ds4_qwen4exp_mtp_argmax(
                row_logits + (size_t)t * REF_VOCAB, REF_VOCAB);
    }
    m->row_argmax_n = n;
    return 0;
}

/* Exercise the production compact-logit seam with the same reduced model.
 * The target computation is still ref_verify_rows; only its return transport
 * changes, exactly as on CUDA. */
static int ref_verify_rows_top1(void *ctx, const int *tokens, uint32_t n,
                                uint32_t pos0, float *hc_rows,
                                int *row_top1) {
    refmodel *m = ctx;
    if (ref_verify_rows(ctx, tokens, n, pos0, hc_rows,
                        &m->compact_logits[0][0]) != 0) return -1;
    for (uint32_t t = 0; t < n; t++) {
        row_top1[t] = ds4_qwen4exp_mtp_argmax(m->compact_logits[t], REF_VOCAB);
    }
    return 0;
}

static int ref_read_logit_row(void *ctx, uint32_t row, float *logits) {
    refmodel *m = ctx;
    if (row >= (uint32_t)DS4_QWEN4EXP_MTP_MAX_COMMIT) return -1;
    m->n_read_logit++;
    memcpy(logits, m->compact_logits[row], REF_VOCAB * sizeof(float));
    return 0;
}

static int ref_decode_token(void *ctx, int token, uint32_t pos,
                            float *hc_row, float *logits) {
    refmodel *m = ctx;
    m->n_decode++;
    uint64_t h = 0;
    if (ref_step(m, token, pos, 1u, &h) != 0) return -1;
    ref_hidden_to_hc(h, hc_row);
    ref_logits(h, logits);
    m->last_decode_argmax = ds4_qwen4exp_mtp_argmax(logits, REF_VOCAB);
    m->last_frontier_argmax = m->last_decode_argmax;
    return 0;
}

static int ref_head_logits(void *ctx, const float *hc_row, float *logits) {
    refmodel *m = ctx;
    m->n_head++;
    ref_logits(ref_hc_to_hidden(hc_row), logits);
    m->last_head_argmax = ds4_qwen4exp_mtp_argmax(logits, REF_VOCAB);
    return 0;
}

/*
 * The head's own cache is written here, one row per position.  The cursor
 * check is the head-cache invariant: every call must land exactly on the row
 * after the last one written, so a rollback that left the cache long or short
 * faults by name on the next seed rather than quietly drafting from a stale
 * row.  Rows above the committed frontier are the chain's speculation and are
 * expected; the next round's truncate is what removes them.
 */
static int ref_draft_step(void *ctx, int next_token, const float *hc_row,
                          uint32_t pos, int *draft_out, float *multi_out) {
    refmodel *m = ctx;
    m->n_draft++;
    if (pos >= REF_CAP) return ref_fault(m, "head position %u past the cap", pos);
    if (m->head_len != pos) {
        return ref_fault(m, "head cursor is %u at position %u", m->head_len, pos);
    }
    const uint64_t h = ref_hc_to_hidden(hc_row);
    m->head[pos] = mix64(h ^ (uint64_t)(uint32_t)next_token);
    m->head_stamp[pos] = 1u;
    m->head_len = pos + 1u;
    for (uint32_t i = 0; i < m->head_len; i++) {
        if (!m->head_stamp[i]) {
            return ref_fault(m, "the head cache has a hole at %u of %u",
                             i, m->head_len);
        }
    }
    /* The row the NEXT chain step reads in place of a target row.  It has to
     * be a function of the cache row just written, because that is what makes
     * the chain a recursion rather than three drafts from one state. */
    if (multi_out) ref_hidden_to_hc(m->head[pos], multi_out);

    /*
     * Where this step sits.  `pos` is the head cache row being written, and
     * the committed frontier is m->kv_len:
     *
     *   pos + 1 <  kv_len   a SEED over a row this round committed; its draft
     *                       is discarded, so no oracle is owed.
     *   pos + 1 == kv_len   chain step 0, drafting from the frontier.
     *   pos + 1 >  kv_len   chain step k, drafting from k speculative tokens
     *                       past the frontier.
     */
    if (pos + 1u < m->kv_len) { *draft_out = 0; return 0; }
    const uint32_t k = pos + 1u - m->kv_len;
    if (k == 0u) m->chain_len = 0u;
    if (k != m->chain_len) {
        return ref_fault(m, "chain step at %u expected %u carried tokens, "
                         "the head has %u", pos, k, m->chain_len);
    }
    if (m->chain_len >= (uint32_t)(DS4_QWEN4EXP_IMPLEMENTED_DEPTH + 1)) {
        return ref_fault(m, "chain of %u is longer than the envelope",
                         m->chain_len);
    }

    /*
     * The draft ORACLE, deliberately wrong on a third of the states.  A hash
     * drafter over this vocabulary accepts about one round in sixty and would
     * leave the accepting branch barely exercised; an oracle at two thirds
     * runs both branches hundreds of times over a 256-token leg, and at depth
     * 3 it makes a full four-token accept happen about three rounds in ten.
     *
     * It runs the target's own step on a throwaway copy of the state, walked
     * forward over the chain's speculative tokens first, so it reads nothing
     * the cycle has not already committed or itself just drafted.
     */
    refmodel *shadow = malloc(sizeof(*shadow));
    if (!shadow) return ref_fault(m, "out of memory");
    *shadow = *m;
    int rc = 0;
    uint64_t h2 = 0;
    for (uint32_t j = 0; j < m->chain_len && rc == 0; j++) {
        rc = ref_step(shadow, m->chain_tok[j], shadow->kv_len, 1u, &h2);
    }
    if (rc == 0) rc = ref_step(shadow, next_token, shadow->kv_len, 1u, &h2);
    const int faulted = shadow->faulted;
    const char *why = faulted ? shadow->fault : NULL;
    int guess = 0;
    if (rc == 0) {
        float logits[REF_VOCAB];
        ref_logits(h2, logits);
        guess = ds4_qwen4exp_mtp_argmax(logits, REF_VOCAB);
        /* Wrong on a third of the states, chosen by the state itself so the
         * accept/reject pattern differs from prompt to prompt. */
        if (mix64(h2 ^ 0xd1b54a32d192ed03ULL) % 3u == 0u) {
            guess = (guess + 1) % (int)REF_VOCAB;
        }
    }
    if (faulted) (void)ref_fault(m, "draft shadow step: %s", why);
    free(shadow);
    if (rc != 0) return -1;
    m->chain_tok[m->chain_len++] = next_token;
    *draft_out = guess;
    return 0;
}

/* ---- the rollback objects ---------------------------------------------- */

/*
 * The SELECT_ROW verbs.  Each adopts the slot the verify left after `row`.
 * A missing slot is a refusal, not a silent no-op: the cycle only ever asks
 * for a row it told the forward to snapshot, so an unwritten slot means the
 * two disagree about the width of the round.
 */
static const void *rb_slot(refmodel *m, uint32_t row) {
    if (row >= (uint32_t)DS4_QWEN4EXP_IMPLEMENTED_DEPTH) return NULL;
    if (!m->slot[row].written) return NULL;
    return &m->slot[row];
}

static int rb_recurrent_select(void *ctx, uint32_t row) {
    refmodel *m = ctx;
    m->selects++;
    /* One row too far: the state including a token the round rejected. */
    const uint32_t take =
        (m->broken == BREAK_SELECT_ONE_ROW &&
         row + 1u < (uint32_t)DS4_QWEN4EXP_IMPLEMENTED_DEPTH &&
         m->slot[row + 1u].written) ? row + 1u : row;
    if (!rb_slot(m, take)) return -1;
    if (m->broken != BREAK_GDN_RECURRENT) m->recurrent = m->slot[take].recurrent;
    return 0;
}
static int rb_conv_select(void *ctx, uint32_t row) {
    refmodel *m = ctx;
    if (!rb_slot(m, row)) return -1;
    if (m->broken != BREAK_GDN_CONV) {
        memcpy(m->conv, m->slot[row].conv, sizeof(m->conv));
    }
    return 0;
}
static int rb_ple_hist_select(void *ctx, uint32_t row) {
    refmodel *m = ctx;
    if (!rb_slot(m, row)) return -1;
    if (m->broken != BREAK_PLE) {
        memcpy(m->ple_hist, m->slot[row].ple_hist, sizeof(m->ple_hist));
    }
    return 0;
}
/* The convolution state is its own object with its own id: on the real session
 * it lives on the device while the n-gram history lives on the host, and a set
 * that rolled one back and not the other used to satisfy the check. */
static int rb_ple_conv_select(void *ctx, uint32_t row) {
    refmodel *m = ctx;
    if (!rb_slot(m, row)) return -1;
    if (m->broken != BREAK_PLE_CONV) {
        memcpy(m->ple, m->slot[row].ple, sizeof(m->ple));
    }
    return 0;
}

static int rb_kv_truncate(void *ctx, uint32_t pos) {
    refmodel *m = ctx;
    if (m->broken == BREAK_QSA_KV) return 0;
    /* One row short: the KV keeps a single rejected row.  ref_step's cursor
     * check then fires on the replay, which is the point -- a rollback that
     * lands one row past the accepted length is not a smaller version of a
     * rollback that does nothing, it is a different bug. */
    m->kv_len = (m->broken == BREAK_QSA_KV_ONE_ROW && m->kv_len > pos)
              ? pos + 1u : pos;
    return 0;
}
/* One step: the tape AND the blocks pooled from it.  Dropping the tape alone
 * leaves a block built from a rejected row, which is exactly the failure the
 * design warns about. */
static int rb_tape_truncate(void *ctx, uint32_t pos) {
    refmodel *m = ctx;
    m->tape_len = pos;
    if (m->broken != BREAK_INDEXER_POOL) m->pool_len = pos / REF_POOL;
    return 0;
}
static int rb_head_truncate(void *ctx, uint32_t pos) {
    refmodel *m = ctx;
    if (m->broken == BREAK_HEAD_CACHE) return 0;
    uint32_t to = pos;
    /* One row short: the cache keeps a single speculative row, which is
     * exactly the bug a no-op mutant cannot stand in for. */
    if (m->broken == BREAK_HEAD_CACHE_ONE_ROW && m->head_len > pos) to = pos + 1u;
    for (uint32_t i = to; i < m->head_len; i++) m->head_stamp[i] = 0u;
    if (m->head_len > to) m->head_len = to;
    return 0;
}

/* The batched seam entry, bound the way the graph binds it: the same rows
 * through the same one-row step, in order, keeping only the last row's
 * outputs.  The exactness property below therefore runs the cycle the way the
 * engine runs it -- seeds folded into the first draft -- and every head-cache
 * invariant ref_draft_step checks per row is checked per row here too. */
static int ref_draft_rows(void *ctx, const int *next_tokens,
                          const float *hc_rows, uint32_t pos0, uint32_t n,
                          int *draft_out, float *multi_out) {
    if (n == 0) return -1;
    for (uint32_t t = 0; t < n; t++) {
        int d = -1;
        const int last = (t + 1u == n);
        if (ref_draft_step(ctx, next_tokens[t],
                           hc_rows + (size_t)t * REF_HC_DIM, pos0 + t, &d,
                           last ? multi_out : NULL) != 0) {
            return -1;
        }
        if (last) *draft_out = d;
    }
    return 0;
}

static int g_ref_batched_draft = 1;

static int ref_build(refmodel *m, ds4_qwen4exp_mtp_model *model,
                     ds4_qwen4exp_rollback_set *set) {
    memset(model, 0, sizeof(*model));
    model->ctx = m;
    model->hc_dim = REF_HC_DIM;
    model->n_vocab = REF_VOCAB;
    model->verify_rows = ref_verify_rows;
    model->verify_rows_top1 = ref_verify_rows_top1;
    model->read_logit_row = ref_read_logit_row;
    model->decode_token = ref_decode_token;
    model->head_logits = ref_head_logits;
    model->draft_step = ref_draft_step;
    model->draft_rows = g_ref_batched_draft ? ref_draft_rows : NULL;

    ds4_qwen4exp_rollback_init(set);
    const struct { ds4_qwen4exp_state_id id; ds4_qwen4exp_rollback_object o; } objs[] = {
        { DS4_QWEN4EXP_STATE_GDN_RECURRENT, { m, rb_recurrent_select, NULL } },
        { DS4_QWEN4EXP_STATE_GDN_CONV,      { m, rb_conv_select, NULL } },
        { DS4_QWEN4EXP_STATE_PLE_HISTORY,   { m, rb_ple_hist_select, NULL } },
        { DS4_QWEN4EXP_STATE_PLE_CONV,      { m, rb_ple_conv_select, NULL } },
        { DS4_QWEN4EXP_STATE_QSA_KV,        { m, NULL, rb_kv_truncate } },
        { DS4_QWEN4EXP_STATE_QSA_INDEXER_TAPE,
                                            { m, NULL, rb_tape_truncate } },
        { DS4_QWEN4EXP_STATE_MTP_HEAD_CACHE,
                                            { m, NULL, rb_head_truncate } },
    };
    for (size_t i = 0; i < sizeof(objs) / sizeof(objs[0]); i++) {
        if (ds4_qwen4exp_rollback_register(set, objs[i].id, &objs[i].o,
                                           g_err, sizeof(g_err)) != 0) {
            printf("  register failed: %s\n", g_err);
            return -1;
        }
    }
    return ds4_qwen4exp_rollback_check(set, g_err, sizeof(g_err));
}

/* ========================================================================
 * The two legs
 * ======================================================================== */

/* The serial leg: one decode per token, argmax, feed it back. */
/*
 * Every carried object, compared field by field against a serial leg advanced
 * to the same length.  This is the "rejected rows leave no trace" contract as
 * an assertion rather than as a token-stream inference: a stale KV row or an
 * undecayed recurrent state can survive several rounds before it moves a
 * token, and by then the first divergence names the wrong round.
 *
 * The head cache is NOT compared here -- a serial leg never runs the head --
 * and is checked separately against the target hidden states the committed
 * rows must have been built from.  Returns NULL when they match, or the name
 * of the first object that does not.
 */
static const char *ref_carried_diff(const refmodel *a, const refmodel *b) {
    if (a->recurrent != b->recurrent)                       return "gdn recurrent";
    if (memcmp(a->conv, b->conv, sizeof(a->conv)) != 0)     return "gdn conv";
    if (memcmp(a->ple, b->ple, sizeof(a->ple)) != 0)        return "ple conv";
    if (memcmp(a->ple_hist, b->ple_hist, sizeof(a->ple_hist)) != 0) {
        return "ple n-gram history";
    }
    if (a->kv_len != b->kv_len)                             return "qsa kv length";
    if (memcmp(a->kv, b->kv, a->kv_len * sizeof(a->kv[0])) != 0) {
        return "qsa kv rows";
    }
    if (a->tape_len != b->tape_len)                         return "indexer tape length";
    if (memcmp(a->tape, b->tape, a->tape_len * sizeof(a->tape[0])) != 0) {
        return "indexer tape rows";
    }
    if (a->pool_len != b->pool_len)                         return "indexer pool length";
    if (memcmp(a->pool, b->pool, a->pool_len * sizeof(a->pool[0])) != 0) {
        return "indexer pooled blocks";
    }
    return NULL;
}

static int run_serial(int first_token, int n, int *out) {
    refmodel m;
    ref_reset(&m, BREAK_NONE, 0);
    float hc[REF_HC_DIM];
    float logits[REF_VOCAB];
    int pending = first_token;
    for (int i = 0; i < n; i++) {
        if (ref_decode_token(&m, pending, (uint32_t)i, hc, logits) != 0) {
            printf("  serial fault: %s\n", m.fault);
            return -1;
        }
        pending = ds4_qwen4exp_mtp_argmax(logits, REF_VOCAB);
        out[i] = pending;
    }
    return 0;
}

/* The MTP leg, driven the way harness/protocol-adapter drives it: a cycle per
 * round, produced tokens are the committed ones after the fed token plus the
 * frontier argmax, and the per-round acceptance length is what the cycle
 * returned. */
typedef struct {
    int      committed_total;
    int      rounds;
    int      acceptance_lengths[512];
    uint64_t drafted, accepted, quenches, disagreements;
    /* Rejecting rounds where the two answers differed AND the frontier was
     * checked against the replay's.  Non-vacuity for that assertion. */
    uint64_t follow_checks;
    /* Rounds whose rollback was checked against an explicit replay, and the
     * spread of acceptance lengths they covered.  The second is the
     * non-vacuity guard on the first: a check that only ever saw full accepts
     * never exercised a select at all. */
    uint64_t replay_checks;
    /* Head cache rows over the committed prefix, by where the head read its
     * hidden state: the TARGET's row (a seed) or its OWN previous row (a chain
     * step that the round then confirmed).  The second is the point of keeping
     * the accepted drafts' rows: a cycle that re-seeded every committed token
     * would report zero of them. */
    uint64_t head_rows_from_target;
    uint64_t head_rows_from_chain;
    uint64_t accept_hist[DS4_QWEN4EXP_MTP_MAX_COMMIT + 1];
    int      faulted;
    char     err[512];
} mtp_run;

static int run_mtp(int first_token, int n, int *out, mtp_run *run,
                   ref_break broken, int batch_variant, int depth) {
    refmodel m;
    ds4_qwen4exp_mtp_model model;
    ds4_qwen4exp_rollback_set set;
    ds4_qwen4exp_mtp_state st;

    memset(run, 0, sizeof(*run));
    ref_reset(&m, broken, batch_variant);
    if (ref_build(&m, &model, &set) != 0) return -1;
    if (ds4_qwen4exp_mtp_state_init(&st, depth, &set, REF_HC_DIM, REF_VOCAB,
                                    run->err, sizeof(run->err)) != 0) {
        run->faulted = 1;
        return -1;
    }

    float logits[REF_VOCAB];
    int pending = first_token;
    uint32_t pos = 0;
    int produced = 0;
    int rc = 0;

    /* The serial shadow the carried state is compared against, walked forward
     * by exactly the tokens the MTP leg commits.  It is a second model, not a
     * replay of the first: comparing the leg to itself would prove nothing. */
    refmodel *sm = malloc(sizeof(*sm));
    uint64_t *ser_hidden = malloc(REF_CAP * sizeof(*ser_hidden));
    int *ser_next = malloc(REF_CAP * sizeof(*ser_next));
    if (!sm || !ser_hidden || !ser_next) {
        snprintf(run->err, sizeof(run->err), "out of memory");
        run->faulted = 1;
        free(sm); free(ser_hidden); free(ser_next);
        ds4_qwen4exp_mtp_state_free(&st);
        return -1;
    }
    ref_reset(sm, BREAK_NONE, 0);
    int ser_fed = first_token;
    uint32_t ser_pos = 0;

    while (produced < n) {
        int committed[DS4_QWEN4EXP_MTP_MAX_COMMIT];
        const int budget = n - produced;
        const uint64_t verify_before = m.n_verify;
        const uint64_t selects_before = m.selects;
        const uint32_t pos_before = pos;
        /* The round's starting state, kept so the rollback can be checked
         * against a replay that never used a slot. */
        refmodel *pre = malloc(sizeof(*pre));
        if (!pre) {
            snprintf(run->err, sizeof(run->err), "out of memory");
            run->faulted = 1; rc = -1; break;
        }
        *pre = m;
        const int got = ds4_qwen4exp_mtp_cycle(&st, &model, pending,
                                               pos, budget, committed,
                                               DS4_QWEN4EXP_MTP_MAX_COMMIT,
                                               logits, run->err,
                                               sizeof(run->err));
        if (got < 0) {
            if (m.faulted) {
                snprintf(run->err, sizeof(run->err), "reference model: %s",
                         m.fault);
            }
            run->faulted = 1;
            rc = -1;
            free(pre);
            break;
        }
        if (committed[0] != pending) {
            snprintf(run->err, sizeof(run->err),
                     "cycle committed %d first, expected the fed token %d",
                     committed[0], pending);
            run->faulted = 1;
            rc = -1;
            free(pre);
            break;
        }
        /* committed[1..] plus the frontier argmax are the round's output. */
        for (int i = 1; i < got; i++) out[produced++] = committed[i];
        const int fed = pending;
        pending = ds4_qwen4exp_mtp_argmax(logits, REF_VOCAB);
        out[produced++] = pending;

        /*
         * WHERE THE FRONTIER COMES FROM, asserted.
         *
         * A rejecting round is one that selected a state slot.  Its committed
         * tokens are the fed token plus whatever prefix was accepted, and the
         * token the caller feeds next is the target's own choice at the LAST
         * ACCEPTED ROW.  An accepting round takes its frontier from the
         * verify's last row instead.  Both are the row the round committed
         * through -- row `got - 1` of the verify -- and the ONE forward left
         * every row's logits, so the frontier is read rather than recomputed.
         *
         * This is a CHANGE OF MEANING from the replay design and is worth
         * stating rather than quietly following.  There, the frontier had to
         * be the one-row replay's answer and NOT the wide verify's, because
         * the two could differ numerically and the replay's was the one the
         * rolled-back state agreed with.  There is no replay now: the state
         * the round adopts is a slot that same wide forward left, so the wide
         * forward's answer is the only answer, and it is the one the state
         * agrees with by construction.
         *
         * Checking committed[0] alone would prove nothing -- both paths set
         * accepted[0] to the fed token -- which is why the frontier is checked
         * against the specific row the round committed through.
         */
        if (verify_before != m.n_verify) {
            const int rejecting = selects_before != m.selects;
            if (committed[0] != fed) {
                snprintf(run->err, sizeof(run->err),
                         "round committed %d, not the fed token %d",
                         committed[0], fed);
                run->faulted = 1; rc = -1; free(pre); break;
            }
            if ((uint32_t)got > m.row_argmax_n) {
                snprintf(run->err, sizeof(run->err),
                         "round committed %d tokens through a verify of %u "
                         "rows", got, m.row_argmax_n);
                run->faulted = 1; rc = -1; free(pre); break;
            }
            const int want = m.row_argmax[got - 1];
            if (pending != want) {
                snprintf(run->err, sizeof(run->err),
                         "%s round left frontier %d; the row it committed "
                         "through chose %d",
                         rejecting ? "rejecting" : "accepting", pending, want);
                run->faulted = 1; rc = -1; free(pre); break;
            }
            if (rejecting) run->follow_checks++;
        }
        /* The adapter's acceptance length: the tokens after the fed one plus
         * the frontier argmax, which is what the cycle returned. */
        run->acceptance_lengths[run->rounds] = got;
        run->rounds++;
        run->committed_total += got;
        run->accept_hist[got]++;
        pos += (uint32_t)got;

        /*
         * SELECTING THE STATE EQUALS REPLAYING IT.
         *
         * This is the claim the whole design rests on: the state the verify
         * left after row `a` is the state an (a + 1)-row feed leaves, so
         * adopting the slot is exact and the shorter forward the cycle used to
         * run was redundant.  Replay the round's committed tokens from the
         * state it started in, on a model that never touched a slot, and the
         * two must agree object for object.
         *
         * Run on EVERY round, not only rejecting ones: a full accept must not
         * select anything, and this is what would catch it if it did.
         */
        if (!batch_variant) {
            refmodel *replay = malloc(sizeof(*replay));
            if (!replay) {
                snprintf(run->err, sizeof(run->err), "out of memory");
                run->faulted = 1; rc = -1; free(pre); break;
            }
            *replay = *pre;
            int broke = 0;
            for (int i = 0; i < got && !broke; i++) {
                uint64_t h = 0;
                if (ref_step(replay, committed[i], pos_before + (uint32_t)i,
                             1u, &h) != 0) {
                    snprintf(run->err, sizeof(run->err),
                             "replay fault: %s", replay->fault);
                    run->faulted = 1; rc = -1; broke = 1;
                }
            }
            const char *rdiff = broke ? NULL : ref_carried_diff(&m, replay);
            free(replay);
            if (rdiff) {
                snprintf(run->err, sizeof(run->err),
                         "after round %d the selected %s differs from a replay "
                         "of the %d committed token(s)",
                         run->rounds, rdiff, got);
                run->faulted = 1; rc = -1;
            }
            if (!broke && !rdiff) run->replay_checks++;
            if (rc != 0) { free(pre); break; }
        }
        free(pre);
        pre = NULL;

        /*
         * Walk the serial shadow by the same tokens and compare EVERY carried
         * object.  A rejecting round that left one speculative row anywhere
         * shows up here, on the round that left it.
         *
         * Not under the batch-variance control: there the tower deliberately
         * answers differently at two widths, so the leg's stream is EXPECTED
         * to leave the serial leg's and its carried state with it.  Comparing
         * them would assert the opposite of what that control demonstrates.
         */
        if (!batch_variant) {
            float shc[REF_HC_DIM], slog[REF_VOCAB];
            const uint32_t ser_pos0 = ser_pos;
            int broke = 0;
            for (int i = 0; i < got && !broke; i++) {
                if (ref_decode_token(sm, ser_fed, ser_pos, shc, slog) != 0) {
                    snprintf(run->err, sizeof(run->err),
                             "serial shadow fault: %s", sm->fault);
                    run->faulted = 1; rc = -1; broke = 1; break;
                }
                ser_hidden[ser_pos] = ref_hc_to_hidden(shc);
                ser_fed = ds4_qwen4exp_mtp_argmax(slog, REF_VOCAB);
                ser_next[ser_pos] = ser_fed;
                ser_pos++;
            }
            if (broke) break;
            const char *diff = ref_carried_diff(&m, sm);
            if (diff) {
                snprintf(run->err, sizeof(run->err),
                         "after round %d (%u tokens committed) the %s differs "
                         "from the serial leg", run->rounds, ser_pos, diff);
                run->faulted = 1; rc = -1; break;
            }
            /*
             * The head cache over the committed prefix.  Row p folds the token
             * at p + 1 into a hidden state, and there are exactly TWO the head
             * is allowed to have read it from: the TARGET's row at p, which is
             * what a seed passes, and the head's OWN row at p - 1, which is
             * what a chain step passes and what a round that accepted the
             * draft above it keeps.  Anything else -- a row for a different
             * token, a chain row built on a parent the round rejected -- fails
             * here even when the tower state is clean, and row 0 has only the
             * target to come from, so the chain rows are anchored rather than
             * self-certifying.
             *
             * Skipped at depth 0, which never runs the head.
             */
            if (depth < 1) {
                /* nothing to compare */
            } else if (m.head_len < ser_pos) {
                snprintf(run->err, sizeof(run->err),
                         "the head cache covers %u rows at frontier %u",
                         m.head_len, ser_pos);
                run->faulted = 1; rc = -1; break;
            }
            for (uint32_t q = 0; depth >= 1 && q < ser_pos; q++) {
                const uint64_t tok = (uint64_t)(uint32_t)ser_next[q];
                const uint64_t from_target = mix64(ser_hidden[q] ^ tok);
                const uint64_t from_chain =
                    q > 0u ? mix64(m.head[q - 1u] ^ tok) : from_target;
                const int target = m.head[q] == from_target;
                const int chain = m.head[q] == from_chain;
                if (!m.head_stamp[q] || (!target && !chain)) {
                    snprintf(run->err, sizeof(run->err),
                             "head cache row %u of %u came from neither the "
                             "target's row nor the head's own", q, ser_pos);
                    run->faulted = 1; rc = -1; break;
                }
                if (q >= ser_pos0) {
                    if (target) run->head_rows_from_target++;
                    else        run->head_rows_from_chain++;
                }
            }
            if (rc != 0) break;
        }
    }
    run->drafted = st.counters.drafted;
    run->accepted = st.counters.accepted;
    run->quenches = st.counters.quenches;
    run->disagreements = st.counters.verify_replay_disagreements;
    if (rc == 0 &&
        ds4_qwen4exp_mtp_counters_check(&st.counters, run->err,
                                        sizeof(run->err)) != 0) {
        run->faulted = 1;
        rc = -1;
    }
    ds4_qwen4exp_mtp_state_free(&st);
    free(sm); free(ser_hidden); free(ser_next);
    return rc;
}

/* ========================================================================
 * Tests
 * ======================================================================== */

#define N_TOKENS 256
static const int g_prompts[] = { 1, 7, 13, 29, 41, 58 };
#define N_PROMPTS ((int)(sizeof(g_prompts) / sizeof(g_prompts[0])))

/*
 * THE CONTRACT.  At every implemented depth the emitted stream must be
 * byte-identical to the depth-0 serial stream and the verify/replay
 * disagreement counter must be 0.
 *
 * Byte-identity is not a nice-to-have here: the scored benchmark reads the
 * depth-N stream against a golden authored from the serial path, so one
 * differing token is a scoring failure rather than a quality question.  It is
 * reachable because the tower is row-invariant at every width the cycle uses
 * -- widths 2 to N + 1 for a verify, 2 to N for a replay -- which
 * tests/test_qwen4exp_graph pins separately.
 *
 * run_mtp() additionally compares every carried object against a serial shadow
 * after each round, so a round that leaves a speculative row behind is named
 * on the round that left it rather than on the token that eventually moved.
 */
static void test_exactness(void) {
    printf("exactness: greedy MTP equals greedy serial over %d tokens, "
           "depths 1 to %d\n", N_TOKENS, DS4_QWEN4EXP_IMPLEMENTED_DEPTH);
    for (int depth = 1; depth <= DS4_QWEN4EXP_IMPLEMENTED_DEPTH; depth++) {
        for (int p = 0; p < N_PROMPTS; p++) {
            int serial[N_TOKENS], mtp[N_TOKENS];
            mtp_run run;
            CHECK(run_serial(g_prompts[p], N_TOKENS, serial) == 0,
                  "serial leg failed for prompt %d", g_prompts[p]);
            if (run_mtp(g_prompts[p], N_TOKENS, mtp, &run, BREAK_NONE, 0,
                        depth) != 0) {
                CHECK(0, "depth %d prompt %d: %s", depth, g_prompts[p],
                      run.err);
                continue;
            }
            int first_diff = -1;
            for (int i = 0; i < N_TOKENS; i++) {
                if (serial[i] != mtp[i]) { first_diff = i; break; }
            }
            CHECK(first_diff < 0,
                  "depth %d prompt %d diverges at token %d: serial %d, mtp %d",
                  depth, g_prompts[p], first_diff,
                  first_diff < 0 ? -1 : serial[first_diff],
                  first_diff < 0 ? -1 : mtp[first_diff]);
            CHECK(run.disagreements == 0,
                  "depth %d prompt %d: %llu verify/replay disagreements",
                  depth, g_prompts[p],
                  (unsigned long long)run.disagreements);

            /* Counter consistency, as the adapter computes it. */
            int sum = 0, deepest = 0;
            for (int i = 0; i < run.rounds; i++) {
                CHECK(run.acceptance_lengths[i] >= 1 &&
                      run.acceptance_lengths[i] <= depth + 1,
                      "depth %d prompt %d round %d has acceptance length %d",
                      depth, g_prompts[p], i, run.acceptance_lengths[i]);
                if (run.acceptance_lengths[i] > deepest) {
                    deepest = run.acceptance_lengths[i];
                }
                sum += run.acceptance_lengths[i];
            }
            CHECK(sum == run.committed_total,
                  "depth %d prompt %d: acceptance lengths sum to %d, %d "
                  "committed", depth, g_prompts[p], sum, run.committed_total);
            /* NON-VACUITY.  A depth-N run that never once committed N + 1
             * tokens exercised the deep accept path zero times, and the
             * identity above would be the depth-1 identity in disguise. */
            CHECK(deepest == depth + 1,
                  "depth %d prompt %d never committed %d tokens in a round "
                  "(deepest %d): the deep accept path did not run",
                  depth, g_prompts[p], depth + 1, deepest);
            CHECK(run.drafted <= (uint64_t)run.rounds * (uint64_t)depth,
                  "depth %d prompt %d: %llu drafts over %d rounds",
                  depth, g_prompts[p], (unsigned long long)run.drafted,
                  run.rounds);
            CHECK(run.accepted <= run.drafted,
                  "depth %d prompt %d: %llu accepted of %llu drafted",
                  depth, g_prompts[p], (unsigned long long)run.accepted,
                  (unsigned long long)run.drafted);
            CHECK(run.quenches == 0, "depth %d prompt %d: %llu quench events",
                  depth, g_prompts[p], (unsigned long long)run.quenches);
            /* Every round's rollback was checked against a replay, and the
             * rounds spanned every acceptance length this depth can produce --
             * so the select path was exercised at each `a`, not only at the
             * one the oracle happens to favour. */
            CHECK(run.replay_checks == (uint64_t)run.rounds,
                  "depth %d prompt %d: %llu of %d rounds checked against a "
                  "replay", depth, g_prompts[p],
                  (unsigned long long)run.replay_checks, run.rounds);
            /* THE HEAD KEEPS ITS OWN ROWS.  A round that accepted a draft
             * keeps the chain row that produced it instead of re-feeding the
             * token through the head.  From depth 2 up that is most of the
             * committed prefix, and zero would be the old re-seeding cycle.
             * At depth 1 there is nothing to show: the chain is one step, that
             * step reads the TARGET's row, and every row is target-sourced
             * either way. */
            CHECK(depth < 2 || run.head_rows_from_chain > 0,
                  "depth %d prompt %d: every head cache row was re-seeded from "
                  "the target", depth, g_prompts[p]);
            for (int k = 1; k <= depth + 1; k++) {
                CHECK(run.accept_hist[k] > 0,
                      "depth %d prompt %d never committed %d token(s), so the "
                      "select at that acceptance length never ran",
                      depth, g_prompts[p], k);
            }
            printf("  depth %d prompt %2d: head rows %llu target / %llu "
                   "chain\n", depth, g_prompts[p],
                   (unsigned long long)run.head_rows_from_target,
                   (unsigned long long)run.head_rows_from_chain);
            printf("  depth %d prompt %2d: %3d rounds, %3llu drafted, "
                   "%3llu accepted, %d committed, %.2f tok/round\n",
                   depth, g_prompts[p], run.rounds,
                   (unsigned long long)run.drafted,
                   (unsigned long long)run.accepted, run.committed_total,
                   (double)run.committed_total / (double)run.rounds);
        }
    }
}

/* Every rollback object must be load bearing: break one and the property must
 * stop holding, either by diverging or by the model refusing outright. */
static void test_rollback_negative_controls(void) {
    printf("negative controls: each rollback object is load bearing\n");
    int serial[N_TOKENS];
    CHECK(run_serial(g_prompts[0], N_TOKENS, serial) == 0, "serial leg failed");

    /*
     * BREAK_HEAD_CACHE -- the no-op truncate -- is not on this list at depth
     * 1 on purpose: there the chain never leaves the frontier, so the truncate
     * has nothing to remove and disabling it changes nothing.  It IS on the
     * list from depth 2 up, where the chain writes speculative rows and a
     * truncate that does not run leaves them for the next round's seed.
     *
     * The one-row mutants are the deliverable's mutation proof: a rollback
     * that lands one row past the accepted length rather than doing nothing at
     * all.  Both must be caught, and caught BY NAME rather than by an
     * eventual divergence.
     */
    static const ref_break bearing[] = {
        BREAK_GDN_RECURRENT, BREAK_GDN_CONV, BREAK_QSA_KV,
        BREAK_INDEXER_POOL, BREAK_PLE,
        BREAK_QSA_KV_ONE_ROW, BREAK_HEAD_CACHE_ONE_ROW,
        BREAK_SELECT_ONE_ROW, BREAK_VERIFY_ROW_LOGITS,
    };
    for (int depth = 1; depth <= DS4_QWEN4EXP_IMPLEMENTED_DEPTH; depth++) {
        for (size_t k = 0; k < sizeof(bearing) / sizeof(bearing[0]); k++) {
            const ref_break b = bearing[k];
            /* At depth 1 the head cache never goes above the frontier, so
             * neither head mutant has a row to get wrong. */
            if (depth == 1 && b == BREAK_HEAD_CACHE_ONE_ROW) continue;
            /* At depth 1 there is exactly one slot, so "one row too far" has
             * nowhere to go. */
            if (depth == 1 && b == BREAK_SELECT_ONE_ROW) continue;
            int mtp[N_TOKENS];
            mtp_run run;
            memset(mtp, 0, sizeof(mtp));
            const int rc = run_mtp(g_prompts[0], N_TOKENS, mtp, &run, b, 0,
                                   depth);
            int diverged = 0;
            for (int i = 0; i < N_TOKENS && !diverged; i++) {
                if (serial[i] != mtp[i]) diverged = 1;
            }
            CHECK(rc != 0 || diverged,
                  "depth %d with %s the MTP stream still matched serial",
                  depth, ref_break_name(b));
            printf("  depth %d  %-46s -> %s\n", depth, ref_break_name(b),
                   rc != 0 ? "refused" : (diverged ? "diverged" : "NO EFFECT"));
            if (rc != 0) printf("            %s\n", run.err);
        }
        if (depth >= 2) {
            int mtp[N_TOKENS];
            mtp_run run;
            memset(mtp, 0, sizeof(mtp));
            const int rc = run_mtp(g_prompts[0], N_TOKENS, mtp, &run,
                                   BREAK_HEAD_CACHE, 0, depth);
            CHECK(rc != 0,
                  "depth %d with %s the run still completed",
                  depth, ref_break_name(BREAK_HEAD_CACHE));
            printf("  depth %d  %-46s -> %s\n", depth,
                   ref_break_name(BREAK_HEAD_CACHE),
                   rc != 0 ? "refused" : "NO EFFECT");
            if (rc != 0) printf("            %s\n", run.err);
        }
    }
}

/*
 * The head cache's own control.
 *
 * Every round must leave the head cache covering exactly the committed prefix.
 * The design's warning is about the OTHER rollback: a session rewind moves the
 * frontier backwards, and a head cache that is restored and then bumped rather
 * than truncated in one step re-exposes rows it no longer covers.  So this
 * drives a rewind, and with the head truncate disabled the reference model's
 * cursor check must catch it.
 */
static void test_head_cache_boundary(void) {
    printf("head cache: it covers exactly the committed prefix\n");
    for (int broken = 0; broken <= 1; broken++) {
        refmodel m;
        ds4_qwen4exp_mtp_model model;
        ds4_qwen4exp_rollback_set set;
        ds4_qwen4exp_mtp_state st;
        ref_reset(&m, broken ? BREAK_HEAD_CACHE : BREAK_NONE, 0);
        CHECK(ref_build(&m, &model, &set) == 0, "reference build failed");
        CHECK(ds4_qwen4exp_mtp_state_init(&st, 1, &set, REF_HC_DIM, REF_VOCAB,
                                          g_err, sizeof(g_err)) == 0,
              "state init failed: %s", g_err);
        float logits[REF_VOCAB];
        int committed[DS4_QWEN4EXP_MTP_MAX_COMMIT];
        int pending = g_prompts[2];
        uint32_t pos = 0;
        int holes = 0;
        while (pos < 64u) {
            const int got = ds4_qwen4exp_mtp_cycle(&st, &model, pending,
                                                   pos, 2, committed, 2,
                                                   logits, g_err, sizeof(g_err));
            if (got < 0) { CHECK(0, "cycle failed: %s", g_err); break; }
            pos += (uint32_t)got;
            pending = ds4_qwen4exp_mtp_argmax(logits, REF_VOCAB);
            if (m.head_len != pos) holes++;
        }
        CHECK(holes == 0,
              "the head cache left the committed prefix %d time(s)", holes);

        /*
         * The rewind: put the frontier back and drive the cycle again.
         *
         * Only the TRUNCATE objects move.  There is no restore() any more, and
         * there never really was one that took a position: the old verb
         * returned the running state to the last round's snapshot and ignored
         * its `pos` argument entirely, so an arbitrary rewind was never
         * something this contract could serve.  A session that genuinely
         * rewinds re-prefills, which is the only thing that puts running state
         * at an arbitrary length.  What this drives is the case the head cache
         * has to survive: the frontier moved and the caches were cut, so the
         * cache must not still be seeded past it.
         */
        const uint32_t back = pos / 2u;
        ds4_qwen4exp_mtp_invalidate(&st);
        for (int i = 0; i < DS4_QWEN4EXP_STATE_COUNT; i++) {
            const ds4_qwen4exp_rollback_object *o = &set.obj[i];
            if (o->truncate) (void)o->truncate(o->ctx, back);
        }
        const int got = ds4_qwen4exp_mtp_cycle(&st, &model, pending,
                                               back, 2, committed, 2, logits,
                                               g_err, sizeof(g_err));
        if (broken) {
            CHECK(got < 0, "a stale head cache survived a rewind to %u", back);
            printf("  with %s -> refused\n", ref_break_name(BREAK_HEAD_CACHE));
        } else {
            CHECK(got == 1, "the rewound cycle failed: %s", g_err);
            printf("  clean rewind to %u -> ok\n", back);
        }
        ds4_qwen4exp_mtp_state_free(&st);
    }
}

/*
 * ROW INVARIANCE IS LOAD BEARING, as a negative control.
 *
 * The exactness contract rests on one property of the tower: row t of an
 * n-row verify equals a one-row decode of the same position.  Everything else
 * follows from it -- the accepted drafts are what serial would have chosen,
 * the state slot the forward leaves after row a is the state an (a + 1)-row
 * feed leaves, and the frontier the round hands back is the row it committed
 * through.
 *
 * So a tower that answers differently at two widths must produce a leg that
 * DIVERGES from serial.  That is what this drives, and it is the honest
 * replacement for what this test used to assert.
 *
 * It used to say "counted, not refused": the cycle compared the wide verify's
 * argmax against a one-row replay's on every rejecting round and incremented
 * verify_replay_disagreements where they differed.  That check is gone with
 * the replay it depended on.  Removing it removed the only RUNTIME detector
 * for a row-invariance violation, and that is a real trade, not a free one:
 * on this tower the property is pinned at exactly zero by
 * tests/test_qwen4exp_graph across every width the cycle uses, and a backend
 * that broke it would now be caught there and in the exactness legs above
 * rather than by a rising counter in production.
 */
static void test_row_invariance_is_load_bearing(void) {
    printf("row invariance: a batch-variant tower must diverge from serial\n");
    int mtp[N_TOKENS], serial[N_TOKENS];
    mtp_run run;
    CHECK(run_serial(g_prompts[0], N_TOKENS, serial) == 0, "serial leg failed");
    for (int depth = 1; depth <= DS4_QWEN4EXP_IMPLEMENTED_DEPTH; depth++) {
        memset(mtp, 0, sizeof(mtp));
        const int rc = run_mtp(g_prompts[0], N_TOKENS, mtp, &run, BREAK_NONE, 1,
                               depth);
        /* The leg still RUNS.  A numeric residual is not a fault the cycle can
         * see, and it must not turn into a crash or an early stop; what it
         * turns into is a wrong answer, which is the point. */
        CHECK(rc == 0, "depth %d: a batch-variant leg failed outright: %s",
              depth, run.err);
        if (rc != 0) continue;
        CHECK(run.committed_total == N_TOKENS,
              "depth %d: the leg stopped early: %d of %d tokens",
              depth, run.committed_total, N_TOKENS);
        int diverged = 0, at = -1;
        for (int i = 0; i < N_TOKENS && !diverged; i++) {
            if (serial[i] != mtp[i]) { diverged = 1; at = i; }
        }
        CHECK(diverged,
              "depth %d: a batch-variant tower still matched serial, so the "
              "exactness result above does not depend on row invariance",
              depth);
        printf("  depth %d: %d rounds, diverges from serial at token %d\n",
               depth, run.rounds, at);
    }
}

static void test_depth_zero_is_serial(void) {
    printf("depth 0: the same cycle runs a serial leg\n");
    int serial[N_TOKENS], mtp[N_TOKENS];
    mtp_run run;
    CHECK(run_serial(g_prompts[1], N_TOKENS, serial) == 0, "serial leg failed");
    CHECK(run_mtp(g_prompts[1], N_TOKENS, mtp, &run, BREAK_NONE, 0, 0) == 0,
          "depth 0 leg failed: %s", run.err);
    CHECK(memcmp(serial, mtp, sizeof(serial)) == 0, "depth 0 diverged");
    CHECK(run.drafted == 0, "depth 0 drafted %llu times",
          (unsigned long long)run.drafted);
    CHECK(run.rounds == N_TOKENS, "depth 0 ran %d rounds for %d tokens",
          run.rounds, N_TOKENS);
}

static void test_depth_envelope(void) {
    printf("depth envelope\n");
    /* Pinned literals, not the constants themselves: the envelope is a ruling
     * -- 6 since the depth-6 merge -- and a build that moves it has to move
     * this line too.  MAX_COMMIT is the widest verify that follows from it. */
    CHECK(DS4_QWEN4EXP_IMPLEMENTED_DEPTH == 6,
          "DS4_QWEN4EXP_IMPLEMENTED_DEPTH is %d", DS4_QWEN4EXP_IMPLEMENTED_DEPTH);
    CHECK(DS4_QWEN4EXP_MTP_MAX_COMMIT == 7,
          "DS4_QWEN4EXP_MTP_MAX_COMMIT is %d", DS4_QWEN4EXP_MTP_MAX_COMMIT);

    CHECK(ds4_qwen4exp_mtp_depth_from_draft_tokens(1, g_err, sizeof(g_err)) == 0,
          "DS4_MTP_DRAFT_TOKENS=1 was not read as a serial leg");
    for (int dt = 2; dt <= DS4_QWEN4EXP_IMPLEMENTED_DEPTH + 1; dt++) {
        CHECK(ds4_qwen4exp_mtp_depth_from_draft_tokens(dt, g_err,
                                                       sizeof(g_err)) == dt - 1,
              "DS4_MTP_DRAFT_TOKENS=%d was not read as depth %d", dt, dt - 1);
    }

    /* The envelope moved from "2 and up is refused" to "8 and up is refused";
     * the refusal still has to name the depth asked for and the envelope
     * constant, so a future move cannot quietly become a silent clamp. */
    for (int dt = DS4_QWEN4EXP_IMPLEMENTED_DEPTH + 2;
         dt <= DS4_QWEN4EXP_IMPLEMENTED_DEPTH + 3; dt++) {
        g_err[0] = '\0';
        CHECK(ds4_qwen4exp_mtp_depth_from_draft_tokens(dt, g_err,
                                                       sizeof(g_err)) < 0,
              "DS4_MTP_DRAFT_TOKENS=%d was accepted", dt);
        char want[64];
        snprintf(want, sizeof(want), "depth %d", dt - 1);
        CHECK(strstr(g_err, want) != NULL,
              "the refusal for DS4_MTP_DRAFT_TOKENS=%d does not name %s: %s",
              dt, want, g_err);
        CHECK(strstr(g_err, "DS4_QWEN4EXP_IMPLEMENTED_DEPTH") != NULL,
              "the refusal for DS4_MTP_DRAFT_TOKENS=%d does not name the "
              "envelope: %s", dt, g_err);
        printf("  draft_tokens %d -> %s\n", dt, g_err);
    }
    CHECK(ds4_qwen4exp_mtp_depth_from_draft_tokens(0, g_err, sizeof(g_err)) < 0,
          "DS4_MTP_DRAFT_TOKENS=0 was accepted");

    refmodel m;
    ds4_qwen4exp_mtp_model model;
    ds4_qwen4exp_rollback_set set;
    ds4_qwen4exp_mtp_state st;
    ref_reset(&m, BREAK_NONE, 0);
    CHECK(ref_build(&m, &model, &set) == 0, "reference build failed");
    for (int d = 0; d <= DS4_QWEN4EXP_IMPLEMENTED_DEPTH; d++) {
        CHECK(ds4_qwen4exp_mtp_state_init(&st, d, &set, REF_HC_DIM, REF_VOCAB,
                                          g_err, sizeof(g_err)) == 0,
              "a depth-%d state was refused: %s", d, g_err);
        ds4_qwen4exp_mtp_state_free(&st);
    }
    g_err[0] = '\0';
    CHECK(ds4_qwen4exp_mtp_state_init(&st, DS4_QWEN4EXP_IMPLEMENTED_DEPTH + 1,
                                      &set, REF_HC_DIM, REF_VOCAB,
                                      g_err, sizeof(g_err)) < 0,
          "a depth-%d state was built", DS4_QWEN4EXP_IMPLEMENTED_DEPTH + 1);
    CHECK(strstr(g_err, "DS4_QWEN4EXP_IMPLEMENTED_DEPTH") != NULL,
          "the state refusal does not name the envelope: %s", g_err);
    printf("  state depth %d -> %s\n", DS4_QWEN4EXP_IMPLEMENTED_DEPTH + 1,
           g_err);
}

static void test_no_yield_guard(void) {
    printf("quench: no yield guard on the scored path\n");
    unsetenv("DS4_QWEN_MTP_QUENCH");
    CHECK(ds4_qwen4exp_mtp_check_no_yield_guard(g_err, sizeof(g_err)) == 0,
          "an unset DS4_QWEN_MTP_QUENCH was refused: %s", g_err);
    setenv("DS4_QWEN_MTP_QUENCH", "0", 1);
    CHECK(ds4_qwen4exp_mtp_check_no_yield_guard(g_err, sizeof(g_err)) == 0,
          "DS4_QWEN_MTP_QUENCH=0 was refused: %s", g_err);
    setenv("DS4_QWEN_MTP_QUENCH", "1", 1);
    CHECK(ds4_qwen4exp_mtp_check_no_yield_guard(g_err, sizeof(g_err)) < 0,
          "DS4_QWEN_MTP_QUENCH=1 was accepted");
    printf("  refused: %s\n", g_err);
    unsetenv("DS4_QWEN_MTP_QUENCH");
}

static void test_rollback_contract(void) {
    printf("rollback contract\n");
    ds4_qwen4exp_rollback_set set;
    ds4_qwen4exp_rollback_init(&set);
    CHECK(ds4_qwen4exp_rollback_check(&set, g_err, sizeof(g_err)) < 0,
          "an empty rollback set passed the check");
    CHECK(strstr(g_err, "gdn recurrent state") != NULL,
          "the refusal does not name the missing object: %s", g_err);

    refmodel m;
    ref_reset(&m, BREAK_NONE, 0);

    /* A select-row object that supplies truncate() is refused. */
    ds4_qwen4exp_rollback_object bad = { &m, rb_recurrent_select,
                                         rb_kv_truncate };
    CHECK(ds4_qwen4exp_rollback_register(&set, DS4_QWEN4EXP_STATE_GDN_RECURRENT,
                                         &bad, g_err, sizeof(g_err)) < 0,
          "a select-row object with truncate() was registered");
    /* A truncate-class object that supplies select_row() is refused. */
    ds4_qwen4exp_rollback_object bad2 = { &m, rb_recurrent_select, NULL };
    CHECK(ds4_qwen4exp_rollback_register(&set, DS4_QWEN4EXP_STATE_QSA_KV,
                                         &bad2, g_err, sizeof(g_err)) < 0,
          "a truncate-class object without truncate() was registered");
    CHECK(ds4_qwen4exp_state_kind(DS4_QWEN4EXP_STATE_GDN_RECURRENT) ==
          DS4_QWEN4EXP_ROLLBACK_SELECT_ROW,
          "the recurrent state must roll back by selecting a row");

    CHECK(ds4_qwen4exp_state_kind(DS4_QWEN4EXP_STATE_MTP_HEAD_CACHE) ==
          DS4_QWEN4EXP_ROLLBACK_TRUNCATE,
          "the head cache must roll back to the boundary in one step");

    /*
     * An incomplete set is caught at OPEN.  Only a rejecting round uses the
     * rollback, so a cycle-time check would pass the first round or two and
     * refuse in the middle of a leg; the state holds the set instead, and
     * init() is where the hole is named.
     */
    ds4_qwen4exp_mtp_model model;
    ds4_qwen4exp_rollback_set full;
    ds4_qwen4exp_mtp_state st;
    ref_reset(&m, BREAK_NONE, 0);
    CHECK(ref_build(&m, &model, &full) == 0, "reference build failed");

    ds4_qwen4exp_rollback_set holed = full;
    holed.registered[DS4_QWEN4EXP_STATE_PLE_HISTORY] = false;
    CHECK(ds4_qwen4exp_mtp_state_init(&st, 1, &holed, REF_HC_DIM, REF_VOCAB,
                                      g_err, sizeof(g_err)) < 0,
          "a state was built with the ple history unregistered");
    CHECK(strstr(g_err, "ple n-gram history") != NULL,
          "the refusal does not name the missing object: %s", g_err);

    /* The convolution state is a second, separate id.  A set that registers
     * the n-gram history and forgets the conv state must still be refused --
     * that is the hole one combined id would have hidden. */
    ds4_qwen4exp_rollback_set no_conv = full;
    no_conv.registered[DS4_QWEN4EXP_STATE_PLE_CONV] = false;
    CHECK(ds4_qwen4exp_mtp_state_init(&st, 1, &no_conv, REF_HC_DIM, REF_VOCAB,
                                      g_err, sizeof(g_err)) < 0,
          "a state was built with the ple conv state unregistered");
    CHECK(strstr(g_err, "ple conv state") != NULL,
          "the refusal does not name the missing conv state: %s", g_err);
    CHECK(ds4_qwen4exp_mtp_state_init(&st, 1, NULL, REF_HC_DIM, REF_VOCAB,
                                      g_err, sizeof(g_err)) < 0,
          "a state was built with no rollback set at all");

    CHECK(ds4_qwen4exp_mtp_state_init(&st, 1, &full, REF_HC_DIM, REF_VOCAB,
                                      g_err, sizeof(g_err)) == 0,
          "state init failed on a complete set: %s", g_err);
    float logits[REF_VOCAB];
    int committed[DS4_QWEN4EXP_MTP_MAX_COMMIT];
    /* Round 1 has no draft yet, so it takes the plain path and seeds one. */
    CHECK(ds4_qwen4exp_mtp_cycle(&st, &model, 5, 0, 2, committed,
                                 2, logits, g_err, sizeof(g_err)) == 1,
          "the first cycle did not commit exactly one token: %s", g_err);
    ds4_qwen4exp_mtp_state_free(&st);
}

/* A budget of one must never commit two, or the adapter overruns its count. */
static void test_budget(void) {
    printf("budget\n");
    refmodel m;
    ds4_qwen4exp_mtp_model model;
    ds4_qwen4exp_rollback_set set;
    ds4_qwen4exp_mtp_state st;
    ref_reset(&m, BREAK_NONE, 0);
    CHECK(ref_build(&m, &model, &set) == 0, "reference build failed");
    CHECK(ds4_qwen4exp_mtp_state_init(&st, 1, &set, REF_HC_DIM, REF_VOCAB,
                                      g_err, sizeof(g_err)) == 0,
          "state init failed: %s", g_err);
    float logits[REF_VOCAB];
    int committed[DS4_QWEN4EXP_MTP_MAX_COMMIT];
    int pending = 3;
    uint32_t pos = 0;
    for (int i = 0; i < 64; i++) {
        const int got = ds4_qwen4exp_mtp_cycle(&st, &model, pending,
                                               pos, 1, committed, 2, logits,
                                               g_err, sizeof(g_err));
        CHECK(got == 1, "a budget of 1 committed %d tokens: %s", got, g_err);
        if (got != 1) break;
        pos += 1;
        pending = ds4_qwen4exp_mtp_argmax(logits, REF_VOCAB);
    }
    CHECK(st.counters.drafted == 0, "a budget of 1 drafted %llu times",
          (unsigned long long)st.counters.drafted);
    ds4_qwen4exp_mtp_state_free(&st);
}

/* A greedy compact caller needs only the reducer's exact token.  Verify that
 * opting into lazy materialization leaves the selected distribution on the
 * model seam while publishing enough metadata to fetch it later. */
static void test_deferred_frontier_logits(void) {
    printf("deferred compact frontier logits\n");
    refmodel m;
    ds4_qwen4exp_mtp_model model;
    ds4_qwen4exp_rollback_set set;
    ds4_qwen4exp_mtp_state st;
    ref_reset(&m, BREAK_NONE, 0);
    CHECK(ref_build(&m, &model, &set) == 0, "reference build failed");
    model.defer_frontier_logits = true;
    CHECK(ds4_qwen4exp_mtp_state_init(&st, 1, &set, REF_HC_DIM, REF_VOCAB,
                                      g_err, sizeof(g_err)) == 0,
          "state init failed: %s", g_err);

    float logits[REF_VOCAB];
    int committed[DS4_QWEN4EXP_MTP_MAX_COMMIT];
    CHECK(ds4_qwen4exp_mtp_cycle(&st, &model, 3, 0, 2, committed, 2,
                                  logits, g_err, sizeof(g_err)) == 1,
          "seed cycle failed: %s", g_err);
    const int next = ds4_qwen4exp_mtp_argmax(logits, REF_VOCAB);
    for (uint32_t i = 0; i < REF_VOCAB; i++) logits[i] = -1234.0f;
    const int got = ds4_qwen4exp_mtp_cycle(
        &st, &model, next, 1, 2, committed, 2, logits, g_err, sizeof(g_err));
    CHECK(got > 0, "compact cycle failed: %s", g_err);
    CHECK(m.n_read_logit == 0,
          "compact cycle eagerly read %llu frontier rows",
          (unsigned long long)m.n_read_logit);
    CHECK(st.frontier_top1_valid && st.frontier_logits_deferred,
          "compact cycle did not publish a deferred frontier");
    CHECK(st.frontier_top1 == ds4_qwen4exp_mtp_argmax(
              m.compact_logits[st.frontier_row], REF_VOCAB),
          "deferred top-1 does not match selected target row");
    CHECK(logits[0] == -1234.0f,
          "deferred cycle unexpectedly overwrote the host distribution");
    CHECK(model.read_logit_row(model.ctx, st.frontier_row, logits) == 0,
          "lazy frontier materialization failed");
    CHECK(ds4_qwen4exp_mtp_argmax(logits, REF_VOCAB) == st.frontier_top1,
          "materialized frontier does not match cached top-1");
    ds4_qwen4exp_mtp_state_free(&st);
}

/* ========================================================================
 * The head wiring
 * ========================================================================
 *
 * The head composes five calls that other lanes own, so what is checked here
 * is the COMPOSITION: which primitive runs in which order, over which mapping,
 * at which offset and width, and that the eh_proj step really does broadcast
 * the embedding half across the hyper-connection streams and add the hidden
 * half per stream.
 *
 * The QSA/MoE block and the norms are PASS-THROUGHS under this test, the way
 * L7's integration test stands them in until L4 is merged: the norms copy, the
 * block is the identity, and the mixer sums the streams.  The two matmuls are
 * real f32 matmuls against host weight tables, because the algebra above is
 * exactly what they have to prove.  The GPU tensor API is host memory here --
 * ds4_gpu_tensor is opaque, so a test may define it -- which is what lets this
 * run on a machine with no GPU at all.
 *
 * COMPARING AGAINST L2's MLX DUMPS, when they land
 * ------------------------------------------------
 * The dump emits the projection as TWO ops, `mtp_fc_embedding` and
 * `mtp_fc_hidden`, each 2560 -> 2560: the first on the normalized embedding,
 * the second on the normalized hidden, summed afterwards.  This engine fuses
 * them into one eh_proj matmul over 5120 inputs, so split its input rows to
 * line the two up:
 *
 *     rows [0, 2560)     -> mtp_fc_embedding
 *     rows [2560, 5120)  -> mtp_fc_hidden
 *
 * (That is the order the shipped head carries; see
 * DS4_QWEN4EXP_EH_PROJ_EMBED_FIRST in ds4_qwen4exp_mtp.h.)  Check each half
 * against its own op, then check the sum against the input `mtp_step` records
 * -- which is this head's `hyper`, before the block runs, and is what
 * `multi_out` returns here.
 *
 * Bands come from the OP CLASS, not from the layer: both halves and their sum
 * are dense GEMMs at real quantization, so the criterion is relative Frobenius
 * error, not the tighter elementwise band the norms and gates are held to.
 * Read the band off the op-class table rather than reusing a neighbouring
 * layer's number.
 *
 * Nothing here reads a dump yet; this is the mapping so that comparison is
 * mechanical when it exists.
 */

struct ds4_gpu_tensor {
    uint64_t bytes;
    unsigned char *data;
    int is_view;
};

ds4_gpu_tensor *ds4_gpu_tensor_alloc(uint64_t bytes) {
    ds4_gpu_tensor *t = calloc(1, sizeof(*t));
    if (!t) return NULL;
    t->data = calloc(1, bytes ? (size_t)bytes : 1u);
    if (!t->data) { free(t); return NULL; }
    t->bytes = bytes;
    return t;
}
void ds4_gpu_tensor_free(ds4_gpu_tensor *t) {
    if (!t) return;
    if (!t->is_view) free(t->data);
    free(t);
}
int ds4_gpu_tensor_write(ds4_gpu_tensor *t, uint64_t off, const void *src,
                         uint64_t bytes) {
    if (!t || off + bytes > t->bytes) return 0;
    memcpy(t->data + off, src, (size_t)bytes);
    return 1;
}
int ds4_gpu_tensor_read(const ds4_gpu_tensor *t, uint64_t off, void *dst,
                        uint64_t bytes) {
    if (!t || off + bytes > t->bytes) return 0;
    memcpy(dst, t->data + off, (size_t)bytes);
    return 1;
}
int ds4_gpu_tensor_copy(ds4_gpu_tensor *dst, uint64_t dst_off,
                        const ds4_gpu_tensor *src, uint64_t src_off,
                        uint64_t bytes) {
    if (!dst || !src) return 0;
    if (dst_off + bytes > dst->bytes || src_off + bytes > src->bytes) return 0;
    memmove(dst->data + dst_off, src->data + src_off, (size_t)bytes);
    return 1;
}
int ds4_gpu_indexer_topk_tensor(ds4_gpu_tensor *selected,
                                const ds4_gpu_tensor *scores,
                                uint32_t n_comp, uint32_t n_tokens,
                                uint32_t top_k) {
    if (!selected || !scores || top_k != 1u ||
        scores->bytes < (uint64_t)n_comp * n_tokens * sizeof(float) ||
        selected->bytes < (uint64_t)n_tokens * sizeof(uint32_t)) {
        return 0;
    }
    uint32_t *out = (uint32_t *)selected->data;
    const float *in = (const float *)scores->data;
    for (uint32_t t = 0; t < n_tokens; t++) {
        const float *row = in + (size_t)t * n_comp;
        float best_v = -INFINITY;
        uint32_t best_i = 0u;
        for (uint32_t i = 0; i < n_comp; i++) {
            const float v = row[i];
            if (v > best_v || (v == best_v && i < best_i)) {
                best_v = v;
                best_i = i;
            }
        }
        out[t] = best_i;
    }
    return 1;
}
int ds4_gpu_begin_commands(void) { return 1; }
int ds4_gpu_end_commands(void) { return 1; }
int ds4_gpu_synchronize(void) { return 1; }

/* ---- the recorded call log --------------------------------------------- */

#define HEAD_N_EMBD    4u
#define HEAD_N_HC      4u
#define HEAD_HC_DIM    (HEAD_N_EMBD * HEAD_N_HC)
#define HEAD_N_LOWRANK 2u
#define HEAD_N_VOCAB   8u
#define HEAD_BLOCK_IL  48u
#define HEAD_ROWS      2u

/* Offsets into the two fake mappings, in bytes. */
#define OFF_ENORM        0u
#define OFF_HNORM        (OFF_ENORM + HEAD_N_EMBD * 4u)
#define OFF_EH_PROJ      (OFF_HNORM + HEAD_HC_DIM * 4u)
#define OFF_HC_NORM      (OFF_EH_PROJ + 2u * HEAD_N_EMBD * HEAD_N_EMBD * 4u)
#define OFF_HC_DOWN      (OFF_HC_NORM + HEAD_HC_DIM * 4u)
#define OFF_HC_UP        (OFF_HC_DOWN + HEAD_HC_DIM * HEAD_N_LOWRANK * 4u)
#define HEAD_MAP_FLOATS  ((OFF_HC_UP + HEAD_N_LOWRANK * HEAD_HC_DIM * 4u) / 4u)

#define OFF_TOKEN_EMBD   0u
#define OFF_OUTPUT       (HEAD_N_VOCAB * HEAD_N_EMBD * 4u)
#define TARGET_MAP_FLOATS ((OFF_OUTPUT + HEAD_N_EMBD * HEAD_N_VOCAB * 4u) / 4u)

typedef struct {
    char     log[16][64];
    int      n_log;
    /* rms_norm */
    uint32_t norm_n[4], norm_group[4], norm_rows[4];
    uint64_t norm_offset[4];
    const void *norm_map[4];
    int      n_norm;
    /* matmul */
    uint64_t mm_in[4], mm_out[4], mm_offset[4], mm_ntok[4];
    const void *mm_map[4];
    int      n_mm;
    /* embed */
    uint32_t embed_n_hc, embed_n_tokens;
    uint64_t embed_offset;
    const void *embed_map;
    /* block */
    uint32_t block_il, block_pos0, block_tokens;
    void    *block_graph, *block_cache;
    /* mixer */
    int      mixer_inject_null;
    uint32_t mixer_rows, mixer_n_hc, mixer_lowrank;
    uint64_t mixer_norm_off, mixer_down_off, mixer_up_off, mixer_inject_off;
    bool mixer_one_mapping;
    const void *mixer_map;
} head_log;

static head_log g_log;
static const float *g_head_map;
static const float *g_target_map;
static const float *g_forced_lm_logits;
static uint32_t g_forced_lm_rows;

static void log_call(const char *what) {
    if (g_log.n_log < 16) {
        snprintf(g_log.log[g_log.n_log], sizeof(g_log.log[0]), "%s", what);
        g_log.n_log++;
    }
}

static const float *map_at(const void *map, uint64_t offset) {
    return (const float *)((const unsigned char *)map + offset);
}

/* Pass-through: copies x to out and records the widths.  The norm's numerics
 * are L5b's business and are tested there. */
static int stub_rms_norm(ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
                         const void *map, uint64_t map_size, uint64_t offset,
                         uint32_t n, uint32_t group, uint32_t rows,
                         float eps, float weight_bias, int round_bf16) {
    (void)map_size; (void)eps; (void)weight_bias; (void)round_bf16;
    log_call("rms_norm");
    if (g_log.n_norm < 4) {
        g_log.norm_n[g_log.n_norm] = n;
        g_log.norm_group[g_log.n_norm] = group;
        g_log.norm_rows[g_log.n_norm] = rows;
        g_log.norm_offset[g_log.n_norm] = offset;
        g_log.norm_map[g_log.n_norm] = map;
        g_log.n_norm++;
    }
    const uint64_t bytes = (uint64_t)n * rows * sizeof(float);
    memcpy(out->data, x->data, (size_t)bytes);
    return 1;
}

/* A real f32 matmul: out[t][o] = sum_i x[t][i] * W[i * out_dim + o].
 *
 * The borrowed LM head may address a RANGE of output rows: the head shifts
 * weight_offset past whole Q8_0 rows, ds4_qwen4exp_q8_0_row_bytes(in_dim)
 * apart, the way the real kernel's row addressing does.  This stub's output
 * table is f32 input-major [in_dim][HEAD_N_VOCAB], so a ranged call decodes
 * its first row from the offset and gathers that row's columns; the full call
 * is first-row 0 and reduces to the same arithmetic, term for term. */
static int stub_matmul(ds4_gpu_tensor *out, const void *map, uint64_t map_size,
                       uint64_t offset, uint64_t in_dim, uint64_t out_dim,
                       const ds4_gpu_tensor *x, uint64_t n_tok) {
    (void)map_size;
    log_call("matmul");
    if (g_log.n_mm < 4) {
        g_log.mm_in[g_log.n_mm] = in_dim;
        g_log.mm_out[g_log.n_mm] = out_dim;
        g_log.mm_offset[g_log.n_mm] = offset;
        g_log.mm_ntok[g_log.n_mm] = n_tok;
        g_log.mm_map[g_log.n_mm] = map;
        g_log.n_mm++;
    }
    if (map == g_target_map && offset >= OFF_OUTPUT) {
        const uint64_t row_bytes =
            ds4_qwen4exp_q8_0_row_bytes((uint32_t)in_dim);
        const uint64_t skip = offset - OFF_OUTPUT;
        if (skip % row_bytes != 0u) return 0;
        const uint32_t r0 = (uint32_t)(skip / row_bytes);
        if ((uint64_t)r0 + out_dim > HEAD_N_VOCAB) return 0;
        if (g_forced_lm_logits) {
            if (n_tok > g_forced_lm_rows) return 0;
            float *os = (float *)out->data;
            for (uint64_t t = 0; t < n_tok; t++) {
                for (uint64_t o = 0; o < out_dim; o++) {
                    os[t * out_dim + o] =
                        g_forced_lm_logits[t * HEAD_N_VOCAB + r0 + o];
                }
            }
            return 1;
        }
        const float *w = map_at(map, OFF_OUTPUT);
        const float *xs = (const float *)x->data;
        float *os = (float *)out->data;
        for (uint64_t t = 0; t < n_tok; t++) {
            for (uint64_t o = 0; o < out_dim; o++) {
                float acc = 0.0f;
                for (uint64_t i = 0; i < in_dim; i++) {
                    acc += xs[t * in_dim + i] *
                           w[i * HEAD_N_VOCAB + r0 + o];
                }
                os[t * out_dim + o] = acc;
            }
        }
        return 1;
    }
    const float *w = map_at(map, offset);
    const float *xs = (const float *)x->data;
    float *os = (float *)out->data;
    for (uint64_t t = 0; t < n_tok; t++) {
        for (uint64_t o = 0; o < out_dim; o++) {
            float acc = 0.0f;
            for (uint64_t i = 0; i < in_dim; i++) {
                acc += xs[t * in_dim + i] * w[i * out_dim + o];
            }
            os[t * out_dim + o] = acc;
        }
    }
    return 1;
}

/* Gather rows and tile them into n_hc streams, the L5b contract. */
static int stub_embed(ds4_gpu_tensor *out_hc, ds4_gpu_tensor *rows_scratch,
                      const ds4_gpu_tensor *tokens, const void *map,
                      uint64_t map_size, uint64_t offset, uint32_t weight_type,
                      uint32_t n_vocab, uint32_t n_tokens, uint32_t n_embd,
                      uint32_t n_hc) {
    (void)map_size; (void)weight_type;
    log_call("embed");
    g_log.embed_n_hc = n_hc;
    g_log.embed_n_tokens = n_tokens;
    g_log.embed_offset = offset;
    g_log.embed_map = map;
    const float *tab = map_at(map, offset);
    const int32_t *ids = (const int32_t *)tokens->data;
    float *rows = (float *)rows_scratch->data;
    float *dst = (float *)out_hc->data;
    for (uint32_t t = 0; t < n_tokens; t++) {
        if ((uint32_t)ids[t] >= n_vocab) return 0;
        memcpy(rows + (size_t)t * n_embd, tab + (size_t)ids[t] * n_embd,
               n_embd * sizeof(float));
        for (uint32_t s = 0; s < n_hc; s++) {
            memcpy(dst + ((size_t)t * n_hc + s) * n_embd,
                   rows + (size_t)t * n_embd, n_embd * sizeof(float));
        }
    }
    return 1;
}

/* Pass-through: sums the streams.  The gated mixer's numerics are L5b's. */
static int stub_hc_mixer(ds4_gpu_tensor *mixed, ds4_gpu_tensor *inject,
                         ds4_gpu_tensor *normed, ds4_gpu_tensor *lowrank,
                         ds4_gpu_tensor *wide, const ds4_gpu_tensor *hyper,
                         const ds4_gpu_qwen4exp_slab *norm_weight,
                         const ds4_gpu_qwen4exp_slab *down_weight,
                         const ds4_gpu_qwen4exp_slab *up_weight,
                         const ds4_gpu_qwen4exp_slab *inject_weight,
                         uint32_t n_embd, uint32_t n_hc,
                         uint32_t n_lowrank, uint32_t rows, float eps,
                         float weight_bias, int round_bf16) {
    (void)normed; (void)lowrank; (void)wide;
    (void)eps; (void)weight_bias; (void)round_bf16;
    log_call("hc_mixer");
    g_log.mixer_inject_null = inject == NULL;
    g_log.mixer_rows = rows;
    g_log.mixer_n_hc = n_hc;
    g_log.mixer_lowrank = n_lowrank;
    /* One slab per weight now.  The head's three come from its own GGUF, so
     * they must all name the head mapping -- checked here rather than assumed,
     * since a slab pointed at the target's mapping would read the right offset
     * out of the wrong file. */
    g_log.mixer_norm_off = norm_weight->offset;
    g_log.mixer_down_off = down_weight->offset;
    g_log.mixer_up_off = up_weight->offset;
    g_log.mixer_inject_off = inject_weight ? inject_weight->offset : 0;
    g_log.mixer_one_mapping =
        norm_weight->map == down_weight->map &&
        norm_weight->map == up_weight->map &&
        norm_weight->map_size == down_weight->map_size &&
        norm_weight->map_size == up_weight->map_size;
    g_log.mixer_map = norm_weight->map;
    const float *h = (const float *)hyper->data;
    float *out = (float *)mixed->data;
    for (uint32_t t = 0; t < rows; t++) {
        for (uint32_t d = 0; d < n_embd; d++) {
            float acc = 0.0f;
            for (uint32_t s = 0; s < n_hc; s++) {
                acc += h[((size_t)t * n_hc + s) * n_embd + d];
            }
            out[(size_t)t * n_embd + d] = acc;
        }
    }
    return 1;
}

/* Pass-through: the identity, which is what L7 stands in for the qwen4exp
 * block until its own block forward is wired.  Returns NONZERO on success --
 * the one convention every member of the hook struct follows. */
static int stub_block(void *graph, void *cache, ds4_gpu_tensor *hyper,
                      uint32_t il, uint32_t pos0, uint32_t n_tokens) {
    (void)hyper;
    log_call("block");
    g_log.block_il = il;
    g_log.block_pos0 = pos0;
    g_log.block_tokens = n_tokens;
    g_log.block_graph = graph;
    g_log.block_cache = cache;
    return 1;
}

static void test_head_wiring(void) {
    printf("head wiring: nextn projections, the 49th block, the borrowed head\n");

    static float head_map[HEAD_MAP_FLOATS];
    static float target_map[TARGET_MAP_FLOATS];
    for (uint32_t i = 0; i < HEAD_MAP_FLOATS; i++) {
        head_map[i] = (float)((int)(mix64(i + 1u) % 17u) - 8) * 0.125f;
    }
    for (uint32_t i = 0; i < TARGET_MAP_FLOATS; i++) {
        target_map[i] = (float)((int)(mix64(i + 1000u) % 19u) - 9) * 0.0625f;
    }
    g_head_map = head_map;
    g_target_map = target_map;
    memset(&g_log, 0, sizeof(g_log));

    int graph_marker = 0, cache_marker = 0;
    ds4_qwen4exp_mtp_head h;
    memset(&h, 0, sizeof(h));
    h.head_map = head_map;   h.head_size = sizeof(head_map);
    h.target_map = target_map; h.target_size = sizeof(target_map);
    h.enorm_offset = OFF_ENORM;
    h.hnorm_offset = OFF_HNORM;
    h.eh_proj_offset = OFF_EH_PROJ;
    h.eh_proj_in_dim = 2u * HEAD_N_EMBD;
    h.hc_head_norm_offset = OFF_HC_NORM;
    h.hc_head_down_offset = OFF_HC_DOWN;
    h.hc_head_up_offset = OFF_HC_UP;
    h.token_embd_offset = OFF_TOKEN_EMBD;
    h.token_embd_type = 8u; /* Q8_0 in the shipped file */
    h.output_offset = OFF_OUTPUT;
    h.block_index = HEAD_BLOCK_IL;
    h.n_embd = HEAD_N_EMBD;
    h.n_hc = HEAD_N_HC;
    h.n_lowrank = HEAD_N_LOWRANK;
    h.n_vocab = HEAD_N_VOCAB;
    h.max_tokens = HEAD_ROWS;
    h.rms_eps = 1.0e-6f;
    h.weight_bias = 1.0f;
    h.round_bf16 = 1;
    h.hooks.rms_norm = stub_rms_norm;
    h.hooks.matmul_q8_0 = stub_matmul;
    h.hooks.embed = stub_embed;
    h.hooks.hc_mixer = stub_hc_mixer;
    h.hooks.block = stub_block;
    h.graph = &graph_marker;
    h.cache = &cache_marker;

    /* The 49th block is graph code and the head cannot run without it.  It was
     * legitimately NULL while the block was unwritten; now that the default
     * hooks bind it, an unbound `block` means a caller overrode the hooks and
     * dropped it, which must refuse by name rather than crash at the call. */
    {
        ds4_qwen4exp_mtp_head no_block = h;
        no_block.hooks.block = NULL;
        CHECK(ds4_qwen4exp_mtp_head_init(&no_block, g_err, sizeof(g_err)) != 0,
              "a head with no block hook initialised");
        CHECK(strstr(g_err, "block") != NULL,
              "the refusal does not name the missing block hook: %s", g_err);
    }

    CHECK(ds4_qwen4exp_mtp_head_init(&h, g_err, sizeof(g_err)) == 0,
          "head init failed: %s", g_err);

    const int next_tokens[HEAD_ROWS] = { 3, 5 };
    float multi_in[HEAD_ROWS * HEAD_HC_DIM];
    for (uint32_t i = 0; i < HEAD_ROWS * HEAD_HC_DIM; i++) {
        multi_in[i] = (float)((int)(mix64(i + 77u) % 13u) - 6) * 0.25f;
    }
    int draft[HEAD_ROWS] = { -1, -1 };
    float multi_out[HEAD_ROWS * HEAD_HC_DIM];
    CHECK(ds4_qwen4exp_mtp_head_forward(&h, next_tokens, multi_in, 12u,
                                        HEAD_ROWS, draft, multi_out,
                                        g_err, sizeof(g_err)) == 0,
          "head forward failed: %s", g_err);

    /* Call order. */
    static const char *want[] = { "embed", "rms_norm", "rms_norm", "matmul",
                                  "block", "hc_mixer", "matmul" };
    const int n_want = (int)(sizeof(want) / sizeof(want[0]));
    CHECK(g_log.n_log == n_want, "the head made %d calls, expected %d",
          g_log.n_log, n_want);
    for (int i = 0; i < n_want && i < g_log.n_log; i++) {
        CHECK(strcmp(g_log.log[i], want[i]) == 0,
              "call %d was %s, expected %s", i, g_log.log[i], want[i]);
    }

    /* The embedding is the TARGET's, gathered as plain rows. */
    CHECK(g_log.embed_map == (const void *)target_map,
          "the embedding did not come from the target mapping");
    CHECK(g_log.embed_offset == OFF_TOKEN_EMBD, "wrong token_embd offset");
    CHECK(g_log.embed_n_hc == 1u,
          "the embedding gather asked for %u streams, expected 1",
          g_log.embed_n_hc);

    /* enorm is over n_embd; hnorm is UNGROUPED over the whole hc row. */
    CHECK(g_log.n_norm == 2, "%d norms, expected 2", g_log.n_norm);
    CHECK(g_log.norm_offset[0] == OFF_ENORM, "wrong enorm offset");
    CHECK(g_log.norm_n[0] == HEAD_N_EMBD && g_log.norm_group[0] == HEAD_N_EMBD,
          "enorm ran over n=%u group=%u, expected %u/%u",
          g_log.norm_n[0], g_log.norm_group[0], HEAD_N_EMBD, HEAD_N_EMBD);
    CHECK(g_log.norm_offset[1] == OFF_HNORM, "wrong hnorm offset");
    CHECK(g_log.norm_n[1] == HEAD_HC_DIM && g_log.norm_group[1] == HEAD_HC_DIM,
          "hnorm ran over n=%u group=%u; it must be one statistic over the "
          "whole %u-wide row", g_log.norm_n[1], g_log.norm_group[1],
          HEAD_HC_DIM);
    CHECK(g_log.norm_map[0] == (const void *)head_map &&
          g_log.norm_map[1] == (const void *)head_map,
          "the nextn norms did not come from the head mapping");

    /* The head mixer takes one slab per weight.  All three of its weights live
     * in the head GGUF, so all three slabs must name the head mapping: a slab
     * left pointing at the target's mapping would read the right offset out of
     * the wrong file and produce plausible numbers. */
    CHECK(g_log.mixer_one_mapping,
          "the head mixer's three slabs do not agree on one mapping");
    CHECK(g_log.mixer_map == (const void *)head_map,
          "the head mixer did not read the head mapping");

    /* eh_proj: one matmul over rows * streams concatenated rows. */
    CHECK(g_log.mm_map[0] == (const void *)head_map, "eh_proj is not in the head map");
    CHECK(g_log.mm_offset[0] == OFF_EH_PROJ, "wrong eh_proj offset");
    CHECK(g_log.mm_in[0] == 2ull * HEAD_N_EMBD && g_log.mm_out[0] == HEAD_N_EMBD,
          "eh_proj ran %llu -> %llu, expected %u -> %u",
          (unsigned long long)g_log.mm_in[0],
          (unsigned long long)g_log.mm_out[0], 2u * HEAD_N_EMBD, HEAD_N_EMBD);
    CHECK(g_log.mm_ntok[0] == (uint64_t)HEAD_ROWS * HEAD_N_HC,
          "eh_proj ran %llu rows, expected %u",
          (unsigned long long)g_log.mm_ntok[0], HEAD_ROWS * HEAD_N_HC);

    /* The 49th block, on the head's own graph and cache. */
    CHECK(g_log.block_il == HEAD_BLOCK_IL, "the block ran at index %u, expected %u",
          g_log.block_il, HEAD_BLOCK_IL);
    CHECK(g_log.block_pos0 == 12u && g_log.block_tokens == HEAD_ROWS,
          "the block ran at position %u over %u rows",
          g_log.block_pos0, g_log.block_tokens);
    CHECK(g_log.block_graph == &graph_marker && g_log.block_cache == &cache_marker,
          "the block did not get the head's graph and cache");

    /* The head's mixer has no inject head. */
    CHECK(g_log.mixer_inject_null, "the head's mixer was given an inject head");
    CHECK(g_log.mixer_inject_off == 0, "the head's mixer got an inject offset");
    CHECK(g_log.mixer_norm_off == OFF_HC_NORM &&
          g_log.mixer_down_off == OFF_HC_DOWN &&
          g_log.mixer_up_off == OFF_HC_UP, "wrong hc_head offsets");
    CHECK(g_log.mixer_rows == HEAD_ROWS && g_log.mixer_n_hc == HEAD_N_HC &&
          g_log.mixer_lowrank == HEAD_N_LOWRANK, "wrong mixer widths");

    /* The LM head is the TARGET's. */
    CHECK(g_log.mm_map[1] == (const void *)target_map,
          "the LM head did not come from the target mapping");
    CHECK(g_log.mm_offset[1] == OFF_OUTPUT, "wrong output offset");
    CHECK(g_log.mm_in[1] == HEAD_N_EMBD && g_log.mm_out[1] == HEAD_N_VOCAB,
          "the LM head ran %llu -> %llu",
          (unsigned long long)g_log.mm_in[1],
          (unsigned long long)g_log.mm_out[1]);

    /* The algebra: hyper[t][s] = W_e . embed(next[t]) + W_h . multi[t][s].
     * With the norms passing through, that is the whole broadcast-and-add. */
    const float *eh = map_at(head_map, OFF_EH_PROJ);
    const float *emb = map_at(target_map, OFF_TOKEN_EMBD);
    const float *out_w = map_at(target_map, OFF_OUTPUT);
    float worst = 0.0f;
    for (uint32_t t = 0; t < HEAD_ROWS; t++) {
        float sample[HEAD_N_EMBD];
        for (uint32_t d = 0; d < HEAD_N_EMBD; d++) sample[d] = 0.0f;
        for (uint32_t s = 0; s < HEAD_N_HC; s++) {
            for (uint32_t o = 0; o < HEAD_N_EMBD; o++) {
                float acc = 0.0f;
                for (uint32_t i = 0; i < HEAD_N_EMBD; i++) {
                    acc += emb[(size_t)next_tokens[t] * HEAD_N_EMBD + i] *
                           eh[i * HEAD_N_EMBD + o];
                }
                for (uint32_t i = 0; i < HEAD_N_EMBD; i++) {
                    acc += multi_in[(size_t)t * HEAD_HC_DIM + s * HEAD_N_EMBD + i] *
                           eh[(HEAD_N_EMBD + i) * HEAD_N_EMBD + o];
                }
                const float got = multi_out[((size_t)t * HEAD_N_HC + s) *
                                            HEAD_N_EMBD + o];
                const float d = got - acc > 0 ? got - acc : acc - got;
                if (d > worst) worst = d;
                sample[o] += acc;
            }
        }
        float logits[HEAD_N_VOCAB];
        for (uint32_t v = 0; v < HEAD_N_VOCAB; v++) {
            float acc = 0.0f;
            for (uint32_t i = 0; i < HEAD_N_EMBD; i++) {
                acc += sample[i] * out_w[i * HEAD_N_VOCAB + v];
            }
            logits[v] = acc;
        }
        const int want_draft = ds4_qwen4exp_mtp_argmax(logits, HEAD_N_VOCAB);
        CHECK(draft[t] == want_draft,
              "row %u drafted %d, the composition gives %d",
              t, draft[t], want_draft);
    }
    CHECK(worst < 1.0e-5f,
          "the eh_proj broadcast-and-add is off by %g; the embedding half must "
          "reach every stream and the hidden half only its own", (double)worst);
    printf("  7 calls in order, eh_proj broadcast exact to %g\n", (double)worst);

    /* Seeds still reach the block, but only the last row reaches its
     * stateless final mixer and vocabulary projection.  The wide call is
     * the independent oracle for the proposal and preserved hyper row. */
    {
        int draft_last[1] = { -1 };
        float multi_last[HEAD_HC_DIM];
        const int calls_before = g_log.n_log;
        CHECK(ds4_qwen4exp_mtp_head_forward_last(&h, next_tokens, multi_in,
                                                 12u, HEAD_ROWS, draft_last,
                                                 multi_last,
                                                 g_err, sizeof(g_err)) == 0,
              "last-row head forward failed: %s", g_err);
        CHECK(g_log.n_log - calls_before == n_want,
              "the last-row forward made %d calls, expected %d",
              g_log.n_log - calls_before, n_want);
        CHECK(g_log.block_tokens == HEAD_ROWS && g_log.mixer_rows == 1u &&
                  g_log.mm_ntok[3] == 1u,
              "last-only must keep all block rows and project one logit row");
        CHECK(draft_last[0] == draft[HEAD_ROWS - 1u],
              "the last-row forward drafted %d, the wide forward's last row "
              "drafted %d", draft_last[0], draft[HEAD_ROWS - 1u]);
        CHECK(memcmp(multi_last,
                     multi_out + (size_t)(HEAD_ROWS - 1u) * HEAD_HC_DIM,
                     sizeof(multi_last)) == 0,
              "the last-row forward's multi row differs from the wide "
              "forward's last row");
        printf("  last-row entry agrees with the wide forward's final row\n");
    }

    /* The CUDA top-1 reducer seeds every lane with -infinity and ignores NaNs.
     * The historical CPU scan instead seeds from entry zero, so a NaN there
     * pins the proposal to token zero.  Exercise the real head forward around
     * that one semantic difference, while retaining coverage for later NaNs,
     * ties, signed zero and infinities. */
    {
        static const float cases[][HEAD_ROWS * HEAD_N_VOCAB] = {
            {
                NAN, -INFINITY, 2.0f, 9.0f, 8.0f, -0.0f, +0.0f, 1.0f,
                -INFINITY, -0.0f, +0.0f, NAN, 5.0f, 5.0f, -INFINITY, 4.0f,
            },
            {
                -INFINITY, -INFINITY, NAN, -INFINITY,
                -INFINITY, -INFINITY, -INFINITY, -INFINITY,
                1.0f, INFINITY, NAN, INFINITY, -INFINITY, 0.0f, -0.0f, 3.0f,
            },
            {
                -0.0f, +0.0f, NAN, -0.0f, +0.0f, -INFINITY, -0.0f, +0.0f,
                NAN, -INFINITY, 3.0f, 11.0f, 10.0f, +0.0f, -0.0f, 9.0f,
            },
        };
        for (uint32_t c = 0; c < sizeof(cases) / sizeof(cases[0]); c++) {
            int got[HEAD_ROWS] = { -1, -1 };
            int got_last[1] = { -1 };
            g_forced_lm_logits = cases[c];
            g_forced_lm_rows = HEAD_ROWS;
            CHECK(ds4_qwen4exp_mtp_head_forward(&h, next_tokens, multi_in,
                                                 12u, HEAD_ROWS, got, NULL,
                                                 g_err, sizeof(g_err)) == 0,
                  "special-value head forward failed: %s", g_err);
            for (uint32_t t = 0; t < HEAD_ROWS; t++) {
                const int want = ds4_qwen4exp_mtp_argmax(
                    cases[c] + (size_t)t * HEAD_N_VOCAB, HEAD_N_VOCAB);
                CHECK(got[t] == want,
                      "special-value case %u row %u drafted %d, CPU reference %d",
                      c, t, got[t], want);
            }
            /* Inject the same final-row distribution at the narrowed
             * vocabulary projection's output.  Its physical row is zero. */
            g_forced_lm_logits = cases[c] +
                (size_t)(HEAD_ROWS - 1u) * HEAD_N_VOCAB;
            g_forced_lm_rows = 1u;
            CHECK(ds4_qwen4exp_mtp_head_forward_last(
                      &h, next_tokens, multi_in, 12u, HEAD_ROWS, got_last,
                      NULL, g_err, sizeof(g_err)) == 0,
                  "special-value last-row forward failed: %s", g_err);
            CHECK(got_last[0] == got[HEAD_ROWS - 1u],
                  "special-value case %u last-only drafted %d, multi-row %d",
                  c, got_last[0], got[HEAD_ROWS - 1u]);
        }
        g_forced_lm_logits = NULL;
        g_forced_lm_rows = 0u;
        printf("  GPU-style top-1 matches CPU proposals for NaNs, ties, "
               "signed zero and infinities\n");
    }

    /* A row count above the built capacity is refused by name. */
    CHECK(ds4_qwen4exp_mtp_head_forward(&h, next_tokens, multi_in, 0u,
                                        HEAD_ROWS + 1u, draft, NULL,
                                        g_err, sizeof(g_err)) < 0,
          "the head accepted more rows than it was built for");
    ds4_qwen4exp_mtp_head_free(&h);
    CHECK(h.t_top1 == NULL && h.top1_host == NULL,
          "the head left its top-1 scratch allocated after free");

    /* The shipped head's eh_proj is [5120, 2560]: the embedding half occupies
     * the first n_embd input rows and the hidden half the rest, so a head that
     * does not take 2 * n_embd inputs is not this head. */
    ds4_qwen4exp_mtp_head narrow = h;
    narrow.eh_proj_in_dim = HEAD_N_EMBD;
    CHECK(ds4_qwen4exp_mtp_head_init(&narrow, g_err, sizeof(g_err)) < 0,
          "a head with a half-width eh_proj was built");
    CHECK(strstr(g_err, "eh_proj") != NULL,
          "the refusal does not name eh_proj: %s", g_err);
}

/* ========================================================================
 * The draft vocabulary shortlist
 * ======================================================================== */

/*
 * The same head, the same stubs, one environment knob:
 * DS4_QWEN4EXP_DRAFT_VOCAB_PREFIX narrows the DRAFT's borrowed-LM-head rows to
 * [0, prefix) plus the DS4_QWEN4EXP_DRAFT_VOCAB_TAIL added-token rows at the
 * top of the table.  What must hold:
 *
 *   (a) a prefix covering the whole vocabulary drafts exactly what the
 *       off-mode head drafts, token for token;
 *   (b) an armed prefix drafts the argmax of the FULL logits restricted to
 *       the shortlist ids, ties to the LOWEST id -- the packed sweep is the
 *       first-max rule over the ids in ascending order;
 *   (c) the restriction is real: a winner outside the list is NOT drafted;
 *   (d) the projection reads the ranges as row ranges of the SAME weight --
 *       two calls, the second at output_offset advanced past n_vocab - tail
 *       Q8_0 rows -- and never a copied or gathered table;
 *   (e) init refuses a prefix or tail that overruns or overlaps.
 */

/* Build and init a head over the stub tables, with the environment as the
 * caller left it.  The maps are refilled deterministically so every
 * configuration sees the same weights.  Returns 0 on success. */
static int build_shortlist_head(ds4_qwen4exp_mtp_head *h) {
    static float head_map[HEAD_MAP_FLOATS];
    static float target_map[TARGET_MAP_FLOATS];
    static int graph_marker, cache_marker;
    for (uint32_t i = 0; i < HEAD_MAP_FLOATS; i++) {
        head_map[i] = (float)((int)(mix64(i + 1u) % 17u) - 8) * 0.125f;
    }
    for (uint32_t i = 0; i < TARGET_MAP_FLOATS; i++) {
        target_map[i] = (float)((int)(mix64(i + 1000u) % 19u) - 9) * 0.0625f;
    }
    g_head_map = head_map;
    g_target_map = target_map;
    memset(&g_log, 0, sizeof(g_log));

    memset(h, 0, sizeof(*h));
    h->head_map = head_map;     h->head_size = sizeof(head_map);
    h->target_map = target_map; h->target_size = sizeof(target_map);
    h->enorm_offset = OFF_ENORM;
    h->hnorm_offset = OFF_HNORM;
    h->eh_proj_offset = OFF_EH_PROJ;
    h->eh_proj_in_dim = 2u * HEAD_N_EMBD;
    h->hc_head_norm_offset = OFF_HC_NORM;
    h->hc_head_down_offset = OFF_HC_DOWN;
    h->hc_head_up_offset = OFF_HC_UP;
    h->token_embd_offset = OFF_TOKEN_EMBD;
    h->token_embd_type = 8u; /* Q8_0 in the shipped file */
    h->output_offset = OFF_OUTPUT;
    h->block_index = HEAD_BLOCK_IL;
    h->n_embd = HEAD_N_EMBD;
    h->n_hc = HEAD_N_HC;
    h->n_lowrank = HEAD_N_LOWRANK;
    h->n_vocab = HEAD_N_VOCAB;
    h->max_tokens = HEAD_ROWS;
    h->rms_eps = 1.0e-6f;
    h->weight_bias = 1.0f;
    h->round_bf16 = 1;
    h->hooks.rms_norm = stub_rms_norm;
    h->hooks.matmul_q8_0 = stub_matmul;
    h->hooks.embed = stub_embed;
    h->hooks.hc_mixer = stub_hc_mixer;
    h->hooks.block = stub_block;
    h->graph = &graph_marker;
    h->cache = &cache_marker;
    return ds4_qwen4exp_mtp_head_init(h, g_err, sizeof(g_err));
}

/* The composition oracle: the FULL-vocabulary logits the head's own algebra
 * produces for one row over the stub tables -- the same walk
 * test_head_wiring checks the wide forward against. */
static void oracle_logits(const int *next_tokens, const float *multi_in,
                          uint32_t row, float *logits) {
    const float *eh = map_at(g_head_map, OFF_EH_PROJ);
    const float *emb = map_at(g_target_map, OFF_TOKEN_EMBD);
    const float *out_w = map_at(g_target_map, OFF_OUTPUT);
    float sample[HEAD_N_EMBD];
    for (uint32_t d = 0; d < HEAD_N_EMBD; d++) sample[d] = 0.0f;
    for (uint32_t s = 0; s < HEAD_N_HC; s++) {
        for (uint32_t o = 0; o < HEAD_N_EMBD; o++) {
            float acc = 0.0f;
            for (uint32_t i = 0; i < HEAD_N_EMBD; i++) {
                acc += emb[(size_t)next_tokens[row] * HEAD_N_EMBD + i] *
                       eh[i * HEAD_N_EMBD + o];
            }
            for (uint32_t i = 0; i < HEAD_N_EMBD; i++) {
                acc += multi_in[(size_t)row * HEAD_HC_DIM + s * HEAD_N_EMBD + i] *
                       eh[(HEAD_N_EMBD + i) * HEAD_N_EMBD + o];
            }
            sample[o] += acc;
        }
    }
    for (uint32_t v = 0; v < HEAD_N_VOCAB; v++) {
        float acc = 0.0f;
        for (uint32_t i = 0; i < HEAD_N_EMBD; i++) {
            acc += sample[i] * out_w[i * HEAD_N_VOCAB + v];
        }
        logits[v] = acc;
    }
}

/* The argmax of the full logits over the shortlist only, ties to the lowest
 * id: the ids in ascending order with a strict > -- which is what the packed
 * GPU sweep implements, since the packed order IS the id order. */
static int shortlist_argmax(const float *logits, uint32_t prefix,
                            uint32_t tail) {
    int best = -1;
    for (uint32_t k = 0; k < prefix + tail; k++) {
        const uint32_t id = k < prefix ? k : HEAD_N_VOCAB - tail + (k - prefix);
        if (best < 0 || logits[id] > logits[best]) best = (int)id;
    }
    return best;
}

static void test_draft_vocab_shortlist(void) {
    printf("draft vocabulary shortlist\n");
    const int next_tokens[HEAD_ROWS] = { 3, 5 };
    float multi_in[HEAD_ROWS * HEAD_HC_DIM];
    for (uint32_t i = 0; i < HEAD_ROWS * HEAD_HC_DIM; i++) {
        multi_in[i] = (float)((int)(mix64(i + 77u) % 13u) - 6) * 0.25f;
    }

    /* (a) A prefix covering the whole vocabulary drafts what the off-mode
     * head drafts: the launch parameters are the full call's, so the forward
     * cannot tell the two configurations apart. */
    int off_draft[HEAD_ROWS] = { -1, -1 };
    int full_draft[HEAD_ROWS] = { -1, -1 };
    unsetenv("DS4_QWEN4EXP_DRAFT_VOCAB_PREFIX");
    unsetenv("DS4_QWEN4EXP_DRAFT_VOCAB_TAIL");
    {
        ds4_qwen4exp_mtp_head h;
        CHECK(build_shortlist_head(&h) == 0, "off-mode head init: %s", g_err);
        CHECK(h.draft_vocab_prefix == 0 && h.draft_vocab_tail == 0,
              "an unset prefix armed %u/%u",
              h.draft_vocab_prefix, h.draft_vocab_tail);
        CHECK(ds4_qwen4exp_mtp_head_forward(&h, next_tokens, multi_in, 12u,
                                            HEAD_ROWS, off_draft, NULL,
                                            g_err, sizeof(g_err)) == 0,
              "off-mode forward failed: %s", g_err);
        ds4_qwen4exp_mtp_head_free(&h);
    }
    setenv("DS4_QWEN4EXP_DRAFT_VOCAB_PREFIX", "8", 1);
    setenv("DS4_QWEN4EXP_DRAFT_VOCAB_TAIL", "0", 1);
    {
        ds4_qwen4exp_mtp_head h;
        CHECK(build_shortlist_head(&h) == 0,
              "whole-vocabulary prefix was refused: %s", g_err);
        CHECK(ds4_qwen4exp_mtp_head_forward(&h, next_tokens, multi_in, 12u,
                                            HEAD_ROWS, full_draft, NULL,
                                            g_err, sizeof(g_err)) == 0,
              "whole-vocabulary forward failed: %s", g_err);
        CHECK(g_log.n_mm == 2 && g_log.mm_out[1] == HEAD_N_VOCAB &&
              g_log.mm_offset[1] == OFF_OUTPUT,
              "a whole-vocabulary prefix must keep the one full-width LM-head "
              "call");
        ds4_qwen4exp_mtp_head_free(&h);
    }
    CHECK(memcmp(off_draft, full_draft, sizeof(off_draft)) == 0,
          "a whole-vocabulary prefix drafted {%d, %d}, the off-mode head "
          "{%d, %d}", full_draft[0], full_draft[1],
          off_draft[0], off_draft[1]);
    printf("  prefix %u + tail %u drafts the off-mode tokens\n",
           HEAD_N_VOCAB, 0u);

    /* (b)+(c) Armed prefixes over the real stub weights: every row drafts
     * the shortlist argmax of the full logits.  The last pair is the
     * adjacent-range boundary (prefix ends exactly where the tail begins). */
    static const struct { uint32_t prefix, tail; } cfg[] = {
        { 5u, 2u }, { 6u, 2u }, { 4u, 0u }, { 1u, 7u }, { 7u, 1u },
    };
    for (size_t c = 0; c < sizeof(cfg) / sizeof(cfg[0]); c++) {
        char pbuf[16], tbuf[16];
        snprintf(pbuf, sizeof(pbuf), "%u", cfg[c].prefix);
        snprintf(tbuf, sizeof(tbuf), "%u", cfg[c].tail);
        setenv("DS4_QWEN4EXP_DRAFT_VOCAB_PREFIX", pbuf, 1);
        setenv("DS4_QWEN4EXP_DRAFT_VOCAB_TAIL", tbuf, 1);
        ds4_qwen4exp_mtp_head h;
        CHECK(build_shortlist_head(&h) == 0,
              "prefix %u tail %u was refused: %s",
              cfg[c].prefix, cfg[c].tail, g_err);
        CHECK(h.draft_vocab_prefix == cfg[c].prefix &&
              h.draft_vocab_tail == cfg[c].tail,
              "prefix %u tail %u armed %u/%u", cfg[c].prefix, cfg[c].tail,
              h.draft_vocab_prefix, h.draft_vocab_tail);
        CHECK((h.t_logits_prefix != NULL) == (cfg[c].tail != 0u),
              "prefix %u tail %u staged its buffers wrong",
              cfg[c].prefix, cfg[c].tail);

        int draft[HEAD_ROWS] = { -1, -1 };
        int draft_last[1] = { -1 };
        CHECK(ds4_qwen4exp_mtp_head_forward(&h, next_tokens, multi_in, 12u,
                                            HEAD_ROWS, draft, NULL,
                                            g_err, sizeof(g_err)) == 0,
              "prefix %u forward failed: %s", cfg[c].prefix, g_err);
        CHECK(ds4_qwen4exp_mtp_head_forward_last(&h, next_tokens, multi_in,
                                                 12u, HEAD_ROWS, draft_last,
                                                 NULL, g_err,
                                                 sizeof(g_err)) == 0,
              "prefix %u last-row forward failed: %s", cfg[c].prefix, g_err);
        CHECK(draft_last[0] == draft[HEAD_ROWS - 1u],
              "prefix %u: the last-row entry drafted %d, the wide forward's "
              "last row %d", cfg[c].prefix, draft_last[0],
              draft[HEAD_ROWS - 1u]);
        float logits[HEAD_ROWS][HEAD_N_VOCAB];
        for (uint32_t t = 0; t < HEAD_ROWS; t++) {
            oracle_logits(next_tokens, multi_in, t, logits[t]);
            const int want = shortlist_argmax(logits[t], cfg[c].prefix,
                                              cfg[c].tail);
            CHECK(draft[t] == want,
                  "prefix %u tail %u row %u drafted %d, the shortlist argmax "
                  "of the full logits is %d",
                  cfg[c].prefix, cfg[c].tail, t, draft[t], want);
        }
        ds4_qwen4exp_mtp_head_free(&h);
        printf("  prefix %u + tail %u -> drafts the shortlist argmax\n",
               cfg[c].prefix, cfg[c].tail);
    }

    /* (d) The 5+2 configuration's call pattern: the LM head is TWO row-range
     * calls over the one weight -- prefix at output_offset, tail at
     * output_offset advanced past n_vocab - tail whole Q8_0 rows. */
    setenv("DS4_QWEN4EXP_DRAFT_VOCAB_PREFIX", "5", 1);
    setenv("DS4_QWEN4EXP_DRAFT_VOCAB_TAIL", "2", 1);
    {
        ds4_qwen4exp_mtp_head h;
        CHECK(build_shortlist_head(&h) == 0, "5+2 head init: %s", g_err);
        int draft[HEAD_ROWS] = { -1, -1 };
        CHECK(ds4_qwen4exp_mtp_head_forward(&h, next_tokens, multi_in, 12u,
                                            HEAD_ROWS, draft, NULL,
                                            g_err, sizeof(g_err)) == 0,
              "5+2 forward failed: %s", g_err);
        static const char *want[] = {
            "embed", "rms_norm", "rms_norm", "matmul", "block", "hc_mixer",
            "matmul", "matmul",
        };
        const int n_want = (int)(sizeof(want) / sizeof(want[0]));
        CHECK(g_log.n_log == n_want, "the 5+2 head made %d calls, expected %d",
              g_log.n_log, n_want);
        for (int i = 0; i < n_want && i < g_log.n_log; i++) {
            CHECK(strcmp(g_log.log[i], want[i]) == 0,
                  "call %d was %s, expected %s", i, g_log.log[i], want[i]);
        }
        const uint64_t row_bytes =
            ds4_qwen4exp_q8_0_row_bytes(HEAD_N_EMBD);
        CHECK(g_log.n_mm == 3, "the 5+2 head made %d matmuls, expected 3",
              g_log.n_mm);
        CHECK(g_log.mm_map[1] == (const void *)g_target_map &&
              g_log.mm_map[2] == (const void *)g_target_map,
              "a ranged LM-head call left the target mapping");
        CHECK(g_log.mm_out[1] == 5u && g_log.mm_offset[1] == OFF_OUTPUT,
              "the prefix call ran width %llu at offset %llu",
              (unsigned long long)g_log.mm_out[1],
              (unsigned long long)g_log.mm_offset[1]);
        CHECK(g_log.mm_out[2] == 2u &&
              g_log.mm_offset[2] ==
                  OFF_OUTPUT + (uint64_t)(HEAD_N_VOCAB - 2u) * row_bytes,
              "the tail call ran width %llu at offset %llu, expected %llu",
              (unsigned long long)g_log.mm_out[2],
              (unsigned long long)g_log.mm_offset[2],
              (unsigned long long)(OFF_OUTPUT +
                                   (uint64_t)(HEAD_N_VOCAB - 2u) * row_bytes));
        CHECK(g_log.mm_ntok[1] == HEAD_ROWS && g_log.mm_ntok[2] == HEAD_ROWS,
              "a ranged LM-head call changed the row count");
        printf("  5+2: the LM head is two row-range calls over one weight\n");
        ds4_qwen4exp_mtp_head_free(&h);
    }

    /* (c) again, with the logits forced: a winner outside the list is not
     * drafted, a tie across the ranges goes to the LOWER id, and a NaN at
     * token zero pins the proposal to zero exactly as the full sweep does. */
    setenv("DS4_QWEN4EXP_DRAFT_VOCAB_PREFIX", "5", 1);
    setenv("DS4_QWEN4EXP_DRAFT_VOCAB_TAIL", "2", 1);
    {
        static const float cases[][HEAD_ROWS * HEAD_N_VOCAB] = {
            /* full winner id 5 (outside 0..4 and 6..7): draft the best of the
             * list, id 7's 10.0, not the 11.0 at id 5. */
            { 1, 2, 3, 9, 2, 11, 4, 10,
              0, 8, 1, 2, 3, 4, 5, 6 },
            /* ids 4 and 6 tie at the top: the lower id wins. */
            { 0, 1, 2, 3, 8, 0, 8, 3,
              4, 4, 4, 4, 4, 4, 4, 4 },
            /* the winner is inside the prefix: unchanged. */
            { 0, 1, 9, 2, 3, 4, 5, 6,
              7, 6, 5, 4, 3, 2, 1, 0 },
            /* a NaN at token zero pins the proposal to zero. */
            { NAN, 5, 4, 3, 2, 1, 6, 7,
              7, 6, 5, 4, 3, 2, 1, 0 },
        };
        for (uint32_t c = 0; c < sizeof(cases) / sizeof(cases[0]); c++) {
            ds4_qwen4exp_mtp_head h;
            CHECK(build_shortlist_head(&h) == 0, "forced case init: %s", g_err);
            int got[HEAD_ROWS] = { -1, -1 };
            g_forced_lm_logits = cases[c];
            g_forced_lm_rows = HEAD_ROWS;
            CHECK(ds4_qwen4exp_mtp_head_forward(&h, next_tokens, multi_in,
                                                12u, HEAD_ROWS, got, NULL,
                                                g_err, sizeof(g_err)) == 0,
                  "forced case %u forward failed: %s", c, g_err);
            for (uint32_t t = 0; t < HEAD_ROWS; t++) {
                const int want = shortlist_argmax(
                    cases[c] + (size_t)t * HEAD_N_VOCAB, 5u, 2u);
                CHECK(got[t] == want,
                      "forced case %u row %u drafted %d, the shortlist argmax "
                      "is %d", c, t, got[t], want);
            }
            printf("  forced case %u -> {%d, %d}\n", c, got[0], got[1]);
            g_forced_lm_logits = NULL;
            g_forced_lm_rows = 0u;
            ds4_qwen4exp_mtp_head_free(&h);
        }
    }

    /* (e) init refuses, by name, a prefix or tail that cannot be laid out. */
    static const struct { const char *p, *t; const char *must_name; } bad[] = {
        { "9", "0", "DS4_QWEN4EXP_DRAFT_VOCAB_PREFIX" },
        { "5", "4", "exceeds the vocabulary" },
        { "2", "9", "DS4_QWEN4EXP_DRAFT_VOCAB_TAIL" },
        { "abc", "2", "DS4_QWEN4EXP_DRAFT_VOCAB_PREFIX" },
        { "5", "2x", "DS4_QWEN4EXP_DRAFT_VOCAB_TAIL" },
    };
    for (size_t k = 0; k < sizeof(bad) / sizeof(bad[0]); k++) {
        setenv("DS4_QWEN4EXP_DRAFT_VOCAB_PREFIX", bad[k].p, 1);
        setenv("DS4_QWEN4EXP_DRAFT_VOCAB_TAIL", bad[k].t, 1);
        ds4_qwen4exp_mtp_head h;
        g_err[0] = '\0';
        CHECK(build_shortlist_head(&h) < 0,
              "prefix \"%s\" tail \"%s\" was accepted", bad[k].p, bad[k].t);
        CHECK(strstr(g_err, bad[k].must_name) != NULL,
              "the refusal for prefix \"%s\" tail \"%s\" does not name %s: %s",
              bad[k].p, bad[k].t, bad[k].must_name, g_err);
        ds4_qwen4exp_mtp_head_free(&h);
        printf("  prefix %s tail %s -> %s\n", bad[k].p, bad[k].t, g_err);
    }

    /* Leave the environment off: nothing later in this suite may inherit a
     * shortlist it did not ask for. */
    unsetenv("DS4_QWEN4EXP_DRAFT_VOCAB_PREFIX");
    unsetenv("DS4_QWEN4EXP_DRAFT_VOCAB_TAIL");
}

int main(void) {
    printf("qwen4exp MTP tests\n\n");
    test_exactness();
    printf("\n");
    test_rollback_negative_controls();
    printf("\n");
    /* The same two, with the seam's batched entry unbound: the cycle then
     * seeds and drafts one head row per call, which is the path a graph that
     * binds no draft_rows takes.  Both bindings must give one stream. */
    g_ref_batched_draft = 0;
    printf("(again, one head row per call: draft_rows unbound)\n");
    test_exactness();
    printf("\n");
    test_rollback_negative_controls();
    printf("\n");
    g_ref_batched_draft = 1;
    test_head_cache_boundary();
    printf("\n");
    test_row_invariance_is_load_bearing();
    printf("\n");
    test_depth_zero_is_serial();
    printf("\n");
    test_depth_envelope();
    printf("\n");
    test_no_yield_guard();
    printf("\n");
    test_rollback_contract();
    printf("\n");
    test_budget();
    printf("\n");
    test_deferred_frontier_logits();
    printf("\n");
    test_head_wiring();
    printf("\n");
    test_draft_vocab_shortlist();
    printf("\n");
    if (g_failures) {
        printf("FAILED: %d check(s)\n", g_failures);
        return 1;
    }
    printf("all qwen4exp MTP checks passed\n");
    return 0;
}
