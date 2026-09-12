/*
 * qwen4exp multi-token prediction: the depth-1 cycle and the head wiring.
 * See ds4_qwen4exp_mtp.h for the contract this implements.
 */

#include "ds4_qwen4exp_mtp.h"

#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* A monotonic nanosecond reading for the phase counters.  Monotonic and not
 * wall so that a clock step cannot make a phase look negative. */
static uint64_t mtp_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static int mtp_fail(char *err, size_t errlen, const char *fmt, ...) {
    if (err && errlen) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(err, errlen, fmt, ap);
        va_end(ap);
    }
    return -1;
}

int ds4_qwen4exp_mtp_argmax(const float *logits, uint32_t n_vocab) {
    int best = 0;
    float bv = logits[0];
    for (uint32_t i = 1; i < n_vocab; i++) {
        /* Strict >: the lowest id wins a tie, which is what ds4s_argmax
         * documents and what the serial leg does. */
        if (logits[i] > bv) { bv = logits[i]; best = (int)i; }
    }
    return best;
}

/* ------------------------------------------------------------------------
 * Depth envelope
 * ------------------------------------------------------------------------ */

int ds4_qwen4exp_mtp_depth_from_draft_tokens(int draft_tokens,
                                             char *err, size_t errlen) {
    if (draft_tokens < 1) {
        return mtp_fail(err, errlen,
                        "qwen4exp MTP: DS4_MTP_DRAFT_TOKENS=%d is below 1; "
                        "1 is a serial leg and 2 is draft depth 1",
                        draft_tokens);
    }
    const int depth = draft_tokens - 1;
    if (depth > DS4_QWEN4EXP_IMPLEMENTED_DEPTH) {
        return mtp_fail(err, errlen,
                        "qwen4exp MTP: DS4_MTP_DRAFT_TOKENS=%d asks for draft "
                        "depth %d; this build implements depths 1 to %d "
                        "(DS4_QWEN4EXP_IMPLEMENTED_DEPTH=%d).  Depth %d would "
                        "need a verify wider than the chain the head is sized "
                        "for.  Set DS4_MTP_DRAFT_TOKENS between 2 and %d, or 1 "
                        "for a serial leg",
                        draft_tokens, depth, DS4_QWEN4EXP_IMPLEMENTED_DEPTH,
                        DS4_QWEN4EXP_IMPLEMENTED_DEPTH, depth,
                        DS4_QWEN4EXP_IMPLEMENTED_DEPTH + 1);
    }
    return depth;
}

int ds4_qwen4exp_mtp_check_no_yield_guard(char *err, size_t errlen) {
    const char *raw = getenv("DS4_QWEN_MTP_QUENCH");
    if (raw && raw[0] && strcmp(raw, "0") != 0) {
        return mtp_fail(err, errlen,
                        "qwen4exp MTP: DS4_QWEN_MTP_QUENCH=\"%s\" arms an "
                        "adaptive disable, and a scored leg runs its declared "
                        "configuration end to end.  Set DS4_QWEN_MTP_QUENCH=0 "
                        "or leave it unset", raw);
    }
    return 0;
}

/* ------------------------------------------------------------------------
 * The rollback contract
 * ------------------------------------------------------------------------ */

ds4_qwen4exp_rollback_kind ds4_qwen4exp_state_kind(ds4_qwen4exp_state_id id) {
    switch (id) {
    case DS4_QWEN4EXP_STATE_GDN_RECURRENT:
    case DS4_QWEN4EXP_STATE_GDN_CONV:
    case DS4_QWEN4EXP_STATE_PLE_HISTORY:
    case DS4_QWEN4EXP_STATE_PLE_CONV:
        return DS4_QWEN4EXP_ROLLBACK_SELECT_ROW;
    case DS4_QWEN4EXP_STATE_QSA_KV:
    case DS4_QWEN4EXP_STATE_QSA_INDEXER_TAPE:
    case DS4_QWEN4EXP_STATE_MTP_HEAD_CACHE:
    case DS4_QWEN4EXP_STATE_COUNT:
    default:
        return DS4_QWEN4EXP_ROLLBACK_TRUNCATE;
    }
}

const char *ds4_qwen4exp_state_name(ds4_qwen4exp_state_id id) {
    switch (id) {
    case DS4_QWEN4EXP_STATE_GDN_RECURRENT:    return "gdn recurrent state";
    case DS4_QWEN4EXP_STATE_GDN_CONV:         return "gdn conv history";
    case DS4_QWEN4EXP_STATE_QSA_KV:           return "qsa kv cache";
    case DS4_QWEN4EXP_STATE_QSA_INDEXER_TAPE: return "qsa indexer tape";
    case DS4_QWEN4EXP_STATE_PLE_HISTORY:      return "ple n-gram history";
    case DS4_QWEN4EXP_STATE_PLE_CONV:         return "ple conv state";
    case DS4_QWEN4EXP_STATE_MTP_HEAD_CACHE:   return "mtp head cache";
    case DS4_QWEN4EXP_STATE_COUNT:
    default:                                  return "unknown state";
    }
}

void ds4_qwen4exp_rollback_init(ds4_qwen4exp_rollback_set *set) {
    memset(set, 0, sizeof(*set));
}

int ds4_qwen4exp_rollback_register(ds4_qwen4exp_rollback_set *set,
                                   ds4_qwen4exp_state_id id,
                                   const ds4_qwen4exp_rollback_object *obj,
                                   char *err, size_t errlen) {
    if ((unsigned)id >= (unsigned)DS4_QWEN4EXP_STATE_COUNT) {
        return mtp_fail(err, errlen,
                        "qwen4exp MTP rollback: state id %d is out of range",
                        (int)id);
    }
    const char *name = ds4_qwen4exp_state_name(id);
    const ds4_qwen4exp_rollback_kind kind = ds4_qwen4exp_state_kind(id);
    if (kind == DS4_QWEN4EXP_ROLLBACK_SELECT_ROW) {
        if (!obj->select_row) {
            return mtp_fail(err, errlen,
                            "qwen4exp MTP rollback: %s is select-row class and "
                            "needs select_row()", name);
        }
        if (obj->truncate) {
            return mtp_fail(err, errlen,
                            "qwen4exp MTP rollback: %s is select-row class and "
                            "must not supply truncate(); it carries no "
                            "position to truncate to", name);
        }
    } else {
        if (!obj->truncate) {
            return mtp_fail(err, errlen,
                            "qwen4exp MTP rollback: %s is truncate class and "
                            "needs truncate()", name);
        }
        if (obj->select_row) {
            return mtp_fail(err, errlen,
                            "qwen4exp MTP rollback: %s is truncate class and "
                            "must not supply select_row(); an append-only "
                            "cache is rolled back by dropping its tail", name);
        }
    }
    set->obj[id] = *obj;
    set->registered[id] = true;
    return 0;
}

int ds4_qwen4exp_rollback_check(const ds4_qwen4exp_rollback_set *set,
                                char *err, size_t errlen) {
    for (int i = 0; i < DS4_QWEN4EXP_STATE_COUNT; i++) {
        if (!set->registered[i]) {
            return mtp_fail(err, errlen,
                            "qwen4exp MTP rollback: %s is not registered; a "
                            "rejecting round would leave it holding the "
                            "rejected token",
                            ds4_qwen4exp_state_name((ds4_qwen4exp_state_id)i));
        }
    }
    return 0;
}

static int rollback_select_row(const ds4_qwen4exp_rollback_set *set,
                               uint32_t row, char *err, size_t errlen) {
    for (int i = 0; i < DS4_QWEN4EXP_STATE_COUNT; i++) {
        const ds4_qwen4exp_rollback_object *o = &set->obj[i];
        if (!o->select_row) continue;
        if (o->select_row(o->ctx, row) != 0) {
            return mtp_fail(err, errlen,
                            "qwen4exp MTP rollback: %s could not adopt the "
                            "state after verify row %u",
                            ds4_qwen4exp_state_name((ds4_qwen4exp_state_id)i),
                            row);
        }
    }
    return 0;
}

static int rollback_truncate_all(const ds4_qwen4exp_rollback_set *set,
                                 uint32_t pos, char *err, size_t errlen) {
    for (int i = 0; i < DS4_QWEN4EXP_STATE_COUNT; i++) {
        const ds4_qwen4exp_rollback_object *o = &set->obj[i];
        if (!o->truncate) continue;
        if (o->truncate(o->ctx, pos) != 0) {
            return mtp_fail(err, errlen,
                            "qwen4exp MTP rollback: truncate of %s to position "
                            "%u failed",
                            ds4_qwen4exp_state_name((ds4_qwen4exp_state_id)i),
                            pos);
        }
    }
    return 0;
}

/* ------------------------------------------------------------------------
 * State and counters
 * ------------------------------------------------------------------------ */

int ds4_qwen4exp_mtp_state_init(ds4_qwen4exp_mtp_state *st, int depth,
                                const ds4_qwen4exp_rollback_set *rollback,
                                uint32_t hc_dim, uint32_t n_vocab,
                                char *err, size_t errlen) {
    memset(st, 0, sizeof(*st));
    if (!rollback) {
        return mtp_fail(err, errlen,
                        "qwen4exp MTP: a rollback set is required; a rejecting "
                        "round has nothing to undo without one");
    }
    if (ds4_qwen4exp_rollback_check(rollback, err, errlen) != 0) return -1;
    st->rollback = rollback;
    if (depth < 0 || depth > DS4_QWEN4EXP_IMPLEMENTED_DEPTH) {
        return mtp_fail(err, errlen,
                        "qwen4exp MTP: draft depth %d is not implemented "
                        "(DS4_QWEN4EXP_IMPLEMENTED_DEPTH=%d)",
                        depth, DS4_QWEN4EXP_IMPLEMENTED_DEPTH);
    }
    if (hc_dim == 0 || n_vocab == 0) {
        return mtp_fail(err, errlen,
                        "qwen4exp MTP: hc_dim %u and n_vocab %u must both be "
                        "positive", hc_dim, n_vocab);
    }
    st->depth = depth;
    st->hc_dim = hc_dim;
    st->n_vocab = n_vocab;
    ds4_qwen4exp_mtp_invalidate(st);
    st->hc_scratch = malloc((size_t)DS4_QWEN4EXP_MTP_HC_ROWS * hc_dim *
                            sizeof(float));
    st->logits_rows = malloc((size_t)DS4_QWEN4EXP_MTP_MAX_COMMIT *
                             (size_t)n_vocab * sizeof(float));
    if (!st->hc_scratch || !st->logits_rows) {
        ds4_qwen4exp_mtp_state_free(st);
        return mtp_fail(err, errlen, "qwen4exp MTP: out of memory");
    }
    return 0;
}

void ds4_qwen4exp_mtp_state_free(ds4_qwen4exp_mtp_state *st) {
    if (!st) return;
    free(st->hc_scratch);
    free(st->logits_rows);
    st->hc_scratch = NULL;
    st->logits_rows = NULL;
}

void ds4_qwen4exp_mtp_invalidate(ds4_qwen4exp_mtp_state *st) {
    for (int k = 0; k < DS4_QWEN4EXP_IMPLEMENTED_DEPTH; k++) st->pending[k] = -1;
    st->n_pending = 0;
    st->pending_parent = -1;
    st->frontier_top1_valid = false;
    st->frontier_logits_deferred = false;
}

int ds4_qwen4exp_mtp_counters_check(const ds4_qwen4exp_mtp_counters *c,
                                    char *err, size_t errlen) {
    uint64_t rounds = 0, committed = 0;
    for (int i = 1; i <= DS4_QWEN4EXP_MTP_MAX_COMMIT; i++) {
        rounds += c->commit_hist[i];
        committed += c->commit_hist[i] * (uint64_t)i;
    }
    if (c->commit_hist[0] != 0) {
        return mtp_fail(err, errlen,
                        "qwen4exp MTP counters: %llu rounds committed nothing",
                        (unsigned long long)c->commit_hist[0]);
    }
    if (rounds != c->rounds) {
        return mtp_fail(err, errlen,
                        "qwen4exp MTP counters: the commit histogram covers "
                        "%llu rounds but %llu ran",
                        (unsigned long long)rounds,
                        (unsigned long long)c->rounds);
    }
    if (committed != c->committed) {
        return mtp_fail(err, errlen,
                        "qwen4exp MTP counters: acceptance lengths sum to %llu "
                        "but %llu tokens were committed",
                        (unsigned long long)committed,
                        (unsigned long long)c->committed);
    }
    /* A round offers at most one chain, so the drafts it can carry are bounded
     * by the deepest chain this build allows -- not by 1, which was the same
     * number only while depth 1 was the only depth. */
    if (c->drafted > c->rounds * (uint64_t)DS4_QWEN4EXP_IMPLEMENTED_DEPTH) {
        return mtp_fail(err, errlen,
                        "qwen4exp MTP counters: %llu draft tokens over %llu "
                        "rounds, which cannot carry more than %d each",
                        (unsigned long long)c->drafted,
                        (unsigned long long)c->rounds,
                        DS4_QWEN4EXP_IMPLEMENTED_DEPTH);
    }
    if (c->accepted > c->drafted) {
        return mtp_fail(err, errlen,
                        "qwen4exp MTP counters: %llu drafts accepted of %llu "
                        "proposed", (unsigned long long)c->accepted,
                        (unsigned long long)c->drafted);
    }
    /* Only a REJECTING round can disagree, and there are drafted - accepted
     * of those.  Checked after the accepted <= drafted test above, so the
     * subtraction cannot wrap. */
    if (c->verify_replay_disagreements > c->drafted - c->accepted) {
        return mtp_fail(err, errlen,
                        "qwen4exp MTP counters: %llu verify/replay "
                        "disagreements over %llu rejecting rounds",
                        (unsigned long long)c->verify_replay_disagreements,
                        (unsigned long long)(c->drafted - c->accepted));
    }
    /* Every round commits the fed token, and every accepted draft adds exactly
     * one more.  At depth 1 this was "accepted == commit_hist[2]"; the sum
     * form is the same statement at any depth. */
    if (c->committed != c->rounds + c->accepted) {
        return mtp_fail(err, errlen,
                        "qwen4exp MTP counters: %llu rounds and %llu accepted "
                        "drafts should commit %llu tokens, not %llu",
                        (unsigned long long)c->rounds,
                        (unsigned long long)c->accepted,
                        (unsigned long long)(c->rounds + c->accepted),
                        (unsigned long long)c->committed);
    }
    if (c->quenches != 0) {
        return mtp_fail(err, errlen,
                        "qwen4exp MTP counters: %llu quench events on a path "
                        "that has no yield guard",
                        (unsigned long long)c->quenches);
    }
    return 0;
}

/* ------------------------------------------------------------------------
 * The cycle
 * ------------------------------------------------------------------------ */

/*
 * Drop the head cache rows above `pos`.
 *
 * At depth 2 and up a chain re-enters the head on its own output, so the rows
 * it wrote above the frontier came from the HEAD's hidden state, not the
 * target's.  They are speculation and they go before the committed rows are
 * seeded -- on the accepting path as much as the rejecting one, because a
 * round that accepted k drafts still leaves rows above pos + k that nothing
 * confirmed.  At depth 1 the chain never leaves the frontier and this is a
 * no-op, which is why the depth-1 cycle never needed it.
 */
static int mtp_head_cache_truncate(const ds4_qwen4exp_rollback_set *set,
                                   uint32_t pos, char *err, size_t errlen) {
    const ds4_qwen4exp_rollback_object *o =
        &set->obj[DS4_QWEN4EXP_STATE_MTP_HEAD_CACHE];
    if (o->truncate && o->truncate(o->ctx, pos) != 0) {
        return mtp_fail(err, errlen,
                        "qwen4exp MTP rollback: truncate of %s to position %u "
                        "failed",
                        ds4_qwen4exp_state_name(
                                DS4_QWEN4EXP_STATE_MTP_HEAD_CACHE), pos);
    }
    return 0;
}

/*
 * Draft the next chain, keeping the head cache rows the round confirmed.
 *
 * The head consumes (row at p, token at p + 1) and owns cache row p, so the
 * chain that ran LAST round already wrote a row per step: step k took the
 * token at pos + k and wrote row pos - 1 + k.  Those rows came from the head's
 * OWN multi stream rather than the target's, and they are the rows this round
 * keeps -- one per committed position -- instead of dropping them and feeding
 * every accepted token back through the head.  A round that accepted `n`
 * drafts therefore starts its chain at row pos + n with the rows below it
 * already written, and the truncate is what removes the steps above `n`, whose
 * tokens the target did not take.
 *
 * ONE row can still be missing, and only on a round that accepted its WHOLE
 * chain: the verify carries the fed token AND every draft, so a full accept
 * commits one token past the last row the chain wrote -- the bonus the extra
 * verify row yielded -- and nothing has fed that token to the head.  Its row
 * is seeded here, from the target row at pos + n - 1 and the token at pos + n,
 * which is the same call the old per-token seed made and the only one left.
 * `st->head_rows` is what says whether it is owed; below `pos` the rows are
 * the prefill's and this never claims to have written them.
 *
 * Every later chain step reads the previous step's `multi` row in place of a
 * target row it does not have.  Those rows land above the frontier and are
 * exactly what the next round's truncate removes.
 */
static int mtp_draft_chain(ds4_qwen4exp_mtp_state *st,
                           const ds4_qwen4exp_mtp_model *model,
                           const float *hc_rows, const int *toks,
                           int n, uint32_t pos, int next_fed,
                           char *err, size_t errlen) {
    const uint64_t t0 = mtp_now_ns();
    ds4_qwen4exp_mtp_invalidate(st);
    /* The chain's first row: the head has to hold every row below it. */
    const uint32_t start = pos + (uint32_t)n;
    if (mtp_head_cache_truncate(st->rollback, start, err, errlen) != 0) {
        return -1;
    }
    if (st->head_rows > start) st->head_rows = start;
    if (st->depth < 1) {
        st->counters.draft_ns += mtp_now_ns() - t0;
        return 0;
    }

    const uint32_t j0 = st->head_rows < pos ? pos : st->head_rows;
    float *const ping = st->hc_scratch +
                        (size_t)DS4_QWEN4EXP_MTP_MAX_COMMIT * st->hc_dim;
    const float *cur_hc = hc_rows + (size_t)n * st->hc_dim;
    int cur_tok = next_fed;
    uint32_t p = pos + (uint32_t)n;
    int k = 0;

    if (model->draft_rows) {
        /*
         * The seed rows and chain step 0 in ONE head forward.  Rows j0 .. start
         * take the tokens toks[j0 - pos + 1 .. n] and then next_fed, over the
         * hc rows j0 - pos .. n -- which sit side by side in hc_rows, so the
         * head reads them as one slab.  That is at most n + 1 <= depth + 1
         * rows, the width the head was built for.  The rows are the same rows
         * the per-row loop below would write, in the same order, from the
         * same inputs; only the launches, the synchronising readbacks and the
         * seed rows' unread argmaxes are gone.
         */
        const uint32_t k0 = j0 - pos;
        const uint32_t seeds = start - j0;
        int rows_tok[DS4_QWEN4EXP_MTP_MAX_COMMIT];
        for (uint32_t i = 0; i < seeds; i++) rows_tok[i] = toks[k0 + i + 1u];
        rows_tok[seeds] = next_fed;
        float *multi_out = (1 < st->depth) ? ping : NULL;
        int draft = -1;
        if (model->draft_rows(model->ctx, rows_tok,
                              hc_rows + (size_t)k0 * st->hc_dim,
                              j0, seeds + 1u, &draft, multi_out) != 0) {
            return mtp_fail(err, errlen,
                            "qwen4exp MTP: %u-row head forward at position %u "
                            "failed", seeds + 1u, j0);
        }
        st->pending[0] = draft;
        st->n_pending = 1;
        cur_tok = draft;
        cur_hc = multi_out;
        p = start + 1u;
        st->head_rows = p;
        k = 1;
    } else {
        for (uint32_t j = j0; j < start; j++) {
            const uint32_t kk = j - pos;   /* 0 <= kk < n, so both reads are in */
            int discard = -1;
            if (model->draft_step(model->ctx, toks[kk + 1u],
                                  hc_rows + (size_t)kk * st->hc_dim,
                                  j, &discard, NULL) != 0) {
                return mtp_fail(err, errlen,
                                "qwen4exp MTP: head seed for token %d at "
                                "position %u failed", toks[kk + 1u], j);
            }
        }
        st->head_rows = start;
    }

    for (; k < st->depth; k++) {
        /* The last step's `multi` row would have no reader. */
        float *multi_out = (k + 1 < st->depth)
                         ? ping + (size_t)(k & 1) * st->hc_dim : NULL;
        int draft = -1;
        if (model->draft_step(model->ctx, cur_tok, cur_hc, p, &draft,
                              multi_out) != 0) {
            return mtp_fail(err, errlen,
                            "qwen4exp MTP: draft step %d for token %d at "
                            "position %u failed", k, cur_tok, p);
        }
        st->pending[k] = draft;
        st->n_pending = k + 1;
        cur_tok = draft;
        cur_hc = multi_out;
        p += 1u;
        st->head_rows = p;
    }
    st->pending_parent = next_fed;
    st->counters.draft_ns += mtp_now_ns() - t0;
    return 0;
}

/* Commit one token: the plain path, and the rejecting round's replay when
 * nothing was accepted.  Runs a one-row decode at `pos`, seeds the next chain
 * from it and returns 1. */
static int mtp_commit_one(ds4_qwen4exp_mtp_state *st,
                          const ds4_qwen4exp_mtp_model *model,
                          int first_token, uint32_t pos,
                          int *accepted, float *logits,
                          int *next_out, char *err, size_t errlen) {
    float *hc0 = st->hc_scratch;
    const uint64_t t0 = mtp_now_ns();
    const int drc = model->decode_token(model->ctx, first_token, pos, hc0,
                                        logits);
    st->counters.verify_ns += mtp_now_ns() - t0;
    if (drc != 0) {
        return mtp_fail(err, errlen,
                        "qwen4exp MTP: target decode of token %d at position "
                        "%u failed", first_token, pos);
    }
    const int next = ds4_qwen4exp_mtp_argmax(logits, model->n_vocab);
    if (next_out) *next_out = next;
    if (mtp_draft_chain(st, model, hc0, NULL, 0, pos, next,
                        err, errlen) != 0) {
        return -1;
    }
    accepted[0] = first_token;
    st->counters.committed += 1;
    st->counters.commit_hist[1] += 1;
    return 1;
}

int ds4_qwen4exp_mtp_cycle(ds4_qwen4exp_mtp_state *st,
                           const ds4_qwen4exp_mtp_model *model,
                           int first_token,
                           uint32_t pos, int budget,
                           int *accepted, int accepted_cap,
                           float *logits,
                           char *err, size_t errlen) {
    if (!st->hc_scratch || !st->logits_rows) {
        return mtp_fail(err, errlen, "qwen4exp MTP: state is not initialised");
    }
    /* `logits` is not optional.  Every path below takes an argmax over it, and
     * the caller's next sample reads it: a NULL here is a caller that has no
     * frontier distribution to sample from, so its leg would repeat one token
     * for ever.  Refuse by name -- the argmax would otherwise dereference it. */
    if (!logits) {
        return mtp_fail(err, errlen,
                        "qwen4exp MTP: the cycle was given no logit buffer to "
                        "leave the frontier distribution in");
    }
    if (accepted_cap < 1 || budget < 1) {
        return mtp_fail(err, errlen,
                        "qwen4exp MTP: budget %d and capacity %d leave no room "
                        "for the fed token", budget, accepted_cap);
    }
    if (model->hc_dim != st->hc_dim || model->n_vocab != st->n_vocab) {
        return mtp_fail(err, errlen,
                        "qwen4exp MTP: the model reports hc_dim %u vocab %u, "
                        "the state was built for hc_dim %u vocab %u",
                        model->hc_dim, model->n_vocab, st->hc_dim, st->n_vocab);
    }
    st->counters.rounds += 1;
    st->frontier_top1_valid = false;
    st->frontier_logits_deferred = false;

    /* A chain belongs to the token it was drafted from.  Anything else -- a
     * rewind, a different sampled token -- makes the whole chain stale, not
     * just its head: every link after the first was drafted on the assumption
     * that the one before it stood. */
    if (st->n_pending > 0 && first_token != st->pending_parent) {
        ds4_qwen4exp_mtp_invalidate(st);
    }
    /* How many of the carried drafts this round may use.  The caller's budget
     * and capacity shorten the chain rather than overrunning either. */
    int n = st->n_pending;
    if (n > budget - 1) n = budget - 1;
    if (n > accepted_cap - 1) n = accepted_cap - 1;
    if (n < 1 || st->depth < 1) {
        return mtp_commit_one(st, model, first_token, pos,
                              accepted, logits, NULL, err, errlen);
    }

    const ds4_qwen4exp_rollback_set *rollback = st->rollback;
    int toks[DS4_QWEN4EXP_MTP_MAX_COMMIT];
    _Static_assert(DS4_QWEN4EXP_MTP_MAX_COMMIT ==
                           DS4_QWEN4EXP_IMPLEMENTED_DEPTH + 1,
                   "the verify carries the fed token plus the whole chain");
    toks[0] = first_token;
    for (int k = 0; k < n; k++) toks[k + 1] = st->pending[k];
    ds4_qwen4exp_mtp_invalidate(st);

    /* No round-start snapshot.  The verify forward itself leaves the state
     * after each drafted row in a slot, so there is nothing to copy first and
     * nothing to rewind to afterwards. */
    float *const hc = st->hc_scratch;
    float *const row_logits = st->logits_rows;
    int row_top1[DS4_QWEN4EXP_MTP_MAX_COMMIT];
    const bool compact_logits =
        model->verify_rows_top1 != NULL && model->read_logit_row != NULL;
    st->counters.drafted += (uint64_t)n;
    const uint64_t verify_t0 = mtp_now_ns();
    const int vrc = compact_logits
        ? model->verify_rows_top1(model->ctx, toks, (uint32_t)n + 1u, pos,
                                  hc, row_top1)
        : model->verify_rows(model->ctx, toks, (uint32_t)n + 1u, pos,
                             hc, row_logits);
    st->counters.verify_ns += mtp_now_ns() - verify_t0;
    if (vrc != 0) {
        return mtp_fail(err, errlen,
                        "qwen4exp MTP: %d-row verify at position %u failed",
                        n + 1, pos);
    }

    /*
     * The longest matching prefix.  Row j's argmax is the target's own token
     * for position pos + j + 1, so draft j stands only if the target chose it
     * AND every draft before it stood -- which the break enforces, because a
     * row that follows a rejected token is conditioned on a prefix the stream
     * never had.
     *
     * The rows come out of the verify itself: one forward runs the final mixer
     * and the LM head over every row, so the loop is a scan over logits that
     * already exist rather than a head call per comparison.  Row n's argmax is
     * never compared: nothing drafted position pos + n + 1, and that argmax is
     * the bonus token the caller feeds next.
     */
    int a = 0;
    int first_mismatch = -1;
    while (a < n) {
        const int chosen = compact_logits
            ? row_top1[a]
            : ds4_qwen4exp_mtp_argmax(
                    row_logits + (size_t)a * st->n_vocab, st->n_vocab);
        if (chosen != toks[a + 1]) { first_mismatch = chosen; break; }
        a++;
    }
    st->counters.accepted += (uint64_t)a;

    /* The target head has already produced every row.  A greedy-only compact
     * seam can carry the exact GPU winner forward and leave the full selected
     * distribution resident until an API actually asks for it. */
    const int compact_frontier_top1 = compact_logits
        ? (a == n ? row_top1[n] : first_mismatch) : -1;
    const bool defer_frontier =
        compact_logits && model->defer_frontier_logits &&
        compact_frontier_top1 >= 0 &&
        (uint32_t)compact_frontier_top1 < st->n_vocab;
    if (compact_logits) {
        if (!defer_frontier &&
            model->read_logit_row(model->ctx, (uint32_t)a, logits) != 0) {
            return mtp_fail(err, errlen,
                            "qwen4exp MTP: target logit row %d read failed", a);
        }
    } else {
        memcpy(logits, row_logits + (size_t)a * st->n_vocab,
               (size_t)st->n_vocab * sizeof(float));
    }

    if (a == n) {
        /* Every draft is the target's own greedy argmax, so the whole run is
         * exactly what the serial leg would have produced and every verified
         * row is a committed row.  Nothing to roll back, and `logits` already
         * holds the frontier row's distribution. */
        for (int k = 0; k <= n; k++) accepted[k] = toks[k];
        st->counters.committed += (uint64_t)(n + 1);
        st->counters.commit_hist[n + 1] += 1;
        const int next_fed = compact_logits
            ? compact_frontier_top1
            : ds4_qwen4exp_mtp_argmax(logits, st->n_vocab);
        if (mtp_draft_chain(st, model, hc, toks, n, pos, next_fed,
                            err, errlen) != 0) {
            return -1;
        }
        if (compact_logits) {
            st->frontier_row = (uint32_t)a;
            st->frontier_top1 = compact_frontier_top1;
            st->frontier_top1_valid = compact_frontier_top1 >= 0 &&
                (uint32_t)compact_frontier_top1 < st->n_vocab;
            st->frontier_logits_deferred = defer_frontier;
        }
        return n + 1;
    }

    /*
     * Reject at row `a`.  Rows a + 1 .. n carry a continuation the target did
     * not choose, so the round keeps only what it accepted.
     *
     * Nothing is rewound and nothing is replayed.  The verify forward mirrored
     * the running state after each drafted row into a slot, so every
     * SELECT_ROW object adopts slot `a` and every TRUNCATE object drops its
     * tail at the accepted length.  That is exact rather than close: the
     * recurrence is token-serial, so the state the forward left after row `a`
     * IS the state an (a + 1)-row feed leaves, which is what the shorter
     * forward used to recompute at the cost of a whole extra pass.
     */
    const uint64_t rb_t0 = mtp_now_ns();
    const int srrc = rollback_select_row(rollback, (uint32_t)a, err, errlen);
    const int trrc = srrc == 0
        ? rollback_truncate_all(rollback, pos + (uint32_t)a + 1u, err, errlen)
        : 0;
    st->counters.rollback_ns += mtp_now_ns() - rb_t0;
    if (srrc != 0 || trrc != 0) return -1;

    /* The frontier distribution is ROW a's, which the verify already left in
     * the row block.  No head call and no forward: the token the caller feeds
     * next is the first one the target chose that the chain had not
     * drafted. */
    for (int k = 0; k <= a; k++) accepted[k] = toks[k];
    st->counters.committed += (uint64_t)(a + 1);
    st->counters.commit_hist[a + 1] += 1;
    if (mtp_draft_chain(st, model, hc, toks, a, pos, first_mismatch,
                        err, errlen) != 0) {
        return -1;
    }
    if (compact_logits) {
        st->frontier_row = (uint32_t)a;
        st->frontier_top1 = compact_frontier_top1;
        st->frontier_top1_valid = compact_frontier_top1 >= 0 &&
            (uint32_t)compact_frontier_top1 < st->n_vocab;
        st->frontier_logits_deferred = defer_frontier;
    }
    return a + 1;
}

/* ------------------------------------------------------------------------
 * The head
 * ------------------------------------------------------------------------ */

#ifndef DS4_NO_GPU

static ds4_gpu_tensor *mtp_alloc(uint64_t bytes, bool *ok) {
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(bytes);
    if (!t) *ok = false;
    return t;
}

/* One unsigned decimal environment variable.  Unset or empty keeps the
 * fallback; anything that is not exactly a decimal count in uint32 range is a
 * named refusal, because a truncated parse would silently arm the wrong
 * shortlist rather than fail. */
static int mtp_env_u32(const char *name, uint32_t fallback, uint32_t *out,
                       char *err, size_t errlen) {
    *out = fallback;
    const char *raw = getenv(name);
    if (!raw || !raw[0]) return 0;
    char *end = NULL;
    errno = 0;
    const unsigned long v = strtoul(raw, &end, 10);
    if (errno != 0 || end == raw || !end || *end != '\0' ||
        v > 0xFFFFFFFFul) {
        return mtp_fail(err, errlen,
                        "qwen4exp MTP head: %s=\"%s\" is not an unsigned "
                        "decimal count below 2^32", name, raw);
    }
    *out = (uint32_t)v;
    return 0;
}

/*
 * Read the draft shortlist ONCE, at head init, and validate it against the
 * vocabulary.  A prefix of 0 (the default) is OFF and the draft runs over the
 * whole vocabulary; the tail is inert without a prefix and is not validated
 * then, because a setting that arms nothing cannot mis-launch anything.
 *
 * The two ranges must be disjoint and inside the table: prefix first, then
 * [n_vocab - tail, n_vocab).  That is what makes the packed argmax order the
 * token-id order -- the top-1's first-max tie rule keeps picking the lowest id
 * -- and what keeps every packed position a distinct id.
 */
static int mtp_head_draft_vocab(ds4_qwen4exp_mtp_head *h,
                                char *err, size_t errlen) {
    uint32_t prefix = 0, tail = 0;
    if (mtp_env_u32("DS4_QWEN4EXP_DRAFT_VOCAB_PREFIX", DS4_QWEN4EXP_DRAFT_VOCAB_PREFIX_DEFAULT, &prefix,
                    err, errlen) != 0 ||
        mtp_env_u32("DS4_QWEN4EXP_DRAFT_VOCAB_TAIL",
                    DS4_QWEN4EXP_DRAFT_VOCAB_TAIL_DEFAULT, &tail,
                    err, errlen) != 0) {
        return -1;
    }
    /* The built-in default is sized for the production vocabulary; on a
     * smaller table (the reduced test artifacts) it falls back to the whole
     * vocabulary instead of refusing.  An explicit setting is validated. */
    if (getenv("DS4_QWEN4EXP_DRAFT_VOCAB_PREFIX") == NULL &&
        getenv("DS4_QWEN4EXP_DRAFT_VOCAB_TAIL") == NULL &&
        (uint64_t)prefix + (uint64_t)tail > (uint64_t)h->n_vocab) {
        prefix = 0u;
    }
    if (prefix == 0u) {
        h->draft_vocab_prefix = 0u;
        h->draft_vocab_tail = 0u;
        return 0;
    }
    if (prefix > h->n_vocab) {
        return mtp_fail(err, errlen,
                        "qwen4exp MTP head: DS4_QWEN4EXP_DRAFT_VOCAB_PREFIX=%u "
                        "exceeds the vocabulary %u",
                        prefix, h->n_vocab);
    }
    if (tail > h->n_vocab) {
        return mtp_fail(err, errlen,
                        "qwen4exp MTP head: DS4_QWEN4EXP_DRAFT_VOCAB_TAIL=%u "
                        "exceeds the vocabulary %u",
                        tail, h->n_vocab);
    }
    if ((uint64_t)prefix + (uint64_t)tail > (uint64_t)h->n_vocab) {
        return mtp_fail(err, errlen,
                        "qwen4exp MTP head: DS4_QWEN4EXP_DRAFT_VOCAB_PREFIX=%u "
                        "plus DS4_QWEN4EXP_DRAFT_VOCAB_TAIL=%u exceeds the "
                        "vocabulary %u; the prefix range and the added-token "
                        "range at the top of the table would overlap",
                        prefix, tail, h->n_vocab);
    }
    h->draft_vocab_prefix = prefix;
    h->draft_vocab_tail = tail;
    return 0;
}

int ds4_qwen4exp_mtp_head_init(ds4_qwen4exp_mtp_head *h,
                               char *err, size_t errlen) {
    if (!h->hooks.rms_norm || !h->hooks.hc_mixer || !h->hooks.embed ||
        !h->hooks.matmul_q8_0 || !h->hooks.block) {
        return mtp_fail(err, errlen,
                        "qwen4exp MTP head: rms_norm, hc_mixer, embed, "
                        "matmul_q8_0 and block must all be bound");
    }
    if (!h->head_map || !h->target_map) {
        return mtp_fail(err, errlen,
                        "qwen4exp MTP head: the head mapping carries the nextn "
                        "tensors and the target mapping the borrowed "
                        "token_embd and output; both are needed");
    }
    if (h->eh_proj_in_dim != 2u * h->n_embd) {
        return mtp_fail(err, errlen,
                        "qwen4exp MTP head: eh_proj takes %u inputs, this head "
                        "needs %u -- the embedding half occupies rows [0, %u) "
                        "and the hidden half the rest, so a different width is "
                        "a different head",
                        h->eh_proj_in_dim, 2u * h->n_embd, h->n_embd);
    }
    if (h->n_embd == 0 || h->n_hc == 0 || h->n_lowrank == 0 ||
        h->n_vocab == 0 || h->max_tokens == 0) {
        return mtp_fail(err, errlen,
                        "qwen4exp MTP head: n_embd %u, n_hc %u, n_lowrank %u, "
                        "n_vocab %u and max_tokens %u must all be positive",
                        h->n_embd, h->n_hc, h->n_lowrank, h->n_vocab,
                        h->max_tokens);
    }
    if (mtp_head_draft_vocab(h, err, errlen) != 0) return -1;
    const uint64_t rows = h->max_tokens;
    const uint64_t n_embd = h->n_embd;
    const uint64_t hc_dim = (uint64_t)h->n_hc * n_embd;
    const uint64_t f = sizeof(float);
    bool ok = true;

    h->t_tokens       = mtp_alloc(rows * sizeof(int32_t), &ok);
    h->t_embed_rows   = mtp_alloc(rows * n_embd * f, &ok);
    h->t_embed_out    = mtp_alloc(rows * n_embd * f, &ok);
    h->t_e_normed     = mtp_alloc(rows * n_embd * f, &ok);
    h->t_h_normed     = mtp_alloc(rows * hc_dim * f, &ok);
    h->t_ehx          = mtp_alloc(rows * h->n_hc * 2ull * n_embd * f, &ok);
    h->t_hyper        = mtp_alloc(rows * hc_dim * f, &ok);
    h->t_mix_normed   = mtp_alloc(rows * hc_dim * f, &ok);
    h->t_mix_lowrank  = mtp_alloc(rows * h->n_lowrank * f, &ok);
    h->t_mix_wide     = mtp_alloc(rows * hc_dim * f, &ok);
    h->t_sample       = mtp_alloc(rows * n_embd * f, &ok);
    h->t_logits       = mtp_alloc(rows * h->n_vocab * f, &ok);
    /* Shortlist staging, sized by the armed setting alone.  Off mode (the
     * default) allocates neither, and a prefix with no tail needs none
     * either -- its one range lands straight in t_logits -- so the
     * full-vocabulary path keeps exactly the allocation profile it has
     * always had. */
    if (h->draft_vocab_prefix && h->draft_vocab_tail) {
        h->t_logits_prefix = mtp_alloc(rows * h->draft_vocab_prefix * f, &ok);
        if (ok) {
            h->t_logits_tail = mtp_alloc(rows * h->draft_vocab_tail * f, &ok);
        }
    }
    h->t_top1         = mtp_alloc(rows * sizeof(uint32_t), &ok);
    h->top1_host      = malloc((size_t)rows * sizeof(uint32_t));
    if (!ok || !h->top1_host) {
        ds4_qwen4exp_mtp_head_free(h);
        return mtp_fail(err, errlen,
                        "qwen4exp MTP head: scratch allocation failed for %u "
                        "rows", h->max_tokens);
    }
    return 0;
}

void ds4_qwen4exp_mtp_head_free(ds4_qwen4exp_mtp_head *h) {
    if (!h) return;
    ds4_gpu_tensor *all[] = {
        h->t_tokens, h->t_embed_rows, h->t_embed_out, h->t_e_normed,
        h->t_h_normed, h->t_ehx, h->t_hyper, h->t_mix_normed,
        h->t_mix_lowrank, h->t_mix_wide, h->t_sample, h->t_logits,
        h->t_logits_prefix, h->t_logits_tail, h->t_top1,
    };
    for (size_t i = 0; i < sizeof(all) / sizeof(all[0]); i++) {
        ds4_gpu_tensor_free(all[i]);
    }
    h->t_tokens = h->t_embed_rows = h->t_embed_out = h->t_e_normed = NULL;
    h->t_h_normed = h->t_ehx = h->t_hyper = h->t_mix_normed = NULL;
    h->t_mix_lowrank = h->t_mix_wide = h->t_sample = h->t_logits = NULL;
    h->t_logits_prefix = h->t_logits_tail = NULL;
    h->t_top1 = NULL;
    free(h->top1_host);
    h->top1_host = NULL;
}

/*
 * PER-STAGE TIMING FOR ONE HEAD STEP.  Off unless DS4_MTP_HEAD_TIME is set.
 *
 * A head step is a chain of asynchronous launches closed by one synchronising
 * readback, so a host clock around an individual stage reads the launch and not
 * the work.  With the variable set, every stage is followed by a device
 * synchronise and the split is accumulated; the totals print once at exit.
 *
 * The synchronises make the step SLOWER, so this attributes time and never
 * measures it: the depth table's draft milliseconds are the measurement.
 */
enum {
    MTP_HEAD_T_TOKEN = 0,
    MTP_HEAD_T_MULTI_IN,
    MTP_HEAD_T_EMBED,
    MTP_HEAD_T_ENORM,
    MTP_HEAD_T_HNORM,
    MTP_HEAD_T_EHX,
    MTP_HEAD_T_EH_PROJ,
    MTP_HEAD_T_BLOCK,
    MTP_HEAD_T_MIXER,
    MTP_HEAD_T_LM_HEAD,
    MTP_HEAD_T_TOP1,
    MTP_HEAD_T_END,
    MTP_HEAD_T_TOP1_IN,
    MTP_HEAD_T_LOGIT0_IN,
    MTP_HEAD_T_MULTI_OUT,
    MTP_HEAD_T_N
};

static const char *const mtp_head_stage_names[MTP_HEAD_T_N] = {
    "token upload", "multi upload", "embed", "enorm", "hnorm",
    "ehx copies", "eh_proj", "block", "hc mixer", "lm head",
    "gpu top-1", "end commands", "top-1 readback", "logit-0 readback",
    "multi readback"
};

static uint64_t mtp_head_stage_ns[MTP_HEAD_T_N];
static uint64_t mtp_head_stage_calls;
static int      mtp_head_timing = -1;

static void mtp_head_time_dump(void) {
    if (mtp_head_stage_calls == 0) return;
    const double n = (double)mtp_head_stage_calls;
    uint64_t total = 0;
    for (int i = 0; i < MTP_HEAD_T_N; i++) total += mtp_head_stage_ns[i];
    fprintf(stderr, "ds4: MTP head stage split over %llu steps, ms each\n",
            (unsigned long long)mtp_head_stage_calls);
    for (int i = 0; i < MTP_HEAD_T_N; i++) {
        fprintf(stderr, "ds4:   %-15s %8.3f  %5.1f%%\n",
                mtp_head_stage_names[i],
                (double)mtp_head_stage_ns[i] / n / 1e6,
                total ? 100.0 * (double)mtp_head_stage_ns[i] / (double)total
                      : 0.0);
    }
    fprintf(stderr, "ds4:   %-15s %8.3f\n", "TOTAL",
            (double)total / n / 1e6);
}

void ds4_qwen4exp_mtp_head_time_snapshot(uint64_t *ns, size_t cap,
                                         uint64_t *calls) {
    _Static_assert(MTP_HEAD_T_N == DS4_QWEN4EXP_MTP_HEAD_STAGES,
                   "the head stage count is published in the header");
    for (size_t i = 0; ns && i < cap; i++) {
        ns[i] = i < (size_t)MTP_HEAD_T_N ? mtp_head_stage_ns[i] : 0u;
    }
    if (calls) *calls = mtp_head_stage_calls;
}

static int mtp_head_time_on(void) {
    if (mtp_head_timing < 0) {
        const char *v = getenv("DS4_MTP_HEAD_TIME");
        mtp_head_timing = (v && *v && strcmp(v, "0") != 0) ? 1 : 0;
        if (mtp_head_timing) atexit(mtp_head_time_dump);
    }
    return mtp_head_timing;
}

#define MTP_HEAD_TICK(slot)                                                   \
    do {                                                                      \
        if (timing) {                                                         \
            (void)ds4_gpu_synchronize();                                      \
            const uint64_t mtp_head_now = mtp_now_ns();                       \
            mtp_head_stage_ns[(slot)] += mtp_head_now - tmark;                 \
            tmark = mtp_head_now;                                             \
        }                                                                     \
    } while (0)

/* The forward proper.  Seed rows must update the head block's caches, but
 * their final mixer and vocabulary projections have no consumer when only
 * the last proposal is requested.  Narrow those stateless operations within
 * the decode-order envelope; wider diagnostic calls retain their dispatch. */
static int mtp_head_forward_impl(ds4_qwen4exp_mtp_head *h,
                                 const int *next_tokens,
                                 const float *multi_in,
                                 uint32_t pos0, uint32_t n_tokens,
                                 int *draft_out, float *multi_out,
                                 bool last_only,
                                 char *err, size_t errlen) {
    if (n_tokens == 0 || n_tokens > h->max_tokens) {
        return mtp_fail(err, errlen,
                        "qwen4exp MTP head: %u rows, built for 1..%u",
                        n_tokens, h->max_tokens);
    }
    const uint32_t n_embd = h->n_embd;
    const uint32_t n_hc = h->n_hc;
    const uint64_t hc_dim = (uint64_t)n_hc * n_embd;
    const uint64_t f = sizeof(float);
    const uint64_t embd_bytes = (uint64_t)n_embd * f;
    const uint32_t first_row = last_only ? n_tokens - 1u : 0u;
    const uint32_t out_rows = n_tokens - first_row;
    const bool narrow_logits = last_only && n_tokens > 1u &&
        n_tokens <= (uint32_t)DS4_QWEN4EXP_MTP_MAX_COMMIT;
    const uint32_t logit_rows = narrow_logits ? 1u : n_tokens;
    const uint32_t logit_first = narrow_logits ? 0u : first_row;
    /* The draft shortlist, fixed at init.  Zero keeps the whole vocabulary;
     * armed, the DRAFT's borrowed-LM-head projection narrows to rows
     * [0, prefix) plus the tail range, and `draft_width` is the width of one
     * PACKED row: the prefix ids first, the tail ids behind them.  Packing
     * prefix-first keeps the packed order the token-id order -- validated at
     * init (the ranges are disjoint and prefix is below the tail base) -- so
     * the top-1's first-max tie rule still picks the lowest id. */
    const uint32_t draft_prefix = h->draft_vocab_prefix;
    const uint32_t draft_tail = draft_prefix ? h->draft_vocab_tail : 0u;
    const uint32_t draft_width = draft_prefix
        ? draft_prefix + draft_tail : h->n_vocab;
    const int timing = mtp_head_time_on();
    uint64_t tmark = timing ? mtp_now_ns() : 0;

    /* The ids the embedding gather reads.  int is the caller's type; the
     * kernel takes int32, and the two agree on every target this builds for. */
    int32_t ids_stack[8];
    int32_t *ids = ids_stack;
    if (n_tokens > sizeof(ids_stack) / sizeof(ids_stack[0])) {
        ids = malloc((size_t)n_tokens * sizeof(int32_t));
        if (!ids) return mtp_fail(err, errlen, "qwen4exp MTP head: out of memory");
    }
    for (uint32_t t = 0; t < n_tokens; t++) ids[t] = (int32_t)next_tokens[t];

    const char *stage = "token upload";
    bool ok = ds4_gpu_tensor_write(h->t_tokens, 0, ids,
                                   (uint64_t)n_tokens * sizeof(int32_t)) != 0;
    if (ids != ids_stack) free(ids);
    MTP_HEAD_TICK(MTP_HEAD_T_TOKEN);
    if (ok) {
        stage = "multi-stream upload";
        ok = ds4_gpu_tensor_write(h->t_hyper, 0, multi_in,
                                  (uint64_t)n_tokens * hc_dim * f) != 0;
    }
    MTP_HEAD_TICK(MTP_HEAD_T_MULTI_IN);
    if (ok) ok = ds4_gpu_begin_commands() != 0;

    /* e = fc_embedding(enorm(embed(next))).  The embedding is the TARGET's;
     * n_hc = 1 asks the tiling gather for plain rows. */
    if (ok) {
        stage = "embedding";
        ok = h->hooks.embed(h->t_embed_out, h->t_embed_rows, h->t_tokens,
                            h->target_map, h->target_size,
                            h->token_embd_offset, h->token_embd_type,
                            h->n_vocab, n_tokens, n_embd, 1u) != 0;
    }
    MTP_HEAD_TICK(MTP_HEAD_T_EMBED);
    if (ok) {
        stage = "enorm";
        ok = h->hooks.rms_norm(h->t_e_normed, h->t_embed_out,
                               h->head_map, h->head_size, h->enorm_offset,
                               n_embd, n_embd, n_tokens,
                               h->rms_eps, h->weight_bias, h->round_bf16) != 0;
    }
    MTP_HEAD_TICK(MTP_HEAD_T_ENORM);
    /* h = fc_hidden(hnorm(multi)).  hnorm is UNGROUPED: one statistic over the
     * whole n_hc * n_embd row, unlike every hyper-connection norm. */
    if (ok) {
        stage = "hnorm";
        ok = h->hooks.rms_norm(h->t_h_normed, h->t_hyper,
                               h->head_map, h->head_size, h->hnorm_offset,
                               (uint32_t)hc_dim, (uint32_t)hc_dim, n_tokens,
                               h->rms_eps, h->weight_bias, h->round_bf16) != 0;
    }
    MTP_HEAD_TICK(MTP_HEAD_T_HNORM);
    /* One matmul does the broadcast-and-add: eh_proj is [fc_embedding;
     * fc_hidden] stacked, so row (t, s) = [e_normed(t) | h_normed(t, s)]
     * against it yields fc_embedding(e) + fc_hidden(h_s) for every stream. */
    if (ok) {
        stage = "eh_proj rows";
        if (h->hooks.ehx_pack) {
            ok = h->hooks.ehx_pack(h->t_ehx, h->t_e_normed, h->t_h_normed,
                                   n_tokens, n_hc, n_embd) != 0;
        } else {
            for (uint32_t t = 0; ok && t < n_tokens; t++) {
                for (uint32_t s = 0; ok && s < n_hc; s++) {
                    const uint64_t dst =
                        ((uint64_t)t * n_hc + s) * 2ull * embd_bytes;
                    ok = ds4_gpu_tensor_copy(
                             h->t_ehx, dst, h->t_e_normed,
                             (uint64_t)t * embd_bytes, embd_bytes) != 0 &&
                         ds4_gpu_tensor_copy(
                             h->t_ehx, dst + embd_bytes, h->t_h_normed,
                             ((uint64_t)t * hc_dim + (uint64_t)s * n_embd) * f,
                             embd_bytes) != 0;
                }
            }
        }
    }
    MTP_HEAD_TICK(MTP_HEAD_T_EHX);
    if (ok) {
        stage = "eh_proj";
        ok = h->hooks.matmul_q8_0(h->t_hyper, h->head_map, h->head_size,
                                  h->eh_proj_offset, 2ull * n_embd, n_embd,
                                  h->t_ehx, (uint64_t)n_tokens * n_hc) != 0;
    }
    MTP_HEAD_TICK(MTP_HEAD_T_EH_PROJ);
    if (ok) {
        stage = "block";
        ok = h->hooks.block(h->graph, h->cache, h->t_hyper, h->block_index,
                            pos0, n_tokens) != 0;
    }
    MTP_HEAD_TICK(MTP_HEAD_T_BLOCK);
    /* t_h_normed's previous contents were consumed by eh_proj.  Reuse it for
     * the final hyper row so the stateless tail needs neither tensor views
     * nor an extra allocation.  Keep t_hyper intact for multi_out. */
    if (ok && narrow_logits) {
        stage = "last head row";
        ok = ds4_gpu_tensor_copy(h->t_h_normed, 0, h->t_hyper,
                                  (uint64_t)first_row * hc_dim * f,
                                  hc_dim * f) != 0;
    }
    /* The head's own mixer: a gated residual with no inject head, the same
     * shape as the tower's final mixer. */
    if (ok) {
        stage = "hc head mixer";
        /* All three come from the head GGUF, which the --mtp path opens as one
         * file, so the three slabs name one mapping here.  They are still
         * built per tensor: the mixer takes slabs because a tower mixer's
         * weights can straddle shards, and the head must not be the one place
         * that reintroduces a single-mapping assumption. */
        const ds4_gpu_qwen4exp_slab norm_slab = {
            h->head_map, h->head_size, h->hc_head_norm_offset, 0, 0, 0 };
        const ds4_gpu_qwen4exp_slab down_slab = {
            h->head_map, h->head_size, h->hc_head_down_offset, 0, 0, 0 };
        const ds4_gpu_qwen4exp_slab up_slab = {
            h->head_map, h->head_size, h->hc_head_up_offset, 0, 0, 0 };
        ok = h->hooks.hc_mixer(h->t_sample, NULL, h->t_mix_normed,
                               h->t_mix_lowrank, h->t_mix_wide,
                               narrow_logits ? h->t_h_normed : h->t_hyper,
                               &norm_slab, &down_slab, &up_slab, NULL,
                               n_embd, n_hc, h->n_lowrank, logit_rows,
                               h->rms_eps, h->weight_bias, h->round_bf16) != 0;
    }
    MTP_HEAD_TICK(MTP_HEAD_T_MIXER);
    /* The borrowed LM head, in the target's mapping.  The shortlist runs the
     * SAME kernel over two row ranges of output.weight: Q8_0 rows are
     * ds4_qwen4exp_q8_0_row_bytes() apart in the mapping, so a range is the
     * same weight_offset shifted past the rows it skips, and the weight is
     * read where it lies -- never copied, gathered or re-represented.  A
     * range's rows get the same per-element reduction the full call gives
     * them (the decode-order ladder is row-exact by construction, which is
     * the property the whole speculative cycle stands on), so a shortlist
     * id's logit is the logit the full projection produces. */
    if (ok) {
        stage = "borrowed lm head";
        ok = h->hooks.matmul_q8_0(
                draft_tail ? h->t_logits_prefix : h->t_logits,
                h->target_map, h->target_size, h->output_offset, n_embd,
                draft_prefix ? draft_prefix : h->n_vocab,
                h->t_sample, logit_rows) != 0;
        if (ok && draft_tail) {
            stage = "borrowed lm head tail";
            ok = h->hooks.matmul_q8_0(
                    h->t_logits_tail, h->target_map, h->target_size,
                    h->output_offset + (uint64_t)(h->n_vocab - draft_tail) *
                        ds4_qwen4exp_q8_0_row_bytes(n_embd),
                    n_embd, draft_tail, h->t_sample, logit_rows) != 0;
        }
        if (ok && draft_tail) {
            /* Pack each row's two ranges into one contiguous shortlist row:
             * prefix first, tail behind it.  Sources are other tensors and
             * the destinations do not overlap, so plain stream-ordered copies
             * are enough -- there is no in-place shuffle to reason about. */
            stage = "shortlist pack";
            for (uint32_t r = 0; r < logit_rows; r++) {
                ok = ds4_gpu_tensor_copy(
                         h->t_logits, (uint64_t)r * draft_width * f,
                         h->t_logits_prefix, (uint64_t)r * draft_prefix * f,
                         (uint64_t)draft_prefix * f) != 0 &&
                     ds4_gpu_tensor_copy(
                         h->t_logits,
                         ((uint64_t)r * draft_width + draft_prefix) * f,
                         h->t_logits_tail, (uint64_t)r * draft_tail * f,
                         (uint64_t)draft_tail * f) != 0;
                if (!ok) break;
            }
        }
    }
    MTP_HEAD_TICK(MTP_HEAD_T_LM_HEAD);
    if (ok) {
        stage = "gpu top-1";
        /* The head exposes only draft ids.  Keep the LM-head arithmetic intact,
         * reduce each finite logit row on the device and read back one id
         * instead of the full vocabulary row for a host scan.  A shortlist
         * row is packed, so the id comes out in packed positions and is
         * rebased below. */
        ok = ds4_gpu_indexer_topk_tensor(h->t_top1, h->t_logits,
                                         draft_width, logit_rows, 1u) != 0;
    }
    MTP_HEAD_TICK(MTP_HEAD_T_TOP1);
    if (ok) ok = ds4_gpu_end_commands() != 0;
    else (void)ds4_gpu_synchronize();
    MTP_HEAD_TICK(MTP_HEAD_T_END);

    /* A narrowed projection writes its sole result at logit row zero;
     * multi_out still reads the original last hyper row. */
    if (ok) {
        stage = "top-1 readback";
        ok = ds4_gpu_tensor_read(h->t_top1,
                                 (uint64_t)logit_first * sizeof(uint32_t),
                                 h->top1_host,
                                 (uint64_t)out_rows * sizeof(uint32_t)) != 0;
    }
    MTP_HEAD_TICK(MTP_HEAD_T_TOP1_IN);
    if (ok) {
        stage = "logit-0 readback";
        for (uint32_t t = 0; ok && t < out_rows; t++) {
            float logit0;
            const uint64_t row = (uint64_t)logit_first + t;
            ok = ds4_gpu_tensor_read(h->t_logits,
                                     row * (uint64_t)draft_width * f,
                                     &logit0, sizeof(logit0)) != 0;
            /* The former CPU scan seeded its comparison with row[0].  A NaN
             * there therefore kept token zero regardless of later values;
             * the generic GPU reducer deliberately ignores NaNs.  Preserve
             * the MTP proposal contract without changing that shared reducer. */
            if (ok) {
                uint32_t logit0_bits;
                memcpy(&logit0_bits, &logit0, sizeof(logit0_bits));
                if ((logit0_bits & 0x7fffffffu) > 0x7f800000u) {
                    h->top1_host[t] = 0u;
                }
            }
        }
    }
    MTP_HEAD_TICK(MTP_HEAD_T_LOGIT0_IN);
    if (ok && multi_out) {
        stage = "multi readback";
        ok = ds4_gpu_tensor_read(h->t_hyper,
                                 (uint64_t)first_row * hc_dim * f, multi_out,
                                 (uint64_t)out_rows * hc_dim * f) != 0;
    }
    MTP_HEAD_TICK(MTP_HEAD_T_MULTI_OUT);
    if (!ok) {
        return mtp_fail(err, errlen,
                        "qwen4exp MTP head: %s failed at position %u over %u "
                        "rows", stage, pos0, n_tokens);
    }
    for (uint32_t t = 0; t < out_rows; t++) {
        /* Unpack the top-1's packed position back into a token id: below the
         * prefix it IS the id; above it, rebase into the tail range.  Off
         * mode (prefix 0) packed nothing and the position is the id. */
        uint32_t id = h->top1_host[t];
        if (draft_prefix && id >= draft_prefix) {
            id = h->n_vocab - draft_tail + (id - draft_prefix);
        }
        draft_out[t] = (int)id;
    }
    if (timing) mtp_head_stage_calls++;
    return 0;
}

int ds4_qwen4exp_mtp_head_forward(ds4_qwen4exp_mtp_head *h,
                                  const int *next_tokens,
                                  const float *multi_in,
                                  uint32_t pos0, uint32_t n_tokens,
                                  int *draft_out, float *multi_out,
                                  char *err, size_t errlen) {
    return mtp_head_forward_impl(h, next_tokens, multi_in, pos0, n_tokens,
                                 draft_out, multi_out, false, err, errlen);
}

int ds4_qwen4exp_mtp_head_forward_last(ds4_qwen4exp_mtp_head *h,
                                       const int *next_tokens,
                                       const float *multi_in,
                                       uint32_t pos0, uint32_t n_tokens,
                                       int *draft_out, float *multi_out,
                                       char *err, size_t errlen) {
    return mtp_head_forward_impl(h, next_tokens, multi_in, pos0, n_tokens,
                                 draft_out, multi_out, true, err, errlen);
}

#endif /* DS4_NO_GPU */

#ifdef DS4_NO_GPU
void ds4_qwen4exp_mtp_head_time_snapshot(uint64_t *ns, size_t cap,
                                         uint64_t *calls) {
    for (size_t i = 0; ns && i < cap; i++) ns[i] = 0u;
    if (calls) *calls = 0u;
}
#endif
