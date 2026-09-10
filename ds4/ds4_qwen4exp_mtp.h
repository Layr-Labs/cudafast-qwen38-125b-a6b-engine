#ifndef DS4_QWEN4EXP_MTP_H
#define DS4_QWEN4EXP_MTP_H

/*
 * qwen4exp -- multi-token prediction at draft depths 1 to 6.
 *
 * Two layers live here.
 *
 *   1. THE CYCLE.  Per round the head drafts a chain of N tokens, the target
 *      runs on [fed token, draft_0 .. draft_{N-1}] as ONE verify of N + 1
 *      rows, the drafts the target's own greedy argmax confirms are accepted
 *      as a prefix, and the round commits between 1 and N + 1 tokens.  The
 *      cycle reaches the model only through ds4_qwen4exp_mtp_model below, so
 *      it holds no graph state and is exercised on any host.
 *
 *   2. THE HEAD.  The nextn projections and norms, the 49th block and the
 *      target's borrowed LM head.  It reaches the block and the shared GPU
 *      primitives through ds4_qwen4exp_mtp_gpu_hooks, whose member types are
 *      the L4/L5a/L5b/L7 prototypes; binding a real function to a hook is
 *      itself the check that the signature still agrees.
 *
 * WHY THIS IS NOT ds4_session_eval_speculative_argmax().  That is upstream's
 * only PUBLIC speculative entry point, and the rollback it drives --
 * spec_frontier_snapshot / _restore / _commit_prefix -- is static inside ds4.c
 * and knows one kind of state, the compact KV frontier.  qwen4exp carries four
 * more kinds that a KV frontier cannot express: the GDN fp32 recurrent state
 * and its conv history, which are overwritten in place and carry no position;
 * the QSA indexer's exact key tape and the pooled blocks derived from it; and
 * the PLE conv state and n-gram history.  So this is a minimal public extension
 * of the same shape -- snapshot, restore, commit-or-drop a prefix -- with the
 * object set named rather than assumed.  See ds4_qwen4exp_rollback_set.
 *
 * The cycle shape is ds4's, not MLX's.  MLX feeds [pending] + drafts.dropLast()
 * and so yields at most `depth` tokens a round, which at depth 1 is one token
 * a round and no gain.  ds4 yields between 1 and depth + 1, which is what the
 * protocol adapter's acceptance_lengths expect.
 *
 * WHAT THE LEG EMITS.  A draft is committed only where it equals the target's
 * greedy argmax as the BATCHED VERIFY computes it.  That is standard
 * speculative decoding: the verify's argmax is the answer, and the draft is
 * accepted or rejected against it.  Nothing here re-derives the token from a
 * one-row decode before emitting it.
 *
 * WHAT SCORES IT.  EXACTNESS AGAINST OUR OWN DEPTH-0 SERIAL STREAM.  The
 * goldens for this engine are authored from the serial path, and the scored
 * benchmark reads the depth-N stream against them, so a single differing token
 * is a scoring failure rather than a quality question.  Byte-identity with
 * llama.cpp is NOT the contract and never was -- it runs a different quant
 * path -- but identity with our own serial leg is, at every depth 1 through 6 alike.
 *
 * This replaces the earlier framing, which read serial identity as a tripwire
 * and the per-depth oracle golden as the scoring truth.  That framing was
 * right while the tower's batch-shape residual was non-zero and a correct
 * engine could have failed a serial comparison.  The residual is now zero at
 * every width the cycle uses, the residual's cause is fixed by construction
 * rather than by measurement, and the requirement is the stronger one.
 *
 * SERIAL IDENTITY IS REACHABLE because the tower is row invariant.  A batched
 * verify and a one-row decode of the same row agree only
 * if every op under them reduces the same way at one row and at two.  Two did
 * not: the upstream Q8_0 matmul ds4_gpu_matmul_q8_0_tensor (ds4_metal.m:18193,
 * against the decode-order entry at :18263) and ds4_gpu_matmul_f32_tensor,
 * which takes cuBLAS SGEMM above one row on CUDA and a matvec at one row on
 * Metal.  Every other op runs one threadgroup or block per token and never sees
 * the batch.
 *
 * Both are routed inside the cycle's width -- see ds4_qwen4exp_matmul.h -- so
 * row t of an n-row call IS a one-row call, by construction rather than by
 * measurement.  From identical state with no rollback in between the difference
 * is now ZERO on the pre-final-mixer row and zero on the logits, on Metal and
 * on CUDA; CUDA measured 2.37e-2 before the f32 half of it.
 * tests/test_qwen4exp_graph asserts both are exactly zero, so an op that starts
 * tiering by row count again goes red there rather than surfacing later as a
 * raised verify/replay disagreement rate.
 *
 * A backend that could not reach zero would not meet the contract above, which
 * is the point of pinning it at exactly zero rather than at a tolerance: there
 * is no width at which the cycle is allowed to be approximately right.
 *
 * The runtime check stays, as a COUNTER.  On a rejecting round the cycle
 * compares the wide verify's argmax at the mismatching row with the narrower
 * replay's and, where they differ, increments `verify_replay_disagreements`
 * and carries on.  The replay's answer stands: the round commits the accepted
 * prefix and leaves the replay's distribution in `logits`, which is what the
 * caller samples its next fed token from, so the emitted stream and the fed
 * tokens cannot drift apart.
 *
 * The counter is diagnostic in the CYCLE and a hard requirement in the TESTS,
 * which is not a contradiction: the cycle cannot tell a numeric residual from
 * a broken rollback and must not kill a leg over the first, while the suite
 * knows the residual is zero and fails on any count at all.
 *
 * It refuses on neither.  A disagreement has two possible causes and the cycle
 * cannot separate them from where it stands: the batch-shape residual, which
 * is ordinary and must not kill a leg, or a rollback object that did not
 * restore to the round's start, which is a real fault.  The second stays
 * covered -- the rollback mutants in tests/test_qwen4exp_mtp and the checkpoint
 * and position assertions in tests/test_qwen4exp_graph catch it at test time,
 * and at runtime it shows as a disagreement RATE far above the residual, which
 * is what the counter exposes.  See ds4_session_qwen4exp_spec_counters().
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* ------------------------------------------------------------------------
 * Depth envelope
 * ------------------------------------------------------------------------ */

/*
 * The draft chain this build implements.  Depth N drafts N tokens by
 * re-entering the head on its OWN `multi` output -- the recursion the head is
 * defined by -- and verifies the N drafts plus the fed token in one forward of
 * N + 1 rows.  There is no tree: the chain is linear, so the verify is a
 * single sequence and the accept is its longest matching prefix.  Depth 7 and
 * beyond is refused by name.
 */
#define DS4_QWEN4EXP_IMPLEMENTED_DEPTH 6

/* The most tokens one cycle can commit: the fed token plus the draft chain.
 * Also the widest verify the cycle runs, which is what sizes the head's row
 * budget and the session's prefill floor. */
#define DS4_QWEN4EXP_MTP_MAX_COMMIT (DS4_QWEN4EXP_IMPLEMENTED_DEPTH + 1)

/* Rows of hc_dim the cycle keeps: one verify's worth, plus the two the head's
 * recursion ping-pongs between.  A chain step reads the previous step's
 * `multi` row while writing its own, so they cannot be one buffer. */
#define DS4_QWEN4EXP_MTP_HC_ROWS (DS4_QWEN4EXP_MTP_MAX_COMMIT + 2)

/*
 * tools/serve-up.sh sets DS4_MTP_DRAFT_TOKENS = declared draft length + 1,
 * because ds4 counts the fed token; harness/protocol-adapter reads the same
 * variable and arms the drafter at 2 or more.  So draft_tokens - 1 is the
 * depth: 1 = serial, 2 = depth 1, ..., 7 = depth 6.
 *
 * Returns the depth, or -1 with `err` filled in.  Depth 0 (serial) is
 * accepted: a serial leg opens the same engine with the drafter unarmed.
 */
int ds4_qwen4exp_mtp_depth_from_draft_tokens(int draft_tokens,
                                             char *err, size_t errlen);

/*
 * Refuse a scored leg whose environment arms an adaptive disable.
 *
 * Upstream's only adaptive disable is the DSpark scheduler
 * (ds4_dspark_scheduler_* in ds4.c), reached from
 * ds4_session_eval_dspark_speculative_argmax and from nowhere else; this
 * cycle, like ds4_session_glm_spec_cycle_impl, never enters it, so the quench
 * counter below stays 0 by construction.  What this call catches is the
 * adapter-side switch: DS4_QWEN_MTP_QUENCH, which the protocol adapter faults
 * on after the fact.  Refusing at open is cheaper than a voided leg.
 *
 * Returns 0 when nothing can quench this path, -1 with `err` filled in.
 */
int ds4_qwen4exp_mtp_check_no_yield_guard(char *err, size_t errlen);

/* ------------------------------------------------------------------------
 * The rollback contract
 * ------------------------------------------------------------------------ */

/*
 * A rejecting round has written state for a token that is not committed.  Every
 * piece of session state falls into one of two classes, and the class decides
 * which entry points the object must supply.
 *
 *   SNAPSHOT   the object is overwritten in place and carries no position, so
 *              it cannot be rewound: it must be copied out before the verify
 *              and copied back on rejection.  snapshot() and restore() are
 *              required, truncate() must be NULL.
 *
 *   TRUNCATE   the object is addressed by position and only ever appended to,
 *              so a rejection drops the rows at or after the boundary.
 *              truncate() is required, snapshot() and restore() must be NULL.
 *
 * truncate(pos) must drop EVERYTHING at or after `pos` in ONE step, including
 * state derived from dropped rows -- the indexer's pooled blocks are the case
 * that bites: block b pools tape rows [b*ratio, (b+1)*ratio), so truncate must
 * keep only the blocks with (b+1)*ratio <= pos.  Restoring a tape and bumping
 * an offset afterwards re-exposes rows the tape no longer covers.
 */
typedef enum {
    /* Running state with no position of its own: overwritten in place every
     * token, so it can only be rolled back to a copy the VERIFY FORWARD ITSELF
     * left behind.  The forward mirrors the state after each drafted row into
     * a slot, and the rollback is "adopt slot a".  There is no rewind and no
     * shorter replay forward: the recurrence is token-serial, so the state
     * after row k is the state a (k + 1)-row feed leaves, and selecting it and
     * replaying it are the same value. */
    DS4_QWEN4EXP_ROLLBACK_SELECT_ROW = 0,
    /* Append-only, addressed by position: rolling back is dropping the tail. */
    DS4_QWEN4EXP_ROLLBACK_TRUNCATE = 1,
} ds4_qwen4exp_rollback_kind;

/*
 * The state a qwen4exp session carries across a token, and the class each one
 * belongs to.  Every id must be registered before a cycle runs; a graph that
 * genuinely has no such object registers a no-op and says so in a comment,
 * rather than leaving a hole the cycle cannot see.
 */
typedef enum {
    /* [n_gdn_layer][n_value_head][state][state] fp32, rewritten every token. */
    DS4_QWEN4EXP_STATE_GDN_RECURRENT = 0,
    /* The 3-row conv history behind the 4-tap causal conv1d. */
    DS4_QWEN4EXP_STATE_GDN_CONV,
    /* QSA K/V rows, one per position per full-attention block. */
    DS4_QWEN4EXP_STATE_QSA_KV,
    /* The indexer's exact unrotated key tape AND its pooled blocks. */
    DS4_QWEN4EXP_STATE_QSA_INDEXER_TAPE,
    /* The n-gram TOKEN history: the two previous ids the hash folds in. */
    DS4_QWEN4EXP_STATE_PLE_HISTORY,
    /* The PLE 9-row dilation-3 convolution state.  Its own id, beside
     * GDN_CONV: it is a different object with a different owner, it lives on
     * the device where the history lives on the host, and one id covering both
     * would let a graph register one of them and satisfy the check. */
    DS4_QWEN4EXP_STATE_PLE_CONV,
    /* The MTP head's own cache stack, which never touches PLE. */
    DS4_QWEN4EXP_STATE_MTP_HEAD_CACHE,
    DS4_QWEN4EXP_STATE_COUNT,
} ds4_qwen4exp_state_id;

/* The class each id must register as.  Fixed by the shape of the state, not a
 * preference: registering the wrong class is refused. */
ds4_qwen4exp_rollback_kind ds4_qwen4exp_state_kind(ds4_qwen4exp_state_id id);

/* The id's name, for refusals.  Never NULL. */
const char *ds4_qwen4exp_state_name(ds4_qwen4exp_state_id id);

typedef struct {
    void *ctx;
    /*
     * Adopt the object as it stood after verify row `row`, 0-based.
     * SELECT_ROW only.
     *
     * The cycle calls this only when it accepted FEWER than every draft, so
     * `row` is always a row the forward was asked to snapshot: 0 <= row < N.
     * A full accept keeps the live state, which is already the state after the
     * last row, and calls nothing.
     */
    int (*select_row)(void *ctx, uint32_t row);
    /* Drop everything at or after `pos`, derived state included.  TRUNCATE
     * only, and in one step: see the note above. */
    int (*truncate)(void *ctx, uint32_t pos);
} ds4_qwen4exp_rollback_object;

typedef struct {
    ds4_qwen4exp_rollback_object obj[DS4_QWEN4EXP_STATE_COUNT];
    bool registered[DS4_QWEN4EXP_STATE_COUNT];
} ds4_qwen4exp_rollback_set;

void ds4_qwen4exp_rollback_init(ds4_qwen4exp_rollback_set *set);

/* Register one object.  Refuses when the entry points do not match the class
 * the id requires. */
int ds4_qwen4exp_rollback_register(ds4_qwen4exp_rollback_set *set,
                                   ds4_qwen4exp_state_id id,
                                   const ds4_qwen4exp_rollback_object *obj,
                                   char *err, size_t errlen);

/* Refuse by name when any id is still unregistered. */
int ds4_qwen4exp_rollback_check(const ds4_qwen4exp_rollback_set *set,
                                char *err, size_t errlen);

/* ------------------------------------------------------------------------
 * The model seam
 * ------------------------------------------------------------------------ */

/*
 * The family entry points the cycle drives.  L7 fills these with the qwen4exp
 * graph; the exactness test fills them with a reduced reference model.
 *
 * BATCH INVARIANCE.  Row t of verify_rows(tokens, n, pos0) must equal what
 * decode_token(tokens[t], pos0 + t) produces from the same state, bit for bit,
 * for every n.  An accepting round writes the state for up to N + 1 committed
 * tokens from one pass while the serial leg writes it from that many one-row
 * passes, and a partially accepting round replays its accepted prefix at a
 * NARROWER width than it verified; if any of those differ the legs' token
 * streams diverge later, silently.  Depth 3 verifies four rows, so invariance
 * is required at every width up to N + 1.
 * The cycle catches the violation on the first rejecting round -- it compares
 * the row-0 argmax from the verify against the argmax from the replay -- and
 * refuses rather than reporting an inexact leg as a result.
 */
typedef struct {
    void *ctx;

    /* Hyper-connection width, n_hc * n_embd (10240), and the shared vocabulary.
     * Both are read from the TARGET: the head borrows its embedding and head. */
    uint32_t hc_dim;
    uint32_t n_vocab;

    /* Advance the target over `n` rows starting at `pos0`, leaving the state
     * as if every row were committed.  Writes `n` rows of `hc_dim` floats --
     * the PRE-FINAL-MIXER hyper-connection stream, which is what the head
     * consumes -- into `hc_rows`, and EVERY row's logits into `row_logits`
     * ([n][n_vocab] f32).  Returns 0 on success.
     *
     * Every row, not just the last: the accept loop compares each drafted
     * position against the target's own argmax there, and one forward already
     * holds every row's state.  The final mixer and the LM head over n rows
     * read the 676 MiB output weight once; a per-row call after the fact reads
     * it once per comparison.  Row t's logits must equal what a one-row
     * decode of row t leaves -- the same batch invariance the stream itself
     * requires, extended to the head. */
    int (*verify_rows)(void *ctx, const int *tokens, uint32_t n, uint32_t pos0,
                       float *hc_rows, float *row_logits);

    /* One row at `pos`: the serial decode step, and the replay a rejecting
     * round runs.  Same outputs for a single row. */
    int (*decode_token)(void *ctx, int token, uint32_t pos,
                        float *hc_row, float *logits);

    /* The target's LM head over one pre-final-mixer row.
     *
     * The cycle no longer calls it: the verify returns every row's logits, so
     * the accept loop reads them rather than re-running the head.  It stays on
     * the seam because it is the ONE-ROW half of the batch invariance the
     * verify's wide head now depends on, and the diagnostics compare the two.
     * Required, and required to agree. */
    int (*head_logits)(void *ctx, const float *hc_row, float *logits);

    /*
     * One MTP draft step: the head's argmax for the token after `next_token`,
     * given the pre-final-mixer row at `pos`.  See
     * ds4_qwen4exp_mtp_head_forward() for the wiring behind this.
     *
     * `multi_out`, when not NULL, receives the head's own hyper stream for the
     * row it just wrote -- hc_dim floats.  That row is what the NEXT step of a
     * depth-2-or-deeper chain passes back as `hc_row`, standing in for the
     * target row the chain does not have.  The step must fill it whatever else
     * it does: a test hook that overrides the drafted token still owes the
     * caller a real `multi` row, because the row after it is computed from it.
     */
    int (*draft_step)(void *ctx, int next_token, const float *hc_row,
                      uint32_t pos, int *draft_out, float *multi_out);

    /*
     * OPTIONAL: several consecutive head rows in ONE forward.  Row t takes
     * next_tokens[t] and hc_rows[t] (hc_dim floats each) at position pos0 + t,
     * exactly as n calls to draft_step would in that order; only the LAST
     * row's draft and `multi` row come back, because every row before it is a
     * seed whose draft nobody reads.  The chain uses it to fold the seed rows
     * a round owes the head cache into the round's first draft step, so an
     * accepting round costs one head forward instead of one per row -- the
     * head's block runs the rows together the way the target's verify does.
     * NULL means the cycle seeds and drafts through draft_step, one row per
     * call, which is the same cache and the same drafts at more launches.
     */
    int (*draft_rows)(void *ctx, const int *next_tokens, const float *hc_rows,
                      uint32_t pos0, uint32_t n, int *draft_out,
                      float *multi_out);
} ds4_qwen4exp_mtp_model;

/* ------------------------------------------------------------------------
 * Counters and state
 * ------------------------------------------------------------------------ */

/*
 * ds4s_spec_counters() reports (drafts, hits, quenches); those are `drafted`,
 * `accepted` and `quenches` here.  The adapter takes deltas across a leg and
 * faults on any quench, so `quenches` staying 0 is load bearing.
 *
 * `verify_replay_disagreements` rides alongside them and is DIAGNOSTIC, never
 * a fault: see ds4_session_qwen4exp_spec_counters() in ds4.h.  The four
 * counters above keep the meaning they always had.
 */
typedef struct {
    uint64_t rounds;    /* cycles run                                        */
    /* DRAFT TOKENS carried into a verify, not rounds: at depth N a round
     * offers up to N of them, and an acceptance rate is drafts over hits.  At
     * depth 1 the two readings coincide, which is why this counted rounds
     * before there was a depth to tell them apart. */
    uint64_t drafted;
    uint64_t accepted;  /* draft TOKENS the target's own argmax confirmed    */
    uint64_t committed; /* tokens committed, the fed token included          */
    uint64_t quenches;  /* always 0: no yield guard on this path             */
    /*
     * STRUCTURALLY ZERO, and kept because ds4s_spec_counters() reports a fixed
     * tuple and benchd reads it.
     *
     * It counted rejecting rounds where the batched verify and the one-row
     * replay disagreed about the same position.  There is no replay any more:
     * a partially accepting round adopts the state slot the verify forward
     * itself left after the last accepted row, so there is no second answer to
     * disagree with.  What the counter used to guard -- that selecting the
     * state equals recomputing it -- is now pinned in the tests, which compare
     * the selected slot against an explicit replay at every depth and every
     * acceptance count.  A non-zero value here would mean the cycle took a
     * path this build does not have.
     */
    uint64_t verify_replay_disagreements;
    /* How many rounds committed 1 token, 2 tokens, ...  Index 0 is unused. */
    uint64_t commit_hist[DS4_QWEN4EXP_MTP_MAX_COMMIT + 1];
    /*
     * Where a round's wall time goes, in nanoseconds.  DIAGNOSTIC, and the
     * three together are the round: the target's verify (or its one-row decode
     * on the no-draft path), the head's seed and chain steps, and the rollback
     * a partially accepting round performs.
     *
     * They exist because "is depth N worth it" is not answerable from
     * acceptance alone.  A deeper chain buys tokens per round and pays for
     * them in verify width and in head steps, and only the split says which
     * side won.  Reading them costs one monotonic clock read per phase.
     */
    uint64_t verify_ns;
    uint64_t draft_ns;
    uint64_t rollback_ns;
} ds4_qwen4exp_mtp_counters;

typedef struct {
    /* Registered and checked once, by ds4_qwen4exp_mtp_state_init().  The set
     * must outlive the state; nothing here copies it. */
    const ds4_qwen4exp_rollback_set *rollback;
    int      depth;          /* 0 = serial, N = N drafts per verify         */
    /* The carried chain, pending[0] first.  pending[k] is the head's guess at
     * the token k + 1 places after `pending_parent`. */
    int      pending[DS4_QWEN4EXP_IMPLEMENTED_DEPTH];
    int      n_pending;      /* 0 when nothing is carried                   */
    int      pending_parent; /* the token the chain was drafted from        */
    /* Head cache rows the cycle has written: rows [0, head_rows).  The chain
     * writes one per step and a round keeps the ones its accepted drafts
     * produced, so this says whether the next chain's first row has everything
     * below it -- and, on a round that accepted its whole chain, that the
     * bonus token's row is still owed.  Rows below the round's own position
     * belong to the prefill and are never counted as written here. */
    uint32_t head_rows;
    float   *hc_scratch;     /* DS4_QWEN4EXP_MTP_HC_ROWS rows of hc_dim     */
    /* The verify's own logits, one row per verified row: the accept loop reads
     * row a straight out of it and the frontier row is copied from it. */
    float   *logits_rows;    /* MAX_COMMIT rows of n_vocab                  */
    uint32_t hc_dim;
    uint32_t n_vocab;
    ds4_qwen4exp_mtp_counters counters;
} ds4_qwen4exp_mtp_state;

/* Refuses when the depth is outside the envelope, when the widths are zero, or
 * when `rollback` is incomplete: an object that only a rejecting round would
 * have used must be missed at open, not on the round that needed it. */
int  ds4_qwen4exp_mtp_state_init(ds4_qwen4exp_mtp_state *st, int depth,
                                 const ds4_qwen4exp_rollback_set *rollback,
                                 uint32_t hc_dim, uint32_t n_vocab,
                                 char *err, size_t errlen);
void ds4_qwen4exp_mtp_state_free(ds4_qwen4exp_mtp_state *st);

/* Drop the carried draft.  Call after a rewind, a prefix change or anything
 * else that makes the parent token no longer the frontier. */
void ds4_qwen4exp_mtp_invalidate(ds4_qwen4exp_mtp_state *st);

/* The counters are self-consistent: sum over commit_hist equals rounds, the
 * weighted sum equals committed, drafted <= rounds, accepted <= drafted, and
 * quenches == 0.  Refuses by name on any violation. */
int ds4_qwen4exp_mtp_counters_check(const ds4_qwen4exp_mtp_counters *c,
                                    char *err, size_t errlen);

/*
 * One cycle.
 *
 * Commits `first_token`, and when a chain of N drafts is carried for it
 * verifies [first_token, draft_0 .. draft_{N-1}] in ONE pass of N + 1 rows.
 * Row j's argmax is the target's own token after row j, so the drafts that
 * survive are the longest prefix each of whose members the target chose; the
 * first token the target chose instead is the one the caller feeds next, and
 * it is left in `logits` rather than committed.
 *
 * A round that accepts fewer than all N drafts rolls every carried object back
 * to the accepted length WITHOUT a second forward.  The verify mirrors the
 * running state after each drafted row into a slot, so the SELECT_ROW objects
 * adopt slot `a` and the TRUNCATE objects drop their tails at pos + a + 1.
 * Rejected rows therefore leave no trace in any carried object -- not a KV
 * row, not a conv tap, not a head cache row -- and the round costs one forward
 * rather than two.  The frontier distribution is row a's, which the accept
 * scan already computed, so even the one-row replay is gone.
 *
 * Writes the committed tokens to `accepted[0..)` starting with `first_token`,
 * leaves the logits of the position AFTER them in `logits` (n_vocab floats),
 * and returns how many tokens it committed: 1 to N + 1, or -1 with `err`
 * filled in.
 *
 * `budget` is how many tokens the caller still wants; the cycle never commits
 * more, so budget 1 takes the plain path and a budget below N + 1 shortens the
 * chain this round rather than overshooting it.  `pos` is the frontier, the number
 * of tokens already committed.
 *
 * There is no end-of-sequence parameter.  The cycle commits the target's own
 * argmax whatever that token is, and leaves the head cache covering exactly
 * the committed prefix -- with the rows the accepted drafts already wrote,
 * plus the one row a full accept leaves owed.  Skipping the head step after an
 * end token would save one draft and leave a hole that the next round reads if
 * the caller keeps going.  Stopping is the caller's decision and it already
 * owns it.  At depth 0 the head is never run, so there is no cache to keep
 * whole.
 *
 * The return value IS the round's acceptance length as the protocol adapter
 * reports it: the adapter counts the committed tokens after the fed one plus
 * the frontier argmax it leaves behind, which is the same number.
 */
int ds4_qwen4exp_mtp_cycle(ds4_qwen4exp_mtp_state *st,
                           const ds4_qwen4exp_mtp_model *model,
                           int first_token,
                           uint32_t pos, int budget,
                           int *accepted, int accepted_cap,
                           float *logits,
                           char *err, size_t errlen);
/* `logits` is REQUIRED, and it is an out-parameter as much as a scratch: the
 * cycle leaves the distribution for the position AFTER everything it committed
 * in it, which is what the caller samples its next fed token from.  A caller
 * that passes its own frontier buffer needs to do nothing else; one that passes
 * NULL is refused by name rather than left sampling a stale distribution. */

/* ------------------------------------------------------------------------
 * The head
 * ------------------------------------------------------------------------ */

/* The head is graph code and needs the GPU tensor API; the cycle above does
 * not.  ds4.c guards its own ds4_gpu.h include the same way, so a CPU build
 * gets the cycle, the depth envelope and the rollback contract and nothing
 * that would ask for a backend. */
#ifndef DS4_NO_GPU

#include "ds4_gpu.h"

/*
 * The 49th block, as a hook.
 *
 * The head's block is an ordinary qwen4exp full-attention block -- the L4 QSA
 * sequence, the L5a MoE trio and the two L5b hyper-connection mixers, in the
 * order Qwen4ExpDecoderLayer runs them -- so it is L7's graph code, not the
 * head's.  It is always full attention (the head file's compress_ratios says 0
 * at that index and is wrong) and it never carries PLE.
 *
 * `hyper` is [n_tokens][n_hc][n_embd] and is rewritten in place.  `cache` is
 * the head's OWN cache stack, never the tower's.
 *
 * Returns NONZERO on success and 0 on failure, the ds4_gpu convention every
 * other hook here follows and the same sense as ds4_qwen4exp_graph_forward's
 * bool.  It is the opposite of the 0-means-success convention the rest of
 * ds4.c uses for engine entry points, which is why it is stated here.
 */
typedef int (*ds4_qwen4exp_block_forward_fn)(
        void           *graph,
        void           *cache,
        ds4_gpu_tensor *hyper,
        uint32_t        il,
        uint32_t        pos0,
        uint32_t        n_tokens);

/* The graph's implementation of the above, defined in ds4_qwen4exp_graph.inc.
 * Declared HERE rather than only in ds4_qwen4exp_graph.h because that header
 * names ds4.c's internal ds4_tensor/ds4_model types and cannot be included
 * from ds4_qwen4exp_mtp_hooks.c.  ds4.c sees both declarations, so a signature
 * drift between them is a compile error rather than a silent mismatch. */
int ds4_qwen4exp_graph_head_block(void *graph, void *cache,
                                  ds4_gpu_tensor *hyper, uint32_t il,
                                  uint32_t pos0, uint32_t n_tokens);

/*
 * The shared primitives the head composes, typed from the declarations the
 * other lanes expose.  Binding the real function to the member is the check
 * that the signature still agrees, so a drift in L5b's prototype becomes a
 * compile error at L7's merge rather than a silent mismatch.
 *
 * EVERY member returns NONZERO on success and 0 on failure, `block` included.
 * That is the ds4_gpu convention, not ds4.c's 0-means-success one; a hook that
 * has it backwards makes the head report failure on every good call.
 *
 *   rms_norm     ds4_gpu_qwen4exp_rms_norm_tensor            (L5b)
 *   hc_mixer     ds4_gpu_qwen4exp_hc_mixer_tensor            (L5b)
 *   embed        ds4_gpu_qwen4exp_embed_tokens_hc_tensor     (L5b)
 *   matmul_q8_0  the decode-order entry, ds4_qwen4exp_matmul.h
 *   block        L7's qwen4exp block forward
 */
typedef struct {
    int (*rms_norm)(ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
                    const void *model_map, uint64_t model_size,
                    uint64_t weight_offset, uint32_t n, uint32_t group,
                    uint32_t rows, float eps, float weight_bias,
                    int round_bf16);
    /* One slab per weight: the mixer's four tensors can land in different
     * shards of a split GGUF, and the head reads two mappings of its own. */
    int (*hc_mixer)(ds4_gpu_tensor *mixed, ds4_gpu_tensor *inject,
                    ds4_gpu_tensor *normed_scratch,
                    ds4_gpu_tensor *lowrank_scratch,
                    ds4_gpu_tensor *wide_scratch, const ds4_gpu_tensor *hyper,
                    const ds4_gpu_qwen4exp_slab *norm_weight,
                    const ds4_gpu_qwen4exp_slab *down_weight,
                    const ds4_gpu_qwen4exp_slab *up_weight,
                    const ds4_gpu_qwen4exp_slab *inject_weight,
                    uint32_t n_embd, uint32_t n_hc, uint32_t n_lowrank,
                    uint32_t rows, float eps, float weight_bias,
                    int round_bf16);
    int (*embed)(ds4_gpu_tensor *out_hc, ds4_gpu_tensor *rows_scratch,
                 const ds4_gpu_tensor *tokens, const void *model_map,
                 uint64_t model_size, uint64_t weight_offset,
                 uint32_t weight_type, uint32_t n_vocab, uint32_t n_tokens,
                 uint32_t n_embd, uint32_t n_hc);
    /* Optional backend packer for [embedding | hidden-stream] rows.  NULL
     * keeps the portable tensor-copy sequence used by the tests and by
     * backends without a fused implementation. */
    int (*ehx_pack)(ds4_gpu_tensor *out, const ds4_gpu_tensor *embedding,
                    const ds4_gpu_tensor *hidden, uint32_t n_tokens,
                    uint32_t n_hc, uint32_t n_embd);
    int (*matmul_q8_0)(ds4_gpu_tensor *out, const void *model_map,
                       uint64_t model_size, uint64_t weight_offset,
                       uint64_t in_dim, uint64_t out_dim,
                       const ds4_gpu_tensor *x, uint64_t n_tok);
    ds4_qwen4exp_block_forward_fn block;
} ds4_qwen4exp_mtp_gpu_hooks;

/*
 * The head reads TWO mappings.  The nextn tensors and blk.<block_index>.* come
 * from the --mtp GGUF, which is a separate draft model opened through
 * ds4_engine_options.mtp_path.  That file sets nextn_shared_target_tensors and
 * carries no token_embd and no output, so the embedding and the LM head come
 * from the TARGET's mapping.  Nothing here assumes the two mappings share an
 * owner, and neither pointer is written.
 */
typedef struct {
    const void *head_map;
    uint64_t    head_size;
    const void *target_map;
    uint64_t    target_size;

    /* Offsets into head_map. */
    uint64_t enorm_offset;         /* blk.N.nextn.enorm.weight        F32   */
    uint64_t hnorm_offset;         /* blk.N.nextn.hnorm.weight        F32   */
    uint64_t eh_proj_offset;       /* blk.N.nextn.eh_proj.weight      Q8_0  */
    uint32_t eh_proj_in_dim;       /* its dim[0]: must be 2 * n_embd (5120) */
    uint64_t hc_head_norm_offset;  /* blk.N.nextn.hc_head_norm.weight F32   */
    uint64_t hc_head_down_offset;  /* blk.N.nextn.hc_head_down.weight Q8_0  */
    uint64_t hc_head_up_offset;    /* blk.N.nextn.hc_head_up.weight   Q8_0  */

    /* Offsets into target_map: the borrowed pair. */
    uint64_t token_embd_offset;
    uint32_t token_embd_type;      /* ggml type id, Q8_0 in the shipped file */
    uint64_t output_offset;

    uint32_t block_index;          /* n_layer: 48 in the production shape    */
    uint32_t n_embd;
    uint32_t n_hc;
    uint32_t n_lowrank;
    uint32_t n_vocab;
    uint32_t max_tokens;           /* rows one forward may carry             */

    float rms_eps;
    float weight_bias;             /* 1 for zero-centered weights, else 0    */
    int   round_bf16;

    ds4_qwen4exp_mtp_gpu_hooks hooks;
    void *graph;                   /* passed straight to hooks.block         */
    void *cache;                   /* the head's own cache stack             */

    /* Owned by init(), released by free(). */
    ds4_gpu_tensor *t_tokens;
    ds4_gpu_tensor *t_embed_rows;
    ds4_gpu_tensor *t_embed_out;
    ds4_gpu_tensor *t_e_normed;
    ds4_gpu_tensor *t_h_normed;
    ds4_gpu_tensor *t_ehx;
    ds4_gpu_tensor *t_hyper;
    ds4_gpu_tensor *t_mix_normed;
    ds4_gpu_tensor *t_mix_lowrank;
    ds4_gpu_tensor *t_mix_wide;
    ds4_gpu_tensor *t_sample;
    ds4_gpu_tensor *t_logits;
    ds4_gpu_tensor *t_top1;
    uint32_t       *top1_host;
} ds4_qwen4exp_mtp_head;

/*
 * The eh_proj row order: EMBEDDING HALF FIRST.
 *
 * MLX applies two separate Linear layers, fc_embedding and fc_hidden.  The
 * converter that produced the shipped head (llama.cpp PR #27836) fuses them
 * with torch.cat([fc_embedding, fc_hidden], dim=1) and its graph feeds
 * ggml_concat(e_norm, h_norm, 0), so the embedding half occupies input rows
 * [0, n_embd) and the hidden half [n_embd, 2 * n_embd).  ds4's GLM head builds
 * its concat the same way (glm_graph_mtp_step writes enorm into
 * concat[0..n_embd) and hnorm above it), and the shipped tensor is
 * blk.48.nextn.eh_proj.weight [5120, 2560] Q8_0.
 *
 * The order is settled, so nothing here re-derives it; what init() checks is
 * that the file agrees on the WIDTH, because a head whose eh_proj is not
 * 2 * n_embd rows wide is not this head and the halves would not line up.
 */
#define DS4_QWEN4EXP_EH_PROJ_EMBED_FIRST 1

/* Bind every hook but `block` to the function the owning lane exports; see
 * ds4_qwen4exp_mtp_hooks.c.  Fill in `block` with the graph's own qwen4exp
 * block forward and the head is ready for init(). */
void ds4_qwen4exp_mtp_default_hooks(ds4_qwen4exp_mtp_gpu_hooks *hooks);

int  ds4_qwen4exp_mtp_head_init(ds4_qwen4exp_mtp_head *h, char *err, size_t errlen);
void ds4_qwen4exp_mtp_head_free(ds4_qwen4exp_mtp_head *h);

/*
 * One head forward.
 *
 *   e      = fc_embedding(enorm(embed(next)))                   [rows][n_embd]
 *   h      = fc_hidden(hnorm(multi))                     [rows][n_hc][n_embd]
 *   hyper  = e broadcast over the streams + h            [rows][n_hc][n_embd]
 *   hyper  = block(hyper)                                             in place
 *   sample = hc_head_mixer(hyper)                                [rows][n_embd]
 *   logits = target.output * sample                             [rows][n_vocab]
 *
 * `enorm` is over n_embd; `hnorm` is UNGROUPED over the whole n_hc * n_embd
 * vector, on one statistic, unlike the hyper-connection norms.  Both fc layers
 * are the two halves of eh_proj, so the broadcast-and-add is one matmul over
 * rows * n_hc concatenated rows.
 *
 * `multi_in` is the target's PRE-final-mixer stream, rows of n_hc * n_embd
 * floats.  `draft_out` receives the argmax per row.  `multi_out`, when not
 * NULL, receives `hyper` -- the stream a depth-2 chain would feed back; at
 * depth 1 nothing reads it.
 */
int ds4_qwen4exp_mtp_head_forward(ds4_qwen4exp_mtp_head *h,
                                  const int *next_tokens,
                                  const float *multi_in,
                                  uint32_t pos0, uint32_t n_tokens,
                                  int *draft_out, float *multi_out,
                                  char *err, size_t errlen);

/* The same forward over `n_tokens` consecutive rows, handing back only the
 * LAST row's argmax (one int) and, when `multi_out` is not NULL, only its
 * `hyper` row (hc_dim floats).  Every row still runs the block and writes its
 * own cache row; what is narrower is the readback and the argmax.  This is the
 * entry the seam's draft_rows binds to. */
int ds4_qwen4exp_mtp_head_forward_last(ds4_qwen4exp_mtp_head *h,
                                       const int *next_tokens,
                                       const float *multi_in,
                                       uint32_t pos0, uint32_t n_tokens,
                                       int *draft_out, float *multi_out,
                                       char *err, size_t errlen);

/* Greedy argmax with the canonical lowest-id tie-break the shim's ds4s_argmax
 * documents.  Shared so the head and the cycle cannot break ties apart. */
#endif /* DS4_NO_GPU */

int ds4_qwen4exp_mtp_argmax(const float *logits, uint32_t n_vocab);

#endif /* DS4_QWEN4EXP_MTP_H */
