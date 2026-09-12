/* ds4_shim: the flat C surface the Rust `ds4-engine` backend links.
 *
 * It wraps the ds4 engine and session API in ds4/ds4.h behind a handful of
 * plain functions so the Rust side declares a few extern functions and no
 * struct layouts. One handle = one engine + one session, which is exactly one
 * benchd phase (fresh engine per phase).
 *
 * The submodule is the Layr-Labs ds4 port, which carries the qwen4exp family:
 * the multi-shard GGUF loader, the qwen4exp graph, the nextn MTP head behind
 * `--mtp-model` and the depth-1 speculative cycle. The serial verbs below are
 * the ones the port routes to that graph; every other session verb refuses by
 * name inside the port itself.
 *
 * Every function returns 0 on success or a negative value on failure unless
 * stated otherwise. Failure text is available through ds4s_last_error(). */
#ifndef DS4_SHIM_H
#define DS4_SHIM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ds4s_handle ds4s_handle;

/* Open the engine on the CUDA backend and create one session of `ctx_size`
 * tokens. `model_path` is the FIRST shard of the target
 * (`...-00001-of-0000N.gguf`); the engine reads `split.count` from it and maps
 * the rest itself. `mtp_head_path` is the separate draft-head GGUF -- the port
 * reaches it through `ds4_engine_options.mtp_path`, which is what its
 * `--mtp-model` flag sets -- or NULL for a serve with no draft head.
 * `mtp_draft_tokens` follows the ds4 convention: 1 = no speculation, 2 = one
 * draft token per verify cycle. It is passed through unchanged. The port
 * implements depths 1 to 3, so with a head armed it refuses a draft-token
 * count outside 1 to 4 at open, as it refuses a DS4_QWEN_MTP_QUENCH that is
 * not 0.
 * Returns NULL on failure; ds4s_open_error() then carries the reason. */
ds4s_handle *ds4s_open(const char *model_path, const char *mtp_head_path,
                       int mtp_draft_tokens, int ctx_size, int n_threads);
void ds4s_close(ds4s_handle *h);

/* Why the last ds4s_open() in this process failed (never NULL). Readable when
 * ds4s_open() returned NULL and there is therefore no handle to ask. */
const char *ds4s_open_error(void);

/* The last failure text for this handle (never NULL). */
const char *ds4s_last_error(const ds4s_handle *h);

/* Vocabulary size of the loaded model. */
int ds4s_vocab_size(const ds4s_handle *h);

/* The model's end-of-sequence token id. */
int32_t ds4s_eos_token(const ds4s_handle *h);

/* Full reset: on the qwen4exp path ds4_session_invalidate() puts every cache,
 * recurrent state, conv history, indexer tape and n-gram history back to
 * position 0, so the next ds4s_sync forwards the whole prompt again. Called at
 * every phase start: one process serves one phase, but a warmup and a timed
 * leg on the same prefix must never share a free sync. */
void ds4s_invalidate(ds4s_handle *h);

/* Synchronize the session to the full token prefix `tokens[0..n)`. A prefix of
 * the live session evaluates only the suffix. After a successful return the
 * logits for the last position are ready. */
int ds4s_sync(ds4s_handle *h, const int32_t *tokens, size_t n);

/* Append one token and evaluate it (a teacher-forced step). Logits for the new
 * position are ready afterwards. */
int ds4s_eval(ds4s_handle *h, int32_t token);

/* The greedy token of the current logits, canonical lowest-id tie-break. */
int32_t ds4s_argmax(const ds4s_handle *h);

/* The top-k logits of the current position, highest first, lowest-id
 * tie-break. Writes up to `k` ids and logits and returns how many. */
int ds4s_top_logits(const ds4s_handle *h, int k, int32_t *ids, float *logits);

/* One speculative cycle: commit `first_token`, let the draft head propose, verify
 * against the target, and return the number of committed tokens written to
 * `out[0..cap)` (`out[0] == first_token`). `budget` is how many tokens the
 * caller still wants; the engine never commits more than that. Returns a
 * negative value on failure. Logits for the new frontier are ready afterwards.
 *
 * The port routes this to ds4_qwen4exp_mtp_cycle, which commits the target's
 * own GREEDY argmax and so emits the serial leg's token stream exactly; at
 * depth 1 it commits 1 or 2 tokens per round. */
int ds4s_eval_speculative(ds4s_handle *h, int32_t first_token, int budget, int32_t *out,
                          int cap);

/* The speculative cycle's counters, read from the port itself
 * (ds4_session_qwen4exp_spec_counters). Read before and after a phase and take
 * the difference.
 *
 * `drafts` and `hits` are the port's `drafted` and `accepted`: rounds that
 * carried a draft into a verify, and drafts the target's own argmax confirmed.
 * `quenches` is 0 by construction -- this path enters no adaptive disable, and
 * the port refuses a DS4_QWEN_MTP_QUENCH that is not 0 at open.
 *
 * `disagreements` is DIAGNOSTIC and is the port's
 * `verify_replay_disagreements`: on a rejecting round the cycle compares the
 * batched verify's row-0 argmax with a one-row replay of the same position and
 * counts where they differ. The replay stands; it is never a refusal. A low
 * rate is the tower's batch-shape residual; a high one means a rollback object
 * is not restoring the round's starting state. Nothing here enforces a
 * ceiling.
 *
 * Every out-pointer may be NULL. A handle whose session has no counters --
 * a non-qwen4exp session, or a CPU build -- reports zeros. */
void ds4s_spec_counters(const ds4s_handle *h, uint64_t *drafts, uint64_t *hits,
                        uint64_t *quenches, uint64_t *disagreements);

/* DIAGNOSTIC profile readout (ds4_session_qwen4exp_profile_text/_words):
 * cumulative engine-side timing splits, a one-line text summary and six
 * packed 64-bit words plus two doubles.  Zeros / empty text when the engine
 * ran with no profiling switch set or the session is not qwen4exp. */
int ds4s_profile_text(const ds4s_handle *h, char *buf, size_t cap);
int ds4s_profile_words(const ds4s_handle *h, uint64_t words[6], double f[2]);

#ifdef __cplusplus
}
#endif

#endif
