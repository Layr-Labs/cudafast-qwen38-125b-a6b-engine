/* See ds4_shim.h. Compiled against the ds4 submodule's ds4.h and linked with
 * the engine's CUDA core objects into libds4qwen.so (tools/ds4/build.sh). */
#include "ds4_shim.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ds4.h"

struct ds4s_handle {
    ds4_engine *engine;
    ds4_session *session;
    char err[512];
};

/* Why the last open failed, for the caller that got NULL back. */
static char g_open_err[512];

static void set_err(ds4s_handle *h, const char *msg) {
    if (!h) return;
    snprintf(h->err, sizeof(h->err), "%s", msg ? msg : "unknown ds4 failure");
}

static void set_open_err(const char *msg) {
    snprintf(g_open_err, sizeof(g_open_err), "%s", msg ? msg : "unknown ds4 open failure");
    fprintf(stderr, "ds4_shim: %s\n", g_open_err);
}

ds4s_handle *ds4s_open(const char *model_path, const char *mtp_head_path,
                       int mtp_draft_tokens, int ctx_size, int n_threads) {
    g_open_err[0] = '\0';
    if (!model_path || !model_path[0] || ctx_size <= 0) {
        set_open_err("ds4s_open needs a model path and a positive context size");
        return NULL;
    }
    ds4s_handle *h = calloc(1, sizeof(*h));
    if (!h) {
        set_open_err("out of memory allocating the ds4 handle");
        return NULL;
    }
    ds4_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path = model_path;
    opt.mtp_path = (mtp_head_path && mtp_head_path[0]) ? mtp_head_path : NULL;
    opt.backend = DS4_BACKEND_CUDA;
    opt.n_threads = n_threads;
    opt.context_size = ctx_size;
    /* Passed through, not clamped: the port refuses a value below 1 by name
     * when a head is armed (ds4_qwen4exp_mtp_depth_from_draft_tokens), and a
     * clamp here would make that refusal unreachable. */
    opt.mtp_draft_tokens = mtp_draft_tokens;
    opt.mtp_margin = 3.0f;
    if (ds4_engine_open(&h->engine, &opt) != 0 || !h->engine) {
        char msg[512];
        snprintf(msg, sizeof(msg), "ds4_engine_open failed for %s", model_path);
        set_open_err(msg);
        free(h);
        return NULL;
    }
    if (ds4_session_create(&h->session, h->engine, ctx_size) != 0 || !h->session) {
        char msg[128];
        snprintf(msg, sizeof(msg), "ds4_session_create(ctx=%d) failed", ctx_size);
        set_open_err(msg);
        ds4_engine_close(h->engine);
        free(h);
        return NULL;
    }
    ds4_session_set_progress(h->session, NULL, NULL);
    ds4_session_set_display_progress(h->session, NULL, NULL);
    /* ARM THE DRAFTER HERE, NOT ON THE FIRST TIMED TOKEN.
     *
     * The engine builds the MTP head's cache slot, its scratch tensors and its
     * device residency on the first speculative cycle. On the scored free-run
     * phase that cycle is the first token benchd's DECODE clock covers, so a
     * one-off setup -- device allocations, host callocs, uploads of zeros --
     * was being priced as decode. It runs at open instead, before any phase
     * connects and before any window opens.
     *
     * It computes nothing: no token is evaluated, no prompt is read, no
     * position moves, and every buffer holds the same zeros the lazy path
     * wrote. A session with no head bound has nothing to arm and returns 0.
     *
     * A refusal is NOT fatal here. The engine puts itself back and the first
     * cycle re-runs the same setup and reports the same error in the same
     * place, so an open that could serve the serial route still serves it. */
    {
        char spec_err[512] = {0};
        if (ds4_session_qwen4exp_spec_prepare(h->session, spec_err,
                                              sizeof(spec_err)) != 0) {
            fprintf(stderr,
                    "ds4_shim: the MTP drafter could not be armed at open (%s); "
                    "it will be built on the first speculative cycle\n",
                    spec_err[0] ? spec_err : "no reason given");
        }
    }
    return h;
}

void ds4s_close(ds4s_handle *h) {
    if (!h) return;
    if (h->session) ds4_session_free(h->session);
    if (h->engine) ds4_engine_close(h->engine);
    free(h);
}

const char *ds4s_open_error(void) {
    return g_open_err[0] ? g_open_err : "";
}

const char *ds4s_last_error(const ds4s_handle *h) {
    return h && h->err[0] ? h->err : "";
}

int ds4s_vocab_size(const ds4s_handle *h) {
    return h && h->engine ? ds4_engine_vocab_size(h->engine) : 0;
}

int32_t ds4s_eos_token(const ds4s_handle *h) {
    return h && h->engine ? (int32_t)ds4_token_eos(h->engine) : -1;
}

void ds4s_invalidate(ds4s_handle *h) {
    if (h && h->session) ds4_session_invalidate(h->session);
}

int ds4s_sync(ds4s_handle *h, const int32_t *tokens, size_t n) {
    if (!h || !tokens || n == 0) return -1;
    ds4_tokens prompt = {0};
    for (size_t i = 0; i < n; i++) ds4_tokens_push(&prompt, (int)tokens[i]);
    char err[256] = {0};
    const int rc = ds4_session_sync(h->session, &prompt, err, sizeof(err));
    ds4_tokens_free(&prompt);
    if (rc != 0) {
        set_err(h, err[0] ? err : "ds4_session_sync failed");
        return -1;
    }
    return 0;
}

int ds4s_eval(ds4s_handle *h, int32_t token) {
    if (!h) return -1;
    char err[256] = {0};
    if (ds4_session_eval(h->session, (int)token, err, sizeof(err)) != 0) {
        set_err(h, err[0] ? err : "ds4_session_eval failed");
        return -1;
    }
    return 0;
}

int32_t ds4s_argmax(const ds4s_handle *h) {
    if (!h) return -1;
    return (int32_t)ds4_session_argmax(h->session);
}

int ds4s_top_logits(const ds4s_handle *h, int k, int32_t *ids, float *logits) {
    if (!h || !ids || !logits || k <= 0) return 0;
    ds4_token_score *scores = calloc((size_t)k, sizeof(*scores));
    if (!scores) {
        set_err((ds4s_handle *)h, "out of memory allocating the top-k buffer");
        return 0;
    }
    const int n = ds4_session_top_logprobs(h->session, scores, k);
    if (n <= 0) set_err((ds4s_handle *)h, "ds4_session_top_logprobs returned no finite logits");
    int written = 0;
    for (int i = 0; i < n && i < k; i++) {
        if (scores[i].id < 0) break;
        ids[written] = (int32_t)scores[i].id;
        logits[written] = scores[i].logit;
        written++;
    }
    free(scores);
    return written;
}

int ds4s_eval_speculative(ds4s_handle *h, int32_t first_token, int budget, int32_t *out,
                          int cap) {
    if (!h || !out || cap <= 0 || budget <= 0) return -1;
    /* The engine writes at most `want` entries, and `want` is capped at the
     * buffer's size, so no depth the pin may grow to can overrun it. */
    int accepted[17];
    int want = cap < 17 ? cap : 17;
    if (want > budget) want = budget;
    char err[256] = {0};
    /* The port routes this to ds4_qwen4exp_mtp_cycle: it verifies [fed token,
     * draft] in one two-row pass, commits the draft only where it equals the
     * target's own greedy argmax, and rolls the seven qwen4exp state objects
     * back to the accepted boundary on a rejection. The cycle commits 1 or 2
     * tokens and never 0.
     *
     * eos_token -1 never matches: the benchmark commits an exact token count
     * and does not stop at end-of-sequence. */
    const int n = ds4_session_eval_speculative_argmax(h->session, (int)first_token, budget, -1,
                                                      accepted, want, err, sizeof(err));
    if (n <= 0) {
        set_err(h, err[0] ? err : "ds4_session_eval_speculative_argmax failed");
        return -1;
    }
    const int written = n < cap ? n : cap;
    for (int i = 0; i < written; i++) out[i] = (int32_t)accepted[i];
    return written;
}

void ds4s_spec_counters(const ds4s_handle *h, uint64_t *drafts, uint64_t *hits,
                        uint64_t *quenches, uint64_t *disagreements) {
    /* The port owns these. ds4_session_qwen4exp_spec_counters() returns
     * non-zero for a session that runs no qwen4exp cycle, and leaves `c` as
     * zeroed here, which is the honest reading for a leg that never drafted. */
    ds4_spec_counters c;
    memset(&c, 0, sizeof(c));
    if (h && h->session) (void)ds4_session_qwen4exp_spec_counters(h->session, &c);
    if (drafts) *drafts = c.drafts;
    if (hits) *hits = c.hits;
    if (quenches) *quenches = c.quenches;
    if (disagreements) *disagreements = c.verify_replay_disagreements;
}
