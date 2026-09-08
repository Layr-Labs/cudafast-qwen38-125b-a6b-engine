/* A synthetic ds4_shim.h engine, for proving the RESIDENT SERVER off a GPU.
 *
 * ds4-resident (harness/protocol-adapter/ds4_shim/ds4_resident.c) is the
 * process that owns the weights for a benchmark window. Everything it does
 * that this repository can go wrong at -- loading once, resetting the session
 * per phase, carrying counters, tearing down clean, refusing to be held by an
 * idle phase -- is socket and lifecycle behaviour, not inference. This file
 * supplies the ds4s_* symbols so the SAME server source links and runs with no
 * CUDA, no driver and no 77 GiB checkpoint, and tools/test-ds4-resident.sh
 * drives it end to end.
 *
 * THE TOKENS ARE NOT INFERENCE. They are a fixed function of the prefix, which
 * is what makes the test assert exact values. Nothing here is ever linked into
 * the scored binary: tools/ds4/build.sh links the resident against
 * libds4qwen.so, and only tools/test-ds4-resident.sh compiles this. */
#include "ds4_shim.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define STUB_VOCAB 1024
#define STUB_EOS 99

struct ds4s_handle {
    uint64_t state;
    int mtp_armed;
    char err[256];
};

static uint64_t g_drafts;
static uint64_t g_hits;
static uint64_t g_quenches;

/* One load per process, counted so the test can prove it. */
static int g_opens;

static uint64_t fold(uint64_t state, int32_t token) {
    return (state ^ (uint64_t)(uint32_t)token) * 1099511628211ull + 12345ull;
}

static int32_t frontier(const ds4s_handle *h) {
    return (int32_t)(h->state % (STUB_VOCAB - 2)) + 1;
}

ds4s_handle *ds4s_open(const char *model_path, const char *mtp_head_path, int mtp_draft_tokens,
                       int ctx_size, int n_threads) {
    (void)n_threads;
    if (!model_path || !model_path[0] || ctx_size <= 0) return NULL;
    g_opens++;
    fprintf(stderr, "stub-engine: OPEN #%d model=%s head=%s draft_tokens=%d ctx=%d\n", g_opens,
            model_path, mtp_head_path ? mtp_head_path : "(none)", mtp_draft_tokens, ctx_size);
    fflush(stderr);
    ds4s_handle *h = calloc(1, sizeof(*h));
    if (!h) return NULL;
    h->state = 0x9e3779b97f4a7c15ull;
    h->mtp_armed = mtp_draft_tokens >= 2 && mtp_head_path && mtp_head_path[0];
    return h;
}

void ds4s_close(ds4s_handle *h) {
    if (!h) return;
    fprintf(stderr, "stub-engine: CLOSE\n");
    fflush(stderr);
    free(h);
}

const char *ds4s_open_error(void) { return "stub-engine refused the open"; }

const char *ds4s_last_error(const ds4s_handle *h) { return h && h->err[0] ? h->err : ""; }

int ds4s_vocab_size(const ds4s_handle *h) { return h ? STUB_VOCAB : 0; }

int32_t ds4s_eos_token(const ds4s_handle *h) { return h ? STUB_EOS : -1; }

void ds4s_invalidate(ds4s_handle *h) {
    if (!h) return;
    h->state = 0x9e3779b97f4a7c15ull;
    fprintf(stderr, "stub-engine: INVALIDATE\n");
    fflush(stderr);
}

int ds4s_sync(ds4s_handle *h, const int32_t *tokens, size_t n) {
    if (!h || !tokens || n == 0) return -1;
    /* A sync is a full re-forward of the prefix from the invalidated state,
     * which is what makes a leaked prefix visible: the same prompt on a dirty
     * session would fold twice and land elsewhere. */
    h->state = 0x9e3779b97f4a7c15ull;
    for (size_t i = 0; i < n; i++) h->state = fold(h->state, tokens[i]);
    return 0;
}

int ds4s_eval(ds4s_handle *h, int32_t token) {
    if (!h) return -1;
    h->state = fold(h->state, token);
    return 0;
}

int32_t ds4s_argmax(const ds4s_handle *h) { return h ? frontier(h) : -1; }

int ds4s_top_logits(const ds4s_handle *h, int k, int32_t *ids, float *logits) {
    if (!h || !ids || !logits || k <= 0) return 0;
    const int32_t top = frontier(h);
    for (int i = 0; i < k; i++) {
        ids[i] = (top + i) % STUB_VOCAB;
        logits[i] = 8.5f - (float)i;
    }
    return k;
}

int ds4s_eval_speculative(ds4s_handle *h, int32_t first_token, int budget, int32_t *out, int cap) {
    if (!h || !out || cap <= 0 || budget <= 0) return -1;
    if (ds4s_eval(h, first_token) != 0) return -1;
    out[0] = first_token;
    int n = 1;
    /* An armed drafter proposes on every cycle and is accepted on the even
     * ones, so a leg exercises both the hit and the miss path. */
    if (h->mtp_armed) {
        g_drafts++;
        if (budget >= 2 && cap >= 2 && (h->state & 1u) == 0) {
            const int32_t drafted = frontier(h);
            if (ds4s_eval(h, drafted) != 0) return -1;
            out[n++] = drafted;
            g_hits++;
        }
    }
    return n;
}

void ds4s_spec_counters(const ds4s_handle *h, uint64_t *drafts, uint64_t *hits,
                        uint64_t *quenches, uint64_t *disagreements) {
    (void)h;
    if (drafts) *drafts = g_drafts;
    if (hits) *hits = g_hits;
    if (quenches) *quenches = g_quenches;
    /* The stub's cycle never rejects, so it never replays and never disagrees.
     * A fixed 0 keeps the wire field exercised end to end. */
    if (disagreements) *disagreements = 0;
}
