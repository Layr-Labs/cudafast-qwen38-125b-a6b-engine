/* ds4-resident: the ONE process that holds the model for a benchmark window.
 *
 * WHY IT EXISTS. benchd's CUDA residency is FreshPerPhase: it spawns a fresh
 * `cuda-engine` for warmup, timed prefill, timed decode and correctness. With
 * the engine linked in-process every one of those phases loaded the ~77 GiB
 * body again. This binary loads it ONCE, and the per-phase `cuda-engine`
 * processes connect to it as clients. The weights are resident exactly once,
 * for the whole window, and a phase costs a socket connect.
 *
 * WHAT IT IS NOT. It is not upstream's `ds4-server`. That server speaks
 * OpenAI/Anthropic chat over HTTP: text in, text out, no logprobs anywhere in
 * it, no teacher-forced eval, no per-cycle speculative accounting, and its own
 * sampler and stop-string machinery inside the reply path. None of the four
 * things the scored adapter needs survives that wire (docs/ds4-resident.md
 * carries the mapping table and the evidence). So this is the minimum that
 * carries the adapter's contract without loss: the ds4_shim.h C surface, one
 * verb per line, over a Unix socket, one client at a time.
 *
 * THE WIRE. NDJSON both ways over AF_UNIX/SOCK_STREAM. One request object per
 * line, one response object per line, strictly in order. Every response
 * carries "ok"; a failed one also carries "error" with the engine's own text.
 * The verbs are ds4_shim.h's functions, one for one:
 *
 *   hello              -> vocab size, EOS, whether the drafter is armed, the
 *                         window's identity, and the load epoch
 *   invalidate         -> ds4s_invalidate
 *   sync {tokens}      -> ds4s_sync            + ds4s_argmax
 *   eval {token}       -> ds4s_eval            + ds4s_argmax
 *   argmax             -> ds4s_argmax
 *   top_logits {k}     -> ds4s_top_logits
 *   eval_speculative {first_token,budget}
 *                      -> ds4s_eval_speculative + ds4s_argmax
 *   spec_run {first_token,count}
 *                      -> ds4s_eval_speculative + ds4s_argmax, cycle after
 *                         cycle until count tokens are committed, in ONE reply
 *                         carrying every cycle's tokens and frontier argmax
 *                         (offered in hello as "spec_run")
 *   spec_counters      -> ds4s_spec_counters
 *   bye                -> acknowledge, then close
 *
 * EAGER ARGMAX. Every state-advancing verb returns the frontier argmax with
 * its own reply. ds4s_argmax is a pure read of logits the call just left
 * ready, so returning it changes no semantics -- it removes one round trip per
 * decoded token from the timed window, which is the only reason it is there.
 *
 * PHASE RESET. Each accepted connection is one benchd phase. The server calls
 * ds4s_invalidate the moment it accepts, BEFORE the client's first byte, so a
 * phase can never inherit the previous phase's live prefix even if the client
 * forgets to drain. The client's own drain_to_zero invalidates again; that is
 * idempotent and deliberate.
 *
 * ONE CLIENT AT A TIME, deliberately. The scored series is single-stream and
 * benchd runs one phase at a time. A second connection waits in the listen
 * backlog rather than interleaving evals on a session that has exactly one
 * frontier.
 *
 * IDLE CEILING. A connection that sends nothing for DS4_RESIDENT_PHASE_TIMEOUT_S
 * is dropped and the server goes back to accepting, so a wedged phase cannot
 * hold the window's weights hostage.
 *
 * THE REQUEST PARSER is deliberately small: it reads the field shapes THIS
 * repository's client emits (harness/protocol-adapter/src/resident.rs) and
 * nothing else. Requests carry no free-form strings, so a scan for a quoted
 * key cannot be fooled by one. A line it cannot read is an ok:false reply, not
 * a crash and not a guess.
 *
 * Environment:
 *   DS4_RESIDENT_SOCKET          the Unix socket path to bind (required)
 *   DS4_MODEL                    first GGUF shard (required)
 *   DS4_MTP_PATH                 native MTP draft head, or unset for serial
 *   DS4_MTP_DRAFT_TOKENS         1 = serial, N+1 = draft depth N (default 1)
 *   DS4_CTX_SIZE                 session context in tokens (default 8192)
 *   DS4_THREADS                  host threads (default 0 = engine default)
 *   DS4_ENGINE_IDENT             the identity string the hello echoes
 *   DS4_RESIDENT_READY_FILE      touched once the model is open AND the socket
 *                                is listening; the serve script polls it
 *   DS4_RESIDENT_PHASE_TIMEOUT_S idle ceiling on one connection (default 1800)
 *   DS4_RESIDENT_NO_SPEC_RUN     1 = do not offer spec_run, so clients drive one
 *                                speculative cycle per request (A/B valve)
 *   DS4_RESIDENT_NO_AB           1 = skip the load-time A/B self-profile
 *                                (self_profile_ab below); the identity is then
 *                                exactly what it was without it
 */
#include "ds4_shim.h"

#include <errno.h>
#include <inttypes.h>
#include <poll.h>
#include <signal.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

#define RESIDENT_MAX_TOP_K 64
#define RESIDENT_MAX_SPEC 17
#define RESIDENT_MAX_RUN 65536

static volatile sig_atomic_t g_stop;
static char g_socket_path[512];

static void log_line(const char *fmt, ...) __attribute__((format(printf, 1, 2)));

static void log_line(const char *fmt, ...) {
    va_list ap;
    fputs("ds4-resident: ", stderr);
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
    fflush(stderr);
}

static void on_signal(int sig) {
    (void)sig;
    g_stop = 1;
}

/* --- environment ---------------------------------------------------------- */

static const char *env_or(const char *name, const char *fallback) {
    const char *v = getenv(name);
    return (v && v[0]) ? v : fallback;
}

static int env_int(const char *name, int fallback) {
    const char *v = getenv(name);
    if (!v || !v[0]) return fallback;
    char *end = NULL;
    const long parsed = strtol(v, &end, 10);
    if (end == v || (end && *end)) {
        log_line("%s=\"%s\" is not an integer", name, v);
        exit(2);
    }
    return (int)parsed;
}

/* --- the request parser --------------------------------------------------- */

/* Point at the character after `"key":` in `line`, or NULL. */
static const char *field(const char *line, const char *key) {
    char pattern[64];
    const int n = snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    if (n < 0 || (size_t)n >= sizeof(pattern)) return NULL;
    const char *at = strstr(line, pattern);
    if (!at) return NULL;
    at += (size_t)n;
    while (*at == ' ') at++;
    if (*at != ':') return NULL;
    at++;
    while (*at == ' ') at++;
    return at;
}

/* Copy the string value of `key` into `out`. Returns 0 on success. */
static int field_string(const char *line, const char *key, char *out, size_t outlen) {
    const char *at = field(line, key);
    if (!at || *at != '"') return -1;
    at++;
    size_t i = 0;
    while (*at && *at != '"') {
        if (i + 1 >= outlen) return -1;
        out[i++] = *at++;
    }
    if (*at != '"') return -1;
    out[i] = '\0';
    return 0;
}

/* Read the integer value of `key`. Returns 0 on success. */
static int field_int(const char *line, const char *key, long long *out) {
    const char *at = field(line, key);
    if (!at) return -1;
    char *end = NULL;
    const long long v = strtoll(at, &end, 10);
    if (end == at) return -1;
    *out = v;
    return 0;
}

/* Read the integer array value of `key` into a freshly allocated buffer.
 * Returns the count, or -1. `*out` is set only on success (it may be NULL for
 * an empty array). */
static long field_int_array(const char *line, const char *key, int32_t **out) {
    const char *at = field(line, key);
    *out = NULL;
    if (!at || *at != '[') return -1;
    at++;
    size_t cap = 64, n = 0;
    int32_t *buf = malloc(cap * sizeof(*buf));
    if (!buf) return -1;
    while (*at && *at != ']') {
        while (*at == ' ' || *at == ',') at++;
        if (*at == ']' || !*at) break;
        char *end = NULL;
        const long long v = strtoll(at, &end, 10);
        if (end == at) { free(buf); return -1; }
        if (n == cap) {
            cap *= 2;
            int32_t *grown = realloc(buf, cap * sizeof(*buf));
            if (!grown) { free(buf); return -1; }
            buf = grown;
        }
        buf[n++] = (int32_t)v;
        at = end;
        while (*at == ' ') at++;
    }
    if (*at != ']') { free(buf); return -1; }
    *out = buf;
    return (long)n;
}

/* --- the response writer -------------------------------------------------- */

/* A growable line buffer, used for both directions. */
typedef struct {
    char *data;
    size_t len;
    size_t cap;
} buf;

static bool buf_reserve(buf *b, size_t extra) {
    if (b->len + extra + 1 <= b->cap) return true;
    size_t cap = b->cap ? b->cap : 1024;
    while (cap < b->len + extra + 1) cap *= 2;
    char *grown = realloc(b->data, cap);
    if (!grown) return false;
    b->data = grown;
    b->cap = cap;
    return true;
}

static bool buf_puts(buf *b, const char *s) {
    const size_t n = strlen(s);
    if (!buf_reserve(b, n)) return false;
    memcpy(b->data + b->len, s, n);
    b->len += n;
    b->data[b->len] = '\0';
    return true;
}

static bool buf_printf(buf *b, const char *fmt, ...) {
    va_list ap;
    char scratch[512];
    va_start(ap, fmt);
    const int n = vsnprintf(scratch, sizeof(scratch), fmt, ap);
    va_end(ap);
    if (n < 0 || (size_t)n >= sizeof(scratch)) return false;
    return buf_puts(b, scratch);
}

/* JSON-escape `s` into `b`, quotes included. The only strings this server
 * emits are engine error text and the identity, so the escape set is the one
 * JSON requires plus a control-character fallback. */
static bool buf_json_string(buf *b, const char *s) {
    if (!buf_puts(b, "\"")) return false;
    for (const unsigned char *p = (const unsigned char *)(s ? s : ""); *p; p++) {
        bool ok;
        switch (*p) {
        case '"':  ok = buf_puts(b, "\\\""); break;
        case '\\': ok = buf_puts(b, "\\\\"); break;
        case '\n': ok = buf_puts(b, "\\n"); break;
        case '\r': ok = buf_puts(b, "\\r"); break;
        case '\t': ok = buf_puts(b, "\\t"); break;
        default:
            ok = (*p < 0x20) ? buf_printf(b, "\\u%04x", *p)
                             : buf_printf(b, "%c", *p);
            break;
        }
        if (!ok) return false;
    }
    return buf_puts(b, "\"");
}

static bool write_all(int fd, const char *data, size_t len) {
    while (len > 0) {
        const ssize_t n = write(fd, data, len);
        if (n < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        data += n;
        len -= (size_t)n;
    }
    return true;
}

static bool send_line(int fd, buf *b) {
    if (!buf_puts(b, "\n")) return false;
    return write_all(fd, b->data, b->len);
}

static bool send_error(int fd, const char *why) {
    buf out = {0};
    bool ok = buf_puts(&out, "{\"ok\":false,\"error\":") && buf_json_string(&out, why) &&
              buf_puts(&out, "}") && send_line(fd, &out);
    free(out.data);
    return ok;
}

/* --- the verbs ------------------------------------------------------------ */

typedef struct {
    ds4s_handle *h;
    const char *ident;
    const char *model_path;
    const char *mtp_head_path;
    int draft_tokens;
    int ctx_size;
    uint64_t load_epoch;
    bool spec_run;
} resident;

/* Serve one request line. Returns 1 to keep the connection, 0 to close it,
 * -1 on a write failure. */
static int serve_line(const resident *r, int fd, const char *line) {
    char op[32];
    if (field_string(line, "op", op, sizeof(op)) != 0) {
        return send_error(fd, "request carries no \"op\" string") ? 1 : -1;
    }

    buf out = {0};
    bool ok = true;
    int keep = 1;

    if (!strcmp(op, "hello")) {
        ok = buf_printf(&out,
                        "{\"ok\":true,\"vocab_size\":%d,\"eos_token\":%d,\"mtp_armed\":%s,"
                        "\"draft_tokens\":%d,\"ctx_size\":%d,\"load_epoch\":%" PRIu64 ","
                        "\"spec_run\":%s,",
                        ds4s_vocab_size(r->h), (int)ds4s_eos_token(r->h),
                        r->draft_tokens >= 2 ? "true" : "false", r->draft_tokens, r->ctx_size,
                        r->load_epoch, r->spec_run ? "true" : "false") &&
             buf_puts(&out, "\"ident\":") && buf_json_string(&out, r->ident) &&
             buf_puts(&out, ",\"model_path\":") && buf_json_string(&out, r->model_path) &&
             buf_puts(&out, ",\"mtp_head_path\":") && buf_json_string(&out, r->mtp_head_path) &&
             buf_puts(&out, "}");
    } else if (!strcmp(op, "invalidate")) {
        ds4s_invalidate(r->h);
        ok = buf_puts(&out, "{\"ok\":true}");
    } else if (!strcmp(op, "sync")) {
        int32_t *tokens = NULL;
        const long n = field_int_array(line, "tokens", &tokens);
        if (n < 0) {
            free(out.data);
            free(tokens);
            return send_error(fd, "sync needs a \"tokens\" integer array") ? 1 : -1;
        }
        const int rc = ds4s_sync(r->h, tokens, (size_t)n);
        free(tokens);
        if (rc != 0) {
            free(out.data);
            return send_error(fd, ds4s_last_error(r->h)) ? 1 : -1;
        }
        ok = buf_printf(&out, "{\"ok\":true,\"token\":%d}", (int)ds4s_argmax(r->h));
    } else if (!strcmp(op, "eval")) {
        long long token = 0;
        if (field_int(line, "token", &token) != 0) {
            free(out.data);
            return send_error(fd, "eval needs a \"token\" integer") ? 1 : -1;
        }
        if (ds4s_eval(r->h, (int32_t)token) != 0) {
            free(out.data);
            return send_error(fd, ds4s_last_error(r->h)) ? 1 : -1;
        }
        ok = buf_printf(&out, "{\"ok\":true,\"token\":%d}", (int)ds4s_argmax(r->h));
    } else if (!strcmp(op, "argmax")) {
        ok = buf_printf(&out, "{\"ok\":true,\"token\":%d}", (int)ds4s_argmax(r->h));
    } else if (!strcmp(op, "top_logits")) {
        long long k = 0;
        if (field_int(line, "k", &k) != 0 || k <= 0 || k > RESIDENT_MAX_TOP_K) {
            free(out.data);
            return send_error(fd, "top_logits needs a \"k\" integer in 1..64") ? 1 : -1;
        }
        int32_t ids[RESIDENT_MAX_TOP_K];
        float logits[RESIDENT_MAX_TOP_K];
        const int n = ds4s_top_logits(r->h, (int)k, ids, logits);
        ok = buf_puts(&out, "{\"ok\":true,\"ids\":[");
        for (int i = 0; ok && i < n; i++) ok = buf_printf(&out, "%s%d", i ? "," : "", (int)ids[i]);
        ok = ok && buf_puts(&out, "],\"logits\":[");
        /* %.9g round-trips a float exactly, so the gate's logits cross the
         * socket without a rounding step of the wire's own invention. */
        for (int i = 0; ok && i < n; i++)
            ok = buf_printf(&out, "%s%.9g", i ? "," : "", (double)logits[i]);
        ok = ok && buf_puts(&out, "]}");
    } else if (!strcmp(op, "eval_speculative")) {
        long long first = 0, budget = 0;
        if (field_int(line, "first_token", &first) != 0 ||
            field_int(line, "budget", &budget) != 0) {
            free(out.data);
            return send_error(fd, "eval_speculative needs \"first_token\" and \"budget\"") ? 1 : -1;
        }
        int32_t committed[RESIDENT_MAX_SPEC];
        const int n = ds4s_eval_speculative(r->h, (int32_t)first, (int)budget, committed,
                                            RESIDENT_MAX_SPEC);
        if (n < 0) {
            free(out.data);
            return send_error(fd, ds4s_last_error(r->h)) ? 1 : -1;
        }
        ok = buf_puts(&out, "{\"ok\":true,\"tokens\":[");
        for (int i = 0; ok && i < n; i++)
            ok = buf_printf(&out, "%s%d", i ? "," : "", (int)committed[i]);
        ok = ok && buf_printf(&out, "],\"token\":%d}", (int)ds4s_argmax(r->h));
    } else if (!strcmp(op, "spec_run")) {
        long long first = 0, count = 0;
        if (!r->spec_run) {
            free(out.data);
            return send_error(fd, "spec_run is not offered (DS4_RESIDENT_NO_SPEC_RUN)") ? 1 : -1;
        }
        if (field_int(line, "first_token", &first) != 0 || field_int(line, "count", &count) != 0 ||
            count <= 0 || count > RESIDENT_MAX_RUN) {
            free(out.data);
            return send_error(fd, "spec_run needs \"first_token\" and a \"count\" in 1..65536") ? 1
                                                                                               : -1;
        }
        /* THE WHOLE FREE RUN IN ONE REQUEST. This is the client's per-cycle
         * loop moved to this side of the socket: feed the pending token with
         * the budget still wanted, record the committed tokens and the frontier
         * argmax the cycle left, and feed that frontier next. Every cycle's
         * tokens and frontier go back, so the client checks and assembles
         * exactly what one request per cycle would have given it. What goes
         * away is the socket round trip after each cycle, which the GPU waited
         * through idle. A cycle the client refuses (nothing committed, or not
         * starting with the fed token) ends the loop, so the refusal follows
         * the same cycle it always did. */
        buf frontiers = {0};
        int32_t pending = (int32_t)first;
        long long produced = 0;
        ok = buf_puts(&out, "{\"ok\":true,\"rounds\":[");
        for (int cycle = 0; ok && produced < count; cycle++) {
            int32_t committed[RESIDENT_MAX_SPEC];
            const int n = ds4s_eval_speculative(r->h, pending, (int)(count - produced), committed,
                                                RESIDENT_MAX_SPEC);
            if (n < 0) {
                free(out.data);
                free(frontiers.data);
                return send_error(fd, ds4s_last_error(r->h)) ? 1 : -1;
            }
            const int32_t frontier = ds4s_argmax(r->h);
            ok = buf_puts(&out, cycle ? ",[" : "[");
            for (int i = 0; ok && i < n; i++)
                ok = buf_printf(&out, "%s%d", i ? "," : "", (int)committed[i]);
            ok = ok && buf_puts(&out, "]") &&
                 buf_printf(&frontiers, "%s%d", cycle ? "," : "", (int)frontier);
            if (n == 0 || committed[0] != pending) break;
            produced += n;
            pending = frontier;
        }
        ok = ok && buf_puts(&out, "],\"frontiers\":[") &&
             (frontiers.len == 0 || buf_puts(&out, frontiers.data)) &&
             buf_printf(&out, "],\"token\":%d}", (int)ds4s_argmax(r->h));
        free(frontiers.data);
    } else if (!strcmp(op, "spec_counters")) {
        uint64_t drafts = 0, hits = 0, quenches = 0, disagreements = 0;
        ds4s_spec_counters(r->h, &drafts, &hits, &quenches, &disagreements);
        ok = buf_printf(&out,
                        "{\"ok\":true,\"drafts\":%" PRIu64 ",\"hits\":%" PRIu64
                        ",\"quenches\":%" PRIu64 ",\"disagreements\":%" PRIu64 "}",
                        drafts, hits, quenches, disagreements);
    } else if (!strcmp(op, "bye")) {
        ok = buf_puts(&out, "{\"ok\":true}");
        keep = 0;
    } else {
        free(out.data);
        char why[96];
        snprintf(why, sizeof(why), "unknown op \"%s\"", op);
        return send_error(fd, why) ? 1 : -1;
    }

    if (!ok) {
        free(out.data);
        return send_error(fd, "the server could not build its response") ? 1 : -1;
    }
    const bool sent = send_line(fd, &out);
    free(out.data);
    return sent ? keep : -1;
}

/* --- one phase ------------------------------------------------------------ */

/* Serve one accepted connection until EOF, `bye`, the idle ceiling, or a write
 * failure. The connection IS the phase, so it starts with an invalidate. */
static void serve_connection(const resident *r, int fd, int timeout_s) {
    ds4s_invalidate(r->h);

    buf in = {0};
    for (;;) {
        char *nl = memchr(in.data, '\n', in.len);
        if (!nl) {
            struct pollfd p = {.fd = fd, .events = POLLIN};
            const int ready = poll(&p, 1, timeout_s > 0 ? timeout_s * 1000 : -1);
            if (ready == 0) {
                log_line("phase idle for %ds; dropping the connection", timeout_s);
                break;
            }
            if (ready < 0) {
                if (errno == EINTR) {
                    if (g_stop) break;
                    continue;
                }
                log_line("poll failed: %s", strerror(errno));
                break;
            }
            if (!buf_reserve(&in, 65536)) break;
            const ssize_t n = read(fd, in.data + in.len, in.cap - in.len - 1);
            if (n < 0) {
                if (errno == EINTR) continue;
                log_line("read failed: %s", strerror(errno));
                break;
            }
            if (n == 0) break; /* the phase's client exited */
            in.len += (size_t)n;
            in.data[in.len] = '\0';
            continue;
        }
        *nl = '\0';
        const size_t consumed = (size_t)(nl - in.data) + 1;
        const int keep = serve_line(r, fd, in.data);
        memmove(in.data, in.data + consumed, in.len - consumed);
        in.len -= consumed;
        in.data[in.len] = '\0';
        if (keep <= 0) break;
    }
    free(in.data);
}

/* --- load-time A/B self-profile ------------------------------------------- */

/* WHY.  The ranked box runs no profiler and the resident's stderr never
 * reaches the run log; the one channel out is the identity this process
 * publishes, which benchd seals as engine_backend.  So, once, after the one
 * load and BEFORE the socket binds, time the scored decode path with each of
 * six decode arms stood down in turn and append the per-round means to the
 * identity: an A/B of every arm on the ranked hardware, in one process, from
 * one binary.
 *
 * WHY NO PHASE CAN SEE IT.  (1) It runs before bind(), so no client can
 * connect while it runs.  (2) It ends with ds4s_ab_set(0) -- the engine's arm
 * mask back at the shipped value, and 0 always lands -- then ds4s_rewarm, the
 * shim's own open-time warm-up replayed from an empty graph cache, then
 * ds4s_invalidate.  (3) Every phase starts with ds4s_invalidate anyway
 * (serve_connection).  (4) Every mask switch retires every captured decode
 * graph, so no executable captured under a stood-down arm survives.
 *
 * Optional entries: the CI stub engine (tools/ds4/resident-stub-engine.c) has
 * neither, and a resident linked against it must still link and serve -- it
 * then skips the profile and publishes the identity it always did.  On ELF
 * (CI, the box, where they come from libds4qwen.so) they are weak references,
 * which link as NULL when nothing defines them.  Mach-O's static linker
 * refuses an undefined weak reference, and tools/test-ds4-resident.sh and
 * tools/three-flows-dry-run.sh build this file over the stub on a macOS
 * laptop, so there they are looked up at run time instead and this file never
 * names them to the linker. */
typedef int (*ab_set_fn)(uint32_t off_mask);
typedef int (*ab_rewarm_fn)(ds4s_handle *h);
static ab_set_fn g_ab_set;
static ab_rewarm_fn g_ab_rewarm;
#if defined(__APPLE__)
#include <dlfcn.h>
static void ab_resolve(void) {
    void *p = dlsym(RTLD_DEFAULT, "ds4s_ab_set");
    memcpy(&g_ab_set, &p, sizeof(p));
    p = dlsym(RTLD_DEFAULT, "ds4s_rewarm");
    memcpy(&g_ab_rewarm, &p, sizeof(p));
}
#else
#pragma weak ds4s_ab_set
#pragma weak ds4s_rewarm
static void ab_resolve(void) {
    g_ab_set = ds4s_ab_set;
    g_ab_rewarm = ds4s_rewarm;
}
#endif

static double mono_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* One decode step the way a phase's spec_run takes it: a speculative cycle fed
 * the pending token, whose frontier argmax is the next pending token.  Writes
 * the committed tokens to out[0..n) and returns n, or -1 on failure. */
static int ab_step(ds4s_handle *h, int32_t *pending, int32_t *out) {
    const int n = ds4s_eval_speculative(h, *pending, RESIDENT_MAX_RUN, out, RESIDENT_MAX_SPEC);
    if (n <= 0 || out[0] != *pending) return -1;
    *pending = ds4s_argmax(h);
    return n;
}

enum {
    AB_PROMPT = 256,  /* synthetic prompt rows (one prefill per run) */
    AB_WARM = 6,      /* rounds per run that re-warm and recapture the graphs
                       * (both GDN replay parities and the head graphs need
                       * two visits each before they replay, so captures stay
                       * out of the timed window) */
    AB_TIMED = 16,    /* timed rounds per run */
    AB_CFGS = 6,      /* C0 all on, C1 all off, C2..C5 leave-one-group-out */
    AB_PASSES = 2,    /* C0..C5, then C5..C0: linear drift cancels */
    AB_RUNS = AB_CFGS * AB_PASSES,
};

/* C0 = every arm on (the scored path), C1 = every arm off, C2..C5 = one arm
 * group off each: the GDN replay PDL chain, every early-trigger edge class
 * (HC up, routed gate/up, roll projection, routed down, QSA split chain), the
 * early shared-expert fork, and the QSA indexer fork.  The key names the
 * group stood down.  The narrow router launch is only in C1 (it measured
 * neutral on its own in PR #5307) to keep the load-time GPU work short. */
#define DS4S_AB_TRIG_GROUP (DS4S_AB_TRIG_HCUP | DS4S_AB_TRIG_GU | \
                            DS4S_AB_TRIG_ROLL | DS4S_AB_TRIG_DOWN | \
                            DS4S_AB_TRIG_QSA)
static const uint32_t ab_mask[AB_CFGS] = {
    0u,
    DS4S_AB_ALL,
    DS4S_AB_GDN_PDL,
    DS4S_AB_TRIG_GROUP,
    DS4S_AB_EARLY_FORK,
    DS4S_AB_IDX_FORK,
};
static const char *const ab_key[AB_CFGS] = {
    "on", "off", "-gdn", "-trig", "-efk", "-idx",
};

typedef struct {
    int cfg;
    int pass;
    double ms_per_round;
    int committed;
    uint64_t hash;
    double wall_s;
} ab_run;

/* 64-bit FNV-1a over the committed ids, four little-endian bytes each. */
static uint64_t fnv1a_tokens(uint64_t hash, const int32_t *ids, int n) {
    for (int i = 0; i < n; i++) {
        const uint32_t v = (uint32_t)ids[i];
        for (int b = 0; b < 4; b++) {
            hash ^= (uint64_t)((v >> (8 * b)) & 0xffu);
            hash *= 1099511628211ull;
        }
    }
    return hash;
}

/* One run: mask, full reset, the prompt, AB_WARM untimed rounds (the first
 * rounds after a mask switch warm and capture the decode graphs afresh), then
 * AB_TIMED timed rounds.  Returns NULL on success or the failing stage. */
static const char *ab_one_run(ds4s_handle *h, const int32_t *prompt, uint32_t mask,
                              ab_run *run) {
    const double w0 = mono_s();
    if (g_ab_set(mask) != 0) return "set";
    ds4s_invalidate(h);
    if (ds4s_sync(h, prompt, AB_PROMPT) != 0) return "sync";
    int32_t pending = ds4s_argmax(h);
    int32_t out[RESIDENT_MAX_SPEC];
    for (int i = 0; i < AB_WARM; i++)
        if (ab_step(h, &pending, out) < 0) return "warm";
    uint64_t hash = 14695981039346656037ull;
    int committed = 0;
    const double t0 = mono_s();
    for (int i = 0; i < AB_TIMED; i++) {
        const int n = ab_step(h, &pending, out);
        if (n < 0) return "timed";
        committed += n;
        hash = fnv1a_tokens(hash, out, n);
    }
    const double t1 = mono_s();
    run->ms_per_round = (t1 - t0) * 1e3 / AB_TIMED;
    run->committed = committed;
    run->hash = hash;
    run->wall_s = t1 - w0;
    return NULL;
}

static const char *self_profile_ab_run(ds4s_handle *h, ab_run *runs, int *n_runs) {
    static int32_t prompt[AB_PROMPT];
    const int vocab = ds4s_vocab_size(h);
    if (vocab < 8192) return "vocab";
    /* #5300's fixed pseudo-random prompt of ordinary token ids (xorshift32
     * from the same seed), 512 rows: the decode it leads to routes experts
     * like any other text, and the timing needs no tokenizer. */
    uint32_t x = 0x9e3779b9u;
    for (int i = 0; i < AB_PROMPT; i++) {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        prompt[i] = (int32_t)(1000u + x % (uint32_t)(vocab / 2));
    }
    *n_runs = 0;
    for (int pass = 0; pass < AB_PASSES; pass++) {
        for (int k = 0; k < AB_CFGS; k++) {
            const int cfg = (pass & 1) ? AB_CFGS - 1 - k : k;
            ab_run *run = &runs[*n_runs];
            memset(run, 0, sizeof(*run));
            run->cfg = cfg;
            run->pass = pass;
            const char *stage = ab_one_run(h, prompt, ab_mask[cfg], run);
            if (stage) return stage;
            (*n_runs)++;
        }
    }
    return NULL;
}

/* Run once after the load and before the socket binds; see the block comment
 * above.  Writes "ab[...]" (or "ab[fail:<stage>]") to out, or leaves it empty
 * when the profile does not apply: no MTP head (draft_tokens < 2), an engine
 * without the entries (the CI stub), or DS4_RESIDENT_NO_AB=1. */
static void self_profile_ab(ds4s_handle *h, int draft_tokens, char *out, size_t cap) {
    out[0] = '\0';
    ab_resolve();
    if (draft_tokens < 2 || !g_ab_set || !g_ab_rewarm) return;
    if (env_int("DS4_RESIDENT_NO_AB", 0) != 0) {
        log_line("A/B self-profile skipped (DS4_RESIDENT_NO_AB)");
        return;
    }
    const double t0 = mono_s();
    ab_run runs[AB_RUNS];
    int n_runs = 0;
    const char *stage = self_profile_ab_run(h, runs, &n_runs);
    const double t_prof = mono_s() - t0;

    /* ALWAYS, success or not: the shipped arms, a warm cache, a clean session. */
    const int restore_rc = g_ab_set(0);
    const double t1 = mono_s();
    const int rewarm_rc = g_ab_rewarm(h);
    ds4s_invalidate(h);
    const double t_rewarm = mono_s() - t1;
    if (!stage && restore_rc != 0) stage = "restore";
    if (!stage && rewarm_rc != 0) stage = "rewarm";

    log_line("A/B self-profile: %d run(s), prompt %d, warm %d, timed %d rounds per run",
             n_runs, (int)AB_PROMPT, (int)AB_WARM, (int)AB_TIMED);
    log_line("  run pass cfg   mask  ms/round committed hash             wall_s");
    for (int i = 0; i < n_runs; i++) {
        const ab_run *r = &runs[i];
        log_line("  %3d %4d %-5s 0x%02x %9.3f %9d %016" PRIx64 " %6.2f", i, r->pass,
                 ab_key[r->cfg], (unsigned)ab_mask[r->cfg], r->ms_per_round, r->committed,
                 r->hash, r->wall_s);
    }
    log_line("A/B self-profile wall: %.1f s profile + %.1f s restore/rewarm = %.1f s%s%s",
             t_prof, t_rewarm, t_prof + t_rewarm, stage ? "; FAILED at " : "",
             stage ? stage : "");

    if (stage) {
        snprintf(out, cap, "ab[fail:%s]", stage);
        return;
    }

    double sum[AB_CFGS] = {0};
    double pass_ms[AB_CFGS][AB_PASSES] = {{0}};
    int same = 1;
    long total_committed = 0;
    for (int i = 0; i < n_runs; i++) {
        sum[runs[i].cfg] += runs[i].ms_per_round;
        pass_ms[runs[i].cfg][runs[i].pass] = runs[i].ms_per_round;
        total_committed += runs[i].committed;
        if (runs[i].hash != runs[0].hash) same = 0;
    }
    /* nz: the largest pass-to-pass difference of any one config -- the noise
     * floor a difference between two configs has to clear. */
    double nz = 0.0;
    for (int c = 0; c < AB_CFGS; c++) {
        const double d = pass_ms[c][0] - pass_ms[c][1];
        if (d > nz) nz = d;
        if (-d > nz) nz = -d;
    }
    size_t len = 0;
    int w = snprintf(out, cap, "ab[n=%d r=%d", (int)AB_PROMPT, (int)AB_TIMED);
    for (int c = 0; w >= 0 && (size_t)w < cap - len && c < AB_CFGS; c++) {
        len += (size_t)w;
        w = snprintf(out + len, cap - len, " %s=%.3f", ab_key[c], sum[c] / AB_PASSES);
    }
    if (w >= 0 && (size_t)w < cap - len) {
        len += (size_t)w;
        w = snprintf(out + len, cap - len, " same=%d t/rd=%.2f nz=%.3f]", same,
                     (double)total_committed / ((double)n_runs * AB_TIMED), nz);
    }
    if (w < 0 || (size_t)w >= cap - len) snprintf(out, cap, "ab[fail:format]");
}

/* --- main ----------------------------------------------------------------- */

static void unlink_socket(void) {
    if (g_socket_path[0]) unlink(g_socket_path);
}

int main(void) {
    setvbuf(stderr, NULL, _IOLBF, 0);

    const char *socket_path = env_or("DS4_RESIDENT_SOCKET", NULL);
    if (!socket_path) {
        log_line("DS4_RESIDENT_SOCKET is unset; it names the Unix socket to bind");
        return 2;
    }
    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    if (strlen(socket_path) >= sizeof(addr.sun_path)) {
        log_line("DS4_RESIDENT_SOCKET is longer than the %zu bytes a Unix socket path holds",
                 sizeof(addr.sun_path) - 1);
        return 2;
    }
    memcpy(addr.sun_path, socket_path, strlen(socket_path));

    const char *model_path = env_or("DS4_MODEL", NULL);
    if (!model_path) {
        log_line("DS4_MODEL is unset; export the path of the first GGUF shard");
        return 2;
    }
    const char *mtp_head_path = env_or("DS4_MTP_PATH", "");
    const int draft_tokens = env_int("DS4_MTP_DRAFT_TOKENS", 1);
    const int ctx_size = env_int("DS4_CTX_SIZE", 8192);
    const int n_threads = env_int("DS4_THREADS", 0);
    const int timeout_s = env_int("DS4_RESIDENT_PHASE_TIMEOUT_S", 1800);
    const int no_spec_run = env_int("DS4_RESIDENT_NO_SPEC_RUN", 0);
    const char *ident = env_or("DS4_ENGINE_IDENT", "ds4");
    const char *ready_file = env_or("DS4_RESIDENT_READY_FILE", NULL);

    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = on_signal;
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);
    signal(SIGPIPE, SIG_IGN);

    /* THE ONE LOAD. Everything after this is a socket connect. */
    log_line("loading %s (draft_tokens=%d ctx=%d)%s%s", model_path, draft_tokens, ctx_size,
             mtp_head_path[0] ? " mtp_head=" : " serial (no draft head)", mtp_head_path);
    ds4s_handle *h = ds4s_open(model_path, mtp_head_path[0] ? mtp_head_path : NULL, draft_tokens,
                               ctx_size, n_threads);
    if (!h) {
        log_line("the engine did not open: %s", ds4s_open_error());
        return 1;
    }
    log_line("model loaded once for this window (vocab=%d eos=%d)", ds4s_vocab_size(h),
             (int)ds4s_eos_token(h));

    /* Append the device's occupancy and bandwidth limits to the identity the
     * hello echoes, so they reach the run's metrics.  Shared memory per SM and
     * peak DRAM bandwidth decide whether a decode kernel is occupancy-capped
     * or bandwidth-capped, and neither is observable on a box whose profiler
     * refuses to attach.  This is a pure read of device properties, done once
     * HERE -- after the one load, before the socket binds -- so no timed phase
     * can see it.  On failure the string is empty and the identity is exactly
     * what it was, which keeps the `ds4-resident load_epoch=` prefix and the
     * ident that benchd seals unchanged for a non-CUDA build. */
    char ident_buf[768];
    const char *limits = ds4s_hw_limits();
    if (limits && limits[0]) {
        const int n = snprintf(ident_buf, sizeof(ident_buf), "%s %s", ident, limits);
        if (n > 0 && (size_t)n < sizeof(ident_buf)) ident = ident_buf;
        log_line("device limits: %s", limits);
    }

    /* The load-time A/B self-profile (self_profile_ab), appended AFTER the
     * pieces above, which stay exactly as they were.  Still before bind(), so
     * no phase can see it; the session and the engine's arm mask are back at
     * the shipped state when it returns.  Empty when it does not apply. */
    char ab[640];
    self_profile_ab(h, draft_tokens, ab, sizeof(ab));
    char ident_ab_buf[sizeof(ident_buf) + sizeof(ab) + 1];
    if (ab[0]) {
        const int n = snprintf(ident_ab_buf, sizeof(ident_ab_buf), "%s %s", ident, ab);
        if (n > 0 && (size_t)n < sizeof(ident_ab_buf)) ident = ident_ab_buf;
        log_line("A/B self-profile: %s", ab);
    }

    /* Bind only after the load, so a connect that succeeds means the weights
     * are already resident and a phase never waits on the loader. */
    const int listener = socket(AF_UNIX, SOCK_STREAM, 0);
    if (listener < 0) {
        log_line("socket() failed: %s", strerror(errno));
        ds4s_close(h);
        return 1;
    }
    unlink(socket_path);
    if (bind(listener, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        log_line("bind(%s) failed: %s", socket_path, strerror(errno));
        close(listener);
        ds4s_close(h);
        return 1;
    }
    snprintf(g_socket_path, sizeof(g_socket_path), "%s", socket_path);
    /* The socket is the window's private control channel: owner only. */
    if (chmod(socket_path, 0600) != 0) {
        log_line("chmod(%s, 0600) failed: %s", socket_path, strerror(errno));
        unlink_socket();
        close(listener);
        ds4s_close(h);
        return 1;
    }
    if (listen(listener, 8) != 0) {
        log_line("listen() failed: %s", strerror(errno));
        unlink_socket();
        close(listener);
        ds4s_close(h);
        return 1;
    }

    if (ready_file) {
        FILE *fp = fopen(ready_file, "w");
        if (!fp) {
            log_line("cannot write the ready file %s: %s", ready_file, strerror(errno));
            unlink_socket();
            close(listener);
            ds4s_close(h);
            return 1;
        }
        fprintf(fp, "%s\n", socket_path);
        fclose(fp);
    }
    log_line("listening on %s; phases connect, the weights stay put", socket_path);
    log_line("spec_run %s", no_spec_run ? "off (DS4_RESIDENT_NO_SPEC_RUN): one cycle per request"
                                        : "on: a speculative free run is one request");

    const resident r = {
        .h = h,
        .ident = ident,
        .model_path = model_path,
        .mtp_head_path = mtp_head_path,
        .draft_tokens = draft_tokens,
        .ctx_size = ctx_size,
        .load_epoch = (uint64_t)getpid(),
        .spec_run = no_spec_run == 0,
    };

    uint64_t phase = 0;
    while (!g_stop) {
        const int fd = accept(listener, NULL, NULL);
        if (fd < 0) {
            if (errno == EINTR) continue;
            log_line("accept() failed: %s", strerror(errno));
            break;
        }
        phase++;
        log_line("phase %" PRIu64 " connected", phase);
        serve_connection(&r, fd, timeout_s);
        close(fd);
        log_line("phase %" PRIu64 " closed", phase);
    }

    log_line("shutting down after %" PRIu64 " phase(s) on one load", phase);
    unlink_socket();
    close(listener);
    ds4s_close(h);
    if (ready_file) unlink(ready_file);
    return 0;
}
