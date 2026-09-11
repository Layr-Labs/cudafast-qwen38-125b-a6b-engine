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
#include <unistd.h>

#define RESIDENT_MAX_TOP_K 64
#define RESIDENT_MAX_SPEC 17

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
                        "\"draft_tokens\":%d,\"ctx_size\":%d,\"load_epoch\":%" PRIu64 ",",
                        ds4s_vocab_size(r->h), (int)ds4s_eos_token(r->h),
                        r->draft_tokens >= 2 ? "true" : "false", r->draft_tokens, r->ctx_size,
                        r->load_epoch) &&
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

    const resident r = {
        .h = h,
        .ident = ident,
        .model_path = model_path,
        .mtp_head_path = mtp_head_path,
        .draft_tokens = draft_tokens,
        .ctx_size = ctx_size,
        .load_epoch = (uint64_t)getpid(),
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
