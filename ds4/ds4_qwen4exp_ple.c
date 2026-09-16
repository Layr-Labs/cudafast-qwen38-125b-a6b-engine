/* Host side of the Qwen4-Exp per-layer embedding (PLE) n-gram table.
 * See ds4_qwen4exp_ple.h for the contract.
 *
 * This file is self-contained on purpose.  It walks the GGUF header itself
 * instead of reaching into the loader in ds4.c, because the table is opened
 * from a shard set that the engine mapping does not necessarily cover, and
 * because the tests must exercise it without a model. */

#include "ds4_qwen4exp_ple.h"

#include <fcntl.h>
#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

/* =========================================================================
 * Errors.
 * ========================================================================= */

static void ple_err(char *err, size_t err_size, const char *fmt, ...)
    __attribute__((format(printf, 3, 4)));

static void ple_err(char *err, size_t err_size, const char *fmt, ...) {
    if (!err || err_size == 0) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, err_size, fmt, ap);
    va_end(ap);
}

/* =========================================================================
 * GGUF header walk.
 * =========================================================================
 *
 * Read only, over a mapping.  Only the header is touched: tensor bytes stay
 * where they are. */

#define PLE_GGUF_MAGIC 0x46554747u /* "GGUF", little endian. */

enum {
    PLE_GGUF_UINT8   = 0,
    PLE_GGUF_INT8    = 1,
    PLE_GGUF_UINT16  = 2,
    PLE_GGUF_INT16   = 3,
    PLE_GGUF_UINT32  = 4,
    PLE_GGUF_INT32   = 5,
    PLE_GGUF_FLOAT32 = 6,
    PLE_GGUF_BOOL    = 7,
    PLE_GGUF_STRING  = 8,
    PLE_GGUF_ARRAY   = 9,
    PLE_GGUF_UINT64  = 10,
    PLE_GGUF_INT64   = 11,
    PLE_GGUF_FLOAT64 = 12,
};

/* GGUF tensor type of IQ4_NL. */
#define PLE_GGUF_TYPE_IQ4_NL 20

#define PLE_TABLE_TENSOR "per_layer_token_embd.weight"

typedef struct {
    const uint8_t *base;
    uint64_t       size;
    uint64_t       pos;
    bool           ok;
} ple_cursor;

typedef struct {
    const char *data;
    uint64_t    len;
} ple_str;

static bool ple_read(ple_cursor *c, void *dst, uint64_t n) {
    if (!c->ok) return false;
    if (n > c->size || c->pos > c->size - n) { c->ok = false; return false; }
    memcpy(dst, c->base + c->pos, (size_t)n);
    c->pos += n;
    return true;
}

static bool ple_skip(ple_cursor *c, uint64_t n) {
    if (!c->ok) return false;
    if (n > c->size || c->pos > c->size - n) { c->ok = false; return false; }
    c->pos += n;
    return true;
}

static bool ple_u32(ple_cursor *c, uint32_t *v) { return ple_read(c, v, 4); }
static bool ple_u64(ple_cursor *c, uint64_t *v) { return ple_read(c, v, 8); }

static bool ple_string(ple_cursor *c, ple_str *s) {
    uint64_t n = 0;
    if (!ple_u64(c, &n)) return false;
    if (n > c->size || c->pos > c->size - n) { c->ok = false; return false; }
    s->data = (const char *)(c->base + c->pos);
    s->len  = n;
    c->pos += n;
    return true;
}

static bool ple_str_is(ple_str s, const char *lit) {
    size_t n = strlen(lit);
    return s.len == (uint64_t)n && memcmp(s.data, lit, n) == 0;
}

static uint64_t ple_scalar_size(uint32_t type) {
    switch (type) {
    case PLE_GGUF_UINT8: case PLE_GGUF_INT8: case PLE_GGUF_BOOL:      return 1;
    case PLE_GGUF_UINT16: case PLE_GGUF_INT16:                        return 2;
    case PLE_GGUF_UINT32: case PLE_GGUF_INT32: case PLE_GGUF_FLOAT32: return 4;
    case PLE_GGUF_UINT64: case PLE_GGUF_INT64: case PLE_GGUF_FLOAT64: return 8;
    default:                                                          return 0;
    }
}

static bool ple_skip_value(ple_cursor *c, uint32_t type, int depth) {
    if (depth > 8) { c->ok = false; return false; }

    uint64_t scalar = ple_scalar_size(type);
    if (scalar != 0) return ple_skip(c, scalar);

    if (type == PLE_GGUF_STRING) {
        ple_str ignored;
        return ple_string(c, &ignored);
    }
    if (type == PLE_GGUF_ARRAY) {
        uint32_t item = 0;
        uint64_t len  = 0;
        if (!ple_u32(c, &item) || !ple_u64(c, &len)) return false;
        uint64_t item_size = ple_scalar_size(item);
        if (item_size != 0) {
            if (len > UINT64_MAX / item_size) { c->ok = false; return false; }
            return ple_skip(c, len * item_size);
        }
        for (uint64_t i = 0; i < len; i++) {
            if (!ple_skip_value(c, item, depth + 1)) return false;
        }
        return true;
    }
    c->ok = false;
    return false;
}

/* Decode one integer array element of any integer width, refusing negatives.
 * The pinned checkpoint stores the PLE arrays as uint64; the other widths are
 * accepted so a re-quantization that narrows them still loads. */
static bool ple_int_elem(ple_cursor *c, uint32_t type, uint64_t *out) {
    switch (type) {
    case PLE_GGUF_UINT32: {
        uint32_t v = 0;
        if (!ple_u32(c, &v)) return false;
        *out = v;
        return true;
    }
    case PLE_GGUF_INT32: {
        int32_t v = 0;
        if (!ple_read(c, &v, 4) || v < 0) return false;
        *out = (uint64_t)v;
        return true;
    }
    case PLE_GGUF_UINT64:
        return ple_u64(c, out);
    case PLE_GGUF_INT64: {
        int64_t v = 0;
        if (!ple_read(c, &v, 8) || v < 0) return false;
        *out = (uint64_t)v;
        return true;
    }
    default:
        return false;
    }
}

/* What one shard contributed. */
typedef struct {
    bool     has_ngram_size;      uint32_t ngram_size;
    bool     has_heads_per_ngram; uint32_t heads_per_ngram;
    bool     has_row_dim;         uint32_t row_dim;
    bool     has_eos;             uint64_t eos;
    bool     has_conv_kernel;     uint32_t conv_kernel;
    bool     has_layers;          uint64_t layers[DS4_PLE_MAX_LAYERS];
    uint64_t layer_count;
    bool     has_vocab;           uint64_t vocab;

    bool     has_multipliers;     uint64_t multipliers[DS4_PLE_MAX_NGRAM];
    uint64_t multiplier_count;
    bool     has_sizes;           uint64_t sizes[DS4_PLE_MAX_HEADS];
    uint64_t size_count;
    bool     has_offsets;         uint64_t offsets[DS4_PLE_MAX_HEADS];
    uint64_t offset_count;

    bool     has_tensor;
    size_t   tensor_path;
    uint64_t tensor_offset;       /* absolute byte offset in its shard */
    uint64_t tensor_dim0;
    uint64_t tensor_dim1;
    uint32_t tensor_type;
} ple_scan;

/* Read a bounded integer array into `dst`. `cap` is the room; the length is
 * reported so the caller can refuse a mismatch rather than silently clamp. */
static bool ple_int_array(ple_cursor *c, uint64_t *dst, size_t cap,
                          uint64_t *count_out) {
    uint32_t item = 0;
    uint64_t len  = 0;
    if (!ple_u32(c, &item) || !ple_u64(c, &len)) return false;
    *count_out = len;
    if (len > cap) return false;
    for (uint64_t i = 0; i < len; i++) {
        if (!ple_int_elem(c, item, &dst[i])) return false;
    }
    return true;
}

static bool ple_scan_file(const uint8_t *map, uint64_t size, size_t path_index,
                          const char *path, ple_scan *s,
                          char *err, size_t err_size) {
    ple_cursor c = { map, size, 0, true };

    uint32_t magic = 0, version = 0;
    uint64_t n_tensors = 0, n_kv = 0;
    if (!ple_u32(&c, &magic) || magic != PLE_GGUF_MAGIC) {
        ple_err(err, err_size, "ds4_ple: %s is not a GGUF file", path);
        return false;
    }
    if (!ple_u32(&c, &version) || version < 2 || version > 3) {
        ple_err(err, err_size, "ds4_ple: %s has GGUF version %u; 2 or 3 expected",
                path, version);
        return false;
    }
    if (!ple_u64(&c, &n_tensors) || !ple_u64(&c, &n_kv)) {
        ple_err(err, err_size, "ds4_ple: %s has a truncated GGUF header", path);
        return false;
    }

    uint64_t alignment = 32;

    for (uint64_t i = 0; i < n_kv; i++) {
        ple_str key;
        uint32_t type = 0;
        if (!ple_string(&c, &key) || !ple_u32(&c, &type)) {
            ple_err(err, err_size, "ds4_ple: %s has a truncated metadata key", path);
            return false;
        }

        bool handled = true;
        if (ple_str_is(key, "general.alignment") && type == PLE_GGUF_UINT32) {
            uint32_t v = 0;
            if (!ple_u32(&c, &v) || v == 0 || (v & (v - 1)) != 0) {
                ple_err(err, err_size, "ds4_ple: %s has an invalid general.alignment", path);
                return false;
            }
            alignment = v;
        } else if (ple_str_is(key, "qwen4exp.ple.ngram_size") && type == PLE_GGUF_UINT32) {
            if (!ple_u32(&c, &s->ngram_size)) goto truncated;
            s->has_ngram_size = true;
        } else if (ple_str_is(key, "qwen4exp.ple.heads_per_ngram") && type == PLE_GGUF_UINT32) {
            if (!ple_u32(&c, &s->heads_per_ngram)) goto truncated;
            s->has_heads_per_ngram = true;
        } else if (ple_str_is(key, "qwen4exp.embedding_length_per_layer_input") &&
                   type == PLE_GGUF_UINT32) {
            if (!ple_u32(&c, &s->row_dim)) goto truncated;
            s->has_row_dim = true;
        } else if (ple_str_is(key, "qwen4exp.ple.conv_kernel") && type == PLE_GGUF_UINT32) {
            if (!ple_u32(&c, &s->conv_kernel)) goto truncated;
            s->has_conv_kernel = true;
        } else if (ple_str_is(key, "qwen4exp.ple.layers") && type == PLE_GGUF_ARRAY) {
            if (!ple_int_array(&c, s->layers, DS4_PLE_MAX_LAYERS, &s->layer_count)) {
                ple_err(err, err_size,
                        "ds4_ple: %s has an unreadable qwen4exp.ple.layers of "
                        "%" PRIu64 " entries", path, s->layer_count);
                return false;
            }
            s->has_layers = true;
        } else if (ple_str_is(key, "qwen4exp.ple.eos_token_id")) {
            ple_cursor v = c;
            if (!ple_int_elem(&v, type, &s->eos)) {
                ple_err(err, err_size,
                        "ds4_ple: %s stores qwen4exp.ple.eos_token_id as a "
                        "non-integer or negative value", path);
                return false;
            }
            c = v;
            s->has_eos = true;
        } else if (ple_str_is(key, "qwen4exp.ple.layer_multipliers") && type == PLE_GGUF_ARRAY) {
            if (!ple_int_array(&c, s->multipliers, DS4_PLE_MAX_NGRAM, &s->multiplier_count)) {
                ple_err(err, err_size,
                        "ds4_ple: %s has an unreadable qwen4exp.ple.layer_multipliers "
                        "of %" PRIu64 " entries", path, s->multiplier_count);
                return false;
            }
            s->has_multipliers = true;
        } else if (ple_str_is(key, "qwen4exp.ple.head_vocab_sizes") && type == PLE_GGUF_ARRAY) {
            if (!ple_int_array(&c, s->sizes, DS4_PLE_MAX_HEADS, &s->size_count)) {
                ple_err(err, err_size,
                        "ds4_ple: %s has an unreadable qwen4exp.ple.head_vocab_sizes "
                        "of %" PRIu64 " entries", path, s->size_count);
                return false;
            }
            s->has_sizes = true;
        } else if (ple_str_is(key, "qwen4exp.ple.head_offsets") && type == PLE_GGUF_ARRAY) {
            if (!ple_int_array(&c, s->offsets, DS4_PLE_MAX_HEADS, &s->offset_count)) {
                ple_err(err, err_size,
                        "ds4_ple: %s has an unreadable qwen4exp.ple.head_offsets "
                        "of %" PRIu64 " entries", path, s->offset_count);
                return false;
            }
            s->has_offsets = true;
        } else if (ple_str_is(key, "tokenizer.ggml.tokens") && type == PLE_GGUF_ARRAY) {
            /* Only the length is wanted; the strings are stepped over. */
            ple_cursor v = c;
            uint32_t item = 0;
            uint64_t len  = 0;
            if (!ple_u32(&v, &item) || !ple_u64(&v, &len)) goto truncated;
            s->vocab     = len;
            s->has_vocab = true;
            handled      = false; /* fall through to the generic skip */
        } else {
            handled = false;
        }

        if (!handled && !ple_skip_value(&c, type, 0)) {
            goto truncated;
        }
        if (!c.ok) goto truncated;
    }

    for (uint64_t i = 0; i < n_tensors; i++) {
        ple_str name;
        uint32_t ndim = 0, type = 0;
        uint64_t dims[4] = {1, 1, 1, 1};
        uint64_t offset = 0;

        if (!ple_string(&c, &name) || !ple_u32(&c, &ndim) || ndim > 4) {
            ple_err(err, err_size, "ds4_ple: %s has an unreadable tensor descriptor", path);
            return false;
        }
        for (uint32_t d = 0; d < ndim; d++) {
            if (!ple_u64(&c, &dims[d])) goto truncated;
        }
        if (!ple_u32(&c, &type) || !ple_u64(&c, &offset)) goto truncated;

        if (!s->has_tensor && ple_str_is(name, PLE_TABLE_TENSOR)) {
            s->has_tensor    = true;
            s->tensor_path   = path_index;
            s->tensor_offset = offset; /* made absolute below */
            s->tensor_dim0   = dims[0];
            s->tensor_dim1   = dims[1];
            s->tensor_type   = type;
        }
    }

    if (s->has_tensor && s->tensor_path == path_index) {
        uint64_t data_pos = (c.pos + alignment - 1) / alignment * alignment;
        s->tensor_offset += data_pos;
    }
    return true;

truncated:
    ple_err(err, err_size, "ds4_ple: %s has truncated GGUF metadata", path);
    return false;
}

/* =========================================================================
 * Mapping.
 * ========================================================================= */

typedef struct {
    int            fd;
    const uint8_t *map;
    uint64_t       size;
} ple_mapping;

static bool ple_map_open(const char *path, ple_mapping *m, char *err, size_t err_size) {
    m->fd = -1; m->map = NULL; m->size = 0;

    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        ple_err(err, err_size, "ds4_ple: cannot open %s", path);
        return false;
    }
#if defined(F_RDAHEAD)
    /* Suppress descriptor readahead where the platform offers it; the random
     * advice on the mapping below is the load-bearing knob. */
    (void)fcntl(fd, F_RDAHEAD, 0);
#endif
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0) {
        close(fd);
        ple_err(err, err_size, "ds4_ple: cannot size %s", path);
        return false;
    }
    void *base = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) {
        close(fd);
        ple_err(err, err_size, "ds4_ple: cannot map %s", path);
        return false;
    }
#if defined(POSIX_MADV_RANDOM)
    /* A step faults in sixteen small scattered rows.  Under default readahead
     * every first touch pulls a page cluster the row read never uses, so the
     * device waits on the disk. */
    (void)posix_madvise(base, (size_t)st.st_size, POSIX_MADV_RANDOM);
#endif
    m->fd   = fd;
    m->map  = (const uint8_t *)base;
    m->size = (uint64_t)st.st_size;
    return true;
}

static void ple_map_close(ple_mapping *m) {
    if (m->map) munmap((void *)m->map, (size_t)m->size);
    if (m->fd >= 0) close(m->fd);
    m->map = NULL;
    m->fd  = -1;
    m->size = 0;
}

/* =========================================================================
 * Constants.
 * ========================================================================= */

/* Every scalar and array the hash needs, in the order the refusal reports
 * them. */
static bool ple_require(const ple_scan *s, char *err, size_t err_size) {
    static const struct { size_t flag_offset; const char *key; } required[] = {
        { offsetof(ple_scan, has_ngram_size),      "qwen4exp.ple.ngram_size" },
        { offsetof(ple_scan, has_heads_per_ngram), "qwen4exp.ple.heads_per_ngram" },
        { offsetof(ple_scan, has_row_dim),         "qwen4exp.embedding_length_per_layer_input" },
        { offsetof(ple_scan, has_eos),             "qwen4exp.ple.eos_token_id" },
        { offsetof(ple_scan, has_conv_kernel),     "qwen4exp.ple.conv_kernel" },
        { offsetof(ple_scan, has_layers),          "qwen4exp.ple.layers" },
        { offsetof(ple_scan, has_vocab),           "tokenizer.ggml.tokens" },
        { offsetof(ple_scan, has_multipliers),     "qwen4exp.ple.layer_multipliers" },
        { offsetof(ple_scan, has_sizes),           "qwen4exp.ple.head_vocab_sizes" },
        { offsetof(ple_scan, has_offsets),         "qwen4exp.ple.head_offsets" },
    };
    for (size_t i = 0; i < sizeof(required) / sizeof(required[0]); i++) {
        const bool *flag = (const bool *)((const char *)s + required[i].flag_offset);
        if (!*flag) {
            ple_err(err, err_size,
                    "ds4_ple: the checkpoint has no %s; the n-gram hash constants "
                    "are never defaulted", required[i].key);
            return false;
        }
    }
    return true;
}

static bool ple_finish_constants(const ple_scan *s, ds4_ple_constants *out,
                                 char *err, size_t err_size) {
    if (!ple_require(s, err, err_size)) return false;

    memset(out, 0, sizeof(*out));

    if (s->ngram_size < 2 || s->ngram_size > DS4_PLE_MAX_NGRAM) {
        ple_err(err, err_size, "ds4_ple: ngram_size %u is outside 2..%d",
                s->ngram_size, DS4_PLE_MAX_NGRAM);
        return false;
    }
    if (s->heads_per_ngram == 0) {
        ple_err(err, err_size, "ds4_ple: heads_per_ngram is zero");
        return false;
    }
    uint64_t heads = (uint64_t)(s->ngram_size - 1) * s->heads_per_ngram;
    if (heads > DS4_PLE_MAX_HEADS) {
        ple_err(err, err_size, "ds4_ple: %" PRIu64 " n-gram heads exceed the %d supported",
                heads, DS4_PLE_MAX_HEADS);
        return false;
    }
    if (s->multiplier_count != s->ngram_size) {
        ple_err(err, err_size,
                "ds4_ple: qwen4exp.ple.layer_multipliers has %" PRIu64
                " entries; ngram_size is %u", s->multiplier_count, s->ngram_size);
        return false;
    }
    if (s->size_count != heads || s->offset_count != heads) {
        ple_err(err, err_size,
                "ds4_ple: head_vocab_sizes has %" PRIu64 " and head_offsets %" PRIu64
                " entries; %" PRIu64 " heads expected",
                s->size_count, s->offset_count, heads);
        return false;
    }
    if (s->row_dim == 0 || s->row_dim % DS4_PLE_IQ4_NL_BLOCK_ELEMS != 0) {
        ple_err(err, err_size,
                "ds4_ple: row width %u is not a multiple of the IQ4_NL block of %d",
                s->row_dim, DS4_PLE_IQ4_NL_BLOCK_ELEMS);
        return false;
    }
    if (s->conv_kernel == 0) {
        ple_err(err, err_size, "ds4_ple: qwen4exp.ple.conv_kernel is zero");
        return false;
    }
    if (s->layer_count == 0) {
        ple_err(err, err_size, "ds4_ple: qwen4exp.ple.layers is empty");
        return false;
    }
    if (s->vocab == 0 || s->vocab > INT32_MAX) {
        ple_err(err, err_size, "ds4_ple: vocabulary of %" PRIu64 " tokens is unusable", s->vocab);
        return false;
    }
    if (s->eos >= s->vocab) {
        ple_err(err, err_size,
                "ds4_ple: end-of-sequence token %" PRIu64 " is outside the "
                "vocabulary of %" PRIu64, s->eos, s->vocab);
        return false;
    }

    /* The reference computes the hash in signed 64-bit and takes a signed
     * remainder.  Signed and unsigned agree only while every token times every
     * multiplier stays non-negative, so that is checked here rather than
     * assumed. */
    const uint64_t max_token = s->vocab - 1;
    for (uint32_t i = 0; i < s->ngram_size; i++) {
        uint64_t m = s->multipliers[i];
        if (m == 0) {
            ple_err(err, err_size, "ds4_ple: layer multiplier %u is zero", i);
            return false;
        }
        if (max_token != 0 && m > (uint64_t)INT64_MAX / max_token) {
            ple_err(err, err_size,
                    "ds4_ple: layer multiplier %u (%" PRIu64 ") times the highest token "
                    "%" PRIu64 " overflows a signed 64-bit product, so the reference "
                    "remainder is not reproducible", i, m, max_token);
            return false;
        }
        out->multipliers[i] = m;
    }

    uint64_t running = 0;
    for (uint64_t h = 0; h < heads; h++) {
        if (s->sizes[h] == 0) {
            ple_err(err, err_size, "ds4_ple: head %" PRIu64 " has a zero vocab size", h);
            return false;
        }
        if (s->offsets[h] != running) {
            ple_err(err, err_size,
                    "ds4_ple: head %" PRIu64 " starts at %" PRIu64 " but the heads before "
                    "it cover %" PRIu64 " rows", h, s->offsets[h], running);
            return false;
        }
        if (s->sizes[h] > UINT64_MAX - running) {
            ple_err(err, err_size, "ds4_ple: the head table overflows");
            return false;
        }
        out->head_vocab_sizes[h] = s->sizes[h];
        out->head_offsets[h]     = s->offsets[h];
        running += s->sizes[h];
    }

    out->ngram_size      = s->ngram_size;
    out->heads_per_ngram = s->heads_per_ngram;
    out->head_count      = (uint32_t)heads;
    out->row_dim         = s->row_dim;
    out->vocab_size      = (uint32_t)s->vocab;
    out->eos_token_id    = (int32_t)s->eos;
    out->conv_kernel     = s->conv_kernel;
    out->ple_layer_count = (uint32_t)s->layer_count;
    for (uint64_t i = 0; i < s->layer_count; i++) {
        if (s->layers[i] > UINT32_MAX) {
            ple_err(err, err_size, "ds4_ple: PLE layer index %" PRIu64 " is out of range",
                    s->layers[i]);
            return false;
        }
        out->ple_layers[i] = (uint32_t)s->layers[i];
    }
    out->row_total       = running;
    out->table_rows      = running;
    return true;
}

static bool ple_scan_paths(const char *const *paths, size_t path_count,
                           ple_scan *scan, ple_mapping *keep,
                           char *err, size_t err_size) {
    if (!paths || path_count == 0) {
        ple_err(err, err_size, "ds4_ple: no GGUF shard was given");
        return false;
    }
    memset(scan, 0, sizeof(*scan));
    if (keep) { keep->fd = -1; keep->map = NULL; keep->size = 0; }

    for (size_t i = 0; i < path_count; i++) {
        ple_mapping m;
        if (!ple_map_open(paths[i], &m, err, err_size)) {
            if (keep && keep->map) ple_map_close(keep);
            return false;
        }
        bool had_tensor = scan->has_tensor;
        bool ok = ple_scan_file(m.map, m.size, i, paths[i], scan, err, err_size);
        bool want = keep && !had_tensor && scan->has_tensor && ok;

        if (!ok) {
            ple_map_close(&m);
            if (keep && keep->map) ple_map_close(keep);
            return false;
        }
        if (want) {
            if (keep->map) ple_map_close(keep);
            *keep = m;
        } else {
            ple_map_close(&m);
        }
    }
    return true;
}

bool ds4_ple_constants_read(const char *const *gguf_paths, size_t path_count,
                            ds4_ple_constants *out, char *err, size_t err_size) {
    if (err && err_size) err[0] = '\0';
    if (!out) return false;

    ple_scan scan;
    if (!ple_scan_paths(gguf_paths, path_count, &scan, NULL, err, err_size)) return false;
    return ple_finish_constants(&scan, out, err, err_size);
}

/* =========================================================================
 * Row ids.
 * =========================================================================
 *
 * One row id per head.  `heads_per_ngram` heads read the 2-token history and
 * `heads_per_ngram` more read the 3-token history:
 *
 *   bigram  = current * m0 ^ previous_1 * m1
 *   trigram = bigram             ^ previous_2 * m2
 *   row     = mixed % head_vocab_size[head] + head_offset[head]
 *
 * END-OF-SEQUENCE RULE.  A shift never crosses a sequence boundary or the
 * start of the sequence: `previous_1` is the end-of-sequence token when there
 * is no earlier token, and `previous_2` is the end-of-sequence token whenever
 * `previous_1` is, which is exactly the reference's segment test written as a
 * recurrence. */

void ds4_ple_history_reset(const ds4_ple_constants *c, ds4_ple_history *h) {
    if (!c || !h) return;
    for (int i = 0; i < DS4_PLE_MAX_NGRAM; i++) h->previous[i] = c->eos_token_id;
}

void ds4_ple_row_ids(const ds4_ple_constants *c, ds4_ple_history *h,
                     const int32_t *tokens, size_t count, uint64_t *out) {
    if (!c || !h || (count != 0 && (!tokens || !out))) return;

    const uint32_t ngram = c->ngram_size;
    const uint32_t hpn   = c->heads_per_ngram;
    const int32_t  eos   = c->eos_token_id;

    for (size_t t = 0; t < count; t++) {
        const int32_t cur = tokens[t];

        uint64_t mixed = (uint64_t)(uint32_t)cur * c->multipliers[0];
        uint64_t *row  = out + t * c->head_count;

        for (uint32_t n = 2; n <= ngram; n++) {
            mixed ^= (uint64_t)(uint32_t)h->previous[n - 1] * c->multipliers[n - 1];
            const uint32_t low = (n - 2) * hpn;
            for (uint32_t k = 0; k < hpn; k++) {
                const uint32_t head = low + k;
                row[head] = mixed % c->head_vocab_sizes[head] + c->head_offsets[head];
            }
        }

        /* Advance: previous[1] becomes the token just consumed, and every
         * deeper slot inherits the slot above it unless the shift now crosses
         * an end-of-sequence token. */
        for (uint32_t p = ngram - 1; p >= 2; p--) {
            h->previous[p] = (cur == eos) ? eos : h->previous[p - 1];
        }
        if (ngram >= 2) h->previous[1] = cur;
    }
}

/* =========================================================================
 * IQ4_NL dequantization.
 * =========================================================================
 *
 * Ported from ggml `dequantize_row_iq4_nl` in ggml/src/ggml-quants.c, with the
 * non-linear code book `kvalues_iq4nl` of ggml/src/ggml-common.h:
 *
 *   for each block: d = fp16_to_fp32(x[i].d)
 *       for j in 0..15: y[j]      = d * kvalues_iq4nl[qs[j] & 0xf]
 *                       y[j + 16] = d * kvalues_iq4nl[qs[j] >> 4] */

static const int8_t ple_kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

static float ple_fp16_to_fp32(uint16_t h) {
    const uint32_t sign     = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t exponent = (h >> 10) & 0x1Fu;
    const uint32_t mantissa = h & 0x3FFu;
    uint32_t bits;

    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            /* Subnormal: normalize it into a float32 exponent. */
            uint32_t e = 0;
            uint32_t m = mantissa;
            while ((m & 0x400u) == 0) { m <<= 1; e++; }
            m &= 0x3FFu;
            bits = sign | ((127u - 15u - e + 1u) << 23) | (m << 13);
        }
    } else if (exponent == 0x1Fu) {
        bits = sign | 0x7F800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 127u - 15u) << 23) | (mantissa << 13);
    }

    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

void ds4_ple_dequant_iq4_nl(const void *blocks, size_t block_count, float *out) {
    const uint8_t *p = (const uint8_t *)blocks;
    if (!p || !out) return;

    for (size_t b = 0; b < block_count; b++) {
        uint16_t half;
        memcpy(&half, p, sizeof(half));
        const float d = ple_fp16_to_fp32(half);
        const uint8_t *qs = p + 2;
        float *y = out + b * DS4_PLE_IQ4_NL_BLOCK_ELEMS;

        for (int j = 0; j < DS4_PLE_IQ4_NL_BLOCK_ELEMS / 2; j++) {
            y[j]      = d * (float)ple_kvalues_iq4nl[qs[j] & 0x0F];
            y[j + 16] = d * (float)ple_kvalues_iq4nl[qs[j] >> 4];
        }
        p += DS4_PLE_IQ4_NL_BLOCK_BYTES;
    }
}

/* =========================================================================
 * Table.
 * ========================================================================= */

struct ds4_ple_table {
    ds4_ple_constants constants;
    ple_mapping       mapping;
    uint64_t          tensor_offset;
    size_t            quant_row_bytes;
    size_t            row_floats;

    /* Hot set: a fixed arena of dequantized rows in least-recently-used
     * order.  Sized once from the ceiling, so there is no growth path and no
     * allocation on the hot path. */
    uint64_t  capacity;
    uint64_t  buckets;      /* power of two, 0 when the hot set is off */
    float    *arena;
    uint64_t *slot_row;
    int32_t  *lru_prev;
    int32_t  *lru_next;
    int32_t  *slot_chain;   /* next slot in the same hash bucket */
    int32_t  *bucket_head;
    int32_t   head;
    int32_t   tail;
    uint64_t  used;

    uint64_t ceiling_bytes;
    uint64_t resident_bytes;
    uint64_t hits;
    uint64_t misses;
    uint64_t evictions;
};

uint64_t ds4_ple_splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

/* =========================================================================
 * Reference derivation of the hash constants.  See the header: this is a
 * tripwire on the artifact's key-values and never a source of values.
 * ========================================================================= */

void ds4_ple_derive_multipliers(uint32_t ngram_size, uint32_t vocab_size,
                                uint32_t ple_layer_ordinal, int64_t seed,
                                uint64_t *out) {
    if (!out || ngram_size == 0) return;
    const uint64_t gamma = 0x9E3779B97F4A7C15ull;
    const uint64_t int64_max = 0x7FFFFFFFFFFFFFFFull;
    const uint64_t vocab = vocab_size ? vocab_size : 1u;
    uint64_t half = (int64_max / vocab) / 2ull;
    if (half == 0) half = 1ull;
    const uint64_t base = (uint64_t)seed + (uint64_t)DS4_PLE_HASH_SEED_PRIME_1 *
                                           (uint64_t)ple_layer_ordinal;
    for (uint32_t i = 0; i < ngram_size; i++) {
        const uint64_t mixed =
            ds4_ple_splitmix64(base + gamma * (uint64_t)(i + 1u));
        out[i] = 2ull * (mixed % half) + 1ull;
    }
}

static bool ple_is_prime(uint64_t v) {
    if (v < 2ull) return false;
    if (v % 2ull == 0ull) return v == 2ull;
    for (uint64_t d = 3ull; d * d <= v; d += 2ull) {
        if (v % d == 0ull) return false;
    }
    return true;
}

bool ds4_ple_head_vocab_follows_rule(const uint64_t *sizes, uint32_t count) {
    if (!sizes || count == 0) return false;
    if (!ple_is_prime(sizes[0])) return false;
    for (uint32_t h = 1; h < count; h++) {
        uint64_t next = sizes[h - 1] + 1ull;
        while (!ple_is_prime(next)) next++;
        if (sizes[h] != next) return false;
    }
    return true;
}

static void ple_lru_detach(ds4_ple_table *t, int32_t slot) {
    int32_t p = t->lru_prev[slot];
    int32_t n = t->lru_next[slot];
    if (p != -1) t->lru_next[p] = n; else if (t->head == slot) t->head = n;
    if (n != -1) t->lru_prev[n] = p; else if (t->tail == slot) t->tail = p;
    t->lru_prev[slot] = -1;
    t->lru_next[slot] = -1;
}

static void ple_lru_push_front(ds4_ple_table *t, int32_t slot) {
    t->lru_prev[slot] = -1;
    t->lru_next[slot] = t->head;
    if (t->head != -1) t->lru_prev[t->head] = slot;
    t->head = slot;
    if (t->tail == -1) t->tail = slot;
}

static int32_t ple_lookup(ds4_ple_table *t, uint64_t row) {
    if (t->buckets == 0) return -1;
    uint64_t b = ds4_ple_splitmix64(row) & (t->buckets - 1);
    for (int32_t s = t->bucket_head[b]; s != -1; s = t->slot_chain[s]) {
        if (t->slot_row[s] == row) return s;
    }
    return -1;
}

static void ple_chain_remove(ds4_ple_table *t, int32_t slot) {
    uint64_t b = ds4_ple_splitmix64(t->slot_row[slot]) & (t->buckets - 1);
    int32_t *link = &t->bucket_head[b];
    while (*link != -1) {
        if (*link == slot) { *link = t->slot_chain[slot]; break; }
        link = &t->slot_chain[*link];
    }
    t->slot_chain[slot] = -1;
}

static void ple_chain_insert(ds4_ple_table *t, int32_t slot) {
    uint64_t b = ds4_ple_splitmix64(t->slot_row[slot]) & (t->buckets - 1);
    t->slot_chain[slot] = t->bucket_head[b];
    t->bucket_head[b] = slot;
}

/* Dequantize one row of the mapping into `dst`. */
static void ple_read_row(const ds4_ple_table *t, uint64_t row, float *dst) {
    const uint8_t *src = t->mapping.map + t->tensor_offset + row * t->quant_row_bytes;
    ds4_ple_dequant_iq4_nl(src, t->constants.row_dim / DS4_PLE_IQ4_NL_BLOCK_ELEMS, dst);
}

static void ple_size_hot_set(ds4_ple_table *t, uint64_t ceiling) {
    const uint64_t row_bytes = (uint64_t)t->row_floats * sizeof(float);
    /* One slot costs its values plus its row id, its two list links and its
     * chain link; the bucket array is four bytes per slot, at most doubled by
     * the power-of-two rounding, so eight bytes are budgeted for it. */
    const uint64_t per_slot = row_bytes + sizeof(uint64_t) + 3 * sizeof(int32_t);
    const uint64_t budgeted = per_slot + 8;

    t->capacity = 0;
    t->buckets  = 0;
    if (ceiling < budgeted) return;

    uint64_t capacity = ceiling / budgeted;
    if (capacity > (uint64_t)INT32_MAX) capacity = (uint64_t)INT32_MAX;

    uint64_t buckets = 1;
    while (buckets < capacity) buckets <<= 1;

    while (capacity > 0 && capacity * per_slot + buckets * sizeof(int32_t) > ceiling) {
        capacity--;
    }
    if (capacity == 0) return;

    t->capacity       = capacity;
    t->buckets        = buckets;
    t->resident_bytes = capacity * per_slot + buckets * sizeof(int32_t);
}

bool ds4_ple_table_open(const char *const *gguf_paths, size_t path_count,
                        uint64_t cache_bytes, ds4_ple_table **out,
                        char *err, size_t err_size) {
    if (err && err_size) err[0] = '\0';
    if (!out) return false;
    *out = NULL;

    ple_scan scan;
    ple_mapping keep;
    if (!ple_scan_paths(gguf_paths, path_count, &scan, &keep, err, err_size)) return false;

    if (!scan.has_tensor) {
        ple_map_close(&keep);
        ple_err(err, err_size,
                "ds4_ple: none of the %zu shards holds %s",
                path_count, PLE_TABLE_TENSOR);
        return false;
    }

    ds4_ple_table *t = calloc(1, sizeof(*t));
    if (!t) {
        ple_map_close(&keep);
        ple_err(err, err_size, "ds4_ple: out of memory");
        return false;
    }
    t->mapping = keep;

    if (!ple_finish_constants(&scan, &t->constants, err, err_size)) {
        ds4_ple_table_close(t);
        return false;
    }

    if (scan.tensor_type != PLE_GGUF_TYPE_IQ4_NL) {
        ple_err(err, err_size, "ds4_ple: %s is GGUF type %u; IQ4_NL (%d) expected",
                PLE_TABLE_TENSOR, scan.tensor_type, PLE_GGUF_TYPE_IQ4_NL);
        ds4_ple_table_close(t);
        return false;
    }
    if (scan.tensor_dim0 != t->constants.row_dim) {
        ple_err(err, err_size,
                "ds4_ple: %s rows are %" PRIu64 " values; the checkpoint declares %u",
                PLE_TABLE_TENSOR, scan.tensor_dim0, t->constants.row_dim);
        ds4_ple_table_close(t);
        return false;
    }
    if (scan.tensor_dim1 < t->constants.row_total) {
        ple_err(err, err_size,
                "ds4_ple: %s holds %" PRIu64 " rows; the head table names %" PRIu64,
                PLE_TABLE_TENSOR, scan.tensor_dim1, t->constants.row_total);
        ds4_ple_table_close(t);
        return false;
    }
    t->constants.table_rows = scan.tensor_dim1;

    t->quant_row_bytes = (size_t)(t->constants.row_dim / DS4_PLE_IQ4_NL_BLOCK_ELEMS) *
                         DS4_PLE_IQ4_NL_BLOCK_BYTES;
    t->row_floats      = t->constants.row_dim;
    t->tensor_offset   = scan.tensor_offset;

    uint64_t need = scan.tensor_dim1 * (uint64_t)t->quant_row_bytes;
    if (t->tensor_offset > t->mapping.size || need > t->mapping.size - t->tensor_offset) {
        ple_err(err, err_size,
                "ds4_ple: %s needs %" PRIu64 " bytes at offset %" PRIu64
                " but the shard is %" PRIu64 " bytes",
                PLE_TABLE_TENSOR, need, t->tensor_offset, t->mapping.size);
        ds4_ple_table_close(t);
        return false;
    }

    t->ceiling_bytes = cache_bytes;
    ple_size_hot_set(t, cache_bytes);

    if (t->capacity > 0) {
        t->arena       = malloc((size_t)t->capacity * t->row_floats * sizeof(float));
        t->slot_row    = malloc((size_t)t->capacity * sizeof(uint64_t));
        t->lru_prev    = malloc((size_t)t->capacity * sizeof(int32_t));
        t->lru_next    = malloc((size_t)t->capacity * sizeof(int32_t));
        t->slot_chain  = malloc((size_t)t->capacity * sizeof(int32_t));
        t->bucket_head = malloc((size_t)t->buckets * sizeof(int32_t));
        if (!t->arena || !t->slot_row || !t->lru_prev || !t->lru_next ||
            !t->slot_chain || !t->bucket_head) {
            ple_err(err, err_size,
                    "ds4_ple: cannot reserve the %" PRIu64 "-byte n-gram hot set",
                    t->resident_bytes);
            ds4_ple_table_close(t);
            return false;
        }
        for (uint64_t i = 0; i < t->buckets; i++) t->bucket_head[i] = -1;
    }
    if (t->resident_bytes > t->ceiling_bytes) {
        /* The sizing above cannot produce this; the check is here so the
         * ceiling is a property the code states, not one a reader infers. */
        ple_err(err, err_size,
                "ds4_ple: the hot set would hold %" PRIu64 " bytes against a ceiling of "
                "%" PRIu64, t->resident_bytes, t->ceiling_bytes);
        ds4_ple_table_close(t);
        return false;
    }

    t->head = -1;
    t->tail = -1;
    *out = t;
    return true;
}

void ds4_ple_table_close(ds4_ple_table *t) {
    if (!t) return;
    ple_map_close(&t->mapping);
    free(t->arena);
    free(t->slot_row);
    free(t->lru_prev);
    free(t->lru_next);
    free(t->slot_chain);
    free(t->bucket_head);
    free(t);
}

const ds4_ple_constants *ds4_ple_table_constants(const ds4_ple_table *t) {
    return t ? &t->constants : NULL;
}

size_t ds4_ple_table_quant_row_bytes(const ds4_ple_table *t) {
    return t ? t->quant_row_bytes : 0;
}

bool ds4_ple_table_quant_row(const ds4_ple_table *t, uint64_t id, void *out) {
    if (!t || !out || id >= t->constants.table_rows) return false;
    memcpy(out, t->mapping.map + t->tensor_offset + id * t->quant_row_bytes,
           t->quant_row_bytes);
    return true;
}

bool ds4_ple_table_rows(ds4_ple_table *t, const uint64_t *ids, size_t count, float *out) {
    if (!t || (count != 0 && (!ids || !out))) return false;

    for (size_t i = 0; i < count; i++) {
        const uint64_t row = ids[i];
        float *dst = out + i * t->row_floats;
        if (row >= t->constants.table_rows) return false;

        if (t->capacity == 0) {
            t->misses++;
            ple_read_row(t, row, dst);
            continue;
        }

        int32_t slot = ple_lookup(t, row);
        if (slot >= 0) {
            t->hits++;
            if (t->head != slot) { ple_lru_detach(t, slot); ple_lru_push_front(t, slot); }
            memcpy(dst, t->arena + (size_t)slot * t->row_floats,
                   t->row_floats * sizeof(float));
            continue;
        }

        t->misses++;
        if (t->used < t->capacity) {
            slot = (int32_t)t->used++;
        } else {
            slot = t->tail;
            ple_chain_remove(t, slot);
            ple_lru_detach(t, slot);
            t->evictions++;
        }
        t->slot_row[slot] = row;
        ple_chain_insert(t, slot);
        ple_lru_push_front(t, slot);

        float *cached = t->arena + (size_t)slot * t->row_floats;
        ple_read_row(t, row, cached);
        memcpy(dst, cached, t->row_floats * sizeof(float));
    }
    return true;
}

void ds4_ple_table_prefetch(ds4_ple_table *t, const uint64_t *ids, size_t count) {
    if (!t || !ids) return;
#if defined(POSIX_MADV_WILLNEED)
    const size_t page = (size_t)sysconf(_SC_PAGESIZE);
    if (page == 0) return;
    for (size_t i = 0; i < count; i++) {
        const uint64_t row = ids[i];
        if (row >= t->constants.table_rows) continue;
        if (t->capacity > 0 && ple_lookup(t, row) >= 0) continue;

        uint64_t start = t->tensor_offset + row * t->quant_row_bytes;
        uint64_t end   = start + t->quant_row_bytes;
        start -= start % page;
        end = (end + page - 1) / page * page;
        if (end > t->mapping.size) end = t->mapping.size;
        (void)posix_madvise((void *)(t->mapping.map + start), (size_t)(end - start),
                            POSIX_MADV_WILLNEED);
    }
#else
    (void)count;
#endif
}

void ds4_ple_table_stats_get(const ds4_ple_table *t, ds4_ple_table_stats *out) {
    if (!out) return;
    memset(out, 0, sizeof(*out));
    if (!t) return;
    out->hits           = t->hits;
    out->misses         = t->misses;
    out->evictions      = t->evictions;
    out->resident_rows  = t->used;
    out->resident_bytes = t->resident_bytes;
    out->ceiling_bytes  = t->ceiling_bytes;
    out->capacity_rows  = t->capacity;
}
