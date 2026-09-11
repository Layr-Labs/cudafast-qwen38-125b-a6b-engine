#ifndef DS4_QWEN4EXP_PLE_H
#define DS4_QWEN4EXP_PLE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Host side of the Qwen4-Exp per-layer embedding (PLE) n-gram table.
 *
 * WHAT THIS IS.  Qwen 3.8 Flash-Next hashes the recent 2- and 3-token history
 * into row ids and reads one table row per head.  The table of the pinned
 * checkpoint is `per_layer_token_embd.weight`, IQ4_NL, 160 x 320,001,536.  A
 * row is 160 values, that is 5 IQ4_NL blocks of 32, that is 90 bytes.  The
 * whole tensor is 26.8 GiB, so it never enters resident memory as a whole.
 * This module memory maps it, reads the rows a step needs, and keeps a bounded
 * set of dequantized rows.
 *
 * WHAT THIS IS NOT.  There is no arithmetic of the PLE block here: the
 * convolution, the key/value projections and the norms are the device half and
 * live in the backend files.  This module produces row ids and row values.
 *
 * ALL CONSTANTS COME FROM THE CHECKPOINT.  The multipliers, the per-head vocab
 * sizes, the per-head offsets, the end-of-sequence token and the row width are
 * read from GGUF metadata.  A missing or malformed key is a refusal, never a
 * default. */

#ifdef __cplusplus
extern "C" {
#endif

/* Room for the pinned checkpoint (ngram_size 3, 16 heads) with margin. */
#define DS4_PLE_MAX_NGRAM 8
#define DS4_PLE_MAX_HEADS 64
#define DS4_PLE_MAX_LAYERS 8

/* IQ4_NL block geometry, from ggml: 32 values per block, 18 bytes per block. */
#define DS4_PLE_IQ4_NL_BLOCK_ELEMS 32
#define DS4_PLE_IQ4_NL_BLOCK_BYTES 18

/* Default hot-set ceiling. */
#define DS4_PLE_DEFAULT_CACHE_BYTES (2048ull * 1024ull * 1024ull)

/* Longest refusal message this module produces. */
#define DS4_PLE_ERROR_SIZE 512

/* Hash and geometry constants, read off the checkpoint.
 *
 * GGUF keys, verified against unsloth/Qwen3.8-Flash-Next-GGUF UD-Q4_K_XL:
 *   qwen4exp.ple.ngram_size                   -> ngram_size
 *   qwen4exp.ple.heads_per_ngram              -> heads_per_ngram
 *   qwen4exp.ple.eos_token_id                 -> eos_token_id
 *   qwen4exp.ple.conv_kernel                  -> conv_kernel
 *   qwen4exp.ple.layers                       -> ple_layers
 *   qwen4exp.ple.layer_multipliers            -> multipliers
 *   qwen4exp.ple.head_vocab_sizes             -> head_vocab_sizes
 *   qwen4exp.ple.head_offsets                 -> head_offsets
 *   qwen4exp.embedding_length_per_layer_input -> row_dim
 *   tokenizer.ggml.tokens                     -> vocab_size (array length) */
typedef struct {
    uint32_t ngram_size;
    uint32_t heads_per_ngram;
    uint32_t head_count;   /* (ngram_size - 1) * heads_per_ngram */
    uint32_t row_dim;      /* values in one table row */
    uint32_t vocab_size;
    int32_t  eos_token_id;
    /* The device half of the block needs these two; they are read here so one
     * refusal covers every PLE key of the checkpoint. */
    uint32_t conv_kernel;
    uint32_t ple_layer_count;
    uint32_t ple_layers[DS4_PLE_MAX_LAYERS];
    uint64_t multipliers[DS4_PLE_MAX_NGRAM];
    uint64_t head_vocab_sizes[DS4_PLE_MAX_HEADS];
    uint64_t head_offsets[DS4_PLE_MAX_HEADS];
    uint64_t row_total;    /* offsets[last] + sizes[last]: rows the hash can name */
    uint64_t table_rows;   /* rows the tensor holds; >= row_total (padded) */
} ds4_ple_constants;

/* Carried n-gram history of one sequence.
 *
 * `previous[p]` is the token `p` positions back, with the end-of-sequence rule
 * already applied: it is the end-of-sequence token when the shift would cross
 * a sequence boundary or run off the start.  `previous[0]` is unused; the
 * current token is passed in. */
typedef struct {
    int32_t previous[DS4_PLE_MAX_NGRAM];
} ds4_ple_history;

typedef struct {
    uint64_t hits;
    uint64_t misses;
    uint64_t evictions;
    uint64_t resident_rows;
    uint64_t resident_bytes;   /* every byte the hot set owns, index included */
    uint64_t ceiling_bytes;
    uint64_t capacity_rows;
} ds4_ple_table_stats;

typedef struct ds4_ple_table ds4_ple_table;

/* ---------------------------------------------------------------- constants */

/* Read and validate the PLE constants from one or more GGUF shards.
 *
 * Every shard listed is parsed; the keys may live in any of them, which is
 * what the published four-way split needs (the metadata shard carries the
 * keys, a data shard carries the tensor).  Returns false and fills `err` when
 * a key is missing, has the wrong type or length, or when the head table is
 * not self-consistent. */
bool ds4_ple_constants_read(const char *const *gguf_paths,
                            size_t             path_count,
                            ds4_ple_constants *out,
                            char              *err,
                            size_t             err_size);

/* ------------------------------------------- reference hash derivation */

/* The reference derivation of the n-gram hash constants, for use as a
 * TRIPWIRE and never as a source of values.
 *
 * The model implementation (transformers `Qwen4ExpTextNGramEmbedding`) builds
 * the multipliers from a splitmix64 stream seeded by the configuration seed
 * and the ORDINAL of the block in `ple_layer_ids`, and the per-head
 * vocabularies from consecutive primes after `ngram_vocab_size_base - 1`.
 * The GGUF key-values of a correctly converted artifact reproduce both.
 *
 * Ports have been observed not to: the MLX runner defaults the seed to 0
 * instead of 1234 and so derives a different multiplier triple, and a
 * published PLE manifest lists a third.  A mismatch is otherwise silent --
 * every shape and byte count still validates -- so the engine consumes the
 * key-values and checks them against this derivation at load.  */
#define DS4_PLE_HASH_SEED_DEFAULT 1234
#define DS4_PLE_HASH_SEED_PRIME_1 10007

/* splitmix64, the one definition: the derivation below and the hot set's
 * bucket index both call it. */
uint64_t ds4_ple_splitmix64(uint64_t v);

/* multipliers[i] = 2 * (splitmix64(base_seed + GAMMA * (i + 1)) % half) + 1,
 * base_seed = seed + PRIME_1 * ple_layer_ordinal,
 * half      = max(1, (INT64_MAX / vocab_size) / 2).
 *
 * The half bound is what keeps `token * multiplier` non-negative in signed
 * int64, so the modulo that follows is unambiguous.  Writes `ngram_size`
 * values. */
void ds4_ple_derive_multipliers(uint32_t  ngram_size,
                                uint32_t  vocab_size,
                                uint32_t  ple_layer_ordinal,
                                int64_t   seed,
                                uint64_t *out);

/* True when `sizes` is `count` consecutive primes in ascending order, which
 * is the per-head vocabulary rule stated without needing
 * `ngram_vocab_size_base`: the reference takes the (global head index + 1)-th
 * prime after that base, so successive heads are successive primes whatever
 * the base and whatever the layer ordinal. */
bool ds4_ple_head_vocab_follows_rule(const uint64_t *sizes, uint32_t count);

/* --------------------------------------------------------------- row ids */

/* Start a sequence: every history slot is the end-of-sequence token. */
void ds4_ple_history_reset(const ds4_ple_constants *c, ds4_ple_history *h);

/* Row ids for `count` tokens, `head_count` ids per token, row-major
 * [count][head_count].  `h` carries `previous_1`/`previous_2` across calls, so
 * one prefill batch and the single-token decode steps that follow produce the
 * same ids a single long call would.  Pass a freshly reset history for a new
 * sequence.  `h` is advanced past the last token. */
void ds4_ple_row_ids(const ds4_ple_constants *c,
                     ds4_ple_history         *h,
                     const int32_t           *tokens,
                     size_t                   count,
                     uint64_t                *out);

/* ------------------------------------------------------------- dequant */

/* Dequantize `block_count` IQ4_NL blocks into `block_count * 32` floats.
 *
 * Ported from ggml `dequantize_row_iq4_nl` (ggml/src/ggml-quants.c) with the
 * `kvalues_iq4nl` table of ggml/src/ggml-common.h.  Block layout: one f16
 * scale then 16 packed bytes; the low nibbles fill the first 16 values of the
 * block and the high nibbles the last 16. */
void ds4_ple_dequant_iq4_nl(const void *blocks, size_t block_count, float *out);

/* --------------------------------------------------------------- table */

/* Map the table and size the hot set.
 *
 * `cache_bytes` is the ceiling of the hot set in bytes, index included; pass
 * DS4_PLE_DEFAULT_CACHE_BYTES for the default and 0 to turn the hot set off.
 * The ceiling is never exceeded: the arena is sized once and never grows.
 *
 * The tensor is mapped, not read.  The mapping is advised random so a row
 * fault does not drag in a readahead cluster the step never uses. */
bool ds4_ple_table_open(const char *const *gguf_paths,
                        size_t             path_count,
                        uint64_t           cache_bytes,
                        ds4_ple_table    **out,
                        char              *err,
                        size_t             err_size);

void ds4_ple_table_close(ds4_ple_table *t);

const ds4_ple_constants *ds4_ple_table_constants(const ds4_ple_table *t);

/* Bytes one quantized row occupies in the file: row_dim / 32 * 18. */
size_t ds4_ple_table_quant_row_bytes(const ds4_ple_table *t);

/* Dequantized rows for `count` row ids, row-major [count][row_dim].
 *
 * EXACTNESS.  A hit and a miss return the same values: the hot set stores the
 * dequantization of the mapped bytes and nothing else, so the ceiling, the
 * eviction order and the hot-set state can never change a returned value.
 * Returns false only on an out-of-range row id. */
bool ds4_ple_table_rows(ds4_ple_table  *t,
                        const uint64_t *ids,
                        size_t          count,
                        float          *out);

/* The quantized bytes of one row, straight from the mapping.  For tests and
 * for a device path that dequantizes on its own. */
bool ds4_ple_table_quant_row(const ds4_ple_table *t, uint64_t id, void *out);

/* Ask the kernel to bring the rows in.  A step calls this with the row ids of
 * the next token so the page-ins overlap the work in flight.  A hint only: it
 * changes no value this module returns. */
void ds4_ple_table_prefetch(ds4_ple_table *t, const uint64_t *ids, size_t count);

void ds4_ple_table_stats_get(const ds4_ple_table *t, ds4_ple_table_stats *out);

#ifdef __cplusplus
}
#endif

#endif /* DS4_QWEN4EXP_PLE_H */
