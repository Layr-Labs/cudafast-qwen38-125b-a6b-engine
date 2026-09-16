#ifndef DS4_QWEN4EXP_H
#define DS4_QWEN4EXP_H

#include "ds4_qwen4exp_moe_types.h"
#include "ds4_qwen4exp_hc_types.h"

/*
 * qwen4exp -- Qwen3.8-Flash-Next 125B-A6B loader surface.
 *
 * This is an internal header.  Include it from ds4.c after ds4_tensor and
 * ds4_model are declared; it names those types.  ds4_qwen4exp.inc holds the
 * implementation and is included from the same place.
 *
 * What this layer owns
 * --------------------
 *   - reading the GGUF metadata of general.architecture=qwen4exp and refusing
 *     by name when it disagrees with the track fixture geometry,
 *   - binding every blk.N.* and non-layer tensor to a checked slot,
 *   - a memory plan grouped by tensor family, printed when the model opens,
 *   - a refusal when free unified memory cannot hold the resident bytes plus
 *     a fixed headroom.
 *
 * What this layer never does
 * --------------------------
 *   - copy tensor bytes.  Every slot below is a pointer into the mmapped GGUF
 *     tensor directory; payload bytes are reached with tensor_data().
 *   - read the per-layer n-gram table into RAM.  It is 28.8 GiB in the
 *     production artifact and stays SSD/page-cache resident behind
 *     ds4_qwen4exp_ple_row().
 *
 * Layer schedule.  48 blocks.  Block il is a sparse full-attention (QSA) block
 * when (il + 1) % full_attention_interval == 0, i.e. 3, 7, ... 47; the other 36
 * are gated-delta-net (GDN) blocks.  Use ds4_qwen4exp_layer_is_full_attention()
 * or the per-layer is_full_attention flag; never re-derive the rule.
 */

/* Highest per-layer n-gram head count this build accepts. */
#define DS4_QWEN4EXP_MAX_PLE_HEADS 64

/* Longest n-gram this build carries hash multipliers for; the pinned
 * checkpoint uses 3. */
#define DS4_QWEN4EXP_MAX_NGRAM 8

/* The n-gram table height is the head-vocabulary sum rounded up to a row
 * boundary; this is the largest pad the loader treats as legitimate. */
#define DS4_QWEN4EXP_PLE_ROW_PAD 256

/* Free unified memory must exceed the resident bytes by this much. */
#define DS4_QWEN4EXP_MEMORY_HEADROOM_BYTES (10ull * 1024ull * 1024ull * 1024ull)

/* The widest prompt chunk a session prefills in one forward, and so the widest
 * forward its activation scratch is sized for.  It is also the ceiling on the
 * session batch: ds4_session_qwen4exp_rows() stages a chunk through a stack
 * buffer of this many token ids, and refuses anything wider.
 *
 * It is a PREFILL number and has nothing to do with the speculative cycle's
 * verify width, which is two.  Sizing the batch from the cycle instead is what
 * made a 1024-token prompt prefill in 512 forwards of two rows and run at
 * decode speed. */
#define DS4_QWEN4EXP_MAX_PREFILL_ROWS 1024u

/* Memory plan buckets.  Every bound tensor is charged to exactly one. */
typedef enum {
    DS4_QWEN4EXP_MEM_EXPERTS = 0, /* routed expert gate/up/down slabs        */
    DS4_QWEN4EXP_MEM_PLE,         /* per_layer_token_embd -- SSD resident    */
    DS4_QWEN4EXP_MEM_DENSE,       /* attention/SSM projections, shared expert */
    DS4_QWEN4EXP_MEM_EMBED,       /* token_embd and output                   */
    DS4_QWEN4EXP_MEM_HC,          /* hyper-connection mix/inject/norm        */
    DS4_QWEN4EXP_MEM_GDN,         /* gated-delta-net parameters              */
    DS4_QWEN4EXP_MEM_ROUTER,      /* ffn_gate_inp and its shared-expert gate */
    DS4_QWEN4EXP_MEM_INDEXER,     /* sparse indexer projections and norms    */
    DS4_QWEN4EXP_MEM_COUNT,
} ds4_qwen4exp_mem_family;

typedef struct {
    uint64_t bytes[DS4_QWEN4EXP_MEM_COUNT];
    uint64_t tensors[DS4_QWEN4EXP_MEM_COUNT];
    uint64_t total_bytes;    /* sum over every family                       */
    uint64_t resident_bytes; /* total_bytes minus the SSD-resident PLE table */
    uint64_t ssd_bytes;      /* the PLE family                              */
    uint64_t bound_tensors;  /* how many GGUF tensors were bound            */
} ds4_qwen4exp_mem_plan;

/*
 * The per-layer n-gram embedding table.
 *
 * Bound as an mmap handle and never read into RAM.  `base` points straight
 * into the shard mapping that holds the tensor, so a row gather is a pointer
 * add and the kernel decides what stays in the page cache.
 */
typedef struct {
    ds4_tensor *tensor;   /* per_layer_token_embd.weight                    */
    const uint8_t *base;  /* first byte of row 0 inside its shard mapping   */
    uint64_t rows;        /* 320,001,536 in the production artifact         */
    uint64_t covered_rows;/* rows the head vocabularies actually cover      */
    uint64_t row_dim;     /* 160 values per row                             */
    uint64_t row_bytes;   /* quantized bytes per row (90 for IQ4_NL/160)    */
    uint64_t bytes;       /* rows * row_bytes                               */
    uint32_t type;        /* GGUF tensor type of the table                  */
    uint32_t heads;       /* per-n-gram head count (16)                     */
    uint64_t head_offset[DS4_QWEN4EXP_MAX_PLE_HEADS];
    uint64_t head_vocab[DS4_QWEN4EXP_MAX_PLE_HEADS];

    /* The n-gram hash constants, read from the artifact's key-values and
     * checked at bind against the reference derivation
     * (ds4_ple_derive_multipliers).  `have_hash` is false when the file
     * carries no multipliers, which is what a reduced synthetic fixture
     * looks like; the graph refuses by name rather than guessing. */
    bool     have_hash;
    uint32_t ngram_size;
    int32_t  eos_token_id;
    uint64_t multipliers[DS4_QWEN4EXP_MAX_NGRAM];
} ds4_qwen4exp_ple_table;

/*
 * One block.  Slots that do not apply to the block type stay NULL:
 * GDN blocks have attn_qkv/attn_gate/ssm_*, QSA blocks have attn_q/k/v/output
 * plus indexer_*, and only the PLE block carries ple_*.  Everything else --
 * the MoE, the shared expert, the router and the hyper-connection tensors --
 * is present on every block.
 */
typedef struct {
    bool is_full_attention;
    bool has_ple;

    /* GDN (linear attention) */
    ds4_tensor *attn_qkv;      /* Q8_0  [n_embd, gdn_qkv_dim]   */
    ds4_tensor *attn_gate;     /* Q8_0  [n_embd, gdn_inner]     */
    ds4_tensor *ssm_out;       /* Q8_0  [gdn_inner, n_embd]     */
    ds4_tensor *ssm_conv1d;    /* F32   [gdn_conv, gdn_qkv_dim] */
    ds4_tensor *ssm_alpha;     /* F32   [n_embd, gdn_value_head]*/
    ds4_tensor *ssm_beta;      /* F32   [n_embd, gdn_value_head]*/
    ds4_tensor *ssm_a;         /* F32   [gdn_value_head]        */
    ds4_tensor *ssm_dt_bias;   /* F32   [gdn_value_head]        */
    ds4_tensor *ssm_norm;      /* F32   [gdn_head_dim]          */

    /* QSA (sparse full attention) */
    ds4_tensor *attn_q;        /* Q8_0  [n_embd, n_head*head_dim*2] */
    ds4_tensor *attn_k;        /* Q8_0  [n_embd, n_head_kv*head_dim] */
    ds4_tensor *attn_v;        /* Q8_0  [n_embd, n_head_kv*head_dim] */
    ds4_tensor *attn_output;   /* Q8_0  [n_head*head_dim, n_embd]  */
    ds4_tensor *attn_q_norm;   /* F32   [head_dim]              */
    ds4_tensor *attn_k_norm;   /* F32   [head_dim]              */
    ds4_tensor *indexer_q_proj; /* BF16 [n_embd, idx_head*idx_dim] */
    ds4_tensor *indexer_k_proj; /* BF16 [n_embd, idx_kv_head*idx_dim] */
    ds4_tensor *indexer_q_norm; /* F32  [idx_dim]               */
    ds4_tensor *indexer_k_norm; /* F32  [idx_dim]               */

    /* Mixture of experts, every block */
    ds4_tensor *ffn_gate_inp;       /* F32  [n_embd, n_expert]        */
    /* Routed expert types vary per block; see ds4_qwen4exp_moe_types.h. */
    ds4_tensor *ffn_gate_exps;      /* Q4_K or Q5_K or Q6_K [n_embd, n_ff_exp, n_expert] */
    ds4_tensor *ffn_up_exps;        /* Q4_K or Q5_K or Q6_K [n_embd, n_ff_exp, n_expert] */
    ds4_tensor *ffn_down_exps;      /* Q5_1 or Q8_0 [n_ff_exp, n_embd, n_expert] */
    ds4_tensor *ffn_gate_inp_shexp; /* F32  [n_embd]                  */
    ds4_tensor *ffn_gate_shexp;     /* Q8_0 [n_embd, n_ff_shexp]      */
    ds4_tensor *ffn_up_shexp;       /* Q8_0 [n_embd, n_ff_shexp]      */
    ds4_tensor *ffn_down_shexp;     /* Q8_0 [n_ff_shexp, n_embd]      */

    /* Hyper connections, every block.  hc_dim = n_hc * n_embd. */
    ds4_tensor *hc_attn_down;   /* Q8_0 [hc_dim, hc_lowrank] */
    ds4_tensor *hc_attn_up;     /* Q8_0 [hc_lowrank, hc_dim] */
    ds4_tensor *hc_attn_inject; /* F32  [hc_dim, n_hc]       */
    ds4_tensor *hc_attn_norm;   /* F32  [hc_dim]             */
    ds4_tensor *hc_ffn_down;
    ds4_tensor *hc_ffn_up;
    ds4_tensor *hc_ffn_inject;
    ds4_tensor *hc_ffn_norm;

    /* Per-layer n-gram embedding, PLE block only */
    ds4_tensor *ple_conv1d;     /* F32  [ple_conv, hc_dim]  */
    ds4_tensor *ple_key;        /* Q8_0 [n_embd, hc_dim]    */
    ds4_tensor *ple_value;      /* Q8_0 [n_embd, ple_embd]  */
    ds4_tensor *ple_norm_conv;  /* F32  [hc_dim]            */
    ds4_tensor *ple_norm_key;   /* F32  [hc_dim]            */
    ds4_tensor *ple_norm_query; /* F32  [hc_dim]            */

    /* Which convention each norm weight in this block uses, classified at bind
     * from the tensor's own values: 1.0 when the checkpoint stores zero-centered
     * `w` and the kernel must apply `y * (1 + w)`, 0.0 when the converter
     * already baked the offset and the weight is `1 + w`.  See
     * qwen4exp_norm_weight_offset(); design section 2 requires this to be a
     * bind-time classification and not a per-call constant. */
    float hc_attn_norm_offset;
    float hc_ffn_norm_offset;
    float attn_q_norm_offset;
    float attn_k_norm_offset;
    float indexer_q_norm_offset;
    float indexer_k_norm_offset;
    float ple_norm_key_offset;
    float ple_norm_query_offset;
    float ple_norm_conv_offset;
} ds4_qwen4exp_layer_weights;

/*
 * The multi-token-prediction head, a second GGUF loaded through --mtp.
 *
 * It holds one extra block at index n_layer (48) plus the nextn projections.
 * Two things about it do not follow the main model:
 *   - the block is ALWAYS full attention, even though the file's
 *     attention.compress_ratios says 0 at that index, and
 *   - hc_attn_inject / hc_ffn_inject ship as Q8_0 here and F32 in the target.
 * It carries no token_embd and no output: nextn_shared_target_tensors is true
 * and it borrows the target model's embedding and LM head.
 */
typedef struct {
    ds4_qwen4exp_layer_weights block; /* blk.<n_layer>.*, full attention   */
    ds4_tensor *eh_proj;      /* Q8_0 [2 * n_embd, n_embd] */
    ds4_tensor *enorm;        /* F32  [n_embd]             */
    ds4_tensor *hnorm;        /* F32  [hc_dim]             */
    ds4_tensor *hc_head_down; /* Q8_0 [hc_dim, hc_lowrank] */
    ds4_tensor *hc_head_up;   /* Q8_0 [hc_lowrank, hc_dim] */
    ds4_tensor *hc_head_norm; /* F32  [hc_dim]             */
    ds4_qwen4exp_mem_plan plan;
    uint32_t block_index;
} ds4_qwen4exp_mtp_weights;

typedef struct {
    ds4_tensor *token_embd;      /* Q8_0 [n_embd, n_vocab]    */
    ds4_tensor *output;          /* Q8_0 [n_embd, n_vocab]    */
    ds4_tensor *output_hc_down;  /* Q8_0 [hc_dim, hc_lowrank] */
    ds4_tensor *output_hc_up;    /* Q8_0 [hc_lowrank, hc_dim] */
    ds4_tensor *output_hc_norm;  /* F32  [hc_dim]             */
    float       output_hc_norm_offset;

    ds4_qwen4exp_ple_table ple;
    ds4_qwen4exp_mem_plan  plan;

    uint32_t n_layer;
    ds4_qwen4exp_layer_weights layer[DS4_MAX_LAYER];
} ds4_qwen4exp_weights;

/* Runtime geometry read out of the GGUF.  The fixed geometry lives in
 * g_ds4_shape (DS4_N_* macros); this carries only the values a reduced-layer
 * test file is allowed to shrink, plus the derived widths L3-L7 need. */
typedef struct {
    uint32_t n_layer;
    uint32_t n_vocab;
    uint32_t ple_layer;      /* zero-based block that carries ple_*      */
    uint64_t ple_rows;
    uint32_t hc_dim;         /* n_hc * n_embd                            */
    uint32_t gdn_qkv_dim;    /* (2*key_head + value_head) * head_dim     */
    uint32_t gdn_inner;      /* value_head * head_dim                    */
    uint32_t qsa_q_dim;      /* n_head * head_dim * 2 (value and gate)   */
    uint32_t qsa_kv_dim;     /* n_head_kv * head_dim                     */
    uint32_t indexer_q_dim;  /* indexer_head * indexer_head_dim          */
    uint32_t indexer_k_dim;  /* indexer_kv_head * indexer_head_dim       */
    bool     synthetic_reduced;
} ds4_qwen4exp_config;

/* Set by config_validate_qwen4exp_model(); read by the binder and by L3-L7. */
extern ds4_qwen4exp_config g_ds4_qwen4exp;

/* Row pointer inside the mmapped n-gram table.  No copy, no allocation.
 * Returns NULL when `row` is out of range. */
static inline const uint8_t *ds4_qwen4exp_ple_row(
        const ds4_qwen4exp_ple_table *t, uint64_t row) {
    if (!t || !t->base || row >= t->rows) return NULL;
    return t->base + row * t->row_bytes;
}

#endif /* DS4_QWEN4EXP_H */
