/*
 * qwen4exp serial graph: session state, memory plan and the forward.
 * See ds4_qwen4exp_graph.h for the contract and the layer schedule.
 *
 * Every weight offset below is resolved through QW_MAP/QW_MAPSZ/QW_OFF, which
 * pick the SHARD the tensor lives in.  The production artifact is a four-way
 * split, so passing m->map here would address the wrong file for every tensor
 * outside shard 0.
 */

#include "ds4_qwen4exp_model_registration.h"

/* ======================================================================
 * Sizing.
 * ====================================================================== */

void ds4_qwen4exp_session_plan_compute(ds4_qwen4exp_session_plan *plan,
                                       uint32_t n_ctx,
                                       uint32_t n_batch) {
    const ds4_qwen4exp_config *cfg = &g_ds4_qwen4exp;
    const uint64_t E = DS4_N_EMBD;
    const uint64_t H = DS4_N_HC;
    const uint64_t B = n_batch;
    const uint64_t f = sizeof(float);

    memset(plan, 0, sizeof(*plan));
    plan->n_ctx = n_ctx;
    plan->n_batch = n_batch;

    for (uint32_t il = 0; il < cfg->n_layer; il++) {
        if (ds4_qwen4exp_layer_is_full_attention(il)) plan->n_qsa_layer++;
        else plan->n_gdn_layer++;
    }

    /* GDN: the recurrent state is [value_head][head_dim][state] fp32 and the
     * convolution keeps DS4_N_GDN_CONV - 1 previous rows of the fused qkv. */
    const uint64_t gdn_state_one =
        (uint64_t)DS4_N_GDN_VALUE_HEAD * DS4_N_GDN_HEAD_DIM * DS4_N_GDN_STATE * f;
    const uint64_t gdn_conv_one =
        (uint64_t)(DS4_N_GDN_CONV - 1u) * cfg->gdn_qkv_dim * f;
    plan->gdn_recurrent_bytes = gdn_state_one * plan->n_gdn_layer;
    plan->gdn_conv_bytes = gdn_conv_one * plan->n_gdn_layer;
    /* The speculative cycle's per-row state slots: one per DRAFT, because the
     * last verified row's state is the live state and needs no slot.  Charged
     * unconditionally so a serve plan does not shrink when the drafter is
     * unarmed and then overrun when it is. */
    plan->spec_snapshot_slots = DS4_QWEN4EXP_IMPLEMENTED_DEPTH;

    /* QSA: k and v caches at [n_ctx][kv_head][head_dim], plus the indexer's
     * exact unrotated key tape and the pooled blocks built from it. */
    const uint64_t kv_one = (uint64_t)n_ctx * cfg->qsa_kv_dim * f;
    const uint64_t tape_one =
        (uint64_t)n_ctx * DS4_N_INDEXER_KV_HEAD * DS4_N_INDEXER_HEAD_DIM * f;
    const uint64_t blocks = (uint64_t)n_ctx / DS4_N_INDEXER_COMPRESS_RATIO;
    const uint64_t pool_one =
        blocks * DS4_N_INDEXER_KV_HEAD * DS4_N_INDEXER_HEAD_DIM * f;
    const uint64_t knorm_one = (uint64_t)DS4_N_INDEXER_HEAD_DIM * f;
    plan->qsa_kv_bytes = 2ull * kv_one * plan->n_qsa_layer;
    plan->qsa_indexer_bytes = (tape_one + pool_one + knorm_one) * plan->n_qsa_layer;

    /* PLE: the depthwise convolution is DILATED by the n-gram size, so its
     * rolling window is (ple_conv - 1) * n_ngram rows -- nine on the pinned
     * checkpoint -- and it runs over the WIDE hyper-connection row, n_hc *
     * n_embd channels, not over the 2560-wide n-gram embedding.  The n-gram
     * id history itself is host side and tiny, and the 26.8 GiB table stays
     * mapped on the solid-state disk. */
    plan->ple_state_bytes = (uint64_t)DS4_QWEN4EXP_PLE_STATE_ROWS *
                            (uint64_t)DS4_N_HC * DS4_N_EMBD * f;

    const uint64_t top_k = (uint64_t)DS4_N_INDEXER_TOP_K / DS4_N_INDEXER_COMPRESS_RATIO;
    const uint64_t max_selected =
        DS4_QWEN4EXP_MAX_SELECTED(top_k, (uint64_t)DS4_N_INDEXER_COMPRESS_RATIO);
    const uint64_t q_width = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;

    uint64_t act = 0;
    act += B * sizeof(int32_t);                    /* tokens              */
    act += 3ull * B * H * E * f;                   /* hyper, normed, wide */
    act += 2ull * B * H * E * f;                   /* PLE key/gated, query/conv */
    act += B * DS4_N_PLE_EMBD * f;                 /* gathered n-gram rows */
    act += B * E * f;                              /* PLE value projection */
    act += B * DS4_N_HC_LOWRANK * f;               /* lowrank             */
    act += 3ull * B * E * f;                       /* mixed, block_out, embed_rows */
    act += B * H * f;                              /* inject              */
    act += B * cfg->gdn_qkv_dim * f;
    act += 2ull * B * DS4_N_GDN_VALUE_HEAD * f;    /* alpha, beta         */
    act += 2ull * B * cfg->gdn_inner * f;          /* gate, out           */
    const uint64_t q8_blocks = (E + 31ull) / 32ull;
    const uint64_t q8_qbytes = B * q8_blocks * 32ull;
    const uint64_t q8_soff = (q8_qbytes + 15ull) & ~15ull;
    const uint64_t q8_sbytes = B * q8_blocks * sizeof(float);
    act += q8_soff + q8_sbytes;                    /* qsa_mixed_q8        */
    act += B * cfg->qsa_q_dim * f;                 /* doubled query       */
    act += 3ull * B * q_width * f;                 /* q, gate, attn out   */
    act += 2ull * B * cfg->qsa_kv_dim * f;
    act += B * cfg->indexer_q_dim * f;
    act += B * cfg->indexer_k_dim * f;
    act += B * blocks * f;                         /* block scores        */
    act += B * top_k * sizeof(int32_t);
    act += B * max_selected * sizeof(int32_t);
    act += B * sizeof(int32_t);                    /* counts              */
    act += B * DS4_N_EXPERT * f;                   /* router logits       */
    act += B * DS4_N_EXPERT_USED * sizeof(int32_t);
    act += B * DS4_N_EXPERT_USED * f;
    act += B * DS4_N_EXPERT_USED * DS4_N_FF_EXP * f; /* routed mid        */
    /* The routed down projection's per-slot partials, and the largest
     * activation here: 100 MiB at batch 1024.  It buys the down weights the
     * sharing across rows the gate and up pair already had, which is three
     * quarters of the experts' weight traffic. */
    act += B * DS4_N_EXPERT_USED * DS4_N_EMBD * f;  /* routed down partial */
    act += B * DS4_N_FF_SHEXP * f;
    act += B * f;                                  /* shared gate scale   */
    act += (uint64_t)DS4_N_ROT / 2u * f;           /* inverse frequencies */
    /* The logit buffer carries one row per VERIFY row: the accept loop reads
     * every compared row's distribution out of one forward, so the LM head
     * runs once over the whole chain instead of once per row afterwards. */
    act += (uint64_t)DS4_QWEN4EXP_MTP_MAX_COMMIT * cfg->n_vocab * f;
    plan->activation_bytes = act;

    plan->spec_snapshot_bytes =
        (uint64_t)plan->spec_snapshot_slots *
        (plan->gdn_recurrent_bytes + plan->gdn_conv_bytes +
         plan->ple_state_bytes);

    plan->total_bytes = plan->gdn_recurrent_bytes + plan->gdn_conv_bytes +
                        plan->qsa_kv_bytes + plan->qsa_indexer_bytes +
                        plan->ple_state_bytes + plan->spec_snapshot_bytes +
                        plan->activation_bytes;
}

void ds4_qwen4exp_session_plan_print(const ds4_qwen4exp_session_plan *plan) {
    const double gib = 1024.0 * 1024.0 * 1024.0;
    fprintf(stderr,
            "ds4: qwen4exp session plan, context %u, batch %u, "
            "%u GDN and %u QSA layers\n",
            plan->n_ctx, plan->n_batch, plan->n_gdn_layer, plan->n_qsa_layer);
    fprintf(stderr, "ds4:   GDN recurrent state  %8.3f GiB\n",
            (double)plan->gdn_recurrent_bytes / gib);
    fprintf(stderr, "ds4:   GDN convolution      %8.3f GiB\n",
            (double)plan->gdn_conv_bytes / gib);
    fprintf(stderr, "ds4:   QSA key/value cache  %8.3f GiB\n",
            (double)plan->qsa_kv_bytes / gib);
    fprintf(stderr, "ds4:   QSA indexer cache    %8.3f GiB\n",
            (double)plan->qsa_indexer_bytes / gib);
    fprintf(stderr, "ds4:   PLE history          %8.3f GiB\n",
            (double)plan->ple_state_bytes / gib);
    fprintf(stderr, "ds4:   spec state slots     %8.3f GiB (%u rows)\n",
            (double)plan->spec_snapshot_bytes / gib,
            plan->spec_snapshot_slots);
    fprintf(stderr, "ds4:   activations          %8.3f GiB\n",
            (double)plan->activation_bytes / gib);
    fprintf(stderr, "ds4:   session total        %8.3f GiB\n",
            (double)plan->total_bytes / gib);
}

#ifndef DS4_NO_GPU

#define QW_MAP(m, t)   ((const void *)(m)->shard[(t)->map_index].map)
#define QW_MAPSZ(m, t) ((m)->shard[(t)->map_index].size)
#define QW_OFF(t)      ((t)->abs_offset)

/* Bytes one row of `t` occupies: dim[0] elements at the tensor's type. */
static uint64_t qwen4exp_row_bytes(const ds4_tensor *t) {
    uint64_t bytes = 0;
    if (!t || t->ndim == 0) return 0;
    if (!tensor_nbytes(t->type, t->dim[0], &bytes)) return 0;
    return bytes;
}

/* Bytes one expert slab of a 3-D expert tensor occupies. */
static uint64_t qwen4exp_expert_bytes(const ds4_tensor *t) {
    uint64_t bytes = 0;
    if (!t || t->ndim < 3) return 0;
    if (!tensor_nbytes(t->type, t->dim[0] * t->dim[1], &bytes)) return 0;
    return bytes;
}

/* The GPU view of one expert weight tensor: the shard mapping that holds it,
 * where it starts in that mapping, its type and its strides.  Building it per
 * tensor is what lets one block's experts live in different shards. */
#define QW_SLAB(m, t) ((ds4_gpu_qwen4exp_slab){ \
    .map = QW_MAP((m), (t)), .map_size = QW_MAPSZ((m), (t)), \
    .offset = QW_OFF(t), .expert_bytes = qwen4exp_expert_bytes(t), \
    .row_bytes = qwen4exp_row_bytes(t), .type = (t)->type })


/*
 * The n-gram hash constants the block runs on.
 *
 * NEVER RE-DERIVED, AND THE ARTIFACT IS THE AUTHORITY.  The values are the
 * artifact's own key-values, already read and CHECKED against the reference
 * derivation at bind (qwen4exp_read_ple_hash_constants,
 * QWEN4EXP-NGRAM-HASH-CONSTANTS-MISMATCH).  Re-deriving them here would make
 * a wrongly converted file silently gather a different set of rows: the
 * reference implementation seeds its splitmix64 stream with 1234, the MLX
 * runner defaults that seed to 0 and derives a different triple, and a
 * published PLE manifest lists a third, while every shape and byte count
 * still validates (design section 6, risk 5).  The MLX dumps stay the
 * reference for the block's ARITHMETIC -- the projections, the norms, the
 * gate and the convolution -- and not for the row ids.
 */
static bool qwen4exp_ple_read_constants(const ds4_model            *m,
                                        const ds4_qwen4exp_weights *w,
                                        ds4_ple_constants          *out) {
    memset(out, 0, sizeof(*out));
    (void)m;
    if (!w || !w->ple.have_hash) {
        fprintf(stderr, "ds4: qwen4exp carries no n-gram hash constants; the "
                        "artifact has no qwen4exp.ple.layer_multipliers\n");
        return false;
    }

    out->ngram_size = w->ple.ngram_size;
    out->heads_per_ngram = DS4_N_NGRAM_HEADS_PER;
    out->head_count = DS4_N_PLE_HEAD;
    out->row_dim = DS4_N_PLE_ROW_DIM;
    out->conv_kernel = DS4_N_PLE_CONV;
    out->eos_token_id = w->ple.eos_token_id;
    if (out->ngram_size > DS4_PLE_MAX_NGRAM ||
        out->head_count > DS4_PLE_MAX_HEADS ||
        out->head_count != (out->ngram_size - 1u) * out->heads_per_ngram) {
        fprintf(stderr, "ds4: qwen4exp PLE geometry %u-gram over %u heads per "
                        "n-gram does not give the %u heads this build sizes for\n",
                out->ngram_size, out->heads_per_ngram, out->head_count);
        return false;
    }
    for (uint32_t i = 0; i < out->ngram_size; i++) {
        out->multipliers[i] = w->ple.multipliers[i];
    }

    for (uint32_t h = 0; h < out->head_count; h++) {
        if (w->ple.head_vocab[h] == 0) {
            fprintf(stderr, "ds4: qwen4exp PLE head %u has no vocabulary; the "
                            "artifact describes the head partition by rule and "
                            "this build needs the table\n", h);
            return false;
        }
        out->head_vocab_sizes[h] = w->ple.head_vocab[h];
        out->head_offsets[h] = w->ple.head_offset[h];
    }
    out->table_rows = w->ple.rows;
    out->row_total = w->ple.covered_rows;
    return true;
}

struct ds4_qwen4exp_session {
    ds4_qwen4exp_session_plan plan;
    uint32_t pos;

    /* FALSE means every state object below is byte-for-byte what a reset
     * leaves: the caches, the recurrent states, the convolution histories,
     * the indexer tape and the n-gram history are all zero (or, for the
     * history, the reset value) and `pos` is 0.  It is TRUE from the moment
     * anything that writes state runs, and back to FALSE only at the end of a
     * full reset.
     *
     * WHY IT EXISTS.  A phase does three resets before its first forward: the
     * resident invalidates on accept, the worker's drain invalidates again,
     * and the sync that opens the prefill resets a third time -- and the last
     * two land INSIDE benchd's timed prefill window.  Zeroing buffers that
     * are already zero is not a cheaper reset, it is the same reset; the flag
     * makes the second and third ones free.  Skipping is exact by
     * construction: the skipped work writes the values that are already
     * there. */
    bool state_dirty;

    /* Per-layer state.  Only the slots the layer type uses are non-NULL. */
    ds4_gpu_tensor *gdn_conv[DS4_MAX_LAYER];
    ds4_gpu_tensor *gdn_state[DS4_MAX_LAYER];
    ds4_gpu_tensor *qsa_k[DS4_MAX_LAYER];
    ds4_gpu_tensor *qsa_v[DS4_MAX_LAYER];
    ds4_gpu_tensor *idx_tape[DS4_MAX_LAYER];
    ds4_gpu_tensor *idx_pool[DS4_MAX_LAYER];

    /* Per-layer indexer norm weights, uploaded once: the pool update takes the
     * k_layernorm weight as a tensor rather than a model offset. */
    ds4_gpu_tensor *idx_q_norm[DS4_MAX_LAYER];
    ds4_gpu_tensor *idx_k_norm[DS4_MAX_LAYER];
    ds4_gpu_tensor *attn_q_norm[DS4_MAX_LAYER];
    ds4_gpu_tensor *attn_k_norm[DS4_MAX_LAYER];

    /* Per-layer n-gram embedding.  The table itself is never copied: the
     * host hashes the history into row ids, reads the rows straight out of
     * the shard mapping and uploads only `ple_rows`, so the device holds the
     * gathered block, the scratch and the rolling window and nothing else.
     * The history and the window ride the session, so a reset restores them
     * with the caches. */
    ds4_ple_history   ple_history;
    ds4_ple_constants ple_constants;
    bool              ple_ready;      /* the hash constants were readable */
    uint64_t         *ple_ids;        /* host [n_batch][head_count]       */
    float            *ple_rows_host;  /* host [n_batch][ple_embd]         */
    ds4_gpu_tensor   *ple_conv_state; /* [state_rows][n_hc * n_embd]      */
    ds4_gpu_tensor   *ple_rows;       /* [n_batch][ple_embd]              */
    ds4_gpu_tensor   *ple_key;        /* key projection, then the gate    */
    ds4_gpu_tensor   *ple_aux;        /* query, then the convolution input */
    ds4_gpu_tensor   *ple_value;      /* value projection, [n_batch][embd] */

    /* Rollback bookkeeping.  A rejecting round has written state for a token
     * that is not committed; these are the snapshots and cursors that undo it.
     * The GDN state and the PLE conv state are SNAPSHOT class -- overwritten in
     * place, no position of their own -- so they are copied out before a verify
     * and copied back on rejection. */
    /* PER-ROW STATE SLOTS, DS4_QWEN4EXP_IMPLEMENTED_DEPTH of them, written by
     * the verify forward and selected from by a partially accepting round.
     * Slot k holds the state as it stood after verify row k. */
    ds4_gpu_tensor *gdn_conv_snapshot[DS4_MAX_LAYER];
    ds4_gpu_tensor *gdn_state_snapshot[DS4_MAX_LAYER];
    ds4_gpu_tensor *ple_conv_snapshot;
    ds4_ple_history ple_history_rows[DS4_QWEN4EXP_IMPLEMENTED_DEPTH];
    /* How many slots the CURRENT forward is writing.  Armed by
     * ds4_qwen4exp_graph_verify_rows and zero everywhere else, so a prefill
     * chunk of 512 never asks for 512 slots. */
    uint32_t        spec_snapshot_rows;
    /* How many rows of logits the CURRENT forward leaves.  1 everywhere but
     * the speculative verify, which wants every compared row's distribution
     * from the one forward rather than a borrowed-head call per row.  Armed by
     * ds4_qwen4exp_graph_verify_rows and cleared right after it. */
    uint32_t        spec_logit_rows;
    /* Compact speculative return: the wide LM head still computes every row,
     * but argmax runs before the command batch closes and only these indices
     * cross to the host until acceptance selects one full frontier row. */
    uint32_t        spec_top1_rows;
    uint32_t        indexer_valid_blocks;
    uint32_t        head_cache_pos;

    /* Shared, position-independent. */
    ds4_gpu_tensor *d_pos;
    ds4_gpu_tensor *inv_freq;

    /* Activations.  All sized for n_batch tokens and reused every layer. */
    ds4_gpu_tensor *tokens;
    ds4_gpu_tensor *hyper;      /* [tok][hc][embd] -- the residual stream */
    ds4_gpu_tensor *normed;
    ds4_gpu_tensor *lowrank;
    ds4_gpu_tensor *wide;
    ds4_gpu_tensor *mixed;      /* block input, [tok][embd]              */
    ds4_gpu_tensor *inject;     /* [tok][hc]                             */
    ds4_gpu_tensor *block_out;  /* [tok][embd]                           */
    ds4_gpu_tensor *embed_rows;

    ds4_gpu_tensor *gdn_qkv;
    ds4_gpu_tensor *gdn_alpha;
    ds4_gpu_tensor *gdn_beta;
    ds4_gpu_tensor *gdn_gate;
    ds4_gpu_tensor *gdn_out;

    ds4_gpu_tensor *qsa_mixed_q8;
    ds4_gpu_tensor *qsa_doubled;
    ds4_gpu_tensor *qsa_q;
    ds4_gpu_tensor *qsa_gate;
    ds4_gpu_tensor *qsa_kin;
    ds4_gpu_tensor *qsa_vin;
    ds4_gpu_tensor *qsa_out;
    ds4_gpu_tensor *idx_q;
    ds4_gpu_tensor *idx_k;
    ds4_gpu_tensor *idx_scores;
    ds4_gpu_tensor *idx_topk;
    ds4_gpu_tensor *idx_selected;
    ds4_gpu_tensor *idx_counts;

    ds4_gpu_tensor *moe_logits;
    ds4_gpu_tensor *moe_selected;
    ds4_gpu_tensor *moe_weights;
    ds4_gpu_tensor *moe_mid;
    ds4_gpu_tensor *moe_down_partial;
    ds4_gpu_tensor *shexp_mid;
    ds4_gpu_tensor *shexp_gate;

    ds4_gpu_tensor *logits;
    ds4_gpu_tensor *spec_top1;

#ifdef DS4_TEST_HOOKS
    /* A copy of the stream taken INSIDE the batch, immediately before the final
     * mixer reads it.  Comparing it with what the public accessor hands back
     * afterwards is a content check: it catches a mixer that wrote through the
     * buffer as well as an accessor that returned a copy.  Pointer identity
     * catches neither. */
    ds4_gpu_tensor *final_mixer_snapshot;
#endif
};

/* ======================================================================
 * Open, reset, close.
 * ====================================================================== */

/* `resident_weight_bytes` is what the weights NEED to be resident;
 * `already_resident_bytes` is what the engine has already put there.
 *
 * Only the difference has still to be found.  The engine uploads the whole
 * resident set before any session opens, and on a device whose free-memory
 * reading drops by exactly that amount -- CUDA's unified memory does -- adding
 * the weights to the requirement counts them twice and refuses a session that
 * would have fitted.  A 76.86 GiB model on a 118.6 GiB box read 40.17 GiB free
 * and was told it needed 76.86 + 0.33 + 10.00.
 *
 * Metal never hit this because its free-memory reading never drops: the reader
 * counts external (file-backed) pages as available, and a mapped GGUF's pages
 * are exactly that, so `already_resident_bytes` is 0 there and the arithmetic
 * below is unchanged. */
static bool qwen4exp_session_guard(const ds4_qwen4exp_session_plan *plan,
                                   uint64_t resident_weight_bytes,
                                   uint64_t already_resident_bytes) {
    const uint64_t outstanding =
        already_resident_bytes >= resident_weight_bytes
            ? 0u
            : resident_weight_bytes - already_resident_bytes;
    const uint64_t need = outstanding + plan->total_bytes;
    if (need > UINT64_MAX - DS4_QWEN4EXP_MEMORY_HEADROOM_BYTES) {
        fprintf(stderr, "ds4: qwen4exp memory budget overflows\n");
        return false;
    }
    const uint64_t required = need + DS4_QWEN4EXP_MEMORY_HEADROOM_BYTES;
    const uint64_t have = qwen4exp_free_memory_bytes();
    if (have == 0) {
        /* No reading available: say so rather than pretend the check passed. */
        fprintf(stderr,
                "ds4: qwen4exp cannot read free memory; skipping the session "
                "memory guard\n");
        return true;
    }
    if (have < required) {
        const double gib = 1024.0 * 1024.0 * 1024.0;
        fprintf(stderr,
                "ds4: qwen4exp refuses to open a session: needs %.2f GiB of "
                "weights (%.2f GiB already resident, %.2f GiB still to find) "
                "plus %.2f GiB of session state plus %.2f GiB of headroom, and "
                "only %.2f GiB is free\n",
                (double)resident_weight_bytes / gib,
                (double)already_resident_bytes / gib,
                (double)outstanding / gib,
                (double)plan->total_bytes / gib,
                (double)DS4_QWEN4EXP_MEMORY_HEADROOM_BYTES / gib,
                (double)have / gib);
        return false;
    }
    return true;
}

/* Every device buffer the session owns comes from here, so a test can prove a
 * refusal allocated nothing by watching this counter. */
static uint64_t g_qwen4exp_alloc_count = 0;

static ds4_gpu_tensor *qwen4exp_alloc_zeroed(uint64_t bytes, bool *ok) {
    if (!*ok) return NULL;
    g_qwen4exp_alloc_count++;
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(bytes);
    if (!t) {
        *ok = false;
        return NULL;
    }
    /* Every state buffer starts at zero: an unzeroed recurrent state or key
     * cache reads as data from a previous sequence. */
    void *zeros = calloc(1, (size_t)bytes);
    if (!zeros || !ds4_gpu_tensor_write(t, 0, zeros, bytes)) *ok = false;
    free(zeros);
    return t;
}

ds4_qwen4exp_session *ds4_qwen4exp_session_open(
        const ds4_qwen4exp_weights *w,
        const ds4_model            *m,
        uint32_t                    n_ctx,
        uint32_t                    n_batch,
        uint64_t                    resident_weight_bytes,
        uint64_t                    already_resident_bytes) {
    const ds4_qwen4exp_config *cfg = &g_ds4_qwen4exp;

    if (!w || !m || n_ctx == 0 || n_batch == 0) {
        fprintf(stderr, "ds4: qwen4exp session needs a bound model, a context "
                        "and a batch\n");
        return NULL;
    }
    if (n_ctx > DS4_QWEN4EXP_MAX_CTX) {
        fprintf(stderr,
                "ds4: qwen4exp context %u exceeds the %u this build sizes for\n",
                n_ctx, DS4_QWEN4EXP_MAX_CTX);
        return NULL;
    }
    if (n_ctx % DS4_N_INDEXER_COMPRESS_RATIO != 0) {
        fprintf(stderr,
                "ds4: qwen4exp context %u must be a multiple of the indexer "
                "compress ratio %u\n",
                n_ctx, DS4_N_INDEXER_COMPRESS_RATIO);
        return NULL;
    }
    if (n_batch > n_ctx) {
        fprintf(stderr, "ds4: qwen4exp batch %u exceeds the context %u\n",
                n_batch, n_ctx);
        return NULL;
    }

    /* The backend has to be up before a single buffer is asked for.  Metal
     * initialises lazily inside its own entry points so this never showed
     * there; CUDA does not, and every ds4_gpu_tensor_alloc_on() then fails at
     * its `device_id >= g_n_gpus` guard and returns NULL WITHOUT a message --
     * the session then reports "buffers could not be allocated" with no cause.
     * Say it here instead, once, by name. */
    if (!ds4_gpu_init()) {
        fprintf(stderr,
                "ds4: qwen4exp session: the GPU backend did not initialise\n");
        return NULL;
    }
#if !defined(__APPLE__) && !defined(DS4_ROCM_BUILD)
    if (g_n_gpus <= 0) {
        fprintf(stderr,
                "ds4: qwen4exp session: the backend reports %d devices, so no "
                "session buffer can be allocated; ds4_gpu_init() has not run or "
                "found no device\n", g_n_gpus);
        return NULL;
    }
#endif

    ds4_qwen4exp_session_plan plan;
    ds4_qwen4exp_session_plan_compute(&plan, n_ctx, n_batch);
    ds4_qwen4exp_session_plan_print(&plan);

    /* The guard runs BEFORE the first allocation, on weights plus session. */
    if (!qwen4exp_session_guard(&plan, resident_weight_bytes,
                                already_resident_bytes)) return NULL;

    ds4_qwen4exp_session *s = calloc(1, sizeof(*s));
    if (!s) {
        fprintf(stderr, "ds4: out of memory allocating the qwen4exp session\n");
        return NULL;
    }
    s->plan = plan;

    const uint64_t E = DS4_N_EMBD;
    const uint64_t H = DS4_N_HC;
    const uint64_t B = n_batch;
    const uint64_t f = sizeof(float);
    const uint64_t blocks = (uint64_t)n_ctx / DS4_N_INDEXER_COMPRESS_RATIO;
    const uint64_t top_k = (uint64_t)DS4_N_INDEXER_TOP_K / DS4_N_INDEXER_COMPRESS_RATIO;
    const uint64_t max_selected =
        DS4_QWEN4EXP_MAX_SELECTED(top_k, (uint64_t)DS4_N_INDEXER_COMPRESS_RATIO);
    const uint64_t q_width = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;

    /* Register every shard mapping with the backend before the first kernel.
     * Metal keeps its existing no-copy views. ROCm keeps its existing mapping
     * path. CUDA registration is isolated in a backend-free policy helper so
     * its allocation decisions can be tested without a device. */
#if !defined(__APPLE__) && !defined(DS4_ROCM_BUILD)
    ds4_qwen4exp_registration_shard reg_shards[DS4_MAX_GGUF_SHARDS];
    for (uint32_t i = 0; i < m->n_shards; i++) {
        reg_shards[i] = (ds4_qwen4exp_registration_shard){
            .map = m->shard[i].map,
            .size = m->shard[i].size,
        };
    }
    ds4_qwen4exp_registration_tensor *reg_tensors = xmalloc(
            (size_t)(m->n_tensors ? m->n_tensors : 1) * sizeof(reg_tensors[0]));
    for (uint64_t ti = 0; ti < m->n_tensors; ti++) {
        const ds4_tensor *t = &m->tensors[ti];
        reg_tensors[ti] = (ds4_qwen4exp_registration_tensor){
            .shard_index = t->map_index,
            .offset = t->abs_offset,
            .bytes = t->bytes,
            .ssd_resident = ds4_streq(t->name, "per_layer_token_embd.weight"),
        };
    }
    const ds4_qwen4exp_registration_ops reg_ops = {
        .lookup = ds4_gpu_model_range_device_ptr,
        .cache = ds4_gpu_cache_model_range,
        .replaced = ds4_gpu_model_range_replaced,
    };
    ds4_qwen4exp_registration_error reg_error;
    const bool registered = ds4_qwen4exp_register_cuda_model(
            reg_shards, m->n_shards, reg_tensors, m->n_tensors,
            &reg_ops, &reg_error);
    free(reg_tensors);
    if (!registered) {
        fprintf(stderr,
                "ds4: qwen4exp could not register CUDA target shard %u "
                "at offset %" PRIu64 " for %" PRIu64 " bytes: %s\n",
                reg_error.shard_index, reg_error.offset, reg_error.bytes,
                ds4_qwen4exp_registration_error_name(reg_error.kind));
        free(s);
        return NULL;
    }
#else
    for (uint32_t i = 0; i < m->n_shards; i++) {
        const ds4_model_shard *sh = &m->shard[i];
        if (!sh->map || sh->tensor_data_pos >= sh->size) continue;
        const uint64_t off = sh->tensor_data_pos;
        const uint64_t len = sh->size - off;

        /* The overlap Metal gives its views has to cover the largest tensor IN
         * THIS SHARD, not the largest in the model: a shard smaller than the
         * global maximum is refused outright ("max tensor span is larger than a
         * mapped tensor span"), which is what an eight-way split of the test
         * model hits while a three-way split does not. */
        uint64_t shard_max_tensor = 0;
        for (uint64_t ti = 0; ti < m->n_tensors; ti++) {
            const ds4_tensor *t = &m->tensors[ti];
            if (t->map_index == i && t->bytes > shard_max_tensor) {
                shard_max_tensor = t->bytes;
            }
        }

        int rc;
#if defined(__APPLE__)
        rc = ds4_gpu_set_model_map_range(sh->map, sh->size, off, len,
                                         shard_max_tensor);
#else
        rc = (i == 0)
            ? ds4_gpu_set_model_map(sh->map, sh->size)
            : ds4_gpu_set_aux_model_map_range(sh->map, sh->size, off, len);
#endif
        if (!rc) {
            fprintf(stderr,
                    "ds4: qwen4exp could not register shard %u with the "
                    "backend\n", i);
            free(s);
            return NULL;
        }
    }
#endif

    bool ok = true;
    for (uint32_t il = 0; il < cfg->n_layer && ok; il++) {
        if (ds4_qwen4exp_layer_is_full_attention(il)) {
            s->qsa_k[il] = qwen4exp_alloc_zeroed((uint64_t)n_ctx * cfg->qsa_kv_dim * f, &ok);
            s->qsa_v[il] = qwen4exp_alloc_zeroed((uint64_t)n_ctx * cfg->qsa_kv_dim * f, &ok);
            s->idx_tape[il] = qwen4exp_alloc_zeroed(
                (uint64_t)n_ctx * DS4_N_INDEXER_KV_HEAD * DS4_N_INDEXER_HEAD_DIM * f, &ok);
            s->idx_pool[il] = qwen4exp_alloc_zeroed(
                blocks * DS4_N_INDEXER_KV_HEAD * DS4_N_INDEXER_HEAD_DIM * f, &ok);
            s->idx_q_norm[il] = qwen4exp_alloc_zeroed(
                (uint64_t)DS4_N_INDEXER_HEAD_DIM * f, &ok);
            s->idx_k_norm[il] = qwen4exp_alloc_zeroed(
                (uint64_t)DS4_N_INDEXER_HEAD_DIM * f, &ok);
            s->attn_q_norm[il] = qwen4exp_alloc_zeroed((uint64_t)DS4_N_HEAD_DIM * f, &ok);
            s->attn_k_norm[il] = qwen4exp_alloc_zeroed((uint64_t)DS4_N_HEAD_DIM * f, &ok);
            /* These three norms take their weight as a device tensor rather
             * than a model offset, so upload them once at session open. */
            if (ok) {
                const ds4_qwen4exp_layer_weights *lw = &w->layer[il];
                ok = lw->indexer_q_norm && lw->indexer_k_norm &&
                     lw->attn_q_norm && lw->attn_k_norm &&
                     ds4_gpu_tensor_write(s->idx_q_norm[il], 0,
                                          tensor_data(m, lw->indexer_q_norm),
                                          (uint64_t)DS4_N_INDEXER_HEAD_DIM * f) &&
                     ds4_gpu_tensor_write(s->idx_k_norm[il], 0,
                                          tensor_data(m, lw->indexer_k_norm),
                                          (uint64_t)DS4_N_INDEXER_HEAD_DIM * f) &&
                     ds4_gpu_tensor_write(s->attn_q_norm[il], 0,
                                          tensor_data(m, lw->attn_q_norm),
                                          (uint64_t)DS4_N_HEAD_DIM * f) &&
                     ds4_gpu_tensor_write(s->attn_k_norm[il], 0,
                                          tensor_data(m, lw->attn_k_norm),
                                          (uint64_t)DS4_N_HEAD_DIM * f);
            }
        } else {
            const uint64_t conv_bytes =
                (uint64_t)(DS4_N_GDN_CONV - 1u) * cfg->gdn_qkv_dim * f;
            const uint64_t state_bytes =
                (uint64_t)DS4_N_GDN_VALUE_HEAD * DS4_N_GDN_HEAD_DIM *
                DS4_N_GDN_STATE * f;
            s->gdn_conv[il] = qwen4exp_alloc_zeroed(conv_bytes, &ok);
            s->gdn_state[il] = qwen4exp_alloc_zeroed(state_bytes, &ok);
            /* The rollback slots.  Both states are overwritten in place every
             * token, so a partially accepting round can only reach the length
             * it accepted from a copy the forward itself left behind. */
            s->gdn_conv_snapshot[il] = qwen4exp_alloc_zeroed(
                conv_bytes * DS4_QWEN4EXP_IMPLEMENTED_DEPTH, &ok);
            s->gdn_state_snapshot[il] = qwen4exp_alloc_zeroed(
                state_bytes * DS4_QWEN4EXP_IMPLEMENTED_DEPTH, &ok);
        }
    }

    /* The PLE block.  Every artifact this build accepts names exactly one PLE
     * block, so the buffers are unconditional and the memory plan charges for
     * them; only the gathered rows, the block scratch and the rolling window
     * live on the device, never the table. */
    {
        s->ple_conv_state = qwen4exp_alloc_zeroed(
            (uint64_t)DS4_QWEN4EXP_PLE_STATE_ROWS * H * E * f, &ok);
        s->ple_rows  = qwen4exp_alloc_zeroed(B * DS4_N_PLE_EMBD * f, &ok);
        s->ple_conv_snapshot = qwen4exp_alloc_zeroed(
            ds4_gpu_tensor_bytes(s->ple_conv_state) *
                DS4_QWEN4EXP_IMPLEMENTED_DEPTH, &ok);
        s->ple_key   = qwen4exp_alloc_zeroed(B * H * E * f, &ok);
        s->ple_aux   = qwen4exp_alloc_zeroed(B * H * E * f, &ok);
        s->ple_value = qwen4exp_alloc_zeroed(B * E * f, &ok);
        if (ok) {
            s->ple_ids = calloc((size_t)B * DS4_N_PLE_HEAD, sizeof(uint64_t));
            s->ple_rows_host = calloc((size_t)B * DS4_N_PLE_EMBD, sizeof(float));
            ok = s->ple_ids != NULL && s->ple_rows_host != NULL;
        }
        if (ok) {
            s->ple_ready = qwen4exp_ple_read_constants(m, w, &s->ple_constants);
            ds4_ple_history_reset(&s->ple_constants, &s->ple_history);
        }
    }

    s->d_pos = qwen4exp_alloc_zeroed(sizeof(uint32_t), &ok);
    s->inv_freq = qwen4exp_alloc_zeroed((uint64_t)DS4_N_ROT / 2u * f, &ok);
    if (ok) {
        float freqs[DS4_MAX_KDA_HEAD_DIM];
        ds4_gpu_qwen4exp_rope_inv_freq(freqs, DS4_N_ROT, DS4_ROPE_FREQ_BASE);
        ok = ds4_gpu_tensor_write(s->inv_freq, 0, freqs,
                                  (uint64_t)DS4_N_ROT / 2u * f);
    }

    s->tokens      = qwen4exp_alloc_zeroed(B * sizeof(int32_t), &ok);
    s->hyper       = qwen4exp_alloc_zeroed(B * H * E * f, &ok);
    s->normed      = qwen4exp_alloc_zeroed(B * H * E * f, &ok);
    s->wide        = qwen4exp_alloc_zeroed(B * H * E * f, &ok);
    s->lowrank     = qwen4exp_alloc_zeroed(B * DS4_N_HC_LOWRANK * f, &ok);
    s->mixed       = qwen4exp_alloc_zeroed(B * E * f, &ok);
    s->block_out   = qwen4exp_alloc_zeroed(B * E * f, &ok);
    s->embed_rows  = qwen4exp_alloc_zeroed(B * E * f, &ok);
    s->inject      = qwen4exp_alloc_zeroed(B * H * f, &ok);

    s->gdn_qkv   = qwen4exp_alloc_zeroed(B * cfg->gdn_qkv_dim * f, &ok);
    s->gdn_alpha = qwen4exp_alloc_zeroed(B * DS4_N_GDN_VALUE_HEAD * f, &ok);
    s->gdn_beta  = qwen4exp_alloc_zeroed(B * DS4_N_GDN_VALUE_HEAD * f, &ok);
    s->gdn_gate  = qwen4exp_alloc_zeroed(B * cfg->gdn_inner * f, &ok);
    s->gdn_out   = qwen4exp_alloc_zeroed(B * cfg->gdn_inner * f, &ok);

    const uint64_t q8_blocks = (E + 31ull) / 32ull;
    const uint64_t q8_qbytes = B * q8_blocks * 32ull;
    const uint64_t q8_soff = (q8_qbytes + 15ull) & ~15ull;
    const uint64_t q8_sbytes = B * q8_blocks * sizeof(float);
    s->qsa_mixed_q8 = qwen4exp_alloc_zeroed(q8_soff + q8_sbytes, &ok);
    s->qsa_doubled = qwen4exp_alloc_zeroed(B * cfg->qsa_q_dim * f, &ok);
    s->qsa_q       = qwen4exp_alloc_zeroed(B * q_width * f, &ok);
    s->qsa_gate    = qwen4exp_alloc_zeroed(B * q_width * f, &ok);
    s->qsa_out     = qwen4exp_alloc_zeroed(B * q_width * f, &ok);
    s->qsa_kin     = qwen4exp_alloc_zeroed(B * cfg->qsa_kv_dim * f, &ok);
    s->qsa_vin     = qwen4exp_alloc_zeroed(B * cfg->qsa_kv_dim * f, &ok);
    s->idx_q       = qwen4exp_alloc_zeroed(B * cfg->indexer_q_dim * f, &ok);
    s->idx_k       = qwen4exp_alloc_zeroed(B * cfg->indexer_k_dim * f, &ok);
    s->idx_scores  = qwen4exp_alloc_zeroed(B * blocks * f, &ok);
    s->idx_topk    = qwen4exp_alloc_zeroed(B * top_k * sizeof(int32_t), &ok);
    s->idx_selected = qwen4exp_alloc_zeroed(B * max_selected * sizeof(int32_t), &ok);
    s->idx_counts  = qwen4exp_alloc_zeroed(B * sizeof(int32_t), &ok);

    s->moe_logits   = qwen4exp_alloc_zeroed(B * DS4_N_EXPERT * f, &ok);
    s->moe_selected = qwen4exp_alloc_zeroed(B * DS4_N_EXPERT_USED * sizeof(int32_t), &ok);
    s->moe_weights  = qwen4exp_alloc_zeroed(B * DS4_N_EXPERT_USED * f, &ok);
    s->moe_mid      = qwen4exp_alloc_zeroed(B * DS4_N_EXPERT_USED * DS4_N_FF_EXP * f, &ok);
    s->moe_down_partial =
        qwen4exp_alloc_zeroed(B * DS4_N_EXPERT_USED * DS4_N_EMBD * f, &ok);
    s->shexp_mid    = qwen4exp_alloc_zeroed(B * DS4_N_FF_SHEXP * f, &ok);
    s->shexp_gate   = qwen4exp_alloc_zeroed(B * f, &ok);
    s->logits       = qwen4exp_alloc_zeroed(
            (uint64_t)DS4_QWEN4EXP_MTP_MAX_COMMIT * cfg->n_vocab * f, &ok);
    s->spec_top1     = qwen4exp_alloc_zeroed(
            (uint64_t)DS4_QWEN4EXP_MTP_MAX_COMMIT * sizeof(int32_t), &ok);
#ifdef DS4_TEST_HOOKS
    s->final_mixer_snapshot = qwen4exp_alloc_zeroed(B * H * E * f, &ok);
#endif

    if (!ok) {
        fprintf(stderr, "ds4: qwen4exp session buffers could not be allocated\n");
        ds4_qwen4exp_session_close(s);
        return NULL;
    }
    return s;
}

void ds4_qwen4exp_session_close(ds4_qwen4exp_session *s) {
    if (!s) return;
    /* Graph nodes retain session buffer addresses, even across resets. */
    ds4_gpu_decode_graphs_invalidate();
    for (uint32_t il = 0; il < DS4_MAX_LAYER; il++) {
        ds4_gpu_tensor_free(s->gdn_conv[il]);
        ds4_gpu_tensor_free(s->gdn_state[il]);
        ds4_gpu_tensor_free(s->gdn_conv_snapshot[il]);
        ds4_gpu_tensor_free(s->gdn_state_snapshot[il]);
        ds4_gpu_tensor_free(s->qsa_k[il]);
        ds4_gpu_tensor_free(s->qsa_v[il]);
        ds4_gpu_tensor_free(s->idx_tape[il]);
        ds4_gpu_tensor_free(s->idx_pool[il]);
        ds4_gpu_tensor_free(s->idx_q_norm[il]);
        ds4_gpu_tensor_free(s->idx_k_norm[il]);
        ds4_gpu_tensor_free(s->attn_q_norm[il]);
        ds4_gpu_tensor_free(s->attn_k_norm[il]);
    }
    ds4_gpu_tensor_free(s->ple_conv_state);
    ds4_gpu_tensor_free(s->ple_conv_snapshot);
    ds4_gpu_tensor_free(s->ple_rows);
    ds4_gpu_tensor_free(s->ple_key);
    ds4_gpu_tensor_free(s->ple_aux);
    ds4_gpu_tensor_free(s->ple_value);
    free(s->ple_ids);
    free(s->ple_rows_host);
    ds4_gpu_tensor_free(s->d_pos);
    ds4_gpu_tensor_free(s->inv_freq);
    ds4_gpu_tensor_free(s->tokens);
    ds4_gpu_tensor_free(s->hyper);
    ds4_gpu_tensor_free(s->normed);
    ds4_gpu_tensor_free(s->wide);
    ds4_gpu_tensor_free(s->lowrank);
    ds4_gpu_tensor_free(s->mixed);
    ds4_gpu_tensor_free(s->block_out);
    ds4_gpu_tensor_free(s->embed_rows);
    ds4_gpu_tensor_free(s->inject);
    ds4_gpu_tensor_free(s->gdn_qkv);
    ds4_gpu_tensor_free(s->gdn_alpha);
    ds4_gpu_tensor_free(s->gdn_beta);
    ds4_gpu_tensor_free(s->gdn_gate);
    ds4_gpu_tensor_free(s->gdn_out);
    ds4_gpu_tensor_free(s->qsa_mixed_q8);
    ds4_gpu_tensor_free(s->qsa_doubled);
    ds4_gpu_tensor_free(s->qsa_q);
    ds4_gpu_tensor_free(s->qsa_gate);
    ds4_gpu_tensor_free(s->qsa_out);
    ds4_gpu_tensor_free(s->qsa_kin);
    ds4_gpu_tensor_free(s->qsa_vin);
    ds4_gpu_tensor_free(s->idx_q);
    ds4_gpu_tensor_free(s->idx_k);
    ds4_gpu_tensor_free(s->idx_scores);
    ds4_gpu_tensor_free(s->idx_topk);
    ds4_gpu_tensor_free(s->idx_selected);
    ds4_gpu_tensor_free(s->idx_counts);
    ds4_gpu_tensor_free(s->moe_logits);
    ds4_gpu_tensor_free(s->moe_selected);
    ds4_gpu_tensor_free(s->moe_weights);
    ds4_gpu_tensor_free(s->moe_down_partial);
    ds4_gpu_tensor_free(s->moe_mid);
    ds4_gpu_tensor_free(s->shexp_mid);
    ds4_gpu_tensor_free(s->shexp_gate);
    ds4_gpu_tensor_free(s->logits);
    ds4_gpu_tensor_free(s->spec_top1);
#ifdef DS4_TEST_HOOKS
    ds4_gpu_tensor_free(s->final_mixer_snapshot);
#endif
    free(s);
}

const ds4_qwen4exp_session_plan *ds4_qwen4exp_session_plan_of(
        const ds4_qwen4exp_session *s) {
    return s ? &s->plan : NULL;
}

uint32_t ds4_qwen4exp_session_pos(const ds4_qwen4exp_session *s) {
    return s ? s->pos : 0u;
}

ds4_gpu_tensor *ds4_qwen4exp_session_hyper(const ds4_qwen4exp_session *s) {
    return s ? s->hyper : NULL;
}

bool ds4_qwen4exp_session_read_hyper(const ds4_qwen4exp_session *s,
                                     uint32_t                    first,
                                     uint32_t                    n_tokens,
                                     float                      *out) {
    if (!s || !s->hyper || !out || n_tokens == 0) {
        fprintf(stderr, "ds4: qwen4exp hyper read-back called with %s%s%s\n",
                (s && s->hyper) ? "" : "no session stream ",
                out ? "" : "no destination ", n_tokens ? "" : "no rows");
        return false;
    }
    if ((uint64_t)first + n_tokens > s->plan.n_batch) {
        fprintf(stderr,
                "ds4: qwen4exp hyper read-back of rows %u..%u passes the "
                "session batch %u\n", first, first + n_tokens, s->plan.n_batch);
        return false;
    }
    const uint64_t row = (uint64_t)DS4_N_HC * DS4_N_EMBD * sizeof(float);
    return ds4_gpu_tensor_read(s->hyper, (uint64_t)first * row, out,
                               (uint64_t)n_tokens * row) != 0;
}

static void qwen4exp_zero_tensor(ds4_gpu_tensor *t) {
    if (!t) return;
    const uint64_t bytes = ds4_gpu_tensor_bytes(t);
    /* Zero on the device: a reset used to calloc every tensor on the host
     * and upload the zeros, ~24 ms per session reset for the layer state. */
    if ((bytes & 3u) == 0 && ds4_gpu_tensor_fill_f32(t, 0.0f, bytes / 4u)) {
        return;
    }
    void *zeros = calloc(1, (size_t)bytes);
    if (!zeros) return;
    (void)ds4_gpu_tensor_write(t, 0, zeros, bytes);
    free(zeros);
}

void ds4_qwen4exp_session_reset(ds4_qwen4exp_session *s) {
    if (!s) return;
    if (!s->state_dirty) {
        /* Already at the reset state and nothing has written since, so every
         * store below would write the value the buffer already holds.  `pos`
         * is set anyway: it costs nothing and keeps this the single place the
         * position is cleared. */
        s->pos = 0;
        return;
    }
    for (uint32_t il = 0; il < DS4_MAX_LAYER; il++) {
        qwen4exp_zero_tensor(s->gdn_conv[il]);
        qwen4exp_zero_tensor(s->gdn_state[il]);
        qwen4exp_zero_tensor(s->qsa_k[il]);
        qwen4exp_zero_tensor(s->qsa_v[il]);
        qwen4exp_zero_tensor(s->idx_tape[il]);
        qwen4exp_zero_tensor(s->idx_pool[il]);
    }
    qwen4exp_zero_tensor(s->ple_conv_state);
    ds4_ple_history_reset(&s->ple_constants, &s->ple_history);
    s->pos = 0;
    s->state_dirty = false;
}

/* ======================================================================
 * The layer body.
 * ====================================================================== */

/*
 * The PLE block, blk.1 in the production artifact.
 *
 * WHERE IT SITS.  First in the layer, ahead of the attention mixer, and a plain
 * residual add over the WHOLE hyper-connection stream:
 *
 *     stream = stream + ple(stream, ids, previousContext, cache)
 *
 * MLX Qwen4Exp.swift:113-121, before `attnHyperConnection.mixWithInject`.  It
 * is not an inject and it does not go through hc_inject: the addend has the
 * full [n_tokens][n_hc][n_embd] shape of the stream itself.
 *
 * CONTRACT for the lane writing the kernels.  On entry `s->hyper` holds the
 * stream for `n_tokens` tokens at positions `s->pos ..< s->pos + n_tokens`.
 * On success the same buffer holds `stream + ple(stream)` and nothing else in
 * the session has moved except the PLE convolution history and the n-gram id
 * history.  `s->mixed`, `s->block_out` and `s->inject` are free scratch here;
 * the attention mixer overwrites all three immediately afterwards.
 *
 * The host half is written and tested (ds4_qwen4exp_ple.c): the signed-int64
 * n-gram hash, the per-head vocabularies and offsets, the EOS-bounded history,
 * and the IQ4_NL row gather out of the SSD-resident table.  A live block calls
 * ds4_ple_row_ids() for the 16 row ids of each token.
 *
 * THE BLOCK INDEX is the zero-based one the loader bound `blk.N.ple_*` to,
 * which is 1 on this checkpoint and is MLX's `pleLayerIndices.first`.  The
 * hash seed uses the POSITION in ple_layer_ids and not the block number, and
 * that seeding is already baked into the artifact's own multipliers, which
 * this never re-derives.
 *
 * Two halves.
 *
 * HOST.  The n-gram table is 26.8 GiB and stays mapped on the solid-state
 * disk.  ds4_ple_row_ids() hashes the recent history into one row id per head
 * (signed int64 throughout, with the end-of-sequence shift rule), the row
 * bytes are read straight out of the shard mapping through
 * ds4_qwen4exp_ple_row(), dequantized, and the [tok][ple_embd] block is the
 * only thing uploaded.  Nothing here keeps a second copy of the table.
 *
 * DEVICE.  ds4_gpu_qwen4exp_ple_block_tensor: the two Q8_0 projections, the
 * three grouped norms, the gate and the dilated depthwise convolution, in the
 * reference's order.
 *
 * The refusal stays for what the kernels do not cover: a table quantized as
 * anything but IQ4_NL, a row width that is not a whole number of IQ4_NL
 * blocks, a convolution the rolling window is not sized for, and an artifact
 * whose hash constants could not be read.
 */
/* Threaded host gather for the prefill width.
 *
 * The n-gram row gather is a host-side scan: 16 random 90-byte reads out of
 * the SSD-mapped IQ4_NL table per token, each one a page fault when the page
 * cache does not hold the row, plus a 160-value dequant.  At the prefill
 * width that is 16K serial faults while the device sits idle, and the fault
 * latency -- not the arithmetic -- dominates a cold chunk.
 *
 * The scan is embarrassingly parallel: a token's row ids depend only on the
 * token itself and the (ngram - 1) tokens before it, so a segment's entry
 * history is reproduced exactly by warming a fresh history over the tokens
 * that precede the segment (a reset history is all EOS, which is what the
 * sequential scan starts from too).  Each thread then reads and dequantizes
 * into its own disjoint slice of ple_rows_host, byte for byte what the serial
 * loop would have written.  The tail segment's history becomes the session's
 * carried history, which is the state the sequential scan would have ended
 * with.  No number the device sees changes; only the wall time does.
 *
 * The armed-slots path (spec_snapshot_rows > 0) stays serial: it is one to
 * four tokens and it must interleave history captures per row.
 *
 * DS4_QWEN4EXP_PLE_GATHER_THREADS=0 puts the serial loop back (the A/B
 * knob); a positive value pins the thread count.  Unset, the gather takes
 * the online CPUs capped at 16 -- enough to keep the SSD queue deep without
 * starving the decode path's own helpers. */
typedef struct {
    const ds4_qwen4exp_ple_table *ple;
    const ds4_qwen4exp_session   *s;
    const int32_t                *tokens;
    uint32_t                      n_tokens;
    uint32_t                      t0;
    uint32_t                      t1;
    uint32_t                      blocks_per_row;
    int                           ok;
} qw_ple_gather_seg;

static void *qw_ple_gather_worker(void *arg) {
    qw_ple_gather_seg *seg = (qw_ple_gather_seg *)arg;
    ds4_ple_history local;
    /* The segment's ENTRY history.  A segment that starts inside the call
     * rebuilds it from the tokens it can see: the state the sequential scan
     * would hold at t0 is a function of the (ngram - 1) tokens before t0,
     * so warming a reset history over exactly those tokens (output
     * discarded) reproduces it.  The segment at t0 == 0 is DIFFERENT on
     * every call after the first chunk: its entry state is the history the
     * previous chunk left in the session, which the token array of THIS
     * chunk cannot name -- it seeds from the carried history instead.  On
     * the first chunk the two agree, because a reset session is all EOS and
     * that is what a fresh history is. */
    const uint32_t warm = seg->s->ple_constants.ngram_size - 1u;
    if (seg->t0 > 0u && warm > 0u) {
        uint64_t scratch[DS4_PLE_MAX_NGRAM * DS4_PLE_MAX_HEADS];
        ds4_ple_history_reset(&seg->s->ple_constants, &local);
        const uint32_t from = seg->t0 > warm ? seg->t0 - warm : 0u;
        (void)ds4_ple_row_ids(&seg->s->ple_constants, &local,
                              seg->tokens + from, seg->t0 - from, scratch);
    } else if (seg->t0 == 0u) {
        local = seg->s->ple_history;
    } else {
        ds4_ple_history_reset(&seg->s->ple_constants, &local);
    }
    seg->ok = 1;
    for (uint32_t t = seg->t0; t < seg->t1; t++) {
        ds4_ple_row_ids(&seg->s->ple_constants, &local, seg->tokens + t, 1u,
                        seg->s->ple_ids + (uint64_t)t * DS4_N_PLE_HEAD);
        for (uint32_t h = 0; h < DS4_N_PLE_HEAD; h++) {
            const uint64_t row =
                seg->s->ple_ids[(uint64_t)t * DS4_N_PLE_HEAD + h];
            const uint8_t *quant = ds4_qwen4exp_ple_row(seg->ple, row);
            if (!quant) { seg->ok = 0; return NULL; }
            ds4_ple_dequant_iq4_nl(
                quant, seg->blocks_per_row,
                seg->s->ple_rows_host + (uint64_t)t * DS4_N_PLE_EMBD +
                    (uint64_t)h * DS4_N_PLE_ROW_DIM);
        }
    }
    return NULL;
}

static bool qwen4exp_graph_ple_gather(ds4_qwen4exp_session *s,
                                      const ds4_qwen4exp_weights *w,
                                      const int32_t *tokens,
                                      uint32_t n_tokens,
                                      uint32_t il) {
    if (!s->ple_ready) {
        fprintf(stderr,
                "ds4: qwen4exp block %u carries the per-layer n-gram embedding "
                "but the artifact's hash constants could not be read\n", il);
        return false;
    }
    if (w->ple.type != DS4_TENSOR_IQ4_NL) {
        fprintf(stderr,
                "ds4: qwen4exp block %u needs the n-gram table as IQ4_NL; this "
                "artifact stores it as type %u, which no gather covers\n",
                il, w->ple.type);
        return false;
    }
    if (w->ple.row_dim != DS4_N_PLE_ROW_DIM ||
        DS4_N_PLE_ROW_DIM % DS4_PLE_IQ4_NL_BLOCK_ELEMS != 0u ||
        (uint64_t)DS4_N_PLE_HEAD * DS4_N_PLE_ROW_DIM != DS4_N_PLE_EMBD) {
        fprintf(stderr,
                "ds4: qwen4exp block %u has %" PRIu64 " values per n-gram row "
                "over %u heads, which does not make the %u-wide embedding this "
                "build gathers\n",
                il, w->ple.row_dim, DS4_N_PLE_HEAD, DS4_N_PLE_EMBD);
        return false;
    }
    if ((DS4_N_PLE_CONV - 1u) * DS4_N_NGRAM != DS4_QWEN4EXP_PLE_STATE_ROWS) {
        fprintf(stderr,
                "ds4: qwen4exp block %u needs a %u-row convolution window "
                "((%u - 1) * %u) and this build sizes %u\n",
                il, (DS4_N_PLE_CONV - 1u) * DS4_N_NGRAM, DS4_N_PLE_CONV,
                DS4_N_NGRAM, DS4_QWEN4EXP_PLE_STATE_ROWS);
        return false;
    }

    /* Row ids for this call's tokens; the history is advanced past the last
     * one, so a prefill chunk and the decode steps after it name the same
     * rows a single long call would. */

    const double gather_t0 =
        getenv("DS4_QWEN4EXP_PLE_GATHER_TIME") ? now_sec() : 0.0;
    const uint32_t block_elems = DS4_PLE_IQ4_NL_BLOCK_ELEMS;
    const uint32_t blocks_per_row = DS4_N_PLE_ROW_DIM / block_elems;

    /* When slots are armed the ids are produced ONE ROW AT A TIME so the
     * n-gram history after each row can be kept beside them.  The contract on
     * ds4_ple_row_ids is that a run of single-token calls names the same rows
     * a single long call would, so this is the same ids either way; splitting
     * it is how the per-row history is observed, not a different computation. */
    if (s->spec_snapshot_rows > 0) {
        for (uint32_t t = 0; t < n_tokens; t++) {
            ds4_ple_row_ids(&s->ple_constants, &s->ple_history, tokens + t, 1u,
                            s->ple_ids + (uint64_t)t * DS4_N_PLE_HEAD);
            if (t < s->spec_snapshot_rows) {
                s->ple_history_rows[t] = s->ple_history;
            }
        }
        for (uint32_t t = 0; t < n_tokens; t++) {
            for (uint32_t h = 0; h < DS4_N_PLE_HEAD; h++) {
                const uint64_t row =
                    s->ple_ids[(uint64_t)t * DS4_N_PLE_HEAD + h];
                const uint8_t *quant = ds4_qwen4exp_ple_row(&w->ple, row);
                if (!quant) {
                    fprintf(stderr,
                            "ds4: qwen4exp block %u hashed token %u head %u to row "
                            "%" PRIu64 ", past the %" PRIu64 " the table holds\n",
                            il, t, h, row, w->ple.rows);
                    return false;
                }
                ds4_ple_dequant_iq4_nl(
                    quant, blocks_per_row,
                    s->ple_rows_host + (uint64_t)t * DS4_N_PLE_EMBD +
                        (uint64_t)h * DS4_N_PLE_ROW_DIM);
            }
        }
    } else {
        /* The prefill width: split the scan across worker threads when the
         * chunk is wide enough to be worth the spawn and the run has not
         * turned the pool off.  Every thread writes ids and rows for a
         * disjoint token range; the tail thread's local history -- the state
         * a sequential scan would have ended with -- is copied back after
         * the join, so the session still carries the same history forward. */
        long nthreads = 1;
        const char * const thr_env =
            getenv("DS4_QWEN4EXP_PLE_GATHER_THREADS");
        if (thr_env && thr_env[0]) {
            char *end = NULL;
            long v = strtol(thr_env, &end, 10);
            if (end != thr_env && v >= 0) nthreads = v;
        } else {
            nthreads = sysconf(_SC_NPROCESSORS_ONLN);
            if (nthreads > 16) nthreads = 16;
            if (nthreads < 1) nthreads = 1;
        }
        if (nthreads <= 1 || n_tokens < 64u) nthreads = 1;

        if (nthreads == 1) {
            ds4_ple_row_ids(&s->ple_constants, &s->ple_history, tokens,
                            n_tokens, s->ple_ids);
            for (uint32_t t = 0; t < n_tokens; t++) {
                for (uint32_t h = 0; h < DS4_N_PLE_HEAD; h++) {
                    const uint64_t row =
                        s->ple_ids[(uint64_t)t * DS4_N_PLE_HEAD + h];
                    const uint8_t *quant = ds4_qwen4exp_ple_row(&w->ple, row);
                    if (!quant) {
                        fprintf(stderr,
                                "ds4: qwen4exp block %u hashed token %u head %u "
                                "to row %" PRIu64 ", past the %" PRIu64 " the "
                                "table holds\n",
                                il, t, h, row, w->ple.rows);
                        return false;
                    }
                    ds4_ple_dequant_iq4_nl(
                        quant, blocks_per_row,
                        s->ple_rows_host + (uint64_t)t * DS4_N_PLE_EMBD +
                            (uint64_t)h * DS4_N_PLE_ROW_DIM);
                }
            }
        } else {
            if ((uint32_t)nthreads > n_tokens) nthreads = (long)n_tokens;
            qw_ple_gather_seg *segs =
                calloc((size_t)nthreads, sizeof(*segs));
            pthread_t *tids = calloc((size_t)nthreads, sizeof(*tids));
            if (!segs || !tids) {
                free(segs);
                free(tids);
                fprintf(stderr,
                        "ds4: qwen4exp PLE gather could not stage %ld threads\n",
                        nthreads);
                return false;
            }
            const uint32_t per = (n_tokens + (uint32_t)nthreads - 1u) /
                                 (uint32_t)nthreads;
            int spawned = 0;
            for (long i = 0; i < nthreads; i++) {
                const uint32_t t0 = (uint32_t)i * per;
                if (t0 >= n_tokens) break;
                uint32_t t1 = t0 + per;
                if (t1 > n_tokens) t1 = n_tokens;
                segs[i].ple = &w->ple;
                segs[i].s = s;
                segs[i].tokens = tokens;
                segs[i].n_tokens = n_tokens;
                segs[i].t0 = t0;
                segs[i].t1 = t1;
                segs[i].blocks_per_row = blocks_per_row;
                segs[i].ok = 0;
                if (i == 0) {
                    qw_ple_gather_worker(&segs[0]);
                    spawned = 1;
                } else if (pthread_create(&tids[i], NULL,
                                          qw_ple_gather_worker,
                                          &segs[i]) == 0) {
                    spawned++;
                } else {
                    segs[i].ok = -1;   /* run it inline after the joins */
                }
            }
            for (long i = 1; i < nthreads; i++) {
                if (segs[i].t0 < n_tokens && segs[i].ok != -1) {
                    pthread_join(tids[i], NULL);
                }
            }
            for (long i = 0; i < nthreads; i++) {
                if (segs[i].t0 >= n_tokens) break;
                if (segs[i].ok == -1) qw_ple_gather_worker(&segs[i]);
                if (segs[i].ok != 1) {
                    fprintf(stderr,
                            "ds4: qwen4exp block %u gather segment %ld hashed "
                            "past the %" PRIu64 " rows the table holds\n",
                            il, i, w->ple.rows);
                    free(segs);
                    free(tids);
                    return false;
                }
            }
            free(segs);
            free(tids);
            /* The carried history is the state after the LAST token; re-warm
             * a fresh history over the final (ngram - 1) tokens and leave it
             * in the session -- exactly what the sequential scan leaves. */
            ds4_ple_history_reset(&s->ple_constants, &s->ple_history);
            const uint32_t tail =
                s->ple_constants.ngram_size - 1u;
            if (tail > 0u && n_tokens > tail) {
                uint64_t scratch[DS4_PLE_MAX_NGRAM * DS4_PLE_MAX_HEADS];
                (void)ds4_ple_row_ids(&s->ple_constants, &s->ple_history,
                                      tokens + (n_tokens - tail), tail,
                                      scratch);
            }
        }
    }
    if (gather_t0 != 0.0) {
        fprintf(stderr,
                "ds4: qwen4exp PLE gather %u rows took %.3f ms\n",
                n_tokens, (now_sec() - gather_t0) * 1000.0);
    }
    if (!ds4_gpu_tensor_write(s->ple_rows, 0, s->ple_rows_host,
                              (uint64_t)n_tokens * DS4_N_PLE_EMBD * sizeof(float))) {
        fprintf(stderr, "ds4: qwen4exp n-gram row upload failed at block %u\n", il);
        return false;
    }
    return true;
}

/* The device half: the two Q8_0 projections, the three grouped norms, the gate
 * and the dilated convolution, over the rows the gather uploaded.  This is the
 * `ple_block` op boundary of the MLX reference dump. */
static bool qwen4exp_graph_ple_block(ds4_qwen4exp_session *s,
                                     const ds4_model *m,
                                     const ds4_qwen4exp_layer_weights *l,
                                     uint32_t n_tokens,
                                     uint32_t il) {
    /* One slab per tensor.  The six tensors of this block are resolved
     * against the shard mapping that actually holds each of them: the
     * production artifact is a four-way split, a boundary can fall inside a
     * block, and one mapping with six offsets would read plausible numbers
     * out of the wrong file. */
    const ds4_gpu_qwen4exp_slab key_slab = QW_SLAB(m, l->ple_key);
    const ds4_gpu_qwen4exp_slab value_slab = QW_SLAB(m, l->ple_value);
    const ds4_gpu_qwen4exp_slab norm_key_slab = QW_SLAB(m, l->ple_norm_key);
    const ds4_gpu_qwen4exp_slab norm_query_slab = QW_SLAB(m, l->ple_norm_query);
    const ds4_gpu_qwen4exp_slab norm_conv_slab = QW_SLAB(m, l->ple_norm_conv);
    const ds4_gpu_qwen4exp_slab conv_slab = QW_SLAB(m, l->ple_conv1d);

    if (!ds4_gpu_qwen4exp_ple_block_tensor(
            s->hyper, s->ple_conv_state, s->ple_conv_snapshot,
            s->spec_snapshot_rows,
            s->ple_key, s->ple_aux, s->ple_value,
            s->ple_rows, &key_slab, &value_slab, &norm_key_slab,
            &norm_query_slab, &norm_conv_slab, &conv_slab,
            DS4_N_EMBD, DS4_N_HC, DS4_N_PLE_EMBD,
            DS4_N_PLE_CONV, DS4_N_NGRAM, n_tokens, DS4_RMS_EPS,
            l->ple_norm_key_offset, l->ple_norm_query_offset,
            l->ple_norm_conv_offset, 1)) {
        fprintf(stderr, "ds4: qwen4exp PLE block %u refused\n", il);
        return false;
    }
    return true;
}

/* The GDN block.  head_layout is TILED for anything the loader bound: the
 * llama.cpp converter reorders the value heads into ggml's broadcast order and
 * the binder does not undo it, so value head j reads key head j % n_key_head. */
static bool qwen4exp_graph_gdn_block(ds4_qwen4exp_session *s,
                                     const ds4_qwen4exp_layer_weights *l,
                                     const ds4_model *m,
                                     uint32_t il,
                                     uint32_t n_tokens) {
    const ds4_qwen4exp_config *cfg = &g_ds4_qwen4exp;

    if (!ds4_qwen4exp_matmul_q8_0(s->gdn_qkv, QW_MAP(m, l->attn_qkv),
                                    QW_MAPSZ(m, l->attn_qkv), QW_OFF(l->attn_qkv),
                                    DS4_N_EMBD, cfg->gdn_qkv_dim,
                                    s->mixed, n_tokens) ||
        !ds4_qwen4exp_matmul_q8_0(s->gdn_gate, QW_MAP(m, l->attn_gate),
                                    QW_MAPSZ(m, l->attn_gate), QW_OFF(l->attn_gate),
                                    DS4_N_EMBD, cfg->gdn_inner,
                                    s->mixed, n_tokens) ||
        !ds4_qwen4exp_matmul_f32(s->gdn_alpha, QW_MAP(m, l->ssm_alpha),
                                   QW_MAPSZ(m, l->ssm_alpha), QW_OFF(l->ssm_alpha),
                                   DS4_N_EMBD, DS4_N_GDN_VALUE_HEAD,
                                   s->mixed, n_tokens) ||
        !ds4_qwen4exp_matmul_f32(s->gdn_beta, QW_MAP(m, l->ssm_beta),
                                   QW_MAPSZ(m, l->ssm_beta), QW_OFF(l->ssm_beta),
                                   DS4_N_EMBD, DS4_N_GDN_VALUE_HEAD,
                                   s->mixed, n_tokens)) {
        fprintf(stderr, "ds4: qwen4exp GDN projections failed at block %u\n", il);
        return false;
    }

    /* One slab per tensor.  These four are four GGUF tensors; resolving
     * ssm_a, ssm_dt.bias and ssm_norm through the convolution's mapping would
     * read the right offsets out of the wrong shard on a split set. */
    const ds4_gpu_qwen4exp_slab conv_slab = QW_SLAB(m, l->ssm_conv1d);
    const ds4_gpu_qwen4exp_slab a_log_slab = QW_SLAB(m, l->ssm_a);
    const ds4_gpu_qwen4exp_slab dt_bias_slab = QW_SLAB(m, l->ssm_dt_bias);
    const ds4_gpu_qwen4exp_slab ssm_norm_slab = QW_SLAB(m, l->ssm_norm);

    const int rc = (n_tokens > 1u)
        ? ds4_gpu_qwen4exp_gdn_prefill(
              s->gdn_out, s->gdn_conv[il], s->gdn_state[il],
              s->gdn_conv_snapshot[il], s->gdn_state_snapshot[il],
              s->spec_snapshot_rows, s->gdn_qkv,
              s->gdn_alpha, s->gdn_beta, s->gdn_gate,
              &conv_slab, &a_log_slab, &dt_bias_slab, &ssm_norm_slab,
              DS4_N_GDN_KEY_HEAD, DS4_N_GDN_VALUE_HEAD, n_tokens,
              DS4_QWEN4EXP_GDN_HEADS_TILED, DS4_RMS_EPS, DS4_RMS_EPS)
        : ds4_gpu_qwen4exp_gdn_decode(
              s->gdn_out, s->gdn_conv[il], s->gdn_state[il], s->gdn_qkv,
              s->gdn_alpha, s->gdn_beta, s->gdn_gate,
              &conv_slab, &a_log_slab, &dt_bias_slab, &ssm_norm_slab,
              DS4_N_GDN_KEY_HEAD, DS4_N_GDN_VALUE_HEAD, 1u,
              DS4_QWEN4EXP_GDN_HEADS_TILED, DS4_RMS_EPS, DS4_RMS_EPS);
    if (!rc) {
        fprintf(stderr, "ds4: qwen4exp GDN block %u refused\n", il);
        return false;
    }

    if (!ds4_qwen4exp_matmul_q8_0(s->block_out, QW_MAP(m, l->ssm_out),
                                    QW_MAPSZ(m, l->ssm_out), QW_OFF(l->ssm_out),
                                    cfg->gdn_inner, DS4_N_EMBD,
                                    s->gdn_out, n_tokens)) {
        fprintf(stderr, "ds4: qwen4exp GDN output projection failed at block %u\n", il);
        return false;
    }
    return true;
}

static bool qwen4exp_graph_qsa_block(ds4_qwen4exp_session *s,
                                     const ds4_qwen4exp_layer_weights *l,
                                     const ds4_model *m,
                                     uint32_t il,
                                     uint32_t n_tokens) {
    const ds4_qwen4exp_config *cfg = &g_ds4_qwen4exp;
    const uint32_t head_dim = DS4_N_HEAD_DIM;
    const uint32_t q_width = DS4_N_HEAD * head_dim;
    const uint32_t ratio = DS4_N_INDEXER_COMPRESS_RATIO;
    const uint32_t top_k = DS4_N_INDEXER_TOP_K / ratio;
    const uint32_t max_selected = DS4_QWEN4EXP_MAX_SELECTED(top_k, ratio);
    const uint32_t n_blocks = s->plan.n_ctx / ratio;
    const uint32_t pos0 = s->pos;

    /* Pre-quantize s->mixed ONCE for Q, K, and V projections. */
    const uint64_t q8_blocks = (uint64_t)DS4_N_EMBD / 32u;
    const uint64_t q8_qbytes = (uint64_t)n_tokens * q8_blocks * 32u;
    const uint64_t q8_soff = (q8_qbytes + 15u) & ~15ull;

    if (!ds4_qwen4exp_quantize_q8_0(s->qsa_mixed_q8, 0, q8_soff, s->mixed,
                                    DS4_N_EMBD, n_tokens)) {
        fprintf(stderr, "ds4: qwen4exp QSA pre-quantize failed at block %u\n", il);
        return false;
    }

    /* Projections.  q is DOUBLED: half query, half output gate. */
    if (!ds4_qwen4exp_matmul_q8_0_preq(s->qsa_doubled, QW_MAP(m, l->attn_q),
                                       QW_MAPSZ(m, l->attn_q), QW_OFF(l->attn_q),
                                       DS4_N_EMBD, cfg->qsa_q_dim,
                                       s->qsa_mixed_q8, 0, q8_soff, n_tokens) ||
        !ds4_qwen4exp_matmul_q8_0_preq(s->qsa_kin, QW_MAP(m, l->attn_k),
                                       QW_MAPSZ(m, l->attn_k), QW_OFF(l->attn_k),
                                       DS4_N_EMBD, cfg->qsa_kv_dim,
                                       s->qsa_mixed_q8, 0, q8_soff, n_tokens) ||
        !ds4_qwen4exp_matmul_q8_0_preq(s->qsa_vin, QW_MAP(m, l->attn_v),
                                       QW_MAPSZ(m, l->attn_v), QW_OFF(l->attn_v),
                                       DS4_N_EMBD, cfg->qsa_kv_dim,
                                       s->qsa_mixed_q8, 0, q8_soff, n_tokens)) {
        fprintf(stderr, "ds4: qwen4exp QSA projections failed at block %u\n", il);
        return false;
    }

    /* Fused Q-Prep: split doubled-query, per-head RMS norm, and RoPE in 1 launch */
    if (!ds4_gpu_qwen4exp_qsa_prep_q_fused_dpos_tensor(
            s->qsa_q, s->qsa_gate, s->qsa_doubled, s->attn_q_norm[il],
            s->inv_freq, n_tokens, DS4_N_HEAD, head_dim, DS4_N_ROT, pos0,
            DS4_RMS_EPS, l->attn_q_norm_offset, s->d_pos)) {
        fprintf(stderr, "ds4: qwen4exp fused Q-prep failed at block %u\n", il);
        return false;
    }

    /* Fused KV-Prep & Append: per-head RMS norm on K, RoPE on K, and direct write into
     * k_cache and v_cache in 1 launch (no sync CPU memcpy) */
    if (!ds4_gpu_qwen4exp_qsa_prep_kv_append_fused_dpos_tensor(
            s->qsa_k[il], s->qsa_v[il], s->qsa_kin, s->qsa_kin, s->qsa_vin,
            s->attn_k_norm[il], s->inv_freq, pos0, n_tokens, DS4_N_HEAD_KV,
            head_dim, DS4_N_ROT, s->plan.n_ctx, DS4_RMS_EPS, l->attn_k_norm_offset,
            s->d_pos)) {
        fprintf(stderr, "ds4: qwen4exp fused KV prep & append failed at block %u\n", il);
        return false;
    }

    /* Indexer: BF16 projections, the exact unrotated key tape, pooled blocks,
     * block scores, top-k, and the "keep OR own" expansion.
     *
     * The k projection feeds the pool update, which runs on every call so the
     * tape stays exact for the positions that DO index later.  The q
     * projection feeds only the norm/rope/scores/select chain below, and that
     * chain runs only when `indexed`.  Below the budget its result was written
     * and never read, so it moves under the same condition.  This removes a
     * computation whose output no reader observes; it does not change one. */
    const bool indexed = (uint64_t)pos0 + n_tokens > DS4_N_INDEXER_TOP_K;

    if (!ds4_qwen4exp_matmul_bf16(s->idx_k, QW_MAP(m, l->indexer_k_proj),
                                  QW_MAPSZ(m, l->indexer_k_proj),
                                  QW_OFF(l->indexer_k_proj),
                                  DS4_N_EMBD, cfg->indexer_k_dim,
                                  s->mixed, n_tokens)) {
        fprintf(stderr, "ds4: qwen4exp indexer projections failed at block %u\n", il);
        return false;
    }
    if (indexed &&
        !ds4_qwen4exp_matmul_bf16(s->idx_q, QW_MAP(m, l->indexer_q_proj),
                                  QW_MAPSZ(m, l->indexer_q_proj),
                                  QW_OFF(l->indexer_q_proj),
                                  DS4_N_EMBD, cfg->indexer_q_dim,
                                  s->mixed, n_tokens)) {
        fprintf(stderr, "ds4: qwen4exp indexer projections failed at block %u\n", il);
        return false;
    }
    if (!ds4_gpu_qwen4exp_qsa_indexer_pool_update_dpos_tensor(
            s->idx_pool[il], s->idx_tape[il], s->idx_k, s->idx_k_norm[il],
            s->inv_freq, pos0, n_tokens, s->plan.n_ctx,
            DS4_N_INDEXER_HEAD_DIM, ratio, DS4_N_ROT, DS4_RMS_EPS,
            l->indexer_k_norm_offset, s->d_pos)) {
        fprintf(stderr, "ds4: qwen4exp indexer pool update failed at block %u\n", il);
        return false;
    }

    const ds4_gpu_tensor *sel = NULL;
    const ds4_gpu_tensor *counts = NULL;
    if (indexed) {
        /* The indexer's query norm.  It was bound and never applied: the k side
         * is normalised inside the pool update, and the q side simply had no
         * call.  llama.cpp applies index_q_norm before the rope, so it goes
         * here, head-wise over the indexer's own head dimension. */
        if (!ds4_gpu_qwen4exp_head_rms_norm_tensor(
                s->idx_q, s->idx_q, s->idx_q_norm[il],
                n_tokens * DS4_N_INDEXER_HEAD, DS4_N_INDEXER_HEAD_DIM,
                DS4_RMS_EPS, l->indexer_q_norm_offset) ||
            !ds4_gpu_qwen4exp_rope_head_tensor(s->idx_q, s->inv_freq, n_tokens,
                                               DS4_N_INDEXER_HEAD, DS4_N_INDEXER_HEAD_DIM,
                                               DS4_N_ROT, pos0) ||
            !ds4_gpu_qwen4exp_qsa_indexer_scores_tensor(
                s->idx_scores, s->idx_q, s->idx_pool[il], n_tokens, n_blocks,
                DS4_N_INDEXER_HEAD, DS4_N_INDEXER_HEAD_DIM, pos0, ratio) ||
            !ds4_gpu_indexer_topk_tensor(s->idx_topk, s->idx_scores, n_blocks,
                                         n_tokens, top_k) ||
            !ds4_gpu_qwen4exp_qsa_indexer_select_tensor(
                s->idx_selected, s->idx_counts, s->idx_scores, s->idx_topk,
                n_tokens, n_blocks, top_k, pos0, ratio, max_selected)) {
            fprintf(stderr, "ds4: qwen4exp indexer selection failed at block %u\n", il);
            return false;
        }
        sel = s->idx_selected;
        counts = s->idx_counts;
    }

    const float scale = 1.0f / sqrtf((float)head_dim);
    if (!ds4_gpu_qwen4exp_qsa_attention_dpos_tensor(
            s->qsa_out, s->qsa_q, s->qsa_k[il], s->qsa_v[il], sel, counts,
            n_tokens, DS4_N_HEAD, DS4_N_HEAD_KV, head_dim, pos0,
            s->plan.n_ctx, max_selected, scale, s->d_pos) ||
        !ds4_gpu_qwen4exp_qsa_output_gate_tensor(s->qsa_out, s->qsa_gate,
                                                 n_tokens * q_width)) {
        fprintf(stderr, "ds4: qwen4exp QSA attention failed at block %u\n", il);
        return false;
    }

    if (!ds4_qwen4exp_matmul_q8_0(s->block_out, QW_MAP(m, l->attn_output),
                                    QW_MAPSZ(m, l->attn_output),
                                    QW_OFF(l->attn_output),
                                    q_width, DS4_N_EMBD, s->qsa_out, n_tokens)) {
        fprintf(stderr, "ds4: qwen4exp QSA output projection failed at block %u\n", il);
        return false;
    }
    return true;
}

/* Which head mapping has already been announced resident, so a per-session
 * setup call says it once rather than once a row. */
static const void *g_qwen4exp_head_resident = NULL;

/* The head block's own stage split, on the same DS4_MTP_HEAD_TIME switch as
 * the head step's.  One run gives both, and the block's six stages sum to the
 * step's "block" line.  As there, the synchronises make the block slower, so it
 * attributes time and does not measure it. */
enum {
    QW_HB_T_ATTN_MIX = 0,
    QW_HB_T_QSA,
    QW_HB_T_ATTN_INJECT,
    QW_HB_T_FFN_MIX,
    QW_HB_T_MOE,
    QW_HB_T_FFN_INJECT,
    QW_HB_T_N
};

static const char *const qw_hb_stage_names[QW_HB_T_N] = {
    "attn mixer", "QSA", "attn inject", "ffn mixer", "MoE", "ffn inject"
};

static uint64_t qw_hb_stage_ns[QW_HB_T_N];
static uint64_t qw_hb_stage_calls;
static int      qw_hb_timing = -1;

static uint64_t qw_hb_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static void qw_hb_time_dump(void) {
    if (qw_hb_stage_calls == 0) return;
    const double n = (double)qw_hb_stage_calls;
    uint64_t total = 0;
    for (int i = 0; i < QW_HB_T_N; i++) total += qw_hb_stage_ns[i];
    fprintf(stderr, "ds4: MTP head BLOCK stage split over %llu calls, ms each\n",
            (unsigned long long)qw_hb_stage_calls);
    for (int i = 0; i < QW_HB_T_N; i++) {
        fprintf(stderr, "ds4:   %-13s %8.3f  %5.1f%%\n",
                qw_hb_stage_names[i], (double)qw_hb_stage_ns[i] / n / 1e6,
                total ? 100.0 * (double)qw_hb_stage_ns[i] / (double)total : 0.0);
    }
    fprintf(stderr, "ds4:   %-13s %8.3f\n", "TOTAL", (double)total / n / 1e6);
}

static void qw_moe_time_dump(void);   /* below, same switch */

static int qw_hb_time_on(void) {
    if (qw_hb_timing < 0) {
        const char *v = getenv("DS4_MTP_HEAD_TIME");
        qw_hb_timing = (v && *v && strcmp(v, "0") != 0) ? 1 : 0;
        if (qw_hb_timing) { atexit(qw_hb_time_dump); atexit(qw_moe_time_dump); }
    }
    return qw_hb_timing;
}

#define QW_HB_TICK(slot)                                                      \
    do {                                                                      \
        if (timing) {                                                         \
            (void)ds4_gpu_synchronize();                                      \
            const uint64_t qw_hb_now = qw_hb_now_ns();                        \
            qw_hb_stage_ns[(slot)] += qw_hb_now - tmark;                      \
            tmark = qw_hb_now;                                                \
        }                                                                     \
    } while (0)

/* The MoE's four stages, with the TOWER'S OWN MoE at width one as the control.
 *
 * The head's block costs about eight times what one of the target's layers
 * costs in a serial decode, and 5.2 ms of its 6.3 ms is the MoE.  Whether that
 * is the head's block doing something the tower's does not, or the MoE simply
 * costing that at width one, is not answerable from the head's number alone.
 * So both are recorded: lane 1 is the call inside the head block, lane 0 a
 * tower layer at the SAME width.  Tower calls at other widths are not
 * recorded -- they are a different question.
 *
 * Same DS4_MTP_HEAD_TIME switch as the step's and the block's, same
 * synchronise-per-stage caveat: this attributes time, it does not measure it.
 */
enum { QW_MOE_T_ROUTER = 0, QW_MOE_T_SELECT, QW_MOE_T_ROUTED,
       QW_MOE_T_SHARED, QW_MOE_T_N };

static const char *const qw_moe_stage_names[QW_MOE_T_N] = {
    "router f32", "router select", "routed experts", "shared expert"
};

static uint64_t qw_moe_stage_ns[2][QW_MOE_T_N];
static uint64_t qw_moe_stage_calls[2];
static int      qw_moe_in_head_block = 0;

static void qw_moe_time_dump(void) {
    for (int lane = 0; lane < 2; lane++) {
        if (qw_moe_stage_calls[lane] == 0) continue;
        const double n = (double)qw_moe_stage_calls[lane];
        uint64_t total = 0;
        for (int i = 0; i < QW_MOE_T_N; i++) total += qw_moe_stage_ns[lane][i];
        fprintf(stderr,
                "ds4: MoE stage split, %s, %llu calls at width 1, ms each\n",
                lane ? "HEAD block " : "tower layer",
                (unsigned long long)qw_moe_stage_calls[lane]);
        for (int i = 0; i < QW_MOE_T_N; i++) {
            fprintf(stderr, "ds4:   %-15s %8.3f\n", qw_moe_stage_names[i],
                    (double)qw_moe_stage_ns[lane][i] / n / 1e6);
        }
        fprintf(stderr, "ds4:   %-15s %8.3f\n", "TOTAL",
                (double)total / n / 1e6);
    }
}

#define QW_MOE_TICK(slot)                                                     \
    do {                                                                      \
        if (moe_timing) {                                                     \
            (void)ds4_gpu_synchronize();                                      \
            const uint64_t qw_moe_now = qw_hb_now_ns();                       \
            qw_moe_stage_ns[moe_lane][(slot)] += qw_moe_now - moe_mark;       \
            moe_mark = qw_moe_now;                                            \
        }                                                                     \
    } while (0)

static bool qwen4exp_graph_moe_block(ds4_qwen4exp_session *s,
                                     const ds4_qwen4exp_layer_weights *l,
                                     const ds4_model *m,
                                     uint32_t il,
                                     uint32_t n_tokens) {
    /* Router: dense F32 logits, then a top-k with softmax over the SELECTED
     * logits alone.  No bias, no renormalisation, no scale. */
    /* The router's logits take the decode-order route too.  THIS FIXTURE
     * CANNOT SEE THE DIFFERENCE -- its routed expert slabs are holes, so the
     * softmax weights this feeds scale zeros -- but on a real checkpoint they
     * scale live experts, and the top-k selection they drive is discrete, so a
     * row-count difference here can move which experts a row uses. */
    const int moe_timing = qw_hb_time_on() && n_tokens == 1u;
    const int moe_lane = qw_moe_in_head_block ? 1 : 0;
    uint64_t moe_mark = moe_timing ? qw_hb_now_ns() : 0;

    if (!ds4_qwen4exp_matmul_f32(s->moe_logits, QW_MAP(m, l->ffn_gate_inp),
                                   QW_MAPSZ(m, l->ffn_gate_inp),
                                   QW_OFF(l->ffn_gate_inp),
                                   DS4_N_EMBD, DS4_N_EXPERT, s->mixed, n_tokens)) {
        fprintf(stderr, "ds4: qwen4exp router failed at block %u\n", il);
        return false;
    }
    QW_MOE_TICK(QW_MOE_T_ROUTER);
    if (!ds4_gpu_qwen4exp_router_select_tensor(s->moe_selected, s->moe_weights,
                                               s->moe_logits, DS4_N_EXPERT,
                                               DS4_N_EXPERT_USED, n_tokens)) {
        fprintf(stderr, "ds4: qwen4exp router select failed at block %u\n", il);
        return false;
    }
    QW_MOE_TICK(QW_MOE_T_SELECT);

    /* Expert types vary per block and per artifact: UD-Q4_K_XL stores block 2
     * gate/up at Q5_K and five blocks' down at Q8_0, the MQ recipes go further,
     * and the MTP head is Q8_0 throughout.  There is no type check here: the
     * loader binds these tensors from ds4_qwen4exp_moe_types.h and the kernels
     * expand the same table, so a bound expert tensor is one the GEMM decodes.
     *
     * Every slab carries its own shard mapping, so a block whose expert
     * tensors landed in different shards runs like any other: blocks 11 and 41
     * of UD-Q4_K_XL do. */
    const ds4_gpu_qwen4exp_slab gate_slab = QW_SLAB(m, l->ffn_gate_exps);
    const ds4_gpu_qwen4exp_slab up_slab = QW_SLAB(m, l->ffn_up_exps);
    const ds4_gpu_qwen4exp_slab down_slab = QW_SLAB(m, l->ffn_down_exps);
    if (!ds4_gpu_qwen4exp_routed_moe_tensor(
            s->block_out, s->moe_mid, s->moe_down_partial,
            &gate_slab, &up_slab, &down_slab,
            DS4_N_EMBD, DS4_N_FF_EXP, DS4_N_EMBD,
            s->moe_selected, s->moe_weights, DS4_N_EXPERT, DS4_N_EXPERT_USED,
            s->mixed, n_tokens, DS4_N_EXPERT_USED * DS4_N_FF_EXP)) {
        fprintf(stderr, "ds4: qwen4exp routed experts failed at block %u\n", il);
        return false;
    }
    QW_MOE_TICK(QW_MOE_T_ROUTED);

    /* The shared expert ADDS into block_out: routed + sigmoid(gate) * shared. */
    const ds4_gpu_qwen4exp_slab sh_router_slab = QW_SLAB(m, l->ffn_gate_inp_shexp);
    const ds4_gpu_qwen4exp_slab sh_gate_slab = QW_SLAB(m, l->ffn_gate_shexp);
    const ds4_gpu_qwen4exp_slab sh_up_slab = QW_SLAB(m, l->ffn_up_shexp);
    const ds4_gpu_qwen4exp_slab sh_down_slab = QW_SLAB(m, l->ffn_down_shexp);
    if (!ds4_gpu_qwen4exp_shared_expert_tensor(
            s->block_out, s->shexp_mid, s->shexp_gate,
            &sh_router_slab, &sh_gate_slab, &sh_up_slab, &sh_down_slab,
            DS4_N_EMBD, DS4_N_FF_SHEXP, DS4_N_EMBD, s->mixed, n_tokens)) {
        fprintf(stderr, "ds4: qwen4exp shared expert failed at block %u\n", il);
        return false;
    }
    QW_MOE_TICK(QW_MOE_T_SHARED);
    if (moe_timing) qw_moe_stage_calls[moe_lane]++;
    return true;
}

/* One gated residual step: mix down to the block input, run the block, inject
 * the block output back into every hyper-connection stream. */
#ifdef DS4_TEST_HOOKS
/* The slice sequence one forward actually ran, recorded at each slice's own
 * call site.  A test asserts the whole sequence, so a slice that is dropped,
 * duplicated or moved -- the PLE block sliding from the head of the layer to
 * the tail, say -- fails rather than passing because the forward still
 * returned true. */
typedef enum {
    QW_SLICE_EMBED = 1,
    QW_SLICE_PLE,
    QW_SLICE_ATTN_MIX,
    QW_SLICE_GDN,
    QW_SLICE_QSA,
    QW_SLICE_ATTN_INJECT,
    QW_SLICE_FFN_MIX,
    QW_SLICE_MOE,
    QW_SLICE_FFN_INJECT,
    QW_SLICE_FINAL_MIX,
    QW_SLICE_HEAD,
} qwen4exp_slice;

#define DS4_QWEN4EXP_MAX_TRACE 1024
static uint8_t  g_qwen4exp_trace[DS4_QWEN4EXP_MAX_TRACE];
static uint32_t g_qwen4exp_trace_len = 0;
static bool     g_qwen4exp_trace_overflow = false;

/* Where a forward's time goes, by slice.
 *
 * The trace records the ORDER of the slices; with DS4_QWEN4EXP_TIME_SLICES set
 * it also records their COST.  Each trace point closes the command batch,
 * waits for the device and opens a new one, so the wall time between two trace
 * points is that slice's GPU work and nothing else.  That serialisation is the
 * measurement: it removes the overlap between slices, so the numbers are a
 * ranking of where the work is, not a prediction of the total.  The unmeasured
 * run is the one that states throughput.
 *
 * ds4_gpu_end_commands / ds4_gpu_begin_commands around a profiled region is
 * what the Metal Q8 prefill profiler already does.  Diagnostic only, and off
 * unless the variable is set.
 *
 * The table prints at the end of the forward that fills it. */
static const char *const g_qwen4exp_slice_name[] = {
    "?", "embed", "ple", "attn_mix", "gdn", "qsa", "attn_inject",
    "ffn_mix", "moe", "ffn_inject", "final_mix", "head",
};
#define DS4_QWEN4EXP_N_SLICE \
    ((int)(sizeof(g_qwen4exp_slice_name) / sizeof(g_qwen4exp_slice_name[0])))

static double   g_qwen4exp_slice_sec[DS4_QWEN4EXP_N_SLICE];
static uint32_t g_qwen4exp_slice_hits[DS4_QWEN4EXP_N_SLICE];
static double   g_qwen4exp_slice_mark = 0.0;
static int      g_qwen4exp_time_slices = -1;

static void qwen4exp_trace(qwen4exp_slice slice) {
    if (g_qwen4exp_trace_len < DS4_QWEN4EXP_MAX_TRACE) {
        g_qwen4exp_trace[g_qwen4exp_trace_len++] = (uint8_t)slice;
    } else {
        g_qwen4exp_trace_overflow = true;
    }

    if (g_qwen4exp_time_slices < 0) {
        g_qwen4exp_time_slices = getenv("DS4_QWEN4EXP_TIME_SLICES") != NULL;
    }
    if (!g_qwen4exp_time_slices) return;
    if (!ds4_gpu_end_commands() || !ds4_gpu_synchronize() ||
        !ds4_gpu_begin_commands()) {
        return;
    }
    const double now = now_sec();
    if (g_qwen4exp_slice_mark > 0.0 && (int)slice < DS4_QWEN4EXP_N_SLICE) {
        g_qwen4exp_slice_sec[(int)slice] += now - g_qwen4exp_slice_mark;
        g_qwen4exp_slice_hits[(int)slice]++;
    }
    g_qwen4exp_slice_mark = now;
}

static void qwen4exp_trace_report(uint32_t n_tokens) {
    if (g_qwen4exp_time_slices <= 0) return;
    double total = 0.0;
    for (int i = 1; i < DS4_QWEN4EXP_N_SLICE; i++) total += g_qwen4exp_slice_sec[i];
    if (!(total > 0.0)) return;
    fprintf(stderr, "ds4: qwen4exp slice times over %u rows, %.3f ms serialised "
                    "total\n", n_tokens, total * 1000.0);
    for (int i = 1; i < DS4_QWEN4EXP_N_SLICE; i++) {
        if (!g_qwen4exp_slice_hits[i]) continue;
        fprintf(stderr, "ds4:   %-12s %8.3f ms  %5.1f%%  over %u calls\n",
                g_qwen4exp_slice_name[i], g_qwen4exp_slice_sec[i] * 1000.0,
                100.0 * g_qwen4exp_slice_sec[i] / total,
                g_qwen4exp_slice_hits[i]);
        g_qwen4exp_slice_sec[i] = 0.0;
        g_qwen4exp_slice_hits[i] = 0;
    }
    g_qwen4exp_slice_mark = 0.0;
}

#define QW_TRACE(slice) qwen4exp_trace(slice)
#define QW_TRACE_REPORT(n) qwen4exp_trace_report(n)
#else
#define QW_TRACE(slice) ((void)0)
#define QW_TRACE_REPORT(n) ((void)0)
#endif


static bool qwen4exp_graph_residual(ds4_qwen4exp_session *s,
                                    const ds4_model *m,
                                    const ds4_tensor *norm,
                                    const ds4_tensor *down,
                                    const ds4_tensor *up,
                                    const ds4_tensor *inject,
                                    float norm_weight_offset,
                                    uint32_t n_tokens) {
#ifdef DS4_TEST_HOOKS
    /* The tower's final mixer is the only call with no inject head.  Copy what
     * it is about to read, inside this batch so the ordering is the GPU's and
     * not the host's. */
    if (!inject && s->final_mixer_snapshot) {
        (void)ds4_gpu_tensor_copy(s->final_mixer_snapshot, 0, s->hyper, 0,
                                  (uint64_t)n_tokens * DS4_N_HC * DS4_N_EMBD *
                                  sizeof(float));
    }
#endif
    /* One slab per tensor.  These four weights are four GGUF tensors and a
     * shard boundary can fall between any two of them; resolving them all
     * through the norm's mapping would read the right offsets out of the wrong
     * shard on a split set and refuse nowhere. */
    const ds4_gpu_qwen4exp_slab norm_slab = QW_SLAB(m, norm);
    const ds4_gpu_qwen4exp_slab down_slab = QW_SLAB(m, down);
    const ds4_gpu_qwen4exp_slab up_slab = QW_SLAB(m, up);
    const ds4_gpu_qwen4exp_slab inject_slab =
        inject ? QW_SLAB(m, inject) : (ds4_gpu_qwen4exp_slab){0};

    return ds4_gpu_qwen4exp_hc_mixer_tensor(
        s->mixed, inject ? s->inject : NULL, s->normed, s->lowrank, s->wide,
        s->hyper, &norm_slab, &down_slab, &up_slab,
        inject ? &inject_slab : NULL,
        DS4_N_EMBD, DS4_N_HC, DS4_N_HC_LOWRANK, n_tokens,
        DS4_HC_EPS, norm_weight_offset, 1) != 0;
}

/* Two fixed launch sequences per layer.  The GDN reads and updates tensor
 * state in place and takes no position argument; QSA, PLE and token uploads
 * stay outside capture.  Every replay still computes on the current inputs.
 * Only small decode/verify widths capture; prefill retains its eager kernels.
 */
static bool qwen4exp_graph_layer_island_encode(
        ds4_qwen4exp_session *s, const ds4_qwen4exp_layer_weights *l,
        const ds4_model *m, uint32_t il, uint32_t n_tokens, uint32_t island) {
    if (island == 0u) {
        /* --- attention side ------------------------------------------- */
        if (!qwen4exp_graph_residual(s, m, l->hc_attn_norm, l->hc_attn_down,
                                     l->hc_attn_up, l->hc_attn_inject,
                                     l->hc_attn_norm_offset, n_tokens)) {
            fprintf(stderr, "ds4: qwen4exp attention mixer failed at block %u\n", il);
            return false;
        }
        QW_TRACE(QW_SLICE_ATTN_MIX);
        if (!l->is_full_attention) {
            if (!qwen4exp_graph_gdn_block(s, l, m, il, n_tokens)) return false;
            QW_TRACE(QW_SLICE_GDN);
        }
    } else {
        if (!ds4_gpu_qwen4exp_hc_inject_tensor(s->hyper, s->hyper, s->block_out,
                                               s->inject, DS4_N_EMBD, DS4_N_HC,
                                               n_tokens)) {
            fprintf(stderr, "ds4: qwen4exp attention inject failed at block %u\n", il);
            return false;
        }
        QW_TRACE(QW_SLICE_ATTN_INJECT);

        /* --- feed-forward side ---------------------------------------- */
        if (!qwen4exp_graph_residual(s, m, l->hc_ffn_norm, l->hc_ffn_down,
                                     l->hc_ffn_up, l->hc_ffn_inject,
                                     l->hc_ffn_norm_offset, n_tokens)) {
            fprintf(stderr, "ds4: qwen4exp FFN mixer failed at block %u\n", il);
            return false;
        }
        QW_TRACE(QW_SLICE_FFN_MIX);
        if (!qwen4exp_graph_moe_block(s, l, m, il, n_tokens)) {
            return false;
        }
        QW_TRACE(QW_SLICE_MOE);
        if (!ds4_gpu_qwen4exp_hc_inject_tensor(s->hyper, s->hyper, s->block_out,
                                               s->inject, DS4_N_EMBD, DS4_N_HC,
                                               n_tokens)) {
            fprintf(stderr, "ds4: qwen4exp FFN inject failed at block %u\n", il);
            return false;
        }
        QW_TRACE(QW_SLICE_FFN_INJECT);
    }
    return true;
}

static bool qwen4exp_graph_layer_island(
        ds4_qwen4exp_session *s, const ds4_qwen4exp_layer_weights *l,
        const ds4_model *m, uint32_t il, uint32_t n_tokens, uint32_t island) {
    /* Profilers synchronize between operations, which is forbidden inside
     * capture.  They must also execute the individual call sites on replay. */
    const bool capture_ok =
        n_tokens <= (uint32_t)DS4_QWEN4EXP_MTP_MAX_COMMIT &&
        !qw_hb_time_on() && getenv("DS4_QWEN4EXP_TIME_SLICES") == NULL &&
        ds4_gpu_decode_graphs_supported();
    if (capture_ok) {
        ds4_decode_graph_key key;
        memset(&key, 0, sizeof(key));
        key.il = il;
        key.island = island;
        /* Snapshot writes are part of GDN: an ordinary two-row feed and a
         * two-row verify must not share an executable with different writes. */
        key.variant = n_tokens | (s->spec_snapshot_rows << 8u);
        key._pad = 0x5157454eu;  /* QWEN: separate from other model families */
        key.cur_hc = s->hyper;
        key.after_attn_hc = s;
        key.after_ffn_hc = (void *)l;
        key.attn_norm = (void *)m;
        const int state = ds4_gpu_decode_graph_begin(&key);
        if (state == 1) {
            /* Preserve the diagnostic slice sequence when host encoding is
             * skipped.  The synchronizing profiler takes the eager path. */
            if (island == 0u) {
                QW_TRACE(QW_SLICE_ATTN_MIX);
                if (!l->is_full_attention) QW_TRACE(QW_SLICE_GDN);
            } else {
                QW_TRACE(QW_SLICE_ATTN_INJECT);
                QW_TRACE(QW_SLICE_FFN_MIX);
                QW_TRACE(QW_SLICE_MOE);
                QW_TRACE(QW_SLICE_FFN_INJECT);
            }
            return true;
        }
        if (state == 0) {
#ifdef DS4_TEST_HOOKS
            const uint32_t trace_len = g_qwen4exp_trace_len;
            const bool trace_overflow = g_qwen4exp_trace_overflow;
#endif
            if (qwen4exp_graph_layer_island_encode(s, l, m, il, n_tokens,
                                                   island)) {
                if (ds4_gpu_decode_graph_end(&key) == 0) return true;
            } else {
                ds4_gpu_decode_graph_abort(&key);
            }
#ifdef DS4_TEST_HOOKS
            g_qwen4exp_trace_len = trace_len;
            g_qwen4exp_trace_overflow = trace_overflow;
#endif
            /* Failed capture executed nothing.  The backend retires it;
             * re-encode this call eagerly exactly once, including on error. */
        }
    }
    return qwen4exp_graph_layer_island_encode(s, l, m, il, n_tokens, island);
}


/* GDN layers have nothing between island 0 and island 1.  Capture both as
 * one executable so each of the 36 GDN layers pays one graph launch per
 * decode/verify width instead of two.  QSA layers keep the 0/2/1 split
 * because island 2 still sits between the halves. */
static bool qwen4exp_graph_gdn_full_layer(
        ds4_qwen4exp_session *s, const ds4_qwen4exp_layer_weights *l,
        const ds4_model *m, uint32_t il, uint32_t n_tokens) {
    const bool capture_ok =
        n_tokens <= (uint32_t)DS4_QWEN4EXP_MTP_MAX_COMMIT &&
        !qw_hb_time_on() && getenv("DS4_QWEN4EXP_TIME_SLICES") == NULL &&
        ds4_gpu_decode_graphs_supported();
    if (capture_ok) {
        ds4_decode_graph_key key;
        memset(&key, 0, sizeof(key));
        key.il = il;
        key.island = 0u;
        key.variant = n_tokens | (s->spec_snapshot_rows << 8u);
        key._pad = 0x47444e46u; /* GDNF: full GDN layer, distinct from half */
        key.cur_hc = s->hyper;
        key.after_attn_hc = s;
        key.after_ffn_hc = (void *)l;
        key.attn_norm = (void *)m;
        const int state = ds4_gpu_decode_graph_begin(&key);
        if (state == 1) {
            QW_TRACE(QW_SLICE_ATTN_MIX);
            QW_TRACE(QW_SLICE_GDN);
            QW_TRACE(QW_SLICE_ATTN_INJECT);
            QW_TRACE(QW_SLICE_FFN_MIX);
            QW_TRACE(QW_SLICE_MOE);
            QW_TRACE(QW_SLICE_FFN_INJECT);
            return true;
        }
        if (state == 0) {
#ifdef DS4_TEST_HOOKS
            const uint32_t trace_len = g_qwen4exp_trace_len;
            const bool trace_overflow = g_qwen4exp_trace_overflow;
#endif
            const bool ok =
                qwen4exp_graph_layer_island_encode(s, l, m, il, n_tokens, 0u) &&
                qwen4exp_graph_layer_island_encode(s, l, m, il, n_tokens, 1u);
            if (ok && ds4_gpu_decode_graph_end(&key) == 0) return true;
            ds4_gpu_decode_graph_abort(&key);
#ifdef DS4_TEST_HOOKS
            g_qwen4exp_trace_len = trace_len;
            g_qwen4exp_trace_overflow = trace_overflow;
#endif
        }
    }
    return qwen4exp_graph_layer_island_encode(s, l, m, il, n_tokens, 0u) &&
           qwen4exp_graph_layer_island_encode(s, l, m, il, n_tokens, 1u);
}

static bool qwen4exp_graph_qsa_island(
        ds4_qwen4exp_session *s, const ds4_qwen4exp_layer_weights *l,
        const ds4_model *m, uint32_t il, uint32_t n_tokens) {
    const bool qsa_capture_ok =
        n_tokens <= (uint32_t)DS4_QWEN4EXP_MTP_MAX_COMMIT &&
        (uint64_t)s->pos + n_tokens <= DS4_N_INDEXER_TOP_K &&
        !qw_hb_time_on() && getenv("DS4_QWEN4EXP_TIME_SLICES") == NULL &&
        ds4_gpu_decode_graphs_supported();
    if (qsa_capture_ok) {
        ds4_decode_graph_key key;
        memset(&key, 0, sizeof(key));
        key.il = il;
        key.island = 2u;
        key.variant = n_tokens | (s->spec_snapshot_rows << 8u);
        key._pad = 0x5157454eu;
        key.cur_hc = s->hyper;
        key.after_attn_hc = s;
        key.after_ffn_hc = (void *)l;
        key.attn_norm = (void *)m;
        const int state = ds4_gpu_decode_graph_begin(&key);
        if (state == 1) {
            QW_TRACE(QW_SLICE_QSA);
            return true;
        }
        if (state == 0) {
#ifdef DS4_TEST_HOOKS
            const uint32_t trace_len = g_qwen4exp_trace_len;
            const bool trace_overflow = g_qwen4exp_trace_overflow;
#endif
            if (qwen4exp_graph_qsa_block(s, l, m, il, n_tokens)) {
                if (ds4_gpu_decode_graph_end(&key) == 0) return true;
            } else {
                ds4_gpu_decode_graph_abort(&key);
            }
#ifdef DS4_TEST_HOOKS
            g_qwen4exp_trace_len = trace_len;
            g_qwen4exp_trace_overflow = trace_overflow;
#endif
        }
    }
    return qwen4exp_graph_qsa_block(s, l, m, il, n_tokens);
}

/* ======================================================================
 * The forward.
 * ====================================================================== */

/* Close the open command batch and report the failure.  Every refusal after
 * the batch opens goes through here. */
#ifdef DS4_TEST_HOOKS
/* PER-LAYER RESIDUAL DUMP, to sit beside llama.cpp's eval-callback.
 *
 * The forward runs the whole tower inside ONE command batch, so a mid-tower
 * read has to close the batch, synchronise and reopen it.  That is far too
 * expensive for a real run and is exactly what a diagnostic wants: the sum of
 * |hyper| over the rows in flight, after every layer, so the first layer whose
 * value departs from llama's names the culprit rather than the last one. */
static int      g_qwen4exp_layer_dump = -1;
static double   g_qwen4exp_layer_sum[DS4_MAX_LAYER];
static uint32_t g_qwen4exp_layer_dump_rows = 0;

static int qwen4exp_layer_dump_on(void) {
    if (g_qwen4exp_layer_dump < 0) {
        const char *e = getenv("DS4_QWEN4EXP_LAYER_DUMP");
        g_qwen4exp_layer_dump = (e && e[0] && e[0] != '0') ? 1 : 0;
    }
    return g_qwen4exp_layer_dump;
}

static void qwen4exp_layer_dump(ds4_qwen4exp_session *s, uint32_t il,
                                uint32_t n_tokens) {
    if (!qwen4exp_layer_dump_on() || il >= DS4_MAX_LAYER) return;
    /* LATCH THE FIRST forward, which is the PREFILL.  The decode steps that
     * follow would otherwise overwrite it, and the question is whether the
     * prompt's own rows already disagree -- a step-0 near-tie says the
     * perturbation is in the prefill, before any token was generated. */
    if (g_qwen4exp_layer_dump_rows != 0 &&
        n_tokens != g_qwen4exp_layer_dump_rows) {
        return;
    }
    if (!ds4_gpu_end_commands() || !ds4_gpu_synchronize()) return;
    const uint64_t hc = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t n = (uint64_t)n_tokens * hc;
    float *buf = malloc((size_t)n * sizeof(float));
    if (buf && ds4_gpu_tensor_read(s->hyper, 0, buf,
                                   (uint64_t)n * sizeof(float))) {
        double acc = 0.0;
        for (uint64_t i = 0; i < n; i++) acc += fabs((double)buf[i]);
        g_qwen4exp_layer_sum[il] = acc;
        g_qwen4exp_layer_dump_rows = n_tokens;
    }
    free(buf);
    (void)ds4_gpu_begin_commands();
}

int ds4_qwen4exp_test_layer_sums(double *out, int cap, int *rows_out) {
    if (!qwen4exp_layer_dump_on()) return 0;
    const ds4_qwen4exp_config *cfg = &g_ds4_qwen4exp;
    int n = (int)cfg->n_layer;
    if (n > cap) n = cap;
    for (int i = 0; i < n; i++) out[i] = g_qwen4exp_layer_sum[i];
    if (rows_out) *rows_out = (int)g_qwen4exp_layer_dump_rows;
    return n;
}
#endif /* DS4_TEST_HOOKS */

static bool qwen4exp_forward_fail(void) {
    (void)ds4_gpu_end_commands();
    return false;
}

bool ds4_qwen4exp_graph_forward(ds4_qwen4exp_session       *s,
                                const ds4_qwen4exp_weights *w,
                                const ds4_model            *m,
                                const int32_t              *tokens,
                                uint32_t                    n_tokens,
                                float                      *logits_out) {
    const ds4_qwen4exp_config *cfg = &g_ds4_qwen4exp;
    const double wall_t0 =
        getenv("DS4_QWEN4EXP_TIME_SLICES") != NULL ? now_sec() : 0.0;

    /* NO SILENT FALSE.  Every refusal below this line names itself, and so
     * does this one: an argument that is missing is a wiring fault in the
     * caller, and the caller has no other way to find out which. */
    if (!s || !w || !m || !tokens ||
        (!logits_out && s->spec_top1_rows == 0u) || n_tokens == 0) {
        fprintf(stderr,
                "ds4: qwen4exp forward called with %s%s%s%s%s%s\n",
                s ? "" : "no session ", w ? "" : "no weights ",
                m ? "" : "no model ", tokens ? "" : "no token buffer ",
                logits_out ? "" : "no logit buffer ",
                n_tokens ? "" : "no rows");
        return false;
    }
    if (n_tokens > s->plan.n_batch) {
        fprintf(stderr,
                "ds4: qwen4exp forward of %u tokens exceeds the session batch %u\n",
                n_tokens, s->plan.n_batch);
        return false;
    }
    if ((uint64_t)s->pos + n_tokens > s->plan.n_ctx) {
        fprintf(stderr,
                "ds4: qwen4exp forward would pass the session context %u\n",
                s->plan.n_ctx);
        return false;
    }

    /* From here on this call writes the caches, the recurrent states, the
     * convolution histories and `pos`; see `state_dirty`. */
    s->state_dirty = true;

#ifdef DS4_TEST_HOOKS
    g_qwen4exp_trace_len = 0;
    g_qwen4exp_trace_overflow = false;
#endif

    /* REFUSE AN ID THE EMBEDDING TABLE DOES NOT HOLD, here, where the ids are
     * still on the host and the check is free.
     *
     * glm_embed_tokens_q8_0_kernel indexes the table by the id and takes no
     * vocabulary bound -- an out-of-range id is an illegal access on CUDA and
     * silently wrong rows on Metal, neither of which names the id that caused
     * it.  A clamp here would be worse than the fault: it would answer with
     * the wrong row and no message. */
    for (uint32_t i = 0; i < n_tokens; i++) {
        if (tokens[i] >= 0 && (uint32_t)tokens[i] < cfg->n_vocab) continue;
        fprintf(stderr,
                "ds4: qwen4exp token %u of this forward is id %d, which is "
                "outside the %u-token vocabulary\n",
                i, tokens[i], cfg->n_vocab);
        return false;
    }

    if (!ds4_gpu_tensor_write(s->tokens, 0, tokens,
                              (uint64_t)n_tokens * sizeof(int32_t))) {
        fprintf(stderr, "ds4: qwen4exp token upload failed\n");
        return false;
    }

    /* The PLE host half runs BEFORE the batch opens: the row upload is a
     * memcpy into a shared buffer, and a memcpy between an encode and a commit
     * would land ahead of work already encoded.  Nothing here touches the
     * device beyond that one write. */
    {
        const ds4_qwen4exp_layer_weights *ple = &w->layer[cfg->ple_layer];
        if (ple->has_ple &&
            !qwen4exp_graph_ple_gather(s, w, tokens, n_tokens, cfg->ple_layer)) {
            fprintf(stderr,
                    "ds4: qwen4exp n-gram row gather failed for block %u\n",
                    cfg->ple_layer);
            return false;
        }
    }

    /* One command batch for the whole forward.  Every op below joins it rather
     * than taking its own command buffer, and the blits that append to the KV
     * caches only work inside a batch at all.  qwen4exp_forward_fail() ends the
     * batch on the way out so a refusal does not leave one open. */
    if (s->d_pos) {
        ds4_gpu_qwen4exp_update_dpos(s->d_pos, s->pos);
    }
    if (!ds4_gpu_begin_commands()) {
        fprintf(stderr, "ds4: qwen4exp could not open a command batch\n");
        return false;
    }

    /* Layer-0 seed: gather the embedding row and tile it into every stream. */
    if (!ds4_gpu_qwen4exp_embed_tokens_hc_tensor(
            s->hyper, s->embed_rows, s->tokens,
            QW_MAP(m, w->token_embd), QW_MAPSZ(m, w->token_embd),
            QW_OFF(w->token_embd), w->token_embd->type, cfg->n_vocab,
            n_tokens, DS4_N_EMBD, DS4_N_HC)) {
        fprintf(stderr, "ds4: qwen4exp embedding gather failed\n");
        return qwen4exp_forward_fail();
    }
    QW_TRACE(QW_SLICE_EMBED);

    for (uint32_t il = 0; il < cfg->n_layer; il++) {
        const ds4_qwen4exp_layer_weights *l = &w->layer[il];

        /* --- per-layer n-gram embedding, one block only ----------------
         * FIRST in the layer, ahead of the attention mixer, and a plain
         * residual add onto the whole hyper stream rather than an inject:
         * MLX Qwen4Exp.swift:113-121 is
         *     stream = stream + ple(stream, ids:, previousContext:, cache:)
         * before mixWithInject.  Running it after the FFN inject instead
         * would feed this block's n-gram contribution to the NEXT block. */
        if (l->has_ple) {
            if (!qwen4exp_graph_ple_block(s, m, l, n_tokens, il)) {
                return qwen4exp_forward_fail();
            }
            QW_TRACE(QW_SLICE_PLE);
        }

        if (l->is_full_attention) {
            if (!qwen4exp_graph_layer_island(s, l, m, il, n_tokens, 0u)) {
                return qwen4exp_forward_fail();
            }
            if (!qwen4exp_graph_qsa_island(s, l, m, il, n_tokens)) {
                return qwen4exp_forward_fail();
            }
            QW_TRACE(QW_SLICE_QSA);
            if (!qwen4exp_graph_layer_island(s, l, m, il, n_tokens, 1u)) {
                return qwen4exp_forward_fail();
            }
        } else if (!qwen4exp_graph_gdn_full_layer(s, l, m, il, n_tokens)) {
            return qwen4exp_forward_fail();
        }
#ifdef DS4_TEST_HOOKS
        qwen4exp_layer_dump(s, il, n_tokens);
#endif
    }

    /* The final mixer has no inject head: it collapses n_hc * n_embd to
     * n_embd and stands in for the model.norm this checkpoint does not have. */
    if (!qwen4exp_graph_residual(s, m, w->output_hc_norm, w->output_hc_down,
                                 w->output_hc_up, NULL,
                                 w->output_hc_norm_offset, n_tokens)) {
        fprintf(stderr, "ds4: qwen4exp final mixer failed\n");
        return qwen4exp_forward_fail();
    }
    QW_TRACE(QW_SLICE_FINAL_MIX);

    /*
     * The LM head, over one row or over all of them.
     *
     * A prefill reads only the last row, and n_vocab is 248320, so a row it
     * will not read is wasted work.  A SPECULATIVE VERIFY reads every row: the
     * accept loop compares each drafted position against the target's own
     * argmax there, and running the head per row afterwards re-reads the
     * 676 MiB output weight once per comparison.  Over the chain the head is
     * the same arithmetic either way -- the routed entry gives every row the
     * one-row reduction order at any width -- so the wide call is the same
     * numbers with the weight read once.
     */
    const uint32_t logit_rows = s->spec_logit_rows ? n_tokens : 1u;
    if (logit_rows == 1u) {
        const uint64_t last = (uint64_t)(n_tokens - 1u) * DS4_N_EMBD * sizeof(float);
        if (!ds4_gpu_tensor_copy(s->embed_rows, 0, s->mixed, last,
                                       (uint64_t)DS4_N_EMBD * sizeof(float))) {
            fprintf(stderr, "ds4: qwen4exp head row copy failed\n");
            return qwen4exp_forward_fail();
        }
    }
    if (!ds4_qwen4exp_matmul_q8_0(s->logits, QW_MAP(m, w->output),
                                    QW_MAPSZ(m, w->output), QW_OFF(w->output),
                                    DS4_N_EMBD, cfg->n_vocab,
                                    logit_rows == 1u ? s->embed_rows : s->mixed,
                                    logit_rows)) {
        fprintf(stderr, "ds4: qwen4exp LM head failed\n");
        return qwen4exp_forward_fail();
    }
    if (s->spec_top1_rows != 0u &&
        (s->spec_top1_rows != logit_rows ||
         !ds4_gpu_indexer_topk_tensor(s->spec_top1, s->logits,
                                      cfg->n_vocab, logit_rows, 1u))) {
        fprintf(stderr, "ds4: qwen4exp compact verify top-1 failed\n");
        return qwen4exp_forward_fail();
    }
    QW_TRACE(QW_SLICE_HEAD);
    QW_TRACE_REPORT(n_tokens);
    /* Commit and wait: the logits are read on the host from here on. */
    if (!ds4_gpu_end_commands() || !ds4_gpu_synchronize()) {
        fprintf(stderr, "ds4: qwen4exp command batch did not complete\n");
        return false;
    }
    if (logits_out &&
        !ds4_gpu_tensor_read(s->logits, 0, logits_out,
                             (uint64_t)logit_rows * cfg->n_vocab *
                                     sizeof(float))) {
        fprintf(stderr, "ds4: qwen4exp logit read failed\n");
        return false;
    }

    s->pos += n_tokens;
    if (wall_t0 != 0.0) {
        fprintf(stderr, "ds4: qwen4exp forward wall: %u rows in %.3f ms\n",
                n_tokens, (now_sec() - wall_t0) * 1000.0);
    }
    return true;
}


/*
 * Adopt the per-GDN-layer state as it stood after verify row `row`.
 *
 * `recurrent` and `conv` select which object, because they are two rollback
 * ids and the cycle may ask for one without the other.  The slot the forward
 * wrote is copied over the live buffer; that copy is the whole rollback, and
 * it is exact because the recurrence is token-serial, so the state the forward
 * left after row k is the state a (k + 1)-row feed would have left.
 */
static bool qwen4exp_session_select_layers(ds4_qwen4exp_session *s,
                                           uint32_t row,
                                           bool recurrent, bool conv) {
    const ds4_qwen4exp_config *cfg = &g_ds4_qwen4exp;
    if (s) s->state_dirty = true;   /* see `state_dirty` */
    if (row >= DS4_QWEN4EXP_IMPLEMENTED_DEPTH) return false;
    if (!ds4_gpu_begin_commands()) return false;
    bool ok = true;
    for (uint32_t il = 0; il < cfg->n_layer && ok; il++) {
        if (recurrent && s->gdn_state[il]) {
            const uint64_t bytes = ds4_gpu_tensor_bytes(s->gdn_state[il]);
            ok = ds4_gpu_tensor_copy(s->gdn_state[il], 0,
                                     s->gdn_state_snapshot[il],
                                     (uint64_t)row * bytes, bytes) != 0;
        }
        if (ok && conv && s->gdn_conv[il]) {
            const uint64_t bytes = ds4_gpu_tensor_bytes(s->gdn_conv[il]);
            ok = ds4_gpu_tensor_copy(s->gdn_conv[il], 0,
                                     s->gdn_conv_snapshot[il],
                                     (uint64_t)row * bytes, bytes) != 0;
        }
    }
    if (!ds4_gpu_end_commands() || !ds4_gpu_synchronize()) return false;
    return ok;
}

/* ======================================================================
 * The MTP head's 49th block.
 *
 * It is an ordinary qwen4exp FULL-ATTENTION block -- the QSA sequence, the MoE
 * trio and the two hyper-connection mixers, in the order the tower runs them.
 * So it is this file's code, called on the head's weights and the head's cache
 * rather than the tower's, and it adds no kernel of its own.
 *
 * The head's block is layer index n_layer (48): one past the tower's last, in
 * the SAME session, so its key/value caches, its indexer tape and its pooled
 * blocks are the session's per-layer slots at that index and nothing else in
 * the session can reach them.  That is what "the head's own cache stack" means
 * here.
 *
 * It is ALWAYS full attention.  The head file's attention.compress_ratios says
 * 0 at that index and is wrong; the binder already ignores it, and this
 * refuses rather than silently running a gated-delta-net block that has no
 * weights.
 * ====================================================================== */

/* Give the session the cache slot and the uploaded per-head norms for the
 * head's block.  Separate from session open because only a session that
 * actually loaded a head needs them. */
bool ds4_qwen4exp_session_add_head_block(ds4_qwen4exp_session             *s,
                                         const ds4_qwen4exp_layer_weights *l,
                                         const ds4_model                  *m,
                                         uint32_t                          il) {
    const ds4_qwen4exp_config *cfg = &g_ds4_qwen4exp;
    const uint64_t f = sizeof(float);
    if (!s || !l || !m) return false;
    if (il >= DS4_MAX_LAYER) {
        fprintf(stderr,
                "ds4: qwen4exp head block index %u is past DS4_MAX_LAYER\n", il);
        return false;
    }
    if (il < cfg->n_layer) {
        fprintf(stderr,
                "ds4: qwen4exp head block index %u collides with the tower's "
                "layers (0..%u)\n", il, cfg->n_layer - 1u);
        return false;
    }
    if (!l->is_full_attention) {
        fprintf(stderr,
                "ds4: qwen4exp head block %u is not bound as full attention\n", il);
        return false;
    }
    if (s->qsa_k[il]) return true;   /* already set up */

    /* Conservative: the slot's own buffers come out of qwen4exp_alloc_zeroed
     * and so match a reset, but this also uploads the head's three norms and
     * registers its mapping, so let the next reset run in full rather than
     * reason about which of those a reset covers. */
    s->state_dirty = true;

    /* Register the HEAD's mapping with the backend.  The session registered the
     * target's shards at open; the head is a second GGUF and every weight the
     * block reads lives in it, so without this the first op that wraps a model
     * range -- the attention mixer's norm -- fails with nothing said.
     *
     * RESIDENT, not mapped.  The target's weights are uploaded to the device
     * by the loader's tensor-span pass; the head's went in as an AUXILIARY
     * range, which on an integrated device with pageable access is the host
     * mmap itself, borrowed.  That is the right answer for a range too large
     * to hold and the wrong one here: the head is 2.6 GiB, and its ten routed
     * experts are a different ten every step, so the read is scattered over
     * the whole file rather than over the handful of tensors the attention
     * projections reuse.  The head's routed experts measured 7.283 ms per step
     * against the tower's 0.249 ms for the same call at the same width -- a
     * twenty-nine-fold gap on comparable bytes, which is where the weights
     * live and not what the kernel does.
     *
     * So the head asks for the cache FIRST, the way the target's spans do, and
     * keeps the mapping as the fallback.  The bytes are the same bytes: this
     * is a copy, not a repack, so nothing the block computes moves.
     *
     * The cache call is idempotent -- a second session finds the range already
     * held -- so this runs per session and uploads once. */
    for (uint32_t i = 0; i < m->n_shards; i++) {
        const ds4_model_shard *sh = &m->shard[i];
        if (!sh->map || sh->tensor_data_pos >= sh->size) continue;
        const uint64_t off = sh->tensor_data_pos;
        const uint64_t len = sh->size - off;
        uint64_t shard_max_tensor = 0;
        for (uint64_t ti = 0; ti < m->n_tensors; ti++) {
            const ds4_tensor *t = &m->tensors[ti];
            if (t->map_index == i && t->bytes > shard_max_tensor) {
                shard_max_tensor = t->bytes;
            }
        }
        int rc;
#if defined(__APPLE__)
        rc = ds4_gpu_set_model_map_range(sh->map, sh->size, off, len,
                                         shard_max_tensor);
#else
        /* THE LOADER MAY ALREADY HOLD IT.  When the head was armed at open,
         * the loader's support-model pass made every head tensor device
         * resident, tensor by tensor.  Caching the whole shard span here
         * would not find one range covering it and would upload the 2.6 GiB
         * a SECOND time through the fd route -- measured as a 401 ms first
         * speculative round and a 148 ms invalidate that landed inside the
         * ranked flow's timed prefill window (2026-09-04).  So ask first:
         * if every head tensor on this shard already has a device range,
         * the head is resident and there is nothing to copy. */
        int already = 0;
        {
            uint64_t seen = 0, cached = 0;
            for (uint64_t ti = 0; ti < m->n_tensors; ti++) {
                const ds4_tensor *t = &m->tensors[ti];
                if (t->map_index != i || t->bytes == 0) continue;
                seen++;
                if (ds4_gpu_model_range_device_ptr(sh->map, t->abs_offset,
                                                   t->bytes)) cached++;
            }
            already = seen > 0 && cached == seen;
        }
        char label[64];
        snprintf(label, sizeof(label), "qwen4exp-mtp-head:shard%u", i);
        rc = already ? 1
                     : ds4_gpu_cache_model_range(sh->map, sh->size, off, len,
                                                 label);
        if (rc) {
            if (g_qwen4exp_head_resident != sh->map) {
                g_qwen4exp_head_resident = sh->map;
                fprintf(stderr,
                        "ds4: qwen4exp MTP head weights resident: %.2f GiB\n",
                        (double)len / 1073741824.0);
            }
        } else {
            fprintf(stderr,
                    "ds4: qwen4exp MTP head shard %u (%.2f GiB) would not fit "
                    "on the device; reading it through the host mapping\n",
                    i, (double)len / 1073741824.0);
            rc = ds4_gpu_set_aux_model_map_range(sh->map, sh->size, off, len);
        }
#endif
        if (!rc) {
            fprintf(stderr,
                    "ds4: qwen4exp could not register the MTP head's shard %u "
                    "with the backend\n", i);
            return false;
        }
    }

    const uint32_t n_ctx = s->plan.n_ctx;
    const uint64_t blocks = (uint64_t)n_ctx / DS4_N_INDEXER_COMPRESS_RATIO;
    bool ok = true;
    s->qsa_k[il] = qwen4exp_alloc_zeroed((uint64_t)n_ctx * cfg->qsa_kv_dim * f, &ok);
    s->qsa_v[il] = qwen4exp_alloc_zeroed((uint64_t)n_ctx * cfg->qsa_kv_dim * f, &ok);
    s->idx_tape[il] = qwen4exp_alloc_zeroed(
        (uint64_t)n_ctx * DS4_N_INDEXER_KV_HEAD * DS4_N_INDEXER_HEAD_DIM * f, &ok);
    s->idx_pool[il] = qwen4exp_alloc_zeroed(
        blocks * DS4_N_INDEXER_KV_HEAD * DS4_N_INDEXER_HEAD_DIM * f, &ok);
    s->idx_k_norm[il] = qwen4exp_alloc_zeroed((uint64_t)DS4_N_INDEXER_HEAD_DIM * f, &ok);
    s->attn_q_norm[il] = qwen4exp_alloc_zeroed((uint64_t)DS4_N_HEAD_DIM * f, &ok);
    s->attn_k_norm[il] = qwen4exp_alloc_zeroed((uint64_t)DS4_N_HEAD_DIM * f, &ok);
    if (ok) {
        /* From the HEAD's mapping, not the target's: these three come out of
         * the --mtp GGUF like the rest of blk.<il>.*. */
        ok = l->indexer_k_norm && l->attn_q_norm && l->attn_k_norm &&
             ds4_gpu_tensor_write(s->idx_k_norm[il], 0,
                                  tensor_data(m, l->indexer_k_norm),
                                  (uint64_t)DS4_N_INDEXER_HEAD_DIM * f) &&
             ds4_gpu_tensor_write(s->attn_q_norm[il], 0,
                                  tensor_data(m, l->attn_q_norm),
                                  (uint64_t)DS4_N_HEAD_DIM * f) &&
             ds4_gpu_tensor_write(s->attn_k_norm[il], 0,
                                  tensor_data(m, l->attn_k_norm),
                                  (uint64_t)DS4_N_HEAD_DIM * f);
    }
    if (!ok) {
        fprintf(stderr, "ds4: qwen4exp head block %u cache setup failed\n", il);
        return false;
    }
    return true;
}

/* Encode the MTP block once, either eagerly or into the decode graph stream.
 * All arguments that become kernel parameters are already stable here. */
static bool qwen4exp_graph_head_block_encode(
        ds4_qwen4exp_session *s,
        const ds4_qwen4exp_layer_weights *l,
        const ds4_model *m, uint32_t il, uint32_t n_tokens,
        const char **failed_stage) {
    const int timing = qw_hb_time_on();
    uint64_t tmark = timing ? qw_hb_now_ns() : 0;
    const char *stage = "attention mixer";
    bool ok = qwen4exp_graph_residual(s, m, l->hc_attn_norm, l->hc_attn_down,
                                      l->hc_attn_up, l->hc_attn_inject,
                                      l->hc_attn_norm_offset, n_tokens);
    QW_HB_TICK(QW_HB_T_ATTN_MIX);
    if (ok) { stage = "QSA"; ok = qwen4exp_graph_qsa_block(s, l, m, il, n_tokens); }
    QW_HB_TICK(QW_HB_T_QSA);
    if (ok) {
        stage = "attention inject";
        ok = ds4_gpu_qwen4exp_hc_inject_tensor(s->hyper, s->hyper, s->block_out,
                                               s->inject, DS4_N_EMBD, DS4_N_HC,
                                               n_tokens) != 0;
    }
    QW_HB_TICK(QW_HB_T_ATTN_INJECT);
    if (ok) {
        stage = "FFN mixer";
        ok = qwen4exp_graph_residual(s, m, l->hc_ffn_norm, l->hc_ffn_down,
                                     l->hc_ffn_up, l->hc_ffn_inject,
                                     l->hc_ffn_norm_offset, n_tokens);
    }
    QW_HB_TICK(QW_HB_T_FFN_MIX);
    if (ok) {
        stage = "MoE";
        qw_moe_in_head_block = 1;
        ok = qwen4exp_graph_moe_block(s, l, m, il, n_tokens);
        qw_moe_in_head_block = 0;
    }
    QW_HB_TICK(QW_HB_T_MOE);
    if (ok) {
        stage = "FFN inject";
        ok = ds4_gpu_qwen4exp_hc_inject_tensor(s->hyper, s->hyper, s->block_out,
                                               s->inject, DS4_N_EMBD, DS4_N_HC,
                                               n_tokens) != 0;
    }
    QW_HB_TICK(QW_HB_T_FFN_INJECT);
    if (timing && ok) qw_hb_stage_calls++;

    if (failed_stage) *failed_stage = stage;
    return ok;
}

/*
 * ds4_qwen4exp_block_forward_fn.  `graph` is the head's weight context and
 * `cache` is the session that owns the slot; `hyper` is the head's own stream,
 * rewritten in place.  Returns NONZERO on success -- the ds4_gpu convention,
 * not ds4.c's.
 *
 * Device-positioned QSA makes the complete head block position-independent.
 * Capturing it as island 3 removes the attention, mixer and MoE launch train
 * from every MTP draft round.  The session's hyper pointer and position are
 * lent to the head for the call and restored afterwards.
 */
int ds4_qwen4exp_graph_head_block(void *graph, void *cache,
                                  ds4_gpu_tensor *hyper, uint32_t il,
                                  uint32_t pos0, uint32_t n_tokens) {
    const ds4_qwen4exp_head_block_ctx *c = graph;
    ds4_qwen4exp_session *s = cache;
    if (!c || !s || !c->l || !c->m || !hyper || n_tokens == 0) return 0;
    if (n_tokens > s->plan.n_batch) return 0;
    if (il >= DS4_MAX_LAYER || !s->qsa_k[il]) {
        fprintf(stderr,
                "ds4: qwen4exp head block %u has no cache slot; "
                "ds4_qwen4exp_session_add_head_block() was not called\n", il);
        return 0;
    }
    if (!c->l->is_full_attention) return 0;

    /* The head's QSA block appends to the slot's key/value caches and its
     * indexer tape; see `state_dirty`. */
    s->state_dirty = true;

    ds4_gpu_tensor *saved_hyper = s->hyper;
    const uint32_t saved_pos = s->pos;
    s->hyper = hyper;
    s->pos = pos0;
    if (s->d_pos) ds4_gpu_qwen4exp_update_dpos(s->d_pos, s->pos);

    const bool capture_ok =
        n_tokens <= (uint32_t)DS4_QWEN4EXP_MTP_MAX_COMMIT &&
        (uint64_t)s->pos + n_tokens <= DS4_N_INDEXER_TOP_K &&
        !qw_hb_time_on() && getenv("DS4_QWEN4EXP_TIME_SLICES") == NULL &&
        ds4_gpu_decode_graphs_supported();
    ds4_decode_graph_key key;
    int graph_state = -1;
    if (capture_ok) {
        memset(&key, 0, sizeof(key));
        key.il = il;
        key.island = 3u;
        key.variant = n_tokens | (s->spec_snapshot_rows << 8u);
        key._pad = 0x4d545048u;
        key.cur_hc = hyper;
        key.after_attn_hc = s;
        key.after_ffn_hc = (void *)c->l;
        key.attn_norm = (void *)c->m;
        graph_state = ds4_gpu_decode_graph_begin(&key);
    }

    const char *stage = "attention mixer";
    bool ok = graph_state == 1;
    if (graph_state != 1) {
        ok = qwen4exp_graph_head_block_encode(
                s, c->l, c->m, il, n_tokens, &stage);
        if (graph_state == 0) {
            if (ok && ds4_gpu_decode_graph_end(&key) == 0) {
                ok = true;
            } else {
                if (!ok) ds4_gpu_decode_graph_abort(&key);
                /* A failed capture executed no kernels.  Re-encode the block
                 * eagerly once so this proposal still advances its caches. */
                ok = qwen4exp_graph_head_block_encode(
                        s, c->l, c->m, il, n_tokens, &stage);
            }
        }
    }

    s->hyper = saved_hyper;
    s->pos = saved_pos;
    if (s->d_pos) ds4_gpu_qwen4exp_update_dpos(s->d_pos, s->pos);
    if (!ok) {
        fprintf(stderr, "ds4: qwen4exp head block %u failed at the %s\n",
                il, stage);
    }
    return ok ? 1 : 0;
}

/* ======================================================================
 * The MTP model seam.
 *
 * ds4_qwen4exp_mtp_cycle() drives the target through four callbacks.  Three of
 * them are this graph; the fourth is the head.  They live here because they
 * need the session's buffers, and because verify_rows and decode_token MUST be
 * the same code path -- the cycle's batch-invariance requirement is that row t
 * of an n-row verify equals a one-row decode from the same state, bit for bit,
 * and the cheapest way to guarantee that is to have one implementation.
 * ====================================================================== */

/* Run `n_tokens` rows and hand back both the pre-final-mixer stream and
 * `logit_rows` rows of logits -- the last row's, or every row's.  This IS
 * ds4_qwen4exp_graph_forward plus a read of the rows the head consumes; there
 * is no second forward path. */
bool ds4_qwen4exp_graph_verify_rows(ds4_qwen4exp_session       *s,
                                    const ds4_qwen4exp_weights *w,
                                    const ds4_model            *m,
                                    const int32_t              *tokens,
                                    uint32_t                    n_tokens,
                                    float                      *hc_rows,
                                    float                      *logits,
                                    uint32_t                    logit_rows) {
    /* One row or all of them, and nothing between: the caller sized `logits`
     * from this number, so a third value would be a buffer overrun rather than
     * a narrower read. */
    if (logit_rows != 1u && logit_rows != n_tokens) {
        fprintf(stderr,
                "ds4: qwen4exp verify of %u rows asked for %u rows of logits; "
                "it leaves the last row's or every row's\n",
                n_tokens, logit_rows);
        return false;
    }
    if (logit_rows > (uint32_t)DS4_QWEN4EXP_MTP_MAX_COMMIT) {
        fprintf(stderr,
                "ds4: qwen4exp verify asked for %u rows of logits; the logit "
                "buffer holds %d\n", logit_rows,
                DS4_QWEN4EXP_MTP_MAX_COMMIT);
        return false;
    }
    /*
     * Arm the per-row state slots for the drafted rows: the last row's state
     * is the live state, so a verify of n rows needs n - 1 of them.  Armed
     * here and nowhere else, so a prefill chunk of 512 never asks for 512
     * slots.
     *
     * Only a call at the CYCLE's width arms them.  This entry point is also
     * the batch-invariance sweep's, which walks widths well past the chain,
     * and a diagnostic sweep never rolls back -- so a wider call runs with no
     * slots rather than being refused.  The cycle cannot reach that branch:
     * its token buffer is DS4_QWEN4EXP_MTP_MAX_COMMIT wide and a _Static_assert
     * ties that to the depth envelope, and if it ever did, select_row would
     * refuse by name on an unwritten slot rather than roll back to a stale
     * one.
     */
    s->spec_snapshot_rows =
        (n_tokens > 1u && n_tokens <= (uint32_t)DS4_QWEN4EXP_MTP_MAX_COMMIT)
            ? n_tokens - 1u : 0u;
    s->spec_logit_rows = logit_rows > 1u ? logit_rows : 0u;
    const bool forward_ok =
        ds4_qwen4exp_graph_forward(s, w, m, tokens, n_tokens, logits);
    s->spec_snapshot_rows = 0u;
    s->spec_logit_rows = 0u;
    if (!forward_ok) return false;
    if (hc_rows && !ds4_qwen4exp_session_read_hyper(s, 0, n_tokens, hc_rows)) {
        fprintf(stderr,
                "ds4: qwen4exp could not read %u pre-final-mixer rows back "
                "(session batch %u)\n", n_tokens, s->plan.n_batch);
        return false;
    }
    return true;
}

bool ds4_qwen4exp_graph_verify_top1_rows(ds4_qwen4exp_session       *s,
                                         const ds4_qwen4exp_weights *w,
                                         const ds4_model            *m,
                                         const int32_t              *tokens,
                                         uint32_t                    n_tokens,
                                         float                      *hc_rows,
                                         int                        *row_top1) {
    if (!s || !row_top1 || n_tokens == 0u ||
        n_tokens > (uint32_t)DS4_QWEN4EXP_MTP_MAX_COMMIT) {
        return false;
    }
    s->spec_snapshot_rows = n_tokens > 1u ? n_tokens - 1u : 0u;
    s->spec_logit_rows = n_tokens > 1u ? n_tokens : 0u;
    s->spec_top1_rows = n_tokens;
    const bool forward_ok =
        ds4_qwen4exp_graph_forward(s, w, m, tokens, n_tokens, NULL);
    s->spec_snapshot_rows = 0u;
    s->spec_logit_rows = 0u;
    s->spec_top1_rows = 0u;
    if (!forward_ok) return false;
    if (hc_rows && !ds4_qwen4exp_session_read_hyper(s, 0, n_tokens, hc_rows)) {
        fprintf(stderr, "ds4: qwen4exp compact verify hyper read failed\n");
        return false;
    }
    return ds4_gpu_tensor_read(s->spec_top1, 0, row_top1,
                               (uint64_t)n_tokens * sizeof(int32_t)) != 0;
}

bool ds4_qwen4exp_graph_read_logit_row(ds4_qwen4exp_session *s,
                                       uint32_t row,
                                       float *logits) {
    const ds4_qwen4exp_config *cfg = &g_ds4_qwen4exp;
    if (!s || !logits || row >= (uint32_t)DS4_QWEN4EXP_MTP_MAX_COMMIT) {
        return false;
    }
    const uint64_t bytes = (uint64_t)cfg->n_vocab * sizeof(float);
    return ds4_gpu_tensor_read(s->logits, (uint64_t)row * bytes,
                               logits, bytes) != 0;
}

/* The target's LM head over ONE supplied pre-final-mixer row.
 *
 * The verify leaves every row's logits now, so the accept loop no longer asks
 * for a row separately.  This stays as the ONE-ROW half of that head's batch
 * invariance: the row is uploaded into the session's own hyper buffer, which
 * the caller has already read out, and the final mixer and head run over that
 * one row -- the same two ops the forward ends with, so a row that came from a
 * verify produces exactly the logits that verify produced for it, and the
 * diagnostics compare the two. */
bool ds4_qwen4exp_graph_head_logits(ds4_qwen4exp_session       *s,
                                    const ds4_qwen4exp_weights *w,
                                    const ds4_model            *m,
                                    const float                *hc_row,
                                    float                      *logits) {
    const ds4_qwen4exp_config *cfg = &g_ds4_qwen4exp;
    if (!s || !w || !m || !hc_row || !logits) {
        fprintf(stderr,
                "ds4: qwen4exp head-row logits called with %s%s%s%s%s\n",
                s ? "" : "no session ", w ? "" : "no weights ",
                m ? "" : "no model ", hc_row ? "" : "no input row ",
                logits ? "" : "no logit buffer");
        return false;
    }

    const uint64_t row_bytes =
        (uint64_t)DS4_N_HC * DS4_N_EMBD * sizeof(float);
    if (!ds4_gpu_tensor_write(s->hyper, 0, hc_row, row_bytes)) {
        fprintf(stderr, "ds4: qwen4exp head row upload failed\n");
        return false;
    }
    if (!ds4_gpu_begin_commands()) {
        fprintf(stderr, "ds4: qwen4exp could not open a command batch\n");
        return false;
    }
    if (!qwen4exp_graph_residual(s, m, w->output_hc_norm, w->output_hc_down,
                                 w->output_hc_up, NULL,
                                 w->output_hc_norm_offset, 1u)) {
        fprintf(stderr, "ds4: qwen4exp final mixer failed on the head row\n");
        return qwen4exp_forward_fail();
    }
    if (!ds4_gpu_tensor_copy(s->embed_rows, 0, s->mixed, 0,
                             (uint64_t)DS4_N_EMBD * sizeof(float)) ||
        !ds4_qwen4exp_matmul_q8_0(s->logits, QW_MAP(m, w->output),
                                    QW_MAPSZ(m, w->output), QW_OFF(w->output),
                                    DS4_N_EMBD, cfg->n_vocab, s->embed_rows, 1u)) {
        fprintf(stderr, "ds4: qwen4exp LM head failed on the head row\n");
        return qwen4exp_forward_fail();
    }
    if (!ds4_gpu_end_commands() || !ds4_gpu_synchronize()) {
        fprintf(stderr, "ds4: qwen4exp command batch did not complete\n");
        return false;
    }
    return ds4_gpu_tensor_read(s->logits, 0, logits,
                               (uint64_t)cfg->n_vocab * sizeof(float)) != 0;
}

#ifdef DS4_TEST_HOOKS
/* Which carried state a chunked prefill left behind, one checksum per object.
 *
 * A prompt prefilled in chunks must leave the same state as the same prompt
 * prefilled whole: every one of these advances per ROW, so a chunk boundary is
 * not supposed to be observable.  When it is, this says WHICH object stopped
 * agreeing rather than leaving the reader to infer it from a changed argmax.
 *
 * The checksum is order sensitive on purpose: two states that hold the same
 * values in different rows are not the same state. */
static double qwen4exp_state_checksum(const ds4_gpu_tensor *t) {
    if (!t) return 0.0;
    const uint64_t bytes = ds4_gpu_tensor_bytes(t);
    const uint64_t n = bytes / sizeof(float);
    if (n == 0) return 0.0;
    float *buf = malloc((size_t)bytes);
    if (!buf) return -1.0;
    double acc = 0.0;
    if (ds4_gpu_tensor_read((ds4_gpu_tensor *)t, 0, buf, bytes)) {
        for (uint64_t i = 0; i < n; i++) {
            const double v = (double)buf[i];
            acc += v * (double)((i % 1021u) + 1u);
        }
    } else {
        acc = -2.0;
    }
    free(buf);
    return acc;
}

/*
 * The same carried state, ONE OBJECT PER ENTRY rather than one sum per layer.
 *
 * ds4_qwen4exp_test_state_checksums() adds a layer's six objects together, so
 * "block 2 differs" cannot say whether the convolution history, the recurrent
 * state, the KV cache or the indexer tape moved first.  A width sweep needs
 * that: the objects advance through different kernels, and which one leads
 * names the kernel.
 *
 * Order is the tower's: layer 0's objects, then layer 1's, and so on, then the
 * two global PLE objects and the KV length.  The first differing index is
 * therefore the first object in causal order that stopped agreeing.
 */
enum { DS4_QWEN4EXP_STATE_OBJECTS_PER_LAYER = 6 };

int ds4_qwen4exp_test_state_object_count(const ds4_qwen4exp_session *s) {
    if (!s) return 0;
    return (int)(g_ds4_qwen4exp.n_layer *
                 DS4_QWEN4EXP_STATE_OBJECTS_PER_LAYER) + 3;
}

int ds4_qwen4exp_test_state_object(const ds4_qwen4exp_session *s, int index,
                                   char *name, size_t name_len, double *sum) {
    if (!s || index < 0 || index >= ds4_qwen4exp_test_state_object_count(s)) {
        return 0;
    }
    static const char *const kind[DS4_QWEN4EXP_STATE_OBJECTS_PER_LAYER] = {
        "gdn_conv", "gdn_state", "qsa_k", "qsa_v", "idx_tape", "idx_pool"
    };
    const int per = DS4_QWEN4EXP_STATE_OBJECTS_PER_LAYER;
    const int layer_entries = (int)g_ds4_qwen4exp.n_layer * per;
    if (index < layer_entries) {
        const uint32_t il = (uint32_t)(index / per);
        const int k = index % per;
        const ds4_gpu_tensor *t = NULL;
        switch (k) {
        case 0: t = s->gdn_conv[il];  break;
        case 1: t = s->gdn_state[il]; break;
        case 2: t = s->qsa_k[il];     break;
        case 3: t = s->qsa_v[il];     break;
        case 4: t = s->idx_tape[il];  break;
        default: t = s->idx_pool[il]; break;
        }
        if (name && name_len) {
            snprintf(name, name_len, "blk%u.%s", il, kind[k]);
        }
        /* A slot the block type does not own is NULL and checksums to zero on
         * both sides, so it never reports a difference of its own. */
        if (sum) *sum = qwen4exp_state_checksum(t);
        return 1;
    }
    switch (index - layer_entries) {
    case 0:
        if (name && name_len) snprintf(name, name_len, "ple_conv_state");
        if (sum) *sum = qwen4exp_state_checksum(s->ple_conv_state);
        return 1;
    case 1: {
        if (name && name_len) snprintf(name, name_len, "ple_history");
        double h = 0.0;
        for (uint32_t i = 0; i < DS4_PLE_MAX_NGRAM; i++) {
            h += (double)s->ple_history.previous[i] * (double)(i + 1u);
        }
        if (sum) *sum = h;
        return 1;
    }
    default:
        /* The KV cache LENGTH, which the contents checksum cannot see: two
         * caches holding the same rows but disagreeing on how many are live
         * are not the same cache. */
        if (name && name_len) snprintf(name, name_len, "kv_length");
        if (sum) *sum = (double)s->pos;
        return 1;
    }
}

void ds4_qwen4exp_test_state_checksums(const ds4_qwen4exp_session *s,
                                       double *out, int cap) {
    /* PER LAYER, in tower order, then the two global objects.  Summing a kind
     * of state across layers -- which this did at first -- reports the ARRAY
     * order, not the causal one, so "gdn_conv differs" said nothing about
     * WHERE the divergence started.  Laid out per layer, the first index that
     * differs is the first block that differs. */
    for (int i = 0; i < cap; i++) out[i] = 0.0;
    if (!s || cap < 2) return;
    const ds4_qwen4exp_config *cfg = &g_ds4_qwen4exp;
    for (uint32_t il = 0; il < cfg->n_layer && (int)il + 2 <= cap; il++) {
        out[il] = qwen4exp_state_checksum(s->gdn_conv[il]) +
                  qwen4exp_state_checksum(s->gdn_state[il]) +
                  qwen4exp_state_checksum(s->qsa_k[il]) +
                  qwen4exp_state_checksum(s->qsa_v[il]) +
                  qwen4exp_state_checksum(s->idx_tape[il]) +
                  qwen4exp_state_checksum(s->idx_pool[il]);
    }
    if ((int)cfg->n_layer + 1 < cap) {
        out[cfg->n_layer] = qwen4exp_state_checksum(s->ple_conv_state);
        double h = 0.0;
        for (uint32_t i = 0; i < DS4_PLE_MAX_NGRAM; i++) {
            h += (double)s->ple_history.previous[i] * (double)(i + 1u);
        }
        out[cfg->n_layer + 1] = h;
    }
}
#endif /* DS4_TEST_HOOKS */

/* ----------------------------------------------------------------------
 * The rollback set.
 *
 * Every object the cycle must be able to undo, registered from the session
 * that owns it.  ds4_qwen4exp_mtp_state_init() refuses when one is missing, so
 * the registrations below are the whole contract for this graph.
 * ---------------------------------------------------------------------- */

static int qwen4exp_rb_gdn_recurrent_select(void *ctx, uint32_t row) {
    return qwen4exp_session_select_layers(ctx, row, true, false) ? 0 : -1;
}
static int qwen4exp_rb_gdn_conv_select(void *ctx, uint32_t row) {
    return qwen4exp_session_select_layers(ctx, row, false, true) ? 0 : -1;
}
static int qwen4exp_rb_ple_hist_select(void *ctx, uint32_t row) {
    ds4_qwen4exp_session *s = ctx;
    if (row >= DS4_QWEN4EXP_IMPLEMENTED_DEPTH) return -1;
    s->state_dirty = true;   /* see `state_dirty` */
    s->ple_history = s->ple_history_rows[row];
    return 0;
}
static int qwen4exp_rb_ple_conv_select(void *ctx, uint32_t row) {
    ds4_qwen4exp_session *s = ctx;
    if (row >= DS4_QWEN4EXP_IMPLEMENTED_DEPTH) return -1;
    s->state_dirty = true;   /* see `state_dirty` */
    const uint64_t bytes = ds4_gpu_tensor_bytes(s->ple_conv_state);
    if (!ds4_gpu_begin_commands()) return -1;
    const bool ok = ds4_gpu_tensor_copy(s->ple_conv_state, 0,
                                        s->ple_conv_snapshot,
                                        (uint64_t)row * bytes, bytes) != 0;
    if (!ds4_gpu_end_commands() || !ds4_gpu_synchronize()) return -1;
    return ok ? 0 : -1;
}
static int qwen4exp_rb_qsa_kv_truncate(void *ctx, uint32_t pos) {
    ds4_qwen4exp_session *s = ctx;
    /* The caches are addressed by position and only appended to, so dropping
     * rows is bookkeeping: the next write starts at `pos`. */
    if (pos > s->plan.n_ctx) return -1;
    s->state_dirty = true;   /* see `state_dirty` */
    s->pos = pos;
    return 0;
}
static int qwen4exp_rb_qsa_indexer_truncate(void *ctx, uint32_t pos) {
    ds4_qwen4exp_session *s = ctx;
    /* One step, tape and pooled blocks together: block b pools tape rows
     * [b*ratio, (b+1)*ratio), so only blocks entirely below `pos` survive.
     * Rebuilding the pool from a truncated tape is what keeps a later offset
     * bump from re-exposing rows the tape no longer covers. */
    if (pos > s->plan.n_ctx) return -1;
    s->state_dirty = true;   /* see `state_dirty` */
    s->indexer_valid_blocks = pos / DS4_N_INDEXER_COMPRESS_RATIO;
    return 0;
}
static int qwen4exp_rb_head_cache_truncate(void *ctx, uint32_t pos) {
    ds4_qwen4exp_session *s = ctx;
    /* The head keeps its own cache stack; the session only records how far it
     * is seeded so a rejecting round cannot leave it ahead of the target. */
    s->state_dirty = true;   /* see `state_dirty` */
    s->head_cache_pos = pos;
    return 0;
}

bool ds4_qwen4exp_session_rollback_set(ds4_qwen4exp_session      *s,
                                       ds4_qwen4exp_rollback_set *set,
                                       char *err, size_t errlen) {
    if (!s || !set) return false;
    ds4_qwen4exp_rollback_init(set);
    const struct {
        ds4_qwen4exp_state_id id;
        ds4_qwen4exp_rollback_object obj;
    } objs[] = {
        { DS4_QWEN4EXP_STATE_GDN_RECURRENT,
          { s, qwen4exp_rb_gdn_recurrent_select, NULL } },
        { DS4_QWEN4EXP_STATE_GDN_CONV,
          { s, qwen4exp_rb_gdn_conv_select, NULL } },
        { DS4_QWEN4EXP_STATE_PLE_HISTORY,
          { s, qwen4exp_rb_ple_hist_select, NULL } },
        { DS4_QWEN4EXP_STATE_PLE_CONV,
          { s, qwen4exp_rb_ple_conv_select, NULL } },
        { DS4_QWEN4EXP_STATE_QSA_KV,
          { s, NULL, qwen4exp_rb_qsa_kv_truncate } },
        { DS4_QWEN4EXP_STATE_QSA_INDEXER_TAPE,
          { s, NULL, qwen4exp_rb_qsa_indexer_truncate } },
        { DS4_QWEN4EXP_STATE_MTP_HEAD_CACHE,
          { s, NULL, qwen4exp_rb_head_cache_truncate } },
    };
    for (size_t i = 0; i < sizeof(objs) / sizeof(objs[0]); i++) {
        if (ds4_qwen4exp_rollback_register(set, objs[i].id, &objs[i].obj,
                                           err, errlen) != 0) {
            return false;
        }
    }
    return ds4_qwen4exp_rollback_check(set, err, errlen) == 0;
}

#ifdef DS4_TEST_HOOKS
/*
 * Test entry point: open a synthetic qwen4exp file, bind it, open a session and
 * run `runs` forwards of the same tokens with a reset between each, writing the
 * last-token logits of every run into `logits_out` back to back.
 *
 * The PLE block runs as an identity: its kernels are not written.  Everything
 * else -- the embedding gather, both hyper-connection mixers, the GDN blocks,
 * the QSA block with its indexer, the MoE and the LM head -- is the production
 * path.
 *
 * Returns the number of runs that completed.
 */



/* Run the head's 49th block end to end on a real head GGUF.
 *
 * The MTP cycle test cannot cover this: it links no backend, supplies its own
 * host ds4_gpu_tensor and runs on every host without a device.  The block is
 * device code over two real mappings, so its coverage lives here.
 *
 * Runs the block twice on the same input and reports whether it succeeded,
 * whether the two runs agree bit for bit, and whether the stream moved at all
 * -- a block that quietly did nothing would otherwise pass the first two. */
int ds4_qwen4exp_test_head_block_run(const char *target_path,
                                     const char *head_path,
                                     uint32_t    n_tokens,
                                     int        *ran_out,
                                     int        *deterministic_out,
                                     int        *changed_out) {
    static ds4_model target_model, head_model;
    static ds4_qwen4exp_weights target_w;
    static ds4_qwen4exp_mtp_weights head_w;

    if (ran_out) *ran_out = 0;
    if (deterministic_out) *deterministic_out = 0;
    if (changed_out) *changed_out = 0;
    if (!ds4_gpu_init()) return -1;

    ds4_qwen4exp_test_set_free_memory_override(
            64ull * 1024ull * 1024ull * 1024ull, true);
    model_open(&target_model, target_path, true, false);
    config_validate_qwen4exp_model(&target_model);
    weights_bind_qwen4exp(&target_w, &target_model);

    model_open(&head_model, head_path, true, false);
    config_validate_qwen4exp_mtp_model(&head_model);
    weights_bind_qwen4exp_mtp(&head_w, &head_model);

    ds4_qwen4exp_session *s = ds4_qwen4exp_session_open(
            &target_w, &target_model, 256u, n_tokens,
            target_w.plan.resident_bytes, 0u);
    ds4_qwen4exp_test_set_free_memory_override(0, false);
    if (!s) {
        model_close(&head_model);
        model_close(&target_model);
        return -2;
    }

    int rc = -3;
    const uint64_t wide = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t bytes = (uint64_t)n_tokens * wide * sizeof(float);
    float *in = calloc((size_t)n_tokens * wide, sizeof(float));
    float *a = calloc((size_t)n_tokens * wide, sizeof(float));
    float *b = calloc((size_t)n_tokens * wide, sizeof(float));
    ds4_gpu_tensor *hyper = ds4_gpu_tensor_alloc(bytes);

    if (in && a && b && hyper &&
        ds4_qwen4exp_session_add_head_block(s, &head_w.block, &head_model,
                                            head_w.block_index)) {
        for (uint64_t i = 0; i < (uint64_t)n_tokens * wide; i++) {
            in[i] = (float)((int)((i * 2654435761u) % 17u) - 8) * 0.125f;
        }
        ds4_qwen4exp_head_block_ctx ctx = { &head_w.block, &head_model };

        int ok1 = 0, ok2 = 0;
        if (ds4_gpu_tensor_write(hyper, 0, in, bytes) &&
            ds4_gpu_begin_commands()) {
            ok1 = ds4_qwen4exp_graph_head_block(&ctx, s, hyper,
                                                head_w.block_index, 0u, n_tokens);
            if (!ds4_gpu_end_commands() || !ds4_gpu_synchronize()) ok1 = 0;
        }
        if (ok1) (void)ds4_gpu_tensor_read(hyper, 0, a, bytes);

        if (ok1 && ds4_gpu_tensor_write(hyper, 0, in, bytes) &&
            ds4_gpu_begin_commands()) {
            ok2 = ds4_qwen4exp_graph_head_block(&ctx, s, hyper,
                                                head_w.block_index, 0u, n_tokens);
            if (!ds4_gpu_end_commands() || !ds4_gpu_synchronize()) ok2 = 0;
        }
        if (ok2) (void)ds4_gpu_tensor_read(hyper, 0, b, bytes);

        if (ran_out) *ran_out = (ok1 && ok2) ? 1 : 0;
        if (ok1 && ok2) {
            if (deterministic_out) {
                *deterministic_out = memcmp(a, b, (size_t)bytes) == 0 ? 1 : 0;
            }
            if (changed_out) {
                *changed_out = memcmp(a, in, (size_t)bytes) != 0 ? 1 : 0;
            }
            rc = 0;
        }
    }

    ds4_gpu_tensor_free(hyper);
    free(b); free(a); free(in);
    ds4_qwen4exp_session_close(s);
    model_close(&head_model);
    model_close(&target_model);
    return rc;
}

/* THE HEAD'S LM HEAD IS THE TARGET'S LM HEAD, BIT FOR BIT.
 *
 * The head borrows the target's `output` weight, so the only thing that can
 * separate its logits from the ones the tower would have produced for the same
 * row is which matmul ENTRY the head's hook is bound to.  The upstream entry
 * ds4_gpu_matmul_q8_0_tensor picks its kernel by row count; the tower routes
 * through ds4_qwen4exp_matmul.h, which gives every row the one-row reduction at
 * every width.  Binding the head to the upstream one put the eh_proj's n_hc
 * rows on the wide side of that line.
 *
 * So this asks the hook itself, the way the head gets it, two questions:
 *
 *   one row      the hook's logits equal the tower's for the same row
 *   row invariant row t of an n-row call equals a one-row call on row t
 *
 * Both are memcmp, not a tolerance: there is no width at which the head is
 * allowed to be approximately the tower.  A rebinding back to the upstream
 * entry fails the second, which is the one that was actually wrong.
 */
int ds4_qwen4exp_test_head_lm_head_identity(const char *target_path,
                                            uint32_t    n_rows,
                                            int        *one_row_same_out,
                                            int        *row_invariant_out) {
    static ds4_model target_model;
    static ds4_qwen4exp_weights target_w;

    if (one_row_same_out) *one_row_same_out = 0;
    if (row_invariant_out) *row_invariant_out = 0;
    if (n_rows == 0u) return -1;
    if (!ds4_gpu_init()) return -1;

    ds4_qwen4exp_test_set_free_memory_override(
            64ull * 1024ull * 1024ull * 1024ull, true);
    model_open(&target_model, target_path, true, false);
    config_validate_qwen4exp_model(&target_model);
    weights_bind_qwen4exp(&target_w, &target_model);
    ds4_qwen4exp_test_set_free_memory_override(0, false);

    const uint32_t n_vocab = g_ds4_qwen4exp.n_vocab;
    const uint64_t in_bytes = (uint64_t)n_rows * DS4_N_EMBD * sizeof(float);
    const uint64_t out_bytes = (uint64_t)n_rows * n_vocab * sizeof(float);

    ds4_qwen4exp_mtp_gpu_hooks hooks;
    memset(&hooks, 0, sizeof(hooks));
    ds4_qwen4exp_mtp_default_hooks(&hooks);

    int rc = -3;
    float *x = calloc((size_t)n_rows * DS4_N_EMBD, sizeof(float));
    float *hook_wide = calloc((size_t)n_rows * n_vocab, sizeof(float));
    float *hook_one  = calloc((size_t)n_rows * n_vocab, sizeof(float));
    float *tower_one = calloc((size_t)n_vocab, sizeof(float));
    ds4_gpu_tensor *t_x = ds4_gpu_tensor_alloc(in_bytes);
    ds4_gpu_tensor *t_out = ds4_gpu_tensor_alloc(out_bytes);

    if (x && hook_wide && hook_one && tower_one && t_x && t_out) {
        /* Rows that differ from each other, so a kernel that broadcast row 0
         * over the batch would be caught rather than agreeing by accident. */
        for (uint64_t i = 0; i < (uint64_t)n_rows * DS4_N_EMBD; i++) {
            x[i] = (float)((int)((i * 2654435761u) % 23u) - 11) * 0.0625f;
        }

        const void *map = QW_MAP(&target_model, target_w.output);
        const uint64_t mapsz = QW_MAPSZ(&target_model, target_w.output);
        const uint64_t off = QW_OFF(target_w.output);

        int ok = ds4_gpu_tensor_write(t_x, 0, x, in_bytes) != 0;

        /* The hook at the head's own width. */
        if (ok) {
            ok = hooks.matmul_q8_0(t_out, map, mapsz, off, DS4_N_EMBD, n_vocab,
                                   t_x, n_rows) != 0 &&
                 ds4_gpu_synchronize() != 0 &&
                 ds4_gpu_tensor_read(t_out, 0, hook_wide, out_bytes) != 0;
        }

        /* The same rows one at a time, through the hook and through the
         * tower's own entry.  A one-row view of the input keeps the two calls
         * reading the same floats. */
        for (uint32_t r = 0; ok && r < n_rows; r++) {
            const uint64_t xoff = (uint64_t)r * DS4_N_EMBD * sizeof(float);
            ds4_gpu_tensor *xv = ds4_gpu_tensor_view(
                    t_x, xoff, (uint64_t)DS4_N_EMBD * sizeof(float));
            if (!xv) { ok = 0; break; }
            ok = hooks.matmul_q8_0(t_out, map, mapsz, off, DS4_N_EMBD, n_vocab,
                                   xv, 1u) != 0 &&
                 ds4_gpu_synchronize() != 0 &&
                 ds4_gpu_tensor_read(t_out, 0,
                                     hook_one + (size_t)r * n_vocab,
                                     (uint64_t)n_vocab * sizeof(float)) != 0;
            if (ok && r == 0u) {
                ok = ds4_qwen4exp_matmul_q8_0(t_out, map, mapsz, off,
                                              DS4_N_EMBD, n_vocab, xv, 1u) != 0 &&
                     ds4_gpu_synchronize() != 0 &&
                     ds4_gpu_tensor_read(t_out, 0, tower_one,
                                         (uint64_t)n_vocab * sizeof(float)) != 0;
            }
            ds4_gpu_tensor_free(xv);
        }

        if (ok) {
            if (one_row_same_out) {
                *one_row_same_out =
                    memcmp(hook_one, tower_one,
                           (size_t)n_vocab * sizeof(float)) == 0 ? 1 : 0;
            }
            if (row_invariant_out) {
                *row_invariant_out =
                    memcmp(hook_wide, hook_one, (size_t)out_bytes) == 0 ? 1 : 0;
            }
            rc = 0;
        }
    }

    ds4_gpu_tensor_free(t_out);
    ds4_gpu_tensor_free(t_x);
    free(tower_one); free(hook_one); free(hook_wide); free(x);
    model_close(&target_model);
    return rc;
}

uint32_t ds4_qwen4exp_test_graph_run(const char    *path,
                                     uint32_t       n_ctx,
                                     const int32_t *tokens,
                                     uint32_t       n_tokens,
                                     uint32_t       runs,
                                     uint64_t       free_bytes,
                                     float         *logits_out,
                                     float         *hyper_out,
                                     uint32_t      *hyper_row_floats_out,
                                     int           *hyper_is_mixer_input_out,
                                     uint8_t       *trace_out,
                                     uint32_t      *trace_len_out,
                                     uint32_t       trace_cap,
                                     uint32_t      *n_vocab_out) {
    static ds4_model model;
    static ds4_qwen4exp_weights weights;

    /* The synthetic set is 6.7 GiB resident and the guard wants 10 GiB of
     * headroom on top, which no development Mac has free while it is also
     * running a build.  The test states the reading it wants; the refusal
     * itself is exercised by the session-guard hook below. */
    ds4_qwen4exp_test_set_free_memory_override(free_bytes, true);
    model_open(&model, path, true, false);
    config_validate_qwen4exp_model(&model);
    weights_bind_qwen4exp(&weights, &model);
    if (n_vocab_out) *n_vocab_out = g_ds4_qwen4exp.n_vocab;
    const uint32_t hyper_row_floats = (uint32_t)DS4_N_HC * DS4_N_EMBD;
    if (hyper_row_floats_out) *hyper_row_floats_out = hyper_row_floats;
    if (hyper_is_mixer_input_out) *hyper_is_mixer_input_out = 0;
    if (trace_len_out) *trace_len_out = 0;

    ds4_qwen4exp_session *s = ds4_qwen4exp_session_open(
            &weights, &model, n_ctx, n_tokens, weights.plan.resident_bytes,
            0u);
    if (!s) {
        model_close(&model);
        return 0;
    }

    uint32_t done = 0;
    for (uint32_t r = 0; r < runs; r++) {
        ds4_qwen4exp_session_reset(s);
        if (!ds4_qwen4exp_graph_forward(s, &weights, &model, tokens, n_tokens,
                                        logits_out + (uint64_t)r * g_ds4_qwen4exp.n_vocab)) {
            break;
        }
        /* The pre-final-mixer rows, which is what the native MTP head reads.
         * Taken through the public accessor, after the forward, so the test
         * exercises the same call L9 will. */
        if (hyper_out) {
            const uint64_t off = (uint64_t)r * n_tokens * hyper_row_floats;
            if (!ds4_qwen4exp_session_read_hyper(s, 0, n_tokens, hyper_out + off)) {
                break;
            }
        }
        /* Content check: what the final mixer read, snapshotted inside the
         * batch, against what the accessor returns after it.  Equal means the
         * accessor hands back the mixer's input AND the mixer did not write
         * through it. */
        if (hyper_is_mixer_input_out && s->final_mixer_snapshot) {
            const uint64_t bytes =
                (uint64_t)n_tokens * hyper_row_floats * sizeof(float);
            float *snap = (float *)malloc((size_t)bytes);
            float *live = (float *)malloc((size_t)bytes);
            int same = 0;
            if (snap && live &&
                ds4_gpu_tensor_read(s->final_mixer_snapshot, 0, snap, bytes) &&
                ds4_qwen4exp_session_read_hyper(s, 0, n_tokens, live)) {
                same = memcmp(snap, live, (size_t)bytes) == 0 ? 1 : 0;
            }
            free(live);
            free(snap);
            *hyper_is_mixer_input_out = same;
        }
        if (trace_out && trace_len_out && !g_qwen4exp_trace_overflow) {
            const uint32_t n = g_qwen4exp_trace_len < trace_cap
                             ? g_qwen4exp_trace_len : trace_cap;
            memcpy(trace_out, g_qwen4exp_trace, n);
            *trace_len_out = g_qwen4exp_trace_len;
        }
        done++;
    }

    ds4_qwen4exp_session_close(s);
    model_close(&model);
    ds4_qwen4exp_test_set_free_memory_override(0, false);
    return done;
}

/* How many device buffers the session allocator has handed out since the
 * process started.  A refused open must not move it. */
uint64_t ds4_qwen4exp_test_graph_alloc_count(void) {
    return g_qwen4exp_alloc_count;
}

/* Open a session and immediately close it, reporting whether it opened and
 * what the plan said.  Used to check the memory guard without a forward.
 *
 * The free-memory override is applied AFTER the weights are bound, because the
 * loader has its own guard on the same reading and would refuse the load first;
 * this hook is about the session half of the budget. */
bool ds4_qwen4exp_test_graph_session_open(const char *path,
                                          uint32_t    n_ctx,
                                          uint32_t    n_batch,
                                          uint64_t    bind_free_bytes,
                                          uint64_t    session_free_bytes,
                                          uint64_t    already_resident_bytes,
                                          ds4_qwen4exp_session_plan *plan_out) {
    static ds4_model model;
    static ds4_qwen4exp_weights weights;

    ds4_qwen4exp_test_set_free_memory_override(bind_free_bytes, true);
    model_open(&model, path, true, false);
    config_validate_qwen4exp_model(&model);
    weights_bind_qwen4exp(&weights, &model);

    if (plan_out) {
        ds4_qwen4exp_session_plan_compute(plan_out, n_ctx, n_batch);
    }
    ds4_qwen4exp_test_set_free_memory_override(session_free_bytes, true);
    ds4_qwen4exp_session *s = ds4_qwen4exp_session_open(
            &weights, &model, n_ctx, n_batch, weights.plan.resident_bytes,
            already_resident_bytes);
    ds4_qwen4exp_test_set_free_memory_override(0, false);
    const bool opened = s != NULL;
    ds4_qwen4exp_session_close(s);
    model_close(&model);
    return opened;
}
#endif /* DS4_TEST_HOOKS */

#else /* DS4_NO_GPU */

/*
 * CPU-only build.  Every op in this tower is a device kernel, and upstream's
 * CPU reference covers GLM 5.2 alone, so there is nothing to fall back to.
 * The plan still computes -- it is arithmetic over the shape preset, and the
 * loader test reads it -- but a session refuses by name.
 */

ds4_qwen4exp_session *ds4_qwen4exp_session_open(
        const ds4_qwen4exp_weights *w,
        const ds4_model            *m,
        uint32_t                    n_ctx,
        uint32_t                    n_batch,
        uint64_t                    resident_weight_bytes,
        uint64_t                    already_resident_bytes) {
    (void)w; (void)m; (void)n_ctx; (void)n_batch; (void)resident_weight_bytes;
    (void)already_resident_bytes;
    fprintf(stderr,
            "ds4: qwen4exp needs a graph backend; this build has none\n");
    return NULL;
}

void ds4_qwen4exp_session_close(ds4_qwen4exp_session *s) { (void)s; }

const ds4_qwen4exp_session_plan *ds4_qwen4exp_session_plan_of(
        const ds4_qwen4exp_session *s) {
    (void)s;
    return NULL;
}

void ds4_qwen4exp_session_reset(ds4_qwen4exp_session *s) { (void)s; }

uint32_t ds4_qwen4exp_session_pos(const ds4_qwen4exp_session *s) {
    (void)s;
    return 0;
}

ds4_gpu_tensor *ds4_qwen4exp_session_hyper(const ds4_qwen4exp_session *s) {
    (void)s;
    return NULL;
}

bool ds4_qwen4exp_session_read_hyper(const ds4_qwen4exp_session *s,
                                     uint32_t first, uint32_t n_tokens,
                                     float *out) {
    (void)s; (void)first; (void)n_tokens; (void)out;
    return false;
}

bool ds4_qwen4exp_graph_forward(ds4_qwen4exp_session       *s,
                                const ds4_qwen4exp_weights *w,
                                const ds4_model            *m,
                                const int32_t              *tokens,
                                uint32_t                    n_tokens,
                                float                      *logits_out) {
    (void)s; (void)w; (void)m; (void)tokens; (void)n_tokens; (void)logits_out;
    return false;
}

#endif /* DS4_NO_GPU */
