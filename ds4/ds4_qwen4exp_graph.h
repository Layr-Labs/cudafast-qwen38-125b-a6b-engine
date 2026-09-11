#ifndef DS4_QWEN4EXP_GRAPH_H
#define DS4_QWEN4EXP_GRAPH_H

/*
 * qwen4exp -- serial prefill and decode over the bound weights.
 *
 * Internal header.  Include it from ds4.c after ds4_qwen4exp.h, whose
 * ds4_qwen4exp_weights and ds4_qwen4exp_config it names, and after ds4_gpu.h.
 * ds4_qwen4exp_graph.inc holds the implementation.
 *
 * Where this sits
 * ---------------
 * qwen4exp is a variant of the GLM_DSA family, so it reaches the same session
 * and sampler machinery as GLM 5.3.  What it does not share is the layer body:
 * the tensor names, the gated residual, the linear-attention block and the
 * router all differ, so the forward lives here rather than branching the GLM
 * graph in a dozen places.
 *
 * The layer schedule, 48 blocks, 4-periodic:
 *
 *     PLE residual add                   block 1 only, FIRST in the layer
 *     HC pre-mix (attention side)        every block
 *       GDN block   il % 4 != 3          36 blocks
 *       QSA block   il % 4 == 3          12 blocks
 *     HC inject                          every block
 *     HC pre-mix (FFN side)              every block
 *       MoE: router, experts, shared     every block
 *     HC inject                          every block
 *
 * The PLE slot leads the layer and is a plain `stream = stream + ple(stream)`
 * over the whole hyper-connection stream, not an inject: MLX Qwen4Exp.swift
 * lines 113-121 apply it before `attnHyperConnection.mixWithInject`.  Running
 * it at the tail instead would hand this block's n-gram contribution to the
 * NEXT block and leave block 1's own attention and MoE reading a stream
 * without it.
 *
 * then the final mixer, which has no inject head and stands in for the
 * `model.norm` this checkpoint does not carry, and the LM head.
 *
 * What is real and what is hooked is recorded next to each call in the .inc.
 * Every op the tower calls is written on both backends: Metal in metal/, CUDA
 * in ds4_cuda_qwen4exp.cu.  The PLE block refuses by name for what its kernels
 * do not cover: an n-gram table that is not IQ4_NL, a row width that is not a
 * whole number of IQ4_NL blocks, a convolution window this build is not sized
 * for, and an artifact whose hash constants could not be read.
 *
 * Memory safety
 * -------------
 * Session state is sized before anything is allocated, added to the resident
 * weight bytes, and checked against free unified memory with the same 10 GiB
 * headroom the loader uses.  ds4_qwen4exp_session_open() refuses by name and
 * allocates nothing when the budget does not fit.
 */

/* Highest context this build sizes a session for.  The production artifact
 * declares 262144; a session asks for what it needs and is refused above this. */
#define DS4_QWEN4EXP_MAX_CTX 262144u

/* The QSA indexer selects whole blocks of `compress_ratio` tokens and the
 * expansion also carries the query's own incomplete block, so a query can name
 * up to top_k * ratio + ratio - 1 tokens. */
#define DS4_QWEN4EXP_MAX_SELECTED(top_k, ratio) ((top_k) * (ratio) + (ratio) - 1u)

/* Rows the PLE depthwise convolution carries between calls:
 * (ple_conv_kernel - 1) * ngram_size, because the convolution is DILATED by
 * the n-gram size.  Nine on the pinned checkpoint; the block refuses by name
 * when the artifact asks for another. */
#define DS4_QWEN4EXP_PLE_STATE_ROWS 9u

/* Per-family session bytes, the KV/state half of the memory plan. */
typedef struct {
    uint64_t gdn_recurrent_bytes; /* [n_value_head][dim][state] f32 per GDN layer */
    uint64_t gdn_conv_bytes;      /* [conv - 1][conv_dim] f32 per GDN layer      */
    uint64_t qsa_kv_bytes;        /* k and v caches per QSA layer                */
    uint64_t qsa_indexer_bytes;   /* exact key tape and pooled blocks per QSA layer */
    uint64_t ple_state_bytes;     /* dilated conv window, [9][n_hc * n_embd]     */
    /* The speculative cycle's per-row state slots.  A verify of N + 1 rows
     * mirrors the carried state after each of its first N rows, so a round
     * that accepts `a` drafts adopts slot `a` instead of rewinding and running
     * a shorter forward again.  One slot per DRAFT: the last row's state is
     * the live state.  Charged whether or not a drafter is armed, so the plan
     * does not shrink under a serial leg and then overrun under an MTP one. */
    uint32_t spec_snapshot_slots;
    uint64_t spec_snapshot_bytes;
    uint64_t activation_bytes;    /* every scratch buffer one forward needs      */
    uint64_t total_bytes;
    uint32_t n_ctx;
    uint32_t n_batch;
    uint32_t n_gdn_layer;
    uint32_t n_qsa_layer;
} ds4_qwen4exp_session_plan;

/*
 * One sequence's state.  Every buffer is device-resident and lives for the
 * session; nothing here is allocated per forward.
 */
typedef struct ds4_qwen4exp_session ds4_qwen4exp_session;

/* Compute the session plan without allocating.  `n_batch` is the widest
 * prefill chunk the session will be asked for. */
void ds4_qwen4exp_session_plan_compute(ds4_qwen4exp_session_plan *plan,
                                       uint32_t n_ctx,
                                       uint32_t n_batch);

/* Print the plan the way the loader prints the weight plan. */
void ds4_qwen4exp_session_plan_print(const ds4_qwen4exp_session_plan *plan);

/*
 * Open a session.  `resident_weight_bytes` is the loader's plan.resident_bytes,
 * and `already_resident_bytes` is how much of it the engine has ALREADY put on
 * the device, which the guard must not ask for a second time;
 * the guard is applied to the sum, so the two halves of the budget are checked
 * together and not one after the other.
 *
 * Returns NULL after printing a named refusal when the geometry is out of
 * range, when free memory cannot hold the budget, or when a device buffer
 * cannot be allocated.  Nothing is left allocated on any failure path.
 */
ds4_qwen4exp_session *ds4_qwen4exp_session_open(
        const ds4_qwen4exp_weights *w,
        const ds4_model            *m,
        uint32_t                    n_ctx,
        uint32_t                    n_batch,
        uint64_t                    resident_weight_bytes,
        uint64_t                    already_resident_bytes);

void ds4_qwen4exp_session_close(ds4_qwen4exp_session *s);

const ds4_qwen4exp_session_plan *ds4_qwen4exp_session_plan_of(
        const ds4_qwen4exp_session *s);

/* Drop every cache and state back to the start of the sequence.  The buffers
 * stay allocated: this is what a free-run repeat calls between runs. */
void ds4_qwen4exp_session_reset(ds4_qwen4exp_session *s);

uint32_t ds4_qwen4exp_session_pos(const ds4_qwen4exp_session *s);

/*
 * Run `n_tokens` tokens through the tower and leave `n_vocab` f32 logits for
 * the LAST token in `logits_out`.
 *
 * One entry for prefill and decode: n_tokens > 1 takes the GDN prefill kernel
 * and the batched attention path, n_tokens == 1 the recurrent decode step.
 * The caller supplies the token ids; the session advances its position by
 * n_tokens on success.
 *
 * Returns false after a named message when an op refuses.  A refusal leaves
 * the session position unchanged but the state buffers undefined, so the
 * caller must reset before reusing the session.
 */
bool ds4_qwen4exp_graph_forward(ds4_qwen4exp_session       *s,
                                const ds4_qwen4exp_weights *w,
                                const ds4_model            *m,
                                const int32_t              *tokens,
                                uint32_t                    n_tokens,
                                float                      *logits_out);

/*
 * The PRE-FINAL-MIXER hyper-connection stream, [n_tokens][n_hc][n_embd] f32.
 *
 * This is what the native MTP head consumes: it reads the target's stream
 * BEFORE the final mixer collapses n_hc * n_embd to n_embd, and keeps its own
 * cache stack.  After a successful ds4_qwen4exp_graph_forward() the buffer
 * still holds the rows for that call's tokens, because the final mixer writes
 * to `mixed` and never touches `hyper`.
 *
 * ds4_qwen4exp_session_hyper() hands back the device tensor for a caller that
 * stays on the GPU.  ds4_qwen4exp_session_read_hyper() copies `n_tokens` rows
 * out to the host, starting at row `first`; it returns false on a bad range or
 * a failed read.  Both return NULL/false before the first forward.
 */
ds4_gpu_tensor *ds4_qwen4exp_session_hyper(const ds4_qwen4exp_session *s);

bool ds4_qwen4exp_session_read_hyper(const ds4_qwen4exp_session *s,
                                     uint32_t                    first,
                                     uint32_t                    n_tokens,
                                     float                      *out);


/* ------------------------------------------------------------------------
 * The MTP model seam
 * ------------------------------------------------------------------------
 *
 * ds4_qwen4exp_mtp_cycle() drives the target through callbacks; these are the
 * graph's half.  verify_rows and decode_token are ONE code path, because the
 * cycle requires row t of an n-row verify to equal a one-row decode from the
 * same state bit for bit.
 */

/* `n_tokens` rows: the pre-final-mixer stream into `hc_rows`
 * ([n_tokens][n_hc * n_embd] f32, may be NULL) and `logit_rows` rows of logits
 * into `logits`.  `logit_rows` is 1 -- the LAST row's, which is what a prefill
 * or a serial step reads -- or n_tokens, which is what the speculative accept
 * loop reads: every compared row's distribution out of the one forward, with
 * the LM head's 676 MiB weight read once instead of once per row.  Advances
 * the session by n_tokens. */
bool ds4_qwen4exp_graph_verify_rows(ds4_qwen4exp_session       *s,
                                    const ds4_qwen4exp_weights *w,
                                    const ds4_model            *m,
                                    const int32_t              *tokens,
                                    uint32_t                    n_tokens,
                                    float                      *hc_rows,
                                    float                      *logits,
                                    uint32_t                    logit_rows);

/* Speculative fast path: run the same wide head, retain its logits on device,
 * and return only each row's greedy winner.  The cycle subsequently reads one
 * selected frontier row with read_logit_row(). */
bool ds4_qwen4exp_graph_verify_top1_rows(ds4_qwen4exp_session       *s,
                                         const ds4_qwen4exp_weights *w,
                                         const ds4_model            *m,
                                         const int32_t              *tokens,
                                         uint32_t                    n_tokens,
                                         float                      *hc_rows,
                                         int                        *row_top1);

/*
 * The depth-1 folded round: verify_top1_rows plus the head's drafts over the
 * same rows, in ONE command batch, ONE synchronize and ONE packed readback.
 * Row j of the head forward takes (row_top1[j], hyper row j) at position
 * pos + j, which is exactly the forward the rejecting outcome drafts from
 * (row a) and the accepting one seeds and drafts from (rows 0..1); the cycle
 * keeps the draft the acceptance outcome selects and truncates the head cache
 * to pos + a + 1.  The hc rows never cross to the host on this path: the
 * head stages its inputs straight off the verify's device buffers.  Advances
 * the session by n_tokens on success, like the verify above.
 */
bool ds4_qwen4exp_graph_verify_top1_draft_rows(
        ds4_qwen4exp_session *s, const ds4_qwen4exp_weights *w,
        const ds4_model *m, ds4_qwen4exp_mtp_head *h,
        const int32_t *tokens, uint32_t n_tokens,
        int *row_top1, int *row_drafts);
bool ds4_qwen4exp_graph_read_logit_row(ds4_qwen4exp_session *s,
                                       uint32_t row,
                                       float *logits);

/* The target's LM head over ONE supplied pre-final-mixer row.  Runs the final
 * mixer and the head, the same two ops the forward ends with, so a row that
 * came out of a verify yields exactly the logits that verify would have. */
bool ds4_qwen4exp_graph_head_logits(ds4_qwen4exp_session       *s,
                                    const ds4_qwen4exp_weights *w,
                                    const ds4_model            *m,
                                    const float                *hc_row,
                                    float                      *logits);

/* Register every rollback object this session owns and check the set is
 * complete.  Refuses by name through `err` when a registration is rejected or
 * an id is left unregistered. */
bool ds4_qwen4exp_session_rollback_set(ds4_qwen4exp_session      *s,
                                       ds4_qwen4exp_rollback_set *set,
                                       char *err, size_t errlen);


/* ------------------------------------------------------------------------
 * The MTP head's 49th block
 * ------------------------------------------------------------------------
 *
 * An ordinary qwen4exp FULL-ATTENTION block run on the head's weights and the
 * head's cache: the QSA sequence, the MoE trio and the two hyper-connection
 * mixers, adding no kernel of its own.  The head's block is layer index
 * n_layer (48) in the SAME session, so its caches are the per-layer slots at
 * that index and nothing in the tower can reach them.
 */
typedef struct {
    const ds4_qwen4exp_layer_weights *l;   /* blk.<n_layer>.* from the head   */
    const ds4_model                  *m;   /* the --mtp GGUF, not the target  */
} ds4_qwen4exp_head_block_ctx;

/* Allocate the head block's cache slot and upload its three per-head norms.
 * Refuses by name on a bad index or a block not bound as full attention. */
bool ds4_qwen4exp_session_add_head_block(ds4_qwen4exp_session             *s,
                                         const ds4_qwen4exp_layer_weights *l,
                                         const ds4_model                  *m,
                                         uint32_t                          il);

/* ds4_qwen4exp_block_forward_fn: `graph` is a ds4_qwen4exp_head_block_ctx and
 * `cache` is the session holding the slot.  NONZERO on success. */
int ds4_qwen4exp_graph_head_block(void *graph, void *cache,
                                  ds4_gpu_tensor *hyper, uint32_t il,
                                  uint32_t pos0, uint32_t n_tokens);

#endif /* DS4_QWEN4EXP_GRAPH_H */

#ifdef DS4_TEST_HOOKS
/* Per-subsystem checksums of the state a prefill leaves behind; see the
 * definition for the order.  Test builds only. */
void ds4_qwen4exp_test_state_checksums(const ds4_qwen4exp_session *s,
                                       double *out, int cap);

/* The same state one object at a time, in tower order, so the FIRST differing
 * index names the first object that stopped agreeing.  `index` runs to
 * ds4_qwen4exp_test_state_object_count() - 1; returns 0 when it does not. */
int ds4_qwen4exp_test_state_object_count(const ds4_qwen4exp_session *s);
int ds4_qwen4exp_test_state_object(const ds4_qwen4exp_session *s, int index,
                                   char *name, size_t name_len, double *sum);

/* The state bytes a later round would READ -- the adopted snapshot row when
 * the rollback table armed one, the live buffers otherwise: conv then
 * recurrent per GDN layer, then the PLE window, then the host n-gram history
 * struct.  `out` NULL returns the byte count; a short cap refuses. */
int64_t ds4_qwen4exp_test_adopted_state_bytes(const ds4_qwen4exp_session *s,
                                              uint8_t *out, uint64_t cap);
#endif
