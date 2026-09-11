#ifndef DS4_GPU_H
#define DS4_GPU_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * GPU Tensor and Command Lifetime.
 * =========================================================================
 *
 * Opaque device tensor used by the DS4-specific GPU executor.
 *
 * The public GPU API is tensor-resident: activations, KV state, and scratch
 * buffers stay device-owned across the whole prefill/decode command sequence.
 */
#ifndef DS4_GPU_TENSOR_DEFINED
#define DS4_GPU_TENSOR_DEFINED
typedef struct ds4_gpu_tensor ds4_gpu_tensor;
#endif

#ifndef DS4_GPU_ATTENTION_DECODE_ROW_DEFINED
#define DS4_GPU_ATTENTION_DECODE_ROW_DEFINED
#define DS4_GPU_ATTENTION_DECODE_BATCH_MAX 32u
typedef struct {
    uint64_t raw_kv;
    uint64_t comp_kv;
    uint64_t topk;
    uint32_t pos;
    uint32_t n_raw;
    uint32_t raw_cap;
    uint32_t raw_start;
    uint32_t n_comp;
    uint32_t top_k;
    uint32_t window;
    uint32_t ratio;
    uint32_t indexed;
} ds4_gpu_attention_decode_row;
#endif

int ds4_gpu_init(void);
void ds4_gpu_cleanup(void);

ds4_gpu_tensor *ds4_gpu_tensor_alloc(uint64_t bytes);
ds4_gpu_tensor *ds4_gpu_tensor_alloc_managed(uint64_t bytes);
ds4_gpu_tensor *ds4_gpu_tensor_view(const ds4_gpu_tensor *base, uint64_t offset, uint64_t bytes);
void ds4_gpu_tensor_free(ds4_gpu_tensor *tensor);
uint64_t ds4_gpu_tensor_bytes(const ds4_gpu_tensor *tensor);
void *ds4_gpu_tensor_contents(ds4_gpu_tensor *tensor);
int ds4_gpu_tensor_fill_f32(ds4_gpu_tensor *tensor, float value, uint64_t count);
int ds4_gpu_tensor_write(ds4_gpu_tensor *tensor, uint64_t offset, const void *data, uint64_t bytes);
int ds4_gpu_tensor_read(const ds4_gpu_tensor *tensor, uint64_t offset, void *data, uint64_t bytes);
int ds4_gpu_tensor_copy(ds4_gpu_tensor *dst, uint64_t dst_offset,
                          const ds4_gpu_tensor *src, uint64_t src_offset,
                          uint64_t bytes);
int ds4_gpu_tensor_copy_f32_to_f16(ds4_gpu_tensor *dst, uint64_t dst_offset,
                                   const ds4_gpu_tensor *src, uint64_t src_offset,
                                   uint64_t count);
int ds4_gpu_moe_handoff_pack_tensor(
        ds4_gpu_tensor       *packed,
        const ds4_gpu_tensor *ffn_norm,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t              n_embd,
        uint32_t              n_expert);
int ds4_gpu_pack_slot_rows_f32_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *slots,
        uint32_t                n_rows,
        uint32_t                width,
        uint32_t                n_slots,
        uint32_t                slot_cap);

int ds4_gpu_begin_commands(void);
int ds4_gpu_flush_encoder(void);
int ds4_gpu_flush_commands(void);
int ds4_gpu_commands_active(void);
#ifdef __APPLE__
int ds4_gpu_parallel_ffn_finish(void);
void ds4_gpu_parallel_ffn_abort(void);
int ds4_gpu_parallel_ffn_start(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        ds4_gpu_tensor       *shared_out,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              gate_offset,
        uint64_t              up_offset,
        uint64_t              down_offset,
        uint32_t              model_dim,
        uint32_t              shared_dim,
        const ds4_gpu_tensor *x,
        float                 clamp);
#endif
int ds4_gpu_signal_selected_readback_ready(uint64_t *event_value);
int ds4_gpu_commit_and_wait_selected_readback(uint64_t event_value, const char *label);
int ds4_gpu_wait_selected_readback_ready(uint64_t event_value, const char *label);
#ifdef DS4_ROCM_BUILD
int ds4_gpu_tensor_read_after_selected_event(const ds4_gpu_tensor *tensor,
                                             uint64_t offset,
                                             void *data,
                                             uint64_t bytes,
                                             uint64_t event_value,
                                             const char *label);
#endif
int ds4_gpu_end_commands(void);
int ds4_gpu_synchronize(void);

int ds4_gpu_set_model_map(const void *model_map, uint64_t model_size);

/* Forget every device-side range, view and cache derived from `model_map`.
 *
 * A GGUF is mmapped, used, and munmapped, and the next open can land on the
 * SAME address with the same size.  Both backends short-circuit
 * ds4_gpu_set_model_map() on exactly that pair, so without this call the new
 * mapping silently inherits the previous mapping's registry: on CUDA a set of
 * device ranges and host registrations, on Metal a set of MTLBuffers created
 * over host memory that no longer exists.  Nothing about the identity of a
 * mapping can be recovered from its address once it has been unmapped, so the
 * owner of the mapping has to say when it goes away.  ds4_model_close() does.
 *
 * Safe to call with a base the backend never saw. */
void ds4_gpu_forget_model_map(const void *model_map);

/* How many model-derived device ranges the backend is holding.  For tests:
 * after every close of every shard the count must be zero. */
int ds4_gpu_model_range_count(void);
int ds4_gpu_set_model_fd(int fd);
int ds4_gpu_set_model_fd_for_map(int fd, const void *model_map);
/* Register ONE mapped shard's fd, additively: a GGUF split set calls this once
 * per shard, and every entry stays live so a range is read through the file it
 * actually lives in.  ds4_gpu_set_model_fd_for_map can only name one file, and
 * on a split set that left every other shard's ranges to be copied out of the
 * mapping a page fault at a time. */
int ds4_gpu_set_model_shard_fd(int fd, const void *model_map);
int ds4_gpu_build_derived_artifacts(const void *model_map, uint64_t model_size,
                                    const char *model_path);
int ds4_gpu_model_range_replaced(const void *model_map, uint64_t offset,
                                 uint64_t bytes);
/* Where a cached model range lives on the device, or NULL when it was never
 * cached.  CUDA only; the other backends do not copy the model. */
const void *ds4_gpu_model_range_device_ptr(const void *model_map,
                                           uint64_t offset, uint64_t bytes);
int ds4_gpu_set_model_map_range(const void *model_map, uint64_t model_size, uint64_t map_offset, uint64_t map_size, uint64_t max_tensor_bytes);
/* Add a secondary GGUF mapping without replacing the primary model mapping. */
int ds4_gpu_set_aux_model_map_range(const void *model_map,
                                    uint64_t model_size,
                                    uint64_t map_offset,
                                    uint64_t map_size);
int ds4_gpu_set_model_map_spans(const void *model_map, uint64_t model_size, const uint64_t *offsets, const uint64_t *sizes, uint32_t count, uint64_t max_tensor_bytes);
int ds4_gpu_cache_model_range(const void *model_map, uint64_t model_size, uint64_t offset, uint64_t bytes, const char *label);
int ds4_gpu_cache_q8_f16_range(const void *model_map, uint64_t model_size, uint64_t offset, uint64_t bytes, uint64_t in_dim, uint64_t out_dim, const char *label);
int ds4_gpu_q8_cache_suppressed(void);
void ds4_gpu_set_q8_cache_suppressed(int suppressed);
#ifdef DS4_ROCM_BUILD
void ds4_gpu_release_q8_f16_cache(void);
#endif

/* Model-file ranges assigned to CUDA devices by the multi-GPU placement
 * planner. Metal keeps these declarations for the shared engine interface. */
#ifndef DS4_MAX_GPUS
#define DS4_MAX_GPUS 16
#endif
typedef struct {
    uint64_t source_offset;
    uint64_t bytes;
    int target_device;
} ds4_tensor_range;

int ds4_gpu_device_cache_tensors(int device_id,
                                 const ds4_tensor_range *ranges,
                                 int n_ranges);
int ds4_gpu_register_support_map(const void *map, uint64_t size, uint64_t bias);
int ds4_gpu_device_cache_support_tensors(int device_id,
                                         int entry_device_id,
                                         const ds4_tensor_range *ranges,
                                         int n_ranges,
                                         int from_main_map);
uint64_t ds4_gpu_tier_free_vram(int logical_tier);
int ds4_gpu_lookup_cache(uint64_t source_offset, uint64_t bytes,
                         int *out_device_id, void **out_device_ptr);
int ds4_gpu_lookup_cache_device(uint64_t source_offset, uint64_t bytes);

int ds4_gpu_pro_q4_expert_table_auto_available(void);
int ds4_gpu_preload_q4_expert_tables(const void *model_map, uint64_t model_size,
                                     uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset,
                                     uint64_t gate_expert_bytes, uint64_t down_expert_bytes,
                                     uint32_t n_total_expert);
int ds4_gpu_should_use_managed_kv_cache(uint64_t kv_cache_bytes, uint64_t context_bytes);
void ds4_gpu_set_quality(bool quality);
void ds4_gpu_set_glm_model(bool enabled);
void ds4_gpu_set_ssd_streaming(bool enabled);
void ds4_gpu_set_glm_streaming_prefill_full_layer(bool enabled);
#ifdef __APPLE__
int ds4_gpu_device_is_pre_m5_apple_silicon(void);
int ds4_gpu_device_is_m5_apple_silicon(void);
int ds4_gpu_set_decode_pipeline_fast_lookup(int enabled);
/* Strict test oracle for the fixed decode mul_mv pipeline lookup cache. */
int ds4_gpu_test_decode_pipeline_fast_lookup(void);
/* Strict test oracle for the extended decode mul_mv_ext (nsg + nxpsg) cache. */
int ds4_gpu_test_decode_pipeline_fast_lookup_ext(void);
/* Strict test oracle for the generated resident-prefill MXFP4 half LUT. */
int ds4_gpu_test_mxfp4_down_half_lut(uint16_t *legacy_bits,
                                     uint16_t *lut_bits);
enum {
    DS4_GPU_TEST_MXFP4_PAIR_TAIL_CULL = 1u << 0,
    DS4_GPU_TEST_MXFP4_PAIR_COMPACT_TILE = 1u << 1,
    DS4_GPU_TEST_MXFP4_MAP_SCATTER = 1u << 2,
    DS4_GPU_TEST_MXFP4_DOWN_TAIL_CULL = 1u << 3,
    DS4_GPU_TEST_MXFP4_DOWN_HALF_LUT = 1u << 4,
    DS4_GPU_TEST_OUTPUT_HC_WEIGHTS4 = 1u << 5,
    DS4_GPU_TEST_HC_RMS_SCALE_PROJ = 1u << 6,
};
void ds4_gpu_test_set_flags(uint32_t flags);
void ds4_gpu_release_zero_prefix_prefill_mask_cache(void);
#else
static inline int ds4_gpu_device_is_pre_m5_apple_silicon(void) { return 0; }
static inline int ds4_gpu_device_is_m5_apple_silicon(void) { return 0; }
#endif
void ds4_gpu_set_streaming_expert_cache_budget(uint32_t experts);
void ds4_gpu_set_streaming_expert_cache_expert_bytes(uint64_t bytes);
uint64_t ds4_gpu_recommended_working_set_size(void);
uint32_t ds4_gpu_stream_expert_cache_configured_count(void);
uint32_t ds4_gpu_stream_expert_cache_current_count(void);
typedef struct ds4_gpu_stream_expert_table {
    const void *model_map;
    uint64_t    model_size;
    uint32_t    layer;
    uint32_t    n_total_expert;
    uint64_t    gate_offset;
    uint64_t    up_offset;
    uint64_t    down_offset;
    uint64_t    gate_expert_bytes;
    uint64_t    down_expert_bytes;
} ds4_gpu_stream_expert_table;
/* Reset only the prompt-local eviction heuristic.  The resident SSD expert
 * cache itself is intentionally kept warm across sessions. */
void ds4_gpu_stream_expert_cache_reset_route_hotness(void);
void ds4_gpu_stream_expert_cache_release_resident(void);
uint32_t ds4_gpu_stream_expert_cache_budget_for_expert_size(
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes);
int ds4_gpu_stream_expert_cache_seed_selected(
        const ds4_gpu_stream_expert_table *table,
        const int32_t                     *selected_ids,
        uint32_t                           n_selected);
int ds4_gpu_stream_expert_cache_begin_selected_load(
        const ds4_gpu_stream_expert_table *table,
        const int32_t                     *selected_ids,
        uint32_t                           n_selected);
int ds4_gpu_glm_stream_expert_cache_begin_selected_load_tensor(
        const ds4_gpu_stream_expert_table *table,
        const ds4_gpu_tensor              *selected,
        uint32_t                           n_selected);
#ifdef __APPLE__
/* The async selected-load worker registers itself so Metal cache paths never
 * wait on command buffers from that thread (they fail the load instead and
 * the caller retries synchronously). */
void ds4_gpu_stream_expert_cache_note_service_thread(void);
#endif
#if defined(DS4_ROCM_BUILD) || (!defined(DS4_NO_GPU) && !defined(__APPLE__))
int ds4_gpu_stream_expert_cache_prepare_selected_batch(
        const ds4_gpu_stream_expert_table *table,
        const int32_t                     *selected_ids,
        uint32_t                           n_tokens,
        uint32_t                           n_selected);
#endif
#ifdef DS4_ROCM_BUILD
int ds4_gpu_stream_expert_cache_load_layer(
        const ds4_gpu_stream_expert_table *table);
int ds4_gpu_stream_expert_cache_seed_from_layer_selected(
        const ds4_gpu_stream_expert_table *table,
        const ds4_gpu_tensor             *selected,
        uint32_t                          n_tokens,
        uint32_t                          n_seed_tokens,
        uint32_t                          n_selected);
int ds4_gpu_stream_expert_cache_finish_pending_batch(void);
int ds4_gpu_stream_expert_cache_release_layer_cache(void);
#endif
int ds4_gpu_stream_expert_cache_seed_experts(
        const ds4_gpu_stream_expert_table *table,
        const int32_t                     *expert_ids,
        const uint32_t                    *expert_priorities,
        uint32_t                           n_experts);
#ifdef __APPLE__
/* Seed from mapped weights with blits appended to the active command buffer. */
int ds4_gpu_stream_expert_cache_seed_experts_gpu_copy(
        const ds4_gpu_stream_expert_table *table,
        const int32_t                     *expert_ids,
        const uint32_t                    *expert_priorities,
        uint32_t                           n_experts);
#endif
void ds4_gpu_print_memory_report(const char *label);

/* Tensor-parallel per-layer gates (Metal only).  The encoder calls
 * ds4_gpu_tp_gate_encode() right after the kernels that produce a partial
 * block output in the TP slab: it closes the current encoder, makes the GPU
 * signal a shared event, queues the exchange on a service thread, and makes
 * the GPU wait for the CPU-signaled release before the combine kernel runs.
 * Sequence values are assigned internally and increase monotonically; both
 * ranks encode the identical gate sequence so values pair up by
 * construction.  The exchange callback runs on the service thread and must
 * return nonzero on success. */
typedef int (*ds4_gpu_tp_exchange_fn)(void *ud, uint32_t layer, uint32_t gate, uint64_t seq);
/* Bind one rank of the two-way split. slab is the transport slab tensor and
 * gpu_flags_off is the offset of its GPU-written gate-ready flag words. */
int ds4_gpu_tp_init(uint32_t rank,
                    ds4_gpu_tensor *slab, uint64_t gpu_flags_off,
                    ds4_gpu_tp_exchange_fn fn, void *ud);
void ds4_gpu_tp_shutdown(void);
/* Multi-session TP reuses slab slots across several encoded graph tapes.
 * Shared-event arrival is required in that mode to make each partial vector
 * CPU-visible before the transport thread reads it. */
void ds4_gpu_tp_set_session_batch_mode(int enabled);
/* The coordinator-only DSpark support model does not participate in TP.
 * Suspend ownership only while encoding it; base-model verification remains
 * split across both ranks. */
void ds4_gpu_tp_suspend_expert_sharding(int suspend);
int ds4_gpu_tp_gate_encode(uint32_t layer, uint32_t gate);
/* Verify-block batch gates: one exchange per layer moving `rows` partial
 * rows at once (speculative verify).  The callback runs on the gate service
 * thread with the same ud as the row-gate exchange fn. */
typedef int (*ds4_gpu_tp_batch_exchange_fn)(void *ud, uint32_t layer,
                                            uint32_t rows, uint64_t seq);
void ds4_gpu_tp_set_batch_exchange(ds4_gpu_tp_batch_exchange_fn fn);
int ds4_gpu_tp_batch_gate_encode(uint32_t layer, uint32_t rows);
/* Prefill batch gates: the service thread exchanges `bytes` between two
 * CPU-visible bounce tensors directly (payloads far beyond slab slots). */
typedef int (*ds4_gpu_tp_big_exchange_fn)(void *ud, uint32_t layer,
                                          uint64_t seq, const void *out,
                                          void *in, uint64_t bytes);
void ds4_gpu_tp_set_big_exchange(ds4_gpu_tp_big_exchange_fn fn);
int ds4_gpu_tp_big_gate_encode(uint32_t layer, uint32_t rows,
                               const ds4_gpu_tensor *out_t,
                               ds4_gpu_tensor *in_t,
                               uint64_t bytes);
/* Split big gate: kick publishes the GPU arrival marker (batch shared
 * event, whose completion semantics make the bounce payload visible to
 * the exchange thread) and queues the exchange, returning the gate seq
 * (0 on failure); wait encodes the release.  Multiple kicks may be in
 * flight; waiting on the last seq covers all earlier kicks (monotonic
 * release event, in-order service thread). */
uint64_t ds4_gpu_tp_big_gate_kick(uint32_t layer, uint32_t rows,
                                  const ds4_gpu_tensor *out_t,
                                  ds4_gpu_tensor *in_t,
                                  uint64_t bytes);
int ds4_gpu_tp_big_gate_wait(uint64_t seq);
/* Pause/resume the DVFS keep-alive around work that keeps the GPU busy.
 * No-op when TP is not bound. */
void ds4_gpu_tp_keepalive_pause(int paused);
/* Split attention heads across the two TP ranks in the GLM batch-prefill
 * attention kernels (qk-low, attention-lora, value-project). The caller
 * zeroes the unowned head range of the heads buffer and combines the
 * attn-output partials over the TP big-gate exchange. */
void ds4_gpu_tp_set_attn_head_split(int enabled);
/* Skip the whole-file model residency set (TP sharding: only the
 * owned ranges are warmed; the rest must never be paged in). Call before
 * the model is mapped. */
void ds4_gpu_model_residency_skip(int skip);
/* Nonzero after any gate exchange failed; the eval must abort. */
int ds4_gpu_tp_failed(void);

/* Tensor-parallel sliced projections (Metal decode path only).
 *
 * ds4_gpu_matmul_q8_0_kslice_tensor computes a k-range partial matvec:
 * out[out_dim] = W[:, k_off : k_off + k_cnt] @ x[x_elem_off : +k_cnt] where
 * W rows span full_in_dim quantized Q8_0 elements.  k offsets/counts must be
 * multiples of 32 (Q8_0 block).  Partial results from both ranks sum to the
 * full projection.
 *
 * ds4_gpu_attention_output_q8_tp_tensor is the group-sliced attention output
 * pair: low projection for groups [group0, group0+group_cnt) plus the
 * matching k-slice of the expand projection, producing this rank's partial
 * attention block output (n_tokens == 1 only). */
int ds4_gpu_matmul_q8_0_kslice_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                full_in_dim,
        uint64_t                k_off,
        uint64_t                k_cnt,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                x_elem_off);
/* CUDA multi-row variant. Each input row contains only the owned contiguous
 * K slice, while each output row spans the full projection width. */
int ds4_gpu_matmul_q8_0_kslice_rows_tensor(
        ds4_gpu_tensor       *out,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint64_t              full_in_dim,
        uint64_t              out_dim,
        uint64_t              k_off,
        uint64_t              k_cnt,
        const ds4_gpu_tensor *x,
        uint64_t              n_rows);
int ds4_gpu_matmul_quant_kslice_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                weight_type,
        uint64_t                full_in_dim,
        uint64_t                k_off,
        uint64_t                k_cnt,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                x_elem_off);
int ds4_gpu_attention_output_q8_tp_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *low,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                out_b_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups_total,
        uint32_t                group0,
        uint32_t                group_cnt,
        uint64_t                out_dim,
        const ds4_gpu_tensor *heads);

/* =========================================================================
 * Embeddings and Indexer Helpers.
 * =========================================================================
 *
 * These kernels seed HC state from token embeddings and implement the ratio-4
 * compressed-attention indexer that chooses visible compressed rows.
 */

int ds4_gpu_embed_token_hc_tensor(
        ds4_gpu_tensor *out_hc,
        const void       *model_map,
        uint64_t          model_size,
        uint64_t          weight_offset,
        uint32_t          n_vocab,
        uint32_t          token,
        uint32_t          n_embd,
        uint32_t          n_hc);

int ds4_gpu_embed_tokens_hc_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *tokens,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n_vocab,
        uint32_t                n_tokens,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ds4_gpu_embed_token_q8_0_tensor(
        ds4_gpu_tensor *out,
        const void       *model_map,
        uint64_t          model_size,
        uint64_t          weight_offset,
        uint32_t          n_vocab,
        uint32_t          token,
        uint32_t          n_embd);

int ds4_gpu_embed_tokens_q8_0_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *tokens,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n_vocab,
        uint32_t                n_tokens,
        uint32_t                n_embd);

int ds4_gpu_embed_token_quant_tensor(
        ds4_gpu_tensor *out,
        const void       *model_map,
        uint64_t          model_size,
        uint64_t          weight_offset,
        uint32_t          weight_type,
        uint32_t          n_vocab,
        uint32_t          token,
        uint32_t          n_embd);

int ds4_gpu_embed_tokens_quant_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *tokens,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                weight_type,
        uint32_t                n_vocab,
        uint32_t                n_tokens,
        uint32_t                n_embd);

int ds4_gpu_indexer_score_one_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_head,
        uint32_t                head_dim,
        float                   scale);

int ds4_gpu_indexer_scores_prefill_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                ratio,
        float                   scale);

int ds4_gpu_indexer_scores_decode_batch_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                ratio,
        float                   scale);

int ds4_gpu_dspark_markov_argmax_tensor(ds4_gpu_tensor *out_idx,
                                        const ds4_gpu_tensor *logits_row,
                                        const void *model_map,
                                        uint64_t model_size,
                                        uint64_t w1_offset,
                                        uint64_t w2_offset,
                                        uint32_t prev_token,
                                        uint32_t vocab,
                                        uint32_t rank);
int ds4_gpu_indexer_topk_tensor(
        ds4_gpu_tensor       *selected,
        const ds4_gpu_tensor *scores,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                top_k);

/* =========================================================================
 * Qwen4-Exp (Qwen 3.8 Flash-Next) QSA block.
 *
 * Kernels live in metal/qwen4exp_qsa.metal and ds4_cuda.cu.  Every tensor is
 * f32 and contiguous; the caller owns the projections, which are ordinary
 * matmuls.  A layer runs, in order:
 *
 *   split_qkv -> head_rms_norm(q) -> rope_head(q)
 *             -> head_rms_norm(k) -> rope_head(k) -> write the KV cache
 *   indexer:  head_rms_norm(index q) -> rope_head(index q)
 *             -> indexer_pool_update(raw index k)
 *             -> indexer_scores -> ds4_gpu_indexer_topk_tensor
 *             -> indexer_select
 *   qsa_attention(selected) -> output_gate -> o_proj
 *
 * While the visible context still fits the token budget the indexer is
 * skipped: call qsa_attention with `selected == NULL` for the dense causal
 * set, which is what MLX's `nil` keep mask means.
 * =========================================================================
 */

/* Score written for a block the query cannot see.  Finite on purpose: the
 * Metal library compiles with fast math, so a kernel cannot depend on
 * infinities surviving the argsort or the softmax rescale.  Anything at or
 * below DS4_QWEN4EXP_QSA_MASKED_LIMIT is masked. */
#define DS4_QWEN4EXP_QSA_MASKED_SCORE (-3.0e38f)
#define DS4_QWEN4EXP_QSA_MASKED_LIMIT (-1.0e30f)

/* Split the fused `attn_qkv` row.  The query half is head-major with the
 * output gate interleaved per head (the doubled `q_proj`), NOT two flat
 * halves.  `q` and `gate` are [n_tokens, n_head, head_dim]; `k` and `v` are
 * [n_tokens, n_kv_head, head_dim]. */
int ds4_gpu_qwen4exp_qsa_split_qkv_tensor(
        ds4_gpu_tensor       *q,
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *k,
        ds4_gpu_tensor       *v,
        const ds4_gpu_tensor *fused,
        uint32_t              n_tokens,
        uint32_t              n_head,
        uint32_t              n_kv_head,
        uint32_t              head_dim);

/* Split the DOUBLED query projection, for the layout the loader actually
 * binds: `blk.N.attn_q.weight` doubled (2560 -> 12288) with `attn_k` and
 * `attn_v` as their own tensors, so one matmul per tensor already lays k and v
 * out the way the attention kernel wants them.  The query row is head-major
 * with the gate interleaved per head, exactly as in the fused row above. */
/* Fused Q-Prep: split doubled-query, per-head RMS norm, and partial RoPE. */
int ds4_gpu_qwen4exp_qsa_prep_q_fused_dpos_tensor(
        ds4_gpu_tensor       *q,
        ds4_gpu_tensor       *gate,
        const ds4_gpu_tensor *doubled,
        const ds4_gpu_tensor *weight,
        const ds4_gpu_tensor *inv_freq,
        uint32_t              n_tokens,
        uint32_t              n_head,
        uint32_t              head_dim,
        uint32_t              rot_dim,
        uint32_t              pos0,
        float                 eps,
        float                 weight_offset,
        const ds4_gpu_tensor *d_pos);

int ds4_gpu_qwen4exp_qsa_prep_kv_append_fused_dpos_tensor(
        ds4_gpu_tensor       *k_cache,
        ds4_gpu_tensor       *v_cache,
        ds4_gpu_tensor       *k_out,
        const ds4_gpu_tensor *raw_k,
        const ds4_gpu_tensor *raw_v,
        const ds4_gpu_tensor *weight,
        const ds4_gpu_tensor *inv_freq,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              n_head_kv,
        uint32_t              head_dim,
        uint32_t              rot_dim,
        uint32_t              cache_cap,
        float                 eps,
        float                 weight_offset,
        const ds4_gpu_tensor *d_pos);

int ds4_gpu_qwen4exp_rope_head_dpos_tensor(
        ds4_gpu_tensor       *x,
        const ds4_gpu_tensor *inv_freq,
        uint32_t              n_tokens,
        uint32_t              n_head,
        uint32_t              head_dim,
        uint32_t              rot_dim,
        uint32_t              pos0,
        const ds4_gpu_tensor *d_pos);

int ds4_gpu_qwen4exp_qsa_indexer_pool_update_dpos_tensor(
        ds4_gpu_tensor       *pool,
        ds4_gpu_tensor       *tape,
        const ds4_gpu_tensor *raw_k,
        const ds4_gpu_tensor *k_norm_weight,
        const ds4_gpu_tensor *inv_freq,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_cap,
        uint32_t              head_dim,
        uint32_t              pool_size,
        uint32_t              rot_dim,
        float                 eps,
        float                 weight_offset,
        const ds4_gpu_tensor *d_pos);

int ds4_gpu_qwen4exp_qsa_attention_dpos_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *k_cache,
        const ds4_gpu_tensor *v_cache,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *counts,
        uint32_t              n_tokens,
        uint32_t              n_head,
        uint32_t              n_kv_head,
        uint32_t              head_dim,
        uint32_t              pos0,
        uint32_t              cache_cap,
        uint32_t              max_selected,
        float                 scale,
        const ds4_gpu_tensor *d_pos);

int ds4_gpu_qwen4exp_qsa_prep_q_fused_tensor(
        ds4_gpu_tensor       *q,
        ds4_gpu_tensor       *gate,
        const ds4_gpu_tensor *doubled,
        const ds4_gpu_tensor *weight,
        const ds4_gpu_tensor *inv_freq,
        uint32_t              n_tokens,
        uint32_t              n_head,
        uint32_t              head_dim,
        uint32_t              rot_dim,
        uint32_t              pos0,
        float                 eps,
        float                 weight_offset);

/* Fused KV-Prep & Append: per-head RMS norm on K, partial RoPE on K, direct write
 * of roped K into k_cache, direct write of raw V into v_cache, and write to k_out. */
int ds4_gpu_qwen4exp_qsa_prep_kv_append_fused_tensor(
        ds4_gpu_tensor       *k_cache,
        ds4_gpu_tensor       *v_cache,
        ds4_gpu_tensor       *k_out,
        const ds4_gpu_tensor *raw_k,
        const ds4_gpu_tensor *raw_v,
        const ds4_gpu_tensor *weight,
        const ds4_gpu_tensor *inv_freq,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              n_head_kv,
        uint32_t              head_dim,
        uint32_t              rot_dim,
        uint32_t              cache_cap,
        float                 eps,
        float                 weight_offset);

int ds4_gpu_qwen4exp_qsa_split_doubled_q_tensor(
        ds4_gpu_tensor       *q,
        ds4_gpu_tensor       *gate,
        const ds4_gpu_tensor *doubled,
        uint32_t              n_tokens,
        uint32_t              n_head,
        uint32_t              head_dim);

/* `y = x * rsqrt(mean(x^2) + eps) * (weight_offset + w)` over `head_dim`.
 * `n_rows` counts head vectors: n_tokens * n_head.  `weight_offset` is 1 for
 * a zero-centered checkpoint and 0 for one that bakes the offset in. */
int ds4_gpu_qwen4exp_head_rms_norm_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *weight,
        uint32_t              n_rows,
        uint32_t              head_dim,
        float                 eps,
        float                 weight_offset);

/* Inverse rope frequencies, `rot_dim / 2` floats: `base ** (-2*pair/rot_dim)`,
 * built in double precision.  The kernels take the table rather than the base
 * because they scale each entry by the token position, where a one-ulp
 * difference in another engine's exp turns into a milliradian. */
void ds4_gpu_qwen4exp_rope_inv_freq(float *dst, uint32_t rot_dim, float freq_base);

/* Partial rope in place over the LEADING `rot_dim` entries of every head
 * vector, half-split NeoX.  Row `t` sits at position `pos0 + t`. */
int ds4_gpu_qwen4exp_rope_head_tensor(
        ds4_gpu_tensor       *x,
        const ds4_gpu_tensor *inv_freq,
        uint32_t              n_tokens,
        uint32_t              n_head,
        uint32_t              head_dim,
        uint32_t              rot_dim,
        uint32_t              pos0);

/* Append `raw_k` to the indexer's exact key tape and rebuild the blocks that
 * became complete.  A block is the fp32 mean of `pool_size` tape rows, then
 * `k_layernorm`, then partial rope at position `pool_size * block`. */
int ds4_gpu_qwen4exp_qsa_indexer_pool_update_tensor(
        ds4_gpu_tensor       *pool,
        ds4_gpu_tensor       *tape,
        const ds4_gpu_tensor *raw_k,
        const ds4_gpu_tensor *k_norm_weight,
        const ds4_gpu_tensor *inv_freq,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_cap,
        uint32_t              head_dim,
        uint32_t              pool_size,
        uint32_t              rot_dim,
        float                 eps,
        float                 weight_offset);

/* Block scores, `sum over index heads of relu(q . k) / sqrt(head_dim)`, with
 * `-INFINITY` where the block is not entirely in the query's past.  Feed the
 * result to ds4_gpu_indexer_topk_tensor, then to the selection below. */
int ds4_gpu_qwen4exp_qsa_indexer_scores_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *pool,
        uint32_t              n_tokens,
        uint32_t              n_blocks,
        uint32_t              n_head,
        uint32_t              head_dim,
        uint32_t              pos0,
        uint32_t              pool_size);

/* Expand the block top-k into an ASCENDING token id list per query: the
 * selected blocks' tokens followed by the tail of the query's own incomplete
 * block ("keep OR own").  `selected` is [n_tokens, max_selected] int32 padded
 * with -1, `counts` is [n_tokens] int32.  `max_selected` must be at least
 * top_k * pool_size + pool_size - 1.
 *
 * Ties: the incoming top-k order is whatever the descending bitonic argsort
 * produced, so this pass sorts the surviving ids ascending and the OUTPUT is
 * tie-order independent.  Which of several equal-scored blocks lands inside
 * the budget still depends on the sort. */
int ds4_gpu_qwen4exp_qsa_indexer_select_tensor(
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *counts,
        const ds4_gpu_tensor *scores,
        const ds4_gpu_tensor *topk,
        uint32_t              n_tokens,
        uint32_t              n_blocks,
        uint32_t              top_k,
        uint32_t              pos0,
        uint32_t              pool_size,
        uint32_t              max_selected);

/* Attention over the selected set with an f32 softmax.  `selected == NULL`
 * runs the dense causal set.  Caches are [cache_cap, n_kv_head, head_dim];
 * `q` and `out` are [n_tokens, n_head, head_dim]. */
int ds4_gpu_qwen4exp_qsa_attention_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *k_cache,
        const ds4_gpu_tensor *v_cache,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *counts,
        uint32_t              n_tokens,
        uint32_t              n_head,
        uint32_t              n_kv_head,
        uint32_t              head_dim,
        uint32_t              pos0,
        uint32_t              cache_cap,
        uint32_t              max_selected,
        float                 scale);

/* `out *= sigmoid(gate)`, the gate carried by the doubled `q_proj`. */
int ds4_gpu_qwen4exp_qsa_output_gate_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *gate,
        uint32_t              n_values);

int ds4_gpu_indexer_top1_value_tensor(
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *values,
        const ds4_gpu_tensor *scores,
        uint32_t              n_comp,
        uint32_t              n_tokens,
        uint32_t              index_offset);

int ds4_gpu_matmul_q8_0_top1_tensor(
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *values,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint64_t              in_dim,
        uint64_t              out_dim,
        const ds4_gpu_tensor *x,
        uint32_t              index_offset);

int ds4_gpu_set_decode_fast_attention(int enabled);
int ds4_gpu_set_decode_score_vec4(int enabled);

/* GPU argmax over n_vocab F32 logits. Writes the winning index as int32 at
 * out_idx[0]. Tie-break: lower index wins (matches host sample_argmax). */
int ds4_gpu_argmax_tensor(
        ds4_gpu_tensor       *out_idx,
        const ds4_gpu_tensor *logits,
        uint32_t                n_vocab);

int ds4_gpu_dsv4_topk_mask_tensor(
        ds4_gpu_tensor       *mask,
        const ds4_gpu_tensor *topk,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                top_k);

/* =========================================================================
 * Dense Projections, Norms, RoPE, and KV Rounding.
 * =========================================================================
 *
 * The graph uses these primitives for Q/KV projections, HC/output projections,
 * attention output projections, and DS4's tail-only RoPE.
 */

int ds4_gpu_matmul_q8_0_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

int ds4_gpu_matmul_q8_0_decode_mpp_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

int ds4_gpu_matmul_q8_0_decode_mpp_model_view_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

int ds4_gpu_matmul_q8_0_rows_scalar_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

int ds4_gpu_matmul_quant_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                weight_type,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

int ds4_gpu_matmul_quant_decode_mpp_model_view_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                weight_type,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

int ds4_gpu_matmul_quant_rows_scalar_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                weight_type,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

/* Optional fused GPU operations.
 *
 * These are acceleration hooks, not required backend primitives.  A backend
 * that does not provide the fused kernel must still define the symbol and
 * return 0.  Callers then use the portable sequence of required primitives.
 * Backends that return nonzero from a fused half-output operation must also
 * implement the matching half-input HC expansion helpers below.
 */
int ds4_gpu_matmul_q8_0_pair_tensor(
        ds4_gpu_tensor       *out0,
        ds4_gpu_tensor       *out1,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight0_offset,
        uint64_t                weight1_offset,
        uint64_t                in_dim,
        uint64_t                out0_dim,
        uint64_t                out1_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

int ds4_gpu_matmul_q4_K_pair_decode_tensor(
        ds4_gpu_tensor       *out0,
        ds4_gpu_tensor       *out1,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight0_offset,
        uint64_t              weight1_offset,
        uint64_t              in_dim,
        uint64_t              out_dim,
        const ds4_gpu_tensor *x);

/* Multi-row decode projections that preserve the one-row reduction order. */
int ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(
        ds4_gpu_tensor       *out,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint64_t              in_dim,
        uint64_t              out_dim,
        const ds4_gpu_tensor *x,
        uint32_t              n_rows);
/* The same projection over an input the CALLER has already quantized, in the
 * exact layout the internal quantize writes: `q_offset` bytes into `q` are
 * n_rows * (in_dim/32) Q8_0 blocks of 32 int8, and `s_offset` bytes in are the
 * matching n_rows * (in_dim/32) f32 block scales.  Both offsets must be
 * 16-byte aligned.
 *
 * Same weights, same bounds and the SAME launch ladder as the entry above --
 * it is one function with the quantize step lifted out -- so a caller that
 * produced xq/xscale some other way lands on identical arithmetic.  The fused
 * hyper-connection norm is that caller: it already holds the normalized value
 * in a register, and writing it to DRAM only for a quantize kernel to read it
 * straight back was 95.6 MB of round trip per mixer call at a 1024-row chunk. */
int ds4_gpu_matmul_q8_0_preq_rows_exact_tensor(
        ds4_gpu_tensor       *out,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint64_t              in_dim,
        uint64_t              out_dim,
        const ds4_gpu_tensor *q,
        uint64_t              q_offset,
        uint64_t              s_offset,
        uint32_t              n_rows);
/* Quantize an f32 input tensor into Q8_0 blocks and f32 scales in the exact
 * layout expected by ds4_gpu_matmul_q8_0_preq_rows_exact_tensor. */
int ds4_gpu_quantize_q8_0_decode_rows_exact_tensor(
        ds4_gpu_tensor       *q,
        uint64_t              q_offset,
        uint64_t              s_offset,
        const ds4_gpu_tensor *x,
        uint64_t              in_dim,
        uint32_t              n_rows);
int ds4_gpu_matmul_f32_decode_rows_exact_tensor(
        ds4_gpu_tensor       *out,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint64_t              in_dim,
        uint64_t              out_dim,
        const ds4_gpu_tensor *x,
        uint32_t              n_rows);
int ds4_gpu_matmul_q8_0_pair_decode_rows_exact_tensor(
        ds4_gpu_tensor       *out0,
        ds4_gpu_tensor       *out1,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight0_offset,
        uint64_t              weight1_offset,
        uint64_t              in_dim,
        uint64_t              out0_dim,
        uint64_t              out1_dim,
        const ds4_gpu_tensor *x,
        uint32_t              n_rows);

int ds4_gpu_matmul_q8_0_f16_out_tensor(
        ds4_gpu_tensor       *out_h,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

int ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        float                   clamp);

int ds4_gpu_router_shared_gate_up_q8_0_tensor(
        ds4_gpu_tensor       *router_logits,
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              router_weight_offset,
        uint64_t              gate_offset,
        uint64_t              up_offset,
        uint64_t              in_dim,
        uint64_t              router_out_dim,
        uint64_t              out_dim,
        const ds4_gpu_tensor *x,
        float                 clamp,
        bool                  router_only);
#ifdef __APPLE__
int ds4_gpu_router_project_select_fused_tensor(
        ds4_gpu_tensor       *router_logits,
        ds4_gpu_tensor       *probs,
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *weights,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              router_weight_offset,
        uint64_t              bias_offset,
        bool                  has_bias,
        const ds4_gpu_tensor *x);
#endif
int ds4_gpu_shared_mid_swiglu_q8_0_decode_exact_tensor(
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        float                   clamp,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *prequant,
        uint32_t                expert_split,
        bool                    home_rank);

int ds4_gpu_shared_mid_swiglu_q8_0_tensor(
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        float                   clamp);

int ds4_gpu_shared_gate_up_swiglu_q8_0_model_view_tensor(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        float                   clamp);

int ds4_gpu_shared_gate_up_swiglu_q8_0_rows_tensor(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok,
        float                   clamp);

int ds4_gpu_shared_gate_up_swiglu_q8_0_rows_scalar_tensor(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok,
        float                   clamp);

int ds4_gpu_matmul_f16_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

/* CUDA batch path: fold an input RMS normalization into the FP16 activation
 * conversion used by the following projection. Returns 0 without touching
 * out when the optimized path is unavailable. */
int ds4_gpu_matmul_f16_rms_fold_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok,
        float                   norm_eps);

/* Exact multi-row form of the DeepSeek 4096x256 F16 router projection. */
int ds4_gpu_matmul_f16_router_rows_exact_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        const ds4_gpu_tensor *x,
        uint32_t                n_rows);

int ds4_gpu_matmul_f16_pair_tensor(
        ds4_gpu_tensor       *out_a,
        ds4_gpu_tensor       *out_b,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_a_offset,
        uint64_t                weight_b_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

/* Optional Metal decode fusion. Returns 1 when the paired projection and
 * recurrent compressor-state store were encoded, 0 when the optimized path
 * is unavailable, and -1 on an attempted-path error. */
int ds4_gpu_matmul_f16_pair_compressor_store_tensor(
        ds4_gpu_tensor       *out_kv,
        ds4_gpu_tensor       *out_score,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_kv_offset,
        uint64_t                weight_score_offset,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                in_dim,
        uint32_t                width,
        const ds4_gpu_tensor *x,
        uint32_t                ratio,
        uint32_t                pos);

int ds4_gpu_matmul_f16_quad_compressor_store_tensor(
        ds4_gpu_tensor       *out0_kv,
        ds4_gpu_tensor       *out0_score,
        ds4_gpu_tensor       *out1_kv,
        ds4_gpu_tensor       *out1_score,
        ds4_gpu_tensor       *state0_kv,
        ds4_gpu_tensor       *state0_score,
        ds4_gpu_tensor       *state1_kv,
        ds4_gpu_tensor       *state1_score,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight0_kv_offset,
        uint64_t              weight0_score_offset,
        uint64_t              weight1_kv_offset,
        uint64_t              weight1_score_offset,
        uint64_t              ape0_offset,
        uint32_t              ape0_type,
        uint64_t              ape1_offset,
        uint32_t              ape1_type,
        uint64_t              in_dim,
        uint32_t              width0,
        uint32_t              width1,
        const ds4_gpu_tensor *x,
        uint32_t              ratio,
        uint32_t              pos);

/* Decode-only M5 fusion: emit-path compressor row finalize (norm + rope +
 * fp8/commit + indexer qat) in one dispatch.  Bit-exact vs the separate
 * dispatches.  Returns 1 when fused, 0 to fall back. */
int ds4_gpu_dsv4_comp_row_finalize_tensor(
        ds4_gpu_tensor       *attn_stage,
        ds4_gpu_tensor       *attn_cache,
        uint32_t              attn_comp_row,
        uint64_t              attn_norm_offset,
        ds4_gpu_tensor       *index_cache,
        uint32_t              index_comp_row,
        uint64_t              index_norm_offset,
        ds4_gpu_tensor       *attn_state_kv,
        ds4_gpu_tensor       *attn_state_score,
        ds4_gpu_tensor       *index_state_kv,
        ds4_gpu_tensor       *index_state_score,
        const void           *model_map,
        uint64_t              model_size,
        uint32_t              pos,
        uint32_t              n_rot,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow,
        float                 rms_eps);

/* Decode-only M5 fusion: q_a/kv Q8 pair projection + F16 quad compressor
 * projection/store in one dispatch.  Bit-exact vs the separate dispatches.
 * Returns 1 when fused, 0 to fall back, -1 on error. */
int ds4_gpu_qkv_pair_quad_compressor_store_tensor(
        ds4_gpu_tensor       *qr,
        ds4_gpu_tensor       *kv_raw,
        ds4_gpu_tensor       *out0_kv,
        ds4_gpu_tensor       *out0_score,
        ds4_gpu_tensor       *out1_kv,
        ds4_gpu_tensor       *out1_score,
        ds4_gpu_tensor       *state0_kv,
        ds4_gpu_tensor       *state0_score,
        ds4_gpu_tensor       *state1_kv,
        ds4_gpu_tensor       *state1_score,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              q_a_offset,
        uint64_t              kv_offset,
        uint64_t              weight0_kv_offset,
        uint64_t              weight0_score_offset,
        uint64_t              weight1_kv_offset,
        uint64_t              weight1_score_offset,
        uint64_t              ape0_offset,
        uint32_t              ape0_type,
        uint64_t              ape1_offset,
        uint32_t              ape1_type,
        uint32_t              in_dim,
        uint32_t              q_rank,
        uint32_t              kv_dim,
        uint32_t              width0,
        uint32_t              width1,
        const ds4_gpu_tensor *x,
        uint32_t              ratio,
        uint32_t              pos);

int ds4_gpu_matmul_f32_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

int ds4_gpu_repeat_hc_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *row,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ds4_gpu_repeat_hc_rows_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *rows,
        uint32_t                n_tokens,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ds4_gpu_rms_norm_plain_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        uint32_t                n,
        float                   eps);

int ds4_gpu_rms_norm_plain_rows_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        uint32_t                n,
        uint32_t                rows,
        float                   eps);

int ds4_gpu_rms_norm_weight_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n,
        float                   eps);

int ds4_gpu_rms_norm_weight_rows_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n,
        uint32_t                rows,
        float                   eps);

int ds4_gpu_add_rms_norm_weight_tensor(
        ds4_gpu_tensor       *norm_out,
        ds4_gpu_tensor       *sum_out,
        const ds4_gpu_tensor *a,
        const ds4_gpu_tensor *b,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n,
        float                   eps);

int ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(
        ds4_gpu_tensor       *q_out,
        const ds4_gpu_tensor *q,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                q_weight_offset,
        uint32_t                q_n,
        ds4_gpu_tensor       *kv_out,
        const ds4_gpu_tensor *kv,
        uint64_t                kv_weight_offset,
        uint32_t                kv_n,
        uint32_t                rows,
        float                   eps);

int ds4_gpu_dsv4_qkv_rms_norm_kv_rope_fp8_store_tensor(
        ds4_gpu_tensor       *q_out,
        const ds4_gpu_tensor *q,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              q_weight_offset,
        uint32_t              q_n,
        ds4_gpu_tensor       *kv_out,
        const ds4_gpu_tensor *kv,
        uint64_t              kv_weight_offset,
        uint32_t              kv_n,
        ds4_gpu_tensor       *raw_cache,
        uint64_t              raw_cap,
        uint32_t              raw_row,
        uint32_t              n_rot,
        uint32_t              pos0,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow,
        float                 eps);

int ds4_gpu_dsv4_qkv_rms_norm_rows_kv_rope_tensor(
        ds4_gpu_tensor       *q_out,
        const ds4_gpu_tensor *q,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                q_weight_offset,
        uint32_t                q_n,
        ds4_gpu_tensor       *kv_out,
        const ds4_gpu_tensor *kv,
        uint64_t                kv_weight_offset,
        uint32_t                kv_n,
        uint32_t                rows,
        uint32_t                kv_n_head,
        uint32_t                kv_head_dim,
        uint32_t                n_rot,
        uint32_t                pos0,
        uint32_t                n_ctx_orig,
        bool                    inverse,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   eps);

int ds4_gpu_head_rms_norm_tensor(
        ds4_gpu_tensor *x,
        uint32_t          n_tok,
        uint32_t          n_head,
        uint32_t          head_dim,
        float             eps);

int ds4_gpu_head_rms_norm_rope_tail_tensor(
        ds4_gpu_tensor *x,
        uint32_t          n_tok,
        uint32_t          n_head,
        uint32_t          head_dim,
        uint32_t          n_rot,
        uint32_t          pos0,
        uint32_t          n_ctx_orig,
        bool              inverse,
        float             freq_base,
        float             freq_scale,
        float             ext_factor,
        float             attn_factor,
        float             beta_fast,
        float             beta_slow,
        float             eps);

int ds4_gpu_attn_q_b_f16_head_rms_rope_tail_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *q_half,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint64_t              in_dim,
        uint64_t              out_dim,
        const ds4_gpu_tensor *x,
        uint32_t              n_tok,
        uint32_t              n_head,
        uint32_t              head_dim,
        uint32_t              n_rot,
        uint32_t              pos0,
        uint32_t              n_ctx_orig,
        bool                  inverse,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow,
        float                 eps);

int ds4_gpu_dsv4_fp8_kv_quantize_tensor(
        ds4_gpu_tensor *x,
        uint32_t          n_tok,
        uint32_t          head_dim,
        uint32_t          n_rot);

int ds4_gpu_dsv4_indexer_qat_tensor(
        ds4_gpu_tensor *x,
        uint32_t          n_rows,
        uint32_t          head_dim);



int ds4_gpu_rope_tail_tensor(
        ds4_gpu_tensor *x,
        uint32_t          n_tok,
        uint32_t          n_head,
        uint32_t          head_dim,
        uint32_t          n_rot,
        uint32_t          pos0,
        uint32_t          n_ctx_orig,
        bool              inverse,
        float             freq_base,
        float             freq_scale,
        float             ext_factor,
        float             attn_factor,
        float             beta_fast,
        float             beta_slow);

int ds4_gpu_glm_rope_tail_tensor(
        ds4_gpu_tensor *x,
        uint32_t        n_tokens,
        uint32_t        n_head,
        uint32_t        head_dim,
        uint32_t        rot_dim,
        uint32_t        pos0,
        uint32_t        n_ctx_orig,
        float           freq_base,
        float           freq_scale,
        float           ext_factor,
        float           attn_factor,
        float           beta_fast,
        float           beta_slow);

int ds4_gpu_glm_kv_lora_rms_norm_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *kv_raw,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              n_tokens,
        uint32_t              kv_raw_dim,
        uint32_t              kv_lora_dim,
        float                 eps);

int ds4_gpu_glm_k_b_project_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *kv_norm,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              n_tokens,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              n_head);

int ds4_gpu_glm_k_b_project_typed_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *kv_norm,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              weight_type,
        uint32_t              n_tokens,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              n_head);

int ds4_gpu_glm_store_compact_kv_tensor(
        ds4_gpu_tensor       *kv_lora_cache,
        ds4_gpu_tensor       *k_rope_cache,
        const ds4_gpu_tensor *kv_norm,
        const ds4_gpu_tensor *kv_raw,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_cap,
        uint32_t              kv_raw_dim,
        uint32_t              kv_lora_dim,
        uint32_t              qk_rope,
        bool                  cache_f16);

int ds4_gpu_glm_qkv_norm_store_compact_kv_tensor(
        ds4_gpu_tensor       *q_out,
        const ds4_gpu_tensor *q,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              q_weight_offset,
        uint32_t              q_n,
        ds4_gpu_tensor       *kv_lora_cache,
        ds4_gpu_tensor       *k_rope_cache,
        const ds4_gpu_tensor *kv_raw,
        uint64_t              kv_weight_offset,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_cap,
        uint32_t              kv_raw_dim,
        uint32_t              kv_lora_dim,
        uint32_t              qk_rope,
        bool                  cache_f16,
        float                 eps);

int ds4_gpu_glm_store_indexer_k_tensor(
        ds4_gpu_tensor       *indexer_key_cache,
        const ds4_gpu_tensor *raw_k,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint64_t              bias_offset,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_cap,
        uint32_t              head_dim,
        uint32_t              rot_dim,
        uint32_t              n_ctx_orig,
        float                 eps,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow,
        bool                  cache_f16);

/* GLM-5.3 pools four normalized indexer keys with a learned, per-channel
 * softmax. Partial pools are retained in tail_k/tail_gate across calls. */
int ds4_gpu_glm53_indexer_pool_update_tensor(
        ds4_gpu_tensor       *pool_cache,
        ds4_gpu_tensor       *tail_k,
        ds4_gpu_tensor       *tail_gate,
        const ds4_gpu_tensor *raw_k,
        const ds4_gpu_tensor *gate,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              norm_weight_offset,
        uint64_t              norm_bias_offset,
        uint64_t              ape_offset,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_cap,
        uint32_t              head_dim,
        uint32_t              pool_size,
        float                 eps,
        bool                  cache_f16);

int ds4_gpu_glm53_expand_pool_selection_tensor(
        ds4_gpu_tensor       *raw_selected,
        const ds4_gpu_tensor *pool_selected,
        uint32_t              n_tokens,
        uint32_t              pos0,
        uint32_t              selected_pools,
        uint32_t              index_topk,
        uint32_t              pool_size,
        uint32_t              output_width);

int ds4_gpu_glm_build_kv_cache_tensor(
        ds4_gpu_tensor       *key_cache,
        ds4_gpu_tensor       *value_cache,
        const ds4_gpu_tensor *kv_raw,
        const ds4_gpu_tensor *k_nope,
        const ds4_gpu_tensor *value,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_cap,
        uint32_t              n_head,
        uint32_t              kv_raw_dim,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              value_dim,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow,
        bool                  cache_f16);

int ds4_gpu_glm_build_kv_cache_flash_tensor(
        ds4_gpu_tensor       *key_cache,
        ds4_gpu_tensor       *value_cache,
        const ds4_gpu_tensor *kv_raw,
        const ds4_gpu_tensor *k_nope,
        const ds4_gpu_tensor *value,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_cap,
        uint32_t              n_head,
        uint32_t              kv_raw_dim,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              value_dim,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow,
        bool                  cache_f16);

int ds4_gpu_glm_attention_full_tensor(
        ds4_gpu_tensor       *heads,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *key_cache,
        const ds4_gpu_tensor *value_cache,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_len,
        uint32_t              cache_cap,
        uint32_t              n_head,
        uint32_t              qk_dim,
        uint32_t              value_dim,
        bool                  cache_f16);

int ds4_gpu_glm_fill_selected_range_tensor(
        ds4_gpu_tensor *selected,
        uint32_t        n_selected);

int ds4_gpu_glm_fill_selected_range_batch_tensor(
        ds4_gpu_tensor *selected,
        uint32_t        n_tokens,
        uint32_t        pos0,
        uint32_t        n_selected,
        uint32_t        pad_row);

int ds4_gpu_glm_indexer_rope_tail_tensor(
        ds4_gpu_tensor *x,
        uint32_t        n_tokens,
        uint32_t        n_head,
        uint32_t        head_dim,
        uint32_t        rot_dim,
        uint32_t        pos0,
        uint32_t        n_ctx_orig,
        float           freq_base,
        float           freq_scale,
        float           ext_factor,
        float           attn_factor,
        float           beta_fast,
        float           beta_slow);

int ds4_gpu_glm_indexer_score_one_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *indexer_key_cache,
        uint32_t              n_rows,
        uint32_t              n_head,
        uint32_t              head_dim,
        float                 scale,
        bool                  cache_f16);

int ds4_gpu_glm_indexer_scores_batch_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *indexer_key_cache,
        uint32_t              n_rows,
        uint32_t              n_tokens,
        uint32_t              pos0,
        uint32_t              n_head,
        uint32_t              head_dim,
        float                 scale,
        bool                  cache_f16);

int ds4_gpu_glm53_indexer_scores_batch_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *indexer_key_cache,
        uint32_t              n_rows,
        uint32_t              n_tokens,
        uint32_t              pos0,
        uint32_t              pool_size,
        uint32_t              n_head,
        uint32_t              head_dim,
        float                 scale,
        bool                  cache_f16);

int ds4_gpu_glm_qk_lowrank_q8_0_tensor(
        ds4_gpu_tensor       *qk_low,
        const ds4_gpu_tensor *q,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_dim);

int ds4_gpu_glm_qk_lowrank_q8_0_batch_tensor(
        ds4_gpu_tensor       *qk_low,
        const ds4_gpu_tensor *q,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              n_tokens,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_dim);

int ds4_gpu_glm_qk_lowrank_typed_tensor(
        ds4_gpu_tensor       *qk_low,
        const ds4_gpu_tensor *q,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              weight_type,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_dim);

int ds4_gpu_glm_qk_lowrank_typed_batch_tensor(
        ds4_gpu_tensor       *qk_low,
        const ds4_gpu_tensor *q,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              weight_type,
        uint32_t              n_tokens,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_dim);

int ds4_gpu_glm_value_project_q8_0_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const ds4_gpu_tensor *lora,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              n_tokens,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              value_dim);

int ds4_gpu_glm_value_project_typed_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const ds4_gpu_tensor *lora,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              weight_type,
        uint32_t              n_tokens,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              value_dim);

int ds4_gpu_glm_attention_indexed_decode_tensor(
        ds4_gpu_tensor       *heads,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              value_weight_offset,
        const ds4_gpu_tensor *selected,
        uint32_t              n_selected,
        uint32_t              cache_cap,
        bool                  cache_f16,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              value_dim,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow);

int ds4_gpu_rope_tail_decode_rows_tensor(
        ds4_gpu_tensor                     *x,
        const ds4_gpu_attention_decode_row *rows,
        uint32_t                            n_rows,
        uint32_t                            n_head,
        uint32_t                            head_dim,
        uint32_t                            n_rot,
        uint32_t                            n_ctx_orig,
        bool                                inverse,
        float                               freq_base,
        float                               freq_scale,
        float                               ext_factor,
        float                               attn_factor,
        float                               beta_fast,
        float                               beta_slow);

int ds4_gpu_glm_attention_indexed_decode_typed_tensor(
        ds4_gpu_tensor       *heads,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              value_weight_offset,
        uint32_t              value_weight_type,
        const ds4_gpu_tensor *selected,
        uint32_t              n_selected,
        uint32_t              cache_cap,
        bool                  cache_f16,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              value_dim,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow);

int ds4_gpu_glm_attention_indexed_decode_split_group8_tensor(
        ds4_gpu_tensor       *heads,
        ds4_gpu_tensor       *partial_lora,
        ds4_gpu_tensor       *partial_ms,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              value_weight_offset,
        const ds4_gpu_tensor *selected,
        uint32_t              n_selected,
        bool                  selected_rows_valid,
        uint32_t              cache_cap,
        bool                  cache_f16,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              value_dim,
        uint32_t              n_ctx_orig,
        uint32_t              block_rows,
        uint32_t              n_blocks,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow);

int ds4_gpu_glm_attention_indexed_decode_split_group8_typed_tensor(
        ds4_gpu_tensor       *heads,
        ds4_gpu_tensor       *partial_lora,
        ds4_gpu_tensor       *partial_ms,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              value_weight_offset,
        uint32_t              value_weight_type,
        const ds4_gpu_tensor *selected,
        uint32_t              n_selected,
        bool                  selected_rows_valid,
        uint32_t              cache_cap,
        bool                  cache_f16,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              value_dim,
        uint32_t              n_ctx_orig,
        uint32_t              block_rows,
        uint32_t              n_blocks,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow);

int ds4_gpu_glm_attention_indexed_batch_tensor(
        ds4_gpu_tensor       *heads,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              value_weight_offset,
        const ds4_gpu_tensor *selected,
        uint32_t              n_tokens,
        uint32_t              n_selected,
        uint32_t              cache_cap,
        bool                  cache_f16,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              value_dim,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow);

int ds4_gpu_glm_attention_indexed_batch_typed_tensor(
        ds4_gpu_tensor       *heads,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              value_weight_offset,
        uint32_t              value_weight_type,
        const ds4_gpu_tensor *selected,
        uint32_t              n_tokens,
        uint32_t              n_selected,
        uint32_t              cache_cap,
        bool                  cache_f16,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              value_dim,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow);

int ds4_gpu_sort_i32_rows_asc_tensor(
        ds4_gpu_tensor       *dst,
        const ds4_gpu_tensor *src,
        uint32_t              row_width,
        uint32_t              n_rows);

int ds4_gpu_glm_attention_indexed_batch_lora_tensor(
        ds4_gpu_tensor       *lora_out,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        const ds4_gpu_tensor *selected,
        uint32_t              n_tokens,
        uint32_t              n_selected,
        uint32_t              cache_cap,
        bool                  cache_f16,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow);

int ds4_gpu_glm_attention_indexed_batch_lora_causal_tensor(
        ds4_gpu_tensor       *lora_out,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        uint32_t              n_tokens,
        uint32_t              pos0,
        uint32_t              n_selected,
        uint32_t              cache_cap,
        bool                  cache_f16,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow);

/* Dense causal MLA over the shared compact latent cache. qk_low and lora_out
 * are [token, head, kv_lora_dim]; the F16 cache is shared by all heads. */
int ds4_gpu_glm_attention_dense_compact_lora_causal_tensor(
        ds4_gpu_tensor       *lora_out,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        uint32_t              q_row0,
        uint32_t              n_q,
        uint32_t              n_kv,
        uint32_t              cache_cap,
        bool                  cache_f16,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_dim);

int ds4_gpu_glm_attention_indexed_batch_lora_valid_tensor(
        ds4_gpu_tensor       *lora_out,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        const ds4_gpu_tensor *selected,
        uint32_t              n_tokens,
        uint32_t              n_selected,
        uint32_t              cache_cap,
        bool                  cache_f16,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow);

int ds4_gpu_glm_attention_flash_staged_tensor(
        ds4_gpu_tensor       *heads,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *key_cache,
        const ds4_gpu_tensor *value_cache,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_len,
        uint32_t              cache_cap,
        uint32_t              n_head,
        uint32_t              qk_dim,
        uint32_t              value_dim,
        bool                  cache_f16);

int ds4_gpu_glm_attention_flash_tensor(
        ds4_gpu_tensor       *heads,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *key_cache,
        const ds4_gpu_tensor *value_cache,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_len,
        uint32_t              cache_cap,
        uint32_t              n_head,
        uint32_t              qk_dim,
        uint32_t              value_dim,
        bool                  cache_f16);

/* Release decode fused KV finalizer: after the standalone RoPE kernel, this
 * performs DS4's FP8 non-RoPE KV round trip and writes the F16-rounded raw
 * attention cache row in one dispatch. */
int ds4_gpu_kv_fp8_store_raw_tensor(
        ds4_gpu_tensor *kv,
        ds4_gpu_tensor *raw_cache,
        uint32_t          raw_cap,
        uint32_t          row,
        uint32_t          head_dim,
        uint32_t          n_rot);

/* Exact multi-session form of the decode KV finalizer. KV rows are
 * contiguous, while each output row is written to its session-private cache. */
int ds4_gpu_kv_fp8_store_raw_decode_rows_tensor(
        ds4_gpu_tensor        *kv,
        ds4_gpu_tensor *const *raw_caches,
        const uint32_t        *raw_caps,
        const uint32_t        *raw_rows,
        uint32_t               n_rows,
        uint32_t               head_dim,
        uint32_t               n_rot);

/* Reference/raw-cache primitive kept for prefill and diagnostics.  Decode uses
 * ds4_gpu_kv_fp8_store_raw_tensor unless a diagnostic reference path is
 * explicitly selected by the graph driver. */
int ds4_gpu_store_raw_kv_tensor(
        ds4_gpu_tensor       *raw_cache,
        const ds4_gpu_tensor *kv,
        uint32_t                raw_cap,
        uint32_t                row,
        uint32_t                head_dim);

int ds4_gpu_store_raw_kv_batch_tensor(
        ds4_gpu_tensor       *raw_cache,
        const ds4_gpu_tensor *kv,
        uint32_t                raw_cap,
        uint32_t                pos0,
        uint32_t                n_tokens,
        uint32_t                head_dim);

/* =========================================================================
 * KV Compression and Attention.
 * =========================================================================
 *
 * Compressed layers maintain rolling score/KV state and append pooled rows at
 * ratio boundaries.  Attention kernels consume raw SWA rows, compressed rows,
 * and optional indexer masks.
 */

int ds4_gpu_compressor_update_tensor(
        const ds4_gpu_tensor *kv_cur,
        const ds4_gpu_tensor *sc_cur,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        ds4_gpu_tensor       *comp_cache,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos,
        uint32_t                comp_row,
        uint32_t                n_rot,
        uint32_t                n_ctx_orig,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   rms_eps,
        bool                    state_already_stored,
        bool                    decode_one_token,
        bool                    defer_finalize);

int ds4_gpu_compressor_store_batch_tensor(
        const ds4_gpu_tensor *kv,
        const ds4_gpu_tensor *sc,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos0,
        uint32_t                n_tokens);

int ds4_gpu_compressor_prefill_tensor(
        ds4_gpu_tensor       *comp_cache,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const ds4_gpu_tensor *kv,
        const ds4_gpu_tensor *sc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos0,
        uint32_t                n_tokens,
        uint32_t                n_rot,
        uint32_t                n_ctx_orig,
        bool                    quantize_fp8,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   rms_eps);

int ds4_gpu_compressor_prefill_ratio4_replay_tensor(
        ds4_gpu_tensor       *comp_cache,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const ds4_gpu_tensor *kv,
        const ds4_gpu_tensor *sc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                pos0,
        uint32_t                n_tokens,
        uint32_t                n_rot,
        uint32_t                n_ctx_orig,
        bool                    quantize_fp8,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   rms_eps);

int ds4_gpu_compressor_prefill_state_ratio4_tensor(
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const ds4_gpu_tensor *kv_tail,
        const ds4_gpu_tensor *sc_tail,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint32_t                head_dim,
        uint32_t                pos0);

int ds4_gpu_attention_decode_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        uint32_t                n_comp,
        const ds4_gpu_tensor *comp_mask,
        uint32_t                use_mask,
        uint32_t                n_head,
        uint32_t                head_dim);

int ds4_gpu_attention_decode_heads_rope_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        uint32_t                n_comp,
        const ds4_gpu_tensor *comp_mask,
        uint32_t                use_mask,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                n_rot,
        uint32_t                pos0,
        uint32_t                n_ctx_orig,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        int                    *fused_inv_rope);

/* Multi-session decode over contiguous Q/head rows and private KV caches.
 * The row table is copied into CUDA launch parameters, so no device-side
 * descriptor upload or synchronization is required. */
int ds4_gpu_attention_decode_rows_rope_tensor(
        ds4_gpu_tensor                       *heads,
        const void                           *model_map,
        uint64_t                              model_size,
        uint64_t                              sinks_offset,
        const ds4_gpu_tensor                 *q,
        const ds4_gpu_attention_decode_row   *rows,
        uint32_t                              n_rows,
        uint32_t                              n_head,
        uint32_t                              head_dim,
        uint32_t                              n_rot,
        uint32_t                              n_ctx_orig,
        float                                 freq_base,
        float                                 freq_scale,
        float                                 ext_factor,
        float                                 attn_factor,
        float                                 beta_fast,
        float                                 beta_slow);
/* Diagnostic/public form of the dk=512 gathered decode-attention KV staging
 * step. The compressed source must be F16; dst writes chronological raw-ring
 * rows followed by compressed rows and must not overlap either source. */
int ds4_gpu_flash_kv_stage_f16_tensor(
        ds4_gpu_tensor       *dst,
        const ds4_gpu_tensor *raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                n_raw,
        const ds4_gpu_tensor *comp,
        uint32_t                comp_is_f16,
        uint32_t                n_comp,
        uint32_t                head_dim);

int ds4_gpu_attention_prefill_raw_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        uint32_t                n_tokens,
        uint32_t                window,
        uint32_t                n_head,
        uint32_t                head_dim);

/* Rectangular raw prefill attention: q is a view of the n_q query rows at
 * token positions [q_row0, q_row0 + n_q) of the chunk, raw_kv keeps all
 * n_kv rows, heads receives n_q output rows.  Used by the TP prefill row
 * split; the square entry above is the q_row0 = 0, n_q = n_kv case. */
int ds4_gpu_attention_prefill_raw_heads_range_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        uint32_t                q_row0,
        uint32_t                n_q,
        uint32_t                n_kv,
        uint32_t                window,
        uint32_t                n_head,
        uint32_t                head_dim);

int ds4_gpu_attention_decode_raw_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                window,
        uint32_t                n_head,
        uint32_t                head_dim);

int ds4_gpu_attention_noncausal_raw_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        uint32_t                n_tokens,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                n_head,
        uint32_t                head_dim);

int ds4_gpu_attention_decode_mixed_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        const ds4_gpu_tensor *comp_mask,
        uint32_t                use_comp_mask,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim);

int ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        const ds4_gpu_tensor *topk,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                n_comp,
        uint32_t                top_k,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim);

int ds4_gpu_attention_prefill_static_mixed_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        uint32_t                n_tokens,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim);

/* Rectangular static-mixed prefill attention: q is a view of the n_q query
 * rows at token positions [q_row0, q_row0 + n_q) of the chunk, while raw_kv
 * keeps all n_tokens rows and comp_kv all n_comp compressed keys.  Used by
 * the TP prefill row split; the square entry above is q_row0 = 0,
 * n_q = n_tokens. */
int ds4_gpu_attention_prefill_static_mixed_heads_range_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        uint32_t                q_row0,
        uint32_t                n_q,
        uint32_t                n_tokens,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim);

int ds4_gpu_attention_prefill_masked_mixed_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        const ds4_gpu_tensor *comp_mask,
        uint32_t                n_tokens,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim);

/* DeepSeek Vision-Exp attention over the current prefill chunk. The raw cache
 * is chronological from raw_start and may include the preceding SWA rows.
 * Synthetic image spans in tokens are made bidirectional as specified by the
 * checkpoint; text and compressed keys retain the normal causal masks. */
int ds4_gpu_attention_visual_mixed_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        const ds4_gpu_tensor *comp_mask,
        uint32_t                use_comp_mask,
        const int32_t          *tokens,
        uint32_t                vocab_size,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim);

int ds4_gpu_attention_output_q8_batch_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *low,
        ds4_gpu_tensor       *group_tmp,
        ds4_gpu_tensor       *low_tmp,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                out_b_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        uint64_t                out_dim,
        const ds4_gpu_tensor *heads,
        uint32_t                n_tokens);
int ds4_gpu_attention_output_q4_K_batch_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *low,
        ds4_gpu_tensor       *group_tmp,
        ds4_gpu_tensor       *low_tmp,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                out_b_offset,
        uint32_t                out_b_type,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        uint64_t                out_dim,
        const ds4_gpu_tensor *heads,
        uint32_t                n_tokens);

int ds4_gpu_attention_output_q8_batch_f16_tensor(
        ds4_gpu_tensor       *out_h,
        ds4_gpu_tensor       *low,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                out_b_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        uint64_t                out_dim,
        const ds4_gpu_tensor *heads,
        uint32_t                n_tokens);

int ds4_gpu_attention_output_low_q8_tensor(
        ds4_gpu_tensor       *low,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        const ds4_gpu_tensor *heads);
int ds4_gpu_attention_output_low_q4_K_slice_tensor(
        ds4_gpu_tensor       *low,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                group0,
        uint32_t                group_cnt,
        const ds4_gpu_tensor *heads);

int ds4_gpu_attention_output_low_q8_rows_exact_tensor(
        ds4_gpu_tensor       *low,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups_total,
        uint32_t                group0,
        uint32_t                group_cnt,
        const ds4_gpu_tensor *heads,
        uint32_t                n_rows);

int ds4_gpu_attention_output_q8_tp_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *low,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                out_b_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups_total,
        uint32_t                group0,
        uint32_t                group_cnt,
        uint64_t                out_dim,
        const ds4_gpu_tensor *heads);

/* =========================================================================
 * Router, Shared Expert, and Routed MoE.
 * =========================================================================
 *
 * These kernels implement the FFN body: router probabilities/top-k or hash
 * routing, shared SwiGLU, and the IQ2_XXS/Q2_K/Q4_K routed experts.
 */

int ds4_gpu_swiglu_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *gate,
        const ds4_gpu_tensor *up,
        uint32_t                n,
        float                   clamp,
        float                   weight);

int ds4_gpu_add_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *a,
        const ds4_gpu_tensor *b,
        uint32_t                n);

int ds4_gpu_add3_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *a,
        const ds4_gpu_tensor *b,
        const ds4_gpu_tensor *c,
        uint32_t                n);

int ds4_gpu_directional_steering_project_tensor(
        ds4_gpu_tensor       *x,
        const ds4_gpu_tensor *directions,
        uint32_t                layer,
        uint32_t                width,
        uint32_t                rows,
        float                   scale);

int ds4_gpu_router_select_tensor(
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *weights,
        ds4_gpu_tensor       *probs,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                bias_offset,
        uint64_t                hash_offset,
        uint32_t                hash_rows,
        uint32_t                token,
        uint32_t                n_expert,
        uint32_t                n_expert_used,
        float                   expert_weight_scale,
        uint32_t                n_expert_groups,
        uint32_t                n_group_used,
        bool                    has_bias,
        bool                    hash_mode,
        const ds4_gpu_tensor *logits);

int ds4_gpu_router_select_batch_tensor(
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *weights,
        ds4_gpu_tensor       *probs,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                bias_offset,
        uint64_t                hash_offset,
        uint32_t                hash_rows,
        uint32_t                n_expert_groups,
        uint32_t                n_group_used,
        bool                    has_bias,
        bool                    hash_mode,
        const ds4_gpu_tensor *logits,
        const ds4_gpu_tensor *tokens,
        uint32_t                n_expert,
        uint32_t                n_expert_used,
        float                   expert_weight_scale,
        uint32_t                n_tokens);

/* DeepSeek Vision-Exp prefill may mix ordinary vocabulary IDs and synthetic
 * image IDs in one batch. Text rows keep the normal/hash route; image rows use
 * the checkpoint's visual selection bias. Routing weights always come from
 * the original, unbiased scores. */
int ds4_gpu_router_select_batch_visual_tensor(
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *weights,
        ds4_gpu_tensor       *probs,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                bias_offset,
        uint64_t                hash_offset,
        uint32_t                hash_rows,
        bool                    has_bias,
        bool                    hash_mode,
        const void             *vision_map,
        uint64_t                vision_size,
        uint64_t                visual_bias_offset,
        const ds4_gpu_tensor *logits,
        const ds4_gpu_tensor *tokens,
        uint32_t                vocab_size,
        uint32_t                n_expert,
        uint32_t                n_expert_used,
        float                   expert_weight_scale,
        uint32_t                n_tokens);

int ds4_gpu_glm_router_select_tensor(
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *weights,
        ds4_gpu_tensor       *probs,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                bias_offset,
        const ds4_gpu_tensor *logits,
        uint32_t                n_expert,
        uint32_t                n_expert_used,
        float                   expert_weight_scale);

int ds4_gpu_glm_router_select_batch_tensor(
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *weights,
        ds4_gpu_tensor       *probs,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                bias_offset,
        const ds4_gpu_tensor *logits,
        uint32_t                n_expert,
        uint32_t                n_expert_used,
        float                   expert_weight_scale,
        uint32_t                n_tokens);

int ds4_gpu_glm_routed_moe_one_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint32_t                gate_type,
        uint32_t                up_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                up_expert_bytes,
        uint64_t                up_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t                n_total_expert,
        uint32_t                n_expert,
        float                   swiglu_clamp,
        uint32_t                layer_index,
        const ds4_gpu_tensor *x,
        bool                    force_resident);

int ds4_gpu_glm_routed_moe_batch_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint32_t                gate_type,
        uint32_t                up_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                up_expert_bytes,
        uint64_t                up_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t                n_total_expert,
        uint32_t                n_expert,
        float                   swiglu_clamp,
        uint32_t                layer_index,
        const ds4_gpu_tensor *x,
        uint32_t                n_tokens,
        uint32_t                mid_token_stride,
        bool                    force_resident);

/*
 * qwen4exp routed MoE (see metal/qwen4exp_moe.metal).  The router takes raw
 * float32 logits, has no bias and softmaxes the selected logits only, so it is
 * a separate entry point from the GLM router.  Weight element ids are the ggml
 * type ids of ds4_qwen4exp_moe_types.h, the one table the loader accepts from
 * and the expert GEMM decodes.
 *
 * Call order for one MoE block, all three on the same command buffer:
 *   1. ds4_gpu_qwen4exp_router_select_tensor
 *   2. ds4_gpu_qwen4exp_routed_moe_tensor         -- writes out
 *   3. ds4_gpu_qwen4exp_shared_expert_tensor      -- adds into out
 */

/* One expert weight tensor: the shard mapping holding it, where it starts in
 * that mapping, its ggml type and its strides.  Every slab carries its own
 * mapping because one block's expert tensors can land in different shards of a
 * split GGUF, as blocks 11 and 41 of UD-Q4_K_XL do. */
typedef struct {
    const void *map;
    uint64_t    map_size;
    uint64_t    offset;
    uint64_t    expert_bytes;   /* one expert's slab; 0 for a 2-D tensor */
    uint64_t    row_bytes;
    uint32_t    type;
} ds4_gpu_qwen4exp_slab;
int ds4_gpu_qwen4exp_router_select_tensor(
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *weights,
        const ds4_gpu_tensor *logits,
        uint32_t              n_expert,
        uint32_t              n_expert_used,
        uint32_t              n_tokens);

int ds4_gpu_qwen4exp_routed_moe_tensor(
        ds4_gpu_tensor              *out,
        ds4_gpu_tensor              *mid,
        /* Per-slot partials of the down projection, n_tokens *
         * n_expert_used * out_dim floats.  A block owns one expert so the down
         * weight is read once for all of that expert's tokens; a token's slots
         * then land in different blocks and are summed afterwards in ascending
         * order.  Sized in the session memory plan. */
        ds4_gpu_tensor              *down_partial,
        const ds4_gpu_qwen4exp_slab *gate,
        const ds4_gpu_qwen4exp_slab *up,
        const ds4_gpu_qwen4exp_slab *down,
        uint32_t                     in_dim,
        uint32_t                     mid_dim,
        uint32_t                     out_dim,
        const ds4_gpu_tensor        *selected,
        const ds4_gpu_tensor        *weights,
        uint32_t                     n_total_expert,
        uint32_t                     n_expert_used,
        const ds4_gpu_tensor        *x,
        uint32_t                     n_tokens,
        uint32_t                     mid_token_stride);

int ds4_gpu_qwen4exp_shared_expert_tensor(
        ds4_gpu_tensor              *out,
        ds4_gpu_tensor              *mid,
        ds4_gpu_tensor              *gate_scale,
        const ds4_gpu_qwen4exp_slab *router,
        const ds4_gpu_qwen4exp_slab *gate,
        const ds4_gpu_qwen4exp_slab *up,
        const ds4_gpu_qwen4exp_slab *down,
        uint32_t                     in_dim,
        uint32_t                     mid_dim,
        uint32_t                     out_dim,
        const ds4_gpu_tensor        *x,
        uint32_t                     n_tokens);

/* What a decode-width routed MoE call hands to the shared expert that
 * follows it on the same stream. */
typedef struct {
    /* The quantised-x region the routed call wrote: quants at xq, scales at
     * xq + n_tokens * (in_dim / 32) * 32, sums after the scales -- the
     * layout both entries derive identically from the region start. */
    const int8_t *xq;
    /* Dead scratch from the same call: its pair-list metadata region
     * (counts/offsets/cursor/active/pairs), every consumer of which ran
     * inside that call.  A caller running after it on the same stream may
     * seat its own scratch at `seat` for `seat_bytes` bytes -- in
     * particular the shared expert's quantised mid, which must never
     * overlap the `xq` region this struct still hands out.  They cannot:
     * the metadata ends exactly where the xq region begins. */
    int8_t  *seat;
    uint64_t seat_bytes;
} ds4_gpu_qwen4exp_moe_handoff;

int ds4_gpu_qwen4exp_shared_expert_preq_tensor(
        ds4_gpu_tensor              *out,
        ds4_gpu_tensor              *mid,
        ds4_gpu_tensor              *gate_scale,
        const ds4_gpu_qwen4exp_slab *router,
        const ds4_gpu_qwen4exp_slab *gate,
        const ds4_gpu_qwen4exp_slab *up,
        const ds4_gpu_qwen4exp_slab *down,
        uint32_t                     in_dim,
        uint32_t                     mid_dim,
        uint32_t                     out_dim,
        const ds4_gpu_tensor        *x,
        uint32_t                     n_tokens,
        /* reuse: a routed MoE call's handoff (see above) -- the region THAT
         * call quantised x into, seated at its own layout offset in the
         * tier scratch, never this entry's base.  NULL quantises x here.
         * The caller owes the same x tensor, n_tokens and in_dim the routed
         * call used, and ordering on one stream.  A seat too small for this
         * entry's quantised mid declines the reuse to NULL's chain. */
        const ds4_gpu_qwen4exp_moe_handoff *reuse,
        /* gate_ready: the router's F32 launch already wrote gate_scale (the
         * ds4_gpu_qwen4exp_router_logits_gate_tensor fold), so this entry
         * skips its own gate kernel instead of repeating it. */
        int                          gate_ready);

/* The decode-width router tail, fused.  Whether the two fuses below engage at
 * all: a width below eight (the small-group envelope) and no
 * DS4_QWEN4EXP_NO_ROUTER_FUSE in the environment.  The entries re-check their
 * own conditions and decline to the unfused kernels, which return the same
 * bits, so a model outside the envelope still runs correctly. */
int ds4_gpu_qwen4exp_router_tail_fuse_on(uint32_t n_tokens);

/* The router's F32 logits with the shared expert's sigmoid gate folded in as
 * one extra block-row of the same GEMV launch.  Writes logits[0..n_rows) and
 * shexp_gate[0..n_rows); the caller then passes gate_ready 1 to the shared
 * expert so the standalone gate kernel is not launched again. */
int ds4_gpu_qwen4exp_router_logits_gate_tensor(
        ds4_gpu_tensor              *logits,
        ds4_gpu_tensor              *shexp_gate,
        const void                  *router_map,
        uint64_t                     router_map_size,
        uint64_t                     router_offset,
        const ds4_gpu_qwen4exp_slab *shexp_router,
        uint64_t                     in_dim,
        uint64_t                     out_dim,
        const ds4_gpu_tensor        *x,
        uint32_t                     n_rows);

/* The routed MoE with the router tail folded in: BUILDS selected and weights
 * from logits (the warp top-k, its softmax and the small metadata scan in one
 * launch at the widths ds4_gpu_qwen4exp_router_tail_fuse_on names) instead of
 * reading them.  Where the fuse declines, the selection is built by the
 * standalone router dispatch and the metadata by today's paths, unchanged.
 * handoff, when not NULL, receives this call's quantised-x region and dead
 * seat (see ds4_gpu_qwen4exp_moe_handoff) -- what the shared expert's preq
 * entry reuses; the regions are seated at THIS call's layout offsets, never
 * at a scratch base. */
int ds4_gpu_qwen4exp_router_tail_moe_tensor(
        ds4_gpu_tensor              *out,
        ds4_gpu_tensor              *mid,
        ds4_gpu_tensor              *down_partial,
        const ds4_gpu_qwen4exp_slab *gate,
        const ds4_gpu_qwen4exp_slab *up,
        const ds4_gpu_qwen4exp_slab *down,
        uint32_t                     in_dim,
        uint32_t                     mid_dim,
        uint32_t                     out_dim,
        ds4_gpu_tensor              *selected,
        ds4_gpu_tensor              *weights,
        const ds4_gpu_tensor        *logits,
        uint32_t                     n_total_expert,
        uint32_t                     n_expert_used,
        const ds4_gpu_tensor        *x,
        uint32_t                     n_tokens,
        uint32_t                     mid_token_stride,
        ds4_gpu_qwen4exp_moe_handoff *handoff);

int ds4_gpu_glm_routed_moe_batch_direct_scalar_q4_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint32_t                gate_type,
        uint32_t                up_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                up_expert_bytes,
        uint64_t                up_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t                n_total_expert,
        uint32_t                n_expert,
        float                   swiglu_clamp,
        uint32_t                layer_index,
        const ds4_gpu_tensor *x,
        uint32_t                n_tokens,
        uint32_t                mid_token_stride);

int ds4_gpu_routed_moe_set_selected_override(const int32_t *selected, uint32_t n_selected);
void ds4_gpu_set_glm_mtp_verify_mode(bool enabled);
#ifdef DS4_ROCM_BUILD
int ds4_gpu_dspark_gfx1151_fast_path(void);
void ds4_gpu_set_dspark_verify_mode(bool enabled);
#endif

int ds4_gpu_matmul_q8_0_kslice_hc_expand_add_tensor(
        ds4_gpu_tensor       *out_hc,
        ds4_gpu_tensor       *block_out,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint64_t              in_dim,
        uint64_t              out_dim,
        uint64_t              in_start,
        uint64_t              in_count,
        const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *block_add,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t              n_embd,
        uint32_t              n_hc);

int ds4_gpu_routed_moe_one_owned_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        ds4_gpu_tensor       *experts,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              gate_offset,
        uint64_t              up_offset,
        uint64_t              down_offset,
        uint32_t              gate_type,
        uint32_t              down_type,
        uint64_t              gate_expert_bytes,
        uint64_t              gate_row_bytes,
        uint64_t              down_expert_bytes,
        uint64_t              down_row_bytes,
        uint32_t              expert_in_dim,
        uint32_t              expert_mid_dim,
        uint32_t              out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t              n_total_expert,
        uint32_t              n_expert,
        uint32_t              resident_expert_base,
        uint32_t              resident_expert_count,
        float                 clamp,
        const ds4_gpu_tensor *x,
        ds4_gpu_tensor       *down_output,
        bool                  pack_fixed3,
        ds4_gpu_tensor       *shared_prequant);

int ds4_gpu_routed_moe_batch_owned_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        ds4_gpu_tensor       *experts,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              gate_offset,
        uint64_t              up_offset,
        uint64_t              down_offset,
        uint32_t              gate_type,
        uint32_t              down_type,
        uint64_t              gate_expert_bytes,
        uint64_t              gate_row_bytes,
        uint64_t              down_expert_bytes,
        uint64_t              down_row_bytes,
        uint32_t              expert_in_dim,
        uint32_t              expert_mid_dim,
        uint32_t              out_dim,
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *weights,
        uint32_t              n_total_expert,
        uint32_t              n_expert,
        uint32_t              resident_expert_base,
        uint32_t              resident_expert_count,
        float                 clamp,
        const ds4_gpu_tensor *x,
        uint32_t              layer_index,
        uint32_t              n_tokens,
        bool                 *mid_is_f16);

int ds4_gpu_routed_moe_owned_slots_combine_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *home_slots,
        const ds4_gpu_tensor *peer_slots,
        const ds4_gpu_tensor *selected,
        uint32_t              out_dim,
        uint32_t              expert_split);

int ds4_gpu_routed_moe_owned_slots_combine_rows_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *home_slots,
        const ds4_gpu_tensor *peer_slots,
        const ds4_gpu_tensor *selected,
        uint32_t              out_dim,
        uint32_t              expert_split,
        uint32_t              rows);

int ds4_gpu_routed_moe_owned_packed_combine_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *home_slots,
        const ds4_gpu_tensor *peer_packed,
        const ds4_gpu_tensor *selected,
        uint32_t              out_dim,
        uint32_t              expert_split);

int ds4_gpu_routed_moe_one_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        ds4_gpu_tensor       *experts,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint32_t                gate_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t                n_total_expert,
        uint32_t                n_expert,
        float                   clamp,
        const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *add_in,
        uint32_t                layer_index,
        bool                    force_resident);

int ds4_gpu_routed_moe_batch_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        ds4_gpu_tensor       *experts,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint32_t                gate_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t                n_total_expert,
        uint32_t                n_expert,
        float                   clamp,
        const ds4_gpu_tensor *x,
        uint32_t                layer_index,
        uint32_t                n_tokens,
        bool                   *mid_is_f16,
        bool                    force_resident);

/* =========================================================================
 * Hyper-Connection Kernels.
 * =========================================================================
 *
 * HC kernels reduce four residual streams before a sublayer and expand the
 * sublayer output back into four streams afterward.
 */

int ds4_gpu_hc_split_sinkhorn_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *mix,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint32_t                n_hc,
        uint32_t                sinkhorn_iters,
        float                   eps);

int ds4_gpu_hc_weighted_sum_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *weights,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ds4_gpu_hc_weighted_sum_split_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc);

/* Release decode fused HC pre-sublayer operation: split the HC mixer and
 * immediately reduce four HC streams into the active 4096-wide sublayer row. */
int ds4_gpu_hc_split_weighted_sum_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *split,
        const ds4_gpu_tensor *mix,
        const ds4_gpu_tensor *residual_hc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint32_t                n_embd,
        uint32_t                n_hc,
        uint32_t                sinkhorn_iters,
        float                   eps);

int ds4_gpu_hc_split_weighted_sum_norm_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *norm_out,
        ds4_gpu_tensor       *split,
        const ds4_gpu_tensor *mix,
        const ds4_gpu_tensor *residual_hc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint64_t                norm_weight_offset,
        uint32_t                n_embd,
        uint32_t                n_hc,
        uint32_t                sinkhorn_iters,
        float                   eps,
        float                   norm_eps);

int ds4_gpu_hc_rms_norm_mix_f16_available(void);
int ds4_gpu_hc_rms_norm_mix_f16_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              n,
        uint32_t              out_dim,
        float                 eps);

/* Batched HC RMSNorm followed by its narrow F16 mixer projection. On the
 * tuned Metal path, scale_scratch stores one float per row instead of the
 * full normalized HC tensor; other shapes retain the established fallback. */
int ds4_gpu_hc_rms_scale_project_f16_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *scale_scratch,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                in_dim,
        uint32_t                out_dim,
        const ds4_gpu_tensor *x,
        uint32_t                n_rows,
        float                   eps);

#ifdef __APPLE__
int ds4_gpu_hc_rms_norm_mix_split_norm_f16_tensor(
        ds4_gpu_tensor       *mix,
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *norm_out,
        ds4_gpu_tensor       *split,
        const ds4_gpu_tensor *residual_hc,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              mix_weight_offset,
        uint64_t              scale_offset,
        uint64_t              base_offset,
        uint64_t              norm_weight_offset,
        uint32_t              n,
        uint32_t              mix_dim,
        uint32_t              n_embd,
        uint32_t              n_hc,
        uint32_t              sinkhorn_iters,
        float                 eps,
        float                 hc_eps,
        float                 norm_eps);

#endif
int ds4_gpu_output_hc_weights_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *pre,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint32_t                n_hc,
        float                   eps);

int ds4_gpu_hc_expand_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *block_out,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *post,
        const ds4_gpu_tensor *comb,
        uint32_t                n_embd,
        uint32_t                n_hc);
int ds4_gpu_hc_expand_add_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *block_out,
        const ds4_gpu_tensor *block_add,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *post,
        const ds4_gpu_tensor *comb,
        uint32_t                n_embd,
        uint32_t                n_hc);


int ds4_gpu_hc_expand_add_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *block_out,
        const ds4_gpu_tensor *block_add,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *post,
        const ds4_gpu_tensor *comb,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ds4_gpu_hc_expand_split_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *block_out,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ds4_gpu_hc_expand_split_half_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *block_out_h,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ds4_gpu_hc_expand_add_split_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *block_out,
        const ds4_gpu_tensor *block_add,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ds4_gpu_hc_expand_add_split_half_add_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *block_out,
        const ds4_gpu_tensor *block_add_h,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ds4_gpu_shared_down_hc_expand_q8_0_tensor(
        ds4_gpu_tensor       *out_hc,
        ds4_gpu_tensor       *shared_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *shared_mid,
        const ds4_gpu_tensor *routed_out,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ds4_gpu_shared_down_hc_expand_add_q8_0_tensor(
        ds4_gpu_tensor       *out_hc,
        ds4_gpu_tensor       *shared_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *shared_mid,
        const ds4_gpu_tensor *routed_out,
        const ds4_gpu_tensor *routed_add,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ds4_gpu_shared_down_hc_expand_owned_q8_0_tensor(
        ds4_gpu_tensor       *out_hc,
        ds4_gpu_tensor       *shared_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *shared_mid,
        const ds4_gpu_tensor *home_slots,
        const ds4_gpu_tensor *peer_packed,
        const ds4_gpu_tensor *selected,
        uint32_t                expert_split,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ds4_gpu_matmul_q8_0_hc_expand_tensor(
        ds4_gpu_tensor       *out_hc,
        ds4_gpu_tensor       *block_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ds4_gpu_glm53_embedding_bf16(
        ds4_gpu_tensor       *out,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        const ds4_gpu_tensor *token_ids,
        uint32_t              n_tokens,
        uint32_t              n_embd,
        uint32_t              n_vocab);

int ds4_gpu_glm53_matmul_bf16(
        ds4_gpu_tensor       *out,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              in_dim,
        uint32_t              out_dim,
        const ds4_gpu_tensor *x,
        uint32_t              n_rows);

int ds4_gpu_glm53_matmul_bf16_qkv(
        ds4_gpu_tensor       *out_q,
        ds4_gpu_tensor       *out_k,
        ds4_gpu_tensor       *out_v,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_q_offset,
        uint64_t              weight_k_offset,
        uint64_t              weight_v_offset,
        uint32_t              in_dim,
        uint32_t              out_dim,
        const ds4_gpu_tensor *x);

#ifndef DS4_GLM53_VISION_TYPES_DEFINED
#define DS4_GLM53_VISION_TYPES_DEFINED
#define DS4_GLM53_VISION_LAYERS 24u

typedef struct {
    uint64_t norm1;
    uint64_t qkv_weight;
    uint64_t qkv_bias;
    uint64_t q_norm;
    uint64_t k_norm;
    uint64_t attn_proj_weight;
    uint64_t attn_proj_bias;
    uint64_t norm2;
    uint64_t gate_weight;
    uint64_t gate_bias;
    uint64_t up_weight;
    uint64_t up_bias;
    uint64_t down_weight;
    uint64_t down_bias;
} ds4_glm53_vision_layer_weights;

typedef struct {
    uint64_t patch_weight;
    uint64_t patch_bias;
    uint64_t post_norm;
    uint64_t downsample_weight;
    uint64_t downsample_bias;
    uint64_t merger_proj;
    uint64_t merger_norm;
    uint64_t merger_norm_bias;
    uint64_t merger_gate;
    uint64_t merger_up;
    uint64_t merger_down;
    ds4_glm53_vision_layer_weights layer[DS4_GLM53_VISION_LAYERS];
} ds4_glm53_vision_weights;
#endif

/* Encode normalized, block-major image patches into 4096-wide language-model
 * embeddings. GPU implementations keep every intermediate on device. */
int ds4_gpu_glm53_vision_encode(
        float                          *out,
        const float                    *patches,
        uint32_t                        grid_h,
        uint32_t                        grid_w,
        const void                     *model_map,
        uint64_t                        model_size,
        const ds4_glm53_vision_weights *weights);

#ifndef DS4_DEEPSEEK4_VISION_TYPES_DEFINED
#define DS4_DEEPSEEK4_VISION_TYPES_DEFINED
#define DS4_DEEPSEEK4_VISION_LAYERS 32u
#define DS4_DEEPSEEK4_LANGUAGE_LAYERS 43u
#define DS4_DEEPSEEK4_MTP_LAYERS 3u

typedef struct {
    uint64_t norm1;
    uint64_t qkv_weight;
    uint64_t qkv_bias;
    uint64_t attn_proj_weight;
    uint64_t attn_proj_bias;
    uint64_t norm2;
    uint64_t mlp_w1;
    uint64_t mlp_w2;
} ds4_deepseek4_vision_layer_weights;

typedef struct {
    uint64_t patch_weight;
    uint64_t patch_bias;
    uint64_t post_norm;
    uint64_t aligner_w1;
    uint64_t aligner_w1_bias;
    uint64_t aligner_w2;
    uint64_t aligner_w2_bias;
    uint64_t image_start;
    uint64_t image_pad;
    uint64_t image_newline;
    uint64_t image_end;
    uint64_t visual_router_bias[DS4_DEEPSEEK4_LANGUAGE_LAYERS];
    uint64_t mtp_visual_router_bias[DS4_DEEPSEEK4_MTP_LAYERS];
    uint64_t hash_router_bias[3];
    ds4_deepseek4_vision_layer_weights layer[DS4_DEEPSEEK4_VISION_LAYERS];
} ds4_deepseek4_vision_weights;
#endif

/* Encode row-major normalized 14x14 RGB patches. The output is the natural
 * row-major 3x3-aligned grid; N-layout permutation and sentinels are applied
 * by the prompt layer once the image's token position is known. */
int ds4_gpu_deepseek4_vision_encode(
        float                              *out,
        const float                        *patches,
        uint32_t                            grid_h,
        uint32_t                            grid_w,
        const void                         *model_map,
        uint64_t                            model_size,
        const ds4_deepseek4_vision_weights *weights);

/* Replace token rows with projected image embeddings and repeat each row into
 * every GLM hyperconnection stream. Must be called in an active command batch. */
int ds4_gpu_glm53_scatter_image_hc(
        ds4_gpu_tensor       *hc,
        const ds4_gpu_tensor *image,
        uint32_t              dst_row,
        uint32_t              image_row,
        uint32_t              rows,
        uint32_t              total_rows,
        uint32_t              n_embd,
        uint32_t              n_hc);

/* GLM-5.3 Kimi Delta Attention. Recurrent and convolution state stay FP32. */
int ds4_gpu_glm53_kda_decode(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *conv_state,
        ds4_gpu_tensor       *recurrent_state,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *k,
        const ds4_gpu_tensor *v,
        const ds4_gpu_tensor *raw_gate,
        const ds4_gpu_tensor *raw_beta,
        const ds4_gpu_tensor *output_gate,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              q_conv_offset,
        uint64_t              k_conv_offset,
        uint64_t              v_conv_offset,
        uint64_t              a_log_offset,
        uint64_t              dt_bias_offset,
        uint64_t              output_norm_offset,
        uint32_t              n_heads,
        uint32_t              n_rows,
        float                 gate_lower_bound,
        float                 norm_eps);

int ds4_gpu_glm53_kda_prefill(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *conv_state,
        ds4_gpu_tensor       *recurrent_state,
        ds4_gpu_tensor       *q,
        ds4_gpu_tensor       *k,
        ds4_gpu_tensor       *v,
        ds4_gpu_tensor       *raw_gate,
        const ds4_gpu_tensor *raw_beta,
        const ds4_gpu_tensor *output_gate,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              q_conv_offset,
        uint64_t              k_conv_offset,
        uint64_t              v_conv_offset,
        uint64_t              a_log_offset,
        uint64_t              dt_bias_offset,
        uint64_t              output_norm_offset,
        uint32_t              n_heads,
        uint32_t              n_tokens,
        float                 gate_lower_bound,
        float                 norm_eps);

/* Which key head a GDN value head reads.  The reference model stores value
 * heads grouped by key head, so value head j belongs to key head
 * j / (n_value_head / n_key_head).  llama.cpp's converter reorders them into
 * tiled order for ggml's broadcast (conversion/qwen.py
 * `_LinearAttentionVReorderBase._reorder_v_heads`), so in a converted GGUF
 * value head j belongs to key head j % n_key_head.  The converter reorders
 * the value rows of `attn_qkv`, all of `attn_gate`, `ssm_alpha`, `ssm_beta`,
 * `ssm_a` and `ssm_dt.bias`, the value channels of `ssm_conv1d` and the
 * columns of `ssm_out` together, so one flag covers every one of them.
 * GGUF-loaded weights are TILED. */
typedef enum {
    DS4_QWEN4EXP_GDN_HEADS_GROUPED = 0,
    DS4_QWEN4EXP_GDN_HEADS_TILED = 1,
} ds4_qwen4exp_gdn_head_layout;

/* Qwen4exp gated delta net (GDN).  `qkv` is the fused `attn_qkv` projection
 * output laid out [rows][tokens][conv_dim] with the query, key and value
 * regions in that order and conv_dim = (2 * n_key_head + n_value_head) * 128;
 * the call rewrites it in place with the convolved, activated and normalised
 * values.  `raw_alpha` and `raw_beta` are the `ssm_alpha` and `ssm_beta`
 * projection outputs, one value per token and value head.  Convolution and
 * recurrent state stay FP32, so a chunked sequence reproduces the
 * single-chunk result bit for bit.  Prefill runs one row of `n_tokens`
 * tokens; decode runs `n_rows` rows of one token, each with its own
 * convolution and recurrent state. */
/*
 * PER-ROW STATE SNAPSHOTS.
 *
 * `conv_snapshot` and `state_snapshot` hold `n_snapshot_rows` slots, each the
 * size of the live buffer beside it, and the prefill mirrors the carried state
 * into slot k after token k.  The speculative cycle uses them to adopt the
 * state as it stood after the last ACCEPTED row instead of rewinding and
 * running a shorter forward again; the recurrence is token-serial, so the two
 * are the same value.  `n_snapshot_rows` must be strictly less than
 * `n_tokens` -- the last row's state is the live state and needs no slot --
 * and 0 asks for none, which is what every non-speculative call passes.
 */
int ds4_gpu_qwen4exp_gdn_prefill(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *conv_state,
        ds4_gpu_tensor       *recurrent_state,
        ds4_gpu_tensor       *conv_snapshot,
        ds4_gpu_tensor       *state_snapshot,
        uint32_t              n_snapshot_rows,
        ds4_gpu_tensor       *qkv,
        const ds4_gpu_tensor *raw_alpha,
        const ds4_gpu_tensor *raw_beta,
        const ds4_gpu_tensor *output_gate,
        /* One slab per tensor: a shard boundary can fall between any two of
         * ssm_conv1d, ssm_a, ssm_dt.bias and ssm_norm. */
        const ds4_gpu_qwen4exp_slab *conv_weight,
        const ds4_gpu_qwen4exp_slab *a_log,
        const ds4_gpu_qwen4exp_slab *dt_bias,
        const ds4_gpu_qwen4exp_slab *output_norm,
        uint32_t              n_key_head,
        uint32_t              n_value_head,
        uint32_t              n_tokens,
        uint32_t              head_layout,
        float                 qk_norm_eps,
        float                 norm_eps);

int ds4_gpu_qwen4exp_gdn_decode(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *conv_state,
        ds4_gpu_tensor       *recurrent_state,
        ds4_gpu_tensor       *qkv,
        const ds4_gpu_tensor *raw_alpha,
        const ds4_gpu_tensor *raw_beta,
        const ds4_gpu_tensor *output_gate,
        /* One slab per tensor: a shard boundary can fall between any two of
         * ssm_conv1d, ssm_a, ssm_dt.bias and ssm_norm. */
        const ds4_gpu_qwen4exp_slab *conv_weight,
        const ds4_gpu_qwen4exp_slab *a_log,
        const ds4_gpu_qwen4exp_slab *dt_bias,
        const ds4_gpu_qwen4exp_slab *output_norm,
        uint32_t              n_key_head,
        uint32_t              n_value_head,
        uint32_t              n_rows,
        uint32_t              head_layout,
        float                 qk_norm_eps,
        float                 norm_eps);

/*
 * REJECT-PATH ADOPTION, resolved on the device.
 *
 * A rejecting round adopts the per-row snapshot slot `row` the verify forward
 * left behind instead of copying it over the live buffers (216 MiB per
 * rejecting round on the production shape).  The adopted row lives in ONE
 * device-resident uint32 at a fixed tensor address -- the discipline `d_pos`
 * uses -- and the kernels above read it when they run, so a captured graph
 * stays valid while adoption changes between replays: the argument a graph
 * bakes is the scalar's ADDRESS, never its value.
 *
 * DS4_QWEN4EXP_STATE_ADOPT_LIVE means the live buffers hold the state to read;
 * any other value is a snapshot row index.  A NULL `adopt_row` always means
 * LIVE, which is the copy-era behavior and what the kill switch and the
 * backends without these entries pass.  The WRITE side is never redirected:
 * a round's final stores land in the live buffers and its snapshot stores in
 * the snapshot tensors, distinct allocations, so an end-write can never
 * clobber a snapshot row that the same round's reject still needs.  The
 * kernels that read an adopted row load their state elements once, before any
 * store they issue, so a round may safely read the snapshot row it is about
 * to rewrite.  Adoption is single-sequence: it arms only on one-row calls.
 *
 * The scalar is also handed in only by a forward that can carry an armed
 * adoption -- the rejecting round's re-feed, at most one speculative commit
 * wide.  A prefill never follows a reject without the session reset that
 * clears adoption, so it passes NULL: LIVE either way to the kernels, and
 * the sign qwen4exp_cuda_gdn_run uses to keep the token-parallel
 * convolution available, which a handed-in scalar refuses.
 */
#define DS4_QWEN4EXP_STATE_ADOPT_LIVE 0xFFFFFFFFu

#if defined(DS4_ROCM_BUILD) || (!defined(DS4_NO_GPU) && !defined(__APPLE__))
int ds4_gpu_qwen4exp_gdn_prefill_adopt(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *conv_state,
        ds4_gpu_tensor       *recurrent_state,
        ds4_gpu_tensor       *conv_snapshot,
        ds4_gpu_tensor       *state_snapshot,
        uint32_t              n_snapshot_rows,
        ds4_gpu_tensor       *qkv,
        const ds4_gpu_tensor *raw_alpha,
        const ds4_gpu_tensor *raw_beta,
        const ds4_gpu_tensor *output_gate,
        const ds4_gpu_qwen4exp_slab *conv_weight,
        const ds4_gpu_qwen4exp_slab *a_log,
        const ds4_gpu_qwen4exp_slab *dt_bias,
        const ds4_gpu_qwen4exp_slab *output_norm,
        uint32_t              n_key_head,
        uint32_t              n_value_head,
        uint32_t              n_tokens,
        uint32_t              head_layout,
        float                 qk_norm_eps,
        float                 norm_eps,
        const ds4_gpu_tensor *adopt_row);

/* The decode twin reads an adopted row too (a rejecting round's replay
 * continues from the adopted state), so it takes the snapshot tensors the
 * plain entry never needed; it still writes no snapshots of its own. */
int ds4_gpu_qwen4exp_gdn_decode_adopt(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *conv_state,
        ds4_gpu_tensor       *recurrent_state,
        ds4_gpu_tensor       *conv_snapshot,
        ds4_gpu_tensor       *state_snapshot,
        ds4_gpu_tensor       *qkv,
        const ds4_gpu_tensor *raw_alpha,
        const ds4_gpu_tensor *raw_beta,
        const ds4_gpu_tensor *output_gate,
        const ds4_gpu_qwen4exp_slab *conv_weight,
        const ds4_gpu_qwen4exp_slab *a_log,
        const ds4_gpu_qwen4exp_slab *dt_bias,
        const ds4_gpu_qwen4exp_slab *output_norm,
        uint32_t              n_key_head,
        uint32_t              n_value_head,
        uint32_t              n_rows,
        uint32_t              head_layout,
        float                 qk_norm_eps,
        float                 norm_eps,
        const ds4_gpu_tensor *adopt_row);
#endif

/* Decode-island CUDA graph capture (CUDA backend; Metal/ROCm/CPU stub it
 * out and stay eager).  Design ported from the Entrpi/ds4 batched-serving
 * fork's per-layer decode graph capture.  The key identifies a captured
 * island: layer, island index, and the activation buffers whose addresses
 * the captured kernels bake in.  ds4_cuda.cu mirrors this struct
 * byte-for-byte (it does not include this header); keep both in sync. */
typedef struct ds4_decode_graph_key {
    uint32_t il;
    uint32_t island;    /* 0/1: layer halves; 2: QSA; 3: complete MTP block */
    uint32_t variant;
    uint32_t _pad;
    void    *cur_hc;
    void    *after_attn_hc;
    void    *after_ffn_hc;
    void    *attn_norm;
} ds4_decode_graph_key;

int  ds4_gpu_decode_graphs_supported(void);
/* 1: replayed (island already executed; skip encoding it)
 * 0: capturing (encode the island, then call _end)
 * -1: run eagerly */
int  ds4_gpu_decode_graph_begin(const ds4_decode_graph_key *key);
/* 0: capture committed and launched; -1: capture failed (entry retired;
 * the caller must re-encode the island eagerly -- no work was executed). */
int  ds4_gpu_qwen4exp_update_dpos(
        ds4_gpu_tensor *d_pos,
        uint32_t        pos);

int  ds4_gpu_decode_graph_end(const ds4_decode_graph_key *key);
void ds4_gpu_decode_graph_abort(const ds4_decode_graph_key *key);
void ds4_gpu_decode_graphs_invalidate(void);

/* =========================================================================
 * Qwen4exp Hyper-Connections, Norms, RoPE, Embedding and Head.
 * =========================================================================
 *
 * The qwen4exp gated residual reuses DS4's [token][hc][embd] activation
 * layout but not its Sinkhorn mixer: there is no combination matrix, the
 * pre-reduction is a MEAN of a full-width low-rank gate rather than a
 * weighted sum of per-stream scalars, and the inject is diagonal.  See
 * metal/qwen4exp_hc.metal for the divergences and ds4_qwen4exp_hc_ref.h for
 * the f32 reference these kernels are checked against.
 */

/* Zero-centered RMS norm, optionally grouped.
 *
 * `group` equal to `n` is the ordinary per-row norm.  `group` equal to the
 * hidden size gives each hyper-connection stream its own statistic while the
 * weight still indexes the flat row, which is the hc_norm form; the MTP
 * head's pre-norm over the hidden stream is UNGROUPED over all n_hc*n_embd.
 * `weight_bias` is 0 when the checkpoint bakes the offset and 1 when it
 * stores zero-centered weights.  `round_bf16` reproduces MLX's cast of the
 * normalized value to the activation dtype before the weight multiply. */
int ds4_gpu_qwen4exp_rms_norm_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              n,
        uint32_t              group,
        uint32_t              rows,
        float                 eps,
        float                 weight_bias,
        int                   round_bf16);

/* silu(x * scale) in place, over `n` values. */
int ds4_gpu_qwen4exp_scale_silu_tensor(
        ds4_gpu_tensor *x,
        uint32_t        n,
        float           scale);

/* Block input: mean over the streams of sigmoid(wide) * normed.  `wide` is
 * the RAW low-rank up projection; its sigmoid is fused in. */
int ds4_gpu_qwen4exp_hc_mix_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *normed,
        const ds4_gpu_tensor *wide,
        uint32_t              n_embd,
        uint32_t              n_hc,
        uint32_t              rows);

/* inject[t][h] = 2 * sigmoid(dot(W[h], normed[t]) / n_hc). */
int ds4_gpu_qwen4exp_hc_inject_weights_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *normed,
        /* The weight as a slab, so its TYPE and row stride travel with it.  The
         * target stores this tensor F32 and the MTP head stores it Q8_0; both
         * decode here, from the one table in ds4_qwen4exp_hc_types.h. */
        const ds4_gpu_qwen4exp_slab *weight,
        uint32_t              n_embd,
        uint32_t              n_hc,
        uint32_t              rows);

/* out[t][h][d] = residual[t][h][d] + block[t][d] * inject[t][h].
 * Safe in place with out == residual. */
int ds4_gpu_qwen4exp_hc_inject_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *block_out,
        const ds4_gpu_tensor *inject,
        uint32_t              n_embd,
        uint32_t              n_hc,
        uint32_t              rows);

/* The whole gated residual mixer: norm, low-rank gate, mean, inject weights.
 *
 * `inject` and `inject_weight_offset` are the inject head; pass NULL and 0
 * for the tower's final mixer, which has none and stands in for the
 * `model.norm` this checkpoint does not carry.  `hyper` is never written:
 * it is the residual the inject adds back into, and it is also the
 * pre-final-mixer stream the native MTP head consumes.
 *
 * The tower's head path is this call with `inject` NULL, giving the collapsed
 * 2560-wide hidden state, followed by ds4_gpu_matmul_q8_0_tensor against the
 * `output` tensor; there is no separate LM-head entry point because there is
 * nothing qwen4exp-specific left to do at that point. */
int ds4_gpu_qwen4exp_hc_mixer_tensor(
        ds4_gpu_tensor       *mixed,
        ds4_gpu_tensor       *inject,
        ds4_gpu_tensor       *normed_scratch,
        ds4_gpu_tensor       *lowrank_scratch,
        ds4_gpu_tensor       *wide_scratch,
        const ds4_gpu_tensor *hyper,
        /* One slab per tensor: a shard boundary can fall between any two of a
         * mixer's four weights. */
        const ds4_gpu_qwen4exp_slab *norm_weight,
        const ds4_gpu_qwen4exp_slab *down_weight,
        const ds4_gpu_qwen4exp_slab *up_weight,
        const ds4_gpu_qwen4exp_slab *inject_weight,
        uint32_t              n_embd,
        uint32_t              n_hc,
        uint32_t              n_lowrank,
        uint32_t              rows,
        float                 eps,
        float                 weight_bias,
        int                   round_bf16);

/* The same mixer, built ONLY out of the per-op wrappers -- no backend fusion.
 *
 * Nothing in the engine calls it.  It exists so a test can require the entry
 * above, which does fuse where the backend can, to agree with the op-by-op
 * chain bit for bit at the production shapes. */
int ds4_gpu_qwen4exp_hc_mixer_unfused_tensor(
        ds4_gpu_tensor       *mixed,
        ds4_gpu_tensor       *inject,
        ds4_gpu_tensor       *normed_scratch,
        ds4_gpu_tensor       *lowrank_scratch,
        ds4_gpu_tensor       *wide_scratch,
        const ds4_gpu_tensor *hyper,
        const ds4_gpu_qwen4exp_slab *norm_weight,
        const ds4_gpu_qwen4exp_slab *down_weight,
        const ds4_gpu_qwen4exp_slab *up_weight,
        const ds4_gpu_qwen4exp_slab *inject_weight,
        uint32_t              n_embd,
        uint32_t              n_hc,
        uint32_t              n_lowrank,
        uint32_t              rows,
        float                 eps,
        float                 weight_bias,
        int                   round_bf16);

/* Token embedding gather tiled into the hyper-connection streams: the
 * layer-0 seed.  `rows_scratch` holds the gathered [n_tokens][n_embd] rows. */
int ds4_gpu_qwen4exp_embed_tokens_hc_tensor(
        ds4_gpu_tensor       *out_hc,
        ds4_gpu_tensor       *rows_scratch,
        const ds4_gpu_tensor *tokens,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              weight_type,
        uint32_t              n_vocab,
        uint32_t              n_tokens,
        uint32_t              n_embd,
        uint32_t              n_hc);

/* CUDA's MTP head packs all [embedding | hidden-stream] rows in one launch.
 * Other backends may leave the optional MTP hook unbound and use tensor
 * copies, so this entry is implemented only by the CUDA Qwen translation
 * unit. */
int ds4_gpu_qwen4exp_ehx_pack_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *embedding,
        const ds4_gpu_tensor *hidden,
        uint32_t              n_tokens,
        uint32_t              n_hc,
        uint32_t              n_embd);

/* =========================================================================
 * Qwen4exp per-layer embedding (PLE) block.
 * =========================================================================
 *
 * The n-gram table is 26.8 GiB and stays on the solid-state disk: the host
 * hashes the recent history into row ids, reads the rows out of the mapping
 * (ds4_qwen4exp_ple.c) and uploads only the gathered [rows][ple_embd] block.
 * Nothing here holds a second copy of the table, and the only PLE state that
 * lives on the device is the gathered rows, the block scratch and the
 * (conv_kernel - 1) * dilation rolling convolution window.
 *
 * Kernels live in metal/qwen4exp_ple.metal and ds4_cuda_qwen4exp.cu; the
 * double reference they are checked against is ds4_qwen4exp_ple_ref.h.
 */

/* out[t][h][d] = sigmoid(signed_sqrt(dot(key[t][h], query[t][h]) / sqrt(E)))
 *                * value[t][d]
 *
 * `signed_sqrt(v)` is sqrt(max(|v|, 1e-6)) * sign(v), the reference's own
 * floor and its own sign rule.  `value` is 2560 wide and shared by every
 * hyper-connection stream.  Safe in place with out == key. */
int ds4_gpu_qwen4exp_ple_gate_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *key_hc,
        const ds4_gpu_tensor *query_hc,
        const ds4_gpu_tensor *value,
        uint32_t              n_embd,
        uint32_t              n_hc,
        uint32_t              rows);

/* The dilated depthwise short convolution, its rolling state and the two adds
 * that close the block:
 *
 *   full(i)      = i < S ? state[i][c] : conv_in[i - S][c],  S = (K-1)*dilation
 *   hyper[t][c] += gated[t][c] + silu(sum_k weight[c][k] * full(t + dilation*k))
 *   state[j][c]  = full(rows + j)
 *
 * `weight` is the F32 [channels][conv_kernel] checkpoint tensor, taps
 * contiguous, and tap conv_kernel - 1 multiplies the current row.  `hyper` is
 * accumulated into, which is the reference's `stream = stream + ple(stream)`.
 * `state` is [S][channels] f32 and is advanced. */
int ds4_gpu_qwen4exp_ple_conv_tensor(
        ds4_gpu_tensor       *hyper,
        ds4_gpu_tensor       *conv_state,
        /* `n_snapshot_rows` slots of [S][channels], slot k the window as it
         * stands after token k.  See the note on ds4_gpu_qwen4exp_gdn_prefill;
         * 0 asks for none. */
        ds4_gpu_tensor       *conv_snapshot,
        uint32_t              n_snapshot_rows,
        const ds4_gpu_tensor *gated,
        const ds4_gpu_tensor *conv_in,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              channels,
        uint32_t              conv_kernel,
        uint32_t              dilation,
        uint32_t              rows,
        /* Adoption: read the rolling window from snapshot row `adopt_row`
         * instead of `conv_state`.  See the note above the GDN entries; NULL
         * and DS4_QWEN4EXP_STATE_ADOPT_LIVE read the live window, which is
         * what the backends that do not resolve adoption always do. */
        const ds4_gpu_tensor *adopt_row);

/* The whole PLE block over already-gathered n-gram rows, composed out of the
 * two kernels above, the grouped RMS norm and the Q8_0 matmul.  This is the
 * `ple_block` op boundary of the MLX reference dump.
 *
 * Every weight arrives as its own slab, so the six tensors of the block are
 * each resolved against the shard mapping that actually holds them: a split
 * GGUF can put a shard boundary inside one block, and one mapping with six
 * offsets would read plausible numbers out of the wrong file.  `expert_bytes`
 * and `row_bytes` are unused here; the block has no 3-D tensor.
 *
 * `key_scratch` is [rows][n_hc*n_embd] and holds the key projection and then
 * the gated stream; `aux_scratch` is the same width and holds the query and
 * then the convolution input; `value_scratch` is [rows][n_embd].  `hyper` is
 * read for the query and accumulated into at the end, and it is the stream
 * ENTERING the layer: the reference applies this block before the layer's
 * attention mixer. */
int ds4_gpu_qwen4exp_ple_block_tensor(
        ds4_gpu_tensor              *hyper,
        ds4_gpu_tensor              *conv_state,
        ds4_gpu_tensor              *conv_snapshot,
        uint32_t                     n_snapshot_rows,
        ds4_gpu_tensor              *key_scratch,
        ds4_gpu_tensor              *aux_scratch,
        ds4_gpu_tensor              *value_scratch,
        const ds4_gpu_tensor        *ngram_rows,
        const ds4_gpu_qwen4exp_slab *key_weight,
        const ds4_gpu_qwen4exp_slab *value_weight,
        const ds4_gpu_qwen4exp_slab *norm_key,
        const ds4_gpu_qwen4exp_slab *norm_query,
        const ds4_gpu_qwen4exp_slab *norm_conv,
        const ds4_gpu_qwen4exp_slab *conv_weight,
        uint32_t              n_embd,
        uint32_t              n_hc,
        uint32_t              ple_embd,
        uint32_t              conv_kernel,
        uint32_t              dilation,
        uint32_t              rows,
        float                 eps,
        float                 norm_key_bias,
        float                 norm_query_bias,
        float                 norm_conv_bias,
        int                   round_bf16,
        /* Adoption, handed straight to the closing convolution.  NULL and
         * DS4_QWEN4EXP_STATE_ADOPT_LIVE read the live window. */
        const ds4_gpu_tensor *adopt_row);

#ifdef __cplusplus
}
#endif

#endif
