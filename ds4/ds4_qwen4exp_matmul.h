/*
 * The one decode-order routing rule for qwen4exp Q8_0 projections.
 *
 * ds4_gpu_matmul_q8_0_tensor picks its reduction strategy by row count
 * (ds4_metal.m:18193), so the same weights and the same input row give
 * slightly different numbers at one row and at two.  It is the ONLY kernel in
 * this port that does: every qwen4exp kernel proper runs one threadgroup per
 * token and reduces within it, so its rows do not see the batch at all.
 *
 * ds4_gpu_matmul_q8_0_decode_rows_exact_tensor (ds4_metal.m:18263) is the
 * entry upstream provides for the multi-row decode case, and at one row it IS
 * the decode path -- so routing through it inside the speculative cycle's
 * width leaves the serial leg unchanged and closes the gap between a batched
 * verify and a one-row decode of the same row.  Above that width the call is
 * prefill, which has no such requirement and wants the throughput.
 *
 * With this and the F32 rule below, the tower is bit-invariant across row count
 * on Metal and on CUDA; tests/test_qwen4exp_graph asserts exactly zero.
 *
 * This lives in its own header because the three users are three translation
 * units: ds4_qwen4exp_graph.inc (compiled into ds4.c), and
 * ds4_qwen4exp_hc_host.inc and ds4_qwen4exp_ple_host.inc (compiled into
 * ds4_metal.m and ds4_cuda_qwen4exp.cu).  One rule, one definition.
 */
#ifndef DS4_QWEN4EXP_MATMUL_H
#define DS4_QWEN4EXP_MATMUL_H

#include <stdint.h>

#include "ds4_gpu.h"
#include "ds4_qwen4exp_mtp.h"

static inline int ds4_qwen4exp_matmul_q8_0(ds4_gpu_tensor       *out,
                                           const void           *map,
                                           uint64_t              map_size,
                                           uint64_t              offset,
                                           uint64_t              in_dim,
                                           uint64_t              out_dim,
                                           const ds4_gpu_tensor *x,
                                           uint32_t              rows) {
    return ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(
            out, map, map_size, offset, in_dim, out_dim, x, rows);
}

static inline int ds4_qwen4exp_quantize_q8_0(ds4_gpu_tensor       *q,
                                             uint64_t              q_offset,
                                             uint64_t              s_offset,
                                             const ds4_gpu_tensor *x,
                                             uint64_t              in_dim,
                                             uint32_t              rows) {
    return ds4_gpu_quantize_q8_0_decode_rows_exact_tensor(
            q, q_offset, s_offset, x, in_dim, rows);
}

static inline int ds4_qwen4exp_matmul_q8_0_preq(ds4_gpu_tensor       *out,
                                                const void           *map,
                                                uint64_t              map_size,
                                                uint64_t              offset,
                                                uint64_t              in_dim,
                                                uint64_t              out_dim,
                                                const ds4_gpu_tensor *q,
                                                uint64_t              q_offset,
                                                uint64_t              s_offset,
                                                uint32_t              rows) {
    return ds4_gpu_matmul_q8_0_preq_rows_exact_tensor(
            out, map, map_size, offset, in_dim, out_dim, q, q_offset, s_offset, rows);
}

/* The same rule for the F32 projections.
 *
 * ds4_gpu_matmul_f32_tensor tiers by row count on BOTH backends, and harder
 * than the Q8_0 entry does: CUDA takes cuBLAS SGEMM above one row and a
 * hand-written kernel at one, while Metal has a matvec at one row, a
 * small-batch kernel to eight, and a general one above.  The qwen4exp tower
 * calls it for the gated delta net's alpha and beta projections, whose outputs
 * become the decay and beta gates and multiply into the recurrent state, and
 * for the MoE router's logits, whose top-k selection is discrete.
 *
 * ds4_gpu_matmul_f32_decode_rows_exact_tensor gives every row the one-row
 * reduction order at ANY width -- one launch on CUDA, one encoder of per-row
 * dispatches on Metal -- so this needs no width condition at all.  It replaced
 * a host loop that allocated and freed two tensor views per row, which was
 * correct but could not be used at prefill widths.
 */
static inline int ds4_qwen4exp_matmul_f32(ds4_gpu_tensor       *out,
                                          const void           *map,
                                          uint64_t              map_size,
                                          uint64_t              offset,
                                          uint64_t              in_dim,
                                          uint64_t              out_dim,
                                          const ds4_gpu_tensor *x,
                                          uint32_t              rows) {
    return ds4_gpu_matmul_f32_decode_rows_exact_tensor(
            out, map, map_size, offset, in_dim, out_dim, x, rows);
}

/* The indexer's BF16 projections.
 *
 * ds4_gpu_glm53_matmul_bf16 tiers by row count at EIGHT, and on BOTH backends
 * with the same number: ds4_metal.m:44625 `use_mv = n_rows <= 8u` picks the
 * per-row matvec at or below eight and a tiled matmul above it, and
 * ds4_cuda.cu:27252 picks a per-row kernel at or below eight and cuBLAS
 * GemmEx above it.  The two do not agree bit for bit, so a nine-row prefill
 * writes an indexer key tape that the same nine rows fed one at a time do not,
 * and the pooled blocks built from that tape inherit it.  The tape and the
 * pool are CARRIED state: they decide which tokens a later query may attend
 * to, so a chunk boundary that moves them is observable at the output as soon
 * as the context passes the indexer's token budget.
 *
 * The rows of a matmul are independent, so running the call in groups of eight
 * is the same arithmetic the one-row path would do, at any width, and needs no
 * new kernel on either backend.  Eight rather than one because eight is the
 * widest group both backends already treat as decode order, so the fix costs
 * ceil(rows / 8) dispatches instead of rows of them.
 *
 * This is the third rule of the same kind, and the last op in the tower that
 * chose a strategy by row count.
 *
 * The loop is the PORTABLE spelling of the rule, not the only one.  Where a
 * backend can issue the same groups in one dispatch it should, because the
 * cost of the loop is entirely dispatch: the per-row kernel reduces each
 * (row, column) pair inside one warp and never reads the row count, so the
 * groups are a partition of the launch grid and nothing else.  CUDA does this
 * in ds4_gpu_glm53_matmul_bf16_rows_exact -- same kernel, same operands, same
 * reduction tree, grid.y raised from eight to rows -- which is bit-identical
 * to the loop by construction rather than by comparison.  A backend without
 * one returns -1 and gets the loop below, unchanged.
 *
 * At or below eight rows nothing moves at all: that call was already a single
 * dispatch, and it is the one inside decode's CUDA graph capture.
 */
#define DS4_QWEN4EXP_BF16_DECODE_ROWS 8u

static inline int ds4_qwen4exp_matmul_bf16(ds4_gpu_tensor       *out,
                                           const void           *map,
                                           uint64_t              map_size,
                                           uint64_t              offset,
                                           uint32_t              in_dim,
                                           uint32_t              out_dim,
                                           const ds4_gpu_tensor *x,
                                           uint32_t              rows) {
    if (rows <= DS4_QWEN4EXP_BF16_DECODE_ROWS) {
        return ds4_gpu_glm53_matmul_bf16(out, map, map_size, offset,
                                         in_dim, out_dim, x, rows);
    }
    const int one_dispatch = ds4_gpu_glm53_matmul_bf16_rows_exact(
            out, map, map_size, offset, in_dim, out_dim, x, rows);
    if (one_dispatch >= 0) return one_dispatch;
    for (uint32_t at = 0; at < rows; at += DS4_QWEN4EXP_BF16_DECODE_ROWS) {
        const uint32_t left = rows - at;
        const uint32_t take = left < DS4_QWEN4EXP_BF16_DECODE_ROWS
            ? left : DS4_QWEN4EXP_BF16_DECODE_ROWS;
        ds4_gpu_tensor *xv = ds4_gpu_tensor_view(
                x, (uint64_t)at * in_dim * sizeof(float),
                (uint64_t)take * in_dim * sizeof(float));
        ds4_gpu_tensor *ov = ds4_gpu_tensor_view(
                out, (uint64_t)at * out_dim * sizeof(float),
                (uint64_t)take * out_dim * sizeof(float));
        const int ok = xv && ov &&
            ds4_gpu_glm53_matmul_bf16(ov, map, map_size, offset,
                                      in_dim, out_dim, xv, take);
        ds4_gpu_tensor_free(ov);
        ds4_gpu_tensor_free(xv);
        if (!ok) return 0;
    }
    return 1;
}

#endif /* DS4_QWEN4EXP_MATMUL_H */
