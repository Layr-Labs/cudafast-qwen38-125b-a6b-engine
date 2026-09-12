/*
 * The MTP head's default hook binding.
 *
 * This is where the head's typed hooks meet the real functions the other lanes
 * export, and the assignment IS the check: if L5b changes a prototype or L4
 * renames an entry point, this file stops compiling instead of the head
 * quietly calling something with the wrong shape.
 *
 * It is a translation unit of its own so the MTP cycle and the head can be
 * linked without a backend -- tests/test_qwen4exp_mtp supplies its own
 * pass-through hooks and its own GPU tensor API -- while the engine build gets
 * the real ones by linking this alongside ds4_metal.o or ds4_cuda.o.
 *
 * `matmul_q8_0` is the DECODE-ORDER entry, not the upstream one.  The upstream
 * ds4_gpu_matmul_q8_0_tensor picks its kernel by row count: at one row it is the
 * decode kernel, and above one row it takes a fused-dequant or a cuBLAS GEMM
 * over an expanded copy of the weight.  The head calls it twice per step -- the
 * eh_proj at n_hc rows and the borrowed LM head at one -- so the eh_proj was on
 * the wide side of that line while the tower's own projections, which route
 * through ds4_qwen4exp_matmul.h, were not.
 *
 * That cost both things at once.  Numerically, row s of the head's four-row
 * eh_proj was not the row a one-row call would have produced, in the one op of
 * the head that sees a batch at all.  In time, the wide side measured 6.9 ms
 * per head step against a weight of 13.9 MiB -- 36%% of the step for a read
 * that is 0.05 ms of bandwidth.  Routing it through ds4_qwen4exp_matmul_q8_0
 * gives the head the tower's rule: every row is the one-row reduction, and the
 * four rows share one read of the weight.
 *
 * At one row the two entries are already the same arithmetic -- the same
 * per-row kernel over the same quantisation, whose two spellings reduce an
 * exact maximum -- so the LM head's numbers do not move.
 *
 * `block` is the graph's head-block runner.  It was NULL while the 49th block
 * was unwritten; it is not any more, and leaving it NULL now would make the
 * head refuse at its own guard rather than run.  A caller with a different
 * graph still overrides it after this returns.
 */

#include "ds4_qwen4exp_mtp.h"

#ifndef DS4_NO_GPU
#include "ds4_qwen4exp_matmul.h"
#endif

#ifndef DS4_NO_GPU

/* The CUDA Qwen object supplies this symbol.  Metal and ROCm builds leave the
 * weak reference empty and retain the portable tensor-copy packer in
 * ds4_qwen4exp_mtp.c. */
#if !defined(__APPLE__) && (defined(__GNUC__) || defined(__clang__))
extern int ds4_gpu_mtp_select_vocab(ds4_gpu_tensor *, ds4_gpu_tensor *, const ds4_gpu_tensor *,
    uint32_t, uint32_t, int, uint32_t, uint32_t, uint32_t *) __attribute__((weak));
extern int ds4_gpu_mtp_indexed_q8(ds4_gpu_tensor *, const void *, uint64_t, uint64_t,
    uint64_t, uint64_t, const ds4_gpu_tensor *, const ds4_gpu_tensor *, uint32_t) __attribute__((weak));
extern int ds4_gpu_qwen4exp_ehx_pack_tensor(
        ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *,
        uint32_t, uint32_t, uint32_t) __attribute__((weak));
#endif

/* The hook takes uint64_t rows because that is the upstream entry's type; the
 * decode-order entry takes uint32_t.  The head is built for at most
 * DS4_QWEN4EXP_MTP_MAX_COMMIT rows and the widest call is that many times n_hc,
 * so the narrowing cannot lose a row -- but a caller that hands over more than
 * a uint32_t holds is refused here rather than truncated into a short matmul. */
static int mtp_matmul_q8_0_decode_rows(ds4_gpu_tensor *out,
                                       const void *model_map,
                                       uint64_t model_size,
                                       uint64_t weight_offset,
                                       uint64_t in_dim, uint64_t out_dim,
                                       const ds4_gpu_tensor *x,
                                       uint64_t n_tok) {
    if (n_tok == 0u || n_tok > UINT32_MAX) return 0;
    return ds4_qwen4exp_matmul_q8_0(out, model_map, model_size, weight_offset,
                                    in_dim, out_dim, x, (uint32_t)n_tok);
}

void ds4_qwen4exp_mtp_default_hooks(ds4_qwen4exp_mtp_gpu_hooks *hooks) {
    hooks->rms_norm    = ds4_gpu_qwen4exp_rms_norm_tensor;
    hooks->hc_mixer    = ds4_gpu_qwen4exp_hc_mixer_tensor;
    hooks->embed       = ds4_gpu_qwen4exp_embed_tokens_hc_tensor;
#if !defined(__APPLE__) && (defined(__GNUC__) || defined(__clang__))
    hooks->ehx_pack    = ds4_gpu_qwen4exp_ehx_pack_tensor;
    hooks->select_vocab = ds4_gpu_mtp_select_vocab;
    hooks->matmul_indexed = ds4_gpu_mtp_indexed_q8;
#else
    hooks->ehx_pack    = NULL;
    hooks->select_vocab = NULL;
    hooks->matmul_indexed = NULL;
#endif
    hooks->matmul_q8_0 = mtp_matmul_q8_0_decode_rows;
    hooks->block       = ds4_qwen4exp_graph_head_block;
}

#endif /* DS4_NO_GPU */
