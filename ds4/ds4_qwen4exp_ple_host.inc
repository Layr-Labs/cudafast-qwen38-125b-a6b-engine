/* Backend-independent qwen4exp PLE block composite.
 *
 * Included at the end of ds4_metal.m and ds4_cuda_qwen4exp.cu, next to
 * ds4_qwen4exp_hc_host.inc and for the same reason: the op ORDER of the block
 * is written once, out of the per-op wrappers each backend already provides,
 * so a divergence from Qwen4ExpPLELayer would have to be introduced in one
 * place rather than two.
 *
 * The order below is Qwen4ExpNGram.swift:327-339 line for line.
 */

#include "ds4_qwen4exp_matmul.h"

#ifdef __cplusplus
extern "C" {
#endif

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
        int                   round_bf16) {
    if (!hyper || !conv_state || !key_scratch || !aux_scratch ||
        !value_scratch || !ngram_rows || !key_weight || !value_weight ||
        !norm_key || !norm_query || !norm_conv || !conv_weight ||
        n_embd == 0 || n_hc == 0 || ple_embd == 0 || conv_kernel < 2u ||
        dilation == 0 || rows == 0) {
        return 0;
    }

    const uint32_t wide = n_hc * n_embd;

    /* key = normKey(keyProj(embedded)), reshaped [hc][embd].  The reshape is
     * free: the 10240-wide row already IS the four streams end to end, which
     * is the layout every other qwen4exp activation uses. */
    if (!ds4_qwen4exp_matmul_q8_0(key_scratch, key_weight->map,
                                    key_weight->map_size, key_weight->offset,
                                    ple_embd, wide, ngram_rows, rows)) {
        return 0;
    }
    if (!ds4_gpu_qwen4exp_rms_norm_tensor(key_scratch, key_scratch,
                                          norm_key->map, norm_key->map_size,
                                          norm_key->offset,
                                          wide, n_embd, rows, eps,
                                          norm_key_bias, round_bf16)) {
        return 0;
    }

    /* value = valueProj(embedded): 2560 wide, no norm, shared by the streams. */
    if (!ds4_qwen4exp_matmul_q8_0(value_scratch, value_weight->map,
                                    value_weight->map_size,
                                    value_weight->offset, ple_embd, n_embd,
                                    ngram_rows, rows)) {
        return 0;
    }

    /* query = normQuery(hidden): the stream ENTERING the layer, grouped so
     * each hyper-connection stream carries its own statistic.  `hyper` is not
     * written here; the block's only write to it is the accumulate below. */
    if (!ds4_gpu_qwen4exp_rms_norm_tensor(aux_scratch, hyper,
                                          norm_query->map,
                                          norm_query->map_size,
                                          norm_query->offset,
                                          wide, n_embd, rows, eps,
                                          norm_query_bias, round_bf16)) {
        return 0;
    }

    /* gated = sigmoid(signed_sqrt(dot(key, query) / sqrt(E))) * value, in
     * place over the key rows: the gate kernel finishes reading a stream
     * before it writes it. */
    if (!ds4_gpu_qwen4exp_ple_gate_tensor(key_scratch, key_scratch,
                                          aux_scratch, value_scratch,
                                          n_embd, n_hc, rows)) {
        return 0;
    }

    /* normConv(gated) over the query rows, which are dead from here. */
    if (!ds4_gpu_qwen4exp_rms_norm_tensor(aux_scratch, key_scratch,
                                          norm_conv->map, norm_conv->map_size,
                                          norm_conv->offset,
                                          wide, n_embd, rows, eps,
                                          norm_conv_bias, round_bf16)) {
        return 0;
    }

    /* hyper += gated + silu(conv1d(normConv(gated))), and roll the window. */
    return ds4_gpu_qwen4exp_ple_conv_tensor(hyper, conv_state, conv_snapshot,
                                            n_snapshot_rows, key_scratch,
                                            aux_scratch, conv_weight->map,
                                            conv_weight->map_size,
                                            conv_weight->offset, wide,
                                            conv_kernel, dilation, rows);
}

#ifdef __cplusplus
}
#endif
