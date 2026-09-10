/* Backend-independent qwen4exp host composites.
 *
 * Included at the end of ds4_metal.m and ds4_cuda.cu.  Both entry points are
 * built only out of the per-op wrappers each backend already provides, so
 * there is one copy of the op order and one place where a divergence from
 * Qwen4ExpGatedResidual would have to be introduced.
 */

#include "ds4_qwen4exp_matmul.h"

#ifdef __cplusplus
extern "C" {
#endif

int ds4_gpu_qwen4exp_hc_mixer_tensor(
        ds4_gpu_tensor       *mixed,
        ds4_gpu_tensor       *inject,
        ds4_gpu_tensor       *normed_scratch,
        ds4_gpu_tensor       *lowrank_scratch,
        ds4_gpu_tensor       *wide_scratch,
        const ds4_gpu_tensor *hyper,
        /* One slab per tensor.  The four weights of a mixer are four separate
         * GGUF tensors and a shard boundary can fall between any two of them,
         * so a single mapping with four offsets would read the right offsets
         * out of the wrong file -- plausible numbers, no refusal anywhere.
         * Same rule the expert and PLE paths already follow. */
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
        int                   round_bf16) {
    if (!mixed || !normed_scratch || !lowrank_scratch || !wide_scratch ||
        !hyper || !norm_weight || !down_weight || !up_weight ||
        n_embd == 0 || n_hc == 0 || n_lowrank == 0 ||
        rows == 0) {
        return 0;
    }

    const uint32_t wide = n_hc * n_embd;

    /* normed = hcNorm(hyper): grouped, one statistic per stream, one weight
     * over the flat row.  `hyper` itself is never written -- it is the
     * residual, and for the tower's last mixer it is also the stream the
     * native MTP head reads. */
    if (!ds4_gpu_qwen4exp_rms_norm_tensor(normed_scratch, hyper,
                                          norm_weight->map,
                                          norm_weight->map_size,
                                          norm_weight->offset,
                                          wide, n_embd, rows, eps,
                                          weight_bias, round_bf16)) {
        return 0;
    }

    /* Low-rank gate: 10240 -> 320, /n_hc, silu, 320 -> 10240.  The sigmoid
     * that closes it is fused into the mix reduction. */
    if (!ds4_qwen4exp_matmul_q8_0(lowrank_scratch, down_weight->map,
                                    down_weight->map_size,
                                    down_weight->offset, wide, n_lowrank,
                                    normed_scratch, rows)) {
        return 0;
    }
    if (!ds4_gpu_qwen4exp_scale_silu_tensor(lowrank_scratch,
                                            rows * n_lowrank,
                                            1.0f / (float)n_hc)) {
        return 0;
    }
    if (!ds4_qwen4exp_matmul_q8_0(wide_scratch, up_weight->map,
                                    up_weight->map_size,
                                    up_weight->offset, n_lowrank, wide,
                                    lowrank_scratch, rows)) {
        return 0;
    }
    if (!ds4_gpu_qwen4exp_hc_mix_tensor(mixed, normed_scratch, wide_scratch,
                                        n_embd, n_hc, rows)) {
        return 0;
    }

    /* The tower's final mixer has no inject head. */
    if (!inject) return 1;

    return ds4_gpu_qwen4exp_hc_inject_weights_tensor(inject, normed_scratch,
                                                     inject_weight,
                                                     n_embd, n_hc, rows);
}

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
        uint32_t              n_hc) {
    if (!out_hc || !rows_scratch || !tokens || !model_map ||
        n_vocab == 0 || n_tokens == 0 || n_embd == 0 || n_hc == 0) {
        return 0;
    }

    /* Upstream's quantized gather transfers unchanged; only the tile into the
     * hyper-connection streams is qwen4exp's, and DS4 already has that too. */
    if (!ds4_gpu_embed_tokens_quant_tensor(rows_scratch, tokens, model_map,
                                           model_size, weight_offset,
                                           weight_type, n_vocab, n_tokens,
                                           n_embd)) {
        return 0;
    }
    return ds4_gpu_repeat_hc_rows_tensor(out_hc, rows_scratch, n_tokens,
                                         n_embd, n_hc);
}

#ifdef __cplusplus
}
#endif
