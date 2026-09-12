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
#include <stdlib.h>
#include <string.h>

#ifndef DS4_NO_GPU
#include "ds4_qwen4exp_matmul.h"
#endif

#ifndef DS4_NO_GPU

/* The CUDA Qwen object supplies this symbol.  Metal and ROCm builds leave the
 * weak reference empty and retain the portable tensor-copy packer in
 * ds4_qwen4exp_mtp.c. */
#if !defined(__APPLE__) && (defined(__GNUC__) || defined(__clang__))
extern int ds4_gpu_mtp_native_screen_init(uint32_t, uint64_t *, uint32_t *) __attribute__((weak));
extern int ds4_gpu_mtp_native_screen(ds4_gpu_tensor *, ds4_gpu_tensor *, ds4_gpu_tensor *,
    const void *, uint64_t, uint64_t, uint32_t, uint32_t, uint32_t, uint32_t,
    const ds4_gpu_tensor *) __attribute__((weak));
extern int ds4_gpu_mtp_native_map(ds4_gpu_tensor *, const ds4_gpu_tensor *,
    const ds4_gpu_tensor *, uint32_t, uint32_t) __attribute__((weak));
extern int ds4_gpu_qwen4exp_ehx_pack_tensor(
        ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *,
        uint32_t, uint32_t, uint32_t) __attribute__((weak));
#endif

#if !defined(__APPLE__) && (defined(__GNUC__) || defined(__clang__))
/* The opaque tensor handles are immutable, head-owned allocations. No tensor
 * contents query here: that API synchronizes CUDA. Binding changes retire all
 * executables before a new warm pass; values written into these buffers do not
 * enter the key and are consumed anew on every replay. */
typedef struct {
    const void *map, *graph, *cache;
    ds4_gpu_tensor *tensor[6];
    uint64_t bytes[6], size, norm, down, up;
    uint32_t embd, hc, lowrank, max_tokens, block;
    float eps, bias;
    int round;
} mtp_tail_binding;

static void mtp_tail_graph_release(ds4_qwen4exp_mtp_head *h) {
    if (h->tail_graph_state) {
        (void)ds4_gpu_synchronize();
        ds4_gpu_decode_graphs_invalidate();
        free(h->tail_graph_state);
    }
    h->tail_graph_state = NULL;
    h->tail_graph_release = NULL;
}

static int mtp_mix_tail(ds4_qwen4exp_mtp_head *h, uint32_t first_row,
                        uint32_t rows, int copy_last_row) {
    /* Reserve the otherwise unused QSA island of the production head layer.
     * The transformer head itself owns island 3; target layers are 0..47.
     * Other shapes/backends and custom mixer hooks retain the eager contract. */
    if (!h || h->hooks.hc_mixer != ds4_gpu_qwen4exp_hc_mixer_tensor ||
        h->block_index != 48u || rows != 1u || first_row > 1u ||
        first_row >= h->max_tokens ||
        h->n_embd != 2560u || h->n_hc != 4u ||
        getenv("DS4_MTP_NO_TAIL_GRAPH") != NULL ||
        /* These diagnostics choose live projection dispatch. A graph captured
         * with all absent must never mask a subsequent explicit override. */
        getenv("DS4_CUDA_NO_Q8_DP4A") != NULL ||
        getenv("DS4_QWEN4EXP_NO_ROW_TILE") != NULL ||
        getenv("DS4_Q8_NO_STREAM_LOADS") != NULL ||
        getenv("DS4_Q8_NO_HC_WARP_PAIR") != NULL ||
        getenv("DS4_QWEN4EXP_PAIR_LANES_R2") != NULL ||
        getenv("DS4_QWEN4EXP_Q8_WIDE_BLOCKS") != NULL ||
        !ds4_gpu_decode_graphs_supported())
        return ds4_qwen4exp_mtp_head_mix_eager(h, first_row, rows, copy_last_row);

    mtp_tail_binding binding;
    memset(&binding, 0, sizeof(binding));
    binding.map = h->head_map; binding.size = h->head_size;
    binding.graph = h->graph; binding.cache = h->cache;
    binding.norm = h->hc_head_norm_offset;
    binding.down = h->hc_head_down_offset; binding.up = h->hc_head_up_offset;
    binding.embd = h->n_embd; binding.hc = h->n_hc;
    binding.lowrank = h->n_lowrank; binding.max_tokens = h->max_tokens;
    binding.block = h->block_index; binding.eps = h->rms_eps;
    binding.bias = h->weight_bias; binding.round = h->round_bf16;
    ds4_gpu_tensor *tensor[6] = {h->t_hyper, h->t_h_normed, h->t_mix_normed,
                               h->t_mix_lowrank, h->t_mix_wide, h->t_sample};
    for (unsigned i = 0; i < 6u; ++i) {
        binding.tensor[i] = tensor[i];
        binding.bytes[i] = ds4_gpu_tensor_bytes(tensor[i]);
    }
    if (h->tail_graph_state &&
        memcmp(h->tail_graph_state, &binding, sizeof(binding)) != 0) {
        if (!ds4_gpu_synchronize()) return 0;
        ds4_gpu_decode_graphs_invalidate();
        memcpy(h->tail_graph_state, &binding, sizeof(binding));
    } else if (!h->tail_graph_state) {
        h->tail_graph_state = malloc(sizeof(binding));
        if (!h->tail_graph_state)
            return ds4_qwen4exp_mtp_head_mix_eager(h, first_row, rows, copy_last_row);
        memcpy(h->tail_graph_state, &binding, sizeof(binding));
        h->tail_graph_release = mtp_tail_graph_release;
    }
    ds4_decode_graph_key key;
    memset(&key, 0, sizeof(key));
    key.il = 48u; key.island = 2u;
    key.variant = copy_last_row ? first_row + 1u : 0u;
    key._pad = 0x4d545054u; /* MTPT: separate from any QSA key. */
    key.cur_hc = h; key.after_attn_hc = h->tail_graph_state;
    key.after_ffn_hc = h->t_hyper; key.attn_norm = h->t_sample;
    const int mode = ds4_gpu_decode_graph_begin(&key);
    if (mode == 1) return 1;
    const int ok = ds4_qwen4exp_mtp_head_mix_eager(h, first_row, rows, copy_last_row);
    if (mode != 0) return ok;
    if (!ok) {
        ds4_gpu_decode_graph_abort(&key);
        return 0; /* Preserve a real copy/mixer error; do not retry it. */
    }
    if (ds4_gpu_decode_graph_end(&key) == 0) return 1;
    /* Failed finalization has executed no work. Preserve the eager error path
     * and keep projection/selection strictly outside the capture interval. */
    return ds4_qwen4exp_mtp_head_mix_eager(h, first_row, rows, copy_last_row);
}
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
    hooks->mix_tail = mtp_mix_tail;
    hooks->native_init = ds4_gpu_mtp_native_screen_init;
    hooks->native_screen = ds4_gpu_mtp_native_screen;
    hooks->native_map = ds4_gpu_mtp_native_map;
#else
    hooks->mix_tail    = NULL;
    hooks->ehx_pack    = NULL;
    hooks->native_init = NULL;
    hooks->native_screen = NULL;
    hooks->native_map = NULL;
#endif
    hooks->matmul_q8_0 = mtp_matmul_q8_0_decode_rows;
    hooks->block       = ds4_qwen4exp_graph_head_block;
}

#endif /* DS4_NO_GPU */
