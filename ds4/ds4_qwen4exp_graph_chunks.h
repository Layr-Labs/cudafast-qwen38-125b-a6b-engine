#ifndef DS4_QWEN4EXP_GRAPH_CHUNKS_H
#define DS4_QWEN4EXP_GRAPH_CHUNKS_H

/*
 * The decode-round chunk schedule: which consecutive layer range each captured
 * piece of the target stack covers.  Kept pure over (n_layer, chunks, piece)
 * so a host test can pin the shapes with no GPU and no engine state; the chunk
 * loop in ds4_qwen4exp_graph.inc is the only engine-side caller, and the
 * read-once valve that picks the schedule lives beside it there.
 *
 * uniform:    per = ceil(n_layer/chunks), piece c = [c*per, c*per+per).  The
 *             pre-headstart arithmetic, unchanged; what
 *             DS4_CUDA_DECODE_GRAPH_NO_HEADSTART=1 restores, and what a stack
 *             of fewer than two layers, or a single piece, falls back to.
 *
 * headstart:  piece 0 = [0,1) exactly; the other chunks-1 pieces split the
 *             n_layer-1 remaining layers, rest_per =
 *             ceil((n_layer-1)/(chunks-1)), piece c>=1 =
 *             [1+(c-1)*rest_per, 1+c*rest_per).
 *
 * WHY A ONE-LAYER FIRST PIECE.  Only the first cudaGraphLaunch of a round
 * meets an idle GPU; every later one queues behind device work already
 * running.  A one-layer piece 0 (~31 nodes, ~20-30 us of host launch time)
 * starts the device that much sooner than a uniform 48/4 piece 0 (~374 nodes,
 * ~0.16-0.25 ms), and piece 0's ~0.8 ms of device work hides the ~0.2-0.33 ms
 * launch of the 16-layer piece behind it.  The default 4 pieces over 48
 * layers is [1,16,16,15].
 *
 * The schedule is a table: pieces are consecutive layer ranges whose work
 * reaches the stream in stack order whether replayed or encoded eagerly, so
 * the kernel sequence is identical to the uniform split's and to the single
 * whole-stack capture's.  The chunk cache keys a piece by its INDEX, never by
 * its range, so the schedule must be constant for the life of the process --
 * which is why the valve selecting it is read once.
 */

#include <stdbool.h>
#include <stdint.h>

/* Range [il_first, il_last) of piece `chunk` of `n_layer` layers cut into
 * `chunks` >= 1 consecutive pieces.  Returns false when the piece is empty
 * (il_first >= n_layer): the starts are strictly increasing, so the caller
 * stops -- and for n_layer >= 1 piece 0 is never empty, while a later piece
 * is empty only when the piece before it clamped its end to n_layer, so the
 * visited pieces always cover the stack exactly. */
static inline bool qw_decode_graph_chunk_range(uint32_t chunk,
                                               uint32_t n_layer,
                                               uint32_t chunks,
                                               bool headstart,
                                               uint32_t *il_first,
                                               uint32_t *il_last) {
    if (headstart && (n_layer < 2u || chunks < 2u)) headstart = false;
    if (headstart) {
        /* Piece 0 = [0,1); the other chunks-1 pieces split the rest.  Here
         * n_layer >= 2 and chunks >= 2, so the divisor chunks-1 is >= 1. */
        const uint32_t rest_per = (n_layer - 1u + chunks - 2u) / (chunks - 1u);
        if (chunk == 0u) {
            *il_first = 0u;
            *il_last = 1u;
        } else {
            *il_first = 1u + (chunk - 1u) * rest_per;
            *il_last = *il_first + rest_per;
        }
    } else {
        /* The uniform split, literally the pre-headstart arithmetic. */
        const uint32_t per = (n_layer + chunks - 1u) / chunks;
        *il_first = chunk * per;
        *il_last = *il_first + per;
    }
    if (*il_first >= n_layer) return false;
    if (*il_last > n_layer) *il_last = n_layer;
    return true;
}

#endif /* DS4_QWEN4EXP_GRAPH_CHUNKS_H */
