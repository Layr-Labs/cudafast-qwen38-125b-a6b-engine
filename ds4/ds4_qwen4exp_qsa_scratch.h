#ifndef DS4_QWEN4EXP_QSA_SCRATCH_H
#define DS4_QWEN4EXP_QSA_SCRATCH_H

#include <stdint.h>

/* Shared by the CPU session planner and CUDA allocation check. */
static inline uint64_t ds4_qwen4exp_qsa_split_bytes(
        uint32_t n_tokens, uint32_t n_head, uint32_t head_dim,
        uint32_t max_count) {
    if (!n_tokens || !n_head || !max_count || head_dim < 32u ||
        head_dim > 1024u || (head_dim & (head_dim - 1u)) != 0u) return 0;
    const uint64_t tiles = ((uint64_t)max_count + head_dim - 1u) / head_dim;
    return (uint64_t)n_tokens * n_head * tiles *
           (2ull * head_dim + 2ull) * sizeof(float);
}

#endif
