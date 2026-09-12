/*
 * Bridge from ds4_cuda_qwen4exp.cu back into the CUDA backend.
 *
 * The Qwen4-Exp kernels live in their own translation unit because that unit
 * is compiled without --use_fast_math (see the Makefile and the header of
 * ds4_cuda_qwen4exp.cu).  It duplicates the leaf helpers it needs, the way
 * ds4_rocm.cu duplicates them for ROCm, and reaches the two helpers that read
 * backend state through the calls below.  ds4_cuda.cu defines both.
 */
#ifndef DS4_CUDA_QWEN4EXP_CUH
#define DS4_CUDA_QWEN4EXP_CUH

#include <stdint.h>
#include <cuda_runtime.h>

/* cuda_decode_stream(): the legacy NULL stream in eager mode, the capture
 * stream while a decode-island capture or replay is in flight. */
cudaStream_t ds4_cuda_qwen4exp_decode_stream(void);

/* cuda_resolve_weight_ptr(): the mapped model on one GPU, the per-device
 * weight cache when the model is split across tiers. */
const char *ds4_cuda_qwen4exp_weight_ptr(
        const void *model_map,
        uint64_t    offset,
        uint64_t    bytes,
        int         logical_tier,
        const char *label);

/* 1 when the Q8_0 row-exact matmul takes the int8 MMA tile at this width. */
int ds4_cuda_qwen4exp_q8_mma_active(uint32_t n_rows);

#endif /* DS4_CUDA_QWEN4EXP_CUH */
