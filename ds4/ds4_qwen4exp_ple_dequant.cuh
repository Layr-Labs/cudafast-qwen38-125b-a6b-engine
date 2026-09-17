/*
 * Device-side IQ4_NL dequantisation for the Qwen4Exp PLE gather.
 *
 * The host PLE path reads 90-byte IQ4_NL rows out of the mmap'd n-gram
 * table, expands them to 640-byte float rows on the CPU, then uploads the
 * floats.  The expansion is 7.1x: every gather moves seven times the bytes
 * the table actually stores, and the CPU does work the device can do in a
 * few microseconds.  This header carries the kernel and its launcher so the
 * gather can upload the raw quant rows and dequantise them in place on the
 * device.
 *
 * IQ4_NL block layout (ggml block_iq4_nl, QK4_NL = 32):
 *   bytes [0,2)   fp16 scale
 *   bytes [2,18)  16 nibble bytes: value j in [0,16) uses the LOW nibble of
 *                 qs[j], value j+16 uses the HIGH nibble of qs[j].
 *   value = scale * kvalues_iq4nl[nibble]
 *
 * The launcher is plain CUDA C++ taking raw device pointers; the ds4_gpu_*
 * wrapper that extracts tensor pointers lives in ds4_cuda_qwen4exp.cu.
 */

#ifndef DS4_QWEN4EXP_PLE_DEQUANT_CUH
#define DS4_QWEN4EXP_PLE_DEQUANT_CUH

#include <cuda_fp16.h>
#include <cstdint>
#include <cstddef>

#ifndef DS4_PLE_IQ4_NL_BLOCK_BYTES
#define DS4_PLE_IQ4_NL_BLOCK_BYTES 18   /* fp16 scale + 16 nibble bytes */
#endif

__device__ __constant__ static const int8_t
    ds4_ple_kvalues_iq4nl[16] = {
        -127, -104, -83, -65, -49, -35, -22, -10,
        1, 13, 25, 38, 53, 69, 89, 113
    };

/*
 * One block per row, one thread per output value.  Row layout is a run of
 * 18-byte blocks; row_vals is a multiple of 32 (160 on this checkpoint).
 * src/dst are device pointers; src rows are packed back to back.
 */
__global__ static void ds4_ple_iq4nl_dequant_kernel(
        const uint8_t * __restrict src,
        float * __restrict dst,
        int row_bytes,
        int row_vals) {
    const int row = blockIdx.x;
    const uint8_t * __restrict s = src + (size_t)row * (size_t)row_bytes;
    float * __restrict d = dst + (size_t)row * (size_t)row_vals;
    for (int v = threadIdx.x; v < row_vals; v += blockDim.x) {
        const int b = v >> 5;          /* 32 values per block */
        const int j = v & 31;
        const uint8_t *blk = s + (size_t)b * DS4_PLE_IQ4_NL_BLOCK_BYTES;
        const float scale = __half2float(*(const __half *)blk);
        const uint8_t byte = blk[2 + (j & 15)];
        const int nib = (j < 16) ? (byte & 0xF) : (byte >> 4);
        d[v] = scale * (float)ds4_ple_kvalues_iq4nl[nib];
    }
}

/*
 * Launch the dequant for n_rows packed IQ4_NL rows.  Returns 0 on a bad
 * argument; the launch itself is stream-ordered like every other ds4
 * kernel.  threads is clamped to the row width so a 160-value row uses
 * exactly five warps.
 */
static inline int ds4_ple_iq4nl_dequant_launch(
        const void *src, float *dst, int n_rows,
        int row_bytes, int row_vals, cudaStream_t stream) {
    if (!src || !dst || n_rows <= 0 || row_vals <= 0 || (row_vals & 31)) {
        return 0;
    }
    int threads = row_vals < 256 ? row_vals : 256;
    ds4_ple_iq4nl_dequant_kernel<<<n_rows, threads, 0, stream>>>(
        (const uint8_t *)src, dst, row_bytes, row_vals);
    return 1;
}

#endif /* DS4_QWEN4EXP_PLE_DEQUANT_CUH */
