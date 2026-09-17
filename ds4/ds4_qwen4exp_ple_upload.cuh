/*
 * Pinned staging and stream-ordered upload for the Qwen4Exp PLE gather.
 *
 * The gather's host staging buffer is a plain calloc, so the blocking
 * ds4_gpu_tensor_write that ships the dequantized rows does two things the
 * decode loop pays for: the driver stages the payload through an internal
 * bounce buffer (pageable copies cannot DMA), and the call stalls the host
 * until the copy completes even though every consumer of the rows is
 * stream-ordered behind it anyway.
 *
 * These helpers give the CUDA backend three pieces:
 *   - a page-locked (pinned) staging allocation, so the copy is a direct
 *     DMA with no bounce stage;
 *   - an async H2D on the decode stream, so the copy is ordered ahead of
 *     the PLE block kernels that read the rows without stalling the host;
 *   - an event the next gather waits on before it rewrites the staging
 *     buffer, which is what makes the async copy safe: the host can run
 *     ahead, but it can never overwrite bytes a copy is still reading.
 *
 * The wrappers that adapt these to ds4_gpu_tensor live in
 * ds4_cuda_qwen4exp.cu beside the other PLE entry points.
 */

#ifndef DS4_QWEN4EXP_PLE_UPLOAD_CUH
#define DS4_QWEN4EXP_PLE_UPLOAD_CUH

#include <cstddef>

/* Page-locked host allocation.  Returns NULL on failure so the caller can
 * fall back to a pageable buffer; the free side must match the alloc. */
static inline void *ds4_ple_pinned_alloc(size_t bytes) {
    void *p = NULL;
    if (bytes == 0 ||
        cudaHostAlloc(&p, bytes, cudaHostAllocDefault) != cudaSuccess) {
        return NULL;
    }
    return p;
}

static inline void ds4_ple_pinned_free(void *p) {
    if (p) cudaFreeHost(p);
}

/* One event tracks the most recent async upload.  DisableTiming keeps it
 * the cheapest kind: it exists only to order the next staging write after
 * the copy, never to measure anything. */
static cudaEvent_t ds4_ple_upload_event = NULL;

/* Queue an H2D copy on `stream` and record the event behind it.  The copy
 * is ordered after everything already on the stream and before whatever
 * the caller launches next, so the PLE block kernels see the rows with no
 * host-side wait.  Returns 0 on any failure. */
static inline int ds4_ple_upload_async(void *dst, const void *src,
                                       size_t bytes, cudaStream_t stream) {
    if (!dst || !src || bytes == 0) return 0;
    if (!ds4_ple_upload_event &&
        cudaEventCreateWithFlags(&ds4_ple_upload_event,
                                 cudaEventDisableTiming) != cudaSuccess) {
        return 0;
    }
    if (cudaMemcpyAsync(dst, src, bytes,
                        cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        return 0;
    }
    if (cudaEventRecord(ds4_ple_upload_event, stream) != cudaSuccess) {
        return 0;
    }
    return 1;
}

/* Block until the last queued upload (and the stream work ahead of it) has
 * completed.  Called before the staging buffer is rewritten; with no copy
 * ever queued it is a no-op. */
static inline int ds4_ple_upload_wait(void) {
    if (!ds4_ple_upload_event) return 1;
    return cudaEventSynchronize(ds4_ple_upload_event) == cudaSuccess;
}

#endif /* DS4_QWEN4EXP_PLE_UPLOAD_CUH */
