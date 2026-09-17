// Double-buffered pinned staging for the PLE gather-row upload.
//
// The decode loop repacks the host n-gram row staging buffer every step and
// hands it to the device gather.  A single unpinned staging buffer serializes
// the host against the copy engine twice over: the blocking cudaMemcpy stages
// through a driver bounce buffer AND stalls the host until the copy drains.
// Pinned buffers remove the bounce; two of them remove the stall — the host
// packs into the buffer the copy engine is not reading, and the only wait is
// on an event recorded two uploads ago, which in steady state has long
// completed.
//
// The per-buffer events exist only to license host reuse.  Device-side
// consumers stay stream-ordered: the upload is issued on the consumer's own
// stream, so no device-side wait is needed at all.

#ifndef DS4_QWEN4EXP_PLE_STAGE_CUH
#define DS4_QWEN4EXP_PLE_STAGE_CUH

#include <cuda_runtime.h>
#include <stddef.h>

struct ds4_ple_stage {
    void       *buf[2];   // pinned staging buffers
    cudaEvent_t ev[2];    // recorded after each buffer's upload
    int         par;      // index of the buffer to pack next
    size_t      bytes;    // per-buffer capacity
    int         live;     // nonzero once both buffers and events exist
};

static inline void ds4_ple_stage_init(ds4_ple_stage *s) {
    s->buf[0] = s->buf[1] = NULL;
    s->ev[0]  = s->ev[1]  = 0;
    s->par    = 0;
    s->bytes  = 0;
    s->live   = 0;
}

// Allocate both pinned buffers and their events.  Returns 0 on success; on
// any failure releases what it took and returns nonzero so the caller can
// fall back to the plain unpinned path.
static inline int ds4_ple_stage_create(ds4_ple_stage *s, size_t bytes) {
    ds4_ple_stage_init(s);
    s->bytes = bytes;
    for (int i = 0; i < 2; i++) {
        if (cudaHostAlloc(&s->buf[i], bytes, cudaHostAllocDefault) != cudaSuccess)
            goto fail;
        if (cudaEventCreateWithFlags(&s->ev[i], cudaEventDisableTiming) != cudaSuccess)
            goto fail;
    }
    s->live = 1;
    return 0;
fail:
    for (int i = 0; i < 2; i++) {
        if (s->ev[i])  cudaEventDestroy(s->ev[i]);
        if (s->buf[i]) cudaFreeHost(s->buf[i]);
    }
    ds4_ple_stage_init(s);
    return 1;
}

// Release both buffers and events.  The recorded events are drained first so
// no in-flight upload still reads a buffer being freed.
static inline void ds4_ple_stage_destroy(ds4_ple_stage *s) {
    if (!s->live) return;
    for (int i = 0; i < 2; i++) {
        cudaEventSynchronize(s->ev[i]);
        cudaEventDestroy(s->ev[i]);
        cudaFreeHost(s->buf[i]);
    }
    ds4_ple_stage_init(s);
}

// Return the buffer the host packs into this step.  A synchronizing event
// that was never recorded completes immediately, so the first two calls never
// block; afterwards each call waits only on the upload issued two steps ago.
static inline void *ds4_ple_stage_acquire(ds4_ple_stage *s) {
    int i = s->par;
    cudaEventSynchronize(s->ev[i]);
    return s->buf[i];
}

// Issue the asynchronous upload of the just-packed buffer to `dst` on
// `stream`, record the buffer's event, and flip parity.  The consumer runs on
// the same stream, so no device-side wait is needed: the copy is ordered
// ahead of the gather kernels by the stream itself.  Returns the memcpy
// status; on failure parity is left so the buffer stays owned by the copy
// engine for the caller's abort path.
static inline cudaError_t ds4_ple_stage_upload(ds4_ple_stage *s, void *dst,
                                               size_t bytes,
                                               cudaStream_t stream) {
    int i = s->par;
    cudaError_t err = cudaMemcpyAsync(dst, s->buf[i], bytes,
                                      cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) return err;
    cudaEventRecord(s->ev[i], stream);
    s->par = i ^ 1;
    return cudaSuccess;
}

#endif // DS4_QWEN4EXP_PLE_STAGE_CUH
