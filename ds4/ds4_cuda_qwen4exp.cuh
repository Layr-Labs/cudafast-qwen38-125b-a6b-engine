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

/* ------------------------------------------------------------------------
 * Programmatic Dependent Launch (PDL) for the decode-round norm ->
 * pair-projection edges, in its complete form.
 *
 * A decode round pays ~2.1 ms/round to graph dependency edges, and the
 * norm/quantize -> pair-projection subset is the one where the consumer's
 * first WEIGHT loads sit serialized behind a tiny producer although their
 * addresses do not depend on the producer's output.  The three parts:
 *
 *   1. the producer (the tiny norm/quant kernel) calls
 *      QWEN4EXP_PDL_TRIGGER() at the top of its body -- the dependent may
 *      LAUNCH now;
 *   2. the consumer (the pair projection) is launched through
 *      QWEN4EXP_LAUNCH_PDL, which sets the programmatic stream
 *      serialization attribute -- the recorded edge relaxes to the trigger;
 *   3. the consumer issues its first WEIGHT loads into registers, THEN
 *      calls QWEN4EXP_PDL_SYNC(), and reads activations only after it.
 *
 * The fence waits for the producer's full completion and flushed writes, so
 * the data dependency moves from the graph edge into the kernel body: the
 * edge is not violated, it is delegated (that is what the attribute names).
 * Weight tensors are read-only session slabs, so a weight load that runs
 * before the fence returns identical bytes; an activation load that ran
 * before it could read a half-written value, which is why the fence, and
 * every consumer below keeps every activation read after it.
 *
 * THE DEADLOCK RULE, on the trigger: a producer whose grid exceeds what the
 * device can hold at once may never carry one.  The dependent's blocks come
 * up at the trigger, stall at the fence, and hold the SM slots the
 * producer's own later waves still need -- nothing retires and nothing
 * schedules.  A trigger is therefore only ever added to a producer that is
 * single-wave at every width where a PSS-attributed consumer can follow it
 * on the stream.  That bound lives in the producers' bodies, not in a
 * convention at the launch sites: every trigger below is row-gated to the
 * same <= 2 the converted launch sites fire at, and the decode quantizer --
 * which the graph-capture ceiling allows up to 7 rows -- bounds its grid to
 * one wave as well, so a verify or prefill launch never carries a live
 * trigger at all.
 *
 * THE .NC RULE, on the fence: cudaGridDependencySynchronize() orders the
 * acquire it emits, but ld.global.nc (const/__restrict__-qualified) loads
 * are not ordered by it -- the compiler may hoist one above the fence and
 * read stale data (llama.cpp PR 24030 documented the SASS).  Converted
 * consumers therefore carry no __restrict__ on their ACTIVATION pointers.
 * Weight pointers keep it wherever they had it: hoisting a weight load
 * above the fence is the whole feature.
 *
 * Capture: the launch attribute is recorded into the kernel node
 * (CUDA >= 12.3; the box toolkit is 13.0), so decode-graph islands capture
 * and replay with the programmatic edges intact.  An older toolkit falls
 * back to the plain triple-chevron launch, which no kernel depends on
 * receiving the attribute.  The fence is a no-op in a plainly-launched
 * kernel and the trigger is a no-op with no dependent launch, so the valve
 * below restores the exact pre-PDL behaviour.
 * ------------------------------------------------------------------------ */

/* The fence and the trigger.  Hopper (9.0) is the first architecture with
 * griddepcontrol, and CUDA 11.8 (CUDART_VERSION 11080) is the first runtime
 * that declares the two intrinsics.  Host passes see no __CUDA_ARCH__ and a
 * toolkit older than 11.8 fails the version arm, so both compile to nothing
 * there -- such a toolkit takes the plain-launch arm of
 * QWEN4EXP_LAUNCH_PDL below, where a fence and a trigger are no-ops
 * anyway. */
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && CUDART_VERSION >= 11080
#define QWEN4EXP_PDL_SYNC() cudaGridDependencySynchronize()
#define QWEN4EXP_PDL_TRIGGER() cudaTriggerProgrammaticLaunchCompletion()
#else
#define QWEN4EXP_PDL_SYNC() ((void)0)
#define QWEN4EXP_PDL_TRIGGER() ((void)0)
#endif

/* The valve, resolved once per process and shared by both translation units
 * (defined in ds4_cuda.cu, next to cuda_q8_mma_available).  DS4_QWEN4EXP_
 * NO_PDL_PREFETCH set to a non-zero value drops the launch attribute: plain
 * launches, the triggers fire into nothing, the fences are no-ops.  A device
 * too old to have compiled the fence in (compute capability < 9.0) is off the
 * same way. */
int ds4_qwen4exp_pdl_enabled(void);

#if CUDART_VERSION >= 12030
/* One launch path for the converted consumers: cudaLaunchKernelEx, with the
 * programmatic stream serialization attribute when PDL is on and without it
 * when it is not -- a zero-attribute cudaLaunchKernelEx is the plain launch.
 * Failures reach the caller's cudaGetLastError() check unchanged. */
#define QWEN4EXP_LAUNCH_PDL(KERNEL, GRID, BLOCK, SMEM, STREAM, ...)        \
    do {                                                                   \
        cudaLaunchAttribute qw_attr[1];                                    \
        qw_attr[0].id =                                                    \
            cudaLaunchAttributeProgrammaticStreamSerialization;            \
        qw_attr[0].val.programmaticStreamSerializationAllowed = 1;          \
        const int qw_pdl = ds4_qwen4exp_pdl_enabled();                     \
        cudaLaunchConfig_t qw_cfg;                                         \
        qw_cfg.gridDim = (GRID);                                           \
        qw_cfg.blockDim = (BLOCK);                                         \
        qw_cfg.dynamicSmemBytes = (SMEM);                                  \
        qw_cfg.stream = (STREAM);                                          \
        qw_cfg.attrs = qw_pdl ? qw_attr : NULL;                            \
        qw_cfg.numAttrs = qw_pdl ? 1 : 0;                                  \
        (void)cudaLaunchKernelEx(&qw_cfg, KERNEL, __VA_ARGS__);            \
    } while (0)
#else
#define QWEN4EXP_LAUNCH_PDL(KERNEL, GRID, BLOCK, SMEM, STREAM, ...)        \
    do {                                                                   \
        (void)ds4_qwen4exp_pdl_enabled();                                  \
        KERNEL<<<(GRID), (BLOCK), (SMEM), (STREAM)>>>(__VA_ARGS__);        \
    } while (0)
#endif

#endif /* DS4_CUDA_QWEN4EXP_CUH */
