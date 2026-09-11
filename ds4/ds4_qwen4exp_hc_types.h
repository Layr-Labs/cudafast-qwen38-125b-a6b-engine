#ifndef DS4_QWEN4EXP_HC_TYPES_H
#define DS4_QWEN4EXP_HC_TYPES_H

/*
 * The one qwen4exp hyper-connection INJECT weight type table.
 *
 * Same contract as ds4_qwen4exp_moe_types.h and for the same reason: the
 * loader builds its accepted-type list from this macro and the Metal and CUDA
 * inject kernels build their dispatch switch from it, so the two cannot drift.
 * A type admitted here without a matching ds4_qwen4exp_<name>_value accessor
 * fails to compile on both backends; a type left out is refused by name at load
 * time rather than reaching a kernel that cannot read it.
 *
 * The set is narrower than the MoE's because only two types occur on
 * blk.N.hc_{attn,ffn}_inject.weight:
 *
 *   f32   the TARGET stores it dense
 *   q8_0  the MTP HEAD stores it quantised
 *
 * That difference is the whole reason this table exists.  The loader accepted
 * both from the start, while the kernel read dense F32 only, so a head block
 * reached the kernel and it returned zero -- the same shape of gap L5c closed
 * on the expert path.
 *
 * This file is read at run time and prepended to the Metal library source, so
 * it must stay valid Metal: preprocessor directives only.  It expands to the
 * accessors metal/qwen4exp_moe.metal already defines, which is why that file is
 * concatenated ahead of metal/qwen4exp_hc.metal.
 */
#define DS4_QWEN4EXP_HC_INJECT_TYPES(X) \
    X(f32,  0) \
    X(q8_0, 8)

#endif /* DS4_QWEN4EXP_HC_TYPES_H */
