#ifndef DS4_QWEN4EXP_MOE_TYPES_H
#define DS4_QWEN4EXP_MOE_TYPES_H

/*
 * The one qwen4exp mixture-of-experts weight type table.
 *
 * The loader builds its accepted-type list from this macro, and the Metal and
 * CUDA expert GEMMs build their dequant switch and their dispatch check from
 * the same macro.  The two therefore cannot drift: a type added here without a
 * matching ds4_qwen4exp_<name>_value accessor is rejected by the compiler on
 * both backends, and a type left out is refused by name at load time and again
 * at dispatch.
 *
 * The two backends reject it at different moments.  CUDA fails at build time,
 * when nvcc compiles ds4_cuda_qwen4exp.cu.  Metal compiles its library from
 * source at run time, so a missing accessor fails there when ds4_gpu_init()
 * builds the library -- an ordinary Metal compile error naming the accessor,
 * before any kernel dispatches, not a wrong number later.
 *
 * The ids are ggml type ids.  The set is what the shipped artifacts carry on a
 * routed or shared expert tensor:
 *
 *   f32   the shared expert router row
 *   q5_1  routed down on most blocks of UD-Q4_K_XL
 *   q8_0  the whole shared expert, and routed down on five blocks
 *   q4_K  routed gate/up on most blocks of UD-Q4_K_XL
 *   q5_K  routed gate/up on block 2 of UD-Q4_K_XL, and on 44 blocks of MQ-Q6
 *   q6_K  routed gate/up on four blocks of MQ-Q6
 *
 * tests/test_qwen4exp_moe.c carries the measured per-block table.
 *
 * This file is also read at run time and prepended to the Metal library
 * source, so it must stay valid Metal: preprocessor directives only.
 */
#define DS4_QWEN4EXP_MOE_TYPES(X) \
    X(f32,   0) \
    X(q5_1,  7) \
    X(q8_0,  8) \
    X(q4_K, 12) \
    X(q5_K, 13) \
    X(q6_K, 14)

#endif /* DS4_QWEN4EXP_MOE_TYPES_H */
