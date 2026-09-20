/* SIMD inner half of the IQ4_NL PLE row dequantization.
 *
 * One IQ4_NL block is 18 bytes: a 2-byte fp16 scale followed by 16 nibble
 * bytes.  The 32 outputs of a block are
 *
 *     y[j]    = d * kvalues_iq4nl[qs[j] & 0x0f]     j = 0..15
 *     y[j+16] = d * kvalues_iq4nl[qs[j] >> 4]
 *
 * The scalar loop in ds4_qwen4exp_ple.c walks that one nibble at a time
 * through a 256-entry high-nibble table.  Both ISAs this engine ships on
 * have a 16-entry byte LUT in one instruction -- vqtbl1q_s8 on AArch64,
 * pshufb on SSSE3 -- so the whole nibble expansion is two table lookups
 * and a widening multiply, eight stores per block instead of thirty-two.
 *
 * The scale conversion stays scalar and stays in the caller: it is one
 * clz-based fp16->fp32 per block, already fast, and keeping it there means
 * this header never duplicates the rounding rule.
 *
 * Every path writes the same 32 floats the scalar loop writes, in the same
 * order.  Compiled out entirely where neither ISA is present.
 */
#ifndef DS4_PLE_DEQUANT_SIMD_H
#define DS4_PLE_DEQUANT_SIMD_H

#include <stdint.h>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>

#define DS4_PLE_DEQUANT_SIMD 1

/* `kv` is the 16-entry kvalues_iq4nl table loaded once per row. */
static inline void ds4_ple_iq4nl_block_simd(const uint8_t *qs, float d,
                                            int8x16_t kv, float *out) {
    const uint8x16_t b  = vld1q_u8(qs);
    const int8x16_t  lo = vqtbl1q_s8(kv, vandq_u8(b, vdupq_n_u8(0x0f)));
    const int8x16_t  hi = vqtbl1q_s8(kv, vshrq_n_u8(b, 4));

    const int16x8_t lo0 = vmovl_s8(vget_low_s8(lo));
    const int16x8_t lo1 = vmovl_s8(vget_high_s8(lo));
    const int16x8_t hi0 = vmovl_s8(vget_low_s8(hi));
    const int16x8_t hi1 = vmovl_s8(vget_high_s8(hi));

    vst1q_f32(out +  0, vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo0))), d));
    vst1q_f32(out +  4, vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo0))), d));
    vst1q_f32(out +  8, vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo1))), d));
    vst1q_f32(out + 12, vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo1))), d));
    vst1q_f32(out + 16, vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi0))), d));
    vst1q_f32(out + 20, vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi0))), d));
    vst1q_f32(out + 24, vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi1))), d));
    vst1q_f32(out + 28, vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi1))), d));
}

#elif defined(__SSSE3__)
#include <tmmintrin.h>

#define DS4_PLE_DEQUANT_SIMD 1


/* Sign-extend the low four i8 lanes of `v` to s32, SSE2 only: put the byte
 * in the high half of a 16-bit lane, arithmetic-shift down, repeat into
 * 32-bit lanes.  Avoids the SSE4.1 pmovsx dependency so __SSSE3__ alone
 * gates this path. */
static inline __m128i ds4_ple_s8x4_s32(__m128i v) {
    const __m128i z = _mm_setzero_si128();
    __m128i w = _mm_unpacklo_epi8(z, v);       /* s16 = byte << 8, lanes 0..7 */
    w = _mm_srai_epi16(w, 8);                  /* sign-extended s16 */
    w = _mm_unpacklo_epi16(z, w);              /* s32 = s16 << 16, lanes 0..3 */
    return _mm_srai_epi32(w, 16);
}

/* `kv` is the 16-entry kvalues_iq4nl table loaded once per row.  pshufb
 * selects on the low four bits of each index byte when bit 7 is clear, so
 * the masked nibbles index it exactly like the scalar table does. */
static inline void ds4_ple_iq4nl_block_simd(const uint8_t *qs, float d,
                                            __m128i kv, float *out) {
    const __m128i b   = _mm_loadu_si128((const __m128i *)qs);
    const __m128i msk = _mm_set1_epi8(0x0f);
    const __m128i lo  = _mm_shuffle_epi8(kv, _mm_and_si128(b, msk));
    const __m128i hi  = _mm_shuffle_epi8(
        kv, _mm_and_si128(_mm_srli_epi16(b, 4), msk));
    const __m128  dv  = _mm_set1_ps(d);

    _mm_storeu_ps(out +  0, _mm_mul_ps(_mm_cvtepi32_ps(ds4_ple_s8x4_s32(lo)), dv));
    _mm_storeu_ps(out +  4, _mm_mul_ps(_mm_cvtepi32_ps(ds4_ple_s8x4_s32(_mm_srli_si128(lo, 4))), dv));
    _mm_storeu_ps(out +  8, _mm_mul_ps(_mm_cvtepi32_ps(ds4_ple_s8x4_s32(_mm_srli_si128(lo, 8))), dv));
    _mm_storeu_ps(out + 12, _mm_mul_ps(_mm_cvtepi32_ps(ds4_ple_s8x4_s32(_mm_srli_si128(lo, 12))), dv));
    _mm_storeu_ps(out + 16, _mm_mul_ps(_mm_cvtepi32_ps(ds4_ple_s8x4_s32(hi)), dv));
    _mm_storeu_ps(out + 20, _mm_mul_ps(_mm_cvtepi32_ps(ds4_ple_s8x4_s32(_mm_srli_si128(hi, 4))), dv));
    _mm_storeu_ps(out + 24, _mm_mul_ps(_mm_cvtepi32_ps(ds4_ple_s8x4_s32(_mm_srli_si128(hi, 8))), dv));
    _mm_storeu_ps(out + 28, _mm_mul_ps(_mm_cvtepi32_ps(ds4_ple_s8x4_s32(_mm_srli_si128(hi, 12))), dv));
}

#else
#define DS4_PLE_DEQUANT_SIMD 0
#endif

#endif /* DS4_PLE_DEQUANT_SIMD_H */
