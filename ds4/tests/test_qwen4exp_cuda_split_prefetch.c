/*
 * Host parity check for qwen4exp_moe_gateup_split_kernel's one-buffer raw
 * payload prefetch.  The production Q4_K load and both decoders are extracted
 * from ds4_cuda_qwen4exp.cu by qwen4exp_cuda_split_prefetch_parity.sh.
 */

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define __device__
#define __forceinline__
#define CUDA_QK_K 256
#define DS4_QWEN4EXP_WIDE_PAYLOAD 0

enum {
    DS4_QWEN4EXP_TY_f32 = 0,
    DS4_QWEN4EXP_TY_q5_1 = 7,
    DS4_QWEN4EXP_TY_q8_0 = 8,
    DS4_QWEN4EXP_TY_q4_K = 12,
    DS4_QWEN4EXP_TY_q5_K = 13,
    DS4_QWEN4EXP_TY_q6_K = 14,
};

static uint32_t host_funnelshift_r(uint32_t lo, uint32_t hi, uint32_t shift) {
    return shift == 0u ? lo : (lo >> shift) | (hi << (32u - shift));
}
#define __funnelshift_r(lo, hi, shift) host_funnelshift_r((lo), (hi), (shift))

static float dev_f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1fu;
    const uint32_t man = h & 0x3ffu;
    union { uint32_t u; float f; } bits;
    if (exp == 0) {
        if (man == 0) { bits.u = sign; return bits.f; }
        union { uint32_t u; float f; } base;
        bits.u = sign | 0x38800000u | (man << 13);
        base.u = sign | 0x38800000u;
        return bits.f - base.f;
    }
    if (exp == 31) { bits.u = sign | 0x7f800000u | (man << 13); return bits.f; }
    bits.u = sign | ((exp + 112u) << 23) | (man << 13);
    return bits.f;
}

#include "bodies.inc"

typedef struct {
    uint32_t g;
    int8_t wq[32];
    float wa[2];
    float wb[2];
} decoded_group;

static int decode_one(const char *row, uint32_t g, const uint32_t *raw,
                      decoded_group *out) {
    int halves = 1;
    decoded_group oracle;
    memset(out, 0, sizeof(*out));
    memset(&oracle, 0, sizeof(oracle));
    out->g = g;
    oracle.g = g;
    dev_qwen4exp_group_decode_w(DS4_QWEN4EXP_TY_q4_K, row, g, raw,
                                out->wq, out->wa, out->wb);
    dev_qwen4exp_group_decode(DS4_QWEN4EXP_TY_q4_K, row, g,
                              oracle.wq, oracle.wa, oracle.wb, &halves);
    return halves == 1 && memcmp(out, &oracle, sizeof(*out)) == 0;
}

static int baseline(const char *row, uint32_t groups, uint32_t lane,
                    decoded_group *out) {
    int n = 0;
    for (uint32_t g = lane; g < groups; g += 32u) {
        uint32_t raw[8];
        const bool have_raw = qw_raw_load(DS4_QWEN4EXP_TY_q4_K,
                                          row, g, raw);
        if (!decode_one(row, g, have_raw ? raw : NULL, &out[n++])) return -1;
    }
    return n;
}

static int prefetched(const char *row, uint32_t groups, uint32_t lane,
                      decoded_group *out) {
    int n = 0;
    uint32_t raw[8];
    uint32_t g = lane;
    bool have_raw = g < groups &&
        qw_raw_load(DS4_QWEN4EXP_TY_q4_K, row, g, raw);
    for (; g < groups; g += 32u) {
        if (!decode_one(row, g, have_raw ? raw : NULL, &out[n++])) return -1;
        const uint32_t next_g = g + 32u;
        have_raw = next_g < groups &&
            qw_raw_load(DS4_QWEN4EXP_TY_q4_K, row, next_g, raw);
    }
    return n;
}

static uint64_t rng_state = 0x6a09e667f3bcc909ull;
static uint32_t rng_u32(void) {
    rng_state = rng_state * 6364136223846793005ull + 1442695040888963407ull;
    return (uint32_t)(rng_state >> 32);
}

static int run_case(char *row, uint32_t groups, bool expect_staged) {
    decoded_group a[16], b[16];
    for (uint32_t lane = 0; lane < 32u; lane++) {
        if (groups > lane) {
            uint32_t raw[8];
            const bool staged = qw_raw_load(DS4_QWEN4EXP_TY_q4_K,
                                            row, lane, raw);
            if (staged != expect_staged) return 0;
        }
        const int na = baseline(row, groups, lane, a);
        const int nb = prefetched(row, groups, lane, b);
        if (na < 0 || nb < 0 || na != nb ||
            memcmp(a, b, (size_t)na * sizeof(*a)) != 0) return 0;
    }
    return 1;
}

int main(void) {
    static const uint32_t group_counts[] = {
        0u, 1u, 31u, 32u, 33u, 63u, 64u, 65u, 319u, 320u
    };
    enum { MAX_GROUPS = 320, BLOCKS = (MAX_GROUPS + 7) / 8 };
    uint32_t storage[(sizeof(cuda_block_q4_K) * BLOCKS + 8) / 4];
    char *aligned = (char *)(void *)storage;
    char *misaligned = aligned + 2;
    int cases = 0;

    for (int trial = 0; trial < 64; trial++) {
        for (size_t i = 0; i < sizeof(storage); i++)
            ((uint8_t *)(void *)storage)[i] = (uint8_t)rng_u32();
        for (size_t i = 0; i < sizeof(group_counts) / sizeof(group_counts[0]); i++) {
            if (!run_case(aligned, group_counts[i], true) ||
                !run_case(misaligned, group_counts[i], false)) {
                fprintf(stderr, "qwen4exp split prefetch parity: FAIL "
                                "trial=%d groups=%u\n", trial, group_counts[i]);
                return 1;
            }
            cases += 2;
        }
    }
    printf("qwen4exp split prefetch parity: PASS "
           "(%d aligned/fallback lane traversals; exact Q4_K decode)\n", cases);
    return 0;
}
