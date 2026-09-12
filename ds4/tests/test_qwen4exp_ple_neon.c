/* ARM64 functional parity, not a performance benchmark. Link the production
 * object and a DS4_PLE_IQ4_NEON=0 copy whose only global symbol is renamed
 * ds4_ple_dequant_iq4_nl_reference (objcopy --keep-global-symbol). */
#include "ds4_qwen4exp_ple.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if !defined(__aarch64__) || !defined(__ARM_NEON)
#error "Run this parity test as ARM64; x86 fallback is not NEON evidence"
#endif
extern void ds4_ple_dequant_iq4_nl_reference(const void *, size_t, float *);
int main(void) {
    enum { BLOCKS = 3, INPUT = BLOCKS * DS4_PLE_IQ4_NL_BLOCK_BYTES + 8,
           VALUES = BLOCKS * DS4_PLE_IQ4_NL_BLOCK_ELEMS + 16 };
    uint8_t input[INPUT], saved[INPUT];
    float reference[VALUES], candidate[VALUES];
    unsigned cases = 0;
    for (unsigned alignment = 0; alignment < 4; alignment++) {
        for (uint32_t h = 0; h < 65536u; h++) {
            memset(input, 0xa6, sizeof(input));
            for (unsigned block = 0; block < BLOCKS; block++) {
                uint8_t *p = input + alignment + block * DS4_PLE_IQ4_NL_BLOCK_BYTES;
                const uint16_t half = (uint16_t)(h ^ (block == 1 ? 0x8000u : block == 2 ? 0x0355u : 0u));
                memcpy(p, &half, sizeof(half));
                for (unsigned j = 0; j < 16; j++)
                    p[2+j] = (uint8_t)(((j+block)&15u) | (((15u-j+block)&15u)<<4));
            }
            memcpy(saved, input, sizeof(input));
            memset(reference, 0x5a, sizeof(reference));
            memset(candidate, 0x5a, sizeof(candidate));
            const unsigned guard = 1u + alignment;
            ds4_ple_dequant_iq4_nl_reference(input + alignment, BLOCKS, reference + guard);
            ds4_ple_dequant_iq4_nl(input + alignment, BLOCKS, candidate + guard);
            if (memcmp(reference, candidate, sizeof(reference)) || memcmp(input, saved, sizeof(input))) {
                fprintf(stderr, "PLE NEON mismatch: half=%04x input offset=%u\n", h, alignment);
                return 1;
            }
            cases++;
        }
    }
    const unsigned offsets[] = {0, 32, 64, 68, 128, 256};
    for (unsigned offset = 0; offset < sizeof(offsets)/sizeof(offsets[0]); offset++) {
        union { float alignment; uint8_t bytes[1024]; } a, b;
        for (unsigned j = 0; j < sizeof(a.bytes); j++) a.bytes[j] = (uint8_t)(j * 37u + 11u);
        memcpy(&b, &a, sizeof(a));
        ds4_ple_dequant_iq4_nl_reference(a.bytes + 64, BLOCKS,
                                        (float *)(void *)(a.bytes + offsets[offset]));
        ds4_ple_dequant_iq4_nl(b.bytes + 64, BLOCKS,
                              (float *)(void *)(b.bytes + offsets[offset]));
        if (memcmp(&a, &b, sizeof(a))) {
            fprintf(stderr, "PLE alias parity offset=%u failed\n", offsets[offset]);
            return 1;
        }
    }
    memset(candidate, 0x5a, sizeof(candidate));
    memcpy(reference, candidate, sizeof(reference));
    ds4_ple_dequant_iq4_nl(input, 0, candidate);
    ds4_ple_dequant_iq4_nl(NULL, 1, candidate);
    ds4_ple_dequant_iq4_nl(input, 1, NULL);
    if (memcmp(reference, candidate, sizeof(reference))) return 1;
    printf("PLE ARM64 NEON: %u complete-buffer comparisons, all half bit patterns, offsets0..3, guards, overlap parity and input immutability PASS\n", cases);
    return 0;
}
