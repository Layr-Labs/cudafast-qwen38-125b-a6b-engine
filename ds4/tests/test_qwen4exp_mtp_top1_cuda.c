#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum { N_VOCAB = 8, N_ROWS = 6 };

static uint32_t cpu_seeded_argmax(const float *row) {
    uint32_t bits;
    memcpy(&bits, &row[0], sizeof(bits));
    if ((bits & 0x7fffffffu) > 0x7f800000u) return 0u;

    uint32_t best = 0u;
    float best_v = row[0];
    for (uint32_t i = 1u; i < N_VOCAB; i++) {
        if (row[i] > best_v) {
            best_v = row[i];
            best = i;
        }
    }
    return best;
}

int main(void) {
    static const float logits[N_ROWS][N_VOCAB] = {
        { NAN, 100.0f, INFINITY, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f },
        { -INFINITY, -INFINITY, NAN, -INFINITY,
          -INFINITY, -INFINITY, -INFINITY, -INFINITY },
        { 1.0f, INFINITY, NAN, INFINITY, -INFINITY, 0.0f, -0.0f, 3.0f },
        { -0.0f, +0.0f, NAN, -0.0f, +0.0f, -INFINITY, -0.0f, +0.0f },
        { 2.0f, NAN, 5.0f, 5.0f, 4.0f, 1.0f, 0.0f, -INFINITY },
        { -3.0f, -2.0f, -1.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f },
    };
    uint32_t got[N_ROWS] = { 0 };
    int rc = 1;

    if (!ds4_gpu_init()) {
        fputs("qwen4exp MTP top-1 CUDA test: GPU init failed\n", stderr);
        return 1;
    }
    ds4_gpu_tensor *in = ds4_gpu_tensor_alloc(sizeof(logits));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(sizeof(got));
    if (in && out &&
        ds4_gpu_tensor_write(in, 0, logits, sizeof(logits)) &&
        ds4_gpu_qwen4exp_mtp_top1_tensor(out, in, N_VOCAB, N_ROWS) &&
        ds4_gpu_synchronize() &&
        ds4_gpu_tensor_read(out, 0, got, sizeof(got))) {
        rc = 0;
        for (uint32_t row = 0u; row < N_ROWS; row++) {
            const uint32_t want = cpu_seeded_argmax(logits[row]);
            if (got[row] != want) {
                fprintf(stderr,
                        "qwen4exp MTP top-1 row %u: got %u, expected %u\n",
                        row, got[row], want);
                rc = 1;
            }
        }
    }
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(in);
    ds4_gpu_cleanup();
    if (rc == 0) puts("qwen4exp MTP top-1 CUDA test: OK");
    return rc;
}
