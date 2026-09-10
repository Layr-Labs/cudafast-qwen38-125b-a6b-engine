/*
 * Print what OUR PLE table reader makes of a row: which bytes it reads and
 * what it dequantizes them to.
 *
 * The other half of the check is tests/qwen4exp_ple_row_addressing.py, which
 * answers the same question through gguf-py.  Neither half can see the other's
 * answer, which is the point: a row index has to become the same byte range on
 * both sides, and the multi-shard artifact is where an offset mistake reads
 * plausible numbers out of the wrong file rather than refusing.
 *
 * usage: test_qwen4exp_ple_row_addressing <row> [<row>...] -- <shard>...
 */
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ds4_qwen4exp_ple.h"

int main(int argc, char **argv) {
    int split = -1;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--") == 0) { split = i; break; }
    }
    if (split < 2 || split + 1 >= argc) {
        fprintf(stderr,
                "usage: %s <row> [<row>...] -- <shard.gguf>...\n", argv[0]);
        return 2;
    }

    const size_t n_paths = (size_t)(argc - split - 1);
    const char **paths = calloc(n_paths, sizeof(*paths));
    if (!paths) return 1;
    for (size_t i = 0; i < n_paths; i++) paths[i] = argv[split + 1 + i];

    char err[DS4_PLE_ERROR_SIZE];
    ds4_ple_table *t = NULL;
    if (!ds4_ple_table_open(paths, n_paths, 0, &t, err, sizeof(err))) {
        fprintf(stderr, "ple table open failed: %s\n", err);
        free(paths);
        return 1;
    }
    const ds4_ple_constants *c = ds4_ple_table_constants(t);
    const size_t row_bytes = ds4_ple_table_quant_row_bytes(t);

    printf("row_dim %u\n", c->row_dim);
    printf("quant_row_bytes %zu\n", row_bytes);
    printf("table_rows %" PRIu64 "\n", c->table_rows);
    printf("row_total %" PRIu64 "\n", c->row_total);
    printf("head_vocab_sizes");
    for (uint32_t i = 0; i < c->head_count; i++) {
        printf(" %" PRIu64, c->head_vocab_sizes[i]);
    }
    printf("\nhead_offsets");
    for (uint32_t i = 0; i < c->head_count; i++) {
        printf(" %" PRIu64, c->head_offsets[i]);
    }
    printf("\n");

    uint8_t *raw = malloc(row_bytes);
    float *values = malloc((size_t)c->row_dim * sizeof(float));
    if (!raw || !values) { ds4_ple_table_close(t); free(paths); return 1; }

    for (int i = 1; i < split; i++) {
        const uint64_t id = strtoull(argv[i], NULL, 10);
        if (!ds4_ple_table_quant_row(t, id, raw)) {
            fprintf(stderr, "row %" PRIu64 " is out of range\n", id);
            ds4_ple_table_close(t); free(paths); return 1;
        }
        printf("row %" PRIu64 " bytes ", id);
        for (size_t b = 0; b < row_bytes; b++) printf("%02x", raw[b]);
        printf("\n");

        if (!ds4_ple_table_rows(t, &id, 1, values)) {
            fprintf(stderr, "row %" PRIu64 " read failed\n", id);
            ds4_ple_table_close(t); free(paths); return 1;
        }
        printf("row %" PRIu64 " values", id);
        /* %.9g round-trips a float exactly, so the diff is on the values and
         * not on how they were printed. */
        for (uint32_t v = 0; v < c->row_dim; v++) printf(" %.9g", values[v]);
        printf("\n");
    }

    free(values); free(raw);
    ds4_ple_table_close(t);
    free(paths);
    return 0;
}
