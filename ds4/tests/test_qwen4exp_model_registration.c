#include "ds4_qwen4exp_model_registration.h"

#include <stdio.h>
#include <string.h>

typedef struct {
    const void *map;
    uint64_t offset;
    uint64_t bytes;
} range_record;

typedef struct {
    const void *map;
    uint64_t size;
    uint64_t offset;
    uint64_t bytes;
} cache_record;

static range_record g_resident[8];
static int g_resident_count;
static range_record g_replaced[8];
static int g_replaced_count;
static cache_record g_cache[8];
static int g_cache_count;
static int g_lookup_count;
static int g_fail_cache;

static void reset_mocks(void) {
    memset(g_resident, 0, sizeof(g_resident));
    memset(g_replaced, 0, sizeof(g_replaced));
    memset(g_cache, 0, sizeof(g_cache));
    g_resident_count = 0;
    g_replaced_count = 0;
    g_cache_count = 0;
    g_lookup_count = 0;
    g_fail_cache = 0;
}

static int contains(const range_record *r, const void *map,
                    uint64_t offset, uint64_t bytes) {
    if (r->map != map || offset < r->offset) return 0;
    if (bytes > UINT64_MAX - offset || r->bytes > UINT64_MAX - r->offset) return 0;
    return offset + bytes <= r->offset + r->bytes;
}

static const void *mock_lookup(const void *map, uint64_t offset, uint64_t bytes) {
    g_lookup_count++;
    for (int i = 0; i < g_resident_count; i++) {
        if (contains(&g_resident[i], map, offset, bytes)) return (const void *)1;
    }
    return NULL;
}

static int mock_cache(const void *map, uint64_t size, uint64_t offset,
                      uint64_t bytes, const char *label) {
    (void)label;
    g_cache[g_cache_count++] = (cache_record){map, size, offset, bytes};
    return !g_fail_cache;
}

static int mock_replaced(const void *map, uint64_t offset, uint64_t bytes) {
    for (int i = 0; i < g_replaced_count; i++) {
        if (contains(&g_replaced[i], map, offset, bytes)) return 1;
    }
    return 0;
}


static const ds4_qwen4exp_registration_ops ops = {
    .lookup = mock_lookup,
    .cache = mock_cache,
    .replaced = mock_replaced,
};

#define CHECK(cond, message) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: %s (line %d)\n", message, __LINE__); \
        return 1; \
    } \
} while (0)

static int test_reuse_and_non_ple_only(void) {
    unsigned char ple_shard[1], weight_shard[1];
    const ds4_qwen4exp_registration_shard shards[] = {
        {NULL, 64},                         /* metadata only */
        {ple_shard, 1000},                  /* PLE only */
        {weight_shard, 1000},              /* target weights */
    };
    const ds4_qwen4exp_registration_tensor tensors[] = {
        {1, 100, 500, true},                    /* never leave SSD */
        {1, 700, 0, false},                     /* zero byte */
        {2, 100, 50, false},                    /* already cached */
        {2, 300, 100, false},                   /* only prefix cached */
        {2, 600, 80, false},                    /* replaced artifact */
    };

    reset_mocks();
    g_resident[g_resident_count++] = (range_record){weight_shard, 100, 50};
    g_resident[g_resident_count++] = (range_record){weight_shard, 300, 20};
    g_replaced[g_replaced_count++] = (range_record){weight_shard, 600, 80};

    ds4_qwen4exp_registration_error error;
    CHECK(ds4_qwen4exp_register_cuda_model(
              shards, 3, tensors, 5, &ops, &error),
          "registration succeeds");
    CHECK(g_lookup_count == 2,
          "only non-PLE, non-replaced, nonzero tensors are queried");
    CHECK(g_cache_count == 1, "only the partially cached tensor is allocated");
    CHECK(!(g_cache[0].offset == 200 && g_cache[0].bytes == 800),
          "registration never allocates the containing shard span");
    CHECK(g_cache[0].map == weight_shard && g_cache[0].size == 1000,
          "allocation uses the tensor's own shard mapping");
    CHECK(g_cache[0].offset == 300 && g_cache[0].bytes == 100,
          "allocation covers exactly the missing tensor bytes");
    return 0;
}

static int test_ple_only_and_metadata_only(void) {
    unsigned char ple_shard[1];
    const ds4_qwen4exp_registration_shard shards[] = {
        {NULL, 32},
        {ple_shard, 900},
    };
    const ds4_qwen4exp_registration_tensor tensors[] = {
        {1, 120, 700, true},
    };
    reset_mocks();
    CHECK(ds4_qwen4exp_register_cuda_model(
              shards, 2, tensors, 1, &ops, NULL),
          "metadata-only and PLE-only shards succeed");
    CHECK(g_lookup_count == 0 && g_cache_count == 0,
          "metadata and PLE-only shards allocate nothing");
    return 0;
}

static int test_invalid_inputs_and_allocation_failure(void) {
    unsigned char shard[1];
    const ds4_qwen4exp_registration_shard shards[] = {
        {shard, 1000},
    };
    ds4_qwen4exp_registration_error error;

    reset_mocks();
    const ds4_qwen4exp_registration_tensor invalid_shard[] = {
        {1, 100, 20, false},
    };
    CHECK(!ds4_qwen4exp_register_cuda_model(
               shards, 1, invalid_shard, 1, &ops, &error),
          "invalid tensor shard is rejected");
    CHECK(error.kind == DS4_QWEN4EXP_REGISTRATION_INVALID_SHARD &&
              error.tensor_index == 0 && error.shard_index == 1,
          "invalid shard failure identifies the tensor");
    CHECK(g_cache_count == 0, "invalid shard allocates nothing");

    reset_mocks();
    const ds4_qwen4exp_registration_tensor out_of_bounds[] = {
        {0, 950, 100, false},
    };
    CHECK(!ds4_qwen4exp_register_cuda_model(
               shards, 1, out_of_bounds, 1, &ops, &error),
          "out-of-bounds tensor range is rejected");
    CHECK(error.kind == DS4_QWEN4EXP_REGISTRATION_INVALID_RANGE &&
              error.offset == 950 && error.bytes == 100,
          "range failure reports exact shard-relative bytes");
    CHECK(g_cache_count == 0, "invalid range allocates nothing");

    reset_mocks();
    g_fail_cache = 1;
    const ds4_qwen4exp_registration_tensor missing[] = {
        {0, 400, 60, false},
    };
    CHECK(!ds4_qwen4exp_register_cuda_model(
               shards, 1, missing, 1, &ops, &error),
          "failed exact allocation propagates");
    CHECK(error.kind == DS4_QWEN4EXP_REGISTRATION_CACHE_FAILED &&
              error.tensor_index == 0 && error.shard_index == 0 &&
              error.offset == 400 && error.bytes == 60,
          "allocation failure identifies the exact tensor range");
    CHECK(g_cache_count == 1 && g_cache[0].offset == 400 &&
              g_cache[0].bytes == 60,
          "failed allocation attempted only the required bytes");
    return 0;
}

int main(void) {
    if (test_reuse_and_non_ple_only() != 0) return 1;
    if (test_ple_only_and_metadata_only() != 0) return 1;
    if (test_invalid_inputs_and_allocation_failure() != 0) return 1;
    puts("test_qwen4exp_model_registration PASS");
    return 0;
}
