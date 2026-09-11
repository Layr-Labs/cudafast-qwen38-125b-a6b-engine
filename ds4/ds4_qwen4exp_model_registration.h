#ifndef DS4_QWEN4EXP_MODEL_REGISTRATION_H
#define DS4_QWEN4EXP_MODEL_REGISTRATION_H

#include <stdbool.h>
#include <stdint.h>

typedef struct {
    const void *map;
    uint64_t size;
} ds4_qwen4exp_registration_shard;

typedef struct {
    uint32_t shard_index;
    uint64_t offset;
    uint64_t bytes;
    bool ssd_resident;
} ds4_qwen4exp_registration_tensor;

typedef struct {
    const void *(*lookup)(const void *map, uint64_t offset, uint64_t bytes);
    int (*cache)(const void *map, uint64_t size, uint64_t offset,
                 uint64_t bytes, const char *label);
    int (*replaced)(const void *map, uint64_t offset, uint64_t bytes);
} ds4_qwen4exp_registration_ops;

typedef enum {
    DS4_QWEN4EXP_REGISTRATION_OK = 0,
    DS4_QWEN4EXP_REGISTRATION_INVALID_ARGUMENT,
    DS4_QWEN4EXP_REGISTRATION_INVALID_SHARD,
    DS4_QWEN4EXP_REGISTRATION_INVALID_RANGE,
    DS4_QWEN4EXP_REGISTRATION_CACHE_FAILED,
} ds4_qwen4exp_registration_error_kind;

typedef struct {
    ds4_qwen4exp_registration_error_kind kind;
    uint64_t tensor_index;
    uint32_t shard_index;
    uint64_t offset;
    uint64_t bytes;
} ds4_qwen4exp_registration_error;

/* CUDA target registration policy. Kept backend-free so its allocation
 * decisions can be exercised with a host-only mock. */
static inline bool ds4_qwen4exp_register_cuda_model(
        const ds4_qwen4exp_registration_shard *shards,
        uint32_t n_shards,
        const ds4_qwen4exp_registration_tensor *tensors,
        uint64_t n_tensors,
        const ds4_qwen4exp_registration_ops *ops,
        ds4_qwen4exp_registration_error *error) {
    if (error) *error = (ds4_qwen4exp_registration_error){0};
    if (!shards || (n_tensors != 0 && !tensors) || !ops ||
        !ops->lookup || !ops->cache) {
        if (error) error->kind = DS4_QWEN4EXP_REGISTRATION_INVALID_ARGUMENT;
        return false;
    }

    for (uint64_t i = 0; i < n_tensors; i++) {
        const ds4_qwen4exp_registration_tensor *t = &tensors[i];
        if (t->bytes == 0) continue;
        if (t->shard_index >= n_shards) {
            if (error) {
                error->kind = DS4_QWEN4EXP_REGISTRATION_INVALID_SHARD;
                error->tensor_index = i;
                error->shard_index = t->shard_index;
                error->offset = t->offset;
                error->bytes = t->bytes;
            }
            return false;
        }
        const ds4_qwen4exp_registration_shard *sh = &shards[t->shard_index];
        if (!sh->map || t->offset > sh->size ||
            t->bytes > sh->size - t->offset) {
            if (error) {
                error->kind = DS4_QWEN4EXP_REGISTRATION_INVALID_RANGE;
                error->tensor_index = i;
                error->shard_index = t->shard_index;
                error->offset = t->offset;
                error->bytes = t->bytes;
            }
            return false;
        }

        /* The PLE n-gram table is intentionally SSD-resident. It is the one
         * target tensor this session policy must never upload. */
        if (t->ssd_resident) continue;
        if (ops->replaced && ops->replaced(sh->map, t->offset, t->bytes)) {
            continue;
        }
        if (ops->lookup(sh->map, t->offset, t->bytes)) continue;

        /* Cache the tensor's own shard-relative bytes, never the containing
         * shard. The cache API is idempotent and safely fills a full tensor
         * when an earlier cached range covers only part of it. */
        if (!ops->cache(sh->map, sh->size, t->offset, t->bytes,
                        "qwen4exp-target-tensor")) {
            if (error) {
                error->kind = DS4_QWEN4EXP_REGISTRATION_CACHE_FAILED;
                error->tensor_index = i;
                error->shard_index = t->shard_index;
                error->offset = t->offset;
                error->bytes = t->bytes;
            }
            return false;
        }
    }
    return true;
}

static inline const char *ds4_qwen4exp_registration_error_name(
        ds4_qwen4exp_registration_error_kind kind) {
    switch (kind) {
    case DS4_QWEN4EXP_REGISTRATION_INVALID_ARGUMENT: return "invalid arguments";
    case DS4_QWEN4EXP_REGISTRATION_INVALID_SHARD: return "invalid shard index";
    case DS4_QWEN4EXP_REGISTRATION_INVALID_RANGE: return "invalid tensor range";
    case DS4_QWEN4EXP_REGISTRATION_CACHE_FAILED: return "tensor allocation failed";
    case DS4_QWEN4EXP_REGISTRATION_OK: return "none";
    }
    return "unknown";
}
#endif
