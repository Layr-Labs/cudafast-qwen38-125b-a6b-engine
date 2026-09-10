/* The startup upload reads every shard through ITS OWN file, and the two
 * routes it can take upload the same bytes.
 *
 * A GGUF split set is several files.  The CUDA loader's fd route used to key a
 * single (fd, mapping) pair on the model's first shard, so on a split set every
 * range outside that one file declined the route and was copied out of the
 * mapping with cudaMemcpy -- a page fault per page, 167 MB/s on a Spark.  The
 * per-shard registry fixes that, and introduces the one mistake worth pinning:
 * reading the right offset out of the WRONG file, which yields plausible bytes
 * and refuses nowhere.
 *
 * So: two files with content that differs at every offset, both mapped, both
 * registered, ranges from each cached and read back off the device and compared
 * with the file they came from.  The run prints a digest over every uploaded
 * byte.  tests/test_gpu_model_shard_fd.sh runs it twice -- once on the fd route
 * and once with DS4_CUDA_NO_FD_CACHE=1, which is the mapping route -- and the
 * two digests must agree.  That is what says the faster route moved bytes and
 * not numbers. */

#include "ds4_gpu.h"

#include <cuda_runtime.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#define CHECK(cond, msg)                                              \
    do {                                                              \
        if (!(cond)) {                                                \
            fprintf(stderr, "FAIL: %s (line %d)\n", (msg), __LINE__); \
            return 1;                                                 \
        }                                                             \
    } while (0)

#define SHARDS 3
#define SHARD_BYTES (12u * 1024u * 1024u)

/* Distinct at every offset across shards: shard s and shard t disagree on byte
 * i whenever s != t, so a range read out of the wrong file cannot match. */
static unsigned char shard_byte(unsigned shard, uint64_t off) {
    return (unsigned char)((off * 131u + shard * 97u + 17u) & 0xffu);
}

static uint64_t fnv1a(uint64_t h, const unsigned char *p, size_t n) {
    for (size_t i = 0; i < n; i++) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

int main(void) {
    int dev_count = 0;
    (void)cudaGetDeviceCount(&dev_count);
    if (dev_count < 1) {
        fprintf(stderr, "test_gpu_model_shard_fd: no CUDA devices; skipping\n");
        return 0;
    }
    CHECK(ds4_gpu_init(), "ds4_gpu_init");

    char paths[SHARDS][256];
    int fds[SHARDS];
    const unsigned char *maps[SHARDS];
    unsigned char *chunk = (unsigned char *)malloc(1u << 20);
    CHECK(chunk != NULL, "chunk alloc");

    const char *tmp = getenv("TMPDIR");
    if (!tmp || !tmp[0]) tmp = "/tmp";
    for (unsigned s = 0; s < SHARDS; s++) {
        snprintf(paths[s], sizeof(paths[s]),
                 "%s/ds4-shard-fd-%d-%u.bin", tmp, (int)getpid(), s);
        FILE *f = fopen(paths[s], "wb");
        CHECK(f != NULL, "create shard file");
        for (uint64_t off = 0; off < SHARD_BYTES; off += (1u << 20)) {
            for (uint64_t i = 0; i < (1u << 20); i++) {
                chunk[i] = shard_byte(s, off + i);
            }
            CHECK(fwrite(chunk, 1, 1u << 20, f) == (1u << 20), "write shard");
        }
        fclose(f);
        fds[s] = open(paths[s], O_RDONLY);
        CHECK(fds[s] >= 0, "open shard");
        void *m = mmap(NULL, SHARD_BYTES, PROT_READ, MAP_PRIVATE, fds[s], 0);
        CHECK(m != MAP_FAILED, "mmap shard");
        maps[s] = (const unsigned char *)m;
    }

    /* Mirrors the engine: the model's own fd names shard 0 only, and every
     * shard is then registered on top of it. */
    CHECK(ds4_gpu_set_model_fd_for_map(fds[0], maps[0]), "set_model_fd_for_map");
    for (unsigned s = 0; s < SHARDS; s++) {
        CHECK(ds4_gpu_set_model_shard_fd(fds[s], maps[s]), "set_model_shard_fd");
    }

    /* Ranges that are not shard-symmetric: a different offset and length in
     * each file, so a swapped fd cannot line up by accident. */
    const uint64_t offs[SHARDS] = {0, 4u << 20, 1u << 20};
    const uint64_t lens[SHARDS] = {6u << 20, 8u << 20, 3u << 20};

    uint64_t digest = 1469598103934665603ull;
    unsigned char *readback = (unsigned char *)malloc(8u << 20);
    CHECK(readback != NULL, "readback alloc");

    for (unsigned s = 0; s < SHARDS; s++) {
        char label[32];
        snprintf(label, sizeof(label), "shard%u", s);
        CHECK(ds4_gpu_cache_model_range(maps[s], SHARD_BYTES, offs[s], lens[s], label),
              "cache_model_range");
        const void *dev =
            ds4_gpu_model_range_device_ptr(maps[s], offs[s], lens[s]);
        CHECK(dev != NULL, "range was not cached");
        CHECK(cudaMemcpy(readback, dev, (size_t)lens[s],
                         cudaMemcpyDeviceToHost) == cudaSuccess,
              "read the upload back");
        for (uint64_t i = 0; i < lens[s]; i++) {
            if (readback[i] != shard_byte(s, offs[s] + i)) {
                fprintf(stderr,
                        "FAIL: shard %u byte %llu is 0x%02x, the file says "
                        "0x%02x\n",
                        s, (unsigned long long)i, readback[i],
                        shard_byte(s, offs[s] + i));
                return 1;
            }
        }
        digest = fnv1a(digest, readback, (size_t)lens[s]);
    }

    printf("digest=%016llx\n", (unsigned long long)digest);
    fprintf(stderr, "test_gpu_model_shard_fd: %d shards, %u ranges, PASS\n",
            SHARDS, SHARDS);

    free(readback);
    free(chunk);
    for (unsigned s = 0; s < SHARDS; s++) {
        (void)munmap((void *)maps[s], SHARD_BYTES);
        (void)close(fds[s]);
        (void)unlink(paths[s]);
    }
    return 0;
}
