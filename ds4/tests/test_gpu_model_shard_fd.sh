#!/bin/sh
# The fd route and the mapping route must upload the same bytes.
#
# Run the shard test twice: once as it ships, and once with the fd route taken
# out, which is the cudaMemcpy-out-of-the-mapping path the loader used before
# the per-shard fd registry.  The digests are over every uploaded byte and must
# agree exactly.
set -e

BIN=$(cd "$(dirname "$0")" && pwd)/test_gpu_model_shard_fd

# The uploads here are a few MiB; the default 1792 MiB arena chunk is for a
# 77 GiB model and would take that much device memory to move 17.
export DS4_CUDA_WEIGHT_ARENA_CHUNK_MB=256

FD=$("$BIN")
MAP=$(DS4_CUDA_NO_FD_CACHE=1 "$BIN")

echo "fd route:      $FD"
echo "mapping route: $MAP"

if [ -z "$FD" ]; then
    echo "no digest from the fd route" >&2
    exit 1
fi
if [ "$FD" != "$MAP" ]; then
    echo "the two upload routes disagree" >&2
    exit 1
fi
echo "test_gpu_model_shard_fd.sh: both routes agree"
