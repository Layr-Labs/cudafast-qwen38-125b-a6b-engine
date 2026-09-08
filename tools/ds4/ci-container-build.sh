#!/usr/bin/env bash
# Build and prove the CUDA engine OFF-BOX: inside the pinned aarch64 CUDA devel
# container, with no GPU and no driver. This is the body of the `cuda-build`
# job in .github/workflows/ci.yml, kept as a script so the identical command
# reproduces the CI result on any aarch64 machine with docker:
#
#   docker run --rm -v "${PWD}:${PWD}" -w "${PWD}" \
#     nvcr.io/nvidia/cuda@sha256:7d2f6a8c2071d911524f95061a0db363e24d27aa51ec831fcccf9e76eb72bc92 \
#     bash tools/ds4/ci-container-build.sh
#
# WHAT IT PROVES. The VENDORED ds4 tree compiles with nvcc for the arch the
# ranked box runs, the qwen4exp kernels included, THROUGH ./setup.sh -- the same
# entry point the box and a participant use; the driver API resolves;
# libds4qwen.so links under --no-undefined, so a missing engine symbol fails
# here instead of when benchd spawns the worker inside the GPU-locked window;
# every kernel cubin in that library carries sm_121a SASS; the shim compiles
# against the port's public ds4.h; and the adapter links and its tests pass
# with --features ds4-engine.
#
# WHAT IT DOES NOT PROVE. Nothing about EXECUTION. There is no device in this
# container, and the CUDA driver library is the toolkit's stub, which cannot
# run. Numerics, kernel correctness and performance stay the ranked box's job.
#
# TWO OFF-BOX-ONLY LINKER SETTINGS, neither of which belongs in
# tools/ds4/build.sh:
#
#   * the libcuda.so.1 symlink. The toolkit ships its driver stub as
#     libcuda.so ONLY, while libds4qwen.so records DT_NEEDED libcuda.so.1 (the
#     stub's soname). Linking the adapter against that library makes ld look
#     for a FILE called libcuda.so.1, which off-box does not exist, and the
#     link fails on cuMemCreate and the rest of the driver API.
#   * RUSTFLAGS -rpath-link, which lets ld satisfy that DT_NEEDED while
#     linking the adapter. A -L does not cover a dependency-of-a-dependency;
#     only -rpath-link, a runpath or LD_LIBRARY_PATH does.
#
# On the box the driver package supplies /usr/lib/aarch64-linux-gnu/libcuda.so.1
# in a default search directory, so both are unnecessary there -- and a stubs
# -L in the repository could shadow the real driver.
#
# NO LIBRARY_PATH OVERRIDE, deliberately. The nvidia/cuda image already exports
# LIBRARY_PATH=/usr/local/cuda/lib64/stubs, so plain `-lcuda` resolves against
# the stub here and tools/ds4/build.sh takes its NORMAL path. That leaves its
# stub-fallback branch unexercised, so step 5 below builds a second time with
# LIBRARY_PATH removed, which is the only way to make `-lcuda` fail to resolve
# and put that branch on a runner. Step 5 costs a second full build (build.sh
# has no link-only entry point and wipes its source copy on every run), so it
# runs LAST and the FIRST build stays the artifact every assert above reads.
set -euo pipefail

log() { printf 'ds4/ci-container-build.sh: %s\n' "$*"; }
die() { printf 'ds4/ci-container-build.sh: %s\n' "$*" >&2; exit 1; }

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
STUBS="${CUDA_HOME}/targets/sbsa-linux/lib/stubs"
# The RC host's toolchain. Pinned rather than floating so a rust release cannot
# turn this gate red on its own schedule.
RUST_VERSION="${RUST_VERSION:-1.98.0}"
export RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo
export PATH="/opt/cargo/bin:${PATH}"

# --- 0. the container's own prerequisites -----------------------------------
# The devel image carries nvcc, gcc and make but not git (build.sh copies the
# vendored tree with git ls-files), curl, or jq -- and jq is needed because this
# job runs ./setup.sh, the documented path, which reads the track contract.
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends build-essential git curl ca-certificates jq
curl -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path \
  --default-toolchain "${RUST_VERSION}" --profile minimal
# The checkout is bind-mounted from the runner, so it is not owned by root here.
git config --global --add safe.directory '*'
"${CUDA_HOME}/bin/nvcc" --version | tail -1
rustc --version

[[ -f "${STUBS}/libcuda.so" ]] || die "no toolkit driver stub at ${STUBS}/libcuda.so"
ln -sf libcuda.so "${STUBS}/libcuda.so.1"
export RUSTFLAGS="-Clink-arg=-Wl,-rpath-link=${STUBS}"

# --- 1. the build the ranked box runs, BY THE DOCUMENTED PATH ---------------
# ./setup.sh is what benchmark.json's setupCommand runs and what README tells a
# participant to run, so that is what is exercised here rather than the build
# script underneath it: it runs tools/ds4/build.sh and then stages the adapter,
# which is the pair the box depends on. The checkpoint verification is skipped
# (no 173 GiB of weights in a CI container); nothing else is.
#
# THIS ALSO PROVES THE VENDORING. The workspace is a plain checkout with NO
# submodule initialised -- there is no submodule any more -- so a build that
# succeeds here is a build from the vendored ds4/ tree that arrived with the
# clone. Before vendoring this job had to fetch an internal repository and
# skipped when it could not.
DS4_BUILD_JOBS="${DS4_BUILD_JOBS:-$(nproc)}"
export DS4_BUILD_JOBS
log "building with -j${DS4_BUILD_JOBS} through ./setup.sh (the documented path)"
[[ ! -f .gitmodules ]] \
  || die ".gitmodules exists: the engine is supposed to be vendored, not a submodule"
[[ -f ds4/ds4.c && -f ds4/VENDOR.json ]] \
  || die "the vendored ds4 tree is not in the checkout (ds4/ds4.c, ds4/VENDOR.json)"
time env MLXFAST_SKIP_WEIGHTS_DOWNLOAD=1 ./setup.sh
[[ -x .build/release/mlxfast-runtime-worker ]] \
  || die "./setup.sh did not stage the adapter at the benchd-resolved path"
log "./setup.sh built the vendored engine and staged the adapter"

# --- 2. the weight owner, and the engine's own server ------------------------
# THE RESIDENT IS THE TOPOLOGY. benchd spawns a worker per phase; ds4-resident
# is the one process that loads the checkpoint and the one every worker
# connects to. If it stopped linking, the ranked run would go back to a load
# per phase. build.sh links it above; this asserts the product and that it
# resolves against libds4qwen.so rather than carrying its own copy of anything.
[[ -x .build/ds4/ds4-resident ]] \
  || die "tools/ds4/build.sh produced no .build/ds4/ds4-resident; the serve would have no weight owner"
readelf -d .build/ds4/ds4-resident | grep -q 'NEEDED.*libds4qwen' \
  || die "ds4-resident does not link libds4qwen.so; it is not driving the pinned engine"
log "ds4-resident links libds4qwen.so"

# The cuda-spark target builds ds4-server alongside our own binaries, at no
# extra cost, and it is asserted here so the decision NOT to use it stays a
# decision rather than a build accident. It cannot serve the scored path: the
# string "logit" does not occur in the pin's ds4_server.c even once, its
# request surface takes text and sampling parameters (never token ids, never a
# teacher-forced eval), and its usage JSON reports no speculative accounting.
# docs/ds4-resident.md carries the mapping table.
[[ -x .build/ds4/src/ds4-server ]] \
  || die "cuda-spark did not produce ds4-server; the pin's Makefile has drifted"
! grep -q 'logit' .build/ds4/src/ds4_server.c \
  || die "the pin's ds4_server.c now mentions logits; re-evaluate whether it can carry the scored contract (docs/ds4-resident.md)"
log "ds4-server built (unused: no logits on its wire), $(stat -c %s .build/ds4/src/ds4-server) bytes"

# --- 3. what the library needs, and which SASS it carries -------------------
readelf -d .build/ds4/libds4qwen.so | grep NEEDED
elf_list="$(cuobjdump --list-elf .build/ds4/libds4qwen.so)"
printf '%s\n' "${elf_list}"
kernels="$(printf '%s\n' "${elf_list}" | grep -c 'sm_121a\.cubin' || true)"
[[ "${kernels}" -gt 0 ]] || die "libds4qwen.so carries no sm_121a kernel cubin at all"
# The ONE expected non-sm_121a entry: nvcc's own device-link step emits a cubin
# at the compiler default arch (sm_75). It holds no engine kernel -- cuobjdump
# reports its command line as the `-arch sm_75 ... -l cudart,cublas,cuda,m,
# cudadevrt` link, not a translation unit. Anything else at another arch is a
# real finding: a kernel that would have to be JIT-compiled, or worse, refused.
wrong_arch="$(printf '%s\n' "${elf_list}" \
  | grep 'cubin' | grep -v 'sm_121a\.cubin' | grep -v 'libds4qwen\.1\.sm_75\.cubin' || true)"
[[ -z "${wrong_arch}" ]] \
  || die "cubin(s) built for the wrong arch (sm_121a expected):"$'\n'"${wrong_arch}"
log "${kernels} kernel cubins, all sm_121a"

# --- 4. the adapter's tests, none of which need a device --------------------
DS4_LIB_DIR="$(cd .build/ds4 && pwd)" cargo test --release --features ds4-engine \
  --manifest-path harness/protocol-adapter/Cargo.toml

# --- 5. build.sh's stub-fallback branch, exercised on purpose ---------------
# build.sh probes whether `-lcuda` resolves and, when it does not, links
# against the toolkit stub instead. That branch is what keeps a box or
# container WITHOUT the driver dev symlink building, and until now no run ever
# took it: this image exports LIBRARY_PATH=<stubs>, so the probe always
# succeeded. Removing LIBRARY_PATH takes the stub off the default linker path
# and forces the probe to fail, which is the only honest way to reach the
# branch. Everything above has already been asserted against the first build,
# so this run's output is the product -- it may overwrite .build/ds4 freely.
log "second build with LIBRARY_PATH removed, to exercise build.sh's stub fallback"
fallback_log="$(mktemp)"
env -u LIBRARY_PATH tools/ds4/build.sh 2>&1 | tee "${fallback_log}" \
  || die "build.sh failed with LIBRARY_PATH removed: its stub fallback does not carry the link"
# The two outcomes build.sh can report, quoted from tools/ds4/build.sh. The
# first is the branch being taken; the second is it giving up, which would exit
# nonzero above but is checked by name so a silent rewording cannot pass here.
grep -q 'libcuda.so dev name not on the default linker path; linking against the toolkit stub' \
  "${fallback_log}" \
  || die "build.sh did not report taking its stub fallback with LIBRARY_PATH removed; the probe still resolved -lcuda, so the branch is still unexercised"
! grep -q 'no libcuda.so dev name resolvable and no toolkit stub' "${fallback_log}" \
  || die "build.sh refused: it found neither a libcuda.so dev name nor a toolkit stub"
[[ -f .build/ds4/libds4qwen.so ]] \
  || die "the fallback build reported success but produced no libds4qwen.so"
log "stub fallback exercised: build.sh linked libds4qwen.so against the toolkit stub"

log "done"
