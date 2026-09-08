#!/usr/bin/env bash
# Build the CUDA engine: the ds4 submodule's core objects, the shim library the
# adapter links, and the adapter itself.
#
# ds4/ is the VENDORED Layr-Labs ds4 port -- plain files in this repository, not
# a submodule, so a participant can read, edit and submit it. ds4/VENDOR.json
# records which signed commit of git@github.com:Layr-Labs/ds4.git the tree came
# from. It carries the qwen4exp family: the multi-shard GGUF loader, the
# qwen4exp kernels and graph, the nextn MTP head behind --mtp-model and the
# depth-1 speculative cycle, and it keeps the engine's own Makefile and build
# targets, so this script is a plain build of that Makefile plus our shim:
#
#   1. copy ds4/ to .build/ds4/src, so the source tree stays clean of build
#      output -- and so a participant's edit is built from THEIR tree;
#   2. build the engine's `cuda-spark` target in that copy, which is its own
#      name for CUDA_ARCH=sm_121 (the GB10). Position-independent code is
#      requested through CC and NVCC, which the Makefile declares with `?=`, so
#      the objects can go into a shared library;
#   3. compile harness/protocol-adapter/ds4_shim/ds4_shim.c and link it with
#      the core objects into .build/ds4/libds4qwen.so;
#   4. link .build/ds4/ds4-resident, THE PROCESS THAT OWNS THE WEIGHTS. benchd
#      spawns a worker per phase; the resident is the one process that loads
#      the checkpoint, and every worker connects to it (tools/serve-up.sh);
#   5. cargo build the adapter with --features ds4-engine against that library
#      and stage it for benchd (tools/stage-cuda-engine.sh).
#
# Step 2 also produces the engine's own binaries, ds4-server among them
# (Makefile: cuda-spark builds ds4 ds4-server ds4-bench ds4-eval ds4-agent).
# Nothing here uses ds4-server: it speaks OpenAI/Anthropic chat over HTTP and
# carries no logits, no token-id input and no speculative counters. See
# docs/ds4-resident.md.
#
# `--cpu-check` runs step 1 and then only syntax-checks ds4.c, the shim and the
# resident server with -DDS4_NO_GPU. It needs no CUDA and is what CI runs.
#
# Environment:
#   CUDA_HOME        CUDA toolkit root (default /usr/local/cuda)
#   DS4_CUDA_ARCH    make CUDA_ARCH value (default sm_121, the GB10)
#   DS4_BUILD_JOBS   make -j (default nproc)
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." >/dev/null && pwd -P)"
cd "${ROOT_DIR}"
log() { printf 'ds4/build.sh: %s\n' "$*"; }
die() { printf 'ds4/build.sh: %s\n' "$*" >&2; exit 1; }

MODE="build"
if [[ "${1:-}" == "--cpu-check" ]]; then MODE="cpu-check"; shift; fi
[[ $# -eq 0 ]] || die "unknown argument: $1"

SUBMODULE="${ROOT_DIR}/ds4"   # the vendored engine tree
OUT="${ROOT_DIR}/.build/ds4"
SRC="${OUT}/src"
SHIM="${ROOT_DIR}/harness/protocol-adapter/ds4_shim"

[[ -f "${SUBMODULE}/ds4.h" && -f "${SUBMODULE}/ds4.c" ]] \
  || die "ds4/ carries no engine source (ds4.h and ds4.c); the vendored tree is incomplete -- re-vendor with tools/ds4/vendor-sync.sh"

# --- 1. fresh source copy ---------------------------------------------------
mkdir -p "${OUT}"
rm -rf "${SRC}"
mkdir -p "${SRC}"
( cd "${SUBMODULE}" && git ls-files -z ) | ( cd "${SUBMODULE}" && tar --null -cf - -T - ) | ( cd "${SRC}" && tar -xf - )
vendored_sha="$(jq -r '.fork.sha // empty' "${SUBMODULE}/VENDOR.json" 2>/dev/null || echo unknown)"
log "copied the vendored ds4 tree (${vendored_sha:0:12}, per ds4/VENDOR.json) to ${SRC}"

# --- cpu-check: syntax only, no CUDA ----------------------------------------
if [[ "${MODE}" == "cpu-check" ]]; then
  # macOS libm has no sincos(); the engine's Metal build maps it to __sincos.
  host_defs=()
  [[ "$(uname -s)" == "Darwin" ]] && host_defs=( "-Dsincos(a,b,c)=__sincos((a),(b),(c))" )
  cc -fsyntax-only -std=c99 -D_GNU_SOURCE -DDS4_NO_GPU -Wall -Wextra "${host_defs[@]}" "${SRC}/ds4.c"
  cc -fsyntax-only -std=c11 -D_GNU_SOURCE -Wall -Wextra -I "${SRC}" "${SHIM}/ds4_shim.c"
  cc -fsyntax-only -std=c11 -D_GNU_SOURCE -Wall -Wextra -I "${SHIM}" "${SHIM}/ds4_resident.c"
  log "cpu-check passed: ds4.c (-DDS4_NO_GPU), ds4_shim.c and ds4_resident.c parse against the pinned header"
  exit 0
fi

# --- 2. the engine's CUDA build ------------------------------------------------
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
[[ -x "${CUDA_HOME}/bin/nvcc" ]] || die "nvcc not found at ${CUDA_HOME}/bin/nvcc (set CUDA_HOME)"
DS4_CUDA_ARCH="${DS4_CUDA_ARCH:-sm_121}"
JOBS="${DS4_BUILD_JOBS:-$(nproc)}"
# The pin's CUDA CORE_OBJS (Makefile: the non-Darwin branch). These are the
# objects the shared library carries; the `cuda-spark` target builds them on
# the way to its own five binaries. The last four are the port's: the qwen4exp
# CUDA kernels, the PLE n-gram table, the MTP cycle and its hook bindings.
CORE_OBJS=(
  ds4.o ds4_image.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_cuda.o ds4_layer_pack.o
  cuda/mmq/ds4_ggml_stubs.o cuda/mmq/ds4_mmq.o cuda/mmq/ds4_mmq_d2r.o
  cuda/mmq/quantize.o cuda/mmq/mmid.o cuda/mmq/mmvq.o cuda/mmq/ds4_repack.o
  ds4_cuda_qwen4exp.o ds4_qwen4exp_ple.o ds4_qwen4exp_mtp.o ds4_qwen4exp_mtp_hooks.o
)
log "building the engine's cuda-spark target (CUDA_ARCH=${DS4_CUDA_ARCH}, -j${JOBS})"
# CC and NVCC are `?=` in the Makefile, so a command-line value wins and reaches
# the sub-make cuda-spark spawns. -fPIC is the ONLY change: the arch flags stay
# the Makefile's own, derived from CUDA_ARCH. The qwen4exp translation unit
# keeps its own QWEN4EXP_NVCCFLAGS (NVCCFLAGS minus --use_fast_math), which is
# derived from the untouched NVCCFLAGS and so is not disturbed here.
make -C "${SRC}" -j"${JOBS}" \
  CUDA_HOME="${CUDA_HOME}" CUDA_ARCH="${DS4_CUDA_ARCH}" \
  CC="cc -fPIC" NVCC="${CUDA_HOME}/bin/nvcc -Xcompiler -fPIC" \
  cuda-spark
for o in "${CORE_OBJS[@]}"; do
  [[ -f "${SRC}/${o}" ]] || die "cuda-spark did not produce ${o}; the CORE_OBJS list has drifted from the pin's Makefile"
done

# --- 3. shim + shared library -----------------------------------------------
cc -fPIC -O2 -std=c11 -D_GNU_SOURCE -Wall -Wextra -I "${SRC}" -c -o "${OUT}/ds4_shim.o" "${SHIM}/ds4_shim.c"
objs=()
for o in "${CORE_OBJS[@]}"; do objs+=( "${SRC}/${o}" ); done
# --no-undefined resolves the driver API (cuMemCreate, ...) at link time, so
# the linker must find a libcuda.so dev name. The driver package normally
# installs it; a box or container without it links against the toolkit's stub,
# which carries the same soname (libcuda.so.1), so the runtime still loads the
# real driver.
# The toolkit's target directory is named by the host machine: sbsa-linux on
# the aarch64 box, x86_64-linux on a participant's x86 workstation.
case "$(uname -m)" in
  aarch64|arm64) cuda_target_dir="${CUDA_HOME}/targets/sbsa-linux"; multiarch_dir="/usr/lib/aarch64-linux-gnu" ;;
  x86_64) cuda_target_dir="${CUDA_HOME}/targets/x86_64-linux"; multiarch_dir="/usr/lib/x86_64-linux-gnu" ;;
  *) die "unsupported host machine $(uname -m); ds4 builds on aarch64 or x86_64 Linux" ;;
esac
cuda_link_dirs=( -L"${cuda_target_dir}/lib" -L"${CUDA_HOME}/lib64" )
if ! printf 'int main(void){return 0;}\n' | cc -x c - -o /dev/null "${cuda_link_dirs[@]}" -lcuda 2>/dev/null; then
  stubs="${cuda_target_dir}/lib/stubs"
  [[ -f "${stubs}/libcuda.so" ]] || die "no libcuda.so dev name resolvable and no toolkit stub at ${stubs}; install the driver dev symlink (ls -l ${multiarch_dir}/libcuda.so)"
  log "libcuda.so dev name not on the default linker path; linking against the toolkit stub ${stubs}"
  cuda_link_dirs+=( -L"${stubs}" )
fi
# --no-undefined: an unresolved engine symbol fails HERE, not when benchd spawns
# the worker inside the GPU-locked window.
"${CUDA_HOME}/bin/nvcc" -shared -cudart shared -Xcompiler -fPIC -Xlinker --no-undefined \
  -o "${OUT}/libds4qwen.so" "${OUT}/ds4_shim.o" "${objs[@]}" \
  "${cuda_link_dirs[@]}" \
  -lcudart -lcublas -lcuda -lm -Xcompiler -pthread
log "linked ${OUT}/libds4qwen.so"
# The engine's own binaries came out of the same cuda-spark run. ds4-server is
# NOT the scored serve; its presence is asserted so a Makefile change that
# stopped building it is visible here rather than in a rumour.
[[ -x "${SRC}/ds4-server" ]] \
  || die "cuda-spark did not produce ds4-server; the pin's Makefile has drifted"
log "ds4-server built at ${SRC}/ds4-server (unused by the scored path; see docs/ds4-resident.md)"

# --- 4. the resident engine: the ONE process that holds the weights ----------
# Plain C against the shim's own header, linked to the shared library. No CUDA
# of its own: every engine call goes through libds4qwen.so.
cc -O2 -std=c11 -D_GNU_SOURCE -Wall -Wextra -Werror -I "${SHIM}" \
  -o "${OUT}/ds4-resident" "${SHIM}/ds4_resident.c" \
  -L "${OUT}" -lds4qwen -Wl,-rpath,"${OUT}"
log "linked ${OUT}/ds4-resident (weight owner; tools/serve-up.sh boots one per window)"

# --- 5. the adapter ---------------------------------------------------------
command -v cargo >/dev/null 2>&1 || die "cargo is required to build the cuda-engine adapter"
DS4_LIB_DIR="${OUT}" cargo build --release --features ds4-engine \
  --manifest-path "${ROOT_DIR}/harness/protocol-adapter/Cargo.toml" --bin cuda-engine
"${ROOT_DIR}/tools/stage-cuda-engine.sh"
log "done: adapter staged against ${OUT}/libds4qwen.so"
