#!/usr/bin/env bash
# Stage the scored CUDA engine into the location the benchmarker (benchd)
# resolves.
#
# WHY THIS EXISTS. benchd resolves the scored engine at a FIXED workspace-
# relative path -- <workspace>/.build/release/mlxfast-runtime-worker -- and
# spawns it, speaking Engine Protocol v1 over stdio. That path is a CONTRACT
# with the benchmarker, not a preference: the resolution rule lives benchd-side
# (the pinned prebuilt / dist channel), so this repository honours the path
# rather than renaming it.
#
# WHAT SITS THERE NOW. The MLX/Metal Swift runtime worker was scrubbed. The
# scored engine on this track is the Engine Protocol v1 adapter over the pinned
# ds4 engine -- the `cuda-engine` binary built from harness/protocol-adapter with
# `--features ds4-engine`. It speaks the SAME Engine Protocol v1 wire the old
# worker did, so the only change is which binary sits at the resolved path and
# what it wires to (the pinned ds4 engine linked in-process,
# whose endpoints benchd/the gate export in the environment). There is no
# mlx.metallib any more: the adapter carries no Metal kernels.
#
# This step copies the FINISHED adapter binary into .build/release under the
# name benchd resolves. It runs AFTER the cargo build.
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null && pwd -P)"
cd "${ROOT_DIR}"

repository_path() {
  local candidate="$1"
  if [[ "${candidate}" == /* ]]; then
    printf '%s\n' "${candidate}"
  else
    printf '%s/%s\n' "${ROOT_DIR}" "${candidate#./}"
  fi
}

BUILD_CONFIGURATION="${CUDA_ENGINE_CARGO_PROFILE:-release}"

# SOURCE -- the adapter's cargo target directory. Resolved the same way an
# operator override is honoured, so a pre-built binary can be pointed at
# directly.
CUDA_ENGINE_BIN="$(repository_path \
  "${CUDA_ENGINE_EXECUTABLE:-harness/protocol-adapter/target/${BUILD_CONFIGURATION}/cuda-engine}")"

# DESTINATION -- the FIXED location benchd resolves. Not env-overridable: it is
# a contract with the benchmarker, not a preference. The name is retained from
# the MLX era on purpose -- benchd's resolver still spawns exactly this path.
STAGED_ENGINE_BIN="$(repository_path ".build/${BUILD_CONFIGURATION}/mlxfast-runtime-worker")"

if [[ ! -x "${CUDA_ENGINE_BIN}" ]]; then
  echo "stage-cuda-engine.sh: cuda-engine binary missing or not executable: ${CUDA_ENGINE_BIN}" >&2
  echo "stage-cuda-engine.sh: build it first:" >&2
  echo "  tools/ds4/build.sh   (or: DS4_LIB_DIR=.build/ds4 cargo build --release --features ds4-engine --manifest-path harness/protocol-adapter/Cargo.toml)" >&2
  exit 1
fi

mkdir -p "$(dirname "${STAGED_ENGINE_BIN}")"

# `-ef` is true only when both paths already resolve to the same file, which
# happens when an operator override points the source straight at the benchd
# path -- then the copy is a no-op (and cp would error copying a file onto
# itself).
if [[ ! "${CUDA_ENGINE_BIN}" -ef "${STAGED_ENGINE_BIN}" ]]; then
  cp -f "${CUDA_ENGINE_BIN}" "${STAGED_ENGINE_BIN}"
fi
# benchd checks the execute bit on the resolved binary; guarantee it survives
# the copy regardless of the caller's umask.
chmod +x "${STAGED_ENGINE_BIN}"

echo "stage-cuda-engine.sh: staged ${STAGED_ENGINE_BIN} (cuda-engine, Engine Protocol v1 over the ds4 engine) for benchd"
