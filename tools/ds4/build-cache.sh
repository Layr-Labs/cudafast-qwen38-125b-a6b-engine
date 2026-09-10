#!/usr/bin/env bash
# A build cache for the ds4 engine, so a ranked dispatch does not rebuild it.
#
# WHY. benchmark.yml checks out fresh and runs ./setup.sh, which runs
# tools/ds4/build.sh: 14 nvcc translation units plus the cargo adapter, about
# three minutes at -j20 on the box, every dispatch, for a tree that usually has
# not changed. setup.sh already honours MLXFAST_SKIP_ENGINE_BUILD=1 when the
# engine is staged; this script is what makes that flag SAFE to set
# automatically.
#
# THE KEY IS THE WHOLE DESIGN. A cache that can return a binary built from
# different inputs is worse than no cache: it would be discovered as a wrong
# number inside a GPU-locked window, not as a build failure. So the key is a
# sha256 over everything that reaches the compiler:
#
#   1. every byte of the VENDORED ENGINE TREE ds4/. It used to be the gitlink,
#      which was one sha for a submodule nobody here could edit. The engine is
#      vendored now and PARTICIPANT-EDITABLE, so the content is the only honest
#      key: a participant's edit to a kernel MUST miss the cache and rebuild,
#      and a gitlink cannot see that edit at all;
#   2. every byte of harness/protocol-adapter/ and tools/ds4/, and Cargo.lock;
#   3. the engine Makefile's own nvcc flag lines (CUDA_ARCH, NVCCFLAGS,
#      QWEN4EXP_NVCCFLAGS, CORE_OBJS, MMQ_OBJS) plus the DS4_CUDA_ARCH passed;
#   4. `nvcc --version` and `rustc --version`;
#   5. THE WORKSPACE PATH. harness/protocol-adapter/build.rs bakes the absolute
#      .build/ds4 directory into the adapter as an rpath, so a binary restored
#      into a DIFFERENT workspace would look for libds4qwen.so where it is not,
#      or -- worse -- find a stale one. The path is an input, not a detail.
#
# Anything not in that list must not change what the build produces. Anything
# that does change it must be in that list.
#
# NO actions/cache, NO tarballs: a directory per key under a box-side root, and
# sha256. The root is CUDAFAST_ENGINE_CACHE_DIR, staged by converge like the
# other box assets; it defaults to ~/.cache/cudafast-engine-build.
#
# Usage:
#   tools/ds4/build-cache.sh key                the key, 64 hex characters
#   tools/ds4/build-cache.sh path               the entry directory for the key
#   tools/ds4/build-cache.sh restore            copy a hit into the workspace
#                                               and stage it; exit 1 on a miss
#   tools/ds4/build-cache.sh save               copy this workspace's artefacts
#                                               into the entry for the key
#
# `restore` is the only one that stages, and it stages through the REAL
# tools/stage-cuda-engine.sh, so the cached path and the built path put the
# same bytes at the same benchd-resolved location by construction.
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." >/dev/null && pwd -P)"
cd "${ROOT_DIR}"
log() { printf 'ds4/build-cache.sh: %s\n' "$*"; }
die() { printf 'ds4/build-cache.sh: %s\n' "$*" >&2; exit 2; }

CACHE_ROOT="${CUDAFAST_ENGINE_CACHE_DIR:-${HOME}/.cache/cudafast-engine-build}"

# The artefacts an engine build leaves behind, workspace-relative. The staged
# worker is NOT in this list: `restore` produces it with stage-cuda-engine.sh
# from the cargo binary, the same way a real build does.
ARTEFACTS=(
  ".build/ds4/libds4qwen.so"
  ".build/ds4/ds4-resident"
  "harness/protocol-adapter/target/release/cuda-engine"
)

# Where the restored manifest lands, for setup.sh's tripwire to read.
MANIFEST_REL=".build/engine-cache/MANIFEST"

sha256_of_stdin() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum | cut -d ' ' -f 1
  else shasum -a 256 | cut -d ' ' -f 1; fi
}

sha256_of_file() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum -- "$1" | cut -d ' ' -f 1
  else shasum -a 256 -- "$1" | cut -d ' ' -f 1; fi
}

# The signed port commit the tree was vendored FROM, for the record. It is
# provenance, not identity: a participant edit keeps this sha and changes the
# content, so the key below hashes the content and this only labels it.
ds4_vendor_base() {
  jq -r '.fork.sha // empty' "${ROOT_DIR}/ds4/VENDOR.json" 2>/dev/null || printf 'unknown\n'
}

# Every tracked byte under the adapter and the ds4 tooling, plus Cargo.lock.
# `git ls-files` is the enumerator so an untracked scratch file in the tree
# cannot change the key, and the paths are sorted so the order is stable.
source_digest() {
  local files rel
  files="$(git -C "${ROOT_DIR}" ls-files \
    'ds4/*' 'harness/protocol-adapter/*' 'tools/ds4/*' | LC_ALL=C sort)"
  [[ -n "${files}" ]] || die "no adapter or ds4 tooling sources are tracked; the key would not cover the build"
  {
    while IFS= read -r rel; do
      [[ -n "${rel}" ]] || continue
      # A tracked path that is not a readable file makes the digest a lie, so
      # it stops the key rather than hashing to nothing.
      [[ -f "${ROOT_DIR}/${rel}" ]] \
        || die "tracked source ${rel} is missing from the working tree; refusing to key a build on it"
      printf '%s\t%s\n' "${rel}" "$(sha256_of_file "${ROOT_DIR}/${rel}")"
    done <<< "${files}"
    # The count is part of the digest, so a file DISAPPEARING from the list
    # changes the key even if nothing else does.
    printf 'files\t%d\n' "$(printf '%s\n' "${files}" | wc -l | tr -d '[:space:]')"
  } | sha256_of_stdin
}

# The engine Makefile's own flag lines. Hashed in full by source_digest above
# as well; kept explicit so the key stays readable and a flag move is visible
# in the manifest rather than only inside a digest.
makefile_flags() {
  local mk="${ROOT_DIR}/ds4/Makefile"
  [[ -f "${mk}" ]] || { printf 'ds4-tree-absent\n'; return 0; }
  grep -E '^(CORE_OBJS|MMQ_OBJS|NVCCFLAGS|QWEN4EXP_NVCCFLAGS|NVCC_ARCH_FLAGS|CUDA_ARCH)[[:space:]]*[:?]?=' "${mk}" || true
}

cache_key() {
  local nvcc_v rustc_v
  nvcc_v="$(nvcc --version 2>/dev/null | tail -1 || true)"
  rustc_v="$(rustc --version 2>/dev/null || true)"
  {
    printf 'cudafast-engine-build-cache/v1\n'
    printf 'sources\t%s\n' "$(source_digest)"
    printf 'cuda_arch\t%s\n' "${DS4_CUDA_ARCH:-sm_121}"
    printf 'cargo_profile\t%s\n' "${CUDA_ENGINE_CARGO_PROFILE:-release}"
    printf 'nvcc\t%s\n' "${nvcc_v}"
    printf 'rustc\t%s\n' "${rustc_v}"
    printf 'workspace\t%s\n' "${ROOT_DIR}"
    printf 'makefile_flags\n%s\n' "$(makefile_flags)"
  } | sha256_of_stdin
}

write_manifest() {
  local dest="$1" key="$2"
  mkdir -p "$(dirname "${dest}")"
  {
    printf 'key\t%s\n' "${key}"
    printf 'ds4_vendor_base\t%s\n' "$(ds4_vendor_base)"
    printf 'workspace\t%s\n' "${ROOT_DIR}"
    printf 'cuda_arch\t%s\n' "${DS4_CUDA_ARCH:-sm_121}"
    printf 'nvcc\t%s\n' "$(nvcc --version 2>/dev/null | tail -1 || true)"
    printf 'rustc\t%s\n' "$(rustc --version 2>/dev/null || true)"
    printf 'saved_at\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "${dest}"
}

manifest_field() {
  awk -F'\t' -v k="$2" '$1 == k { print $2; exit }' "$1" 2>/dev/null || true
}

case "${1:-}" in
key)
  cache_key
  ;;

path)
  printf '%s/%s\n' "${CACHE_ROOT}" "$(cache_key)"
  ;;

restore)
  key="$(cache_key)"
  entry="${CACHE_ROOT}/${key}"
  if [[ ! -d "${entry}" ]]; then
    log "miss: no entry at ${entry}"
    exit 1
  fi
  # A hit is only a hit if the entry is COMPLETE and its manifest still agrees
  # with this workspace. A half-written entry -- a save interrupted by the job
  # timeout -- must read as a miss, not as an engine.
  for rel in "${ARTEFACTS[@]}"; do
    if [[ ! -f "${entry}/${rel}" ]]; then
      log "miss: entry ${key:0:12} is incomplete (${rel} absent); building"
      exit 1
    fi
  done
  if [[ ! -f "${entry}/MANIFEST" ]]; then
    log "miss: entry ${key:0:12} has no MANIFEST; building"
    exit 1
  fi
  want_base="$(manifest_field "${entry}/MANIFEST" ds4_vendor_base)"
  have_base="$(ds4_vendor_base)"
  if [[ "${want_base}" != "${have_base}" ]]; then
    die "REFUSING the cache entry ${key:0:12}: its MANIFEST records vendor base ${want_base:-none} but this tree records ${have_base}. The content digest should have separated these, so a match here means the key is not covering the engine tree."
  fi
  want_workspace="$(manifest_field "${entry}/MANIFEST" workspace)"
  if [[ "${want_workspace}" != "${ROOT_DIR}" ]]; then
    die "REFUSING the cache entry ${key:0:12}: it was built in ${want_workspace:-an unrecorded directory} and this workspace is ${ROOT_DIR}. The adapter carries the build directory as an rpath, so those binaries would resolve libds4qwen.so somewhere else."
  fi
  for rel in "${ARTEFACTS[@]}"; do
    mkdir -p "${ROOT_DIR}/$(dirname "${rel}")"
    cp -f "${entry}/${rel}" "${ROOT_DIR}/${rel}"
  done
  chmod +x "${ROOT_DIR}/.build/ds4/ds4-resident" \
           "${ROOT_DIR}/harness/protocol-adapter/target/release/cuda-engine"
  mkdir -p "$(dirname "${ROOT_DIR}/${MANIFEST_REL}")"
  cp -f "${entry}/MANIFEST" "${ROOT_DIR}/${MANIFEST_REL}"
  # Stage through the REAL tool, so the cached path and the built path put the
  # same bytes at the same benchd-resolved location.
  "${ROOT_DIR}/tools/stage-cuda-engine.sh"
  log "restored ${key:0:12} from ${entry} and staged the adapter"
  ;;

save)
  key="$(cache_key)"
  entry="${CACHE_ROOT}/${key}"
  for rel in "${ARTEFACTS[@]}"; do
    [[ -f "${ROOT_DIR}/${rel}" ]] \
      || die "nothing to save: ${rel} is absent; run tools/ds4/build.sh first"
  done
  # Write to a scratch directory and rename, so a save that dies half way
  # cannot leave an entry another job would read as a hit.
  mkdir -p "${CACHE_ROOT}"
  staging="$(mktemp -d "${CACHE_ROOT}/.staging.XXXXXX")"
  trap 'rm -rf "${staging}"' EXIT
  for rel in "${ARTEFACTS[@]}"; do
    mkdir -p "${staging}/$(dirname "${rel}")"
    cp -f "${ROOT_DIR}/${rel}" "${staging}/${rel}"
  done
  write_manifest "${staging}/MANIFEST" "${key}"
  rm -rf "${entry}"
  mv "${staging}" "${entry}"
  trap - EXIT
  # The tripwire reads this from the workspace on a build-path run too, so a
  # later skip on the same workspace is checked the same way as a restore.
  write_manifest "${ROOT_DIR}/${MANIFEST_REL}" "${key}"
  log "saved ${key:0:12} to ${entry}"
  ;;

*)
  die "usage: tools/ds4/build-cache.sh {key|path|restore|save}"
  ;;
esac
