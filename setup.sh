#!/usr/bin/env bash
# setup.sh -- build the CUDA engine and verify the pinned target snapshot.
#
# benchmark.json's setupCommand runs `./tools/fetch-benchd.sh && ./setup.sh`;
# benchmark.yml calls it on the ranked box before the GPU-locked measurement.
#
# Steps:
#   1. tools/ds4/build.sh -- the vendored ds4 tree's CUDA core objects, the weight
#      owner, the shim library and the cuda-engine adapter (staged for benchd
#      by tools/stage-cuda-engine.sh). Needs nvcc and cargo.
#   2. verify the pinned GGUF target snapshot at MLXFAST_TARGET_SNAPSHOT_DIR
#      against the pins in fixtures/qwen3_8_125b_a6b_track.json: every main
#      shard and the native MTP draft head, byte count first, then sha256.
#
# Environment:
#   MLXFAST_TARGET_SNAPSHOT_DIR   the on-box target snapshot: the GGUF shards
#                                 plus the native MTP draft head, flat beside
#                                 them (organizer-staged; there is no default)
#   MLXFAST_SKIP_ENGINE_BUILD=1   skip step 1 (a pre-built engine is staged).
#                                 When the staged engine came from
#                                 tools/ds4/build-cache.sh it carries a
#                                 MANIFEST, and the vendor base it records must
#                                 equal ds4/VENDOR.json's or the skip is
#                                 REFUSED -- see step 1.
#   MLXFAST_SKIP_WEIGHTS_DOWNLOAD=1 / SKIP_MODEL_DOWNLOAD=1
#                                 skip step 2 (no scored run is possible)
#   MLXFAST_SKIP_WEIGHTS_SHA256=1 verify byte counts only (the ranked box
#                                 verifies the full digest at staging time)
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null && pwd -P)"
cd "${ROOT_DIR}"
log() { printf 'setup.sh: %s\n' "$*"; }
die() { printf 'setup.sh: %s\n' "$*" >&2; exit 1; }

CONTRACT="${ROOT_DIR}/fixtures/qwen3_8_125b_a6b_track.json"
command -v jq >/dev/null 2>&1 || die "jq is required"

# --- 1. engine build --------------------------------------------------------
if [[ "${MLXFAST_SKIP_ENGINE_BUILD:-0}" == "1" ]]; then
  log "MLXFAST_SKIP_ENGINE_BUILD=1: not building the engine; a pre-built cuda-engine must be staged"
  [[ -x "${ROOT_DIR}/.build/release/mlxfast-runtime-worker" ]] \
    || die "no staged engine at .build/release/mlxfast-runtime-worker"
  # THE PIN TRIPWIRE. A cached engine is only the right engine for the tree it
  # was built from. tools/ds4/build-cache.sh keys on the vendored engine's
  # CONTENT and leaves a MANIFEST beside the artefacts; if one is there its
  # vendor base must agree with ds4/VENDOR.json, or the run refuses HERE rather
  # than measuring the wrong engine inside the GPU-locked window. No MANIFEST
  # means the engine was staged by hand, which is the operator's own business
  # and is left alone.
  #
  # The CONTENT is what the cache key covers, so a participant's edit under
  # ds4/ misses the cache and is rebuilt. This check is the second line: it
  # catches an entry restored across a base change, which the key would already
  # have separated.
  cache_manifest="${ROOT_DIR}/.build/engine-cache/MANIFEST"
  if [[ -f "${cache_manifest}" ]]; then
    staged_ds4="$(awk -F'\t' '$1 == "ds4_vendor_base" { print $2; exit }' "${cache_manifest}")"
    current_ds4="$(jq -r '.fork.sha // empty' "${ROOT_DIR}/ds4/VENDOR.json" 2>/dev/null || true)"
    [[ -n "${staged_ds4}" ]] \
      || die "the staged engine's MANIFEST records no ds4 vendor base: ${cache_manifest}"
    [[ "${staged_ds4}" == "${current_ds4}" ]] \
      || die "the staged engine was built from ds4 vendor base ${staged_ds4} but this tree records ${current_ds4}; refusing to run a cached binary against a different engine base (delete .build and re-run to rebuild)"
    log "staged engine matches the vendored ds4 base ${current_ds4:0:12}"
  fi
else
  log "building the ds4 engine and the cuda-engine adapter (tools/ds4/build.sh)"
  "${ROOT_DIR}/tools/ds4/build.sh"
  # Stage (again, idempotently) so the staged path is this script's own
  # contract with benchd, whatever the build script did.
  "${ROOT_DIR}/tools/stage-cuda-engine.sh"
fi

# --- 2. target snapshot verification -----------------------------------------
if [[ "${MLXFAST_SKIP_WEIGHTS_DOWNLOAD:-0}" == "1" || "${SKIP_MODEL_DOWNLOAD:-0}" == "1" ]]; then
  log "MLXFAST_SKIP_WEIGHTS_DOWNLOAD/SKIP_MODEL_DOWNLOAD set: not verifying the checkpoint (no scored run is possible)"
else
  SNAPSHOT_DIR="${MLXFAST_TARGET_SNAPSHOT_DIR:-}"
  [[ -n "${SNAPSHOT_DIR}" ]] \
    || die "MLXFAST_TARGET_SNAPSHOT_DIR is unset: point it at the on-box GGUF target snapshot (organizer-staged; there is no default)"
  [[ -d "${SNAPSHOT_DIR}" ]] || die "the target snapshot directory does not exist: ${SNAPSHOT_DIR}"
  log "verifying the pinned GGUF target snapshot at ${SNAPSHOT_DIR} against fixtures/qwen3_8_125b_a6b_track.json"
  # Byte counts first (cheap, names a truncated file), then sha256 of every
  # file. The files are hashed CONCURRENTLY, one process per file, bounded by
  # MLXFAST_VERIFY_JOBS (default 8): the same digests, compared against the
  # same pins, in a fraction of the wall-clock. Nothing is skipped.
  count=0
  pins="$(jq -r '.target.files[] | [.path, (.bytes|tostring), .sha256] | @tsv' "${CONTRACT}")"
  while IFS=$'\t' read -r rel want_bytes want_sha; do
    [[ -n "${rel}" ]] || continue
    f="${SNAPSHOT_DIR}/${rel}"
    [[ -f "${f}" ]] || die "pinned checkpoint file is missing: ${rel}"
    got_bytes="$(wc -c < "${f}" | tr -d '[:space:]')"
    [[ "${got_bytes}" == "${want_bytes}" ]] \
      || die "byte-count mismatch for ${rel}: staged ${got_bytes}, pinned ${want_bytes}"
    count=$((count + 1))
  done <<< "${pins}"
  [[ "${count}" -gt 0 ]] || die "the contract pins no target files (.target.files is empty)"
  if [[ "${MLXFAST_SKIP_WEIGHTS_SHA256:-0}" != "1" ]]; then
    verify_dir="$(mktemp -d)"
    jobs_max="${MLXFAST_VERIFY_JOBS:-8}"
    running=0
    while IFS=$'\t' read -r rel want_bytes want_sha; do
      [[ -n "${rel}" ]] || continue
      (
        got_sha="$(sha256sum "${SNAPSHOT_DIR}/${rel}" | cut -d ' ' -f 1)"
        if [[ "${got_sha}" == "${want_sha}" ]]; then
          printf 'ok\t%s\n' "${rel}"
        else
          printf 'mismatch\t%s\t%s\t%s\n' "${rel}" "${got_sha}" "${want_sha}"
        fi
      ) > "${verify_dir}/$(printf '%s' "${rel}" | tr '/' '_').result" &
      running=$((running + 1))
      if [[ "${running}" -ge "${jobs_max}" ]]; then
        wait -n
        running=$((running - 1))
      fi
    done <<< "${pins}"
    wait
    verified=0
    for r in "${verify_dir}"/*.result; do
      IFS=$'\t' read -r status rel got want < "${r}"
      if [[ "${status}" != "ok" ]]; then
        rm -rf "${verify_dir}"
        die "sha256 mismatch for ${rel}: staged ${got}, pinned ${want}"
      fi
      verified=$((verified + 1))
    done
    rm -rf "${verify_dir}"
    [[ "${verified}" == "${count}" ]] || die "sha256 verified ${verified} of ${count} pinned files"
  fi
  [[ -e "${SNAPSHOT_DIR}/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf" ]] \
    || die "the native MTP draft head is missing: ${SNAPSHOT_DIR}/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf"
  if [[ "${MLXFAST_SKIP_WEIGHTS_SHA256:-0}" == "1" ]]; then
    log "target snapshot verified: ${count} files by byte count (sha256 skipped by MLXFAST_SKIP_WEIGHTS_SHA256=1)"
  else
    log "target snapshot verified: ${count} files by byte count and sha256"
  fi
fi

log "setup complete: cuda-engine staged for benchd; ds4 engine built and the GGUF target verified per the steps above"
