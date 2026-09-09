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
#   2. verify the pinned GGUF target snapshot against the pins in
#      fixtures/qwen3_8_125b_a6b_track.json: every main shard and the native
#      MTP draft head, byte count first, then sha256. With
#      MLXFAST_TARGET_SNAPSHOT_DIR set (a ranked box, organizer-staged) the
#      directory and every file must already be there: nothing is fetched,
#      nothing is deleted, and every run hashes the set. With it UNSET (a
#      participant's own machine) the snapshot lives under this checkout,
#      setup downloads each missing file from the pinned public model
#      repository (resumable, kept only when both pins match), and a verified
#      set leaves a marker keyed on the pins and each file's size, mtime and
#      inode, so a rerun there reads no shard.
#
# Environment:
#   MLXFAST_TARGET_SNAPSHOT_DIR   the target snapshot: the GGUF shards plus the
#                                 native MTP draft head, flat beside them.
#                                 Set: verify only. Unset: the default
#                                 reference_weights/Qwen3.8-Flash-Next-GGUF
#                                 under this checkout, which setup fills.
#   MLXFAST_TARGET_BASE_URL       where a missing file is fetched from on the
#                                 default path. Default: the fixture's
#                                 upstream_model_id at upstream_revision on
#                                 huggingface.co. A file:// base works.
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

# The public model repository the target was pinned from, at that revision
# (the fixture's upstream_model_id / upstream_revision; README, "The pinned
# artifacts"). The shards live under the variant directory and the draft head
# under MTP/; the snapshot keeps them flat.
DEFAULT_SNAPSHOT_DIR="${ROOT_DIR}/reference_weights/Qwen3.8-Flash-Next-GGUF"

resolve_target_source() {
  # Read once, in step 2, when the default path may fetch.
  UPSTREAM_MODEL_ID="$(jq -r '.target.upstream_model_id // empty' "${CONTRACT}")"
  UPSTREAM_REVISION="$(jq -r '.target.upstream_revision // empty' "${CONTRACT}")"
  UPSTREAM_VARIANT="$(jq -r '.target.upstream_variant // empty' "${CONTRACT}")"
  [[ -n "${UPSTREAM_MODEL_ID}" && -n "${UPSTREAM_REVISION}" && -n "${UPSTREAM_VARIANT}" ]] \
    || die "the contract names no upstream_model_id / upstream_revision / upstream_variant; cannot fetch a missing file"
  TARGET_BASE_URL="${MLXFAST_TARGET_BASE_URL:-https://huggingface.co/${UPSTREAM_MODEL_ID}/resolve/${UPSTREAM_REVISION}}"
}

source_path_for() {
  # The repository path of one pinned file: the draft head under MTP/, a
  # shard under the variant directory.
  case "$1" in
    mtp-*.gguf) printf 'MTP/%s\n' "$1" ;;
    *) printf '%s/%s\n' "${UPSTREAM_VARIANT}" "$1" ;;
  esac
}

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | cut -d ' ' -f 1
  else
    shasum -a 256 "$1" | cut -d ' ' -f 1
  fi
}

file_identity() {
  # size, mtime and inode: what the verified marker records per file.
  stat -c '%s %Y %i' "$1" 2>/dev/null || stat -f '%z %m %i' "$1"
}

download_pinned_file() {
  # download_pinned_file REL WANT_BYTES WANT_SHA -- fetch one pinned file into
  # the snapshot, resumable, and keep it only when both pins match.
  local rel="$1" want_bytes="$2" want_sha="$3"
  local dest="${SNAPSHOT_DIR}/${rel}" partial="${SNAPSHOT_DIR}/${rel}.partial"
  local url got_bytes got_sha
  url="${TARGET_BASE_URL}/$(source_path_for "${rel}")"
  command -v curl >/dev/null 2>&1 || die "curl is required to download ${rel}"
  log "downloading ${rel} (${want_bytes} bytes) from ${url}"
  curl -fL --retry 5 --retry-delay 5 -C - -o "${partial}" "${url}" \
    || die "download failed for ${rel}; rerun ./setup.sh to resume from ${partial}"
  got_bytes="$(wc -c < "${partial}" | tr -d '[:space:]')"
  if [[ "${got_bytes}" != "${want_bytes}" ]]; then
    rm -f "${partial}"
    die "downloaded ${rel} is ${got_bytes} bytes, pinned ${want_bytes}; discarded"
  fi
  got_sha="$(sha256_of "${partial}")"
  if [[ "${got_sha}" != "${want_sha}" ]]; then
    rm -f "${partial}"
    die "downloaded ${rel} has sha256 ${got_sha}, pinned ${want_sha}; discarded"
  fi
  mv -f "${partial}" "${dest}"
  log "downloaded and verified ${rel}"
}

snapshot_marker_path() {
  # Keyed on the pins themselves, so a fixture change invalidates every marker.
  local key
  key="$(jq -c '.target.files' "${CONTRACT}" | sha256_of /dev/stdin | cut -c 1-16)"
  printf '%s/.verified-%s\n' "${SNAPSHOT_DIR}" "${key}"
}

snapshot_identity() {
  # One line per pinned file: path, size, mtime, inode.
  local rel
  while IFS=$'\t' read -r rel _ _; do
    [[ -n "${rel}" ]] || continue
    [[ -f "${SNAPSHOT_DIR}/${rel}" ]] || return 1
    printf '%s %s\n' "${rel}" "$(file_identity "${SNAPSHOT_DIR}/${rel}")"
  done <<< "${pins}"
}

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
  pins="$(jq -r '.target.files[] | [.path, (.bytes|tostring), .sha256] | @tsv' "${CONTRACT}")"
  [[ -n "${pins}" ]] || die "the contract pins no target files (.target.files is empty)"
  if [[ -n "${SNAPSHOT_DIR}" ]]; then
    # A staged snapshot: present, complete, and hashed on every run. The ranked
    # path never fetches, deletes or short-cuts organizer material.
    local_snapshot=0
    [[ -d "${SNAPSHOT_DIR}" ]] || die "the target snapshot directory does not exist: ${SNAPSHOT_DIR}"
  else
    local_snapshot=1
    SNAPSHOT_DIR="${DEFAULT_SNAPSHOT_DIR}"
    log "MLXFAST_TARGET_SNAPSHOT_DIR is unset; using ${SNAPSHOT_DIR}"
    mkdir -p "${SNAPSHOT_DIR}" || die "cannot create the target snapshot directory: ${SNAPSHOT_DIR}"
    # One setup at a time fills the directory: two would append into the same
    # partial file.
    setup_lock="${SNAPSHOT_DIR}/.setup-lock"
    mkdir "${setup_lock}" 2>/dev/null \
      || die "another setup is filling ${SNAPSHOT_DIR}, or a previous one left ${setup_lock}; remove it when no setup is running"
    trap 'rmdir "${setup_lock}" 2>/dev/null || true' EXIT
    resolve_target_source
    # A file that is absent, or the wrong size, is fetched. A present file of
    # the right size is left for the digest pass below to judge.
    while IFS=$'\t' read -r rel want_bytes want_sha; do
      [[ -n "${rel}" ]] || continue
      f="${SNAPSHOT_DIR}/${rel}"
      if [[ -f "${f}" ]]; then
        got_bytes="$(wc -c < "${f}" | tr -d '[:space:]')"
        [[ "${got_bytes}" == "${want_bytes}" ]] && continue
        log "${rel} is ${got_bytes} bytes, pinned ${want_bytes}; fetching it again"
        rm -f "${f}"
      fi
      download_pinned_file "${rel}" "${want_bytes}" "${want_sha}"
    done <<< "${pins}"
  fi

  # On the default path, a rerun on a machine that already holds the verified
  # set reads no shard: the marker written after a full pass records each
  # file's size, mtime and inode, and a set that still matches it is the set
  # that was verified. A staged snapshot takes no marker and hashes every run.
  marker="$(snapshot_marker_path)"
  if [[ "${local_snapshot}" == "1" && "${MLXFAST_SKIP_WEIGHTS_SHA256:-0}" != "1" && -f "${marker}" ]] \
      && identity="$(snapshot_identity)" && [[ "${identity}" == "$(cat "${marker}")" ]]; then
    log "target snapshot at ${SNAPSHOT_DIR} verified by marker: $(wc -l < "${marker}" | tr -d '[:space:]') pinned files unchanged since the last full verification"
    verified_by_marker=1
  else
    verified_by_marker=0
  fi

  log "verifying the pinned GGUF target snapshot at ${SNAPSHOT_DIR} against fixtures/qwen3_8_125b_a6b_track.json"
  # Byte counts first (cheap, names a truncated file), then sha256 of every
  # file. The files are hashed CONCURRENTLY, one process per file, bounded by
  # MLXFAST_VERIFY_JOBS (default 8): the same digests, compared against the
  # same pins, in a fraction of the wall-clock. Nothing is skipped.
  count=0
  while IFS=$'\t' read -r rel want_bytes want_sha; do
    [[ -n "${rel}" ]] || continue
    f="${SNAPSHOT_DIR}/${rel}"
    [[ -f "${f}" ]] || die "pinned checkpoint file is missing: ${rel}"
    got_bytes="$(wc -c < "${f}" | tr -d '[:space:]')"
    [[ "${got_bytes}" == "${want_bytes}" ]] \
      || die "byte-count mismatch for ${rel}: staged ${got_bytes}, pinned ${want_bytes}"
    count=$((count + 1))
  done <<< "${pins}"
  if [[ "${MLXFAST_SKIP_WEIGHTS_SHA256:-0}" != "1" && "${verified_by_marker}" != "1" ]]; then
    verify_dir="$(mktemp -d)"
    jobs_max="${MLXFAST_VERIFY_JOBS:-8}"
    running=0
    while IFS=$'\t' read -r rel want_bytes want_sha; do
      [[ -n "${rel}" ]] || continue
      (
        got_sha="$(sha256_of "${SNAPSHOT_DIR}/${rel}")"
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
    if [[ "${local_snapshot}" == "1" ]]; then
      # The marker is a rerun convenience on the default path, never a gate.
      identity="$(snapshot_identity)"
      printf '%s\n' "${identity}" > "${marker}.tmp.$$"
      mv -f "${marker}.tmp.$$" "${marker}"
      log "wrote the verification marker $(basename "${marker}")"
    fi
  fi
  [[ -e "${SNAPSHOT_DIR}/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf" ]] \
    || die "the native MTP draft head is missing: ${SNAPSHOT_DIR}/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf"
  if [[ "${MLXFAST_SKIP_WEIGHTS_SHA256:-0}" == "1" ]]; then
    log "target snapshot verified: ${count} files by byte count (sha256 skipped by MLXFAST_SKIP_WEIGHTS_SHA256=1)"
  elif [[ "${verified_by_marker}" == "1" ]]; then
    log "target snapshot verified: ${count} files by byte count and the marker of the last full sha256 pass"
  else
    log "target snapshot verified: ${count} files by byte count and sha256"
  fi
  log "target snapshot ready at ${SNAPSHOT_DIR}; local runs: tools/local-baseline.sh"
fi

log "setup complete: cuda-engine staged for benchd; ds4 engine built and the GGUF target verified per the steps above"
