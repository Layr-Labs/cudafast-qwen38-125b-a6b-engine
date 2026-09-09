#!/usr/bin/env bash
# setup.sh step 2 against a tiny pinned snapshot served from a file:// base.
# No GPU, toolchain, checkpoint, credentials, or network is used.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
WORK="$(cd "$(mktemp -d)" && pwd -P)"
trap 'rm -rf "${WORK}"' EXIT

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d ' ' -f 1; else shasum -a 256 "$1" | cut -d ' ' -f 1; fi
}

# The checkout under test: the real setup.sh, a fixture whose target pins name
# five tiny files, and a staged engine so step 1 is skipped.
ROOT="${WORK}/checkout with spaces"
mkdir -p "${ROOT}/fixtures" "${ROOT}/tools" "${ROOT}/.build/release"
cp "${REPO_ROOT}/setup.sh" "${ROOT}/"
printf '#!/usr/bin/env bash\nexit 0\n' > "${ROOT}/.build/release/mlxfast-runtime-worker"
chmod +x "${ROOT}/.build/release/mlxfast-runtime-worker"

# The source: the repository layout the pinned revision has, shards under the
# variant directory and the head under MTP/.
SOURCE="${WORK}/source"
mkdir -p "${SOURCE}/UD-Q4_K_XL" "${SOURCE}/MTP"
FILES=(
  Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
  Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf
  Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf
  Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf
  mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf
)
source_path() { case "$1" in mtp-*) echo "MTP/$1" ;; *) echo "UD-Q4_K_XL/$1" ;; esac; }
pins='[]'
for f in "${FILES[@]}"; do
  head -c $((RANDOM + 1000)) /dev/urandom > "${SOURCE}/$(source_path "${f}")"
  pins="$(jq -c --arg p "${f}" --argjson b "$(wc -c < "${SOURCE}/$(source_path "${f}")" | tr -d ' ')" \
    --arg s "$(sha256_of "${SOURCE}/$(source_path "${f}")")" '. + [{path:$p, bytes:$b, sha256:$s}]' <<<"${pins}")"
done
write_fixture() { jq -n --argjson files "$1" '{target:{files:$files}}' > "${ROOT}/fixtures/qwen3_8_125b_a6b_track.json"; }
write_fixture "${pins}"

# run_setup NAME [ENV=VALUE...]: runs setup.sh from the checkout with the
# engine build skipped; captures stdout+stderr; sets rc.
run_setup() {
  local name="$1"; shift
  rc=0
  (
    cd "${ROOT}"
    env -u MLXFAST_TARGET_SNAPSHOT_DIR -u MLXFAST_TARGET_BASE_URL \
      MLXFAST_SKIP_ENGINE_BUILD=1 MLXFAST_TARGET_BASE_URL="file://${SOURCE}" "$@" ./setup.sh
  ) > "${WORK}/${name}.log" 2>&1 || rc=$?
}
downloads() { grep -c 'downloading ' "${WORK}/$1.log" || true; }
DEFAULT_DIR="${ROOT}/reference_weights/Qwen3.8-Flash-Next-GGUF"

# 1. Fresh machine, no MLXFAST_TARGET_SNAPSHOT_DIR: the default directory is
#    filled from the source, verified, and marked.
run_setup fresh
[[ "${rc}" == 0 ]] || { cat "${WORK}/fresh.log"; exit 1; }
grep -q 'MLXFAST_TARGET_SNAPSHOT_DIR is unset; using' "${WORK}/fresh.log"
[[ "$(downloads fresh)" == 5 ]]
for f in "${FILES[@]}"; do cmp -s "${DEFAULT_DIR}/${f}" "${SOURCE}/$(source_path "${f}")"; done
grep -q 'by byte count and sha256' "${WORK}/fresh.log"
grep -q 'wrote the verification marker' "${WORK}/fresh.log"
marker="$(ls "${DEFAULT_DIR}"/.verified-*)"
[[ "$(wc -l < "${marker}" | tr -d ' ')" == 5 ]]
[[ -z "$(ls "${DEFAULT_DIR}"/*.partial 2>/dev/null)" ]]

# 2. Rerun on the same machine: nothing is fetched (the base is unreachable),
#    no shard is hashed, the marker answers.
run_setup rerun MLXFAST_TARGET_BASE_URL="file://${WORK}/nowhere"
[[ "${rc}" == 0 ]] || { cat "${WORK}/rerun.log"; exit 1; }
[[ "$(downloads rerun)" == 0 ]]
grep -q 'verified by marker: 5 pinned files unchanged' "${WORK}/rerun.log"
grep -q 'by byte count and the marker' "${WORK}/rerun.log"

# 3. One truncated file: only that file is fetched again; the set verifies in
#    full and a new marker is written.
truncate_target="${DEFAULT_DIR}/${FILES[2]}"
head -c 10 "${truncate_target}" > "${truncate_target}.tmp" && mv "${truncate_target}.tmp" "${truncate_target}"
run_setup truncated
[[ "${rc}" == 0 ]] || { cat "${WORK}/truncated.log"; exit 1; }
[[ "$(downloads truncated)" == 1 ]]
grep -q "${FILES[2]} is 10 bytes, pinned" "${WORK}/truncated.log"
cmp -s "${truncate_target}" "${SOURCE}/$(source_path "${FILES[2]}")"
grep -q 'by byte count and sha256' "${WORK}/truncated.log"

# 4. A file whose bytes changed but whose size did not: the marker no longer
#    matches (mtime, inode), the full pass runs and refuses by name.
tamper_target="${DEFAULT_DIR}/${FILES[0]}"
size="$(wc -c < "${tamper_target}" | tr -d ' ')"
head -c "${size}" /dev/urandom > "${tamper_target}.tmp" && mv "${tamper_target}.tmp" "${tamper_target}"
run_setup tampered
[[ "${rc}" == 1 ]]
grep -q "sha256 mismatch for ${FILES[0]}" "${WORK}/tampered.log"
rm -f "${tamper_target}"

# 5. A corrupt source for a missing file: the download is discarded, nothing is
#    left in the snapshot, and setup fails closed.
good_source="${SOURCE}/$(source_path "${FILES[0]}")"
cp "${good_source}" "${WORK}/good.bak"
head -c "${size}" /dev/urandom > "${good_source}"
run_setup corrupt_source
[[ "${rc}" == 1 ]]
grep -q "downloaded ${FILES[0]} has sha256 .*; discarded" "${WORK}/corrupt_source.log"
[[ ! -e "${tamper_target}" && ! -e "${tamper_target}.partial" ]]
cp "${WORK}/good.bak" "${good_source}"

# 6. The pins change: a marker written under the old pins is not trusted.
run_setup restore
[[ "${rc}" == 0 ]] || { cat "${WORK}/restore.log"; exit 1; }
new_pins="$(jq -c '.[0].sha256 = ("0" * 64)' <<<"${pins}")"
write_fixture "${new_pins}"
run_setup repinned
[[ "${rc}" == 1 ]]
! grep -q 'verified by marker' "${WORK}/repinned.log"
grep -q "sha256 mismatch for ${FILES[0]}" "${WORK}/repinned.log"
write_fixture "${pins}"

# 7. An explicit MLXFAST_TARGET_SNAPSHOT_DIR (a box) holding the set with no
#    marker: verified in full, nothing fetched, marker written there.
BOX="${WORK}/box snapshot"
mkdir -p "${BOX}"
for f in "${FILES[@]}"; do cp "${SOURCE}/$(source_path "${f}")" "${BOX}/${f}"; done
run_setup box MLXFAST_TARGET_SNAPSHOT_DIR="${BOX}" MLXFAST_TARGET_BASE_URL="file://${WORK}/nowhere"
[[ "${rc}" == 0 ]] || { cat "${WORK}/box.log"; exit 1; }
[[ "$(downloads box)" == 0 ]]
grep -q 'by byte count and sha256' "${WORK}/box.log"
[[ -n "$(ls "${BOX}"/.verified-* 2>/dev/null)" ]]

# 8. The download skip keeps its meaning: nothing fetched, nothing verified.
rm -rf "${DEFAULT_DIR}"
run_setup skipped MLXFAST_SKIP_WEIGHTS_DOWNLOAD=1
[[ "${rc}" == 0 ]]
[[ "$(downloads skipped)" == 0 && ! -d "${DEFAULT_DIR}" ]]
grep -q 'not verifying the checkpoint' "${WORK}/skipped.log"

echo 'test-setup-onboarding.sh: all 8 cases passed'
