#!/usr/bin/env bash
#
# test-fetch-benchd-recorder.sh -- the SECOND CHANNEL BINARY in
# tools/fetch-benchd.sh.
#
# WHY THIS TEST EXISTS. The dist channel published one binary, and the
# manifest's six top-level fields WERE the benchd pin. bench pull request
# #255 adds `record-correctness-golden` beside benchd and declares it in a
# `binaries` object, without moving those six fields. So TWO MANIFEST SHAPES
# ARE LIVE at once -- the qwen3.8-125b-a6b-v1 channel still serves the legacy
# one until its republish -- and the resolver has to read both:
#
#   * the legacy shape must behave EXACTLY as it did: benchd alone, no
#     recorder fetched, no recorder required;
#   * the new shape must fetch the recorder from the same platform directory,
#     verify it against ITS OWN entry, and stage it beside benchd;
#   * a recorder that does not verify must be refused BY NAME, with nothing
#     installed -- a channel half-obtained is a publish defect, not a smaller
#     channel.
#
# Fully hermetic: no network, no toolchain, no benchd. Each case builds a
# throwaway dist directory (four-byte "binaries" carrying real ELF magic, plus
# a hand-written manifest) and drives the REAL resolver against it through
# BENCHD_DIST_LOCAL.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FETCH="${REPO_ROOT}/tools/fetch-benchd.sh"
BRANCH="qwen3.8-125b-a6b-v1"
TRIPLE="aarch64-unknown-linux-gnu"
RECORDER="record-correctness-golden"

WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

passed=0
failed=0

ok() {
  echo "ok    $1"
  passed=$((passed + 1))
}

bad() {
  echo "FAIL  $1"
  echo "      $2"
  failed=$((failed + 1))
}

sha_of()   { shasum -a 256 "$1" | awk '{print $1}'; }
bytes_of() { wc -c < "$1" | tr -d '[:space:]'; }

# The six-field manifest the channel published before the second binary, and
# still publishes on this branch today.
write_legacy_manifest() {
  local dir="$1" bin="$1/benchd"
  {
    echo '{'
    echo '  "version": "0.0.0-test",'
    echo "  \"branch\": \"${BRANCH}\","
    echo '  "source_commit": "da3dcb844075ef4608c4f5f63400d30ad0a5720e",'
    echo "  \"target_triple\": \"${TRIPLE}\","
    echo "  \"sha256\": \"$(sha_of "${bin}")\","
    echo "  \"bytes\": $(bytes_of "${bin}")"
    echo '}'
  } > "${dir}/benchd.manifest.json"
}

# The #255 manifest: the SAME six fields, describing benchd, plus one line
# per binary. The per-binary entries are written on ONE LINE each, which is
# what keeps the anchored six-field sed blind to their nested keys.
#   write_binaries_manifest <dir> [<recorder sha256> <recorder bytes>]
# The two optional arguments override what the entry CLAIMS, so a test can
# declare a recorder that the staged file does not match.
write_binaries_manifest() {
  local dir="$1" bin="$1/benchd" rec="$1/${RECORDER}"
  local rec_sha="${2:-}" rec_bytes="${3:-}"
  [[ -n "${rec_sha}" ]]   || rec_sha="$(sha_of "${rec}")"
  [[ -n "${rec_bytes}" ]] || rec_bytes="$(bytes_of "${rec}")"
  {
    echo '{'
    echo '  "version": "0.0.0-test",'
    echo "  \"branch\": \"${BRANCH}\","
    echo '  "source_commit": "da3dcb844075ef4608c4f5f63400d30ad0a5720e",'
    echo "  \"target_triple\": \"${TRIPLE}\","
    echo "  \"sha256\": \"$(sha_of "${bin}")\","
    echo "  \"bytes\": $(bytes_of "${bin}"),"
    echo '  "binaries": {'
    echo "    \"benchd\": {\"sha256\": \"$(sha_of "${bin}")\", \"bytes\": $(bytes_of "${bin}")},"
    echo "    \"${RECORDER}\": {\"sha256\": \"${rec_sha}\", \"bytes\": ${rec_bytes}}"
    echo '  }'
    echo '}'
  } > "${dir}/benchd.manifest.json"
}

# Resolve a prepared dist directory into a fresh install directory.
resolve() {
  local case_name="$1"
  env \
    BENCHD_BRANCH="${BRANCH}" \
    BENCHD_BIN_DIR="${WORK}/${case_name}/benchd-bin" \
    BENCHD_DIST_LOCAL="${WORK}/${case_name}/dist" \
    BENCHD_EXPECT_TARGET_TRIPLE="${TRIPLE}" \
    "${FETCH}" 2>&1
}

# A case directory holding an ELF benchd and, optionally, an ELF recorder.
new_case() {
  local case_name="$1" with_recorder="$2"
  local dist="${WORK}/${case_name}/dist"
  mkdir -p "${dist}" "${WORK}/${case_name}/benchd-bin"
  printf '\177ELFbenchd' > "${dist}/benchd"
  if [[ "${with_recorder}" == "with-recorder" ]]; then
    printf '\177ELFrecorder-bytes' > "${dist}/${RECORDER}"
  fi
  printf '%s' "${dist}"
}

# -- 1. THE LEGACY MANIFEST IS UNCHANGED BEHAVIOUR ----------------------------
dist="$(new_case legacy no-recorder)"
write_legacy_manifest "${dist}"
out="$(resolve legacy)"
rc=$?
if [[ "${rc}" -eq 0 ]]; then
  ok "a legacy six-field manifest resolves"
else
  bad "a legacy six-field manifest resolves" "exit ${rc}: ${out}"
fi
if [[ -f "${WORK}/legacy/benchd-bin/benchd" && -f "${WORK}/legacy/benchd-bin/benchd.manifest.json" ]]; then
  ok "the legacy pair installs"
else
  bad "the legacy pair installs" "${out}"
fi
if [[ ! -e "${WORK}/legacy/benchd-bin/${RECORDER}" ]]; then
  ok "no recorder is staged from a manifest that declares none"
else
  bad "no recorder is staged from a manifest that declares none" "a recorder appeared"
fi
case "${out}" in
  *"declares no ${RECORDER}"*) ok "the resolve says out loud that this channel carries benchd alone" ;;
  *) bad "the resolve says out loud that this channel carries benchd alone" "${out}" ;;
esac

# -- 2. THE #255 MANIFEST STAGES BOTH BINARIES --------------------------------
dist="$(new_case both with-recorder)"
write_binaries_manifest "${dist}"
out="$(resolve both)"
rc=$?
if [[ "${rc}" -eq 0 ]]; then
  ok "a manifest declaring ${RECORDER} resolves"
else
  bad "a manifest declaring ${RECORDER} resolves" "exit ${rc}: ${out}"
fi
if [[ -f "${WORK}/both/benchd-bin/${RECORDER}" ]]; then
  ok "the recorder is staged beside benchd in the install directory"
else
  bad "the recorder is staged beside benchd in the install directory" "${out}"
fi
if [[ -x "${WORK}/both/benchd-bin/${RECORDER}" ]]; then
  ok "the staged recorder is executable"
else
  bad "the staged recorder is executable" "mode $(ls -l "${WORK}/both/benchd-bin/${RECORDER}" 2>&1)"
fi
if [[ "$(cat "${WORK}/both/benchd-bin/${RECORDER}")" == "$(cat "${dist}/${RECORDER}")" ]]; then
  ok "the staged recorder is the channel's bytes"
else
  bad "the staged recorder is the channel's bytes" "content differs"
fi
# The resolver's STDOUT contract is one path: benchd's. The recorder is read
# from beside it, never from stdout.
stdout_only="$(env \
  BENCHD_BRANCH="${BRANCH}" \
  BENCHD_BIN_DIR="${WORK}/both/benchd-bin" \
  BENCHD_EXPECT_TARGET_TRIPLE="${TRIPLE}" \
  "${FETCH}" 2>/dev/null)"
if [[ "${stdout_only}" == "${WORK}/both/benchd-bin/benchd" ]]; then
  ok "stdout is still exactly one path, benchd's"
else
  bad "stdout is still exactly one path, benchd's" "'${stdout_only}'"
fi

# -- 3. THE SIX LEGACY FIELDS STILL READ AS SINGLE VALUES ---------------------
# The whole two-shape design rests on the per-binary entries being ONE LINE, so
# the anchored six-field expression cannot see their nested sha256/bytes keys.
# Read them with the LEGACY expression, verbatim, out of the new manifest.
legacy_field() {
  sed -n "s/^[[:space:]]*\"$2\"[[:space:]]*:[[:space:]]*\"\{0,1\}\([^\",]*\)\"\{0,1\}[[:space:]]*,\{0,1\}[[:space:]]*\$/\1/p" "$1"
}
new_manifest="${WORK}/both/benchd-bin/benchd.manifest.json"
for key in sha256 bytes branch source_commit target_triple; do
  n="$(legacy_field "${new_manifest}" "${key}" | wc -l | tr -d '[:space:]')"
  if [[ "${n}" -eq 1 ]]; then
    ok "the legacy reader still finds exactly one '${key}' in the #255 manifest"
  else
    bad "the legacy reader still finds exactly one '${key}' in the #255 manifest" "found ${n}"
  fi
done

# -- 4. A TAMPERED RECORDER IS REFUSED BY NAME --------------------------------
# The declared entry is written from the honest bytes, then the file is edited
# to the SAME LENGTH -- so the byte count agrees and only the digest catches it.
dist="$(new_case tampered with-recorder)"
write_binaries_manifest "${dist}"
printf '\177ELFrecorder-BYTES' > "${dist}/${RECORDER}"
out="$(resolve tampered)"
rc=$?
if [[ "${rc}" -ne 0 ]]; then
  ok "a recorder whose bytes are not the declared ones is refused"
else
  bad "a recorder whose bytes are not the declared ones is refused" "exit 0: ${out}"
fi
case "${out}" in
  *"RECORDER MISMATCH"*sha256*) ok "the refusal names RECORDER MISMATCH and the digest" ;;
  *) bad "the refusal names RECORDER MISMATCH and the digest" "${out}" ;;
esac
if [[ ! -e "${WORK}/tampered/benchd-bin/benchd" ]]; then
  ok "benchd is NOT installed when the recorder beside it fails to verify"
else
  bad "benchd is NOT installed when the recorder beside it fails to verify" "benchd was installed"
fi

# -- 5. A TRUNCATED RECORDER IS CAUGHT ON THE BYTE COUNT ----------------------
dist="$(new_case truncated with-recorder)"
write_binaries_manifest "${dist}"
printf '\177ELF' > "${dist}/${RECORDER}"
out="$(resolve truncated)"
rc=$?
if [[ "${rc}" -ne 0 ]]; then
  ok "a truncated recorder is refused"
else
  bad "a truncated recorder is refused" "exit 0: ${out}"
fi
case "${out}" in
  *"RECORDER MISMATCH"*"byte count"*) ok "the refusal names the byte count" ;;
  *) bad "the refusal names the byte count" "${out}" ;;
esac

# -- 6. A DECLARED RECORDER THAT IS NOT THERE IS REFUSED BY NAME --------------
dist="$(new_case absent with-recorder)"
write_binaries_manifest "${dist}"
rm -f "${dist}/${RECORDER}"
out="$(resolve absent)"
rc=$?
if [[ "${rc}" -ne 0 ]]; then
  ok "a manifest that declares a recorder the dist does not carry is refused"
else
  bad "a manifest that declares a recorder the dist does not carry is refused" "exit 0: ${out}"
fi
case "${out}" in
  *"RECORDER MISSING"*) ok "the refusal names RECORDER MISSING" ;;
  *) bad "the refusal names RECORDER MISSING" "${out}" ;;
esac

# -- 7. A HALF-DECLARED ENTRY IS REFUSED, NOT READ AS LEGACY ------------------
# A digest with no byte count would verify against nothing. It must refuse
# rather than fall back to "this manifest declares no recorder".
dist="$(new_case half with-recorder)"
write_binaries_manifest "${dist}"
python3 - "${dist}/benchd.manifest.json" "${RECORDER}" <<'PY'
import re, sys
path, name = sys.argv[1], sys.argv[2]
text = open(path).read()
text = re.sub(r'("%s": \{"sha256": "[0-9a-f]{64}"), "bytes": \d+\}' % re.escape(name),
              r'\1, "bytes": }', text)
open(path, "w").write(text)
PY
out="$(resolve half)"
rc=$?
if [[ "${rc}" -ne 0 ]]; then
  ok "a half-declared recorder entry is refused"
else
  bad "a half-declared recorder entry is refused" "exit 0: ${out}"
fi

# -- 8. THE OFFLINE PATH CHECKS THE RECORDER TOO ------------------------------
# The ranked box gets the files placed, not downloaded. An installed set whose
# manifest declares a recorder that is missing must refuse there too, or the
# offline path becomes the way an unverified set gets used.
mkdir -p "${WORK}/preplaced/benchd-bin"
printf '\177ELFbenchd' > "${WORK}/preplaced/benchd-bin/benchd"
printf '\177ELFrecorder-bytes' > "${WORK}/preplaced/benchd-bin/${RECORDER}"
write_binaries_manifest "${WORK}/preplaced/benchd-bin"
out="$(env \
  BENCHD_BRANCH="${BRANCH}" \
  BENCHD_BIN_DIR="${WORK}/preplaced/benchd-bin" \
  BENCHD_EXPECT_TARGET_TRIPLE="${TRIPLE}" \
  "${FETCH}" 2>&1)"
rc=$?
if [[ "${rc}" -eq 0 ]]; then
  ok "an already-installed set carrying both binaries is reused"
else
  bad "an already-installed set carrying both binaries is reused" "exit ${rc}: ${out}"
fi
rm -f "${WORK}/preplaced/benchd-bin/${RECORDER}"
out="$(env \
  BENCHD_BRANCH="${BRANCH}" \
  BENCHD_BIN_DIR="${WORK}/preplaced/benchd-bin" \
  BENCHD_EXPECT_TARGET_TRIPLE="${TRIPLE}" \
  "${FETCH}" 2>&1)"
rc=$?
if [[ "${rc}" -ne 0 ]]; then
  ok "an already-installed set missing its declared recorder is refused, not reused"
else
  bad "an already-installed set missing its declared recorder is refused, not reused" "exit 0: ${out}"
fi
case "${out}" in
  *"RECORDER MISSING"*) ok "the offline refusal names RECORDER MISSING" ;;
  *) bad "the offline refusal names RECORDER MISSING" "${out}" ;;
esac

echo "fetch-benchd recorder: ${passed} passed, ${failed} failed"
[[ "${failed}" -eq 0 ]]
