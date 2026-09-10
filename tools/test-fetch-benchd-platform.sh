#!/usr/bin/env bash
#
# test-fetch-benchd-platform.sh -- the platform gate in tools/fetch-benchd.sh.
#
# WHY THIS TEST EXISTS. The shared bench channel publishes one benchd pair per
# platform, and both pairs carry the same six manifest fields. Every check the
# resolver had before this gate -- branch, byte count, sha256 -- PASSES on the
# wrong pair. Only `target_triple` and the binary's own container format
# separate them, so those two checks are the whole gate and they need a test
# that fails when either one stops working.
#
# Fully hermetic: no network, no toolchain, no benchd. Each case builds a
# throwaway dist directory (a four-byte "binary" carrying real ELF or Mach-O
# magic, plus a hand-written manifest) and drives the REAL resolver against it
# through BENCHD_DIST_LOCAL.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FETCH="${REPO_ROOT}/tools/fetch-benchd.sh"
BRANCH="qwen3.8-125b-a6b-v1"

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

# A four-byte file whose first four bytes are the named container's magic.
# Enough for the gate: it reads exactly those four bytes.
write_binary() {
  local path="$1" format="$2"
  case "${format}" in
    elf) printf '\177ELF' > "${path}" ;;
    macho) printf '\317\372\355\376' > "${path}" ;;
    *) printf 'JUNK' > "${path}" ;;
  esac
}

# The channel's manifest, values-only, one key per line -- the shape
# scripts/build-dist.sh writes and the resolver's sed parser expects.
write_manifest() {
  local dir="$1" triple="$2" binary="${1}/benchd"
  local sha bytes
  sha="$(shasum -a 256 "${binary}" | awk '{print $1}')"
  bytes="$(wc -c < "${binary}" | tr -d '[:space:]')"
  {
    echo '{'
    echo '  "version": "0.0.0-test",'
    echo "  \"branch\": \"${BRANCH}\","
    echo '  "source_commit": "da3dcb844075ef4608c4f5f63400d30ad0a5720e",'
    if [[ -n "${triple}" ]]; then
      echo "  \"target_triple\": \"${triple}\","
    fi
    echo "  \"sha256\": \"${sha}\","
    echo "  \"bytes\": ${bytes}"
    echo '}'
  } > "${dir}/benchd.manifest.json"
}

# Build a dist directory and resolve it. Echoes the resolver's combined output;
# returns its exit code.
resolve() {
  local case_name="$1" format="$2" manifest_triple="$3" expect_triple="$4"
  local dist="${WORK}/${case_name}/dist" bin_dir="${WORK}/${case_name}/benchd-bin"
  mkdir -p "${dist}" "${bin_dir}"
  write_binary "${dist}/benchd" "${format}"
  write_manifest "${dist}" "${manifest_triple}"
  env \
    BENCHD_BRANCH="${BRANCH}" \
    BENCHD_BIN_DIR="${bin_dir}" \
    BENCHD_DIST_LOCAL="${dist}" \
    BENCHD_EXPECT_TARGET_TRIPLE="${expect_triple}" \
    "${FETCH}" 2>&1
}

# -- 1. the host's own pair installs --------------------------------------
out="$(resolve accept elf aarch64-unknown-linux-gnu aarch64-unknown-linux-gnu)"
rc=$?
if [[ "${rc}" -eq 0 ]]; then
  ok "the pair whose target_triple matches the host installs"
else
  bad "the pair whose target_triple matches the host installs" "exit ${rc}: ${out}"
fi
if [[ -f "${WORK}/accept/benchd-bin/benchd" ]]; then
  ok "the accepted binary is installed at BENCHD_BIN_DIR/benchd"
else
  bad "the accepted binary is installed at BENCHD_BIN_DIR/benchd" "not present"
fi
if [[ -f "${WORK}/accept/benchd-bin/benchd.manifest.json" ]]; then
  ok "the manifest is installed beside it"
else
  bad "the manifest is installed beside it" "not present"
fi

# -- 2. the macOS pair on a Linux host is REFUSED BY NAME ------------------
# This is the case the gate exists for: every other check passes.
out="$(resolve wrong_platform macho aarch64-apple-darwin aarch64-unknown-linux-gnu)"
rc=$?
if [[ "${rc}" -ne 0 ]]; then
  ok "the macOS pair is refused on a Linux host"
else
  bad "the macOS pair is refused on a Linux host" "exit 0: ${out}"
fi
case "${out}" in
  *WRONG\ PLATFORM*aarch64-apple-darwin*aarch64-unknown-linux-gnu*)
    ok "the refusal names both triples" ;;
  *) bad "the refusal names both triples" "${out}" ;;
esac
if [[ ! -f "${WORK}/wrong_platform/benchd-bin/benchd" ]]; then
  ok "nothing is installed on a platform refusal"
else
  bad "nothing is installed on a platform refusal" "a binary was installed"
fi

# -- 3. a MIS-STAMPED pair is refused on the bytes -------------------------
# The manifest claims Linux; the bytes are Mach-O. The triple check passes and
# the format check is what catches it.
out="$(resolve mis_stamped macho aarch64-unknown-linux-gnu aarch64-unknown-linux-gnu)"
rc=$?
if [[ "${rc}" -ne 0 ]]; then
  ok "a manifest that mis-states the platform is refused on the bytes"
else
  bad "a manifest that mis-states the platform is refused on the bytes" "exit 0: ${out}"
fi
case "${out}" in
  *MIS-STAMPED*) ok "the mis-stamp refusal says so by name" ;;
  *) bad "the mis-stamp refusal says so by name" "${out}" ;;
esac

# -- 4. a manifest with no target_triple is refused ------------------------
out="$(resolve no_triple elf "" aarch64-unknown-linux-gnu)"
rc=$?
if [[ "${rc}" -ne 0 ]]; then
  ok "a manifest carrying no target_triple is refused"
else
  bad "a manifest carrying no target_triple is refused" "exit 0: ${out}"
fi

# -- 5. a host the channel publishes nothing for is refused ----------------
out="$(resolve unknown_host elf aarch64-unknown-linux-gnu x86_64-apple-darwin)"
rc=$?
if [[ "${rc}" -ne 0 ]]; then
  ok "a host with no published dist lane is refused"
else
  bad "a host with no published dist lane is refused" "exit 0: ${out}"
fi

# -- 5b. an x86_64 Linux host installs the x86_64 pair ---------------------
out="$(resolve x86_host elf x86_64-unknown-linux-gnu x86_64-unknown-linux-gnu)"
rc=$?
if [[ "${rc}" -eq 0 && -f "${WORK}/x86_host/benchd-bin/benchd" ]]; then
  ok "an x86_64 Linux host installs the x86_64-unknown-linux-gnu pair"
else
  bad "an x86_64 Linux host installs the x86_64-unknown-linux-gnu pair" "exit ${rc}: ${out}"
fi

# -- 5c. the x86_64 pair is refused on the aarch64 box ---------------------
out="$(resolve x86_on_box elf x86_64-unknown-linux-gnu aarch64-unknown-linux-gnu)"
rc=$?
if [[ "${rc}" -ne 0 ]]; then
  ok "the x86_64 pair is refused on the aarch64 box"
else
  bad "the x86_64 pair is refused on the aarch64 box" "exit 0: ${out}"
fi

# -- 6. the check also runs on an ALREADY-INSTALLED pair -------------------
# The offline path must not become the way a wrong-platform binary gets used.
mkdir -p "${WORK}/preplaced/benchd-bin"
write_binary "${WORK}/preplaced/benchd-bin/benchd" macho
write_manifest "${WORK}/preplaced/benchd-bin" aarch64-apple-darwin
out="$(env \
  BENCHD_BRANCH="${BRANCH}" \
  BENCHD_BIN_DIR="${WORK}/preplaced/benchd-bin" \
  BENCHD_EXPECT_TARGET_TRIPLE=aarch64-unknown-linux-gnu \
  "${FETCH}" 2>&1)"
rc=$?
if [[ "${rc}" -ne 0 ]]; then
  ok "an already-installed wrong-platform pair is refused, not reused"
else
  bad "an already-installed wrong-platform pair is refused, not reused" "exit 0: ${out}"
fi

echo "fetch-benchd platform gate: ${passed} passed, ${failed} failed"
[[ "${failed}" -eq 0 ]]
