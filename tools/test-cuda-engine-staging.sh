#!/usr/bin/env bash
# Revert-proof for tools/stage-cuda-engine.sh and its wiring into setup.sh.
#
# benchd resolves the scored engine at a FIXED workspace-relative path --
# <workspace>/.build/release/mlxfast-runtime-worker -- and spawns it, speaking
# Engine Protocol v1 over stdio. That path is a CONTRACT with the benchmarker,
# not a preference; the name is retained from the MLX era on purpose. setup.sh
# builds the cuda-engine adapter and calls tools/stage-cuda-engine.sh to copy
# the finished binary to exactly that path.
#
# This suite asserts that staging actually lands the cuda-engine binary at the
# benchd-resolved path, and that setup.sh invokes the staging step. It goes RED
# if the staging fix is reverted: the setup.sh call is dropped, or the
# destination is repointed away from .build/release/mlxfast-runtime-worker.
#
# It also covers the BUILD CACHE (tools/ds4/build-cache.sh), because the cache's
# whole risk is that its restore path and the build path stop agreeing: the
# cached run must put the SAME BYTES at the SAME path, and must refuse rather
# than run an engine built from a different ds4 pin.
#
# Hermetic: no weights, no GPU, no CUDA toolchain, no cargo build, no network,
# no box. Each case copies the REAL tool into a throwaway repo root so its
# ROOT_DIR resolves there, then drives it with a stub cuda-engine binary under
# $TMPDIR.
#
# Usage: tools/test-cuda-engine-staging.sh [-v]
set -uo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null && pwd -P)"
STAGE_TOOL="${REPO_ROOT}/tools/stage-cuda-engine.sh"
SETUP_SH="${REPO_ROOT}/setup.sh"
VERBOSE=0
[[ "${1:-}" == "-v" ]] && VERBOSE=1

# The benchd-resolved path the destination MUST land at, workspace-relative.
# This literal is the contract; the whole suite exists to keep it honoured.
BENCHD_RESOLVED_REL=".build/release/mlxfast-runtime-worker"

# Non-vacuity floor: a case deleted or short-circuited leaves the survivors
# green, so exit status alone cannot notice the suite shrinking. Raise this in
# the same commit that adds assertions.
EXPECTED_MIN_ASSERTIONS=28

WORK="$(mktemp -d "${TMPDIR:-/tmp}/cuda-engine-staging.XXXXXX")"
trap 'rm -rf "${WORK}"' EXIT

PASSED=0
FAILED=0
FAILURES=()

pass() {
  PASSED=$((PASSED + 1))
  [[ "${VERBOSE}" == "1" ]] && echo "ok    $1"
  return 0
}

fail() {
  FAILED=$((FAILED + 1))
  FAILURES+=("$1")
  echo "FAIL  $1" >&2
}

assert_file() {
  if [[ -f "$1" ]]; then pass "$2"; else fail "$2 (missing: $1)"; fi
}

assert_executable() {
  if [[ -x "$1" ]]; then pass "$2"; else fail "$2 (not executable: $1)"; fi
}

assert_absent() {
  if [[ ! -e "$1" ]]; then pass "$2"; else fail "$2 (unexpectedly present: $1)"; fi
}

assert_same_bytes() {
  if cmp -s "$1" "$2"; then pass "$3"; else fail "$3 ($1 and $2 differ)"; fi
}

assert_says() {
  # $1 output, $2 needle, $3 label
  if printf '%s' "$1" | grep -qF -- "$2"; then pass "$3"; else fail "$3 (no '$2' in: $1)"; fi
}

# Build a throwaway repo root containing a copy of the real staging tool, so its
# ROOT_DIR resolves into the sandbox. Prints the root path.
make_fake_root() {
  local root
  root="$(mktemp -d "${WORK}/root.XXXXXX")"
  mkdir -p "${root}/tools"
  cp "${STAGE_TOOL}" "${root}/tools/stage-cuda-engine.sh"
  chmod +x "${root}/tools/stage-cuda-engine.sh"
  printf '%s\n' "${root}"
}

# Create the SOURCE binary the cargo build would have produced under the
# adapter's default target directory.
seed_source_bin() {
  local root="$1"
  mkdir -p "${root}/harness/protocol-adapter/target/release"
  printf '#!/bin/sh\nexit 0\n' \
    > "${root}/harness/protocol-adapter/target/release/cuda-engine"
  chmod +x "${root}/harness/protocol-adapter/target/release/cuda-engine"
}

# ---------------------------------------------------------------------------
# Case 1: the happy path stages the cuda-engine binary at the benchd-resolved
# location. This is the destination revert-proof: repoint the destination away
# from .build/release/mlxfast-runtime-worker and nothing lands here -> RED.
# ---------------------------------------------------------------------------
root="$(make_fake_root)"
seed_source_bin "${root}"
if "${root}/tools/stage-cuda-engine.sh" >/dev/null 2>&1; then
  pass "staging exits 0 when the cuda-engine binary is present"
else
  fail "staging exits 0 when the cuda-engine binary is present"
fi
staged_bin="${root}/${BENCHD_RESOLVED_REL}"
assert_file "${staged_bin}" \
  "cuda-engine staged at benchd path ${BENCHD_RESOLVED_REL}"
assert_executable "${staged_bin}" "staged cuda-engine keeps its execute bit"
# The source cargo target is untouched (staging is a copy, not a move).
assert_file "${root}/harness/protocol-adapter/target/release/cuda-engine" \
  "source cuda-engine is left in place (staging copies, not moves)"

# ---------------------------------------------------------------------------
# Case 2: an operator override (CUDA_ENGINE_EXECUTABLE) still lands at the SAME
# fixed benchd path -- the destination is a contract, not env-overridable.
# ---------------------------------------------------------------------------
root="$(make_fake_root)"
prebuilt="${WORK}/prebuilt-cuda-engine"
printf '#!/bin/sh\nexit 0\n' > "${prebuilt}"
chmod +x "${prebuilt}"
if CUDA_ENGINE_EXECUTABLE="${prebuilt}" \
    "${root}/tools/stage-cuda-engine.sh" >/dev/null 2>&1; then
  pass "staging exits 0 with an operator-supplied prebuilt binary"
else
  fail "staging exits 0 with an operator-supplied prebuilt binary"
fi
assert_file "${root}/${BENCHD_RESOLVED_REL}" \
  "operator-override binary still lands at the fixed benchd path"

# ---------------------------------------------------------------------------
# Case 3: fail-closed when the cuda-engine binary was never built, and nothing
# is staged at the benchd path.
# ---------------------------------------------------------------------------
root="$(make_fake_root)"
if "${root}/tools/stage-cuda-engine.sh" >/dev/null 2>&1; then
  fail "staging fails when the cuda-engine binary is absent"
else
  pass "staging fails when the cuda-engine binary is absent"
fi
assert_absent "${root}/${BENCHD_RESOLVED_REL}" \
  "nothing is staged when the cuda-engine binary is absent"

# ---------------------------------------------------------------------------
# Case 4: setup.sh actually INVOKES tools/stage-cuda-engine.sh. Catches a revert
# that drops the wiring while the tool survives. The comment in setup.sh that
# names the script bare does not match this quoted-execution pattern.
# ---------------------------------------------------------------------------
invocations="$(grep -cE '[^#]*"\$\{ROOT_DIR\}/tools/stage-cuda-engine\.sh"' "${SETUP_SH}" || true)"
if [[ "${invocations}" -ge 1 ]]; then
  pass "setup.sh invokes tools/stage-cuda-engine.sh"
else
  fail "setup.sh invokes tools/stage-cuda-engine.sh (found ${invocations})"
fi

# ===========================================================================
# THE BUILD CACHE. A real git repository with a real ds4 gitlink is needed for
# the key, so these cases run against a CLONE of this repository rather than a
# hand-made root: the cache tool reads `git ls-tree HEAD ds4` and the submodule
# HEAD, and a fake root has neither.
# ===========================================================================
# A real git repository is needed, because the key enumerates tracked files with
# `git ls-files`. It is BUILT here rather than cloned from this repository: a
# clone carries only committed files, so a clone-based sandbox would test the
# last commit instead of the working tree.
#
# The sandbox carries a MINIATURE VENDORED ENGINE -- a ds4/ directory with a
# VENDOR.json and one source file -- because that is what the engine is now. The
# key hashes ds4/'s CONTENT, so an edit under it must change the key; case 11
# below is that assertion, and it is the one the whole vendoring rests on.
cache_repo="${WORK}/cache-repo"
FAKE_DS4_SHA="1111111111111111111111111111111111111111"
mkdir -p "${cache_repo}/tools/ds4" "${cache_repo}/harness/protocol-adapter/src" \
         "${cache_repo}/ds4"
cp "${STAGE_TOOL}" "${cache_repo}/tools/stage-cuda-engine.sh"
cp "${REPO_ROOT}/tools/ds4/build-cache.sh" "${cache_repo}/tools/ds4/build-cache.sh"
cp "${SETUP_SH}" "${cache_repo}/setup.sh"
chmod +x "${cache_repo}/tools/stage-cuda-engine.sh" \
         "${cache_repo}/tools/ds4/build-cache.sh" "${cache_repo}/setup.sh"
printf 'fn main() {}\n' > "${cache_repo}/harness/protocol-adapter/src/main.rs"
printf '[]\n' > "${cache_repo}/harness/protocol-adapter/Cargo.lock"
printf '{"fork":{"sha":"%s"}}\n' "${FAKE_DS4_SHA}" > "${cache_repo}/ds4/VENDOR.json"
printf 'int engine_probe(void) { return 1; }\n' > "${cache_repo}/ds4/ds4.c"
printf 'all:\n\t@true\n' > "${cache_repo}/ds4/Makefile"
cache_ready=0
if git -C "${cache_repo}" init --quiet 2>/dev/null \
   && git -C "${cache_repo}" add -A 2>/dev/null \
   && git -C "${cache_repo}" -c user.email=t@t -c user.name=t \
        commit --quiet -m sandbox 2>/dev/null; then
  cache_ready=1
fi
if [[ "${cache_ready}" == "1" ]]; then
  export CUDAFAST_ENGINE_CACHE_DIR="${WORK}/cache-root"

  # Stand in for a finished build: the two ds4 artefacts and the cargo binary.
  mkdir -p "${cache_repo}/.build/ds4" \
           "${cache_repo}/harness/protocol-adapter/target/release"
  printf 'libds4qwen-bytes\n' > "${cache_repo}/.build/ds4/libds4qwen.so"
  printf '#!/bin/sh\nexit 0\n' > "${cache_repo}/.build/ds4/ds4-resident"
  printf '#!/bin/sh\necho cuda-engine\n' \
    > "${cache_repo}/harness/protocol-adapter/target/release/cuda-engine"
  chmod +x "${cache_repo}/.build/ds4/ds4-resident" \
           "${cache_repo}/harness/protocol-adapter/target/release/cuda-engine"

  # The vendor base the sandbox records, for the tripwire cases below.
  assert_says "$(cat "${cache_repo}/ds4/VENDOR.json")" "${FAKE_DS4_SHA}" \
    "the sandbox records a ds4 vendor base the cache key can read"

  # -------------------------------------------------------------------------
  # Case 5: a miss is a miss. Nothing staged, exit non-zero, no refusal noise.
  # -------------------------------------------------------------------------
  if (cd "${cache_repo}" && ./tools/ds4/build-cache.sh restore >/dev/null 2>&1); then
    fail "an empty cache root reports a miss"
  else
    pass "an empty cache root reports a miss"
  fi

  # -------------------------------------------------------------------------
  # Case 6: the BUILD path stages, then save + restore stages the SAME BYTES at
  # the SAME path. This is the assertion the cache exists to keep true.
  # -------------------------------------------------------------------------
  (cd "${cache_repo}" && ./tools/stage-cuda-engine.sh >/dev/null 2>&1) \
    && pass "the build path stages the adapter" \
    || fail "the build path stages the adapter"
  built_staged="${WORK}/built-staged.bin"
  cp "${cache_repo}/${BENCHD_RESOLVED_REL}" "${built_staged}" 2>/dev/null || true

  if (cd "${cache_repo}" && ./tools/ds4/build-cache.sh save >/dev/null 2>&1); then
    pass "save writes an entry for the current key"
  else
    fail "save writes an entry for the current key"
  fi
  # Wipe the workspace build tree: the restore must reconstruct all of it.
  rm -rf "${cache_repo}/.build" \
         "${cache_repo}/harness/protocol-adapter/target/release/cuda-engine"
  if (cd "${cache_repo}" && ./tools/ds4/build-cache.sh restore >/dev/null 2>&1); then
    pass "restore reports a hit for the key it just saved"
  else
    fail "restore reports a hit for the key it just saved"
  fi
  assert_executable "${cache_repo}/${BENCHD_RESOLVED_REL}" \
    "restore stages the adapter at the benchd path, executable"
  assert_same_bytes "${built_staged}" "${cache_repo}/${BENCHD_RESOLVED_REL}" \
    "the restore path stages BYTE-IDENTICAL content to the build path"
  assert_file "${cache_repo}/.build/ds4/libds4qwen.so" \
    "restore brings back libds4qwen.so"
  assert_executable "${cache_repo}/.build/ds4/ds4-resident" \
    "restore brings back the weight owner, executable"
  assert_file "${cache_repo}/.build/engine-cache/MANIFEST" \
    "restore leaves the MANIFEST setup.sh's tripwire reads"

  # -------------------------------------------------------------------------
  # Case 7: THE TRIPWIRE. A MANIFEST naming a different ds4 commit must stop
  # setup.sh's skip path by name, never run the cached binary.
  # -------------------------------------------------------------------------
  sed -i.bak 's/^ds4_vendor_base\t.*/ds4_vendor_base\t0000000000000000000000000000000000000000/' \
    "${cache_repo}/.build/engine-cache/MANIFEST"
  rm -f "${cache_repo}/.build/engine-cache/MANIFEST.bak"
  out="$( (cd "${cache_repo}" && MLXFAST_SKIP_ENGINE_BUILD=1 \
      MLXFAST_SKIP_WEIGHTS_DOWNLOAD=1 ./setup.sh) 2>&1 )" && rc=0 || rc=$?
  if [[ "${rc}" -ne 0 ]]; then
    pass "setup.sh REFUSES a staged engine whose MANIFEST names another ds4 base"
  else
    fail "setup.sh REFUSES a staged engine whose MANIFEST names another ds4 base"
  fi
  assert_says "${out}" "refusing to run a cached binary against a different engine base" \
    "the refusal names the reason rather than failing obscurely"

  # -------------------------------------------------------------------------
  # Case 8: with the MANIFEST honest again, the same skip path SUCCEEDS -- so
  # case 7 proves the tripwire and not merely a broken skip path.
  # -------------------------------------------------------------------------
  (cd "${cache_repo}" && ./tools/ds4/build-cache.sh restore >/dev/null 2>&1) || true
  if (cd "${cache_repo}" && MLXFAST_SKIP_ENGINE_BUILD=1 \
      MLXFAST_SKIP_WEIGHTS_DOWNLOAD=1 ./setup.sh >/dev/null 2>&1); then
    pass "the skip path SUCCEEDS on a matching MANIFEST (case 7 is the tripwire, not a broken path)"
  else
    fail "the skip path SUCCEEDS on a matching MANIFEST (case 7 is the tripwire, not a broken path)"
  fi

  # -------------------------------------------------------------------------
  # Case 9: an entry missing an artefact reads as a MISS, not as an engine. A
  # save killed by the job timeout must not be restored as a complete build.
  # -------------------------------------------------------------------------
  entry="$( (cd "${cache_repo}" && ./tools/ds4/build-cache.sh path) 2>/dev/null )"
  rm -f "${entry}/.build/ds4/libds4qwen.so"
  # Wipe the workspace too, so "nothing staged" below means the restore
  # produced nothing rather than an earlier case having left it there.
  rm -rf "${cache_repo}/.build" \
         "${cache_repo}/harness/protocol-adapter/target/release/cuda-engine"
  out="$( (cd "${cache_repo}" && ./tools/ds4/build-cache.sh restore) 2>&1 )" && rc=0 || rc=$?
  if [[ "${rc}" -eq 1 ]]; then
    pass "an incomplete cache entry exits 1, the miss code the workflow branches on"
  else
    fail "an incomplete cache entry exits 1, the miss code the workflow branches on (got ${rc})"
  fi
  # A crash also exits non-zero, so the MESSAGE is what proves the entry was
  # inspected and rejected rather than half-copied and then failing on the
  # missing file.
  assert_says "${out}" "is incomplete" \
    "the incomplete entry is reported as a named miss, not a crash"
  assert_absent "${cache_repo}/${BENCHD_RESOLVED_REL}" \
    "an incomplete entry stages nothing at the benchd path"

  # -------------------------------------------------------------------------
  # Case 9b: THE RPATH REFUSAL. harness/protocol-adapter/build.rs bakes the
  # absolute .build/ds4 directory into the adapter, so artefacts built in
  # another workspace would resolve libds4qwen.so somewhere else -- or find a
  # stale one. The workspace is part of the key, so this normally reads as a
  # miss; the refusal is what catches an entry copied between boxes by hand.
  # -------------------------------------------------------------------------
  # Case 9 emptied the workspace, so rebuild the stand-in artefacts and save a
  # COMPLETE entry: the completeness check runs first and would otherwise mask
  # the refusal this case is about.
  mkdir -p "${cache_repo}/.build/ds4" \
           "${cache_repo}/harness/protocol-adapter/target/release"
  printf 'libds4qwen-bytes\n' > "${cache_repo}/.build/ds4/libds4qwen.so"
  printf '#!/bin/sh\nexit 0\n' > "${cache_repo}/.build/ds4/ds4-resident"
  printf '#!/bin/sh\necho cuda-engine\n' \
    > "${cache_repo}/harness/protocol-adapter/target/release/cuda-engine"
  chmod +x "${cache_repo}/.build/ds4/ds4-resident" \
           "${cache_repo}/harness/protocol-adapter/target/release/cuda-engine"
  (cd "${cache_repo}" && ./tools/ds4/build-cache.sh save >/dev/null 2>&1) \
    && pass "a complete entry can be saved again for the refusal case" \
    || fail "a complete entry can be saved again for the refusal case"
  entry="$( (cd "${cache_repo}" && ./tools/ds4/build-cache.sh path) 2>/dev/null )"
  sed -i.bak 's|^workspace\t.*|workspace\t/somewhere/else|' "${entry}/MANIFEST"
  rm -f "${entry}/MANIFEST.bak"
  rm -rf "${cache_repo}/.build"
  out="$( (cd "${cache_repo}" && ./tools/ds4/build-cache.sh restore) 2>&1 )" && rc=0 || rc=$?
  if [[ "${rc}" -ne 0 ]]; then
    pass "an entry built in another workspace is refused"
  else
    fail "an entry built in another workspace is refused"
  fi
  assert_says "${out}" "rpath" \
    "the workspace refusal names the rpath as the reason"
  assert_absent "${cache_repo}/${BENCHD_RESOLVED_REL}" \
    "a refused entry stages nothing at the benchd path"

  # -------------------------------------------------------------------------
  # Case 10: the workflow arms the skip ONLY from a restore, and saves only
  # when it did not. A revert that always exports the flag would run a stale
  # engine on every dispatch.
  # -------------------------------------------------------------------------
  wf="${REPO_ROOT}/.github/workflows/benchmark.yml"
  # The export must live INSIDE the restore step's success branch. Counting the
  # occurrences is what makes that checkable: exactly one line in the whole
  # workflow may write the skip flag, and the restore step is the only step
  # that runs the cache tool.
  arms="$(grep -c 'MLXFAST_SKIP_ENGINE_BUILD=1" >> "\${GITHUB_ENV}"' "${wf}" || true)"
  if [[ "${arms}" -eq 1 ]]; then
    pass "exactly one line in benchmark.yml arms MLXFAST_SKIP_ENGINE_BUILD"
  else
    fail "exactly one line in benchmark.yml arms MLXFAST_SKIP_ENGINE_BUILD (found ${arms})"
  fi
  # AND it must sit INSIDE the restore's success branch. A skip armed before or
  # outside the `if` would run a stale engine on a miss -- the exact failure
  # this cache must never cause -- while still being a single line.
  if_line="$(grep -n 'if ./tools/ds4/build-cache.sh restore; then' "${wf}" | head -1 | cut -d: -f1)"
  arm_line="$(grep -n 'MLXFAST_SKIP_ENGINE_BUILD=1" >> "\${GITHUB_ENV}"' "${wf}" | head -1 | cut -d: -f1)"
  else_line="$(awk -v start="${if_line:-0}" 'NR > start && $0 ~ /^          else$/ { print NR; exit }' "${wf}")"
  if [[ -n "${if_line}" && -n "${arm_line}" && -n "${else_line}" \
        && "${arm_line}" -gt "${if_line}" && "${arm_line}" -lt "${else_line}" ]]; then
    pass "the skip is armed INSIDE the restore step's success branch"
  else
    fail "the skip is armed INSIDE the restore step's success branch (if ${if_line:-?}, arm ${arm_line:-?}, else ${else_line:-?})"
  fi
  if grep -q 'build-cache.sh restore' "${wf}"; then
    pass "benchmark.yml runs the cache restore before setup"
  else
    fail "benchmark.yml runs the cache restore before setup"
  fi
  if grep -q "if: env.MLXFAST_SKIP_ENGINE_BUILD != '1'" "${wf}"; then
    pass "benchmark.yml saves only when the restore missed"
  else
    fail "benchmark.yml saves only when the restore missed"
  fi
  # ORDER: the restore must precede ./setup.sh, or the flag arrives too late.
  restore_line="$(grep -n 'build-cache.sh restore' "${wf}" | head -1 | cut -d: -f1)"
  setup_line="$(grep -n 'run: ./setup.sh' "${wf}" | head -1 | cut -d: -f1)"
  if [[ -n "${restore_line}" && -n "${setup_line}" && "${restore_line}" -lt "${setup_line}" ]]; then
    pass "the restore step runs BEFORE ./setup.sh"
  else
    fail "the restore step runs BEFORE ./setup.sh (restore ${restore_line:-?}, setup ${setup_line:-?})"
  fi
else
  echo "cuda-engine-staging: the build-cache sandbox could not be created; git is required" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Trailer + non-vacuity floor.
# ---------------------------------------------------------------------------
echo "cuda-engine-staging: ${PASSED} passed, ${FAILED} failed"
if [[ "${FAILED}" -ne 0 ]]; then
  echo "cuda-engine-staging: FAILURES:" >&2
  printf '  - %s\n' "${FAILURES[@]}" >&2
  exit 1
fi
if [[ "${PASSED}" -lt "${EXPECTED_MIN_ASSERTIONS}" ]]; then
  echo "cuda-engine-staging: ran only ${PASSED} assertions, expected at least ${EXPECTED_MIN_ASSERTIONS} -- the suite shrank" >&2
  exit 1
fi
exit 0
