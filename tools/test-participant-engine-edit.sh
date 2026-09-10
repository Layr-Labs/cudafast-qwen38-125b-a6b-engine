#!/usr/bin/env bash
# The vendoring proof: a participant may EDIT THE ENGINE, and the gates agree.
#
# ds4/ used to be a submodule pinned to the internal Layr-Labs/ds4. A participant
# could not read it, fork it, diff it or submit a change to it. It is vendored
# now -- plain files, declared in benchmark.json editablePaths -- and this suite
# is what keeps that true. It asserts the properties the change rests on:
#
#   1. AN ENGINE EDIT IS IN THE SURFACE. benchmark.json editablePaths carries
#      ds4/, and that manifest is what Yukon archives and benchd enforces.
#   2. AN ENGINE EDIT IS BUILT, NOT CACHED PAST. The build-cache key hashes the
#      vendored tree's CONTENT, so an edit misses the cache and is rebuilt. A
#      key that ignored ds4/ would serve the participant a binary built from
#      somebody else's engine, which is the worst failure this change could have.
#   3. AN ENGINE EDIT STILL BUILDS. tools/ds4/build.sh --cpu-check passes on the
#      edited tree.
#   4. THE HARNESS HASH DOES NOT MOVE. benchd seals a harness hash over a FIXED
#      root set that does NOT include ds4/ -- so an engine edit leaves the
#      harness identity alone, while a harness edit changes it. That is the
#      point: the engine is the participant's, the harness is not.
#
# THE ROSTER BELOW IS A HAND-COPY of benchd's HARNESS_HASH_ROOTS
# (crates/bench-core/src/harness_hash.rs). benchd owns the real one; this is a
# transcription, exactly as benchd's own freeze tests transcribe rosters. If the
# two drift, this suite is testing a hash nobody computes -- so a roster change
# in benchd is a change HERE in the same breath.
#
# Hermetic: no weights, no GPU, no CUDA, no network, no box. The build check is
# the CPU syntax pass, which needs only cc.
#
# Usage: tools/test-participant-engine-edit.sh [-v]
set -uo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null && pwd -P)"
VERBOSE=0
[[ "${1:-}" == "-v" ]] && VERBOSE=1

EXPECTED_MIN_ASSERTIONS=9
PASSED=0; FAILED=0; FAILURES=()
pass() { PASSED=$((PASSED+1)); [[ "${VERBOSE}" == "1" ]] && echo "ok    $1"; return 0; }
fail() { FAILED=$((FAILED+1)); FAILURES+=("$1"); echo "FAIL  $1" >&2; }

WORK="$(mktemp -d "${TMPDIR:-/tmp}/participant-edit.XXXXXX")"
trap 'rm -rf "${WORK}"' EXIT

# benchd's roster, transcribed. `ds4` is deliberately NOT here.
harness_hash() {
  python3 - "$1" <<'PY'
import hashlib, os, sys
root = sys.argv[1]
ROOTS = ["Package.swift", "Sources", "Tests", "benchmark.json", "benchmark.sh",
         "setup.sh", "tools", "README.md", "TASK.md", "harness", "vllm"]
paths = []
for r in ROOTS:
    p = os.path.join(root, r)
    if not os.path.exists(p):
        continue
    if os.path.isdir(p):
        for dirpath, dirnames, filenames in os.walk(p):
            # hidden entries are skipped, subtree and all; symlinks are excluded
            dirnames[:] = [d for d in dirnames
                           if not d.startswith('.')
                           and not os.path.islink(os.path.join(dirpath, d))]
            for fn in filenames:
                if fn.startswith('.'):
                    continue
                fp = os.path.join(dirpath, fn)
                if os.path.islink(fp) or not os.path.isfile(fp):
                    continue
                paths.append(fp)
    else:
        paths.append(p)
h = hashlib.sha256()
for p in sorted(paths):
    try:
        with open(p, 'rb') as fh:
            data = fh.read()
    except OSError:
        continue
    h.update(p.encode()); h.update(data)
print(h.hexdigest())
PY
}

# ---------------------------------------------------------------------------
# A sandbox repository: the real manifest, a miniature engine.
# ---------------------------------------------------------------------------
SB="${WORK}/repo"
mkdir -p "${SB}/ds4" "${SB}/harness" "${SB}/tools" "${SB}/fixtures"
# The REAL manifest, so editablePaths is the shipping list and not a fixture.
cp "${REPO_ROOT}/benchmark.json" "${SB}/benchmark.json"
printf 'int engine(void) { return 1; }\n' > "${SB}/ds4/ds4.c"
printf '{"fork":{"sha":"1111111111111111111111111111111111111111"}}\n' > "${SB}/ds4/VENDOR.json"
printf 'harness\n' > "${SB}/harness/adapter.rs"
printf 'tool\n' > "${SB}/tools/thing.sh"
printf '{}\n' > "${SB}/fixtures/contract.json"
git -C "${SB}" init --quiet
git -C "${SB}" add -A
git -C "${SB}" -c user.email=t@t -c user.name=t commit --quiet -m base

# ---------------------------------------------------------------------------
# 1. The engine is in the surface: the real manifest lists ds4.
# ---------------------------------------------------------------------------
if grep -q '"ds4"' "${SB}/benchmark.json"; then
  pass "benchmark.json editablePaths carries ds4"
else
  fail "benchmark.json editablePaths carries ds4"
fi

# ---------------------------------------------------------------------------
# 2. The harness hash: an engine edit does not move it, a harness edit does.
# ---------------------------------------------------------------------------
H0="$(harness_hash "${SB}")"
if [[ "${H0}" =~ ^[0-9a-f]{64}$ ]]; then
  pass "the harness hash is a sha256 over the roster"
else
  fail "the harness hash is a sha256 over the roster (got '${H0}')"
fi

printf '/* participant tuning */\n' >> "${SB}/ds4/ds4.c"
H_DS4="$(harness_hash "${SB}")"
if [[ "${H_DS4}" == "${H0}" ]]; then
  pass "a ds4/ ENGINE edit leaves the harness hash unchanged"
else
  fail "a ds4/ ENGINE edit leaves the harness hash unchanged (${H0:0:12} -> ${H_DS4:0:12})"
fi
git -C "${SB}" checkout --quiet -- ds4/ds4.c

printf 'participant harness change\n' >> "${SB}/harness/adapter.rs"
H_HARNESS="$(harness_hash "${SB}")"
if [[ "${H_HARNESS}" != "${H0}" ]]; then
  pass "a harness/ edit CHANGES the harness hash"
else
  fail "a harness/ edit CHANGES the harness hash (both ${H0:0:12})"
fi
git -C "${SB}" checkout --quiet -- harness/adapter.rs

# NON-VACUITY: the roster must really be hashing content, not returning a
# constant. `tools` is a root too, so an edit there must move it as well.
printf 'x\n' >> "${SB}/tools/thing.sh"
H_TOOLS="$(harness_hash "${SB}")"
if [[ "${H_TOOLS}" != "${H0}" ]]; then
  pass "a tools/ edit CHANGES the harness hash (the roster is live, not a constant)"
else
  fail "a tools/ edit CHANGES the harness hash (the roster is live, not a constant)"
fi
git -C "${SB}" checkout --quiet -- tools/thing.sh

if [[ "$(harness_hash "${SB}")" == "${H0}" ]]; then
  pass "reverting every edit returns the harness hash to its original value"
else
  fail "reverting every edit returns the harness hash to its original value"
fi

# The transcription must match benchd's roster ORDER AND MEMBERSHIP if a benchd
# checkout is reachable. Absent one, the hand-copy stands on its comment.
BENCHD_ROSTER_SRC="${BENCHD_SRC_DIR:-}/crates/bench-core/src/harness_hash.rs"
if [[ -n "${BENCHD_SRC_DIR:-}" && -f "${BENCHD_ROSTER_SRC}" ]]; then
  want="$(sed -n '/pub const HARNESS_HASH_ROOTS/,/\];/p' "${BENCHD_ROSTER_SRC}" \
    | grep -oE '"[^"]+"' | tr -d '"' | tr '\n' ' ')"
  have="Package.swift Sources Tests benchmark.json benchmark.sh setup.sh tools README.md TASK.md harness vllm "
  if [[ "${want}" == "${have}" ]]; then
    pass "the transcribed roster matches benchd's HARNESS_HASH_ROOTS"
  else
    fail "the transcribed roster matches benchd's HARNESS_HASH_ROOTS (benchd: ${want})"
  fi
  if [[ "${want}" != *"ds4"* ]]; then
    pass "benchd's roster does NOT include ds4 (an engine edit cannot move the harness hash)"
  else
    fail "benchd's roster now includes ds4; this suite's premise has changed"
  fi
else
  echo "test-participant-engine-edit: NOTE -- BENCHD_SRC_DIR unset, the roster transcription was not cross-checked" >&2
fi

# ---------------------------------------------------------------------------
# 3 + 4. In THIS repository: an engine edit misses the cache and still builds.
# ---------------------------------------------------------------------------
if [[ -f "${REPO_ROOT}/ds4/ds4_qwen4exp_mtp.c" && -f "${REPO_ROOT}/ds4/VENDOR.json" ]]; then
  probe="${REPO_ROOT}/ds4/ds4_qwen4exp_mtp.c"
  cp "${probe}" "${WORK}/probe.orig"
  restore() { cp "${WORK}/probe.orig" "${probe}"; }

  key_before="$("${REPO_ROOT}/tools/ds4/build-cache.sh" key 2>/dev/null)"
  printf '\n/* participant tuning probe */\n' >> "${probe}"
  key_after="$("${REPO_ROOT}/tools/ds4/build-cache.sh" key 2>/dev/null)"
  if [[ -n "${key_before}" && "${key_before}" != "${key_after}" ]]; then
    pass "an engine edit CHANGES the build-cache key, so it is rebuilt and not served from cache"
  else
    fail "an engine edit CHANGES the build-cache key (${key_before:0:12} -> ${key_after:0:12})"
  fi

  if "${REPO_ROOT}/tools/ds4/build.sh" --cpu-check >"${WORK}/build.out" 2>&1; then
    pass "the edited engine tree still passes tools/ds4/build.sh --cpu-check"
  else
    fail "the edited engine tree still passes tools/ds4/build.sh --cpu-check ($(tail -2 "${WORK}/build.out"))"
  fi

  restore
  if [[ "$("${REPO_ROOT}/tools/ds4/build-cache.sh" key 2>/dev/null)" == "${key_before}" ]]; then
    pass "reverting the engine edit returns the cache key to its original value"
  else
    fail "reverting the engine edit returns the cache key to its original value"
  fi
else
  fail "this repository carries no vendored ds4 tree to edit (ds4/VENDOR.json and ds4/*.c expected)"
fi

echo "participant-engine-edit: ${PASSED} passed, ${FAILED} failed"
if [[ "${FAILED}" -ne 0 ]]; then
  printf '  - %s\n' "${FAILURES[@]}" >&2
  exit 1
fi
if [[ "${PASSED}" -lt "${EXPECTED_MIN_ASSERTIONS}" ]]; then
  echo "participant-engine-edit: ran only ${PASSED} assertions, expected at least ${EXPECTED_MIN_ASSERTIONS} -- the suite shrank" >&2
  exit 1
fi
exit 0
