#!/usr/bin/env bash
#
# test-stage-baseline-workspace.sh -- the plan the reference-tree stager prints.
#
# tools/stage-baseline-workspace.sh puts the REFERENCE tree on a ranked box: a
# clone from a LOCAL bundle or mirror, detached at the fixture's
# baseline_reference_commit, then built. The clone and the build need a source
# repository and a CUDA toolchain, so this suite drives the --dry-run plan and
# the refusals, which is the whole surface that can be checked off the box.
#
# Hermetic: no clone, no build, no toolchain, no network.
#
# Cases:
#   1. --dry-run prints the clone, the detached checkout at the fixture's pinned
#      commit, the HEAD re-read and both build steps -- and creates nothing.
#   2. the clone is --no-local and --no-checkout: the staged tree is a real
#      object copy pinned to a commit, never a hardlink farm on a branch tip.
#   3. a missing --source refuses by name.
#   4. a --source that is not on this box refuses by name.
#   5. an existing destination refuses by name: a staged reference tree is never
#      reset, moved or re-fetched in place.
#   6. a fixture with no baseline_reference_commit refuses by name.
#   7. an unknown argument exits 2.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGER="${REPO_ROOT}/tools/stage-baseline-workspace.sh"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

failures=0
fail() { echo "FAIL: $*" >&2; failures=$((failures + 1)); }
ok()   { echo "ok: $*"; }

command -v jq >/dev/null 2>&1 || { echo "test-stage-baseline-workspace.sh: jq is required" >&2; exit 1; }

FIXTURE="${REPO_ROOT}/fixtures/qwen3_8_125b_a6b_track.json"
REF_COMMIT="$(jq -r '.baseline_reference_commit' "${FIXTURE}")"

# A stand-in for the organizer-staged local source. It is never read: --dry-run
# clones nothing, and every refusal below fires before the clone.
SOURCE="${WORK}/reference.bundle"
: > "${SOURCE}"
DEST="${WORK}/reference"

# --- case 1 + 2: the plan ---------------------------------------------------
PLAN="${WORK}/plan.out"
"${STAGER}" "${DEST}" --source "${SOURCE}" --dry-run > "${PLAN}" 2>&1
[ $? -eq 0 ] || { fail "case 1: --dry-run exited non-zero"; sed 's/^/    /' "${PLAN}" >&2; }

for needle in "git clone --no-local --no-checkout '${SOURCE}' '${DEST}'" \
              "git -C '${DEST}' checkout --detach ${REF_COMMIT}" \
              "git -C '${DEST}' rev-parse HEAD" \
              "${DEST}/tools/ds4/build.sh" \
              "${DEST}/tools/stage-cuda-engine.sh"; do
  grep -qF -- "${needle}" "${PLAN}" || fail "case 1: the plan does not carry '${needle}'"
done
grep -qF -- "reference commit  ${REF_COMMIT}" "${PLAN}" \
  || fail "case 1: the plan does not name the fixture's pinned reference commit"
ok "case 1: the plan clones, detaches at ${REF_COMMIT:0:12}, re-reads HEAD and builds"

grep -qF -- "--no-local" "${PLAN}" \
  && grep -qF -- "--no-checkout" "${PLAN}" \
  && grep -qF -- "checkout --detach ${REF_COMMIT}" "${PLAN}" \
  && ok "case 2: the clone is a real object copy pinned to a commit, not a branch tip" \
  || fail "case 2: the clone is not --no-local --no-checkout at the pinned commit"

[ -e "${DEST}" ] && fail "case 1: --dry-run created the destination"
ok "case 1: --dry-run created nothing"

# --- case 3: no source ------------------------------------------------------
env -u MLXFAST_BASELINE_SOURCE "${STAGER}" "${DEST}" --dry-run > "${WORK}/nosrc.out" 2>&1
[ $? -eq 2 ] || fail "case 3: a missing --source must refuse with exit 2"
grep -q 'REFUSE missing-source' "${WORK}/nosrc.out" \
  && ok "case 3: a missing source refuses by name" \
  || fail "case 3: the refusal did not name missing-source"
grep -q 'holds no credential' "${WORK}/nosrc.out" \
  && ok "case 3: the refusal says the stager holds no credential" \
  || fail "case 3: the refusal does not state the no-credential rule"

# --- case 4: a source that is not on this box -------------------------------
"${STAGER}" "${DEST}" --source "${WORK}/absent.bundle" --dry-run > "${WORK}/badsrc.out" 2>&1
[ $? -eq 2 ] || fail "case 4: an absent --source must refuse with exit 2"
grep -q 'REFUSE missing-source' "${WORK}/badsrc.out" \
  && ok "case 4: a source that is not staged refuses by name" \
  || fail "case 4: the refusal did not name missing-source"

# --- case 5: the destination already exists ---------------------------------
mkdir -p "${DEST}"
"${STAGER}" "${DEST}" --source "${SOURCE}" > "${WORK}/exists.out" 2>&1
[ $? -eq 2 ] || fail "case 5: an existing destination must refuse with exit 2"
grep -q 'REFUSE destination-exists' "${WORK}/exists.out" \
  && ok "case 5: an existing reference tree is never reset in place" \
  || fail "case 5: the refusal did not name destination-exists"
rmdir "${DEST}"

# --- case 6: a fixture with no reference commit -----------------------------
NOREF="${WORK}/no-ref-fixture.json"
jq 'del(.baseline_reference_commit)' "${FIXTURE}" > "${NOREF}"
"${STAGER}" "${DEST}" --source "${SOURCE}" --fixture "${NOREF}" --dry-run > "${WORK}/noref.out" 2>&1
[ $? -eq 2 ] || fail "case 6: a fixture with no baseline_reference_commit must refuse with exit 2"
grep -q 'REFUSE missing-reference-commit' "${WORK}/noref.out" \
  && ok "case 6: an unpinned reference commit refuses by name" \
  || fail "case 6: the refusal did not name missing-reference-commit"

# --- case 7: an unknown argument --------------------------------------------
"${STAGER}" "${DEST}" --nope > "${WORK}/arg.out" 2>&1
[ $? -eq 2 ] || fail "case 7: an unknown argument must exit 2"
grep -q 'unknown argument --nope' "${WORK}/arg.out" \
  && ok "case 7: an unknown argument refuses by name" \
  || fail "case 7: the refusal did not name the argument"

if [ "${failures}" -eq 0 ]; then
  echo "PASS: test-stage-baseline-workspace.sh"
  exit 0
fi
echo "FAIL: ${failures} failure(s)" >&2
exit 1
