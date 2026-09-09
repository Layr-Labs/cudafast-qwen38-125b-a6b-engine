#!/usr/bin/env bash
# Unit test for .github/scripts/overlay-editable-paths.sh (issue #36).
#
# WHAT THIS PROVES, and it is the part worth having: a REJECTED submission
# leaves the trusted checkout exactly as it found it. The overlay deletes the
# trusted copy of an editable path before it writes the submission's copy over
# it, so every reason to refuse a path has to be found BEFORE that delete. The
# pre-copy check used to look for symlinks only, so a FIFO (or a socket, or a
# device node) at an editable root or inside one got past it, the trusted copy
# was already gone, and the copy then blocked or refused -- the post-copy
# validate_overlay_tree never ran. The refusal was right and it cost the
# verifier its content.
#
# It also pins path_identity's OUTPUT SHAPE, which is the other half of the same
# issue. The function used to try BSD `stat -f` first and GNU `stat -c` second.
# On GNU coreutils `-f` means "file system status" and takes no format, so the
# format string was read as a FILE operand: it errored on stderr, the path
# itself SUCCEEDED and printed a multi-line filesystem-status block on stdout,
# the status was 1, and the fallback then appended the real device:inode to that
# block. The identity string was a paragraph naming the path, so two names for
# the SAME inode compared unequal and the identity arm of the forbidden-path
# guard was dead on Linux -- which is what the hosted job and the ranked box
# both run. The hardlink case below is that arm, exercised through the real
# script: with the old order it does not refuse on GNU, and it does refuse here
# under BSD, so the case is a tripwire on the platform that matters.
#
# Hermetic: bash, jq, find and mkfifo. No git (TRUSTED_MAIN_SHA is unset, which
# skips the stale-file report and every git read with it), no credential, no
# network, no toolchain, no box.
#
# Usage: tools/test-overlay-editable-paths.sh
# Exit:  0 all cases pass, 1 a case failed (printed with a FAIL prefix)
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null && pwd -P)"
OVERLAY="${REPO_ROOT}/.github/scripts/overlay-editable-paths.sh"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/overlay-editable-paths.XXXXXX")"
trap 'rm -rf "${WORK}"' EXIT

# The stale-editable-file report is the script's only git reader, and it returns
# before any git call when this is empty. Dropped from the environment so the
# suite cannot inherit a value from a caller and start needing a repository.
unset TRUSTED_MAIN_SHA

fails=0

pass()    { printf 'PASS %s\n' "$1"; }
failure() { printf 'FAIL %s: %s\n' "$1" "$2"; fails=$((fails + 1)); }

# A fixture is a trusted checkout (contract + content) beside a submission
# worktree, both git-free. The contract is written here rather than copied from
# the repository so the cases stay readable and do not move when the real
# editablePaths list does.
new_fixture() {
  local root
  root="$(mktemp -d "${WORK}/fixture.XXXXXX")"
  mkdir -p "${root}/trusted/src" "${root}/sub/src"
  cat > "${root}/trusted/benchmark.json" <<'JSON'
{
  "editablePaths": ["src", "config.txt"],
  "optionalEditablePaths": ["optional.txt"]
}
JSON
  printf 'trusted kernel\n' > "${root}/trusted/src/kernel.txt"
  printf 'trusted config\n' > "${root}/trusted/config.txt"
  printf 'submitted kernel\n' > "${root}/sub/src/kernel.txt"
  printf 'submitted config\n' > "${root}/sub/config.txt"
  printf '%s' "${root}"
}

# EVERY overlay run IS BOUNDED, because the defect this suite is about ends in a
# `cp` that never returns: a FIFO reaching the copy blocks on open forever. A
# regression has to turn this suite RED in a minute, not sit on the job's whole
# timeout budget. `timeout` ships with GNU coreutils, so the hosted runner and
# the box always have it; a macOS checkout without it runs unbounded, which is
# fine interactively.
TIMEOUT_BIN=""
for candidate in timeout gtimeout; do
  if command -v "${candidate}" >/dev/null 2>&1; then
    TIMEOUT_BIN="${candidate}"
    break
  fi
done

# run_overlay ROOT -- runs the REAL script in the fixture's trusted checkout and
# leaves its combined output in ${WORK}/out. Returns the script's status.
run_overlay() {
  local root="$1" rc=0
  local -a cmd=(env "CONTRACT_PATH=benchmark.json" "SUBMISSION_WORKTREE=${root}/sub" "${OVERLAY}")
  if [[ -n "${TIMEOUT_BIN}" ]]; then
    cmd=("${TIMEOUT_BIN}" -k 5 60 "${cmd[@]}")
  fi
  ( cd "${root}/trusted" && "${cmd[@]}" ) > "${WORK}/out" 2>&1 || rc=$?
  if [[ "${rc}" -eq 124 || "${rc}" -eq 137 ]]; then
    echo "the overlay did not return within 60s; it blocked, which is what a shape it should have refused does to the copy" >> "${WORK}/out"
  fi
  return "${rc}"
}

# assert_refusal LABEL ROOT NEEDLE
assert_refusal() {
  local label="$1" root="$2" needle="$3" rc=0
  run_overlay "${root}" || rc=$?
  if [[ "${rc}" -ne 1 ]]; then
    failure "${label}" "expected exit 1, got ${rc}: $(tr '\n' ' ' < "${WORK}/out")"
    return
  fi
  if ! grep -qF -- "${needle}" "${WORK}/out"; then
    failure "${label}" "diagnostic did not mention '${needle}': $(tr '\n' ' ' < "${WORK}/out")"
    return
  fi
  pass "${label}"
}

# assert_accepted LABEL ROOT
assert_accepted() {
  local label="$1" root="$2" rc=0
  run_overlay "${root}" || rc=$?
  if [[ "${rc}" -ne 0 ]]; then
    failure "${label}" "expected exit 0, got ${rc}: $(tr '\n' ' ' < "${WORK}/out")"
    return
  fi
  pass "${label}"
}

# assert_content LABEL PATH WANTED
assert_content() {
  local label="$1" path="$2" want="$3" got="absent"
  if [[ -f "${path}" ]]; then
    got="$(cat "${path}")"
  fi
  if [[ "${got}" == "${want}" ]]; then
    pass "${label}"
  else
    failure "${label}" "got '${got}', wanted '${want}'"
  fi
}

# --- 1. a FIFO AT an editable root ------------------------------------------
# The trusted copy of that root must survive the refusal.
root="$(new_fixture)"
rm -f "${root}/sub/config.txt"
mkfifo "${root}/sub/config.txt"
assert_refusal "fifo at an editable root is refused" "${root}" \
  "must contain only regular files and directories"
assert_content "fifo at an editable root leaves the trusted copy intact" \
  "${root}/trusted/config.txt" "trusted config"

# --- 2. a FIFO NESTED inside an editable directory --------------------------
# `src` is the FIRST entry in the fixture contract, so a correct refusal happens
# before the overlay has touched anything at all.
root="$(new_fixture)"
mkfifo "${root}/sub/src/pipe"
assert_refusal "fifo nested in an editable directory is refused" "${root}" \
  "must contain only regular files and directories"
assert_content "fifo nested in an editable directory leaves the trusted tree intact" \
  "${root}/trusted/src/kernel.txt" "trusted kernel"
assert_content "a refusal on the first entry does not touch a later entry" \
  "${root}/trusted/config.txt" "trusted config"

# --- 3. the control: an ordinary submission still overlays ------------------
# Without this the two refusals above would pass just as well against a script
# that refused everything.
root="$(new_fixture)"
assert_accepted "an ordinary submission overlays" "${root}"
assert_content "the submitted directory replaced the trusted one" \
  "${root}/trusted/src/kernel.txt" "submitted kernel"
assert_content "the submitted file replaced the trusted one" \
  "${root}/trusted/config.txt" "submitted config"

# --- 4. path_identity returns a device:inode, not a paragraph ---------------
# The function definition is lifted out of the SHIPPED script and evaluated
# here, so this reads the real `stat` line on whatever platform runs it: GNU
# coreutils on the hosted runner and on the ranked box, BSD stat on a
# contributor's macOS checkout. A multi-line filesystem-status block -- what the
# old BSD-first order produced under GNU -- fails the match.
identity_source="$(sed -n '/^path_identity() {$/,/^}$/p' "${OVERLAY}")"
if [[ -z "${identity_source}" ]]; then
  failure "path_identity is extractable from the shipped script" \
    "no path_identity definition found in ${OVERLAY}"
else
  pass "path_identity is extractable from the shipped script"
  eval "${identity_source}"

  probe="${WORK}/identity-probe"
  printf 'probe\n' > "${probe}"
  identity="$(path_identity "${probe}")"
  if [[ "${identity}" =~ ^[0-9]+:[0-9]+$ ]]; then
    pass "path_identity of a file is exactly device:inode (${identity})"
  else
    failure "path_identity of a file is exactly device:inode" \
      "got '$(printf '%s' "${identity}" | tr '\n' ' ')'"
  fi

  identity="$(path_identity "${WORK}")"
  if [[ "${identity}" =~ ^[0-9]+:[0-9]+$ ]]; then
    pass "path_identity of a directory is exactly device:inode (${identity})"
  else
    failure "path_identity of a directory is exactly device:inode" \
      "got '$(printf '%s' "${identity}" | tr '\n' ' ')'"
  fi

  rc=0
  identity="$(path_identity "${WORK}/does-not-exist")" || rc=$?
  if [[ "${rc}" -ne 0 && -z "${identity}" ]]; then
    pass "path_identity of an absent path is empty and fails"
  else
    failure "path_identity of an absent path is empty and fails" \
      "rc=${rc} output='${identity}'"
  fi
fi

# --- 5. the identity arm of the forbidden-path guard, end to end ------------
# A second NAME for the protected file. ASCII case folding does not normalise
# `alias.pin` to `benchd.pin`, so the folded-string arm cannot see it; only the
# device:inode comparison can, and that comparison is only meaningful when
# path_identity returns an identity. This is the case that goes red on GNU if
# the `stat` order is ever put back the way it was.
root="$(new_fixture)"
cat > "${root}/trusted/benchmark.json" <<'JSON'
{
  "editablePaths": ["alias.pin", "src"],
  "optionalEditablePaths": []
}
JSON
printf 'the pinned scorer\n' > "${root}/trusted/benchd.pin"
ln "${root}/trusted/benchd.pin" "${root}/trusted/alias.pin"
printf 'attacker pin\n' > "${root}/sub/alias.pin"
assert_refusal "a second hardlinked name for the pinned scorer is refused" "${root}" \
  "covers the measurement-harness surface"
assert_content "the refused alias left the pinned scorer intact" \
  "${root}/trusted/benchd.pin" "the pinned scorer"

if [[ "${fails}" -eq 0 ]]; then
  echo "OK: all overlay-editable-paths cases passed"
  exit 0
fi
echo "FAILED: ${fails} case(s)"
exit 1
