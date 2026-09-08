#!/usr/bin/env bash
# Overlay only benchmark.json editablePaths from an untrusted submission
# checkout onto the trusted checkout (the current working directory).
#
# Re-implementation of Layr-Labs/qwen-3.8-mtp-challenge@bfab0de
# .github/scripts/overlay-editable-paths.sh:1-186. Rule-for-rule derivation is
# in docs/submission-restriction-spec.md. Two additions this repository needs
# are marked ADDITION below; everything else is the original's semantics.
set -euo pipefail

: "${SUBMISSION_WORKTREE:?SUBMISSION_WORKTREE is required}"

CONTRACT_PATH="${CONTRACT_PATH:-benchmark.json}"

# ADDITION (pinned measurement harness). benchd is what MEASURES a submission.
# It used to be a SHA-pinned source submodule at `benchd`, pointed at by
# `.gitmodules`; it is now a channel-resolved PREBUILT binary -- the retired `benchd.pin` spelling named
# {branch, commit, sha256, bytes} and tools/fetch-benchd.sh resolves it into
# `benchd-bin/`. An editable entry reaching the pin or the resolved binary would
# let a submission repoint or replace its own scorer, so the overlay refuses to
# act on such an entry by construction rather than trusting the manifest linter
# to have caught it. The two submodule spellings are KEPT so a reintroduced
# gitlink is covered on arrival. The original had nothing here to protect.
#
# Kept in lockstep with enforce-modifiable-surface.sh FORBIDDEN_SURFACE_PATHS
# and lint-benchmark-manifest.py FORBIDDEN_EDITABLE: the three layers that read
# this editable surface must not disagree about which spellings reach the scorer.
FORBIDDEN_OVERLAY_PATHS=("benchd" ".gitmodules" "benchd.pin" "benchd-bin")

# CASE FOLDING. The ranked box is macOS and APFS is case-INSENSITIVE by
# default, so `BENCHD.PIN` names the same file as `benchd.pin`. A byte-comparison
# guard passes the spelling, and the `rm -rf "${target_path}"` below then
# deletes the real pin before the overlay writes the submission's copy over it. Both halves of the guard below exist because neither is
# sufficient alone: the folded-string test catches an entry whose target does
# not exist yet, and the filesystem-identity test catches a spelling that ASCII
# folding does not normalise (Unicode case folding, HFS+ decomposition) but the
# filesystem still resolves to the protected path.
fold_case() { printf '%s' "$1" | tr '[:upper:]' '[:lower:]'; }

# Filesystem identity of an existing path, as device:inode. Empty (status 1)
# when the path does not exist.
path_identity() {
  local path="$1"
  [[ -e "${path}" ]] || return 1
  stat -f '%d:%i' "${path}" 2>/dev/null || stat -c '%d:%i' "${path}" 2>/dev/null
}

# True when `path` names, contains, or lives inside a forbidden path on THIS
# filesystem -- by case-folded spelling or by resolved identity.
reaches_forbidden_path() {
  local path="$1" forbidden folded_path folded_forbidden forbidden_identity prefix identity
  folded_path="$(fold_case "${path}")"
  for forbidden in "${FORBIDDEN_OVERLAY_PATHS[@]}"; do
    folded_forbidden="$(fold_case "${forbidden}")"
    if [[ "${folded_path}" == "${folded_forbidden}" \
       || "${folded_path}" == "${folded_forbidden}/"* \
       || "${folded_forbidden}" == "${folded_path}/"* ]]; then
      return 0
    fi
    forbidden_identity="$(path_identity "${forbidden}")" || continue
    # Walk the entry's own prefixes: `BENCHD/crates` is forbidden because
    # `BENCHD` resolves to the submodule, even though the full path does not
    # exist.
    prefix="${path}"
    while [[ -n "${prefix}" && "${prefix}" != "." && "${prefix}" != "/" ]]; do
      if identity="$(path_identity "${prefix}")" && [[ "${identity}" == "${forbidden_identity}" ]]; then
        return 0
      fi
      prefix="$(dirname -- "${prefix}")"
    done
  done
  return 1
}

if [[ ! -d "${SUBMISSION_WORKTREE}" ]]; then
  echo "::error::submission worktree not found at ${SUBMISSION_WORKTREE}" >&2
  exit 1
fi
if [[ ! -f "${CONTRACT_PATH}" ]]; then
  echo "::error::trusted benchmark contract missing at ${CONTRACT_PATH}" >&2
  exit 1
fi
if ! command -v jq >/dev/null 2>&1; then
  echo "::error::jq is required to read editablePaths from ${CONTRACT_PATH}" >&2
  exit 1
fi

validate_contract_path() {
  local path="$1"
  # A leading ':' would be git pathspec magic wherever this path is reused as
  # one, not a path. (The original overlay omitted this test; its sibling
  # run-submission-static-review.sh:139 carried it. Applied here so the two
  # validators cannot disagree about what a legal editable path is.)
  if [[ -z "${path}" || "${path}" == /* || "${path}" == :* || "${path}" == *\\* ]]; then
    echo "::error::invalid editable path '${path}' in ${CONTRACT_PATH}" >&2
    exit 1
  fi
  case "/${path}/" in
    *"/../"*|*"/./"*)
      echo "::error::invalid editable path '${path}' in ${CONTRACT_PATH}" >&2
      exit 1
      ;;
  esac
  if reaches_forbidden_path "${path}"; then
    echo "::error::editable path '${path}' in ${CONTRACT_PATH} covers the measurement-harness surface (benchd-bin, or the retired benchd.pin spelling) or a retired submodule spelling; refusing to overlay it" >&2
    exit 1
  fi
}

OPTIONAL_EDITABLE_PATHS=()
while IFS= read -r optional_path; do
  [[ -n "${optional_path}" ]] && OPTIONAL_EDITABLE_PATHS+=("${optional_path}")
done < <(jq -r '(.optionalEditablePaths // [])[]' "${CONTRACT_PATH}")

is_optional_editable_path() {
  local candidate="$1" optional
  for optional in ${OPTIONAL_EDITABLE_PATHS[@]+"${OPTIONAL_EDITABLE_PATHS[@]}"}; do
    [[ "${candidate}" == "${optional}" ]] && return 0
  done
  return 1
}

validate_overlay_tree() {
  local path="$1"
  if find "${path}" -type l -print -quit | grep -q .; then
    echo "::error file=${path}::overlaid editable paths must not contain symlinks" >&2
    exit 1
  fi
  if find "${path}" ! -type f ! -type d -print -quit | grep -q .; then
    echo "::error file=${path}::overlaid editable paths must contain only regular files and directories" >&2
    exit 1
  fi
  if find "${path}" -type f -links +1 -print -quit | grep -q .; then
    echo "::error file=${path}::overlaid editable paths must not contain hardlinked files" >&2
    exit 1
  fi
  if find "${path}" \( -perm -4000 -o -perm -2000 \) -print -quit | grep -q .; then
    echo "::error file=${path}::overlaid editable paths must not contain setuid or setgid files" >&2
    exit 1
  fi
}

### Stale-editable-file detection (report-only) ###############################
#
# The overlay takes the SUBMISSION's copy of every editablePath wholesale. That
# is correct for files the participant actually edited, but it also carries
# over their copy of editable files they never touched -- so an operator change
# to an editable file made AFTER the candidate branched is silently replaced by
# the pre-change content. Two consequences: this ranked run builds and measures
# the older file, not trusted main's; and on promotion the older content lands
# back on main, reverting the operator edit.
#
# Report-only on purpose. Failing here would reject a legitimate submission for
# something the participant did not do, and dropping the file from the overlay
# would change what the ranked run measures -- both are policy calls, not this
# script's. Skipped silently when TRUSTED_MAIN_SHA is unset or the merge base
# cannot be resolved.
TRUSTED_MAIN_SHA="${TRUSTED_MAIN_SHA:-}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null && pwd -P)"
HARDENED_GIT="${SCRIPT_DIR}/hardened-git.sh"

submission_git() {
  if [[ -x "${HARDENED_GIT}" ]]; then
    "${HARDENED_GIT}" -C "${SUBMISSION_WORKTREE}" "$@"
  else
    git -C "${SUBMISSION_WORKTREE}" "$@"
  fi
}

path_is_editable() {
  local candidate="$1" allowed
  while IFS= read -r allowed; do
    [[ -n "${allowed}" ]] || continue
    if [[ "${candidate}" == "${allowed}" || "${candidate}" == "${allowed}/"* ]]; then
      return 0
    fi
  done < <(jq -r '.editablePaths[]' "${CONTRACT_PATH}")
  return 1
}

report_stale_editable_overlays() {
  [[ -n "${TRUSTED_MAIN_SHA}" ]] || return 0
  local merge_base stale=0 path
  merge_base="$(submission_git merge-base HEAD "${TRUSTED_MAIN_SHA}" 2>/dev/null || true)"
  if [[ -z "${merge_base}" ]]; then
    echo "::warning::could not resolve a merge base between the submission and trusted main ${TRUSTED_MAIN_SHA}; skipping the stale-editable-file check" >&2
    return 0
  fi
  # Candidate already sits on the trusted tip: nothing can be stale.
  [[ "${merge_base}" != "${TRUSTED_MAIN_SHA}" ]] || return 0

  while IFS= read -r path; do
    [[ -n "${path}" ]] || continue
    path_is_editable "${path}" || continue
    # Untouched by the submission since it branched -> the overlay is about to
    # replace trusted main's newer content with the pre-branch content.
    if submission_git diff --quiet "${merge_base}" HEAD -- "${path}" 2>/dev/null; then
      echo "::warning file=${path}::${path} changed on trusted main after this submission branched, and the submission still carries the pre-change content; the overlay will use the OLDER file (and promoting this submission would revert the change on main)" >&2
      stale=$((stale + 1))
    fi
  done < <(submission_git diff --name-only "${merge_base}" "${TRUSTED_MAIN_SHA}" 2>/dev/null || true)

  if [[ "${stale}" -gt 0 ]]; then
    echo "::warning::${stale} editable file(s) are overlaid at their pre-branch content because trusted main moved after this submission branched; re-sync the submission with current main to pick the changes up" >&2
  fi
  echo "benchmark: stale-editable-file check complete (merge_base=${merge_base} stale=${stale})"
}

report_stale_editable_overlays

# The allowlist, the optional set and the exemptions all come from the TRUSTED
# contract in the working directory -- never from the submission worktree -- so
# a submission cannot widen its own surface, make its own missing files
# optional, or exempt itself from anything by shipping a benchmark.json.
#
# Q8 (issue #20): the loop used to read editablePaths in its own header via a
# process substitution. A trusted contract with `editablePaths: []`, no
# editablePaths key, or one that does not parse yielded zero iterations, the
# loop body never ran, and the success trailer below still printed -- so a
# pipeline measured the UNMODIFIED trusted tree and attributed the score to a
# submission that overlaid nothing. Read the allowlist up front and fail closed
# on a contract that does not parse or lists nothing to overlay.
if ! jq -e 'type == "object"' >/dev/null 2>&1 "${CONTRACT_PATH}"; then
  echo "::error::${CONTRACT_PATH} does not parse as a JSON object; refusing to report a successful overlay that moved nothing" >&2
  exit 1
fi
EDITABLE_PATHS=()
while IFS= read -r editable_path; do
  [[ -n "${editable_path}" ]] && EDITABLE_PATHS+=("${editable_path}")
done < <(jq -r '.editablePaths[]' "${CONTRACT_PATH}")
if (( ${#EDITABLE_PATHS[@]} == 0 )); then
  echo "::error::${CONTRACT_PATH} lists no editablePaths; refusing to report a successful overlay that moved nothing" >&2
  exit 1
fi

for editable_path in "${EDITABLE_PATHS[@]}"; do
  validate_contract_path "${editable_path}"

  source_path="${SUBMISSION_WORKTREE}/${editable_path}"
  target_path="${editable_path}"

  if [[ ! -e "${source_path}" ]]; then
    # OPTIONAL editable paths may legitimately be absent from a submission, and
    # absence means "keep the trusted checkout's copy" -- never "delete it".
    #
    # WHY THIS EXISTS. Submissions arrive as archives with REPLACE semantics
    # over editablePaths: whatever the archive does not carry is gone. That is
    # correct for source, where a missing file means a stale clone and the loud
    # refusal below is the right answer. It is wrong for the MTP head
    # declaration, which is a small opt-in file most archives will simply not
    # contain -- and the contract says an ABSENT mtp-head.manifest.json means
    # "use the organizer-pinned head". Failing the overlay would turn the
    # default case into a refusal.
    if is_optional_editable_path "${editable_path}"; then
      echo "benchmark: optional editable path ${editable_path} absent from the submission; keeping the trusted copy (the contract default)"
      continue
    fi
    echo "::error file=${editable_path}::submitted editable path is missing" >&2
    exit 1
  fi
  if find "${source_path}" -type l -print -quit | grep -q .; then
    echo "::error file=${editable_path}::submitted editable paths must not contain symlinks" >&2
    exit 1
  fi

  rm -rf "${target_path}"
  mkdir -p "$(dirname "${target_path}")"
  if [[ -d "${source_path}" ]]; then
    mkdir -p "${target_path}"
    (cd "${source_path}" && tar -cf - .) | (cd "${target_path}" && tar -xf -)
  else
    cp "${source_path}" "${target_path}"
  fi
  validate_overlay_tree "${target_path}"

  echo "benchmark: overlaid editable path ${editable_path}"
done

echo "benchmark: trusted harness retained; submitted editable paths overlaid"
