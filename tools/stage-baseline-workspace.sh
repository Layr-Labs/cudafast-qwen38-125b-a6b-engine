#!/usr/bin/env bash
# stage-baseline-workspace.sh -- put the REFERENCE tree on this box, built.
#
# WHAT THE REFERENCE TREE IS. The ranked path is PAIRED: every scored run
# measures a SERIAL CONTROL leg beside the candidate leg, on the same box, in the
# same job, and divides one by the other. The control leg runs on the tree this
# script stages -- the fixture's `baseline_reference_commit`, the promoted
# baseline every box's control leg runs, so one board compares one control.
#
# NO CREDENTIAL. A ranked box holds no key (bundles-no-keys), so this script
# never authenticates and never reaches a host that would ask it to. It clones
# from a LOCAL source that the organizer has already staged on the box:
#
#   * a git BUNDLE file (git bundle create ... --all), or
#   * a local MIRROR directory (a bare or non-bare clone on the box).
#
# Either is a plain path on the filesystem. `git clone` over a local path copies
# objects directly and opens no network connection.
#
# WHAT IT LEAVES BEHIND. A directory at <dir> whose HEAD is the pinned reference
# commit, with tools/ds4/build.sh run inside it, so the tree carries its own
# .build/ds4/{libds4qwen.so,ds4-resident} and its staged adapter at
# .build/release/mlxfast-runtime-worker. tools/ranked-box-preflight.sh section 8
# checks exactly those, so a half-staged tree is refused before the GPU window
# rather than at the moment the control leg tries to boot.
#
# IT IS NOT AN UPDATER. An existing <dir> is REFUSED rather than reset, moved or
# re-fetched: re-pointing a reference tree in place is how a board silently
# changes its control. Remove it deliberately, then stage again.
#
# Usage:  tools/stage-baseline-workspace.sh <dir> [options]
#           <dir>   where the reference tree is created (must not exist)
#
# Exit codes
#   0  the tree is staged, at the pinned commit, and built
#   2  refusal before anything is written (argument, tool, source, fixture)
#   4  the clone, the checkout or the build failed
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null && pwd -P)"

FIXTURE="${REPO_DIR}/fixtures/qwen3_8_125b_a6b_track.json"
SOURCE="${MLXFAST_BASELINE_SOURCE:-}"
DRY_RUN=0
DEST=""

usage() {
  cat <<'EOF'
usage: stage-baseline-workspace.sh <dir> [options]

  <dir>            where the reference tree is created. It must not exist:
                   an existing tree is refused, never reset or re-fetched

  --source PATH    the LOCAL git bundle file or mirror directory to clone from
                   (default: $MLXFAST_BASELINE_SOURCE). No credential is used
                   and no network host is contacted
  --fixture FILE   track fixture (default: this repo's)
  --dry-run        print the commands and stop. Clones nothing, builds nothing
                   and writes nothing
  -h, --help       this text

ENV:
  MLXFAST_BASELINE_SOURCE  the default --source
EOF
}

refuse() { # refuse NAME MESSAGE...
  local name="$1"; shift
  printf 'stage-baseline-workspace: REFUSE %s: %s\n' "${name}" "$*" >&2
  exit 2
}

while [ $# -gt 0 ]; do
  case "$1" in
    --source)  SOURCE="$2"; shift 2 ;;
    --fixture) FIXTURE="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    -*) printf 'stage-baseline-workspace: unknown argument %s\n' "$1" >&2; usage >&2; exit 2 ;;
    *)
      if [ -z "${DEST}" ]; then DEST="$1"
      else printf 'stage-baseline-workspace: unexpected argument %s\n' "$1" >&2; usage >&2; exit 2
      fi
      shift ;;
  esac
done

[ -n "${DEST}" ] || refuse missing-argument "<dir> is required: where the reference tree is created"
command -v git >/dev/null 2>&1 || refuse missing-tool "git is required to clone and verify the reference tree"
command -v jq >/dev/null 2>&1 || refuse missing-tool "jq is required to read the pinned reference commit"
[ -r "${FIXTURE}" ] || refuse missing-fixture "cannot read the track fixture ${FIXTURE}"

REF_COMMIT="$(jq -r '.baseline_reference_commit // empty' "${FIXTURE}")"
case "${REF_COMMIT}" in
  [0-9a-f]*) [ "${#REF_COMMIT}" -eq 40 ] || REF_COMMIT="" ;;
  *) REF_COMMIT="" ;;
esac
[ -n "${REF_COMMIT}" ] || refuse missing-reference-commit \
  "${FIXTURE} carries no 40-hex baseline_reference_commit; there is no reference tree to stage"

[ -n "${SOURCE}" ] || refuse missing-source \
  "--source is required (or set MLXFAST_BASELINE_SOURCE): a local git bundle file or mirror directory on this box. This script holds no credential and contacts no host"
[ -e "${SOURCE}" ] || refuse missing-source \
  "--source ${SOURCE} does not exist on this box; the organizer stages the bundle or mirror out of band"
if [ -d "${SOURCE}" ]; then
  SOURCE_KIND="mirror directory"
elif [ -f "${SOURCE}" ]; then
  SOURCE_KIND="git bundle"
else
  refuse bad-source "--source ${SOURCE} is neither a file (bundle) nor a directory (mirror)"
fi

# AN EXISTING TREE IS REFUSED. Re-pointing a reference tree in place is how a
# board silently changes its control leg.
if [ "${DRY_RUN}" -eq 0 ] && [ -e "${DEST}" ]; then
  refuse destination-exists \
    "${DEST} already exists. This script never resets, moves or re-fetches a staged reference tree, because that changes what every scored run on this box is divided by. Remove it deliberately, then stage again"
fi

BUILD="${DEST}/tools/ds4/build.sh"
STAGE="${DEST}/tools/stage-cuda-engine.sh"

if [ "${DRY_RUN}" -eq 1 ]; then
  cat <<EOF
stage-baseline-workspace: DRY RUN -- nothing is cloned, built or written

reference commit  ${REF_COMMIT}
source            ${SOURCE}  (${SOURCE_KIND})
destination       ${DEST}

  git clone --no-local --no-checkout '${SOURCE}' '${DEST}'
  git -C '${DEST}' checkout --detach ${REF_COMMIT}
  git -C '${DEST}' rev-parse HEAD          # must print ${REF_COMMIT}
  ${BUILD}
  ${STAGE}
EOF
  exit 0
fi

# --no-local keeps the clone a real object copy rather than a hardlink farm into
# the source, so a later change to the source cannot reach into the staged tree.
# --no-checkout because the checkout below is the pinned commit, not a branch
# tip: a branch that has moved must not decide what the control leg runs.
git clone --no-local --no-checkout "${SOURCE}" "${DEST}" \
  || { printf 'stage-baseline-workspace: the clone from %s failed; nothing is staged\n' "${SOURCE}" >&2; exit 4; }
git -C "${DEST}" checkout --detach "${REF_COMMIT}" \
  || { printf 'stage-baseline-workspace: %s does not carry commit %s; stage a source that does\n' "${SOURCE}" "${REF_COMMIT}" >&2; exit 4; }

HAVE="$(git -C "${DEST}" rev-parse HEAD)"
[ "${HAVE}" = "${REF_COMMIT}" ] \
  || { printf 'stage-baseline-workspace: HEAD is %s, expected %s\n' "${HAVE}" "${REF_COMMIT}" >&2; exit 4; }
printf 'stage-baseline-workspace: %s is at %s\n' "${DEST}" "${REF_COMMIT}" >&2

[ -x "${BUILD}" ] \
  || { printf 'stage-baseline-workspace: the staged tree has no executable tools/ds4/build.sh\n' >&2; exit 4; }
"${BUILD}" || { printf 'stage-baseline-workspace: the ds4 build failed in %s\n' "${DEST}" >&2; exit 4; }
[ -x "${STAGE}" ] \
  || { printf 'stage-baseline-workspace: the staged tree has no executable tools/stage-cuda-engine.sh\n' >&2; exit 4; }
"${STAGE}" || { printf 'stage-baseline-workspace: staging the adapter failed in %s\n' "${DEST}" >&2; exit 4; }

for rel in .build/ds4/libds4qwen.so .build/ds4/ds4-resident .build/release/mlxfast-runtime-worker; do
  [ -f "${DEST}/${rel}" ] \
    || { printf 'stage-baseline-workspace: the build left no %s; the reference tree cannot serve the control leg\n' "${rel}" >&2; exit 4; }
done

# HEAD IS RE-READ AFTER THE BUILD. The build writes into .build/ only, and the
# preflight refuses a workspace with uncommitted changes, so a build that dirtied
# a tracked file must be caught here rather than on the next ranked dispatch.
HAVE="$(git -C "${DEST}" rev-parse HEAD)"
[ "${HAVE}" = "${REF_COMMIT}" ] \
  || { printf 'stage-baseline-workspace: HEAD moved to %s during the build\n' "${HAVE}" >&2; exit 4; }
git -C "${DEST}" diff --quiet HEAD \
  || { printf 'stage-baseline-workspace: the build left tracked changes in %s; the preflight refuses a dirty reference tree\n' "${DEST}" >&2; exit 4; }

printf 'stage-baseline-workspace: staged and built %s at %s\n' "${DEST}" "${REF_COMMIT}" >&2
printf '  export MLXFAST_BASELINE_WORKSPACE=%s\n' "${DEST}" >&2
