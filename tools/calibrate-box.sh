#!/usr/bin/env bash
# calibrate-box.sh -- capture THIS box's serial-control health band.
#
# WHAT A CALIBRATION IS ON THIS TRACK. The ranked path is PAIRED: every scored
# run measures a SERIAL CONTROL leg on the staged reference tree next to the
# candidate leg, and divides one by the other. The control leg is therefore
# already the denominator, and nothing here becomes one.
#
# WHAT THIS FILE IS FOR, then, is the question the ratio cannot answer: DID THE
# CONTROL LEG LAND WHERE THIS BOX LANDS? A box with a degraded link, a hot
# chassis or a half-applied driver still produces a ratio, and that ratio looks
# normal. So each box carries a band -- the mean and the spread of its own
# control leg -- and benchd refuses a run whose control leg falls outside it.
# The band is READ, never divided by.
#
# WHAT RUNS, IN ORDER
#
#   1. The two ranked environment variables and the reference tree are checked.
#      The workspace must be at the fixture's baseline_reference_commit and must
#      carry its build: a band captured on another tree describes another engine.
#   2. The GPU lock is taken and HELD for the whole calibration. serve-up.sh does
#      not take it, and a resident engine holds GPU memory for as long as this
#      driver runs.
#   3. `benchd calibrate-baseline` runs the control leg --passes times under the
#      full official methodology (cool gate per pass, same prompt, the fixture's
#      live golden), writes the file, and refuses when the per-axis CV exceeds
#      1 % -- the box was not stable, so no band describes it.
#
# THIS DRIVER BOOTS NO SERVE. benchd owns the residency, and it boots the
# resident INSIDE THE PASS LOOP: for each pass it runs the REFERENCE tree's own
# `tools/serve-up.sh --boot --spec serial --draft-len 0` from that workspace,
# reads the socket back, injects it into that pass's worker spawn, measures the
# pass, and `--stop`s the resident before the next pass boots. ONE RESIDENT PER
# PASS, never one shared across the four.
#
# THAT IS DELIBERATE, and it is what makes the band describe the scored leg. A
# scored run boots its control leg's resident once and measures it once, so a
# pass measured on a resident that four passes had already warmed would not be
# the same measurement. Paying the load per pass buys a band whose passes are
# each shaped like the leg they certify -- and it is also what makes the CV
# meaningful, because load-to-load variation is inside the sample instead of
# hidden by a shared residency.
#
# A driver that booted its own serve would be measuring a different arrangement
# than the one it certifies, which is why this one boots none.
#
# IT APPLIES NOTHING BUT THE FILE IT IS ASKED TO WRITE. No fixture, no golden and
# no constant is touched: there is no stored pair on this track to touch.
#
# HALT. The driver leads its own process group and writes run.pid and run.pgid
# beside the output file, so tools/qwen4exp-g1-halt.sh stops the driver,
# serve-up.sh, the resident engine and benchd together.
#
# Usage:  tools/calibrate-box.sh <box> <out> [options]
#           <box>   the runner name this band is attributed to (RUNNER_NAME)
#           <out>   where the calibration file is written
#
# Exit codes
#   0  the band was captured and written
#   2  refusal before any load (argument, tool, environment, reference tree)
#   3  another run holds the GPU lock
#   4  benchd refused the capture or the stability gate
#
# THE ENGINE IS NAMED RELATIVE TO THE WORKSPACE. benchd resolves it inside the
# reference tree, so the value is the staged adapter's path UNDER that tree and
# never an absolute path into this checkout -- an absolute path here would point
# the control leg at the candidate's binary.
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null && pwd -P)"

# --- own process group ------------------------------------------------------
# Same discipline as tools/qwen4exp-calibrate.sh: a halt must reach serve-up.sh,
# the resident engine and benchd together, and a signal for the run must never
# reach the operator's shell. The re-exec happens before the arguments are read,
# so the caller's argv rides through unchanged.
if [ "${CALIBRATE_BOX_SETSID:-0}" != "1" ] && command -v setsid >/dev/null 2>&1; then
  self_pgid="$(ps -o pgid= -p $$ 2>/dev/null | tr -d '[:space:]' || true)"
  if [ "${self_pgid}" != "$$" ]; then
    export CALIBRATE_BOX_SETSID=1
    exec setsid --wait "${BASH_SOURCE[0]}" "$@"
  fi
fi
export CALIBRATE_BOX_SETSID=1

FIXTURE="${REPO_DIR}/fixtures/qwen3_8_125b_a6b_track.json"
MANIFEST="${REPO_DIR}/benchmark.json"
PASSES=4
LOCK_PATH=/tmp/mtplx-gpu-exclusive.lock
LOCK_WAIT=600
DRY_RUN=0
BOX=""
OUT=""

# The staged adapter, RELATIVE to the reference workspace. It is the path
# tools/stage-cuda-engine.sh writes, and benchd resolves it inside the workspace
# it is given -- so the control leg runs the reference tree's binary.
ENGINE_REL=".build/release/mlxfast-runtime-worker"

usage() {
  cat <<'EOF'
usage: calibrate-box.sh <box> <out> [options]

  <box>            the runner name this band belongs to. It must equal
                   RUNNER_NAME when the job sets one -- a band that cannot be
                   attributed to a box is a band from anywhere
  <out>            where the calibration file is written

  --passes N       control-leg passes (default 4). At least 2: the CV is
                   undefined below two
  --fixture FILE   track fixture (default: this repo's)
  --lock PATH      GPU lock (default /tmp/mtplx-gpu-exclusive.lock)
  --lock-wait SEC  seconds to wait for the lock (default 600)
  --dry-run        print every command and stop. Takes no lock, boots no
                   engine, loads nothing and writes nothing
  -h, --help       this text

ENV:
  MLXFAST_BASELINE_WORKSPACE     the built reference tree (required)
  MLXFAST_TARGET_SNAPSHOT_DIR    the pinned target snapshot (required)
  MLXFAST_QWEN38_GOLDEN_DIR      staged goldens (default: this repo's correctness_prompts)
  BENCHD                         a benchd to use as-is (default: tools/fetch-benchd.sh)
  BENCHD_BIN_DIR                 where benchd.manifest.json is read for the
                                 benchd source commit (default: <repo>/benchd-bin)
  MLXFAST_BENCHD_SOURCE_COMMIT   the benchd source commit, when the manifest
                                 cannot be read
EOF
}

refuse() { # refuse NAME MESSAGE...
  local name="$1"; shift
  printf 'calibrate-box: REFUSE %s: %s\n' "${name}" "$*" >&2
  exit 2
}

need_tool() { # need_tool NAME WHAT_IT_IS_FOR
  command -v "$1" >/dev/null 2>&1 || refuse missing-tool "$1 is required for $2"
}

is_uint() { case "$1" in ''|*[!0-9]*) return 1 ;; *) return 0 ;; esac; }

while [ $# -gt 0 ]; do
  case "$1" in
    --passes)    PASSES="$2"; shift 2 ;;
    --fixture)   FIXTURE="$2"; shift 2 ;;
    --lock)      LOCK_PATH="$2"; shift 2 ;;
    --lock-wait) LOCK_WAIT="$2"; shift 2 ;;
    --dry-run)   DRY_RUN=1; shift ;;
    -h|--help)   usage; exit 0 ;;
    -*) printf 'calibrate-box: unknown argument %s\n' "$1" >&2; usage >&2; exit 2 ;;
    *)
      if [ -z "${BOX}" ]; then BOX="$1"
      elif [ -z "${OUT}" ]; then OUT="$1"
      else printf 'calibrate-box: unexpected argument %s\n' "$1" >&2; usage >&2; exit 2
      fi
      shift ;;
  esac
done

[ -n "${BOX}" ] || refuse missing-argument "<box> is required: the runner name this band is attributed to"
[ -n "${OUT}" ] || refuse missing-argument "<out> is required: where the calibration file is written"
is_uint "${PASSES}" || refuse bad-argument "--passes must be a non-negative integer, got '${PASSES}'"
[ "${PASSES}" -ge 2 ] || refuse bad-argument "--passes ${PASSES} is not a calibration: the sample CV is undefined below two passes, so there would be no evidence the mean describes this box"
is_uint "${LOCK_WAIT}" || refuse bad-argument "--lock-wait must be a non-negative integer, got '${LOCK_WAIT}'"

# THE BOX NAMES ITSELF. A band belongs to one machine. When the job knows its
# runner name, the argument must be that name.
if [ -n "${RUNNER_NAME:-}" ] && [ "${RUNNER_NAME}" != "${BOX}" ]; then
  refuse box-mismatch "this job runs on runner '${RUNNER_NAME}' but the band was asked for box '${BOX}'; a band captured here describes this box and no other"
fi

need_tool jq "reading the track fixture and the manifest"
need_tool git "verifying the reference workspace HEAD"
[ -r "${FIXTURE}" ] || refuse missing-fixture "cannot read the track fixture ${FIXTURE}"
[ -r "${MANIFEST}" ] || refuse missing-manifest "cannot read ${MANIFEST}"

# --- the reference tree -----------------------------------------------------
WORKSPACE="${MLXFAST_BASELINE_WORKSPACE:-}"
[ -n "${WORKSPACE}" ] || refuse missing-baseline-workspace \
  "MLXFAST_BASELINE_WORKSPACE is unset; the control leg runs on the staged reference tree and there is nothing else to calibrate"
[ -d "${WORKSPACE}" ] || refuse missing-baseline-workspace \
  "MLXFAST_BASELINE_WORKSPACE=${WORKSPACE} is not a directory; stage it with tools/stage-baseline-workspace.sh"

WANT_REF_COMMIT="$(jq -r '.baseline_reference_commit // empty' "${FIXTURE}")"
case "${WANT_REF_COMMIT}" in
  [0-9a-f]*) [ "${#WANT_REF_COMMIT}" -eq 40 ] || WANT_REF_COMMIT="" ;;
  *) WANT_REF_COMMIT="" ;;
esac
[ -n "${WANT_REF_COMMIT}" ] || refuse missing-reference-commit \
  "${FIXTURE} carries no 40-hex baseline_reference_commit; there is no reference tree to calibrate against"
HAVE_REF_COMMIT="$(git -C "${WORKSPACE}" rev-parse HEAD 2>/dev/null || true)"
[ "${HAVE_REF_COMMIT}" = "${WANT_REF_COMMIT}" ] || refuse reference-commit-mismatch \
  "the reference workspace is at ${HAVE_REF_COMMIT:-no git HEAD} but the fixture pins ${WANT_REF_COMMIT}; a band captured on another tree describes another engine"

# THE REFERENCE TREE MUST BE ABLE TO SERVE THE LEG. benchd boots it through the
# workspace's own serve script and spawns the workspace's own adapter, so both
# have to be there before the lock is taken and the box is committed.
[ -x "${WORKSPACE}/tools/serve-up.sh" ] || refuse missing-tool \
  "${WORKSPACE}/tools/serve-up.sh is not executable; benchd boots the control leg through the REFERENCE tree's own serve script"
[ -x "${WORKSPACE}/.build/ds4/ds4-resident" ] || refuse baseline-workspace-not-built \
  "the reference workspace has no executable .build/ds4/ds4-resident; run tools/ds4/build.sh there, or re-stage with tools/stage-baseline-workspace.sh"
[ -x "${WORKSPACE}/${ENGINE_REL}" ] || refuse baseline-workspace-not-built \
  "the reference workspace has no executable ${ENGINE_REL}; run tools/stage-cuda-engine.sh there, or re-stage with tools/stage-baseline-workspace.sh"

# A PRESET SERVE SPEC IS REFUSED. The control leg is SERIAL, and a value in the
# environment would reach the serve-up.sh benchd boots and arm a drafter for it.
# serve-up.sh --boot refuses a disagreeing value itself; this refuses earlier,
# before the lock and before anything is committed.
[ -z "${SERVE_UP_SPECULATIVE:-}" ] || refuse preset-serve-spec \
  "SERVE_UP_SPECULATIVE=${SERVE_UP_SPECULATIVE} is preset in the environment; the control leg is serial and its serve spec comes from nowhere else"

# AN INHERITED RESIDENT SOCKET IS REFUSED. benchd boots the control leg's
# resident and injects its socket; one already in the environment would point
# every pass's worker at a resident this calibration never booted.
for var in DS4_RESIDENT_SOCKET BENCH_WORKER_RESIDENT_SOCKET; do
  eval "value=\${${var}:-}"
  [ -z "${value}" ] || refuse inherited-resident-socket \
    "${var}=${value} is set in the environment; benchd boots the control leg's resident and injects its socket, so an inherited one would measure a resident this calibration never booted"
done

WEIGHTS_DIR="${MLXFAST_TARGET_SNAPSHOT_DIR:-}"
[ -n "${WEIGHTS_DIR}" ] || refuse missing-weights \
  "MLXFAST_TARGET_SNAPSHOT_DIR is unset; the reference serve has no target snapshot to load"
[ -d "${WEIGHTS_DIR}" ] || refuse missing-weights \
  "MLXFAST_TARGET_SNAPSHOT_DIR=${WEIGHTS_DIR} is not a directory"

TRACK_ID="$(jq -r '.trackId // empty' "${MANIFEST}")"
[ -n "${TRACK_ID}" ] || refuse missing-track-id "${MANIFEST} carries no trackId"
export MLXFAST_QWEN_MTP_TRACK_ID="${TRACK_ID}"

# --- the golden the band is captured on -------------------------------------
# The fixture's live_golden, the ONE prompt the scored run times, resolved from
# the staged pool exactly as tools/qwen38-125b-a6b-measure-and-score.sh resolves
# it. A band captured on another prompt does not describe the scored leg.
LIVE_GOLDEN="$(jq -r '.live_golden // empty' "${FIXTURE}")"
[ -n "${LIVE_GOLDEN}" ] || refuse missing-live-golden "${FIXTURE} declares no live_golden"
GOLDEN_DIR="${MLXFAST_QWEN38_GOLDEN_DIR:-${REPO_DIR}/correctness_prompts/qwen3.8-125b-a6b-cuda-v1}"
GOLDEN_PATH="${GOLDEN_DIR}/${LIVE_GOLDEN}.golden.json"
[ -r "${GOLDEN_PATH}" ] || refuse missing-golden \
  "cannot read the live golden ${GOLDEN_PATH}; stage the pool, or set MLXFAST_QWEN38_GOLDEN_DIR"

# --- benchd -----------------------------------------------------------------
# Honoured as-is when the caller sets it, exactly as measure-and-score.sh does.
# A DRY RUN RESOLVES NOTHING: tools/fetch-benchd.sh writes into benchd-bin/ and
# may reach the dist channel to do it, and --dry-run promises to write nothing.
BENCHD_UNRESOLVED='<tools/fetch-benchd.sh, resolved at run time>'
if [ -z "${BENCHD:-}" ]; then
  if [ "${DRY_RUN}" -eq 1 ]; then
    BENCHD="${BENCHD_UNRESOLVED}"
  else
    BENCHD="$("${REPO_DIR}/tools/fetch-benchd.sh")"
  fi
fi
if [ "${BENCHD}" != "${BENCHD_UNRESOLVED}" ]; then
  [ -x "${BENCHD}" ] || refuse missing-benchd "benchd not found at ${BENCHD}"
fi

# --- the benchd source commit ------------------------------------------------
# The band records WHICH BENCHD MEASURED IT, because a band captured by one
# scoring binary does not describe what another one measures. The value is not
# invented here: it is the source_commit of the manifest that sits beside the
# pinned benchd, which is the same pair tools/fetch-benchd.sh verifies and
# tools/ranked-box-preflight.sh section 6 checks. MLXFAST_BENCHD_SOURCE_COMMIT
# is honoured when a box has the value but not the manifest.
BENCHD_SOURCE_COMMIT="${MLXFAST_BENCHD_SOURCE_COMMIT:-}"
if [ -z "${BENCHD_SOURCE_COMMIT}" ]; then
  BENCHD_MANIFEST="${BENCHD_BIN_DIR:-${REPO_DIR}/benchd-bin}/benchd.manifest.json"
  [ -r "${BENCHD_MANIFEST}" ] || refuse missing-benchd-manifest \
    "cannot read ${BENCHD_MANIFEST}; the band must record which benchd measured it, and this is where that is written. Stage the benchd pair, or set MLXFAST_BENCHD_SOURCE_COMMIT"
  BENCHD_SOURCE_COMMIT="$(jq -r '.source_commit // empty' "${BENCHD_MANIFEST}")"
fi
case "${BENCHD_SOURCE_COMMIT}" in
  [0-9a-f]*) [ "${#BENCHD_SOURCE_COMMIT}" -eq 40 ] || BENCHD_SOURCE_COMMIT="" ;;
  *) BENCHD_SOURCE_COMMIT="" ;;
esac
[ -n "${BENCHD_SOURCE_COMMIT}" ] || refuse unattributable-benchd \
  "the benchd source commit is not a 40-hex sha; a band captured by an unattributable benchd cannot be traced to the code that measured it"

# EVERY VALUE THE FILE WILL CARRY IS PASSED EXPLICITLY. benchd falls back to
# RUNNER_NAME and MLXFAST_BENCHD_SOURCE_COMMIT when the flags are absent, but a
# fallback is a value nobody chose: passing them puts the whole band's identity
# in the command line the run log records, and makes a disagreement a refusal
# here rather than a surprise in the file.
CALIBRATE_ARGV=(
  "${BENCHD}" calibrate-baseline
  --baseline-workspace "${WORKSPACE}"
  --engine "${ENGINE_REL}"
  --weights "${WEIGHTS_DIR}"
  --golden "${GOLDEN_PATH}"
  --out "${OUT}"
  --passes "${PASSES}"
  --box "${BOX}"
  --track "${TRACK_ID}"
  --prompt "${LIVE_GOLDEN}"
  --reference-commit "${WANT_REF_COMMIT}"
  --benchd-source-commit "${BENCHD_SOURCE_COMMIT}"
)

# --- dry run ----------------------------------------------------------------
if [ "${DRY_RUN}" -eq 1 ]; then
  cat <<EOF
calibrate-box: DRY RUN -- no lock, no engine, nothing written

track                ${TRACK_ID}
box                  ${BOX}
reference workspace  ${WORKSPACE} (HEAD ${HAVE_REF_COMMIT})
reference commit     ${WANT_REF_COMMIT}
weights              ${WEIGHTS_DIR}
golden               ${GOLDEN_PATH}
prompt               ${LIVE_GOLDEN}
passes               ${PASSES}
engine (in workspace) ${ENGINE_REL}
benchd               ${BENCHD}
benchd source commit ${BENCHD_SOURCE_COMMIT}
output               ${OUT}
GPU lock             ${LOCK_PATH} (wait ${LOCK_WAIT}s)

benchd boots the control leg's resident itself, from the reference tree, ONCE
PER PASS -- booted, measured, stopped, then the next pass:
  ${WORKSPACE}/tools/serve-up.sh --boot --spec serial --draft-len 0 --socket-out <FILE>
  ${WORKSPACE}/tools/serve-up.sh --stop --socket <SOCKET>

the calibration:
  ${CALIBRATE_ARGV[*]}
EOF
  exit 0
fi

# --- the lock, then the one reference resident ------------------------------
need_tool flock "holding the GPU lock"
need_tool setsid "putting the run in its own process group; without it halt would have to signal the operator's group"

OUT_DIR="$(dirname -- "${OUT}")"
mkdir -p "${OUT_DIR}"
printf '%s\n' "$$" > "${OUT_DIR}/run.pid"
ps -o pgid= -p $$ 2>/dev/null | tr -d '[:space:]' > "${OUT_DIR}/run.pgid" || true

exec 9>>"${LOCK_PATH}" || refuse lock-unwritable "cannot open the GPU lock ${LOCK_PATH}"
if ! flock -w "${LOCK_WAIT}" 9; then
  printf 'calibrate-box: REFUSE gpu-lock-busy: %s is held by another run after %ss. Nothing has been loaded.\n' \
    "${LOCK_PATH}" "${LOCK_WAIT}" >&2
  exit 3
fi
printf 'calibrate-box: GPU lock held: %s\n' "${LOCK_PATH}" >&2

# Nothing this driver starts outlives it. benchd stops the control leg's resident
# with `serve-up.sh --stop` on success and on failure alike, and this covers a
# benchd -- or a resident it did not reach -- that outlived its parent.
kill_run_group() {
  local pids
  pids="$(pgrep -g "$$" 2>/dev/null | grep -v "^$$\$" || true)"
  if [ -n "${pids}" ]; then
    # Deliberate word splitting: pids is a newline-separated list.
    # shellcheck disable=SC2086
    kill -KILL ${pids} 2>/dev/null || true
  fi
}
trap kill_run_group EXIT

printf 'calibrate-box: capturing %s passes of the SERIAL control leg on %s (%s)\n' \
  "${PASSES}" "${BOX}" "${LIVE_GOLDEN}" >&2
if ! "${CALIBRATE_ARGV[@]}"; then
  printf 'calibrate-box: benchd refused the capture or the stability gate; nothing was pinned\n' >&2
  exit 4
fi
printf 'calibrate-box: wrote %s\n' "${OUT}" >&2
