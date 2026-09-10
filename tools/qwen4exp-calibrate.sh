#!/usr/bin/env bash
# qwen4exp-calibrate.sh -- the box driver for the CUDA track's SERIAL baseline
# calibration (lane L12c).
#
# The official baseline is the serial denominator every scored speculative leg
# is divided by. David's rule is that it matches the official measurement path:
# it is measured by that path itself, on this box, with one weights load, and
# its legs must agree to within 1% before their mean is pinned.
#
# WHAT RUNS, IN ORDER
#
#   1. tools/spec-declaration.sh describe must say `serial`. The calibration
#      authors the serial reference; a box whose declared spec is a draft depth
#      would measure the wrong leg.
#   2. The GPU lock is taken and HELD for the whole calibration. serve-up.sh
#      does not take it -- its own comment says the caller owns the window --
#      and a resident engine holds GPU memory for as long as this driver runs.
#   3. serve-up.sh boots ONE ds4-resident, SERIAL (SERVE_UP_SPECULATIVE=0), and
#      this driver re-execs itself inside that wrapper (--inner). Every golden's
#      legs then run against the SAME residency: the weights load once for the
#      whole calibration, not once per golden and not once per leg.
#   4. Inside, `benchd weights-digest` hashes the ~105 GiB tree ONCE and the
#      digest is handed to every pass, so no pass re-hashes it.
#   5. Per golden: `benchd iterate --capture-baseline <REC> --capture-passes
#      W,A,A,A,A`. That is benchd's own capture flow, unchanged -- the same
#      iterate_flow_windowed, decode window and golden-oracle workload the
#      scored run times. Pass W is the WARM-UP leg and its record is discarded;
#      the four A passes merge into one record of four legs.
#   6. `benchd calibrate-baseline` gates every record's per-axis sample CV at
#      1%, and prints the pair, the bench-core constants patch and the golden
#      `baseline_*_seconds_per_token` fields.
#
# IT APPLIES NOTHING. The report and the patch land in the run directory. Pinning
# a scored denominator is David's call and a reviewed PR against benchd; this
# driver never edits a constant, a golden or a fixture.
#
# THE CAPTURE MODE MUST BE ARMED. `--capture-baseline` runs only while the
# platform's baseline is PENDING, so it refuses by name on a benchd whose
# OFFICIAL_BASELINE_CUDA still carries the retired vLLM pair. That is the point:
# the capture instrument and the scoring binary are opposites, and a benchd that
# would score cannot capture.
#
# HALT. The driver leads its own process group and writes run.pid and run.pgid
# into the run directory, so tools/qwen4exp-g1-halt.sh RUNDIR stops the driver,
# serve-up.sh, the resident engine and benchd together.
#
# EXIT CODES
#   0  every golden's legs passed the CV gate; the report and patch are written
#   2  refusal before any load (argument, tool, pin, declared spec)
#   3  another run holds the GPU lock
#   4  a capture or the CV gate refused
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null && pwd -P)"

# --- own process group ------------------------------------------------------
# Same discipline as qwen4exp-g1-boot.sh: halt must reach serve-up.sh, the
# resident engine and benchd together, and a signal for the run must never
# reach the operator's shell. The re-exec happens before the arguments are read,
# so the caller's argv rides through unchanged. The --inner re-exec below is
# already inside this group and must not re-do it.
if [ "${QWEN4EXP_CALIBRATE_SETSID:-0}" != "1" ] && command -v setsid >/dev/null 2>&1; then
  self_pgid="$(ps -o pgid= -p $$ 2>/dev/null | tr -d '[:space:]' || true)"
  if [ "${self_pgid}" != "$$" ]; then
    export QWEN4EXP_CALIBRATE_SETSID=1
    exec setsid --wait "${BASH_SOURCE[0]}" "$@"
  fi
fi
export QWEN4EXP_CALIBRATE_SETSID=1

# --- defaults ---------------------------------------------------------------
WEIGHTS_DIR="${MLXFAST_TARGET_SNAPSHOT_DIR:-}"
ENGINE_BIN="${MLXFAST_ENGINE_BIN:-${REPO_DIR}/.build/release/mlxfast-runtime-worker}"
FIXTURE="${REPO_DIR}/fixtures/qwen3_8_125b_a6b_track.json"
MANIFEST="${REPO_DIR}/benchmark.json"
RUN_ROOT="${REPO_DIR}/.build/calibration-runs"
LEGS=4
PIN=""
LOCK_PATH=/tmp/mtplx-gpu-exclusive.lock
LOCK_WAIT=600
DRY_RUN=0
INNER=0
RUN=""

# The two collaborators, overridable the way measure-and-score.sh already lets a
# caller set BENCHD: an operator points them at a staged build, and the
# mechanics test points them at stubs. Neither is hash-checked here.
SERVE_UP="${QWEN4EXP_CALIBRATE_SERVE_UP:-${REPO_DIR}/tools/serve-up.sh}"
SPEC_DECLARATION="${QWEN4EXP_CALIBRATE_SPEC_DECLARATION:-${REPO_DIR}/tools/spec-declaration.sh}"

usage() {
  cat <<'EOF'
usage: qwen4exp-calibrate.sh --weights DIR [options]

  --weights DIR    the pinned target snapshot (GGUF shards + MTP head).
                   Default: $MLXFAST_TARGET_SNAPSHOT_DIR
  --engine PATH    the benchd runtime-worker adapter
                   (default <repo>/.build/release/mlxfast-runtime-worker)
  --legs N         timed legs per golden AFTER the warm-up leg (default 4).
                   The sample CV needs at least 2
  --pin NAME       the golden whose pair becomes the platform constant
                   (default: the fixture's live_golden)
  --run-dir DIR    parent of the timestamped run dir
                   (default <repo>/.build/calibration-runs)
  --fixture FILE   track fixture (default: this repo's)
  --lock PATH      GPU lock (default /tmp/mtplx-gpu-exclusive.lock)
  --lock-wait SEC  seconds to wait for the lock (default 600)
  --dry-run        print every command and stop. Takes no lock, boots no
                   engine, loads nothing, writes nothing, and fetches no
                   benchd (an unset BENCHD prints as unresolved)
  -h, --help       this text

ENV:
  BENCHD                              a benchd to use as-is (default: tools/fetch-benchd.sh)
  MLXFAST_QWEN38_GOLDEN_DIR           the staged goldens. Required: they are box material
  QWEN4EXP_CALIBRATE_SERVE_UP         serve-up.sh to use
  QWEN4EXP_CALIBRATE_SPEC_DECLARATION spec-declaration.sh to use
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --weights)   WEIGHTS_DIR="$2"; shift 2 ;;
    --engine)    ENGINE_BIN="$2"; shift 2 ;;
    --legs)      LEGS="$2"; shift 2 ;;
    --pin)       PIN="$2"; shift 2 ;;
    --run-dir)   RUN_ROOT="$2"; shift 2 ;;
    --fixture)   FIXTURE="$2"; shift 2 ;;
    --lock)      LOCK_PATH="$2"; shift 2 ;;
    --lock-wait) LOCK_WAIT="$2"; shift 2 ;;
    --dry-run)   DRY_RUN=1; shift ;;
    --inner)     INNER=1; shift ;;
    --run)       RUN="$2"; shift 2 ;;
    -h|--help)   usage; exit 0 ;;
    *) printf 'qwen4exp-calibrate: unknown argument %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

refuse() { # refuse NAME MESSAGE...
  local name="$1"; shift
  printf 'qwen4exp-calibrate: REFUSE %s: %s\n' "${name}" "$*" >&2
  exit 2
}

need_tool() { # need_tool PATH_OR_NAME WHAT_IT_IS_FOR
  if [ "${1#/}" != "$1" ]; then
    [ -x "$1" ] || refuse missing-tool "$1 is required for $2"
  else
    command -v "$1" >/dev/null 2>&1 || refuse missing-tool "$1 is required for $2"
  fi
}

is_uint() { case "$1" in ''|*[!0-9]*) return 1 ;; *) return 0 ;; esac; }

[ -n "${WEIGHTS_DIR}" ] || refuse missing-argument "--weights DIR is required (or set MLXFAST_TARGET_SNAPSHOT_DIR)"
is_uint "${LEGS}" || refuse bad-argument "--legs must be a non-negative integer, got '${LEGS}'"
is_uint "${LOCK_WAIT}" || refuse bad-argument "--lock-wait must be a non-negative integer, got '${LOCK_WAIT}'"
[ "${LEGS}" -ge 2 ] || refuse bad-argument "--legs ${LEGS} cannot be calibrated: the sample CV is undefined below two legs, so there would be no evidence the mean describes the box"

need_tool jq "reading the track fixture and the manifest"
[ -r "${FIXTURE}" ] || refuse missing-fixture "cannot read the track fixture ${FIXTURE}"
[ -r "${MANIFEST}" ] || refuse missing-manifest "cannot read ${MANIFEST}"
[ -x "${SERVE_UP}" ] || refuse missing-tool "${SERVE_UP} is required for the one-load resident engine"
[ -x "${SPEC_DECLARATION}" ] || refuse missing-tool "${SPEC_DECLARATION} is required for the declared spec"

# --- benchd ---------------------------------------------------------------
# Honoured as-is when the caller sets it, exactly as measure-and-score.sh does;
# otherwise resolved from the dist channel, which verifies it against its own
# manifest.
#
# A DRY RUN RESOLVES NOTHING. tools/fetch-benchd.sh writes the binary into
# benchd-bin/ and may reach the dist channel to do it, and --dry-run promises to
# take no lock, boot nothing and write nothing -- a promise an operator relies on
# when checking the plan from a box that is not theirs. So the dry run prints
# what it WOULD resolve instead of resolving it. A caller-set BENCHD is still
# honoured and still checked, because that path fetches nothing.
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
  # Exported so the --inner re-exec uses the SAME binary the outer half resolved
  # and verified, instead of resolving the channel a second time.
  export BENCHD
fi

# --- the declared spec must be SERIAL ---------------------------------------
# The calibration authors the serial reference. A box whose declared spec is a
# draft depth would boot a speculative serve and measure the wrong leg -- and
# benchd would then refuse it by name at CALIBRATION-SPEC-ARMED after paying for
# the load. Refuse here instead, before anything boots.
SPEC_DESC="$("${SPEC_DECLARATION}" describe)"
[ "${SPEC_DESC}" = "serial" ] || refuse declared-spec-not-serial \
  "the declared spec is '${SPEC_DESC}', but the official baseline is the SERIAL denominator every scored speculative leg is divided by. Calibrate on a serial declaration"

TRACK_ID="$(jq -r '.trackId // empty' "${MANIFEST}")"
[ -n "${TRACK_ID}" ] || refuse missing-track-id "${MANIFEST} carries no trackId"
export MLXFAST_QWEN_MTP_TRACK_ID="${TRACK_ID}"

LIVE_GOLDEN="$(jq -r '.live_golden // empty' "${FIXTURE}")"
[ -n "${LIVE_GOLDEN}" ] || refuse missing-live-golden "${FIXTURE} declares no live_golden"
PIN="${PIN:-${LIVE_GOLDEN}}"

# The timed pool, as `<name>\t<path>\t<sha256>\t<bytes>` lines. The names, pins
# and byte counts all come from the fixture; this driver invents none of them.
GOLDEN_DIR="${MLXFAST_QWEN38_GOLDEN_DIR:-}"
[ -n "${GOLDEN_DIR}" ] || refuse missing-golden-dir \
  "MLXFAST_QWEN38_GOLDEN_DIR is unset; the timed-pool goldens are staged on the box out of band and this driver fetches nothing"
POOL="$(jq -r --arg dir "${GOLDEN_DIR}" '
  .timed_prompt_pool[]
  | (.r2_path | split("/") | last) as $base
  | ($base | sub("\\.golden\\.json$"; "")) as $name
  | [$name, $dir + "/" + $base, .sha256, (.bytes | tostring)]
  | @tsv' "${FIXTURE}")"
[ -n "${POOL}" ] || refuse empty-pool "${FIXTURE} declares no timed_prompt_pool"

grep -q "^${PIN}	" <<<"${POOL}" \
  || refuse unknown-pin "--pin '${PIN}' names no golden in the fixture's timed_prompt_pool"

# The pass spec: one warm-up pass then LEGS timed passes, all over one residency.
# W's record is discarded -- a first leg on a just-loaded engine measures the
# load, not the box.
PASSES="W$(for _ in $(seq 1 "${LEGS}"); do printf ',A'; done)"

# Build the capture invocation for one golden into the global ITER_ARGV. A global
# rather than a printed-and-re-split list: a path with a space must survive, and
# this shell may be bash 3.2 (no mapfile).
ITER_ARGV=()
iterate_argv() { # iterate_argv GOLDEN_PATH SHA256 BYTES RECORD_BASE [DIGEST]
  ITER_ARGV=(
    "${BENCHD}" iterate
    --engine "${ENGINE_BIN}"
    --weights "${WEIGHTS_DIR}"
    --golden "$1"
    --golden-sha256 "$2"
    --golden-bytes "$3"
    --mode local-iterate
    --capture-baseline "$4"
    --capture-passes "${PASSES}"
  )
  [ -n "${5:-}" ] && ITER_ARGV+=(--weights-digest "$5")
  return 0
}

# --- dry run ----------------------------------------------------------------
if [ "${DRY_RUN}" -eq 1 ] && [ "${INNER}" -eq 0 ]; then
  cat <<EOF
qwen4exp-calibrate: DRY RUN -- no lock, no engine, nothing written

track                ${TRACK_ID}
declared spec        ${SPEC_DESC}
weights              ${WEIGHTS_DIR}
engine               ${ENGINE_BIN}
benchd               ${BENCHD}
passes per golden    ${PASSES}  (W discarded, ${LEGS} timed legs)
pinned golden        ${PIN}
run dir              ${RUN_ROOT}/<UTC timestamp>-calib
GPU lock             ${LOCK_PATH} (wait ${LOCK_WAIT}s)

one resident engine for the whole calibration:
  SERVE_UP_WEIGHTS_DIR=${WEIGHTS_DIR} SERVE_UP_SPECULATIVE=0 ${SERVE_UP} \\
    ${BASH_SOURCE[0]} --inner --run <RUN>

  ${BENCHD} weights-digest --weights ${WEIGHTS_DIR}
EOF
  while IFS=$'\t' read -r name path sha bytes; do
    iterate_argv "${path}" "${sha}" "${bytes}" "<RUN>/legs/${name}.json" '<DIGEST>'
    printf '\n  %s\n' "${name}"
    printf '    %s\n' "${ITER_ARGV[*]}"
  done <<<"${POOL}"
  printf '\nthen, over the merged A records:\n  %s calibrate-baseline --track %s --pin %s --record <RUN>/records/<name>.json ... --json-out <RUN>/calibration.json\n' \
    "${BENCHD}" "${TRACK_ID}" "${PIN}"
  exit 0
fi

# --- the inner half: everything that needs the resident engine ---------------
# Re-exec'd by the outer half inside serve-up.sh, so every golden's legs run
# against the ONE residency serve-up booted. It takes no lock (the outer half
# holds it) and boots nothing.
if [ "${INNER}" -eq 1 ]; then
  [ -n "${RUN}" ] || refuse missing-argument "--inner requires --run DIR"
  LOG="${RUN}/driver.log"
  log() { printf '[%s] %s\n' "$(date -u +%H:%M:%SZ)" "$*" | tee -a "${LOG}"; }
  mkdir -p "${RUN}/legs" "${RUN}/records"

  # The ~105 GiB tree is hashed ONCE for the whole calibration and the digest is
  # handed to every pass (`--weights-digest`), instead of once per pass.
  DIGEST="$("${BENCHD}" weights-digest --weights "${WEIGHTS_DIR}")"
  log "weights digest ${DIGEST}"

  RECORDS=()
  while IFS=$'\t' read -r name path sha bytes; do
    [ -r "${path}" ] || refuse missing-golden "cannot read the golden ${path} for ${name}"
    log "capturing ${name}: ${PASSES}"
    iterate_argv "${path}" "${sha}" "${bytes}" "${RUN}/legs/${name}.json" "${DIGEST}"
    if ! "${ITER_ARGV[@]}" >>"${LOG}" 2>&1; then
      printf 'qwen4exp-calibrate: capture failed for %s; see %s\n' "${name}" "${LOG}" >&2
      exit 4
    fi
    # The four A passes merged into one record; W's is the warm-up and is left
    # where it fell. The copy carries the golden's NAME, which is what the report
    # and the golden-field block are keyed by.
    cp "${RUN}/legs/${name}.A.json" "${RUN}/records/${name}.json"
    RECORDS+=(--record "${RUN}/records/${name}.json")
  done <<<"${POOL}"

  log "gating ${#RECORDS[@]} record(s) at CV <= 1% and printing the pin"
  if ! "${BENCHD}" calibrate-baseline \
        --track "${TRACK_ID}" \
        --pin "${PIN}" \
        "${RECORDS[@]}" \
        --json-out "${RUN}/calibration.json" \
        | tee "${RUN}/calibration-report.txt"; then
    printf 'qwen4exp-calibrate: the CV gate refused; see %s\n' "${LOG}" >&2
    exit 4
  fi
  # The patch alone, for the reviewed benchd PR. Extracted from the JSON report,
  # never re-derived here.
  jq -r '.constants_patch' "${RUN}/calibration.json" > "${RUN}/official-baseline.patch"
  log "report ${RUN}/calibration-report.txt"
  log "patch  ${RUN}/official-baseline.patch  (APPLIED NOTHING -- David gates the pin)"
  exit 0
fi

# --- the outer half: run dir, lock, one resident engine ---------------------
RUN="${RUN_ROOT%/}/$(date -u +%Y%m%dT%H%M%SZ)-calib"
mkdir -p "${RUN}"
LOG="${RUN}/driver.log"
log() { printf '[%s] %s\n' "$(date -u +%H:%M:%SZ)" "$*" | tee -a "${LOG}"; }

printf '%s\n' "$$" > "${RUN}/run.pid"
ps -o pgid= -p $$ 2>/dev/null | tr -d '[:space:]' > "${RUN}/run.pgid" || true

log "run dir ${RUN}"
log "track ${TRACK_ID}, declared spec ${SPEC_DESC}, pinned golden ${PIN}"
log "passes per golden ${PASSES} (W discarded, ${LEGS} timed legs)"
log "benchd ${BENCHD}"

# The lock is HELD for the whole calibration: serve-up.sh does not take it, and
# the resident engine holds GPU memory for as long as this driver runs. Both
# tools are needed only HERE -- the dry run takes no lock and the inner half runs
# inside the group the outer half already leads -- so they are checked here
# rather than at the top, where they would refuse a dry run on a box that will
# never take a lock.
need_tool flock "holding the GPU lock"
need_tool setsid "putting the run in its own process group; without it halt would have to signal the operator's group"
exec 9>>"${LOCK_PATH}" || refuse lock-unwritable "cannot open the GPU lock ${LOCK_PATH}"
if ! flock -w "${LOCK_WAIT}" 9; then
  printf 'qwen4exp-calibrate: REFUSE gpu-lock-busy: %s is held by another run after %ss. Nothing has been loaded.\n' "${LOCK_PATH}" "${LOCK_WAIT}" | tee -a "${LOG}" >&2
  exit 3
fi
log "GPU lock held: ${LOCK_PATH}"

# Nothing this driver starts outlives it: serve-up.sh tears the resident down on
# its own exit, and this covers a benchd or a sampler that outlived its parent.
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

# ONE resident engine, SERIAL, for every golden's every leg.
env SERVE_UP_WEIGHTS_DIR="${WEIGHTS_DIR}" \
    SERVE_UP_SPECULATIVE=0 \
    "${SERVE_UP}" \
    "${BASH_SOURCE[0]}" --inner --run "${RUN}" \
      --weights "${WEIGHTS_DIR}" --engine "${ENGINE_BIN}" \
      --legs "${LEGS}" --pin "${PIN}" --fixture "${FIXTURE}"
