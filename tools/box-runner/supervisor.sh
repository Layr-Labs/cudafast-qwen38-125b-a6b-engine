#!/usr/bin/env bash
# supervisor.sh -- mint -> run -> reap loop that makes a box serve GitHub
# Actions jobs for this repository's ranked benchmark, one job at a time.
#
# Adapted from the box-3-proven qwen/bench lineage
# (the operator's machine-scripts repository, READ-ONLY reference -- this
# script does not modify that repo or anything in it)
# with its JIT registration pattern kept and its orphan bug fixed. Quoting
# the box-3 lesson this script exists to fix, verbatim from that reference's
# header (supervisor.sh:2-30 there):
#
#   > Loop, one iteration per job:
#   >   ...
#   >   4. Start actions-runner with that JIT config as the `runner` user and
#   >      wait for it to exit. JIT runners are single-job by construction
#   >      (equivalent to --ephemeral): the registration is consumed by
#   >      exactly one job.
#   >   ...
#   > One job at a time is guaranteed by the JIT/ephemeral registration; the
#   > supervisor never runs two runner processes concurrently and adds no
#   > other concurrency.
#
# and the spawn line itself (supervisor.sh:158 there):
#
#   > su - "${RUNNER_USER}" -c "... ${timeout_bin} ./run.sh"
#
# THE BUG: `su -` starts a NEW LOGIN SESSION. When launchd stops the
# supervisor daemon (crash, `launchctl bootout`, an operator's kickstart),
# the signal targets the DAEMON'S OWN process group -- and the su-spawned
# runner session was never a member of it, so it is not reached even though
# both were alive at that instant. The job tree becomes a live orphan that
# outlives the daemon that was supposed to manage it, and nothing before had
# a way to find or stop it short of an operator manually hunting `ps`.
#
# THE FIX (this script + lib/pidtree.sh): the job's root pid is written to
# PID_FILE the instant it is known, BEFORE this process waits on it, so the
# tree is discoverable by ANY later invocation -- not just this one, and not
# only while this process is alive. `stop` (below) reads that pid and asks
# lib/pidtree.sh to walk the LIVE PARENT-PID CHAIN from it (a kernel fact
# independent of session id or process group, so `su -`'s session boundary
# does not hide anything from it) and kill every descendant found. `run`
# calls the exact same reap at startup, before minting a new job, so
# restarting the daemon after an ungraceful death (the box-3 scenario)
# actually recovers instead of leaking a second concurrent job tree. See
# lib/pidtree.sh's header for the full mechanism.
#
# Also carried forward unchanged from the reference (supervisor.sh:130-142
# there), because it is a distinct, still-load-bearing hardening and this
# script keeps the same technique: the JIT config crosses to the runner OFF
# argv, via the ACTIONS_RUNNER_INPUT_JITCONFIG env var (the actions-runner's
# own env-input mechanism), so no other uid can scrape it from `ps`. Neither
# the App key nor any token this process's gh-app-mint.sh mints ever enters
# the runner's environment or argv.
#
# Usage:
#   supervisor.sh [run]     Run the mint/spawn/reap loop (default; foreground
#                            -- a launchd plist installed by install.sh is
#                            what keeps it running as a daemon).
#   supervisor.sh stop      Standalone entrypoint: if a job tree is tracked
#                            (PID_FILE), kill it (TERM, wait, KILL survivors)
#                            and clear the tracking file. Safe to call any
#                            time, including when this script's own `run`
#                            process is no longer alive -- that is precisely
#                            the box-3 recovery case above.
#   supervisor.sh status     Print `idle`, or `running: ...` plus one `ps`
#                            line per live process in the current job's
#                            tree -- the between-jobs quiescence check.
#
# Configuration: see box-runner.env.example for every variable below, and
# tools/box-runner/README.md for the parameterization surface (RUNNER_LABELS
# in particular is coupled to the sibling participant-contract workflow's
# `runs-on:` and has no default here for that reason).
set -u

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null && pwd -P)"
# shellcheck source=tools/box-runner/lib/pidtree.sh
. "${SCRIPT_DIR}/lib/pidtree.sh"

# --- Configuration (override via BOX_RUNNER_CONFIG, or export directly) -----
CONFIG_ENV="${BOX_RUNNER_CONFIG:-${SCRIPT_DIR}/box-runner.env}"
# shellcheck disable=SC1090  # operator-editable file; path resolved above
[ -f "${CONFIG_ENV}" ] && . "${CONFIG_ENV}"

RUNNER_USER="${RUNNER_USER:-runner}"          # unprivileged runner account
RUNNER_DIR="${RUNNER_DIR:-}"                  # actions-runner install, owned by RUNNER_USER
GH_APP_MINT="${GH_APP_MINT:-${SCRIPT_DIR}/gh-app-mint.sh}"
RUNNER_NAME_PREFIX="${RUNNER_NAME_PREFIX:-qwen38-125b-a6b-box}"
# NO DEFAULT. Must equal (a subset of) the `runs-on:` labels in
# .github/workflows/benchmark.yml, today
# `[self-hosted, macOS, qwen3.8-125b-a6b-cuda-v1]` (gh-app-mint.sh adds
# `self-hosted` itself). A wrong or missing value silently strands every job
# dispatched against this runner rather than failing loudly at dispatch time,
# so this refuses to start instead of guessing.
RUNNER_LABELS="${RUNNER_LABELS:-}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/log}"
STATE_DIR="${STATE_DIR:-${SCRIPT_DIR}/state}"
PID_FILE="${STATE_DIR}/job.pid"
PAUSE_BETWEEN_JOBS="${PAUSE_BETWEEN_JOBS:-15}"
IDLE_RECYCLE_SECONDS="${IDLE_RECYCLE_SECONDS:-3600}"     # re-mint JIT config if no job arrives
STOP_TERM_TIMEOUT_SECONDS="${STOP_TERM_TIMEOUT_SECONDS:-10}" # grace before SIGKILL in stop/reap
ALERT_HOOK="${ALERT_HOOK:-}"                              # optional executable, called with 1 msg arg

# TEST / DRY-RUN ONLY -- NEVER set either of these in a production install.
#   RUNNER_SU_ENABLED=0  skips the `su - RUNNER_USER` privilege drop and runs
#                         RUNNER_DIR/run.sh directly as the invoking user, so
#                         the spawn/stop/status/kill-tree logic can be
#                         exercised without root or a provisioned `runner`
#                         account. install.sh never sets this; test/*.sh
#                         does. The tree-tracking and kill-tree mechanism
#                         exercised is identical either way.
#   BOX_RUNNER_MAX_ITERATIONS  bounds the loop to N iterations (0 = run
#                         forever, the production default) so a test can
#                         assert on a finite run.
RUNNER_SU_ENABLED="${RUNNER_SU_ENABLED:-1}"
BOX_RUNNER_MAX_ITERATIONS="${BOX_RUNNER_MAX_ITERATIONS:-0}"

mkdir -p "${LOG_DIR}" "${STATE_DIR}"

log() {
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) supervisor: $*" >> "${LOG_DIR}/supervisor.log"
}

# alert MESSAGE -- must never itself fail the caller. Always visible on
# stderr (not just the log file): under launchd, StandardErrorPath already
# captures it into a log either way, but a direct/manual/CI invocation has
# nothing else watching LOG_DIR, and "refuse with a named error" needs to
# actually be seen to mean anything.
alert() {
  log "ALERT: $*"
  echo "box-runner: ALERT: $*" >&2
  if [ -n "${ALERT_HOOK}" ] && [ -x "${ALERT_HOOK}" ]; then
    "${ALERT_HOOK}" "$*" || log "alert hook failed"
  fi
}

# fail MESSAGE -- named, fail-closed configuration/credential error. Every
# refusal in this script goes through here so it is alerted AND logged, and
# so the exit code is uniformly EX_CONFIG (78, sysexits.h) for a config
# problem as opposed to a runtime one.
fail() {
  alert "$*"
  exit 78
}

require_config() {
  [ -n "${RUNNER_DIR}" ] || fail "RUNNER_DIR is required (no default -- path to the actions-runner install)"
  [ -e "${RUNNER_DIR}/run.sh" ] || fail "actions-runner install missing: ${RUNNER_DIR}/run.sh (run install.sh first)"
  [ -e "${GH_APP_MINT}" ] || fail "gh-app mint helper missing: ${GH_APP_MINT}"
  [ -x "${GH_APP_MINT}" ] || fail "gh-app mint helper is not executable: ${GH_APP_MINT}"
  [ -n "${RUNNER_LABELS}" ] || fail "RUNNER_LABELS is required (no default -- see the header comment above and README.md)"
}

# resolve_timeout_bin -- prints "timeout N" / "gtimeout N" if a timeout(1) is
# on PATH (GNU coreutils; macOS ships neither by default, Homebrew's
# coreutils installs gtimeout), else prints nothing. IDLE_RECYCLE_SECONDS is
# meaningless without one -- the loop still functions, it just never
# self-recycles an idle JIT registration.
resolve_timeout_bin() {
  if command -v timeout >/dev/null 2>&1; then
    printf 'timeout %s' "${IDLE_RECYCLE_SECONDS}"
  elif command -v gtimeout >/dev/null 2>&1; then
    printf 'gtimeout %s' "${IDLE_RECYCLE_SECONDS}"
  fi
}

# --- JIT registration ---------------------------------------------------------
# Delegates to gh-app-mint.sh, which builds an RS256 JWT from the App private
# key, mints a scoped 1h installation token (revoked on its own exit), and
# calls generate-jitconfig. The App key is read only inside that helper; this
# process only ever sees the single-use encoded JIT config on its stdout.
mint_jit_config() {
  local name="$1"
  "${GH_APP_MINT}" jitconfig "${name}" "${RUNNER_LABELS}"
}

# --- Spawn ---------------------------------------------------------------------
RUNNER_EXIT_STATUS=""

# spawn_job NAME JIT_CONFIG
#   Starts exactly one job and blocks until it exits, setting
#   RUNNER_EXIT_STATUS. Writes PID_FILE with the tracked root pid BEFORE
#   waiting -- not after -- so a concurrent `stop`/`status` call can find a
#   job as soon as it exists, and so a supervisor that dies mid-wait still
#   leaves a discoverable PID_FILE for the next `run`'s startup reap.
spawn_job() {
  local name="$1" jit_config="$2"
  local jit_file="${STATE_DIR}/jitconfig.$$"
  local umask_prev
  umask_prev="$(umask)"
  umask 177
  printf '%s' "${jit_config}" > "${jit_file}"
  umask "${umask_prev}"

  local timeout_bin job_log
  timeout_bin="$(resolve_timeout_bin)"
  job_log="${LOG_DIR}/runner-$(date -u +%Y%m%dT%H%M%SZ).log"
  local child_pid

  if [ "${RUNNER_SU_ENABLED}" = "1" ]; then
    chown "${RUNNER_USER}" "${jit_file}" 2>/dev/null \
      || fail "cannot chown JIT config to ${RUNNER_USER} (supervisor must run as root in production)"
    # su - gives a clean login environment for the runner and drops every
    # supervisor privilege/env var -- and, per the header above, is exactly
    # what puts the runner in a new session that lib/pidtree.sh's pid-chain
    # walk (not process-group signaling) is designed to still reach.
    # rm -f run-helper.sh before `cp -f` inside run.sh: `cp -f` onto an
    # existing file keeps its old (possibly non-executable) mode.
    # shellcheck disable=SC2086  # timeout_bin is intentionally "cmd N" or empty
    su - "${RUNNER_USER}" -c "umask 022; cd '${RUNNER_DIR}' && rm -f run-helper.sh && export ACTIONS_RUNNER_INPUT_JITCONFIG=\"\$(cat '${jit_file}')\" && rm -f '${jit_file}' && ${timeout_bin} ./run.sh" \
      >> "${job_log}" 2>&1 &
    child_pid=$!
  else
    # TEST/DRY-RUN path only -- see RUNNER_SU_ENABLED above.
    (
      cd "${RUNNER_DIR}" || exit 90
      export ACTIONS_RUNNER_INPUT_JITCONFIG
      ACTIONS_RUNNER_INPUT_JITCONFIG="$(cat "${jit_file}")"
      rm -f "${jit_file}"
      # shellcheck disable=SC2086  # timeout_bin is intentionally "cmd N" or empty
      exec ${timeout_bin} ./run.sh
    ) >> "${job_log}" 2>&1 &
    child_pid=$!
  fi

  printf '%s\n' "${child_pid}" > "${PID_FILE}"
  log "spawned ${name} pid=${child_pid} log=${job_log}"

  wait "${child_pid}"
  RUNNER_EXIT_STATUS=$?
  rm -f "${jit_file}"
  # status 124 = idle recycle timeout (no job arrived): the JIT registration
  # is discarded and a fresh one is minted next iteration. Any other status
  # still goes through reap_between_jobs before the next job.
  log "runner exited status=${RUNNER_EXIT_STATUS} name=${name} log=${job_log}"
}

# reap_between_jobs -- defensive kill-tree call for right after a job's
# tracked root pid has been wait(2)'d, so `status` is guaranteed idle before
# the next mint. In the ordinary case (a job that cleaned up after itself)
# this finds nothing alive and is a no-op; it exists for the narrow window
# in which the root has just exited but a background process it forked is
# still alive. Once the root itself is gone, pidtree_collect can no longer
# find that straggler by pid-chain (its ppid link to a since-reaped parent
# no longer resolves) -- that is a known, documented limitation (see
# README.md "Known limitation"). The mechanism this script GUARANTEES is the
# box-3 case itself: `stop`, called (by an operator, or by this loop's own
# startup self-heal) while the tracked root is still alive, always finds and
# kills its whole tree, orphaned session or not.
reap_between_jobs() {
  local pid
  pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
  if [ -n "${pid}" ]; then
    pidtree_kill_tree "${pid}" "${STOP_TERM_TIMEOUT_SECONDS}" \
      || alert "reap: tree rooted at pid ${pid} did not fully die"
  fi
  rm -f "${PID_FILE}"
}

# --- stop / status (standalone entrypoints) -------------------------------------
cmd_stop() {
  if [ ! -f "${PID_FILE}" ]; then
    echo "box-runner: idle (no ${PID_FILE})"
    return 0
  fi
  local pid before after
  pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
  if [ -z "${pid}" ] || ! pidtree_alive "${pid}"; then
    echo "box-runner: idle (stale ${PID_FILE}; pid ${pid:-<empty>} not alive); clearing"
    rm -f "${PID_FILE}"
    return 0
  fi
  before="$(pidtree_collect "${pid}" | wc -l | tr -d ' ')"
  log "stop: killing tree rooted at pid ${pid} (${before} process(es))"
  if pidtree_kill_tree "${pid}" "${STOP_TERM_TIMEOUT_SECONDS}"; then
    rm -f "${PID_FILE}"
    log "stop: tree rooted at ${pid} fully stopped (${before} process(es))"
    echo "box-runner: stopped (${before} process(es) killed)"
    return 0
  fi
  after="$(pidtree_collect "${pid}" | wc -l | tr -d ' ')"
  alert "stop: tree rooted at ${pid} did not fully die (${after} process(es) survived SIGKILL)"
  echo "box-runner: FAILED to fully stop tree at pid ${pid} (${after} survivor(s)); see ${LOG_DIR}/supervisor.log" >&2
  return 1
}

cmd_status() {
  if [ ! -f "${PID_FILE}" ]; then
    echo "idle"
    return 0
  fi
  local pid
  pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
  if [ -z "${pid}" ] || ! pidtree_alive "${pid}"; then
    echo "idle (stale pidfile ${PID_FILE})"
    return 0
  fi
  echo "running: tree rooted at pid ${pid}"
  pidtree_describe "${pid}"
}

on_term() {
  log "received TERM/INT; stopping current job tree and exiting"
  cmd_stop || true
  exit 143
}

# --- run (main loop) -------------------------------------------------------------
cmd_run() {
  require_config

  # Self-heal: a previous `run` may have died without reaching
  # reap_between_jobs (crash, SIGKILL, `launchctl bootout` hitting only THIS
  # process and not its su-detached job tree -- the box-3 bug quoted above).
  # If PID_FILE names a still-alive tree, stop it before minting a new job:
  # this is what turns "restart the daemon" into "actually recover" instead
  # of leaking a second, concurrent job tree.
  if [ -f "${PID_FILE}" ]; then
    log "startup: found ${PID_FILE} from a previous run; reaping before minting a new job"
    cmd_stop || alert "startup reap did not fully succeed; continuing anyway"
  fi

  trap on_term TERM INT

  log "starting; runner_user=${RUNNER_USER} runner_dir=${RUNNER_DIR} labels=${RUNNER_LABELS} su_enabled=${RUNNER_SU_ENABLED}"

  local iteration=0
  while :; do
    iteration=$((iteration + 1))
    if [ "${BOX_RUNNER_MAX_ITERATIONS}" != "0" ] && [ "${iteration}" -gt "${BOX_RUNNER_MAX_ITERATIONS}" ]; then
      log "reached BOX_RUNNER_MAX_ITERATIONS=${BOX_RUNNER_MAX_ITERATIONS} (test/dry-run bound); stopping"
      break
    fi

    local runner_name jit_config
    runner_name="${RUNNER_NAME_PREFIX}-$(date -u +%Y%m%d%H%M%S)-$$"
    if ! jit_config="$(mint_jit_config "${runner_name}")" || [ -z "${jit_config}" ]; then
      alert "JIT config mint failed (iteration ${iteration}); pausing 300s"
      sleep 300
      continue
    fi
    log "minted JIT config for ${runner_name} (iteration ${iteration})"

    spawn_job "${runner_name}" "${jit_config}"
    reap_between_jobs

    sleep "${PAUSE_BETWEEN_JOBS}"
  done
}

# --- dispatch --------------------------------------------------------------------
case "${1:-run}" in
  run)    cmd_run ;;
  stop)   cmd_stop ;;
  status) cmd_status ;;
  *) echo "usage: supervisor.sh {run|stop|status}" >&2; exit 64 ;;
esac
