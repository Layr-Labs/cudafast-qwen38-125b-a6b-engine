#!/usr/bin/env bash
# qwen4exp-g1-halt.sh -- stop a qwen4exp-g1-boot.sh run from outside it.
#
# A run holds the GPU lock and, once the boot fires, about 80 GiB of host
# memory. To kill the driver alone is not enough. What matters is ds4, and ds4
# is a grandchild: driver -> /usr/bin/time -> ds4. This script stops the whole
# tree two ways, then proves that nothing is left.
#
#   1. The ppid walk, tools/box-runner/lib/pidtree.sh. It follows the live
#      parent-pid links, which the kernel keeps whatever a descendant does to
#      its session or its process group. It therefore reaches a re-sessioned
#      child that a signal to the process group would miss. The library is in
#      this repo for that failure, so the script uses it instead of writing the
#      walk again.
#   2. The process group, as the backstop. qwen4exp-g1-boot.sh re-execs itself
#      under setsid, so the driver, ds4 and the samplers share one process group
#      that holds nothing else on the box. The group is still safe to kill after
#      the driver itself has died and the ppid links are gone.
#
# The script refuses to signal a process group it cannot show to be a run. It
# never signals its own group.
#
# usage: qwen4exp-g1-halt.sh RUN_DIR [--timeout SECONDS]
#
# exit 0  the run is gone, or it was never running
# exit 1  a process of the tree survived SIGKILL
# exit 2  the directory does not name a run, or the group is not a run group
set -euo pipefail

TIMEOUT=10
RUN_DIR=""
while [ $# -gt 0 ]; do
  case "$1" in
    --timeout) TIMEOUT="$2"; shift 2 ;;
    -h|--help) sed -n '2,28p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) RUN_DIR="$1"; shift ;;
  esac
done

die() { printf 'qwen4exp-g1-halt: %s\n' "$*" >&2; exit 2; }

[ -n "${RUN_DIR}" ] || die "usage: qwen4exp-g1-halt.sh RUN_DIR"
[ -d "${RUN_DIR}" ] || die "${RUN_DIR} is not a directory"

PIDFILE="${RUN_DIR%/}/run.pid"
PGIDFILE="${RUN_DIR%/}/run.pgid"
[ -r "${PIDFILE}" ]  || die "${RUN_DIR} carries no run.pid"
[ -r "${PGIDFILE}" ] || die "${RUN_DIR} carries no run.pgid; the driver did not record its process group"

ROOT="$(tr -d '[:space:]' < "${PIDFILE}")"
PGID="$(tr -d '[:space:]' < "${PGIDFILE}")"
case "${ROOT}"  in ''|*[!0-9]*) die "run.pid does not hold a pid" ;; esac
case "${PGID}"  in ''|*[!0-9]*) die "run.pgid does not hold a process group id" ;; esac

# Never signal the group this script runs in.
SELF_PGID="$(ps -o pgid= -p $$ 2>/dev/null | tr -d '[:space:]' || true)"
[ "${PGID}" != "${SELF_PGID}" ] \
  || die "run.pgid ${PGID} is this script's own process group; refusing to signal it"

members() { ps -o pid= -g "${PGID}" 2>/dev/null | tr -d ' ' | grep -v '^$' || true; }

if [ -z "$(members)" ]; then
  printf 'qwen4exp-g1-halt: run %s is already stopped; no process of group %s is alive\n' "${ROOT}" "${PGID}"
  exit 0
fi

# The group must be shown to be a run. If the leader lives, its command line
# names the driver. If the leader is gone, every survivor names the run
# directory. Anything else is somebody's reused pgid, and the script stops.
leader_args="$(ps -o args= -p "${PGID}" 2>/dev/null || true)"
if [ -n "${leader_args}" ]; then
  case "${leader_args}" in
    *qwen4exp-g1-boot.sh*) : ;;
    *) die "process group ${PGID} is led by '${leader_args}', which is not qwen4exp-g1-boot.sh; refusing to signal it" ;;
  esac
else
  while IFS= read -r pid; do
    [ -n "${pid}" ] || continue
    args="$(ps -o args= -p "${pid}" 2>/dev/null || true)"
    case "${args}" in
      *"${RUN_DIR%/}"*|*qwen4exp-g1-boot.sh*) : ;;
      *) die "process ${pid} in group ${PGID} is '${args}', which names neither the run directory nor the driver; refusing to signal the group" ;;
    esac
  done < <(members)
fi

LIB="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null && pwd -P)/box-runner/lib/pidtree.sh"
if [ -r "${LIB}" ]; then
  # shellcheck source=/dev/null
  . "${LIB}"
else
  printf 'qwen4exp-g1-halt: pidtree.sh is missing at %s; the process group is the only stop\n' "${LIB}" >&2
fi

printf 'qwen4exp-g1-halt: run %s, driver pid %s, pgid %s\n' "${RUN_DIR}" "${ROOT}" "${PGID}"

if command -v pidtree_describe >/dev/null 2>&1; then
  printf 'qwen4exp-g1-halt: tree before\n'
  pidtree_describe "${ROOT}" || true
  pidtree_kill_tree "${ROOT}" "${TIMEOUT}" || true
fi

kill -TERM -- "-${PGID}" 2>/dev/null || true
waited=0
while [ "${waited}" -lt "$(( TIMEOUT * 5 ))" ]; do
  [ -n "$(members)" ] || break
  sleep 0.2
  waited=$(( waited + 1 ))
done
kill -KILL -- "-${PGID}" 2>/dev/null || true
sleep 0.3

survivors="$(members | wc -l | tr -d '[:space:]')"
if [ "${survivors}" -gt 0 ]; then
  printf 'qwen4exp-g1-halt: FAILED -- %s process(es) of group %s survived SIGKILL\n' "${survivors}" "${PGID}" >&2
  exit 1
fi
printf 'qwen4exp-g1-halt: halted -- no process of run %s is alive\n' "${ROOT}"
exit 0
