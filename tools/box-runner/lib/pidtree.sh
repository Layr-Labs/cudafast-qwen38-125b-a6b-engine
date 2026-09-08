#!/usr/bin/env bash
# lib/pidtree.sh -- process-tree tracking and kill-tree primitives.
#
# THE ORPHAN BUG THIS FILE EXISTS TO FIX (box-3-proven; see the retired qwen
# supervisor in the operator's machine-scripts repository, supervisor.sh:158,
# read-only reference for this repo). That script spawned the ephemeral
# actions-runner with:
#
#   su - "${RUNNER_USER}" -c '... timeout ./run.sh'
#
# `su -` starts a NEW LOGIN SESSION: the runner process (and anything it goes
# on to fork -- including a detached grandchild, which is exactly the shape a
# hung or misbehaving job step can leave behind) lands in a session/process
# group that is DIFFERENT from the supervisor daemon's own. When launchd
# stops the daemon (crash, `launchctl bootout`, a manual kickstart -k), the
# signal macOS delivers targets the DAEMON'S process group -- and a
# `su -`-spawned session was never a member of it, so it is not reached
# EVEN THOUGH the daemon and the runner subtree were both still alive at that
# instant. The subtree survives as a live orphan across restarts. This is not
# a hypothetical: it is the box-3-proven failure mode named in this repo's
# task brief.
#
# THE FIX. Do not rely on process-group signaling at all -- it is exactly the
# mechanism that misses a re-sessioned descendant. Instead, track the actual
# PID of whatever this script spawns as job "root", and when it is time to
# stop, walk the LIVE PARENT-PID CHAIN from that root (ps/pgrep ppid links,
# which the kernel maintains independently of session id or process group,
# and which are unaffected by `su -`'s session boundary) to find every
# still-alive descendant, then signal each one explicitly. This works exactly
# when it needs to: called BETWEEN jobs, while the root (the `su`/runner
# process) is still alive, so nothing has had a chance to be orphaned onto
# pid 1 yet -- the walk finds the whole tree before any of it can escape.
#
# Persisting the root pid to a file (done by the caller, not this library) is
# what lets `supervisor.sh stop` work as a STANDALONE entrypoint even after
# the supervisor loop itself has died ungracefully (a SIGKILL cannot be
# trapped, so the daemon cannot always clean up after itself): a later,
# separate invocation reads the pid from disk and performs the same live
# ppid-chain walk, unaffected by the fact that the process that originally
# recorded it is gone.
#
# Every function in this file is read-only about WHERE state lives; it only
# ever takes a pid on its command line and returns/acts on live process
# state. No paths, no env vars, no secrets.
set -u

# pidtree_children PID
#   Print the pid of every LIVE direct child of PID, one per line. Silent
#   (empty output, exit 0) if PID has no children or does not exist.
pidtree_children() {
  pgrep -P "$1" 2>/dev/null || true
}

# pidtree_alive PID
#   True (exit 0) iff PID currently exists and is signalable by this process.
pidtree_alive() {
  kill -0 "$1" 2>/dev/null
}

# pidtree_collect ROOT_PID
#   Breadth-first walk of the live ppid tree rooted at ROOT_PID. Prints one
#   pid per line: ROOT_PID first (if alive), then every live descendant,
#   deduplicated. Prints nothing if ROOT_PID is not alive. This is the
#   session/process-group-independent walk described above -- it finds a
#   `su -`-detached or otherwise re-sessioned descendant exactly as reliably
#   as an ordinary child, because it never consults session id or pgid.
pidtree_collect() {
  local root="$1"
  pidtree_alive "${root}" || return 0

  # Plain indexed arrays only (bash 3.2 compatible -- macOS ships 3.2 as
  # /bin/bash and this library must run there unmodified).
  local -a queue=("${root}")
  local -a seen=()
  local -a out=()
  local pid child already idx

  while [ "${#queue[@]}" -gt 0 ]; do
    pid="${queue[0]}"
    queue=("${queue[@]:1}")

    already=0
    for idx in "${seen[@]:-}"; do
      if [ "${idx}" = "${pid}" ]; then
        already=1
        break
      fi
    done
    [ "${already}" -eq 1 ] && continue
    seen+=("${pid}")

    pidtree_alive "${pid}" || continue
    out+=("${pid}")

    while IFS= read -r child; do
      [ -n "${child}" ] && queue+=("${child}")
    done < <(pidtree_children "${pid}")
  done

  if [ "${#out[@]}" -gt 0 ]; then
    printf '%s\n' "${out[@]}"
  fi
}

# pidtree_describe ROOT_PID
#   Human-readable listing of every live pid in ROOT_PID's tree, for the
#   `status` entrypoint. One `ps` line per process; prints nothing (exit 0)
#   if the tree is empty (the between-jobs quiescent state).
pidtree_describe() {
  local root="$1" pid
  local -a pids=()
  while IFS= read -r pid; do
    [ -n "${pid}" ] && pids+=("${pid}")
  done < <(pidtree_collect "${root}")
  [ "${#pids[@]}" -eq 0 ] && return 0
  # -o field set is BSD/macOS-and-Linux-compatible; one `ps -p` call per pid
  # keeps this portable (GNU ps and BSD ps disagree on comma-vs-space -p
  # lists in some versions).
  for pid in "${pids[@]}"; do
    ps -o pid,ppid,pgid,stat,command -p "${pid}" 2>/dev/null | tail -n +2
  done
}

# pidtree_signal SIGNAL PID...
#   Best-effort: send SIGNAL to every listed pid. Never fails the caller --
#   a pid that already exited between collection and signaling is not an
#   error.
pidtree_signal() {
  local sig="$1"; shift
  local pid
  for pid in "$@"; do
    kill -s "${sig}" "${pid}" 2>/dev/null || true
  done
}

# pidtree_kill_tree ROOT_PID [TERM_TIMEOUT_SECONDS]
#   The `stop` primitive. Collects ROOT_PID's live tree, sends TERM to all of
#   it, polls (0.2s steps) for up to TERM_TIMEOUT_SECONDS (default 10) for
#   the tree to disappear, then re-collects (a process can fork a grandchild
#   in the grace window) and sends KILL to whatever is still alive. Prints
#   nothing; returns 0 if the tree is fully gone afterward, 1 if something
#   survived KILL (e.g. a zombie awaiting reap by an unrelated parent, or a
#   pid this uid cannot signal).
pidtree_kill_tree() {
  local root="$1" timeout="${2:-10}"
  local -a tree=()
  local pid waited

  while IFS= read -r pid; do
    [ -n "${pid}" ] && tree+=("${pid}")
  done < <(pidtree_collect "${root}")

  [ "${#tree[@]}" -eq 0 ] && return 0

  pidtree_signal TERM "${tree[@]}"

  waited=0
  while [ "${waited}" -lt "$((timeout * 5))" ]; do
    tree=()
    while IFS= read -r pid; do
      [ -n "${pid}" ] && tree+=("${pid}")
    done < <(pidtree_collect "${root}")
    [ "${#tree[@]}" -eq 0 ] && return 0
    sleep 0.2
    waited=$((waited + 1))
  done

  # Re-collect one last time (catches anything forked during the grace
  # window) and escalate to KILL.
  tree=()
  while IFS= read -r pid; do
    [ -n "${pid}" ] && tree+=("${pid}")
  done < <(pidtree_collect "${root}")
  [ "${#tree[@]}" -eq 0 ] && return 0
  pidtree_signal KILL "${tree[@]}"

  # Final settle check.
  sleep 0.3
  pidtree_collect "${root}" | grep -q . && return 1
  return 0
}
