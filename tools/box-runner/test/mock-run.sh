#!/usr/bin/env bash
# mock-run.sh -- stand-in for actions-runner's run.sh, used ONLY by
# test-box-runner.sh's orphan-kill regression test. supervisor.sh's
# RUNNER_DIR points at this file's directory in test mode
# (RUNNER_SU_ENABLED=0), so spawn_job execs this instead of a real runner.
#
# On start: records its own pid to <state-dir>/mock.pid, then spawns a
# DETACHED grandchild (mock-orphan-child.sh, backgrounded and disowned,
# under `setsid` when available) that records ITS OWN pid to
# <state-dir>/orphan.pid before sleeping. That grandchild plays the role of
# whatever a real job's `su - runner -c '... ./run.sh'` subtree can leave
# behind: something the supervisor's tracked root spawned that would NOT be
# reaped by naively killing only the pid supervisor.sh remembered, or by a
# process-group signal that never reaches a re-sessioned descendant (the
# box-3 orphan bug -- see supervisor.sh's and lib/pidtree.sh's headers).
# mock-run.sh itself then sleeps, standing in for the actions-runner
# listening for a job.
#
# Usage: mock-run.sh [state-dir]
#   The state dir may also be given via the MOCK_STATE_DIR env var (checked
#   first) -- supervisor.sh execs this script with no arguments, exactly as
#   it execs the real run.sh, so the end-to-end supervisor.sh test exports
#   MOCK_STATE_DIR into supervisor.sh's own environment before starting it
#   and relies on that env var surviving into the spawned job's environment.
set -u
SELF_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null && pwd -P)"
STATE_DIR="${MOCK_STATE_DIR:-${1:?usage: mock-run.sh <state-dir> (or set MOCK_STATE_DIR)}}"
mkdir -p "${STATE_DIR}"
echo "$$" > "${STATE_DIR}/mock.pid"

if command -v setsid >/dev/null 2>&1; then
  setsid "${SELF_DIR}/mock-orphan-child.sh" "${STATE_DIR}" \
    > "${STATE_DIR}/orphan.log" 2>&1 < /dev/null &
else
  "${SELF_DIR}/mock-orphan-child.sh" "${STATE_DIR}" \
    > "${STATE_DIR}/orphan.log" 2>&1 < /dev/null &
fi
disown 2>/dev/null || true

exec sleep 300
