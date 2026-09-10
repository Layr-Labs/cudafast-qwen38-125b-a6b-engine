#!/usr/bin/env bash
# mock-orphan-child.sh -- helper spawned (via setsid, when available) by
# mock-run.sh to stand in for a detached grandchild an in-progress job left
# running. See mock-run.sh and test-box-runner.sh for how this is used.
set -u
STATE_DIR="${1:?usage: mock-orphan-child.sh <state-dir>}"
echo "$$" > "${STATE_DIR}/orphan.pid"
exec sleep 300
