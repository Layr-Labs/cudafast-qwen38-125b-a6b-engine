#!/usr/bin/env bash
# mock-gh-app-mint.sh -- stub for GH_APP_MINT used by test-box-runner.sh.
# Real gh-app-mint.sh talks to the GitHub API and needs a real App key; this
# stub proves out supervisor.sh's spawn/stop/status/kill-tree logic without
# any network access or credential. It only implements enough of the real
# script's interface (the `jitconfig <name> <labels>` case) for that.
set -u
case "${1:-}" in
  jitconfig) printf 'FAKE-JIT-CONFIG-for-%s-labels-%s' "${2:-}" "${3:-}" ;;
  *) echo "mock-gh-app-mint: unsupported subcommand '${1:-}'" >&2; exit 64 ;;
esac
