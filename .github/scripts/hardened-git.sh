#!/usr/bin/env bash
# Run git against an UNTRUSTED worktree with a neutralized configuration so a
# planted .git/config, hooks, filters, fsmonitor, or pager in a submission
# cannot execute as the invoking uid.
#
# Re-implementation of Layr-Labs/qwen-3.8-mtp-challenge@bfab0de
# .github/scripts/hardened-git.sh:1-46, unchanged in behaviour.
#
# The submission-restriction gates read commits and blobs out of the submission
# checkout, whose repo-local .git/config is attacker-influenced on submission
# branches; repo-local settings such as core.fsmonitor / core.hooksPath /
# core.pager can run arbitrary programs as the git-invoking uid. The `-c`
# overrides below take precedence over repo-local config and neutralize those
# vectors; GIT_CONFIG_NOSYSTEM + /dev/null global/system config drop the
# system/global layers; HOME=/var/empty removes any per-user config; and the
# empty base environment (env -i, restoring only PATH) strips inherited GIT_*
# knobs.
set -euo pipefail

if [[ "$#" -eq 0 ]]; then
  echo "usage: hardened-git.sh GIT_ARG [GIT_ARG...]" >&2
  exit 2
fi

git_bin="$(command -v git)"
if [[ -z "${git_bin}" ]]; then
  echo "hardened-git.sh: git not found on PATH" >&2
  exit 1
fi

# safe.directory='*' only bypasses git's cross-uid ownership refusal (env -i +
# HOME=/var/empty dropped the runner gitconfig that normally grants it); it
# enables no config/hook execution -- those vectors are neutralized above it.
exec env -i \
  PATH="${PATH}" \
  GIT_CONFIG_GLOBAL=/dev/null \
  GIT_CONFIG_SYSTEM=/dev/null \
  GIT_CONFIG_NOSYSTEM=1 \
  GIT_TERMINAL_PROMPT=0 \
  HOME=/var/empty \
  "${git_bin}" \
    -c core.fsmonitor=false \
    -c core.hooksPath=/dev/null \
    -c core.pager=cat \
    -c protocol.ext.allow=never \
    -c safe.directory='*' \
    "$@"
