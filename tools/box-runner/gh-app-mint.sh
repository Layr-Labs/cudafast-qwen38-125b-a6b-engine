#!/usr/bin/env bash
# gh-app-mint.sh -- mint short-lived GitHub App credentials for JIT
# self-hosted-runner registration. NO PAT: the only durable secret this
# script ever touches is the GitHub App private key, and only a PATH to it
# (GH_APP_KEY) -- never key material -- lives in this file or any env var it
# sets. Everything this script emits is short-lived (installation token,
# ~1h, and explicitly REVOKED by this script once used) or single-use (the
# JIT runner config it prints).
#
# Adapted from the box-3-proven qwen/bench lineage
# (the operator's machine-scripts repository, read-only reference -- this
# script does not modify that repo) for tools/box-runner/. Two load-bearing
# changes from that lineage:
#
#   1. FAIL-CLOSED, NO BAKED-IN IDENTITY. The original carried literal
#      fallback defaults for GH_APP_ID / GH_REPO / GH_INSTALLATION_ID, so a
#      missing config file silently degraded to a hardcoded box-3 identity --
#      its own header documents that this is exactly how a different box
#      once kept registering against the wrong repository after a
#      promotion, because editing the config file alone had no effect while
#      the literal fallback was silently winning. This script refuses to run
#      at all, with a named error, if GH_APP_ID, GH_APP_KEY, or
#      GH_RUNNER_REPO is unset: there is no repository or identity this
#      script will register against by accident or by copied default.
#   2. GH_RUNNER_REPO, not a hardcoded GH_REPO. The public repository name
#      for this track is not settled yet (see tools/box-runner/README.md's
#      coupling note); every caller (supervisor.sh, install.sh, an operator)
#      must supply it via env, and this script has no opinion about it.
#
# Usage:
#   gh-app-mint.sh installation-id            -> prints the installation id
#   gh-app-mint.sh token                      -> prints a 1h installation token
#   gh-app-mint.sh jitconfig <runner-name> <labels-csv>
#                                              -> prints base64 JIT config (single use)
#
# Required env (fail-closed; none of these three has a default):
#   GH_APP_ID           GitHub App id.
#   GH_APP_KEY          Path to the App's RS256 private key. Only the path
#                        lives here -- the file's permissions and custody are
#                        the operator's responsibility (install.sh never
#                        writes or downloads this file).
#   GH_RUNNER_REPO       "owner/repo" this runner registers against.
#
# Optional env:
#   GH_INSTALLATION_ID   Skip the /repos/{repo}/installation lookup if the
#                        installation id is already known.
#   GH_RUNNER_GROUP       Runner group id for jitconfig (default: 1, GitHub's
#                        default group -- not a secret).
#   GH_WORK_FOLDER        work_folder for generate-jitconfig (default under
#                        /Users/Shared, namespaced to this track so it never
#                        collides with an unrelated runner's job root).
#   BOX_RUNNER_APP_ENV    Optional env file sourced BEFORE the required vars
#                        above are read, so a box can pin its own values
#                        without editing this script (mirrors the original
#                        lineage's on-box gh-app.env precedence,
#                        generalized to any operator-chosen path/name via
#                        install.sh). A missing/unreadable file is not an
#                        error -- it simply means every value above must
#                        already be in the environment.
#
# Credential-exposure hardening carried over unchanged from the box-3
# lineage (see the original for the incident this was hardened against):
#   * The RS256 JWT and the installation token are NEVER placed on a curl
#     command line -- curl_auth() feeds the Authorization header to curl via
#     `--config -` (stdin), so no secret ever appears in this process's argv,
#     `ps` output, or a crash dump.
#   * The installation token minted for jitconfig is SCOPED to exactly this
#     one repository (administration:write + metadata:read), so a stolen
#     jitconfig-path token cannot touch any other repo or hold broader
#     permissions.
#   * That token is REVOKED (DELETE /installation/token) on every exit path
#     (normal return, INT, TERM), shrinking its live window from the ~1h
#     GitHub grants to the few seconds this script actually uses it.
#   * No secret is ever written to disk or a durable env var. No `set -x`,
#     no curl `-v` (either would echo the bearer token).
set -euo pipefail

fail() {
  echo "gh-app-mint: $*" >&2
  exit 78 # EX_CONFIG (sysexits.h) -- a configuration problem, not a runtime one.
}

# Optional box-local overrides, sourced BEFORE the required-var checks below
# so a plain assignment in the file wins over anything already exported (same
# precedence rule as the box-3 lineage). Guarded: a missing/unreadable file
# is not an error, and the `&&` list keeps a false `[ -r ... ]` from tripping
# `set -e`.
# shellcheck disable=SC1090  # path is operator/box-supplied, not knowable here
[ -n "${BOX_RUNNER_APP_ENV:-}" ] && [ -r "${BOX_RUNNER_APP_ENV}" ] && . "${BOX_RUNNER_APP_ENV}"

[ -n "${GH_APP_ID:-}" ]     || fail "GH_APP_ID is required (no default -- refusing to guess an App identity)"
[ -n "${GH_APP_KEY:-}" ]    || fail "GH_APP_KEY is required (path to the App's RS256 private key; no default path)"
[ -f "${GH_APP_KEY}" ]      || fail "GH_APP_KEY does not exist: ${GH_APP_KEY}"
[ -n "${GH_RUNNER_REPO:-}" ] || fail "GH_RUNNER_REPO is required (owner/repo; no default -- see tools/box-runner/README.md)"

# Bare repository name (no owner) for the access_tokens `repositories` scope.
GH_REPO_NAME="${GH_RUNNER_REPO##*/}"
GH_RUNNER_GROUP="${GH_RUNNER_GROUP:-1}"
GH_WORK_FOLDER="${GH_WORK_FOLDER:-/Users/Shared/qwen38-125b-a6b-box-runner-jobs/_work}"
API="https://api.github.com"

b64url() { openssl base64 -A | tr '+/' '-_' | tr -d '='; }

# curl wrapper that supplies the bearer credential on STDIN (never argv). The
# secret goes into a curl --config blob read from a pipe; every other option
# (URL, -X, -d, -H) is non-secret and stays on the command line.
curl_auth() {
  local bearer="$1"; shift
  printf 'header = "Authorization: Bearer %s"\n' "${bearer}" \
    | curl -fsS --max-time 20 --config - "$@"
}

# The internal installation token minted for the jitconfig flow. Held in a
# global so the EXIT trap can revoke it regardless of function scope. Set
# ONLY on the jitconfig path; the standalone `token` subcommand leaves it
# empty so its returned token stays alive for the caller.
MINTED_TOKEN=""

revoke_minted_token() {
  # Best-effort: a revoke failure must never change our exit status or leak
  # the token (output discarded). 204 No Content on success.
  [ -n "${MINTED_TOKEN}" ] && [ "${MINTED_TOKEN}" != "null" ] || return 0
  curl_auth "${MINTED_TOKEN}" -X DELETE \
    -H "Accept: application/vnd.github+json" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    "${API}/installation/token" >/dev/null 2>&1 || true
  MINTED_TOKEN=""
  return 0
}
# Revoke on EVERY termination path: normal/`set -e` exit, and INT/TERM (so a
# killed mint still tears down the token it minted). revoke is idempotent, so
# the double-fire from a signal that also runs EXIT is a harmless no-op.
trap revoke_minted_token EXIT
trap 'revoke_minted_token; exit 130' INT
trap 'revoke_minted_token; exit 143' TERM

make_jwt() {
  local now iat exp header payload signing_input sig
  now="$(date +%s)"
  iat="$((now - 60))"  # backdate 60s for clock skew
  exp="$((now + 540))" # 9 min (<= GitHub's 10 min cap)
  header="$(printf '{"alg":"RS256","typ":"JWT"}' | b64url)"
  payload="$(printf '{"iat":%s,"exp":%s,"iss":"%s"}' "${iat}" "${exp}" "${GH_APP_ID}" | b64url)"
  signing_input="${header}.${payload}"
  sig="$(printf '%s' "${signing_input}" \
    | openssl dgst -sha256 -sign "${GH_APP_KEY}" -binary \
    | b64url)"
  printf '%s.%s' "${signing_input}" "${sig}"
}

api_get() {
  local jwt="$1" path="$2"
  curl_auth "${jwt}" \
    -H "Accept: application/vnd.github+json" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    "${API}${path}"
}

resolve_installation_id() {
  local jwt; jwt="$(make_jwt)"
  api_get "${jwt}" "/repos/${GH_RUNNER_REPO}/installation" | jq -r '.id'
}

# Mint an installation token. With no argument the token carries the full
# installation surface (used by the standalone `token` subcommand). Passing
# "scoped" restricts it to exactly what generate-jitconfig needs on this one
# repo (administration:write + metadata:read), so a stolen jitconfig-path
# token cannot touch other repos or hold broader permissions.
mint_installation_token() {
  local scope="${1:-full}" jwt inst_id body
  jwt="$(make_jwt)"
  inst_id="${GH_INSTALLATION_ID:-$(resolve_installation_id)}"
  [ -n "${inst_id}" ] && [ "${inst_id}" != "null" ] || fail "could not resolve installation id for ${GH_RUNNER_REPO}"
  if [ "${scope}" = "scoped" ]; then
    body="$(printf '{"repositories":["%s"],"permissions":{"administration":"write","metadata":"read"}}' "${GH_REPO_NAME}")"
    curl_auth "${jwt}" -X POST \
      -H "Accept: application/vnd.github+json" \
      -H "X-GitHub-Api-Version: 2022-11-28" \
      -d "${body}" \
      "${API}/app/installations/${inst_id}/access_tokens" \
      | jq -r '.token'
  else
    curl_auth "${jwt}" -X POST \
      -H "Accept: application/vnd.github+json" \
      -H "X-GitHub-Api-Version: 2022-11-28" \
      "${API}/app/installations/${inst_id}/access_tokens" \
      | jq -r '.token'
  fi
}

gen_jitconfig() {
  local name="$1" labels="$2" group_json labels_json
  # Repo- and permission-scoped token; revoked by the EXIT trap once done.
  MINTED_TOKEN="$(mint_installation_token scoped)"
  [ -n "${MINTED_TOKEN}" ] && [ "${MINTED_TOKEN}" != "null" ] || fail "could not mint installation token"
  # JIT-config runners get EXACTLY the labels passed here -- unlike config.sh
  # runners, the API does NOT auto-add self-hosted/OS/arch. Force-include
  # self-hosted so `runs-on: [self-hosted, ...]` can match, then de-dupe.
  labels_json="$(printf '%s' "self-hosted,${labels}" | jq -R 'split(",") | unique_by(ascii_downcase)')"
  # runner_group_id is REQUIRED by generate-jitconfig.
  group_json=",\"runner_group_id\":${GH_RUNNER_GROUP}"
  curl_auth "${MINTED_TOKEN}" -X POST \
    -H "Accept: application/vnd.github+json" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    "${API}/repos/${GH_RUNNER_REPO}/actions/runners/generate-jitconfig" \
    -d "{\"name\":\"${name}\",\"labels\":${labels_json},\"work_folder\":\"${GH_WORK_FOLDER}\"${group_json}}" \
    | jq -r '.encoded_jit_config'
  # MINTED_TOKEN is revoked on exit by revoke_minted_token (covers this
  # success path and any error path above).
}

case "${1:-}" in
  installation-id) resolve_installation_id ;;
  token)           mint_installation_token full ;;
  jitconfig)       gen_jitconfig "${2:?runner name}" "${3:?labels csv}" ;;
  *) echo "usage: gh-app-mint.sh {installation-id|token|jitconfig <name> <labels>}" >&2; exit 64 ;;
esac
