#!/usr/bin/env bash
# install.sh -- idempotent box installation for tools/box-runner/.
#
# Does exactly four things, every one of them safe to re-run:
#   1. Download and CHECKSUM-VERIFY the actions-runner tarball, pinned by
#      exact version + per-platform sha256 below -- refuses to extract
#      anything whose downloaded bytes do not match the pin.
#   2. Create the directories supervisor.sh needs (RUNNER_DIR, LOG_DIR,
#      STATE_DIR) and, if running as root, hand RUNNER_DIR to RUNNER_USER.
#   3. Seed box-runner.env from box-runner.env.example WITHOUT overwriting an
#      existing one.
#   4. Render com.qwen38.box-runner.plist.template into a ready-to-copy
#      plist. This script NEVER calls `launchctl` itself -- it prints the
#      exact bootstrap command for an operator to run by hand. Installing
#      the runner binary and installing a running daemon are deliberately
#      two different, separately-reviewable actions.
#
# Nothing here touches GitHub App credentials: gh-app-mint.sh's GH_APP_ID /
# GH_APP_KEY / GH_RUNNER_REPO are supervisor/gh-app-mint's business, set via
# env or BOX_RUNNER_APP_ENV, never written or read by this script.
#
# Usage:
#   tools/box-runner/install.sh
#   RUNNER_DIR=/custom/path RUNNER_USER=myuser tools/box-runner/install.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null && pwd -P)"

fail() { echo "install.sh: $*" >&2; exit 78; }
warn() { echo "install.sh: WARNING: $*" >&2; }
info() { echo "install.sh: $*"; }

# --- Pinned actions-runner release ------------------------------------------
# Pinned 2026-08-24 from `gh api repos/actions/runner/releases/latest`
# (tag_name v2.336.0); sha256 values copied verbatim from that release's
# published body (the "BEGIN SHA <platform>" markers GitHub renders into the
# release notes), not recomputed locally -- re-verify against
# https://github.com/actions/runner/releases/tag/v2.336.0 before trusting
# this pin for a real install, and bump both the version and every hash
# together when upgrading; a stale hash for a new version is a refusal, not
# a silent extract of unverified bytes (see download_and_verify below).
ACTIONS_RUNNER_VERSION="${ACTIONS_RUNNER_VERSION:-2.336.0}"

pinned_sha256() {
  case "$1" in
    osx-arm64) printf '%s' "8e8839c49b7060b6b2154f4931f815df330c27f167d53ef2239ee3dfce28b079" ;;
    osx-x64)   printf '%s' "f79c43232761ca495fc18df550bb2865aa99984b37c173c0aa1f8c09d0d548fe" ;;
    *)         return 1 ;;
  esac
}

# platform_tag -- the actions-runner release asset's platform suffix for
# this host. The ranked box is Apple Silicon (osx-arm64); osx-x64 stays
# pinned above for a local Intel Mac dev/test box but is not the ranked
# target. Anything else (Linux, Windows) refuses: this installer is
# macOS-only by design, matching the ranked M5 hardware.
platform_tag() {
  local os arch
  os="$(uname -s)"
  arch="$(uname -m)"
  [ "${os}" = "Darwin" ] || fail "unsupported OS '${os}' -- this installer targets macOS only (the ranked box is an Apple Silicon Mac)"
  case "${arch}" in
    arm64) printf 'osx-arm64' ;;
    x86_64) printf 'osx-x64' ;;
    *) fail "unsupported architecture '${arch}'" ;;
  esac
}

compute_sha256() {
  local file="$1"
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "${file}" | awk '{print $1}'
  elif command -v sha256sum >/dev/null 2>&1; then
    sha256sum "${file}" | awk '{print $1}'
  else
    fail "neither shasum nor sha256sum is on PATH; cannot verify the downloaded tarball"
  fi
}

# --- Configuration -----------------------------------------------------------
RUNNER_USER="${RUNNER_USER:-runner}"
RUNNER_DIR="${RUNNER_DIR:-/Users/runner/actions-runner}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/log}"
STATE_DIR="${STATE_DIR:-${SCRIPT_DIR}/state}"

IS_ROOT=0
[ "$(id -u)" -eq 0 ] && IS_ROOT=1

# --- Step 1: actions-runner install, checksum-pinned and idempotent --------
install_actions_runner() {
  local tag version_marker
  tag="$(platform_tag)"
  version_marker="${RUNNER_DIR}/.box-runner-installed-version"

  if [ -x "${RUNNER_DIR}/run.sh" ] && [ -f "${version_marker}" ] \
     && [ "$(cat "${version_marker}" 2>/dev/null)" = "${ACTIONS_RUNNER_VERSION}" ]; then
    info "actions-runner ${ACTIONS_RUNNER_VERSION} already installed at ${RUNNER_DIR} (idempotent no-op)"
    return 0
  fi

  local want_sha
  want_sha="$(pinned_sha256 "${tag}")" \
    || fail "no pinned sha256 for platform '${tag}' -- refusing to download an unverifiable tarball; add a pin above first"

  local asset url tmp_dir tarball got_sha
  asset="actions-runner-${tag}-${ACTIONS_RUNNER_VERSION}.tar.gz"
  url="https://github.com/actions/runner/releases/download/v${ACTIONS_RUNNER_VERSION}/${asset}"
  tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/box-runner-install.XXXXXX")"
  # The handler clears its own trap. A bash RETURN trap is GLOBAL, not
  # function-scoped: left installed, this one survives past
  # install_actions_runner and re-fires when main() returns, by which point
  # `tmp_dir` -- a local of this function -- no longer exists, and under
  # `set -euo pipefail` that unbound expansion kills the script. That is
  # exactly what a first real install hit: every install action succeeded,
  # then install.sh exited 1 on the way out (re-runs took the idempotent
  # `return 0` above, before this trap is ever set, so they exited 0 and hid
  # it). Clearing the trap inside the handler keeps the cleanup running
  # exactly once, on this function's return, and leaves nothing behind.
  trap 'rm -rf "${tmp_dir}"; trap - RETURN' RETURN
  tarball="${tmp_dir}/${asset}"

  info "downloading ${url}"
  curl -fsSL --max-time 300 -o "${tarball}" "${url}" \
    || fail "download failed: ${url}"

  got_sha="$(compute_sha256 "${tarball}")"
  [ "${got_sha}" = "${want_sha}" ] \
    || fail "checksum mismatch for ${asset}: got ${got_sha}, want ${want_sha} -- refusing to extract. Do NOT relax this check; re-pin deliberately if the release truly changed."
  info "checksum verified (sha256 ${got_sha})"

  mkdir -p "${RUNNER_DIR}"
  tar -xzf "${tarball}" -C "${RUNNER_DIR}"
  printf '%s' "${ACTIONS_RUNNER_VERSION}" > "${version_marker}"
  info "extracted actions-runner ${ACTIONS_RUNNER_VERSION} to ${RUNNER_DIR}"

  if [ "${IS_ROOT}" -eq 1 ]; then
    if id "${RUNNER_USER}" >/dev/null 2>&1; then
      chown -R "${RUNNER_USER}" "${RUNNER_DIR}"
      info "chowned ${RUNNER_DIR} to ${RUNNER_USER}"
    else
      fail "running as root but RUNNER_USER '${RUNNER_USER}' does not exist -- create the account first (out of scope for this script), then re-run install.sh"
    fi
  else
    warn "not running as root: left ${RUNNER_DIR} owned by $(id -un). Production installs run as root so this step can chown it to RUNNER_USER (${RUNNER_USER}); re-run with sudo before going live."
  fi
}

# --- Step 2: directories ------------------------------------------------------
ensure_dirs() {
  mkdir -p "${LOG_DIR}" "${STATE_DIR}"
  info "log dir:   ${LOG_DIR}"
  info "state dir: ${STATE_DIR}"
}

# --- Step 3: config file, seeded once, never overwritten --------------------
ensure_env_file() {
  local target="${SCRIPT_DIR}/box-runner.env"
  if [ -f "${target}" ]; then
    info "box-runner.env already exists at ${target} (not overwritten)"
  else
    cp "${SCRIPT_DIR}/box-runner.env.example" "${target}"
    info "seeded ${target} from box-runner.env.example -- edit RUNNER_LABELS before starting the daemon (see the file's own comments)"
  fi
}

# --- Step 4: launchd plist, rendered but NOT loaded ---------------------------
render_plist() {
  local template="${SCRIPT_DIR}/com.qwen38.box-runner.plist.template"
  local out="${SCRIPT_DIR}/com.qwen38.box-runner.plist"
  [ -f "${template}" ] || fail "missing plist template: ${template}"

  sed \
    -e "s#__SUPERVISOR_PATH__#${SCRIPT_DIR}/supervisor.sh#g" \
    -e "s#__LOG_DIR__#${LOG_DIR}#g" \
    -e "s#__CONFIG_ENV_PATH__#${SCRIPT_DIR}/box-runner.env#g" \
    -e "s#__SCRIPT_DIR__#${SCRIPT_DIR}#g" \
    "${template}" > "${out}"
  info "rendered ${out}"

  cat <<EOF

install.sh does NOT load the daemon. To go live, an operator runs (as root):

  sudo cp "${out}" /Library/LaunchDaemons/com.qwen38.box-runner.plist
  sudo chown root:wheel /Library/LaunchDaemons/com.qwen38.box-runner.plist
  sudo chmod 644 /Library/LaunchDaemons/com.qwen38.box-runner.plist
  sudo launchctl bootstrap system /Library/LaunchDaemons/com.qwen38.box-runner.plist

To take it back down:

  sudo launchctl bootout system/com.qwen38.box-runner

launchctl bootout sends its signal to the DAEMON's own process group only --
it will NOT reach a job tree the daemon spawned via 'su - ${RUNNER_USER}'
(that is the orphan bug supervisor.sh's header documents and lib/pidtree.sh
fixes). Always run '${SCRIPT_DIR}/supervisor.sh stop' immediately before or
after a bootout to reap any in-flight job tree; see README.md "Restore /
uninstall".
EOF
}

main() {
  info "RUNNER_DIR=${RUNNER_DIR} RUNNER_USER=${RUNNER_USER} platform=$(platform_tag) version=${ACTIONS_RUNNER_VERSION}"
  install_actions_runner
  ensure_dirs
  ensure_env_file
  render_plist
  info "done. Edit ${SCRIPT_DIR}/box-runner.env (RUNNER_LABELS is required) and set GH_APP_ID / GH_APP_KEY / GH_RUNNER_REPO before starting the daemon."
}

# Allow test/*.sh to source this file for its pure helper functions
# (platform_tag, pinned_sha256, compute_sha256) without running main() --
# main() downloads real bytes off the network and requires a real macOS
# host, neither of which belongs in a unit test.
if [ "${BOX_RUNNER_INSTALL_SOURCE_ONLY:-0}" != "1" ]; then
  main "$@"
fi
