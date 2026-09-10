#!/usr/bin/env bash
#
# fetch-benchd.sh -- resolve the `benchd` measurement harness from its release
# channel.
#
# THE PIN IS REMOVED (David ruling 2026-08-27). benchd used to be frozen here by
# ./benchd.pin ({branch, commit, sha256, bytes}), which made every benchd bug
# fix require an engine-repo pin-advance commit before it could reach a box --
# the depth/echo fix sat merged in the bench repo while the served pin still
# carried the bug. That coupling is gone: this repository names only the CHANNEL
# (the bench repo + branch), and the channel's tip is what runs. Fixing a
# measurement bug is now: merge it bench-side, republish dist, done -- zero
# engine commits.
#
# WHAT REPLACES THE PIN'S GUARANTEES, stated honestly:
#
#   * INTEGRITY (the bytes are what the organizer published): the dist channel
#     publishes `benchd.manifest.json` next to the binary ({branch,
#     source_commit, sha256, bytes}, written by the bench repo's
#     scripts/build-dist.sh). The binary is verified against THAT manifest --
#     fetched from the same organizer-controlled channel -- and nothing
#     unverified is ever installed or returned. What this no longer defends
#     against is the channel itself moving, which is the point: the channel is
#     the organizer's bench repo, participants cannot write to it, and its tip
#     is now the intended source of truth.
#   * PROVENANCE (knowing which benchd measured a run): recorded, not pinned.
#     The resolved {branch, source_commit, sha256} is logged loudly on every
#     resolve and the manifest is installed beside the binary
#     (benchd-bin/benchd.manifest.json), so any run's harness identity can be
#     read off the box afterwards.
#   * SUBMISSION-PROOFNESS: unchanged. This script and the channel constants
#     live under tools/, outside editablePaths, and the manifest linter
#     (FORBIDDEN_EDITABLE) still forbids `benchd.pin`/`benchd-bin` spellings,
#     so a submission can neither redirect the fetch nor resurrect a pin.
#
# THE FILE NAMES FOLLOW THE CHANNEL. The channel publishes `benchd` and
# `benchd.manifest.json`. It published `benchctl` / `benchctl.manifest.json`
# until the bench-side rename, and this script asked for those names. A box that
# still stages the old pair resolves NOTHING: the offline path below looks for
# benchd-bin/benchd by name. Move the converge staging unit to the new names.
#
# WHAT IT DOES, in order:
#   1. If benchd-bin/benchd AND benchd-bin/benchd.manifest.json exist and
#      agree (sha256 + bytes), use them, never touching the network -- the
#      OFFLINE path: the ranked box gets both files placed together. A binary
#      with no manifest, or a pair that disagrees, refuses loudly (a bare
#      unattributable binary is exactly what this script must never run).
#      Set BENCHD_REFRESH=1 to discard the local pair and re-resolve the
#      channel tip.
#   2. Otherwise obtain the PAIR -- from BENCHD_DIST_LOCAL (file/dir, for
#      air-gapped boxes) or by downloading manifest-then-binary from the
#      channel -- verify the binary against its manifest, and install both.
#   3. On every path, check the PLATFORM before the binary is installed or
#      returned. See "THE PLATFORM EXPECTATION" below.
#   4. If the manifest DECLARES a second binary, obtain and verify that too.
#      See "THE CHANNEL CARRIES MORE THAN BENCHD" below.
#
# THE CHANNEL CARRIES MORE THAN BENCHD, and the manifest says so.
#
# The channel used to publish one binary, so the manifest's six top-level
# fields WERE the benchd pin. bench pull request #255 adds a second binary --
# `record-correctness-golden`, the golden AUTHOR -- and keeps those six fields
# describing benchd exactly as they were, adding a `binaries` object:
#
#   "binaries": {
#     "benchd": {"sha256": "<64 hex>", "bytes": <int>},
#     "record-correctness-golden": {"sha256": "<64 hex>", "bytes": <int>}
#   }
#
# Every entry is ONE LINE, which is what keeps the anchored six-field sed above
# blind to the nested `sha256`/`bytes` keys (mlxfast-bench scripts/dist-lib.sh
# states the rule and tests it).
#
# BOTH MANIFEST SHAPES ARE LIVE, so both are handled here:
#
#   * NO `binaries` entry for the recorder (the legacy six-field manifest, and
#     what the qwen3.8-125b-a6b-v1 channel still serves until its republish):
#     benchd only, byte for byte the behaviour this script always had. The
#     recorder is not fetched, not required, and not staged.
#   * AN ENTRY IS PRESENT: the recorder is taken from the SAME platform
#     directory as benchd, verified against THAT entry's sha256 and bytes,
#     and staged beside benchd in benchd-bin/. A declared recorder that is
#     missing or does not verify is refused BY NAME -- a half-obtained channel
#     is a publish or placement defect, and installing benchd alone would
#     leave the caller resolving a recorder this script had already found wrong.
#
# The recorder is NOT returned on stdout. This script's stdout contract is one
# path -- benchd's -- so callers read the recorder from beside it.
#
# THE PLATFORM EXPECTATION, and why this repository states it out loud.
#
# The shared dist channel publishes ONE PAIR PER PLATFORM, in two directories
# that hold IDENTICALLY NAMED files:
#
#   dist/benchd                          aarch64-apple-darwin (the MLX box)
#   dist/benchd.manifest.json
#   dist/linux-aarch64/benchd            aarch64-unknown-linux-gnu (this box)
#   dist/linux-aarch64/benchd.manifest.json
#   dist/linux-x86_64/benchd             x86_64-unknown-linux-gnu (participant x86 workstations)
#   dist/linux-x86_64/benchd.manifest.json
#
# Both manifests carry the SAME six fields, so the two pairs are
# indistinguishable except by `target_triple`. That is the whole hazard: the
# macOS pair passes the branch check, the byte count and the digest, and then
# installs a Mach-O binary this track's LINUX box cannot exec. A Mach-O binary
# on Linux does not fail in a way anybody can read either -- the kernel refuses
# the exec format and the caller sees a bare "cannot execute binary file",
# which reads like a broken download rather than the wrong platform.
#
# So the expectation is NAMED, checked twice, and fails closed:
#
#   * the manifest's `target_triple` must equal the expected triple; and
#   * the binary's own container format (ELF or Mach-O, read from its first
#     four bytes) must match the operating system that triple names.
#
# The second check is not redundant. The first trusts the manifest to describe
# the bytes; the second reads the bytes. A mis-stamped publish fails the
# second one.
#
# The expectation also picks the DIRECTORY, so the wrong pair is not reachable
# by accident. It is still checked after the download, because the directory a
# request went to is not proof of what came back.
#
# NOTHING HERE PINS AN IDENTITY. The Linux pair's sha256 and byte count change
# on every republish; they are read from the manifest beside the binary and are
# never hardcoded.
#
# Prints the absolute path of the verified binary on STDOUT (diagnostics go to
# stderr), so callers can do:  BENCHD="$(./tools/fetch-benchd.sh)"
#
# Env:
#   BENCHD_BRANCH       channel branch. Default: qwen3.8-125b-a6b-v1.
#                       THE BRANCH IS THE PROJECT, NOT THE TRACK. David ruling
#                       2026-08-27: the MLX and CUDA tracks of this model share
#                       ONE bench release branch and ONE dist channel, because
#                       they share one benchmarker. The TRACK id stays
#                       platform-specific (qwen3.8-125b-a6b-cuda-v1) and is what
#                       names the leaderboard namespace, the runner labels and
#                       the R2 prefix. Do not conflate the two: this variable
#                       and the manifest branch check are the only places the
#                       PROJECT name belongs.
#   BENCHD_EXPECT_TARGET_TRIPLE
#                       the Rust target triple this host expects the channel to
#                       have published. Default: derived from `uname -s` and
#                       `uname -m`. Set it only to check a dist for a host other
#                       than this one; an unknown host refuses rather than
#                       guessing.
#   BENCHD_BIN_DIR      install directory. Default: <repo>/benchd-bin
#                       (gitignored; a fetched artifact, not repository content).
#   BENCHD_REFRESH      set to 1 to discard an already-installed pair and
#                       re-resolve the channel tip. A staged
#                       record-correctness-golden is discarded with it.
#   BENCHD_DIST_LOCAL   path to an already-obtained dist (a directory holding
#                       benchd + benchd.manifest.json, or the benchd file
#                       with the manifest beside it). Verified, never trusted
#                       bare. A manifest that declares
#                       record-correctness-golden expects that file in the same
#                       directory.
#   BENCHD_DIST_BASE_URL
#                       raw host + repo prefix. Default:
#                       https://raw.githubusercontent.com/Layr-Labs/mlxfast-bench
#                       -- the public bench repository, which is where this
#                       track's release branch and dist channel live. It is
#                       public, so a fetch needs no token.
#   BENCHD_DIST_TOKEN / GITHUB_TOKEN
#                       bearer token for a private channel repo.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BRANCH="${BENCHD_BRANCH:-qwen3.8-125b-a6b-v1}"
DEST_DIR="${BENCHD_BIN_DIR:-${REPO_ROOT}/benchd-bin}"
DEST="${DEST_DIR}/benchd"
DEST_MANIFEST="${DEST_DIR}/benchd.manifest.json"
# The channel's second binary. Named once, here, because the file name is also
# the manifest's `binaries` key and the name in the same platform directory.
RECORDER_NAME="record-correctness-golden"
DEST_RECORDER="${DEST_DIR}/${RECORDER_NAME}"
# The public bench repository is the default channel. See the
# BENCHD_DIST_BASE_URL note above.
BASE_URL="${BENCHD_DIST_BASE_URL:-https://raw.githubusercontent.com/Layr-Labs/mlxfast-bench}"

die() {
  echo "fetch-benchd.sh: $*" >&2
  exit 1
}

command -v shasum >/dev/null 2>&1 || die "shasum is required to verify benchd."

# -- manifest -----------------------------------------------------------------
# benchd.manifest.json is values-only JSON, one key per line (build-dist.sh).
# Parsed with sed rather than jq/python3 so the OFFLINE path on the ranked box
# needs nothing but a shell and shasum.
manifest_field() {
  sed -n "s/^[[:space:]]*\"$2\"[[:space:]]*:[[:space:]]*\"\{0,1\}\([^\",]*\)\"\{0,1\}[[:space:]]*,\{0,1\}[[:space:]]*\$/\1/p" "$1"
}

# Read `sha256` or `bytes` out of ONE per-binary `binaries` entry:
#   manifest_binary_field <manifest> <binary name> <sha256|bytes>
# The entry is one line, so the binary name anchors the match and the field is
# read from inside its braces. Empty output means "this manifest declares no
# such binary" -- the legacy shape -- and is not an error here.
# Mirrors mlxfast-bench scripts/dist-lib.sh `dist_manifest_binary_field`.
manifest_binary_field() {
  sed -n "s/^[[:space:]]*\"$2\"[[:space:]]*:[[:space:]]*{.*\"$3\"[[:space:]]*:[[:space:]]*\"\{0,1\}\([^\",}]*\)\"\{0,1\}.*}[[:space:]]*,\{0,1\}[[:space:]]*\$/\1/p" "$1"
}

# Load + validate a manifest file; sets MF_BRANCH/MF_COMMIT/MF_SHA256/MF_BYTES,
# and MF_RECORDER_SHA256/MF_RECORDER_BYTES when the manifest declares the
# recorder (both empty on the legacy six-field manifest).
# An unparseable manifest must refuse, not degrade: empty fields would make
# every comparison below trivially "match".
load_manifest() {
  local path="$1"
  [[ -f "${path}" ]] || return 1
  MF_BRANCH="$(manifest_field "${path}" branch)"
  MF_COMMIT="$(manifest_field "${path}" source_commit)"
  MF_SHA256="$(manifest_field "${path}" sha256)"
  MF_BYTES="$(manifest_field "${path}" bytes)"
  MF_TRIPLE="$(manifest_field "${path}" target_triple)"
  MF_RECORDER_SHA256="$(manifest_binary_field "${path}" "${RECORDER_NAME}" sha256)"
  MF_RECORDER_BYTES="$(manifest_binary_field "${path}" "${RECORDER_NAME}" bytes)"
  if [[ "${#MF_SHA256}" -ne 64 || -n "${MF_SHA256//[0-9a-f]/}" ]]; then
    echo "fetch-benchd.sh:   manifest sha256 is not 64 lowercase hex characters: '${MF_SHA256}' (${path})" >&2
    return 1
  fi
  if [[ -z "${MF_BYTES}" || -n "${MF_BYTES//[0-9]/}" ]]; then
    echo "fetch-benchd.sh:   manifest bytes is not a positive integer: '${MF_BYTES}' (${path})" >&2
    return 1
  fi
  if [[ "${MF_BRANCH}" != "${BRANCH}" ]]; then
    echo "fetch-benchd.sh:   manifest names branch '${MF_BRANCH}', expected '${BRANCH}' -- wrong channel (${path})" >&2
    return 1
  fi
  # A DECLARED recorder must be declared completely. Half an entry -- a digest
  # with no byte count, or either one malformed -- would verify against nothing,
  # and "verifies against nothing" is the state this script exists to prevent.
  if [[ -n "${MF_RECORDER_SHA256}" || -n "${MF_RECORDER_BYTES}" ]]; then
    if [[ "${#MF_RECORDER_SHA256}" -ne 64 || -n "${MF_RECORDER_SHA256//[0-9a-f]/}" ]]; then
      echo "fetch-benchd.sh:   manifest ${RECORDER_NAME} sha256 is not 64 lowercase hex characters: '${MF_RECORDER_SHA256}' (${path})" >&2
      return 1
    fi
    if [[ -z "${MF_RECORDER_BYTES}" || -n "${MF_RECORDER_BYTES//[0-9]/}" ]]; then
      echo "fetch-benchd.sh:   manifest ${RECORDER_NAME} bytes is not a positive integer: '${MF_RECORDER_BYTES}' (${path})" >&2
      return 1
    fi
  fi
  return 0
}

# Does the loaded manifest declare the recorder? The legacy six-field manifest
# does not, and on it this script does exactly what it always did.
manifest_declares_recorder() {
  [[ -n "${MF_RECORDER_SHA256}" ]]
}

# Verify a recorder file against the LOADED manifest's `binaries` entry. Same
# order and same reporting as matches_manifest: bytes, then digest.
matches_recorder_entry() {
  local path="$1" actual_bytes actual_sha
  [[ -f "${path}" ]] || return 1
  actual_bytes="$(wc -c < "${path}" | tr -d '[:space:]')"
  if [[ "${actual_bytes}" != "${MF_RECORDER_BYTES}" ]]; then
    echo "fetch-benchd.sh:   RECORDER MISMATCH: byte count manifest=${MF_RECORDER_BYTES} actual=${actual_bytes} (${path})" >&2
    return 1
  fi
  actual_sha="$(shasum -a 256 "${path}" | awk '{print $1}')"
  if [[ "${actual_sha}" != "${MF_RECORDER_SHA256}" ]]; then
    echo "fetch-benchd.sh:   RECORDER MISMATCH: sha256 manifest=${MF_RECORDER_SHA256} actual=${actual_sha} (${path})" >&2
    return 1
  fi
  return 0
}

# Verify a binary against the LOADED manifest. Bytes first (cheap, and a length
# mismatch is already disqualifying), then the digest. Reports on stderr so
# callers see WHICH half failed.
matches_manifest() {
  local path="$1" actual_bytes actual_sha
  [[ -f "${path}" ]] || return 1
  actual_bytes="$(wc -c < "${path}" | tr -d '[:space:]')"
  if [[ "${actual_bytes}" != "${MF_BYTES}" ]]; then
    echo "fetch-benchd.sh:   byte count mismatch: manifest=${MF_BYTES} actual=${actual_bytes} (${path})" >&2
    return 1
  fi
  actual_sha="$(shasum -a 256 "${path}" | awk '{print $1}')"
  if [[ "${actual_sha}" != "${MF_SHA256}" ]]; then
    echo "fetch-benchd.sh:   sha256 mismatch: manifest=${MF_SHA256} actual=${actual_sha} (${path})" >&2
    return 1
  fi
  return 0
}

announce() {
  echo "fetch-benchd.sh: benchd identity: branch=${MF_BRANCH} source_commit=${MF_COMMIT} target_triple=${MF_TRIPLE} sha256=${MF_SHA256} bytes=${MF_BYTES}" >&2
  if manifest_declares_recorder; then
    echo "fetch-benchd.sh: ${RECORDER_NAME} identity: sha256=${MF_RECORDER_SHA256} bytes=${MF_RECORDER_BYTES} (staged at ${DEST_RECORDER})" >&2
  else
    echo "fetch-benchd.sh: this manifest declares no ${RECORDER_NAME}; the channel carries benchd alone" >&2
  fi
}

# -- platform -----------------------------------------------------------------
# The triple THIS host needs, derived from uname. An unrecognised host refuses
# rather than guessing: a wrong guess would either reject a correct dist or
# accept a binary this machine cannot run, and both are worse than saying so.
expected_triple() {
  local os machine
  os="$(uname -s)"
  machine="$(uname -m)"
  case "${os}:${machine}" in
    Darwin:arm64) printf 'aarch64-apple-darwin' ;;
    Darwin:x86_64) printf 'x86_64-apple-darwin' ;;
    Linux:aarch64) printf 'aarch64-unknown-linux-gnu' ;;
    Linux:arm64) printf 'aarch64-unknown-linux-gnu' ;;
    Linux:x86_64) printf 'x86_64-unknown-linux-gnu' ;;
    *) printf '' ;;
  esac
}

EXPECT_TRIPLE="${BENCHD_EXPECT_TARGET_TRIPLE:-$(expected_triple)}"
if [[ -z "${EXPECT_TRIPLE}" ]]; then
  die "cannot name the target triple this host expects ($(uname -s)/$(uname -m)); set BENCHD_EXPECT_TARGET_TRIPLE rather than run an unchecked binary."
fi

# THE CHANNEL PUBLISHES ONE DIRECTORY PER PLATFORM, and the platform is what
# picks the directory. The macOS pair is the channel's historical root, `dist/`;
# the Linux pairs are `dist/linux-aarch64/` (bench pull request 219, the ranked
# box) and `dist/linux-x86_64/` (a participant's x86 workstation; no ranked box
# runs it). All carry
# the SAME six-field manifest, so the two pairs are indistinguishable except by
# `target_triple` -- which is exactly why the wrong directory must not be
# reachable by accident: the macOS pair passes every hash check and installs a
# Mach-O the Linux box cannot exec.
dist_subpath_for() {
  case "$1" in
    aarch64-apple-darwin) printf 'dist' ;;
    aarch64-unknown-linux-gnu) printf 'dist/linux-aarch64' ;;
    x86_64-unknown-linux-gnu) printf 'dist/linux-x86_64' ;;
    *) printf '' ;;
  esac
}

DIST_SUBPATH="$(dist_subpath_for "${EXPECT_TRIPLE}")"
if [[ -z "${DIST_SUBPATH}" ]]; then
  die "the channel publishes no dist for '${EXPECT_TRIPLE}'; there is no binary to resolve for this host."
fi

# The container format a triple names. Used to read the BYTES back against the
# manifest's claim, so a mis-stamped publish is caught too.
triple_format() {
  case "$1" in
    *-apple-darwin) printf 'macho' ;;
    *-linux-*) printf 'elf' ;;
    *) printf '' ;;
  esac
}

# The container format a file actually has, from its first four bytes.
#   ELF        7f 45 4c 46
#   Mach-O     cf fa ed fe (64-bit little-endian) / ca fe ba be (universal)
file_format() {
  local magic
  magic="$(od -An -tx1 -N4 -v < "$1" | tr -d ' \n')"
  case "${magic}" in
    7f454c46) printf 'elf' ;;
    cffaedfe|cefaedfe|feedfacf|feedface|cafebabe|bebafeca) printf 'macho' ;;
    *) printf "unknown(${magic})" ;;
  esac
}

# Refuse a binary this host cannot run. Both halves name both values, because
# "wrong platform" is only actionable when the reader can see which two.
matches_platform() {
  local path="$1" want_format have_format
  if [[ -z "${MF_TRIPLE}" ]]; then
    echo "fetch-benchd.sh:   the manifest names no target_triple, so the platform cannot be checked (expected ${EXPECT_TRIPLE})" >&2
    return 1
  fi
  if [[ "${MF_TRIPLE}" != "${EXPECT_TRIPLE}" ]]; then
    echo "fetch-benchd.sh:   WRONG PLATFORM: this pair's target_triple is '${MF_TRIPLE}', this host needs '${EXPECT_TRIPLE}'" >&2
    echo "fetch-benchd.sh:   the channel keeps one directory per platform; this host resolves ${DIST_SUBPATH}/ on branch ${BRANCH}" >&2
    return 1
  fi
  want_format="$(triple_format "${MF_TRIPLE}")"
  if [[ -z "${want_format}" ]]; then
    echo "fetch-benchd.sh:   target_triple '${MF_TRIPLE}' names no container format this script knows; refusing" >&2
    return 1
  fi
  have_format="$(file_format "${path}")"
  if [[ "${have_format}" != "${want_format}" ]]; then
    echo "fetch-benchd.sh:   MIS-STAMPED DIST: target_triple '${MF_TRIPLE}' says ${want_format}, the bytes are ${have_format} (${path})" >&2
    return 1
  fi
  return 0
}

# -- 1. already in place (the offline path) -----------------------------------
if [[ "${BENCHD_REFRESH:-0}" == "1" && -f "${DEST}" ]]; then
  echo "fetch-benchd.sh: BENCHD_REFRESH=1 -- discarding the installed pair to re-resolve the channel tip" >&2
  rm -f "${DEST}" "${DEST_MANIFEST}" "${DEST_RECORDER}"
fi

if [[ -f "${DEST}" ]]; then
  load_manifest "${DEST_MANIFEST}" \
    || die "${DEST} exists but ${DEST_MANIFEST} is missing or malformed; a binary with no manifest is unattributable and will not be run. Place the channel's benchd.manifest.json beside it, or delete the binary and re-run."
  matches_manifest "${DEST}" \
    || die "${DEST} does NOT match its manifest (see the mismatch above); refusing to run or replace it. Delete the pair deliberately, then re-run."
  matches_platform "${DEST}" \
    || die "${DEST} is not for this host (see the platform mismatch above); refusing to run it. A binary for another platform fails at exec with an unreadable message, so it is refused here by name instead."
  if manifest_declares_recorder; then
    [[ -f "${DEST_RECORDER}" ]] \
      || die "RECORDER MISSING: ${DEST_MANIFEST} declares ${RECORDER_NAME} but ${DEST_RECORDER} is not there. The channel publishes both files together; place the recorder beside benchd, or set BENCHD_REFRESH=1 to re-resolve the pair."
    matches_recorder_entry "${DEST_RECORDER}" \
      || die "${DEST_RECORDER} does NOT match the manifest entry for ${RECORDER_NAME} (see the mismatch above); refusing to run or replace it. Delete it deliberately, or set BENCHD_REFRESH=1."
    chmod 755 "${DEST_RECORDER}"
  fi
  chmod 755 "${DEST}"
  announce
  printf '%s\n' "${DEST}"
  exit 0
fi

# -- 2. obtain the pair -------------------------------------------------------
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT
STAGED="${TMP_DIR}/benchd"
STAGED_MANIFEST="${TMP_DIR}/benchd.manifest.json"
STAGED_RECORDER="${TMP_DIR}/${RECORDER_NAME}"

if [[ -n "${BENCHD_DIST_LOCAL:-}" ]]; then
  src="${BENCHD_DIST_LOCAL}"
  [[ -d "${src}" ]] && src="${src}/benchd"
  [[ -f "${src}" ]] || die "BENCHD_DIST_LOCAL does not resolve to a benchd file: ${src}"
  src_manifest="$(dirname "${src}")/benchd.manifest.json"
  [[ -f "${src_manifest}" ]] || die "BENCHD_DIST_LOCAL has no benchd.manifest.json beside the binary (${src_manifest}); an unattributable binary will not be installed."
  echo "fetch-benchd.sh: taking benchd from ${src}" >&2
  cp "${src}" "${STAGED}"
  cp "${src_manifest}" "${STAGED_MANIFEST}"
  load_manifest "${STAGED_MANIFEST}" \
    || die "BENCHD_DIST_LOCAL manifest is malformed or names the wrong channel; refusing."
  # One named source, so a mismatch is a hard refusal: silently continuing past
  # the file the operator explicitly pointed at would hide the real problem.
  matches_manifest "${STAGED}" \
    || die "BENCHD_DIST_LOCAL (${src}) does NOT match its manifest (see the mismatch above); refusing to install or run it."
  matches_platform "${STAGED}" \
    || die "BENCHD_DIST_LOCAL (${src}) is not for this host (see the platform mismatch above); refusing to install or run it."
  if manifest_declares_recorder; then
    src_recorder="$(dirname "${src}")/${RECORDER_NAME}"
    [[ -f "${src_recorder}" ]] \
      || die "RECORDER MISSING: the BENCHD_DIST_LOCAL manifest declares ${RECORDER_NAME} but ${src_recorder} is not there. The channel publishes both files in one directory; a dist that carries only one of them is incomplete."
    cp "${src_recorder}" "${STAGED_RECORDER}"
    matches_recorder_entry "${STAGED_RECORDER}" \
      || die "BENCHD_DIST_LOCAL (${src_recorder}) does NOT match the manifest entry for ${RECORDER_NAME} (see the mismatch above); refusing to install or run it."
  fi
else
  command -v curl >/dev/null 2>&1 || die "curl is required to download benchd (or set BENCHD_DIST_LOCAL)."

  # The channel: this host's dist directory on the bench branch tip (see
  # dist_subpath_for above). The manifest is fetched FIRST,
  # then the binary from the same directory, and the binary must match the
  # manifest -- so a half-updated publish (one file at the old build) refuses
  # rather than installing a pair that disagrees. The `refs/heads/` prefix is
  # required, not cosmetic: branch names contain slashes, and without the
  # explicit ref namespace raw.githubusercontent cannot tell where the ref ends
  # and the path begins.
  DIST_DIR_URL="${BASE_URL}/refs/heads/${BRANCH}/${DIST_SUBPATH}"
  auth_token="${BENCHD_DIST_TOKEN:-${GITHUB_TOKEN:-}}"
  fetch() {
    local url="$1" out="$2"
    if [[ -n "${auth_token}" ]]; then
      curl -fsSL --retry 3 --retry-delay 2 -H "Authorization: Bearer ${auth_token}" -o "${out}" "${url}"
    else
      curl -fsSL --retry 3 --retry-delay 2 -o "${out}" "${url}"
    fi
  }

  echo "fetch-benchd.sh: resolving the ${BRANCH} channel tip (${DIST_DIR_URL})" >&2
  fetch "${DIST_DIR_URL}/benchd.manifest.json" "${STAGED_MANIFEST}" || {
    {
      echo "fetch-benchd.sh: could not fetch the channel manifest (${DIST_DIR_URL}/benchd.manifest.json)."
      if [[ -z "${auth_token}" ]]; then
        echo "  if the channel repo is private, set BENCHD_DIST_TOKEN/GITHUB_TOKEN."
      fi
      echo "  offline alternative: place benchd + benchd.manifest.json at ${DEST_DIR}/,"
      echo "  or point BENCHD_DIST_LOCAL at a directory holding both."
    } >&2
    exit 1
  }
  load_manifest "${STAGED_MANIFEST}" \
    || die "the channel manifest is malformed or names the wrong branch; refusing (this is a publish problem, not resolved by re-running)."
  fetch "${DIST_DIR_URL}/benchd" "${STAGED}" \
    || die "the channel manifest exists but the binary download failed (${DIST_DIR_URL}/benchd)."
  matches_manifest "${STAGED}" \
    || die "the downloaded benchd does NOT match the channel manifest (see the mismatch above) -- a half-updated publish; refusing. Republish dist bench-side, then re-run."
  matches_platform "${STAGED}" \
    || die "the channel's benchd is not for this host (see the platform mismatch above); refusing to install it. Publish the ${EXPECT_TRIPLE} dist bench-side, then re-run."
  # The second binary, from the SAME platform directory, verified against its
  # own `binaries` entry. Only fetched when the manifest declares it, so the
  # legacy channel is one request as before.
  if manifest_declares_recorder; then
    echo "fetch-benchd.sh: the manifest declares ${RECORDER_NAME}; fetching it from the same directory" >&2
    fetch "${DIST_DIR_URL}/${RECORDER_NAME}" "${STAGED_RECORDER}" \
      || die "RECORDER MISSING: the manifest declares ${RECORDER_NAME} but the download failed (${DIST_DIR_URL}/${RECORDER_NAME}). Republish dist bench-side, then re-run."
    matches_recorder_entry "${STAGED_RECORDER}" \
      || die "the downloaded ${RECORDER_NAME} does NOT match the channel manifest entry (see the mismatch above) -- a half-updated publish; refusing. Republish dist bench-side, then re-run."
  fi
fi

# -- 3. install ONLY what was verified ----------------------------------------
mkdir -p "${DEST_DIR}"
chmod 755 "${STAGED}"
mv "${STAGED}" "${DEST}"
mv "${STAGED_MANIFEST}" "${DEST_MANIFEST}"
echo "fetch-benchd.sh: installed benchd at ${DEST} (manifest beside it)" >&2
if manifest_declares_recorder; then
  chmod 755 "${STAGED_RECORDER}"
  mv "${STAGED_RECORDER}" "${DEST_RECORDER}"
  echo "fetch-benchd.sh: installed ${RECORDER_NAME} at ${DEST_RECORDER}" >&2
else
  # A channel that no longer declares the recorder must not leave a stale one
  # staged: it would be a binary this manifest cannot attribute, which is the
  # one thing the offline path refuses to run.
  rm -f "${DEST_RECORDER}"
fi
announce

printf '%s\n' "${DEST}"
