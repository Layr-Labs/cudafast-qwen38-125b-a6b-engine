#!/usr/bin/env bash
#
# fetch-goldens.sh -- fetch one pin-verified object from the track's R2 bucket
# for track qwen3.8-125b-a6b-cuda-v1, keyed on {r2_path, sha256, bytes}.
#
# ---------------------------------------------------------------------------
# READ THIS BEFORE ASSUMING THIS SCRIPT GIVES YOU A LOCAL CALIBRATION GOLDEN.
#
# As of this commit there is NO PUBLIC GOLDEN PIN in this repository, so there
# is nothing here for a participant to fetch yet. That is a finding, not an
# omission in this script:
#
#   * Every {r2_path, sha256, bytes} pin in fixtures/qwen3_8_125b_a6b_track.json
#     is HIDDEN material -- the 8 `timed_prompt_pool[]` tapes and the
#     `hidden_correctness_golden` oracle. They are organizer/box-side by
#     construction: the GETs are CREDENTIALED (SigV4, R2_ACCESS_KEY_ID /
#     R2_SECRET_ACCESS_KEY), so a participant clone cannot fetch them even if
#     it tried, and this script REFUSES to try (see "the hidden guard" below).
#     They are also the wrong FORMAT for local iteration: pool objects are
#     benchd `TimedPromptTapeDocument` tapes with `deny_unknown_fields`, not
#     the `--golden` documents `MLXFAST_CORRECTNESS_GOLDEN_PATH` names.
#
#   * The one correctness golden a participant legitimately has today needs no
#     fetching at all: correctness_prompts/public_longcopy_gate_english_1024_256.json
#     is CHECKED INTO GIT (gemma provenance -- `model_provenance.repository` is
#     the pinned RadixArk/Qwen3.8-Flash-Next-NVFP4 @ 7b719225), and it
#     is what `defaultCorrectnessGoldenPath()` in Sources/MLXFastCLI/main.swift
#     already falls back to via MLXFAST_PUBLIC_CORRECTNESS_GOLDEN_PATH. A
#     `yukon clone` gets it in the clone.
#
# So this script exists ARMED AND UNUSED on the participant path: the transport
# and the pin verification are real and tested, keyed on explicit arguments, so
# that publishing a public local-calibration golden is a one-line invocation
# plus a documented pin -- not another lane. WHICH object that should be, and
# what its pin is, is an ORGANIZER DECISION and is deliberately not invented
# here.
# ---------------------------------------------------------------------------
#
# THE R2 CONVENTION THIS MIRRORS (do not re-derive it):
#   * The base URL lives in the environment variable R2_BUCKET_ENDPOINT and
#     NOWHERE ELSE. It is secret-tier: never hardcode an endpoint, a bucket
#     name, or an account host in this repository. Ask the organizer for the
#     R2 base.
#   * The BUCKET IS PART OF THE ENDPOINT, never part of r2_path. Prefixing the
#     bucket onto the object key is the documented way to waste a day; the
#     earlier tracks' runbooks record it costing three ranked dispatches.
#   * Object keys live under correctness_prompts/<track_id>/, which is also the
#     branch name and the track id -- one string, three roles.
#   * Verify BYTE COUNT FIRST, then sha256, and delete the file on either
#     mismatch: a truncated transfer is the common failure and the byte count
#     names it precisely, where a bare hash mismatch does not.
#   This mirrors .github/scripts/download-r2-object.sh and
#   scripts/verify-cuda-track-pins.sh --fetch on the qwen/cuda tracks; neither
#   exists in this repository, which is why the credentialed transport here is
#   DELEGATED rather than reimplemented (a second hand-rolled SigV4 signer is
#   exactly the wrong thing to own twice).
#
# Usage:
#   tools/fetch-goldens.sh --r2-path KEY --sha256 HEX --bytes N --out FILE
#
# Env:
#   R2_BUCKET_ENDPOINT   REQUIRED. https://<host>[/<bucket>[/<prefix>...]].
#                        Ask the organizer; keep it in .env, never in a repo.
#   R2_ACCESS_KEY_ID     Optional. Set (with the secret below) to take the
#   R2_SECRET_ACCESS_KEY   credentialed path via a delegated downloader.
#                        Unset => a plain anonymous HTTPS GET, which is all a
#                        public object needs.
#   MLXFAST_QWEN38_R2_DOWNLOADER
#                        Path to a `download-r2-object.sh KEY DEST` compatible
#                        signer for the credentialed path. No default: this
#                        repository ships no signer.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
CONTRACT="${SCRIPT_DIR}/fixtures/qwen3_8_125b_a6b_track.json"

R2_PATH=""
WANT_SHA=""
WANT_BYTES=""
OUT_PATH=""
ALLOW_HIDDEN=0

usage() {
  cat <<EOF
Usage: tools/fetch-goldens.sh --r2-path KEY --sha256 HEX --bytes N --out FILE

Fetch one object from the track's R2 bucket and accept it only if it matches
the supplied {sha256, bytes} pin exactly. The file is removed on any mismatch,
so a failed run never leaves half-verified bytes behind.

  --r2-path KEY   Object key, e.g. correctness_prompts/qwen3.8-125b-a6b-cuda-v1/NAME.json
                  The BUCKET IS NOT PART OF THIS -- it is in R2_BUCKET_ENDPOINT.
  --sha256 HEX    64 lowercase hex characters. Required: there is no unpinned fetch.
  --bytes N       Exact byte count. Required, and checked before the hash.
  --out FILE      Destination path.
  --allow-hidden  Organizer/box escape hatch; see below. Needs credentials too.

THE R2 BASE URL COMES FROM THE ENVIRONMENT ONLY:

  export R2_BUCKET_ENDPOINT='https://<host>/<bucket>'   # ASK THE ORGANIZER

It is secret-tier material. It is deliberately absent from this repository and
must stay that way -- keep it in your .env, never in a file you commit.

NO PUBLIC GOLDEN IS PINNED FOR THIS TRACK YET, so there is currently nothing a
participant should be fetching with this. The correctness golden local runs
already use is checked into the clone at
correctness_prompts/public_longcopy_gate_english_1024_256.json.

THE HIDDEN GUARD: every pin in fixtures/qwen3_8_125b_a6b_track.json (the 8
timed_prompt_pool tapes and hidden_correctness_golden) is hidden, box-only
material. This script refuses to fetch any of them by key OR by digest, so a
copy-pasted pin from the contract cannot quietly pull hidden bytes onto a
participant machine. --allow-hidden lifts that refusal for organizer-side
provisioning and additionally requires R2 credentials to be present.
EOF
}

while (( $# > 0 )); do
  case "$1" in
    --r2-path) R2_PATH="${2:-}"; shift 2 ;;
    --sha256)  WANT_SHA="${2:-}"; shift 2 ;;
    --bytes)   WANT_BYTES="${2:-}"; shift 2 ;;
    --out)     OUT_PATH="${2:-}"; shift 2 ;;
    --allow-hidden) ALLOW_HIDDEN=1; shift ;;
    -h|--help|help) usage; exit 0 ;;
    *)
      echo "fetch-goldens.sh: unknown argument '$1'" >&2
      usage >&2
      exit 2
      ;;
  esac
done

for required in R2_PATH WANT_SHA WANT_BYTES OUT_PATH; do
  eval "value=\${${required}}"
  if [[ -z "${value}" ]]; then
    echo "fetch-goldens.sh: missing required argument (--r2-path, --sha256, --bytes, --out are all mandatory)" >&2
    exit 2
  fi
done

# A pin is sha256 AND bytes together; neither alone is a pin. Reject malformed
# input here rather than after spending a download on it.
if ! printf '%s' "${WANT_SHA}" | grep -Eq '^[0-9a-f]{64}$'; then
  echo "fetch-goldens.sh: --sha256 must be 64 lowercase hex characters, got: ${WANT_SHA}" >&2
  exit 2
fi
if ! printf '%s' "${WANT_BYTES}" | grep -Eq '^[1-9][0-9]*$'; then
  echo "fetch-goldens.sh: --bytes must be a positive integer, got: ${WANT_BYTES}" >&2
  exit 2
fi
# The signed key is sent verbatim with no percent-encoding on the credentialed
# path, so restrict the charset the same way the qwen signer does.
if ! printf '%s' "${R2_PATH}" | grep -Eq '^[A-Za-z0-9._/-]+$'; then
  echo "fetch-goldens.sh: --r2-path may only contain [A-Za-z0-9._/-], got: ${R2_PATH}" >&2
  exit 2
fi
case "${R2_PATH}" in
  /*|*/../*|*/..)
    echo "fetch-goldens.sh: --r2-path must be a relative object key without '..' segments" >&2
    exit 2
    ;;
esac

# --- the hidden guard -------------------------------------------------------
# Read the pins the contract declares hidden -- the timed_prompt_pool[] tapes
# and the hidden_correctness_golden oracle -- and refuse to fetch any of them.
# Matching on BOTH key and digest matters: renaming the object on the command
# line must not get around the digest check, and vice versa.
#
# THE GUARD FAILS CLOSED. If the contract cannot be read there is no way to
# know whether a requested pin is hidden, and "cannot tell" must mean "do not
# fetch" -- the alternative is that deleting or renaming one fixture silently
# disarms the only thing standing between a copy-pasted pool digest and hidden
# bytes on a participant's disk. (Caught by tools/test-fetch-goldens.sh, whose
# out-of-tree copy of this script found the guard passing vacuously.)
if [[ ! -r "${CONTRACT}" ]]; then
  echo "fetch-goldens.sh: cannot read the track contract at ${CONTRACT}" >&2
  echo "fetch-goldens.sh: refusing -- the hidden-material guard cannot be evaluated without it" >&2
  exit 1
fi

hidden_pins() {
  awk '
    /^  "timed_prompt_pool": \[/ { in_pool=1; next }
    /^  "hidden_correctness_golden": \{/ { in_hidden=1; next }
    in_pool && /^  \]/ { in_pool=0; next }
    in_hidden && /^  \}/ { in_hidden=0; next }
    (in_pool || in_hidden) && /"(sha256|r2_path)":/ {
      value=$0
      sub(/^[^:]*: *"/, "", value)
      sub(/".*$/, "", value)
      print value
    }
  ' "${CONTRACT}"
}

is_hidden=0
while IFS= read -r pin; do
  [[ -n "${pin}" ]] || continue
  if [[ "${pin}" == "${R2_PATH}" || "${pin}" == "${WANT_SHA}" ]]; then
    is_hidden=1
    break
  fi
done <<EOF
$(hidden_pins)
EOF

if [[ "${is_hidden}" == "1" && "${ALLOW_HIDDEN}" != "1" ]]; then
  cat >&2 <<EOF
fetch-goldens.sh: REFUSING -- that pin is hidden, box-only material.

  requested key : ${R2_PATH}
  requested sha : ${WANT_SHA}

It matches a timed_prompt_pool[] tape or hidden_correctness_golden in
fixtures/qwen3_8_125b_a6b_track.json. Those objects are organizer-side: the GETs
are credentialed, the tapes are a benchd format local --golden modes cannot
load, and the anti-lottery cohort stops being hidden the moment a participant
holds all eight.

If you are the organizer provisioning a box, pass --allow-hidden (credentials
are required as well). If you are looking for a golden to iterate against
locally, the checked-in one is
correctness_prompts/public_longcopy_gate_english_1024_256.json -- no fetch
needed.
EOF
  exit 1
fi

# --- endpoint ---------------------------------------------------------------
if [[ -z "${R2_BUCKET_ENDPOINT:-}" ]]; then
  cat >&2 <<EOF
fetch-goldens.sh: R2_BUCKET_ENDPOINT is not set.

The R2 base URL is secret-tier and is intentionally NOT stored in this
repository. ASK THE ORGANIZER FOR THE R2 BASE, then:

  export R2_BUCKET_ENDPOINT='https://<host>/<bucket>'

Keep it in your .env; never commit it. The bucket belongs in this endpoint,
never in --r2-path.
EOF
  exit 1
fi

endpoint="${R2_BUCKET_ENDPOINT%/}"
if ! printf '%s' "${endpoint}" | grep -Eq '^https://[A-Za-z0-9.-]+(/[A-Za-z0-9._-]+)*$'; then
  # Deliberately does NOT echo the endpoint: it is secret-tier and this message
  # can land in a log.
  echo "fetch-goldens.sh: R2_BUCKET_ENDPOINT is malformed (want https://<host>[/<bucket>...]); value withheld" >&2
  exit 1
fi

mkdir -p "$(dirname "${OUT_PATH}")"

have_credentials=0
if [[ -n "${R2_ACCESS_KEY_ID:-}" && -n "${R2_SECRET_ACCESS_KEY:-}" ]]; then
  have_credentials=1
fi

if [[ "${is_hidden}" == "1" && "${have_credentials}" != "1" ]]; then
  echo "fetch-goldens.sh: --allow-hidden needs R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY; hidden objects are not anonymously readable" >&2
  exit 1
fi

# --- transport --------------------------------------------------------------
# Credentialed objects need a SigV4 signer. This repository ships none on
# purpose, so the signer is delegated: point MLXFAST_QWEN38_R2_DOWNLOADER at a
# `download-r2-object.sh KEY DEST` compatible script. Anonymous objects need
# nothing but curl.
if [[ "${have_credentials}" == "1" ]]; then
  downloader="${MLXFAST_QWEN38_R2_DOWNLOADER:-}"
  if [[ -z "${downloader}" ]]; then
    echo "fetch-goldens.sh: R2 credentials are set but no signer is available." >&2
    echo "fetch-goldens.sh: set MLXFAST_QWEN38_R2_DOWNLOADER to a 'download-r2-object.sh KEY DEST' script." >&2
    echo "fetch-goldens.sh: (this repository deliberately ships no SigV4 signer of its own)" >&2
    exit 1
  fi
  if [[ ! -x "${downloader}" ]]; then
    echo "fetch-goldens.sh: MLXFAST_QWEN38_R2_DOWNLOADER is not executable: ${downloader}" >&2
    exit 1
  fi
  echo "fetch-goldens.sh: fetching ${R2_PATH} (credentialed, delegated signer)" >&2
  if ! "${downloader}" "${R2_PATH}" "${OUT_PATH}"; then
    echo "fetch-goldens.sh: delegated download failed for ${R2_PATH}" >&2
    rm -f "${OUT_PATH}"
    exit 1
  fi
else
  url="${endpoint}/${R2_PATH}"
  echo "fetch-goldens.sh: fetching ${R2_PATH} (anonymous)" >&2
  # --fail so an HTML error page never gets hashed as if it were the object;
  # no --location, matching the qwen signer (a redirect off the pinned
  # endpoint is not a source we agreed to).
  if ! curl --fail --silent --show-error \
       --connect-timeout 30 --max-time 600 \
       --retry 5 --retry-all-errors --retry-delay 2 \
       --output "${OUT_PATH}" "${url}"; then
    echo "fetch-goldens.sh: download failed for ${R2_PATH}" >&2
    echo "fetch-goldens.sh: (a 403 here usually means the object is credentialed, i.e. organizer-side)" >&2
    rm -f "${OUT_PATH}"
    exit 1
  fi
fi

# --- pin verification -------------------------------------------------------
# Byte count FIRST: a truncated transfer is the common failure and this names
# it exactly, where a bare hash mismatch would only say "different".
actual_bytes="$(wc -c < "${OUT_PATH}" | tr -d '[:space:]')"
if [[ "${actual_bytes}" != "${WANT_BYTES}" ]]; then
  rm -f "${OUT_PATH}"
  echo "fetch-goldens.sh: byte-count mismatch for ${R2_PATH} (got ${actual_bytes}, pinned ${WANT_BYTES}); refused" >&2
  exit 1
fi

actual_sha="$(shasum -a 256 "${OUT_PATH}" | awk '{print $1}')"
if [[ "${actual_sha}" != "${WANT_SHA}" ]]; then
  rm -f "${OUT_PATH}"
  echo "fetch-goldens.sh: sha256 mismatch for ${R2_PATH} (got ${actual_sha}, pinned ${WANT_SHA}); refused" >&2
  exit 1
fi

echo "fetch-goldens.sh: verified ${R2_PATH} -> ${OUT_PATH} (sha256 ${actual_sha}, bytes ${actual_bytes})"
