#!/usr/bin/env bash
#
# fetch-goldens.sh -- fetch pin-verified golden objects from the track's R2
# bucket for track qwen3.8-125b-a6b-cuda-v1.
#
# THE SCRIPT HAS TWO MODES.
#
#   1. ONE OBJECT, for participants:
#        tools/fetch-goldens.sh --r2-path KEY --sha256 HEX --bytes N --out FILE
#      It fetches one object and keeps it only when the bytes match the pin.
#      A pin that fixtures/qwen3_8_125b_a6b_track.json declares hidden is
#      refused, by key and by digest, so a pin copied out of the contract
#      cannot pull organizer material onto a participant machine.
#
#   2. THE WHOLE PINNED SET, for the organizer staging a ranked box:
#        tools/fetch-goldens.sh --all --out DIR
#      It reads the contract, then fetches every timed_prompt_pool tape and
#      every live_golden_speculative oracle into DIR. DIR is the directory the
#      box's runner service exports as MLXFAST_QWEN38_GOLDEN_DIR. These objects
#      are organizer material, so this mode needs R2 credentials and refuses
#      without them. Run tools/ranked-box-preflight.sh afterwards: it verifies
#      the staged set against the contract again.
#
# WHY THE GOLDENS ARE NOT IN THIS REPOSITORY. The 8 timed-pool tapes and the 6
# per-depth oracles are organizer material. They are published in R2 at the
# r2_path keys the contract pins, and the ranked box stages them out of band.
# They are never in git. The public goldens under correctness_prompts/ are
# different material: they are participant goldens for local runs, they carry
# no pin here, and they need no fetch.
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
#
# Usage:
#   tools/fetch-goldens.sh --r2-path KEY --sha256 HEX --bytes N --out FILE
#   tools/fetch-goldens.sh --all --out DIR
#
# Env:
#   R2_BUCKET_ENDPOINT   REQUIRED. https://<host>[/<bucket>[/<prefix>...]].
#                        Ask the organizer; keep it in .env, never in a repo.
#   R2_ACCESS_KEY_ID     The credentialed path. REQUIRED by --all. For one
#   R2_SECRET_ACCESS_KEY   object they are optional: unset means a plain
#                        anonymous HTTPS GET, which is all a public object
#                        needs.
#   MLXFAST_QWEN38_R2_DOWNLOADER
#                        Path to a `download-r2-object.sh KEY DEST` compatible
#                        signer. It overrides the vendored signer at
#                        tools/download-r2-object.sh, which is the default.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
CONTRACT="${SCRIPT_DIR}/fixtures/qwen3_8_125b_a6b_track.json"

R2_PATH=""
WANT_SHA=""
WANT_BYTES=""
OUT_PATH=""
ALLOW_HIDDEN=0
FETCH_ALL=0
DEFAULT_DOWNLOADER="${SCRIPT_DIR}/tools/download-r2-object.sh"

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

THE HIDDEN GUARD: every pin in fixtures/qwen3_8_125b_a6b_track.json (the
timed_prompt_pool tapes, the per-depth oracles and hidden_correctness_golden)
is organizer material. This script refuses to fetch any of them by key OR by
digest, so a copy-pasted pin from the contract cannot quietly pull that
material onto a participant machine. --allow-hidden lifts that refusal for one
object and additionally requires R2 credentials to be present.

Usage: tools/fetch-goldens.sh --all --out DIR

Stage the WHOLE pinned set for a ranked box: every timed_prompt_pool tape and
every live_golden_speculative oracle, each verified against its {sha256, bytes}
pin, into DIR. A file that already matches its pin is left alone, so the mode
is safe to re-run. It finishes by finding the hidden_correctness_golden digest
among the staged files.

  --all           Stage the whole pinned set. Needs R2_ACCESS_KEY_ID and
                  R2_SECRET_ACCESS_KEY.
  --out DIR       The staging directory. Export it to the runner service as
                  MLXFAST_QWEN38_GOLDEN_DIR, then run
                  tools/ranked-box-preflight.sh.
EOF
}

while (( $# > 0 )); do
  case "$1" in
    --r2-path) R2_PATH="${2:-}"; shift 2 ;;
    --sha256)  WANT_SHA="${2:-}"; shift 2 ;;
    --bytes)   WANT_BYTES="${2:-}"; shift 2 ;;
    --out)     OUT_PATH="${2:-}"; shift 2 ;;
    --allow-hidden) ALLOW_HIDDEN=1; shift ;;
    --all)     FETCH_ALL=1; shift ;;
    -h|--help|help) usage; exit 0 ;;
    *)
      echo "fetch-goldens.sh: unknown argument '$1'" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "${FETCH_ALL}" == "1" ]]; then
  if [[ -z "${OUT_PATH}" ]]; then
    echo "fetch-goldens.sh: --all needs --out DIR (the staging directory the box exports as MLXFAST_QWEN38_GOLDEN_DIR)" >&2
    exit 2
  fi
  if [[ -n "${R2_PATH}" || -n "${WANT_SHA}" || -n "${WANT_BYTES}" ]]; then
    echo "fetch-goldens.sh: --all stages every pin the contract declares, so --r2-path, --sha256 and --bytes do not apply to it" >&2
    exit 2
  fi
else
  for required in R2_PATH WANT_SHA WANT_BYTES OUT_PATH; do
    eval "value=\${${required}}"
    if [[ -z "${value}" ]]; then
      echo "fetch-goldens.sh: missing required argument (--r2-path, --sha256, --bytes, --out are all mandatory)" >&2
      exit 2
    fi
  done
fi

# A pin is sha256 AND bytes together; neither alone is a pin. Reject malformed
# input here rather than after spending a download on it. --all takes its pins
# from the contract, so it validates them where it reads them.
if [[ "${FETCH_ALL}" == "0" ]]; then
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
fi

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
if [[ "${FETCH_ALL}" == "0" ]]; then
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

If you are the organizer staging a box, use --all --out DIR, which stages the
whole pinned set (credentials are required as well); --allow-hidden lifts this
refusal for one object. If you are looking for a golden to iterate against
locally, the checked-in one is
correctness_prompts/public_longcopy_gate_english_1024_256.json -- no fetch
needed.
EOF
  exit 1
fi
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

have_credentials=0
if [[ -n "${R2_ACCESS_KEY_ID:-}" && -n "${R2_SECRET_ACCESS_KEY:-}" ]]; then
  have_credentials=1
fi

if [[ "${is_hidden}" == "1" && "${have_credentials}" != "1" ]]; then
  echo "fetch-goldens.sh: --allow-hidden needs R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY; hidden objects are not anonymously readable" >&2
  exit 1
fi

# --- the signer -------------------------------------------------------------
# Credentialed objects need a SigV4 signer. The vendored one at
# tools/download-r2-object.sh is the default, so staging a box needs nothing
# else on it. MLXFAST_QWEN38_R2_DOWNLOADER overrides it with any
# `download-r2-object.sh KEY DEST` compatible script.
DOWNLOADER=""
if [[ "${have_credentials}" == "1" ]]; then
  DOWNLOADER="${MLXFAST_QWEN38_R2_DOWNLOADER:-${DEFAULT_DOWNLOADER}}"
  if [[ ! -x "${DOWNLOADER}" ]]; then
    echo "fetch-goldens.sh: the signer is not executable: ${DOWNLOADER}" >&2
    echo "fetch-goldens.sh: (set MLXFAST_QWEN38_R2_DOWNLOADER to a 'download-r2-object.sh KEY DEST' script, or restore tools/download-r2-object.sh)" >&2
    exit 1
  fi
fi

# --- transport --------------------------------------------------------------
# transport_fetch KEY DEST -- one object, no verification. The caller owns the
# pin check, and owns removing DEST when the pin does not hold.
transport_fetch() {
  local key="$1" dest="$2"
  mkdir -p "$(dirname "${dest}")"
  if [[ "${have_credentials}" == "1" ]]; then
    echo "fetch-goldens.sh: fetching ${key} (credentialed, signed)" >&2
    if ! "${DOWNLOADER}" "${key}" "${dest}"; then
      echo "fetch-goldens.sh: signed download failed for ${key}" >&2
      rm -f "${dest}"
      return 1
    fi
  else
    echo "fetch-goldens.sh: fetching ${key} (anonymous)" >&2
    # --fail so an HTML error page never gets hashed as if it were the object;
    # no --location, matching the qwen signer (a redirect off the pinned
    # endpoint is not a source we agreed to).
    if ! curl --fail --silent --show-error \
         --connect-timeout 30 --max-time 600 \
         --retry 5 --retry-all-errors --retry-delay 2 \
         --output "${dest}" "${endpoint}/${key}"; then
      echo "fetch-goldens.sh: download failed for ${key}" >&2
      echo "fetch-goldens.sh: (a 403 here usually means the object is credentialed, i.e. organizer-side)" >&2
      rm -f "${dest}"
      return 1
    fi
  fi
  return 0
}

# --- pin verification -------------------------------------------------------
# verify_pin PATH WANT_SHA WANT_BYTES LABEL -- byte count FIRST, because a
# truncated transfer is the common failure and this names it exactly, where a
# bare hash mismatch would only say "different". The file is REMOVED on either
# mismatch, so a failed run never leaves half-verified bytes behind.
verify_pin() {
  local path="$1" want_sha="$2" want_bytes="$3" label="$4" got_bytes got_sha
  got_bytes="$(wc -c < "${path}" | tr -d '[:space:]')"
  if [[ "${got_bytes}" != "${want_bytes}" ]]; then
    rm -f "${path}"
    echo "fetch-goldens.sh: byte-count mismatch for ${label} (got ${got_bytes}, pinned ${want_bytes}); refused" >&2
    return 1
  fi
  got_sha="$(shasum -a 256 "${path}" | awk '{print $1}')"
  if [[ "${got_sha}" != "${want_sha}" ]]; then
    rm -f "${path}"
    echo "fetch-goldens.sh: sha256 mismatch for ${label} (got ${got_sha}, pinned ${want_sha}); refused" >&2
    return 1
  fi
  return 0
}

# matches_pin PATH WANT_SHA WANT_BYTES -- true when the file is already the
# pinned object. Nothing is removed: this is the idempotence test, not a gate.
matches_pin() {
  local path="$1" want_sha="$2" want_bytes="$3" got_bytes got_sha
  [[ -f "${path}" ]] || return 1
  got_bytes="$(wc -c < "${path}" | tr -d '[:space:]')"
  [[ "${got_bytes}" == "${want_bytes}" ]] || return 1
  got_sha="$(shasum -a 256 "${path}" | awk '{print $1}')"
  [[ "${got_sha}" == "${want_sha}" ]]
}

# ============================================================================
# --all: stage the whole pinned set
# ============================================================================
if [[ "${FETCH_ALL}" == "1" ]]; then
  if [[ "${have_credentials}" != "1" ]]; then
    cat >&2 <<EOF
fetch-goldens.sh: --all needs R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY.

The pinned tapes and oracles are organizer material and the GETs are
credentialed, so there is no anonymous way to stage them. Ask the organizer for
the credentials and the R2 base, then re-run.
EOF
    exit 1
  fi
  command -v jq >/dev/null 2>&1 \
    || { echo "fetch-goldens.sh: --all reads the contract with jq, and jq is not on PATH" >&2; exit 1; }

  mkdir -p "${OUT_PATH}"
  [[ -d "${OUT_PATH}" ]] \
    || { echo "fetch-goldens.sh: --out is not a directory: ${OUT_PATH}" >&2; exit 1; }

  # Every pin the contract declares: the timed-pool tapes and the per-depth
  # oracles. A depth may reuse another depth's tape, so a key is staged once.
  PINS="$(jq -r '
    [ (.timed_prompt_pool // [])[],
      ((.live_golden_speculative // {}) | to_entries[] | .value) ]
    | map(select(type == "object" and (.r2_path | type) == "string"))
    | unique_by(.r2_path)[]
    | [.r2_path, .sha256, (.bytes | tostring)]
    | @tsv' "${CONTRACT}")"
  [[ -n "${PINS}" ]] \
    || { echo "fetch-goldens.sh: the contract declares no timed_prompt_pool or live_golden_speculative pin to stage" >&2; exit 1; }

  staged=0
  kept=0
  while IFS=$'\t' read -r key want_sha want_bytes; do
    [[ -n "${key}" ]] || continue
    if ! printf '%s' "${want_sha}" | grep -Eq '^[0-9a-f]{64}$'; then
      echo "fetch-goldens.sh: ${key} carries no usable sha256 pin (got '${want_sha}'); the contract is unarmed" >&2
      exit 1
    fi
    if ! printf '%s' "${want_bytes}" | grep -Eq '^[1-9][0-9]*$'; then
      echo "fetch-goldens.sh: ${key} carries no usable byte-count pin (got '${want_bytes}'); the contract is unarmed" >&2
      exit 1
    fi
    dest="${OUT_PATH}/${key##*/}"
    if matches_pin "${dest}" "${want_sha}" "${want_bytes}"; then
      echo "fetch-goldens.sh: ${key##*/} is already staged and matches its pin" >&2
      kept=$((kept + 1))
      continue
    fi
    # Land the bytes beside the destination and move them in only after the pin
    # holds, so a partial transfer is never visible as a staged golden.
    partial="${dest}.partial"
    rm -f "${partial}"
    transport_fetch "${key}" "${partial}" || exit 1
    verify_pin "${partial}" "${want_sha}" "${want_bytes}" "${key}" || exit 1
    chmod 0444 "${partial}"
    rm -f "${dest}"
    mv "${partial}" "${dest}"
    staged=$((staged + 1))
  done <<EOF
${PINS}
EOF

  # The hidden correctness oracle is pinned by DIGEST ONLY -- the contract gives
  # it no key -- so it is resolved among the files just staged rather than
  # fetched by name. A set that carries no file with that digest is not a
  # staged box, and saying so here is cheaper than finding out at measure time.
  HIDDEN_SHA="$(jq -r '.hidden_correctness_golden.sha256 // ""' "${CONTRACT}")"
  HIDDEN_BYTES="$(jq -r '.hidden_correctness_golden.bytes // 0 | tostring' "${CONTRACT}")"
  if printf '%s' "${HIDDEN_SHA}" | grep -Eq '^[0-9a-f]{64}$'; then
    hidden_hit=""
    for staged_file in "${OUT_PATH}"/*.json; do
      [[ -e "${staged_file}" ]] || continue
      if matches_pin "${staged_file}" "${HIDDEN_SHA}" "${HIDDEN_BYTES}"; then
        hidden_hit="${staged_file##*/}"
        break
      fi
    done
    if [[ -z "${hidden_hit}" ]]; then
      echo "fetch-goldens.sh: no staged file carries hidden_correctness_golden (sha256 ${HIDDEN_SHA}, ${HIDDEN_BYTES} bytes); the correctness oracle is not staged" >&2
      exit 1
    fi
    echo "fetch-goldens.sh: hidden_correctness_golden resolves to ${hidden_hit}"
  else
    echo "fetch-goldens.sh: the contract declares no hidden_correctness_golden digest; nothing to resolve" >&2
  fi

  echo "fetch-goldens.sh: staged ${staged} object(s), kept ${kept} already-pinned file(s), in ${OUT_PATH}"
  echo "fetch-goldens.sh: export it as MLXFAST_QWEN38_GOLDEN_DIR, then run tools/ranked-box-preflight.sh"
  exit 0
fi

# ============================================================================
# one object
# ============================================================================
transport_fetch "${R2_PATH}" "${OUT_PATH}" || exit 1
verify_pin "${OUT_PATH}" "${WANT_SHA}" "${WANT_BYTES}" "${R2_PATH}" || exit 1
echo "fetch-goldens.sh: verified ${R2_PATH} -> ${OUT_PATH} (sha256 ${WANT_SHA}, bytes ${WANT_BYTES})"
