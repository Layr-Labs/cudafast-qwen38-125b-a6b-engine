#!/usr/bin/env bash
#
# qwen4exp-golden-reauthor.sh -- re-author this track's goldens on the ds4
# engine, for track qwen3.8-125b-a6b-cuda-v1.
#
# WHY. The pinned goldens, staged on this box in MLXFAST_QWEN38_GOLDEN_DIR,
# were recorded on the previous vLLM engine over the NVFP4 checkpoint. They
# carry that model's provenance and that engine's token streams, so they
# describe no run this repository can make. `official_scoring_enabled` is
# `false` until they are re-authored here (docs/participant-contract.md 5.5,
# docs/qwen38-125b-a6b-port-notes.md 5).
#
# WHAT IT PRODUCES, into --out:
#   prompts/<name>.tokens.json    the prompt ids each case is recorded over
#   goldens/<name>.golden.json    the depth-0 (serial) golden of every public
#                                 pool prompt and of the public correctness
#                                 prompt, with its free-run benchmark oracle
#   goldens/<live>.mtp<D>.golden.json   the per-depth oracle for the live golden
#   mtp<D>-exactness.txt          the serial-vs-mtp per-prompt oracle report
#   index.tsv                     name, sha256, bytes -- the staging index
#   fixture-pins.json             the pin patch, PRINTED AND WRITTEN, NEVER APPLIED
#   negative-control.txt          the perturbed-golden refusal, recorded
#
# THE PROMPTS DO NOT MOVE. A golden binds three things: the weights, the
# prompts, and the prompt SHAs. Only the engine and the checkpoint changed, so
# every pool prompt is taken from the pinned golden that already carries it,
# after that golden is verified against the contract's {bytes, sha256}. The
# public correctness prompt has no Qwen capture yet -- the checked-in
# public_longcopy_gate_english_1024_*.json pair was tokenized for Gemma -- so
# its ids come from tokenizing the checked-in text with the target's own
# tokenizer (`ds4 --dump-tokens --raw-prompt`: the model has no chat
# template, so the prompt file's own tokens are sent unchanged).
#
# THE ENGINE IS DRIVEN THROUGH THE ADAPTER, NOT THE CLI. A golden needs token
# ids and teacher-forced argmax per position; the ds4 CLI emits text. The
# Engine Protocol adapter (`cuda-engine`) exposes the teacher-forced verbs, so
# `benchd record-correctness-golden` drives that, and every worker attaches
# to the one resident engine tools/serve-up.sh booted for the window: the
# weights load ONCE per serve.
#
# TWO SERVES, ONE LOAD EACH. The depth-0 goldens describe the SCORED serial
# serve, which loads no draft head, so they are captured under
# SERVE_UP_SPECULATIVE=0. The per-depth oracle needs the head, so it is
# captured under SERVE_UP_SPECULATIVE=1. Capturing both under one speculative
# serve would save a load and would describe a serve the serial leg never runs.
#
# THE TWO REFUSALS, BY NAME:
#   official-scoring-armed  fixtures/...json `official_scoring_enabled` is true.
#                           Re-authoring goldens while the track scores would
#                           move the oracle under a live leaderboard.
#   engine-pin-mismatch     the ds4 submodule gitlink is not the fixture's
#                           `serve_configuration.engine_pin`. A different engine
#                           commit is a different numeric epoch, and goldens
#                           authored on it would not describe the pinned engine.
#
# THE FIXTURE IS NEVER EDITED. The pin patch is printed. Applying it, and
# flipping `official_scoring_enabled`, is David's call.
#
# Usage:
#   tools/qwen4exp-golden-reauthor.sh --weights DIR [--out DIR] [options]
#   tools/qwen4exp-golden-reauthor.sh --dry-run --weights DIR
#   tools/qwen4exp-golden-reauthor.sh --phase NAME --out DIR      (one step)
#
# Options:
#   --weights DIR      the pinned target snapshot: the GGUF shards and the MTP head
#   --out DIR          work directory (default .build/golden-reauthor/<UTC>)
#   --benchd PATH      benchd (default: resolved by tools/fetch-benchd.sh)
#   --recorder PATH    record-correctness-golden. Default: the copy staged
#                      beside the resolved benchd -- tools/fetch-benchd.sh
#                      puts both in benchd-bin/ when the channel manifest
#                      declares the recorder. An explicit --recorder always
#                      wins, so a hand-built one can still be named.
#   --engine PATH      the cuda-engine adapter
#   --ds4 PATH         the ds4 CLI, for --dump-tokens
#   --depth N          the oracle depth to capture (default 1)
#   --steps N          expected_tokens per case (default 1024)
#   --decode-steps N   free-run oracle length (default 128)
#   --lock PATH        the box GPU lock (default /tmp/mtplx-gpu-exclusive.lock)
#   --lock-wait S      seconds to wait for it (default 600)
#   --phase NAME       run ONE phase against an existing --out: prompts,
#                      capture-serial, capture-mtp, validate, negative-control,
#                      patch. The full run wraps capture-serial and capture-mtp
#                      in tools/serve-up.sh and holds the GPU lock around both.
#   --dry-run          print every command and stop
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null && pwd -P)"
FIXTURE="${REPO_ROOT}/fixtures/qwen3_8_125b_a6b_track.json"
PUBLIC_PROMPT="${REPO_ROOT}/correctness_prompts/public_longcopy_gate_english_1024.txt"
PUBLIC_CASE_NAME="public-longcopy-gate-english-1024"

log()    { printf 'golden-reauthor: %s\n' "$*" >&2; }
refuse() { printf 'golden-reauthor: REFUSE %s: %s\n' "$1" "$2" >&2; exit 1; }
die()    { printf 'golden-reauthor: %s\n' "$*" >&2; exit 1; }

# --- arguments --------------------------------------------------------------
WEIGHTS=""
OUT=""
BENCHD_BIN="${BENCHD:-}"
RECORDER_BIN="${BENCHD_RECORD_GOLDEN_BIN:-}"
ENGINE_BIN=""
DS4_BIN=""
DEPTH=1
STEPS=1024
DECODE_STEPS=128
LOCK_PATH=/tmp/mtplx-gpu-exclusive.lock
LOCK_WAIT=600
PHASE=""
DRY_RUN=0

while [ $# -gt 0 ]; do
  case "$1" in
    --weights)       WEIGHTS="$2"; shift 2 ;;
    --out)           OUT="$2"; shift 2 ;;
    --benchd)        BENCHD_BIN="$2"; shift 2 ;;
    --recorder)      RECORDER_BIN="$2"; shift 2 ;;
    --engine)        ENGINE_BIN="$2"; shift 2 ;;
    --ds4)           DS4_BIN="$2"; shift 2 ;;
    --depth)         DEPTH="$2"; shift 2 ;;
    --steps)         STEPS="$2"; shift 2 ;;
    --decode-steps)  DECODE_STEPS="$2"; shift 2 ;;
    --lock)          LOCK_PATH="$2"; shift 2 ;;
    --lock-wait)     LOCK_WAIT="$2"; shift 2 ;;
    --phase)         PHASE="$2"; shift 2 ;;
    --dry-run)       DRY_RUN=1; shift ;;
    -h|--help)       sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//;$d'; exit 0 ;;
    *)               die "unknown argument $1 (try --help)" ;;
  esac
done

case "${PHASE}" in
  ""|prompts|capture-serial|capture-mtp|validate|negative-control|patch) ;;
  *) die "unknown --phase ${PHASE}" ;;
esac

# --- the commands this run would run ----------------------------------------
# Every externally visible step goes through run(), so --dry-run prints exactly
# what a real run executes and nothing else can slip past the printout.
run() {
  printf '  %q' "$@" >&2
  printf '\n' >&2
  [ "${DRY_RUN}" -eq 1 ] && return 0
  "$@"
}

# run(), for a command whose OWN stderr has to be kept. The redirect belongs to
# the command, never to the run() call: `run cmd 2>FILE` would send run()'s
# printout into FILE -- so the command line would never be printed -- and on a
# dry run it would create FILE in a directory the dry run never made.
run_stderr_to() {
  local file="$1"
  shift
  printf '  %q' "$@" >&2
  printf ' 2> %q\n' "${file}" >&2
  [ "${DRY_RUN}" -eq 1 ] && return 0
  "$@" 2>"${file}"
}

# --- tools ------------------------------------------------------------------
need_tool() {
  if [ "${1#/}" != "$1" ]; then
    [ -x "$1" ] || refuse missing-tool "$1 is required for $2"
  else
    command -v "$1" >/dev/null 2>&1 || refuse missing-tool "$1 is required for $2"
  fi
}
need_tool jq      "reading the track contract"
need_tool python3 "the golden edits and the free-run capture"
need_tool shasum  "pinning every artifact by sha256"

[ -r "${FIXTURE}" ] || die "cannot read the track contract ${FIXTURE}"

TRACK_ID="$(jq -r '.track_id' "${FIXTURE}")"
LIVE_GOLDEN="$(jq -r '.live_golden' "${FIXTURE}")"
PROVENANCE_REPO="$(jq -r '.target.upstream_model_id' "${FIXTURE}")"
PROVENANCE_REV="$(jq -r '.target.upstream_revision' "${FIXTURE}")"
# The R2 key prefix the pin patch below writes. It is an object key, never a
# path in this checkout: the goldens are not in git.
GOLDEN_DIR_REL="correctness_prompts/${TRACK_ID}"
# Where the pinned goldens this re-author READS are staged on this box.
# Required on a real run; a dry run reads nothing.
GOLDEN_STAGE="${MLXFAST_QWEN38_GOLDEN_DIR:-}"
if [ "${DRY_RUN}" -eq 0 ] && [ -z "${GOLDEN_STAGE}" ]; then
  die "MLXFAST_QWEN38_GOLDEN_DIR is unset; the pinned goldens this re-author reads are staged on the box out of band"
fi

# --- refusal 1: the track must not be scoring -------------------------------
ARMED="$(jq -r '.official_scoring_enabled' "${FIXTURE}")"
[ "${ARMED}" = "false" ] || refuse official-scoring-armed \
  "fixtures/qwen3_8_125b_a6b_track.json has official_scoring_enabled=${ARMED}. Re-authoring the goldens is changing the oracle a scored run is judged against; it happens while the track is unarmed, and the arm flip is the LAST step, not a step this script may run under"

# --- refusal 2: the engine pin must be the one the goldens will describe -----
WANT_PIN="$(jq -r '.serve_configuration.engine_pin // empty' "${FIXTURE}")"
TARGET_PIN="$(jq -r '.target.engine_pin // empty' "${FIXTURE}")"
HAVE_PIN="$(jq -r '.fork.sha // empty' "${REPO_ROOT}/ds4/VENDOR.json" 2>/dev/null || true)"
[[ "${WANT_PIN}" =~ ^[0-9a-f]{40}$ ]] || refuse engine-pin-mismatch \
  "serve_configuration.engine_pin is unarmed or malformed in the track contract (got '${WANT_PIN}')"
[ "${WANT_PIN}" = "${TARGET_PIN}" ] || refuse engine-pin-mismatch \
  "serve_configuration.engine_pin (${WANT_PIN}) and target.engine_pin (${TARGET_PIN}) disagree; the contract names two engines"
[ "${HAVE_PIN}" = "${WANT_PIN}" ] || refuse engine-pin-mismatch \
  "ds4/VENDOR.json records vendor base ${HAVE_PIN:-absent} but the contract pins ${WANT_PIN}. Goldens authored on a different engine base describe a different numeric epoch"
# THIS TOOL IS THE ORGANIZER'S, not a participant's. The engine is editable, so
# provenance alone is not enough here: goldens must describe the REFERENCE
# engine, and a dirty ds4/ would author them from somebody's edit. A committed
# edit is not visible from here -- re-author from a clean checkout of a
# vendor-sync'd tree, which is what the refusal says.
[ -z "$(git -C "${REPO_ROOT}" status --porcelain -- ds4 2>/dev/null)" ] || refuse engine-pin-mismatch \
  "the vendored ds4 tree has uncommitted modifications; goldens must be authored on the reference engine, not on a local edit (git status --porcelain -- ds4)"

log "track ${TRACK_ID}, live golden ${LIVE_GOLDEN}, engine pin ${WANT_PIN:0:12}, unarmed"

# --- resolved tools ---------------------------------------------------------
ENGINE_BIN="${ENGINE_BIN:-${REPO_ROOT}/harness/protocol-adapter/target/release/cuda-engine}"
DS4_BIN="${DS4_BIN:-${REPO_ROOT}/.build/ds4/ds4}"
if [ -z "${BENCHD_BIN}" ]; then
  BENCHD_BIN="$("${REPO_ROOT}/tools/fetch-benchd.sh")" \
    || refuse missing-tool "tools/fetch-benchd.sh could not resolve benchd"
fi
# THE RECORDER COMES OFF THE CHANNEL, when the channel carries it.
# tools/fetch-benchd.sh stages record-correctness-golden beside benchd and
# verifies it against the manifest's `binaries` entry, so the copy next to the
# resolved benchd is a channel-attributed binary, not a local build. An
# explicit --recorder (or BENCHD_RECORD_GOLDEN_BIN) still wins: it is how a
# recorder built from the benchd source is named while a channel that predates
# the republish is in front of us.
if [ -z "${RECORDER_BIN}" ]; then
  STAGED_RECORDER="$(dirname "${BENCHD_BIN}")/record-correctness-golden"
  if [ -x "${STAGED_RECORDER}" ]; then
    RECORDER_BIN="${STAGED_RECORDER}"
    log "recorder ${RECORDER_BIN} (staged from the channel beside benchd)"
  fi
fi
[ -n "${RECORDER_BIN}" ] || refuse missing-tool \
  "record-correctness-golden was not found beside benchd ($(dirname "${BENCHD_BIN}")) and was not given (--recorder, or BENCHD_RECORD_GOLDEN_BIN). The dist channel publishes the recorder beside benchd from source_commit >= the mlxfast-bench PR #255 republish; a channel manifest older than that carries the six top-level fields only, declares no \`binaries\` entry for the recorder, and tools/fetch-benchd.sh therefore stages benchd alone. Either republish dist bench-side so the manifest declares the recorder, or build record-correctness-golden from the benchd source at the commit the channel names and pass it with --recorder"

# --- work directory ---------------------------------------------------------
OUT="${OUT:-${REPO_ROOT}/.build/golden-reauthor/$(date -u +%Y%m%dT%H%M%SZ)}"
PROMPT_DIR="${OUT}/prompts"
GOLDEN_OUT="${OUT}/goldens"

POOL_NAMES=()
while IFS= read -r line; do POOL_NAMES+=("${line}"); done < <(
  jq -r '.timed_prompt_pool[].r2_path | split("/") | last | sub("\\.golden\\.json$";"")' "${FIXTURE}"
)
[ "${#POOL_NAMES[@]}" -gt 0 ] || die "the track contract declares no timed_prompt_pool"

# The capture order: every pool prompt, then the public correctness prompt.
CASE_NAMES=("${POOL_NAMES[@]}" "${PUBLIC_CASE_NAME}")

# ============================================================================
# PHASES
# ============================================================================

phase_prompts() {
  log "phase prompts -- verifying each pinned golden, then taking its prompt ids"
  run mkdir -p "${PROMPT_DIR}"
  local i=0 name pinned sha bytes have_sha have_bytes
  for name in "${POOL_NAMES[@]}"; do
    pinned="${GOLDEN_STAGE}/${name}.golden.json"
    sha="$(jq -r ".timed_prompt_pool[${i}].sha256" "${FIXTURE}")"
    bytes="$(jq -r ".timed_prompt_pool[${i}].bytes" "${FIXTURE}")"
    i=$((i + 1))
    if [ "${DRY_RUN}" -eq 0 ]; then
      [ -r "${pinned}" ] || die "the pinned golden ${pinned} is missing; its prompt is the input this re-author reads"
      have_bytes="$(wc -c < "${pinned}" | tr -d '[:space:]')"
      [ "${have_bytes}" = "${bytes}" ] || die "${name}: byte count ${have_bytes}, contract pins ${bytes}"
      have_sha="$(shasum -a 256 "${pinned}" | awk '{print $1}')"
      [ "${have_sha}" = "${sha}" ] || die "${name}: sha256 ${have_sha}, contract pins ${sha}"
    fi
    run sh -c "'${REPO_ROOT}/tools/qwen4exp-golden-edit.py' prompt-tokens '${pinned}' > '${PROMPT_DIR}/${name}.tokens.json'"
  done
  log "phase prompts -- tokenizing the public correctness prompt with the target's own tokenizer"
  need_tool "${DS4_BIN}" "tokenizing the public correctness prompt (ds4 --dump-tokens)"
  local shard
  shard="$(printf '%s\n' "${WEIGHTS}"/*-00001-of-*.gguf | head -1)"
  run sh -c "'${DS4_BIN}' -m '${shard}' --dump-tokens --raw-prompt --prompt-file '${PUBLIC_PROMPT}' > '${PROMPT_DIR}/${PUBLIC_CASE_NAME}.dump.txt'"
  run sh -c "'${REPO_ROOT}/tools/qwen4exp-golden-edit.py' head-tokens --dump '${PROMPT_DIR}/${PUBLIC_CASE_NAME}.dump.txt' --count 1024 > '${PROMPT_DIR}/${PUBLIC_CASE_NAME}.tokens.json'"
}

phase_capture_serial() {
  log "phase capture-serial -- depth 0 goldens for ${#CASE_NAMES[@]} prompts, under the serial serve"
  run mkdir -p "${GOLDEN_OUT}"
  local name
  for name in "${CASE_NAMES[@]}"; do
    run "${RECORDER_BIN}" \
      --backend live \
      --worker-bin "${ENGINE_BIN}" \
      --weights "${WEIGHTS}" \
      --prompt-tokens "${PROMPT_DIR}/${name}.tokens.json" \
      --case-name "${name}" \
      --steps "${STEPS}" \
      --benchmark-free-run \
      --benchmark-steps "${DECODE_STEPS}" \
      --track "${TRACK_ID}" \
      --model-provenance-repo "${PROVENANCE_REPO}" \
      --model-provenance-rev "${PROVENANCE_REV}" \
      --out "${GOLDEN_OUT}/${name}.golden.json"
  done
}

phase_capture_mtp() {
  log "phase capture-mtp -- the depth-${DEPTH} oracle and the serial-vs-mtp exactness report, under the speculative serve"
  run sh -c "'${REPO_ROOT}/tools/ds4/free-run-capture.py' --engine '${ENGINE_BIN}' --seed-from '${GOLDEN_OUT}/${LIVE_GOLDEN}.golden.json' --depth '${DEPTH}' --steps '${DECODE_STEPS}' > '${OUT}/mtp${DEPTH}-capture.json'"
  run "${REPO_ROOT}/tools/qwen4exp-golden-edit.py" graft-decode-oracle \
    --golden "${GOLDEN_OUT}/${LIVE_GOLDEN}.golden.json" \
    --capture "${OUT}/mtp${DEPTH}-capture.json" \
    --out "${GOLDEN_OUT}/${LIVE_GOLDEN}.mtp${DEPTH}.golden.json"
  # The oracle proper: every prompt run serial and mtp, token streams compared.
  # A mismatch is a RESULT, not a failure of this run -- it is what the
  # per-depth golden exists to record -- so the report is kept either way.
  run sh -c "'${REPO_ROOT}/tools/ds4/mtp-exactness-gate.py' --engine '${ENGINE_BIN}' --depth '${DEPTH}' --steps '${DECODE_STEPS}' '${GOLDEN_OUT}'/*.golden.json > '${OUT}/mtp${DEPTH}-exactness.txt' 2>&1 || true"
  run sh -c "cat '${OUT}/mtp${DEPTH}-exactness.txt'"
}

phase_validate() {
  log "phase validate -- every artifact through the scoring loader, pinned to its own bytes"
  local f sha bytes
  run sh -c ": > '${OUT}/index.tsv'"
  for f in "${GOLDEN_OUT}"/*.golden.json; do
    if [ "${DRY_RUN}" -eq 0 ]; then
      sha="$(shasum -a 256 "${f}" | awk '{print $1}')"
      bytes="$(wc -c < "${f}" | tr -d '[:space:]')"
    else
      f="${GOLDEN_OUT}/<each>.golden.json"; sha='<sha256>'; bytes='<bytes>'
    fi
    run "${BENCHD_BIN}" validate-golden \
      --golden "${f}" \
      --track "${TRACK_ID}" \
      --golden-sha256 "${sha}" \
      --golden-bytes "${bytes}" \
      --contract "${FIXTURE}"
    run sh -c "printf '%s\t%s\t%s\n' \"\$(basename '${f}' .golden.json)\" '${sha}' '${bytes}' >> '${OUT}/index.tsv'"
    if [ "${DRY_RUN}" -eq 1 ]; then break; fi
  done
}

phase_negative_control() {
  log "phase negative-control -- a one-token edit of the live golden must be REJECTED"
  local live="${GOLDEN_OUT}/${LIVE_GOLDEN}.golden.json"
  local bad="${OUT}/negative-control.golden.json"
  local sha bytes rc
  run "${REPO_ROOT}/tools/qwen4exp-golden-edit.py" perturb-one-token --golden "${live}" --out "${bad}"
  if [ "${DRY_RUN}" -eq 0 ]; then
    sha="$(shasum -a 256 "${live}" | awk '{print $1}')"
    bytes="$(wc -c < "${live}" | tr -d '[:space:]')"
  else
    sha='<the live golden sha256>'; bytes='<the live golden bytes>'
  fi
  set +e
  run_stderr_to "${OUT}/negative-control.txt" \
    "${BENCHD_BIN}" validate-golden \
    --golden "${bad}" \
    --track "${TRACK_ID}" \
    --golden-sha256 "${sha}" \
    --golden-bytes "${bytes}" \
    --contract "${FIXTURE}"
  rc=$?
  set -e
  [ "${DRY_RUN}" -eq 1 ] && return 0
  cat "${OUT}/negative-control.txt" >&2
  [ "${rc}" -eq 1 ] || die "NEGATIVE CONTROL FAILED: validate-golden exited ${rc} on a golden with one changed token; it must exit 1 (reject). The pin does not bind the tokens, so the re-authored set proves nothing"
  log "negative control held: one changed token, byte count unchanged, validate-golden rejected on the integrity pin"
}

phase_patch() {
  log "phase patch -- the pins. THIS SCRIPT DOES NOT EDIT THE FIXTURE"
  if [ "${DRY_RUN}" -eq 1 ]; then
    printf '  # reads %s and writes %s\n' "${OUT}/index.tsv" "${OUT}/fixture-pins.json" >&2
    return 0
  fi
  local live_sha live_bytes mtp_sha mtp_bytes
  live_sha="$(shasum -a 256 "${GOLDEN_OUT}/${LIVE_GOLDEN}.golden.json" | awk '{print $1}')"
  live_bytes="$(wc -c < "${GOLDEN_OUT}/${LIVE_GOLDEN}.golden.json" | tr -d '[:space:]')"
  mtp_sha="$(shasum -a 256 "${GOLDEN_OUT}/${LIVE_GOLDEN}.mtp${DEPTH}.golden.json" | awk '{print $1}')"
  mtp_bytes="$(wc -c < "${GOLDEN_OUT}/${LIVE_GOLDEN}.mtp${DEPTH}.golden.json" | tr -d '[:space:]')"
  {
    printf '{\n  "timed_prompt_pool": [\n'
    local i=0 name sha bytes comma
    for name in "${POOL_NAMES[@]}"; do
      sha="$(shasum -a 256 "${GOLDEN_OUT}/${name}.golden.json" | awk '{print $1}')"
      bytes="$(wc -c < "${GOLDEN_OUT}/${name}.golden.json" | tr -d '[:space:]')"
      i=$((i + 1))
      comma=','; [ "${i}" -eq "${#POOL_NAMES[@]}" ] && comma=''
      printf '    {"r2_path": "%s/%s.golden.json", "sha256": "%s", "bytes": %s}%s\n' \
        "${GOLDEN_DIR_REL}" "${name}" "${sha}" "${bytes}" "${comma}"
    done
    printf '  ],\n'
    printf '  "hidden_correctness_golden": {"sha256": "%s", "bytes": %s},\n' "${live_sha}" "${live_bytes}"
    printf '  "live_golden_speculative": {\n    "mtp%s": {"r2_path": "%s/%s.mtp%s.golden.json", "sha256": "%s", "bytes": %s}\n  }\n' \
      "${DEPTH}" "${GOLDEN_DIR_REL}" "${LIVE_GOLDEN}" "${DEPTH}" "${mtp_sha}" "${mtp_bytes}"
    printf '}\n'
  } > "${OUT}/fixture-pins.json"
  cat "${OUT}/fixture-pins.json"
  cat >&2 <<EOF

golden-reauthor: the patch above is NOT applied. Publishing the goldens to R2
at the ${GOLDEN_DIR_REL}/ keys, writing these pins into
fixtures/qwen3_8_125b_a6b_track.json and flipping official_scoring_enabled are
David's calls. The goldens never enter this repository.

golden-reauthor: TWO PINS THIS RUN CANNOT WRITE.
  * mtp2 and mtp3. The engine implements depths 1 to 3 since e2f86b7, so those
    oracle entries stay as the contract already carries them.
  * baseline_prefill_seconds_per_token / baseline_decode_seconds_per_token.
    record-correctness-golden OMITS both keys -- they are skip_serializing_if
    Option::is_none (mlxfast-bench
    crates/benchctl/src/bin/record-correctness-golden.rs:478-479 at 44b220a --
    the crate carried the old name at that commit), so
    the golden it writes carries no benchmark baseline at all. Today's pinned
    goldens DO carry the pair (botany: 0.0004879835673828125 prefill,
    0.06451959972265625 decode), so re-authoring LOSES it, and an official run
    REQUIRES it from the golden or from --baseline-*
    (crates/benchd/src/main.rs resolve_paired_baselines). The re-authored set
    is not scorable until the calibration lane measures the pair and stamps it.
EOF
}

# ============================================================================
# DISPATCH
# ============================================================================

if [ -n "${PHASE}" ]; then
  case "${PHASE}" in
    prompts)          phase_prompts ;;
    capture-serial)   phase_capture_serial ;;
    capture-mtp)      phase_capture_mtp ;;
    validate)         phase_validate ;;
    negative-control) phase_negative_control ;;
    patch)            phase_patch ;;
  esac
  exit 0
fi

# The full run. --weights is required: it names the artifact both serves load.
[ -n "${WEIGHTS}" ] || die "--weights is required (the pinned target snapshot: the GGUF shards and the MTP head)"
[ "${DRY_RUN}" -eq 1 ] || [ -d "${WEIGHTS}" ] || die "the target snapshot directory is missing: ${WEIGHTS}"

SELF="${REPO_ROOT}/tools/qwen4exp-golden-reauthor.sh"
COMMON=(--out "${OUT}" --weights "${WEIGHTS}" --engine "${ENGINE_BIN}" --ds4 "${DS4_BIN}"
        --benchd "${BENCHD_BIN}" --recorder "${RECORDER_BIN}" --depth "${DEPTH}"
        --steps "${STEPS}" --decode-steps "${DECODE_STEPS}")

log "work directory ${OUT}"
run mkdir -p "${OUT}"

phase_prompts

# --- the GPU lock -----------------------------------------------------------
# Both serves hold GPU memory, so the whole capture window sits inside one hold
# of the box-wide lock. serve-up.sh boots the resident INSIDE this window and
# never takes the lock itself.
if [ "${DRY_RUN}" -eq 1 ]; then
  log "the full run would now hold ${LOCK_PATH} (flock -w ${LOCK_WAIT}) around both serves"
else
  need_tool flock "holding the box GPU lock around both serves"
  exec 9>>"${LOCK_PATH}" || refuse lock-unwritable "cannot open the GPU lock ${LOCK_PATH}"
  flock -w "${LOCK_WAIT}" 9 \
    || refuse gpu-lock-busy "${LOCK_PATH} is held by another run after ${LOCK_WAIT}s. Nothing has been loaded"
  log "GPU lock held: ${LOCK_PATH}"
fi

log "serve 1 of 2 -- serial (SERVE_UP_SPECULATIVE=0), one weight load"
run env "SERVE_UP_WEIGHTS_DIR=${WEIGHTS}" SERVE_UP_SPECULATIVE=0 \
  "${REPO_ROOT}/tools/serve-up.sh" "${SELF}" --phase capture-serial "${COMMON[@]}"
[ "${DRY_RUN}" -eq 1 ] && phase_capture_serial

log "serve 2 of 2 -- mtp${DEPTH} (SERVE_UP_SPECULATIVE=1), one weight load"
run env "SERVE_UP_WEIGHTS_DIR=${WEIGHTS}" SERVE_UP_SPECULATIVE=1 "SERVE_UP_SPEC_DRAFT_LEN=${DEPTH}" \
  "${REPO_ROOT}/tools/serve-up.sh" "${SELF}" --phase capture-mtp "${COMMON[@]}"
[ "${DRY_RUN}" -eq 1 ] && phase_capture_mtp

phase_validate
phase_negative_control
phase_patch

log "done. Artifacts in ${OUT}"
