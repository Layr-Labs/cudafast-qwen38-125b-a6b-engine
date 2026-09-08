#!/usr/bin/env bash
#
# qwen38-125b-a6b-measure-and-score.sh -- benchmark.json's benchmarkCommand /
# preSubmitCommand entry point for track qwen3.8-125b-a6b-cuda-v1.
#
# It drives `benchd iterate --mode official` -- the SOLE scored path -- against
# the track's LIVE golden. `iterate --mode official` is timed-first, spawns the
# sandboxed runtime worker, runs the full correctness set, gates on the official
# floor/bands, and SEALS the artifact itself: it writes score.json in the
# {score, metrics} shape Yukon's ScoreFileSchema reads, its `.sha256` sidecar,
# and the per-mode benchmark-integrity sidecar. There is NO results.json to
# convert -- benchd is the sole writer of the sealed score, so this script no
# longer post-processes one. This script is TRUSTED-side tooling: it is NOT in
# editablePaths, so a submission cannot rewrite the measurement pipeline from
# inside its own archive.
#
# WHY iterate, not measure-job. `benchd measure-job` is RETIRED -- it encoded
# the paired/8-tape/baseline-workspace design that predates the David 2026-08-27
# single-stream ruling. The scored model now is SINGLE-LEG (Laguna): one leg
# (the submission's declared spec -- serial at the MTP-0 launch reference)
# measured over the ONE live golden, scored against the pinned OFFICIAL_BASELINE
# constants benchd resolves from the track id. composite =
# prefill_gain^0.25 * decode_gain^0.75 (score_default_weights). No separate
# baseline workspace is measured; the golden's paired baselines / the platform
# OFFICIAL_BASELINE are the control.
#
# The LIVE golden is read FROM the fixture, never hardcoded: the fixture's
# `live_golden` names it and its `timed_prompt_pool[]` entry pins it
# ({sha256, bytes}). A future live_golden rotation is picked up here with no edit
# to this script. The pins are forwarded to benchd as
# --golden-sha256/--golden-bytes, which re-verifies the raw bytes BEFORE parse
# and refuses on any mismatch (the integrity pin). --contract carries the arm
# gate: benchd refuses, pre-GPU, to seal an official artifact unless the fixture
# declares official_scoring_enabled: true.
#
# The design rationale lives in docs/qwen38-125b-a6b-port-notes.md section 9.
#
# Usage:
#   ./tools/qwen38-125b-a6b-measure-and-score.sh                 # full measure + seal
#   ./tools/qwen38-125b-a6b-measure-and-score.sh --preflight-only # pre-GPU dry-run, no engine
#
# Env:
#   MLXFAST_SCORE_PATH               Where benchd seals the {score, metrics} JSON
#                                     (--score-path). Defaults to score.json
#                                     (benchmark.json always sets this explicitly).
#                                     The .sha256 sidecar and benchmark-integrity
#                                     sidecar are sealed beside it by benchd.
#   MLXFAST_QWEN38_GOLDEN_DIR         Directory holding the staged timed-pool
#                                     golden files. The LIVE golden is resolved
#                                     from it as <live_golden>.golden.json (the
#                                     basename of the fixture entry's r2_path).
#                                     Box-only, staged out of band and
#                                     pin-verified by tools/ranked-box-preflight.sh.
#   MLXFAST_ENGINE_BIN               The Engine Protocol v1 adapter benchd spawns
#                                     (`--engine`). Default: the fixed staged path
#                                     tools/stage-cuda-engine.sh writes,
#                                     .build/release/mlxfast-runtime-worker. The
#                                     adapter opens the ds4 engine in-process
#                                     (weights shared by the owner booted below).
#   MLXFAST_WEIGHTS_PATH             Weights directory, passed as `--weights`.
#                                     Default: the on-box GGUF target snapshot
#                                     ./setup.sh verifies in place,
#                                     MLXFAST_TARGET_SNAPSHOT_DIR.
#   BENCHD                           Path to the benchd binary. Default: the
#                                     binary ./tools/fetch-benchd.sh resolves from
#                                     the dist channel and verifies against the
#                                     channel's benchd.manifest.json
#                                     (benchd-bin/benchd). No cargo build: the
#                                     ranked box has no Rust toolchain, so benchd
#                                     ships prebuilt and pinned by sha256.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${SCRIPT_DIR}"

PREFLIGHT_ONLY=0
for arg in "$@"; do
  case "${arg}" in
    --preflight-only) PREFLIGHT_ONLY=1 ;;
    *)
      echo "qwen38-125b-a6b-measure-and-score.sh: unrecognized argument: ${arg}" >&2
      exit 2
      ;;
  esac
done

# jq is required for reading the fixture (live_golden + pins) and the trackId.
if ! command -v jq >/dev/null 2>&1; then
  echo "qwen38-125b-a6b-measure-and-score.sh: jq is required (it reads the fixture pins and trackId)." >&2
  exit 1
fi

# Resolve the PINNED benchd. benchd ships as a channel PREBUILT
# (./tools/fetch-benchd.sh, verified against the channel's
# benchd.manifest.json; the sha pin is retired -- David ruling 2026-08-27, so
# measurement fixes ship bench-side with no engine commit). fetch-benchd.sh
# accepts an already-present benchd-bin/benchd whose sha256 and bytes match the
# manifest beside it -- the offline path on the ranked box, which has no Rust
# toolchain -- and otherwise downloads and verifies it. It never yields an
# unverified binary, so this refuses rather than measuring against unpinned
# scoring code.
#
# BENCHD= from the caller is honoured and NOT hash-checked: that is a
# deliberate "use this other binary" for benchd development.
if [[ -z "${BENCHD:-}" ]]; then
  BENCHD="$("${SCRIPT_DIR}/tools/fetch-benchd.sh")"
fi
if [[ ! -x "${BENCHD}" ]]; then
  echo "qwen38-125b-a6b-measure-and-score.sh: benchd not found at ${BENCHD}." >&2
  echo "  fetch it: ./tools/fetch-benchd.sh   (resolves the dist channel, verifies the manifest)" >&2
  echo "  or set BENCHD to an existing binary." >&2
  exit 1
fi

CONTRACT="${SCRIPT_DIR}/fixtures/qwen3_8_125b_a6b_track.json"
if [[ ! -f "${CONTRACT}" ]]; then
  echo "qwen38-125b-a6b-measure-and-score.sh: track contract fixture missing: ${CONTRACT}" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# THE TRACK ID. `benchd iterate` REQUIRES MLXFAST_QWEN_MTP_TRACK_ID in every
# mode -- its `-{platform}-v{N}` suffix keys the OFFICIAL_BASELINE pair and the
# acceptance bands, and the official path resolves it BEFORE the timed run. The
# value is benchmark.json's `trackId` (the PLATFORM id). benchd treats
# env != contract as a hard error, so a caller that pre-set a DIFFERENT value is
# refused here rather than silently overridden.
MANIFEST_TRACK_ID="$(jq -r '.trackId // empty' "${SCRIPT_DIR}/benchmark.json" 2>/dev/null || true)"
if [[ -z "${MANIFEST_TRACK_ID}" ]]; then
  echo "qwen38-125b-a6b-measure-and-score.sh: benchmark.json carries no trackId; benchd iterate requires MLXFAST_QWEN_MTP_TRACK_ID and this script will not guess one." >&2
  exit 1
fi
if [[ -n "${MLXFAST_QWEN_MTP_TRACK_ID:-}" && "${MLXFAST_QWEN_MTP_TRACK_ID}" != "${MANIFEST_TRACK_ID}" ]]; then
  echo "qwen38-125b-a6b-measure-and-score.sh: MLXFAST_QWEN_MTP_TRACK_ID is set to '${MLXFAST_QWEN_MTP_TRACK_ID}' but benchmark.json trackId is '${MANIFEST_TRACK_ID}'; the track id is ONE value and this script will not override either." >&2
  exit 1
fi
export MLXFAST_QWEN_MTP_TRACK_ID="${MANIFEST_TRACK_ID}"

# ---------------------------------------------------------------------------
# THE LIVE GOLDEN + ITS PIN, read FROM the fixture (never hardcoded). The
# fixture's `live_golden` names the single golden this track scores over; its
# timed_prompt_pool[] entry -- matched by the basename of its r2_path
# (<live_golden>.golden.json) -- carries the {sha256, bytes} pin. Resolving the
# name first and then looking up the pin means a future live_golden rotation
# needs no edit here.
LIVE_GOLDEN_NAME="$(jq -r '.live_golden // empty' "${CONTRACT}")"
if [[ -z "${LIVE_GOLDEN_NAME}" ]]; then
  echo "qwen38-125b-a6b-measure-and-score.sh: fixture declares no live_golden; there is no golden to score over." >&2
  exit 1
fi

LIVE_GOLDEN_BASENAME="${LIVE_GOLDEN_NAME}.golden.json"
# The pool entry whose r2_path ends in /<live_golden>.golden.json. Its pin is the
# integrity pin forwarded to benchd.
live_golden_entry="$(jq -c --arg base "/${LIVE_GOLDEN_BASENAME}" \
  'first(.timed_prompt_pool[] | select(.r2_path | endswith($base)))' "${CONTRACT}")"
if [[ -z "${live_golden_entry}" || "${live_golden_entry}" == "null" ]]; then
  echo "qwen38-125b-a6b-measure-and-score.sh: live_golden '${LIVE_GOLDEN_NAME}' has no timed_prompt_pool entry (looked for an r2_path ending in /${LIVE_GOLDEN_BASENAME})." >&2
  exit 1
fi
LIVE_GOLDEN_SHA256="$(printf '%s' "${live_golden_entry}" | jq -r '.sha256 // empty')"
LIVE_GOLDEN_BYTES="$(printf '%s' "${live_golden_entry}" | jq -r '.bytes // empty')"
if ! printf '%s' "${LIVE_GOLDEN_SHA256}" | grep -Eq '^[0-9a-f]{64}$' \
  || ! printf '%s' "${LIVE_GOLDEN_BYTES}" | grep -Eq '^[1-9][0-9]*$'; then
  echo "qwen38-125b-a6b-measure-and-score.sh: live_golden '${LIVE_GOLDEN_NAME}' is unarmed or malformed (sha256='${LIVE_GOLDEN_SHA256}', bytes='${LIVE_GOLDEN_BYTES}'); nothing can be pin-verified against it." >&2
  exit 1
fi

# PER-DEPTH TIMED ORACLE. The timed leg replays the engine's FREE-RUN under the
# declared speculative spec, and a free-run trajectory forks from the serial one
# at near-tie argmaxes (docs/participant-contract.md section 11.4(c)). So the
# timed oracle must be authored under the SAME spec the leg replays: the serial
# golden resolved above is the launch reference (MTP-0), and each permitted MTP
# draft depth has its OWN golden. The spec label is derived through the one
# trusted deriver that ALSO boots the serve (tools/spec-declaration.sh describe
# -> serial|mtp1|mtp2|mtp3), so the oracle can never disagree with the draft
# depth that actually ran. This overrides ONLY the timed-oracle file and its pin;
# the correctness gate is untouched, because every per-depth golden carries the
# SAME serial cases[] oracle, so token fidelity is still measured against serial.
SPEC_DESC="$("${SCRIPT_DIR}/tools/spec-declaration.sh" describe)"
if [[ "${SPEC_DESC}" != "serial" ]]; then
  # live_golden_speculative maps a spec label to its per-depth golden's
  # {r2_path, sha256, bytes}. A declared depth with no entry is REFUSED -- there
  # is no timed oracle authored for it, and scoring it against the serial oracle
  # would fail at step 0 (the very divergence this map exists to resolve). The
  # depth is already envelope-checked by spec-declaration.sh; this is the
  # oracle-existence half of the same fail-closed posture.
  spec_entry="$(jq -c --arg k "${SPEC_DESC}" '.live_golden_speculative[$k] // empty' "${CONTRACT}")"
  if [[ -z "${spec_entry}" || "${spec_entry}" == "null" ]]; then
    echo "qwen38-125b-a6b-measure-and-score.sh: declared spec '${SPEC_DESC}' has no live_golden_speculative entry in ${CONTRACT}; no timed oracle is authored for that draft depth. Refusing rather than scoring it against the serial oracle." >&2
    exit 1
  fi
  LIVE_GOLDEN_BASENAME="$(printf '%s' "${spec_entry}" | jq -r '.r2_path // empty')"
  LIVE_GOLDEN_BASENAME="${LIVE_GOLDEN_BASENAME##*/}"
  LIVE_GOLDEN_SHA256="$(printf '%s' "${spec_entry}" | jq -r '.sha256 // empty')"
  LIVE_GOLDEN_BYTES="$(printf '%s' "${spec_entry}" | jq -r '.bytes // empty')"
  if ! printf '%s' "${LIVE_GOLDEN_SHA256}" | grep -Eq '^[0-9a-f]{64}$' \
    || ! printf '%s' "${LIVE_GOLDEN_BYTES}" | grep -Eq '^[1-9][0-9]*$' \
    || [[ -z "${LIVE_GOLDEN_BASENAME}" || "${LIVE_GOLDEN_BASENAME}" != *.golden.json ]]; then
    echo "qwen38-125b-a6b-measure-and-score.sh: live_golden_speculative['${SPEC_DESC}'] is unarmed or malformed (file='${LIVE_GOLDEN_BASENAME}', sha256='${LIVE_GOLDEN_SHA256}', bytes='${LIVE_GOLDEN_BYTES}')." >&2
    exit 1
  fi
  echo "qwen38-125b-a6b-measure-and-score.sh: declared spec ${SPEC_DESC}; timed oracle is ${LIVE_GOLDEN_BASENAME} (sha256 ${LIVE_GOLDEN_SHA256}, ${LIVE_GOLDEN_BYTES} bytes)" >&2
fi

# ---------------------------------------------------------------------------
# The LIVE golden FILE, resolved from the existing box staging convention:
# MLXFAST_QWEN38_GOLDEN_DIR holds the staged pool, and the live golden is
# <live_golden>.golden.json within it. The goldens are box-only, staged out of
# band and pin-verified by tools/ranked-box-preflight.sh.
GOLDEN_DIR="${MLXFAST_QWEN38_GOLDEN_DIR:-}"
LIVE_GOLDEN_PATH=""
if [[ -n "${GOLDEN_DIR}" ]]; then
  LIVE_GOLDEN_PATH="${GOLDEN_DIR}/${LIVE_GOLDEN_BASENAME}"
fi

# ---------------------------------------------------------------------------
# --preflight-only: a pre-GPU DRY RUN that exercises the arm gate and (when the
# golden is staged) the integrity pin, WITHOUT spawning the engine or loading the
# model. No score is written. The full run below enforces both refusals through
# benchd itself; this is the loud early gate.
if [[ "${PREFLIGHT_ONLY}" == "1" ]]; then
  # ARM GATE (mirror of benchd's enforce_official_scoring_enabled): an official
  # run refuses unless the fixture declares official_scoring_enabled: true. false
  # and ABSENT both refuse -- an absent arm state is not an armed one. benchd
  # enforces this itself on the real run; mirroring it here makes the pre-GPU
  # dry-run honest rather than a rubber stamp.
  armed="$(jq -r '.official_scoring_enabled // false' "${CONTRACT}")"
  if [[ "${armed}" != "true" ]]; then
    echo "qwen38-125b-a6b-measure-and-score.sh: preflight REFUSING -- fixtures/qwen3_8_125b_a6b_track.json does not declare official_scoring_enabled: true (got '${armed}')." >&2
    echo "  benchd refuses to seal an official scoring artifact for an unarmed track; this dry-run mirrors that refusal." >&2
    exit 1
  fi
  echo "qwen38-125b-a6b-measure-and-score.sh: preflight -- arm gate OK (official_scoring_enabled: true), live_golden ${LIVE_GOLDEN_NAME} (sha256 ${LIVE_GOLDEN_SHA256}, ${LIVE_GOLDEN_BYTES} bytes)" >&2

  # INTEGRITY PIN: when the live golden is staged, validate-golden re-verifies its
  # raw bytes against the pin AND load-validates it (reference-model pin from
  # --contract), with NO engine spawned. When it is not staged (e.g. an off-box
  # pre-submit), the pin is still enforced by benchd on the real run -- the old
  # measure-job --preflight-only never checked the golden either, so this is not a
  # weakening; it is a stronger check whenever the golden is present.
  if [[ -n "${LIVE_GOLDEN_PATH}" && -f "${LIVE_GOLDEN_PATH}" ]]; then
    exec "${BENCHD}" validate-golden \
      --golden "${LIVE_GOLDEN_PATH}" \
      --golden-sha256 "${LIVE_GOLDEN_SHA256}" \
      --golden-bytes "${LIVE_GOLDEN_BYTES}" \
      --contract "${CONTRACT}"
  fi
  echo "qwen38-125b-a6b-measure-and-score.sh: preflight -- live golden not staged (MLXFAST_QWEN38_GOLDEN_DIR unset or ${LIVE_GOLDEN_BASENAME} absent); benchd re-verifies the {sha256, bytes} pin on the real run." >&2
  exit 0
fi

# ---------------------------------------------------------------------------
# A REAL RUN needs the staged live golden.
if [[ -z "${GOLDEN_DIR}" || ! -d "${GOLDEN_DIR}" ]]; then
  cat >&2 <<EOF
qwen38-125b-a6b-measure-and-score.sh: MLXFAST_QWEN38_GOLDEN_DIR is unset or missing.
  The live golden (${LIVE_GOLDEN_BASENAME}) is staged onto the box out of band
  (docs/qwen38-125b-a6b-port-notes.md section 7) and this job holds no credential
  to fetch it. There is nothing this script can do here except refuse.
EOF
  exit 1
fi
if [[ ! -f "${LIVE_GOLDEN_PATH}" ]]; then
  echo "qwen38-125b-a6b-measure-and-score.sh: live golden not found at ${LIVE_GOLDEN_PATH}" >&2
  echo "  live_golden is '${LIVE_GOLDEN_NAME}' (fixtures/qwen3_8_125b_a6b_track.json); stage ${LIVE_GOLDEN_BASENAME} into MLXFAST_QWEN38_GOLDEN_DIR." >&2
  exit 1
fi

# --engine. `iterate` spawns the Engine Protocol v1 adapter, which connects to
# the ds4 engine in-process. Default to the fixed staged path
# tools/stage-cuda-engine.sh writes, so the documented invocation is
# self-sufficient; MLXFAST_ENGINE_BIN overrides it for engine development.
ENGINE_BIN="${MLXFAST_ENGINE_BIN:-${SCRIPT_DIR}/.build/release/mlxfast-runtime-worker}"
if [[ ! -x "${ENGINE_BIN}" ]]; then
  echo "qwen38-125b-a6b-measure-and-score.sh: engine binary missing or not executable: ${ENGINE_BIN}" >&2
  echo "  build + stage it (./setup.sh, or tools/stage-cuda-engine.sh), or set MLXFAST_ENGINE_BIN." >&2
  exit 1
fi

# --weights. benchd resolves the target weights from `--weights`. Default to the
# on-box GGUF target snapshot ./setup.sh verifies in place
# (MLXFAST_TARGET_SNAPSHOT_DIR); MLXFAST_WEIGHTS_PATH overrides.
WEIGHTS_PATH="${MLXFAST_WEIGHTS_PATH:-${MLXFAST_TARGET_SNAPSHOT_DIR:-}}"
if [[ -z "${WEIGHTS_PATH}" || ! -d "${WEIGHTS_PATH}" ]]; then
  echo "qwen38-125b-a6b-measure-and-score.sh: the target weights directory is missing or unset: '${WEIGHTS_PATH}'" >&2
  echo "  Set MLXFAST_TARGET_SNAPSHOT_DIR to the on-box GGUF target snapshot (./setup.sh verifies it), or MLXFAST_WEIGHTS_PATH to override." >&2
  exit 1
fi

SCORE_PATH="${MLXFAST_SCORE_PATH:-score.json}"

# THE SOLE SCORED PATH: benchd iterate --mode official over the live golden.
# benchd resolves the OFFICIAL_BASELINE (keyed by MLXFAST_QWEN_MTP_TRACK_ID) as
# the paired control -- there is NO separate baseline workspace. The score is the
# composite prefill_gain^0.25 * decode_gain^0.75. benchd SEALS score.json (the
# {score, metrics} shape Yukon reads), score.json.sha256, and the
# benchmark-integrity sidecar itself; this script does no post-conversion.
#
# --golden-sha256/--golden-bytes are the INTEGRITY PIN (benchd re-verifies the
# raw bytes before parse and refuses on mismatch). --contract carries the ARM
# GATE (benchd refuses, pre-GPU, unless official_scoring_enabled: true).
iterate_official=(
  "${BENCHD}" iterate
  --engine "${ENGINE_BIN}"
  --weights "${WEIGHTS_PATH}"
  --golden "${LIVE_GOLDEN_PATH}"
  --mode official
  --score-path "${SCORE_PATH}"
  --golden-sha256 "${LIVE_GOLDEN_SHA256}"
  --golden-bytes "${LIVE_GOLDEN_BYTES}"
  --contract "${CONTRACT}"
)

# The scored leg must REQUEST the declared spec. benchd's single-leg official
# path sends free_decode_begin with no spec unless it is told one, and the
# adapter then resolves serial, so a declared MTP leg would be measured as
# serial while its oracle is the MTP one. benchd carries the declared depth
# with --mtp-depth N; a benchd that does not know the flag cannot run an MTP
# leg, and that is a refusal, never a silent serial run.
if [[ "${SPEC_DESC}" != "serial" ]]; then
  DECLARED_DEPTH="$("${SCRIPT_DIR}/tools/spec-declaration.sh" draft-len)"
  if "${BENCHD}" iterate --help 2>&1 | grep -q -- '--mtp-depth'; then
    iterate_official+=( --mtp-depth "${DECLARED_DEPTH}" )
    echo "qwen38-125b-a6b-measure-and-score.sh: benchd will request the declared spec (--mtp-depth ${DECLARED_DEPTH})" >&2
  else
    echo "qwen38-125b-a6b-measure-and-score.sh: the pinned benchd does not accept --mtp-depth, so it cannot request the declared ${SPEC_DESC} leg; refusing rather than measuring a serial leg against the ${SPEC_DESC} oracle" >&2
    exit 1
  fi
fi

# tools/serve-up.sh boots ONE ds4-resident for this window -- the process that
# owns the weights -- and exports the engine environment every spawned
# cuda-engine reads (DS4_RESIDENT_SOCKET, DS4_MODEL, DS4_MTP_PATH,
# DS4_MTP_DRAFT_TOKENS, ...). benchd spawns a worker per phase and each one
# CONNECTS instead of loading, so the weights load once for the whole window.
# A caller that already exported DS4_MODEL (a diagnostics run) is honoured
# as-is and takes the in-process path with no resident.
#
# THE SERVE SPEC IS DERIVED HERE, not inherited. This script already resolves
# the declaration through tools/spec-declaration.sh above -- it picks the timed
# oracle and the --mtp-depth flag from it -- so it hands the SAME derivation to
# the serve. Leaving SERVE_UP_SPECULATIVE to the caller made benchmark.json's
# benchmarkCommand boot a SERIAL engine while benchd was told --mtp-depth N:
# the declared leg would be measured on an unarmed drafter against the MTP
# oracle. tools/ranked-box-preflight.sh section 7 resolves the same deriver, so
# a box-preset value that disagrees is already refused before this point.
SERVE_SPEC="$("${SCRIPT_DIR}/tools/spec-declaration.sh" speculative)"
SERVE_DRAFT_LEN="$("${SCRIPT_DIR}/tools/spec-declaration.sh" draft-len)"
if [[ -n "${DS4_MODEL:-}" ]]; then
  exec "${iterate_official[@]}"
else
  exec env SERVE_UP_WEIGHTS_DIR="${WEIGHTS_PATH}" \
           SERVE_UP_SPECULATIVE="${SERVE_SPEC}" \
           SERVE_UP_SPEC_DRAFT_LEN="${SERVE_DRAFT_LEN}" \
           "${SCRIPT_DIR}/tools/serve-up.sh" "${iterate_official[@]}"
fi
