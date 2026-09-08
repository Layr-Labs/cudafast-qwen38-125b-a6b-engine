#!/usr/bin/env bash
#
# ranked-box-preflight.sh -- the fail-closed gate the ranked job runs BEFORE
# ./setup.sh and before any measurement, for track qwen3.8-125b-a6b-cuda-v1.
#
# WHAT IT IS FOR. The ranked job holds NO CREDENTIAL by design (bundles-no-keys:
# a box receives staged bundles, never keys), so the hidden material this track
# measures against cannot be fetched by the job itself -- it is staged onto the
# box out of band by the organizer. That moves the whole question from "can we
# download it" to "is what is on this box the pinned material". This script
# answers that question, and refuses when the answer is anything other than
# "yes, exactly".
#
# Every check below aborts before a score.json could exist. There is no
# degraded mode: the alternative to a verified staged asset is a non-zero exit,
# never a substituted, defaulted, or re-fetched one.
#
# THE STAGING CONVENTION IS THE EXISTING ONE, not a new one. The golden path
# comes from the runner process environment under the name
# tools/qwen38-125b-a6b-measure-and-score.sh already reads:
#
#   MLXFAST_QWEN38_GOLDEN_DIR          directory holding the pinned pool goldens
#                                      (the live golden this leg scores over plus
#                                      the rotation set)
#   MLXFAST_BASELINE_WORKSPACE         the built REFERENCE tree the serial-control
#                                      leg runs on (section 8)
#   MLXFAST_BASELINE_CALIBRATION       this box's calibration file, the health
#                                      band for that leg (section 8b)
#
# On a self-hosted runner they are set in the runner service environment by
# whoever stages the box; a `run:` step inherits them. Nothing here reads a
# GitHub secret, and nothing here reaches the network.
#
# PAIRED, WITH A PER-BOX BASELINE (David 2026-09-08). This track scores TWO legs
# on the SAME box in the SAME job, over the ONE live golden: a SERIAL CONTROL leg
# on the organizer-staged reference tree, and the CANDIDATE leg on this checkout
# at its declared draft depth. The score is the ratio of the two. NO PAIR IS
# STORED ANYWHERE -- not in a benchd constant, not in the fixture, not in a
# golden -- so this preflight verifies the staged pool goldens (and the live
# golden among them), the fixture arm state, the pinned benchd against its own
# manifest, the serve identity, AND the two halves the paired path adds: the
# reference workspace is the pinned reference commit and is built, and this box
# has a calibration file that describes THIS box against THAT commit.
#
# THE CALIBRATION IS A HEALTH BAND, NEVER A DENOMINATOR. benchd divides by the
# measured control leg. The band only says whether that leg landed where this box
# has landed before; outside it, the run dies and seals nothing.
#
# WHAT THE PINS ARE. fixtures/qwen3_8_125b_a6b_track.json is trusted-side (not an
# editable path), and its timed_prompt_pool[] carries {r2_path, sha256, bytes}
# per golden plus hidden_correctness_golden's {sha256, bytes}. A staged file is
# accepted only when its byte count AND its sha256 equal the contract's -- byte
# count first, because a truncated stage is the common failure and naming it
# precisely is worth one `wc -c` (the same order tools/fetch-goldens.sh and
# tools/fetch-benchd.sh verify in).
#
# Usage:  tools/ranked-box-preflight.sh
# Exit:   0 every check passed
#         1 a check failed (message on stderr names which)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
CONTRACT="${REPO_ROOT}/fixtures/qwen3_8_125b_a6b_track.json"

fail() {
  echo "ranked-box-preflight: REFUSING -- $*" >&2
  exit 1
}

ok() {
  echo "ok    $*"
}

# --- 0. tools ---------------------------------------------------------------
# jq is already a hard requirement of tools/qwen38-125b-a6b-measure-and-score.sh
# (it reads the fixture pins and the trackId), so a box that cannot run this
# cannot finish a ranked run either.
command -v jq >/dev/null 2>&1 || fail "jq is required to read the track contract"
command -v shasum >/dev/null 2>&1 || fail "shasum is required to verify staged assets"
[[ -r "${CONTRACT}" ]] || fail "cannot read the track contract at ${CONTRACT}"
jq -e . >/dev/null 2>&1 < "${CONTRACT}" || fail "the track contract is not valid JSON: ${CONTRACT}"
ok "track contract readable and parses: fixtures/qwen3_8_125b_a6b_track.json"

# --- 1. the job holds no credential -----------------------------------------
# The workflow references no secret (tools/ci-workflow-egress-scan.sh is the
# static half of that). This is the runtime half: a credential reaching the job
# through the RUNNER's environment would defeat the same invariant without ever
# appearing in the workflow file. R2 keys and a signer are what would let this
# job pull hidden material itself instead of measuring the staged bundle.
for var in R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY MLXFAST_QWEN38_R2_DOWNLOADER BENCHD_DIST_TOKEN; do
  eval "value=\${${var}:-}"
  # shellcheck disable=SC2154  # assigned by the eval above
  [[ -z "${value}" ]] || fail "${var} is set in the ranked job's environment; this job must hold no credential (boxes get staged bundles, never keys)"
done
ok "no R2 credential, signer, or dist token in the job environment"

# --- 2. no measurement-weakening override -----------------------------------
# Each of these is a real, local-debugging-only switch in this tree. On a
# ranked box any of them would silently change what is measured or what is
# accepted, so their presence is a refusal rather than a warning.
#
#   BENCHD                            tools/qwen38-125b-a6b-measure-and-score.sh honours a
#                                     caller-supplied benchd WITHOUT the
#                                     channel-manifest hash check (deliberate, for
#                                     benchd development) -- so on a ranked run
#                                     it is a way to measure against unpinned
#                                     scoring code.
#   MLXFAST_SKIP_WEIGHTS_DOWNLOAD /   setup.sh builds tools only and never
#   SKIP_MODEL_DOWNLOAD               obtains or verifies the checkpoint.
#   MLXFAST_LOCAL_COOL_GATE           disables the thermal gate.
#   MLXFAST_LOCAL_ALLOW_GOLDEN_DRIFT  publishes a timing estimate past a failed
#                                     public correctness gate.
#   DS4_MODEL / DS4_MTP_PATH /        the engine configuration tools/serve-up.sh
#   DS4_MTP_DRAFT_TOKENS /            derives for the window. A preset value is
#   DS4_CTX_SIZE / DS4_EOS_IDS        read instead of the derived one, so the
#                                     engine that runs is not the engine the
#                                     declaration describes: DS4_MODEL takes the
#                                     in-process path with no resident,
#                                     DS4_MTP_PATH and DS4_MTP_DRAFT_TOKENS arm a
#                                     drafter the declaration did not ask for,
#                                     DS4_CTX_SIZE changes the session context
#                                     every timed phase runs in, and DS4_EOS_IDS
#                                     changes where a decode stops. Each of the
#                                     last two changes the measured work while
#                                     every seal still reads normal.
#   DS4_RESIDENT_SOCKET /             the resident each leg's worker connects to.
#   BENCH_WORKER_RESIDENT_SOCKET      benchd boots each leg's resident itself and
#                                     injects that leg's socket into that leg's
#                                     worker spawns, so a socket already in the
#                                     environment is a THIRD resident neither leg
#                                     booted -- and every worker that inherited
#                                     it would measure whatever is on the other
#                                     end, on either leg, with a normal-looking
#                                     seal. benchd refuses one in its own
#                                     environment; this refuses it before the
#                                     box is committed to anything.
#
# MLXFAST_GPU_TEMP_CMD is NOT refused: it is benchd's documented, first-class
# override for the temperature reader, and on a Linux/CUDA box the native reader
# below (nvidia-smi) is what benchd's cool gate reads anyway. It stays an
# OPTIONAL override -- never required, never set here for the job -- so a box
# needs no environment variable at all to get a temperature reading.
for var in BENCHD MLXFAST_SKIP_WEIGHTS_DOWNLOAD SKIP_MODEL_DOWNLOAD MLXFAST_LOCAL_COOL_GATE MLXFAST_LOCAL_ALLOW_GOLDEN_DRIFT MLXFAST_SKIP_ENGINE_BUILD MLXFAST_SKIP_WEIGHTS_SHA256 DS4_MODEL DS4_MTP_PATH DS4_MTP_DRAFT_TOKENS DS4_CTX_SIZE DS4_EOS_IDS DS4_RESIDENT_SOCKET BENCH_WORKER_RESIDENT_SOCKET; do
  eval "value=\${${var}:-}"
  [[ -z "${value}" ]] || fail "${var} is set in the ranked job's environment; it weakens or bypasses what the ranked run measures"
done
ok "no measurement-weakening override in the job environment"

# --- 2b. the GPU temperature reader resolves --------------------------------
# DAVID'S RULING 2026-08-26: no calibration without thermal control, and a
# missing reader REFUSES rather than silently self-disabling.
#
# WHY IT MUST BE LOUD HERE. The pinned benchd's own cool gate SKIPS with a
# warning when no reader resolves -- it returns GateState::SkippedNoReader
# rather than failing. That is correct for a participant's laptop and
# catastrophic for a ranked run: every timed leg would proceed ungated and the
# seal would look normal. Refusing here makes that path unreachable on this box.
#
# THE READER IS NATIVE TO THE PLATFORM, not an environment workaround. This box
# is Linux/CUDA (GB10), so the reader is nvidia-smi, which every CUDA box has --
# no environment variable is needed to get a reading. macOS keeps macmon, the
# reader that platform's benchd cool gate uses. MLXFAST_GPU_TEMP_CMD is an
# OPTIONAL override, honoured first (benchd honours it first too); it is never
# required and never set here for the job.
GPU_TEMP_SOURCE=""
GPU_TEMP_FROZEN_CHECK=0
if [[ -n "${MLXFAST_GPU_TEMP_CMD:-}" ]]; then
  GPU_TEMP_SOURCE="MLXFAST_GPU_TEMP_CMD override"
  read_gpu_temp() { eval "${MLXFAST_GPU_TEMP_CMD}" 2>/dev/null | head -1 | tr -d '[:space:]'; }
elif [[ "$(uname -s)" == "Linux" ]]; then
  command -v nvidia-smi >/dev/null 2>&1 \
    || fail "nvidia-smi not found: this Linux/CUDA box has no GPU temperature reader, so benchd's cool-down gate would silently skip on every timed leg. Install the NVIDIA driver utilities (nvidia-smi) or set MLXFAST_GPU_TEMP_CMD; this run measures nothing without thermal control"
  GPU_TEMP_SOURCE="nvidia-smi"
  read_gpu_temp() {
    nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d '[:space:]'
  }
else
  # macOS: macmon, resolved the same way .github/workflows/benchmark.yml pins it.
  MLXFAST_MACMON="${MLXFAST_MACMON:-${MLXFAST_MACMON_BIN:-/opt/homebrew/bin/macmon}}"
  test -x "${MLXFAST_MACMON}" \
    || fail "macmon missing at ${MLXFAST_MACMON}: no GPU temperature reader on this macOS box, so benchd's cool-down gate would silently skip on every timed leg. Install macmon or point MLXFAST_MACMON_BIN at it; this run measures nothing without thermal control"
  GPU_TEMP_SOURCE="macmon (${MLXFAST_MACMON})"
  GPU_TEMP_FROZEN_CHECK=1
  read_gpu_temp() {
    "${MLXFAST_MACMON}" pipe -s1 2>/dev/null | jq -r '.temp.gpu_temp_avg // empty' | head -1
  }
fi
ok "GPU temperature reader resolved: ${GPU_TEMP_SOURCE}"

# --- 2c. the reader gives a plausible reading -------------------------------
# A sensor that reads at or below the 5C implausibility floor is broken or
# frozen, and a broken sensor passes every cool gate on an arbitrarily hot GPU.
first_temp="$(read_gpu_temp || true)"
[[ -n "${first_temp}" ]] \
  || fail "the temperature reader (${GPU_TEMP_SOURCE}) produced no reading; a reader that cannot be read is a missing reader"
[[ "$(jq -n --argjson t "${first_temp}" '$t > 5')" == "true" ]] \
  || fail "GPU temperature reads ${first_temp}C, at or below the 5C implausibility floor; the sensor is broken or frozen, and a broken sensor passes every cool gate"

# The frozen-sensor multi-sample check applies ONLY to macmon's averaged float,
# which always jitters on a live box. nvidia-smi reports an INTEGER Celsius that
# can legitimately sit on one value across ~20s on a genuinely idle box, so the
# "identical reading = stuck sensor" inference is a false positive there and is
# not run for the native/override readers.
if [[ "${GPU_TEMP_FROZEN_CHECK}" == "1" ]]; then
  distinct_temp_count() {
    printf '%s\n' "$@" | sort -u | wc -l | tr -d ' '
  }
  temps=("${first_temp}")
  for _ in 1 2; do
    sleep 2
    temps+=("$(read_gpu_temp || true)")
  done
  if [[ "$(distinct_temp_count "${temps[@]}")" == "1" ]]; then
    for _ in 1 2 3; do
      sleep 5
      temps+=("$(read_gpu_temp || true)")
    done
    if [[ "$(distinct_temp_count "${temps[@]}")" == "1" ]]; then
      fail "the temperature reader (${GPU_TEMP_SOURCE}) returned the identical value ${first_temp}C on ${#temps[@]} samples across ~20s; treating it as a frozen sensor, because a stuck reading passes every cool gate on an arbitrarily hot GPU"
    fi
  fi
  ok "temperature reader is plausible and moving (samples: ${temps[*]})"
else
  ok "temperature reader is plausible (${GPU_TEMP_SOURCE}: ${first_temp}C)"
fi

# --- 3. the timed pool is armed ---------------------------------------------
# "Armed" is a property of the CONTRACT, checked before anything on disk is
# looked at: a sentinel entry has no digest to verify a staged file against, so
# a staged file would be accepted on its name alone.
#
# THIS TRACK'S SENTINEL IS SPELLED IN FULL, and the check is two checks rather
# than one. The substring form alone was the whole guard, and it accepts ANY
# string ending in PENDING-ORGANIZER -- including the MLX twin's
# `QWEN38-125B-A6B-MLX-PENDING-ORGANIZER`. A fixture copied from the twin and
# only half-swept would therefore read as correctly unarmed here while naming
# the other track's R2 prefix, and the first thing that noticed would be a
# fetch against the wrong prefix.
#
# So: refuse on the substring (an unarmed pool of any spelling), and ALSO
# refuse on the twin's exact sentinel with its own message, because those two
# faults need different fixes -- one waits for the organizer, the other is a
# sweep bug in this repository.
TRACK_SENTINEL="QWEN38-125B-A6B-CUDA-PENDING-ORGANIZER"
FOREIGN_SENTINEL="QWEN38-125B-A6B-MLX-PENDING-ORGANIZER"

if grep -q "${FOREIGN_SENTINEL}" "${CONTRACT}"; then
  fail "the track contract carries the MLX twin's sentinel (${FOREIGN_SENTINEL}), not this track's (${TRACK_SENTINEL}); the fixture was copied and not fully swept"
fi
if grep -q 'PENDING-ORGANIZER' "${CONTRACT}"; then
  fail "the track contract still carries ${TRACK_SENTINEL} sentinels; the timed pool is unarmed and nothing can be pin-verified against it"
fi

# SINGLE-LEG: the pool need only carry the live golden (plus the rotation set),
# so the retired "exactly 8" cohort-size assertion is dropped -- a rigid count
# was the paired eight-tape model. What still must hold is that the pool is
# non-empty, so there IS a live golden to pin-verify against; every entry's pin
# is then checked below and the live golden is asserted present in section 4.
pool_count="$(jq -r '.timed_prompt_pool | length' "${CONTRACT}")"
[[ "${pool_count}" =~ ^[1-9][0-9]*$ ]] || fail "timed_prompt_pool is empty; there is no live golden to pin-verify against"

# A pin is {sha256, bytes} together; neither half alone is one. An entry that
# fails this is unarmed no matter what it is called.
malformed="$(jq -r '
  .timed_prompt_pool
  | to_entries
  | map(select(
      (.value.r2_path | type != "string" or length == 0)
      or (.value.sha256 | type != "string" or test("^[0-9a-f]{64}$") | not)
      or (.value.bytes | type != "number" or . <= 0)
    ))
  | map("timed_prompt_pool[" + (.key | tostring) + "]")
  | join(", ")
' "${CONTRACT}")"
[[ -z "${malformed}" ]] || fail "unarmed or malformed pool pin(s): ${malformed}"

hidden_sha="$(jq -r '.hidden_correctness_golden.sha256 // ""' "${CONTRACT}")"
hidden_bytes="$(jq -r '.hidden_correctness_golden.bytes // 0' "${CONTRACT}")"
printf '%s' "${hidden_sha}" | grep -Eq '^[0-9a-f]{64}$' \
  || fail "hidden_correctness_golden.sha256 is not a 64-hex digest; the token-fidelity oracle is unarmed"
printf '%s' "${hidden_bytes}" | grep -Eq '^[1-9][0-9]*$' \
  || fail "hidden_correctness_golden.bytes is not a positive integer"
ok "timed pool armed: ${pool_count} pinned pool golden(s) + a pinned hidden correctness golden"

# --- 4. the staged tapes match the pins -------------------------------------
# The pinned pool goldens default to this repo's COMMITTED copy -- present in every
# checkout and pin-verified below exactly like a staged copy. An env override wins, so
# a production box points MLXFAST_QWEN38_GOLDEN_DIR at its organizer-staged (hidden)
# goldens; a job that never wired the env then refuses on CONTENT (a pin mismatch)
# rather than ABSENCE. The default does NOT weaken the gate: whatever directory is
# used, every golden is verified byte-then-sha against the contract pins below.
GOLDEN_DIR="${MLXFAST_QWEN38_GOLDEN_DIR:-${REPO_ROOT}/correctness_prompts/qwen3.8-125b-a6b-cuda-v1}"
[[ -d "${GOLDEN_DIR}" ]] \
  || fail "the pinned pool goldens are not present: MLXFAST_QWEN38_GOLDEN_DIR is unset and the committed default ${GOLDEN_DIR} is absent. Under single-leg the scored run reads only the live golden, but the whole pinned pool is verified here -- stage the goldens (or set MLXFAST_QWEN38_GOLDEN_DIR to the staged directory), then re-dispatch"

verify_pin() {
  # verify_pin <path> <want_sha256> <want_bytes> <label>
  local path="$1" want_sha="$2" want_bytes="$3" label="$4" got_bytes got_sha
  [[ -f "${path}" ]] || fail "${label}: staged file is missing: ${path}"
  got_bytes="$(wc -c < "${path}" | tr -d '[:space:]')"
  [[ "${got_bytes}" == "${want_bytes}" ]] \
    || fail "${label}: byte-count mismatch (staged ${got_bytes}, pinned ${want_bytes}): ${path}"
  got_sha="$(shasum -a 256 "${path}" | awk '{print $1}')"
  [[ "${got_sha}" == "${want_sha}" ]] \
    || fail "${label}: sha256 mismatch (staged ${got_sha}, pinned ${want_sha}): ${path}"
}

expected_list=""
while IFS='	' read -r r2_path want_sha want_bytes; do
  [[ -n "${r2_path}" ]] || continue
  name="${r2_path##*/}"
  verify_pin "${GOLDEN_DIR}/${name}" "${want_sha}" "${want_bytes}" "timed-pool tape ${name}"
  expected_list="${expected_list}${name}
"
done <<EOF
$(jq -r '.timed_prompt_pool[] | [.r2_path, .sha256, (.bytes | tostring)] | @tsv' "${CONTRACT}")
EOF
ok "all ${pool_count} staged pool goldens match their contract pins (bytes then sha256)"

# PER-DEPTH TIMED ORACLES. live_golden_speculative maps each permitted MTP draft
# depth (mtp1/mtp2/mtp3) to its OWN timed-oracle golden {r2_path, sha256, bytes};
# tools/qwen38-125b-a6b-measure-and-score.sh selects one by the declared depth.
# Each must be pin-verified here AND allowed to exist in the staging directory
# (the "only pinned *.json" check below refuses any name absent from
# expected_list). A depth may REUSE another depth's file when their trajectories
# are identical, so a name is appended to expected_list at most once. An absent
# map means a serial-only track and this section verifies nothing.
spec_malformed="$(jq -r '
  (.live_golden_speculative // {})
  | to_entries
  | map(select(
      (.value.r2_path | type != "string" or length == 0)
      or (.value.sha256 | type != "string" or test("^[0-9a-f]{64}$") | not)
      or (.value.bytes | type != "number" or . <= 0)
    ))
  | map("live_golden_speculative[" + .key + "]")
  | join(", ")
' "${CONTRACT}")"
[[ -z "${spec_malformed}" ]] || fail "unarmed or malformed per-depth oracle pin(s): ${spec_malformed}"

spec_count=0
while IFS='	' read -r r2_path want_sha want_bytes; do
  [[ -n "${r2_path}" ]] || continue
  name="${r2_path##*/}"
  verify_pin "${GOLDEN_DIR}/${name}" "${want_sha}" "${want_bytes}" "per-depth oracle ${name}"
  if ! printf '%s' "${expected_list}" | grep -Fxq "${name}"; then
    expected_list="${expected_list}${name}
"
  fi
  spec_count=$((spec_count + 1))
done <<EOF
$(jq -r '(.live_golden_speculative // {}) | to_entries[] | [.value.r2_path, .value.sha256, (.value.bytes | tostring)] | @tsv' "${CONTRACT}")
EOF
[[ "${spec_count}" -eq 0 ]] || ok "all ${spec_count} per-depth oracle pin(s) match their staged golden(s)"

# The staging directory must hold ONLY pinned goldens. Under single-leg the
# scored run reads just the live golden, but an unpinned *.json staged where the
# pool lives is still an unattributed golden -- a mis-staged or leftover file
# that has no contract pin behind it -- so it is a refusal rather than a
# warning: this directory carries pinned material only.
unexpected=""
for staged in "${GOLDEN_DIR}"/*.json; do
  [[ -e "${staged}" ]] || continue
  staged_name="${staged##*/}"
  if ! printf '%s' "${expected_list}" | grep -Fxq "${staged_name}"; then
    unexpected="${unexpected} ${staged_name}"
  fi
done
[[ -z "${unexpected}" ]] \
  || fail "MLXFAST_QWEN38_GOLDEN_DIR holds *.json file(s) that are not pinned in timed_prompt_pool:${unexpected} (an unpinned golden staged where the pool lives is unattributed material; this directory carries pinned goldens only)"
ok "no unpinned *.json in the staging directory"

# The hidden correctness oracle is pinned by digest only -- the contract gives
# it no r2_path -- and benchd resolves it itself. If the box names one
# through the existing MLXFAST_CORRECTNESS_GOLDEN_PATH convention
# (Sources/MLXFastCLI/main.swift), it must be the pinned bytes; if it names
# none, this asserts nothing about it rather than inventing a location.
if [[ -n "${MLXFAST_CORRECTNESS_GOLDEN_PATH:-}" ]]; then
  verify_pin "${MLXFAST_CORRECTNESS_GOLDEN_PATH}" "${hidden_sha}" "${hidden_bytes}" "hidden correctness golden"
  ok "MLXFAST_CORRECTNESS_GOLDEN_PATH matches hidden_correctness_golden"
else
  ok "MLXFAST_CORRECTNESS_GOLDEN_PATH unset; benchd resolves the oracle from the contract"
fi

# --- 4b. the live golden the single-leg run scores over is staged -----------
# Single-leg reads exactly ONE golden: the fixture's live_golden, resolved by
# tools/qwen38-125b-a6b-measure-and-score.sh as <live_golden>.golden.json. The
# loop above already pin-verified it AS a pool member; this asserts the
# fixture's live_golden actually NAMES a pinned pool entry and is staged, so a
# live_golden rotation that points at a golden absent from the pool -- or a box
# that staged the pool but not the live golden -- is caught here, pre-GPU,
# rather than at measure time.
LIVE_GOLDEN_NAME="$(jq -r '.live_golden // ""' "${CONTRACT}")"
[[ -n "${LIVE_GOLDEN_NAME}" ]] \
  || fail "the fixture declares no live_golden; there is no golden for the single-leg run to score over"
live_golden_base="${LIVE_GOLDEN_NAME}.golden.json"
printf '%s' "${expected_list}" | grep -Fxq "${live_golden_base}" \
  || fail "live_golden '${LIVE_GOLDEN_NAME}' names no timed_prompt_pool entry (looked for ${live_golden_base}); it carries no pin and cannot be pin-verified"
[[ -f "${GOLDEN_DIR}/${live_golden_base}" ]] \
  || fail "the live golden ${live_golden_base} is not staged in ${GOLDEN_DIR}; it is the one golden the single-leg run scores over"
ok "live golden ${live_golden_base} is pinned and staged (the single-leg scored golden)"

# --- 5. the fixture is armed for official scoring ---------------------------
# benchd refuses, pre-GPU, to seal an official artifact unless the fixture
# declares official_scoring_enabled: true (enforce_official_scoring_enabled).
# false AND absent both refuse -- an absent arm state is not an armed one. The
# measure wrapper's --preflight-only mirrors this; refusing here makes an
# unarmed ranked dispatch fail before setup rather than after the GPU window.
armed="$(jq -r '.official_scoring_enabled // false' "${CONTRACT}")"
[[ "${armed}" == "true" ]] \
  || fail "the track contract does not declare official_scoring_enabled: true (got '${armed}'); benchd refuses to seal an official score for an unarmed track, and so does this gate"
ok "fixture is armed for official scoring (official_scoring_enabled: true)"

# --- 6. the pinned benchd is present and matches its own manifest ---------
# The scored composite is SEALED by benchd, so an unattributable or mismatched
# benchd is a refusal here, before setup. This is the fail-fast, provenance-
# logging half of tools/fetch-benchd.sh's OFFLINE path (its step 1): the
# converge-staged PAIR benchd-bin/{benchd, benchd.manifest.json} is placed
# on the box out of band (bundles-no-keys: this job holds no dist token and
# reaches no network -- section 1 refuses BENCHD_DIST_TOKEN), and fetch-benchd.sh
# re-verifies it authoritatively in the next workflow step (adding the channel
# branch and the target_triple / ELF-container checks). A binary with no manifest
# is unattributable and must never seal a score. BENCHD_BIN_DIR is resolved
# exactly as fetch-benchd.sh resolves it, so a box that stages the pair outside
# the workspace is checked at the same path the harness will read.
BENCHD_DIR="${BENCHD_BIN_DIR:-${REPO_ROOT}/benchd-bin}"
BENCHD_BIN="${BENCHD_DIR}/benchd"
BENCHD_MANIFEST="${BENCHD_DIR}/benchd.manifest.json"
[[ -f "${BENCHD_BIN}" ]] \
  || fail "the pinned benchd is not staged at ${BENCHD_BIN}; this job holds no dist token and does not fetch benchd over the network. Stage the converge pair on the box"
[[ -f "${BENCHD_MANIFEST}" ]] \
  || fail "benchd.manifest.json is missing beside ${BENCHD_BIN}; a binary with no manifest is unattributable and will not seal a score"
jq -e . >/dev/null 2>&1 < "${BENCHD_MANIFEST}" \
  || fail "the benchd manifest is not valid JSON: ${BENCHD_MANIFEST}"
benchd_sha="$(jq -r '.sha256 // ""' "${BENCHD_MANIFEST}")"
benchd_bytes="$(jq -r '.bytes // 0' "${BENCHD_MANIFEST}")"
benchd_commit="$(jq -r '.source_commit // ""' "${BENCHD_MANIFEST}")"
printf '%s' "${benchd_sha}" | grep -Eq '^[0-9a-f]{64}$' \
  || fail "the benchd manifest sha256 is not a 64-hex digest ('${benchd_sha}'); the harness identity is unverifiable"
printf '%s' "${benchd_bytes}" | grep -Eq '^[1-9][0-9]*$' \
  || fail "the benchd manifest bytes is not a positive integer: '${benchd_bytes}'"
verify_pin "${BENCHD_BIN}" "${benchd_sha}" "${benchd_bytes}" "pinned benchd"
[[ -n "${benchd_commit}" ]] \
  || echo "ranked-box-preflight: WARNING -- benchd manifest records no source_commit; provenance is unattributed" >&2
ok "pinned benchd matches its manifest (source_commit ${benchd_commit:-unrecorded}, sha256 ${benchd_sha})"

# --- 7. the serve spec matches the participant declaration (the ARM GATE) ----
# Single-leg scores ONE serve, and that serve's spec IS part of the identity: a
# resident SERIAL serve and a scored MTP serve are different runs. The serve spec
# is DERIVED from the participant declaration (mtp-head.manifest.json `spec`) by
# the single trusted source tools/spec-declaration.sh, which validates it
# fail-closed. This gate resolves the SAME derivation and:
#   * REFUSES an invalid declaration (the helper exits non-zero and names why);
#   * when the declaration ENABLES speculation, requires the fixture to be ARMED
#     for official scoring (official_scoring_enabled: true) -- a valid declaration
#     alone does not open the gate;
#   * REFUSES any box-preset SERVE_UP_SPECULATIVE. Under one leg a preset that
#     agreed with the declaration was harmless. Under the paired path it is not:
#     the value reaches the environment of BOTH legs' serve scripts, and the
#     serial-control leg is serial whatever the candidate declares. So the serve
#     spec comes from the declaration and from nowhere else.
#
# An ABSENT or disabled declaration derives 0 (SERIAL) -- the MTP-0 launch
# reference (David ruling) -- and the historical serial-only refusal is preserved
# for it: an undeclared/disabled config with a box-preset speculative serve still
# refuses. THIS is the gate the native-MTP path moves through: it opens ONLY for a
# VALIDLY declared spec on an ARMED fixture, and its opening is governed by merge
# timing (this hook merges after the separate arm-gate retry seals), not by any
# hardcoded serial assertion here.
if ! decl_spec="$("${REPO_ROOT}/tools/spec-declaration.sh" speculative)"; then
  fail "the speculative-decode declaration (mtp-head.manifest.json) is invalid; the derivation above names why. An undeclared or disabled declaration is valid and derives serial"
fi
decl_draft="$("${REPO_ROOT}/tools/spec-declaration.sh" draft-len)"
[[ -z "${SERVE_UP_SPECULATIVE:-}" ]] \
  || fail "a box pre-set SERVE_UP_SPECULATIVE=${SERVE_UP_SPECULATIVE}; on the paired path that value reaches BOTH legs' serve scripts and the serial-control leg must be serial whatever the candidate declares. The serve spec comes from the declaration only"
if [[ "${decl_spec}" == "1" ]]; then
  armed="$(jq -r '.official_scoring_enabled // false' "${CONTRACT}")"
  [[ "${armed}" == "true" ]] \
    || fail "the declaration enables MTP speculation (num_speculative_tokens=${decl_draft}) but fixtures/qwen3_8_125b_a6b_track.json is not armed (official_scoring_enabled != true); the arm gate opens only for a valid declaration on an armed fixture"
  ok "serve spec is the declared MTP config (num_speculative_tokens=${decl_draft}) and the fixture is armed"
else
  ok "serve spec is the serial launch reference (SERVE_UP_SPECULATIVE=0; declaration absent or disabled)"
fi

# SERVE CONFIGURATION PINS (MTP 0..3). fixtures/qwen3_8_125b_a6b_track.json
# serve_configuration pins the engine BASE every depth's timed oracle was
# authored under, and the engine environment tools/serve-up.sh exports.
#
# DO NOT RE-ADD AN EQUALITY CHECK BETWEEN engine_pin AND THE ENGINE CONTENT.
# (Ruled 2026-09-04.) This once compared the ds4 submodule gitlink to
# serve_configuration.engine_pin. The engine is VENDORED and PARTICIPANT-EDITABLE
# now, and that comparison would refuse EVERY submission -- which is the whole
# point of vendoring it.
#
# WHAT engine_pin MEANS. It is PROVENANCE: the engine base the calibrated
# baseline and the timed oracles were AUTHORED ON. It is not a statement that the
# engine running now is byte-identical to it, and it must never be read as one.
#
# WHAT ACTUALLY GATES A PARTICIPANT. The CORRECTNESS goldens and the calibrated
# baseline. An edit that changes the token stream fails the goldens, wherever the
# change came from -- engine, harness or adapter -- which is a far better gate
# than a sha comparison, because a sha says nothing about the bytes that ran.
#
# So this asserts only that the tree was VENDORED FROM the pinned base
# (ds4/VENDOR.json fork.sha), and leaves the numbers to what measures them.
want_engine_pin="$(jq -r '.serve_configuration.engine_pin // empty' "${CONTRACT}")"
[[ "${want_engine_pin}" =~ ^[0-9a-f]{40}$ ]] \
  || fail "serve_configuration.engine_pin is unarmed or malformed in fixtures/qwen3_8_125b_a6b_track.json (got '${want_engine_pin}')"
[[ -f "${REPO_ROOT}/ds4/VENDOR.json" ]] \
  || fail "ds4/VENDOR.json is absent; the engine tree carries no provenance (re-vendor with tools/ds4/vendor-sync.sh)"
have_engine_pin="$(jq -r '.fork.sha // empty' "${REPO_ROOT}/ds4/VENDOR.json" 2>/dev/null || true)"
[[ "${have_engine_pin}" == "${want_engine_pin}" ]] \
  || fail "ds4/VENDOR.json records vendor base ${have_engine_pin:-absent} but the fixture pins ${want_engine_pin}; the goldens were authored on a different engine base (re-vendor with tools/ds4/vendor-sync.sh, or re-author and re-pin)"
target_engine_pin="$(jq -r '.target.engine_pin // empty' "${CONTRACT}")"
[[ "${target_engine_pin}" == "${want_engine_pin}" ]] \
  || fail "serve_configuration.engine_pin (${want_engine_pin}) and target.engine_pin (${target_engine_pin}) disagree"
[[ -f "${REPO_ROOT}/ds4/ds4.c" && -f "${REPO_ROOT}/ds4/Makefile" ]] \
  || fail "the vendored ds4 tree has no ds4.c or Makefile; setup.sh could not build an engine from it"
ok "ds4 engine base ${want_engine_pin:0:12} matches ds4/VENDOR.json (engine content is participant-editable and is gated by the correctness goldens)"
# The toolchain is part of the numeric epoch: the goldens were authored on this
# nvcc build and this driver floor (fleet-bootstrap pins them on the box).
want_nvcc="$(jq -r '.serve_configuration.toolchain.nvcc_version // empty' "${CONTRACT}")"
want_driver_min="$(jq -r '.serve_configuration.toolchain.driver_min // empty' "${CONTRACT}")"
[[ -n "${want_nvcc}" && -n "${want_driver_min}" ]] \
  || fail "serve_configuration.toolchain.{nvcc_version,driver_min} are unarmed in fixtures/qwen3_8_125b_a6b_track.json"
command -v nvcc >/dev/null 2>&1 || fail "nvcc is not on PATH; the fleet toolchain pin is ${want_nvcc}"
have_nvcc="$(nvcc --version 2>/dev/null | sed -n 's/.*release [0-9.]*, \(V[0-9.]*\).*/\1/p' | head -1)"
[[ "${have_nvcc}" == "${want_nvcc}" ]] \
  || fail "nvcc is ${have_nvcc:-unknown} but the fixture pins ${want_nvcc}; a different toolchain is a different numeric epoch (re-author, then re-pin)"
have_driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d '[:space:]')"
[[ -n "${have_driver}" ]] || fail "nvidia-smi reports no driver version"
if [[ "$(printf '%s\n%s\n' "${want_driver_min}" "${have_driver}" | sort -V | head -1)" != "${want_driver_min}" ]]; then
  fail "driver ${have_driver} is below the fixture floor ${want_driver_min}"
fi
ok "toolchain: nvcc ${have_nvcc}, driver ${have_driver} (floor ${want_driver_min})"
# engine_env. THE SHAPE IS CHECKED FIRST, because the pinned map is empty
# today and a `while read` over zero entries asserts nothing at all: a fixture
# that lost the key entirely, or grew a list where an object belongs, would
# have sailed through the loop below. So the key must EXIST and be an object,
# and only then are its entries (if any) matched against serve-up.sh.
env_type="$(jq -r '.serve_configuration | if has("engine_env") then (.engine_env | type) else "absent" end' "${CONTRACT}")"
[[ "${env_type}" == "object" ]] \
  || fail "serve_configuration.engine_env must be an object (it is '${env_type}'); an absent or wrongly-typed map makes the export check vacuous"
env_count="$(jq -r '.serve_configuration.engine_env | length' "${CONTRACT}")"
while IFS=$'\t' read -r want_key want_val; do
  [[ -n "${want_key}" ]] || continue
  grep -Eq -- "^export ${want_key}=\"?\\\$\{SERVE_UP_[A-Z_]+:-${want_val}\}\"?$|^export ${want_key}=${want_val}$" "${REPO_ROOT}/tools/serve-up.sh" \
    || fail "tools/serve-up.sh no longer exports pinned engine env ${want_key}=${want_val} (serve_configuration.engine_env); the timed oracles were authored with it"
done < <(jq -r '.serve_configuration.engine_env | to_entries[] | [.key, .value] | @tsv' "${CONTRACT}")
ok "serve_configuration.engine_env is an object pinning ${env_count} variable(s), and serve-up.sh exports every one"

# WEIGHT OWNER. The window loads the checkpoint ONCE, in the resident engine
# process (David 2026-08-30: weights load once, persistent worker, passes
# reconnect). benchd spawns a fresh worker per phase, so a serve script that
# stopped booting an owner would silently go back to a load per phase -- more
# than sixteen loads of the 103.7 GiB artifact in one ranked run. The fixture
# names the owner, and this refuses a repository whose serve script does not
# boot it or whose build does not produce it.
want_owner="$(jq -r '.serve_configuration.weight_owner // empty' "${CONTRACT}")"
[[ -n "${want_owner}" && "${want_owner}" != "none" ]] \
  || fail "serve_configuration.weight_owner is unarmed (got '${want_owner}'); this track loads the weights once per window and the owner must be named"
grep -qF -- "\"weight_owner\": \"${want_owner}\"" "${REPO_ROOT}/tools/serve-up.sh" \
  || fail "tools/serve-up.sh does not record weight_owner=${want_owner} in the serve identity; the fixture and the serve script disagree about who holds the weights"
grep -qF -- 'SERVE_UP_RESIDENT_BIN' "${REPO_ROOT}/tools/serve-up.sh" \
  || fail "tools/serve-up.sh no longer boots a resident weight owner; every phase would reload the checkpoint"
grep -qF -- "-o \"\${OUT}/${want_owner}\"" "${REPO_ROOT}/tools/ds4/build.sh" \
  || fail "tools/ds4/build.sh no longer links ${want_owner}; the serve would have no weight owner to boot"
ok "weight owner ${want_owner}: the fixture, the serve script and the build agree the checkpoint loads once per window"
depth_pins="$(jq -r '(.serve_configuration.depths // {}) | to_entries | map(.key + "=" + (.value.num_speculative_tokens|tostring) + "/" + (.value.ds4_mtp_draft_tokens|tostring)) | join(" ")' "${CONTRACT}")"
[[ "${depth_pins}" == "serial=0/1 mtp1=1/2 mtp2=2/3 mtp3=3/4" ]] \
  || fail "serve_configuration.depths must pin exactly serial=0/1 mtp1=1/2 mtp2=2/3 mtp3=3/4 (num_speculative_tokens/ds4_mtp_draft_tokens; got '${depth_pins}')"
ok "serve configuration pins verified (engine env exported by serve-up.sh; depths serial/mtp1/mtp2/mtp3 pinned)"

# --- 8. the paired, per-box baseline ----------------------------------------
# PAIRED WITH A PER-BOX BASELINE (David 2026-09-08). The ranked job measures TWO
# legs on THIS box, in THIS job, over the ONE live golden:
#
#   leg 1  the SERIAL control, on the organizer-staged REFERENCE tree
#          (MLXFAST_BASELINE_WORKSPACE), speculation off, always;
#   leg 2  the candidate, on this checkout, at its declared draft depth.
#
# The score is the ratio of the two legs. NOTHING stores the pair: not a
# constant, not the fixture, not a golden (section 3's goldens are checked for
# that at rest by tools/lint-benchmark-manifest.py). So the two things this
# section must establish are that the REFERENCE TREE IS THE RIGHT TREE and that
# THIS BOX HAS A HEALTH BAND for its leg 1.
#
# THE CALIBRATION IS A HEALTH BAND, NEVER A DENOMINATOR. It says what leg 1 has
# measured on this box before. If leg 1 lands outside the band the run dies and
# seals nothing, because a control leg that moved means the box moved -- and a
# ratio taken on a moved box is not this track's number. It is never divided by.
for var in MLXFAST_BASELINE_WORKSPACE MLXFAST_BASELINE_CALIBRATION; do
  eval "value=\${${var}:-}"
  [[ -n "${value}" ]] \
    || fail "${var} is not set in the ranked job's environment; the paired path measures its serial control on the staged reference tree against this box's band, and neither can be resolved without it"
done
command -v git >/dev/null 2>&1 || fail "git is required to verify the reference workspace HEAD"

BASELINE_WORKSPACE="${MLXFAST_BASELINE_WORKSPACE}"
BASELINE_CALIBRATION="${MLXFAST_BASELINE_CALIBRATION}"
[[ -d "${BASELINE_WORKSPACE}" ]] \
  || fail "MLXFAST_BASELINE_WORKSPACE=${BASELINE_WORKSPACE} is not a directory on this box; stage the reference tree with tools/stage-baseline-workspace.sh"

# The reference tree and the candidate tree are TWO trees. One tree cannot be
# both legs: leg 1 would then measure the submission and the score would be the
# candidate divided by itself.
baseline_real="$(cd "${BASELINE_WORKSPACE}" && pwd -P)"
[[ "${baseline_real}" != "${REPO_ROOT}" ]] \
  || fail "MLXFAST_BASELINE_WORKSPACE points at this checkout (${REPO_ROOT}); the serial control must run on the reference tree, not on the submission"

# THE REFERENCE COMMIT IS THE FIXTURE'S. baseline_reference_commit names the
# promoted baseline every box's leg 1 runs, so one board compares one control.
WANT_REF_COMMIT="$(jq -r '.baseline_reference_commit // empty' "${CONTRACT}")"
[[ "${WANT_REF_COMMIT}" =~ ^[0-9a-f]{40}$ ]] \
  || fail "fixtures/qwen3_8_125b_a6b_track.json carries no 40-hex baseline_reference_commit (got '${WANT_REF_COMMIT}'); the serial-control leg has no reference tree to run on"
have_ref_commit="$(git -C "${BASELINE_WORKSPACE}" rev-parse HEAD 2>/dev/null || true)"
[[ "${have_ref_commit}" == "${WANT_REF_COMMIT}" ]] \
  || fail "the reference workspace is at ${have_ref_commit:-no git HEAD} but the fixture pins ${WANT_REF_COMMIT}; re-stage it with tools/stage-baseline-workspace.sh"
git -C "${BASELINE_WORKSPACE}" diff --quiet HEAD 2>/dev/null \
  || fail "the reference workspace has uncommitted changes; leg 1 must run the pinned reference tree, not an edited one"
ok "reference workspace at ${WANT_REF_COMMIT:0:12}, clean: ${BASELINE_WORKSPACE}"

# THE REFERENCE TREE MUST BE BUILT. A staged-but-unbuilt reference tree fails
# only when serve-up.sh boots it -- after the candidate has already paid for the
# checkpoint and the build. These are exactly the outputs tools/ds4/build.sh
# links and tools/stage-cuda-engine.sh copies.
for rel in .build/ds4/libds4qwen.so .build/ds4/ds4-resident .build/release/mlxfast-runtime-worker; do
  [[ -f "${BASELINE_WORKSPACE}/${rel}" ]] \
    || fail "the reference workspace has no ${rel}; its ds4 build is not staged (run tools/ds4/build.sh and tools/stage-cuda-engine.sh in ${BASELINE_WORKSPACE}, or re-stage with tools/stage-baseline-workspace.sh)"
done
[[ -x "${BASELINE_WORKSPACE}/.build/ds4/ds4-resident" ]] \
  || fail "the reference workspace's .build/ds4/ds4-resident is not executable; serve-up.sh could not boot the serial control leg"
[[ -x "${BASELINE_WORKSPACE}/tools/serve-up.sh" ]] \
  || fail "the reference workspace has no executable tools/serve-up.sh; leg 1 boots the REFERENCE tree's own serve script, not this checkout's"
# WHICH serve-up.sh IS IN THERE IS SETTLED BY THE COMMIT PIN ABOVE, not by a
# grep here: the workspace is at baseline_reference_commit and is clean, so its
# serve script is exactly that commit's. What has to be true of that COMMIT --
# that it speaks the per-leg --boot/--stop verbs benchd boots the control leg
# with -- is a property of the fixture's choice of reference, and
# tools/lint-benchmark-manifest.py checks it at rest, where it can be fixed.
ok "reference workspace carries its ds4 build (libds4qwen.so, ds4-resident) and the staged adapter"

# --- 8b. this box's calibration file ----------------------------------------
[[ -f "${BASELINE_CALIBRATION}" ]] \
  || fail "MLXFAST_BASELINE_CALIBRATION=${BASELINE_CALIBRATION} is not a file; calibrate this box with tools/calibrate-box.sh"
jq -e . >/dev/null 2>&1 < "${BASELINE_CALIBRATION}" \
  || fail "the baseline calibration is not valid JSON: ${BASELINE_CALIBRATION}"

cal_version="$(jq -r '.version // empty' "${BASELINE_CALIBRATION}")"
[[ "${cal_version}" == "1" ]] \
  || fail "the baseline calibration declares version '${cal_version}'; this preflight reads version 1 only"

TRACK_ID="$(jq -r '.track_id // empty' "${CONTRACT}")"
cal_track="$(jq -r '.track_id // empty' "${BASELINE_CALIBRATION}")"
[[ "${cal_track}" == "${TRACK_ID}" ]] \
  || fail "the baseline calibration names track '${cal_track}' but this track is '${TRACK_ID}'; a band from another track does not describe this leg"

# THE BOX NAMES ITSELF. A band is a property of ONE machine, so it is attributed
# to the runner it was captured on. An unnamed runner is refused rather than
# waved through: a band that cannot be attributed is a band from anywhere.
cal_box="$(jq -r '.box // empty' "${BASELINE_CALIBRATION}")"
[[ -n "${cal_box}" ]] || fail "the baseline calibration names no box; a per-box band must say which box"
[[ -n "${RUNNER_NAME:-}" ]] \
  || fail "the baseline calibration names box '${cal_box}' but this job has no RUNNER_NAME; a per-box band cannot be attributed to a runner that does not identify itself"
[[ "${cal_box}" == "${RUNNER_NAME}" ]] \
  || fail "the baseline calibration was captured on box '${cal_box}' but this job runs on '${RUNNER_NAME}'; another box's band does not describe this box"

cal_ref="$(jq -r '.reference_commit // empty' "${BASELINE_CALIBRATION}")"
[[ "${cal_ref}" == "${WANT_REF_COMMIT}" ]] \
  || fail "the baseline calibration was captured against reference commit '${cal_ref}' but the fixture pins ${WANT_REF_COMMIT}; re-calibrate this box against the promoted reference"

LIVE_GOLDEN_NAME_CAL="$(jq -r '.prompt // empty' "${BASELINE_CALIBRATION}")"
[[ "${LIVE_GOLDEN_NAME_CAL}" == "${LIVE_GOLDEN_NAME}" ]] \
  || fail "the baseline calibration was captured on prompt '${LIVE_GOLDEN_NAME_CAL}' but the fixture's live_golden is '${LIVE_GOLDEN_NAME}'; the band must describe the prompt leg 1 runs"

cal_passes="$(jq -r '.passes // empty' "${BASELINE_CALIBRATION}")"
[[ "${cal_passes}" =~ ^[2-9][0-9]*$ ]] \
  || fail "the baseline calibration records ${cal_passes:-no} pass(es); a band needs at least two, because the CV is undefined below two"

# THE NUMBERS. A mean that is zero, negative or absent is not a band, and a CV
# above 1 % means the box was not stable when it was calibrated -- the same gate
# the calibrator applies, re-applied here so a hand-edited file cannot widen it.
for key in prefill_seconds_per_token_mean decode_seconds_per_token_mean; do
  value="$(jq -r --arg k "${key}" '.[$k] // empty' "${BASELINE_CALIBRATION}")"
  [[ -n "${value}" ]] || fail "the baseline calibration carries no ${key}; there is no band centre on that axis"
  [[ "$(jq -n --argjson v "${value}" '$v > 0 and $v < 60')" == "true" ]] \
    || fail "the baseline calibration's ${key} is ${value} s/tok, which is not a plausible per-token time; the band centre is unusable"
done
for key in prefill_cv decode_cv; do
  value="$(jq -r --arg k "${key}" '.[$k] // empty' "${BASELINE_CALIBRATION}")"
  [[ -n "${value}" ]] || fail "the baseline calibration carries no ${key}; there is no evidence the mean describes this box"
  [[ "$(jq -n --argjson v "${value}" '$v >= 0 and $v <= 0.01')" == "true" ]] \
    || fail "the baseline calibration's ${key} is ${value}, above the 1 % stability gate; this box was not stable when it was calibrated, so re-calibrate it"
done

# THE BANDS MUST BE ABLE TO FAIL. A band that does not bracket 1 rejects a
# healthy box; a band wide enough to admit a box running at half speed accepts a
# broken one. Both are refused by name.
for axis in prefill decode; do
  low="$(jq -r --arg k "${axis}_band_low" '.[$k] // empty' "${BASELINE_CALIBRATION}")"
  high="$(jq -r --arg k "${axis}_band_high" '.[$k] // empty' "${BASELINE_CALIBRATION}")"
  [[ -n "${low}" && -n "${high}" ]] \
    || fail "the baseline calibration carries no ${axis}_band_low/${axis}_band_high; leg 1 would be gated by nothing on that axis"
  [[ "$(jq -n --argjson l "${low}" --argjson h "${high}" '$l > 0.5 and $l <= 1 and $h >= 1 and $h < 2 and $l < $h')" == "true" ]] \
    || fail "the ${axis} band [${low}, ${high}] is not a usable band: it must bracket 1 and stay inside (0.5, 2), or it either rejects a healthy box or admits one running at half speed"
done

cal_benchd="$(jq -r '.benchd_source_commit // empty' "${BASELINE_CALIBRATION}")"
[[ "${cal_benchd}" =~ ^[0-9a-f]{40}$ ]] \
  || fail "the baseline calibration records benchd_source_commit '${cal_benchd}'; a band captured by an unattributable benchd cannot be traced to the code that measured it"

# TIMESTAMPS. The band must be YOUNGER than the reference commit it describes --
# a band captured before the reference tree existed describes a different engine
# -- and it may not be dated in the future, which is a clock fault on the box.
cal_at="$(jq -r '.captured_at // empty' "${BASELINE_CALIBRATION}")"
cal_epoch="$(jq -r '(.captured_at // "") | try fromdateiso8601 catch empty' "${BASELINE_CALIBRATION}")"
[[ -n "${cal_epoch}" ]] \
  || fail "the baseline calibration's captured_at ('${cal_at}') is not an RFC 3339 UTC timestamp; the band cannot be dated"
ref_epoch="$(git -C "${BASELINE_WORKSPACE}" show -s --format=%ct HEAD 2>/dev/null || true)"
[[ "${ref_epoch}" =~ ^[0-9]+$ ]] \
  || fail "cannot read the reference commit's date from ${BASELINE_WORKSPACE}; the band's age cannot be checked"
now_epoch="$(date -u +%s)"
(( cal_epoch >= ref_epoch )) \
  || fail "the baseline calibration is dated ${cal_at}, BEFORE reference commit ${WANT_REF_COMMIT:0:12}; a band captured before the reference tree existed describes a different engine"
(( cal_epoch <= now_epoch + 300 )) \
  || fail "the baseline calibration is dated ${cal_at}, in the future; the box clock is wrong and every age check here is meaningless"
cal_sha="$(shasum -a 256 "${BASELINE_CALIBRATION}" | awk '{print $1}')"
ok "box band for ${cal_box}: ${cal_passes} passes on ${LIVE_GOLDEN_NAME_CAL}, captured ${cal_at} against ${WANT_REF_COMMIT:0:12} (sha256 ${cal_sha:0:12})"

echo "ranked-box-preflight: all checks passed"
