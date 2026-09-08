#!/usr/bin/env bash
#
# test-ranked-box-preflight-env.sh -- the ranked preflight's environment gate.
#
# tools/ranked-box-preflight.sh section 1 refuses a CREDENTIAL in the ranked
# job's environment, and section 2 refuses a MEASUREMENT-WEAKENING OVERRIDE.
# Both lists are just names in a `for` loop, and a name that falls off one is
# invisible: the run proceeds, measures something else, and seals a normal-
# looking artifact. So the lists are pinned here, one case per name, by running
# the REAL preflight with that one variable set and requiring a refusal that
# names it.
#
# The control case runs with none of them set and requires the preflight to get
# PAST section 2. Whatever it does after that is not this suite's business --
# the later sections need staged box assets.
#
# Hermetic: no toolchain, weights, GPU, benchd binary, staged golden or network.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFLIGHT="${REPO_ROOT}/tools/ranked-box-preflight.sh"

failures=0
fail() {
  echo "FAIL: $*" >&2
  failures=$((failures + 1))
}

# Every name the two gates refuse. The environment is cleared of all of them for
# each case, so one case sets exactly one.
CREDENTIALS=(R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY MLXFAST_QWEN38_R2_DOWNLOADER BENCHD_DIST_TOKEN)
OVERRIDES=(BENCHD MLXFAST_SKIP_WEIGHTS_DOWNLOAD SKIP_MODEL_DOWNLOAD MLXFAST_LOCAL_COOL_GATE
           MLXFAST_LOCAL_ALLOW_GOLDEN_DRIFT MLXFAST_SKIP_ENGINE_BUILD MLXFAST_SKIP_WEIGHTS_SHA256
           DS4_MODEL DS4_MTP_PATH DS4_MTP_DRAFT_TOKENS DS4_CTX_SIZE DS4_EOS_IDS
           DS4_RESIDENT_SOCKET BENCH_WORKER_RESIDENT_SOCKET)

clear_args=()
for var in "${CREDENTIALS[@]}" "${OVERRIDES[@]}"; do
  clear_args+=(-u "${var}")
done

# The control: with a clean environment the preflight reaches section 3.
control_out="$(env "${clear_args[@]}" "${PREFLIGHT}" 2>&1)"
if ! grep -q 'no R2 credential, signer, or dist token in the job environment' <<<"${control_out}"; then
  fail "control: the preflight did not pass the credential gate on a clean environment; got: ${control_out}"
fi
if ! grep -q 'no measurement-weakening override in the job environment' <<<"${control_out}"; then
  fail "control: the preflight did not pass the override gate on a clean environment; got: ${control_out}"
fi

# One case per name: set it, and require a refusal that names it.
check_refusal() {
  local var="$1" expected="$2" out rc
  out="$(env "${clear_args[@]}" "${var}=set-by-the-test" "${PREFLIGHT}" 2>&1)"
  rc=$?
  if [[ "${rc}" -eq 0 ]]; then
    fail "${var}: the preflight exited 0 with ${var} set in the environment"
    return
  fi
  if ! grep -q "REFUSING -- ${var} is set" <<<"${out}"; then
    fail "${var}: the refusal does not name it; got: $(tail -2 <<<"${out}")"
    return
  fi
  if ! grep -q "${expected}" <<<"${out}"; then
    fail "${var}: refused for the wrong reason (expected '${expected}'); got: $(tail -2 <<<"${out}")"
  fi
}

for var in "${CREDENTIALS[@]}"; do
  check_refusal "${var}" 'this job must hold no credential'
done
for var in "${OVERRIDES[@]}"; do
  check_refusal "${var}" 'it weakens or bypasses what the ranked run measures'
done

# ---------------------------------------------------------------------------
# THE PAIRED, PER-BOX BASELINE GATE (section 8).
#
# Section 8 refuses a ranked box whose REFERENCE TREE is not the pinned one, or
# whose CALIBRATION FILE does not describe this box against that tree. Every one
# of those checks is a line that can silently stop refusing, and the consequence
# is not a failed run: it is a scored run whose serial control ran on the wrong
# engine, or was gated by a band from another machine.
#
# Sections 1 and 2 refuse on an environment variable alone, so the cases above
# need no box. Section 8 sits behind sections 2b to 7, which want a temperature
# reader, a staged benchd pair and a toolchain. So this half builds the SMALLEST
# scaffold that lets the REAL preflight reach section 8 -- stub nvcc/nvidia-smi,
# a fake benchd beside a manifest that describes it, and a clone of this
# repository at the fixture's own baseline_reference_commit -- and then makes ONE
# thing wrong per case.
#
# Still hermetic: no GPU, no weights, no engine, no network. The clone is a
# --shared local clone of this repository.
scaffold_failures_before="${failures}"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

FIXTURE="${REPO_ROOT}/fixtures/qwen3_8_125b_a6b_track.json"
TRACK_ID="$(jq -r '.track_id' "${FIXTURE}")"
REF_COMMIT="$(jq -r '.baseline_reference_commit' "${FIXTURE}")"
LIVE_GOLDEN="$(jq -r '.live_golden' "${FIXTURE}")"
NVCC_PIN="$(jq -r '.serve_configuration.toolchain.nvcc_version' "${FIXTURE}")"
DRIVER_PIN="$(jq -r '.serve_configuration.toolchain.driver_min' "${FIXTURE}")"
RUNNER="test-box-qwen38-125b-a6b-cuda"

# The scaffold is only buildable where the pinned reference commit is reachable
# from this checkout. That is the normal case (it is a commit of this
# repository); when it is not, the suite says so rather than passing quietly.
if ! git -C "${REPO_ROOT}" cat-file -e "${REF_COMMIT}^{commit}" 2>/dev/null; then
  echo "SKIP: the fixture's baseline_reference_commit ${REF_COMMIT} is not in this checkout, so the section 8 scaffold cannot be built here" >&2
  scaffold_ready=0
else
  scaffold_ready=1
fi

if [[ "${scaffold_ready}" == "1" ]]; then
  STUB="${WORK}/bin"
  mkdir -p "${STUB}"
  cat > "${STUB}/nvcc" <<EOF
#!/usr/bin/env bash
printf 'nvcc: NVIDIA (R) Cuda compiler driver\nCuda compilation tools, release 13.0, ${NVCC_PIN}\n'
EOF
  cat > "${STUB}/nvidia-smi" <<EOF
#!/usr/bin/env bash
printf '%s\n' "${DRIVER_PIN}"
EOF
  chmod 755 "${STUB}"/*

  # A benchd that matches the manifest beside it. Section 6 verifies exactly
  # that pair, and nothing here ever runs the binary.
  BENCHD_DIR="${WORK}/benchd-bin"
  mkdir -p "${BENCHD_DIR}"
  printf 'not a real benchd\n' > "${BENCHD_DIR}/benchd"
  chmod 755 "${BENCHD_DIR}/benchd"
  jq -n --arg s "$(shasum -a 256 "${BENCHD_DIR}/benchd" | awk '{print $1}')" \
        --argjson b "$(wc -c < "${BENCHD_DIR}/benchd" | tr -d '[:space:]')" \
        '{sha256: $s, bytes: $b, source_commit: "0000000000000000000000000000000000000000"}' \
    > "${BENCHD_DIR}/benchd.manifest.json"

  # The reference tree: this repository at the pinned commit, plus the build
  # outputs tools/ds4/build.sh links and tools/stage-cuda-engine.sh copies.
  WS="${WORK}/reference"
  git clone --quiet --shared --no-checkout "${REPO_ROOT}" "${WS}" 2>/dev/null
  git -C "${WS}" checkout --quiet --detach "${REF_COMMIT}" 2>/dev/null
  mkdir -p "${WS}/.build/ds4" "${WS}/.build/release"
  : > "${WS}/.build/ds4/libds4qwen.so"
  : > "${WS}/.build/ds4/ds4-resident"
  : > "${WS}/.build/release/mlxfast-runtime-worker"
  chmod +x "${WS}/.build/ds4/ds4-resident" "${WS}/.build/release/mlxfast-runtime-worker"

  # A healthy calibration for this box, against that tree, captured now (after
  # the reference commit and before the clock).
  write_calibration() { # write_calibration PATH [jq filter]
    local path="$1" filter="${2:-.}"
    jq -n --arg track "${TRACK_ID}" --arg box "${RUNNER}" --arg ref "${REF_COMMIT}" \
          --arg prompt "${LIVE_GOLDEN}" --arg at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" '
      {
        version: 1,
        track_id: $track,
        box: $box,
        reference_commit: $ref,
        prompt: $prompt,
        passes: 4,
        prefill_seconds_per_token_mean: 0.0006282488193359375,
        decode_seconds_per_token_mean: 0.0329116748046875,
        prefill_cv: 0.004,
        decode_cv: 0.002,
        prefill_band_low: 0.95,
        prefill_band_high: 1.05,
        decode_band_low: 0.98,
        decode_band_high: 1.02,
        captured_at: $at,
        benchd_source_commit: "1111111111111111111111111111111111111111"
      }' | jq "${filter}" > "${path}"
  }

  CAL_OK="${WORK}/baseline-calibration.json"
  write_calibration "${CAL_OK}"

  # Run the REAL preflight with the scaffold, overriding one thing at a time.
  # KEY=VALUE arguments are set; KEY= clears one.
  run_preflight() {
    env "${clear_args[@]}" \
        PATH="${STUB}:${PATH}" \
        MLXFAST_GPU_TEMP_CMD="echo 42" \
        BENCHD_BIN_DIR="${BENCHD_DIR}" \
        RUNNER_NAME="${RUNNER}" \
        MLXFAST_BASELINE_WORKSPACE="${WS}" \
        MLXFAST_BASELINE_CALIBRATION="${CAL_OK}" \
        "$@" "${PREFLIGHT}" 2>&1
  }

  expect_refusal() { # expect_refusal LABEL NEEDLE [env overrides...]
    local label="$1" needle="$2"; shift 2
    local out rc
    out="$(run_preflight "$@")"
    rc=$?
    if [[ "${rc}" -eq 0 ]]; then
      fail "section 8 / ${label}: the preflight exited 0"
      return
    fi
    if ! grep -qF "${needle}" <<<"${out}"; then
      fail "section 8 / ${label}: the refusal does not say '${needle}'; got: $(tail -2 <<<"${out}")"
      return
    fi
    echo "ok    section 8 / ${label}: refused by name"
  }

  # The control: the whole scaffold is right, so the preflight passes EVERY
  # check. Without this, a section 8 that refused unconditionally would pass
  # every case above it.
  control8="$(run_preflight)"
  if grep -q 'ranked-box-preflight: all checks passed' <<<"${control8}"; then
    echo "ok    section 8 / control: a correctly staged box passes every check"
  else
    fail "section 8 / control: a correctly staged box did not pass; got: $(tail -3 <<<"${control8}")"
  fi

  expect_refusal "workspace unset" \
    "MLXFAST_BASELINE_WORKSPACE is not set" MLXFAST_BASELINE_WORKSPACE=
  expect_refusal "calibration unset" \
    "MLXFAST_BASELINE_CALIBRATION is not set" MLXFAST_BASELINE_CALIBRATION=
  expect_refusal "workspace absent" \
    "is not a directory on this box" "MLXFAST_BASELINE_WORKSPACE=${WORK}/absent"
  expect_refusal "workspace is the candidate" \
    "points at this checkout" "MLXFAST_BASELINE_WORKSPACE=${REPO_ROOT}"
  expect_refusal "runner unnamed" \
    "has no RUNNER_NAME" RUNNER_NAME=

  # A reference tree at the wrong commit.
  WS_OLD="${WORK}/reference-old"
  git clone --quiet --shared --no-checkout "${REPO_ROOT}" "${WS_OLD}" 2>/dev/null
  if git -C "${WS_OLD}" checkout --quiet --detach "${REF_COMMIT}~1" 2>/dev/null; then
    mkdir -p "${WS_OLD}/.build/ds4" "${WS_OLD}/.build/release"
    : > "${WS_OLD}/.build/ds4/libds4qwen.so"
    : > "${WS_OLD}/.build/ds4/ds4-resident"
    : > "${WS_OLD}/.build/release/mlxfast-runtime-worker"
    chmod +x "${WS_OLD}/.build/ds4/ds4-resident"
    expect_refusal "workspace at the wrong commit" \
      "but the fixture pins ${REF_COMMIT}" "MLXFAST_BASELINE_WORKSPACE=${WS_OLD}"
  else
    fail "section 8 / workspace at the wrong commit: cannot check out ${REF_COMMIT}~1"
  fi

  # A reference tree that was staged but never built.
  WS_UNBUILT="${WORK}/reference-unbuilt"
  git clone --quiet --shared --no-checkout "${REPO_ROOT}" "${WS_UNBUILT}" 2>/dev/null
  git -C "${WS_UNBUILT}" checkout --quiet --detach "${REF_COMMIT}" 2>/dev/null
  expect_refusal "workspace not built" \
    "its ds4 build is not staged" "MLXFAST_BASELINE_WORKSPACE=${WS_UNBUILT}"

  # A reference tree with an edit in it.
  WS_DIRTY="${WORK}/reference-dirty"
  cp -R "${WS}" "${WS_DIRTY}"
  printf 'edited\n' >> "${WS_DIRTY}/README.md"
  expect_refusal "workspace edited" \
    "has uncommitted changes" "MLXFAST_BASELINE_WORKSPACE=${WS_DIRTY}"

  # One bad field per calibration file.
  cal_case() { # cal_case LABEL JQ_FILTER NEEDLE
    local label="$1" filter="$2" needle="$3"
    local path="${WORK}/cal-$(printf '%s' "${label}" | tr -c 'a-z0-9' '-').json"
    write_calibration "${path}" "${filter}"
    expect_refusal "${label}" "${needle}" "MLXFAST_BASELINE_CALIBRATION=${path}"
  }

  cal_case "calibration version" '.version = 2' "declares version '2'"
  cal_case "calibration track" '.track_id = "another-track-v1"' "names track 'another-track-v1'"
  cal_case "calibration box" '.box = "another-box"' "was captured on box 'another-box'"
  cal_case "calibration reference commit" \
    '.reference_commit = "0000000000000000000000000000000000000000"' \
    "was captured against reference commit"
  cal_case "calibration prompt" '.prompt = "not-the-live-golden"' \
    "was captured on prompt 'not-the-live-golden'"
  cal_case "calibration passes" '.passes = 1' "a band needs at least two"
  cal_case "calibration mean" '.decode_seconds_per_token_mean = 0' \
    "is not a plausible per-token time"
  cal_case "calibration cv" '.decode_cv = 0.05' "above the 1 % stability gate"
  cal_case "calibration band does not bracket 1" '.decode_band_high = 0.99' \
    "is not a usable band"
  cal_case "calibration band admits a half-speed box" '.prefill_band_high = 2.5' \
    "is not a usable band"
  cal_case "calibration benchd" '.benchd_source_commit = "unknown"' \
    "cannot be traced to the code that measured it"
  cal_case "calibration timestamp unreadable" '.captured_at = "yesterday"' \
    "is not an RFC 3339 UTC timestamp"
  cal_case "calibration older than the reference" '.captured_at = "2000-01-01T00:00:00Z"' \
    "describes a different engine"
  cal_case "calibration dated in the future" '.captured_at = "2999-01-01T00:00:00Z"' \
    "the box clock is wrong"

  if [[ "${failures}" -eq "${scaffold_failures_before}" ]]; then
    echo "test-ranked-box-preflight-env.sh: the section 8 paired-baseline cases and the control passed"
  fi
fi

total=$(( ${#CREDENTIALS[@]} + ${#OVERRIDES[@]} ))
if [[ "${failures}" -eq 0 ]]; then
  echo "test-ranked-box-preflight-env.sh: all ${total} environment refusals, the section 8 paired-baseline cases and both controls passed"
  exit 0
fi
echo "test-ranked-box-preflight-env.sh: ${failures} case(s) failed" >&2
exit 1
