#!/usr/bin/env bash
# Exercise the real local entry point, root proxy, and facade with a stub benchd.
# No model, GPU, toolchain, credentials, or network is used.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
# Canonical (pwd -P): on macOS mktemp answers under /var, a symlink to
# /private/var, and the helper records its cwd resolved.
WORK="$(cd "$(mktemp -d)" && pwd -P)"
trap 'rm -rf "${WORK}"' EXIT
TEST_ROOT="${WORK}/checkout with spaces"
mkdir -p "${TEST_ROOT}/tools" "${TEST_ROOT}/fixtures" "${TEST_ROOT}/correctness_prompts" \
  "${TEST_ROOT}/.build/release" "${TEST_ROOT}/reference_weights/Qwen3.8-Flash-Next-GGUF"
cp "${REPO_ROOT}/benchmark.sh" "${TEST_ROOT}/"
cp "${REPO_ROOT}/tools/"{local-baseline,benchmark,spec-declaration,serve-up}.sh "${TEST_ROOT}/tools/"
cp "${REPO_ROOT}/benchmark.json" "${REPO_ROOT}/mtp-head.manifest.json" "${TEST_ROOT}/"
cp "${REPO_ROOT}/fixtures/qwen3_8_125b_a6b_track.json" "${TEST_ROOT}/fixtures/"
cp "${REPO_ROOT}/correctness_prompts/public-longcopy-gate-english-1024.golden.json" "${TEST_ROOT}/correctness_prompts/"

cat > "${WORK}/benchd" <<'STUB'
#!/usr/bin/env bash
set -euo pipefail
if [[ "${2:-}" == "--help" ]]; then
  echo '--mtp-depth'
  exit 0
fi
printf '%s\n' "$PWD" > "${CAPTURE}.cwd"
printf '%s\n' "$@" > "${CAPTURE}.argv"
printf '%s\n' "${MLXFAST_BASELINE_WORKSPACE-unset}" "${MLXFAST_BASELINE_CALIBRATION-unset}" > "${CAPTURE}.pair"
printf '%s\n' '{"score":null,"passed":true,"metrics":{"decode_seconds_per_token":0.05}}'
exit "${STUB_EXIT:-0}"
STUB
chmod +x "${WORK}/benchd"
cp "${WORK}/benchd" "${TEST_ROOT}/.build/release/mlxfast-runtime-worker"

# DS4_MODEL is the facade's in-process diagnostic hatch: with it the facade
# spawns benchd directly instead of booting a resident through serve-up.sh,
# which is what keeps these cases free of an engine and a snapshot.
# tools/test-benchmark-facade-env.sh pins the serve-up dispatch itself.
run_case() {
  local name="$1"
  shift
  rm -f "${WORK}/${name}.argv"
  rc=0
  (
    cd "${WORK}"
    env -u MLXFAST_ENGINE_BIN -u MLXFAST_CORRECTNESS_GOLDEN_PATH \
      -u MLXFAST_WEIGHTS_PATH -u MLXFAST_SCORE_PATH -u MLXFAST_TARGET_SNAPSHOT_DIR \
      -u MLXFAST_CORRECTNESS_GOLDEN_SHA256 -u MLXFAST_CORRECTNESS_GOLDEN_BYTES \
      -u MLXFAST_QWEN_MTP_TRACK_ID \
      -u SPEC_DECLARATION_MANIFEST -u SPEC_DECLARATION_CONTRACT \
      DS4_MODEL=stub BENCHD="${WORK}/benchd" CAPTURE="${WORK}/${name}" "$@"
  ) > "${WORK}/${name}.stdout" 2> "${WORK}/${name}.stderr" || rc=$?
}

assert_arg() {
  local name="$1" flag="$2" value="$3"
  awk -v flag="${flag}" -v value="${value}" '
    previous == flag && $0 == value { found=1 }
    { previous=$0 }
    END { exit !found }
  ' "${WORK}/${name}.argv"
}

run_case defaults "${TEST_ROOT}/tools/local-baseline.sh"
[[ "${rc}" == 0 ]]
[[ "$(cat "${WORK}/defaults.cwd")" == "${TEST_ROOT}" ]]
assert_arg defaults --mode local-iterate
assert_arg defaults --engine .build/release/mlxfast-runtime-worker
assert_arg defaults --weights reference_weights/Qwen3.8-Flash-Next-GGUF
assert_arg defaults --golden correctness_prompts/public-longcopy-gate-english-1024.golden.json
assert_arg defaults --score-path score.local-iterate.json
grep -Fxq -- '--cool-gate' "${WORK}/defaults.argv"
grep -q 'public local baseline (unranked)' "${WORK}/defaults.stderr"
grep -q 'score=null is expected' "${WORK}/defaults.stderr"
[[ "$(cat "${WORK}/defaults.stdout")" == '{"score":null,"passed":true,"metrics":{"decode_seconds_per_token":0.05}}' ]]

mkdir -p "${TEST_ROOT}/box snapshot"
run_case snapshot_env MLXFAST_TARGET_SNAPSHOT_DIR='box snapshot' "${TEST_ROOT}/tools/local-baseline.sh"
[[ "${rc}" == 0 ]]
assert_arg snapshot_env --weights 'box snapshot'

cp "${WORK}/benchd" "${TEST_ROOT}/custom engine"
cp "${TEST_ROOT}/correctness_prompts/public-longcopy-gate-english-1024.golden.json" "${TEST_ROOT}/custom golden.json"
mkdir -p "${TEST_ROOT}/custom weights"
run_case overrides MLXFAST_ENGINE_BIN='custom engine' \
  MLXFAST_CORRECTNESS_GOLDEN_PATH='custom golden.json' \
  MLXFAST_TARGET_SNAPSHOT_DIR='box snapshot' \
  MLXFAST_WEIGHTS_PATH='custom weights' MLXFAST_SCORE_PATH='custom result.json' \
  "${TEST_ROOT}/tools/local-baseline.sh"
[[ "${rc}" == 0 ]]
assert_arg overrides --engine 'custom engine'
assert_arg overrides --golden 'custom golden.json'
assert_arg overrides --weights 'custom weights'
assert_arg overrides --score-path 'custom result.json'

run_case inherited_pair MLXFAST_BASELINE_WORKSPACE=organizer-reference \
  MLXFAST_BASELINE_CALIBRATION=organizer-calibration.json \
  "${TEST_ROOT}/tools/local-baseline.sh"
[[ "${rc}" == 0 ]]
[[ "$(cat "${WORK}/inherited_pair.pair")" == $'unset\nunset' ]]

run_case failed STUB_EXIT=1 "${TEST_ROOT}/tools/local-baseline.sh"
[[ "${rc}" == 1 ]]
run_case help "${TEST_ROOT}/tools/local-baseline.sh" --help
[[ "${rc}" == 0 && ! -f "${WORK}/help.argv" ]]
run_case wrong_mode "${TEST_ROOT}/tools/local-baseline.sh" --official
[[ "${rc}" == 1 && ! -f "${WORK}/wrong_mode.argv" ]]
run_case missing_golden MLXFAST_CORRECTNESS_GOLDEN_PATH=absent.json "${TEST_ROOT}/tools/local-baseline.sh"
[[ "${rc}" == 1 && ! -f "${WORK}/missing_golden.argv" ]]
grep -q 'correctness golden not found' "${WORK}/missing_golden.stderr"
run_case wrong_pin MLXFAST_CORRECTNESS_GOLDEN_SHA256=bad "${TEST_ROOT}/tools/local-baseline.sh"
[[ "${rc}" == 1 && ! -f "${WORK}/wrong_pin.argv" ]]
grep -q 'sha256 mismatch' "${WORK}/wrong_pin.stderr"
run_case missing_engine MLXFAST_ENGINE_BIN=absent-engine "${TEST_ROOT}/tools/local-baseline.sh"
[[ "${rc}" == 1 && ! -f "${WORK}/missing_engine.argv" ]]
grep -q './setup.sh' "${WORK}/missing_engine.stderr"
run_case missing_snapshot MLXFAST_WEIGHTS_PATH=absent-snapshot "${TEST_ROOT}/tools/local-baseline.sh"
[[ "${rc}" == 1 && ! -f "${WORK}/missing_snapshot.argv" ]]
grep -q 'target snapshot directory not found' "${WORK}/missing_snapshot.stderr"

echo 'test-local-baseline.sh: all 11 cases passed'
