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
           DS4_MODEL DS4_MTP_PATH DS4_MTP_DRAFT_TOKENS DS4_CTX_SIZE DS4_EOS_IDS)

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

total=$(( ${#CREDENTIALS[@]} + ${#OVERRIDES[@]} ))
if [[ "${failures}" -eq 0 ]]; then
  echo "test-ranked-box-preflight-env.sh: all ${total} refusals and the control passed"
  exit 0
fi
echo "test-ranked-box-preflight-env.sh: ${failures} case(s) failed" >&2
exit 1
