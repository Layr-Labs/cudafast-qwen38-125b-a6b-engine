#!/usr/bin/env bash
#
# test-benchmark-facade-env.sh -- what the benchd facade hands to benchd.
#
# TWO PROPERTIES, both facade-specific divergences from the upstream reference:
# the TRACK ID export, and the DECLARED SPEC + RESIDENT dispatch.
#
# `benchd iterate` REQUIRES MLXFAST_QWEN_MTP_TRACK_ID in every mode (its
# `-{platform}-v{N}` suffix keys the official baseline pair; the official path
# resolves it before the gates-only branch), and this repository once exported
# it nowhere, so `./benchmark.sh --local-iterate` and `--official` both refused
# with "no track_id". tools/benchmark.sh now reads benchmark.json's `trackId`
# and exports it before dispatch. This suite pins that, by driving the REAL
# facade with a STUB benchd that records the environment and argv it was
# spawned with. Hermetic: no toolchain, weights, GPU, benchd binary or network.
#
# Cases:
#   1. --local-iterate: the stub sees MLXFAST_QWEN_MTP_TRACK_ID == benchmark.json
#      trackId, non-empty, on an `iterate --mode local-iterate` argv.
#   2. --official (gates-only env): the same export on `--mode official`.
#   3. a manifest with NO trackId: the facade refuses (exit 1, names the field)
#      and benchd is never spawned.
#   4. a caller pre-sets a DIFFERENT MLXFAST_QWEN_MTP_TRACK_ID: refused, never
#      overridden, benchd never spawned.
#   5. a caller pre-sets the SAME value: accepted.
#   6. an mtp1 declaration: the argv carries --mtp-depth 1.
#   7. an mtp1 declaration against a benchd whose help does not list
#      --mtp-depth: refused, and benchd is never spawned for the run.
#   8. no DS4_MODEL: the run is dispatched THROUGH tools/serve-up.sh, so the
#      window boots one resident instead of loading per phase.
#
# Cases 1-5 export DS4_MODEL. That is the facade's in-process diagnostic hatch,
# and it is what keeps those five cases hermetic: with it the facade spawns
# benchd directly, with no serve, no weights and no resident binary. Case 8 is
# the one that drops it, and it asserts the dispatch that the other five skip.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FACADE="${REPO_ROOT}/tools/benchmark.sh"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

failures=0
fail() {
  echo "FAIL: $*" >&2
  failures=$((failures + 1))
}

command -v jq >/dev/null 2>&1 || { echo "test-benchmark-facade-env.sh: jq is required" >&2; exit 1; }

EXPECTED="$(jq -r '.trackId // empty' "${REPO_ROOT}/benchmark.json")"
if [[ -z "${EXPECTED}" ]]; then
  echo "test-benchmark-facade-env.sh: benchmark.json carries no trackId; nothing to pin" >&2
  exit 1
fi

# The stub benchd: answers `iterate --help` with STUB_HELP (the facade greps it
# for --mtp-depth), and otherwise records the track id env (or __UNSET__) and its
# argv, emits an empty JSON payload the way benchd would on stdout, exits 0.
STUB="${WORK}/benchd-stub"
cat > "${STUB}" <<'STUBEOF'
#!/usr/bin/env bash
if [[ "${1:-}" == "iterate" && "${2:-}" == "--help" ]]; then
  printf '%s\n' "${STUB_HELP-}"
  exit 0
fi
printf '%s\n' "${MLXFAST_QWEN_MTP_TRACK_ID-__UNSET__}" > "${STUB_CAPTURE_ENV}"
printf '%s\n' "$@" > "${STUB_CAPTURE_ARGV}"
echo '{}'
exit 0
STUBEOF
chmod 755 "${STUB}"

# The declaration the facade derives the spec from. spec-declaration.sh reads
# SPEC_DECLARATION_MANIFEST, so a case can declare mtp1 without touching the
# repository's own mtp-head.manifest.json.
MTP1_MANIFEST="${WORK}/mtp1.manifest.json"
echo '{"spec":{"enabled":true,"num_speculative_tokens":1}}' > "${MTP1_MANIFEST}"

GOLDEN="${WORK}/golden.json"
echo '{}' > "${GOLDEN}"

# run_facade CASE FACADE_PATH [facade args...] -- runs a facade with the stub
# benchd; captures land in ${WORK}/<case>.env / .argv / .out; sets rc.
run_facade() {
  local case_name="$1" facade="$2"
  shift 2
  rm -f "${WORK}/${case_name}.env" "${WORK}/${case_name}.argv"
  env -u MLXFAST_QWEN_MTP_TRACK_ID \
    STUB_CAPTURE_ENV="${WORK}/${case_name}.env" \
    STUB_CAPTURE_ARGV="${WORK}/${case_name}.argv" \
    STUB_HELP="    --mtp-depth <N>   Request the native-MTP speculative leg" \
    BENCHD="${STUB}" \
    MLXFAST_ENGINE_BIN="${STUB}" \
    DS4_MODEL="${WORK}/in-process-diagnostic.gguf" \
    MLXFAST_CORRECTNESS_GOLDEN_PATH="${GOLDEN}" \
    MLXFAST_SCORE_PATH="${WORK}/${case_name}.score.json" \
    MLXFAST_INTEGRITY_PATH="${WORK}/${case_name}.integrity.json" \
    "${EXTRA_ENV[@]}" \
    "${facade}" "$@" > "${WORK}/${case_name}.out" 2>&1
  rc=$?
}

# Case 1: --local-iterate exports the manifest's trackId.
EXTRA_ENV=(MLXFAST_NO_SANDBOX=1)
run_facade case1 "${FACADE}" --local-iterate
if [[ "${rc}" -ne 0 ]]; then
  fail "case 1: facade exited ${rc} with the stub benchd; output: $(cat "${WORK}/case1.out")"
fi
if [[ ! -f "${WORK}/case1.env" ]]; then
  fail "case 1: benchd was never spawned"
elif [[ "$(cat "${WORK}/case1.env")" != "${EXPECTED}" ]]; then
  fail "case 1: benchd saw MLXFAST_QWEN_MTP_TRACK_ID='$(cat "${WORK}/case1.env")', expected '${EXPECTED}'"
fi
if [[ -f "${WORK}/case1.argv" ]]; then
  if [[ "$(head -n 1 "${WORK}/case1.argv")" != "iterate" ]]; then
    fail "case 1: benchd subcommand is not iterate: $(head -n 1 "${WORK}/case1.argv")"
  fi
  if ! grep -qx -- '--mode' "${WORK}/case1.argv" || ! grep -qx -- 'local-iterate' "${WORK}/case1.argv"; then
    fail "case 1: argv does not carry --mode local-iterate: $(tr '\n' ' ' < "${WORK}/case1.argv")"
  fi
fi

# Case 2: --official, gates-only env (the seam-1 run the ranked box takes).
EXTRA_ENV=(MLXFAST_BENCHMARK_SKIP_TIMED=1 MLXFAST_BENCHMARK_CHECK_GATES=1)
run_facade case2 "${FACADE}" --official
if [[ "${rc}" -ne 0 ]]; then
  fail "case 2: facade exited ${rc} in official mode; output: $(cat "${WORK}/case2.out")"
fi
if [[ ! -f "${WORK}/case2.env" ]]; then
  fail "case 2: benchd was never spawned in official mode"
elif [[ "$(cat "${WORK}/case2.env")" != "${EXPECTED}" ]]; then
  fail "case 2: official mode: benchd saw MLXFAST_QWEN_MTP_TRACK_ID='$(cat "${WORK}/case2.env")', expected '${EXPECTED}'"
fi
if [[ -f "${WORK}/case2.argv" ]] && ! grep -qx -- 'official' "${WORK}/case2.argv"; then
  fail "case 2: argv does not carry --mode official: $(tr '\n' ' ' < "${WORK}/case2.argv")"
fi

# Case 3: a manifest with NO trackId refuses before benchd is spawned. The
# facade is copied into a throwaway tree whose benchmark.json lacks the field;
# it resolves the manifest relative to its own location, so nothing else moves.
mkdir -p "${WORK}/notrack/tools"
cp "${FACADE}" "${WORK}/notrack/tools/benchmark.sh"
jq 'del(.trackId)' "${REPO_ROOT}/benchmark.json" > "${WORK}/notrack/benchmark.json"
EXTRA_ENV=(MLXFAST_NO_SANDBOX=1)
run_facade case3 "${WORK}/notrack/tools/benchmark.sh" --local-iterate
if [[ "${rc}" -eq 0 ]]; then
  fail "case 3: facade ran with a manifest that carries no trackId"
fi
if ! grep -q 'trackId' "${WORK}/case3.out"; then
  fail "case 3: refusal does not name trackId; got: $(cat "${WORK}/case3.out")"
fi
if [[ -f "${WORK}/case3.env" ]]; then
  fail "case 3: benchd was spawned despite the missing trackId"
fi

# Case 4: a caller pre-sets a DIFFERENT track id -> refused, not overridden.
EXTRA_ENV=(MLXFAST_NO_SANDBOX=1 MLXFAST_QWEN_MTP_TRACK_ID=some-other-track-mlx-v9)
run_facade case4 "${FACADE}" --local-iterate
if [[ "${rc}" -eq 0 ]]; then
  fail "case 4: facade ran with a caller-set track id that differs from benchmark.json"
fi
if ! grep -q 'MLXFAST_QWEN_MTP_TRACK_ID' "${WORK}/case4.out"; then
  fail "case 4: refusal does not name the variable; got: $(cat "${WORK}/case4.out")"
fi
if [[ -f "${WORK}/case4.env" ]]; then
  fail "case 4: benchd was spawned despite the track id mismatch"
fi

# Case 5: a caller pre-sets the SAME value -> accepted, same export.
EXTRA_ENV=(MLXFAST_NO_SANDBOX=1 "MLXFAST_QWEN_MTP_TRACK_ID=${EXPECTED}")
run_facade case5 "${FACADE}" --local-iterate
if [[ "${rc}" -ne 0 ]]; then
  fail "case 5: facade refused a caller-set track id equal to benchmark.json's; output: $(cat "${WORK}/case5.out")"
fi
if [[ ! -f "${WORK}/case5.env" ]] || [[ "$(cat "${WORK}/case5.env")" != "${EXPECTED}" ]]; then
  fail "case 5: benchd did not see the expected track id"
fi

# Case 6: an mtp1 declaration puts --mtp-depth 1 on the argv. Without it benchd
# sends free_decode_begin with no spec, the adapter resolves serial, and the
# declared leg is timed as a serial one.
EXTRA_ENV=(MLXFAST_NO_SANDBOX=1 "SPEC_DECLARATION_MANIFEST=${MTP1_MANIFEST}")
run_facade case6 "${FACADE}" --local-iterate
if [[ "${rc}" -ne 0 ]]; then
  fail "case 6: facade exited ${rc} on an mtp1 declaration; output: $(cat "${WORK}/case6.out")"
fi
if [[ ! -f "${WORK}/case6.argv" ]]; then
  fail "case 6: benchd was never spawned"
else
  if ! grep -qx -- '--mtp-depth' "${WORK}/case6.argv"; then
    fail "case 6: argv carries no --mtp-depth under an mtp1 declaration: $(tr '\n' ' ' < "${WORK}/case6.argv")"
  fi
  depth="$(grep -A1 -x -- '--mtp-depth' "${WORK}/case6.argv" | tail -n 1)"
  if [[ "${depth}" != "1" ]]; then
    fail "case 6: --mtp-depth is '${depth}', expected 1"
  fi
fi

# Case 7: the same declaration against a benchd whose help does not list the
# flag. A serial run under an mtp declaration is the failure this refuses.
EXTRA_ENV=(MLXFAST_NO_SANDBOX=1 "SPEC_DECLARATION_MANIFEST=${MTP1_MANIFEST}" STUB_HELP=)
run_facade case7 "${FACADE}" --local-iterate
if [[ "${rc}" -eq 0 ]]; then
  fail "case 7: facade ran an mtp1 declaration on a benchd with no --mtp-depth"
fi
if ! grep -q -- '--mtp-depth' "${WORK}/case7.out"; then
  fail "case 7: refusal does not name --mtp-depth; got: $(cat "${WORK}/case7.out")"
fi
if [[ -f "${WORK}/case7.argv" ]]; then
  fail "case 7: benchd was spawned for the run despite the refusal"
fi

# Case 8: with no DS4_MODEL the run goes THROUGH tools/serve-up.sh, so the window
# boots one resident and every phase connects instead of loading. There is no
# target snapshot here, so serve-up refuses -- and that refusal, in serve-up's
# own words with benchd never spawned, is what proves the dispatch.
rm -f "${WORK}/case8.env" "${WORK}/case8.argv"
# The stub must advertise --mtp-depth here: a tree whose mtp-head.manifest.json
# declares a draft depth (every promoted submission does) otherwise stops at
# the facade's benchd probe, before the dispatch this case proves.
env -u MLXFAST_QWEN_MTP_TRACK_ID -u DS4_MODEL \
  STUB_HELP="    --mtp-depth <N>   Request the native-MTP speculative leg" \
  STUB_CAPTURE_ENV="${WORK}/case8.env" \
  STUB_CAPTURE_ARGV="${WORK}/case8.argv" \
  BENCHD="${STUB}" \
  MLXFAST_ENGINE_BIN="${STUB}" \
  MLXFAST_CORRECTNESS_GOLDEN_PATH="${GOLDEN}" \
  MLXFAST_SCORE_PATH="${WORK}/case8.score.json" \
  MLXFAST_INTEGRITY_PATH="${WORK}/case8.integrity.json" \
  MLXFAST_WEIGHTS_PATH="${WORK}/no-such-snapshot" \
  MLXFAST_NO_SANDBOX=1 \
  "${FACADE}" --local-iterate > "${WORK}/case8.out" 2>&1
rc=$?
if [[ "${rc}" -eq 0 ]]; then
  fail "case 8: the facade ran with no resident and no target snapshot"
fi
if ! grep -q 'serve-up.sh' "${WORK}/case8.out"; then
  fail "case 8: the run was not dispatched through serve-up.sh; got: $(cat "${WORK}/case8.out")"
fi
if [[ -f "${WORK}/case8.argv" ]]; then
  fail "case 8: benchd was spawned without a resident window"
fi

if [[ "${failures}" -eq 0 ]]; then
  echo "test-benchmark-facade-env.sh: all 8 cases passed (trackId=${EXPECTED})"
  exit 0
fi
echo "test-benchmark-facade-env.sh: ${failures} case(s) failed" >&2
exit 1
