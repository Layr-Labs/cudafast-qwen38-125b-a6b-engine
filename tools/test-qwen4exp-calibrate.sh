#!/usr/bin/env bash
#
# test-qwen4exp-calibrate.sh -- the mechanics of the L12c calibration driver.
#
# The driver's value is a wiring claim: ONE resident engine for the whole
# calibration, one weights hash for every pass, benchd's own capture flow per
# golden with no spec anywhere near it, and a CV gate over the merged records --
# applying nothing. This suite pins that wiring by driving the REAL script with
# STUB collaborators (benchd, serve-up.sh, spec-declaration.sh, flock, setsid)
# that record how they were called. Hermetic: no GPU, no lock, no weights, no
# resident engine, no network.
#
# Cases:
#   1. --dry-run prints one capture line per timed-pool golden, each carrying
#      --capture-baseline and --capture-passes, and NONE carrying a spec flag;
#      it creates no run directory and never touches the lock path.
#   2. a full mechanics run boots serve-up EXACTLY ONCE (one weights load for the
#      whole calibration), hashes the weights EXACTLY ONCE and passes that digest
#      to every pass, runs one iterate per golden, and hands every merged record
#      plus --pin to one calibrate-baseline call.
#   3. it writes the report and the patch, and edits no golden, fixture or
#      constant.
#   4. a declared spec that is not serial refuses BY NAME before anything boots.
#   5. --legs 1 refuses: the sample CV is undefined below two legs.
#   6. an unknown argument refuses with exit 2.
#   7. --dry-run with BENCHD UNSET resolves no benchd: it never runs
#      tools/fetch-benchd.sh, which would write into benchd-bin/ and may reach
#      the dist channel to do it.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DRIVER="${REPO_ROOT}/tools/qwen4exp-calibrate.sh"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

failures=0
fail() { echo "FAIL: $*" >&2; failures=$((failures + 1)); }
ok()   { echo "ok: $*"; }

command -v jq >/dev/null 2>&1 || { echo "test-qwen4exp-calibrate.sh: jq is required" >&2; exit 1; }

BIN="${WORK}/bin"
mkdir -p "${BIN}"

# flock/setsid stubs: the driver only needs them to EXIST and to succeed. The
# lock itself is not under test here (the driver's own --dry-run case proves it
# is never taken), and the process-group re-exec is disabled by env below.
cat > "${BIN}/flock" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF
cat > "${BIN}/setsid" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF

# spec-declaration stub: prints whatever STUB_SPEC says.
cat > "${BIN}/spec-declaration.sh" <<'EOF'
#!/usr/bin/env bash
[ "${1:-}" = "describe" ] && { printf '%s\n' "${STUB_SPEC:-serial}"; exit 0; }
printf '0\n'
EOF

# serve-up stub: records ONE line per invocation, then execs the wrapped command
# exactly as the real serve-up.sh does after its readiness probe.
cat > "${BIN}/serve-up.sh" <<'EOF'
#!/usr/bin/env bash
printf 'serve-up weights=%s speculative=%s\n' \
  "${SERVE_UP_WEIGHTS_DIR:-}" "${SERVE_UP_SPECULATIVE:-}" >> "${STUB_SERVE_LOG}"
exec "$@"
EOF

# benchd stub: records every invocation's argv, answers `weights-digest`,
# writes a plausible four-leg capture record for `iterate --capture-baseline`,
# and answers `calibrate-baseline` with a report plus its JSON.
cat > "${BIN}/benchd" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "${STUB_BENCHD_LOG}"
case "${1:-}" in
  weights-digest)
    printf 'deadbeef:1234:7\n'
    ;;
  iterate)
    base=""
    while [ $# -gt 0 ]; do
      [ "$1" = "--capture-baseline" ] && base="$2"
      shift
    done
    [ -n "${base}" ] || exit 1
    cat > "${base%.json}.A.json" <<'REC'
{
  "track_id": "qwen3.8-125b-a6b-cuda-v1",
  "mode": "local-iterate",
  "decode_steps": 128,
  "engine_sha256": "aaaa",
  "weights_sha256": "bbbb",
  "golden_sha256": "cccc",
  "benchd_sha256": "dddd",
  "run_count": 4,
  "runs": []
}
REC
    ;;
  calibrate-baseline)
    out=""
    while [ $# -gt 0 ]; do
      [ "$1" = "--json-out" ] && out="$2"
      shift
    done
    printf 'CV GATE: PASS\nAPPLIED NOTHING\n'
    [ -n "${out}" ] && printf '{"constants_patch":"pub const OFFICIAL_BASELINE_CUDA: Option<OfficialBaseline> = None;\\n"}\n' > "${out}"
    ;;
  *) exit 1 ;;
esac
exit 0
EOF
chmod 755 "${BIN}"/*

WEIGHTS="${WORK}/weights"; mkdir -p "${WEIGHTS}"
LOCK="${WORK}/never-taken.lock"

POOL_NAMES="$(jq -r '.timed_prompt_pool[] | (.r2_path | split("/") | last | sub("\\.golden\\.json$"; ""))' \
  "${REPO_ROOT}/fixtures/qwen3_8_125b_a6b_track.json")"
POOL_COUNT="$(printf '%s\n' "${POOL_NAMES}" | wc -l | tr -d ' ')"
LIVE_GOLDEN="$(jq -r '.live_golden' "${REPO_ROOT}/fixtures/qwen3_8_125b_a6b_track.json")"

drive() { # drive OUTFILE [args...]
  local out="$1"; shift
  env PATH="${BIN}:${PATH}" \
      QWEN4EXP_CALIBRATE_SETSID=1 \
      QWEN4EXP_CALIBRATE_SERVE_UP="${BIN}/serve-up.sh" \
      QWEN4EXP_CALIBRATE_SPEC_DECLARATION="${BIN}/spec-declaration.sh" \
      STUB_SPEC="${STUB_SPEC:-serial}" \
      STUB_SERVE_LOG="${WORK}/serve.log" \
      STUB_BENCHD_LOG="${WORK}/benchd.log" \
      BENCHD="${BIN}/benchd" \
      "${DRIVER}" --weights "${WEIGHTS}" --lock "${LOCK}" "$@" > "${out}" 2>&1
  return $?
}

# --- case 1: --dry-run ------------------------------------------------------
: > "${WORK}/serve.log"; : > "${WORK}/benchd.log"
DRY="${WORK}/dry.out"
drive "${DRY}" --dry-run --run-dir "${WORK}/runs"
[ $? -eq 0 ] || fail "case 1: --dry-run exited non-zero"

missing=0
while IFS= read -r name; do
  grep -q -- "--capture-baseline <RUN>/legs/${name}.json" "${DRY}" || { fail "case 1: no capture line for ${name}"; missing=1; }
done <<< "${POOL_NAMES}"
[ "${missing}" -eq 0 ] && ok "case 1: every one of the ${POOL_COUNT} timed-pool goldens has a capture line"

grep -q -- "--capture-passes W,A,A,A,A" "${DRY}" || fail "case 1: the pass spec is not a warm-up plus four timed legs"
grep -q -- "--mtp-depth" "${DRY}"      && fail "case 1: a spec flag reached the capture; the baseline must be SERIAL"
grep -q -- "--candidate-spec" "${DRY}" && fail "case 1: a spec flag reached the capture; the baseline must be SERIAL"
if grep -q -- "--capture-passes W,A,A,A,A" "${DRY}" \
   && ! grep -q -- "--mtp-depth\|--candidate-spec" "${DRY}"; then
  ok "case 1: the capture is serial, warm-up plus four legs"
fi

[ -e "${LOCK}" ] && fail "case 1: --dry-run created the lock file ${LOCK}"
[ -d "${WORK}/runs" ] && fail "case 1: --dry-run created a run directory"
[ -s "${WORK}/serve.log" ] && fail "case 1: --dry-run booted serve-up"
[ -s "${WORK}/benchd.log" ] && fail "case 1: --dry-run ran benchd"
ok "case 1: --dry-run took no lock, booted no engine and wrote nothing"

# --- case 2 + 3: a full mechanics run ---------------------------------------
: > "${WORK}/serve.log"; : > "${WORK}/benchd.log"
RUNS="${WORK}/runs"
RUN_OUT="${WORK}/run.out"
drive "${RUN_OUT}" --run-dir "${RUNS}"
rc=$?
[ "${rc}" -eq 0 ] || { fail "case 2: the run exited ${rc}"; sed 's/^/    /' "${RUN_OUT}" >&2; }

serve_calls="$(grep -c '^serve-up ' "${WORK}/serve.log" 2>/dev/null || printf '0')"
if [ "${serve_calls}" = "1" ]; then
  ok "case 2: serve-up booted exactly once -- one weights load for the whole calibration"
else
  fail "case 2: serve-up booted ${serve_calls} time(s); the weights must load once"
fi
grep -q 'speculative=0' "${WORK}/serve.log" || fail "case 2: the resident engine was not booted SERIAL"

digest_calls="$(grep -c '^weights-digest ' "${WORK}/benchd.log" 2>/dev/null || printf '0')"
[ "${digest_calls}" = "1" ] \
  && ok "case 2: the weights tree was hashed exactly once" \
  || fail "case 2: weights-digest ran ${digest_calls} time(s); the ~105 GiB tree must be hashed once"

iterate_calls="$(grep -c '^iterate ' "${WORK}/benchd.log" 2>/dev/null || printf '0')"
[ "${iterate_calls}" = "${POOL_COUNT}" ] \
  && ok "case 2: one capture per timed-pool golden (${iterate_calls})" \
  || fail "case 2: ${iterate_calls} captures for ${POOL_COUNT} goldens"

digest_passes="$(grep -c -- '--weights-digest deadbeef:1234:7' "${WORK}/benchd.log" 2>/dev/null || printf '0')"
[ "${digest_passes}" = "${POOL_COUNT}" ] \
  && ok "case 2: every capture reused the one digest" \
  || fail "case 2: ${digest_passes} of ${POOL_COUNT} captures reused the digest"

cal="$(grep '^calibrate-baseline ' "${WORK}/benchd.log" || true)"
[ -n "${cal}" ] || fail "case 2: calibrate-baseline never ran"
records="$(printf '%s\n' "${cal}" | tr ' ' '\n' | grep -c -- '^--record$' || printf '0')"
[ "${records}" = "${POOL_COUNT}" ] \
  && ok "case 2: all ${records} merged records were gated in one call" \
  || fail "case 2: ${records} records gated for ${POOL_COUNT} goldens"
case "${cal}" in
  *"--pin ${LIVE_GOLDEN}"*) ok "case 2: the fixture's live_golden (${LIVE_GOLDEN}) is the pinned record" ;;
  *) fail "case 2: --pin did not default to the fixture's live_golden" ;;
esac

RUN_DIR="$(find "${RUNS}" -maxdepth 1 -name '*-calib' | head -n 1)"
[ -n "${RUN_DIR}" ] || fail "case 3: no run directory was created"
if [ -n "${RUN_DIR}" ]; then
  for artifact in driver.log run.pid run.pgid calibration.json calibration-report.txt official-baseline.patch; do
    [ -s "${RUN_DIR}/${artifact}" ] || fail "case 3: ${artifact} is missing or empty"
  done
  grep -q 'OFFICIAL_BASELINE_CUDA' "${RUN_DIR}/official-baseline.patch" \
    || fail "case 3: the patch does not name the constant it patches"
  ok "case 3: the run wrote its log, pid/pgid, report, JSON and patch"
fi

# APPLIES NOTHING: the tracked tree is byte-identical after the run.
if git -C "${REPO_ROOT}" diff --quiet -- fixtures correctness_prompts benchmark.json; then
  ok "case 3: the run edited no golden, fixture or manifest"
else
  fail "case 3: the run modified tracked benchmark inputs"
fi

# --- case 4: a non-serial declared spec -------------------------------------
: > "${WORK}/serve.log"; : > "${WORK}/benchd.log"
STUB_SPEC=mtp1 drive "${WORK}/spec.out" --run-dir "${WORK}/runs4"
[ $? -eq 2 ] || fail "case 4: a non-serial declaration must refuse with exit 2"
grep -q 'REFUSE declared-spec-not-serial' "${WORK}/spec.out" \
  && ok "case 4: a non-serial declaration refuses by name" \
  || fail "case 4: the refusal did not name declared-spec-not-serial"
[ -s "${WORK}/serve.log" ] && fail "case 4: it booted serve-up before refusing"

# --- case 5: too few legs ---------------------------------------------------
drive "${WORK}/legs.out" --legs 1 --dry-run
[ $? -eq 2 ] || fail "case 5: --legs 1 must refuse with exit 2"
grep -q 'undefined below two legs' "${WORK}/legs.out" \
  && ok "case 5: one leg is not a calibration" \
  || fail "case 5: the refusal did not name the undefined CV"

# --- case 6: unknown argument -----------------------------------------------
drive "${WORK}/arg.out" --nope
[ $? -eq 2 ] || fail "case 6: an unknown argument must exit 2"
grep -q 'unknown argument --nope' "${WORK}/arg.out" \
  && ok "case 6: an unknown argument refuses by name" \
  || fail "case 6: the refusal did not name the argument"

# --- case 7: --dry-run resolves no benchd ---------------------------------
# The stub PATH cannot cover this one: the driver resolves fetch-benchd.sh by
# absolute path out of the repo. So the test unsets BENCHD and puts a TRIPWIRE
# fetch-benchd.sh at that exact path -- one that fails loudly if it is ever run
# -- restoring the real script afterwards whatever happens.
FETCH="${REPO_ROOT}/tools/fetch-benchd.sh"
FETCH_BACKUP="${WORK}/fetch-benchd.sh.real"
restore_fetch() { [ -f "${FETCH_BACKUP}" ] && cp "${FETCH_BACKUP}" "${FETCH}"; }
trap 'restore_fetch; rm -rf "${WORK}"' EXIT

if [ -f "${FETCH}" ]; then
  cp "${FETCH}" "${FETCH_BACKUP}"
  cat > "${FETCH}" <<'TRIPEOF'
#!/usr/bin/env bash
printf 'TRIPWIRE: fetch-benchd.sh ran
' >&2
exit 1
TRIPEOF
  chmod 755 "${FETCH}"

  env -u BENCHD PATH="${BIN}:${PATH}"       QWEN4EXP_CALIBRATE_SETSID=1       QWEN4EXP_CALIBRATE_SERVE_UP="${BIN}/serve-up.sh"       QWEN4EXP_CALIBRATE_SPEC_DECLARATION="${BIN}/spec-declaration.sh"       STUB_SERVE_LOG="${WORK}/serve.log"       STUB_BENCHD_LOG="${WORK}/benchd.log"       "${DRIVER}" --weights "${WEIGHTS}" --lock "${LOCK}" --dry-run       > "${WORK}/nofetch.out" 2>&1
  rc=$?
  restore_fetch

  [ "${rc}" -eq 0 ] || fail "case 7: --dry-run with BENCHD unset exited ${rc}"
  if grep -q 'TRIPWIRE' "${WORK}/nofetch.out"; then
    fail "case 7: --dry-run ran fetch-benchd.sh; it must resolve nothing"
  elif grep -q 'resolved at run time' "${WORK}/nofetch.out"; then
    ok "case 7: --dry-run prints benchd as unresolved and never fetches it"
  else
    fail "case 7: --dry-run did not report benchd as unresolved"
  fi
else
  fail "case 7: ${FETCH} is missing; the tripwire has nothing to replace"
fi

if [ "${failures}" -eq 0 ]; then
  echo "PASS: test-qwen4exp-calibrate.sh"
  exit 0
fi
echo "FAIL: ${failures} failure(s)" >&2
exit 1
