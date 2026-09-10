#!/usr/bin/env bash
#
# test-qwen4exp-golden-reauthor.sh -- prove the mechanics of the golden
# re-author off a GPU, against the adapter's MOCK backend.
#
# WHAT IT PROVES
#   1. --dry-run prints the plan and every command, and runs nothing
#   2. the two named refusals fire: official-scoring-armed, engine-pin-mismatch
#   3. the prompts phase verifies each pinned golden against the contract and
#      takes its 1024 prompt ids
#   4. the capture phases produce a well-formed golden for every pool prompt,
#      for the public correctness prompt, and for the depth-1 oracle
#   5. `benchd validate-golden` ACCEPTS every one of them against the track
#      contract and its own {sha256, bytes}
#   6. the negative control REFUSES: one changed token, byte count unchanged,
#      validate-golden rejects
#   7. the run never edits fixtures/qwen3_8_125b_a6b_track.json
#   8. the recorder is taken from beside the resolved benchd when the channel
#      staged one there, and its absence refuses by name and says what the
#      channel has to publish
#
# WHAT IT IS NOT. The mock's tokens are a fixed function of its input, so the
# goldens this test produces describe nothing. It tests the MECHANICS.
#
# WHAT IT NEEDS. `cuda-engine` (built here), and benchd's `benchd` and
# `record-correctness-golden` at the release-channel commit. On a box, both come
# off the dist channel (`tools/fetch-benchd.sh` stages them together in
# `benchd-bin/`). This test resolves no channel, so name them:
#   BENCHD_RECORD_GOLDEN_BIN=... BENCHD=... tools/test-qwen4exp-golden-reauthor.sh
# or point BENCHD_SRC_DIR at a benchd checkout and this script builds both.
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null && pwd -P)"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

pass() { printf 'test-golden-reauthor: PASS -- %s\n' "$*"; }
fail() { printf 'test-golden-reauthor: FAIL -- %s\n' "$*" >&2; exit 1; }
note() { printf 'test-golden-reauthor: %s\n' "$*"; }

SCRIPT="${ROOT_DIR}/tools/qwen4exp-golden-reauthor.sh"
FIXTURE="${ROOT_DIR}/fixtures/qwen3_8_125b_a6b_track.json"
FIXTURE_SHA_BEFORE="$(shasum -a 256 "${FIXTURE}" | awk '{print $1}')"

# --- the engine: the real adapter, serving its mock backend ------------------
cargo build --quiet --manifest-path "${ROOT_DIR}/harness/protocol-adapter/Cargo.toml" \
  --bin cuda-engine || fail "cuda-engine does not build"
ENGINE="${ROOT_DIR}/harness/protocol-adapter/target/debug/cuda-engine"
pass "cuda-engine builds and serves the mock backend"

# --- benchd: benchd + the golden recorder ---------------------------------
BENCHD_BIN="${BENCHD:-}"
RECORDER_BIN="${BENCHD_RECORD_GOLDEN_BIN:-}"
if [ -z "${BENCHD_BIN}" ] || [ -z "${RECORDER_BIN}" ]; then
  [ -n "${BENCHD_SRC_DIR:-}" ] || fail "set BENCHD and BENCHD_RECORD_GOLDEN_BIN, or BENCHD_SRC_DIR to a benchd checkout at the release-channel commit"
  ( cd "${BENCHD_SRC_DIR}" && cargo build --release --bin benchd --bin record-correctness-golden >/dev/null ) \
    || fail "benchd does not build in ${BENCHD_SRC_DIR}"
  BENCHD_BIN="${BENCHD_BIN:-${BENCHD_SRC_DIR}/target/release/benchd}"
  RECORDER_BIN="${RECORDER_BIN:-${BENCHD_SRC_DIR}/target/release/record-correctness-golden}"
fi
[ -x "${BENCHD_BIN}" ] || fail "benchd is not executable: ${BENCHD_BIN}"
[ -x "${RECORDER_BIN}" ] || fail "record-correctness-golden is not executable: ${RECORDER_BIN}"
note "benchd ${BENCHD_BIN}"
note "recorder ${RECORDER_BIN}"

# --- a target snapshot the mock never reads, and a stub tokenizer -----------
WEIGHTS="${WORK}/weights"
mkdir -p "${WEIGHTS}"
: > "${WEIGHTS}/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"
: > "${WEIGHTS}/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf"

# `ds4 --dump-tokens` tokenizes the public correctness prompt with the target's
# own tokenizer. There is no ds4 here, so a stub emits the shape that command
# emits: the id array on the first line, one decoded piece per line after it.
DS4_STUB="${WORK}/ds4"
cat > "${DS4_STUB}" <<'STUB'
#!/usr/bin/env bash
python3 - <<'PY'
ids = [1000 + (i % 4096) for i in range(1100)]
print("[" + ", ".join(str(i) for i in ids) + "]")
for i in ids:
    print(f"{i:6d}  piece{i}")
PY
STUB
chmod +x "${DS4_STUB}"

COMMON=(--weights "${WEIGHTS}" --engine "${ENGINE}" --ds4 "${DS4_STUB}"
        --benchd "${BENCHD_BIN}" --recorder "${RECORDER_BIN}"
        --steps 128 --decode-steps 128 --depth 1)

# --- 1. the dry run ---------------------------------------------------------
DRY="${WORK}/dry.txt"
"${SCRIPT}" --dry-run --out "${WORK}/dry" "${COMMON[@]}" > "${DRY}" 2>&1 \
  || fail "the dry run exited non-zero"
for needle in 'record-correctness-golden' 'validate-golden' 'free-run-capture.py' \
              'mtp-exactness-gate.py' 'perturb-one-token' 'serve-up.sh' \
              'SERVE_UP_SPECULATIVE=0' 'SERVE_UP_SPECULATIVE=1' 'flock' \
              'negative-control.txt'; do
  grep -q -- "${needle}" "${DRY}" || fail "the dry run never printed ${needle}"
done
# The negative control's OWN validate-golden line must be printed. A redirect on
# the run() call would swallow it and would create the file in a directory the
# dry run never made.
grep -q -- '--golden .*negative-control.golden.json' "${DRY}" \
  || fail "the dry run never printed the negative control's validate-golden line"
if grep -qi 'No such file or directory' "${DRY}"; then fail "the dry run tried to touch the filesystem"; fi
[ -e "${WORK}/dry/negative-control.txt" ] && fail "the dry run created ${WORK}/dry/negative-control.txt"
[ -d "${WORK}/dry/goldens" ] && fail "the dry run created ${WORK}/dry/goldens"
pass "--dry-run prints every command (serve, capture, oracle, validate, negative control) and runs none of them"

# --- 2. the two named refusals ---------------------------------------------
# A fake repository root: the real tools and prompts, a doctored contract.
FAKE="${WORK}/fake"
mkdir -p "${FAKE}/fixtures"
cp -R "${ROOT_DIR}/tools" "${FAKE}/tools"
ln -s "${ROOT_DIR}/correctness_prompts" "${FAKE}/correctness_prompts"

python3 - "${FIXTURE}" "${FAKE}/fixtures/qwen3_8_125b_a6b_track.json" <<'PY'
import json, sys
doc = json.load(open(sys.argv[1]))
doc["official_scoring_enabled"] = True
json.dump(doc, open(sys.argv[2], "w"), indent=2)
PY
if "${FAKE}/tools/qwen4exp-golden-reauthor.sh" --dry-run --out "${WORK}/r1" "${COMMON[@]}" \
     > "${WORK}/refuse1.txt" 2>&1; then
  fail "the script ran with official_scoring_enabled=true"
fi
grep -q 'REFUSE official-scoring-armed' "${WORK}/refuse1.txt" \
  || fail "an armed track was refused, but not by the name official-scoring-armed"
pass "refuses official-scoring-armed when the track is scoring"

# The mismatch must be a REAL one: a contract pin that the tree's `ds4` gitlink
# disagrees with. A fake root that is no git repository at all would refuse for
# the WRONG reason -- there is no HEAD to read -- and would prove nothing about
# the check. So the fake root becomes a git repository whose `ds4` gitlink is a
# DIFFERENT commit, and its contract keeps the pin the real tree carries.
python3 - "${FIXTURE}" "${FAKE}/fixtures/qwen3_8_125b_a6b_track.json" <<'PY'
import json, sys
doc = json.load(open(sys.argv[1]))
doc["official_scoring_enabled"] = False
json.dump(doc, open(sys.argv[2], "w"), indent=2)
PY
OTHER_PIN="abcdef0123456789abcdef0123456789abcdef01"
git -C "${FAKE}" init -q
git -C "${FAKE}" config user.email "test@example.invalid"
git -C "${FAKE}" config user.name "golden-reauthor test"
git -C "${FAKE}" config commit.gpgsign false
git -C "${FAKE}" update-index --add --cacheinfo "160000,${OTHER_PIN},ds4"
git -C "${FAKE}" commit -q -m "a tree whose ds4 gitlink is not the contract pin"
[ "$(git -C "${FAKE}" ls-tree HEAD ds4 | awk '{print $3}')" = "${OTHER_PIN}" ] \
  || fail "the fake tree does not carry the ds4 gitlink this test needs"
CONTRACT_PIN="$(jq -r '.serve_configuration.engine_pin' "${FIXTURE}")"
[ "${CONTRACT_PIN}" != "${OTHER_PIN}" ] || fail "the fake gitlink equals the contract pin; there is no mismatch to test"

if "${FAKE}/tools/qwen4exp-golden-reauthor.sh" --dry-run --out "${WORK}/r2" "${COMMON[@]}" \
     > "${WORK}/refuse2.txt" 2>&1; then
  fail "the script ran with an engine pin the ds4 gitlink does not match"
fi
grep -q 'REFUSE engine-pin-mismatch' "${WORK}/refuse2.txt" \
  || fail "a wrong engine pin was refused, but not by the name engine-pin-mismatch"
grep -q "${OTHER_PIN}" "${WORK}/refuse2.txt" \
  || fail "the refusal does not name the gitlink it read"
grep -q "${CONTRACT_PIN}" "${WORK}/refuse2.txt" \
  || fail "the refusal does not name the contract pin"
pass "refuses engine-pin-mismatch on a REAL mismatch: gitlink ${OTHER_PIN:0:12} against contract pin ${CONTRACT_PIN:0:12}"

# --- 3-4. the phases --------------------------------------------------------
OUT="${WORK}/run"
mkdir -p "${OUT}"
"${SCRIPT}" --phase prompts --out "${OUT}" "${COMMON[@]}" >"${WORK}/prompts.log" 2>&1 \
  || { cat "${WORK}/prompts.log" >&2; fail "the prompts phase failed"; }
POOL_COUNT="$(jq '.timed_prompt_pool | length' "${FIXTURE}")"
FOUND="$(find "${OUT}/prompts" -name '*.tokens.json' | wc -l | tr -d '[:space:]')"
[ "${FOUND}" -eq "$((POOL_COUNT + 1))" ] \
  || fail "expected $((POOL_COUNT + 1)) prompt files (pool + the public correctness prompt), found ${FOUND}"
for f in "${OUT}"/prompts/*.tokens.json; do
  n="$(python3 -c "import json,sys;print(len(json.load(open(sys.argv[1]))))" "${f}")"
  [ "${n}" -eq 1024 ] || fail "$(basename "${f}") carries ${n} ids, not 1024"
done
pass "the prompts phase verified every pinned golden against the contract and took ${FOUND} x 1024 prompt ids"

"${SCRIPT}" --phase capture-serial --out "${OUT}" "${COMMON[@]}" >"${WORK}/serial.log" 2>&1 \
  || { tail -20 "${WORK}/serial.log" >&2; fail "the capture-serial phase failed"; }
GOLDENS="$(find "${OUT}/goldens" -name '*.golden.json' | wc -l | tr -d '[:space:]')"
[ "${GOLDENS}" -eq "$((POOL_COUNT + 1))" ] || fail "expected $((POOL_COUNT + 1)) depth-0 goldens, found ${GOLDENS}"
pass "the capture-serial phase produced ${GOLDENS} depth-0 goldens from the mock"

"${SCRIPT}" --phase capture-mtp --out "${OUT}" "${COMMON[@]}" >"${WORK}/mtp.log" 2>&1 \
  || { tail -20 "${WORK}/mtp.log" >&2; fail "the capture-mtp phase failed"; }
LIVE="$(jq -r '.live_golden' "${FIXTURE}")"
[ -s "${OUT}/goldens/${LIVE}.mtp1.golden.json" ] || fail "no depth-1 oracle was written"
grep -qE 'MATCH|MISMATCH' "${OUT}/mtp1-exactness.txt" || fail "the exactness report holds no per-prompt verdict"
python3 - "${OUT}/goldens/${LIVE}.golden.json" "${OUT}/goldens/${LIVE}.mtp1.golden.json" <<'PY'
import json, sys
a = json.load(open(sys.argv[1])); b = json.load(open(sys.argv[2]))
assert a["cases"] == b["cases"], "the oracle changed cases[]; only the decode oracle may differ"
assert a["benchmark"]["expected_decode_seed_token"] == b["benchmark"]["expected_decode_seed_token"]
assert len(b["benchmark"]["expected_decode_tokens"]) == 128
PY
pass "the capture-mtp phase grafted a 128-token depth-1 oracle and left cases[] untouched"

# --- 5. validation ----------------------------------------------------------
"${SCRIPT}" --phase validate --out "${OUT}" "${COMMON[@]}" >"${WORK}/validate.log" 2>&1 \
  || { tail -20 "${WORK}/validate.log" >&2; fail "validate-golden rejected a golden this run produced"; }
LINES="$(wc -l < "${OUT}/index.tsv" | tr -d '[:space:]')"
[ "${LINES}" -eq "$((POOL_COUNT + 2))" ] \
  || fail "index.tsv holds ${LINES} rows, expected $((POOL_COUNT + 2)) (pool + public + the depth-1 oracle)"
pass "benchd validate-golden ACCEPTED all ${LINES} artifacts against the track contract and their own pins"

# --- 6. the negative control ------------------------------------------------
"${SCRIPT}" --phase negative-control --out "${OUT}" "${COMMON[@]}" >"${WORK}/neg.log" 2>&1 \
  || { tail -20 "${WORK}/neg.log" >&2; fail "the negative control did not hold"; }
grep -q 'REJECT' "${OUT}/negative-control.txt" || fail "the negative control recorded no rejection"
ORIG_BYTES="$(wc -c < "${OUT}/goldens/${LIVE}.golden.json" | tr -d '[:space:]')"
BAD_BYTES="$(wc -c < "${OUT}/negative-control.golden.json" | tr -d '[:space:]')"
[ "${ORIG_BYTES}" = "${BAD_BYTES}" ] \
  || fail "the perturbed golden is ${BAD_BYTES} bytes and the original ${ORIG_BYTES}; the control must change only the token"
pass "the negative control held: one changed token, byte count unchanged, validate-golden rejected it"

# --- 7. the fixture ---------------------------------------------------------
"${SCRIPT}" --phase patch --out "${OUT}" "${COMMON[@]}" >"${OUT}/patch.txt" 2>&1 \
  || fail "the patch phase failed"
grep -q 'timed_prompt_pool' "${OUT}/fixture-pins.json" || fail "the pin patch names no timed_prompt_pool"
[ "$(shasum -a 256 "${FIXTURE}" | awk '{print $1}')" = "${FIXTURE_SHA_BEFORE}" ] \
  || fail "the run edited fixtures/qwen3_8_125b_a6b_track.json; the pins are David-gated"
pass "the pin patch is printed and written, and the track contract is byte-identical"

# --- 8. the recorder comes from beside benchd -----------------------------
# tools/fetch-benchd.sh stages record-correctness-golden beside the benchd it
# resolves, so the run must find it there with no --recorder. STAGED holds a
# copy of both binaries in one directory, which is the shape benchd-bin/ has
# after a resolve against a channel whose manifest declares the recorder.
#
# BENCHD_RECORD_GOLDEN_BIN is how THIS test names the recorder, and the script
# reads it as an override, so it is cleared for the cases below -- they are
# about what the script finds when nothing names one.
STAGED="${WORK}/benchd-bin"
mkdir -p "${STAGED}"
cp "${BENCHD_BIN}" "${STAGED}/benchd"
cp "${RECORDER_BIN}" "${STAGED}/record-correctness-golden"
NO_RECORDER=(--weights "${WEIGHTS}" --engine "${ENGINE}" --ds4 "${DS4_STUB}"
             --steps 128 --decode-steps 128 --depth 1)
UNNAMED=(env -u BENCHD_RECORD_GOLDEN_BIN)

"${UNNAMED[@]}" "${SCRIPT}" --dry-run --out "${WORK}/staged" \
  --benchd "${STAGED}/benchd" "${NO_RECORDER[@]}" > "${WORK}/staged.txt" 2>&1 \
  || { cat "${WORK}/staged.txt" >&2; fail "the run refused although the recorder was staged beside benchd"; }
grep -q 'staged from the channel beside benchd' "${WORK}/staged.txt" \
  || fail "the run did not say it took the recorder from beside benchd"
grep -q -- "${STAGED}/record-correctness-golden" "${WORK}/staged.txt" \
  || fail "the capture commands do not drive the staged recorder"
pass "the staged recorder is used with no --recorder, and the run names the path it took"

# An explicit --recorder still wins over the staged copy.
"${UNNAMED[@]}" "${SCRIPT}" --dry-run --out "${WORK}/override" \
  --benchd "${STAGED}/benchd" --recorder "${RECORDER_BIN}" "${NO_RECORDER[@]}" \
  > "${WORK}/override.txt" 2>&1 \
  || { cat "${WORK}/override.txt" >&2; fail "the explicit --recorder run refused"; }
grep -q -- "${RECORDER_BIN}" "${WORK}/override.txt" \
  || fail "--recorder did not override the staged copy"
if grep -q -- "${STAGED}/record-correctness-golden" "${WORK}/override.txt"; then
  fail "the staged copy was driven although --recorder named another one"
fi
pass "an explicit --recorder overrides the staged copy"

# A benchd with NO recorder beside it: the legacy channel, which publishes
# benchd alone. The refusal must name what the channel has to publish, not
# only that a flag is missing.
BARE="${WORK}/bare-bin"
mkdir -p "${BARE}"
cp "${BENCHD_BIN}" "${BARE}/benchd"
if "${UNNAMED[@]}" "${SCRIPT}" --dry-run --out "${WORK}/bare" \
     --benchd "${BARE}/benchd" "${NO_RECORDER[@]}" > "${WORK}/bare.txt" 2>&1; then
  fail "the run proceeded with no recorder anywhere"
fi
grep -q 'REFUSE missing-tool' "${WORK}/bare.txt" \
  || fail "the missing recorder was refused, but not by the name missing-tool"
grep -q 'source_commit >= the mlxfast-bench PR #255 republish' "${WORK}/bare.txt" \
  || fail "the refusal does not say which channel publishes the recorder"
pass "no recorder anywhere refuses missing-tool and names the republish the channel needs"

printf 'test-golden-reauthor: all checks passed\n'
