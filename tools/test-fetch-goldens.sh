#!/usr/bin/env bash
# Unit test for tools/fetch-goldens.sh. Offline, no GPU, no real R2 -- a
# throwaway python3 http.server on 127.0.0.1 stands in for the bucket, so the
# URL assembly (endpoint + "/" + r2_path) and the pin verification are
# exercised against real bytes over a real socket without any credential, any
# network egress, or any hidden material.
#
# WHAT THIS PROVES, and it is the part worth having: the REFUSAL paths. A
# fetcher that accepts good bytes is easy; one that reliably rejects a
# one-byte-short transfer, a digest mismatch, and a hidden pin -- and leaves no
# partial file behind when it does -- is the thing the pin discipline actually
# rests on.
#
# Usage: tools/test-fetch-goldens.sh
# Exit:  0 all cases pass, 1 a case failed (printed with a FAIL prefix)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
FETCH="${SCRIPT_DIR}/tools/fetch-goldens.sh"
WORK="$(mktemp -d)"

SERVER_PID=""
cleanup() {
  if [[ -n "${SERVER_PID}" ]]; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  rm -rf "${WORK}"
}
trap cleanup EXIT

failures=0
fail() {
  echo "FAIL: $1" >&2
  failures=$((failures + 1))
}
pass() { echo "ok: $1"; }

# --- a stand-in bucket ------------------------------------------------------
# Mirrors the real key convention: correctness_prompts/<track_id>/<name>.json.
KEY="correctness_prompts/qwen3.8-125b-a6b-cuda-v1/test-object.json"
mkdir -p "${WORK}/bucket/$(dirname "${KEY}")"
printf '{"version":1,"note":"synthetic fetch-goldens test object"}\n' \
  > "${WORK}/bucket/${KEY}"

GOOD_SHA="$(shasum -a 256 "${WORK}/bucket/${KEY}" | awk '{print $1}')"
GOOD_BYTES="$(wc -c < "${WORK}/bucket/${KEY}" | tr -d '[:space:]')"

# Pick a free port by binding one and releasing it.
PORT="$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')"
( cd "${WORK}/bucket" && exec python3 -m http.server "${PORT}" --bind 127.0.0.1 >/dev/null 2>&1 ) &
SERVER_PID=$!

# Wait for the socket rather than sleeping a guessed interval.
ready=0
for _ in $(seq 1 50); do
  if curl --fail --silent --max-time 1 -o /dev/null "http://127.0.0.1:${PORT}/${KEY}"; then
    ready=1
    break
  fi
  sleep 0.2
done
if [[ "${ready}" != "1" ]]; then
  echo "FAIL: stand-in bucket did not come up on 127.0.0.1:${PORT}" >&2
  exit 1
fi

ENDPOINT="http://127.0.0.1:${PORT}"

# The endpoint regex deliberately requires https:// AND forbids a port, which
# is right for a real R2 endpoint (https://<host>/<bucket>) and wrong for a
# loopback stand-in (http://127.0.0.1:<port>). The cases below that need a
# completed transfer therefore run against a copy with exactly those two
# characters relaxed -- the 's' made optional and ':' added to the host class.
# Everything else, including the whole verification path, is the real script.
# The unrelaxed regex is covered on the real script by case 7.
#
# The copy lives in a stand-in repo root whose fixtures/ is a symlink to the
# real one, because the script resolves its contract relative to its own
# location and the hidden guard FAILS CLOSED when that contract is unreadable.
# Putting the copy in a bare temp dir would make every hidden case pass for the
# wrong reason (refused because the contract was missing, not because the pin
# was recognised as hidden).
mkdir -p "${WORK}/repo/tools"
ln -s "${SCRIPT_DIR}/fixtures" "${WORK}/repo/fixtures"
RELAXED="${WORK}/repo/tools/fetch-goldens-loopback.sh"
sed 's|\^https://\[A-Za-z0-9.-\]+|^https?://[A-Za-z0-9.:-]+|' "${FETCH}" > "${RELAXED}"
chmod +x "${RELAXED}"
# A silently-failed sed would turn every transfer case into a false FAIL that
# looks like a fetcher bug, so assert the rewrite actually happened.
if cmp -s "${FETCH}" "${RELAXED}" || ! grep -q 'https?://\[A-Za-z0-9.:-\]' "${RELAXED}"; then
  echo "FAIL: could not relax the endpoint regex for loopback testing" >&2
  exit 1
fi

# --- case 1: a correct pin is accepted --------------------------------------
out="${WORK}/case1.json"
if R2_BUCKET_ENDPOINT="${ENDPOINT}" "${RELAXED}" \
     --r2-path "${KEY}" --sha256 "${GOOD_SHA}" --bytes "${GOOD_BYTES}" \
     --out "${out}" >"${WORK}/case1.log" 2>&1; then
  if [[ -s "${out}" ]] && [[ "$(shasum -a 256 "${out}" | awk '{print $1}')" == "${GOOD_SHA}" ]]; then
    pass "correct pin accepted, bytes match"
  else
    fail "case 1: exited 0 but the output file is missing or wrong"
  fi
else
  fail "case 1: refused a correct pin ($(cat "${WORK}/case1.log"))"
fi

# --- case 2: sha256 mismatch is refused, and leaves NO file ------------------
out="${WORK}/case2.json"
BAD_SHA="0000000000000000000000000000000000000000000000000000000000000000"
if R2_BUCKET_ENDPOINT="${ENDPOINT}" "${RELAXED}" \
     --r2-path "${KEY}" --sha256 "${BAD_SHA}" --bytes "${GOOD_BYTES}" \
     --out "${out}" >"${WORK}/case2.log" 2>&1; then
  fail "case 2: accepted a sha256 mismatch"
elif [[ -e "${out}" ]]; then
  fail "case 2: refused but left a partial file at ${out}"
elif grep -q "sha256 mismatch" "${WORK}/case2.log"; then
  pass "sha256 mismatch refused, no file left behind"
else
  fail "case 2: refused for the wrong reason ($(cat "${WORK}/case2.log"))"
fi

# --- case 3: byte-count mismatch is refused (and named as such) --------------
# The truncation case: the hash would also differ, but the byte count must be
# what reports it, because that is the diagnosis the operator needs.
out="${WORK}/case3.json"
if R2_BUCKET_ENDPOINT="${ENDPOINT}" "${RELAXED}" \
     --r2-path "${KEY}" --sha256 "${GOOD_SHA}" --bytes "$((GOOD_BYTES - 1))" \
     --out "${out}" >"${WORK}/case3.log" 2>&1; then
  fail "case 3: accepted a byte-count mismatch"
elif [[ -e "${out}" ]]; then
  fail "case 3: refused but left a partial file at ${out}"
elif grep -q "byte-count mismatch" "${WORK}/case3.log"; then
  pass "byte-count mismatch refused and reported as truncation, not as a hash error"
else
  fail "case 3: refused for the wrong reason ($(cat "${WORK}/case3.log"))"
fi

# --- case 4: a hidden pin is refused, by DIGEST ------------------------------
# POSITIVE CONTROL for the digest arm, run against a SYNTHETIC ARMED contract.
# This track's own contract is not armed yet: every timed_prompt_pool entry and
# the hidden oracle carry the exact PENDING-ORGANIZER sentinel, so the real
# fixture declares no digest for the guard to match. A test that only read the
# real fixture would therefore report the digest arm as passing while it was
# matching nothing at all. This case gives the guard a contract that DOES
# declare a digest and requires it to refuse. Case 5 keeps the coupling to the
# shipped fixture.
mkdir -p "${WORK}/armed/tools" "${WORK}/armed/fixtures"
ARMED_SHA="f8eb3e0faf960154eacef498aa5f1c0be19d11e08054af37cc59a4f28ca5911b"
cat > "${WORK}/armed/fixtures/qwen3_8_125b_a6b_track.json" <<ARMEDJSON
{
  "timed_prompt_pool": [
    {
      "r2_path": "correctness_prompts/qwen3.8-125b-a6b-cuda-v1/armed-pool-object.json",
      "sha256": "1111111111111111111111111111111111111111111111111111111111111111",
      "bytes": 1
    }
  ],
  "hidden_correctness_golden": {
    "sha256": "${ARMED_SHA}",
    "bytes": 16692
  }
}
ARMEDJSON
cp "${RELAXED}" "${WORK}/armed/tools/fetch-goldens-loopback.sh"
chmod +x "${WORK}/armed/tools/fetch-goldens-loopback.sh"

out="${WORK}/case4.json"
if R2_BUCKET_ENDPOINT="${ENDPOINT}" "${WORK}/armed/tools/fetch-goldens-loopback.sh" \
     --r2-path "${KEY}" --sha256 "${ARMED_SHA}" --bytes 16692 \
     --out "${out}" >"${WORK}/case4.log" 2>&1; then
  fail "case 4: fetched a hidden-golden digest"
elif grep -q "hidden, box-only material" "${WORK}/case4.log"; then
  pass "hidden golden refused by digest even under an innocuous key"
else
  fail "case 4: refused for the wrong reason ($(cat "${WORK}/case4.log"))"
fi

# --- case 4b: the same guard refuses a declared POOL key ---------------------
out="${WORK}/case4b.json"
ARMED_KEY="correctness_prompts/qwen3.8-125b-a6b-cuda-v1/armed-pool-object.json"
if R2_BUCKET_ENDPOINT="${ENDPOINT}" "${WORK}/armed/tools/fetch-goldens-loopback.sh" \
     --r2-path "${ARMED_KEY}" --sha256 "${GOOD_SHA}" --bytes "${GOOD_BYTES}" \
     --out "${out}" >"${WORK}/case4b.log" 2>&1; then
  fail "case 4b: fetched a hidden pool key"
elif grep -q "hidden, box-only material" "${WORK}/case4b.log"; then
  pass "hidden pool tape refused by key even under an unrelated digest"
else
  fail "case 4b: refused for the wrong reason ($(cat "${WORK}/case4b.log"))"
fi

# --- case 5: the guard reads THIS REPOSITORY'S shipped fixture ---------------
# The coupling case. Case 4 proves the mechanism on a synthetic contract; this
# proves the shipped script reads the shipped fixture and refuses what that
# fixture declares. The fixture is unarmed, so what it declares today is the
# exact pending sentinel -- and the sentinel must be refused like any other
# declared pin. When the pool is armed this case keeps working unchanged,
# because it reads the value out of the fixture rather than repeating it.
out="${WORK}/case5.json"
SENTINEL_KEY="$(awk '
  /^  "timed_prompt_pool": \[/ { in_pool=1; next }
  in_pool && /^  \]/ { exit }
  in_pool && /"r2_path":/ {
    value=$0
    sub(/^[^:]*: *"/, "", value)
    sub(/".*$/, "", value)
    print value
    exit
  }
' "${SCRIPT_DIR}/fixtures/qwen3_8_125b_a6b_track.json")"
if [[ -z "${SENTINEL_KEY}" ]]; then
  fail "case 5: could not read a timed_prompt_pool r2_path out of the shipped fixture"
elif R2_BUCKET_ENDPOINT="${ENDPOINT}" "${RELAXED}" \
     --r2-path "${SENTINEL_KEY}" --sha256 "${GOOD_SHA}" --bytes "${GOOD_BYTES}" \
     --out "${out}" >"${WORK}/case5.log" 2>&1; then
  fail "case 5: fetched a key the shipped fixture declares hidden"
elif grep -q "hidden, box-only material" "${WORK}/case5.log"; then
  pass "the shipped fixture's own declared pool key is refused"
else
  fail "case 5: refused for the wrong reason ($(cat "${WORK}/case5.log"))"
fi

# --- case 6: no endpoint -> refuse, and say to ask the organizer -------------
out="${WORK}/case6.json"
if env -u R2_BUCKET_ENDPOINT "${FETCH}" \
     --r2-path "${KEY}" --sha256 "${GOOD_SHA}" --bytes "${GOOD_BYTES}" \
     --out "${out}" >"${WORK}/case6.log" 2>&1; then
  fail "case 6: ran without R2_BUCKET_ENDPOINT"
elif grep -q "ASK THE ORGANIZER FOR THE R2 BASE" "${WORK}/case6.log"; then
  pass "missing endpoint refused with the organizer instruction"
else
  fail "case 6: refused for the wrong reason ($(cat "${WORK}/case6.log"))"
fi

# --- case 7: a malformed endpoint must not be echoed (secret-tier) ----------
out="${WORK}/case7.json"
SECRET_ISH="ftp://not-a-valid-endpoint/super-secret-bucket-name"
if R2_BUCKET_ENDPOINT="${SECRET_ISH}" "${FETCH}" \
     --r2-path "${KEY}" --sha256 "${GOOD_SHA}" --bytes "${GOOD_BYTES}" \
     --out "${out}" >"${WORK}/case7.log" 2>&1; then
  fail "case 7: accepted a malformed endpoint"
elif grep -q "super-secret-bucket-name" "${WORK}/case7.log"; then
  fail "case 7: echoed the endpoint into the log (secret-tier leak)"
elif grep -q "value withheld" "${WORK}/case7.log"; then
  pass "malformed endpoint refused without echoing it"
else
  fail "case 7: refused for the wrong reason ($(cat "${WORK}/case7.log"))"
fi

# --- case 8: unpinned / malformed arguments are refused before any I/O -------
out="${WORK}/case8.json"
if R2_BUCKET_ENDPOINT="${ENDPOINT}" "${RELAXED}" \
     --r2-path "${KEY}" --sha256 "not-a-sha" --bytes "${GOOD_BYTES}" \
     --out "${out}" >"${WORK}/case8.log" 2>&1; then
  fail "case 8: accepted a malformed --sha256"
elif grep -q "must be 64 lowercase hex" "${WORK}/case8.log"; then
  pass "malformed --sha256 refused"
else
  fail "case 8: refused for the wrong reason ($(cat "${WORK}/case8.log"))"
fi

out="${WORK}/case8b.json"
if R2_BUCKET_ENDPOINT="${ENDPOINT}" "${RELAXED}" \
     --r2-path "${KEY}" --sha256 "${GOOD_SHA}" --bytes 0 \
     --out "${out}" >"${WORK}/case8b.log" 2>&1; then
  fail "case 8b: accepted --bytes 0 (the sentinel placeholder)"
elif grep -q "must be a positive integer" "${WORK}/case8b.log"; then
  pass "--bytes 0 refused (a zero byte count is the sentinel, never a pin)"
else
  fail "case 8b: refused for the wrong reason ($(cat "${WORK}/case8b.log"))"
fi

# --- case 9: a missing object is refused, with no file left ------------------
out="${WORK}/case9.json"
if R2_BUCKET_ENDPOINT="${ENDPOINT}" "${RELAXED}" \
     --r2-path "correctness_prompts/qwen3.8-125b-a6b-cuda-v1/absent.json" \
     --sha256 "${GOOD_SHA}" --bytes "${GOOD_BYTES}" \
     --out "${out}" >"${WORK}/case9.log" 2>&1; then
  fail "case 9: accepted a 404"
elif [[ -e "${out}" ]]; then
  fail "case 9: refused but left a file at ${out}"
else
  pass "absent object refused, no file left behind"
fi

# --- case 10: an unreadable contract fails CLOSED ---------------------------
# The regression this pins: a copy of the script that cannot see the contract
# must refuse outright, never fetch with the hidden guard silently disarmed.
mkdir -p "${WORK}/norepo/tools"
cp "${RELAXED}" "${WORK}/norepo/tools/fetch-goldens-nocontract.sh"
out="${WORK}/case10.json"
if R2_BUCKET_ENDPOINT="${ENDPOINT}" "${WORK}/norepo/tools/fetch-goldens-nocontract.sh" \
     --r2-path "${KEY}" --sha256 "${GOOD_SHA}" --bytes "${GOOD_BYTES}" \
     --out "${out}" >"${WORK}/case10.log" 2>&1; then
  fail "case 10: fetched with no readable contract (hidden guard was disarmed)"
elif [[ -e "${out}" ]]; then
  fail "case 10: refused but left a file at ${out}"
elif grep -q "hidden-material guard cannot be evaluated" "${WORK}/case10.log"; then
  pass "unreadable contract fails closed (guard cannot be silently disarmed)"
else
  fail "case 10: refused for the wrong reason ($(cat "${WORK}/case10.log"))"
fi

# ============================================================================
# --all: staging the whole pinned set for a ranked box
# ============================================================================
# The same stand-in bucket, three more objects, and a synthetic contract whose
# pins are their real digests. The credentialed branch is exercised through a
# STUB signer -- the vendored one speaks SigV4 to real R2, which this test must
# never do -- so what is under test here is the staging loop: what it fetches,
# what it verifies, what it refuses, and what it leaves behind.
ALL_KEY_A="correctness_prompts/qwen3.8-125b-a6b-cuda-v1/all-pool-a.golden.json"
ALL_KEY_B="correctness_prompts/qwen3.8-125b-a6b-cuda-v1/all-pool-b.golden.json"
ALL_KEY_C="correctness_prompts/qwen3.8-125b-a6b-cuda-v1/all-oracle-mtp1.golden.json"
printf '{"version":1,"case":"all-pool-a"}\n' > "${WORK}/bucket/${ALL_KEY_A}"
printf '{"version":1,"case":"all-pool-b"}\n' > "${WORK}/bucket/${ALL_KEY_B}"
printf '{"version":1,"case":"all-oracle-mtp1"}\n' > "${WORK}/bucket/${ALL_KEY_C}"

pin_sha() { shasum -a 256 "$1" | awk '{print $1}'; }
pin_bytes() { wc -c < "$1" | tr -d '[:space:]'; }

A_SHA="$(pin_sha "${WORK}/bucket/${ALL_KEY_A}")"; A_BYTES="$(pin_bytes "${WORK}/bucket/${ALL_KEY_A}")"
B_SHA="$(pin_sha "${WORK}/bucket/${ALL_KEY_B}")"; B_BYTES="$(pin_bytes "${WORK}/bucket/${ALL_KEY_B}")"
C_SHA="$(pin_sha "${WORK}/bucket/${ALL_KEY_C}")"; C_BYTES="$(pin_bytes "${WORK}/bucket/${ALL_KEY_C}")"

# write_all_contract PATH B_BYTES -- the second tape's byte pin is a parameter
# so the truncation case can make exactly one thing wrong.
write_all_contract() {
  cat > "$1" <<ALLJSON
{
  "timed_prompt_pool": [
    {"r2_path": "${ALL_KEY_A}", "sha256": "${A_SHA}", "bytes": ${A_BYTES}},
    {"r2_path": "${ALL_KEY_B}", "sha256": "${B_SHA}", "bytes": $2}
  ],
  "live_golden_speculative": {
    "mtp1": {"r2_path": "${ALL_KEY_C}", "sha256": "${C_SHA}", "bytes": ${C_BYTES}}
  },
  "hidden_correctness_golden": {"sha256": "${A_SHA}", "bytes": ${A_BYTES}}
}
ALLJSON
}

# The stub signer: the `download-r2-object.sh KEY DEST` contract, over loopback
# and with no credential. The real signer's SigV4 is not this test's subject.
SIGNER="${WORK}/stub-signer.sh"
cat > "${SIGNER}" <<SIGNEREOF
#!/usr/bin/env bash
set -euo pipefail
curl --fail --silent --show-error --output "\$2" "${ENDPOINT}/\$1"
SIGNEREOF
chmod +x "${SIGNER}"

all_root() { # all_root DIR B_BYTES -- a stand-in repo root with the copy + contract
  mkdir -p "$1/tools" "$1/fixtures"
  cp "${RELAXED}" "$1/tools/fetch-goldens-loopback.sh"
  chmod +x "$1/tools/fetch-goldens-loopback.sh"
  write_all_contract "$1/fixtures/qwen3_8_125b_a6b_track.json" "$2"
}

# --- case 11: --all stages and verifies the whole pinned set -----------------
all_root "${WORK}/allrepo" "${B_BYTES}"
ALL_OUT="${WORK}/staged"
if R2_BUCKET_ENDPOINT="${ENDPOINT}" \
   R2_ACCESS_KEY_ID="stub-key" R2_SECRET_ACCESS_KEY="stub-secret" \
   MLXFAST_QWEN38_R2_DOWNLOADER="${SIGNER}" \
   "${WORK}/allrepo/tools/fetch-goldens-loopback.sh" --all --out "${ALL_OUT}" \
   >"${WORK}/case11.log" 2>&1; then
  ok11=1
  for want in all-pool-a.golden.json all-pool-b.golden.json all-oracle-mtp1.golden.json; do
    [[ -f "${ALL_OUT}/${want}" ]] || { fail "case 11: ${want} was not staged"; ok11=0; }
  done
  [[ "$(pin_sha "${ALL_OUT}/all-pool-a.golden.json")" == "${A_SHA}" ]] \
    || { fail "case 11: the staged pool tape does not match its pin"; ok11=0; }
  # No .partial may survive a successful run, and the staged files are read-only.
  compgen -G "${ALL_OUT}/*.partial" >/dev/null \
    && { fail "case 11: a .partial file survived a successful run"; ok11=0; }
  perms="$(ls -l "${ALL_OUT}/all-pool-a.golden.json" | cut -c2-10)"
  [[ "${perms}" == "r--r--r--" ]] \
    || { fail "case 11: staged golden is ${perms}, expected r--r--r-- (0444)"; ok11=0; }
  grep -q "hidden_correctness_golden resolves to all-pool-a.golden.json" "${WORK}/case11.log" \
    || { fail "case 11: the digest-only hidden oracle was not resolved among the staged files"; ok11=0; }
  grep -q "staged 3 object(s)" "${WORK}/case11.log" \
    || { fail "case 11: the run did not report 3 staged objects"; ok11=0; }
  [[ "${ok11}" == "1" ]] && pass "--all staged the whole pinned set, verified it, and resolved the hidden oracle"
else
  fail "case 11: --all refused a healthy set ($(tail -3 "${WORK}/case11.log"))"
fi

# --- case 12: a second run is a no-op ---------------------------------------
# Idempotence is what makes this safe to put in a converge script: a box that is
# already staged must not re-fetch, and must not rewrite files it already holds.
before="$(shasum -a 256 "${ALL_OUT}"/*.json | shasum -a 256 | awk '{print $1}')"
if R2_BUCKET_ENDPOINT="${ENDPOINT}" \
   R2_ACCESS_KEY_ID="stub-key" R2_SECRET_ACCESS_KEY="stub-secret" \
   MLXFAST_QWEN38_R2_DOWNLOADER="${SIGNER}" \
   "${WORK}/allrepo/tools/fetch-goldens-loopback.sh" --all --out "${ALL_OUT}" \
   >"${WORK}/case12.log" 2>&1; then
  after="$(shasum -a 256 "${ALL_OUT}"/*.json | shasum -a 256 | awk '{print $1}')"
  if [[ "${before}" != "${after}" ]]; then
    fail "case 12: the second run changed the staged files"
  elif ! grep -q "staged 0 object(s), kept 3" "${WORK}/case12.log"; then
    fail "case 12: the second run did not report every file as already staged ($(tail -2 "${WORK}/case12.log"))"
  else
    pass "a second --all run fetched nothing and left the staged set untouched"
  fi
else
  fail "case 12: the second --all run refused ($(tail -3 "${WORK}/case12.log"))"
fi

# --- case 13: a one-byte-short object refuses and leaves no .partial ---------
all_root "${WORK}/allrepo-short" "$((B_BYTES + 1))"
SHORT_OUT="${WORK}/staged-short"
if R2_BUCKET_ENDPOINT="${ENDPOINT}" \
   R2_ACCESS_KEY_ID="stub-key" R2_SECRET_ACCESS_KEY="stub-secret" \
   MLXFAST_QWEN38_R2_DOWNLOADER="${SIGNER}" \
   "${WORK}/allrepo-short/tools/fetch-goldens-loopback.sh" --all --out "${SHORT_OUT}" \
   >"${WORK}/case13.log" 2>&1; then
  fail "case 13: --all accepted an object that is a byte short of its pin"
elif compgen -G "${SHORT_OUT}/*.partial" >/dev/null; then
  fail "case 13: refused but left a .partial file in the staging directory"
elif [[ -e "${SHORT_OUT}/all-pool-b.golden.json" ]]; then
  fail "case 13: refused but staged the object anyway"
elif grep -q "byte-count mismatch for ${ALL_KEY_B}" "${WORK}/case13.log"; then
  pass "--all refuses a short object by name and stages no partial bytes"
else
  fail "case 13: refused for the wrong reason ($(tail -3 "${WORK}/case13.log"))"
fi

# --- case 14: --all without credentials refuses, naming them ----------------
# The default signer speaks SigV4, so there is no anonymous way to stage this
# material. The refusal must name what is missing rather than fail inside curl.
NOCRED_OUT="${WORK}/staged-nocred"
if env -u R2_ACCESS_KEY_ID -u R2_SECRET_ACCESS_KEY -u MLXFAST_QWEN38_R2_DOWNLOADER \
   R2_BUCKET_ENDPOINT="${ENDPOINT}" \
   "${WORK}/allrepo/tools/fetch-goldens-loopback.sh" --all --out "${NOCRED_OUT}" \
   >"${WORK}/case14.log" 2>&1; then
  fail "case 14: --all ran with no R2 credentials"
elif ! grep -q "R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY" "${WORK}/case14.log"; then
  fail "case 14: refused without naming the credentials ($(tail -3 "${WORK}/case14.log"))"
elif compgen -G "${NOCRED_OUT}/*.json" >/dev/null; then
  fail "case 14: refused but staged a file anyway"
else
  pass "--all without credentials refuses and names the two variables"
fi

echo
if (( failures > 0 )); then
  echo "${failures} case(s) failed" >&2
  exit 1
fi
echo "all fetch-goldens.sh cases passed"
