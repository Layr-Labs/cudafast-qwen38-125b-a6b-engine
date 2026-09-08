#!/usr/bin/env bash
# test-ds4-resident.sh -- prove the RESIDENT-SERVER topology end to end, off a GPU.
#
# THE CLAIM UNDER TEST is the binding one: the weights load ONCE for a
# benchmark window, and benchd's fresh worker per phase costs a socket connect.
# Everything that carries that claim is lifecycle and socket behaviour, so this
# runs the REAL sources -- tools/serve-up.sh, the real
# harness/protocol-adapter/ds4_shim/ds4_resident.c, and the real `cuda-engine`
# adapter -- against tools/ds4/resident-stub-engine.c, a synthetic ds4s_*
# engine. No CUDA, no driver, no checkpoint, no network.
#
# WHAT IT PROVES
#   1. one load: four phases, exactly one engine open
#   2. reconnect is milliseconds, and the number is printed
#   3. the session resets between phases (no KV leaks across benchd's drain)
#   4. every phase's hello carries the SAME resident, and its identity, in the
#      one string benchd seals (engine_backend)
#   5. the counters cross the wire
#   6. a memory plan that does not fit REFUSES BEFORE ANY LOAD
#   7. teardown leaves no resident and no socket
#   8. an idle phase is dropped on the resident's own ceiling, and the resident
#      survives to serve the next one
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null && pwd -P)"
# The synthetic artifact and the synthetic engine declaration. One writer, so
# this test and the other off-box drivers cannot drift apart.
# shellcheck source=tools/ds4/synthetic-window.sh
. "${ROOT_DIR}/tools/ds4/synthetic-window.sh"
WORK="$(mktemp -d)"
# The socket directory is SHORT and separate, because a Unix socket address
# holds 103 bytes (macOS) or 107 (Linux) and the platform temp root alone is
# already 60-odd of them on macOS. /tmp is the shape a box has.
SOCKDIR="$(mktemp -d /tmp/ds4s.XXXXXX)"
trap 'rm -rf "${WORK}" "${SOCKDIR}"' EXIT

pass() { printf 'test-ds4-resident: PASS -- %s\n' "$*"; }
fail() { printf 'test-ds4-resident: FAIL -- %s\n' "$*" >&2; exit 1; }

# --- build the real server against the synthetic engine ---------------------
SHIM="${ROOT_DIR}/harness/protocol-adapter/ds4_shim"
cc -O2 -std=c11 -D_GNU_SOURCE -Wall -Wextra -Werror -I "${SHIM}" \
  -o "${WORK}/ds4-resident" "${SHIM}/ds4_resident.c" "${ROOT_DIR}/tools/ds4/resident-stub-engine.c" \
  || fail "the resident server does not build against the synthetic engine"
pass "the real ds4_resident.c builds -Werror against a synthetic ds4_shim.h engine"

cargo build --quiet --manifest-path "${ROOT_DIR}/harness/protocol-adapter/Cargo.toml" \
  --bin cuda-engine || fail "cuda-engine does not build"
ENGINE="${ROOT_DIR}/harness/protocol-adapter/target/debug/cuda-engine"
[[ -x "${ENGINE}" ]] || fail "cuda-engine was not produced at ${ENGINE}"

# --- a synthetic window -----------------------------------------------------
WEIGHTS="${WORK}/weights"
# The shards carry a REAL GGUF tensor index, because serve-up.sh's memory plan
# sizes the streamed n-gram table out of it before any boot. The stub engine
# never opens them, so the data section is padding.
synthetic_window_weights "${WEIGHTS}" || fail "cannot write the synthetic artifact"

# The engine's memory declaration. serve-up.sh's plan refuses unless the pinned
# engine says it charges the n-gram table to an SSD-resident family, and it
# takes the headroom from there. This job does not check out the ds4 submodule,
# so the declaration is supplied here. tools/test-serve-up-plan-memory.sh owns
# the cases where it is missing.
ENGINE_HEADER="${WORK}/ds4_qwen4exp.h"
synthetic_window_engine_header "${ENGINE_HEADER}"

# The log directory has the RANKED JOB's shape: <runner>/_work/<repo>/<repo>,
# with this repository's 35-character name, plus .build/ds4. A socket named
# under THAT is longer than the 103 (macOS) or 107 (Linux) bytes a Unix socket
# address holds, so the socket lives in its own short directory instead.
#
# The two assertions below keep the coverage honest, and they are two because
# the socket name carries the run tag now:
#   * a socket named under the log dir would be over the limit, so if the socket
#     ever tracks the log directory again the four-phase window cannot boot;
#   * the socket where it actually lives is under the limit, so a run-tagged
#     name in a short directory has not quietly grown past it either.
REPO_LEAF="cudafast-qwen38-125b-a6b-engine"
LOGDIR="${WORK}/actions-runner/_work/${REPO_LEAF}/${REPO_LEAF}/.build/ds4"
mkdir -p "${LOGDIR}"
# serve-up.sh's own name shape: ds4-resident.<pid>-<UTC stamp>.sock.
SOCKET_LEAF="ds4-resident.$$-$(date -u +%Y%m%dT%H%M%SZ).sock"
SOCKET_UNDER_LOGDIR=$(( ${#LOGDIR} + 1 + ${#SOCKET_LEAF} ))
SOCKET_ACTUAL=$(( ${#SOCKDIR} + 1 + ${#SOCKET_LEAF} ))
[[ "${SOCKET_UNDER_LOGDIR}" -gt 103 ]] \
  || fail "a socket under the box-shaped log dir would be only ${SOCKET_UNDER_LOGDIR} bytes; the test no longer covers the long-path case"
[[ "${SOCKET_ACTUAL}" -le 103 ]] \
  || fail "the socket's own path is ${SOCKET_ACTUAL} bytes, past the 103 a Unix socket address holds"

meminfo() { printf 'MemTotal:       %s kB\nMemAvailable:   %s kB\n' "$1" "$1" > "${WORK}/meminfo.$2"; }
meminfo 268435456 plenty      # 256 GiB available
meminfo 1024      starved     # 1 MiB available

serve() {
  SERVE_UP_WEIGHTS_DIR="${WEIGHTS}" \
  SERVE_UP_LOG_DIR="${LOGDIR}" \
  SERVE_UP_SOCKET_DIR="${SOCKDIR}" \
  SERVE_UP_RESIDENT_BIN="${WORK}/ds4-resident" \
  SERVE_UP_SPECULATIVE=1 \
  SERVE_UP_SPEC_DRAFT_LEN=1 \
  SERVE_UP_HEALTH_TIMEOUT_S=60 \
  SERVE_UP_MEMINFO="$1" \
  SERVE_UP_ENGINE_HEADER="${ENGINE_HEADER}" \
  "${ROOT_DIR}/tools/serve-up.sh" "${@:2}"
}

# --- 6. the memory plan refuses BEFORE any load -----------------------------
# Run this FIRST, on a clean log directory, so "no open happened" is a fact
# about this run and not about ordering.
set +e
starved_out="$(serve "${WORK}/meminfo.starved" true 2>&1)"
starved_rc=$?
set -e
[[ "${starved_rc}" -ne 0 ]] || fail "serve-up.sh accepted a load with 1 MiB free"
grep -q 'memory plan REFUSES the load' <<<"${starved_out}" \
  || fail "the refusal did not name the memory plan: ${starved_out}"
grep -q 'Nothing has been loaded' <<<"${starved_out}" \
  || fail "the refusal did not state that nothing was loaded"
[[ -z "$(find "${LOGDIR}" -name 'ds4-resident.*.log' 2>/dev/null)" ]] \
  || fail "the resident was started despite the memory refusal"
pass "a memory plan that does not fit refuses before any load, and starts no resident"

# --- 1-5. one load, four phases -------------------------------------------
# Each phase is a FRESH cuda-engine, exactly as benchd spawns them: warmup,
# timed prefill, timed decode, correctness.
cat > "${WORK}/phases.sh" <<'PHASES'
set -euo pipefail
engine="$1"; out="$2"
phase() { printf '%s\n' "${@:2}" | "${engine}" > "${out}/phase-$1.out" 2> "${out}/phase-$1.err"; }
phase warmup   '{"id":1,"kind":"prefill","prompt_tokens":[11,12,13]}'
phase prefill  '{"id":1,"kind":"prefill","prompt_tokens":[11,12,13]}'
phase decode   '{"id":1,"kind":"free_decode_begin","seed_tokens":[11,12,13],"spec":{"mode":"mtp","mtp":{"depth":1}}}' \
               '{"id":2,"kind":"free_decode_run","count":6}'
phase correct  '{"id":1,"kind":"correctness_begin","prompt_tokens":[11,12,13]}' \
               '{"id":2,"kind":"correctness_step","token":50}'
PHASES

PHASE_OUT="${WORK}/phases"
mkdir -p "${PHASE_OUT}"
serve "${WORK}/meminfo.plenty" bash "${WORK}/phases.sh" "${ENGINE}" "${PHASE_OUT}" \
  || fail "the four-phase window did not complete"

RESIDENT_LOG="$(find "${LOGDIR}" -name 'ds4-resident.*.log' | head -1)"
[[ -n "${RESIDENT_LOG}" ]] || fail "no resident log was written"

opens="$(grep -c 'stub-engine: OPEN' "${RESIDENT_LOG}" || true)"
[[ "${opens}" == "1" ]] || fail "the engine was opened ${opens} times, not once"
pass "ONE load for four phases (the whole point of the topology)"

# A "phase" to the resident is any accepted connection, so the count includes
# serve-up.sh's health probes as well as the four workers. What is asserted
# here is that every connection was served and closed, and that the four
# WORKERS are among them -- each one says so in its own stderr.
phases_seen="$(grep -c 'phase [0-9]* connected' "${RESIDENT_LOG}" || true)"
closed="$(grep -c 'phase [0-9]* closed' "${RESIDENT_LOG}" || true)"
[[ "${phases_seen}" -ge 4 ]] || fail "the resident saw only ${phases_seen} connections"
[[ "${closed}" == "${phases_seen}" ]] \
  || fail "the resident closed ${closed} of ${phases_seen} connections"
workers="$(grep -l 'attached to the resident engine' "${PHASE_OUT}"/phase-*.err | wc -l | tr -d ' ')"
[[ "${workers}" == "4" ]] || fail "${workers} of 4 workers attached to the resident"
pass "four fresh workers reconnected to the one resident (${phases_seen} connections in all, every one closed)"

# 3. session reset. The resident invalidates on accept, and the worker's own
# drain invalidates again, so a phase can never inherit a live prefix.
invalidates="$(grep -c 'stub-engine: INVALIDATE' "${RESIDENT_LOG}" || true)"
[[ "${invalidates}" -ge 4 ]] || fail "only ${invalidates} session resets across 4 phases"
pass "the session is reset at every phase boundary (${invalidates} resets over 4 phases)"

# The two prefill phases ran the same prompt on the same resident. A leaked
# prefix would fold the prompt onto a dirty state and change the token, so
# equal tokens ARE the no-leak evidence.
warm_token="$(python3 -c 'import json,sys
for l in open(sys.argv[1]):
    r=json.loads(l)
    if r.get("id")==1 and "token" in r: print(r["token"]); break' "${PHASE_OUT}/phase-warmup.out")"
pref_token="$(python3 -c 'import json,sys
for l in open(sys.argv[1]):
    r=json.loads(l)
    if r.get("id")==1 and "token" in r: print(r["token"]); break' "${PHASE_OUT}/phase-prefill.out")"
[[ -n "${warm_token}" && "${warm_token}" == "${pref_token}" ]] \
  || fail "the same prompt gave ${warm_token} in warmup and ${pref_token} in prefill; a prefix leaked across the phase boundary"
pass "the same prompt gives the same token in two phases: no KV leaked across the boundary"

# 4. every phase's hello names the same resident, with the window's identity.
epochs="$(grep -ho 'load epoch [0-9]*' "${PHASE_OUT}"/phase-*.err | sort -u | wc -l | tr -d ' ')"
[[ "${epochs}" == "1" ]] || fail "the phases attached to ${epochs} different residents"
grep -q 'attached to the resident engine' "${PHASE_OUT}/phase-decode.err" \
  || fail "the worker did not report attaching to a resident"
grep -q 'nothing loaded in this process' "${PHASE_OUT}/phase-decode.err" \
  || fail "the worker did not state that it loaded nothing"
# THE SEALED STRING. benchd reads the hello's `backend` and seals it as
# `engine_backend` (bench-runner session.rs -> benchd official.rs
# seal_engine_identity), so this string is the ONLY place the scored artifact
# says which engine produced the number. It must name the topology and carry the
# resident's load_epoch: with the same value in every phase's hello, the "one
# load per window" claim is readable from the artifact alone.
hello_backend_of() { python3 -c 'import json,sys
for l in open(sys.argv[1]):
    r=json.loads(l)
    if r.get("id")==0: print(r.get("backend","")); break' "$1"; }
hello_device_of() { python3 -c 'import json,sys
for l in open(sys.argv[1]):
    r=json.loads(l)
    if r.get("id")==0: print(r.get("device","")); break' "$1"; }
hello_backend="$(hello_backend_of "${PHASE_OUT}/phase-decode.out")"
[[ "${hello_backend}" == ds4-resident\ load_epoch=* ]] \
  || fail "the sealed backend does not name the resident topology and its load epoch (got '${hello_backend}')"
grep -q 'mtp1' <<<"${hello_backend}" \
  || fail "the protocol hello does not carry the resident engine identity (got '${hello_backend}')"
[[ "${hello_backend}" != *mock* ]] \
  || fail "the sealed backend of a resident-attached worker reads as the mock (got '${hello_backend}')"
[[ "$(hello_device_of "${PHASE_OUT}/phase-decode.out")" == "cuda sm_121" ]] \
  || fail "the resident-attached worker sealed an unexpected device"
sealed_epoch="${hello_backend#ds4-resident load_epoch=}"
sealed_epoch="${sealed_epoch%% *}"
resident_pid="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["hello"]["load_epoch"])' "${LOGDIR}/serve-identity.json")"
[[ "${sealed_epoch}" == "${resident_pid}" ]] \
  || fail "the sealed load epoch is ${sealed_epoch}, the window's resident reported ${resident_pid}"
distinct_backends="$(for out in "${PHASE_OUT}"/phase-*.out; do hello_backend_of "${out}"; done | sort -u | wc -l | tr -d ' ')"
[[ "${distinct_backends}" == "1" ]] \
  || fail "the four phases sealed ${distinct_backends} different backend strings; the load repeated"
pass "every phase sealed the SAME backend string, and it carries the resident's load epoch: ${hello_backend}"

# 2. reconnect overhead. Milliseconds, not seconds, is the claim.
worst=0
for err in "${PHASE_OUT}"/phase-*.err; do
  ms="$(sed -n 's/.*in \([0-9.]*\) ms.*/\1/p' "${err}" | head -1)"
  [[ -n "${ms}" ]] || fail "no reconnect timing in ${err}"
  worst="$(awk -v a="${worst}" -v b="${ms}" 'BEGIN{print (b>a)?b:a}')"
done
awk -v w="${worst}" 'BEGIN{exit !(w < 250)}' \
  || fail "the worst per-phase reconnect was ${worst} ms; a reconnect must be milliseconds"
pass "worst per-phase reconnect (connect + hello): ${worst} ms"

# 5. the counters cross the wire.
run_line="$(python3 -c 'import json,sys
for l in open(sys.argv[1]):
    r=json.loads(l)
    if r.get("id")==2: print(json.dumps(r)); break' "${PHASE_OUT}/phase-decode.out")"
python3 - "${run_line}" <<'PY'
import json, sys
r = json.loads(sys.argv[1])
assert r["ok"] is True, r
assert r["committed_total"] == 6, r
assert len(r["tokens"]) == 6, r
assert sum(r["acceptance_lengths"]) == 6, r
assert r["drafted_total"] >= r["accepted_total"] > 0, r
PY
pass "the speculative leg's raw counters crossed the socket: ${run_line}"

# --- 7. clean teardown ------------------------------------------------------
# The socket name carries the run tag, so this asserts the DIRECTORY is empty
# of sockets rather than naming one file.
[[ -z "$(find "${SOCKDIR}" -name 'ds4-resident.*.sock' 2>/dev/null)" ]] \
  || fail "a socket survived teardown in ${SOCKDIR}"
[[ ! -f "${LOGDIR}/ds4-resident.ready" ]] || fail "the ready file survived teardown"
grep -q 'stub-engine: CLOSE' "${RESIDENT_LOG}" || fail "the resident did not close the engine"
grep -q "shutting down after ${phases_seen} phase(s) on one load" "${RESIDENT_LOG}" \
  || fail "the resident did not report a clean ${phases_seen}-phase shutdown on one load"
pass "teardown removed the socket and the ready file, and closed the engine"

# --- 8. the per-phase idle ceiling ------------------------------------------
# A wedged phase must not hold the window's weights. Drive the resident
# directly so the ceiling is the thing under test.
SOCK="${WORK}/timeout.sock"
DS4_RESIDENT_SOCKET="${SOCK}" DS4_MODEL="${WEIGHTS}/target-00001-of-00002.gguf" \
DS4_MTP_PATH="${WEIGHTS}/mtp-head.gguf" DS4_MTP_DRAFT_TOKENS=2 \
DS4_RESIDENT_PHASE_TIMEOUT_S=1 DS4_ENGINE_IDENT="stub" \
  "${WORK}/ds4-resident" > "${WORK}/timeout.log" 2>&1 &
TIMEOUT_PID=$!
for _ in $(seq 1 100); do [[ -S "${SOCK}" ]] && break; sleep 0.1; done
[[ -S "${SOCK}" ]] || fail "the resident never bound ${SOCK}"

python3 - "${SOCK}" <<'PY'
import socket, sys, time
# A phase that connects and says nothing must be dropped, not served forever.
idle = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
idle.settimeout(15)
idle.connect(sys.argv[1])
started = time.monotonic()
if idle.recv(4096) != b"":
    raise SystemExit("the idle phase was answered rather than dropped")
held = time.monotonic() - started
if held > 10:
    raise SystemExit(f"the idle phase held the resident for {held:.1f}s")
idle.close()
# And the resident is still there for the next phase.
nxt = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
nxt.settimeout(15)
nxt.connect(sys.argv[1])
nxt.sendall(b'{"op":"hello"}\n')
line = nxt.recv(65536)
if b'"ok":true' not in line:
    raise SystemExit(f"the resident did not survive the dropped phase: {line!r}")
nxt.close()
print(f"idle phase dropped after {held:.2f}s; the resident served the next one")
PY
kill -TERM "${TIMEOUT_PID}" 2>/dev/null || true
wait "${TIMEOUT_PID}" 2>/dev/null || true
grep -q 'phase idle for 1s; dropping the connection' "${WORK}/timeout.log" \
  || fail "the resident did not report dropping the idle phase"
[[ ! -S "${SOCK}" ]] || fail "SIGTERM left the socket behind"
pass "an idle phase is dropped on the resident's ceiling, and the resident serves the next one"

printf 'test-ds4-resident: all checks passed\n'
