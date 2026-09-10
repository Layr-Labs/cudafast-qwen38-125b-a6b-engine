#!/usr/bin/env bash
# test-serve-up-boot-stop.sh -- the per-leg boot/stop verbs in tools/serve-up.sh.
#
# The paired ranked path needs a resident that OUTLIVES the command that booted
# it: benchd boots each leg's resident from that leg's workspace, measures the
# leg through its own worker spawns, and stops it before the other leg boots.
# That is `--boot ... --socket-out FILE` and `--stop --socket PATH`, and the
# properties that make the pair usable are:
#
#   * the socket path is the FIRST LINE of the file, so one `head -1` reads it;
#   * --boot exits 0 with the resident STILL SERVING;
#   * --stop ends it and cleans up after it;
#   * --stop is IDEMPOTENT, because the caller runs it on success and on failure
#     alike and a second call must not turn a clean failure into two.
#
# This suite drives the REAL tools/serve-up.sh against a STUB resident: a small
# python server that binds the socket, writes the ready file and answers the
# hello a worker sends. The memory plan, the health probe, the identity file and
# the teardown are the real ones.
#
# Hermetic: no GPU, no engine, no checkpoint, no network. The artifact is the
# synthetic GGUF window every other off-box driver here uses.
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVE_UP="${ROOT_DIR}/tools/serve-up.sh"
# shellcheck source=tools/ds4/synthetic-window.sh
. "${ROOT_DIR}/tools/ds4/synthetic-window.sh"

WORK="$(mktemp -d)"
# A short socket directory. A Unix socket address holds 103 bytes on macOS, and
# the default mktemp directory plus serve-up.sh's run-tagged leaf is already
# close to it.
SOCKDIR="$(mktemp -d /tmp/suptest.XXXXXX)"
cleanup() {
  # Never leave a stub resident behind, whatever the suite did.
  for f in "${SOCKDIR}"/*.pid; do
    [[ -e "${f}" ]] || continue
    pid="$(sed -n 1p "${f}" | tr -d '[:space:]')"
    [[ -n "${pid}" ]] && kill -KILL "${pid}" 2>/dev/null
  done
  rm -rf "${WORK}" "${SOCKDIR}"
}
trap cleanup EXIT

failures=0
fail() { echo "FAIL: $*" >&2; failures=$((failures + 1)); }
ok()   { echo "ok: $*"; }

command -v python3 >/dev/null 2>&1 || { echo "test-serve-up-boot-stop.sh: python3 is required" >&2; exit 1; }
command -v jq      >/dev/null 2>&1 || { echo "test-serve-up-boot-stop.sh: jq is required" >&2; exit 1; }

# --- the stub resident ------------------------------------------------------
# It does what serve-up.sh's health probe requires and nothing else: bind the
# socket named by DS4_RESIDENT_SOCKET, write DS4_RESIDENT_READY_FILE, and answer
# every hello with ok:true until it is signalled.
RESIDENT="${WORK}/stub-resident"
cat > "${RESIDENT}" <<'STUB'
#!/usr/bin/env python3
import json, os, socket, sys

sock_path = os.environ["DS4_RESIDENT_SOCKET"]
ready = os.environ["DS4_RESIDENT_READY_FILE"]
srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
try:
    os.unlink(sock_path)
except FileNotFoundError:
    pass
srv.bind(sock_path)
srv.listen(8)
with open(ready, "w", encoding="utf-8") as fh:
    fh.write("ready\n")
reply = json.dumps({
    "ok": True, "vocab_size": 248320, "eos_token": 248046,
    "mtp_armed": bool(os.environ.get("DS4_MTP_PATH")),
    "draft_tokens": int(os.environ.get("DS4_MTP_DRAFT_TOKENS", "1")),
    "ctx_size": int(os.environ.get("DS4_CTX_SIZE", "8192")),
    "load_epoch": 1, "ident": os.environ.get("DS4_ENGINE_IDENT", ""),
}).encode() + b"\n"
while True:
    try:
        conn, _ = srv.accept()
    except OSError:
        break
    try:
        conn.recv(65536)
        conn.sendall(reply)
    except OSError:
        pass
    finally:
        conn.close()
sys.exit(0)
STUB
chmod +x "${RESIDENT}"

# --- the synthetic window ---------------------------------------------------
WEIGHTS="${WORK}/weights"
synthetic_window_weights "${WEIGHTS}" || { echo "cannot write the synthetic window" >&2; exit 1; }
ENGINE_HEADER="${WORK}/ds4_qwen4exp.h"
synthetic_window_engine_header "${ENGINE_HEADER}" || exit 1
printf 'MemTotal:       268435456 kB\nMemAvailable:   268435456 kB\n' > "${WORK}/meminfo"
LOGDIR="${WORK}/logs"

serve() {
  env SERVE_UP_WEIGHTS_DIR="${WEIGHTS}" \
      SERVE_UP_LOG_DIR="${LOGDIR}" \
      SERVE_UP_SOCKET_DIR="${SOCKDIR}" \
      SERVE_UP_RESIDENT_BIN="${RESIDENT}" \
      SERVE_UP_HEALTH_TIMEOUT_S=60 \
      SERVE_UP_MEMINFO="${WORK}/meminfo" \
      SERVE_UP_ENGINE_HEADER="${ENGINE_HEADER}" \
      "$@"
}

hello_ok() { # hello_ok SOCKET -- the probe a worker sends
  python3 - "$1" <<'PY'
import json, socket, sys
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.settimeout(10)
s.connect(sys.argv[1])
s.sendall(b'{"op":"hello"}\n')
line = b""
while not line.endswith(b"\n"):
    chunk = s.recv(65536)
    if not chunk:
        break
    line += chunk
s.close()
sys.exit(0 if json.loads(line.decode()).get("ok") is True else 1)
PY
}

# --- case 1: --boot publishes the socket and leaves the resident serving -----
SOCKET_OUT="${WORK}/leg1/socket.txt"
BOOT_OUT="${WORK}/boot.out"
serve "${SERVE_UP}" --boot --spec serial --draft-len 0 --socket-out "${SOCKET_OUT}" \
  > "${BOOT_OUT}" 2>&1
rc=$?
[[ "${rc}" -eq 0 ]] || { fail "case 1: --boot exited ${rc}"; sed 's/^/    /' "${BOOT_OUT}" >&2; }

SOCKET="$(head -1 "${SOCKET_OUT}" 2>/dev/null)"
if [[ -n "${SOCKET}" && -S "${SOCKET}" ]]; then
  ok "case 1: the socket path is the first line of the socket-out file"
else
  fail "case 1: the socket-out file does not name a bound socket (got '${SOCKET:-}')"
fi
[[ "$(wc -l < "${SOCKET_OUT}" | tr -d ' ')" == "1" ]] \
  && ok "case 1: the socket-out file is exactly the one line" \
  || fail "case 1: the socket-out file is not one line"

RESIDENT_PID="$(sed -n 1p "${SOCKET}.pid" 2>/dev/null | tr -d '[:space:]')"
if [[ -n "${RESIDENT_PID}" ]] && kill -0 "${RESIDENT_PID}" 2>/dev/null; then
  ok "case 1: --boot exited 0 with the resident still running (pid ${RESIDENT_PID})"
else
  fail "case 1: the resident did not survive --boot"
fi
hello_ok "${SOCKET}" \
  && ok "case 1: the handed-over resident answers the hello a worker sends" \
  || fail "case 1: the handed-over resident does not answer a hello"

# The control leg is SERIAL: no drafter armed, one draft token.
if [[ -f "${LOGDIR}/serve-identity.json" ]]; then
  spec="$(jq -r '.spec_config' "${LOGDIR}/serve-identity.json")"
  head="$(jq -r '.mtp_head_file' "${LOGDIR}/serve-identity.json")"
  drafts="$(jq -r '.mtp_draft_tokens' "${LOGDIR}/serve-identity.json")"
  [[ "${spec}" == "serial" && -z "${head}" && "${drafts}" == "1" ]] \
    && ok "case 1: --spec serial armed no drafter (spec_config=serial, no head, draft_tokens=1)" \
    || fail "case 1: --spec serial booted spec_config=${spec} head='${head}' draft_tokens=${drafts}"
else
  fail "case 1: no serve-identity.json was written"
fi

# --- case 2: --stop ends it and cleans up -----------------------------------
STOP_OUT="${WORK}/stop.out"
serve "${SERVE_UP}" --stop --socket "${SOCKET}" > "${STOP_OUT}" 2>&1
rc=$?
[[ "${rc}" -eq 0 ]] || { fail "case 2: --stop exited ${rc}"; sed 's/^/    /' "${STOP_OUT}" >&2; }
for _ in $(seq 1 25); do
  kill -0 "${RESIDENT_PID}" 2>/dev/null || break
  sleep 0.2
done
kill -0 "${RESIDENT_PID}" 2>/dev/null \
  && fail "case 2: the resident survived --stop" \
  || ok "case 2: --stop ended the resident"
[[ ! -e "${SOCKET}" ]] && ok "case 2: --stop removed the socket" || fail "case 2: the socket survived --stop"
[[ ! -e "${SOCKET}.pid" ]] && ok "case 2: --stop removed the pid sidecar" || fail "case 2: the sidecar survived --stop"
[[ ! -e "${LOGDIR}/ds4-resident.ready" ]] \
  && ok "case 2: --stop removed the ready file" \
  || fail "case 2: the ready file survived --stop"

# --- case 3: --stop is idempotent -------------------------------------------
serve "${SERVE_UP}" --stop --socket "${SOCKET}" > "${WORK}/stop2.out" 2>&1
[[ $? -eq 0 ]] \
  && ok "case 3: a second --stop is a no-op that exits 0" \
  || { fail "case 3: the second --stop did not exit 0"; sed 's/^/    /' "${WORK}/stop2.out" >&2; }
grep -q 'nothing to stop' "${WORK}/stop2.out" \
  && ok "case 3: the second --stop says there was nothing to stop" \
  || fail "case 3: the second --stop did not say what it found"

serve "${SERVE_UP}" --stop --socket "${SOCKDIR}/never-booted.sock" > "${WORK}/stop3.out" 2>&1
[[ $? -eq 0 ]] \
  && ok "case 3: --stop on a socket that was never booted exits 0" \
  || fail "case 3: --stop on an unknown socket did not exit 0"

# --- case 4: --spec mtp arms the drafter at the declared depth ---------------
rm -f "${LOGDIR}/serve-identity.json"
SOCKET_OUT2="${WORK}/leg2/socket.txt"
serve "${SERVE_UP}" --boot --spec mtp --draft-len 2 --socket-out "${SOCKET_OUT2}" \
  > "${WORK}/boot2.out" 2>&1
rc=$?
[[ "${rc}" -eq 0 ]] || { fail "case 4: --boot --spec mtp exited ${rc}"; sed 's/^/    /' "${WORK}/boot2.out" >&2; }
SOCKET2="$(head -1 "${SOCKET_OUT2}" 2>/dev/null)"
if [[ -f "${LOGDIR}/serve-identity.json" ]]; then
  spec="$(jq -r '.spec_config' "${LOGDIR}/serve-identity.json")"
  drafts="$(jq -r '.mtp_draft_tokens' "${LOGDIR}/serve-identity.json")"
  head="$(jq -r '.mtp_head_file' "${LOGDIR}/serve-identity.json")"
  [[ "${spec}" == "mtp2" && "${drafts}" == "3" && -n "${head}" ]] \
    && ok "case 4: --spec mtp --draft-len 2 armed the drafter (spec_config=mtp2, draft_tokens=3)" \
    || fail "case 4: booted spec_config=${spec} draft_tokens=${drafts} head='${head}'"
else
  fail "case 4: no serve-identity.json was written for the mtp boot"
fi
[[ -n "${SOCKET2}" ]] && serve "${SERVE_UP}" --stop --socket "${SOCKET2}" >/dev/null 2>&1

# --- case 5: the refusals ---------------------------------------------------
refuse() { # refuse LABEL NEEDLE [extra env...] -- everything after is argv
  local label="$1" needle="$2"; shift 2
  local out rc
  out="$("$@" 2>&1)"
  rc=$?
  if [[ "${rc}" -eq 0 ]]; then
    fail "case 5 / ${label}: exited 0"
    return
  fi
  grep -qF -- "${needle}" <<<"${out}" \
    && ok "case 5 / ${label}: refused by name" \
    || fail "case 5 / ${label}: the refusal does not say '${needle}'; got: $(tail -1 <<<"${out}")"
}

refuse "serial with a draft length" "the serial control leg arms no drafter" \
  env SERVE_UP_WEIGHTS_DIR="${WEIGHTS}" SERVE_UP_SOCKET_DIR="${SOCKDIR}" \
      "${SERVE_UP}" --boot --spec serial --draft-len 1 --socket-out "${WORK}/x.txt"
refuse "mtp with no draft length" "requires --draft-len N with N >= 1" \
  env SERVE_UP_WEIGHTS_DIR="${WEIGHTS}" SERVE_UP_SOCKET_DIR="${SOCKDIR}" \
      "${SERVE_UP}" --boot --spec mtp --socket-out "${WORK}/x.txt"
refuse "no spec" "--boot requires --spec serial|mtp" \
  env SERVE_UP_WEIGHTS_DIR="${WEIGHTS}" SERVE_UP_SOCKET_DIR="${SOCKDIR}" \
      "${SERVE_UP}" --boot --socket-out "${WORK}/x.txt"
refuse "no socket-out" "--boot requires --socket-out FILE" \
  env SERVE_UP_WEIGHTS_DIR="${WEIGHTS}" SERVE_UP_SOCKET_DIR="${SOCKDIR}" \
      "${SERVE_UP}" --boot --spec serial --draft-len 0
refuse "stop with no socket" "--stop requires --socket PATH" \
  env "${SERVE_UP}" --stop
# A FLAG AT THE END OF THE ARGV. `--spec` with nothing after it used to take the
# empty string and then `shift 2` ran off the end, which under `set -u` ended the
# script with a bare exit 1 and no message: the caller saw a failed boot and
# nothing saying why. One case per flag that takes a value.
refuse "--spec with no value" "--spec requires a value" \
  env SERVE_UP_WEIGHTS_DIR="${WEIGHTS}" SERVE_UP_SOCKET_DIR="${SOCKDIR}" \
      "${SERVE_UP}" --boot --spec
refuse "--draft-len with no value" "--draft-len requires a value" \
  env SERVE_UP_WEIGHTS_DIR="${WEIGHTS}" SERVE_UP_SOCKET_DIR="${SOCKDIR}" \
      "${SERVE_UP}" --boot --spec serial --draft-len
refuse "--socket-out with no value" "--socket-out requires a value" \
  env SERVE_UP_WEIGHTS_DIR="${WEIGHTS}" SERVE_UP_SOCKET_DIR="${SOCKDIR}" \
      "${SERVE_UP}" --boot --spec serial --draft-len 0 --socket-out
refuse "--socket with no value" "--socket requires a value" \
  env "${SERVE_UP}" --stop --socket
# THE ONE THAT PROTECTS THE CONTROL LEG: an inherited speculative value cannot
# turn a serial boot into a speculative one.
refuse "environment disagrees with the flag" "in --boot mode the flag is the authority" \
  env SERVE_UP_WEIGHTS_DIR="${WEIGHTS}" SERVE_UP_SOCKET_DIR="${SOCKDIR}" \
      SERVE_UP_SPECULATIVE=1 \
      "${SERVE_UP}" --boot --spec serial --draft-len 0 --socket-out "${WORK}/x.txt"

# --- case 5b: a HANDOVER that fails leaves no stale pid sidecar --------------
# --boot writes <socket>.pid FIRST and publishes the socket second, so the caller
# never holds a socket path it cannot stop. That ordering creates the one window
# this case covers: a failure BETWEEN the sidecar write and the handover. The
# teardown then runs, and a sidecar left behind would name a dead pid and outlive
# every run that could remove it -- nothing later looks for it, because no caller
# ever learned the socket path.
#
# The failure is an unwritable --socket-out. /dev/null is not a directory, so the
# mkdir of its "subdirectory" fails with ENOTDIR after the sidecar exists. A boot
# that dies EARLIER (a resident that never comes up) would not exercise this at
# all: the sidecar would not have been written yet, and the case would pass on a
# teardown that removes nothing.
sidecars_before="$(find "${SOCKDIR}" -name '*.sock.pid' | wc -l | tr -d ' ')"
serve "${SERVE_UP}" --boot --spec serial --draft-len 0 \
  --socket-out /dev/null/not-a-directory/socket.txt \
  > "${WORK}/badhandover.out" 2>&1
[ $? -ne 0 ] \
  && ok "case 5b: a boot whose handover cannot be published fails" \
  || fail "case 5b: --boot exited 0 without publishing its socket"
sidecars_after="$(find "${SOCKDIR}" -name '*.sock.pid' | wc -l | tr -d ' ')"
[ "${sidecars_after}" = "${sidecars_before}" ] \
  && ok "case 5b: the failed handover left no stale pid sidecar behind" \
  || fail "case 5b: the failed handover left $(( sidecars_after - sidecars_before )) stale sidecar(s)"
leftover_socks="$(find "${SOCKDIR}" -name '*.sock' | wc -l | tr -d ' ')"
[ "${leftover_socks}" = "0" ] \
  && ok "case 5b: the failed handover left no socket behind" \
  || fail "case 5b: the failed handover left ${leftover_socks} socket(s)"

# --- case 6: the wrapper form still works -----------------------------------
# Local drivers (tools/benchmark.sh, tools/qwen4exp-calibrate.sh,
# tools/qwen4exp-golden-reauthor.sh) still use it, so the per-leg verbs must be
# additive.
rm -f "${LOGDIR}/serve-identity.json"
WRAP_OUT="${WORK}/wrap.out"
serve "${SERVE_UP}" sh -c 'test -S "${DS4_RESIDENT_SOCKET}" && echo WRAPPED_OK' \
  > "${WRAP_OUT}" 2>&1
rc=$?
[[ "${rc}" -eq 0 ]] && grep -q 'WRAPPED_OK' "${WRAP_OUT}" \
  && ok "case 6: the wrapper form still boots, runs the command and exits with its code" \
  || { fail "case 6: the wrapper form regressed (exit ${rc})"; sed 's/^/    /' "${WRAP_OUT}" >&2; }
# And it still tears its own resident down: nothing is left listening.
leftover=0
for f in "${SOCKDIR}"/*.sock; do
  [[ -e "${f}" ]] && leftover=$((leftover + 1))
done
[[ "${leftover}" -eq 0 ]] \
  && ok "case 6: the wrapper form left no resident behind" \
  || fail "case 6: ${leftover} socket(s) survived the wrapper run"

if [[ "${failures}" -eq 0 ]]; then
  echo "PASS: test-serve-up-boot-stop.sh"
  exit 0
fi
echo "FAIL: ${failures} failure(s)" >&2
exit 1
