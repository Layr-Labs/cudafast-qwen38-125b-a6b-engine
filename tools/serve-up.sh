#!/usr/bin/env bash
# serve-up.sh -- boot the ONE resident ds4 engine for a benchmark window.
#
# THREE FORMS. The first is the PER-LEG pair the paired ranked path uses; the
# third is the original wrapper, which local drivers still use.
#
#   tools/serve-up.sh --boot --spec serial|mtp --draft-len N --socket-out FILE
#   tools/serve-up.sh --stop --socket PATH
#   tools/serve-up.sh CMD [ARGS...]
#
# --boot boots ONE resident FROM THE TREE THIS SCRIPT LIVES IN, waits for the
# healthy hello, writes the resident's socket path as the FIRST LINE of FILE,
# and EXITS 0 WITH THE RESIDENT STILL RUNNING. --stop tears that resident down
# and is idempotent: a second stop, or a stop of a resident that already died,
# is a no-op that exits 0. A caller uses --stop on success and on failure alike.
#
# WHY THE PAIR EXISTS. The ranked run is PAIRED, and each leg has its own tree:
# the serial control leg runs the REFERENCE tree's engine, the candidate leg
# runs the submission's. benchd runs each leg's serve-up FROM THAT LEG'S
# WORKSPACE, and injects the socket it reads back into that leg's worker spawns
# as DS4_RESIDENT_SOCKET and BENCH_WORKER_RESIDENT_SOCKET. The wrapper form
# cannot express that: it owns the whole command, so it can only serve one tree
# for the whole run.
#
# LEG 1 IS ALWAYS `--spec serial --draft-len 0`. The control leg is serial
# whatever the candidate declares, and in --boot mode the FLAGS are the
# authority: a SERVE_UP_SPECULATIVE in the environment that disagrees is
# refused rather than honoured.
#
# ONE RESIDENT AT A TIME. A resident holds ~103.7 GiB, and the memory plan below
# refuses unless that much is free RIGHT NOW. So the two legs are sequential:
# boot leg 1, measure it, stop it, and only then boot leg 2. The plan has to
# hold for each boot on its own, which is exactly what it checks.
#
# WEIGHTS LOAD ONCE PER WINDOW. benchd's CUDA residency is FreshPerPhase: it
# spawns a fresh `cuda-engine` for warmup, timed prefill, timed decode and
# correctness, and an official run does that for every pair of every cohort on
# both legs. With the engine linked into the worker, each of those loaded the
# whole artifact again -- at least sixteen loads of a 103.7 GiB body (plus the
# 2.8 GB draft head on the speculative leg) in one ranked run, which no
# twenty-minute pipeline survives.
#
# So the weights get an OWNER: `ds4-resident` (built by tools/ds4/build.sh from
# harness/protocol-adapter/ds4_shim/ds4_resident.c). This script boots exactly
# one of them, inside the caller's GPU-lock window, and exports
# DS4_RESIDENT_SOCKET. Every per-phase cuda-engine connects to it and loads
# nothing. The MTP draft head is opened by the SAME resident process, so the
# speculative leg does not reload either.
#
# NOT UPSTREAM'S ds4-server. `make cuda-spark` builds upstream's ds4-server and
# this script deliberately does not use it: that server speaks OpenAI/Anthropic
# chat over HTTP, and the word "logit" does not occur in it once. It cannot
# carry top-k logits, token-id input, teacher-forced eval, or per-cycle
# speculative counters. docs/ds4-resident.md carries the mapping table and the
# evidence.
#
# GPU LOCK -- NOT TAKEN HERE. The CALLER owns the box GPU window
# (.github/workflows/benchmark.yml holds the box-wide flock on
# /tmp/mtplx-gpu-exclusive.lock for the whole measurement) and this script
# boots the resident inside it. A resident holds GPU memory, so in resident
# mode it must never outlive that lock window.
#
# Usage:
#   SERVE_UP_WEIGHTS_DIR=<dir> tools/serve-up.sh --boot --spec serial --draft-len 0 --socket-out FILE
#   tools/serve-up.sh --stop --socket PATH
#   SERVE_UP_WEIGHTS_DIR=<dir with the GGUF shards and the MTP head> tools/serve-up.sh CMD [ARGS...]
#
# Environment:
#   SERVE_UP_WEIGHTS_DIR      the pinned target snapshot directory (required;
#                             MLXFAST_TARGET_SNAPSHOT_DIR is honoured as a fallback)
#   SERVE_UP_MODEL_FILE       first GGUF shard (default: the *-00001-of-*.gguf in the dir)
#   SERVE_UP_MTP_HEAD_FILE    native MTP draft head (default: the mtp-*.gguf in the dir)
#   SERVE_UP_SPECULATIVE      0 = serial serve (DS4_MTP_DRAFT_TOKENS=1); 1 = MTP serve
#                             at SERVE_UP_SPEC_DRAFT_LEN drafts. DERIVED by the caller
#                             from mtp-head.manifest.json via tools/spec-declaration.sh.
#                             Default 0 -- the stock serial launch reference.
#   SERVE_UP_SPEC_DRAFT_LEN   declared num_speculative_tokens (default 1 when SPECULATIVE=1)
#   SERVE_UP_CTX_SIZE         resident session context in tokens (default 8192)
#   SERVE_UP_LOG_DIR          where the run's identity, ready file and resident
#                             log are kept (default <repo>/.build/ds4)
#   SERVE_UP_SOCKET_DIR       where the resident's Unix socket is bound (default
#                             ${TMPDIR:-/tmp}). It is SEPARATE from the log dir
#                             because a Unix socket path holds only 103 bytes
#                             (macOS) or 107 (Linux), and the ranked job's
#                             workspace -- <runner>/_work/<repo>/<repo> with a
#                             35-character repository name -- is already over
#                             that on its own. A socket under the log dir made
#                             the resident refuse to bind on the box.
#                             The socket FILE carries the run tag, like the
#                             resident log, so two overlapping windows on one box
#                             do not collide in a shared directory the way they
#                             could not in their own checkouts. The name is about
#                             48 bytes, which leaves the default directories far
#                             inside the limit.
#   SERVE_UP_RESIDENT_BIN     the resident binary (default <repo>/.build/ds4/ds4-resident)
#   SERVE_UP_HEALTH_TIMEOUT_S ceiling on the load + first healthy hello (default 5400)
#   SERVE_UP_MEMINFO          where free RAM is read from (default /proc/meminfo)
#   SERVE_UP_ENGINE_HEADER    where the engine's memory declaration is read from
#                             (default <repo>/ds4/ds4_qwen4exp.h)
#
# The last two variables name WHERE a fact is read from. They are test seams and
# they are NOT knobs: neither can relax the plan. The plan itself has no knob.
# Its headroom comes from the pinned engine's own header, its session budget is
# a constant below, and the refusal cannot be turned off from the outside.
# SERVE_UP_MEM_HEADROOM_GB is gone -- a refusal a caller can switch off is not a
# refusal.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null && pwd -P)"
log() { printf 'serve-up.sh: %s\n' "$*" >&2; }
die() { printf 'serve-up.sh: %s\n' "$*" >&2; exit 1; }

usage() {
  die "usage: SERVE_UP_WEIGHTS_DIR=<dir> tools/serve-up.sh --boot --spec serial|mtp --draft-len N --socket-out FILE
       tools/serve-up.sh --stop --socket PATH
       SERVE_UP_WEIGHTS_DIR=<dir> tools/serve-up.sh CMD [ARGS...]"
}

[[ $# -ge 1 ]] || usage

MODE="wrapper"
BOOT_SPEC=""
BOOT_DRAFT_LEN=""
BOOT_SOCKET_OUT=""
STOP_SOCKET=""

# A FLAG WITH NO VALUE IS REFUSED BY NAME. `--spec` at the end of the argv used
# to take "${2:-}" -- the empty string -- and then `shift 2` ran off the end,
# which under `set -u` ended the script with a bare exit 1 and no message. A
# caller then saw a failed boot and nothing saying why. The value is required
# here, before the shift, so the refusal names the flag that is missing one.
need_value() { # need_value FLAG VALUE_OR_EMPTY
  [[ $# -ge 2 && -n "$2" ]] || die "$1 requires a value"
}

case "$1" in
  --boot)
    MODE="boot"; shift
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --spec)       need_value "$@"; BOOT_SPEC="$2"; shift 2 ;;
        --draft-len)  need_value "$@"; BOOT_DRAFT_LEN="$2"; shift 2 ;;
        --socket-out) need_value "$@"; BOOT_SOCKET_OUT="$2"; shift 2 ;;
        *) die "--boot: unknown argument '$1'" ;;
      esac
    done
    ;;
  --stop)
    MODE="stop"; shift
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --socket) need_value "$@"; STOP_SOCKET="$2"; shift 2 ;;
        *) die "--stop: unknown argument '$1'" ;;
      esac
    done
    ;;
esac

# --- --stop: idempotent teardown of one booted resident ----------------------
# It needs NOTHING the boot needed -- no weights, no engine header, no memory
# plan. Everything it must know is beside the socket, so it runs before any of
# that is resolved and cannot fail for a reason unrelated to stopping.
#
# THE PID AND THE READY FILE COME FROM THE SIDECAR the boot wrote
# (<socket>.pid: the pid on line 1, the ready file on line 2), never from a
# guess about what is listening on the socket. A stop that had to infer its
# target could kill the wrong process on a shared box.
if [[ "${MODE}" == "stop" ]]; then
  [[ -n "${STOP_SOCKET}" ]] || die "--stop requires --socket PATH"
  stop_pidfile="${STOP_SOCKET}.pid"
  stop_pid=""
  stop_ready=""
  if [[ -f "${stop_pidfile}" ]]; then
    stop_pid="$(sed -n 1p "${stop_pidfile}" | tr -d '[:space:]')"
    stop_ready="$(sed -n 2p "${stop_pidfile}")"
  fi
  if [[ -n "${stop_pid}" ]] && kill -0 "${stop_pid}" 2>/dev/null; then
    kill -TERM "${stop_pid}" 2>/dev/null || true
    for _ in $(seq 1 50); do
      kill -0 "${stop_pid}" 2>/dev/null || break
      sleep 0.2
    done
    if kill -0 "${stop_pid}" 2>/dev/null; then
      log "resident pid ${stop_pid} did not stop on SIGTERM; killing it"
      kill -KILL "${stop_pid}" 2>/dev/null || true
    fi
    log "resident pid ${stop_pid} stopped"
  else
    # IDEMPOTENT BY DESIGN. A second --stop, or a --stop of a resident that
    # already exited, is a no-op and exits 0: the caller runs it on success and
    # on failure alike, and a teardown that refused the second call would turn
    # a clean failure path into a second failure.
    log "no running resident for ${STOP_SOCKET}; nothing to stop"
  fi
  rm -f "${STOP_SOCKET}" "${stop_pidfile}" 2>/dev/null || true
  [[ -z "${stop_ready}" ]] || rm -f "${stop_ready}" 2>/dev/null || true
  exit 0
fi

command -v python3 >/dev/null 2>&1 || die "python3 is required (it speaks the resident's health probe)"

# --- --boot: the flags are the authority over the spec -----------------------
if [[ "${MODE}" == "boot" ]]; then
  [[ -n "${BOOT_SOCKET_OUT}" ]] || die "--boot requires --socket-out FILE"
  case "${BOOT_SPEC}" in
    serial)
      # A serial boot with a draft length is a contradiction, not a default to
      # silently ignore: the caller believes it asked for a drafter.
      [[ -z "${BOOT_DRAFT_LEN}" || "${BOOT_DRAFT_LEN}" == "0" ]]         || die "--boot --spec serial takes --draft-len 0 (got '${BOOT_DRAFT_LEN}'); the serial control leg arms no drafter"
      boot_speculative=0
      ;;
    mtp)
      [[ "${BOOT_DRAFT_LEN}" =~ ^[1-9][0-9]*$ ]]         || die "--boot --spec mtp requires --draft-len N with N >= 1 (got '${BOOT_DRAFT_LEN}')"
      boot_speculative=1
      ;;
    "") die "--boot requires --spec serial|mtp" ;;
    *)  die "--boot --spec must be serial or mtp (got '${BOOT_SPEC}')" ;;
  esac
  # THE ENVIRONMENT MAY NOT DISAGREE WITH THE FLAG. On the ranked path leg 1 is
  # serial whatever the candidate declares, so an inherited SERVE_UP_SPECULATIVE
  # that says otherwise is a refusal, never an override.
  if [[ -n "${SERVE_UP_SPECULATIVE:-}" && "${SERVE_UP_SPECULATIVE}" != "${boot_speculative}" ]]; then
    die "--boot --spec ${BOOT_SPEC} derives SERVE_UP_SPECULATIVE=${boot_speculative} but the environment carries ${SERVE_UP_SPECULATIVE}; in --boot mode the flag is the authority and a disagreeing environment is refused"
  fi
  SERVE_UP_SPECULATIVE="${boot_speculative}"
  [[ "${boot_speculative}" == "0" ]] || SERVE_UP_SPEC_DRAFT_LEN="${BOOT_DRAFT_LEN}"
fi

SERVE_UP_WEIGHTS_DIR="${SERVE_UP_WEIGHTS_DIR:-${MLXFAST_TARGET_SNAPSHOT_DIR:-}}"
[[ -n "${SERVE_UP_WEIGHTS_DIR}" ]] || die "set SERVE_UP_WEIGHTS_DIR (or MLXFAST_TARGET_SNAPSHOT_DIR) to the target snapshot"
[[ -d "${SERVE_UP_WEIGHTS_DIR}" ]] || die "target snapshot directory is missing: '${SERVE_UP_WEIGHTS_DIR}'"

if [[ -z "${SERVE_UP_MODEL_FILE:-}" ]]; then
  shopt -s nullglob
  shards=( "${SERVE_UP_WEIGHTS_DIR}"/*-00001-of-*.gguf )
  shopt -u nullglob
  [[ ${#shards[@]} -eq 1 ]] || die "expected exactly one *-00001-of-*.gguf in ${SERVE_UP_WEIGHTS_DIR}, found ${#shards[@]} (set SERVE_UP_MODEL_FILE)"
  SERVE_UP_MODEL_FILE="${shards[0]}"
fi
[[ -f "${SERVE_UP_MODEL_FILE}" ]] || die "model file is missing: ${SERVE_UP_MODEL_FILE}"
if [[ -z "${SERVE_UP_MTP_HEAD_FILE:-}" ]]; then
  shopt -s nullglob
  heads=( "${SERVE_UP_WEIGHTS_DIR}"/mtp-*.gguf )
  shopt -u nullglob
  [[ ${#heads[@]} -eq 1 ]] || die "expected exactly one mtp-*.gguf in ${SERVE_UP_WEIGHTS_DIR}, found ${#heads[@]} (set SERVE_UP_MTP_HEAD_FILE)"
  SERVE_UP_MTP_HEAD_FILE="${heads[0]}"
fi
[[ -f "${SERVE_UP_MTP_HEAD_FILE}" ]] || die "the native MTP draft head is missing: ${SERVE_UP_MTP_HEAD_FILE} (it sits flat beside the first shard)"

SERVE_UP_SPECULATIVE="${SERVE_UP_SPECULATIVE:-0}"
case "${SERVE_UP_SPECULATIVE}" in 0|1) ;; *) die "SERVE_UP_SPECULATIVE must be 0 or 1 (got '${SERVE_UP_SPECULATIVE}')" ;; esac
if [[ "${SERVE_UP_SPECULATIVE}" == "1" ]]; then
  SERVE_UP_SPEC_DRAFT_LEN="${SERVE_UP_SPEC_DRAFT_LEN:-1}"
  [[ "${SERVE_UP_SPEC_DRAFT_LEN}" =~ ^[1-9][0-9]*$ ]] || die "SERVE_UP_SPEC_DRAFT_LEN must be a positive integer (got '${SERVE_UP_SPEC_DRAFT_LEN}')"
  # ds4 counts the fed token: draft_tokens = 1 + declared draft length.
  DS4_MTP_DRAFT_TOKENS=$((SERVE_UP_SPEC_DRAFT_LEN + 1))
  SPEC_LABEL="mtp${SERVE_UP_SPEC_DRAFT_LEN}"
else
  DS4_MTP_DRAFT_TOKENS=1
  SPEC_LABEL="serial"
fi

SERVE_UP_CTX_SIZE="${SERVE_UP_CTX_SIZE:-8192}"
SERVE_UP_LOG_DIR="${SERVE_UP_LOG_DIR:-${SCRIPT_DIR}/.build/ds4}"
SERVE_UP_SOCKET_DIR="${SERVE_UP_SOCKET_DIR:-${TMPDIR:-/tmp}}"
SERVE_UP_RESIDENT_BIN="${SERVE_UP_RESIDENT_BIN:-${SCRIPT_DIR}/.build/ds4/ds4-resident}"
SERVE_UP_HEALTH_TIMEOUT_S="${SERVE_UP_HEALTH_TIMEOUT_S:-5400}"
SERVE_UP_MEMINFO="${SERVE_UP_MEMINFO:-/proc/meminfo}"

# --- the memory plan's constants ---------------------------------------------
# The track fixture is the authority for the artifact. The plan reads the
# streamed table's quantization from it, so the plan and the pinned target
# cannot disagree.
TRACK_FIXTURE="${SCRIPT_DIR}/fixtures/qwen3_8_125b_a6b_track.json"

# The per-layer n-gram table. The engine keeps it mapped on the solid-state
# disk and streams it, so it is NOT part of the resident set.
PLE_TENSOR_NAME="per_layer_token_embd.weight"

# THE PLAN ONLY HOLDS FOR AN ENGINE THAT STREAMS THE TABLE. Subtracting the
# n-gram table LOWERS the gate, so it is valid only against an engine that
# really keeps that table off the resident set. Upstream ds4 has no qwen4exp
# family at all and would load the whole body. So the plan reads the pinned
# engine's own declaration and REFUSES BY NAME when it is not there:
#
#   DS4_QWEN4EXP_MEM_PLE                 the engine charges the table to its own
#                                        family and subtracts it from the
#                                        resident total
#   DS4_QWEN4EXP_MEMORY_HEADROOM_BYTES   the headroom every allocating path owes
#                                        (issue #82), and the value this plan
#                                        uses -- READ from the engine, never
#                                        restated here
#
# Both must be declared in the header below, which is the vendored engine tree
# in this repository. This is why the plan is correct on any tree:
# on a tree whose pin has no qwen4exp port it refuses instead of guessing.
ENGINE_HEADER="${SERVE_UP_ENGINE_HEADER:-${SCRIPT_DIR}/ds4/ds4_qwen4exp.h}"
ENGINE_PLE_CHARGE_SYMBOL="DS4_QWEN4EXP_MEM_PLE"
ENGINE_HEADROOM_SYMBOL="DS4_QWEN4EXP_MEMORY_HEADROOM_BYTES"

# Session budget: KV and GDN recurrent state, the indexer tape, the dilated PLE
# convolution window, and one forward's activations. The engine sizes this
# exactly in ds4_qwen4exp_session_plan_compute(n_ctx, n_batch). Those terms are
# linear in the context, and PR #90 records them well under 2 GiB at ctx 4096,
# so 4 GiB covers a session of twice that context with the same margin.
#
# THE CEILING IS PART OF THE BUDGET. SERVE_UP_CTX_SIZE feeds DS4_CTX_SIZE, so a
# caller could otherwise ask for a longer context and be charged the same 4 GiB.
# A context above the ceiling is REFUSED. Raising the ceiling means raising the
# budget with it, against a measurement from ds4_qwen4exp_session_plan_print().
MEM_SESSION_GIB=4
MEM_SESSION_CTX_CEILING=8192
mkdir -p "${SERVE_UP_LOG_DIR}" "${SERVE_UP_SOCKET_DIR}"

[[ -x "${SERVE_UP_RESIDENT_BIN}" ]] \
  || die "the resident engine binary is missing: ${SERVE_UP_RESIDENT_BIN} (run tools/ds4/build.sh)"

RUN_TAG="$$-$(date -u +%Y%m%dT%H%M%SZ)"

# --- the engine environment the resident and every worker read ---------------
export DS4_MODEL="${SERVE_UP_MODEL_FILE}"
export DS4_MTP_DRAFT_TOKENS
export DS4_CTX_SIZE="${SERVE_UP_CTX_SIZE}"
export DS4_EOS_IDS="${DS4_EOS_IDS:-248046,248044}"
# ds4 takes the draft head as a SEPARATE model file (its `--mtp-model`, which
# the shim reaches through ds4_engine_options.mtp_path), loaded by the resident
# alongside the body. The serial leg arms no drafter, so it loads no head at
# all.
if [[ "${SERVE_UP_SPECULATIVE}" == "1" ]]; then
  export DS4_MTP_PATH="${SERVE_UP_MTP_HEAD_FILE}"
else
  unset DS4_MTP_PATH || true
fi
# The engine identity the resident announces and every worker's hello carries
# (benchd records the hello's backend/device strings): the pinned engine
# commit, the toolchain, and the declared depth.
# Each is best-effort: a box without nvcc or nvidia-smi records "unknown"
# rather than dying silently mid-identity under `set -e` + `pipefail`.
engine_pin="$(jq -r '.fork.sha // empty' "${SCRIPT_DIR}/ds4/VENDOR.json" 2>/dev/null || true)"
nvcc_ver="$(nvcc --version 2>/dev/null | sed -n 's/.*release [0-9.]*, \(V[0-9.]*\).*/\1/p' | head -1 || true)"
driver_ver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d '[:space:]' || true)"
export DS4_ENGINE_IDENT="ds4@${engine_pin:-unknown} nvcc=${nvcc_ver:-unknown} driver=${driver_ver:-unknown} ${SPEC_LABEL} draft_tokens=${DS4_MTP_DRAFT_TOKENS}"

# What this window's resident is: the artifact, the head, the declared depth,
# the context and the engine pin. It is recorded in the identity file as
# provenance for the run, not as a reuse key -- the resident is booted and torn
# down inside ONE window, and there is no path that adopts an existing one.
SERVE_IDENTITY_KEY="${DS4_MODEL}|${DS4_MTP_PATH:-none}|${DS4_MTP_DRAFT_TOKENS}|${SERVE_UP_CTX_SIZE}|${engine_pin:-unknown}"
IDENTITY_FILE="${SERVE_UP_LOG_DIR}/serve-identity.json"
SOCKET_PATH="${SERVE_UP_SOCKET_DIR}/ds4-resident.${RUN_TAG}.sock"
READY_FILE="${SERVE_UP_LOG_DIR}/ds4-resident.ready"
RESIDENT_LOG="${SERVE_UP_LOG_DIR}/ds4-resident.${RUN_TAG}.log"

export DS4_RESIDENT_SOCKET="${SOCKET_PATH}"
export DS4_RESIDENT_READY_FILE="${READY_FILE}"

# --- the health probe: a real hello over the real socket ---------------------
# Not "the process is alive" and not "the file exists": the probe opens the
# socket, sends the hello verb a worker sends, and requires ok:true back. A
# resident that bound its socket but cannot answer is not healthy.
probe_resident() {
  python3 - "$1" <<'PY'
import json, socket, sys
try:
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
    reply = json.loads(line.decode())
except Exception as err:                       # noqa: BLE001 - any failure is unhealthy
    print(f"probe failed: {err}", file=sys.stderr)
    sys.exit(1)
if reply.get("ok") is not True:
    print(f"probe refused: {reply}", file=sys.stderr)
    sys.exit(1)
print(json.dumps({k: reply.get(k) for k in
                  ("vocab_size", "eos_token", "mtp_armed", "draft_tokens", "ctx_size",
                   "load_epoch", "ident")}))
PY
}

# --- the memory plan: refuse BEFORE the load, never during it ----------------
# The resident set is the body's shards MINUS the streamed n-gram table, plus
# the draft head on the speculative leg, plus the session, plus headroom. If the
# box does not have that free RIGHT NOW, this refuses here -- an OOM during a
# load takes the box down with it, and has (2026-08-29, ai-server).
#
# THE NUMBER THAT IS EASY TO GET WRONG IS THE PLE TABLE. The body is 103.69 GiB,
# but not all of it becomes resident: per_layer_token_embd.weight stays mapped
# on the solid-state disk and is streamed. The engine subtracts it
#   (ds4_qwen4exp.inc: plan->ssd_bytes = plan->bytes[DS4_QWEN4EXP_MEM_PLE];
#    plan->resident_bytes = plan->total_bytes - plan->ssd_bytes)
# so this plan subtracts the same tensor. The bytes come from the artifact's own
# GGUF tensor index, through tools/gguf-tensor-bytes.py. They are never guessed
# and never hardcoded: charging the table refuses a load that fits on a 119 GiB
# Spark, and a hardcoded number gets a different artifact wrong.
#
# The subtraction is correct only while the engine streams that table, so the
# plan REFUSES BY NAME when the tensor is absent from the index, or when its
# type is not the one the fixture declares (target.quantization.ple_table).

# Print a byte count in GiB, to two decimal places.
gib() { awk -v b="$1" 'BEGIN{printf "%.2f", b/1073741824}'; }

plan_memory() {
  local shard_report shard_count body_bytes ple_type ple_bytes resident_bytes
  local head_bytes session_bytes headroom_bytes want_bytes avail_kb avail_bytes

  # The context this window will ask the engine for must be one the session
  # budget below actually covers.
  [[ "${SERVE_UP_CTX_SIZE}" =~ ^[1-9][0-9]*$ ]] \
    || die "memory plan REFUSES the load: SERVE_UP_CTX_SIZE must be a positive integer (got '${SERVE_UP_CTX_SIZE}'). Nothing has been loaded."
  if (( SERVE_UP_CTX_SIZE > MEM_SESSION_CTX_CEILING )); then
    die "memory plan REFUSES the load: SERVE_UP_CTX_SIZE is ${SERVE_UP_CTX_SIZE}, and the ${MEM_SESSION_GIB} GiB session budget is only stated to cover ${MEM_SESSION_CTX_CEILING} tokens. Measure the session with ds4_qwen4exp_session_plan_print() and raise MEM_SESSION_GIB and MEM_SESSION_CTX_CEILING together. Nothing has been loaded."
  fi

  # THE ENGINE MUST DECLARE THE BEHAVIOUR THE PLAN RELIES ON, and it supplies
  # the headroom. An engine that does not stream the n-gram table loads the
  # whole body, so the subtraction below would lower the gate under it.
  headroom_bytes="$(python3 - "${ENGINE_HEADER}" "${ENGINE_PLE_CHARGE_SYMBOL}" "${ENGINE_HEADROOM_SYMBOL}" <<'ENGINEPY'
import ast, os, re, sys
path, charge, headroom = sys.argv[1], sys.argv[2], sys.argv[3]
if not os.path.exists(path):
    print(f"engine-declaration-absent: no {path}; the pinned engine declares no "
          f"qwen4exp memory plan, so the n-gram table cannot be treated as streamed",
          file=sys.stderr)
    raise SystemExit(1)
text = open(path, encoding="utf-8", errors="replace").read()
if not re.search(r"\b" + re.escape(charge) + r"\b", text):
    print(f"engine-charge-absent: {path} declares no {charge}; this engine does not "
          f"charge the n-gram table to an SSD-resident family", file=sys.stderr)
    raise SystemExit(1)
m = re.search(r"^[ \t]*#[ \t]*define[ \t]+" + re.escape(headroom) + r"[ \t]+(.+?)[ \t]*$",
              text, re.M)
if not m:
    print(f"engine-headroom-absent: {path} defines no {headroom}", file=sys.stderr)
    raise SystemExit(1)
expr = re.sub(r"(?i)(?<=[0-9])[uUlL]+", "", m.group(1).split("/*")[0].split("//")[0].strip())
allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Mult, ast.Add,
           ast.Sub, ast.LShift, ast.UAdd)
try:
    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, allowed):
            raise ValueError(type(node).__name__)
    value = eval(compile(tree, "<headroom>", "eval"))
except Exception as err:
    print(f"engine-headroom-unreadable: cannot read {headroom} from '{expr}' ({err})",
          file=sys.stderr)
    raise SystemExit(1)
if not isinstance(value, int) or value <= 0:
    print(f"engine-headroom-unreadable: {headroom} is {value!r}", file=sys.stderr)
    raise SystemExit(1)
print(value)
ENGINEPY
)" || die "memory plan REFUSES the load: the pinned engine does not declare the streamed n-gram table and its headroom (the refusal is named above). Nothing has been loaded."

  # Every shard of the family the first shard names, so a 4-shard body is
  # planned as 4 shards and not as one.
  shard_report="$(python3 - "${DS4_MODEL}" <<'PY'
import os, re, sys
first = sys.argv[1]
m = re.match(r"(.*)-(\d{5})-of-(\d{5})\.gguf$", os.path.basename(first))
d = os.path.dirname(first) or "."
if not m:
    print(1); print(os.path.getsize(first)); raise SystemExit
stem, total = m.group(1), int(m.group(3))
size = 0
for i in range(1, total + 1):
    p = os.path.join(d, f"{stem}-{i:05d}-of-{total:05d}.gguf")
    if not os.path.exists(p):
        print(f"missing shard {p}", file=sys.stderr); raise SystemExit(1)
    size += os.path.getsize(p)
print(total); print(size)
PY
)" || die "cannot plan the target's shard bytes (a shard of the pinned family is missing)"
  shard_count="$(printf '%s\n' "${shard_report}" | sed -n 1p)"
  body_bytes="$(printf '%s\n' "${shard_report}" | sed -n 2p)"

  # Which quantization the engine streams for the n-gram table, read from the
  # track fixture. The fixture is the pin, so the plan cannot drift from the
  # target it plans.
  ple_type="$(python3 - "${TRACK_FIXTURE}" <<'PY'
import json, sys
q = json.load(open(sys.argv[1])).get("target", {}).get("quantization", {})
t = q.get("ple_table")
if not isinstance(t, str) or not t:
    print("no target.quantization.ple_table", file=sys.stderr); raise SystemExit(1)
print(t)
PY
)" || die "memory plan REFUSES the load: ${TRACK_FIXTURE} declares no target.quantization.ple_table, so the streamed table cannot be identified. Nothing has been loaded."

  ple_bytes="$("${SCRIPT_DIR}/tools/gguf-tensor-bytes.py" \
      --first-shard "${DS4_MODEL}" --tensor "${PLE_TENSOR_NAME}" --allow-type "${ple_type}")" \
    || die "memory plan REFUSES the load: it cannot size ${PLE_TENSOR_NAME} as a streamed ${ple_type} table in the artifact's GGUF index (the refusal is named above). Nothing has been loaded."
  if (( ple_bytes <= 0 || ple_bytes >= body_bytes )); then
    die "memory plan REFUSES the load: the GGUF index sizes ${PLE_TENSOR_NAME} at ${ple_bytes} bytes against a ${body_bytes}-byte body, which cannot be right. Nothing has been loaded."
  fi
  resident_bytes=$(( body_bytes - ple_bytes ))

  # The head is resident only when this window declares one. The serial leg
  # arms no drafter, loads no head, and is charged for none.
  head_bytes=0
  if [[ -n "${DS4_MTP_PATH:-}" ]]; then head_bytes="$(wc -c < "${DS4_MTP_PATH}" | tr -d '[:space:]')"; fi

  session_bytes=$(( MEM_SESSION_GIB * 1024 * 1024 * 1024 ))
  want_bytes=$(( resident_bytes + head_bytes + session_bytes + headroom_bytes ))

  [[ -r "${SERVE_UP_MEMINFO}" ]] \
    || die "cannot read free memory from ${SERVE_UP_MEMINFO}; refusing to start a ${want_bytes}-byte load blind"
  avail_kb="$(awk '/^MemAvailable:/ {print $2; exit}' "${SERVE_UP_MEMINFO}")"
  [[ -n "${avail_kb}" ]] \
    || die "${SERVE_UP_MEMINFO} carries no MemAvailable line; refusing to start a ${want_bytes}-byte load blind"
  avail_bytes=$(( avail_kb * 1024 ))

  # The plan, as a table, BEFORE the fit check. A refusal is then read next to
  # the numbers that caused it.
  {
    printf 'serve-up.sh: memory plan for this window (GiB)\n'
    printf '  %-24s %9s\n'      "body (${shard_count} shards)" "$(gib "${body_bytes}")"
    printf '  %-24s %9s   %s\n' "n-gram table on SSD" "-$(gib "${ple_bytes}")" \
      "${PLE_TENSOR_NAME} (${ple_type}), mapped and streamed, never resident"
    printf '  %-24s %9s\n'      "resident weights"   "$(gib "${resident_bytes}")"
    printf '  %-24s %9s\n'      "MTP draft head"     "$(gib "${head_bytes}")"
    printf '  %-24s %9s\n'      "session/KV/scratch" "$(gib "${session_bytes}")"
    printf '  %-24s %9s   %s\n' "headroom"           "$(gib "${headroom_bytes}")" \
      "${ENGINE_HEADROOM_SYMBOL}, read from the pinned engine"
    printf '  %-24s %9s\n'      "required"           "$(gib "${want_bytes}")"
    printf '  %-24s %9s   %s\n' "available"          "$(gib "${avail_bytes}")" "${SERVE_UP_MEMINFO}"
  } >&2

  if (( avail_bytes < want_bytes )); then
    die "memory plan REFUSES the load: the resident needs $(gib "${want_bytes}") GiB (resident weights $(gib "${resident_bytes}") GiB + head $(gib "${head_bytes}") GiB + session ${MEM_SESSION_GIB} GiB + headroom $(gib "${headroom_bytes}") GiB) and ${SERVE_UP_MEMINFO} reports $(gib "${avail_bytes}") GiB available. Nothing has been loaded."
  fi
  log "memory plan ACCEPTS the load: $(gib "${want_bytes}") GiB required, $(gib "${avail_bytes}") GiB available; the body is resident exactly once and the n-gram table stays on SSD"
}

# --- teardown ----------------------------------------------------------------
# ALWAYS. The resident holds the GPU and ~103.7 GiB of host memory, so it never
# outlives this script: the window that booted it tears it down, on success and
# on failure alike. There is deliberately no path that leaves it up.
RESIDENT_PID=""
# --boot HANDS THE RESIDENT OVER. It is set only after the resident answers a
# healthy hello and the socket has been published, so every failure path before
# that still tears down. After it, the caller owns the resident and ends it with
# `--stop --socket <PATH>`.
KEEP_RESIDENT=0

teardown() {
  if [[ "${KEEP_RESIDENT}" == "1" ]]; then
    return 0
  fi
  if [[ -n "${RESIDENT_PID}" ]] && kill -0 "${RESIDENT_PID}" 2>/dev/null; then
    kill -TERM "${RESIDENT_PID}" 2>/dev/null || true
    for _ in $(seq 1 50); do
      kill -0 "${RESIDENT_PID}" 2>/dev/null || break
      sleep 0.2
    done
    if kill -0 "${RESIDENT_PID}" 2>/dev/null; then
      log "resident engine did not stop on SIGTERM; killing pid ${RESIDENT_PID}"
      kill -KILL "${RESIDENT_PID}" 2>/dev/null || true
    fi
    log "resident engine torn down; its log is ${RESIDENT_LOG}"
  fi
  # THE PID SIDECAR GOES WITH THE SOCKET. --boot writes <socket>.pid and hands
  # the resident over; if anything fails AFTER that write but before the handover
  # completes, this teardown runs and a stale sidecar would be left naming a dead
  # pid. A later --stop reads it, finds the pid gone, and cleans up -- but the
  # file itself would outlive every run that could remove it.
  rm -f "${SOCKET_PATH}" "${READY_FILE}" "${SOCKET_PATH}.pid" 2>/dev/null || true
}
trap teardown EXIT
trap 'exit 130' INT TERM

# --- boot the one resident ---------------------------------------------------
plan_memory
rm -f "${SOCKET_PATH}" "${READY_FILE}" 2>/dev/null || true
log "booting the resident engine: ${SERVE_UP_MODEL_FILE} (${SPEC_LABEL}); ONE load for the whole window"
"${SERVE_UP_RESIDENT_BIN}" >"${RESIDENT_LOG}" 2>&1 &
RESIDENT_PID=$!

start="$(date +%s)"
while :; do
  if ! kill -0 "${RESIDENT_PID}" 2>/dev/null; then
    log "the resident engine exited before it was healthy; its log:"
    tail -40 "${RESIDENT_LOG}" >&2 || true
    exit 1
  fi
  if [[ -f "${READY_FILE}" ]] && probe_resident "${SOCKET_PATH}" >/dev/null 2>&1; then
    break
  fi
  now="$(date +%s)"
  if (( now - start >= SERVE_UP_HEALTH_TIMEOUT_S )); then
    log "the resident engine was not healthy within ${SERVE_UP_HEALTH_TIMEOUT_S}s; its log:"
    tail -40 "${RESIDENT_LOG}" >&2 || true
    exit 1
  fi
  sleep 2
done
log "resident engine healthy on ${SOCKET_PATH} after $(( $(date +%s) - start ))s; every phase now connects instead of loading"

HELLO_JSON="$(probe_resident "${SOCKET_PATH}")" || die "the resident engine stopped answering before the run started"

# --- the window's identity ---------------------------------------------------
python3 - "${IDENTITY_FILE}" <<PY
import json, sys
json.dump({
    "engine": "ds4",
    "weight_owner": "ds4-resident",
    "resident_socket": "${SOCKET_PATH}",
    "resident_pid": "${RESIDENT_PID}",
    "identity_key": "${SERVE_IDENTITY_KEY}",
    "engine_pin": "${engine_pin:-unknown}",
    "nvcc_version": "${nvcc_ver:-unknown}",
    "driver_version": "${driver_ver:-unknown}",
    "engine_ident": "${DS4_ENGINE_IDENT}",
    "model_file": "${SERVE_UP_MODEL_FILE}",
    "mtp_head_file": "${DS4_MTP_PATH:-}",
    "weights_dir": "${SERVE_UP_WEIGHTS_DIR}",
    "spec_config": "${SPEC_LABEL}",
    "declared_draft_len": ${SERVE_UP_SPEC_DRAFT_LEN:-0},
    "mtp_draft_tokens": ${DS4_MTP_DRAFT_TOKENS},
    "ctx_size": ${SERVE_UP_CTX_SIZE},
    "hello": json.loads('''${HELLO_JSON}'''),
    "started": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
}, open(sys.argv[1], "w"), indent=2)
PY

export SERVE_IDENTITY_FILE="${IDENTITY_FILE}"
export SERVE_IDENTITY_WEIGHT_OWNER="ds4-resident"
export SERVE_IDENTITY_SPEC_CONFIG="${SPEC_LABEL}"
export SERVE_IDENTITY_SPEC_SPECULATIVE="${SERVE_UP_SPECULATIVE}"
export SERVE_IDENTITY_SPEC_DRAFT_LEN="${SERVE_UP_SPEC_DRAFT_LEN:-0}"
export SERVE_IDENTITY_RESIDENT_PID="${RESIDENT_PID}"

# --- --boot: publish the socket and hand the resident over -------------------
# The socket path is the FIRST LINE of the file the caller named, so a caller
# reads it with one `head -1`. The sidecar beside the socket carries what
# --stop needs and nothing else: the pid on line 1, the ready file on line 2.
#
# THE ORDER MATTERS. The sidecar is written BEFORE the socket file is published
# and before this script exits, so there is no window in which the caller holds
# a socket path it cannot stop.
if [[ "${MODE}" == "boot" ]]; then
  printf '%s\n%s\n' "${RESIDENT_PID}" "${READY_FILE}" > "${SOCKET_PATH}.pid"
  socket_out_dir="$(dirname -- "${BOOT_SOCKET_OUT}")"
  [[ "${socket_out_dir}" == "." ]] || mkdir -p "${socket_out_dir}"
  printf '%s\n' "${SOCKET_PATH}" > "${BOOT_SOCKET_OUT}"
  KEEP_RESIDENT=1
  log "resident handed over on ${SOCKET_PATH} (${SPEC_LABEL}); stop it with: tools/serve-up.sh --stop --socket ${SOCKET_PATH}"
  exit 0
fi

log "running: $*"
set +e
"$@"
rc=$?
set -e
exit "${rc}"
