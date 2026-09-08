#!/usr/bin/env bash
# test-serve-up-plan-memory.sh -- prove the memory plan in tools/serve-up.sh.
#
# THE CLAIM UNDER TEST. The resident set is NOT the whole artifact body. The
# per-layer n-gram table (per_layer_token_embd.weight) stays mapped on the
# solid-state disk and is streamed, so the plan subtracts it. Charging it
# refuses a boot that fits: on a 119 GiB Spark the whole body plus the head
# wants ~114 GiB, and the true resident need is ~93 GiB.
#
# WHAT THIS PROVES
#   1. the subtraction, to the kilobyte: the plan ACCEPTS exactly the resident
#      need and REFUSES one kilobyte below it. Without the subtraction the
#      accept case would refuse.
#   2. the plan table names the streamed table and its bytes
#   3. an absent n-gram table REFUSES by name
#   4. an n-gram table of a type the engine does not stream REFUSES by name
#   5. the refusal has NO environment bypass
#   6. the headroom is READ from the pinned engine, not restated here
#   7. an engine that does not declare the streamed table REFUSES by name, so
#      the subtraction can never lower the gate under an engine that would load
#      the whole body
#   8. a context above what the session budget covers REFUSES by name
#
# The artifacts are synthetic GGUF indexes (tools/ds4/synthetic-window.sh),
# so this reads the REAL tool's REAL index reader with no weights, no GPU and
# no network. Hermetic: bash, awk and python3.
#
# Usage: tools/test-serve-up-plan-memory.sh
# Exit:  0 all cases pass, 1 a case failed (printed with a FAIL prefix)
set -uo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null && pwd -P)"
SERVE_UP="${ROOT_DIR}/tools/serve-up.sh"
GIB=$((1024 * 1024 * 1024))

# The synthetic artifact and the synthetic engine declaration. One writer, so
# this test and the other off-box drivers cannot drift apart.
# shellcheck source=tools/ds4/synthetic-window.sh
. "${ROOT_DIR}/tools/ds4/synthetic-window.sh"

# The session budget the plan states, and the context it covers. This test
# restates them on purpose: if either moves, the boundary cases below go red.
SESSION_GIB=4
SESSION_CTX_CEILING=8192

# The headroom is NOT restated. The plan reads it from the pinned engine's own
# header, and this test declares that header, so case 6 can move the number and
# watch the plan move with it.
HEADROOM_GIB=10

WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

fails=0
pass() { printf 'test-serve-up-plan-memory: PASS -- %s\n' "$*"; }
fail() { printf 'test-serve-up-plan-memory: FAIL -- %s\n' "$*" >&2; fails=$((fails + 1)); }

# The engine's memory declaration. The plan refuses unless the pinned engine
# says it charges the n-gram table to an SSD-resident family, and it takes the
# headroom from this file.
ENGINE_HEADER="${WORK}/ds4_qwen4exp.h"
synthetic_window_engine_header "${ENGINE_HEADER}" "${HEADROOM_GIB}"

# A resident binary that exits at once. Every case here stops at or before the
# memory plan, which runs BEFORE the boot, so no case needs a healthy resident.
RESIDENT_STUB="${WORK}/ds4-resident-stub"
printf '#!/bin/sh\nexit 0\n' > "${RESIDENT_STUB}"
chmod +x "${RESIDENT_STUB}"

# The body and the head come from synthetic_window_weights, and its shape is
# what the byte arithmetic below reads.
SHARD_BYTES="${SYNTHETIC_WINDOW_SHARD_BYTES}"
HEAD_BYTES="${SYNTHETIC_WINDOW_HEAD_BYTES}"

# meminfo FILE BYTES -- a /proc/meminfo whose MemAvailable covers BYTES.
meminfo() {
  printf 'MemTotal:       %s kB\nMemAvailable:   %s kB\n' \
    "$(( $2 / 1024 ))" "$(( $2 / 1024 ))" > "$1"
}

# run_serve MEMINFO WEIGHTS [EXTRA_ENV...] -- drive the REAL serve-up.sh on the
# speculative leg (so the head is charged) and print its combined output.
run_serve() {
  local meminfo_file="$1" weights="$2"; shift 2
  env SERVE_UP_ENGINE_HEADER="${ENGINE_HEADER}" "$@" \
    SERVE_UP_WEIGHTS_DIR="${weights}" \
    SERVE_UP_LOG_DIR="${WORK}/logdir" \
    SERVE_UP_RESIDENT_BIN="${RESIDENT_STUB}" \
    SERVE_UP_SPECULATIVE=1 \
    SERVE_UP_SPEC_DRAFT_LEN=1 \
    SERVE_UP_HEALTH_TIMEOUT_S=5 \
    SERVE_UP_MEMINFO="${meminfo_file}" \
    "${SERVE_UP}" true 2>&1
}

# --- the artifact whose numbers this test knows ------------------------------
GOOD="${WORK}/good"
PLE_BYTES="${SYNTHETIC_WINDOW_PLE_BYTES}"
synthetic_window_weights "${GOOD}" \
  || { fail "cannot write the synthetic artifact"; exit 1; }

BODY_BYTES=$(( SHARD_BYTES * 2 ))
RESIDENT_BYTES=$(( BODY_BYTES - PLE_BYTES ))
REQUIRED_BYTES=$(( RESIDENT_BYTES + HEAD_BYTES + SESSION_GIB * GIB + HEADROOM_GIB * GIB ))
WHOLE_BODY_BYTES=$(( BODY_BYTES + HEAD_BYTES + SESSION_GIB * GIB + HEADROOM_GIB * GIB ))

# --- 1a. the plan ACCEPTS exactly the resident need --------------------------
# MemAvailable is set to the required bytes and not one byte more. The plan
# must pass its fit check. It then fails on the stub resident, which is the
# proof that the plan let the boot start.
meminfo "${WORK}/meminfo.exact" "${REQUIRED_BYTES}"
out="$(run_serve "${WORK}/meminfo.exact" "${GOOD}")"
if [[ "${out}" == *"memory plan REFUSES the load"* ]]; then
  fail "the plan refused a box that has exactly the resident need: ${out}"
elif [[ "${out}" != *"memory plan ACCEPTS the load"* ]]; then
  fail "the plan neither accepted nor refused: ${out}"
else
  pass "the plan accepts a box holding exactly the resident need ($(( REQUIRED_BYTES / 1024 )) kB)"
fi

# --- 1b. one kilobyte short REFUSES ------------------------------------------
# 1a and 1b together pin the required bytes to the kilobyte, so the subtraction
# is proven by value and not by reading the printed table.
meminfo "${WORK}/meminfo.short" $(( REQUIRED_BYTES - 1024 ))
out="$(run_serve "${WORK}/meminfo.short" "${GOOD}")"
if [[ "${out}" == *"memory plan REFUSES the load"* && "${out}" == *"Nothing has been loaded"* ]]; then
  pass "one kilobyte below the resident need refuses, and says nothing was loaded"
else
  fail "one kilobyte below the resident need did not refuse: ${out}"
fi

# --- 1c. the OLD plan's number is not the one enforced -----------------------
# A box that holds the whole body would also have passed before this lane. The
# case that matters is the box that holds the RESIDENT need and not the body:
# 1a already proves it accepts. Here the whole-body figure is asserted to be
# strictly larger, so the two numbers can never collapse into one.
if (( WHOLE_BODY_BYTES > REQUIRED_BYTES )); then
  pass "charging the whole body would want $(( (WHOLE_BODY_BYTES - REQUIRED_BYTES) / 1024 )) kB more than the resident need"
else
  fail "the synthetic artifact has no n-gram table to subtract; the test proves nothing"
fi

# --- 2. the plan is printed as a table ---------------------------------------
meminfo "${WORK}/meminfo.plenty" $(( REQUIRED_BYTES * 2 ))
out="$(run_serve "${WORK}/meminfo.plenty" "${GOOD}")"
want_ple_gib="$(awk -v b="${PLE_BYTES}" 'BEGIN{printf "%.2f", b/1073741824}')"
if [[ "${out}" == *"memory plan for this window (GiB)"* \
   && "${out}" == *"n-gram table on SSD"* \
   && "${out}" == *"-${want_ple_gib}"* \
   && "${out}" == *"${SYNTHETIC_WINDOW_PLE_NAME} (${SYNTHETIC_WINDOW_PLE_TYPE})"* \
   && "${out}" == *"resident weights"* \
   && "${out}" == *"required"* ]]; then
  pass "the plan prints a table naming the streamed tensor, its type and its bytes"
else
  fail "the plan table is missing rows: ${out}"
fi

# --- 3. an absent n-gram table REFUSES by name -------------------------------
NO_PLE="${WORK}/no-ple"
synthetic_window_weights "${NO_PLE}" "none" || fail "cannot write the artifact without an n-gram table"
out="$(run_serve "${WORK}/meminfo.plenty" "${NO_PLE}")"
if [[ "${out}" == *"tensor-absent"* && "${out}" == *"per_layer_token_embd.weight"* \
   && "${out}" == *"memory plan REFUSES the load"* ]]; then
  pass "an artifact with no per_layer_token_embd.weight refuses by name"
else
  fail "a missing n-gram table did not refuse by name: ${out}"
fi

# --- 4. a table the engine does not stream REFUSES by name -------------------
# The fixture declares the streamed quantization (target.quantization.ple_table).
# A re-quantized table is a different residency story, so it must stop the boot
# rather than shrink the plan by a number that no longer holds.
WRONG_TYPE="${WORK}/wrong-type"
synthetic_window_weights "${WRONG_TYPE}" "${SYNTHETIC_WINDOW_PLE_NAME}:Q8_0:${SYNTHETIC_WINDOW_PLE_DIMS}" \
  || fail "cannot write the artifact with a re-quantized n-gram table"
out="$(run_serve "${WORK}/meminfo.plenty" "${WRONG_TYPE}")"
if [[ "${out}" == *"tensor-type-not-allowed"* && "${out}" == *"is Q8_0"* \
   && "${out}" == *"memory plan REFUSES the load"* ]]; then
  pass "an n-gram table the engine does not stream refuses by name"
else
  fail "a re-quantized n-gram table did not refuse by name: ${out}"
fi

# --- 5. the refusal has no environment bypass --------------------------------
# The headroom and the session budget are constants in the script. Setting the
# variable the old plan read must change nothing.
out="$(run_serve "${WORK}/meminfo.short" "${GOOD}" SERVE_UP_MEM_HEADROOM_GB=0)"
if [[ "${out}" == *"memory plan REFUSES the load"* ]]; then
  pass "SERVE_UP_MEM_HEADROOM_GB=0 does not relax the refusal"
else
  fail "an environment variable relaxed the memory refusal: ${out}"
fi

# --- 6. the headroom is read FROM the engine ---------------------------------
# The same artifact and a header declaring 12 GiB instead of 10 must want
# exactly 2 GiB more. That proves the number is read and not restated.
BIGGER_HEADER="${WORK}/ds4_qwen4exp_12gib.h"
synthetic_window_engine_header "${BIGGER_HEADER}" $(( HEADROOM_GIB + 2 ))
meminfo "${WORK}/meminfo.exact" "${REQUIRED_BYTES}"
out="$(ENGINE_HEADER="${BIGGER_HEADER}" run_serve "${WORK}/meminfo.exact" "${GOOD}")"
if [[ "${out}" != *"memory plan REFUSES the load"* ]]; then
  fail "a 12 GiB engine headroom did not raise the need: ${out}"
else
  meminfo "${WORK}/meminfo.plus2" $(( REQUIRED_BYTES + 2 * GIB ))
  out="$(ENGINE_HEADER="${BIGGER_HEADER}" run_serve "${WORK}/meminfo.plus2" "${GOOD}")"
  if [[ "${out}" == *"memory plan ACCEPTS the load"* ]]; then
    pass "the headroom is read from the engine: 10 -> 12 GiB moves the need by exactly 2 GiB"
  else
    fail "a 12 GiB engine headroom wanted more than 2 GiB extra: ${out}"
  fi
fi

# --- 7. an engine that does not declare the streamed table REFUSES -----------
# THIS IS THE ONE THAT KEEPS THE PLAN HONEST ON ANY TREE. The subtraction lowers
# the gate, so it is only allowed against an engine that really streams the
# table. Upstream ds4 has no qwen4exp family: against that pin the plan must
# refuse, not subtract.
out="$(ENGINE_HEADER="${WORK}/no-such-header.h" run_serve "${WORK}/meminfo.plenty" "${GOOD}")"
if [[ "${out}" == *"engine-declaration-absent"* && "${out}" == *"memory plan REFUSES the load"* ]]; then
  pass "an engine with no qwen4exp memory declaration refuses by name"
else
  fail "a pin without the qwen4exp declaration did not refuse: ${out}"
fi

UPSTREAM_HEADER="${WORK}/upstream.h"
printf '#define DS4_QWEN4EXP_MEMORY_HEADROOM_BYTES (10ull * 1024ull * 1024ull * 1024ull)\n' \
  > "${UPSTREAM_HEADER}"
out="$(ENGINE_HEADER="${UPSTREAM_HEADER}" run_serve "${WORK}/meminfo.plenty" "${GOOD}")"
if [[ "${out}" == *"engine-charge-absent"* && "${out}" == *"memory plan REFUSES the load"* ]]; then
  pass "an engine that declares no DS4_QWEN4EXP_MEM_PLE charge refuses by name"
else
  fail "an engine that does not charge the table to an SSD family did not refuse: ${out}"
fi

NO_HEADROOM_HEADER="${WORK}/no-headroom.h"
printf 'typedef enum { DS4_QWEN4EXP_MEM_PLE } c;\n' > "${NO_HEADROOM_HEADER}"
out="$(ENGINE_HEADER="${NO_HEADROOM_HEADER}" run_serve "${WORK}/meminfo.plenty" "${GOOD}")"
if [[ "${out}" == *"engine-headroom-absent"* && "${out}" == *"memory plan REFUSES the load"* ]]; then
  pass "an engine that declares no headroom constant refuses by name"
else
  fail "a missing headroom constant did not refuse: ${out}"
fi

# --- 8. a context the session budget does not cover REFUSES ------------------
# SERVE_UP_CTX_SIZE feeds DS4_CTX_SIZE, so a longer context must not be charged
# the same session budget.
out="$(run_serve "${WORK}/meminfo.plenty" "${GOOD}" SERVE_UP_CTX_SIZE=$(( SESSION_CTX_CEILING + 1 )))"
if [[ "${out}" == *"memory plan REFUSES the load"* && "${out}" == *"session budget"* ]]; then
  pass "a context above ${SESSION_CTX_CEILING} tokens refuses instead of being under-charged"
else
  fail "a context above the session ceiling was planned anyway: ${out}"
fi
out="$(run_serve "${WORK}/meminfo.plenty" "${GOOD}" SERVE_UP_CTX_SIZE="${SESSION_CTX_CEILING}")"
if [[ "${out}" == *"memory plan ACCEPTS the load"* ]]; then
  pass "a context at the ceiling is planned"
else
  fail "a context at the ceiling was refused: ${out}"
fi

if [[ ${fails} -eq 0 ]]; then
  echo "OK: all serve-up memory-plan cases passed"
  exit 0
fi
echo "FAILED: ${fails} case(s)"
exit 1
