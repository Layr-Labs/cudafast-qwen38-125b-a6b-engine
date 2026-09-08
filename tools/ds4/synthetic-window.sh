#!/usr/bin/env bash
# shellcheck shell=bash
#
# synthetic-window.sh -- the synthetic window the off-box tools drive.
#
# TEST SUPPORT ONLY. Nothing here is staged, scored, or linked into a measured
# run. It writes the two things tools/serve-up.sh reads BEFORE it boots
# anything, so a host with no artifact, no GPU and no ds4 checkout can drive the
# REAL memory plan:
#
#   1. THE ARTIFACT. A 2-shard GGUF body plus a draft head. The shards carry a
#      REAL GGUF tensor index, because the plan sizes the streamed n-gram table
#      (per_layer_token_embd.weight) out of that index through
#      tools/gguf-tensor-bytes.py. The data section is padding: every reader in
#      this repository reads the index only.
#   2. THE ENGINE DECLARATION. A ds4_qwen4exp.h that declares
#      DS4_QWEN4EXP_MEM_PLE and DS4_QWEN4EXP_MEMORY_HEADROOM_BYTES. The plan
#      refuses unless the pinned engine says it charges the n-gram table to an
#      SSD-resident family, and it reads the headroom from that header.
#
# THIS FILE IS THE ONE PLACE THEY ARE WRITTEN. tools/test-serve-up-plan-memory.sh,
# tools/test-ds4-resident.sh and tools/three-flows-dry-run.sh all source it, so a
# change to the plan's inputs moves one file and every off-box driver moves with
# it.
#
# Usage: source this file. It is not executable on its own.
#   . "<repo>/tools/ds4/synthetic-window.sh"
#   synthetic_window_weights "${WORK}/weights"
#   synthetic_window_engine_header "${WORK}/ds4_qwen4exp.h"

SYNTHETIC_WINDOW_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null && pwd -P)"
SYNTHETIC_WINDOW_MAKE_GGUF="${SYNTHETIC_WINDOW_DIR}/make-synthetic-gguf.py"

# The shape of the synthetic artifact. A caller that asserts on the plan's
# numbers reads these instead of restating them.
SYNTHETIC_WINDOW_SHARD_BYTES=$((1024 * 1024))
SYNTHETIC_WINDOW_HEAD_BYTES=4096
# The streamed n-gram table. The type is the one the track fixture declares as
# target.quantization.ple_table; the plan refuses any other type by name.
SYNTHETIC_WINDOW_PLE_NAME="per_layer_token_embd.weight"
SYNTHETIC_WINDOW_PLE_TYPE="IQ4_NL"
SYNTHETIC_WINDOW_PLE_DIMS="512x1024"
SYNTHETIC_WINDOW_PLE_TENSOR="${SYNTHETIC_WINDOW_PLE_NAME}:${SYNTHETIC_WINDOW_PLE_TYPE}:${SYNTHETIC_WINDOW_PLE_DIMS}"
# The bytes that index entry sizes: 512 x 1024 IQ4_NL elements are
# 524288 / 32 blocks of 18 bytes. Nothing in this file reads it; the caller that
# asserts on the plan's numbers does.
# shellcheck disable=SC2034  # read by the scripts that source this file
SYNTHETIC_WINDOW_PLE_BYTES=$(( 512 * 1024 * 18 / 32 ))
# The default headroom the declaration states, in GiB.
SYNTHETIC_WINDOW_HEADROOM_GIB=10

# synthetic_window_engine_header FILE [HEADROOM_GIB]
#
# Write the pinned engine's memory declaration. The default headroom is
# SYNTHETIC_WINDOW_HEADROOM_GIB; a caller that proves the plan READS the number
# passes a different one.
synthetic_window_engine_header() {
  local out="$1" headroom="${2:-${SYNTHETIC_WINDOW_HEADROOM_GIB}}" dir
  dir="$(dirname -- "${out}")"
  mkdir -p "${dir}" || return 1
  cat > "${out}" <<HEADER
/* synthetic ds4_qwen4exp.h, written by tools/ds4/synthetic-window.sh */
typedef enum {
    DS4_QWEN4EXP_MEM_WEIGHTS = 0,
    DS4_QWEN4EXP_MEM_PLE,
    DS4_QWEN4EXP_MEM__COUNT
} ds4_qwen4exp_mem_class;
#define DS4_QWEN4EXP_MEMORY_HEADROOM_BYTES (${headroom}ull * 1024ull * 1024ull * 1024ull)
HEADER
}

# synthetic_window_weights DIR [PLE_TENSOR]
#
# Write the 2-shard body and the draft head. PLE_TENSOR is a --tensor argument
# for the FIRST shard, and defaults to SYNTHETIC_WINDOW_PLE_TENSOR. Two other
# values a caller uses on purpose:
#   none            a body with NO n-gram table, so the plan refuses by name
#   NAME:TYPE:DIMS  a table of another type, so the plan refuses by name
synthetic_window_weights() {
  local dir="$1" tensor="${2:-${SYNTHETIC_WINDOW_PLE_TENSOR}}"
  if [[ "${tensor}" == "none" ]]; then
    tensor="token_embd.weight:Q8_0:512x1024"
  fi
  mkdir -p "${dir}" || return 1
  "${SYNTHETIC_WINDOW_MAKE_GGUF}" --out "${dir}/target-00001-of-00002.gguf" \
    --tensor "${tensor}" --size-bytes "${SYNTHETIC_WINDOW_SHARD_BYTES}" || return 1
  "${SYNTHETIC_WINDOW_MAKE_GGUF}" --out "${dir}/target-00002-of-00002.gguf" \
    --tensor "blk.0.attn_q.weight:Q4_K:2560x2560" \
    --size-bytes "${SYNTHETIC_WINDOW_SHARD_BYTES}" || return 1
  head -c "${SYNTHETIC_WINDOW_HEAD_BYTES}" /dev/zero > "${dir}/mtp-head.gguf" || return 1
}
