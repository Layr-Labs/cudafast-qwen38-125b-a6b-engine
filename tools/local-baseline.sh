#!/usr/bin/env bash
# Run the public local fixture through the normal benchmark entry point.
# benchd continues to own correctness, cooling, timing, and sealed artifacts.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: tools/local-baseline.sh

Run the public Qwen CUDA local baseline after ./setup.sh. No organizer goldens,
reference workspace, or calibration file are required. This measures local
correctness and timing on one leg; it does not produce a ranked score.

Optional environment overrides (relative paths resolve from the checkout):
  MLXFAST_ENGINE_BIN                .build/release/mlxfast-runtime-worker
  MLXFAST_CORRECTNESS_GOLDEN_PATH    correctness_prompts/public-longcopy-gate-english-1024.golden.json
  MLXFAST_WEIGHTS_PATH              $MLXFAST_TARGET_SNAPSHOT_DIR, else
                                    reference_weights/Qwen3.8-Flash-Next-GGUF
  MLXFAST_SCORE_PATH                score.local-iterate.json

The existing benchmark entry point verifies benchd, boots one resident engine
through tools/serve-up.sh, and enables the cool gate. It runs one leg: the
paired ranked path (tools/qwen38-125b-a6b-measure-and-score.sh) is not involved.
USAGE
}

if [[ $# -gt 0 ]]; then
  if [[ $# -eq 1 && ( "$1" == "--help" || "$1" == "-h" ) ]]; then
    usage
    exit 0
  fi
  usage >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${REPO_ROOT}"

export MLXFAST_ENGINE_BIN="${MLXFAST_ENGINE_BIN:-.build/release/mlxfast-runtime-worker}"
export MLXFAST_CORRECTNESS_GOLDEN_PATH="${MLXFAST_CORRECTNESS_GOLDEN_PATH:-correctness_prompts/public-longcopy-gate-english-1024.golden.json}"
export MLXFAST_WEIGHTS_PATH="${MLXFAST_WEIGHTS_PATH:-${MLXFAST_TARGET_SNAPSHOT_DIR:-reference_weights/Qwen3.8-Flash-Next-GGUF}}"
export MLXFAST_SCORE_PATH="${MLXFAST_SCORE_PATH:-score.local-iterate.json}"

if [[ ! -x "${MLXFAST_ENGINE_BIN}" ]]; then
  echo "local-baseline.sh: engine not executable: ${MLXFAST_ENGINE_BIN}" >&2
  echo "local-baseline.sh: run ./setup.sh to build and stage the engine, or set MLXFAST_ENGINE_BIN" >&2
  exit 1
fi
if [[ ! -d "${MLXFAST_WEIGHTS_PATH}" ]]; then
  echo "local-baseline.sh: target snapshot directory not found: ${MLXFAST_WEIGHTS_PATH}" >&2
  echo "local-baseline.sh: run ./setup.sh to download and verify the snapshot, or set MLXFAST_TARGET_SNAPSHOT_DIR" >&2
  exit 1
fi

{
  echo "local-baseline.sh: public local baseline (unranked)"
  echo "local-baseline.sh: golden=${MLXFAST_CORRECTNESS_GOLDEN_PATH}"
  echo "local-baseline.sh: weights=${MLXFAST_WEIGHTS_PATH}"
  echo "local-baseline.sh: result=${MLXFAST_SCORE_PATH}; score=null is expected without ranked paired scoring"
} >&2

exec "${REPO_ROOT}/benchmark.sh" --local-iterate
