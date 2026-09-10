#!/bin/sh
#
# Host parity proof for the Qwen4-Exp CUDA expert dequantisers.
#
# tests/test_qwen4exp_moe runs the Metal expert GEMM against a double-precision
# reference.  The CUDA twin in ds4_cuda_qwen4exp.cu has no such check on a Mac,
# and a CUDA host is not always available, so this script takes the K-quant
# accessors OUT of that file as they stand -- not a copy of them -- compiles
# them for the host with the __device__ qualifiers defined away, and compares
# them element by element with the same reference.
#
# Extracting rather than copying is the point: if someone edits a Q5_K or Q6_K
# accessor in ds4_cuda_qwen4exp.cu, this runs the edited body.  If the function
# is renamed or moved, the extraction comes up empty and the script fails.
#
# Run it with: make test-qwen4exp-cuda-moe

set -eu

ROOT=$(cd "$(dirname "$0")/.." && pwd)
SRC="$ROOT/ds4_cuda_qwen4exp.cu"
TEST="$ROOT/tests/test_qwen4exp_cuda_moe.c"
CC=${CC:-cc}

if [ ! -f "$SRC" ]; then
    echo "qwen4exp CUDA MoE parity: $SRC is missing" >&2
    exit 1
fi

WORK=$(mktemp -d) || exit 1
trap 'rm -rf "$WORK"' EXIT
OUT="$WORK/bodies.inc"
: >"$OUT"

# One struct: everything from the typedef that ends in the named terminator.
extract_struct() {
    awk -v name="$1" '
        /^typedef struct \{/ { buf = "" }
        { buf = buf $0 "\n" }
        $0 == "} " name ";" { printf "%s", buf; found = 1; exit }
        END { if (!found) exit 1 }
    ' "$SRC" >>"$OUT" || {
        echo "qwen4exp CUDA MoE parity: $1 not found in ds4_cuda_qwen4exp.cu" >&2
        exit 1
    }
    printf '\n' >>"$OUT"
}

# One function: the signature line through the closing brace in column one.
extract_fn() {
    awk -v sig="$1" '
        index($0, sig) == 1 { on = 1 }
        on { print; found = 1 }
        on && $0 == "}" { exit }
        END { if (!found) exit 1 }
    ' "$SRC" >>"$OUT" || {
        echo "qwen4exp CUDA MoE parity: '$1' not found in ds4_cuda_qwen4exp.cu" >&2
        exit 1
    }
    printf '\n' >>"$OUT"
}

extract_struct cuda_block_q5_K
extract_struct cuda_block_q6_K
extract_fn '__device__ static void dev_q4_K_get_scale_min('
extract_fn '__device__ __forceinline__ static float dev_qwen4exp_q5_K_value('
extract_fn '__device__ __forceinline__ static float dev_qwen4exp_q6_K_value('

BIN="$WORK/test_qwen4exp_cuda_moe"
$CC -O2 -Wall -Wextra -Werror -std=c99 -I"$WORK" -o "$BIN" "$TEST" -lm
"$BIN"
