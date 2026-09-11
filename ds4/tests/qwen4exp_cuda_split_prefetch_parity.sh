#!/bin/sh
# Host parity proof for the one-buffer Q4_K split-kernel payload prefetch.

set -eu

ROOT=$(cd "$(dirname "$0")/.." && pwd)
SRC="$ROOT/ds4_cuda_qwen4exp.cu"
TEST="$ROOT/tests/test_qwen4exp_cuda_split_prefetch.c"
CC=${CC:-cc}

WORK=$(mktemp -d) || exit 1
trap 'rm -rf "$WORK"' EXIT
OUT="$WORK/bodies.inc"
: >"$OUT"

extract_struct() {
    awk -v name="$1" '
        /^typedef struct \{/ { buf = "" }
        { buf = buf $0 "\n" }
        $0 == "} " name ";" { printf "%s", buf; found = 1; exit }
        END { if (!found) exit 1 }
    ' "$SRC" >>"$OUT" || {
        echo "qwen4exp split prefetch parity: $1 not found" >&2
        exit 1
    }
    printf '\n' >>"$OUT"
}

extract_fn() {
    awk -v sig="$1" '
        index($0, sig) == 1 { on = 1 }
        on { print; found = 1 }
        on && $0 == "}" { exit }
        END { if (!found) exit 1 }
    ' "$SRC" >>"$OUT" || {
        echo "qwen4exp split prefetch parity: $1 not found" >&2
        exit 1
    }
    printf '\n' >>"$OUT"
}

extract_struct cuda_block_q4_K
extract_struct cuda_block_q5_1
extract_struct cuda_block_q5_K
extract_struct cuda_block_q6_K
extract_fn '__device__ static void dev_q4_K_get_scale_min('
extract_fn '__device__ __forceinline__ static bool qwen4exp_word_aligned('
extract_fn '__device__ __forceinline__ static void qw_load_words8('
extract_fn '__device__ __forceinline__ static void dev_qwen4exp_group_decode('
extract_fn '__device__ __forceinline__ static uint32_t qw_pack4('
extract_fn '__device__ __forceinline__ static void qw_tile_store_group('
extract_fn '__device__ __forceinline__ static bool qw_raw_load('
extract_fn '__device__ __forceinline__ static void dev_qwen4exp_group_decode_w('

BIN="$WORK/test_qwen4exp_cuda_split_prefetch"
$CC -O2 -Wall -Wextra -Werror -Wno-unknown-pragmas -std=c99 \
    -I"$WORK" -o "$BIN" "$TEST"
"$BIN"
