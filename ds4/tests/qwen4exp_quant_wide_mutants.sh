#!/bin/sh
#
# Mutation proof for the eight-warp activation quantiser (prefill widths).
#
# tests/test_qwen4exp_moe requires the eight-warp quantiser to agree with the
# one-warp quantiser BYTE FOR BYTE through the MoE input-reuse chain at 64,
# 65 and 1024 rows and three magnitudes.  This script breaks the wide kernel one
# line at a time and requires that check to notice.  Each mutant rebuilds one
# object plus a relink.  Run it with: make test-qwen4exp-quant-wide-mutants
#
# The last mutant is a deliberate no-op (the tail clamp spelled with min())
# and must SURVIVE.

set -u

ROOT=$(cd "$(dirname "$0")/.." && pwd)
TEST="$ROOT/tests/test_qwen4exp_moe"
SRC="$ROOT/ds4_cuda_qwen4exp.cu"

if [ ! -f "$SRC" ]; then
    echo "qwen4exp quant wide mutants: $SRC is missing" >&2
    exit 1
fi
if ! command -v nvcc >/dev/null 2>&1 && [ ! -x "${CUDA_HOME:-/usr/local/cuda}/bin/nvcc" ]; then
    echo "qwen4exp quant wide mutants: skipped, no CUDA toolchain"
    exit 0
fi

cd "$ROOT" || exit 1
WORK=$(mktemp -d) || exit 1
cp "$SRC" "$WORK/orig.cu" || exit 1
restore() {
    cp "$WORK/orig.cu" "$SRC"
    rm -rf "$WORK"
}
trap 'restore' EXIT INT TERM

build() {
    make CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}" \
        CUDA_ARCH="${DS4_CUDA_ARCH:-sm_121}" \
        tests/test_qwen4exp_moe >"$WORK/build.log" 2>&1
}

if ! build; then
    echo "qwen4exp quant wide mutants: the unmutated tree does not build" >&2
    tail -n 20 "$WORK/build.log" >&2
    exit 1
fi
if "$TEST" >"$WORK/base.log" 2>&1; then
    echo "  baseline                           PASS (as required)"
else
    echo "  baseline                           FAIL -- the unmutated wide quantiser does not agree" >&2
    tail -n 3 "$WORK/base.log" >&2
    exit 1
fi

failures=0

# mutant NAME EXPECT(caught|survives) SED_EXPRESSION
mutant() {
    name=$1
    expect=$2
    expr=$3
    sed "$expr" "$WORK/orig.cu" >"$WORK/m.cu" || exit 1
    if cmp -s "$WORK/orig.cu" "$WORK/m.cu"; then
        echo "  $name: the mutation did not apply, the kernel source moved" >&2
        failures=$((failures + 1))
        return
    fi
    cp "$WORK/m.cu" "$SRC"
    if ! build; then
        echo "  $name: the mutant does not compile" >&2
        tail -n 10 "$WORK/build.log" >&2
        failures=$((failures + 1))
        cp "$WORK/orig.cu" "$SRC"
        return
    fi
    if "$TEST" >"$WORK/$name.log" 2>&1; then
        got=survives
    else
        got=caught
    fi
    cp "$WORK/orig.cu" "$SRC"
    if [ "$got" = "$expect" ]; then
        if [ "$got" = caught ]; then
            printf '  %-34s caught: %s\n' "$name" \
                "$(grep -E 'mismatch|error|failed' "$WORK/$name.log" | head -n 1)"
        else
            printf '  %-34s survives, as expected\n' "$name"
        fi
    elif [ "$expect" = caught ]; then
        printf '  %-34s NOT CAUGHT -- the test passed a broken kernel\n' "$name" >&2
        failures=$((failures + 1))
    else
        printf '  %-34s CAUGHT -- expected an exact no-op\n' "$name" >&2
        failures=$((failures + 1))
    fi
}

# The warp's lane taken from the block-wide thread index: warps 1..7 see
# lanes >= 32 and quantise nothing.
mutant lane_not_masked caught \
    's|    dev_qwen4exp_quantize_group(xq, xscale, xsum, xr, threadIdx.x \& 31u, n,|    dev_qwen4exp_quantize_group(xq, xscale, xsum, xr, threadIdx.x, n,|'

# Seven groups per block instead of eight: the last group of each block is
# never quantised.
mutant seven_groups_per_block caught \
    's|    const uint32_t g = blockIdx.x \* 8u + (threadIdx.x >> 5u);|    const uint32_t g = blockIdx.x * 7u + (threadIdx.x >> 5u);|'

# The group written one slot late.
mutant slot_off_by_one caught \
    '/qwen4exp_quantize_rows_wide_kernel(/,/^}/{
         s|                                (uint64_t)r \* groups + g);|                                (uint64_t)r * groups + g + 1u);|
     }'

# Deliberate no-op: the tail-group clamp spelled with min().  Must survive.
mutant clamp_spelling survives \
    '/qwen4exp_quantize_rows_wide_kernel(/,/^}/{
         s|    const uint32_t n = width - i0 < 32u ? width - i0 : 32u;|    const uint32_t n = min(width - i0, 32u);|
     }'

if [ "$failures" -ne 0 ]; then
    echo "qwen4exp quant wide mutants: $failures mutant(s) misbehaved" >&2
    exit 1
fi
echo "qwen4exp quant wide mutants: every mutant behaved as expected"
