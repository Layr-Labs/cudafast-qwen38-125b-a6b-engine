#!/bin/sh
#
# Mutation proof for the qwen4exp PLE Metal kernels.
#
# tests/test_qwen4exp_ple_kernels checks the gate, the dilated depthwise
# convolution, its rolling window and the whole block.  This script proves
# those checks bite: it rewrites one line of metal/qwen4exp_ple.metal at a
# time, points the engine at the rewritten copy through
# DS4_METAL_QWEN4EXP_PLE_SOURCE, and requires the test to FAIL.  A mutant that
# still passes means the test does not cover that line.
#
# The Metal sources are read at run time, so no rebuild is needed between
# mutants.  Run it with: make test-qwen4exp-ple-kernel-mutants

set -u

ROOT=$(cd "$(dirname "$0")/.." && pwd)
TEST="$ROOT/tests/test_qwen4exp_ple_kernels"
SRC="$ROOT/metal/qwen4exp_ple.metal"

case "$(uname -s)" in
Darwin) ;;
*)
    echo "qwen4exp PLE mutants: skipped, DS4_METAL_QWEN4EXP_PLE_SOURCE is Metal only"
    exit 0
    ;;
esac

if [ ! -x "$TEST" ]; then
    echo "qwen4exp PLE mutants: $TEST is missing, run make tests/test_qwen4exp_ple_kernels" >&2
    exit 1
fi
if [ ! -f "$SRC" ]; then
    echo "qwen4exp PLE mutants: $SRC is missing" >&2
    exit 1
fi

WORK=$(mktemp -d) || exit 1
trap 'rm -rf "$WORK"' EXIT
cd "$ROOT" || exit 1

failures=0

# The unmutated source must pass, otherwise every mutant below proves nothing.
if DS4_METAL_QWEN4EXP_PLE_SOURCE="$SRC" DS4_METAL_MATH_SAFE=1 "$TEST" \
        >"$WORK/base.log" 2>&1; then
    echo "  baseline                      PASS (as required)"
else
    echo "  baseline                      FAIL -- the unmutated kernels do not pass" >&2
    sed -n '$p' "$WORK/base.log" >&2
    exit 1
fi

# mutant NAME SED_EXPRESSION
mutant() {
    name=$1
    expr=$2
    out="$WORK/$name.metal"
    sed "$expr" "$SRC" >"$out" || exit 1
    if cmp -s "$SRC" "$out"; then
        echo "  $name: the mutation did not apply, the kernel source moved" >&2
        failures=$((failures + 1))
        return
    fi
    if DS4_METAL_QWEN4EXP_PLE_SOURCE="$out" DS4_METAL_MATH_SAFE=1 "$TEST" \
            >"$WORK/$name.log" 2>&1; then
        printf '  %-29s NOT CAUGHT -- the test passed a broken kernel\n' "$name" >&2
        failures=$((failures + 1))
    else
        printf '  %-29s caught: %s\n' "$name" "$(grep -v '^ds4: ' "$WORK/$name.log" | tail -n 1)"
    fi
}

# The convolution is DILATED by the n-gram size.  Walk adjacent rows instead:
# the same four taps, read from the wrong rows.
mutant conv_undilated \
    's|const uint i = t + args.dilation \* k;|const uint i = t + k;|'

# The rolling window is shifted by one row: the state carries the rows ending
# one earlier, so only a chunked run disagrees with a single call.
mutant conv_window_shift \
    's|const uint i = args.n_tokens + j;|const uint i = args.n_tokens + j - 1u;|'

# Taps reversed: tap 0 multiplies the current row instead of tap K-1.
mutant conv_tap_order \
    's|weight\[(ulong)c \* args.conv_kernel + k\]|weight[(ulong)c * args.conv_kernel + (args.conv_kernel - 1u - k)]|'

# The gate takes a plain square root of the magnitude: every negative inner
# product gates the wrong way.
mutant gate_unsigned_sqrt \
    's|return v > 0.0f ? magnitude : (v < 0.0f ? -magnitude : 0.0f);|return magnitude;|'

# The gate drops the 1e-6 floor inside the signed square root.  Only the
# floor case of the test reaches this line: at production scale the inner
# product is orders above 1e-6 and the floor changes nothing.
mutant gate_no_floor \
    's|sqrt(max(fabs(v), 1.0e-6f))|sqrt(fabs(v))|'

# The gate drops the 1/sqrt(hidden) scale on the inner product.
mutant gate_no_scale \
    's|qwen4exp_ple_signed_sqrt(sumf \* args.inv_sqrt_embd)|qwen4exp_ple_signed_sqrt(sumf)|'

# The block adds the convolution without the un-convolved gated stream.
mutant conv_drop_residual \
    's|hyper\[index\] += gated\[index\] + acc \* qwen4exp_ple_sigmoid(acc);|hyper[index] += acc * qwen4exp_ple_sigmoid(acc);|'

if [ "$failures" -ne 0 ]; then
    echo "qwen4exp PLE mutants: $failures mutant(s) survived" >&2
    exit 1
fi
echo "qwen4exp PLE mutants: PASS (every mutant was caught)"
