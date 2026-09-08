#!/bin/sh
#
# Mutation proof for the qwen4exp hyper-connection INJECT kernel.
#
# tests/test_qwen4exp_hc_norm checks the inject weights against a double
# reference, and checks that the Q8_0 encoding of the same weights gives a
# bit-identical result.  This script proves
# those checks bite: it rewrites one line of metal/qwen4exp_hc.metal at a
# time, points the engine at the rewritten copy through
# DS4_METAL_QWEN4EXP_HC_SOURCE, and requires the test to FAIL.  A mutant that
# still passes means the test does not cover that line.
#
# The Metal sources are read at run time, so no rebuild is needed between
# mutants.  Run it with: make test-qwen4exp-hc-mutants

set -u

ROOT=$(cd "$(dirname "$0")/.." && pwd)
TEST="$ROOT/tests/test_qwen4exp_hc_norm"
SRC="$ROOT/metal/qwen4exp_hc.metal"

case "$(uname -s)" in
Darwin) ;;
*)
    echo "qwen4exp HC inject mutants: skipped, DS4_METAL_QWEN4EXP_HC_SOURCE is Metal only"
    exit 0
    ;;
esac

if [ ! -x "$TEST" ]; then
    echo "qwen4exp HC inject mutants: $TEST is missing, run make tests/test_qwen4exp_hc_norm" >&2
    exit 1
fi
if [ ! -f "$SRC" ]; then
    echo "qwen4exp HC inject mutants: $SRC is missing" >&2
    exit 1
fi

WORK=$(mktemp -d) || exit 1
trap 'rm -rf "$WORK"' EXIT
cd "$ROOT" || exit 1

failures=0

# The unmutated source must pass, otherwise every mutant below proves nothing.
if DS4_METAL_QWEN4EXP_HC_SOURCE="$SRC" DS4_METAL_MATH_SAFE=1 "$TEST" \
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
    if DS4_METAL_QWEN4EXP_HC_SOURCE="$out" DS4_METAL_MATH_SAFE=1 "$TEST" \
            >"$WORK/$name.log" 2>&1; then
        printf '  %-29s NOT CAUGHT -- the test passed a broken kernel\n' "$name" >&2
        failures=$((failures + 1))
    else
        printf '  %-29s caught: %s\n' "$name" "$(grep -v '^ds4: ' "$WORK/$name.log" | tail -n 1)"
    fi
}

# The Q8_0 decode is the new path.  Read the block scale as if every block
# shared the first one: the magnitudes stay plausible and the F32 comparison
# above still passes, so only the Q8_0 case can catch it.
mutant q8_scale_frozen \
    's|device const char \* wr = weight + (uint64_t)h \* args.weight_row_bytes;|device const char * wr = weight;|'

# Dispatch every inject weight as F32 whatever its type: a Q8_0 row read as
# floats gives values of the wrong magnitude entirely.
mutant inject_type_ignored \
    's|sumf += xr\[i\] \* ds4_qwen4exp_inject_value(args.weight_type, wr, i);|sumf += xr[i] * ds4_qwen4exp_f32_value(wr, i);|'

if [ "$failures" -gt 0 ]; then
    echo "qwen4exp HC inject mutants: $failures mutant(s) were NOT caught" >&2
    exit 1
fi
echo "qwen4exp HC inject mutants: PASS (every mutant was caught)"
