#!/bin/sh
#
# Mutation proof for the pipelined dense Q8_0 MMA tile.
#
# tests/test_qwen4exp_hc_norm requires matmul_q8_0_preq_rows_mma_pipe_kernel
# to agree with matmul_q8_0_preq_rows_mma_kernel BIT FOR BIT at every
# production (K, N) and at widths 8, 16, 64, 1024 and 1017.  That assertion
# is worth exactly as much as its ability to fail, so this script breaks the
# pipelined kernel one line at a time and requires the test to notice: the
# k order, the float chain's association, the pipeline's stage pairing, the
# weight fragment's slots, and the seeded conversion.  The last mutant is a
# deliberate no-op (the commutative operands of one multiply) and must
# SURVIVE, so a change that made it observable would also be reported.
#
# Each mutant rebuilds ds4_cuda.o, minutes each.  The GPU run of each
# mutant is taken under the lab's GPU lock when one is configured
# (DS4_GPU_LOCK=/path), so builds never hold it.
# Run it with: make test-qwen4exp-mma-pipe-mutants

set -u

ROOT=$(cd "$(dirname "$0")/.." && pwd)
TEST="$ROOT/tests/test_qwen4exp_hc_norm"
SRC="$ROOT/ds4_cuda.cu"

if [ ! -f "$SRC" ]; then
    echo "qwen4exp MMA pipe mutants: $SRC is missing" >&2
    exit 1
fi
if ! command -v nvcc >/dev/null 2>&1 && [ ! -x "${CUDA_HOME:-/usr/local/cuda}/bin/nvcc" ]; then
    echo "qwen4exp MMA pipe mutants: skipped, no CUDA toolchain"
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
        tests/test_qwen4exp_hc_norm >"$WORK/build.log" 2>&1
}
run_test() {
    if [ -n "${DS4_GPU_LOCK:-}" ]; then
        flock "$DS4_GPU_LOCK" "$TEST"
    else
        "$TEST"
    fi
}

if ! build; then
    echo "qwen4exp MMA pipe mutants: the unmutated tree does not build" >&2
    tail -n 20 "$WORK/build.log" >&2
    exit 1
fi
if run_test >"$WORK/base.log" 2>&1; then
    echo "  baseline                           PASS (as required)"
else
    echo "  baseline                           FAIL -- the unmutated pipelined tile does not agree" >&2
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
    if run_test >"$WORK/$name.log" 2>&1; then
        got=survives
    else
        got=caught
    fi
    cp "$WORK/orig.cu" "$SRC"
    if [ "$got" = "$expect" ]; then
        if [ "$got" = caught ]; then
            printf '  %-34s caught: %s\n' "$name" \
                "$(grep -E 'differ|error|disagree|non-finite|failed' \
                        "$WORK/$name.log" | head -n 1)"
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

# The k32 steps of a stage walked descending: the same dots, the same
# products, accumulated in the other order.
mutant k_order caught \
    's|        for (int gg = 0; gg < G; gg++) { /\* the stage'"'"'s k32 steps, ascending \*/|        for (int gg = G - 1; gg >= 0; gg--) { /* the stage'"'"'s k32 steps, ascending */|'

# The float chain reassociated: (ws * xs) * dot + acc becomes ws * (xs * dot) + acc.
mutant epilogue_order caught \
    's|                    acc\[mi\]\[ni\]\[0\] = q8_mma_fma_ftz(q8_mma_fmul_ftz(wsp.x, xs\[mi\]\[0\]\[gg\]), q8_mma_dot_to_f32(d\[mi\]\[0\]), acc\[mi\]\[ni\]\[0\]);|                    acc[mi][ni][0] = q8_mma_fma_ftz(wsp.x, q8_mma_fmul_ftz(xs[mi][0][gg], q8_mma_dot_to_f32(d[mi][0])), acc[mi][ni][0]);|'

# The consumers read the buffer after the one the producers published.
mutant stage_skew caught \
    's|        const int buf = (int)(s % (uint64_t)STAGES); /\* consumers: stage s'"'"'s buffer \*/|        const int buf = (int)((s + 1u) % (uint64_t)STAGES); /* consumers: stage s'"'"'s buffer */|'

# The weight fragment'"'"'s two k halves swapped for the odd blocks.
mutant fragment_lane_swap caught \
    's|                    bf\[0\] = pw\[0\];|                    bf[0] = pw[4]; bf[1] = pw[0]; if (0)|'

# The seed the conversion subtracts, one off.
mutant magic_seed caught \
    's|#define Q8_MMA_MAGIC_BITS 0x4B400000|#define Q8_MMA_MAGIC_BITS 0x4B400001|'

# The commutative operands of one multiply, which is the same rounding.
mutant fmul_operands_swapped survives \
    's|                    acc\[mi\]\[ni\]\[1\] = q8_mma_fma_ftz(q8_mma_fmul_ftz(wsp.y, xs\[mi\]\[0\]\[gg\]), q8_mma_dot_to_f32(d\[mi\]\[1\]), acc\[mi\]\[ni\]\[1\]);|                    acc[mi][ni][1] = q8_mma_fma_ftz(q8_mma_fmul_ftz(xs[mi][0][gg], wsp.y), q8_mma_dot_to_f32(d[mi][1]), acc[mi][ni][1]);|'

if [ "$failures" -ne 0 ]; then
    echo "qwen4exp MMA pipe mutants: $failures problem(s)" >&2
    exit 1
fi
echo "qwen4exp MMA pipe mutants: all mutants behaved as required"
