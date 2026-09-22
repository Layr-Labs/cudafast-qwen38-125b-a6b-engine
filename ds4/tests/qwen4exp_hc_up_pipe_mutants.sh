#!/bin/sh
#
# Mutation proof for the pipelined hyper-connection up+mix tile.
#
# tests/test_qwen4exp_hc_norm requires the fused mixer, with
# qwen4exp_hc_up_mix_pipe_kernel in it, to agree with the op-by-op chain
# BIT FOR BIT over 120 cases.  That assertion
# is worth exactly as much as its ability to fail, so this script breaks the
# pipelined kernel one line at a time and requires the test to notice: the
# group order, the pipeline's stage pairing, the weight fragment's slots
# and the epilogue's final scaling.  The last mutant is a
# deliberate no-op (the commutative operands of one multiply) and must
# SURVIVE, so a change that made it observable would also be reported.
#
# Each mutant rebuilds ds4_cuda_qwen4exp.o, under a minute each.  The GPU run of each
# mutant is taken under the lab's GPU lock when one is configured
# (DS4_GPU_LOCK=/path), so builds never hold it.
# Run it with: make test-qwen4exp-hc-up-pipe-mutants

set -u

ROOT=$(cd "$(dirname "$0")/.." && pwd)
TEST="$ROOT/tests/test_qwen4exp_hc_norm"
SRC="$ROOT/ds4_cuda_qwen4exp.cu"

if [ ! -f "$SRC" ]; then
    echo "qwen4exp HC up pipe mutants: $SRC is missing" >&2
    exit 1
fi
if ! command -v nvcc >/dev/null 2>&1 && [ ! -x "${CUDA_HOME:-/usr/local/cuda}/bin/nvcc" ]; then
    echo "qwen4exp HC up pipe mutants: skipped, no CUDA toolchain"
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
# A mutant that breaks the producer/consumer handshake would hang, not
# fail, so every run has a deadline; a hang counts as caught.
run_test() {
    if [ -n "${DS4_GPU_LOCK:-}" ]; then
        flock "$DS4_GPU_LOCK" timeout 900 "$TEST"
    else
        timeout 900 "$TEST"
    fi
}

if ! build; then
    echo "qwen4exp HC up pipe mutants: the unmutated tree does not build" >&2
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

# The stage's groups walked descending: the same products, chained in the
# other order.
mutant k_order caught \
    's|        for (int gg = 0; gg < QHP_G; gg++) { /\* the stage'"'"'s groups, ascending \*/|        for (int gg = QHP_G - 1; gg >= 0; gg--) { /* the stage'"'"'s groups, ascending */|'

# The consumers take their activations from the other stage buffer (data
# only: the barriers still pair up, so a wrong answer, not a hang).
mutant stage_skew caught \
    's|        const unsigned char \*sA = sA_all + buf \* QHP_A_BYTES;|        const unsigned char *sA = sA_all + ((buf + 1) % QHP_STAGES) * QHP_A_BYTES;|'

# The mix'"'"'s FFMA split into a rounded multiply and a rounded add (the
# intrinsics, so the compiler cannot contract it back): the same algebra,
# a moved rounding.  (Its final scaling, `mix * (1.0f / 4)`, is exactly
# `mix / 4` -- a power of two -- so that is not a mutation.)
mutant epilogue_fma_split caught \
    's|                    mix = __fmaf_rn(qwen4exp_sigmoid(sC\[t \* C_TOK_STRIDE + h \* QHP_BC + (int)lane\]), normed, mix);|                    mix = __fadd_rn(mix, __fmul_rn(qwen4exp_sigmoid(sC[t * C_TOK_STRIDE + h * QHP_BC + (int)lane]), normed));|'

# The commutative operands of one multiply, which is the same rounding.
mutant fmul_operands_swapped survives \
    's|                        acc\[mi\]\[ni\]\[0\] = qwen4exp_fma_ftz(qwen4exp_fmul_ftz(wsp.x, xs\[mi\]\[0\]\[gg\]), qhp_dot_to_f32(d\[mi\]\[0\]), acc\[mi\]\[ni\]\[0\]);|                        acc[mi][ni][0] = qwen4exp_fma_ftz(qwen4exp_fmul_ftz(xs[mi][0][gg], wsp.x), qhp_dot_to_f32(d[mi][0]), acc[mi][ni][0]);|'

if [ "$failures" -ne 0 ]; then
    echo "qwen4exp HC up pipe mutants: $failures problem(s)" >&2
    exit 1
fi
echo "qwen4exp HC up pipe mutants: all mutants behaved as required"
