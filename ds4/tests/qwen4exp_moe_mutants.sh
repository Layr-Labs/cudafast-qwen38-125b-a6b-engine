#!/bin/sh
#
# Mutation proof for the qwen4exp MoE Metal kernels.
#
# tests/test_qwen4exp_moe checks the router, the expert GEMMs and the shared
# expert.  This script proves those checks bite: it rewrites one line of
# metal/qwen4exp_moe.metal at a time, points the engine at the rewritten copy
# through DS4_METAL_QWEN4EXP_MOE_SOURCE, and requires the test to FAIL.  A
# mutant that still passes means the test does not cover that line.
#
# The Metal sources are read at run time, so no rebuild is needed between
# mutants.  Run it with: make test-qwen4exp-moe-mutants

set -u

ROOT=$(cd "$(dirname "$0")/.." && pwd)
TEST="$ROOT/tests/test_qwen4exp_moe"
SRC="$ROOT/metal/qwen4exp_moe.metal"

case "$(uname -s)" in
Darwin) ;;
*)
    echo "qwen4exp MoE mutants: skipped, DS4_METAL_QWEN4EXP_MOE_SOURCE is Metal only"
    exit 0
    ;;
esac

if [ ! -x "$TEST" ]; then
    echo "qwen4exp MoE mutants: $TEST is missing, run make tests/test_qwen4exp_moe" >&2
    exit 1
fi
if [ ! -f "$SRC" ]; then
    echo "qwen4exp MoE mutants: $SRC is missing" >&2
    exit 1
fi

WORK=$(mktemp -d) || exit 1
trap 'rm -rf "$WORK"' EXIT
cd "$ROOT" || exit 1

failures=0

# The unmutated source must pass, otherwise every mutant below proves nothing.
if DS4_METAL_QWEN4EXP_MOE_SOURCE="$SRC" "$TEST" >"$WORK/base.log" 2>&1; then
    echo "  baseline                 PASS (as required)"
else
    echo "  baseline                 FAIL -- the unmutated kernels do not pass" >&2
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
    if DS4_METAL_QWEN4EXP_MOE_SOURCE="$out" "$TEST" >"$WORK/$name.log" 2>&1; then
        printf '  %-24s NOT CAUGHT -- the test passed a broken kernel\n' "$name" >&2
        failures=$((failures + 1))
    else
        printf '  %-24s caught: %s\n' "$name" "$(grep -v '^ds4: ' "$WORK/$name.log" | tail -n 1)"
    fi
}

# Q5_1: read the high nibble where the low nibble belongs.
mutant q5_1_nibble_swap \
    's|((uint)xb->qs\[j\] & 0x0Fu)|((uint)xb->qs[j] >> 4u)|'

# Q5_1: take the fifth bit of the upper half from bit j+12 instead of j+16.
# ggml writes this as (qh >> (j + 12)) & 0x10, which is bit j+16; shifting by
# 12 with a 1-bit mask is the plausible transcription error.
mutant q5_1_high_bit_shift \
    's|(qh >> (j + 16u))|(qh >> (j + 12u))|'

# Q5_K: decode the block as Q4_K, which drops the fifth bit plane.  This is
# the "the loader admits Q5_K but the kernel reads Q4_K" bug the expert type
# table exists to prevent.
mutant q5_K_reads_q4_K \
    's|return ds4_glm_q5_K_value((device const block_q5_K \*)row, k);|return ds4_glm_q4_K_value((device const block_q4_K *)row, k);|'

# Q6_K: decode the block as Q5_K.  Both are 256-element superblocks with
# packed scales, so the mistake is silent without a per-type check.
mutant q6_K_reads_q5_K \
    's|return ds4_glm_q6_K_value((device const block_q6_K \*)row, k);|return ds4_glm_q5_K_value((device const block_q5_K *)row, k);|'

# Router: break ties toward the higher expert index.
mutant router_tie_rule \
    's|(sa == sb \&\& a < b)|(sa == sb \&\& a > b)|'

# Router: score GLM-style sigmoid probabilities instead of the raw logits.
# Selection is unchanged (sigmoid is monotonic); only the weights move.
mutant router_sigmoid_probs \
    's|sel_scores\[tid\] = active ? token_logits\[tid\] : -INFINITY;|sel_scores[tid] = active ? 1.0f / (1.0f + exp(-token_logits[tid])) : -INFINITY;|'

# Routed experts: drop the silu from the SwiGLU.
mutant swiglu_no_silu \
    's|(g / (1.0f + exp(-g))) \* scratch\[ntg\] \* weights\[selected_off\]|g * scratch[ntg] * weights[selected_off]|'

# Shared expert: gate on the raw router logit instead of its sigmoid.
mutant shared_gate_no_sigmoid \
    's|gate_out\[token\] = 1.0f / (1.0f + exp(-scratch\[0\]));|gate_out[token] = scratch[0];|'

# Combine: overwrite the routed sum instead of adding the shared expert to it.
mutant combine_overwrite \
    's|out\[off\] += gate_scale\[token\] \* scratch\[0\];|out[off] = gate_scale[token] * scratch[0];|'

if [ "$failures" -ne 0 ]; then
    echo "qwen4exp MoE mutants: $failures mutant(s) survived" >&2
    exit 1
fi
echo "qwen4exp MoE mutants: PASS (every mutant was caught)"
