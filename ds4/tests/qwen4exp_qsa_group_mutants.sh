#!/bin/sh
#
# Mutation proof for the second-cut head-group attention kernel
# (qwen4exp_qsa2_attention_group_kernel in ds4_cuda_qwen4exp.cu).
#
# tests/test_qwen4exp_qsa requires the group kernel to agree with the
# per-head kernel BYTE FOR BYTE over 1024/1024/1024/64/1/1017-row segments,
# dense and sparse.  This rewrites one line of the kernel at a time, rebuilds
# the test and requires it to fail; the deliberate no-ops at the end must
# survive.  Run with: make test-qwen4exp-qsa-group-mutants
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
TEST="$ROOT/tests/test_qwen4exp_qsa"
SRC="$ROOT/ds4_cuda_qwen4exp.cu"
if ! command -v nvcc >/dev/null 2>&1 && [ ! -x "${CUDA_HOME:-/usr/local/cuda}/bin/nvcc" ]; then
    echo "qwen4exp QSA group mutants: skipped, no CUDA toolchain"; exit 0
fi
cd "$ROOT" || exit 1
WORK=$(mktemp -d) || exit 1
cp "$SRC" "$WORK/orig.cu" || exit 1
restore() { cp "$WORK/orig.cu" "$SRC"; rm -rf "$WORK"; }
trap 'restore' EXIT INT TERM
build() {
    make CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}" \
        CUDA_ARCH="${DS4_CUDA_ARCH:-sm_121}" tests/test_qwen4exp_qsa >"$WORK/build.log" 2>&1
}
if ! build; then echo "the unmutated tree does not build" >&2; tail -n 20 "$WORK/build.log" >&2; exit 1; fi
if "$TEST" >"$WORK/base.log" 2>&1; then echo "  baseline                           PASS (as required)"
else echo "  baseline                           FAIL" >&2; tail -n 3 "$WORK/base.log" >&2; exit 1; fi
failures=0
# mutant NAME EXPECT(caught|survives) SED_EXPRESSION   (applied within the qsa2 kernel's lines only)
mutant() {
    name=$1; expect=$2; expr=$3
    sed "/qwen4exp_qsa2_attention_group_kernel(/,/^}/ $expr" "$WORK/orig.cu" >"$WORK/m.cu" || exit 1
    if cmp -s "$WORK/orig.cu" "$WORK/m.cu"; then
        echo "  $name: the mutation did not apply, the kernel source moved" >&2; failures=$((failures + 1)); return
    fi
    cp "$WORK/m.cu" "$SRC"
    if ! build; then echo "  $name: the mutant does not compile" >&2; tail -n 10 "$WORK/build.log" >&2; failures=$((failures + 1)); cp "$WORK/orig.cu" "$SRC"; return; fi
    if "$TEST" >"$WORK/$name.log" 2>&1; then got=survives; else got=caught; fi
    cp "$WORK/orig.cu" "$SRC"
    if [ "$got" = "$expect" ]; then
        if [ "$got" = caught ]; then printf '  %-34s caught: %s\n' "$name" "$(grep -E 'not bit-exact|differ|failed' "$WORK/$name.log" | head -n 1)"
        else printf '  %-34s survives, as expected\n' "$name"; fi
    elif [ "$expect" = caught ]; then printf '  %-34s NOT CAUGHT -- the test passed a broken kernel\n' "$name" >&2; failures=$((failures + 1))
    else printf '  %-34s CAUGHT -- expected an exact no-op\n' "$name" >&2; failures=$((failures + 1)); fi
}

# The score chain: skip one word of the dot (the per-head chain is 64 words).
mutant qsa2_dot_word_dropped caught \
    's|                            dot\[s\]\[h\] = __fmaf_rn(qq.w, kk\[s\]\[i\].w, dot\[s\]\[h\]);|                            if (w + i != 3u) dot[s][h] = __fmaf_rn(qq.w, kk[s][i].w, dot[s][h]);|'
# Two roundings where the chain has one FMA.
mutant qsa2_dot_no_fma caught \
    's|                            dot\[s\]\[h\] = __fmaf_rn(qq.x, kk\[s\]\[i\].x, dot\[s\]\[h\]);|                            dot[s][h] = __fadd_rn(dot[s][h], __fmul_rn(qq.x, kk[s][i].x));|'
# A score written to the wrong tile slot: the sum tree then adds the same
# probabilities in a different order.
mutant qsa2_slot_swapped caught \
    's|                    probs\[h \* nth + slot\] = sc;|                    probs[h * nth + (slot ^ 1u)] = sc;|'
# The tile maximum folded over one scorer warp short.
mutant qsa2_max_fold_short caught \
    's|            for (uint32_t w = 0; w < kwarps; w++) v = fmaxf(v, wmax\[h \* (nth / 32u) + w\]);|            for (uint32_t w = 1; w < kwarps; w++) v = fmaxf(v, wmax[h * (nth / 32u) + w]);|'
# The value chain reads the next key'"'"'s probability.
mutant qsa2_probs_shifted caught \
    's|                        const float4 p4 = \*(const float4 \*)(probs + h \* nth + j + u);|                        const float4 p4 = *(const float4 *)(probs + h * nth + j + u + 4u);|'
# The rescale applied to the running sum as a rounded product and an add.
mutant qsa2_run_sum_no_fma caught \
    's|            run_sum\[h\] = __fmaf_rn(run_sum\[h\], rescale\[h\], tile_sum\[h\]);|            run_sum[h] = __fadd_rn(__fmul_rn(run_sum[h], rescale[h]), tile_sum[h]);|'

# ---- deliberate no-ops -----------------------------------------------------
# A masked key'"'"'s probability is exactly zero and its value is read as
# zero, so fma(0, 0, c) is c: dropping the predicate changes no bit here.
mutant qsa2_masked_fma_unpredicated survives \
    's|                            if (kj\[u\] >= 0) contrib\[c\]\[h\] = __fmaf_rn(p4.x, vv\[c\]\[u\], contrib\[c\]\[h\]);|                            contrib[c][h] = __fmaf_rn(p4.x, vv[c][u], contrib[c][h]);|'
# fmaxf is commutative: the fold'"'"'s operand order cannot move a bit.
mutant qsa2_max_fold_commuted survives \
    's|            for (uint32_t w = 0; w < kwarps; w++) v = fmaxf(v, wmax\[h \* (nth / 32u) + w\]);|            for (uint32_t w = 0; w < kwarps; w++) v = fmaxf(wmax[h * (nth / 32u) + w], v);|'

if [ "$failures" -gt 0 ]; then echo "qwen4exp QSA group mutants: $failures mutant(s) did not behave as required" >&2; exit 1; fi
echo "qwen4exp QSA group mutants: PASS"
