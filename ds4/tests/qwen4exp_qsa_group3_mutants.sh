#!/bin/sh
#
# Mutation proof for the THIRD-CUT head-group attention kernel.
#
# tests/test_qwen4exp_qsa requires the third cut (the dispatch default at the
# production shape) to agree with the per-head kernel BYTE FOR BYTE at 1024,
# 1017, 64 and 1 rows, dense and sparse.  This script breaks the third cut one
# line at a time -- the scorer (qwen4exp_qsa3_score_tile) and the kernel body
# -- and requires that check to notice.  Each mutant rebuilds one object plus
# a relink.  Run it with: make test-qwen4exp-qsa-group3-mutants
#
# The two deliberate no-ops at the end must SURVIVE.

set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
TEST="$ROOT/tests/test_qwen4exp_qsa"
SRC="$ROOT/ds4_cuda_qwen4exp.cu"
if ! command -v nvcc >/dev/null 2>&1 && [ ! -x "${CUDA_HOME:-/usr/local/cuda}/bin/nvcc" ]; then
    echo "qwen4exp QSA third-cut mutants: skipped, no CUDA toolchain"; exit 0
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
KERNEL='/qwen4exp_qsa3_attention_group_kernel(/,/^}/'
# mutant NAME EXPECT(caught|survives) SED_EXPRESSION [RANGE]   (default range: the kernel body)
mutant() {
    name=$1; expect=$2; expr=$3; range=${4:-$KERNEL}
    sed "$range $expr" "$WORK/orig.cu" >"$WORK/m.cu" || exit 1
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

SCORE='/^__device__ __forceinline__ static void qwen4exp_qsa3_score_tile(/,/^}/'

# The score chain: skip one word of the dot (the per-head chain is 64 words).
mutant qsa3_dot_word_dropped caught \
    's|                        dot\[s\]\[h\] = __fmaf_rn(qq.w, kk\[s\]\[i\].w, dot\[s\]\[h\]);|                        if (w + i != 3u) dot[s][h] = __fmaf_rn(qq.w, kk[s][i].w, dot[s][h]);|' "$SCORE"
# Two roundings where the chain has one FMA.
mutant qsa3_dot_no_fma caught \
    's|                        dot\[s\]\[h\] = __fmaf_rn(qq.x, kk\[s\]\[i\].x, dot\[s\]\[h\]);|                        dot[s][h] = __fadd_rn(dot[s][h], __fmul_rn(qq.x, kk[s][i].x));|' "$SCORE"
# A score written to the wrong tile slot: the sum tree then adds the same
# probabilities in a different order.
mutant qsa3_slot_swapped caught \
    's|                probs\[h \* NTH + slot\] = sc;|                probs[h * NTH + (slot ^ 1u)] = sc;|' "$SCORE"
# The tile maximum folded over one per-warp slot short.
mutant qsa3_max_fold_short caught \
    's|                for (uint32_t w = 0; w < 8u; w++) v = fmaxf(v, wmax\[vt \* 8u + w\]);|                for (uint32_t w = 1; w < 8u; w++) v = fmaxf(v, wmax[vt * 8u + w]);|'
# The value chain reads the next key'"'"'s probability.
mutant qsa3_probs_shifted caught \
    's|                        const float4 p4 = \*(const float4 \*)(probs + h \* NTH + j + u);|                        const float4 p4 = *(const float4 *)(probs + h * NTH + j + u + 4u);|'
# The rescale applied to the running sum as a rounded product and an add.
mutant qsa3_run_sum_no_fma caught \
    's|                st_runsum\[vt\] = __fmaf_rn(st_runsum\[vt\], st_rescale\[vt\], tsum\[vt\]);|                st_runsum[vt] = __fadd_rn(__fmul_rn(st_runsum[vt], st_rescale[vt]), tsum[vt]);|'
# The tile-sum tree paired the wrong way: level 128 pairs slot l with l+64.
mutant qsa3_sum_tree_pairs caught \
    's|                float v = ((p\[vl\] + p\[vl + 128u\]) + (p\[vl + 64u\] + p\[vl + 192u\])) +|                float v = ((p[vl] + p[vl + 64u]) + (p[vl + 128u] + p[vl + 192u])) +|'
# The valuers consume the buffer the scorers are writing.
mutant qsa3_buffer_race caught \
    's|            float \*probs = probs0 + cur \* GROUP \* NTH;|            float *probs = probs0 + (cur ^ 1u) * GROUP * NTH;|'

# ---- deliberate no-ops -----------------------------------------------------
# A masked key'"'"'s probability is exactly zero and its value is read as
# zero, so fma(0, 0, c) is c: dropping the predicate changes no bit here.
mutant qsa3_masked_fma_unpredicated survives \
    's|                            if (kj\[u\] >= 0) contrib\[c\]\[h\] = __fmaf_rn(p4.x, vv\[c\]\[u\], contrib\[c\]\[h\]);|                            contrib[c][h] = __fmaf_rn(p4.x, vv[c][u], contrib[c][h]);|'
# fmaxf is commutative: the fold'"'"'s operand order cannot move a bit.
mutant qsa3_max_fold_commuted survives \
    's|                for (uint32_t w = 0; w < 8u; w++) v = fmaxf(v, wmax\[vt \* 8u + w\]);|                for (uint32_t w = 0; w < 8u; w++) v = fmaxf(wmax[vt * 8u + w], v);|'

if [ "$failures" -gt 0 ]; then echo "qwen4exp QSA third-cut mutants: $failures mutant(s) did not behave as required" >&2; exit 1; fi
echo "qwen4exp QSA third-cut mutants: PASS"
