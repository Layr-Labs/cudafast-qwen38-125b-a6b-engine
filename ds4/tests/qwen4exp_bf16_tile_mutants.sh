#!/bin/sh
# Mutation proof for glm53_matvec_bf16_f32_tile_kernel (ds4_cuda.cu): each
# mutant rewrites one line inside the kernel and tests/test_bf16_prefill_grid
# must fail; the controls must survive.  make test-bf16-tile-mutants
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
TEST="$ROOT/tests/test_bf16_prefill_grid"
SRC="$ROOT/ds4_cuda.cu"
cd "$ROOT" || exit 1
WORK=$(mktemp -d) || exit 1
cp "$SRC" "$WORK/orig.cu" || exit 1
restore() { cp "$WORK/orig.cu" "$SRC"; rm -rf "$WORK"; }
trap 'restore' EXIT INT TERM
build() { make CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}" CUDA_ARCH="${DS4_CUDA_ARCH:-sm_121}" tests/test_bf16_prefill_grid >"$WORK/build.log" 2>&1; }
if ! build; then echo "the unmutated tree does not build" >&2; tail -n 20 "$WORK/build.log" >&2; exit 1; fi
if "$TEST" >"$WORK/base.log" 2>&1; then echo "  baseline                           PASS (as required)"
else echo "  baseline                           FAIL" >&2; tail -n 3 "$WORK/base.log" >&2; exit 1; fi
failures=0
mutant() {
    name=$1; expect=$2; expr=$3
    sed "/glm53_matvec_bf16_f32_tile_kernel(/,/^}/ $expr" "$WORK/orig.cu" >"$WORK/m.cu" || exit 1
    if cmp -s "$WORK/orig.cu" "$WORK/m.cu"; then echo "  $name: the mutation did not apply" >&2; failures=$((failures + 1)); return; fi
    cp "$WORK/m.cu" "$SRC"
    if ! build; then echo "  $name: the mutant does not compile" >&2; failures=$((failures + 1)); cp "$WORK/orig.cu" "$SRC"; return; fi
    if "$TEST" >"$WORK/$name.log" 2>&1; then got=survives; else got=caught; fi
    cp "$WORK/orig.cu" "$SRC"
    if [ "$got" = "$expect" ]; then
        if [ "$got" = caught ]; then printf '  %-34s caught: %s\n' "$name" "$(grep -E 'BF16 launch grid' "$WORK/$name.log" | head -n 1)"
        else printf '  %-34s survives, as expected\n' "$name"; fi
    elif [ "$expect" = caught ]; then printf '  %-34s NOT CAUGHT\n' "$name" >&2; failures=$((failures + 1))
    else printf '  %-34s CAUGHT -- expected an exact no-op\n' "$name" >&2; failures=$((failures + 1)); fi
}
# The chain one element short: start at lane + 32 instead of lane.
mutant bf16_tile_chain_short caught \
    's|    for (uint32_t i = lane; i < in_dim; i += 32u) {|    for (uint32_t i = lane + 32u; i < in_dim; i += 32u) {|'
# Two roundings where the walk has one fmaf.
mutant bf16_tile_no_fma caught \
    's|            for (int c = 0; c < TN; c++) sum\[t\]\[c\] = fmaf(w\[c\], xv\[t\], sum\[t\]\[c\]);|            for (int c = 0; c < TN; c++) sum[t][c] = __fadd_rn(sum[t][c], __fmul_rn(w[c], xv[t]));|'
# Row tiles overlap by one.
mutant bf16_tile_rows_overlap caught \
    's|    const uint32_t row0 = blockIdx.y \* (uint32_t)TM;|    const uint32_t row0 = blockIdx.y * (uint32_t)(TM - 1);|'
# The bf16 word widened into the low half.
mutant bf16_tile_word_low caught \
    's|        for (int c = 0; c < TN; c++) w\[c\] = __uint_as_float((uint32_t)wr\[c\]\[i\] << 16);|        for (int c = 0; c < TN; c++) w[c] = __uint_as_float((uint32_t)wr[c][i]);|'
# Control: an FMA'"'"'s product is commutative.
mutant bf16_tile_fma_commuted survives \
    's|            for (int c = 0; c < TN; c++) sum\[t\]\[c\] = fmaf(w\[c\], xv\[t\], sum\[t\]\[c\]);|            for (int c = 0; c < TN; c++) sum[t][c] = fmaf(xv[t], w[c], sum[t][c]);|'
if [ "$failures" -gt 0 ]; then echo "bf16 tile mutants: $failures mutant(s) did not behave as required" >&2; exit 1; fi
echo "bf16 tile mutants: PASS"
