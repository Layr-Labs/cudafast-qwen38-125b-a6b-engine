#!/bin/sh
#
# Mutation proof for the THREE-ROW decode paths (the depth-2 verify width).
#
# The row-3 valve (ds4_qwen4exp_row3_enabled, DS4_QWEN4EXP_NO_ROW3) widens the
# rows <= 2 fast-path gates to rows <= 3: pair-lane Q8 projections, the HC
# down/up pair kernels, the fused GDN and QSA projection launches, the f32
# vector trees, the split routed gate/up, the vector routed down and the vector
# shared expert, all at R = 3.  Four standalone tests and the MoE suite hold
# every one of those against a one-row or eager oracle at width 3, bit for bit.
# That assertion is worth exactly as much as its ability to fail, so this
# script breaks each three-row launch one at a time and requires the tests to
# notice.  Run it with: make test-qwen4exp-row3-mutants
#
# Each mutant rebuilds one CUDA object and relinks the tests that read it.
# The deliberate equivalences at the end are asserted to SURVIVE: the split
# gate/up at R = 2 over a three-pair expert is two passes of the same chains,
# and a valve that ignores its environment still computes the same numbers.

set -u

ROOT=$(cd "$(dirname "$0")/.." && pwd)
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
ARCH="${DS4_CUDA_ARCH:-sm_121}"
DENSE_SRC="$ROOT/ds4_cuda.cu"
MOE_SRC="$ROOT/ds4_cuda_qwen4exp.cu"

if [ ! -x "$CUDA_HOME/bin/nvcc" ] && ! command -v nvcc >/dev/null 2>&1; then
    echo "qwen4exp row3 mutants: skipped, no CUDA toolchain"
    exit 0
fi

cd "$ROOT" || exit 1
WORK=$(mktemp -d) || exit 1
cp "$DENSE_SRC" "$WORK/dense.cu" || exit 1
cp "$MOE_SRC" "$WORK/moe.cu" || exit 1
restore() {
    cp "$WORK/dense.cu" "$DENSE_SRC"
    cp "$WORK/moe.cu" "$MOE_SRC"
    rm -rf "$WORK"
}
trap 'restore' EXIT INT TERM

OBJS="ds4_cuda.o ds4_cuda_qwen4exp.o ds4_image.o cuda/mmq/ds4_ggml_stubs.o cuda/mmq/ds4_mmq.o cuda/mmq/ds4_mmq_d2r.o cuda/mmq/quantize.o cuda/mmq/mmid.o cuda/mmq/mmvq.o cuda/mmq/ds4_repack.o"
STANDALONE="test_q8_decode_pairs test_gdn_projections test_q8_projection_triple test_f32_vector_decode"

build() {
    make CUDA_HOME="$CUDA_HOME" CUDA_ARCH="$ARCH" $OBJS tests/test_qwen4exp_moe >"$WORK/build.log" 2>&1 || return 1
    for t in $STANDALONE; do
        [ -f "$WORK/$t.o" ] || cc -O2 -std=c11 -D_GNU_SOURCE -I. -c -o "$WORK/$t.o" "tests/$t.c" >>"$WORK/build.log" 2>&1 || return 1
        "$CUDA_HOME/bin/nvcc" -O3 -Xcompiler -pthread -o "$WORK/$t" "$WORK/$t.o" $OBJS \
            -lm -Xcompiler -pthread -L"$CUDA_HOME/targets/sbsa-linux/lib" -L"$CUDA_HOME/lib64" -lcudart -lcublas >>"$WORK/build.log" 2>&1 || return 1
    done
    return 0
}

# run TESTNAME: the standalone tests exit non-zero on a mismatch; the MoE
# suite prints and exits non-zero on its first failed requirement.
run() {
    case "$1" in
        moe) ./tests/test_qwen4exp_moe >"$WORK/run.log" 2>&1 ;;
        *)   "$WORK/$1" >"$WORK/run.log" 2>&1 ;;
    esac
}

if ! build; then
    echo "qwen4exp row3 mutants: the unmutated tree does not build" >&2
    tail -n 20 "$WORK/build.log" >&2
    exit 1
fi
for t in $STANDALONE moe; do
    if run "$t"; then
        echo "  baseline $t: PASS (as required)"
    else
        echo "  baseline $t: FAIL -- the unmutated three-row paths do not agree" >&2
        tail -n 5 "$WORK/run.log" >&2
        exit 1
    fi
done

failures=0

# mutant NAME FILE(dense|moe) TEST EXPECT(caught|survives) SED_EXPRESSION
mutant() {
    name=$1; file=$2; test=$3; expect=$4; expr=$5
    if [ "$file" = dense ]; then src="$DENSE_SRC"; orig="$WORK/dense.cu"; else src="$MOE_SRC"; orig="$WORK/moe.cu"; fi
    sed "$expr" "$orig" >"$WORK/m.cu" || exit 1
    if cmp -s "$orig" "$WORK/m.cu"; then
        echo "  $name: the mutation did not apply, the source moved" >&2
        failures=$((failures + 1))
        return
    fi
    cp "$WORK/m.cu" "$src"
    if ! build; then
        echo "  $name: the mutant does not compile" >&2
        tail -n 10 "$WORK/build.log" >&2
        failures=$((failures + 1))
        cp "$orig" "$src"
        return
    fi
    if run "$test"; then outcome=survives; else outcome=caught; fi
    cp "$orig" "$src"
    if [ "$outcome" = "$expect" ]; then
        printf '  %-44s %s (as required)\n' "$name" "$outcome"
    else
        printf '  %-44s %s -- expected %s\n' "$name" "$outcome" "$expect" >&2
        failures=$((failures + 1))
    fi
}

# 1. The general pair-lane projection at three rows launched as the two-row
#    template over one y-block: row 2 is never computed (its canary stands).
mutant "pair_lanes<3> launched as <2>" dense test_q8_decode_pairs caught \
    's/(matmul_q8_0_preq_pair_lanes_kernel<3, false>),/(matmul_q8_0_preq_pair_lanes_kernel<2, false>),/'
# 2. The HC down pair at three rows launched as the two-row instantiation.
mutant "hc_down_pair<3> launched as <2>" dense test_q8_decode_pairs caught \
    's/(matmul_q8_hc_down_pair_kernel<3>), 320, 64, 0,/(matmul_q8_hc_down_pair_kernel<2>), 320, 64, 0,/'
# 3. The HC up warp pair at three rows launched as the two-row instantiation.
mutant "hc_warp_pair<3> launched as <2>" dense test_q8_decode_pairs caught \
    's/(matmul_q8_hc_warp_pair_kernel<3>),/(matmul_q8_hc_warp_pair_kernel<2>),/'
# 4. The fused GDN four-projection launch at three rows as the two-row one.
mutant "gdn_projection<3> launched as <2>" dense test_gdn_projections caught \
    's/QWEN4EXP_LAUNCH_PDL((qwen_gdn_projection_kernel<3>),/QWEN4EXP_LAUNCH_PDL((qwen_gdn_projection_kernel<2>),/'
# 5. The fused QSA triple projection at three rows as the two-row one.
mutant "q8_projection_triple<3> launched as <2>" dense test_q8_projection_triple caught \
    's/QWEN4EXP_LAUNCH_PDL((qwen_q8_projection_triple_kernel<3>),/QWEN4EXP_LAUNCH_PDL((qwen_q8_projection_triple_kernel<2>),/'
# 6. The f32 vector tree (router shape) at three rows as the two-row one.
mutant "f32_vector_tree<3,4,1> launched as <2,4,1>" dense test_f32_vector_decode caught \
    's/QWEN4EXP_LAUNCH_PDL((qwen_f32_vector_tree_kernel<3, 4, 1>),/QWEN4EXP_LAUNCH_PDL((qwen_f32_vector_tree_kernel<2, 4, 1>),/'
# 7. The vector routed down at tile 3 launched as R = 2: one y-block of two
#    rows, the third row's output untouched.
mutant "moe_down vector R=3 launched as R=2" moe moe caught \
    's/if (tile == 3) { QWEN4EXP_DOWN_IMPL(3, DS4_QWEN4EXP_TY_q5_1, true); }/if (tile == 3) { QWEN4EXP_DOWN_IMPL(2, DS4_QWEN4EXP_TY_q5_1, true); }/'
# 8. The split gate/up at R = 3 drops its third pair.
mutant "split gate/up R=3 drops the third pair" moe moe caught \
    's/                    if (r < take) { \\$/                    if (r < take - (R == 3)) { \\/'
# 9. The vector shared expert at tile 3 launched as R = 2.
mutant "shared gate/up tile 3 launched as R=2" moe moe caught \
    's/else if (tile == 3) { QWEN4EXP_SH_GATEUP(3); }/else if (tile == 3) { QWEN4EXP_SH_GATEUP(2); }/'

# Deliberate equivalences.
# 10. The split gate/up at R = 2 over three pairs: the `at += R` loop takes a
#     pair and then a remainder, each row's chain untouched.
mutant "split gate/up R=2 over three rows (equivalent)" moe moe survives \
    's/if (tile == 3) { QWEN4EXP_SPLIT_GATEUP_R(3, V, P); } \\$/if (tile == 3) { QWEN4EXP_SPLIT_GATEUP_R(2, V, P); } \\/'
# 11. A valve that ignores its environment: both paths compute the same bits.
mutant "row-3 valve ignores DS4_QWEN4EXP_NO_ROW3 (equivalent)" dense test_q8_decode_pairs survives \
    's/        enabled = !(e \&\& e\[0\] \&\& e\[0\] != '"'"'0'"'"');/        enabled = 1; (void)e;/'

if [ "$failures" -ne 0 ]; then
    echo "qwen4exp row3 mutants: $failures unexpected outcome(s)" >&2
    exit 1
fi
echo "qwen4exp row3 mutants: every mutant behaved as required"
