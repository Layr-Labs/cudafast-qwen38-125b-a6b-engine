#!/bin/sh
#
# Mutation proof for the FUSED qwen4exp hyper-connection mixer.
#
# tests/test_qwen4exp_hc_norm requires ds4_gpu_qwen4exp_hc_mixer_tensor to
# agree with ds4_gpu_qwen4exp_hc_mixer_unfused_tensor BIT FOR BIT over 84
# shape/head/flag combinations.  That assertion is worth exactly as much as its
# ability to fail, so this script breaks the fused kernels one line at a time
# and requires the test to notice.
#
# The CUDA sources are compiled, not read at run time, so unlike the Metal
# inject mutants each mutant here costs a rebuild of ONE object plus a relink.
# Run it with: make test-qwen4exp-hc-fuse-mutants
#
# A mutant that is NOT caught is either a test gap or a genuine equivalence.
# The deliberate no-ops at the end are the second kind and are asserted to
# survive, so a change that made them observable would also be reported.

set -u

ROOT=$(cd "$(dirname "$0")/.." && pwd)
TEST="$ROOT/tests/test_qwen4exp_hc_norm"
SRC="$ROOT/ds4_cuda_qwen4exp.cu"

if [ ! -f "$SRC" ]; then
    echo "qwen4exp HC fuse mutants: $SRC is missing" >&2
    exit 1
fi
if ! command -v nvcc >/dev/null 2>&1 && [ ! -x "${CUDA_HOME:-/usr/local/cuda}/bin/nvcc" ]; then
    echo "qwen4exp HC fuse mutants: skipped, no CUDA toolchain"
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

if ! build; then
    echo "qwen4exp HC fuse mutants: the unmutated tree does not build" >&2
    tail -n 20 "$WORK/build.log" >&2
    exit 1
fi
if "$TEST" >"$WORK/base.log" 2>&1; then
    echo "  baseline                           PASS (as required)"
else
    echo "  baseline                           FAIL -- the unmutated fused mixer does not agree" >&2
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

# ---- the norm --------------------------------------------------------------

# rsqrtf is the approximate reciprocal square root; the norm needs the
# correctly rounded one, and the unfused kernel says so in a comment.
mutant norm_rsqrt caught \
    's|    return 1.0f / sqrtf(total / (float)group + eps);|    return rsqrtf(total / (float)group + eps);|'

# Fold eps in after the reciprocal instead of before the square root.
mutant norm_eps_after caught \
    's|    return 1.0f / sqrtf(total / (float)group + eps);|    return 1.0f / (sqrtf(total / (float)group) + eps);|'

# Drop MLX'"'"'s cast of the normalized value to the activation dtype.
mutant norm_no_bf16 caught \
    's|    if (round_bf16) normed = qwen4exp_round_bf16(normed);|    if (0) normed = qwen4exp_round_bf16(normed);|'

# Move the weight multiply to before the bf16 cast: same algebra, one rounding
# point moved, which is the failure this whole exercise exists to catch.
mutant norm_weight_before_cast caught \
    's|    return normed \* (weight_bias + w);|    return qwen4exp_round_bf16(x * scale * (weight_bias + w)) \* 1.0f;|'

# The zero-centered checkpoints'"'"' offset.
mutant norm_no_weight_bias caught \
    's|    return normed \* (weight_bias + w);|    return normed * w;|'

# One statistic for the whole token instead of one per stream: publish stream
# 0'"'"'s scale for every stream.
mutant norm_scale_shared caught \
    's|    if (threadIdx.x == 0u) nscale\[(uint64_t)row \* (n / group) + g\] = scale;|    if (threadIdx.x == 0u \&\& g == 0u) { for (uint32_t q = 0; q < n / group; q++) nscale[(uint64_t)row * (n / group) + q] = scale; }|'

# ---- the quantize, and the fast-math seam ----------------------------------

# The correctly rounded divide, which is what this translation unit'"'"'s flags
# would give and is NOT what ds4_cuda.cu'"'"'s --use_fast_math build does.
mutant q8_precise_divide caught \
    's|    const float d = qwen4exp_q8_ftz(a \* QWEN4EXP_Q8_RCP127);|    const float d = qwen4exp_q8_ftz(a / 127.0f);|'

# The correctly rounded reciprocal instead of MUFU.RCP.
mutant q8_precise_reciprocal caught \
    's|    const float id = d != 0.0f ? qwen4exp_q8_rcp_approx(d) : 0.0f;|    const float id = d != 0.0f ? __frcp_rn(d) : 0.0f;|'

# Truncate instead of round to nearest.
mutant q8_truncate caught \
    's|        int q = (int)lrintf(qwen4exp_q8_ftz(vz \* id));|        int q = (int)(qwen4exp_q8_ftz(vz * id));|'

# Address the Q8_0 block by the warp before the loop step rather than after:
# every value is quantized correctly and lands in the wrong block.
mutant q8_block_index caught \
    's|        const uint64_t pair = blk0 + (uint64_t)(k \* warps + warp);|        const uint64_t pair = blk0 + (uint64_t)(warp * (group / blockDim.x) + k);|'

# ---- the mix and the inject ------------------------------------------------

# Reduce the streams high to low.  The values are the same and the sum is not.
mutant mix_stream_order caught \
    's|        const uint64_t idx = row + (uint64_t)h \* n_embd;|        const uint64_t idx = row + (uint64_t)(n_hc - 1u - h) * n_embd;|'

# Two roundings where the kernel has one FFMA.
mutant mix_no_fma caught \
    's|        acc += qwen4exp_sigmoid(wide\[idx\]) \* normed;|        acc = __fadd_rn(acc, __fmul_rn(qwen4exp_sigmoid(wide[idx]), normed));|'

# The sigmoid the mix reduction folds in.
mutant mix_no_sigmoid caught \
    's|        acc += qwen4exp_sigmoid(wide\[idx\]) \* normed;|        acc += wide[idx] * normed;|'

# The inject head'"'"'s factor of two.
mutant inject_no_two caught \
    's|            2.0f \* qwen4exp_sigmoid(total \* (1.0f / (float)n_hc));|            qwen4exp_sigmoid(total * (1.0f / (float)n_hc));|'

# Walk the inject dot channel-outer instead of stream-outer: the same 10240
# products in a different order, which is the classic silent reassociation.
mutant inject_dot_order caught \
    's|            const uint32_t i = hs \* n_embd + k + threadIdx.x;|            const uint32_t i = ((k + threadIdx.x) % n_embd) * n_hc + hs;|'

# The fused mix/inject kernel is only reached above the row threshold, so it
# needs its own mutants.
mutant fused_mix_no_fma caught \
    's|            smix\[d\] += qwen4exp_sigmoid(gr\[i\]) \* normed;|            smix[d] = __fadd_rn(smix[d], __fmul_rn(qwen4exp_sigmoid(gr[i]), normed));|'

mutant fused_inject_row caught \
    's|                            iw + (uint64_t)ho \* weight_row_bytes, i);|                            iw, i);|'

# ---- the up+mix tile and the norm+inject pass ------------------------------

# The tile's K loop: split its FFMA into a rounded product and an add.
mutant wide_kloop_no_fma caught \
    's|acc\[mi\]\[ni\]\[0\] = qwen4exp_fma_ftz(qwen4exp_fmul_ftz(w0, xs\[mi\]\[0\]), (float)d\[0\], acc\[mi\]\[ni\]\[0\]);|acc[mi][ni][0] = __fadd_rn(acc[mi][ni][0], __fmul_rn(qwen4exp_fmul_ftz(w0, xs[mi][0]), (float)d[0]));|'

# Scale the dot first and the weight after: the same three factors, one
# rounding point moved.
mutant wide_kloop_scale_order caught \
    's|acc\[mi\]\[ni\]\[1\] = qwen4exp_fma_ftz(qwen4exp_fmul_ftz(w1, xs\[mi\]\[0\]), (float)d\[1\], acc\[mi\]\[ni\]\[1\]);|acc[mi][ni][1] = qwen4exp_fma_ftz(w1, qwen4exp_fmul_ftz(xs[mi][0], (float)d[1]), acc[mi][ni][1]);|'

# The mix in the epilogue, streams high to low.
mutant wide_mix_stream_order caught \
    's|            for (int h = 0; h < NT; h++) {|            for (int h = NT - 1; h >= 0; h--) {|'

# Two roundings where the mix has one FFMA.
mutant wide_mix_no_fma caught \
    's|                mix = __fmaf_rn(qwen4exp_sigmoid(acc\[mi\]\[h\]\[e\]), normed, mix);|                mix = __fadd_rn(mix, __fmul_rn(qwen4exp_sigmoid(acc[mi][h][e]), normed));|'

# The sigmoid must see the whole projection: apply it to the accumulator one
# stage early by dropping the last stage'"'"'s contribution from it.
mutant wide_mix_partial_sigmoid caught \
    's|                mix = __fmaf_rn(qwen4exp_sigmoid(acc\[mi\]\[h\]\[e\]), normed, mix);|                mix = __fmaf_rn(qwen4exp_sigmoid(acc[mi][h][e] * 0.999f), normed, mix);|'

# The inject dot in the norm pass, streams high to low: every quantized byte
# is still right, only the dot'"'"'s order moves.
mutant wide_inject_stream_order caught \
    's|    for (uint32_t g = 0; g < n_hc; g++) {|    for (uint32_t g = n_hc; g-- > 0;) {|'

# The inject accumulate as a rounded product and an add: the FFMA's addend
# becomes zero and the accumulator is added after.
mutant wide_inject_no_fma caught \
    's|                            iacc\[ho\]);|                            0.0f) + iacc[ho];|'

# ---- deliberate no-ops -----------------------------------------------------

# The tile shape does not enter the tile'"'"'s arithmetic: run the wide shape
# at every width, then the narrow one at every width.
mutant wide_tile_always_big survives \
    's|            if (rows <= 64u) {|            if (rows <= 0u) {|'

mutant wide_tile_always_small survives \
    's|            if (rows <= 64u) {|            if (rows <= 0xffffffffu) {|'


# fmaxf is exact and associative over the finite non-negative values the
# butterfly sees, so reversing it is the same number.
mutant q8_butterfly_reversed survives \
    's|        for (int off = 16; off > 0; off >>= 1) {|        for (int off = 1; off <= 16; off <<= 1) {|'

# 1/n_hc is a power of two here, so the multiply and the divide agree exactly.
mutant mix_divide_by_n_hc survives \
    's|    out\[(uint64_t)t \* n_embd + d\] = acc \* (1.0f / (float)n_hc);|    out[(uint64_t)t * n_embd + d] = acc / (float)n_hc;|'

# The row threshold picks WHICH kernel computes the mix and the inject; the two
# were written to agree bit for bit, and this asserts that they do.
mutant fuse_threshold_one survives \
    's|#define QWEN4EXP_HC_FUSE_MIX_MIN_ROWS 48u|#define QWEN4EXP_HC_FUSE_MIX_MIN_ROWS 1u|'

mutant fuse_threshold_never survives \
    's|#define QWEN4EXP_HC_FUSE_MIX_MIN_ROWS 48u|#define QWEN4EXP_HC_FUSE_MIX_MIN_ROWS 0xffffffffu|'

if [ "$failures" -gt 0 ]; then
    echo "qwen4exp HC fuse mutants: $failures mutant(s) did not behave as required" >&2
    exit 1
fi
echo "qwen4exp HC fuse mutants: PASS"
