#!/bin/sh
#
# Mutation proof for the eight-lane GDN prefill recurrence (octet kernel).
#
# tests/test_qwen4exp_gdn_value_reuse requires the octet kernel to agree with
# the retained single-row recurrence BYTE FOR BYTE over 13 widths, 3 head
# layouts, 4 input scales, eager and captured, on every output field (outputs,
# carried state, state snapshots).  This script breaks the octet kernel one
# line at a time and requires that check to notice.  Each mutant rebuilds one
# object plus a relink.  Run it with: make test-qwen4exp-gdn-octet-mutants
#
# The last mutant is a deliberate no-op (swapping the two operands of one
# commutative add) and must SURVIVE: it is the property the kernel's local
# butterfly levels rest on.

set -u

ROOT=$(cd "$(dirname "$0")/.." && pwd)
TEST="$ROOT/tests/test_qwen4exp_gdn_value_reuse"
SRC="$ROOT/ds4_cuda_qwen4exp.cu"

if [ ! -f "$SRC" ]; then
    echo "qwen4exp GDN octet mutants: $SRC is missing" >&2
    exit 1
fi
if ! command -v nvcc >/dev/null 2>&1 && [ ! -x "${CUDA_HOME:-/usr/local/cuda}/bin/nvcc" ]; then
    echo "qwen4exp GDN octet mutants: skipped, no CUDA toolchain"
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
        tests/test_qwen4exp_gdn_value_reuse >"$WORK/build.log" 2>&1
}

if ! build; then
    echo "qwen4exp GDN octet mutants: the unmutated tree does not build" >&2
    tail -n 20 "$WORK/build.log" >&2
    exit 1
fi
if "$TEST" >"$WORK/base.log" 2>&1; then
    echo "  baseline                           PASS (as required)"
else
    echo "  baseline                           FAIL -- the unmutated octet kernel does not agree" >&2
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
                "$(grep -E 'mismatch|error|failed' "$WORK/$name.log" | head -n 1)"
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

# Local butterfly levels paired the wrong way: (p0+p1)+(p2+p3) is the sum of
# the same four partials but not the tree the 32-lane butterfly takes.
mutant tree_pairs_wrong caught \
    's|        hk\[r\] = __fadd_rn(__fadd_rn(p\[0\], p\[2\]), __fadd_rn(p\[1\], p\[3\]));|        hk[r] = __fadd_rn(__fadd_rn(p[0], p[1]), __fadd_rn(p[2], p[3]));|'

# The dot product contracted from .x instead of .y first.
mutant dot4_x_first caught \
    '/^__device__ static __forceinline__ float qwen4exp_gdn_dot4_pinned(/,/^}/{
         s|    float acc = __fmul_rn(a.y, b.y);|    float acc = __fmul_rn(a.x, b.x);|;
         s|    acc = __fmaf_rn(a.x, b.x, acc);|    acc = __fmaf_rn(a.y, b.y, acc);|
     }'

# Segment shuffles starting at offset 8 cross into the neighbouring row group.
mutant shuffle_crosses_segment caught \
    '/^__device__ static __forceinline__ void qwen4exp_gdn_octet_sum(float v\[R\]) {/,/^}/{
         s|    for (int offset = 4; offset > 0; offset >>= 1) {|    for (int offset = 8; offset > 1; offset >>= 1) {|
     }'

# Delta contracted into one FMA instead of FSUB then FMUL.
mutant delta_contracted caught \
    's|        const float delta_v = __fmul_rn(__fsub_rn(o.v\[r\], hk\[r\]), beta);|        const float delta_v = __fmaf_rn(o.v[r], beta, -hk[r] * beta);|'

# Decay skipped on one column of one quad.
mutant decay_skips_column caught \
    '/^__device__ static __forceinline__ void qwen4exp_gdn_octet_step(/,/^}/{
         s|            h\[r\]\[m\].w = __fmul_rn(h\[r\]\[m\].w, g);|            h[r][m].w = __fmul_rn(h[r][m].w, m == 3u ? 1.0f : g);|
     }'

# Second register set consumed one token late: reads token t+1 operands for
# token t on every other trip.
mutant double_buffer_skew caught \
    's|            qwen4exp_gdn_octet_step<R>(h, ob, res);|            qwen4exp_gdn_octet_step<R>(h, oa, res);|'

# Deliberate no-op: IEEE addition is commutative, so swapping the two sides
# of one local butterfly add changes no bit.  Must survive.
mutant commute_local_add survives \
    's|        hk\[r\] = __fadd_rn(__fadd_rn(p\[0\], p\[2\]), __fadd_rn(p\[1\], p\[3\]));|        hk[r] = __fadd_rn(__fadd_rn(p[2], p[0]), __fadd_rn(p[1], p[3]));|'

if [ "$failures" -ne 0 ]; then
    echo "qwen4exp GDN octet mutants: $failures mutant(s) misbehaved" >&2
    exit 1
fi
echo "qwen4exp GDN octet mutants: every mutant behaved as expected"
