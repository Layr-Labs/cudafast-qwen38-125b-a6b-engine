#!/usr/bin/env bash
# Proves the F32 router projection's exactness sweep bites.
#
# run_router_f32_prefill_exact_case in tests/test_qwen4exp_moe.c claims that
# matmul_f32_warp_tile_kernel is BIT-IDENTICAL to the block tile it replaces,
# and that the top-10-of-512 expert selection driven by its logits is therefore
# unchanged.  A claim like that is only worth what its check catches, so this
# rewrites ONE line of the kernel at a time and requires the sweep to fail.
#
# Each mutant below perturbs the summation ORDER or the term set, which is the
# only thing that can move the bits: the terms themselves are fixed by the
# weight and the activation.  The two marked NO-OP are deliberate controls --
# floating-point addition is commutative, so swapping the operands of one add
# cannot change a single bit and the sweep is EXPECTED to pass on them.  They
# are here so the pass rate below is read as resolution, not as a gap.
#
# Usage: tests/qwen4exp_router_f32_mutants.sh   (run from .build/ds4/src)
set -u
SRC=ds4_cuda.cu
BAK=$(mktemp)
cp "$SRC" "$BAK"
restore() { cp "$BAK" "$SRC"; rm -f "$BAK"; }
trap restore EXIT

MUT_NAME=()
MUT_FROM=()
MUT_TO=()
MUT_EXPECT=()
add() { MUT_NAME+=("$1"); MUT_FROM+=("$2"); MUT_TO+=("$3"); MUT_EXPECT+=("$4"); }

# --- order of the reduction tree -------------------------------------------
add "reassociate A's second level" \
    'for (int c = 0; c < TN; c++) A[t][c] = A[t][c] + (t0[t][c] + t1[t][c]);' \
    'for (int c = 0; c < TN; c++) A[t][c] = (A[t][c] + t0[t][c]) + t1[t][c];' fail
add "reassociate B's second level" \
    'for (int c = 0; c < TN; c++) B[t][c] = B[t][c] + (t0[t][c] + t1[t][c]);' \
    'for (int c = 0; c < TN; c++) B[t][c] = (B[t][c] + t0[t][c]) + t1[t][c];' fail
add "shuffle tree climbs instead of descending" \
    'for (int d = 16; d > 0; d >>= 1) {' \
    'for (int d = 1; d < 32; d <<= 1) {' fail
add "one shuffle level dropped" \
    'for (int d = 16; d > 0; d >>= 1) {' \
    'for (int d = 16; d > 1; d >>= 1) {' fail

# --- which chains pair with which ------------------------------------------
add "pair 2 with 7 instead of 6" \
    'matmul_f32_warp_tile_pair<TM, TN, MC>(t0, t1, 2u, 6u, lane, mcnt, wr, xr);' \
    'matmul_f32_warp_tile_pair<TM, TN, MC>(t0, t1, 2u, 7u, lane, mcnt, wr, xr);' fail
add "A takes chain 1 instead of chain 4" \
    'matmul_f32_warp_tile_pair<TM, TN, MC>(t0, t1, 0u, 4u, lane, mcnt, wr, xr);' \
    'matmul_f32_warp_tile_pair<TM, TN, MC>(t0, t1, 0u, 1u, lane, mcnt, wr, xr);' fail

# --- the chain walk itself --------------------------------------------------
add "chain stride 32 instead of 256" \
    'const uint32_t ia = ia0 + 256u * (uint32_t)m;' \
    'const uint32_t ia = ia0 + 32u * (uint32_t)m;' fail
add "lane block instead of lane stride" \
    'const uint32_t ia0 = lane + 32u * ja;' \
    'const uint32_t ia0 = lane * 8u + ja;' fail
# The runtime chain-count arm (in_dim != 2560) and the specialised arm
# (in_dim == 2560) are separate code, so each needs its own mutant.  The first
# of these SURVIVED until run_router_f32_prefill_exact_case grew a 2048 -> 512
# shape: nothing in the sweep reached the runtime arm.
add "runtime chain one element short" \
    'const uint32_t mcnt = (uint32_t)(in_dim >> 8);' \
    'const uint32_t mcnt = (uint32_t)(in_dim >> 8) - 1u;' fail
add "specialised chain one element short" \
    'for (int m = 0; m < (MC > 0 ? MC : 1); m++) {' \
    'for (int m = 0; m < (MC > 1 ? MC - 1 : 1); m++) {' fail

# --- the rounding of the MAC ------------------------------------------------
add "split the FMA into multiply then add" \
    'a[t][c] += wa[c] * xa[t];
                    b[t][c] += wb[c] * xb[t];' \
    'a[t][c] += __fmul_rn(wa[c], xa[t]);
                    b[t][c] += __fmul_rn(wb[c], xb[t]);' fail

# --- who writes the answer --------------------------------------------------
add "lane 1 stores instead of lane 0" \
    'if (lane == 0u) {' 'if (lane == 1u) {' fail
add "row tiles overlap" \
    'const uint32_t row0 = blockIdx.y * (uint32_t)TM;' \
    'const uint32_t row0 = blockIdx.y * (uint32_t)(TM - 1);' fail
# The eight-warp block: its own row/column placement lines.
add "tile8 row groups overlap" \
    'const uint32_t row0 = (blockIdx.y * (uint32_t)WR + wr_i) * (uint32_t)TM;' \
    'const uint32_t row0 = (blockIdx.y * (uint32_t)WR + wr_i) * (uint32_t)(TM - 1);' fail
add "tile8 last row of each group dropped" \
    'const uint32_t take = n_rows - row0 < (uint32_t)TM ? n_rows - row0
                                                       : (uint32_t)TM;' \
    'const uint32_t take = n_rows - row0 < (uint32_t)TM ? n_rows - row0
                                                       : (uint32_t)TM - 1u;' fail
add "tile8 warp split swapped" \
    'const uint32_t wr_i = warp / (uint32_t)WC, wc_i = warp % (uint32_t)WC;' \
    'const uint32_t wr_i = warp % (uint32_t)WC, wc_i = warp / (uint32_t)WC;' fail

# --- controls: commutative, cannot move a bit -------------------------------
add "NO-OP: commute the final A + B" \
    'float s = A[t][c] + B[t][c];' 'float s = B[t][c] + A[t][c];' pass
add "NO-OP: commute the shuffle add" \
    's = s + __shfl_down_sync(0xffffffffu, s, d);' \
    's = __shfl_down_sync(0xffffffffu, s, d) + s;' pass
# Which of the two warps of a row group takes which column tile is a
# permutation of identical work: every tile is still computed by one warp.
add "NO-OP: tile8 swap the two column groups" \
    'const uint32_t tile = blockIdx.x * (uint32_t)WC + wc_i;' \
    'const uint32_t tile = blockIdx.x * (uint32_t)WC + (wc_i ^ 1u);' pass

caught=0; missed=0; control_ok=0; control_bad=0
for i in "${!MUT_NAME[@]}"; do
    cp "$BAK" "$SRC"
    python3 - "$SRC" "${MUT_FROM[$i]}" "${MUT_TO[$i]}" <<'PY'
import sys
path, frm, to = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(path).read()
n = s.count(frm)
if n == 0:
    sys.stderr.write("MUTANT ANCHOR NOT FOUND: %r\n" % frm[:60]); sys.exit(3)
open(path, 'w').write(s.replace(frm, to))
PY
    [ $? -ne 0 ] && { echo "SKIP (anchor): ${MUT_NAME[$i]}"; continue; }
    if ! make CUDA_ARCH="${CUDA_ARCH:-sm_121}" tests/test_qwen4exp_moe >/dev/null 2>&1; then
        echo "CAUGHT (build): ${MUT_NAME[$i]}"; caught=$((caught+1)); continue
    fi
    if ./tests/test_qwen4exp_moe --router-f32 >/dev/null 2>&1; then
        if [ "${MUT_EXPECT[$i]}" = pass ]; then
            echo "control passed as expected: ${MUT_NAME[$i]}"; control_ok=$((control_ok+1))
        else
            echo "SURVIVED: ${MUT_NAME[$i]}"; missed=$((missed+1))
        fi
    else
        if [ "${MUT_EXPECT[$i]}" = pass ]; then
            echo "CONTROL UNEXPECTEDLY FAILED: ${MUT_NAME[$i]}"; control_bad=$((control_bad+1))
        else
            echo "caught: ${MUT_NAME[$i]}"; caught=$((caught+1))
        fi
    fi
done
cp "$BAK" "$SRC"
make CUDA_ARCH="${CUDA_ARCH:-sm_121}" tests/test_qwen4exp_moe >/dev/null 2>&1
echo "router f32 mutants: $caught caught, $missed survived, $control_ok controls passed, $control_bad controls broke"
[ "$missed" -eq 0 ] && [ "$control_bad" -eq 0 ] && echo "qwen4exp router f32 mutants: PASS" && exit 0
echo "qwen4exp router f32 mutants: FAIL"; exit 1
