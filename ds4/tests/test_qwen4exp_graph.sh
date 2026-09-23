#!/bin/sh
# Build the reduced-layer synthetic qwen4exp GGUF set, run the end-to-end graph
# test against it, then check that the SHIPPED binary opens the same file
# through the production entry with no test hook anywhere.
#
# The files are written sparse, so the set costs a few hundred MiB on disk.
# Set DS4_QWEN4EXP_TEST_DIR to keep them.
#
# The set the forward RUNS on raises --dense-threshold so every tensor the
# tower reads carries a real payload instead of a hole.  That is what makes the
# numbers in this test mean something: a hole decodes to zero, and a tower of
# zeros passes a bit-exactness check without computing anything.  The routed
# expert slabs stay holes -- they are 6.3 of the set's 6.7 GiB, and the MoE
# kernel's numerics are tests/test_qwen4exp_moe's job.
set -e

ROOT=$(cd "$(dirname "$0")/.." && pwd)
WRITER="$ROOT/tools/qwen4exp_synthetic_gguf.py"
BIN="$ROOT/tests/test_qwen4exp_graph"
DS4="$ROOT/ds4"
PY=${PYTHON:-python3}

DIR=${DS4_QWEN4EXP_TEST_DIR:-}
KEEP=1
if [ -z "$DIR" ]; then
    DIR=$(mktemp -d "${TMPDIR:-/tmp}/qwen4exp-graph.XXXXXX")
    KEEP=0
fi

cleanup() {
    if [ "$KEEP" = "0" ]; then rm -rf "$DIR"; fi
}
trap cleanup EXIT

# The default recipe mixes types per block the way the shipped recipes do and
# upgrades one block's routed gate and up slabs to Q5_K.  The loader accepts
# that and the MoE kernel decodes it, so the forward must RUN on it.  The
# second set is the same file with that block back at Q4_K and with dense
# payloads, and it is the one the numeric checks run on.
#
# The third set is the SAME model written as ONE shard.  Every tensor's payload
# is seeded by its name, so the two sets differ only in which file each tensor
# lands in.  The forward over them must agree byte for byte: that is what says
# every weight is resolved against the mapping that actually holds it, and it
# is the only way a per-tensor slab mistake shows up, since reading the right
# offset out of the wrong shard yields plausible numbers and refuses nowhere.
mkdir -p "$DIR/mixed" "$DIR/q4k" "$DIR/q4k1"
"$PY" "$WRITER" --out "$DIR/mixed" --name qw4x --quiet >/dev/null
# EIGHT shards over ~113 tensors puts a boundary inside every block type: the
# hyper-connection group of blocks 1 and 2, the gated-delta-net group of BLOCK 2
# alone, and the expert group of blocks 0 and 3 each straddle a file boundary.
# Three shards left whole groups together, so a weight resolved through a
# sibling's mapping still read the right bytes and the mistake stayed invisible.
"$PY" "$WRITER" --out "$DIR/q4k" --name qw4x --quiet --shards 8 \
    --dense-threshold 67108864 \
    --retype-tensor blk.3.ffn_gate_exps.weight=Q4_K \
    --retype-tensor blk.3.ffn_up_exps.weight=Q4_K >/dev/null
"$PY" "$WRITER" --out "$DIR/q4k1" --name qw4x --quiet --shards 1 \
    --dense-threshold 67108864 \
    --retype-tensor blk.3.ffn_gate_exps.weight=Q4_K \
    --retype-tensor blk.3.ffn_up_exps.weight=Q4_K >/dev/null

# The MTP head, in its own directory: --mtp-head builds the head's tensor list
# too, and the blk.3 retypes above name a tower block the head does not have.
# Dense payloads, because a head block over zeroed weights would leave the
# stream untouched and the "it did something" check could not tell that from a
# block that ran and did nothing.
mkdir -p "$DIR/head"
"$PY" "$WRITER" --out "$DIR/head" --name qw4x --quiet --shards 1 --mtp-head \
    --dense-threshold 67108864 >/dev/null

"$BIN" "$DIR/q4k" "$DIR/q4k1"
"$BIN" --session "$DIR/q4k1" "$DIR/head"
"$BIN" --mixed "$DIR/mixed"

# The production entry: no DS4_TEST_HOOKS in this binary.  `ds4 -m` must reach
# the qwen4exp loader, validate and bind every tensor, print the memory plan and
# then say plainly that the forward is not wired into the session machinery.
# It exits non-zero at that point, so check the message, not the status.
echo
echo "PRODUCTION ENTRY: ds4 -m on the synthetic shard 1"
OUT=$("$DS4" -m "$DIR/mixed/qw4x-00001-of-00003.gguf" -p hi -n 1 2>&1 || true)
if printf '%s' "$OUT" | grep -q "qwen4exp memory plan"; then
    echo "  ok    ds4 -m reached the qwen4exp loader and bound every tensor"
else
    echo "  FAIL  ds4 -m did not reach the qwen4exp loader"
    printf '%s\n' "$OUT" | tail -20
    exit 1
fi

# What happens next depends on how much memory this host has free.  Either the
# open completes and says the forward is not wired into the session machinery,
# or the loader's guard refuses because free memory cannot hold the model plus
# 10 GiB of headroom.  Both are the production entry working; a silent success
# or a crash is not.
if printf '%s' "$OUT" | grep -q "qwen4exp model opened, validated and bound"; then
    echo "  ok    the production entry opened, validated and bound the model"
elif printf '%s' "$OUT" | grep -q "qwen4exp: refusing to load: free unified memory"; then
    echo "  ok    the production entry bound the model and the memory guard then"
    echo "        refused on this host's real free memory"
else
    echo "  FAIL  ds4 -m ended in neither of the two expected states"
    printf '%s\n' "$OUT" | tail -20
    exit 1
fi
