#!/bin/sh
# Build the synthetic qwen4exp GGUF set plus the fault-injected variants, then
# run tests/test_qwen4exp_loader against them.
#
# The files are written sparse, so the whole set costs a few hundred MiB on
# disk while being multi-GiB logically.  Set DS4_QWEN4EXP_TEST_DIR to keep them.
set -e

ROOT=$(cd "$(dirname "$0")/.." && pwd)
WRITER="$ROOT/tools/qwen4exp_synthetic_gguf.py"
BIN="$ROOT/tests/test_qwen4exp_loader"
PY=${PYTHON:-python3}

DIR=${DS4_QWEN4EXP_TEST_DIR:-}
KEEP=1
if [ -z "$DIR" ]; then
    DIR=$(mktemp -d "${TMPDIR:-/tmp}/qwen4exp-loader.XXXXXX")
    KEEP=0
fi

cleanup() {
    if [ "$KEEP" = "0" ]; then rm -rf "$DIR"; fi
}
trap cleanup EXIT

gen() {
    out=$1
    shift
    mkdir -p "$DIR/$out"
    "$PY" "$WRITER" --out "$DIR/$out" --name qw4x --quiet "$@" >/dev/null
}

echo "building synthetic qwen4exp files in $DIR"

# The good file: 4 blocks (3 GDN + 1 QSA), production per-layer shapes, plus
# the MTP head that goes with it (block 4, nextn tensors, shared target
# tensors, no token_embd and no output).
gen good --mtp-head

# Tensor-level faults.
gen missing   --omit-tensor blk.0.ssm_a
gen badtype   --retype-tensor blk.0.attn_qkv.weight=Q4_K
gen badshape  --reshape-tensor blk.0.attn_gate.weight=2560,6100
gen badple    --reshape-tensor per_layer_token_embd.weight=128,200000

# Metadata that disagrees with the fixture geometry baked into the shape preset.
gen badkv_hc        --kv-set qwen4exp.hyper_connection.low_rank=U32:256
gen badkv_experts   --kv-set qwen4exp.expert_count=U32:256
gen badkv_heads     --kv-set qwen4exp.attention.head_count=U32:16
gen badkv_interval  --kv-set qwen4exp.full_attention_interval=U32:8

# The PLE block id convention: the fixture records it one-based, the GGUF
# zero-based.  A file that carries both must keep them consistent.
gen badkv_ple_onebased --kv-set-array qwen4exp.ple.layer_ids_one_based=I32:3

# The n-gram head partition must sum to the table height.  The vocabularies are
# 16 consecutive primes, so the fault is the generator's own list with the LAST
# head enlarged: overriding the whole array with round numbers would disagree
# with head_offsets first and trip a different check.  Derived from the writer
# so it cannot drift when the prime rule moves.
OVERRUN=$("$PY" - "$WRITER" <<'PYEOF'
import importlib.util, sys
spec = importlib.util.spec_from_file_location("g", sys.argv[1])
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)
_, vocabs, _ = g.ple_head_tables(200000)
vocabs[-1] += 1000
print(",".join(str(v) for v in vocabs))
PYEOF
)
gen badkv_ple_rows --kv-set-array "qwen4exp.ple.head_vocab_sizes=U64:$OVERRUN"

# The recorded attention schedule must match the interval rule.
gen badkv_schedule --kv-set-array qwen4exp.attention.compress_ratios=I32:4,0,0,4

echo
"$BIN" "$DIR"

# The engine's own load order: the target fixes the geometry, then the head is
# validated and bound against it.
echo
"$BIN" --open-pair "$DIR/good/qw4x-00001-of-00003.gguf" "$DIR/good/qw4x-mtp.gguf"
