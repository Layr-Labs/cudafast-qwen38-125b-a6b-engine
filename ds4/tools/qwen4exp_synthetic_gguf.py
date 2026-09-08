#!/usr/bin/env python3
"""Write a reduced-layer synthetic qwen4exp GGUF split set.

Why this exists
---------------
The production artifact (unsloth/Qwen3.8-Flash-Next-GGUF, UD-Q4_K_XL) is
1224 tensors over 4 shards and 91 GiB, of which 28.8 GiB is the per-layer
n-gram table.  A 24 GB laptop cannot hold it, so the loader tests run against a
file that keeps every *per-layer* shape at production size and shrinks only
three things:

  * the block count          48  -> --layers (default 4)
  * the vocabulary       248320  -> --vocab  (default 4096)
  * the n-gram table rows 320,001,536 -> --ple-rows (default 200000)

The file declares ``qwen4exp.ds4.synthetic_reduced = true``.  That flag is what
lets the loader accept a smaller block count and vocabulary; a production
checkpoint must never carry it, and the loader prints a warning banner when it
sees it.  Everything else -- 2560 hidden, 512 experts x 640, 24/2 heads x 256,
GDN 16 key / 48 value heads x 128, indexer 4 x 128 top-2048, hyper connections
width 4 low-rank 320, n-gram size 3, PLE row width 160, and every quantization
type -- is exactly what the real file carries.

Disk cost
---------
At the defaults the file is about 6.3 GiB *logically* but is written sparse:
tensors larger than --dense-threshold get their length materialised without
their bytes, so the set costs a few MiB on disk.  A tensor that is written
gets a payload that DECODES to finite, small values (see "Payloads" below); a
tensor that is left as a hole reads back as zeros, which decodes to zero.  Pass
--dense to write every byte (slow, and the full disk cost), or raise
--dense-threshold to fill the tensors a forward test actually reads.

Payloads
--------
Every written byte is drawn from a generator seeded by the tensor name, so the
same file comes out of every run.  The integer fields of a quantized block are
uniform random; the f16 block scales and minimums are chosen for the block
layout so that a decoded weight comes out on the order of PAYLOAD_MAX.  F32 and
BF16
tensors get small normal values, and a tensor whose name carries "norm" gets
values near 1.0.

This matters: a block whose f16 scale is an arbitrary bit pattern decodes to an
infinity, the forward then produces NaN everywhere, and every bit-exactness or
reference check downstream compares NaN with NaN and passes.  The graph test
asserts that the logits and the pre-final-mixer rows are finite, which only
holds because of the ranges below.

How to shrink further
---------------------
  --layers 1        one GDN block only; no QSA block, no PLE block
  --layers 4        3 GDN + 1 QSA (the default; QSA lands on block 3 because
                    full_attention_interval is 4 and cannot be reduced)
  --vocab 256       token_embd and output become negligible
  --ple-rows 32     the n-gram table becomes a few KiB
  --shards 1        one file instead of a split set
  --experts N       shrink the routed experts.  This makes the file FAIL the
                    loader's expert_count check on purpose; it exists for the
                    "refuse on a mis-shaped tensor" test, not for a good file.

To grow instead: --layers 48 --vocab 248320 --ple-rows 320001536 writes a
logically production-sized set, still sparse.  On a laptop that file makes the
memory guard refuse with the real resident number, which is a useful check.

The MTP head
------------
``--mtp-head`` writes a SECOND, single-shard file beside the target set: the
head ds4 loads through ``--mtp`` (``ds4_engine_options.mtp_path``).  It holds
one extra block at index ``--layers``, always full attention, plus the six
``nextn.*`` tensors, and it carries no ``token_embd`` and no ``output``: it
declares ``nextn_shared_target_tensors`` and borrows the target's.  The block
is written with ``hc_*_inject`` at Q8_0, which is what the shipped head does
and what the target does not.

The real tokenizer
------------------
By default the file carries a placeholder vocabulary: 256 byte tokens and no
merges, so every prompt encodes one token per byte.  That is enough to open a
file, and useless for a question about how the ENGINE tokenizes a prompt, which
only the model's own table can answer.  ``--tokenizer PATH`` reads a HuggingFace
``tokenizer.json`` and writes its token table and merge list instead, at the
production vocabulary size.  The weights are still synthetic; a fixture built
this way answers tokenizer questions and nothing else:

  python3 tools/qwen4exp_synthetic_gguf.py --out DIR --shards 1 \
      --tokenizer /path/to/tokenizer.json
  ./ds4 -m DIR/qwen4exp-synthetic.gguf --raw-prompt --dump-tokens \
      --prompt-file PROMPT

``--dump-tokens`` reads the metadata alone, so the reduced layer count and the
sparse tensors never matter.

Fault injection for the refusal tests
-------------------------------------
  --omit-tensor NAME
  --retype-tensor NAME=TYPE
  --reshape-tensor NAME=d0,d1[,d2]
  --kv-set KEY=TYPE:VALUE          (override or add a scalar metadata key)
  --kv-set-array KEY=TYPE:v1,v2    (override or add an array metadata key)
  --kv-drop KEY
"""

import argparse
import json
import os
import random
import struct
import sys

# ---------------------------------------------------------------------------
# GGUF primitives
# ---------------------------------------------------------------------------

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3

(V_U8, V_I8, V_U16, V_I16, V_U32, V_I32, V_F32, V_BOOL, V_STRING, V_ARRAY,
 V_U64, V_I64, V_F64) = range(13)

# name -> (ggml type id, elements per block, bytes per block)
TYPES = {
    "F32":     (0,   1,   4),
    "F16":     (1,   1,   2),
    "Q4_0":    (2,  32,  18),
    "Q4_1":    (3,  32,  20),
    "Q5_0":    (6,  32,  22),
    "Q5_1":    (7,  32,  24),
    "Q8_0":    (8,  32,  34),
    "Q2_K":   (10, 256,  84),
    "Q3_K":   (11, 256, 110),
    "Q4_K":   (12, 256, 144),
    "Q5_K":   (13, 256, 176),
    "Q6_K":   (14, 256, 210),
    "IQ4_NL": (20,  32,  18),
    "IQ4_XS": (23, 256, 136),
    "BF16":   (30,   1,   2),
}


def type_bytes(type_name, elements):
    _, blk, nb = TYPES[type_name]
    return (elements + blk - 1) // blk * nb


# ---------------------------------------------------------------------------
# Payloads
#
# One tile of whole blocks is drawn per tensor and repeated over its length.
# The tile is deterministic in the tensor name and every decoded value is
# finite and small, which is what lets a forward test assert on the numbers.
# ---------------------------------------------------------------------------

PAYLOAD_MAX = 0.05          # the size a decoded weight is scaled to
TILE_BYTES = 256 * 1024     # tile length, rounded up to whole blocks


def f16(x):
    return struct.pack("<e", x)


def scale(rng, span):
    """An f16 block scale that keeps `span` integer units inside PAYLOAD_MAX."""
    return f16(PAYLOAD_MAX / span * rng.uniform(0.5, 1.0))


def offset(rng):
    """An f16 block minimum, itself the size of PAYLOAD_MAX."""
    return f16(rng.uniform(-PAYLOAD_MAX, PAYLOAD_MAX))


# One block per quantized type, in the ggml struct order, with the integer span
# each f16 scale has to cover:
#
#   Q4_0/Q4_1   4-bit weights, +-8 / 0..15
#   Q5_0/Q5_1   5-bit weights, +-16 / 0..31
#   Q8_0        8-bit weights, +-127
#   Q2_K        2-bit weights (0..3) x a 4-bit sub-scale (0..15)
#   Q3_K        3-bit weights (+-4) x a 6-bit signed sub-scale (+-32)
#   Q4_K/Q5_K   4/5-bit weights x a 6-bit sub-scale (0..63)
#   Q6_K        6-bit weights (+-32) x an int8 sub-scale (+-127)
#   IQ4_NL      a nibble into a table whose largest entry is 127
#   IQ4_XS      the same table x a 6-bit signed sub-scale (+-32)

def blk_q4_0(rng):
    return scale(rng, 8) + rng.randbytes(16)


def blk_q4_1(rng):
    return scale(rng, 15) + offset(rng) + rng.randbytes(16)


def blk_q5_0(rng):
    return scale(rng, 16) + rng.randbytes(4 + 16)


def blk_q5_1(rng):
    return scale(rng, 31) + offset(rng) + rng.randbytes(4 + 16)


def blk_q8_0(rng):
    return scale(rng, 127) + rng.randbytes(32)


def blk_q2_k(rng):
    return rng.randbytes(16 + 64) + scale(rng, 3 * 15) + scale(rng, 15)


def blk_q3_k(rng):
    return rng.randbytes(32 + 64 + 12) + scale(rng, 4 * 32)


def blk_q4_k(rng):
    return scale(rng, 15 * 63) + scale(rng, 63) + rng.randbytes(12 + 128)


def blk_q5_k(rng):
    return scale(rng, 31 * 63) + scale(rng, 63) + rng.randbytes(12 + 32 + 128)


def blk_q6_k(rng):
    return rng.randbytes(128 + 64 + 16) + scale(rng, 32 * 127)


def blk_iq4_nl(rng):
    return scale(rng, 127) + rng.randbytes(16)


def blk_iq4_xs(rng):
    return scale(rng, 127 * 32) + rng.randbytes(2 + 4 + 128)


BLOCKS = {
    "Q4_0": blk_q4_0, "Q4_1": blk_q4_1,
    "Q5_0": blk_q5_0, "Q5_1": blk_q5_1,
    "Q8_0": blk_q8_0,
    "Q2_K": blk_q2_k, "Q3_K": blk_q3_k, "Q4_K": blk_q4_k,
    "Q5_K": blk_q5_k, "Q6_K": blk_q6_k,
    "IQ4_NL": blk_iq4_nl, "IQ4_XS": blk_iq4_xs,
}


def float_tile(rng, name, count, pack):
    """A tile of `count` unquantized values: near 1.0 for a norm weight, small
    either side of zero for everything else."""
    if "norm" in name:
        values = [1.0 + rng.uniform(-PAYLOAD_MAX, PAYLOAD_MAX) for _ in range(count)]
    else:
        values = [rng.uniform(-PAYLOAD_MAX, PAYLOAD_MAX) for _ in range(count)]
    return b"".join(pack(v) for v in values)


def bf16(x):
    """The top two bytes of the f32, which is what a bf16 store keeps."""
    return struct.pack("<f", x)[2:]


def payload_tile(name, type_name):
    """A whole number of blocks, at least TILE_BYTES long, to repeat over a
    tensor's length.  Seeded by the tensor name, so it is the same every run."""
    rng = random.Random(name)
    _, _, block_bytes = TYPES[type_name]
    blocks = max(1, (TILE_BYTES + block_bytes - 1) // block_bytes)
    if type_name == "F32":
        return float_tile(rng, name, blocks, lambda v: struct.pack("<f", v))
    if type_name == "F16":
        return float_tile(rng, name, blocks, f16)
    if type_name == "BF16":
        return float_tile(rng, name, blocks, bf16)
    build = BLOCKS[type_name]
    return b"".join(build(rng) for _ in range(blocks))


def w_str(s):
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def w_kv(key, vtype, value):
    out = w_str(key) + struct.pack("<I", vtype)
    if vtype == V_U16:
        out += struct.pack("<H", value)
    elif vtype == V_U32:
        out += struct.pack("<I", value)
    elif vtype == V_I32:
        out += struct.pack("<i", value)
    elif vtype == V_U64:
        out += struct.pack("<Q", value)
    elif vtype == V_F32:
        out += struct.pack("<f", value)
    elif vtype == V_BOOL:
        out += struct.pack("<B", 1 if value else 0)
    elif vtype == V_STRING:
        out += w_str(value)
    else:
        raise ValueError("unsupported scalar KV type %d" % vtype)
    return out


def w_kv_array(key, item_type, values):
    # Collected and joined rather than appended to: a real tokenizer array is a
    # quarter of a million entries, and growing one bytes object per entry is
    # quadratic -- the write never finishes.
    parts = [w_str(key), struct.pack("<I", V_ARRAY),
             struct.pack("<I", item_type), struct.pack("<Q", len(values))]
    for v in values:
        if item_type == V_I32:
            parts.append(struct.pack("<i", v))
        elif item_type == V_U32:
            parts.append(struct.pack("<I", v))
        elif item_type == V_U64:
            parts.append(struct.pack("<Q", v))
        elif item_type == V_STRING:
            parts.append(w_str(v))
        else:
            raise ValueError("unsupported array item type %d" % item_type)
    return b"".join(parts)


def align_up(x, n):
    return (x + n - 1) // n * n


# ---------------------------------------------------------------------------
# Geometry.  These are the production numbers; only the three reducible ones
# come from the command line.
# ---------------------------------------------------------------------------

N_EMBD = 2560
N_HEAD = 24
N_HEAD_KV = 2
HEAD_DIM = 256
N_ROT = 64
CONTEXT = 262144
ROPE_BASE = 1.0e7
RMS_EPS = 1e-6

# The production vocabulary.  --tokenizer emits the real token table, which
# only lines up with the loader's shape at this size, so it sets --vocab here.
N_VOCAB_FULL = 248320
# The model's end-of-sequence id, used when a tokenizer.json carries no
# <|endoftext|> entry to read it from.
EOS_TOKEN_ID = 248044

N_EXPERT_DEFAULT = 512
N_EXPERT_USED = 10
N_FF_EXP = 640
N_FF_SHEXP = 640

FULL_ATTENTION_INTERVAL = 4

GDN_KEY_HEAD = 16
GDN_VALUE_HEAD = 48
GDN_HEAD_DIM = 128
GDN_CONV = 4
GDN_STATE = 128
GDN_INNER = GDN_VALUE_HEAD * GDN_HEAD_DIM              # 6144
GDN_QKV = (2 * GDN_KEY_HEAD + GDN_VALUE_HEAD) * GDN_HEAD_DIM  # 10240

IDX_HEAD = 4
IDX_KV_HEAD = 1
IDX_HEAD_DIM = 128
IDX_TOP_K = 2048
IDX_COMPRESS = 4

N_HC = 4
HC_LOWRANK = 320
HC_DIM = N_HC * N_EMBD                                  # 10240

NGRAM = 3
NGRAM_HEADS_PER = 8
PLE_HEADS = 16
PLE_ROW_DIM = 160
PLE_CONV = 4
PLE_EMBD = 2560
PLE_LAYER = 1                                           # zero based

QSA_Q_DIM = N_HEAD * HEAD_DIM * 2                       # 12288
QSA_KV_DIM = N_HEAD_KV * HEAD_DIM                       # 512


def is_full_attention(il):
    return (il + 1) % FULL_ATTENTION_INTERVAL == 0


def build_tensors(args):
    """Return [(name, type_name, dims)] in the order the real file uses."""
    n_exp = args.experts
    out = []

    # Non-layer tensors first, as in shard 2 of the production artifact.
    out.append(("output.weight", "Q8_0", [N_EMBD, args.vocab]))
    out.append(("output_hc_down.weight", "Q8_0", [HC_DIM, HC_LOWRANK]))
    out.append(("output_hc_norm.weight", "F32", [HC_DIM]))
    out.append(("output_hc_up.weight", "Q8_0", [HC_LOWRANK, HC_DIM]))
    out.append(("per_layer_token_embd.weight", "IQ4_NL",
                [PLE_ROW_DIM, args.ple_rows]))
    out.append(("token_embd.weight", "Q8_0", [N_EMBD, args.vocab]))

    for il in range(args.layers):
        p = "blk.%d." % il
        if is_full_attention(il):
            out.append((p + "attn_q.weight", "Q8_0", [N_EMBD, QSA_Q_DIM]))
            out.append((p + "attn_k.weight", "Q8_0", [N_EMBD, QSA_KV_DIM]))
            out.append((p + "attn_v.weight", "Q8_0", [N_EMBD, QSA_KV_DIM]))
            out.append((p + "attn_output.weight", "Q8_0",
                        [N_HEAD * HEAD_DIM, N_EMBD]))
            out.append((p + "attn_q_norm.weight", "F32", [HEAD_DIM]))
            out.append((p + "attn_k_norm.weight", "F32", [HEAD_DIM]))
            out.append((p + "indexer.q_proj.weight", "BF16",
                        [N_EMBD, IDX_HEAD * IDX_HEAD_DIM]))
            out.append((p + "indexer.k_proj.weight", "BF16",
                        [N_EMBD, IDX_KV_HEAD * IDX_HEAD_DIM]))
            out.append((p + "indexer.q_norm.weight", "F32", [IDX_HEAD_DIM]))
            out.append((p + "indexer.k_norm.weight", "F32", [IDX_HEAD_DIM]))
        else:
            out.append((p + "attn_qkv.weight", "Q8_0", [N_EMBD, GDN_QKV]))
            out.append((p + "attn_gate.weight", "Q8_0", [N_EMBD, GDN_INNER]))
            out.append((p + "ssm_out.weight", "Q8_0", [GDN_INNER, N_EMBD]))
            out.append((p + "ssm_conv1d.weight", "F32", [GDN_CONV, GDN_QKV]))
            out.append((p + "ssm_alpha.weight", "F32", [N_EMBD, GDN_VALUE_HEAD]))
            out.append((p + "ssm_beta.weight", "F32", [N_EMBD, GDN_VALUE_HEAD]))
            out.append((p + "ssm_a", "F32", [GDN_VALUE_HEAD]))
            out.append((p + "ssm_dt.bias", "F32", [GDN_VALUE_HEAD]))
            out.append((p + "ssm_norm.weight", "F32", [GDN_HEAD_DIM]))

        # A per-block mix, the way the shipped recipes mix.  UD-Q4_K_XL stores
        # gate/up at Q5_K on block 2 of 48 and down at Q8_0 on blocks 2, 4, 30,
        # 46 and 47; this set has 4 blocks, so it puts the Q5_K block last and
        # the Q8_0 down block first to keep both away from block 0's defaults.
        gate_up = "Q5_K" if il == args.layers - 1 else "Q4_K"
        down = "Q8_0" if il == 0 else "Q5_1"
        out.append((p + "ffn_gate_inp.weight", "F32", [N_EMBD, n_exp]))
        out.append((p + "ffn_gate_exps.weight", gate_up, [N_EMBD, N_FF_EXP, n_exp]))
        out.append((p + "ffn_up_exps.weight", gate_up, [N_EMBD, N_FF_EXP, n_exp]))
        out.append((p + "ffn_down_exps.weight", down, [N_FF_EXP, N_EMBD, n_exp]))
        out.append((p + "ffn_gate_inp_shexp.weight", "F32", [N_EMBD]))
        out.append((p + "ffn_gate_shexp.weight", "Q8_0", [N_EMBD, N_FF_SHEXP]))
        out.append((p + "ffn_up_shexp.weight", "Q8_0", [N_EMBD, N_FF_SHEXP]))
        out.append((p + "ffn_down_shexp.weight", "Q8_0", [N_FF_SHEXP, N_EMBD]))

        for side in ("attn", "ffn"):
            out.append((p + "hc_%s_down.weight" % side, "Q8_0", [HC_DIM, HC_LOWRANK]))
            out.append((p + "hc_%s_up.weight" % side, "Q8_0", [HC_LOWRANK, HC_DIM]))
            out.append((p + "hc_%s_inject.weight" % side, "F32", [HC_DIM, N_HC]))
            out.append((p + "hc_%s_norm.weight" % side, "F32", [HC_DIM]))

        if il == args.ple_layer:
            out.append((p + "ple_conv1d.weight", "F32", [PLE_CONV, HC_DIM]))
            out.append((p + "ple_key.weight", "Q8_0", [N_EMBD, HC_DIM]))
            out.append((p + "ple_value.weight", "Q8_0", [N_EMBD, PLE_EMBD]))
            out.append((p + "ple_norm_conv.weight", "F32", [HC_DIM]))
            out.append((p + "ple_norm_key.weight", "F32", [HC_DIM]))
            out.append((p + "ple_norm_query.weight", "F32", [HC_DIM]))

    return out


# The n-gram hash constants.  The engine does not trust a file's multipliers:
# it re-derives them at load and refuses a mismatch, because a wrong triple is
# otherwise silent -- every shape and byte count still validates.  A fixture
# that wants the real PLE path has to satisfy the same derivation, so it lives
# here rather than being hard-coded.  Keep in step with
# ds4_ple_derive_multipliers / ds4_ple_splitmix64 in ds4_qwen4exp_ple.c.
PLE_HASH_SEED = 1234
PLE_HASH_PRIME_1 = 10007
PLE_HASH_GAMMA = 0x9E3779B97F4A7C15
MASK64 = (1 << 64) - 1
INT64_MAX = (1 << 63) - 1


def splitmix64(v):
    v = (v + PLE_HASH_GAMMA) & MASK64
    z = v
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return z ^ (z >> 31)


def ple_multipliers(ngram_size, vocab_size, layer_ordinal, seed=PLE_HASH_SEED):
    """multipliers[i] = 2 * (splitmix64(base + GAMMA * (i+1)) % half) + 1.

    `half` is what keeps `token * multiplier` non-negative in signed int64, so
    the modulo that follows the mix is unambiguous."""
    half = max(1, (INT64_MAX // max(1, vocab_size)) // 2)
    base = (seed + PLE_HASH_PRIME_1 * layer_ordinal) & MASK64
    return [2 * (splitmix64((base + PLE_HASH_GAMMA * (i + 1)) & MASK64) % half) + 1
            for i in range(ngram_size)]


def _is_prime(v):
    if v < 2:
        return False
    if v % 2 == 0:
        return v == 2
    d = 3
    while d * d <= v:
        if v % d == 0:
            return False
        d += 2
    return True

def build_mtp_tensors(args):
    """The MTP head's tensors: one full-attention block at index --layers plus
    the nextn projections and norms.  No token_embd and no output."""
    il = args.layers
    p = "blk.%d." % il
    out = []
    out.append((p + "attn_q.weight", "Q8_0", [N_EMBD, QSA_Q_DIM]))
    out.append((p + "attn_k.weight", "Q8_0", [N_EMBD, QSA_KV_DIM]))
    out.append((p + "attn_v.weight", "Q8_0", [N_EMBD, QSA_KV_DIM]))
    out.append((p + "attn_output.weight", "Q8_0", [N_HEAD * HEAD_DIM, N_EMBD]))
    out.append((p + "attn_q_norm.weight", "F32", [HEAD_DIM]))
    out.append((p + "attn_k_norm.weight", "F32", [HEAD_DIM]))
    out.append((p + "indexer.q_proj.weight", "BF16",
                [N_EMBD, IDX_HEAD * IDX_HEAD_DIM]))
    out.append((p + "indexer.k_proj.weight", "BF16",
                [N_EMBD, IDX_KV_HEAD * IDX_HEAD_DIM]))
    out.append((p + "indexer.q_norm.weight", "F32", [IDX_HEAD_DIM]))
    out.append((p + "indexer.k_norm.weight", "F32", [IDX_HEAD_DIM]))

    out.append((p + "ffn_gate_inp.weight", "F32", [N_EMBD, args.experts]))
    out.append((p + "ffn_gate_exps.weight", "Q4_K", [N_EMBD, N_FF_EXP, args.experts]))
    out.append((p + "ffn_up_exps.weight", "Q4_K", [N_EMBD, N_FF_EXP, args.experts]))
    out.append((p + "ffn_down_exps.weight", "Q5_1", [N_FF_EXP, N_EMBD, args.experts]))
    out.append((p + "ffn_gate_inp_shexp.weight", "F32", [N_EMBD]))
    out.append((p + "ffn_gate_shexp.weight", "Q8_0", [N_EMBD, N_FF_SHEXP]))
    out.append((p + "ffn_up_shexp.weight", "Q8_0", [N_EMBD, N_FF_SHEXP]))
    out.append((p + "ffn_down_shexp.weight", "Q8_0", [N_FF_SHEXP, N_EMBD]))

    for side in ("attn", "ffn"):
        out.append((p + "hc_%s_down.weight" % side, "Q8_0", [HC_DIM, HC_LOWRANK]))
        out.append((p + "hc_%s_up.weight" % side, "Q8_0", [HC_LOWRANK, HC_DIM]))
        # Q8_0 here, F32 in the target: the shipped head really does differ.
        out.append((p + "hc_%s_inject.weight" % side, "Q8_0", [HC_DIM, N_HC]))
        out.append((p + "hc_%s_norm.weight" % side, "F32", [HC_DIM]))

    out.append((p + "nextn.eh_proj.weight", "Q8_0", [2 * N_EMBD, N_EMBD]))
    out.append((p + "nextn.enorm.weight", "F32", [N_EMBD]))
    out.append((p + "nextn.hnorm.weight", "F32", [HC_DIM]))
    out.append((p + "nextn.hc_head_down.weight", "Q8_0", [HC_DIM, HC_LOWRANK]))
    out.append((p + "nextn.hc_head_up.weight", "Q8_0", [HC_LOWRANK, HC_DIM]))
    out.append((p + "nextn.hc_head_norm.weight", "F32", [HC_DIM]))
    return out


def build_mtp_kvs(args):
    """The head's metadata.  It shares every geometry key with the target and
    adds the two the head validator reads; block_count counts the extra block,
    and the compress_ratios entry for it is 0 even though the block IS full
    attention, which is why the loader forces the QSA path there."""
    kv, arrays = build_kvs(args)
    kv = [e for e in kv if e[0] not in ("qwen4exp.block_count", "general.name")]
    kv.append(("general.name", V_STRING, "qwen4exp synthetic reduced MTP head"))
    kv.append(("qwen4exp.block_count", V_U32, args.layers + 1))
    kv.append(("qwen4exp.nextn_predict_layers", V_U32, 1))
    kv.append(("qwen4exp.nextn_shared_target_tensors", V_BOOL, True))
    arrays = [a for a in arrays if a[0] != "qwen4exp.attention.compress_ratios"]
    arrays.append(("qwen4exp.attention.compress_ratios", V_I32,
                   [IDX_COMPRESS if is_full_attention(i) else 0
                    for i in range(args.layers)] + [0]))
    return kv, arrays



# ---------------------------------------------------------------------------
# Tokenizer
#
# The engine's vocab_load() requires tokenizer.ggml.tokens and .merges before
# it will open a file at all, and the CLI cannot turn a prompt into ids without
# them -- which is why a fixture with no tokenizer stops at "GGUF tokenizer
# token table is missing or invalid" however good the weights are.
#
# A byte-level GPT-2 vocabulary is enough and is what this family uses: the
# engine maps each raw byte to a printable codepoint (gpt2_byte_to_codepoint in
# ds4.c) and looks the resulting string up.  With all 256 byte tokens present
# and NO merges, every prompt tokenizes, one token per byte, and nothing can
# fail to encode.  The remaining ids are distinct placeholders so the table is
# the vocabulary size the shape preset fixes.
# ---------------------------------------------------------------------------


def gpt2_byte_to_codepoint(b):
    """The engine's mapping, mirrored: keep printable bytes, move the rest
    above 256 in the order they are skipped."""
    if (33 <= b <= 126) or (161 <= b <= 172) or b >= 174:
        return b
    n = 0
    for x in range(256):
        if (33 <= x <= 126) or (161 <= x <= 172) or x >= 174:
            continue
        if x == b:
            return 256 + n
        n += 1
    return b


def build_tokenizer(vocab_size, eos_id):
    """256 byte tokens, then placeholders, with the end-of-sequence token at
    `eos_id` -- the same id the n-gram history is filled with, so the PLE rule
    and the sampler agree on what ends a sequence."""
    if vocab_size < 257:
        raise SystemExit("--vocab must be at least 257 to hold the byte tokens")
    tokens = [chr(gpt2_byte_to_codepoint(b)) for b in range(256)]
    tokens += ["<tok%d>" % i for i in range(256, vocab_size)]
    tokens[eos_id] = "<|endoftext|>"
    return tokens


def load_hf_tokenizer(path, vocab_size):
    """The model's OWN tokenizer, read out of a HuggingFace tokenizer.json into
    the two arrays vocab_load() reads.

    The placeholder vocabulary above encodes one token per byte, which is
    enough to prove the loader accepts a file but produces a token stream no
    real run would ever see.  A question about how the engine tokenizes a
    given prompt can only be answered against the real table, so this reads it.

    Three details decide whether the arrays are right:

      * ids.  ``model.vocab`` is text -> id and ``added_tokens`` carries the
        specials at ids ABOVE it, so neither is in id order and neither alone
        covers the table.  Both are placed by id.
      * holes.  The declared vocabulary is larger than the ids the tokenizer
        file uses (248320 against 248077 here), and vocab_load() takes
        vocab->n_vocab from the LENGTH of the token array, so the unused ids
        must be present.  They are filled with distinct names nothing can
        tokenize to.
      * merges.  bpe_rank() joins the two halves with one space and looks THAT
        string up, so a merge entry is written pre-joined.  tokenizer.json
        writes either the joined string or the pair; both are accepted.

    The byte-level token spellings need no conversion: HuggingFace stores them
    in the same printable-codepoint form byte_encode() produces.

    Returns (tokens, merges, eos_id)."""
    with open(path, encoding="utf-8") as fp:
        spec = json.load(fp)

    model = spec.get("model") or {}
    vocab = model.get("vocab")
    merges_in = model.get("merges")
    if not isinstance(vocab, dict) or merges_in is None:
        raise SystemExit("%s is not a BPE tokenizer.json (need model.vocab and "
                         "model.merges)" % path)

    tokens = [None] * vocab_size

    def place(text, tid):
        if not 0 <= tid < vocab_size:
            raise SystemExit("tokenizer id %d for %r is outside the %d-entry "
                             "vocabulary" % (tid, text, vocab_size))
        tokens[tid] = text

    for text, tid in vocab.items():
        place(text, tid)
    for entry in spec.get("added_tokens") or []:
        place(entry["content"], entry["id"])

    for tid, text in enumerate(tokens):
        if text is None:
            tokens[tid] = "[PAD%d]" % tid

    merges = []
    for m in merges_in:
        merges.append(m if isinstance(m, str) else "%s %s" % (m[0], m[1]))

    # The end of sequence is a SPECIAL, so it is read from added_tokens and not
    # from the merge vocabulary, where the same spelling would be ordinary text.
    eos_id = EOS_TOKEN_ID
    for entry in spec.get("added_tokens") or []:
        if entry["content"] == "<|endoftext|>":
            eos_id = entry["id"]
            break
    return tokens, merges, eos_id


def ple_head_tables(rows):
    """Partition the n-gram table over PLE_HEADS heads.

    The reference takes the (global head index + 1)-th prime after a base, so
    successive head vocabularies are CONSECUTIVE PRIMES, and the engine refuses
    a table that is not -- the rule is checkable without knowing the base.  A
    fixture that split `rows` evenly therefore loaded but could never exercise
    the real n-gram path.

    Returns the offsets, the vocabularies, and the table height they sum to,
    which is what the caller must size the tensor by: `rows` is a target, and
    the primes decide the exact height."""
    start = max(2, rows // PLE_HEADS)
    vocabs, v = [], start
    while len(vocabs) < PLE_HEADS:
        if _is_prime(v):
            vocabs.append(v)
        v += 1
    offsets, run = [], 0
    for size in vocabs:
        offsets.append(run)
        run += size
    return offsets, vocabs, run


def build_kvs(args):
    # Settled once in main(); recomputing here would re-round the already
    # rounded height and produce a different, larger prime set.
    offsets, vocabs = args.ple_head_offsets, args.ple_head_vocabs
    kv = []
    kv.append(("general.architecture", V_STRING, "qwen4exp"))
    kv.append(("general.type", V_STRING, "model"))
    kv.append(("general.name", V_STRING, "qwen4exp synthetic reduced"))
    kv.append(("general.alignment", V_U32, args.alignment))
    kv.append(("general.quantization_version", V_U32, 2))

    # The flag that lets the loader accept a reduced block count and vocabulary.
    kv.append(("qwen4exp.ds4.synthetic_reduced", V_BOOL, True))

    kv.append(("qwen4exp.block_count", V_U32, args.layers))
    kv.append(("qwen4exp.context_length", V_U32, CONTEXT))
    kv.append(("qwen4exp.embedding_length", V_U32, N_EMBD))
    kv.append(("qwen4exp.vocab_size", V_U32, args.vocab))
    kv.append(("qwen4exp.attention.head_count", V_U32, N_HEAD))
    kv.append(("qwen4exp.attention.head_count_kv", V_U32, N_HEAD_KV))
    kv.append(("qwen4exp.attention.key_length", V_U32, HEAD_DIM))
    kv.append(("qwen4exp.attention.value_length", V_U32, HEAD_DIM))
    kv.append(("qwen4exp.attention.layer_norm_rms_epsilon", V_F32, RMS_EPS))
    kv.append(("qwen4exp.rope.freq_base", V_F32, ROPE_BASE))
    kv.append(("qwen4exp.rope.dimension_count", V_U32, N_ROT))
    kv.append(("qwen4exp.expert_count", V_U32, args.experts))
    kv.append(("qwen4exp.expert_used_count", V_U32, N_EXPERT_USED))
    kv.append(("qwen4exp.expert_feed_forward_length", V_U32, N_FF_EXP))
    kv.append(("qwen4exp.expert_shared_feed_forward_length", V_U32, N_FF_SHEXP))
    kv.append(("qwen4exp.full_attention_interval", V_U32, FULL_ATTENTION_INTERVAL))

    kv.append(("qwen4exp.ssm.conv_kernel", V_U32, GDN_CONV))
    kv.append(("qwen4exp.ssm.state_size", V_U32, GDN_STATE))
    kv.append(("qwen4exp.ssm.group_count", V_U32, GDN_KEY_HEAD))
    kv.append(("qwen4exp.ssm.time_step_rank", V_U32, GDN_VALUE_HEAD))
    kv.append(("qwen4exp.ssm.inner_size", V_U32, GDN_INNER))

    kv.append(("qwen4exp.attention.indexer.head_count", V_U32, IDX_HEAD))
    kv.append(("qwen4exp.attention.indexer.key_length", V_U32, IDX_HEAD_DIM))
    kv.append(("qwen4exp.attention.indexer.top_k", V_U32, IDX_TOP_K))

    kv.append(("qwen4exp.hyper_connection.count", V_U32, N_HC))
    kv.append(("qwen4exp.hyper_connection.low_rank", V_U32, HC_LOWRANK))

    kv.append(("qwen4exp.ple.ngram_size", V_U32, NGRAM))
    # The hash constants.  Without them the engine leaves have_hash false and
    # the PLE block refuses by name, which is right for a file that carries no
    # real n-gram table but leaves the kernels untested.  The ordinal is the
    # block's POSITION in qwen4exp.ple.layers, not its layer number.
    kv.append(("tokenizer.ggml.model", V_STRING, "gpt2"))
    kv.append(("tokenizer.ggml.bos_token_id", V_U32, 0))
    kv.append(("tokenizer.ggml.eos_token_id", V_U32, args.tokenizer_eos))
    kv.append(("qwen4exp.ple.seed", V_U32, PLE_HASH_SEED))
    kv.append(("qwen4exp.ple.eos_token_id", V_U32, args.ple_eos))
    kv.append(("qwen4exp.ple.heads_per_ngram", V_U32, NGRAM_HEADS_PER))
    kv.append(("qwen4exp.ple.conv_kernel", V_U32, PLE_CONV))
    kv.append(("qwen4exp.embedding_length_per_layer_input", V_U32, PLE_ROW_DIM))

    arrays = [
        ("qwen4exp.attention.compress_ratios", V_I32,
         [IDX_COMPRESS if is_full_attention(i) else 0 for i in range(args.layers)]),
        ("qwen4exp.ple.layers", V_I32, [args.ple_layer]),
        ("qwen4exp.ple.head_offsets", V_U64, offsets),
        ("qwen4exp.ple.head_vocab_sizes", V_U64, vocabs),
        ("qwen4exp.ple.layer_multipliers", V_U64,
         ple_multipliers(NGRAM, args.vocab, 0)),
        # --tokenizer settles both of these in main(); without it the byte
        # vocabulary stands and there are NO merges, because one token per byte
        # is already a complete, unambiguous encoding and a merge table would
        # only make the fixture's token stream depend on rules nothing here
        # tests.
        ("tokenizer.ggml.tokens", V_STRING, args.tokenizer_tokens),
        ("tokenizer.ggml.merges", V_STRING, args.tokenizer_merges),
    ]
    return kv, arrays


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def parse_kv_set(spec):
    key, rest = spec.split("=", 1)
    kind, value = rest.split(":", 1)
    kind = kind.upper()
    if kind == "U32":
        return (key, V_U32, int(value))
    if kind == "I32":
        return (key, V_I32, int(value))
    if kind == "U64":
        return (key, V_U64, int(value))
    if kind == "F32":
        return (key, V_F32, float(value))
    if kind == "BOOL":
        return (key, V_BOOL, value not in ("0", "false", "False"))
    if kind == "STRING":
        return (key, V_STRING, value)
    raise SystemExit("unknown KV kind %r (use U32/I32/U64/F32/BOOL/STRING)" % kind)


def parse_kv_set_array(spec):
    key, rest = spec.split("=", 1)
    kind, value = rest.split(":", 1)
    kind = kind.upper()
    items = [x for x in value.split(",") if x != ""]
    if kind == "I32":
        return (key, V_I32, [int(x) for x in items])
    if kind == "U32":
        return (key, V_U32, [int(x) for x in items])
    if kind == "U64":
        return (key, V_U64, [int(x) for x in items])
    raise SystemExit("unknown array kind %r (use I32/U32/U64)" % kind)


def apply_faults(tensors, args):
    omit = set(args.omit_tensor or [])
    retype = dict(s.split("=", 1) for s in (args.retype_tensor or []))
    reshape = dict(s.split("=", 1) for s in (args.reshape_tensor or []))

    out = []
    for name, tname, dims in tensors:
        if name in omit:
            continue
        if name in retype:
            tname = retype[name].upper()
            if tname not in TYPES:
                raise SystemExit("unknown tensor type %r" % tname)
        if name in reshape:
            dims = [int(x) for x in reshape[name].split(",")]
        out.append((name, tname, dims))

    for name in omit | set(retype) | set(reshape):
        if name not in {t[0] for t in tensors}:
            raise SystemExit("fault injection names an unknown tensor: %s" % name)
    return out


def shard_path(out_dir, stem, index, count):
    if count == 1:
        return os.path.join(out_dir, "%s.gguf" % stem)
    return os.path.join(out_dir, "%s-%05d-of-%05d.gguf" % (stem, index + 1, count))


def write_shard(path, kv_blob, n_kv, tensors, alignment, dense, threshold,
                verbose):
    """tensors: [(name, type_name, dims, nbytes, rel_offset)]"""
    header = GGUF_MAGIC + struct.pack("<I", GGUF_VERSION)
    header += struct.pack("<Q", len(tensors)) + struct.pack("<Q", n_kv)
    header += kv_blob

    directory = b""
    for name, tname, dims, nbytes, rel in tensors:
        directory += w_str(name) + struct.pack("<I", len(dims))
        for d in dims:
            directory += struct.pack("<Q", d)
        directory += struct.pack("<I", TYPES[tname][0]) + struct.pack("<Q", rel)

    data_start = align_up(len(header) + len(directory), alignment)

    with open(path, "wb") as f:
        f.write(header)
        f.write(directory)
        f.write(b"\0" * (data_start - f.tell()))

        total = 0
        for name, tname, dims, nbytes, rel in tensors:
            f.seek(data_start + rel)
            if nbytes == 0:
                continue
            if dense or nbytes <= threshold:
                # A deterministic payload that decodes to finite, small values,
                # repeated over the tensor.  Both halves matter: a non-zero
                # payload proves the mapping reaches the right offset, and a
                # DECODABLE one lets a forward test assert on the numbers.
                tile = payload_tile(name, tname)
                whole, tail = divmod(nbytes, len(tile))
                for _ in range(whole):
                    f.write(tile)
                if tail:
                    f.write(tile[:tail])
            else:
                # Sparse: materialise the length, not the bytes.  APFS, ext4 and
                # XFS all leave the skipped range as a hole.
                f.write(b"\0" * 64)
                f.seek(data_start + rel + nbytes - 1)
                f.write(b"\0")
            total = max(total, rel + nbytes)

        f.truncate(data_start + align_up(total, alignment))

    if verbose:
        st = os.stat(path)
        print("  %s: %d tensors, %.2f GiB logical, %.1f MiB on disk"
              % (os.path.basename(path), len(tensors),
                 st.st_size / 1024 ** 3,
                 st.st_blocks * 512 / 1024 ** 2))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--name", default="qwen4exp-synthetic", help="file stem")
    ap.add_argument("--layers", type=int, default=4,
                    help="block count (default 4: 3 GDN + 1 QSA)")
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--ple-rows", type=int, default=200000,
                    help="target n-gram table height; the real height is the "
                         "sum of the 16 consecutive primes at or above "
                         "rows/16, because the engine checks that rule")
    ap.add_argument("--ple-eos", type=int, default=None,
                    help="end-of-sequence token the n-gram history is filled "
                         "with (default: the last id in the vocabulary)")
    ap.add_argument("--tokenizer", metavar="PATH",
                    help="emit the model's real tokenizer from a HuggingFace "
                         "tokenizer.json instead of the one-token-per-byte "
                         "placeholder vocabulary.  Implies --vocab %d, the "
                         "only size the real table lines up with"
                         % N_VOCAB_FULL)
    ap.add_argument("--ple-layer", type=int, default=PLE_LAYER)
    ap.add_argument("--experts", type=int, default=N_EXPERT_DEFAULT)
    ap.add_argument("--shards", type=int, default=3)
    ap.add_argument("--alignment", type=int, default=32)
    ap.add_argument("--dense", action="store_true",
                    help="write every byte instead of a sparse file")
    ap.add_argument("--dense-threshold", type=int, default=4 * 1024 * 1024,
                    help="tensors up to this size are always written in full")
    ap.add_argument("--omit-tensor", action="append")
    ap.add_argument("--retype-tensor", action="append", metavar="NAME=TYPE")
    ap.add_argument("--reshape-tensor", action="append", metavar="NAME=d0,d1")
    ap.add_argument("--kv-set", action="append", metavar="KEY=TYPE:VALUE")
    ap.add_argument("--kv-set-array", action="append", metavar="KEY=TYPE:v1,v2")
    ap.add_argument("--kv-drop", action="append", metavar="KEY")
    ap.add_argument("--mtp-head", action="store_true",
                    help="also write <name>-mtp.gguf, the single-shard MTP head")
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args()

    if args.shards < 1:
        raise SystemExit("--shards must be at least 1")
    if args.layers < 1:
        raise SystemExit("--layers must be at least 1")

    # Settle the token table first: it fixes the vocabulary size, which
    # --ple-eos then defaults off.
    if args.tokenizer:
        args.vocab = N_VOCAB_FULL
        (args.tokenizer_tokens,
         args.tokenizer_merges,
         args.tokenizer_eos) = load_hf_tokenizer(args.tokenizer, args.vocab)
    else:
        args.tokenizer_tokens = None
        args.tokenizer_merges = []
        args.tokenizer_eos = None

    # The head vocabularies must be consecutive primes, so the table height is
    # their sum and not the requested target.  Settle it once, here, so the
    # tensor, the key-values and the report cannot disagree.
    if args.ple_eos is None:
        args.ple_eos = args.tokenizer_eos if args.tokenizer else args.vocab - 1
    if args.tokenizer_eos is None:
        # No real table: the byte vocabulary marks the n-gram end of sequence
        # as its own end of sequence, so the two ids stay the same one.
        args.tokenizer_eos = args.ple_eos
    if args.tokenizer_tokens is None:
        args.tokenizer_tokens = build_tokenizer(args.vocab, args.ple_eos)
    if not 0 <= args.ple_eos < args.vocab:
        raise SystemExit("--ple-eos must be inside the vocabulary")
    offsets, vocabs, height = ple_head_tables(args.ple_rows)
    args.ple_head_offsets = offsets
    args.ple_head_vocabs = vocabs
    args.ple_rows = height

    os.makedirs(args.out, exist_ok=True)
    verbose = not args.quiet

    tensors = apply_faults(build_tensors(args), args)
    kv, arrays = build_kvs(args)

    drop = set(args.kv_drop or [])
    overrides = [parse_kv_set(s) for s in (args.kv_set or [])]
    array_overrides = [parse_kv_set_array(s) for s in (args.kv_set_array or [])]
    override_keys = ({k for k, _, _ in overrides} |
                     {k for k, _, _ in array_overrides})

    kv = [e for e in kv if e[0] not in drop and e[0] not in override_keys]
    kv += overrides
    arrays = [a for a in arrays if a[0] not in drop and a[0] not in override_keys]
    arrays += array_overrides

    # Distribute tensors over shards.  Shard 0 is metadata only, exactly as the
    # production file is laid out; the rest carry an even slice of the table.
    if args.shards == 1:
        groups = [tensors]
    else:
        groups = [[]]
        data_shards = args.shards - 1
        per = (len(tensors) + data_shards - 1) // data_shards
        for i in range(data_shards):
            groups.append(tensors[i * per:(i + 1) * per])
        if any(len(g) == 0 for g in groups[1:]):
            raise SystemExit("--shards %d leaves an empty shard for %d tensors"
                             % (args.shards, len(tensors)))

    split_total = len(tensors)
    if verbose:
        print("qwen4exp synthetic: %d blocks, vocab %d, %d experts, "
              "%d PLE rows, %d tensors, %d shard(s)"
              % (args.layers, args.vocab, args.experts, args.ple_rows,
                 split_total, args.shards))

    paths = []
    for si, group in enumerate(groups):
        placed, rel = [], 0
        for name, tname, dims in group:
            elements = 1
            for d in dims:
                elements *= d
            nbytes = type_bytes(tname, elements)
            placed.append((name, tname, dims, nbytes, rel))
            rel = align_up(rel + nbytes, args.alignment)

        blob, n_kv = b"", 0
        if si == 0:
            for k, t, v in kv:
                blob += w_kv(k, t, v)
                n_kv += 1
            for k, t, v in arrays:
                blob += w_kv_array(k, t, v)
                n_kv += 1
        if args.shards > 1:
            # Same encoding llama.cpp's gguf-split writes: u16 / u16 / i32.
            blob += w_kv("split.no", V_U16, si)
            blob += w_kv("split.count", V_U16, args.shards)
            blob += w_kv("split.tensors.count", V_I32, split_total)
            n_kv += 3

        path = shard_path(args.out, args.name, si, args.shards)
        write_shard(path, blob, n_kv, placed, args.alignment, args.dense,
                    args.dense_threshold, verbose)
        paths.append(path)

    if args.mtp_head:
        mtp_tensors = apply_faults(build_mtp_tensors(args), args) \
            if (args.omit_tensor or args.retype_tensor or args.reshape_tensor) \
            else build_mtp_tensors(args)
        mtp_kv, mtp_arrays = build_mtp_kvs(args)
        mtp_kv = [e for e in mtp_kv if e[0] not in drop and e[0] not in override_keys]
        mtp_kv += overrides
        mtp_arrays = [a for a in mtp_arrays
                      if a[0] not in drop and a[0] not in override_keys]
        mtp_arrays += array_overrides

        placed, rel = [], 0
        for name, tname, dims in mtp_tensors:
            elements = 1
            for d in dims:
                elements *= d
            nbytes = type_bytes(tname, elements)
            placed.append((name, tname, dims, nbytes, rel))
            rel = align_up(rel + nbytes, args.alignment)
        blob, n_kv = b"", 0
        for k, t, v in mtp_kv:
            blob += w_kv(k, t, v)
            n_kv += 1
        for k, t, v in mtp_arrays:
            blob += w_kv_array(k, t, v)
            n_kv += 1
        mtp_path = os.path.join(args.out, "%s-mtp.gguf" % args.name)
        if verbose:
            print("qwen4exp synthetic MTP head: block %d, %d tensors, "
                  "shared target tensors" % (args.layers, len(placed)))
        write_shard(mtp_path, blob, n_kv, placed, args.alignment, args.dense,
                    args.dense_threshold, verbose)

    print(paths[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
