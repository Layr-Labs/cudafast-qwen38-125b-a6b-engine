#!/usr/bin/env python3
"""Check DS4_SHAPE_QWEN4EXP in ds4.c against the track fixture.

The loader refuses a GGUF whose metadata disagrees with DS4_SHAPE_QWEN4EXP.
That preset is therefore the fixture geometry, written in C.  This script keeps
the two honest: it parses the preset out of ds4.c and compares it field by
field with fixtures/qwen3_8_125b_a6b_track.json (target.*).

Conventions worth knowing, both verified against the shipped artifacts:

  * target.ple_layer_ids is ONE BASED ([2]).  The GGUF records the same block
    zero based ([1]).  Baekpica's file carries both spellings
    (ple.layer_ids_one_based=[2], ple.checkpoint_layer_ids_zero_based=[1]),
    which is what settles the convention.
  * target.quantization is a per-variant recipe, not geometry.  The loader
    accepts a SET of quantization types per tensor role instead, because the
    shipped files mix them per block.  It is therefore not compared here.

usage: qwen4exp_fixture_check.py [ds4.c] [fixture.json]
"""

import json
import re
import sys

# preset field -> fixture field, with an optional transform
FIELDS = [
    ("n_layer",                  "num_hidden_layers",              None),
    ("n_embd",                   "hidden_size",                    None),
    ("n_vocab",                  "vocab_size",                     None),
    ("n_head",                   "num_attention_heads",            None),
    ("n_head_kv",                "num_key_value_heads",            None),
    ("n_head_dim",               "head_dim",                       None),
    ("n_value_dim",              "head_dim",                       None),
    ("n_expert",                 "num_experts",                    None),
    ("n_expert_used",            "num_experts_per_tok",            None),
    ("n_ff_exp",                 "moe_intermediate_size",          None),
    ("n_ff_shexp",               "shared_expert_intermediate_size", None),
    ("n_indexer_head",           "indexer_n_heads",                None),
    ("n_indexer_head_dim",       "indexer_head_dim",               None),
    ("n_indexer_top_k",          "indexer_budget",                 None),
    ("n_indexer_kv_head",        "indexer_kv_heads",               None),
    ("n_indexer_compress_ratio", "indexer_compress_ratio",         None),
    ("n_hc",                     "hc_count",                       None),
    ("n_hc_lowrank",             "hc_lowrank",                     None),
    ("n_full_attn_interval",     "full_attention_interval",        None),
    ("n_gdn_key_head",           "linear_num_key_heads",           None),
    ("n_gdn_value_head",         "linear_num_value_heads",         None),
    ("n_gdn_head_dim",           "linear_value_head_dim",          None),
    ("n_gdn_state",              "linear_key_head_dim",            None),
    ("n_gdn_conv",               "linear_conv_kernel_dim",         None),
    ("n_ngram",                  "ngram_size",                     None),
    ("n_ngram_heads_per",        "heads_per_ngram",                None),
    ("n_ple_conv",               "ple_conv_kernel_size",           None),
    ("n_ple_embd",               "ple_embed_dim",                  None),
    ("rope_orig_ctx",            "max_position_embeddings",        None),
    ("rope_freq_base",           "rope_theta",                     None),
    # ple_layer_ids is one based in the fixture and zero based in the loader.
    ("n_ple_layer",              "ple_layer_ids",                  lambda v: v[0] - 1),
]

DERIVED = [
    # preset field, fixture-derived expectation, why
    ("n_rot", lambda t: int(t["head_dim"] * t["partial_rotary_factor"]),
     "head_dim * partial_rotary_factor"),
    ("n_gdn_inner", lambda t: t["linear_num_value_heads"] * t["linear_value_head_dim"],
     "linear value heads * value head dim"),
]


def parse_preset(path):
    src = open(path).read()
    m = re.search(r"static const ds4_shape DS4_SHAPE_QWEN4EXP = \{(.*?)\n\};",
                  src, re.S)
    if not m:
        raise SystemExit("DS4_SHAPE_QWEN4EXP not found in %s" % path)
    out = {}
    for field, value in re.findall(r"\.(\w+)\s*=\s*([^,]+),", m.group(1)):
        value = value.strip()
        if value.endswith("f"):
            out[field] = float(value[:-1])
        elif re.fullmatch(r"-?\d+", value):
            out[field] = int(value)
        else:
            out[field] = value
    return out


def main():
    ds4c = sys.argv[1] if len(sys.argv) > 1 else "ds4.c"
    fixture = (sys.argv[2] if len(sys.argv) > 2 else
               "fixtures/qwen3_8_125b_a6b_track.json")

    preset = parse_preset(ds4c)
    target = json.load(open(fixture))["target"]

    if target.get("gguf_architecture") != "qwen4exp":
        raise SystemExit("fixture target.gguf_architecture is not qwen4exp")

    bad = 0
    for field, key, xform in FIELDS:
        if key not in target:
            print("MISSING fixture key %-34s (for %s)" % (key, field))
            bad += 1
            continue
        want = xform(target[key]) if xform else target[key]
        got = preset.get(field)
        if got is None:
            print("MISSING preset field %-32s" % field)
            bad += 1
        elif float(got) != float(want):
            print("MISMATCH %-26s preset=%s fixture.%s=%s"
                  % (field, got, key, want))
            bad += 1

    for field, derive, why in DERIVED:
        want = derive(target)
        got = preset.get(field)
        if got is None or int(got) != int(want):
            print("MISMATCH %-26s preset=%s derived=%s (%s)"
                  % (field, got, want, why))
            bad += 1

    checked = len(FIELDS) + len(DERIVED)
    if bad:
        print("\n%d of %d fields disagree with the fixture" % (bad, checked))
        return 1
    print("DS4_SHAPE_QWEN4EXP agrees with the fixture on all %d fields" % checked)
    return 0


if __name__ == "__main__":
    sys.exit(main())
