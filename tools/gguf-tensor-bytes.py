#!/usr/bin/env python3
"""gguf-tensor-bytes.py -- report the byte size of ONE named tensor.

The size comes from the GGUF tensor index. It is never guessed and never
hardcoded. Only the headers are read: the tensor index sits at the front of
each shard, so no weight byte is touched and nothing is loaded.

WHY THIS TOOL EXISTS. The memory plan in tools/serve-up.sh must know how much
of the artifact body becomes resident. For the qwen4exp target the body is
103.69 GiB, but the per-layer n-gram table (per_layer_token_embd.weight) stays
mapped on the solid-state disk and is streamed. The engine subtracts it
(ds4_qwen4exp.inc: plan->ssd_bytes = plan->bytes[DS4_QWEN4EXP_MEM_PLE];
plan->resident_bytes = plan->total_bytes - plan->ssd_bytes) and every caller
that plans a load must subtract the same bytes. A caller that counts the table
refuses a load that fits. A caller that hardcodes the number gets a different
artifact wrong.

THE ALLOWED TYPES ARE AN ARGUMENT, NOT A DEFAULT. The caller names the ggml
types the engine streams for this tensor. A tensor of any other type is
REFUSED by name. The subtraction is only correct for a tensor the engine keeps
off the resident set, so a re-quantized artifact must stop the boot, not
silently shrink the plan.

Usage:
  gguf-tensor-bytes.py --first-shard PATH --tensor NAME --allow-type NAME [--allow-type NAME]...

Output:
  the tensor size in bytes, on stdout, one line.

Exit:
  0  the tensor was found, its type is allowed, and its size was printed
  2  a refusal, named on stderr (bad argument, unreadable artifact, missing
     tensor, or a type the caller does not allow)
"""

import argparse
import os
import re
import struct
import sys

# ggml type traits: id -> (name, elements per block, bytes per block).
#
# This is the ONE ggml type table in this repository. It mirrors the pinned
# engine's own table, ds4/cuda/mmq/ds4_ggml_stubs.h (ggml_type_size() and
# ggml_blck_size()), which is hand-aligned with llama.cpp ggml/src/ggml.c.
# Ids 4 and 5 (Q4_2, Q4_3) are deprecated upstream and are absent here. An id
# that is not in this table is REFUSED; it is never assumed to be dense.
GGML_TYPES = {
    0:  ("F32",     1,   4),
    1:  ("F16",     1,   2),
    2:  ("Q4_0",    32,  18),
    3:  ("Q4_1",    32,  20),
    6:  ("Q5_0",    32,  22),
    7:  ("Q5_1",    32,  24),
    8:  ("Q8_0",    32,  34),
    9:  ("Q8_1",    32,  36),
    10: ("Q2_K",    256, 84),
    11: ("Q3_K",    256, 110),
    12: ("Q4_K",    256, 144),
    13: ("Q5_K",    256, 176),
    14: ("Q6_K",    256, 210),
    15: ("Q8_K",    256, 292),
    16: ("IQ2_XXS", 256, 66),
    17: ("IQ2_XS",  256, 74),
    18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S",   256, 50),
    20: ("IQ4_NL",  32,  18),
    21: ("IQ3_S",   256, 110),
    22: ("IQ2_S",   256, 82),
    23: ("IQ4_XS",  256, 136),
    24: ("I8",      1,   1),
    25: ("I16",     1,   2),
    26: ("I32",     1,   4),
    27: ("I64",     1,   8),
    28: ("F64",     1,   8),
    29: ("IQ1_M",   256, 56),
    30: ("BF16",    1,   2),
}

TYPE_IDS_BY_NAME = {name: tid for tid, (name, _, _) in GGML_TYPES.items()}


def refuse(message):
    """Print a named refusal and stop. Nothing has been loaded."""
    print("gguf-tensor-bytes: REFUSE {}".format(message), file=sys.stderr)
    raise SystemExit(2)


def read_exact(handle, count):
    """Read exactly count bytes, or refuse."""
    data = handle.read(count)
    if len(data) != count:
        refuse("truncated-gguf: the header of {} ends early".format(handle.name))
    return data


def read_u32(handle):
    return struct.unpack("<I", read_exact(handle, 4))[0]


def read_u64(handle):
    return struct.unpack("<Q", read_exact(handle, 8))[0]


def read_str(handle):
    return read_exact(handle, read_u64(handle)).decode("utf-8", "replace")


def skip_value(handle, kind):
    """Step over one metadata value of the given GGUF value type."""
    if kind in (0, 1, 7):        # uint8, int8, bool
        read_exact(handle, 1)
    elif kind in (2, 3):         # uint16, int16
        read_exact(handle, 2)
    elif kind in (4, 5, 6):      # uint32, int32, float32
        read_exact(handle, 4)
    elif kind in (10, 11, 12):   # uint64, int64, float64
        read_exact(handle, 8)
    elif kind == 8:              # string
        read_str(handle)
    elif kind == 9:              # array
        element = read_u32(handle)
        for _ in range(read_u64(handle)):
            skip_value(handle, element)
    else:
        refuse("unknown-metadata-type: GGUF value type {} in {}".format(kind, handle.name))


def find_tensor(path, tensor_name):
    """Return (type id, element count) for tensor_name in path, or None."""
    with open(path, "rb") as handle:
        if read_exact(handle, 4) != b"GGUF":
            refuse("not-a-gguf: {} does not start with the GGUF magic".format(path))
        read_u32(handle)                       # format version
        n_tensors = read_u64(handle)
        n_kv = read_u64(handle)
        for _ in range(n_kv):
            read_str(handle)
            skip_value(handle, read_u32(handle))
        for _ in range(n_tensors):
            name = read_str(handle)
            dims = [read_u64(handle) for _ in range(read_u32(handle))]
            type_id = read_u32(handle)
            read_u64(handle)                   # offset into the data section
            if name == tensor_name:
                elements = 1
                for dim in dims:
                    elements *= dim
                return type_id, elements
    return None


def shard_family(first_shard):
    """Return every shard of the -NNNNN-of-NNNNN family the first shard names."""
    match = re.match(r"(.*)-(\d{5})-of-(\d{5})\.gguf$", os.path.basename(first_shard))
    if not match:
        return [first_shard]
    folder = os.path.dirname(first_shard) or "."
    stem, total = match.group(1), int(match.group(3))
    return [os.path.join(folder, "{}-{:05d}-of-{:05d}.gguf".format(stem, i, total))
            for i in range(1, total + 1)]


def main():
    parser = argparse.ArgumentParser(add_help=True, description=__doc__)
    parser.add_argument("--first-shard", required=True,
                        help="the artifact's first GGUF shard; the whole -of- family is scanned")
    parser.add_argument("--tensor", required=True, help="the tensor name to size")
    parser.add_argument("--allow-type", action="append", default=[], metavar="NAME",
                        help="a ggml type name the caller allows (repeatable; at least one)")
    args = parser.parse_args()

    if not args.allow_type:
        refuse("no-allowed-types: name at least one --allow-type; an unchecked type is a wrong plan")

    allowed = {}
    for name in args.allow_type:
        key = name.strip().upper()
        if key not in TYPE_IDS_BY_NAME:
            refuse("unknown-allowed-type: '{}' is not a ggml type this tool knows".format(name))
        allowed[TYPE_IDS_BY_NAME[key]] = key

    for path in shard_family(args.first_shard):
        if not os.path.exists(path):
            refuse("missing-shard: {} is not on disk".format(path))

    for path in shard_family(args.first_shard):
        found = find_tensor(path, args.tensor)
        if found is None:
            continue
        type_id, elements = found
        if type_id not in GGML_TYPES:
            refuse("unknown-tensor-type: {} in {} has ggml type id {}"
                   .format(args.tensor, path, type_id))
        type_name, block_elements, block_bytes = GGML_TYPES[type_id]
        if type_id not in allowed:
            refuse("tensor-type-not-allowed: {} in {} is {}, and the caller allows only {}"
                   .format(args.tensor, path, type_name, ", ".join(sorted(allowed.values()))))
        if elements % block_elements:
            refuse("partial-block: {} holds {} elements, which is not a whole number of {} blocks"
                   .format(args.tensor, elements, type_name))
        print(elements // block_elements * block_bytes)
        return

    refuse("tensor-absent: no {} in any shard of {}".format(args.tensor, args.first_shard))


if __name__ == "__main__":
    main()
