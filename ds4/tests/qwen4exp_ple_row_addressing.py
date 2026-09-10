#!/usr/bin/env python3
"""Check our PLE row addressing against gguf-py, on the real artifact.

Run this on a host that HAS the GGUF; it needs `pip install gguf`.

    python3 tests/qwen4exp_ple_row_addressing.py \
        --rows 15277278 319095437 \
        --shards /path/qwen38-*.gguf

What it checks, per row:
  1. WHICH SHARD and WHICH BYTE RANGE the row lands in.  gguf-py finds
     per_layer_token_embd.weight, its quantization type and its payload offset
     inside the shard that carries it; the row offset is then
     `row * row_dim / 32 * 18` for IQ4_NL.  Our reader is asked for the same
     row's bytes and the two byte strings must be equal.  On a multi-shard
     artifact this is the check that bites: reading the right offset out of the
     wrong shard yields plausible numbers and refuses nowhere.
  2. THE VALUES.  gguf-py dequantizes the bytes with its own IQ4_NL
     implementation; ours dequantizes with ds4_ple_dequant_iq4_nl.  Both are
     exact integer-table lookups scaled by one f16, so the two must agree to
     the bit -- this is not a tolerance comparison.

The C half is tests/test_qwen4exp_ple_row_addressing.c; build it with

    cc -O2 -std=c99 -I. -o /tmp/ple_addr \
        tests/test_qwen4exp_ple_row_addressing.c ds4_qwen4exp_ple.c -lm

and pass it with --tool, or let this script build it if `cc` is on the path.
"""

import argparse
import os
import subprocess
import sys


def load_gguf():
    try:
        import gguf  # noqa: F401
        import numpy  # noqa: F401
    except ImportError:
        sys.exit("this check needs gguf-py and numpy: pip install gguf numpy")
    import gguf
    return gguf


def find_tensor(gguf, shards, name):
    """The shard that carries `name`, and that tensor's reader entry."""
    for path in shards:
        reader = gguf.GGUFReader(path, "r")
        for tensor in reader.tensors:
            if tensor.name == name:
                return path, reader, tensor
    sys.exit("no shard carries %s" % name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, nargs="+", required=True)
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--tensor", default="per_layer_token_embd.weight")
    ap.add_argument("--tool", default=None,
                    help="the built C half; built into /tmp if omitted")
    args = ap.parse_args()

    gguf = load_gguf()
    path, reader, tensor = find_tensor(gguf, args.shards, args.tensor)

    qtype = tensor.tensor_type
    if qtype != gguf.GGMLQuantizationType.IQ4_NL:
        sys.exit("this check covers IQ4_NL only; the table is %s" % qtype.name)

    # IQ4_NL: 32 values per block, one f16 scale plus 16 packed bytes.
    block_elems, block_bytes = 32, 18
    row_dim = int(tensor.shape[0])
    if row_dim % block_elems:
        sys.exit("row_dim %d is not a whole number of IQ4_NL blocks" % row_dim)
    row_bytes = row_dim // block_elems * block_bytes
    rows_in_table = int(tensor.shape[1])

    # `tensor.data` is a memoryview onto the payload, so its base is exactly
    # what our reader has to resolve to.  Taking the row out of it keeps the
    # arithmetic gguf-py's rather than restating the header layout here.
    payload = memoryview(tensor.data).cast("B")
    if len(payload) != rows_in_table * row_bytes:
        sys.exit("gguf-py payload is %d bytes, the geometry says %d"
                 % (len(payload), rows_in_table * row_bytes))

    print("tensor      %s" % tensor.name)
    print("shard       %s" % path)
    print("type        %s" % qtype.name)
    print("row_dim     %d" % row_dim)
    print("rows        %d" % rows_in_table)
    print("row_bytes   %d" % row_bytes)

    tool = args.tool
    if not tool:
        tool = "/tmp/ple_addr_%d" % os.getpid()
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        subprocess.run(
            ["cc", "-O2", "-std=c99", "-I", root, "-o", tool,
             os.path.join(root, "tests",
                          "test_qwen4exp_ple_row_addressing.c"),
             os.path.join(root, "ds4_qwen4exp_ple.c"), "-lm"],
            check=True)

    out = subprocess.run(
        [tool] + [str(r) for r in args.rows] + ["--"] + args.shards,
        check=True, capture_output=True, text=True).stdout

    ours_bytes, ours_values = {}, {}
    for line in out.strip().split("\n"):
        parts = line.split()
        if len(parts) > 2 and parts[0] == "row" and parts[2] == "bytes":
            ours_bytes[int(parts[1])] = parts[3]
        elif len(parts) > 2 and parts[0] == "row" and parts[2] == "values":
            ours_values[int(parts[1])] = [float(v) for v in parts[3:]]
        else:
            print("ours        %s" % line)

    failures = 0
    for row in args.rows:
        if row >= rows_in_table:
            print("row %d is past the table (%d rows)" % (row, rows_in_table))
            failures += 1
            continue
        want = bytes(payload[row * row_bytes:(row + 1) * row_bytes]).hex()
        got = ours_bytes.get(row)
        same_bytes = (got == want)
        print("row %d bytes  %s" % (row, "MATCH" if same_bytes else "DIFFER"))
        if not same_bytes:
            print("   gguf-py %s" % want)
            print("   ours    %s" % got)
            failures += 1

        # gguf.quants.dequantize reshapes its argument, so it needs a numpy
        # array rather than a memoryview: newer gguf-py releases stopped
        # accepting the buffer directly.
        import numpy as np
        want_values = gguf.quants.dequantize(
            np.frombuffer(bytes(payload[row * row_bytes:(row + 1) * row_bytes]),
                          dtype=np.uint8),
            qtype).reshape(-1).tolist()
        got_values = ours_values.get(row, [])
        # Both sides are float32 values; ours arrives as a decimal string and
        # gguf-py's as a float32 widened to double, so comparing the doubles
        # exactly fails on the last digits of the decimal round trip.  Compare
        # them AS float32, which is what both actually are.
        same_values = (len(want_values) == len(got_values) and
                       all(np.float32(a) == np.float32(b)
                           for a, b in zip(want_values, got_values)))
        print("row %d values %s" % (row, "MATCH" if same_values else "DIFFER"))
        if not same_values:
            for i, (a, b) in enumerate(zip(want_values, got_values)):
                if np.float32(a) != np.float32(b):
                    print("   first difference at %d: gguf-py %.9g ours %.9g"
                          % (i, a, b))
                    break
            failures += 1

    print("FAIL (%d)" % failures if failures else "PASS")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
