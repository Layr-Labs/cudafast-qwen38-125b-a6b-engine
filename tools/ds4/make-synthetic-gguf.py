#!/usr/bin/env python3
"""make-synthetic-gguf.py -- write a small GGUF file for the tools tests.

TEST SUPPORT ONLY. Nothing here is ever staged, scored, or linked into a
measured run. It exists so tools/test-serve-up-plan-memory.sh and
tools/test-ds4-resident.sh can drive the REAL memory plan against a REAL GGUF
tensor index with no artifact, no GPU, and no load.

The file holds a valid header and a valid tensor index. The data section is
zero padding, because every reader in this repository reads the index only.

Usage:
  make-synthetic-gguf.py --out FILE [--tensor NAME:TYPE:D1[xD2...]]... [--size-bytes N]

  --tensor      one index entry. TYPE is a ggml type name (for example
                IQ4_NL). Repeat the option for more tensors.
  --size-bytes  pad the file to exactly N bytes. The option refuses when the
                header and the index are already larger than N.

Exit:
  0  the file was written
  2  a refusal, named on stderr
"""

import argparse
import importlib.util
import os
import struct
import sys


def _load_reader():
    """Load tools/gguf-tensor-bytes.py as a module.

    The writer takes its type table from the READER. There is one ggml type
    table in this repository, so a test cannot drift from the tool it tests.
    """
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "gguf-tensor-bytes.py")
    spec = importlib.util.spec_from_file_location("gguf_tensor_bytes", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_READER = _load_reader()


def refuse(message):
    print("make-synthetic-gguf: REFUSE {}".format(message), file=sys.stderr)
    raise SystemExit(2)


def gguf_string(text):
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def main():
    parser = argparse.ArgumentParser(add_help=True, description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tensor", action="append", default=[], metavar="NAME:TYPE:DIMS")
    parser.add_argument("--size-bytes", type=int, default=None)
    args = parser.parse_args()

    body = b""
    offset = 0
    for spec in args.tensor:
        parts = spec.split(":")
        if len(parts) != 3:
            refuse("bad-tensor-spec: '{}' is not NAME:TYPE:DIMS".format(spec))
        name, type_name, dims_text = parts
        type_id = _READER.TYPE_IDS_BY_NAME.get(type_name.strip().upper())
        if type_id is None:
            refuse("unknown-type: '{}' is not a ggml type name".format(type_name))
        dims = [int(d) for d in dims_text.split("x")]
        body += gguf_string(name)
        body += struct.pack("<I", len(dims))
        body += b"".join(struct.pack("<Q", d) for d in dims)
        body += struct.pack("<I", type_id)
        body += struct.pack("<Q", offset)
        elements = 1
        for dim in dims:
            elements *= dim
        _, block_elements, block_bytes = _READER.GGML_TYPES[type_id]
        offset += elements // block_elements * block_bytes

    head = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", len(args.tensor)) \
        + struct.pack("<Q", 0)
    blob = head + body

    if args.size_bytes is not None:
        if args.size_bytes < len(blob):
            refuse("size-too-small: the header and the index are {} bytes, and --size-bytes is {}"
                   .format(len(blob), args.size_bytes))
        blob += b"\0" * (args.size_bytes - len(blob))

    with open(args.out, "wb") as handle:
        handle.write(blob)


if __name__ == "__main__":
    main()
