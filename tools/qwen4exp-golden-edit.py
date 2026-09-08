#!/usr/bin/env python3
"""Byte-level edits on a recorded golden, for tools/qwen4exp-golden-reauthor.sh.

Every subcommand here works on the RAW BYTES of a golden that
`benchd record-correctness-golden` already wrote and validated. Nothing
re-serializes the document: a JSON round trip would reformat every field and
change the byte count that the track contract pins, so each edit is a targeted
substitution and everything it does not touch stays byte-identical.

SUBCOMMANDS

  prompt-tokens GOLDEN
      Print `cases[0].prompt_tokens` as a JSON array. This is how the
      re-author run takes the PROMPTS from the pinned goldens: the prompt set
      is the durable half of a golden and the engine change does not move it.

  head-tokens --dump FILE --count N
      Print the first N ids of a `ds4 --dump-tokens` output. The first line of
      that output is the token-id array; the lines after it are the decoded
      pieces, one per token.

  graft-decode-oracle --golden G --capture C --out O
      Write G with `benchmark.expected_decode_tokens` replaced by the token
      stream in capture file C (tools/ds4/free-run-capture.py output). This is
      how a per-depth oracle is authored: the pinned mtp1/mtp2 oracles are the
      serial golden with ONLY that array changed, because teacher-forced
      `cases[]` do not depend on the route. Refuses when C's `seed_token`
      disagrees with G's `expected_decode_seed_token`, which would mean the two
      captures did not see the same seed forward.

  perturb-one-token --golden G --out O
      Write G with ONE id of `cases[0].expected_tokens` changed and the BYTE
      COUNT unchanged (one decimal digit is moved). The negative control: the
      result must be REJECTED by `benchd validate-golden` against G's pin.
"""
import argparse
import json
import re
import sys

ARRAY_RE = rb'("expected_decode_tokens"\s*:\s*)\[[^\]]*\]'
FIRST_EXPECTED_RE = rb'"expected_tokens":\s*\[\s*(\d+)'


def read(path):
    with open(path, "rb") as fh:
        return fh.read()


def write(path, data):
    with open(path, "wb") as fh:
        fh.write(data)


def cmd_prompt_tokens(args):
    doc = json.loads(read(args.golden))
    json.dump(doc["cases"][0]["prompt_tokens"], sys.stdout)
    sys.stdout.write("\n")


def cmd_head_tokens(args):
    with open(args.dump, "r") as fh:
        first = fh.readline()
    ids = [int(t) for t in re.sub(r"[\[\],]", " ", first).split()]
    if len(ids) < args.count:
        sys.exit(
            f"{args.dump}: the prompt tokenizes to {len(ids)} ids, fewer than the "
            f"{args.count} a correctness golden case needs"
        )
    json.dump(ids[: args.count], sys.stdout)
    sys.stdout.write("\n")


def cmd_graft_decode_oracle(args):
    raw = read(args.golden)
    doc = json.loads(raw)
    capture = json.loads(read(args.capture))
    seed = doc["benchmark"]["expected_decode_seed_token"]
    if capture["seed_token"] != seed:
        sys.exit(
            f"the capture's seed token {capture['seed_token']} is not the golden's "
            f"expected_decode_seed_token {seed}: the two free runs did not see the same "
            f"seed forward, so grafting one oracle onto the other would describe no run"
        )
    want = len(doc["benchmark"]["expected_decode_tokens"])
    tokens = capture["tokens"]
    if len(tokens) != want:
        sys.exit(f"the capture holds {len(tokens)} tokens; the golden's oracle is {want} long")
    body = ",\n".join(f"      {t}" for t in tokens)
    replacement = rb"\1[\n" + body.encode() + rb"\n    ]"
    out, count = re.subn(ARRAY_RE, replacement, raw, count=1)
    if count != 1:
        sys.exit(f"{args.golden}: expected exactly one expected_decode_tokens array, matched {count}")
    json.loads(out)
    write(args.out, out)
    print(f"grafted {len(tokens)} oracle tokens into {args.out}")


def cmd_perturb_one_token(args):
    raw = read(args.golden)
    match = re.search(FIRST_EXPECTED_RE, raw)
    if not match:
        sys.exit(f"{args.golden}: found no cases[].expected_tokens array to perturb")
    start, end = match.span(1)
    old = raw[start:end]
    last = old[-1:]
    new = old[:-1] + (b"8" if last == b"9" else bytes([last[0] + 1]))
    out = raw[:start] + new + raw[end:]
    if len(out) != len(raw):
        sys.exit("the perturbation changed the byte count; it must change only the token")
    write(args.out, out)
    print(f"perturbed one expected token {old.decode()} -> {new.decode()} in {args.out} ({len(out)} bytes, unchanged)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prompt-tokens")
    p.add_argument("golden")
    p.set_defaults(fn=cmd_prompt_tokens)

    p = sub.add_parser("head-tokens")
    p.add_argument("--dump", required=True)
    p.add_argument("--count", type=int, required=True)
    p.set_defaults(fn=cmd_head_tokens)

    p = sub.add_parser("graft-decode-oracle")
    p.add_argument("--golden", required=True)
    p.add_argument("--capture", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_graft_decode_oracle)

    p = sub.add_parser("perturb-one-token")
    p.add_argument("--golden", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_perturb_one_token)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
