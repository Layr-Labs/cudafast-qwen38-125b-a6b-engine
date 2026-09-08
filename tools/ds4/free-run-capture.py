#!/usr/bin/env python3
"""Capture one free-run token stream from the staged `cuda-engine`, at a declared spec.

WHY THIS EXISTS. `benchd record-correctness-golden` authors a golden's
`benchmark` block by driving `free_decode_begin` / `free_decode_run`, but it
sends NO `spec`, and an absent spec is `serial`
(harness/protocol-adapter/src/adapter.rs `resolve_spec`). So the recorder can
author the depth-0 (serial) oracle and nothing else. The per-depth oracles the
track contract pins (`live_golden_speculative.mtp1`) need the SAME free-run at
an armed spec, which is what this script captures.

WHAT IT PRINTS. One JSON object on stdout:

    {"seed_token": <int>, "tokens": [<int>, ...], "drafted_total": N, "accepted_total": N}

`tokens` is the committed stream of `free_decode_run(--steps)`. It is the array
that becomes a golden's `benchmark.expected_decode_tokens`; `seed_token` is that
golden's `expected_decode_seed_token`.

HOW IT IS DRIVEN. The engine is spawned once and inherits the caller's
environment, so run it under tools/serve-up.sh: the resident holds the weights
and this process only connects. `--depth 0` sends no spec (the serial control);
`--depth N` sends `{"mode": "mtp", "mtp": {"depth": N}}`.

Usage:
  tools/ds4/free-run-capture.py --engine BIN --seed-from GOLDEN.json \\
      [--depth N] [--steps N]
"""
import argparse
import json
import os
import subprocess
import sys


def spawn(engine):
    proc = subprocess.Popen(
        [engine],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        env=dict(os.environ),
        text=True,
        bufsize=1,
    )
    hello = json.loads(proc.stdout.readline())
    if hello.get("id") != 0 or not hello.get("ok", False):
        raise RuntimeError(f"bad hello: {hello}")
    return proc


def call(proc, req):
    proc.stdin.write(json.dumps(req) + "\n")
    proc.stdin.flush()
    line = proc.stdout.readline()
    if not line:
        raise RuntimeError("engine closed the protocol stream")
    resp = json.loads(line)
    if not resp.get("ok", False):
        raise RuntimeError(f"engine refused {req.get('kind')}: {resp.get('error')}")
    return resp


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--engine", required=True)
    ap.add_argument("--seed-from", required=True, help="golden whose benchmark.decode_seed_tokens is the seed")
    ap.add_argument("--depth", type=int, default=1, help="0 = serial (no spec), N = mtp at depth N")
    ap.add_argument("--steps", type=int, default=128)
    args = ap.parse_args()

    if not os.access(args.engine, os.X_OK):
        sys.exit(f"engine binary is not executable: {args.engine}")
    golden = json.load(open(args.seed_from))
    seed = golden["benchmark"]["decode_seed_tokens"]

    spec = None if args.depth == 0 else {"mode": "mtp", "mtp": {"depth": args.depth}}
    proc = spawn(args.engine)
    try:
        begin = {"id": 1, "kind": "free_decode_begin", "seed_tokens": seed}
        if spec is not None:
            begin["spec"] = spec
        first = call(proc, begin)
        run = call(proc, {"id": 2, "kind": "free_decode_run", "count": args.steps})
    finally:
        proc.stdin.close()
        proc.wait(timeout=600)

    tokens = run.get("tokens") or []
    if len(tokens) != args.steps:
        sys.exit(f"free_decode_run returned {len(tokens)} committed tokens; need exactly {args.steps}")
    json.dump(
        {
            "seed_token": first["seed_token"],
            "tokens": tokens,
            "drafted_total": run.get("drafted_total", 0),
            "accepted_total": run.get("accepted_total", 0),
        },
        sys.stdout,
    )
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
