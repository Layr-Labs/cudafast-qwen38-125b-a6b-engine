#!/usr/bin/env python3
"""Bit-exact MTP gate: the mtp leg must commit the serial leg's token stream.

Drives the staged `cuda-engine` binary over Engine Protocol v1 (NDJSON on
stdio) twice per prompt, once with the drafter off (DS4_MTP_DRAFT_TOKENS=1) and
once with it armed at the declared depth, and compares the committed tokens.
Both spawns inherit the caller's environment, so run it under tools/serve-up.sh
(SERVE_UP_SPECULATIVE=1) to share one weight owner.

Prompts come from the staged goldens: `benchmark.decode_seed_tokens` of every
*.golden.json in the golden directory, or the files named on the command line.

Usage:
  tools/ds4/mtp-exactness-gate.py [--engine BIN] [--steps N] [--depth D] [GOLDEN.json ...]

Exit status 0 when every prompt matches, 1 otherwise. Prints one line per
prompt: match/mismatch, first divergent step, drafts, accepts.
"""
import argparse
import glob
import json
import os
import subprocess
import sys


def spawn(engine, env):
    proc = subprocess.Popen(
        [engine],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        env=env,
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


def free_run(engine, env, seed, steps, spec):
    proc = spawn(engine, env)
    try:
        begin = {"id": 1, "kind": "free_decode_begin", "seed_tokens": seed}
        if spec is not None:
            begin["spec"] = spec
        first = call(proc, begin)
        run = call(proc, {"id": 2, "kind": "free_decode_run", "count": steps})
    finally:
        proc.stdin.close()
        proc.wait(timeout=600)
    return first["seed_token"], run


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("goldens", nargs="*")
    ap.add_argument("--engine", default=os.environ.get("MLXFAST_ENGINE_BIN", ".build/release/mlxfast-runtime-worker"))
    ap.add_argument("--golden-dir", default=os.environ.get("MLXFAST_QWEN38_GOLDEN_DIR", ""))
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--depth", type=int, default=1)
    args = ap.parse_args()

    goldens = args.goldens or sorted(glob.glob(os.path.join(args.golden_dir, "*.golden.json")))
    goldens = [g for g in goldens if ".mtp" not in os.path.basename(g)]
    if not goldens:
        sys.exit("no goldens: pass files or set MLXFAST_QWEN38_GOLDEN_DIR")
    if not os.access(args.engine, os.X_OK):
        sys.exit(f"engine binary is not executable: {args.engine}")

    serial_env = dict(os.environ, DS4_MTP_DRAFT_TOKENS="1")
    mtp_env = dict(os.environ, DS4_MTP_DRAFT_TOKENS=str(args.depth + 1))
    spec = {"mode": "mtp", "mtp": {"depth": args.depth}}

    failures = 0
    for path in goldens:
        golden = json.load(open(path))
        seed = golden["benchmark"]["decode_seed_tokens"]
        name = os.path.basename(path).replace(".golden.json", "")
        s_seed, s_run = free_run(args.engine, serial_env, seed, args.steps, None)
        m_seed, m_run = free_run(args.engine, mtp_env, seed, args.steps, spec)
        s_tokens = [s_seed] + s_run["tokens"]
        m_tokens = [m_seed] + m_run["tokens"]
        drafted = m_run.get("drafted_total", 0)
        accepted = m_run.get("accepted_total", 0)
        if s_tokens == m_tokens:
            print(f"{name}: MATCH {len(m_tokens)} tokens; drafts {drafted}, accepted {accepted}")
        else:
            failures += 1
            first = next(i for i, (a, b) in enumerate(zip(s_tokens, m_tokens)) if a != b)
            print(
                f"{name}: MISMATCH at step {first} (serial {s_tokens[first]}, mtp {m_tokens[first]}); "
                f"drafts {drafted}, accepted {accepted}"
            )
    print(f"{len(goldens) - failures}/{len(goldens)} prompts bit-exact at depth {args.depth}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
