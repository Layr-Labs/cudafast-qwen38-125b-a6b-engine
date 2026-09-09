# Agent guide — Qwen 3.8 125B A6B CUDA engine

This file is the working contract for coding agents in this repository.
`CLAUDE.md` is a symbolic link to this file.

Read [README.md](README.md) first. It states what this repository is, how to
set it up, and what the structure is. This file adds the operational rules that
a person or an agent needs while iterating here.

The ranked track is `qwen3.8-125b-a6b-cuda-v1`.

> **The engine is the ds4 port (`Layr-Labs/ds4`).** The TRACK is the Qwen 3.8 125B
> A6B CUDA one: `benchmark.json`, `fixtures/qwen3_8_125b_a6b_track.json`, the
> scoring ruling, the benchmarker channel and the tooling all name it. The
> scored engine is the Engine Protocol v1 adapter (`harness/protocol-adapter`)
> with the ds4 C/CUDA engine linked in-process; the engine source is the pinned
> VENDORED `ds4/` tree and the participant-editable surface is `ds4/`, `harness/`
> and the MTP head declaration (`mtp-head/`, `mtp-head.manifest.json`);
> `benchmark.json` `editablePaths` is the authority.
> `docs/qwen38-125b-a6b-port-notes.md` is the engineering record. Read it
> before you read anything else here.

## Goal

Make the target text tower decode and prefill faster.
Do not change the observable model behavior beyond what the token-tolerance gate
allows.

## Authorities

| Question | Authority |
|---|---|
| Editable paths, commands, scoring values | `benchmark.json` |
| Pins, the timed pool, scoring semantics | `fixtures/qwen3_8_125b_a6b_track.json` |
| Why the manifest says what it says | `docs/participant-contract.md` |
| The engineering log for this port | `docs/qwen38-125b-a6b-port-notes.md` |
| What a measured run executes | the channel benchmarker (`tools/fetch-benchd.sh`) |

`benchmark.json` and `fixtures/qwen3_8_125b_a6b_track.json` carry pure
configuration. They hold values, paths, commands, and pins. They carry no prose.
Where this file disagrees with the fixture, the fixture wins.

## Lineage

This repository descends from `Layr-Labs/mlxfast-qwen-38-27b-mtp-engine`, which
descends from `Layr-Labs/mlxfast-challenge-dev`. Those repositories rank
different models under different rules. Only this track's rules apply here.

Some fixtures still carry `Gemma`, `Qwen 3.6` or `Laguna` in their names. Those
are not leftovers. They name real foreign checkpoints, and they are the negative
controls and the fixture substrate this track's own gates are tested against.
Renaming one would make the name lie about what it holds.

## Current state

> **NOTE — official scoring is ARMED.**
> `fixtures/qwen3_8_125b_a6b_track.json` sets `official_scoring_enabled` to
> `true`. That flag is the single authority on the arm state, and the
> benchmarker enforces it: it refuses to seal an official scoring artifact while
> the flag is `false` or absent. What is still box work, not repository state, is
> staging the reference workspace and calibrating each ranked box. The launch
> reference stays stock serial (`num_speculative_tokens` 0); the native MTP head
> is the improvement path.

### Scoring is PAIRED, with a per-box baseline

David ruling 2026-09-08. A ranked run measures the number of pairs the fixture
declares in `official_pairs`, which is 2 (David ruling 2026-09-09), on the same
box in the same job, on the fixture's one `live_golden`.

| Leg | What runs | Speculation |
|---|---|---|
| Serial control | the reference tree at the fixture's `baseline_reference_commit` | off, always |
| Candidate | the dispatched commit | the declared draft depth |

The legs run strictly one after the other and each leg loads the model once. Per
role the per-token times are summed over the pairs, and the score is the live
ratio of those sums:
`(ref_prefill / cand_prefill) ^ 0.25 * (ref_decode / cand_decode) ^ 0.75`. Both
speedup floors are 0.95 and the ceiling is 5.0, applied to that aggregate.

> **WARNING — no file holds a baseline pair, and none may.**
> Not a benchmarker constant, not the fixture, not a golden. A golden that
> carries `benchmark.baseline_prefill_seconds_per_token` or
> `benchmark.baseline_decode_seconds_per_token` is refused on the ranked path,
> and `tools/lint-benchmark-manifest.py` refuses one at rest. Do not add the
> field back when you re-author a golden.

Two environment variables carry the paired path, and both are required on a
ranked box:

| Variable | Meaning |
|---|---|
| `MLXFAST_BASELINE_WORKSPACE` | the built reference tree the control leg runs on |
| `MLXFAST_BASELINE_CALIBRATION` | this box's calibration file |

The calibration file is a HEALTH BAND for the control leg. The benchmarker
refuses a run whose control leg falls outside it. It is never a denominator.

Three tools own this path. None of them runs on a laptop.

```bash
tools/stage-baseline-workspace.sh <dir>   # clone the reference tree, build it
tools/calibrate-box.sh <box> <out>        # capture that box's band
tools/ranked-box-preflight.sh             # section 8 refuses either, by name
```

`tools/stage-baseline-workspace.sh --dry-run` and `tools/calibrate-box.sh
--dry-run` print their plans and touch nothing. Use them to read the path
without a box.

#### The per-leg serve verbs

benchd boots each leg's resident itself, from that leg's own workspace:

```bash
tools/serve-up.sh --boot --spec serial|mtp --draft-len N --socket-out FILE
tools/serve-up.sh --stop --socket PATH
```

`--boot` starts ONE resident, waits for the healthy hello, writes the socket
path as the FIRST LINE of `FILE`, and exits 0 with the resident still running.
benchd reads that line and injects the socket into that leg's worker spawns as
`DS4_RESIDENT_SOCKET` and `BENCH_WORKER_RESIDENT_SOCKET`. `--stop` ends the
resident and cleans up after it. `--stop` is idempotent: benchd runs it on
success and on failure alike.

Leg 1 is always `--spec serial --draft-len 0`. Leg 2 gets the candidate's
declared spec.

The wrapper form (`tools/serve-up.sh CMD [ARGS...]`) still works. Local drivers
use it.

> **WARNING — the two serves never overlap.**
> Each leg boots its own resident engine, and each resident holds about
> 103.7 GiB. Leg 1 boots, is measured, and is stopped. Only then does leg 2
> boot. `tools/serve-up.sh`'s memory plan must hold for each serve on its own.
> Do not add a path that keeps both up.

> **WARNING — never export `SERVE_UP_SPECULATIVE` on the ranked path.**
> The value would reach BOTH legs' serve scripts, and the control leg is serial
> whatever the candidate declares. In `--boot` mode the flag wins and a
> disagreeing environment value is REFUSED. The preflight and
> `tools/qwen38-125b-a6b-measure-and-score.sh` refuse a preset value too.

> **WARNING — never export `DS4_RESIDENT_SOCKET` or
> `BENCH_WORKER_RESIDENT_SOCKET` in the job.**
> benchd boots each leg's resident and injects that leg's socket. A socket
> already in the environment is a third resident neither leg booted, and every
> worker that inherited it would measure whatever is on the other end. benchd
> refuses one in its own environment, and so do the preflight and the two
> drivers.

> **NOTE — ranked pipeline and runner availability.**
> `.github/workflows/benchmark.yml` is the configured ranked pipeline: a hosted
> surface check followed by a self-hosted ranked job on
> `[self-hosted, Linux, ARM64, qwen3.8-125b-a6b-cuda-v1]`. A dispatch reaches
> measurement only when a live self-hosted runner advertises all four labels and
> its staged inputs pass `tools/ranked-box-preflight.sh`. Official workflow
> [34370162983](https://github.com/Layr-Labs/cudafast-qwen38-125b-a6b-engine/actions/runs/34370162983)
> completed on 2026-09-09 on Spark 3 (`spark-3`), passing hosted static review
> and GPU measurement. Runner availability and staged box state are runtime
> state, so check the live Actions runner inventory or queued run before another
> dispatch. The job holds no credential by design: organizer-staged goldens and
> box assets are pin-verified rather than fetched or substituted.

There is ONE speculative arm on this track: the MTP head. DFlash was a
Gemma-era second arm and is removed.

## Notes for autonomous agents

These behaviors are expected. They are not bugs.

> **NOTE — the local signals off the box.**
> The GPU, the pinned target snapshot and the ranked goldens are box material.
> Off the box the signals are `cargo test --manifest-path
> harness/protocol-adapter/Cargo.toml`, which runs the adapter against a mock
> transport with no GPU, `tools/ds4/build.sh --cpu-check`, and the shell tests
> `tools/test-*.sh`. The cool gate below runs on the box.

### The cool-down gate

The benchmarker waits for the GPU to cool before it starts a timed run. The
local modes pass `--cool-gate` to the benchmarker automatically. The gate reads
the GPU temperature from the box's native reader. On the CUDA box that reader is
`nvidia-smi`.

**The gate lives in the benchmarker, and only `./benchmark.sh` arms it.**
`./benchmark.sh` passes `--cool-gate` to `benchd`, and `benchd` runs the gate
itself before each timed phase. Prefill and decode are gated separately.

> **WARNING — a timing taken outside `./benchmark.sh` runs UNGATED.**
> `./benchmark.sh` is what passes `--cool-gate`. Any other invocation times
> whatever temperature the GPU happens to be at, and it does not say so.

`./benchmark.sh --local-cool-gate-only` exits 0 without probing anything. The
bare probe is the benchmarker's own entry point.

```bash
benchd-bin/benchd --local-cool-gate-only
```

> **WARNING — a run that pauses on a cool-down message is working, not hung.**
> Do not kill it. Do not treat the wait as a failure.

The gate aborts with a non-zero exit when the GPU stays hot and is not trending
down. That abort means something else is loading the GPU. Free the GPU and
retry. The abort does not mean your change is wrong.

The gate warns and skips when no temperature reader resolves.

> **WARNING — a skipped gate still produces a number.**
> Locally, no reader means no gate, and the run times whatever temperature the
> GPU happens to be at. Treat a timing taken with no reader as unmeasured.

The ranked box does the opposite. A missing or frozen reader is a hard refusal
there, before any measurement (`tools/ranked-box-preflight.sh`, sections 2b and
2c). A ranked run never proceeds without thermal control.

The gate mirrors the ranked runner's fixed per-platform thermal contract: 50 C
on CUDA/GB10 (40 C on Mac). That contract is operator-owned. The benchmarker
owns the exact thresholds; this repository does not set them. The threshold is a
fixed constant inside the benchmarker (sealed as `cool_gate_c` with source
`platform-cool-gate`) and no fixture can move it.

### Measurement discipline

Trust a timing number only from a cool, quiescent machine. Back-to-back runs
heat the GPU and throttle it. A 2-minute to 3-minute pause between local runs is
normal.

> **WARNING — a local score is directional.**
> The ranked run is single-stream (scored batch size 1, David ruling
> 2026-08-27), so it exercises the same width a local test does. What still
> differs is the machine, the pinned prompts and the staged checkpoint. Do not
> read a local score as a prediction of the ranked composite.

Record a same-machine baseline before you optimize. Sync to the latest tip
first. Do not compare a change against a stale branch or an old local run. Rerun
the baseline whenever the base commit changes.

### One model-holding run at a time

The target model is RAM-resident. Two model residencies at once can exhaust a
local machine's memory.

> **WARNING — run one model-holding command at a time.**
> Do not start a second local run while the first is alive. Do not run any
> second model-holding command next to a local test. One resident engine at a
> time is the only arrangement that fits.

No run lock enforces this. The discipline is yours to keep.

`cargo test` on the adapter never loads the real model. It is safe to run
alongside.

Check for an orphaned worker when a run aborts. A worker whose parent process
identifier is 1 is usually an orphan. Verify it, then kill it.

### The near-tie caveat

The checked-in public goldens are greedy continuations captured on ranked
hardware. A near-tie argmax can diverge on other hardware, even for correct
code.

> **WARNING — a local gate failure on other hardware may not be your bug.**
> Check whether an unmodified `main` fails at the same token position on your
> machine. Do that before you treat a local failure as a regression.

Rerun with `MLXFAST_LOCAL_ALLOW_GOLDEN_DRIFT=1` when unmodified `main` fails the
same way. The local mode then still publishes its timing estimate.

The override is local-only. It hides nothing. The score keeps
`passed_correctness: false`, records the diverging tokens, and explains itself in
`metrics.error`.

> **WARNING — never use the override to paper over a real regression.**
> The mismatch is yours when unmodified `main` passes on your machine.

### One ranked machine, one queue

Ranked runs execute serially on a single runner. Duplicate dispatches queue
behind the run in flight. They do not cancel it. Expect delays. Do not dispatch
several ranked runs in parallel and expect concurrent results.

### Know the runnable surface

Only the `benchmark.json` `editablePaths` entries ship in a submission. A change
anywhere else does not upload, even when it helps locally. Official ranking
needs hidden organizer goldens. It is not runnable locally.

## Building

The scored engine is the `cuda-engine` adapter. `./setup.sh` builds and stages
it for you; to rebuild it directly:

```bash
tools/ds4/build.sh
```

This command builds the vendored ds4 engine, links the shim
library, and builds the adapter with its real ds4 backend
(`--features ds4-engine`). It needs `nvcc` and `cargo`. Off the
box, build the adapter alone for the deterministic mock backend, which needs
no GPU:

```bash
cargo build --release --manifest-path harness/protocol-adapter/Cargo.toml --bin cuda-engine
```

```bash
tools/stage-cuda-engine.sh
```

This command copies the finished `cuda-engine` binary to the fixed workspace path
benchd resolves and spawns (`.build/release/mlxfast-runtime-worker`). That name
is retained from the seed on purpose — it is a contract with the benchmarker, not
a preference. There is no `mlx.metallib`: the adapter carries no Metal kernels.
The vendored MLX/Metal kernel tree and its ahead-of-time build step were removed
with the seed's runtime, and the whole Swift package (`Package.swift`,
`Package.resolved`, all of `Sources/`) went with the final de-Swift. `cargo` is
the only toolchain this repository builds with.

## Common commands

```bash
cargo test --manifest-path harness/protocol-adapter/Cargo.toml
```

This command runs the adapter's unit tests against a deterministic mock
backend and a scripted ds4 session — no GPU, no engine, no checkpoint. It is
the local smoke signal.

```bash
tools/ds4/build.sh --cpu-check
```

This command checks that the vendored engine and the adapter's C shim agree,
with no CUDA. CI runs it.

```bash
./tools/fetch-benchd.sh
```

This command resolves and verifies the pinned benchmarker binary.

```bash
./setup.sh
```

This command builds the ds4 engine and the `cuda-engine` adapter, stages the
adapter, then makes the GGUF target snapshot present and verified. With
`MLXFAST_TARGET_SNAPSHOT_DIR` unset, missing files are downloaded from the
pinned public model repository into `reference_weights/`, every file is
checked against the contract's pins, and a rerun there reads no shard. With it
set (a ranked box), setup only verifies the staged directory, on every run.
This track's MTP head is native to the target checkpoint; there is no separate
head-staging step.

```bash
tools/local-baseline.sh
```

This command runs the local test on the public golden. It seals no score.

## Engine tooling

The scored engine is Rust (`harness/protocol-adapter`). Use a stable `cargo`
toolchain; `rust-analyzer` is the standard language server. Point your editor at
`harness/protocol-adapter/` so it reads `Cargo.toml`.

## Where to spend effort

Good changes improve one or more of these.

- CUDA kernel work inside the vendored engine (`ds4/`). Prioritize the kernels
  the prefill and the timed decode window reach.
- Attention dispatch. The full-attention layers and the gated deltanet linear
  layers take different paths.
- The quantized matmul and the MoE gather-GEMM for the routed experts.
- KV-cache handling on the 12 full-attention layers, and the recurrent state on
  the other 36.
- The n-gram / PLE table reader on layer 1.
- Weight loading and reuse. Prepare eagerly at init. Warm kernels before the
  first scored forward. Avoid redundant conversions.
- The speculative cycle: the drafter, the draft depth, and the target verify.

## Wrong strategies

Do not specialize for the public correctness prompt. Keep every change
prompt-independent and model-general. The hidden prompts differ from the public
fixtures.

Do not assume the ranked box has your local machine's memory budget. A strategy
tuned on one machine can move differently on another.

Do not treat a local-only environment override as proof of a valid improvement.
Skipping the checkpoint verification and pointing at a user-specific reference
path are debugging aids. They do not establish a rankable optimization.

Do not draw a conclusion from a tiny local run alone. A local run is a smoke
test. It is especially weak for sequence-length-dependent changes, because it
may not exercise the ranked sequence lengths or the ranked memory pressure.

Be conservative with numeric reassociation. A changed accumulation order can
flip a near-tie greedy argmax.

> **WARNING — the weights are frozen as shipped.**
> Do not re-quantize any weight. Do not re-represent one. Do not change the
> numerical format of one. Do not mirror one. This holds on disk and in memory,
> and it holds even when the result passes every correctness gate. Nothing
> licenses a change of weight format: a lossier target substitutes a degraded
> model instead of optimizing the accepted one.
> The MTP head is NO exception. It is the organizer's pinned weights, a separate
> Q8_0 GGUF staged in the target snapshot beside the shards, and the engine uses
> it exactly as staged. Do not re-quantize it, replace it, or upload head
> weights of your own.
> `mtp-head/` IS an editable path; it holds only its `README.md`, and the byte
> budget bars a weight file there. The declaration
> `mtp-head.manifest.json` is editable and optional. It accepts
> `"source": "pinned"` only, and its live field is `spec`
> (`docs/participant-contract.md` section 4).
> Nothing about the head is participant-tunable except the draft depth.
> A head only proposes tokens; the pinned target decides every emitted token.

> **WARNING — do not add a cache keyed on a request's input tokens whose only
> possible hit is the harness repeating one identical computation.**
> Bit-identical output does not make it legitimate. The benchmark measures
> single-pass inference. An optimization must save work that recurs in
> single-pass production inference. The harness never legitimately issues the
> same whole-prompt forward twice to one worker process. Any such repetition is
> a harness bug, never a contract to rely on. Input-independent caching stays
> fine. Within-request KV reuse stays fine. A change in this category fails the
> static review as bypass behavior.

Do not hardcode hidden prompts, hidden token identifiers, or answers. Do not use
timing shortcuts, protocol injection, network access, or filesystem
exfiltration.

## Before submitting

Run at least these commands.

```bash
cargo test --manifest-path harness/protocol-adapter/Cargo.toml
tools/ds4/build.sh --cpu-check
```

These commands run the adapter's unit tests and check the engine pin and the
shim. No GPU.

```bash
./setup.sh
```

This command builds the ds4 engine and the scored `cuda-engine` adapter, stages
the adapter, and verifies the target snapshot (box steps fail closed off the
box).

```bash
SERVE_UP_SPECULATIVE=1 tools/serve-up.sh tools/ds4/mtp-exactness-gate.py
```

This command proves, prompt by prompt, that the MTP leg commits the serial
leg's token stream. Run it on the box before you
declare MTP.

```bash
./tools/fetch-benchd.sh
```

This command resolves the pinned benchmarker. Run a local test afterwards.

Check the near-tie caveat above when local correctness fails. Prefer a
more conservative optimization when performance improves but correctness turns
fragile.

Use the Yukon CLI for every account operation and every submission operation.
README.md holds the submission commands. Python is not part of the challenge
runtime.
