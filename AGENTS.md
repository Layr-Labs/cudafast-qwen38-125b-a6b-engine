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

The engine's own sources are Gemma-named, because the engine is still the seed
(see the warning at the top). A few fixtures and transform validators carry
`Qwen 3.6` or `Laguna` in their names, and those are not leftovers: they name
real foreign checkpoints that this track's gates are proven against. `Qwen35CheckpointValidation` and `fixtures/qwen3_6_27b_config.json`
build the Qwen-shaped config the Gemma trusted-config gate must REJECT;
`LagunaConfig` is the fixture substrate the generic transform tests run on.
Renaming either would make the name lie about what it holds.

## Current state

> **NOTE — official scoring is UNARMED while the goldens are re-authored.**
> `fixtures/qwen3_8_125b_a6b_track.json` sets `official_scoring_enabled` to
> `false`. The pinned goldens and per-depth oracles were authored on the
> previous (vLLM, NVFP4) engine and do not describe the ds4 engine. Re-arming
> needs goldens re-authored on ds4, the serial baseline re-pinned in the
> benchmarker, and a negative control. The launch reference stays stock serial
> (`num_speculative_tokens` 0); the native MTP head is the improvement path.

> **WARNING — no ranked runner advertises the label yet.**
> `.github/workflows/benchmark.yml` is the real ranked pipeline: a hosted
> surface check, then a self-hosted ranked job on
> `[self-hosted, Linux, ARM64, qwen3.8-125b-a6b-cuda-v1]`. The label is the ruled
> one and the job body is the real Linux/CUDA one (the stale macOS/M5 duplicate
> was deleted with #41). What remains is box work: no runner advertises the label
> yet and the box is not staged, so a dispatch queues or refuses until the box is
> registered and the goldens are staged (test-3 prep). It holds no credential by
> design: the goldens are staged onto the box and pin-verified by
> `tools/ranked-box-preflight.sh`, which refuses rather than fetch or substitute.

There is ONE speculative arm on this track: the MTP head. DFlash was a

## Notes for autonomous agents

These behaviors are expected. They are not bugs.

> **NOTE — the local-iteration guidance below is SEED material.**
> The cool-down gate, the startup memory profile, the non-M5
> near-tie caveat and the `mlxfast-swift` local modes describe the removed
> Apple-Metal runtime. The CUDA engine's local-iteration workflow is deferred to
> a David/organizer ruling with the participant-editable surface. Today the local
> signal is `cargo test --manifest-path harness/protocol-adapter/Cargo.toml`,
> which runs the adapter against a mock transport with no GPU. The ranked cool
> gate itself persists on the CUDA box (`docs/qwen38-125b-a6b-port-notes.md`
> section 7).

### The cool-down gate

The benchmarker waits for the GPU to cool before it starts a timed run. The
local modes pass `--cool-gate` to the benchmarker automatically. The gate reads
the GPU temperature through `macmon`.

**The gate lives in the benchmarker, and only `./benchmark.sh` arms it.**
`./benchmark.sh` passes `--cool-gate` to `benchd`, and `benchd` runs the gate
itself before each timed phase. Prefill and decode are gated separately.

> **WARNING — driving the Swift CLI directly runs UNGATED.**
> `mlxfast-swift --local-iterate` and `--local-submit` do not go through
> `./benchmark.sh`. The seed's Swift harness dispatched its per-phase gate to an
> external helper named by `MLXFAST_LOCAL_COOL_GATE_HELPER`. Nothing sets that
> variable. Unset, the gate returns immediately and times a hot GPU without
> saying so. These local modes belonged to the removed Apple-Metal runtime.

Set the variable to the pinned benchmarker to arm that path.

```bash
MLXFAST_LOCAL_COOL_GATE_HELPER="$PWD/benchd-bin/benchd" mlxfast-swift --local-iterate
```

Prefer `./benchmark.sh`. It is the measured path and it needs no such variable.

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

`./setup.sh` installs `macmon` as a pinned, hash-verified release binary. The
gate warns and skips when `macmon` is absent. Skip the install with
`MLXFAST_SKIP_MACMON_INSTALL=1`.

> **WARNING — a skipped gate still produces a number.**
> Locally, no reader means no gate, and the run times whatever temperature the
> GPU happens to be at. Treat a timing taken without `macmon` as unmeasured.

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
> Do not start a second local run while the first is alive. Do not run a
> model-holding `mlxfast-swift` command next to a local test. These commands are
> `correctness`, `correctness-trace`, `generate-golden`, and
> `generate-gpqa-answers`.

No run lock enforces this. The discipline is yours to keep.

`cargo test` on the adapter never loads the real model. It is safe to run
alongside.

Check for an orphaned worker when a run aborts. A worker whose parent process
identifier is 1 is usually an orphan. Verify it, then kill it.

### The startup memory profile

The runtime selects a low-memory profile automatically below 64 GiB of physical
memory. The profile caps the MLX allocator cache at 6 GiB, shortens command
buffers, and releases free warmup buffers before the worker serves requests.

The profile is pure memory management. It disables no code path and no
output-affecting feature. It announces itself on stderr. Force it either way
with `DARKBLOOM_STARTUP_MEMORY_PROFILE=full|low|auto`.

A machine that is too small fails loudly with an out-of-memory error. It does
not diverge silently from ranked behavior.

### The non-M5 near-tie caveat

The checked-in public goldens are M5-generated greedy continuations. A near-tie
argmax can diverge on another Apple Silicon generation, even for correct code.

> **WARNING — a local gate failure on non-M5 hardware may not be your bug.**
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
adapter, then verifies the GGUF target snapshot (box steps that fail closed off
the box). This track's MTP head is native to the target checkpoint; there is
no separate head-staging step.

## Engine tooling

The scored engine is Rust (`harness/protocol-adapter`). Use a stable `cargo`
toolchain; `rust-analyzer` is the standard language server. Point your editor at
`harness/protocol-adapter/` so it reads `Cargo.toml`.

## Where to spend effort

Good changes improve one or more of these.

- Kernel-level work inside the vendored Metal sources. Prioritize kernels the
  cohort prefill and the timed decode window reach.
- The batching engine. Admission, scheduling, round driving, and stream drain
  are competitive surface.
- Attention dispatch. The sliding-window and full-attention layer types use
  different masks and different head dimensions.
- The quantized matmul and the MoE gather-GEMM for the routed experts.
- KV-cache handling. The sliding-window cache only ever needs the last 1024
  positions.
- Weight loading and reuse. Prepare eagerly at init. Warm kernels before the
  first scored forward. Avoid redundant conversions.
- MLX operation scheduling and synchronization.
- Transform metadata that lets the runtime skip work safely.

## Wrong strategies

Do not specialize for the public correctness prompt. Keep every change
prompt-independent and model-general. The hidden prompts differ from the public
fixtures.

Do not assume the ranked box has your local machine's memory budget. A strategy
tuned on one Apple Silicon generation can move differently on another.

Do not treat a local-only environment override as proof of a valid improvement.
Disabling the sandbox, skipping the transform without verifying `weights/`, and
pointing at a user-specific reference path are debugging aids. They do not
establish a rankable optimization.

Do not draw a conclusion from a tiny local run alone. A local run is a smoke
test. It is especially weak for sequence-length-dependent changes, because it
may not exercise the ranked sequence lengths or the ranked memory pressure.

Be conservative with numeric reassociation. A changed accumulation order can
flip a near-tie greedy argmax.

> **WARNING — the target quantization is frozen as shipped.**
> Do not re-quantize any target weight. Do not re-represent one. Do not change
> the numerical format of one. This holds even when the result passes every
> correctness gate. Nothing licenses a change of target format: a lossier target
> substitutes a degraded model instead of optimizing the accepted one. The MTP
> head is a narrow exception, and the exception is RE-QUANTIZATION ONLY
> (David ruling 2026-08-26) — re-quantize it within its 2 GiB
> declaration cap, but do not replace it and do not upload head weights.
> `mtp-head/` is not an editable path, and a head declaration
> accepts `"source": "pinned"` only. A head re-quantization happens ON LOAD, in
> memory: the head loader quantizes the head's parameters while it binds the
> checkpoint. On this CUDA track the engine is the ds4 engine behind the Rust
> adapter, so the editable seam is the adapter in `harness/`. Nothing on disk
> changes
> (`docs/participant-contract.md` section 4.4).
> They only propose tokens; the pinned target decides every emitted token.
> The target's own quantization is verified on the LOADED model TWICE: once at
> worker startup, and again at the top of every window that gets measured,
> immediately before the measured work starts. The second check is there because
> the first alone verifies a model that later code can still change in place. An
> in-memory re-quantization of the target is refused by name, and the refusal
> stops the worker before any measurement.

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

Check the non-M5 near-tie caveat above when local correctness fails. Prefer a
more conservative optimization when performance improves but correctness turns
fragile.

Use the Yukon CLI for every account operation and every submission operation.
README.md holds the submission commands. Python is not part of the challenge
runtime.
