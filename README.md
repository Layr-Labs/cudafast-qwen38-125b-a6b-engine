# cudafast — Qwen 3.8 125B A6B CUDA

This repository is the engine for the Qwen 3.8 125B A6B CUDA speedup benchmark.
The track identifier is `qwen3.8-125b-a6b-cuda-v1`.

> **The engine is the ds4 port (`Layr-Labs/ds4`).** The scored engine is the
> Engine Protocol v1 adapter (`harness/protocol-adapter`) with the ds4 C/CUDA
> engine linked in-process. The engine source is VENDORED at `ds4/`,
> pinned to one commit of `Layr-Labs/ds4`, our port of `antirez/ds4`;
> `tools/ds4/build.sh` builds it. The target is the `UD-Q4_K_XL` GGUF
> conversion of Qwen 3.8 Flash Next, four shards plus a Q8_0 MTP draft head.
> The port carries the `qwen4exp` family the target needs: the multi-shard
> loader, the qwen4exp kernels and graph, the nextn MTP head behind
> `--mtp-model` and the speculative cycle at depths 1 to 6.
> The participant-editable surface is the vendored engine (`ds4/`), the engine
> adapter (`harness/`) and the MTP head declaration (`mtp-head/`,
> `mtp-head.manifest.json`);
> `benchmark.json` `editablePaths` is the authority. This repository was seeded from the
> Gemma 4 26B A4B MLX engine and then carried a vLLM engine; both are gone.
> `docs/qwen38-125b-a6b-port-notes.md` is the engineering record.

## What this repository is

This repository holds the engine. You optimize the engine. You make the model
do the same work in less time.

The benchmarker measures the engine. The benchmarker is a separate program
called `benchd`. It arrives as a verified prebuilt binary. It owns all
timing, all scoring, and all gates. Nothing in this repository measures or
scores anything.

The ranked run is SINGLE-STREAM. It times each pinned prompt in its own
window, one at a time, on your engine and on a serial control engine. The
score compares the two.

> **NOTE — official scoring is ARMED.**
> `fixtures/qwen3_8_125b_a6b_track.json` sets `official_scoring_enabled` to
> `true`. That flag is the single authority on the arm state, and the
> benchmarker enforces it: it refuses to seal an official scoring artifact while
> the flag is `false` or absent. What is still box work, not repository state, is
> staging the reference workspace and calibrating each ranked box. The launch
> reference stays stock serial (`num_speculative_tokens` 0); the native MTP head
> is the improvement path.

### Lineage

This repository descends from `Layr-Labs/mlxfast-qwen-38-27b-mtp-engine`, which
descends from `Layr-Labs/mlxfast-challenge-dev`. Those repositories rank
different models under different rules. Only this track's rules apply here.

A few fixtures and transform validators still carry `Qwen` or `Laguna` in their
names. Those names point at real foreign checkpoints on purpose. They are the
negative controls and the fixture substrate that this track's own gates are
tested against.

## Requirements

The scored engine is the Engine Protocol v1 adapter with the ds4 engine linked
in-process. `./setup.sh` builds the engine and the adapter, stages the adapter,
and verifies the target snapshot. The steps need different things.

- A CUDA toolkit (`nvcc`, CUDA 13) and a C compiler. `tools/ds4/build.sh`
  builds upstream's `cuda-spark` target for `sm_121` (the GB10) and the shim
  library the adapter links. This is box work. Off the box,
  `tools/ds4/build.sh --cpu-check` proves the pin and the shim agree without
  CUDA.
- A Rust toolchain (`cargo`). The adapter builds with `--features ds4-engine`
  against the shim library. The default (mock) build is portable and needs no
  GPU, so the adapter's own tests run on a laptop and in hosted CI.
- The organizer-staged GGUF target snapshot on the box, pointed to by
  `MLXFAST_TARGET_SNAPSHOT_DIR`: the four target shards and, flat beside them,
  the Q8_0 MTP draft head. `./setup.sh` verifies every file against the
  `{bytes, sha256}` pins in `fixtures/qwen3_8_125b_a6b_track.json` and FAILS
  CLOSED when a file or the sidecar manifest is absent. It never fetches,
  substitutes, or re-quantizes a checkpoint.
- `jq` and Git.

The benchmarker arrives as a prebuilt binary.
`./tools/fetch-benchd.sh` resolves it from the `qwen3.8-125b-a6b-v1` dist
channel and verifies it against the channel's own `benchd.manifest.json`.
That channel is the public bench repository `Layr-Labs/mlxfast-bench`, so the
fetch needs no token.
The channel holds one binary per platform and the resolver REFUSES a binary
whose `target_triple` is not this host's.

## Quickstart

Run these commands in order. One sentence describes each command.

```bash
git clone <repository-url> cudafast-qwen38-125b-a6b-engine
```

This command copies the repository to your machine.

```bash
cd cudafast-qwen38-125b-a6b-engine
```

This command makes the repository your working directory.

```bash
./tools/fetch-benchd.sh
```

This command resolves the benchmarker binary from the dist channel into
`benchd-bin/` and verifies its sha256, its byte count and its platform against
the channel's `benchd.manifest.json` (installed beside the binary).

```bash
./setup.sh
```

This command builds the ds4 engine and the `cuda-engine` adapter, stages the
adapter at the path benchd resolves, then — as box work that fails closed off
the box — verifies the GGUF target snapshot against the contract's
`{bytes, sha256}` pins. See [Requirements](#requirements) for the toolchain
each step needs and the environment variables that skip a step.

The adapter's own unit tests run without a GPU or a checkpoint:

```bash
cargo test --manifest-path harness/protocol-adapter/Cargo.toml
```

This command exercises the protocol loop, the verb translation, and every error
path against a mock transport. It needs no GPU and no checkpoint.

> **NOTE — the ranked run is box work.**
> The GPU, the pinned target snapshot and the ranked goldens all live on the
> box. Off the box the local signals are the adapter tests above, the engine
> syntax check (`tools/ds4/build.sh --cpu-check`) and the shell tests
> (`tools/test-*.sh`). See "Local testing vs the ranked run".

The public prompts in `correctness_prompts/` are Gemma captures kept for reuse
of their 1024-token prompts; the model-identity loader rejects them against the
current target, so they are not runnable against this track today
(`docs/qwen38-125b-a6b-port-notes.md` section 5).

## Repository structure

| Path | What it holds | Status |
|---|---|---|
| `ds4/` | The ds4 engine, vendored from `Layr-Labs/ds4` (our port of `antirez/ds4`). `ds4/VENDOR.json` records the signed base. **Editable.** | Editable |
| `harness/protocol-adapter/` | The CUDA engine: the Engine Protocol v1 adapter (`cuda-engine`) with the ds4 engine linked through `ds4_shim/`. | Editable |
| `fixtures/` | The track contract and the pinned checkpoint manifests. | Trusted |
| `tools/` | Setup, build (`tools/ds4/`), staging, lint, and measurement scripts. | Trusted |
| `benchd-bin/` | Where `./tools/fetch-benchd.sh` installs the verified binary. Git ignores it. | Fetched |
| `mtp-head/` | The MTP head area. It holds only its `README.md`. The head weights are not here. | Editable |
| `correctness_prompts/` | The public prompts and public goldens for local runs (Gemma captures). The track goldens are NOT here: they live in R2 and on the ranked box. | Trusted |
| `benchmark.json` | The Yukon track manifest. It lists every editable path. | Trusted |

`benchmark.json` `editablePaths` lists four entries: `ds4/`, `harness/`,
`mtp-head/` and `mtp-head.manifest.json`.

### The MTP head

This track's MTP head is NATIVE: it is a separate Q8_0 GGUF that the organizer
stages in the target snapshot beside the shards
(`docs/qwen38-125b-a6b-port-notes.md` section 6). `tools/serve-up.sh` loads that
file. No submission downloads or carries a head weight.

The head is used exactly as staged. No custom head is accepted: a head
declaration accepts `"source": "pinned"` only.

### The head directory

`mtp-head/` holds one `README.md` and nothing else. It is an editable path, and
the byte budget bars a weight file in it.

> **NOTE — keep that `README.md` in place.**
> It records what the directory is, and it keeps the directory present in a
> fresh clone.

## What you may change

`benchmark.json` `editablePaths` is the authority. It lists four entries:
`ds4/` (the vendored engine), `harness/` (the engine adapter), and `mtp-head/`
plus `mtp-head.manifest.json` (the MTP head declaration). The rule behind the
list is simple.
Code that **proposes** tokens or computes the forward pass is editable. Code
that **verifies**, **measures**, or **ledgers** stays trusted — the
benchmarker (benchd), the gates, `fixtures/` and this manifest.

### The engine is yours to change

`ds4/` is the inference engine, vendored into this repository as plain files and
**participant-editable**. Change a kernel, the graph, the scheduler, the
Makefile flags -- `benchmark.json` `editablePaths` is the authority, and it
lists `ds4`.

`ds4/VENDOR.json` records the signed commit of `Layr-Labs/ds4` the tree came
from. You do not need that repository: the engine is here.

**How to iterate.** Edit under `ds4/`, then rebuild and test:

```bash
tools/ds4/build.sh
cargo test --manifest-path harness/protocol-adapter/Cargo.toml
```

The build cache keys on the engine's CONTENT, so your edit is always rebuilt and
never served from a previous build.

**What is protected.** `benchmark.json`,
`benchmark.sh`, `fixtures/`, `correctness_prompts/` and `tools/` are the
contract, the gates and the oracles. A commit that touches them is refused by
`.github/scripts/enforce-modifiable-surface.sh` before it is measured. Code that
**proposes** tokens or computes the forward pass is editable; code that
**verifies**, **measures** or **ledgers** is not.

**Your edit must still be correct.** The correctness goldens are the gate: an
engine change that alters the token stream fails them. Speed is what you are
optimizing; the tokens are what you must preserve.

### The organizer-pinned head

The speculative head is the organizer's pinned weights. It is used exactly as
staged. You may not re-quantize it, replace it, or upload head weights of your
own. Nothing about the head is participant-tunable except the draft depth, which
you declare in `mtp-head.manifest.json`.

`mtp-head/` is an editable path, but the editable byte budget bars a head
weight file: a real head weight far exceeds `maxFileBytes` (4473321) and is
refused before any measurement.

The declaration file stays editable and optional: `mtp-head.manifest.json`. Its
live field is `spec`, and `tools/spec-declaration.sh` reads it:

```json
"spec": { "enabled": true, "num_speculative_tokens": 1 }
```

The rest of the file is bounded too. The accepted top-level keys are `version`,
`source`, `max_bytes`, `bytes`, `sha256` and `spec`. An unknown key is refused
by name. `"source"` must be `"pinned"`; `"source": "remote"` and
`"source": "in_branch"` are refused by name. `max_bytes` is an integer from 1 to
2147483648, so a declaration may lower the 2 GiB cap and may not raise it.

A declared `sha256` is optional and the runner does not verify it against the
head bytes. The head bytes are bound one level up: the head is part of the
pinned target snapshot, and `./setup.sh` verifies every staged file against the
contract's `{bytes, sha256}` pins.
`docs/participant-contract.md` section 4 is the authority for the declaration
rules.

An absent declaration is serial: the drafter is off. That is the normal case. A
declaration that is present but broken is a refusal. The runner never falls back
silently.

A head only **proposes** tokens. The organizer-pinned target model decides
every emitted token. The serial control leg always runs with the drafter
off.

### Batch size and draft depth

> **NOTE — batch size is locked at 1. Draft depth is not locked.**
> The scored batch size is 1: this track is scored single-stream (David ruling
> 2026-08-27). You may not tune it.
>
> The draft depth is a free lever, and it is not pinned at 1. You declare it in
> `mtp-head.manifest.json` under `spec.num_speculative_tokens`.
>
> Select a depth from 1 to 6. The pinned engine implements all six
> (`DS4_IMPLEMENTED_DEPTH` in `harness/protocol-adapter/src/ds4_backend.rs`). A
> depth outside 1 to 6 is refused by name, never clamped
> (`permitted_draft_depths` in the track fixture). benchd measures at depth 2
> when the invocation names no depth; an `mtp` block with no `depth` key
> resolves to the ceiling of 6.
>
> Every run seals what actually ran: `effective_spec` for the declared arm and
> depth, `effective_mean_draft_len` for the realized draft length.

### The byte budget

`benchmark.json` `editableSurfaceByteBudget` caps the enforced editable
surface.

| Key | Value |
|---|---|
| `maxTotalBytes` | 19550883 |
| `maxFileBytes` | 4473321 |
| `maxGrowthBytes` | 19550883 |
| `exemptPathMaxBytes` | 512000000 |
| `exemptPathMaxFileBytes` | 100000000 |

Every editable path is enforced. Nothing is exempt.

`exemptPaths` is absent since 2026-08-26. The exemption existed to let head
weights ride in a submission outside the source budget. A submission carries no
head weights any more, so there is nothing to exempt. The two exempt caps stay
declared because both enforcers carry the same numbers as compiled-in fallbacks
and this manifest is what holds them to a reviewed value.

The organizer-staged head is not walked by this budget at all. It sits in the
target snapshot on the box, which is not an editable path, so the walk never
visits it. `max_bytes` bounds what the runner **loads**, and stays at 2 GiB.

### The weights are frozen

The target model's quantization is frozen as shipped, and so is the MTP head's.
Do not re-quantize a weight. Do not re-represent one. Do not change the
numerical format of one. Do not mirror one. This holds on disk and in memory,
and it holds even when the result passes every correctness gate.

Nothing licenses a change of weight format. A lossier target substitutes a
degraded model instead of optimizing the accepted one.

The MTP head is no exception. It is the organizer's pinned weights, and it is
used exactly as staged. No weight on this track may be re-quantized, re-cast or
mirrored, on disk or in memory. A decoder only proposes tokens, and the pinned
target decides every emitted token.

### What you must not change

- Everything that `editablePaths` does not list. Today the editable entries are
  `ds4/` (the engine), `harness/`, `mtp-head/` and `mtp-head.manifest.json`.
- `fixtures/`, `benchmark.json`, the benchmarker (benchd), the gates, the other
  scripts under `tools/` and `.github/`, the tests, and the documents.
- `weights/`, the reference checkpoints, the scores, and the goldens.

Do not hardcode hidden prompts. Do not hardcode hidden token identifiers. Do
not use timing shortcuts, protocol injection, network access, or filesystem
exfiltration.

Do not add a cache keyed on a request's input tokens whose only possible hit is
the harness repeating one identical computation. The benchmark measures
single-pass inference. Input-independent caches stay legal. These are weights,
dequantized tensors, and RoPE or mask tables keyed on shapes and offsets.
Within-request KV reuse also stays legal.

## Local testing vs the ranked run

The ranked run is box work. It needs the GB10 box, the staged target snapshot
and the staged goldens. Off the box this repository gives you three signals.

| Signal | Command | What it proves |
|---|---|---|
| The engine builds | `tools/ds4/build.sh` on the box, `tools/ds4/build.sh --cpu-check` off it | Your engine edit compiles, and the shim agrees with the vendored engine. |
| The adapter is correct | `cargo test --manifest-path harness/protocol-adapter/Cargo.toml` | The protocol loop, the verb translation and every error path, against a mock transport. No GPU. |
| The scripts hold | `tools/test-*.sh` | The setup, serve, preflight and calibration scripts, against stubs. No GPU. |

On the box, `tools/serve-up.sh` boots and stops one resident engine, and
`tools/calibrate-box.sh` records the box's control band. Both are organizer
tools. Read each one before you run it.

> **WARNING — a local signal is directional, not predictive.**
> No local signal measures the ranked composite. The ranked box run is the
> authority on any score.

## Scoring and gates

### The formula

```text
composite = prefill_gain ^ 0.25 * decode_gain ^ 0.75
```

Each component is a gain:

```text
gain = baseline_aggregate / candidate_aggregate
```

The score is serial-anchored. A faster candidate scores above 1.

`aggregate` is the **per-prompt sum**. Each pool prompt is timed in its own
single-stream window, and the elapsed times are added together. Do this for
prefill and for decode separately. Do it on the baseline leg (the serial
control) and on the candidate leg (the built-in MTP), over the same prompts.

> **NOTE — the aggregate is a sum of separate windows.**
> It is not one elapsed time covering all the prompts at once.

### The two legs

The ranked run measures TWO legs. It measures both on the same machine, in the
same job, on the same prompt.

| Leg | What runs | Speculation |
|---|---|---|
| Serial control | the organizer-staged **reference tree** | off, always |
| Candidate | your submission | your declared draft depth |

The reference tree is the engine commit that the fixture pins as
`baseline_reference_commit`. Every ranked box runs the same reference commit, so
one board compares one control.

The score is the ratio of the two legs.

> **NOTE — the baseline is measured, not stored.**
> No file holds a baseline pair. The goldens hold none, the fixture holds none,
> and the benchmarker holds none. A golden that carries
> `benchmark.baseline_prefill_seconds_per_token` or
> `benchmark.baseline_decode_seconds_per_token` is refused.

This is why the two legs are measured together. A stored number describes the
machine that produced it, at the temperature and on the engine of that day. The
control leg describes YOUR run, on the same box, minutes before your candidate
leg.

Each leg loads the model once. The two legs never run at the same time: the
first leg boots its engine, is measured, and is stopped, and only then does the
second leg boot. The box holds about 103.7 GiB of weights, so one engine at a
time is the only arrangement that fits.

The benchmarker boots each leg's engine from that leg's own tree:

```bash
tools/serve-up.sh --boot --spec serial|mtp --draft-len N --socket-out FILE
tools/serve-up.sh --stop --socket PATH
```

The control leg always gets `--spec serial --draft-len 0`, and benchd verifies
its tokens against the serial tape (`--control-golden`), never against the
per-depth tape the candidate leg is scored on.

### The per-box calibration

Each ranked box carries a calibration file. The file records what the control
leg has measured on THAT box before: the mean of each axis, the spread, and a
band around the mean.

The file is a **health band**. The benchmarker reads it to answer one question:
did the control leg land where this box lands? If the control leg falls outside
the band, the run stops and seals no score. A box that has changed does not
produce a number.

> **WARNING — the calibration is never a denominator.**
> The benchmarker divides by the control leg it just measured. It never divides
> by the calibration file.

The organizer writes the file with `tools/calibrate-box.sh`, on the box it
describes. The tool runs the control leg four times on the reference tree and
refuses when the spread is above 1 %. Each pass gets its own resident engine:
the benchmarker boots one, measures the pass, and stops it before the next pass
boots, so every pass is shaped like the single control leg a scored run
measures.

### The measured window

| Quantity | Value |
|---|---|
| Seed tokens per window | 1024 |
| Checked decode steps | 128 |
| Golden shape | 1024 prompt tokens and 129 expected tokens |
| Concurrent streams | 1 |
| Prompts in the pinned pool | 8, timed one window each |

### The parameters

| Parameter | Value |
|---|---|
| `scoredBatchSize` | 1 |
| `prefillGainExponent` | 0.25 |
| `decodeGainExponent` | 0.75 |
| `pairsPerCohort` | 4 |
| `minPairsPerCohort` | 4 |
| `decodeSpeedupFloor` | 0.90 |
| `decodeSpeedupCeiling` | 5.0 |
| `kvBackend` | `contiguous` |

The width is fixed at 1. A width other than the declared one has no certified
series tag, and the benchmarker refuses it rather than run it.

THE B = 1 POINT IS RULED AHEAD OF THE PUBLISHED BENCHMARKER. At the published
channel tip the width certification accepts 8 only, so a fixture declaring 1 is
refused there today. The refusal is fail-closed and correct; this repository
declares the ruled shape and does not work around it.

`kvBackend` is pinned `contiguous` on both legs. The benchmarker refuses when
it cannot honour the pinned backend. It does not degrade to another backend.

### Token fidelity

The benchmarker applies a per-stream token-tolerance gate with a **10% budget**.

> **WARNING — the gate accepts similar output, not identical output.**
> This track does not require your output to match the serial trajectory token
> for token. The block-shaped forward pass diverges from the serial forward
> pass at near-tie argmaxes. The gate prices that divergence against the 10%
> budget. Do not read the gate as lossless.

The checked-in public goldens were captured on ranked hardware. A near-tie
argmax can diverge on other hardware, even for correct code. Before you treat a
local failure as your own regression, check whether an unmodified `main` fails
at the same token position on your machine.

### Current status

> **NOTE — the track is ARMED.**
> Three statements describe the arm state.

1. `fixtures/qwen3_8_125b_a6b_track.json` sets `official_scoring_enabled` to
   `true`. That flag is the single authority on the arm state, and the
   benchmarker enforces it: it refuses to seal an official scoring artifact
   while the flag is `false` or absent.
2. Every one of the 8 timed-pool slots, and the hidden correctness golden, pins
   a real golden by `{r2_path, sha256, bytes}` (the correctness golden pins the
   live golden). `tools/ranked-box-preflight.sh` verifies every staged golden
   against its pin -- byte count then sha256 -- and refuses on any mismatch.
   `live_golden` names the one live scored prompt (`botany`); the other 7 rotate.
3. The launch reference candidate is stock SERIAL (`num_speculative_tokens` 0).
   It is scored against the serial control leg, which runs the same engine on
   the same box, so the expected launch composite is approximately 1.000 within
   bands (prefill +/-5 %, decode +2 % up) -- a null control of the scored
   pipeline. The native MTP head is the participants' improvement path (a later,
   separate submission).

The arm state, the pins and the serial launch reference are repository state now.
Staging the goldens onto the ranked box and registering the runner remain
box/organizer work. Nothing in a submission can change the arm state or the pins
-- the contract fixture is trusted-side and is not an editable path.

> **NOTE — the track goldens are not in this repository.**
> The 8 timed-pool tapes and the 6 per-depth oracles are organizer material.
> They are published in R2 at the `r2_path` keys the contract pins. The ranked
> box stages them out of band into the directory its runner service exports as
> `MLXFAST_QWEN38_GOLDEN_DIR`. `tools/ranked-box-preflight.sh` verifies every
> file there against the contract's `{sha256, bytes}` and refuses an extra
> `*.json`. They are never in git, so your clone does not carry them. Keep every
> change prompt-independent and model-general.
>
> The organizer stages them with the signer this repository vendors:
>
> ```bash
> R2_BUCKET_ENDPOINT=... R2_ACCESS_KEY_ID=... R2_SECRET_ACCESS_KEY=... \
>   tools/fetch-goldens.sh --all --out "$MLXFAST_QWEN38_GOLDEN_DIR"
> tools/ranked-box-preflight.sh
> ```

There is ONE speculative arm on this track. `allowed_modes` declares `serial`
and `mtp` only. DFlash was a Gemma-era second arm and is removed.

## Submitting

Use the Yukon CLI for every account operation and every submission operation.

```bash
export PATH="${HOME}/.local/bin:${PATH}"
```

This command puts `yukon` on your path.

```bash
yukon login <api-key> --api <url>
```

This command authenticates you.

```bash
yukon clone <benchmark-id-or-name>
```

This command clones the benchmark repository.

Before you submit, remove build artifacts from the editable paths. The adapter
under `harness/` is a Cargo crate; its `target/` build tree is large and is not
part of a submission. The archive packages the editable-path directories as they
sit on disk, so a stale `target/` inflates the archive past the 25 MiB limit and
the submission is rejected before it uploads.

```bash
cargo clean --manifest-path harness/protocol-adapter/Cargo.toml
```

This command removes the adapter's build tree. `./setup.sh` and the ranked box
rebuild it from source, so removing it costs you nothing.

```bash
yukon submit --model "<exact model name>" --note-file submission-note.md
```

This command uploads your editable-path archive.

```bash
yukon submissions
```

This command lists your submissions.

A submission archive replaces the editable paths. It rejects generated
artifacts, symlinks, local scores, reference checkpoints, and any source change
outside the editable surface. `yukon submit` does not run a local test first.
No local run blocks the upload. Run the local test yourself before you submit.

## The pinned artifacts

| Artifact | Identity |
|---|---|
| Target model | `unsloth/Qwen3.8-Flash-Next-GGUF`, variant `UD-Q4_K_XL`. It is a GGUF conversion of `Qwen/Qwen3.8-Flash-Next` @ `f5d08274bafd880402bd16f5e3e6c514136ec06c`. |
| Target manifest | `fixtures/qwen3_8_125b_a6b_track.json` `target.files`, which pins each file by bytes and sha256 |
| MTP head | `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`, pinned in the same `target.files` list |
| Engine | vendored at `ds4/` from `Layr-Labs/ds4`, our port of `antirez/ds4`. `ds4/VENDOR.json` records the signed base. |
| Benchmarker | dist channel: branch `qwen3.8-125b-a6b-v1` on the public bench repository `Layr-Labs/mlxfast-bench`. The channel's `benchd.manifest.json` is the authority for the commit, the sha256, the byte count and the platform; `tools/fetch-benchd.sh` enforces all four. Nothing is pinned here, because the channel tip is the intended source. |

The staged snapshot holds the four target shards (111,334,654,784 bytes) and,
flat beside them, the Q8_0 MTP draft head (2,786,568,256 bytes). The fixture
pins all five files.

The model repository is public. It downloads without a token. There is no
organizer-hosted mirror for this checkpoint, so
`MLXFAST_REFERENCE_FALLBACK_BASE_URL` is empty by default.

### The target model

| Property | Value |
|---|---|
| Architecture | `qwen4_exp`. The text tower is `qwen4_exp_text`. |
| Hidden layers | 48, on a four-layer repeat |
| Full-attention layers | 12, at every index where `index % 4 == 3` |
| Linear-attention layers | The other 36. They are gated deltanet and carry a constant-size recurrent state. |
| Full-attention heads | 24 query heads, 2 KV heads, head dimension 256 |
| Rotary | Partial 0.25, `rope_theta` 1e7, interleaved mrope |
| QSA indexer | 4 heads, 1 KV head, dimension 128, budget 2048, compress 4 |
| Hyper-connections | `hc_count` 4, `hc_lowrank` 320 |
| Routed experts | 512, 10 per token, `moe_intermediate` 640, plus a shared expert of the same width |
| Hidden size | 2560 |
| Vocabulary | 248320 |
| Embeddings | Untied |
| Quantization | GGUF mixed: `Q4_K` routed expert gate and up, `Q5_1` routed expert down, `IQ4_NL` for the PLE table, `Q8_0`/F32/BF16 for the dense embedding and output |

## Building the engine

The scored engine is the `cuda-engine` adapter. `./setup.sh` builds and stages
it for you; to rebuild it directly:

```bash
tools/ds4/build.sh
```

This command builds the vendored ds4 engine, links the shim
library, and builds the adapter with its real ds4 backend
(`--features ds4-engine`). Off the box, build the adapter alone
without the feature for the deterministic mock backend, which needs no GPU and
no engine:

```bash
cargo build --release --manifest-path harness/protocol-adapter/Cargo.toml --bin cuda-engine
```

```bash
tools/stage-cuda-engine.sh
```

This command copies the finished `cuda-engine` binary to the fixed workspace path
benchd resolves and spawns (`.build/release/mlxfast-runtime-worker`). That name
is retained from the seed on purpose: it is a contract with the benchmarker, not
a preference, so this repository honours the path rather than renaming it. The
adapter carries no Metal kernels, so there is no `mlx.metallib`.

`cargo` is the only toolchain this repository builds with.

## Continuous integration

`.github/workflows/ci.yml` runs on every pull request and on every push to
`main`. It runs repository hygiene checks and the Rust adapter's `cargo`
build/clippy/test, all on
`ubuntu-latest`. There is no Swift build: the Swift seed package was removed in
the final de-Swift.

CI is advisory. No status check is required. A red run blocks neither a merge
nor a dispatch. `docs/ci-coverage.md` holds the detail.

CI never measures and never scores. CI holds no secret, downloads no weights,
and runs no GPU test. The GPU tests and the checkpoint tests are box-only.
`docs/ci-coverage.md` names them.

`.github/workflows/benchmark.yml` is the ranked pipeline. It triggers on
`workflow_dispatch` only. It holds no secret. Its ranked job runs on
`[self-hosted, Linux, ARM64, qwen3.8-125b-a6b-cuda-v1]` (the GB10 box). Before it measures, it verifies
every box-staged asset against the contract's `{sha256, bytes}` pins. It
publishes no score until the runner is registered, the box is staged, and the
benchmarker emits a composite. Each of those gaps gives a non-zero exit and no
artifact.

## Where to get help

| Question | Authority |
|---|---|
| What the track measures, path by path | `benchmark.json` |
| Pins, the timed pool, scoring values | `fixtures/qwen3_8_125b_a6b_track.json` |
| Why the manifest says what it says | `docs/participant-contract.md` |
| The engineering log for this port | `docs/qwen38-125b-a6b-port-notes.md` |
| The measured window and the decode target | `docs/timed-decode-evaluation.md` |
| What CI covers | `docs/ci-coverage.md` |
| Agent and contributor guidance | `AGENTS.md` |

> **NOTE — the order of authority.**
> The ranked box run is the authority on any score. The contract fixture
> `fixtures/qwen3_8_125b_a6b_track.json` wins over this document. This document
> only explains; it never overrides. If either disagrees with the benchmarker
> about measurement, the benchmarker wins.

## License and attribution

This repository's harness code is licensed per [LICENSE](LICENSE). The pinned
checkpoint is a GGUF conversion of `Qwen/Qwen3.8-Flash-Next`. The model's own
license terms apply to it. This repository distributes no model weights. [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) holds the full
third-party attribution.
