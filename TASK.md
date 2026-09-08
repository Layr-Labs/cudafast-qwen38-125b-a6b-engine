# The task — Qwen 3.8 125B A6B CUDA

Make the target text tower run faster.

> **The engine is the ds4 port (`Layr-Labs/ds4`).** The scored engine is the
> Engine Protocol v1 adapter (`harness/protocol-adapter`) with the ds4 C/CUDA
> engine linked in-process. The engine source is vendored at `ds4/`
> and it is read-only. `docs/qwen38-125b-a6b-port-notes.md` is the engineering
> record.

The ranked track is `qwen3.8-125b-a6b-cuda-v1`. Read [README.md](README.md) for the
setup steps and the repository structure. Read
[`docs/participant-contract.md`](docs/participant-contract.md) for the reasons
behind the rules.

## What you optimize

You optimize the engine. The scored engine is the Engine Protocol v1 adapter
with the ds4 engine linked in-process (`harness/protocol-adapter`); it reports
raw counters and the benchmarker does all timing and scoring. The
participant-editable surface is the engine adapter (`harness/`) and the MTP
head declaration. Today `benchmark.json` `editablePaths` lists three entries:
`harness`, `mtp-head` and `mtp-head.manifest.json`.

The MTP head proposes tokens. The target model decides every emitted token.

## What you may change

`benchmark.json` `editablePaths` is the authority. It lists three entries:
`harness` (the engine adapter), and `mtp-head` plus
`mtp-head.manifest.json` (the MTP head).

Depth 1 is one MTP draft per target-verified cycle. Depths 2 to 6 extend the
same chain in the engine.

The rule behind the list is simple. Code that **proposes** tokens or computes
the forward pass is editable. Code that **verifies**, **measures**, or
**ledgers** stays trusted.

The speculative head is the organizer's pinned weights. You may re-quantize it.
You may not replace it, and you may not upload head weights of your own.

`mtp-head/` is not an editable path, so a submission carries no head weight
file. The declaration file `mtp-head.manifest.json` stays editable, and it
accepts `"source": "pinned"` only. `"source": "remote"` and `"source": "in_branch"` are refused by name.

The head has a 2 GiB declaration cap. The size cap is the only gate on a
declaration. A declared `sha256` is optional, and the runner does not verify it.

A re-quantization happens ON LOAD, in memory. Nothing on disk changes, and no
artifact travels in a submission.

The seed's re-quantization-on-load seam lived in the vendored MLX model fork,
which has been removed. The participant seam that tunes the head on the CUDA
engine is deferred with the engine surface. Do not write into `mtp-head/`: the
ranked worker runs under a sandbox that denies file writes, and the benchmarker
refuses a changed head tree. `docs/participant-contract.md` section 4.4 is the
authority for the declaration rules that remain in force.

> **WARNING — the target quantization is frozen.**
> Do not re-quantize any target weight. Do not re-represent one. Do not change
> the numerical format of one. This holds even when the result passes every
> correctness gate. An editable transform does not license the change. The two
> speculative decoders are a narrow exception, and the exception is
> re-quantization only. You may re-quantize either one, within its 2 GiB
> declaration cap. You may not replace either one.

> **NOTE — batch size is locked. Draft depth is not.**
> The batch size stays 8. You may not tune it.
>
> The draft depth is a free lever, set from your own drafter code, which is
> editable. It is not pinned at 1.
>
> Select a depth from 1 to 6; the non-editable envelope refuses 7 and above, and benchd
> measures at depth 2 when the invocation names no depth.
>
> Each run seals `effective_spec` and `effective_mean_draft_len`, so the depth
> that ran and the draft length it realized are both visible afterwards.

You may not change anything that verifies, measures, or ledgers. This covers the
trusted harness, the target weights, the transform contract, the tokenizer, the
goldens, the gates, and the timing code.

## How to run it

```bash
./tools/fetch-benchd.sh
```

This command resolves and verifies the pinned benchmarker binary.

```bash
./setup.sh
```

This command builds the ds4 engine and the `cuda-engine` adapter, stages the
adapter, then — as box work that fails closed off the box — verifies the GGUF
target snapshot. See [README.md](README.md) for the toolchain each step needs.

```bash
cargo test --manifest-path harness/protocol-adapter/Cargo.toml
```

This command runs the adapter's own unit tests against a mock transport, with no
GPU and no checkpoint. It is the local signal available today; participant local
iteration on the CUDA engine is deferred with the editable surface.

## How it scores

```text
composite = prefill_gain ^ 0.25 * decode_gain ^ 0.75
gain      = baseline_aggregate / candidate_aggregate
```

The score is serial-anchored. A faster candidate scores above 1.

`aggregate` is the per-stream sum. Add each of the 8 concurrent streams' own
elapsed time together. Do this for prefill and for decode separately, on both
legs.

The ranked run is SINGLE-STREAM: scored batch size 1 (David ruling
2026-08-27). Each of the 8 pinned pool prompts is timed in its own window, over
a 1024-token seed and a 128-step decode window. The floor is 0.90. The ceiling
is 5.0. The KV backend is pinned `contiguous`.

The benchmarker applies a per-stream token-tolerance gate with a 10% budget.

> **WARNING — the gate accepts similar output, not identical output.**
> This track does not require token-for-token equality with the serial
> trajectory. The gate prices divergence against the 10% budget.

## The current state

> **NOTE — official scoring is NOT armed.**
> `fixtures/qwen3_8_125b_a6b_track.json` sets `official_scoring_enabled` to
> `false`, and every timed-pool slot is the
> `QWEN38-125B-A6B-CUDA-PENDING-ORGANIZER` sentinel.
> `tools/ranked-box-preflight.sh` refuses a ranked run while any sentinel is
> present, and `tools/qwen38-125b-a6b-measure-and-score.sh` refuses rather
> than emit a score.
>
> Arming the track is organizer work. Nothing in a submission can do it.

There is ONE speculative arm. `allowed_modes` declares `serial` and `mtp`
only. DFlash was a Gemma-era second arm and is removed from this track.

## Local runs are directional

Both the local test and the ranked run are single-stream, so the two exercise
the same width. What still differs is the machine, the pinned prompts and the
staged checkpoint. Treat a local score as a smoke signal, not as a prediction.
The ranked box run is the authority.

## Authorities

| Question | File |
|---|---|
| Editable paths, commands, scoring values | `benchmark.json` |
| Pins, the timed pool, scoring semantics | `fixtures/qwen3_8_125b_a6b_track.json` |
| Why the manifest says what it says | `docs/participant-contract.md` |
| What a measured run executes | the channel benchmarker (`tools/fetch-benchd.sh`) |

Where this document and the contract fixture disagree, the fixture wins. Where
either disagrees with the benchmarker about measurement, the benchmarker wins.
