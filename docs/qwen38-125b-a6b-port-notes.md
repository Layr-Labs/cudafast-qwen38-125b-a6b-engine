# Qwen 3.8 125B A6B CUDA — port notes

Track `qwen3.8-125b-a6b-cuda-v1`. These notes record the port of this
repository from its seed to the CUDA track. They are an engineering log. Where
they disagree with `benchmark.json` or
`fixtures/qwen3_8_125b_a6b_track.json`, those files win.

## 1. What this repository is

This repository was seeded from the Gemma 4 26B A4B MLX engine. The seed
carries a Swift harness, a vendored MLX runner and a vendored Metal kernel
tree. The CUDA engine is not that tree. The CUDA engine is an Engine Protocol
v1 adapter with the ds4 engine linked in-process; the `ds4/` submodule pins it
(sections 10 and 12). Sections 1-9 are the record of the vLLM era.

Read this section before you read the rest of the repository. Many files still
describe the seed.

## 2. The track identity

The canonical name construction is `{model}{version}-{parameters}-{platform}-v{N}`.
For this track that gives:

| Item | Value |
|---|---|
| Track id | `qwen3.8-125b-a6b-cuda-v1` |
| Benchmark name | `cudafast-qwen38-125b-a6b` |
| Leaderboard namespace | `qwen3.8-125b-a6b-cuda-v1` |
| Runner label | `qwen3.8-125b-a6b-cuda-v1` |
| Contract fixture | `fixtures/qwen3_8_125b_a6b_track.json` |
| Organizer sentinel | `QWEN38-125B-A6B-CUDA-PENDING-ORGANIZER` |
| Target checkpoint | `RadixArk/Qwen3.8-Flash-Next-NVFP4` @ `7b719225242aacd3dbd3f9407468c2ee9a9d2594` |

The MLX twin of this track is `qwen3.8-125b-a6b-cuda-v1`. The two tracks rank
the same model on different hardware. They have different track ids, different
leaderboard namespaces and different target checkpoints.

The two tracks SHARE one benchmarker. Section 4 states what that means.

## 3. The scoring ruling

David ruled the scoring on 2026-08-27. The fixture carries the ruling word for
word in `scoring_semantics.ruling_verbatim`. The ruling is:

> score the qwen 3.8 125b-a6b tracks (mlx and cuda) single-stream, paired
> serial vs the built-in mtp, on prefill gains ^ .25 * decode ^ .75

The enforced values that follow from it are:

| Field | Value |
|---|---|
| `scored_batch_size` | 1 |
| `scored_exponents.prefill_gain_exponent` | 0.25 |
| `scored_exponents.decode_gain_exponent` | 0.75 |
| `benchmark.json` `scoring.mode` | `qwen-native-mtp-paired-decode-only` |

`scored_batch_size` 1 is RULED AHEAD OF THE PUBLISHED BENCHMARKER. At the
published channel tip, the width certification accepts B = 8 only, so a
fixture that declares 1 is refused there. The refusal is fail-closed and
correct. This repository declares the ruled shape. It does not work around the
refusal.

`tools/lint-benchmark-manifest.py` pins these values per track. A change to
any of them must move the fixture, the manifest and that linter together.

## 4. The benchmarker channel and the platform expectation

`tools/fetch-benchd.sh` resolves `benchd` from a release channel. There is no
`benchd.pin` file any more.

THE BRANCH IS THE PROJECT, NOT THE TRACK. The MLX and CUDA tracks of this model
share one bench release branch, `qwen3.8-125b-a6b-v1`, on the development bench
repository. They share it because they share one benchmarker. The track id
stays platform-specific.

THE CHANNEL HOLDS ONE PAIR PER PLATFORM. `dist/` carries the
`aarch64-apple-darwin` pair for the MLX box. `dist/linux-aarch64/` carries the
`aarch64-unknown-linux-gnu` pair for this track's box (bench pull request 219).
Both pairs carry the same six manifest fields, so the branch check, the byte
count and the digest all PASS on the wrong pair.

THE PLATFORM IS THEREFORE CHECKED, TWICE:

1. the manifest's `target_triple` must equal the triple this host expects; and
2. the binary's own container format, read from its first four bytes, must
   match the operating system that triple names.

The second check reads the bytes, so it catches a mis-stamped publish that the
first check would accept. The expectation also picks the directory, so the
wrong pair is not reachable by accident; the check still runs after the
download, because the directory a request went to is not proof of what came
back.

NOTHING IS PINNED HERE. The sha256 and the byte count change on every
republish. They are read from the manifest beside the binary.

`BENCHD_EXPECT_TARGET_TRIPLE` names the expectation when you need to check a
dist for another host. A host the channel publishes no lane for refuses rather
than guesses.

`tools/test-fetch-benchd-platform.sh` holds this gate. It is hermetic and runs
in CI.

## 5. The goldens

The top-level fixtures in `correctness_prompts/`
(`public_longcopy_gate_english_1024*.json`) are seed-era captures kept for reuse
of the 1024-token prompts. The model-identity loader rejects them against a Qwen
target; that rejection is the fail-closed direction.

The track goldens live in `correctness_prompts/qwen3.8-125b-a6b-cuda-v1/`. They
are Qwen captures (`model_type: qwen4_exp_text`,
`RadixArk/Qwen3.8-Flash-Next-NVFP4` provenance). There are 8, one per timed-pool
slot. NONE of them carries a baseline pair: the ranked path is PAIRED, and the
serial-control leg measured beside the candidate supplies the denominator
(section 5 of `docs/participant-contract.md`). A golden that carries
`benchmark.baseline_prefill_seconds_per_token` or
`benchmark.baseline_decode_seconds_per_token` is refused on the ranked path, and
`tools/lint-benchmark-manifest.py` refuses one at rest. `index.tsv` pins each by
`{sha256, bytes}`, and `fixtures/qwen3_8_125b_a6b_track.json` pins the same
`{r2_path, sha256, bytes}` in `timed_prompt_pool[]`. `live_golden` names the one
live scored prompt (`botany`); the other 7 rotate.
`hidden_correctness_golden` pins the token-fidelity oracle to the live golden.
`tools/ranked-box-preflight.sh` verifies every staged golden against its pin,
byte count then sha256, and refuses on any mismatch.

LAUNCH STATE (David MTP-0 ruling). The launch reference candidate is stock
SERIAL (`num_speculative_tokens` 0). The CANDIDATE leg's serve follows the
submission's declared spec; the SERIAL CONTROL leg is serial whatever the
candidate declares. benchd boots each leg's resident itself, from that leg's own
workspace, with `tools/serve-up.sh --boot --spec serial|mtp --draft-len N
--socket-out FILE`, and ends it with `--stop --socket PATH`. Leg 1 is always
`--spec serial --draft-len 0`, and in `--boot` mode the flag is the authority: a
disagreeing `SERVE_UP_SPECULATIVE` is refused, not honoured. NOTHING SETS THAT
VARIABLE ON THE RANKED PATH -- not the workflow, not the measure wrapper -- and a
box-preset value is refused by `tools/ranked-box-preflight.sh` section 7 and by
`tools/qwen38-125b-a6b-measure-and-score.sh`.

Because the launch reference is serial and the control leg is serial on the same
box, the launch reference's expected composite is approximately 1.000 within
bands (prefill +/-5 %, decode +2 % up) -- a NULL CONTROL of the scored pipeline;
a value outside approximately 1 +/- band is a finding, not a pass. The native MTP
head is the participants' improvement path (a later, separate submission).
Human-facing numbers show tokens per second: per-token times are stored in
seconds per token (benchd's internal representation), and 0.06451959972265625
s/tok displays as 15.50 tok/s.

## 6. What the DFlash removal took out

This model carries a NATIVE MTP head. The head is a SEPARATE file in the pinned
snapshot, `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`, and ds4 opens it as a
second model through `--mtp-model` / `ds4_engine_options.mtp_path`. The fixture records
that shape in `mtp_head` (`packaging: separate_gguf`, `file`, `engine_option`).
The earlier `tensor_prefix: language_model.mtp.` described the Baekpica fork,
which read the head out of the target checkpoint; that fork is gone and the
line went with it. DFlash was a Gemma-era second speculative arm. This track
does not use it, so it is deleted rather than carried.

The removal took out the DFlash sources, the DFlash tests, the DFlash
fixtures, the DFlash staging script, the DFlash head declaration and its
`editablePaths` entries, the DFlash arm-selection seam and its test, and the
DFlash constants. `allowed_modes` is now `serial` and `mtp` only.

The STAGED MTP assistant head did NOT go with the DFlash removal. This track's
head is native, embedded in the target checkpoint, so `setup-gemma4-assistant.sh`
and the assistant-head loader beside it were dead weight here too. They belonged
to the seed's engine tree, not to the DFlash arm, and they were removed with the
Apple-Metal runtime scrub. Section 7 lists the remaining seed residue.

## 7. What this sweep did NOT move

Name these when you read the tree, because they still describe the seed.

* THE SWIFT MODEL GEOMETRY KEPT FROM THE SEED IS STILL GEMMA.
  `Sources/MLXFastCore` `Constants.swift` and `Sources/MLXFastTransform` carry
  Gemma geometry and the Gemma checkpoint identity, and `requiredGoldenModelType`
  is still `gemma4_text`. These move with the engine port, not with the track
  identity, because a gate holding some fields of one model and some of another
  rejects every checkpoint and explains none of them.
* THE PARTICIPANT-EDITABLE ENGINE SURFACE IS THE ADAPTER.
  The seed's MLX runner surface (Swift model files and vendored Metal kernels)
  was removed with the Apple-Metal runtime. `benchmark.json` `editablePaths`
  now lists three entries: `harness`, `mtp-head` and
  `mtp-head.manifest.json` (section 10).
* THE RANKED PIPELINE IS NOW LINUX-AUTHORED, WITH TWO BOX STEPS LEFT.
  `.github/workflows/benchmark.yml` runs a Linux/aarch64 self-hosted job on the
  GB10 box (`[self-hosted, Linux, ARM64, qwen3.8-125b-a6b-cuda-v1]`). The timed
  step holds the shared box GPU lock (`/tmp/mtplx-gpu-exclusive.lock`, the
  spark-gate lock) and checks quiescence with `/proc/loadavg` and `nvidia-smi`.
  Two things are still box/bench work, not workflow work, and each fails closed
  until it lands: (1) the GPU temperature reader benchd's cool gate uses on the
  GB10 box (the macOS twin used `macmon`; `tools/ranked-box-preflight.sh`
  refuses a run while no reader is present); and (2) `./setup.sh`, the engine
  build and on-box checkpoint verification, which is the engine-body lane's.
  The fixture is ARMED (`official_scoring_enabled: true`), every pool slot is
  pinned and `live_golden` is `botany`, so a dispatch waits on those two box
  steps -- plus the reference workspace and this box's calibration -- not on the
  fixture.
* THE CUDA CHECKPOINT MANIFEST IS NOT AUTHORED. There is no per-file
  `{sha256, bytes}` fixture for `RadixArk/Qwen3.8-Flash-Next-NVFP4`. The box
  stages `target.sha256` and `target.bytes` beside the snapshot
  (the retired spark-gate protocol, section 8). Authoring the fixture needs the
  box.
* THE TARGET LICENCE IS NOT RECORDED. The fixture carries no `license` block,
  because no licence fact for this checkpoint is verified in this repository.

## 7a. Retired-DFlash residue, named rather than left to be found

The DFlash ARM is gone: the routes, the sessions, the spec module and
capability, the loaders, the CLI verbs, the client verbs, the constants, the
fixtures, the tests and the two `editablePaths` entries. What is left is
inert text and one dead wire field. It is listed here so a reader who greps
`dflash` and finds hits knows which hits are expected.

* `TrustedWorkerEffectiveSpec` still declares `dflash: EffectiveDFlash?`. It is
  a sealed struct's dead key: nothing sets it and nothing reads it.
* `Gemma4RuntimeWorker.swift` (near the top) cites
  `experimentalDFlashWorkerHello`, which is deleted.
* Stale comments in `HeadRequantOnLoadTests`, `HarnessHashRootSetTests` and
  `MTPWorkerTwinEqualityTests`.
* `THIRD_PARTY_NOTICES.md` still credits the Laguna DFlash speculator, which
  this tree no longer carries.
* Two never-populated optional response fields for the retired `dflash_*`
  request kinds, and the `dflash-head` spelling kept in the forbidden-directory
  list of `Gemma4BenchmarkManifestTests` -- that one is KEPT on purpose, the way
  the `benchd.pin` spelling is, so a re-added directory is refused on arrival.

Clearing the first four is a follow-up. None of them is reachable and none
changes a measured value.

## 8. Follow-up work

1. Author the Engine Protocol v1 adapter over the pinned vLLM base, and move
   `editablePaths` onto it.
2. DONE for the box orchestration: `.github/workflows/benchmark.yml` now runs
   the Linux/aarch64 GB10 job (labels, GPU lock, quiescence). Two box/bench
   steps remain, both fail-closed: the Linux GPU temperature reader for benchd's
   cool gate, and `./setup.sh` (engine build + on-box checkpoint verify), which
   belongs to the engine-body lane.
3. Author the CUDA checkpoint manifest fixture and a track-fixture test that
   pins the geometry against it. The Gemma track-fixture test was deleted with
   the Gemma fixture; there is no geometry pin to gate until that fixture
   exists.
4. DONE: the 8 track goldens are regenerated as Qwen captures under this track's
   tokenizer, each carrying its serial baseline, pinned in `index.tsv` and the
   fixture's `timed_prompt_pool[]` (`live_golden: botany`). Scoring is armed at
   the serial launch reference (MTP-0).
5. Record the target licence.
6. Finish removing the staged-head residue. The staging script
   (`setup-gemma4-assistant.sh`) and the Swift assistant-head loader were removed
   with the Apple-Metal runtime; `mtp-head/` and
   `fixtures/gemma4_assistant.sha256` remain and are slated for removal with the
   rest of the seed.

## 9. The measurement wrapper (measure-and-score.sh)

`tools/qwen38-125b-a6b-measure-and-score.sh` is `benchmark.json`'s
`benchmarkCommand` and `preSubmitCommand`. This section holds the design
rationale that used to live in the script's header.

**Why the measure-job seam, not the facade.** The qwen manifest's
`benchmarkCommand` targets the `benchd iterate` facade (vendored as
`tools/benchmark.sh`). `benchd iterate` is the single-run legacy path and
never touches `scored_batch_size`, `per_cohort` or `ScoredBatchPoint`; those
live only in `benchd measure-job`. So this track's wrapper targets
`measure-job` directly and then converts its `results.json` into the
`{score, metrics}` shape `src/benchmark/score.ts` requires. The wrapper is
TRUSTED-side: it is not in `editablePaths`, so a submission cannot rewrite the
measurement pipeline from inside its archive.

**The ruled pair count.** `--target-pairs` and `--min-pairs` are both 4, the
ruled contest parameter (David 2026-08-26, "you run it using 4 pairs instead of
2 of 8 batches"). The served benchmarker enforces it: an official run whose
`target_pairs` or `min_pairs` is not `PAIRS_PER_COHORT_TARGET` is refused
pre-GPU, by name. That constant is `4` on the served channel
(qwen3.8-125b-a6b-v1, source_commit `8439d6fe`, verified 2026-08-30), so the
wrapper's `--min-pairs 4 --target-pairs 4` is accepted. The wrapper's literals
are a belt-and-suspenders declaration of the ruled floor at the call site; the
guarantee that a published median covers 4 pairs comes from the benchmarker's
refusal, which no argv can talk past. `benchmark.json` `scoring.pairsPerCohort`
and `tools/lint-benchmark-manifest.py` are the authority on the value and its
supersession chain.

**The thermal contract is benchd's, not the wrapper's.** Every timed phase runs
behind the benchmarker's own per-platform cool-down gate: a 50 °C threshold on
CUDA/GB10 (40 °C on Mac), sealed as `cool_gate_c` with source
`platform-cool-gate`, a 900 s ceiling per phase, prefill and decode gated
separately on fresh workers. The gate is rejected -- with one gated retry -- on
throttling under load, missing telemetry or a token mismatch. Nothing in the
wrapper can relax any of that; the one thing the caller owes the benchmarker is a
usable GPU temperature reader, which `tools/ranked-box-preflight.sh` verifies on
the ranked box before setup (nvidia-smi on Linux/CUDA, macmon on macOS).

## 10. The engine is upstream ds4 (2026-09-02, superseded in part by section 12)

The vLLM engine is gone. The scored engine is the Engine Protocol v1 adapter
(`harness/protocol-adapter`) with the upstream ds4 engine linked in-process.

The engine was first a fork, `Baekpica/ds4-dfm-rs`, carried as the submodule
with two patches over it in `ds4-overlay/`. That fork, the overlay and the
fork-only environment are all gone: the submodule is UPSTREAM `antirez/ds4`.

- Engine source: `antirez/ds4`, the `ds4/` git submodule, pinned at
  `110afdd8886586f18fc9b28bc5533152dd10e728`. Upstream carries two model
  families and four variants; the Qwen 3.8 Flash Next (`qwen4exp`) family is
  NOT among them. The shim therefore refused a `qwen4exp` GGUF by name, so
  this engine could not run the pinned target. **Section 12 supersedes this
  bullet:** the submodule now pins our port and the refusal is gone.
- Build: `tools/ds4/build.sh` copies the pinned tree, builds upstream's own
  `cuda-spark` target (upstream's name for `CUDA_ARCH=sm_121`) with `-fPIC`
  through `CC` and `NVCC`, links the core objects with
  `harness/protocol-adapter/ds4_shim/ds4_shim.c` into
  `.build/ds4/libds4qwen.so`, links `.build/ds4/ds4-resident` against it, and
  builds the adapter with `--features ds4-engine`.
- Target: `unsloth/Qwen3.8-Flash-Next-GGUF`, variant `UD-Q4_K_XL`: four shards
  plus a Q8_0 native MTP draft head. The fixture's `target.files` pins every
  file by bytes and sha256.
- Serving shape: ONE RESIDENT ENGINE FOR EACH WINDOW. benchd starts a fresh
  `cuda-engine` for each phase, and an official run would otherwise load the
  103.7 GiB artifact 16 times or more, which no 20-minute pipeline survives.
  `tools/serve-up.sh` therefore starts one `ds4-resident` process
  (`harness/protocol-adapter/ds4_shim/ds4_resident.c`, linked by
  `tools/ds4/build.sh`) inside the caller's GPU-lock window. That process opens
  the body and the MTP draft head one time and serves the `ds4_shim.h` verbs
  over a Unix socket. Each phase's worker connects and loads nothing; the
  measured reconnect is 0.266 ms. The serve script plans the memory and refuses
  before any load when the RESIDENT set plus headroom does not fit. The n-gram
  table is not part of that set: the engine streams it from the solid-state
  disk, so the plan subtracts it (`docs/ds4-resident.md` section 6). The plan
  refuses outright against a pin whose engine does not declare that behaviour. Upstream's own
  `ds4-server` is built by `cuda-spark` and is NOT used: it speaks
  OpenAI/Anthropic chat over HTTP and carries no logits, no token-id input, no
  teacher-forced eval and no speculative counters. `docs/ds4-resident.md`
  carries the mapping table and the evidence.
- Engine environment: `DS4_MODEL`, `DS4_MTP_PATH`, `DS4_MTP_DRAFT_TOKENS`,
  `DS4_CTX_SIZE`, `DS4_EOS_IDS`, `DS4_ENGINE_IDENT`, plus `DS4_RESIDENT_SOCKET`
  and `DS4_RESIDENT_READY_FILE` for the resident. `DS4_LOCK_FILE` is no longer
  keyed per run: upstream's exclusive instance lock defaults to
  `/tmp/ds4.lock`, and with ONE resident for each window that default is what
  we want -- a second resident on the box would be an OOM, and the lock refuses
  it. The
  fork-only variables (`DS4_QWEN_PLE_*`, `DS4_QWEN_PREFILL_CHUNK`,
  `DS4_QWEN_MTP_QUENCH`, `DS4_SESSION_LAZY_GRAPH`) are gone: upstream reads
  none of them. `DS4_CUDA_WEIGHT_IPC_MANIFEST` is gone for a different reason.
  It was the CUDA IPC handle of an earlier topology: one `ds4_weight_server`
  held the weights, and each per-phase worker attached to that handle and then
  opened its OWN engine session. The topology is now the resident socket.
  `tools/serve-up.sh` starts one `ds4-resident`, which holds the engine, and
  each worker connects on `DS4_RESIDENT_SOCKET` and opens no session of its own
  (`docs/ds4-resident.md`).
- MTP: upstream takes the draft head as a SEPARATE model
  (`ds4_engine_options.mtp_path`), so `serve-up.sh` exports `DS4_MTP_PATH` on
  the speculative leg only and maps the declared `num_speculative_tokens` N to
  `DS4_MTP_DRAFT_TOKENS=N+1`. There is no quench environment variable to set:
  upstream's adaptive disable is its DSpark scheduler, a different code path
  from the one the shim drives, and it publishes no quench counter. The
  adapter's quench tripwire stays armed and the shim reports 0 for it.
- Counters: upstream publishes no speculation counters, so the shim counts
  cycles and accepted drafts itself from what each cycle commits.
- Bit-exactness gate: `tools/ds4/mtp-exactness-gate.py` runs every pool
  prompt serial and MTP and requires identical token streams.

- The scored leg requests its spec. benchd's single-leg path only sends a
  spec when `benchd iterate --mtp-depth N` names one; the measure script
  passes the declared depth and refuses on a benchd without the flag.
  The adapter refused depths 2 and 3 by name while the engine implemented only
  depth 1, so the fixture's `mtp2` and `mtp3` oracle entries were unreachable.
  SUPERSEDED by section 14: the `e2f86b7` sync implements depths 1 to 3.

What this change does NOT do: the goldens under `correctness_prompts/` and the
per-depth oracles were authored on the vLLM engine over the NVFP4 checkpoint.
They do not describe this engine. `official_scoring_enabled` is `false` until
the goldens are re-authored on ds4 and the serial baseline is re-pinned.

## 11. Re-authoring the goldens on ds4 (tooling, 2026-09-03)

`tools/qwen4exp-golden-reauthor.sh` re-authors this track's goldens on the ds4
engine. It captures the artifacts, it validates them, and it prints the pins.
It does not write the fixture and it does not arm the track.

### 11.1 What the tool captures

The tool writes these files into `--out`:

| File | What it holds |
|---|---|
| `prompts/<name>.tokens.json` | the 1024 prompt ids of one case |
| `goldens/<name>.golden.json` | the depth-0 golden of one prompt |
| `goldens/botany.mtp1.golden.json` | the depth-1 oracle of the live golden |
| `mtp1-exactness.txt` | the serial-against-MTP report, one line for each prompt |
| `index.tsv` | the name, the sha256 and the byte count of each artifact |
| `fixture-pins.json` | the pin patch |
| `negative-control.txt` | the refusal of the perturbed golden |

### 11.2 The prompts do not move

A golden binds three things: the weights, the prompts and the prompt SHAs. This
change moves the engine and the checkpoint. It does not move the prompts.

The tool reads each pool prompt from the golden that already carries it. Before
it reads a golden, it compares that golden with the contract pin. It compares
the byte count first, then the sha256. It stops if either value disagrees.

The public correctness prompt has no Qwen capture. The checked-in
`public_longcopy_gate_english_1024_*.json` pair uses the Gemma tokenizer, so its
ids are not valid Qwen ids. The tool tokenizes the checked-in text with the
target tokenizer (`ds4 --dump-tokens`) and keeps the first 1024 ids.

### 11.3 The engine is driven through the adapter

A golden holds token ids and one teacher-forced argmax for each position. The
ds4 CLI writes text, so the CLI cannot author a golden. The Engine Protocol
adapter (`cuda-engine`) has the teacher-forced verbs, so
`benchd record-correctness-golden` drives the adapter.

Each capture spawns a fresh adapter. Each adapter attaches to the one resident
engine that `tools/serve-up.sh` starts. The weights load one time for each
serve.

The run uses two serves. The depth-0 goldens describe the scored serial serve,
which loads no draft head, so they are captured with `SERVE_UP_SPECULATIVE=0`.
The depth-1 oracle needs the head, so it is captured with
`SERVE_UP_SPECULATIVE=1`. One speculative serve for both would save one load and
would describe a serve that the serial leg never runs.

The full run holds the box GPU lock around both serves.

### 11.4 The per-depth oracle

`record-correctness-golden` sends no `spec`, and an absent `spec` is the serial
route (`harness/protocol-adapter/src/adapter.rs`, `resolve_spec`). The recorder
therefore authors the depth-0 oracle only.

`tools/ds4/free-run-capture.py` captures the free run at a declared depth.
`tools/qwen4exp-golden-edit.py graft-decode-oracle` then writes the depth-0
golden again with only `benchmark.expected_decode_tokens` replaced. This is the
shape the contract already pins: the mtp1 and mtp2 goldens differ from the
serial golden in that array alone, because `cases[]` is teacher-forced and does
not change with the route.

`tools/ds4/mtp-exactness-gate.py` then runs each prompt on both routes and
compares the two token streams. A mismatch is a result, not a failure of the
run. The report is kept in both conditions.

### 11.5 The two refusals

| Name | Condition |
|---|---|
| `official-scoring-armed` | the contract has `official_scoring_enabled: true` |
| `engine-pin-mismatch` | the `ds4` gitlink is not `serve_configuration.engine_pin`, or the two contract pins disagree |

The first refusal keeps the tool away from a scoring track. The second keeps the
goldens attached to one engine commit.

### 11.6 The negative control

`tools/qwen4exp-golden-edit.py perturb-one-token` changes one id of
`cases[0].expected_tokens` and keeps the byte count the same. It moves one
decimal digit, so the file length does not change.

`benchd validate-golden` must reject that file against the pin of the golden
it came from. The run stops when the exit status is not 1. This proves that the
`{sha256, bytes}` pin binds the tokens, and not only the file length.

### 11.7 How to run it

Do a dry run first. It prints each command and it runs none of them.

```bash
tools/qwen4exp-golden-reauthor.sh --dry-run --weights <TARGET_SNAPSHOT>
```

Then do the full run on the box.

```bash
tools/qwen4exp-golden-reauthor.sh --weights <TARGET_SNAPSHOT>
```

The tool needs two benchd binaries: `benchd` and `record-correctness-golden`.
`tools/fetch-benchd.sh` resolves both from the release channel and stages them
together in `benchd-bin/`, so the run finds the recorder beside the benchd it
resolved. Nothing more is necessary.

The channel publishes the recorder from `source_commit` at or after the
`mlxfast-bench` pull request #255 republish. A channel manifest older than
that carries the six top-level fields only. It declares no `binaries` entry for
the recorder, `tools/fetch-benchd.sh` stages `benchd` alone, and the run stops
with the named `missing-tool` refusal. On such a channel, build
`record-correctness-golden` from the benchd source at the commit the channel
names, then name it:

```bash
tools/qwen4exp-golden-reauthor.sh \
  --weights <TARGET_SNAPSHOT> \
  --recorder <PATH-TO-record-correctness-golden>
```

`--recorder` always wins over the staged copy.

`tools/test-qwen4exp-golden-reauthor.sh` proves the mechanics off a GPU. It
drives the adapter mock backend, so its goldens describe nothing. It needs
`benchd` and `record-correctness-golden` from a benchd checkout, which CI
cannot build, so CI does not run it.

```bash
BENCHD_SRC_DIR=<benchd checkout> tools/test-qwen4exp-golden-reauthor.sh
```

### 11.8 One thing this tool cannot do

This is a limit of the benchmarker, not a decision of this repository.

(The recorder's omission of a baseline pair used to be listed here as the second
limit, with an instruction to stamp a measured pair into each golden. It is
neither a limit nor an instruction now: writing NO pair is the REQUIRED shape.
The ranked path is paired, the control leg measured beside the candidate is the
denominator, and a golden that carries a pair is refused.)

1. **Only depth 1 is reachable.** The adapter refuses draft depths 2 and 3 by
   name. The `mtp2` and `mtp3` oracle entries stay as the contract carries them.
   SUPERSEDED by section 14: since `e2f86b7` the engine implements depths 1 to
   3 and all three are reachable.

### 11.9 What stays with David

The tool prints the pin patch and writes it to `fixture-pins.json`. It does not
apply it. These steps are David's:

- copy the goldens into `correctness_prompts/qwen3.8-125b-a6b-cuda-v1/`;
- write the pins into `fixtures/qwen3_8_125b_a6b_track.json`;
- set `official_scoring_enabled` to `true`.

## 12. The submodule is our ds4 port (2026-09-03)

Lane L12a of epic #71. The `ds4/` submodule no longer pins upstream: it pins
the Layr-Labs port, which carries the `qwen4exp` family the target needs. The
shim's name refusal is gone, and the repository builds, links and drives the
port.

- Submodule: `git@github.com:Layr-Labs/ds4.git` (`.gitmodules`), pinned at
  `278b799b974cb580e0f96ed3dce68bfbe0d8675b`, the head of the port's
  `qwen4exp/integrate-signed` branch. The same sha is the fixture's
  `serve_configuration.engine_pin` and `target.engine_pin`;
  `tools/ranked-box-preflight.sh` refuses when the fixture and the gitlink
  disagree. The port's `main` is a signed snapshot of upstream `110afdd` with
  an identical tree, so the fork point is recorded rather than described.
- What the port adds: a multi-shard GGUF loader for the unsloth artifact; the
  qwen4exp kernels for Metal and CUDA, the CUDA translation unit built without
  fast math (`QWEN4EXP_NVCCFLAGS`); the serial session verbs the graph serves,
  with every other verb refusing by name; the nextn MTP head behind
  `--mtp-model` (support kind `DS4_SUPPORT_QWEN4EXP_NEXTN`); the depth-1
  speculative cycle behind `ds4_session_eval_speculative_argmax`, greedy only;
  and a memory plan that keeps the PLE table SSD-resident
  (`DS4_QWEN4EXP_MEM_PLE`, `DS4_QWEN4EXP_MEMORY_HEADROOM_BYTES` in
  `ds4_qwen4exp.h`).
- The shim (`harness/protocol-adapter/ds4_shim/ds4_shim.c`): the GGUF
  architecture probe and the `qwen4exp not yet ported` refusal are deleted.
  `ds4s_open` passes the FIRST shard as `model_path` -- the engine reads
  `split.count` from it and maps the rest -- and the head as
  `ds4_engine_options.mtp_path` with `mtp_draft_tokens = depth + 1`. The port
  refused depths 2 and 3 at the time (section 14: it implements 1 to 3 now)
  and a `DS4_QWEN_MTP_QUENCH` that is not 0 at open
  (`ds4_qwen4exp_mtp_depth_from_draft_tokens`,
  `ds4_qwen4exp_mtp_check_no_yield_guard`), so the shim adds no guard of its
  own. `ds4s_eval_speculative` reaches the port's cycle unchanged;
  `ds4s_invalidate` is the port's full reset (every cache, recurrent state,
  conv history, indexer tape and n-gram history back to position 0);
  `ds4s_top_logits` still asks for exactly 8.
- Counters: `278b799` publishes them. `ds4_session_qwen4exp_spec_counters()`
  (`ds4.h`) returns the cycle's own `drafted`, `accepted` and `quenches`, so
  `ds4s_spec_counters()` READS THE PORT and reconstructs nothing; the shim
  keeps no counter state at all. It takes the handle now, because the port's
  counters belong to a session. `quenches` is still 0 by construction.
- The fourth counter, `verify_replay_disagreements`, is DIAGNOSTIC. On a
  rejecting round the cycle compares the batched verify's row-0 argmax with a
  one-row replay of the same position and counts where they differ; the replay
  stands, so it is never a refusal. It crosses the shim, the resident socket
  (`"disagreements"`) and the session seam into
  `FreeRunResult.verify_replay_disagreements`. NOTHING ENFORCES A CEILING. It
  now also travels the Engine Protocol v1 wire, which PR #99 added once benchd
  could parse it; `None` there means NOT REPORTED and is not `Some(0)`. The
  ordering that carries it is stated in `protocol.rs` and
  `docs/ds4-resident.md`: the box's benchd pair must be at or past bench
  `db3b73e`, because an older one parses the free-run response with
  `deny_unknown_fields` and rejects the whole line.
- Build: `tools/ds4/build.sh` is unchanged in shape. Its `CORE_OBJS` list
  gained the port's four objects -- `ds4_cuda_qwen4exp.o`,
  `ds4_qwen4exp_ple.o`, `ds4_qwen4exp_mtp.o`, `ds4_qwen4exp_mtp_hooks.o` --
  and still refuses when the list has drifted from the pin's Makefile.
- The NULL-logits defect this lane first pinned is FIXED at `278b799`. Pin
  `6526f08` passed `NULL` as the cycle's logits buffer and the cycle read it,
  so no `mtp1` leg could run. `278b799` passes `s->logits`
  (`ds4.c:75861-75864` at this pin), so the cycle leaves the frontier's logits
  where `ds4_session_argmax()` reads them, which is what the adapter calls
  after every cycle.
- CI: `Layr-Labs/ds4` is an INTERNAL repository and engine CI holds no
  credential, so a credential-free `git submodule update --init` cannot read
  it. Both ds4 steps in `.github/workflows/ci.yml` therefore probe the pin and,
  when it is unreachable, print a `::warning::` and skip. Until the port is
  readable from this repository's Actions, the off-box CUDA build and the CPU
  syntax check run only where the pin is reachable -- a developer machine or
  the box. `docs/ci-coverage.md` records the gap.

## 13. The engine is vendored, and editable (2026-09-04)

David: a participant must be able to EDIT the ds4 implementation and have their
change built and scored, the way the Gemma track works. A submodule pinned to an
INTERNAL repository gave them none of that -- they could not read it, fork it,
diff it or submit against it, the ranked box needed a git identity to fetch it,
and CI skipped every engine step because it may hold no credential.

- `ds4/` is now plain files: 339 files, 18,244,410 bytes, exported from the
  signed commit the fixture already pinned. `ds4/VENDOR.json` records the fork
  sha and tag, the antirez/ds4 fork point, and the excluded paths.
- Excluded from the export: `gguf-tools/imatrix/dataset`,
  `gguf-tools/quality-testing/data`, `speed-bench`, `dir-steering/out` and
  `misc`. All data, no source: 96 MB of corpora and captures that no build
  target reads, plus a directory the port's own `.gitignore` carries. Every
  source file is kept and every Makefile target still resolves.
- `tools/ds4/vendor-sync.sh` re-vendors from a signed tag, verifies the
  signature, and rewrites VENDOR.json and both fixture `engine_pin` fields
  together. Port development stays in Layr-Labs/ds4 and lands here by script.
- `benchmark.json` `editablePaths` gains `ds4`. Nothing in `ds4/` verifies,
  measures or ledgers, so the whole engine is editable rather than a curated
  subset the way Gemma's vendored framework needs. The byte caps moved with it
  by the 1 MiB-margin method, and
  `.github/scripts/submission-static-review-checks.sh`'s fallbacks with them.
- RULED 2026-09-04. `engine_pin` is PROVENANCE -- the engine base the baseline
  and goldens were authored on -- and NOT a byte-identity claim. Nobody may
  re-add an equality check between it and the engine content: it would refuse
  every submission. Correctness goldens plus the calibrated baseline are what
  gate a participant.
- RULED 2026-09-04. `ds4` must NOT be added to benchd's `HARNESS_HASH_ROOTS`.
  It is the EDITABLE scope, and a harness hash that moved with every
  participant edit would defeat its purpose: the hash exists to say the trusted
  surface is unchanged, so folding the participant's own surface into it makes
  it say nothing. `tools/test-participant-engine-edit.sh` asserts the roster
  still excludes `ds4`, so a future addition turns that suite red.
- `tools/ranked-box-preflight.sh` no longer compares a gitlink to the fixture:
  against an editable engine that would refuse every submission. It checks
  PROVENANCE (VENDOR.json base == fixture pin) and leaves the numbers to the
  correctness goldens. `tools/qwen4exp-golden-reauthor.sh` keeps the strict
  reading and also refuses a dirty `ds4/`, because goldens must describe the
  reference engine.
- The build cache keys on the vendored tree's CONTENT, not a gitlink, so a
  participant's edit misses the cache and is rebuilt.
- `tools/test-participant-engine-edit.sh` is the proof, and it is hermetic.

## 14. The e2f86b7 vendor-sync (2026-09-04)

The first re-vendor through `tools/ds4/vendor-sync.sh`, and the first with real
engine changes behind it.

- Pin `278b799` -> `e2f86b7` (tag `pin-e2f86b7`, signature verified at vendor
  time). 340 files, 18,351,471 bytes.
- THE DEPTH ENVELOPE MOVED. `DS4_QWEN4EXP_IMPLEMENTED_DEPTH` is 3, so depths 1,
  2 and 3 all run and only 4 and above are refused. The adapter's
  `DS4_IMPLEMENTED_DEPTH` moved with it, and a new test reads the vendored
  header and refuses to let the two drift: an adapter below the engine refuses
  depths the engine serves, above it accepts depths the engine refuses at open.
  The fixture's `mtp2` and `mtp3` oracle entries are reachable now.
- THE LOADER. Model loads drop from about 470 s to about 10 s on the box
  (76.86 GiB of spans), with peak device memory 80 GB. That is the L17 shard-fd
  loader; this repository only pins it.
- THE RATE LINE carries counts now, at both `ds4_cli.c` emit sites. The G1
  driver's no-output parser was written for both shapes and was verified
  against the newly vendored source rather than assumed: the seconds print at
  `%.3f`, which the parser already reads.
- `--prompt-tokens` reached the CLI, so the G1 driver sends the golden's 1024
  token prefix on both legs. The prompt file tokenizes to 1054, and the CLI's
  first argmax on the full instruction is EOS -- so without the flag both legs
  generated nothing and the run compared two empty streams. That is what the
  no-output refusal in section 13 catches; this removes the cause.

## 15. The G1 stream comparison excludes the preamble (2026-09-04)

The first real G1 run (3751b81, `g1-runs/20260904T041612Z-g1`) reported
`stream_mismatch` on a pair whose generated tokens were byte-identical. Both
legs exited 0 with 64 generated from 1024 prompt tokens; the whole
`stream.diff` was one line:

    -qwen4exp memory budget: free 119.01 GiB ...
    +qwen4exp memory budget: free 118.64 GiB ...

The driver hashed and compared stdout WHOLE, and the engine prints its memory
plan and a budget line before the first token. That budget line reads the
machine's free memory, so the two legs could not agree on it and never can.

`g1_strip_preamble` drops everything through the LAST budget line. The raw
captures stay on disk; the stripped streams are written beside them as
`<leg>.stream`, and those are what the comparison, the diff and the sealed
`stream_sha256` use. An engine that prints no budget line is compared whole,
which is the old behaviour and the safe direction.

The no-output text fallback now reads the STRIPPED stream. It could not have
fired otherwise: the preamble is always present, so raw stdout always carries
non-whitespace, and the "a banner does not rescue a leg" case it exists for was
unreachable on a real run. That was latent in section 13's work and only a real
run could show it.
