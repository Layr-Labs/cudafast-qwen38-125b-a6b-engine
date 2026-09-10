# Qwen 3.8 125B A6B — participant contract

This document states the terms that bind a submission to track
`qwen3.8-125b-a6b-cuda-v1`.

`benchmark.json` is the Yukon track manifest.
`fixtures/qwen3_8_125b_a6b_track.json` is the track contract fixture. Both files
carry pure configuration: values, paths, commands, and pins. They carry no
prose. This document explains those files. It never overrides them.

## 1. Order of authority

Apply these in order. The higher entry wins.

1. The ranked run on the official runner. It is the authority on any score.
2. `fixtures/qwen3_8_125b_a6b_track.json` and `benchmark.json`.
3. This document.
4. `README.md` and `TASK.md`.

If either configuration file disagrees with this document on a plain value, the
configuration file wins. If either disagrees with the benchmarker about
measurement, the benchmarker wins.

The benchmarker is a prebuilt `benchd` binary resolved from the bench
repository's release channel (the track branch's `dist/`). The channel publishes
`benchd.manifest.json` (`{branch, source_commit, sha256, bytes}`) beside the
binary; `./tools/fetch-benchd.sh` verifies the binary against that manifest,
installs both into `benchd-bin/`, and logs the resolved identity. The harness is
trusted-side: a submission cannot change what measures it. This repository has
no submodules: the engine (`ds4/`) is vendored as plain files and is editable,
and it is not the benchmarker.

## 2. What the track measures

The track measures Qwen 3.8 125B A6B CUDA text-tower inference speed.

You optimize the adapter that drives the ds4 engine and the speculative-decode
arm.

The target model is `unsloth/Qwen3.8-Flash-Next-GGUF`,
variant `UD-Q4_K_XL`: a GGUF conversion of `Qwen/Qwen3.8-Flash-Next`
at revision `f5d08274bafd880402bd16f5e3e6c514136ec06c`. It is a sparse MoE
model.

| Property | Value |
|---|---|
| Architecture | `qwen4_exp`. The text tower is `qwen4_exp_text`. |
| Hidden layers | 48, on a four-layer repeat |
| Full attention | 12 layers, at index `% 4 == 3`: 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47 |
| Linear attention | The other 36 layers. They are gated deltanet and carry a constant-size recurrent state. |
| Full-attention heads | 24 query heads, 2 KV heads, head dimension 256 |
| Rotary | Partial 0.25, `rope_theta` 1e7, interleaved mrope sections [11, 11, 10] |
| QSA indexer | 4 heads, 1 KV head, dimension 128, budget 2048, compress 4 |
| Hyper-connections | `hc_count` 4, `hc_lowrank` 320 |
| MoE | 512 routed experts, 10 per token, `moe_intermediate` 640, plus a shared expert of width 640 behind a shared expert gate |
| n-gram / PLE | Layer index 1. `ngram_size` 3, 8 heads per n-gram, 128 split parts: 384 shard tensors plus 3 int64 buffers. The table is offloaded to SSD behind a bounded LRU. |
| Hidden size | 2560 |
| Vocabulary | 248320, embeddings untied |
| Tokens | eos 248046 and 248044; bos and pad 248044 |
| Quantization | Affine, group size 32, 4 bits. Router gates and multimodal weights are BF16. |
| Final norm | There is no `model.norm` tensor. The final `hyper_connection_mixer` stands in for it. |
| Raw tensors | 3747 across 22 shards. Index `total_size` is 113,209,155,128 bytes. |
| Text tower | 3414 tensors. The vision tower is 333 tensors and the loader skips them. |

There is no sliding-window attention on this model.

The speculative-decode arm is the native MTP head. It is a separate pinned
Q8_0 GGUF (`mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`) that the organizer stages
in the target snapshot beside the shards. `tools/serve-up.sh` passes it to the
engine with `--mtp-model`. It carries 1 hidden layer of hybrid full attention.
It has no embedding and no `lm_head` of its own: it rides the target's.

`kv_backend` is pinned `contiguous` on both legs. The benchmarker refuses when
it cannot honour the pinned backend. It does not degrade to another backend.

## 3. What you may edit

`benchmark.json` `editablePaths` is the authority. It lists four entries:

| Path | What it is |
|---|---|
| `ds4/` | The vendored ds4 engine (see section 3.5). |
| `harness/` | The Engine Protocol v1 adapter (`harness/protocol-adapter`), the scored binary. |
| `mtp-head/` | The MTP head area (see section 4). It holds only its `README.md`. |
| `mtp-head.manifest.json` | The MTP head declaration (see section 4). |

The rule behind the list: anything that only **proposes** tokens or computes
the forward pass is editable. Anything that **verifies**, **measures**, or
**ledgers** stays trusted — the benchmarker (benchd), the gates, the track
contract (`fixtures/`), and this manifest.

The scored engine is the Engine Protocol v1 adapter with the ds4 C/CUDA engine
linked in-process (`harness/protocol-adapter`, `ds4/`). The `ds4/` tree is
`Layr-Labs/ds4`, our port of `antirez/ds4`, vendored as plain files. It is
editable: you submit the engine you edited. `ds4/VENDOR.json` records the
signed base the tree was exported from.

### 3.1 Optional paths

`optionalEditablePaths` lists `mtp-head.manifest.json`.

A submission archive has REPLACE semantics over `editablePaths`. An absent head
declaration means the organizer-pinned head. The overlay therefore skips a missing
optional path instead of failing closed.
Yukon's overlay reads this list from the trusted contract, never from the
submission.

### 3.2 The byte budget

`editableSurfaceByteBudget` caps the editable surface.

| Key | Value |
|---|---|
| `maxTotalBytes` | 19550883 |
| `maxFileBytes` | 4473321 |
| `maxGrowthBytes` | 19550883 |
| `exemptPathMaxBytes` | 512000000 |
| `exemptPathMaxFileBytes` | 100000000 |

Every editable path is enforced. Nothing is exempt.

`exemptPaths` is **absent** since 2026-08-26. The exemption existed for one
reason: to let head weights ride in a submission outside the source budget. A
submission carries no head weights any more, so there is nothing to exempt.

The two exempt caps stay declared. They cannot bind while `exemptPaths` is
absent. They stay because both enforcers carry the same two numbers as
compiled-in fallbacks, and this manifest is what holds those constants to a
reviewed value. `tools/lint-benchmark-manifest.py` check 3b enforces that
equality.

No head weight file is staged in an editable path, so this budget never meets
one. The head sits in the organizer-staged target snapshot, which is outside the
editable surface.

### 3.3 What you may not edit

You may not edit anything that verifies, measures, or ledgers. This covers the
trusted benchmarker (benchd), the target weights, the transform contract, the
tokenizer, the goldens, the gates, and the timing and telemetry code.
`fixtures/` is outside the editable surface. The scoring step reads the contract
from the trusted checkout for that reason.

### 3.4 The weights are frozen

The target model's quantization is frozen as shipped. So is the MTP head's.

A submission must not re-quantize any weight. It must not re-represent one. It
must not change the numerical format of one. It must not mirror one. This holds
on disk and in memory, and it holds even when the result passes every
correctness gate.

No editable path licenses a change of weight format. A lossier target
substitutes a degraded model. It does not optimize the accepted one.

The MTP head is no exception. It is the organizer's pinned weights, and the
engine uses it exactly as staged. You may **not** re-quantize it. You may
**not** replace it. You may **not** upload head weights of your own. Custom
head weights are not accepted on this track.

The head is the organizer's pinned weights, staged with the target snapshot.
`fixtures/qwen3_8_125b_a6b_track.json` names the repository and the variant, and
its `target.files` list pins every shard and the MTP draft head by bytes and
sha256; `./setup.sh` verifies the staged snapshot against that list. The
contract fixture lives in `fixtures/`, which is outside the editable surface.

Two things enforce this, and section 4 states each one:

1. No editable path can hold head weights. The byte budget bars a weight file,
   and a submission that carries one is refused.
2. The head declaration selects nothing. `source` is `"pinned"`, and the
   only head that can load is the organizer-staged one.

Nothing about the head is participant-tunable except the draft depth. Section
4.1 states how you declare it.

The reason is the propose-and-decide split. The head only proposes tokens. The
pinned target model decides every emitted token.

### 3.5 Editing the engine

**What `engine_pin` means (ruled 2026-09-04).** The fixture's
`serve_configuration.engine_pin` and `target.engine_pin` are **provenance**: the
engine base the calibrated baseline and the timed oracles were *authored on*.
They are **not** a claim that the engine you run is byte-identical to that
commit, and nothing checks that — an equality check would refuse every
submission. What gates your engine is the **correctness goldens** and the
**calibrated baseline**: change the token stream and you fail, wherever the
change came from. `tools/ranked-box-preflight.sh` checks only that the tree was
vendored *from* that base.

The engine is VENDORED at `ds4/` -- plain files, exported from a signed commit
of `Layr-Labs/ds4`, our port of `antirez/ds4`. `ds4/VENDOR.json` records that
base and the fixture's `target.engine_pin` names it. `tools/ds4/build.sh` copies
the tree to `.build/ds4/src` and builds the engine's own `cuda-spark` target
from the copy, so YOUR edit is what gets built. The engine IS
participant-editable; `editablePaths` lists it. `tools/ds4/vendor-sync.sh`
re-vendors from a newer signed tag, which is how the organizer advances the
base. CI syntax-checks the engine and the shim on every
pull request (`tools/ds4/build.sh --cpu-check`).

The adapter reaches the engine through
`harness/protocol-adapter/ds4_shim/ds4_shim.h`, a flat C surface over the
engine's session API (`ds4/ds4.h`: `ds4_session_sync`, `ds4_session_eval`,
`ds4_session_argmax`, `ds4_session_top_logprobs`,
`ds4_session_eval_speculative_argmax`). The pinned port carries the `qwen4exp`
model family, so the shim opens the target with no refusal.

**Draft depth.** The pinned engine drafts **up to six** tokens per
target-verified cycle (`ds4_session_eval_speculative_argmax` in `ds4.c`, which
the port routes to `ds4_qwen4exp_mtp_cycle`). Depths **1 to 6 all run**: that is
the whole track envelope (`mtp_head.permitted_draft_depths` in the contract
fixture). The depth that ran is echoed as `effective_spec` rather than clamped.
A depth outside 1 to 6 is refused by name. In
`harness/protocol-adapter/src/ds4_backend.rs`, `MTP_MIN_DEPTH` is 1,
`MTP_MAX_DEPTH` is 6, and `DS4_IMPLEMENTED_DEPTH` is 6. The last one must equal
the vendored engine's `DS4_QWEN4EXP_IMPLEMENTED_DEPTH`.

Depth 1 was the only implemented depth until the `e2f86b7` vendor-sync. Which
depth is *fastest* is yours to find: a deeper draft proposes more per cycle and
accepts less often.

## 4. The MTP head

The track carries one speculative head. It is the organizer's weights.

| Item | Value |
|---|---|
| Declaration | `mtp-head.manifest.json` |
| Where the weights are | A separate pinned Q8_0 GGUF in the target snapshot, flat beside the shards |
| File | `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` |
| Organizer pin | `fixtures/qwen3_8_125b_a6b_track.json` `target.files`, by bytes and sha256 |
| How the engine gets it | `tools/serve-up.sh` passes it with `--mtp-model` |

The declaration file is editable. The head is not.

Nothing in a submission stages a head weight file. There is no head stager and
no head weights directory. `./setup.sh` verifies the target snapshot, and the
head arrives with it.

### 4.1 What you may declare

`mtp-head.manifest.json` is editable and optional. Its live field is `spec`,
which `tools/spec-declaration.sh` reads:

```json
"spec": { "enabled": true, "num_speculative_tokens": 1 }
```

`spec.enabled` is a boolean. `spec.num_speculative_tokens` is an integer. An
enabled depth must be one of the contract's `mtp_head.permitted_draft_depths`,
which are 1 to 6. An absent file, an absent `spec` block, `enabled: false`, or
`num_speculative_tokens: 0` all mean serial: the drafter is off. An unknown key
inside `spec` is refused by name, so a mistyped key never reads as its default.

`source` must be `"pinned"`, and an unknown top-level key is refused by name;
`tools/spec-declaration.sh` checks both. The only head that can load is the
organizer-staged one, so the value selects nothing. `max_bytes`, `bytes` and
`sha256` are recorded, not read; `./setup.sh` verifies the head bytes against
the contract's own pin instead.

The file carries no `arm` key. There is one arm, so there is nothing to select.

A declaration that is present but broken is a refusal. The runner never falls
back silently.

### 4.2 What you may not do

You may not ship head weights. The editable byte budget bars them: a real head
weight file far exceeds `maxFileBytes` (4473321), so
`.github/scripts/submission-static-review-checks.sh` refuses it before any
measurement. Yukon overlays only the declared editable paths from the trusted
contract, and benchd refuses a candidate that diverges outside them.
`mtp-head/` is an editable path and it holds only its `README.md`, but the byte
budget still bars a weight file there.

You may not re-quantize the head. You may not re-cast it, mirror it, or alter it
in any other way, on disk or in memory. The engine loads the staged bytes and
uses them as they are.

You may not edit the target snapshot. It is not an editable path, so any change
to it is outside the surface.

### 4.3 What the size cap does and does not do

`max_bytes` is recorded, not read. Nothing bounds a load by it.

A declared `sha256` is optional, and the runner does not verify it against the
head bytes. It treats a wrong digest and an absent digest alike. That is stated
here plainly because it is a real limit, not a detail: a declared digest is a
statement of intent, not a check.

The head bytes are bound one level up. They are part of the organizer-staged
target snapshot, and `./setup.sh` verifies every staged file against the
`{bytes, sha256}` pins in `target.files`. Both legs load the head out of that
one verified snapshot.

### 4.4 What the head does in a run

The head only **proposes** tokens. The organizer-pinned target model decides
every emitted token.

The serial control leg always runs with the drafter off
(`--spec serial --draft-len 0`). The candidate leg runs at the depth the
declaration names. Section 3.5 states the envelope, and section 7 states what
each run seals.

## 5. Scoring

### 5.1 The formula

```text
composite = prefill_gain ^ 0.25 * decode_gain ^ 0.75
```

Each component is a gain:

```text
gain = baseline_aggregate / candidate_aggregate
```

The score is serial-anchored. A faster candidate scores above 1.

**THE SCORED SHAPE IS SINGLE-STREAM.** David ruling 2026-08-27, relayed by
orchestrator: this track scores a PAIRED serial-against-MTP comparison, ONE
stream at a time, at `scored_batch_size` 1. The batched cohort adaptation is NOT
pursued.

`aggregate` is a **sum over the pairs**, per role. Each gain is therefore a
RATIO OF SUMS, not a mean of per-pair ratios. The scored ranked run times the
ONE prompt `live_golden` names, on every leg.

**THE BASELINE LEG IS MEASURED, NOT STORED.** David ruling 2026-09-08. A ranked
run measures PAIRS OF LEGS. It measures them on the SAME box, in the SAME job,
over the ONE prompt the fixture names in `live_golden`. The fixture's
`official_pairs` sets the count, and it is 2 (David ruling 2026-09-09). Every
pair is the same two legs in the same order:

| Leg | What runs | Speculation |
|---|---|---|
| Serial control | the organizer-staged reference tree | off, always |
| Candidate | your submission | your declared draft depth |

The reference tree is the engine commit that the fixture pins as
`baseline_reference_commit`. The organizer stages it on each ranked box and
builds it there. Every ranked box runs the same reference commit, so one board
compares one control.

```text
composite = (ref_prefill_spt / cand_prefill_spt) ^ 0.25
          * (ref_decode_spt  / cand_decode_spt)  ^ 0.75
```

The legs run STRICTLY ONE AFTER THE OTHER, and each leg loads the model once.
Per role the per-token times are SUMMED over the pairs, and each gain is the
ratio of those two sums. The floors, the ceiling and the acceptance bands apply
to that aggregate, not to one pair. Every control leg is checked against this
box's own baseline calibration (section 5.5.1).

**BOTH SPEEDUP FLOORS ARE 0.95** (David ruling 2026-09-09). A candidate that
regresses prefill or decode by more than 5 percent is refused. The fixture
declares them as `decode_speedup_floor` and `prefill_speedup_floor`, and the
benchmarker enforces the fixture's values. The ceiling stays 5.0.

NO FILE HOLDS A BASELINE PAIR. The goldens hold none, the fixture holds none,
and the benchmarker holds none. A golden that carries
`benchmark.baseline_prefill_seconds_per_token` or
`benchmark.baseline_decode_seconds_per_token` is REFUSED on the ranked path, and
`tools/lint-benchmark-manifest.py` refuses one in this repository.

A stored number describes the machine that produced it, at the temperature and
on the engine of that day. The control leg describes YOUR run: same box, same
hour, same thermal state.

The ruling this track is scored under, verbatim, dated 2026-08-27, and carried
in the fixture's `scoring_semantics.ruling_verbatim`: "score the qwen 3.8
125b-a6b tracks (mlx and cuda) single-stream, paired serial vs the built-in
mtp, on prefill gains ^ .25 * decode ^ .75".

`scoring.mode` is `qwen-native-mtp-paired-decode-only`, which is the
benchmarker's own single-stream paired regime name (`benchd`
`overlay::SCORING_MODE`). It names the measurement methodology, not the
formula.

### 5.1.1 Where the prefill window is, and why you cannot move work out of it

**THE VERBS DO NOT CHANGE.** The single-stream pair is still
`free_decode_begin` followed by `free_decode_run`. There is no new message and
no new field. The benchmarker splits its OWN parent clock at the verb boundary:

| Window | From | To |
|---|---|---|
| prefill | `free_decode_begin` sent | the validated `seed_token` comes back |
| decode | there | `free_decode_run(N)` returns |

`elapsed = prefill + decode`, and `seconds_per_token` is unchanged. Both legs
are bracketed identically.

**WHAT THE ENGINE OWES, and it is not optional:**

1. `free_decode_begin` runs the FULL seed prefill -- the golden's
   `decode_seed_tokens`, all 1024 of them, with the requested spec resolved --
   and replies only after that work has COMPLETED. The reply is the seed token
   (the greedy argmax after the whole seed) and the echoed `effective_spec`.
2. `free_decode_run` does NOT prefill and does not re-run any part of the seed.
   It decodes from the state `begin` left.
3. Nothing prefills before `free_decode_begin` arrives.
4. The hello advertises `free_run_decode` only.
5. Units are unchanged.

**MOVING PREFILL WORK INTO THE RUN IS NOT AN OPTIMISATION. IT IS A SCORING
DEFECT.** Deferred seed work does not disappear: it leaves the prefill window
and lands in the decode window. The whole window is unchanged, so `elapsed` and
`seconds_per_token` look identical -- but the composite weights the two windows
0.25 and 0.75, so shrinking prefill and growing decode by the same amount MOVES
THE COMPOSITE, and it moves it against you. The same applies in reverse to a
leg that did decode work early.

This is engine-side and unobservable on the wire, so it is held by tests rather
than by the protocol: a prefill-window test counts the tokens each verb pushes
through the target and checks the
full-attention cache offsets at the verb boundary, on the serial leg AND the
mtp leg. The mtp leg is the sharper case: it feeds tokens it may take back, so
its forward count legitimately exceeds N, and what must hold is that its
offsets land exactly on `seed + N` -- rollback took back the drafts and nothing
else, and never re-prefilled.

### 5.2 The measured window

| Quantity | Value |
|---|---|
| Seed tokens per stream | 1024 |
| Checked decode steps | 128 |
| Golden shape | 1024 `prompt_tokens` and 129 `expected_tokens` |
| Streams per window | 1 |
| Timed prompts per leg | 1 (the fixture's `live_golden`) |
| Prompts in the pinned correctness pool | 8 |
| Prefill tokens per correctness-pool pass | 8 x 1024 |
| Pairs per ranked job | 2 (the fixture's `official_pairs`) |
| Legs per ranked job | 4 (each pair is serial control, then candidate) |

The correctness-pool rows are not the scored timing. The box stages all 8 pinned
prompts and `tools/ranked-box-preflight.sh` verifies all 8 against the fixture
pins. Each timed leg runs the ONE prompt `live_golden` names, and section 5.1
states what that timing produces.

The legs run one after the other, in one job. Each leg loads the weights once,
and each leg gets its own worker residency.

`MLXFastConstants.correctnessPromptTokens`, `benchmarkPrefillPromptTokens`, and
`benchmarkDecodeSeedTokens` all equal 1024. `benchmarkDecodeSteps` is 128.

Every timed leg runs on a cool, quiescent box. The ranked job waits for the
machine to go idle before the clock starts, and the benchmarker holds each timed
phase behind the fixed 50 C cool-down gate (the CUDA/GB10 value of the
per-platform `platform-cool-gate`; 40 C on Mac). The job refuses to measure at all
when the box has no GPU temperature reader, or when that reader returns a frozen
or implausible value. Only pairs accepted under that gate feed the composite.

### 5.3 The parameters

| Parameter | Value |
|---|---|
| `scoredBatchSize` | 1 |
| `prefillGainExponent` | 0.25 |
| `decodeGainExponent` | 0.75 |
| `pairsPerCohort` | 2 |
| `minPairsPerCohort` | 2 |
| `decodeSpeedupFloor` | 0.95 |
| `prefillSpeedupFloor` | 0.95 |
| `decodeSpeedupCeiling` | 5.0 |
| `kvBackend` | `contiguous` |

The scored run times ONE prompt at a time. There is no sweep and no per-run
choice of width. A width the benchmarker has not certified has no series tag,
and the benchmarker refuses that width rather than run it.

There is NO MEDIAN on this track. Each role's per-token times are summed over
the 2 pairs, and each gain is the ratio of those sums.

### 5.4 Token fidelity

The benchmarker applies a per-stream token-tolerance gate with a **10%
budget**.

This track does not require token-for-token equality with the serial
trajectory. The block-shaped forward pass diverges from the serial forward pass
at near-tie argmaxes. The gate prices that divergence against the 10% budget.
The gate accepts similar output. It does not certify lossless output.

### 5.5 Arming

`fixtures/qwen3_8_125b_a6b_track.json` sets `official_scoring_enabled`. That
flag is the SINGLE authority on this track's arm state, and it is load-bearing.
The pinned benchmarker reads it from the `--contract` fixture. It refuses to
seal an official scoring artifact while the flag is `false`, and it refuses
while the flag is absent, because it treats an absent flag as unarmed rather
than armed. The flag is `true`: the track is ARMED. What remains before a ranked
dispatch is box work, not repository state -- the reference workspace staged and
built, and each ranked box calibrated.

The benchmarker, not the engine, produces the composite. It computes it from
benchd's own parent-clocked prefill and decode windows, summed over the pairs,
at the certified exponent pair. No engine-reported value feeds it, and it does
not depend on per-stream instrumentation. Each record seals exactly one of
`composite` and `composite_absent_reason`. A composite is absent only when the
record accepted no pair, or when a window is degenerate, and the reason names
which.

Refuse, not degrade, stays the standing posture for this track.
`tools/qwen38-125b-a6b-measure-and-score.sh` refuses with a non-zero exit rather
than emit a score when a prerequisite is missing. It does not substitute the
shared-window `raw_ratio_of_means` diagnostic for the ruled composite formula.
The `kv_backend` check and the byte-budget check use it too.

The timed prompt pool is ARMED. All 8 `timed_prompt_pool[]` entries pin a real
golden by `{r2_path, sha256, bytes}`. `live_golden` names the one live scored
prompt, `botany`; the other 7 stay in the pool for rotation.
`hidden_correctness_golden` pins the token-fidelity oracle to the same live
golden. `tools/ranked-box-preflight.sh` verifies every staged golden against its
pin -- byte count first, then sha256 -- and refuses on any mismatch. NO golden
carries a baseline pair: the control leg supplies the denominator, and section
5.1 says why.

**The launch reference candidate is stock SERIAL.** No speculative config,
`num_speculative_tokens` 0. The CANDIDATE leg's serve config FOLLOWS THE
SUBMISSION's declared spec (David MTP-0 ruling). The SERIAL CONTROL leg is
serial whatever the candidate declares. The benchmarker boots each leg's
resident itself, from that leg's own workspace, with
`tools/serve-up.sh --boot --spec serial|mtp --draft-len N --socket-out FILE`,
and ends it with `tools/serve-up.sh --stop --socket PATH`. Leg 1 is always
`--spec serial --draft-len 0`, the flag is the authority, and a disagreeing
`SERVE_UP_SPECULATIVE` in the environment is refused rather than honoured. A
box-preset `SERVE_UP_SPECULATIVE` is refused earlier still, because it would
reach both legs. The native MTP head is the participants' improvement path: a
submission that declares MTP is a separate, later entry.

Because the launch reference is serial, and the control leg is serial on the
same box, the expected launch composite is approximately 1.000, within the
acceptance bands (prefill +/-5 %, decode +2 % up). That run is a NULL CONTROL of
the scored pipeline: a composite outside approximately 1 +/- band is a finding,
not a pass.

### 5.5.1 The per-box calibration

Each ranked box carries a calibration file. `MLXFAST_BASELINE_CALIBRATION` names
it. The file records what the control leg has measured on THAT box before.

```json
{
  "version": 1,
  "track_id": "qwen3.8-125b-a6b-cuda-v1",
  "box": "<the runner name>",
  "reference_commit": "<the fixture's baseline_reference_commit>",
  "prompt": "botany",
  "passes": 4,
  "prefill_seconds_per_token_mean": 0.0006282488193359375,
  "decode_seconds_per_token_mean": 0.0329116748046875,
  "prefill_cv": 0.004,
  "decode_cv": 0.002,
  "prefill_band_low": 0.95,
  "prefill_band_high": 1.05,
  "decode_band_low": 0.98,
  "decode_band_high": 1.02,
  "captured_at": "2026-09-08T00:00:00Z",
  "benchd_source_commit": "<40 hex>"
}
```

The file is a HEALTH BAND. The benchmarker reads it to answer one question: did
the control leg land where this box lands? The measured control leg must satisfy
`mean * band_low <= spt <= mean * band_high` on both axes. Outside the band the
run dies by name and seals no score.

> **WARNING — the calibration is never a denominator.**
> The benchmarker divides by the control leg it just measured. It never divides
> by this file.

The organizer writes the file with `tools/calibrate-box.sh <box> <out>`, on the
box it describes. The tool takes the GPU lock and hands the window to the
benchmarker.

The benchmarker boots the reference tree's serial resident ONCE PER PASS. It
boots the resident, measures one pass against it, stops it, and only then boots
the next pass's resident. It does not share one resident across the four passes.
A scored run boots its control leg once and measures it once, so a pass measured
on an already-warmed resident would not describe the leg the band certifies.

It refuses when the spread on either axis is above 1 %: a box that unstable has
no band.

`tools/ranked-box-preflight.sh` section 8 refuses a calibration file that names
another track, another box, another reference commit or another prompt, that is
older than the reference commit, that is dated in the future, or whose band
cannot fail.

DISPLAY CONVENTION (David standing rule). The calibration file and benchd's
capture are stored in seconds per token -- an internal representation only.
Every human-facing number shows tokens per second (tok/s = 1 / seconds per
token). A decode mean of 0.06451959972265625 s/tok is shown as 15.50 tok/s.

### 5.6 Which goldens you can hold

| Object | Where it lives | Can you have it? |
|---|---|---|
| `correctness_prompts/public_longcopy_gate_english_1024_256.json` and `..._1024_1024.json` | Checked into git | **Yes.** They are Gemma-era captures kept for their 1024-token prompts; they do not load against the Qwen target. |
| `correctness_prompts/public-longcopy-gate-english-1024.golden.json` | Checked into git | **Yes.** A public Qwen capture (`unsloth/Qwen3.8-Flash-Next-GGUF` @ `38bb39ee97821de2c9009abb7e93950eec396e66`) for local runs. |
| `timed_prompt_pool[]`, 8 tapes | R2, at the `r2_path` keys the fixture pins. The ranked box stages them out of band into `MLXFAST_QWEN38_GOLDEN_DIR`. | **No.** They are organizer material and they are never in git. |
| `live_golden_speculative{}`, 6 per-depth oracles | The same: R2 keys, staged on the box. | **No.** Same material, same handling. |
| `hidden_correctness_golden` | The live golden, pinned by digest only. It is one of the staged files. | **No.** It is the token-fidelity oracle and it stays on the box. |

`tools/fetch-goldens.sh` is the organizer-side, pin-verified fetcher for R2
objects. It reads the R2 base from the environment variable
`R2_BUCKET_ENDPOINT` only. That value is secret-tier and is absent from this
repository. The script verifies the byte count first, then the sha256, and
deletes the file on either mismatch. It refuses to fetch anything the contract
declares hidden, and that guard fails closed when it cannot read the contract.

> **NOTE — this repository pins no public golden for that tool to fetch.**
> A participant has nothing to fetch with it today. Whether to publish a public
> local-calibration golden is an organizer decision.

The organizer stages the whole pinned set on a ranked box with the same tool.
`--all` reads the fixture, fetches every tape and every per-depth oracle, and
verifies each one against its `{sha256, bytes}` pin. It signs the requests with
the signer vendored at `tools/download-r2-object.sh`, so it needs R2 credentials
and refuses without them. A file that already matches its pin is left alone, so
the command is safe to re-run:

```bash
R2_BUCKET_ENDPOINT=... R2_ACCESS_KEY_ID=... R2_SECRET_ACCESS_KEY=... \
  tools/fetch-goldens.sh --all --out "$MLXFAST_QWEN38_GOLDEN_DIR"
tools/ranked-box-preflight.sh
```

The preflight then verifies the staged directory against the fixture again and
refuses an extra `*.json` in it.

## 6. Running the benchmark

`benchmarkCommand` targets `benchd iterate --mode official` through
`tools/qwen38-125b-a6b-measure-and-score.sh`. That script is trusted-side tooling. It is
not an editable path, so a submission cannot rewrite the measurement pipeline
from inside its own archive.

The wrapped invocation is:

```text
benchd iterate --mode official \
  --contract fixtures/qwen3_8_125b_a6b_track.json \
  --engine .build/release/mlxfast-runtime-worker \
  --weights $MLXFAST_TARGET_SNAPSHOT_DIR \
  --golden $MLXFAST_QWEN38_GOLDEN_DIR/<live_golden>.golden.json \
  --golden-sha256 <pin> --golden-bytes <pin> \
  --baseline-workspace $MLXFAST_BASELINE_WORKSPACE \
  --baseline-calibration $MLXFAST_BASELINE_CALIBRATION \
  [--mtp-depth <your declared depth>] \
  --score-path score.json
```

The benchmarker seals `score.json` itself. The script does no conversion.

The two ranked environment variables are:

| Variable | Meaning |
|---|---|
| `MLXFAST_BASELINE_WORKSPACE` | the built reference tree the serial control leg runs on, at the fixture's `baseline_reference_commit` |
| `MLXFAST_BASELINE_CALIBRATION` | this box's calibration file, the health band for that leg |

Both are required on the ranked path. The script refuses by name when either is
absent, and `tools/ranked-box-preflight.sh` refuses before that.

`RUNNER_NAME` names the box the band is checked against, and
`MLXFAST_BENCHD_SOURCE_COMMIT` names the benchmarker that captured a band. The
benchmarker reads both from the environment when the matching flag is absent.

The benchmarker boots each leg's resident itself and injects that leg's socket
into that leg's workers. Nothing else in the job may export
`DS4_RESIDENT_SOCKET` or `BENCH_WORKER_RESIDENT_SOCKET`: an inherited socket is
a resident neither leg booted, and it is refused by name.

The organizer stages the reference tree with
`tools/stage-baseline-workspace.sh <dir>`. That tool clones this repository at
the pinned commit from a local bundle or mirror, builds it, and verifies HEAD. It
holds no credential and contacts no host. It refuses an existing directory: a
staged reference tree is never reset in place, because that changes what every
scored run on the box is divided by.

**WHERE THE COMPOSITE LIVES ON THIS TRACK.** A single-stream run has no cohort
record, so benchd seals the composite ONE LEVEL UP: `results.json` carries
`composite` (`{composite_score, composite_speedup_floor,
composite_speedup_floor_met, decode_gain, prefill_gain}`) beside
`composite_scored_exponents`, and exactly one of `composite` /
`composite_absent_reason` is present. The published score is
`composite.composite_score`. benchd's own overlay publishes the same number
with the aggregation discriminator
`shared_window_composite_prefill_decode_gain` and a `single_stream_composite`
block carrying the two gains and the exponent pair.

The decode-only median (`aggregate.raw_decode_speedup_median`) is NOT the score
on this track. It is a different formula -- decode only, no prefill component,
no exponents -- and the emitter refuses a single-stream record that seals no
composite rather than publishing the median in its place. The median is
forwarded in `metrics` as a diagnostic.

Two drift tripwires run at that seam, because benchd reports a floor and an
exponent pair but wires neither to an exit code: the emitter REFUSES a run
whose `composite_speedup_floor_met` is false, and refuses a run whose sealed
`composite_scored_exponents` differ from `benchmark.json`
`scoring.scoredExponents`.

`preSubmitCommand` runs `./tools/qwen38-125b-a6b-measure-and-score.sh --preflight-only`.
That runs the arm gate and the golden integrity pin, and exits without
measuring. It does not need the two ranked variables: it is the pre-submit
check, not the ranked path.

The ranked pipeline is `.github/workflows/benchmark.yml`, which `benchmark.json`
`runner.workflow` names. It triggers on `workflow_dispatch` only. Its hosted
surface-check job gates its ranked job, which runs on the self-hosted labels
`[self-hosted, Linux, ARM64, qwen3.8-125b-a6b-cuda-v1]` — the last label is the
track id, and Linux + ARM64 name the GB10 box.
The ranked job holds no credential. The organizer stages the hidden timed-pool
tapes, the reference workspace and this box's calibration file onto the box. Before `./setup.sh` runs, `tools/ranked-box-preflight.sh`
verifies each tape against this track's `{sha256, bytes}` pins. One ranked run
occupies
the box at a time. A second dispatch queues rather than cancelling the first.

`setupCommand` is `./tools/fetch-benchd.sh && ./setup.sh`. It chains no head
stager, because there is none. The checked-in `mtp-head.manifest.json` declares
`"source": "pinned"`, and the head arrives beside the target shards in the
snapshot that `./setup.sh` verifies.

## 7. The pinned artifacts

| Artifact | Identity |
|---|---|
| Target model | `unsloth/Qwen3.8-Flash-Next-GGUF`, variant `UD-Q4_K_XL` |
| Target manifest | `fixtures/qwen3_8_125b_a6b_track.json` `target.files` (bytes + sha256 per file) |
| Engine | vendored at `ds4/` from `Layr-Labs/ds4` @ `278b799b974cb580e0f96ed3dce68bfbe0d8675b` (`ds4/VENDOR.json`) |
| MTP head | `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`, a pinned Q8_0 GGUF in the target snapshot |

The staged snapshot holds the four main GGUF shards (111,334,654,784 bytes)
and, flat beside them, the Q8_0 MTP draft head (2,786,568,256 bytes). The
fixture pins all five by bytes and sha256.

The model repository is public and downloads without a token.

Participants never supply the target weights. Substituting or re-deriving the
target is a failure.

Batch size is locked at 1. The track is scored single-stream.

You can select the draft depth. It is not pinned at 1.

You declare the depth in `mtp-head.manifest.json` under
`spec.num_speculative_tokens`. That file is an editable path, so the depth is a
free lever.

The permitted values are 1 to 6 (`mtp_head.permitted_draft_depths` in the track
fixture). The envelope is trusted code and is not an editable path. A depth
outside 1 to 6 is refused by name, never clamped, so the serve that boots and
the `effective_spec` the adapter seals can never disagree.

An absent depth does not mean 1. Two layers supply a depth when a request does
not name one. Do not confuse them.

| Request | Result |
|---|---|
| benchd invocation gives no `--mtp-depth` | benchd measures at depth 2 |
| the `mtp` block has no `depth` key | the envelope uses its ceiling of 6 |

Every run seals the depth that operated. Read `effective_spec` for the depth the
run declared. Read `effective_mean_draft_len` for the draft length that the run
realized. The two can differ: a run can declare depth 2 and realize a mean draft
length near 1.

## 8. Prohibited techniques

A submission that uses any of these fails the static review.

- A cache or memo keyed on a request's input tokens whose only possible hit is
  the harness repeating one identical computation. Bit-identical output does
  not make it legitimate. The benchmark measures single-pass inference. An
  optimization must save work that recurs in single-pass production inference.
- Hardcoded hidden prompts, hidden token identifiers, or answers.
- Timing shortcuts, protocol injection, network access, and filesystem
  exfiltration.
- Any change outside `editablePaths`.

Input-independent caching stays legal. This covers weights, dequantized
tensors, and RoPE or mask tables keyed on shapes and offsets. Within-request KV
reuse stays legal.

Keep every change prompt-independent and model-general. The hidden prompts
differ from the public fixtures.

## 9. Submitting

Use the Yukon CLI for every account operation and every submission operation.
`README.md` holds the commands.

A submission archive packages only `editablePaths`. It rejects generated
artifacts, symlinks, local scores, reference checkpoints, and any source change
outside the editable surface. `yukon submit` does not run a local test first,
and no local run blocks the upload.

The ranked run on the official runner is the gate that ranks a submission.

## 10. License

The pinned checkpoint is a GGUF conversion of `Qwen/Qwen3.8-Flash-Next`. The
model's own license terms apply to it. They ship with the checkpoint at its
pinned revision.

This repository distributes no model weights.
