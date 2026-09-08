# Timed Decode Evaluation Target

The ranked decode measurement uses a private prompt. That prompt is independent
of the public correctness fixture, the hidden teacher-forced fixture, and the
hidden GPQA cases. Its stable contract identifier is
`MLXFastConstants.benchmarkEvaluationTargetID`.

The workflow downloads the prompt only after correctness completes and after it
scrubs the hidden material of that correctness step. The workflow then verifies
the prompt against the operator-managed SHA-256 and byte count. It then passes
the prompt explicitly to the trusted measurement wrapper.

The wrapper generates a checked benchmark oracle from that prompt. It does this
separately for the pinned baseline binary and for the candidate binary. The
oracle supplies:

- the 512-token prefill prompt and its next token;
- the 512-token decode seed and its next token;
- the 128 expected tokens checked during the timed decode loop.

Oracle cache identity must include the binary hash, the evaluation target ID,
and the prompt SHA-256. A prompt rotation must therefore miss the old cache,
even when a binary did not otherwise change.

## Target definition

`lowsim-prose-v1` is an organizer-authored prose target. Measurement selects
it, not content class. The ranked self-hosted runner for that track
re-tokenized and re-validated it for the Laguna XS 2.1 tokenizer
(vocab 100352). This was part of the completed model re-pin.

The public contract is its shape and its gate. The text is original. Its first
512 target-tokenizer tokens form the seed. Its 129-token greedy continuation is
the seed next-token plus the 128 checked decode tokens. That continuation must
pass the self-similarity metric below on the ranked M5 hardware.

The specific text, its subject matter, and its genre are private. This matches
the hidden correctness prompts. Publishing the class would only help
submissions specialize against it.

This target remains a representative text-generation workload: a normal
512-token prefill, then a 129-token greedy continuation. The private operator
RUNBOOK holds the authoring guidance for future rotations. That guidance says
which content shapes keep a greedy continuation diverse instead of letting it
collapse into repetition.

Expectations are not a substitute for measurement. Before a target is
activated, the actual greedy continuation generated on the ranked M5 must pass
the metric below. Discard a candidate that fails. Do not tune around it.

## Self-similarity metric

For every continuation token and each order in `{1, 2, 3}`, take the
immediately preceding `k` tokens. Look for earlier occurrences of that suffix
whose following token was already present in the request context. The analyzer
reports:

- recurrence rate for each order;
- hit rate when the most recent matching occurrence supplies the draft token;
- an optimistic hit rate when any matching occurrence had the actual follower;
- a practical aggregate that chooses the longest recurrent suffix and drafts
  the most recent occurrence's follower.

The aggregate never reads future continuation tokens. It approximates a
longest-suffix prompt-lookup candidate generator. The optimistic fields show
how much ambiguity the deterministic tie-break leaves on the table.

Use this command to score a generated base golden or an assembled benchmark
oracle without loading a model:

```bash
.build/release/mlxfast-swift analyze-ngram-similarity \
  --golden /path/to/generated-timed-golden.json \
  --orders 1,2,3 \
  --max-hit-rate 0.03
```

For a base golden, the command scores the first 129 expected tokens. For an
assembled benchmark oracle, it scores
`expected_decode_seed_token + expected_decode_tokens[0..<128]` against the
512-token decode seed.

The activation threshold is a longest-match, most-recent-follower hit rate of
at most `0.03`. Three percent caps an idealized zero-overhead single-token
reuse benefit near 1.03x. Lookup and target-verification overhead should reduce
the realized gain toward 1.0x. Treat the per-order and optimistic rates as
diagnostics. Reject a target with a conspicuous repeated run, even when the
aggregate narrowly passes.

For comparison, the checked-in longcopy fixture currently scores 75 hits in
129 positions (`0.5813953488372093`) under the same aggregate metric. The
threshold is therefore well over an order of magnitude lower than the
repetitive workload it replaces.

This figure is **tokenizer-dependent**. It is a property of how the fixture's
text tokenizes, not of the text alone, so it moves when the target tokenizer
changes. `0.5813953488372093` is the value measured on this target
(`analyze-ngram-similarity --orders 1,2,3` over
`correctness_prompts/public_longcopy_gate_english_1024_{256,1024}.json`, which
agree) after the goldens were regenerated against the pinned Qwen 3.8
checkpoint on 2026-08-28.

This section's own rule is to re-measure rather than carry any of these numbers
forward to a new target. Earlier vintages of the same metric are recorded here
under that rule: `119/129` (`0.9224806201550387`) under the Gemma 4 tokenizer
for the Gemma-era captures of these same two files, `59/129` (`0.45736`) under
the Qwen 3.6 tokenizer for the retired `*_512*` fixtures, and `37/129`
(`0.2868`) under the serial-era tokenizer before that.

## Correctness separation

A change to the timed prompt must not change:

- `correctness_prompts/public_longcopy_gate_english_1024.txt` or either checked-in
  public golden;
- the hidden teacher-forced correctness object or its SHA/byte pins;
- the hidden GPQA reference object, exact-token checks, semantic judge, or TTFT
  gate;
- `correctnessSteps`, behavior budgets, or QA thresholds.

Only these move when this target rotates: the timed prompt object, the
self-generated timed oracle, the prompt-target metadata, and the baseline
timing calibration.
