# mtp-head/ — the MTP head area

This directory is an editable path. It holds only this `README.md`.

The head weights are not here. This track's MTP head is NATIVE: it is a
separate pinned Q8_0 GGUF (`mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`) that the
organizer stages in the target snapshot, flat beside the target shards.
`tools/serve-up.sh` passes that file to the engine with `--mtp-model`.
`./setup.sh` verifies it against the `{bytes, sha256}` pin in
`fixtures/qwen3_8_125b_a6b_track.json` `target.files`.

No submission carries a head weight. The editable byte budget bars one:
`benchmark.json` `editableSurfaceByteBudget.maxFileBytes` is far below a real
head weight file, so a weight staged here is refused before any measurement.

## The head is used as shipped

The head is the organizer's pinned weights. Do not re-quantize it. Do not
re-cast it, mirror it, or alter it in any other way, on disk or in memory. Do
not replace it, and do not upload head weights of your own. Custom head weights
are not accepted on this track.

Nothing about the head is participant-tunable except the draft depth.

## The declaration

The declaration file is `../mtp-head.manifest.json`. It is editable and
optional.

Its live field is `spec`, which `tools/spec-declaration.sh` reads:

```json
"spec": { "enabled": true, "num_speculative_tokens": 1 }
```

An absent file, an absent `spec` block, `enabled: false`, or
`num_speculative_tokens: 0` all mean serial: the drafter is off. An enabled
depth must be one of the contract's `mtp_head.permitted_draft_depths`, which are
1 to 6.

The other keys are recorded, not read. `source` is `"pinned"`: the only head
that can load is the organizer-staged one, and the value selects nothing.
`max_bytes`, `bytes` and `sha256` are not checked against the staged head;
`./setup.sh` verifies the head bytes against the contract's own pin instead.

A declared `sha256` is not verified against the head bytes.
`docs/participant-contract.md` section 4.3 states that limit plainly.

## Why this README is checked in

It records what this directory is, and it keeps the directory present in a fresh
clone.

## What the head does in a run

A head only PROPOSES tokens. The organizer-pinned target decides every emitted
token. The serial control leg of every ranked pair runs with the drafter off.
