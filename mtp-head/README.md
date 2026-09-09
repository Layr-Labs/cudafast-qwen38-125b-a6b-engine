# mtp-head/ — the organizer's staged MTP head weights

This directory holds the organizer-pinned MTP head. It is **not** an editable
path. A submission carries nothing here.

This track's MTP head is NATIVE: it is embedded in the pinned target checkpoint
under `language_model.mtp.*` (`docs/qwen38-125b-a6b-port-notes.md` section 6),
so there is no separate head download. The seed's staging script
(`setup-gemma4-assistant.sh`) and the `fixtures/gemma4_assistant.sha256` manifest
were the removed Apple-Metal runtime's path; this directory and that manifest
remain as seed residue slated for removal with the engine port.

## Custom head weights are not accepted

This is the David ruling of 2026-08-26. It replaces the earlier
bring-your-own-head design.

You may re-quantize this head. You may not replace it, and you may not upload
head weights of your own.

Three things enforce that:

1. `mtp-head` is not in `benchmark.json` `editablePaths`. A submission that
   carries a file here is refused by
   `.github/scripts/enforce-modifiable-surface.sh`, which names the file and
   says it is outside the modifiable surface. The overlay never copies it, and
   the benchmarker's write-divergence gate refuses it as content that diverges
   from the trusted baseline outside the editable surface.
2. `../mtp-head.manifest.json` accepts `"source": "pinned"` only.
   `"source": "in_branch"` and `"source": "remote"` are refused by name.
3. A re-quantization happens on load, in memory, on the benchmark machine.
   No re-quantized file is made, so there is no artifact to travel in a
   submission.

## How to declare a re-quantization

`../mtp-head.manifest.json` stays editable. It is the declaration surface.
Keep `"source": "pinned"`. State the `bytes` you expect the staged artifact to
have, if you want the record. A declaration may lower `max_bytes` and may not
raise it above the 2 GiB track cap.

The quantization recipe itself is read from the head's own `config.json` by the
loader. A `quantization` block there selects which modules load quantized and at
what geometry: `group_size` (positive, at most 65536), `bits` between 2 and 8,
and optional per-layer overrides. See `docs/participant-contract.md` sections
3.4 and 4.

## How to produce a re-quantization

A re-quantization happens ON LOAD, in memory. Nothing in this directory
changes.

The seed's re-quantization-on-load seam lived in the vendored MLX model fork,
which has been removed. The participant seam that tunes the head on the CUDA
engine is deferred with the engine surface to a David/organizer ruling.

Do not write into this directory. The ranked worker runs under a sandbox that
denies file writes, and the benchmarker refuses a head tree that changed.

The target model is not in scope. See `docs/participant-contract.md` section
4.4.

> **NOTE — the MTP loader does not check declare-versus-carry.**
> It neither refuses a head that declares a quantization its tensors do not
> carry, nor one that carries packed tensors it does not declare. An absent
> declaration skips quantization and packed
> weights then fail inside the weight bind with a shape error. A declaration
> with no packed tensor quantizes nothing, silently.

## Why this README is checked in

It documents what the organizer stages here, and it keeps the directory present
in a fresh clone.

The head tree digest excludes a top-level `README.md`, so this file is invisible
to head verification.

## The tree digest rule

This rule mirrors the trusted CLI's provenance sealing.

The digest is a SHA-256 over a concatenation. The concatenation holds one
`"<hex file sha256>  <relative path>\n"` entry for every regular file in the
tree, except a top-level `README.md`. The entries are in `LC_ALL=C` sorted
relative-path order. `bytes` is the byte total of the same file set.

Run this equivalent shell inside the head directory:

```sh
find . -type f ! -name README.md \
  | sed 's|^\./||' \
  | LC_ALL=C sort \
  | while read -r f; do
      printf '%s  %s\n' "$(shasum -a 256 "$f" | awk '{print $1}')" "$f"
    done | shasum -a 256
```

The harness computes this digest and reports it. It does not compare it against
the organizer's pinned digests. `docs/participant-contract.md` section 4.3
states that limit plainly.

A head only PROPOSES tokens. The organizer-pinned target decides every emitted
token. The baseline leg of every ranked pair runs the pinned head.
