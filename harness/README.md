# `harness/` — the CUDA engine's protocol side

This directory holds the Engine Protocol v1 adapter. The adapter is what benchd
talks to. It IS an editable path (`benchmark.json` `editablePaths` lists
`harness`): the adapter is the engine's protocol side, and the measurement is
taken by benchd on its own side of the wire.

## What is here

| Path | What it is |
|---|---|
| `protocol-adapter/` | The NDJSON-over-stdio adapter, its deterministic mock backend, and the ds4 backend (`src/ds4_backend.rs` over `ds4_shim/`). |

## Why it is a standalone cargo crate

The adapter must build and test with no CUDA toolchain and no GPU. Keeping it
out of the engine's build tree is what makes `cargo test` a laptop and hosted-CI
command. The engine behind it is swapped by changing the factory, and the
protocol loop is the same code in both cases.

## What it does not do

It does not measure and it does not score. It reports raw counters; benchd
times the requests and does every division. `free_decode_run` is the clearest
case: ONE request commits N tokens, benchd brackets that request with its own
clock, and the response carries the committed tokens plus three integer
counters and nothing else.

## Status

The ds4 backend is REAL: the ds4 engine from the pinned `ds4/` submodule --
`Layr-Labs/ds4`, our qwen4exp port -- linked in-process through the flat C surface in
`ds4_shim/ds4_shim.h`, behind a session seam so the verb translation, the
free-run counter accounting and every refusal are unit-tested against a
scripted session with no GPU. The `ds4-engine` feature is off by default, so
`cuda-engine` serves the mock unless the box build asks for ds4;
`tools/ds4/build.sh` produces the `libds4qwen.so` the feature links. The real
GPU drive is box-validated by `tools/ds4/mtp-exactness-gate.py`.
