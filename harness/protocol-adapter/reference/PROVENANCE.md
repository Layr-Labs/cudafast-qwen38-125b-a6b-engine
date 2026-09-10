# `reference/` — pinned copy of the authoritative Engine Protocol v1 schema

`engine-protocol-v1.schema.json` in this directory is a **verbatim, read-only
copy** of the single normative wire schema:

- **Source (authoritative):**
  `mlxfast-bench/crates/bench-protocol/schema/engine-protocol-v1.schema.json`
  (JSON Schema Draft 2020-12), embedded in that crate as
  `bench_protocol::JSON_SCHEMA` and kept struct/schema-consistent there by
  `schema_properties_match_struct_fields` (bench-protocol/src/lib.rs).
- **Copied:** 2026-08-18, byte-identical (`diff -q` clean, 7288 bytes).

## Why a vendored copy

The CUDA adapter lives in a different repository from the authoritative schema
and must not depend on the bench repository's build. This copy lets the
adapter's conformance tests (`src/tests.rs`) validate the BASE-V1 response
lines against the real contract, offline, with no cross-repo dependency.

## What this schema does NOT cover

It is BASE V1. The two v1.1 free-run verbs this track is scored on --
`free_decode_begin` and `free_decode_run` -- and their response fields
(`effective_spec`, `acceptance_lengths`, `drafted_total`, `accepted_total`,
`committed_total`) are outside it. Those are mirrored from the Swift reference
worker's own `CodingKeys`, cited in `src/protocol.rs`, and the free-run
responses are checked against that field list rather than against this file.

The schema is NOT extended to admit them. It is evidence of the base contract,
and editing it would destroy the only thing it is good for.

## Rules

- **Read-only.** Do not hand-edit this file. It is evidence of the contract, not
  a design surface. If the upstream schema changes, re-copy it verbatim and note
  the new date/bytes here. Copied into THIS repository 2026-08-28, byte-identical
  to the copy in `Layr-Labs/cudafast-engine`.
- The adapter's wire types (`src/protocol.rs`) mirror
  `bench-protocol`'s `WorkerRequest` / `WorkerResponse` /
  `CorrectnessTraceLogit` / `ExpertStreamingStats`; this schema is what the
  conformance tests use to catch any drift between them.
