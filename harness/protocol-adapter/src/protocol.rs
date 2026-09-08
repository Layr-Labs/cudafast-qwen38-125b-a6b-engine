//! Engine Protocol v1 wire types (NDJSON, one JSON object per line).
//!
//! ## Provenance — a FAITHFUL MIRROR of the authoritative wire
//!
//! These types mirror the single normative definition of the wire,
//! `mlxfast-bench/crates/bench-protocol/src/lib.rs` (`WorkerRequest`,
//! `WorkerResponse`, `CorrectnessTraceLogit`, `ExpertStreamingStats`, and the
//! `RequestKind::is_timed_step` classifier), itself a port of the Swift Codable
//! types. Field names, JSON-key order, `deny_unknown_fields` (closed envelope),
//! and `skip_serializing_if = "Option::is_none"` (omit-not-null) are reproduced
//! exactly, so a canonical line parses-then-reserializes byte-identically and
//! validates against `reference/engine-protocol-v1.schema.json` (a pinned copy
//! of that crate's `schema/engine-protocol-v1.schema.json`).
//!
//! We deliberately do NOT depend on the `bench-protocol` crate: it lives in the
//! bench repo, and vendoring the structs keeps the CUDA adapter
//! self-contained. The `reference/` schema copy + the conformance tests are
//! what guard against drift.
//!
//! ## The v1.1 FREE-RUN extension, and why it is not in the pinned schema
//!
//! `reference/engine-protocol-v1.schema.json` is BASE V1. This track is scored
//! on the FREE-RUN series, whose two verbs -- `free_decode_begin` and
//! `free_decode_run` -- are the v1.1 extension the Swift reference worker
//! already speaks. Their wire fields are mirrored from that worker's own
//! `CodingKeys` (`Sources/MLXFastTrustedHarness/Gemma4RuntimeWorker.swift`:
//! `effective_spec`, `acceptance_lengths`, `drafted_total`, `accepted_total`,
//! `committed_total`) and its assembly invariants
//! (`Sources/MLXFastHarness/RuntimeWorkerGenericDispatch.swift`).
//!
//! The pinned schema is NOT edited to admit them -- it is evidence of the base
//! contract, not a design surface -- so the conformance test validates the
//! BASE-V1 responses against it and checks the free-run responses against the
//! field list above instead.
//!
//! ## BENCHD SPLITS ITS OWN CLOCK
//!
//! `free_decode_run(count)` is ONE request that commits `count` tokens. The
//! adapter reports raw counters for it and NOTHING derived: no rate, no ratio,
//! no elapsed time, no speedup. benchd times the request from its own side and
//! does every division. This is not a style preference -- it is the
//! measurement boundary (`docs/participant-contract.md`), and
//! `tests.rs` enforces it by name.

use serde::{Deserialize, Serialize};

/// Engine Protocol version reported on the `hello`.
pub const PROTOCOL_VERSION: u32 = 1;

/// Top-k logit count carried by `correctness_begin` / `correctness_step`
/// responses (`top_logits[8]`, PROTOCOL.md).
pub const TOP_LOGITS_K: usize = 8;

/// The capability string that gates the free-run verbs. benchd will not issue
/// `free_decode_begin` / `free_decode_run` to a worker whose hello does not
/// advertise it. Mirrored from the Swift reference worker's
/// `runtimeWorkerFreeRunDecodeCapability`
/// (`Sources/MLXFastHarness/RuntimeWorkerGenericDispatch.swift`).
pub const FREE_RUN_DECODE_CAPABILITY: &str = "free_run_decode";

/// The pinned, read-only copy of the authoritative JSON Schema, embedded for
/// the conformance tests.
pub const JSON_SCHEMA: &str = include_str!("../reference/engine-protocol-v1.schema.json");

/// Request kind — the benchmarker -> engine message kinds, as a typed helper
/// over the raw wire `kind` string. `hello` is NOT a request kind (it is the
/// unsolicited `id = 0` response). The first seven mirror `bench-protocol`'s
/// `RequestKind`; the last two are the v1.1 free-run extension (see the module
/// header).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RequestKind {
    Prefill,
    DecodeBegin,
    DecodeStep,
    Correctness,
    CorrectnessBegin,
    CorrectnessStep,
    PhaseDiagnostics,
    FreeDecodeBegin,
    FreeDecodeRun,
}

impl RequestKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            RequestKind::Prefill => "prefill",
            RequestKind::DecodeBegin => "decode_begin",
            RequestKind::DecodeStep => "decode_step",
            RequestKind::Correctness => "correctness",
            RequestKind::CorrectnessBegin => "correctness_begin",
            RequestKind::CorrectnessStep => "correctness_step",
            RequestKind::PhaseDiagnostics => "phase_diagnostics",
            RequestKind::FreeDecodeBegin => "free_decode_begin",
            RequestKind::FreeDecodeRun => "free_decode_run",
        }
    }

    pub fn from_wire(kind: &str) -> Option<RequestKind> {
        match kind {
            "prefill" => Some(RequestKind::Prefill),
            "decode_begin" => Some(RequestKind::DecodeBegin),
            "decode_step" => Some(RequestKind::DecodeStep),
            "correctness" => Some(RequestKind::Correctness),
            "correctness_begin" => Some(RequestKind::CorrectnessBegin),
            "correctness_step" => Some(RequestKind::CorrectnessStep),
            "phase_diagnostics" => Some(RequestKind::PhaseDiagnostics),
            "free_decode_begin" => Some(RequestKind::FreeDecodeBegin),
            "free_decode_run" => Some(RequestKind::FreeDecodeRun),
            _ => None,
        }
    }

    /// Is this a *timed step* the phase-close barrier counts toward
    /// `completed_work`? True for exactly `decode_begin | decode_step |
    /// correctness_begin | correctness_step`. This is the SINGLE source of
    /// truth, mirroring `bench-protocol` `RequestKind::is_timed_step`
    /// (bench-protocol/src/lib.rs:110). This classifies `correctness_*` as TIMED,
    /// per `docs/PROTOCOL-v1.1.md` Amendment 4 (David, 2026-08-18), which reconciled
    /// the signed spec to the authoritative wire.
    ///
    /// THE FREE-RUN PAIR IS TIMED TOO, and counts differently from the
    /// teacher-forced pair. `free_decode_begin` is the phase's first unit (the
    /// seed forward); `free_decode_run` then contributes ONE UNIT PER ROUND,
    /// so a free-run phase closes at `R + 1` where R is the number of rounds.
    ///
    /// R IS NOT THE TOKEN COUNT. benchd enforces `completed_work == R + 1`
    /// against the length of `acceptance_lengths`, and a drafting round commits
    /// several tokens, so R < N on the mtp leg. The two coincide only on the
    /// serial leg, where every round commits exactly one token -- which is why
    /// counting tokens looks right until a drafting leg runs. The adapter
    /// applies the per-round count in `adapter.rs`, because it is not 1.
    pub fn is_timed_step(&self) -> bool {
        matches!(
            self,
            RequestKind::DecodeBegin
                | RequestKind::DecodeStep
                | RequestKind::CorrectnessBegin
                | RequestKind::CorrectnessStep
                | RequestKind::FreeDecodeBegin
                | RequestKind::FreeDecodeRun
        )
    }
}

/// Benchmarker -> engine request. Flat, closed envelope. `id` + `kind` always
/// present; the rest are populated per kind. Mirrors `bench-protocol`
/// `WorkerRequest` field-for-field and in declaration order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct WorkerRequest {
    pub id: i64,
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_tokens: Option<Vec<i64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed_tokens: Option<Vec<i64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub steps: Option<i64>,
    // ---- v1.1 free-run extension ----
    /// `free_decode_run`'s committed-token count N. Unbounded on the wire, so
    /// the adapter bounds it (`adapter.rs`), exactly as the Swift reference
    /// worker does.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub count: Option<i64>,
    /// The speculative spec. Rides ONLY on `decode_begin` and
    /// `free_decode_begin`; anywhere else it is refused rather than ignored,
    /// because a silently-ignored spec is a false statement about what ran.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spec: Option<Spec>,
}

/// The requested speculative spec, a tagged union over `mode`. This track
/// admits `serial` and `mtp` ONLY -- `allowed_modes` in
/// `fixtures/qwen3_8_125b_a6b_track.json` is the authority, and any other mode
/// is refused BY NAME rather than silently run as serial.
///
/// THE ENVELOPE IS OPEN, AND THAT IS WHAT MAKES THE REFUSAL READABLE. This
/// struct does NOT set `deny_unknown_fields`, unlike the request and response
/// envelopes around it. It did, and the effect was that a caller sending a
/// REAL retired-arm spec -- `{"mode":"dflash","dflash":{...}}` -- died in
/// serde while the whole REQUEST LINE was being parsed. The adapter answered
/// `id = -1` with "not a valid WorkerRequest", because at that point it had no
/// id yet: the named "dflash is not a mode this track runs" refusal, which is
/// the message the operator needs, was never reached.
///
/// So the block is parsed permissively and the MODE is what decides. An
/// unknown mode is refused by name, with the request's own id, and any sibling
/// block it carried is simply not read. Nothing is loosened by this: a mode
/// this track does not declare is refused either way, and refusing it with an
/// id and a sentence beats refusing it with a parse error.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct Spec {
    pub mode: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mtp: Option<SpecMTP>,
}

/// The `mtp` block of a spec.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct SpecMTP {
    /// Absent means "the envelope's measured default"; the adapter resolves it
    /// and ECHOES what it resolved, so the caller is never told one depth and
    /// given another.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub depth: Option<i64>,
}

/// The resolved spec the adapter echoes on a decode opener. It is a STATEMENT
/// OF WHAT WILL RUN: the depth here is the depth the round loop uses.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct EffectiveSpec {
    pub mode: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mtp: Option<EffectiveSpecMTP>,
}

/// The resolved `mtp` block.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub struct EffectiveSpecMTP {
    pub depth: i64,
}

/// Engine -> benchmarker response. Flat, closed envelope. `id` + `ok` always
/// present. Mirrors `bench-protocol` `WorkerResponse` field-for-field and in
/// declaration order (so parse-then-reserialize is byte-identical and the
/// schema's `additionalProperties:false` holds). Responses carry NO `kind`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct WorkerResponse {
    pub id: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nonce: Option<String>,
    pub ok: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_logits: Option<Vec<CorrectnessTraceLogit>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed_token: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokens: Option<Vec<i64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expert_stats: Option<ExpertStreamingStats>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peak_ram_gb: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub protocol_version: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completed_work: Option<i64>,
    // ---- v1.1 hello advertisement ----
    /// The spec modes this worker can actually RUN, advertised on the hello.
    ///
    /// LOAD-BEARING, not informational. benchd refuses a spec whose mode is
    /// absent from this list BEFORE the timed seed forward, so a worker that
    /// omits it can run `serial` and nothing else -- the mtp leg of a paired
    /// measurement is refused at the session, and the refusal reads as a
    /// contract error rather than as a missing advertisement.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spec_modes: Option<Vec<String>>,
    /// The optional protocol surfaces this worker implements, advertised on
    /// the hello.
    ///
    /// ALSO LOAD-BEARING. benchd requires `free_run_decode` here before it
    /// will issue `free_decode_begin` / `free_decode_run` at all
    /// (bench-runner `require_free_run_capability`). Without it the scored
    /// series cannot be driven, whatever the engine can do.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capabilities: Option<Vec<String>>,
    // ---- v1.1 free-run extension ----
    /// Echoed on a decode opener: the spec that WILL run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effective_spec: Option<EffectiveSpec>,
    /// `free_decode_run` raw counters. RAW ONLY: one entry per committed round,
    /// the three totals, and nothing derived from them. The seed token rides
    /// in `seed_token` above, which means the same thing it does on
    /// `decode_begin`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub acceptance_lengths: Option<Vec<i64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub drafted_total: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub accepted_total: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub committed_total: Option<i64>,
    /// AUDIT-ONLY, NEVER SCORED: rejecting rounds where the batched verify's
    /// row-0 argmax and the one-row replay of the same position chose
    /// different tokens. The tower is not batch-invariant, so the two argmaxes
    /// can differ; the REPLAY stands (it is the row shape the serial path
    /// would have run), and the divergence is COUNTED rather than refused.
    ///
    /// OPTIONAL, and absent means NOT REPORTED -- which is NOT the claim `0`
    /// makes. A backend that runs no verify and no replay (the mock, and the
    /// serial route, which reads no engine counters at all) reports nothing
    /// here rather than reporting zero disagreements it never looked for.
    ///
    /// THE INVARIANT: only a REJECTING round can disagree, so a reported count
    /// is bounded by the rejected drafts, `drafted_total - accepted_total`.
    /// benchd refuses a leg that reports more
    /// (`bench_core::free_run::FreeRunConsistencyError::VerifyReplayDisagreementsExceedRejected`),
    /// and `adapter.rs` re-checks it here first so the bug is NAMED rather
    /// than costing the run.
    ///
    /// SHIPPING ORDER, and it is not symmetric. benchd seals this as
    /// `spec_verify_replay_disagreements` from bench `db3b73e` (pull request
    /// #258). An OLDER benchd parses the free-run response with
    /// `deny_unknown_fields` and REJECTS the whole line, so a worker that
    /// sends this field to a pre-`db3b73e` pair fails its mtp leg. Nothing on
    /// the wire lets the worker detect that: `hello` travels worker ->
    /// benchmarker, `WorkerRequest` carries no benchmarker version, and there
    /// is no handshake in the other direction to read. The constraint is
    /// therefore an ORDERING one, stated in `docs/ds4-resident.md` §4 and in
    /// this repository's pull request, not a runtime gate this crate can
    /// implement: the box's benchd pair must be >= `db3b73e` before this
    /// adapter is staged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verify_replay_disagreements: Option<u64>,
}

impl WorkerResponse {
    /// A successful response carrying `id`, `ok = true`, and the session nonce.
    pub fn ok(id: i64, nonce: &str) -> Self {
        Self {
            id,
            nonce: Some(nonce.to_string()),
            ok: true,
            ..Self::default()
        }
    }

    /// A failure response: `ok = false` with an `error` string and the nonce.
    pub fn error(id: i64, nonce: Option<&str>, message: impl Into<String>) -> Self {
        Self {
            id,
            nonce: nonce.map(str::to_string),
            ok: false,
            error: Some(message.into()),
            ..Self::default()
        }
    }
}

/// A single top-K logit entry. JSON keys are literally `token` / `logit`.
/// Mirrors `bench-protocol` `CorrectnessTraceLogit` (`token:i64`, `logit:f64`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub struct CorrectnessTraceLogit {
    pub token: i64,
    pub logit: f64,
}

impl CorrectnessTraceLogit {
    pub fn new(token: i64, logit: f64) -> Self {
        Self { token, logit }
    }
}

/// Expert-streaming counters. JSON keys are the `expert_*` names. The dense
/// RAM-resident runtime reports the zero struct. Mirrors `bench-protocol`
/// `ExpertStreamingStats`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ExpertStreamingStats {
    #[serde(rename = "expert_cache_hits")]
    pub cache_hits: u64,
    #[serde(rename = "expert_cache_misses")]
    pub cache_misses: u64,
    #[serde(rename = "expert_cache_evictions")]
    pub cache_evictions: u64,
    #[serde(rename = "expert_bytes_read")]
    pub bytes_read: u64,
    #[serde(rename = "expert_read_seconds")]
    pub read_seconds: f64,
    #[serde(rename = "expert_peak_cached_tensors")]
    pub peak_cached_tensors: u64,
}

impl ExpertStreamingStats {
    /// The all-zero stats reported by the dense RAM-resident runtime.
    pub fn zero() -> Self {
        Self::default()
    }
}
