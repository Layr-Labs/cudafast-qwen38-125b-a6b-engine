//! Engine Protocol v1 adapter for this track's CUDA engine.
//!
//! An NDJSON-over-stdio line protocol adapter: it reads request lines from
//! stdin, drives a generation engine through the [`engine::Engine`] trait, and
//! writes response lines to stdout. The whole protocol surface — framing,
//! ordering, the phase-close barrier, the `completed_work` counter,
//! fresh-engine-per-phase, session-discard-on-error, and the v1.1 free-run
//! pair — is proven against the deterministic [`mock`] backend with no GPU.
//!
//! ## WHAT THIS ENGINE IS, AND IS NOT
//!
//! * It runs SINGLE-STREAM. The scored series on this track is the paired
//!   serial-against-MTP single-stream one (David ruling 2026-08-27), and the
//!   pairing is TWO RUNS OF THE SAME PHASE under two specs. Nothing here
//!   compares them.
//! * Its ONE speculative arm is the NATIVE MTP head, embedded in the pinned
//!   target checkpoint. There is no DFlash arm and no separate drafter to
//!   stage.
//! * The FREE-RUN verbs are one request each: `free_decode_begin` opens the
//!   phase on a resolved spec, `free_decode_run(count)` commits `count`
//!   tokens. benchd splits its own clock around that one request; the wire
//!   does not change to let it. The phase closes at `R + 1` units, where R is
//!   the number of ROUNDS -- not the token count, which is larger whenever the
//!   drafter is accepted.
//! * The hello ADVERTISES `spec_modes` and `capabilities`. benchd gates on
//!   both: no `free_run_decode` capability means no free-run verbs at all, and
//!   a spec mode absent from `spec_modes` is refused before the timed seed
//!   forward.
//! * IT REPORTS RAW COUNTERS ONLY. No rate, no ratio, no elapsed time, no
//!   speedup, on any verb. Measurement and scoring live in benchd.
//!   `tests::free_run_response_carries_no_derived_metric` is the tripwire.
//!   ITS SCOPE IS TOP-LEVEL RESPONSE KEYS, and that is a real limit, accepted
//!   deliberately: `expert_stats` is a benchd-DEFINED sub-struct and one of
//!   its members is `expert_read_seconds`, so descending into nested objects
//!   would fire on benchd's own counter shape. What the tripwire covers is the
//!   surface where THIS ENGINE chooses the key names.
//!
//! See `protocol.rs` for the wire this implements and which half of it is
//! pinned by the vendored schema.

pub mod adapter;
pub mod engine;
pub mod mock;
pub mod protocol;

// The real ds4 backend: the pinned ds4 engine linked in-process, behind a
// session seam so the verb translation unit-tests with no GPU. The FFI half
// compiles only under the `ds4-engine` feature; the seam and the verb logic
// compile everywhere.
pub mod ds4_backend;

// The resident-server client: the per-phase worker's half of the topology
// that keeps the weights loaded ONCE for a whole benchmark window. It needs
// neither CUDA nor the `ds4-engine` feature -- the resident holds the engine,
// this speaks NDJSON to it -- so it is proven with a stub server on any box.
#[cfg(unix)]
pub mod resident;

pub use adapter::{Adapter, FREE_RUN_MAX_COUNT};
pub use engine::{Engine, EngineError, EngineFactory, FreeRunResult, Route, Step};
pub use protocol::{
    CorrectnessTraceLogit, EffectiveSpec, EffectiveSpecMTP, ExpertStreamingStats, RequestKind,
    Spec, SpecMTP, WorkerRequest, WorkerResponse, JSON_SCHEMA, PROTOCOL_VERSION, TOP_LOGITS_K,
};

#[cfg(test)]
mod tests;
