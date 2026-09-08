//! The generation surface the protocol adapter drives.
//!
//! The adapter talks only to the [`Engine`] trait; the deterministic
//! [`crate::mock`] backend and the ds4 backend both implement it,
//! so the protocol plumbing is proven against the mock with no GPU and the same
//! code path wires to real inference on the box by swapping the factory.
//!
//! ## THE ENGINE REPORTS RAW COUNTERS AND NOTHING ELSE
//!
//! Nothing in this trait returns a rate, a ratio, an elapsed time or a
//! speedup. `free_decode_run` returns the committed tokens and three integer
//! counters; benchd times the call from its own side and does every division.
//! Measurement lives in benchd. An engine that reported a derived number would
//! be asking to be believed about its own speed.

use crate::protocol::{CorrectnessTraceLogit, EffectiveSpec, ExpertStreamingStats};

/// The result of one teacher-forced forward: the greedy-selected token plus the
/// top-k logits (engine-ordered, canonical lowest-id tie-break). `decode_step`
/// serializes only `token`; the correctness gates also read `top_logits`.
#[derive(Debug, Clone, PartialEq)]
pub struct Step {
    pub token: i64,
    /// Up to [`crate::protocol::TOP_LOGITS_K`] entries.
    pub top_logits: Vec<CorrectnessTraceLogit>,
}

/// The decode route a resolved spec selects. `Serial` is always runnable;
/// `Mtp` needs the native MTP head, which on this track is EMBEDDED in the
/// pinned target checkpoint. There is no third arm: DFlash was a Gemma-era
/// second arm and this track does not carry it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Route {
    Serial,
    Mtp,
}

impl Route {
    pub fn as_str(&self) -> &'static str {
        match self {
            Route::Serial => "serial",
            Route::Mtp => "mtp",
        }
    }
}

/// What one `free_decode_run(count)` phase committed. EVERY FIELD IS A RAW
/// COUNT. The invariants the Swift reference worker states for the same wire
/// (`RuntimeWorkerGenericDispatch.swift`) hold here too, and the adapter
/// re-checks them before it serializes:
///
/// * `committed_total == count == tokens.len()`;
/// * `sum(acceptance_lengths) == count`;
/// * `drafted_total >= accepted_total`.
///
/// On the SERIAL leg the drafter proposes nothing, so `drafted_total` and
/// `accepted_total` are both 0 and every acceptance length is 1. That is a
/// measurement of the control leg, not a degenerate case to special-case.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct FreeRunResult {
    /// The committed token ids, in commit order.
    pub tokens: Vec<i64>,
    /// One entry per committed round: how many tokens that round committed.
    pub acceptance_lengths: Vec<i64>,
    /// Draft tokens PROPOSED across the phase.
    pub drafted_total: i64,
    /// Draft tokens the target ACCEPTED across the phase.
    pub accepted_total: i64,
    /// Tokens committed across the phase.
    pub committed_total: i64,
    /// DIAGNOSTIC, from the engine's own counter: rejecting rounds where the
    /// batched verify and the one-row replay of the same position chose
    /// different tokens. The replay stands, so this never fails a leg.
    ///
    /// `None` means NOT REPORTED, and that is a different statement from
    /// `Some(0)`. A backend with no verify-and-replay to compare -- the mock,
    /// and the serial route, which reads no engine counters at all -- has
    /// nothing to say here and says nothing. benchd draws the same
    /// distinction: the wire field is omitted, not zeroed.
    pub verify_replay_disagreements: Option<i64>,
}

/// Why an engine call failed. The adapter turns any of these into an
/// `ok:false` + `error` response and discards the session (fail-closed).
#[derive(Debug, Clone, PartialEq)]
pub enum EngineError {
    Fault(String),
    /// The allocator drain did not reach verified-zero at phase start
    /// (`residual_bytes` still resident). Fail-closed.
    DrainNonZero {
        residual_bytes: u64,
    },
    /// The requested spec names a mode this engine cannot run. Refused BY
    /// NAME: the caller is told which mode and what is runnable, never
    /// silently downgraded to serial.
    UnsupportedMode {
        requested: String,
        runnable: String,
    },
    /// The caller NAMED a draft depth outside the envelope. Refused rather
    /// than clamped, because benchd holds the echo to an explicitly requested
    /// depth: a clamped echo reads to it as a divergence and costs the leg.
    DepthOutOfEnvelope {
        requested: i64,
        permitted: String,
    },
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EngineError::Fault(m) => write!(f, "engine fault: {m}"),
            EngineError::DrainNonZero { residual_bytes } => write!(
                f,
                "allocator drain left {residual_bytes} residual bytes (want verified-zero)"
            ),
            EngineError::UnsupportedMode {
                requested,
                runnable,
            } => write!(
                f,
                "spec mode {requested:?} is not runnable on this engine; runnable modes are {runnable}"
            ),
            EngineError::DepthOutOfEnvelope {
                requested,
                permitted,
            } => write!(
                f,
                "requested mtp draft depth {requested} is outside this track's envelope \
                 ({permitted}); an explicitly requested depth is echoed verbatim, so it is \
                 refused rather than clamped. Omit the depth to let the engine choose one."
            ),
        }
    }
}

/// The generation API the adapter calls. One instance backs exactly one phase
/// (fresh-engine-per-phase): the adapter mints a new one at each phase opener
/// and drops it at the phase-close barrier or on session discard.
pub trait Engine {
    /// Drain the allocator to verified-zero at phase start. CUDA analogue:
    /// `cudaDeviceSynchronize` + pool-trim + verified-zero. Returns residual
    /// bytes; the adapter fails the phase closed unless this is 0.
    fn drain_to_zero(&mut self) -> Result<u64, EngineError>;

    /// `prefill`: force full evaluation of the prompt, return the next token.
    fn prefill(&mut self, prompt: &[i64]) -> Result<i64, EngineError>;

    /// `decode_begin`: one seed forward opening the timed decode window;
    /// returns the seed token.
    fn decode_begin(&mut self, seed: &[i64]) -> Result<i64, EngineError>;

    /// The SHARED teacher-forced forward behind both `decode_step` and
    /// `correctness_step`: they must share the code path so the engine cannot
    /// tell it is being timed. Returns the token + top-8 logits; the adapter
    /// serializes only what each kind's response shape needs.
    fn step(&mut self, input: i64) -> Result<Step, EngineError>;

    /// `correctness_begin`: first teacher-forced forward opening the anchor
    /// gate; returns token + top-8 logits.
    fn correctness_begin(&mut self, prompt: &[i64]) -> Result<Step, EngineError>;

    /// `correctness`: free-run greedy generation of `steps` tokens.
    fn correctness_freerun(&mut self, prompt: &[i64], steps: i64) -> Result<Vec<i64>, EngineError>;

    /// Resolve a requested route + depth against what this engine can run, and
    /// open the free-run phase on the seed. Returns the seed token AND the
    /// spec that will actually run, so the adapter echoes a resolved value
    /// rather than the request's.
    ///
    /// A mode this engine cannot run is refused here, by name
    /// ([`EngineError::UnsupportedMode`]).
    ///
    /// DEPTH: THE TWO FORMS ARE NOT THE SAME REQUEST.
    ///
    /// * `requested_depth: Some(d)` is the caller NAMING a depth. benchd holds
    ///   the echo to it -- an explicitly requested depth must come back
    ///   VERBATIM -- so silently clamping an out-of-envelope `d` produces an
    ///   echo divergence, and benchd discards the leg with an opaque error
    ///   rather than reporting the clamp. An out-of-envelope explicit depth is
    ///   therefore REFUSED BY NAME ([`EngineError::DepthOutOfEnvelope`]),
    ///   which is a failure the operator can act on.
    /// * `requested_depth: None` is the caller leaving the depth to the
    ///   ENGINE. There is nothing to honour, so the engine picks a depth
    ///   inside the envelope and echoes what it picked. That form stays as it
    ///   was: engine-decides, and the echo states the decision.
    fn free_decode_begin(
        &mut self,
        seed: &[i64],
        route: Route,
        requested_depth: Option<i64>,
    ) -> Result<(i64, EffectiveSpec), EngineError>;

    /// Commit `count` more tokens on the open free-run phase, returning RAW
    /// COUNTERS ONLY. One request, one phase: benchd brackets this call with
    /// its own clock and does every division itself.
    fn free_decode_run(&mut self, count: i64) -> Result<FreeRunResult, EngineError>;

    /// Expert-streaming counters for the responses that carry them (dense
    /// runtime = zero struct).
    fn expert_stats(&self) -> ExpertStreamingStats {
        ExpertStreamingStats::zero()
    }

    /// Peak RSS in GB (engine-reported, distrusted for scoring — audit only).
    fn peak_ram_gb(&self) -> f64;
}

/// Fresh-engine-per-phase mint point. The adapter calls this at every phase
/// opener so no warm graph/allocator cache survives across phases.
pub trait EngineFactory {
    fn create(&self) -> Box<dyn Engine>;
}

impl<F> EngineFactory for F
where
    F: Fn() -> Box<dyn Engine>,
{
    fn create(&self) -> Box<dyn Engine> {
        self()
    }
}
