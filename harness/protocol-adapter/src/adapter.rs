//! The Engine Protocol v1 run loop and phase state machine.
//!
//! Reads `WorkerRequest` lines from a `BufRead`, drives an [`Engine`] minted
//! fresh per phase, and writes `WorkerResponse` lines to a `Write`.
//!
//! ## THE FREE-RUN PAIR IS THE SCORED PATH ON THIS TRACK
//!
//! `free_decode_begin` opens a single-stream phase on a resolved spec and
//! ECHOES the spec that will run; `free_decode_run(count)` commits `count`
//! tokens in ONE request and returns RAW COUNTERS ONLY. benchd brackets that
//! one request with its own clock and splits it itself -- the adapter emits no
//! rate, no ratio, no elapsed time and no speedup, on any verb. The paired
//! serial-vs-MTP comparison is two runs of this same phase under two specs,
//! and benchd is what compares them.
//!
//! Enforces:
//!
//! * the unsolicited `hello` (`id = 0`) at startup, establishing the session
//!   nonce echoed on every subsequent response;
//! * fresh-engine-per-phase, drained to verified-zero at each phase opener;
//! * the `completed_work` counter — one per *timed step* exactly as
//!   `RequestKind::is_timed_step` defines (decode_begin/decode_step +
//!   correctness_begin/correctness_step are timed; prefill and free-run
//!   correctness are not) — reported at `phase_diagnostics`. This follows the
//!   authoritative wire (`bench-protocol` `is_timed_step`, what benchd validates);
//!   `docs/PROTOCOL-v1.1.md` **Amendment 4** (2026-08-18) reconciles the signed
//!   spec's §2.6/§2.7 prose to this classification (correctness IS timed for the
//!   counter; "not on the timed path" there means the scored speed path only);
//! * the spec's PLACEMENT: it rides on `decode_begin` and `free_decode_begin`
//!   and is refused by name on every other verb, so no request is answered
//!   `ok` while the spec it carried was dropped;
//! * session-discard-on-error and on early EOF (fail-closed);
//! * fail-closed on a step with no matching opener and on a double-open;
//! * one JSON object per line, echoing the request `id`.

use std::io::{BufRead, Write};

use crate::engine::{Engine, EngineError, EngineFactory, Route};
use crate::protocol::{
    EffectiveSpec, ExpertStreamingStats, RequestKind, Spec, WorkerRequest, WorkerResponse,
    FREE_RUN_DECODE_CAPABILITY, PROTOCOL_VERSION,
};

/// The bound on `free_decode_run`'s `count`. The wire leaves N unbounded, so
/// the adapter bounds it rather than letting a caller ask for an arbitrary
/// phase length. Same value the Swift reference worker uses
/// (`MLXFastConstants.freeRunMaxConfiguredTotalTokens`).
pub const FREE_RUN_MAX_COUNT: i64 = 1_536;

/// The bound on `correctness`'s `steps`. The wire leaves it unbounded and the
/// backends ALLOCATE on it (`Vec::with_capacity(steps)`), so an absurd value
/// aborted the process on a capacity overflow instead of answering with a
/// protocol error. The value is the resident session context
/// (`tools/serve-up.sh` `SERVE_UP_CTX_SIZE`, default 8192, and its ceiling):
/// a golden tape longer than the context the engine serves cannot exist.
pub const CORRECTNESS_MAX_STEPS: i64 = 8_192;

/// Which opener started the currently-open phase. Steps must match their
/// opener (`decode_step` only inside a `Decode` phase, `correctness_step` only
/// inside a `CorrectnessAnchor` phase); an opener while a phase is already open
/// is a fail-closed double-open.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PhaseKind {
    Prefill,
    Decode,
    CorrectnessFreeRun,
    CorrectnessAnchor,
    /// The v1.1 free-run phase: `free_decode_begin` opens it,
    /// `free_decode_run` commits N and is the phase's last work unit.
    FreeRun,
}

/// A fatal-to-this-request error. The adapter emits a matching `ok:false`
/// response, discards the session (fail-closed), and continues reading.
#[derive(Debug)]
enum ReqError {
    Malformed(String),
    Protocol(String),
    Engine(EngineError),
}

impl ReqError {
    fn message(&self) -> String {
        match self {
            ReqError::Malformed(m) | ReqError::Protocol(m) => m.clone(),
            ReqError::Engine(e) => e.to_string(),
        }
    }
}

/// The protocol adapter. Generic over the engine factory so the mock and the
/// real backend share every line of this loop.
pub struct Adapter<F: EngineFactory> {
    factory: F,
    nonce: String,
    backend: String,
    device: String,
    /// The current phase's engine, minted at its opener and dropped at the
    /// barrier / on session discard.
    engine: Option<Box<dyn Engine>>,
    /// Which opener started the open phase (`None` between phases).
    phase: Option<PhaseKind>,
    /// The routes this worker advertises as runnable, in hello order. This
    /// track declares `serial` and `mtp` (`allowed_modes` in
    /// `fixtures/qwen3_8_125b_a6b_track.json`), and the ENGINE is what decides
    /// whether a declared mode is actually runnable on a given instance -- so
    /// a mode advertised here can still be refused by name at resolution.
    spec_modes: Vec<Route>,
    /// Monotonic count of timed steps completed in the current phase.
    completed_work: i64,
}

impl<F: EngineFactory> Adapter<F> {
    /// Build an adapter announcing an explicit backend/device identity in its
    /// hello, with a fresh session nonce.
    ///
    /// EVERY backend names ITSELF here. There is no default identity, because a
    /// default is what let the mock announce `cuda`/`cuda`, byte-identically to
    /// a run on the real engine: benchd seals these two strings as
    /// `engine_backend`/`engine_device` (`crates/benchctl/src/official.rs`,
    /// `seal_engine_identity`), so a sealed artifact said nothing about which
    /// engine produced the number. Nothing SCORES on them; they are how a
    /// reader tells a measurement from a protocol exercise.
    pub fn with_backend(factory: F, backend: impl Into<String>, device: impl Into<String>) -> Self {
        Self::with_session(factory, backend, device, generate_nonce())
    }

    /// Build an adapter with an explicit backend/device/nonce (tests pin the
    /// nonce for determinism).
    pub fn with_session(
        factory: F,
        backend: impl Into<String>,
        device: impl Into<String>,
        nonce: impl Into<String>,
    ) -> Self {
        Adapter {
            factory,
            nonce: nonce.into(),
            backend: backend.into(),
            device: device.into(),
            engine: None,
            phase: None,
            spec_modes: vec![Route::Serial, Route::Mtp],
            completed_work: 0,
        }
    }

    /// Narrow the advertised spec modes. Used by tests that pair a
    /// capability-restricted engine with a hello that says so.
    pub fn advertising(mut self, modes: Vec<Route>) -> Self {
        self.spec_modes = modes;
        self
    }

    /// Run the protocol loop to EOF. Returns `Err` only on a hard I/O error;
    /// protocol/engine errors are reported inline (`ok:false`) and the loop
    /// continues after discarding the phase session. On EOF any in-flight
    /// session is discarded fail-closed.
    pub fn run<R: BufRead, W: Write>(&mut self, input: R, mut output: W) -> std::io::Result<()> {
        // Unsolicited startup hello (id = 0): announces protocol version +
        // backend/device, ADVERTISES the runnable spec modes and the optional
        // surfaces, and establishes the session nonce.
        //
        // THE TWO ADVERTISEMENTS ARE NOT DECORATION. benchd gates on both,
        // before any timed work:
        //
        //   * it refuses `free_decode_begin` / `free_decode_run` outright
        //     unless `capabilities` carries `free_run_decode`; and
        //   * it refuses a spec whose `mode` is absent from `spec_modes`,
        //     before the timed seed forward.
        //
        // So a hello that omits them describes an engine that can run the
        // serial control leg and nothing else. The mtp leg -- half of the
        // paired measurement this track is scored on -- would be refused at
        // the session with an error about the contract rather than about the
        // missing advertisement.
        let hello = WorkerResponse {
            id: 0,
            nonce: Some(self.nonce.clone()),
            ok: true,
            expert_stats: Some(ExpertStreamingStats::zero()),
            protocol_version: Some(PROTOCOL_VERSION),
            backend: Some(self.backend.clone()),
            device: Some(self.device.clone()),
            spec_modes: Some(
                self.spec_modes
                    .iter()
                    .map(|r| r.as_str().to_string())
                    .collect(),
            ),
            capabilities: Some(vec![FREE_RUN_DECODE_CAPABILITY.to_string()]),
            ..Default::default()
        };
        self.emit(&mut output, &hello)?;

        for line in input.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let response = self.service(&line);
            self.emit(&mut output, &response)?;
        }

        // Clean EOF: discard any half-advanced in-flight session (no barrier
        // synthesized).
        self.discard_session();
        Ok(())
    }

    /// Service one request line, always producing a `WorkerResponse`. On any
    /// error this discards the phase session (fail-closed) before returning the
    /// `ok:false` response.
    fn service(&mut self, line: &str) -> WorkerResponse {
        // Parse first: an unparseable line answers with id = -1.
        let request: WorkerRequest = match serde_json::from_str(line) {
            Ok(r) => r,
            Err(e) => {
                self.discard_session();
                return WorkerResponse::error(
                    -1,
                    Some(&self.nonce),
                    format!("request line was not a valid WorkerRequest: {e}"),
                );
            }
        };
        let id = request.id;

        match self.dispatch(&request) {
            Ok(resp) => resp,
            Err(err) => {
                // Fail-closed: any error discards the phase's session state.
                self.discard_session();
                let msg = match &err {
                    ReqError::Malformed(_) => format!("malformed request: {}", err.message()),
                    _ => err.message(),
                };
                WorkerResponse::error(id, Some(&self.nonce), msg)
            }
        }
    }

    fn dispatch(&mut self, request: &WorkerRequest) -> Result<WorkerResponse, ReqError> {
        let kind = RequestKind::from_wire(&request.kind).ok_or_else(|| {
            ReqError::Malformed(format!("unknown request kind {:?}", request.kind))
        })?;
        let id = request.id;

        // THE SPEC RIDES ON THE DECODE OPENERS AND NOWHERE ELSE, and a spec
        // that arrives anywhere else is REFUSED rather than dropped. A verb
        // that read no spec but answered `ok` said, by answering, that it ran
        // what was asked for -- and on every verb but these two, it did not.
        // Checked once here, on the resolved kind, so no arm can forget it.
        if request.spec.is_some()
            && !matches!(
                kind,
                RequestKind::DecodeBegin | RequestKind::FreeDecodeBegin
            )
        {
            return Err(ReqError::Malformed(format!(
                "spec is not accepted on {}; it rides only on decode_begin and free_decode_begin",
                kind.as_str()
            )));
        }

        let response = match kind {
            // ---- opener: prefill (single forward, NOT a timed step) ----
            RequestKind::Prefill => {
                let prompt = self.require_tokens(&request.prompt_tokens, "prompt_tokens")?;
                let mut engine = self.open_phase(PhaseKind::Prefill)?;
                let token = engine.prefill(prompt).map_err(ReqError::Engine)?;
                self.install(engine, kind);
                let mut r = self.ok(id);
                r.token = Some(token);
                r
            }

            // ---- opener: decode_begin (seed forward, TIMED) ----
            RequestKind::DecodeBegin => {
                let seed = self.require_tokens(&request.seed_tokens, "seed_tokens")?;
                // DECODE_BEGIN IS THE TEACHER-FORCED v1 VERB, AND IT RUNS
                // SERIAL. It takes a spec so that a caller who names the
                // control leg is answered rather than second-guessed: an
                // absent spec and `{"mode":"serial"}` are the same request.
                // A DRAFTING spec is refused BY NAME here, because the verb
                // has no drafting path -- `engine.decode_begin` takes no route
                // -- and running it as serial would answer `ok` to a request
                // for something else.
                let (route, _) = Self::resolve_spec(request.spec.as_ref())?;
                if route != Route::Serial {
                    return Err(ReqError::Malformed(format!(
                        "spec mode {:?} is not accepted on decode_begin; decode_begin is the \
                         teacher-forced v1 verb and runs serial, and the mtp route runs through \
                         free_decode_begin",
                        route.as_str()
                    )));
                }
                let mut engine = self.open_phase(PhaseKind::Decode)?;
                let seed_token = engine.decode_begin(seed).map_err(ReqError::Engine)?;
                self.install(engine, kind);
                let mut r = self.ok(id);
                r.seed_token = Some(seed_token);
                // THE ECHO IS ALWAYS EMITTED, spec or no spec, and it is the
                // same serial shape the engines build for `free_decode_begin`
                // (`mode: "serial"`, no `mtp` block). It states what ran, which
                // on this verb is serial in every case. benchd checks the echo
                // only when it sent a spec (`require_spec_echo`), so echoing on
                // a no-spec opener adds a true statement and gates nothing.
                r.effective_spec = Some(EffectiveSpec {
                    mode: Route::Serial.as_str().to_string(),
                    mtp: None,
                });
                r
            }

            // ---- step: decode_step (TIMED, requires an open Decode phase) ----
            RequestKind::DecodeStep => {
                let token_in = self.require_token(&request.token)?;
                let engine = self.require_step_engine(kind, PhaseKind::Decode)?;
                let step = engine.step(token_in).map_err(ReqError::Engine)?;
                self.completed_work += 1; // is_timed_step
                let mut r = self.ok(id);
                r.token = Some(step.token); // decode_step response is token-only
                r
            }

            // ---- opener: correctness (free-run greedy, NOT timed) ----
            RequestKind::Correctness => {
                let prompt = self.require_tokens(&request.prompt_tokens, "prompt_tokens")?;
                let steps = request
                    .steps
                    .ok_or_else(|| ReqError::Malformed("correctness missing steps".into()))?;
                // BOUNDED BEFORE THE ENGINE SEES IT, the same posture as
                // `free_decode_run`'s count. The backends size a `Vec` on this
                // number, so an absurd `steps` killed the process with a
                // capacity overflow -- the caller got an exit code where the
                // protocol has an error message, and the session died with it.
                if steps <= 0 {
                    return Err(ReqError::Malformed(format!(
                        "correctness steps must be positive, got {steps}"
                    )));
                }
                if steps > CORRECTNESS_MAX_STEPS {
                    return Err(ReqError::Malformed(format!(
                        "correctness steps {steps} is above the bound {CORRECTNESS_MAX_STEPS}"
                    )));
                }
                let mut engine = self.open_phase(PhaseKind::CorrectnessFreeRun)?;
                let tokens = engine
                    .correctness_freerun(prompt, steps)
                    .map_err(ReqError::Engine)?;
                let peak = engine.peak_ram_gb();
                self.install(engine, kind);
                let mut r = self.ok(id);
                r.tokens = Some(tokens);
                r.peak_ram_gb = Some(peak); // NOTE: plain correctness carries NO expert_stats
                r
            }

            // ---- opener: correctness_begin (teacher-forced, TIMED) ----
            RequestKind::CorrectnessBegin => {
                let prompt = self.require_tokens(&request.prompt_tokens, "prompt_tokens")?;
                let mut engine = self.open_phase(PhaseKind::CorrectnessAnchor)?;
                let step = engine.correctness_begin(prompt).map_err(ReqError::Engine)?;
                let stats = engine.expert_stats();
                let peak = engine.peak_ram_gb();
                self.install(engine, kind);
                self.correctness_gate_response(id, step, stats, peak)
            }

            // ---- step: correctness_step (TIMED, requires CorrectnessAnchor) ----
            RequestKind::CorrectnessStep => {
                let token_in = self.require_token(&request.token)?;
                let engine = self.require_step_engine(kind, PhaseKind::CorrectnessAnchor)?;
                let step = engine.step(token_in).map_err(ReqError::Engine)?;
                let stats = engine.expert_stats();
                let peak = engine.peak_ram_gb();
                self.completed_work += 1; // is_timed_step
                self.correctness_gate_response(id, step, stats, peak)
            }

            // ---- opener: free_decode_begin (seed forward, TIMED) ----
            RequestKind::FreeDecodeBegin => {
                let seed = self.require_tokens(&request.seed_tokens, "seed_tokens")?;
                let (route, depth) = Self::resolve_spec(request.spec.as_ref())?;
                let mut engine = self.open_phase(PhaseKind::FreeRun)?;
                let (seed_token, effective) = engine
                    .free_decode_begin(seed, route, depth)
                    .map_err(ReqError::Engine)?;
                self.install(engine, kind);
                let mut r = self.ok(id);
                r.seed_token = Some(seed_token);
                // THE ECHO IS A STATEMENT OF WHAT WILL RUN. The engine
                // resolved it (clamping the depth if need be); the adapter
                // forwards what came back, never what was asked for.
                r.effective_spec = Some(effective);
                r
            }

            // ---- step: free_decode_run (TIMED, commits N) ----
            RequestKind::FreeDecodeRun => {
                let count = request
                    .count
                    .ok_or_else(|| ReqError::Malformed("free_decode_run missing count".into()))?;
                if count <= 0 {
                    return Err(ReqError::Malformed(format!(
                        "free_decode_run count must be positive, got {count}"
                    )));
                }
                if count > FREE_RUN_MAX_COUNT {
                    return Err(ReqError::Malformed(format!(
                        "free_decode_run count {count} is above the bound {FREE_RUN_MAX_COUNT}"
                    )));
                }
                let engine = self.require_step_engine(kind, PhaseKind::FreeRun)?;
                let result = engine.free_decode_run(count).map_err(ReqError::Engine)?;

                // THE CONSISTENCY TRIPLE, re-checked here before anything is
                // serialized. A backend that returned an inconsistent phase
                // would otherwise publish counters benchd cannot reconcile,
                // and benchd would refuse the RUN rather than name the bug.
                let committed: i64 = result.acceptance_lengths.iter().sum();
                if result.committed_total != count {
                    return Err(ReqError::Protocol(format!(
                        "free_decode_run committed_total {} != count {count}",
                        result.committed_total
                    )));
                }
                if result.tokens.len() as i64 != count {
                    return Err(ReqError::Protocol(format!(
                        "free_decode_run returned {} tokens, expected count {count}",
                        result.tokens.len()
                    )));
                }
                if committed != count {
                    return Err(ReqError::Protocol(format!(
                        "free_decode_run sum(acceptance_lengths) {committed} != count {count}"
                    )));
                }
                if result.drafted_total < result.accepted_total {
                    return Err(ReqError::Protocol(format!(
                        "free_decode_run drafted_total {} < accepted_total {}",
                        result.drafted_total, result.accepted_total
                    )));
                }
                // THE DISAGREEMENT BOUND, when the backend reports one. Only a
                // REJECTING round can disagree, so the count cannot exceed the
                // rejected drafts. benchd enforces exactly this
                // (`FreeRunConsistencyError::VerifyReplayDisagreementsExceedRejected`)
                // and refuses the leg; naming it here costs the same run and
                // says which counter is wrong.
                if let Some(disagreements) = result.verify_replay_disagreements {
                    let rejected = result.drafted_total - result.accepted_total;
                    if disagreements < 0 || disagreements > rejected {
                        return Err(ReqError::Protocol(format!(
                            "free_decode_run verify_replay_disagreements {disagreements} is not \
                             within the rejected drafts (drafted_total {} - accepted_total {} = \
                             {rejected})",
                            result.drafted_total, result.accepted_total
                        )));
                    }
                }
                // NOT CHECKED HERE: that every acceptance length is positive.
                // benchd's own schema gives `acceptance_lengths` a minimum of
                // 0, so a zero-length round is a shape benchd ACCEPTS. An
                // adapter-side rejection of it would be this repository
                // inventing an invariant the contract does not have, and would
                // refuse a run benchd would have scored.

                // COMPLETED_WORK COUNTS ROUNDS, NOT TOKENS.
                //
                // benchd requires `completed_work == R + 1`, where R is the
                // number of ROUNDS the phase ran -- one per
                // `acceptance_lengths` entry (bench-core free_run.rs, "free-run
                // completed_work {} != R+1"). The opener already counted the
                // seed forward as 1; each round is one more unit.
                //
                // THIS IS NOT `count`. On the serial leg the two agree, because
                // every round commits exactly one token and R == N -- which is
                // exactly why counting tokens looks correct until a DRAFTING
                // leg runs. On the mtp leg a round commits several tokens, so
                // R < N, and counting tokens overstates the work by the
                // difference. A 10-token mtp phase over 4 rounds must report 5,
                // not 11.
                self.completed_work += result.acceptance_lengths.len() as i64;

                let mut r = self.ok(id);
                r.tokens = Some(result.tokens);
                r.acceptance_lengths = Some(result.acceptance_lengths);
                r.drafted_total = Some(result.drafted_total);
                r.accepted_total = Some(result.accepted_total);
                r.committed_total = Some(result.committed_total);
                // AUDIT-ONLY, and OMITTED when the backend measured nothing:
                // absent means NOT REPORTED, which is not what `0` says.
                //
                // WHEN THIS FIELD IS EMITTED IT NEEDS A benchd >= db3b73e ON
                // THE BOX. An older benchd parses this response with
                // `deny_unknown_fields` and rejects the line. The worker
                // cannot detect that -- benchd announces no version to it (see
                // `protocol.rs`) -- so the ordering is a deployment
                // constraint, written down in `docs/ds4-resident.md` §4.
                r.verify_replay_disagreements =
                    result.verify_replay_disagreements.map(|d| d as u64);
                r
            }

            // ---- barrier: phase_diagnostics (closes the open phase) ----
            RequestKind::PhaseDiagnostics => {
                let engine = self.engine.as_ref().ok_or_else(|| {
                    ReqError::Protocol("phase_diagnostics with no open phase to close".into())
                })?;
                let stats = engine.expert_stats();
                let peak = engine.peak_ram_gb();
                let completed_work = self.completed_work;
                // Report-then-reset: drop the fresh-per-phase engine, clear the
                // phase, zero the counter.
                self.discard_session();
                let mut r = self.ok(id);
                r.expert_stats = Some(stats);
                r.peak_ram_gb = Some(peak);
                r.completed_work = Some(completed_work);
                r
            }
        };
        Ok(response)
    }

    /// Resolve a requested `spec` into a route plus an optional requested
    /// depth. THE SPEC RIDES ONLY ON A DECODE OPENER, and this track admits
    /// two modes; anything else is refused BY NAME rather than run as serial.
    /// An ABSENT spec is `serial`, which is the v1 behaviour and the control
    /// leg.
    fn resolve_spec(spec: Option<&Spec>) -> Result<(Route, Option<i64>), ReqError> {
        let Some(spec) = spec else {
            return Ok((Route::Serial, None));
        };
        match spec.mode.as_str() {
            "serial" => {
                if spec.mtp.is_some() {
                    return Err(ReqError::Malformed(
                        "spec mode \"serial\" carries an mtp block; the two disagree about what would run".into(),
                    ));
                }
                Ok((Route::Serial, None))
            }
            "mtp" => Ok((Route::Mtp, spec.mtp.and_then(|m| m.depth))),
            other => Err(ReqError::Malformed(format!(
                "spec mode {other:?} is not a mode this track runs; allowed_modes is serial, mtp"
            ))),
        }
    }

    /// A base `ok:true` response carrying `id` and the session nonce.
    fn ok(&self, id: i64) -> WorkerResponse {
        WorkerResponse::ok(id, &self.nonce)
    }

    /// Both correctness gate kinds share the same response shape: token +
    /// top_logits[8] + expert_stats + peak_ram_gb.
    fn correctness_gate_response(
        &self,
        id: i64,
        step: crate::engine::Step,
        stats: ExpertStreamingStats,
        peak: f64,
    ) -> WorkerResponse {
        let mut r = self.ok(id);
        r.token = Some(step.token);
        r.top_logits = Some(step.top_logits);
        r.expert_stats = Some(stats);
        r.peak_ram_gb = Some(peak);
        r
    }

    /// Open a phase: reject a double-open (fail-closed), mint a cold engine, and
    /// drain it to verified-zero, resetting the completed-work counter.
    fn open_phase(&mut self, phase: PhaseKind) -> Result<Box<dyn Engine>, ReqError> {
        if let Some(open) = self.phase {
            return Err(ReqError::Protocol(format!(
                "double-open: {phase:?} arrived while a {open:?} phase was still open (missing phase_diagnostics)"
            )));
        }
        self.completed_work = 0;
        let mut engine = self.factory.create();
        let residual = engine.drain_to_zero().map_err(ReqError::Engine)?;
        if residual != 0 {
            return Err(ReqError::Engine(EngineError::DrainNonZero {
                residual_bytes: residual,
            }));
        }
        Ok(engine)
    }

    // ============================================================================
    // completed_work counting — follows the AUTHORITATIVE classifier
    // ----------------------------------------------------------------------------
    // The adapter counts one unit of completed_work per *timed step*, using
    // `RequestKind::is_timed_step` (mirrored from bench-protocol/src/lib.rs:110,
    // what benchd validates). Under that classifier:
    //   * decode_begin, decode_step, correctness_begin, correctness_step -> TIMED
    //     (each contributes 1): a prefill-only phase reports 0; a decode phase
    //     reports 1 + N steps; a correctness anchor phase reports 1 + N steps.
    //   * prefill, correctness (free-run), phase_diagnostics -> NOT timed.
    //
    // RULED (David, 2026-08-18): `docs/PROTOCOL-v1.1.md` **Amendment 4** reconciles
    // the signed spec to this classification — correctness_begin/_step ARE timed in
    // base v1; the §2.6/§2.7 "timed path" prose refers to the scored speed path, not
    // the completed_work counter. The adapter's behavior here matches the amendment.
    // ============================================================================

    /// Commit a freshly-opened engine as the current phase.
    fn install(&mut self, engine: Box<dyn Engine>, opener: RequestKind) {
        self.engine = Some(engine);
        self.phase = Some(match opener {
            RequestKind::Prefill => PhaseKind::Prefill,
            RequestKind::DecodeBegin => PhaseKind::Decode,
            RequestKind::Correctness => PhaseKind::CorrectnessFreeRun,
            RequestKind::CorrectnessBegin => PhaseKind::CorrectnessAnchor,
            RequestKind::FreeDecodeBegin => PhaseKind::FreeRun,
            // steps/barrier never install a phase.
            RequestKind::DecodeStep
            | RequestKind::CorrectnessStep
            | RequestKind::FreeDecodeRun
            | RequestKind::PhaseDiagnostics => unreachable!("not an opener"),
        });
        if opener.is_timed_step() {
            self.completed_work += 1;
        }
    }

    /// Borrow the open engine for a step, fail-closed unless the open phase was
    /// started by the matching opener (N1: no timed count on unverified state).
    fn require_step_engine(
        &mut self,
        step_kind: RequestKind,
        want: PhaseKind,
    ) -> Result<&mut Box<dyn Engine>, ReqError> {
        match self.phase {
            Some(p) if p == want => Ok(self.engine.as_mut().expect("open phase implies an engine")),
            Some(p) => Err(ReqError::Protocol(format!(
                "{} has no matching opener: current phase is {:?}, not {:?}",
                step_kind.as_str(),
                p,
                want
            ))),
            None => Err(ReqError::Protocol(format!(
                "{} with no open phase",
                step_kind.as_str()
            ))),
        }
    }

    fn require_tokens<'a>(
        &self,
        field: &'a Option<Vec<i64>>,
        name: &str,
    ) -> Result<&'a [i64], ReqError> {
        field
            .as_deref()
            .ok_or_else(|| ReqError::Malformed(format!("missing {name}")))
    }

    fn require_token(&self, field: &Option<i64>) -> Result<i64, ReqError> {
        field.ok_or_else(|| ReqError::Malformed("missing token".into()))
    }

    /// Fail-closed session discard: drop the engine, clear the phase, zero the
    /// counter.
    fn discard_session(&mut self) {
        self.engine = None;
        self.phase = None;
        self.completed_work = 0;
    }

    /// Serialize one response as a single NDJSON line (object + '\n') and flush.
    fn emit<W: Write>(&self, output: &mut W, response: &WorkerResponse) -> std::io::Result<()> {
        let mut line = serde_json::to_vec(response).expect("WorkerResponse serializes");
        line.push(b'\n');
        output.write_all(&line)?;
        output.flush()
    }
}

/// A best-effort unpredictable session nonce, zero-dependency (time + pid,
/// hashed). Adequate for the replay/co-tenant defense on the TCP bridge; tests
/// pin it via [`Adapter::with_session`].
fn generate_nonce() -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::time::{SystemTime, UNIX_EPOCH};

    let mut hasher = DefaultHasher::new();
    std::process::id().hash(&mut hasher);
    if let Ok(dur) = SystemTime::now().duration_since(UNIX_EPOCH) {
        dur.as_nanos().hash(&mut hasher);
    }
    // A second sample decorrelates same-nanosecond starts.
    let addr = &hasher as *const _ as usize;
    addr.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}
