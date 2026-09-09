//! A deterministic, GPU-free mock engine for exercising the adapter's protocol
//! behavior (framing, ordering, barriers, counters, lifecycle, error/EOF).
//!
//! The mock implements the same [`Engine`] trait the ds4 backend does,
//! so every test drives the *actual* adapter loop — only the token source is
//! stubbed. Tokens are a fixed function of the input, so tests assert exact
//! values. A shared [`MockLog`] records lifecycle events (create / drain /
//! forward / drop) so tests can prove fresh-engine-per-phase and
//! session-discard behavior.

use std::sync::{Arc, Mutex};

use crate::engine::{Engine, EngineError, FreeRunResult, Route, Step};
use crate::protocol::{
    CorrectnessTraceLogit, EffectiveSpec, EffectiveSpecMTP, ExpertStreamingStats, TOP_LOGITS_K,
};

/// A lifecycle event recorded by mock engines, in order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Event {
    /// A fresh engine instance was minted by the factory (carries its serial).
    Created(u64),
    /// `drain_to_zero` was called on the given instance.
    Drained(u64),
    /// A forward ran on the given instance: (serial, method, input token).
    Forward(u64, Method, i64),
    /// The engine instance was dropped (phase closed / session discarded).
    Dropped(u64),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    Prefill,
    DecodeBegin,
    Step,
    CorrectnessBegin,
    CorrectnessFreeRun,
    FreeDecodeBegin,
    FreeDecodeRun,
}

/// Shared, ordered lifecycle log. Clone it to hold a handle after the factory
/// has moved into the adapter.
#[derive(Debug, Clone, Default)]
pub struct MockLog {
    events: Arc<Mutex<Vec<Event>>>,
    next_serial: Arc<Mutex<u64>>,
}

impl MockLog {
    pub fn new() -> Self {
        Self::default()
    }

    fn push(&self, e: Event) {
        self.events.lock().unwrap().push(e);
    }

    fn next_serial(&self) -> u64 {
        let mut s = self.next_serial.lock().unwrap();
        let serial = *s;
        *s += 1;
        serial
    }

    /// Snapshot of all events so far, in order.
    pub fn events(&self) -> Vec<Event> {
        self.events.lock().unwrap().clone()
    }

    /// How many distinct engine instances the factory minted.
    pub fn created_count(&self) -> usize {
        self.events()
            .iter()
            .filter(|e| matches!(e, Event::Created(_)))
            .count()
    }
}

/// Tunables for negative tests. Default = a well-behaved engine.
#[derive(Debug, Clone, Default)]
pub struct MockConfig {
    /// Residual bytes `drain_to_zero` reports. Non-zero drives the fail-closed
    /// drain path.
    pub drain_residual: u64,
    /// If set, any forward whose input token equals this value returns a
    /// `Fault` (models a mid-phase engine failure).
    pub fault_on_input: Option<i64>,
    /// Routes this mock engine can run. Default = serial + mtp, the two this
    /// track declares. Narrow it to prove the refusal path.
    pub runnable: Option<Vec<Route>>,
}

/// What the mock announces as its backend, and what benchd then seals as
/// `engine_backend`. It says `mock` and nothing else, so a sealed artifact from
/// a protocol exercise can never be read as a CUDA measurement. The mock
/// announced `cuda` until 2026-09-03, byte-identically to a run on the real
/// engine.
pub const MOCK_BACKEND: &str = "mock";
/// What the mock announces as its device, sealed as `engine_device`. There is
/// no device: the mock's tokens are a function of the input and its timings
/// mean nothing.
pub const MOCK_DEVICE: &str = "none";

/// Deterministic peak RSS the mock reports (audit-only field).
pub const MOCK_PEAK_RAM_GB: f64 = 18.5;
pub const PREFILL_BASE: i64 = 100_000;
pub const SEED_BASE: i64 = 200_000;

/// The MTP envelope's ceiling on this track: permitted depths are 1, 2 and 3
/// (`fixtures/qwen3_8_125b_a6b_track.json` `mtp_head.permitted_draft_depths`).
/// An EXPLICITLY REQUESTED depth outside 1...3 is REFUSED (benchd echoes an
/// explicit request verbatim, so a clamp would read as divergence); an ABSENT
/// depth is the engine's to choose, and it chooses inside this bound.
pub const MTP_MAX_DEPTH: i64 = 6;
/// The floor of the same envelope.
pub const MTP_MIN_DEPTH: i64 = 1;
/// The depth benchd measures when the invocation names none.
pub const MTP_DEFAULT_DEPTH: i64 = 2;

/// A deterministic engine instance. Tokens are a pure function of the input:
///   * `prefill(prompt)`         -> `PREFILL_BASE + prompt.len()`
///   * `decode_begin(seed)`      -> `SEED_BASE + seed.len()`
///   * `step(t)`                 -> `t + 1`  (shared decode_step/correctness_step)
///   * `correctness_begin(p)`    -> `PREFILL_BASE + p.len()` (+ top-8 logits)
///   * `correctness_freerun(p,n)`-> `[PREFILL_BASE, PREFILL_BASE+1, ... n-1]`
pub struct MockEngine {
    serial: u64,
    log: MockLog,
    config: MockConfig,
    /// The open free-run phase's resolved route and depth, set by
    /// `free_decode_begin`. `None` until then, so a `free_decode_run` with no
    /// opener is a fault rather than a silent serial run.
    free_run: Option<(Route, i64)>,
    /// The last committed token, so the committed ids continue rather than
    /// restart.
    last_token: i64,
}

impl MockEngine {
    /// Deterministic descending top-8 logits with the canonical (lowest-id)
    /// tie-break satisfied by construction.
    fn top_logits(token: i64) -> Vec<CorrectnessTraceLogit> {
        (0..TOP_LOGITS_K as i64)
            .map(|i| CorrectnessTraceLogit::new(token + i, 10.0 - i as f64))
            .collect()
    }

    fn record(&self, method: Method, input: i64) -> Result<(), EngineError> {
        self.log.push(Event::Forward(self.serial, method, input));
        if self.config.fault_on_input == Some(input) {
            return Err(EngineError::Fault(format!(
                "mock forced fault on input token {input}"
            )));
        }
        Ok(())
    }
}

impl Drop for MockEngine {
    fn drop(&mut self) {
        self.log.push(Event::Dropped(self.serial));
    }
}

impl Engine for MockEngine {
    fn drain_to_zero(&mut self) -> Result<u64, EngineError> {
        self.log.push(Event::Drained(self.serial));
        Ok(self.config.drain_residual)
    }

    fn prefill(&mut self, prompt: &[i64]) -> Result<i64, EngineError> {
        self.record(Method::Prefill, prompt.len() as i64)?;
        Ok(PREFILL_BASE + prompt.len() as i64)
    }

    fn decode_begin(&mut self, seed: &[i64]) -> Result<i64, EngineError> {
        self.record(Method::DecodeBegin, seed.len() as i64)?;
        Ok(SEED_BASE + seed.len() as i64)
    }

    fn step(&mut self, input: i64) -> Result<Step, EngineError> {
        self.record(Method::Step, input)?;
        let token = input.wrapping_add(1);
        Ok(Step {
            token,
            top_logits: Self::top_logits(token),
        })
    }

    fn correctness_begin(&mut self, prompt: &[i64]) -> Result<Step, EngineError> {
        self.record(Method::CorrectnessBegin, prompt.len() as i64)?;
        let token = PREFILL_BASE + prompt.len() as i64;
        Ok(Step {
            token,
            top_logits: Self::top_logits(token),
        })
    }

    fn correctness_freerun(&mut self, prompt: &[i64], steps: i64) -> Result<Vec<i64>, EngineError> {
        self.record(Method::CorrectnessFreeRun, prompt.len() as i64)?;
        Ok((0..steps).map(|i| PREFILL_BASE + i).collect())
    }

    fn free_decode_begin(
        &mut self,
        seed: &[i64],
        route: Route,
        requested_depth: Option<i64>,
    ) -> Result<(i64, EffectiveSpec), EngineError> {
        self.record(Method::FreeDecodeBegin, seed.len() as i64)?;
        let runnable = self
            .config
            .runnable
            .clone()
            .unwrap_or_else(|| vec![Route::Serial, Route::Mtp]);
        if !runnable.contains(&route) {
            return Err(EngineError::UnsupportedMode {
                requested: route.as_str().to_string(),
                runnable: runnable
                    .iter()
                    .map(|r| r.as_str())
                    .collect::<Vec<_>>()
                    .join(", "),
            });
        }
        // AN EXPLICIT DEPTH IS HONOURED OR REFUSED -- never quietly changed.
        // benchd requires the echo of an explicitly requested depth to match
        // the request verbatim, so clamping one would show up there as an echo
        // divergence and cost the leg with an opaque message. An ABSENT depth
        // asks the engine to choose, so there is nothing to honour and the
        // engine picks the measured default.
        let depth = match route {
            Route::Serial => 0,
            Route::Mtp => match requested_depth {
                Some(requested) => {
                    if !(MTP_MIN_DEPTH..=MTP_MAX_DEPTH).contains(&requested) {
                        return Err(EngineError::DepthOutOfEnvelope {
                            requested,
                            permitted: format!("{MTP_MIN_DEPTH}...{MTP_MAX_DEPTH}"),
                        });
                    }
                    requested
                }
                None => MTP_DEFAULT_DEPTH,
            },
        };
        let seed_token = SEED_BASE + seed.len() as i64;
        self.free_run = Some((route, depth));
        self.last_token = seed_token;
        Ok((
            seed_token,
            EffectiveSpec {
                mode: route.as_str().to_string(),
                mtp: match route {
                    Route::Serial => None,
                    Route::Mtp => Some(EffectiveSpecMTP { depth }),
                },
            },
        ))
    }

    fn free_decode_run(&mut self, count: i64) -> Result<FreeRunResult, EngineError> {
        self.record(Method::FreeDecodeRun, count)?;
        let (route, depth) = self.free_run.ok_or_else(|| {
            EngineError::Fault("free_decode_run with no open free-run phase".into())
        })?;
        // A DETERMINISTIC ROUND SHAPE, chosen so the three invariants are
        // exercised rather than trivially satisfied: the serial leg commits one
        // token per round and drafts nothing; the mtp leg proposes `depth`
        // tokens per round and the target accepts alternating depth / 0, so
        // acceptance lengths vary and `drafted_total > accepted_total`.
        let mut tokens = Vec::new();
        let mut acceptance_lengths = Vec::new();
        let mut drafted_total = 0i64;
        let mut accepted_total = 0i64;
        let mut round = 0i64;
        while (tokens.len() as i64) < count {
            let remaining = count - tokens.len() as i64;
            let committed = match route {
                Route::Serial => 1,
                Route::Mtp => {
                    let accepted = if round % 2 == 0 { depth } else { 0 };
                    drafted_total += depth;
                    accepted_total += accepted;
                    // The verified bonus token always commits, so a round
                    // commits accepted + 1.
                    (accepted + 1).min(remaining)
                }
            };
            let committed = committed.min(remaining);
            for _ in 0..committed {
                self.last_token += 1;
                tokens.push(self.last_token);
            }
            acceptance_lengths.push(committed);
            round += 1;
        }
        Ok(FreeRunResult {
            committed_total: tokens.len() as i64,
            tokens,
            acceptance_lengths,
            drafted_total,
            accepted_total,
            // The mock runs no verify and no replay, so it has no two argmaxes
            // to compare. NOT REPORTED, which is not the claim `0` makes.
            verify_replay_disagreements: None,
        })
    }

    fn expert_stats(&self) -> ExpertStreamingStats {
        ExpertStreamingStats::zero()
    }

    fn peak_ram_gb(&self) -> f64 {
        MOCK_PEAK_RAM_GB
    }
}

/// Factory that mints deterministic [`MockEngine`]s and records each mint in the
/// shared [`MockLog`]. This is what proves fresh-engine-per-phase.
pub struct MockFactory {
    log: MockLog,
    config: MockConfig,
}

impl MockFactory {
    /// A well-behaved factory plus a handle to its lifecycle log.
    pub fn new() -> (Self, MockLog) {
        Self::with_config(MockConfig::default())
    }

    /// A factory whose engines carry `config` (for negative tests).
    pub fn with_config(config: MockConfig) -> (Self, MockLog) {
        let log = MockLog::new();
        (
            MockFactory {
                log: log.clone(),
                config,
            },
            log,
        )
    }
}

impl crate::engine::EngineFactory for MockFactory {
    fn create(&self) -> Box<dyn Engine> {
        let serial = self.log.next_serial();
        self.log.push(Event::Created(serial));
        Box::new(MockEngine {
            serial,
            log: self.log.clone(),
            config: self.config.clone(),
            free_run: None,
            last_token: 0,
        })
    }
}
