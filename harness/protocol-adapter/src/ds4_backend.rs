//! The ds4 backend: the real engine behind `cuda-engine` on the box.
//!
//! The engine is the ds4 port (`Layr-Labs/ds4`) in the `ds4/` submodule, linked
//! in-process through `ds4_shim/` (see `build.rs`). One process serves
//! one benchd phase: the model is opened BEFORE the startup `hello`, so the
//! load never sits inside a timed window, and every verb then drives the same
//! session.
//!
//! Verb translation, in the engine's own terms (`ds4/ds4.h`):
//!
//! * `prefill(prompt)` / `decode_begin(seed)` / `free_decode_begin(seed, ..)`
//!   -> `ds4_session_sync(prompt)`; the next token is the argmax of the logits
//!   the sync leaves behind;
//! * `step(input)` -> `ds4_session_eval(input)` then argmax + top-8 logits;
//!   `correctness_step` and `decode_step` share it;
//! * `free_decode_run(count)` -> on the SERIAL route, `count` plain evals; on
//!   the MTP route, `ds4_session_eval_speculative_argmax` cycles until `count`
//!   tokens are committed. Each cycle commits its input token and, when the
//!   MTP draft matches the target's own argmax, the draft as well. The
//!   target argmax decides every token, so the stream is the serial stream.
//!
//! The counters are the engine's own speculation counters read before and
//! after the phase (`ds4_session_qwen4exp_spec_counters`): `drafts` (rounds
//! that carried a draft into a verify), `hits` (drafts the target accepted),
//! `quenches`, and the diagnostic `verify_replay_disagreements`. Nothing here
//! computes a rate, and nothing here enforces a ceiling on the last one.
//!
//! DEPTH. The pinned engine drafts UP TO THREE tokens per cycle, which is the
//! whole track envelope, so an explicit 1, 2 or 3 all run. A request outside
//! 1..3 is still refused by name ([`EngineError::DepthOutOfEnvelope`]) rather
//! than clamped and echoed wrong. [`DS4_IMPLEMENTED_DEPTH`] tracks the engine's
//! own DS4_QWEN4EXP_IMPLEMENTED_DEPTH and moves with a vendor-sync, not on its
//! own: the two must agree or the adapter refuses depths the engine serves.

use std::sync::{Arc, Mutex};

use crate::engine::{Engine, EngineError, EngineFactory, FreeRunResult, Route, Step};
use crate::protocol::{
    CorrectnessTraceLogit, EffectiveSpec, EffectiveSpecMTP, ExpertStreamingStats, TOP_LOGITS_K,
};

/// The lowest depth the track envelope names (`permitted_draft_depths`).
pub const MTP_MIN_DEPTH: i64 = 1;
/// The highest depth the track envelope names.
pub const MTP_MAX_DEPTH: i64 = 6;
/// The depth the pinned engine implements. MUST equal the vendored engine's
/// `DS4_QWEN4EXP_IMPLEMENTED_DEPTH` (ds4/ds4_qwen4exp_mtp.h): lower and the
/// adapter refuses depths the engine serves, higher and it accepts depths the
/// engine refuses at open. It moved 1 -> 3 with the e2f86b7 vendor-sync and 3 -> 6 for the depth-6 cap.
pub const DS4_IMPLEMENTED_DEPTH: i64 = 6;

/// The speculative cycle's counters, as the engine reports them. Read before
/// and after a leg and take the difference; nothing here computes a rate.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SpecCounters {
    /// Rounds that carried a draft into a verify (the port's `drafted`).
    pub drafts: u64,
    /// Drafts the target's own argmax confirmed (the port's `accepted`).
    pub hits: u64,
    /// Adaptive-disable events. 0 on this path; the tripwire below stays armed.
    pub quenches: u64,
    /// DIAGNOSTIC (the port's `verify_replay_disagreements`): rejecting rounds
    /// where the batched verify's row-0 argmax and the one-row replay of the
    /// same position differed. The replay stands, so this is never a refusal
    /// and nothing here enforces a ceiling; benchd may seal one later.
    pub verify_replay_disagreements: u64,
}

/// The session seam. [`FfiSession`] is the real engine; tests inject a scripted
/// one, so the verb translation and the counter accounting are proven with no
/// GPU. It is the ONLY thing that touches the engine.
pub trait Ds4Session: Send {
    /// Whether the engine was opened with the MTP drafter armed.
    fn mtp_armed(&self) -> bool;
    /// Synchronize the session to `tokens`; logits for the last position follow.
    fn sync(&mut self, tokens: &[i64]) -> Result<(), String>;
    /// Append and evaluate one token; logits for the new position follow.
    fn eval(&mut self, token: i64) -> Result<(), String>;
    /// The greedy token of the current logits.
    fn argmax(&mut self) -> i64;
    /// The top-k logits of the current position, highest first.
    fn top_logits(&mut self, k: usize) -> Vec<CorrectnessTraceLogit>;
    /// One speculative cycle on `first_token` with `budget` tokens still wanted.
    /// Returns the committed tokens (`[first_token]` or `[first_token, draft]`).
    fn eval_speculative(&mut self, first_token: i64, budget: i64) -> Result<Vec<i64>, String>;
    /// The speculative cycle's counters. `&mut` because a session that reaches
    /// the engine over a socket has to send a request to read them.
    fn spec_counters(&mut self) -> SpecCounters;
    /// The model's end-of-sequence token id, when the engine knows one.
    fn eos_token(&self) -> Option<i64> {
        None
    }
    /// Drop the live prefix so the next `sync` forwards the whole prompt.
    fn invalidate(&mut self) {}
    /// DIAGNOSTIC: the engine's packed profile words (six 64-bit words and two
    /// doubles), or `None` when the session has none to report.
    fn profile_words(&mut self) -> Option<([u64; 6], [f64; 2])> {
        None
    }
}

/// The end-of-sequence ids the correctness free-run reference stops on:
/// `DS4_EOS_IDS` when set, else the model's own EOS plus Qwen's two stop ids.
fn eos_ids(engine_eos: Option<i64>) -> Vec<i64> {
    let from_env = std::env::var("DS4_EOS_IDS")
        .ok()
        .map(|s| {
            s.split(',')
                .filter_map(|p| p.trim().parse::<i64>().ok())
                .collect::<Vec<_>>()
        })
        .filter(|v| !v.is_empty());
    let mut ids = from_env.unwrap_or_else(|| vec![248046, 248044]);
    if let Some(eos) = engine_eos {
        if !ids.contains(&eos) {
            ids.push(eos);
        }
    }
    ids
}

/// ds4-backed engine for ONE phase. Cheap to mint: the session it borrows was
/// opened once at process start.
pub struct Ds4Engine {
    session: Arc<Mutex<dyn Ds4Session>>,
    /// The open free-run phase's resolved route + depth and the token the next
    /// cycle feeds. `None` until `free_decode_begin`.
    free_run: Option<FreeRun>,
}

struct FreeRun {
    route: Route,
    pending: i64,
}

impl Ds4Engine {
    pub fn new(session: Arc<Mutex<dyn Ds4Session>>) -> Self {
        Self {
            session,
            free_run: None,
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, dyn Ds4Session + 'static> {
        self.session
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn fault<T>(result: Result<T, String>) -> Result<T, EngineError> {
        result.map_err(EngineError::Fault)
    }

    /// Resolve a requested route + depth against the envelope AND against what
    /// the vendored engine implements. `Serial` is always runnable. `Mtp` needs
    /// the drafter armed at open (`DS4_MTP_DRAFT_TOKENS >= 2`), and an explicit
    /// depth above [`DS4_IMPLEMENTED_DEPTH`] is refused by name.
    pub(crate) fn resolve(
        mtp_armed: bool,
        route: Route,
        requested_depth: Option<i64>,
    ) -> Result<i64, EngineError> {
        match route {
            Route::Serial => Ok(0),
            Route::Mtp => {
                if !mtp_armed {
                    return Err(EngineError::UnsupportedMode {
                        requested: "mtp".to_string(),
                        runnable: "serial (the engine was opened without the MTP drafter; \
                                   set DS4_MTP_DRAFT_TOKENS=2)"
                            .to_string(),
                    });
                }
                let depth = requested_depth.unwrap_or(DS4_IMPLEMENTED_DEPTH);
                if !(MTP_MIN_DEPTH..=MTP_MAX_DEPTH).contains(&depth) {
                    return Err(EngineError::DepthOutOfEnvelope {
                        requested: depth,
                        permitted: format!("{MTP_MIN_DEPTH}...{MTP_MAX_DEPTH}"),
                    });
                }
                if depth > DS4_IMPLEMENTED_DEPTH {
                    return Err(EngineError::DepthOutOfEnvelope {
                        requested: depth,
                        permitted: format!(
                            "{MTP_MIN_DEPTH}...{DS4_IMPLEMENTED_DEPTH} on this engine (the pinned \
                             ds4 drafts up to {DS4_IMPLEMENTED_DEPTH} tokens per cycle)"
                        ),
                    });
                }
                Ok(depth)
            }
        }
    }

    fn step_at_frontier(session: &mut dyn Ds4Session) -> Result<Step, EngineError> {
        let token = session.argmax();
        let top_logits = session.top_logits(TOP_LOGITS_K);
        if top_logits.len() != TOP_LOGITS_K {
            return Err(EngineError::Fault(format!(
                "engine returned {} top logits, the gate needs exactly {TOP_LOGITS_K}",
                top_logits.len()
            )));
        }
        Ok(Step { token, top_logits })
    }
}

impl Engine for Ds4Engine {
    fn drain_to_zero(&mut self) -> Result<u64, EngineError> {
        // One process per phase, so there is no allocator to drain. What a
        // phase start must guarantee is that its first sync does real work:
        // the engine's sync is free when the live prefix already equals the
        // prompt, so drop that prefix here.
        self.lock().invalidate();
        Ok(0)
    }

    fn prefill(&mut self, prompt: &[i64]) -> Result<i64, EngineError> {
        let mut s = self.lock();
        Self::fault(s.sync(prompt))?;
        Ok(s.argmax())
    }

    fn decode_begin(&mut self, seed: &[i64]) -> Result<i64, EngineError> {
        let mut s = self.lock();
        Self::fault(s.sync(seed))?;
        Ok(s.argmax())
    }

    fn step(&mut self, input: i64) -> Result<Step, EngineError> {
        // TEACHER FORCING: the caller's token is evaluated; the model's own
        // choice is returned but never fed. Shared by decode_step AND
        // correctness_step; do NOT split.
        let mut s = self.lock();
        Self::fault(s.eval(input))?;
        Self::step_at_frontier(&mut *s)
    }

    fn correctness_begin(&mut self, prompt: &[i64]) -> Result<Step, EngineError> {
        let mut s = self.lock();
        Self::fault(s.sync(prompt))?;
        Self::step_at_frontier(&mut *s)
    }

    fn correctness_freerun(&mut self, prompt: &[i64], steps: i64) -> Result<Vec<i64>, EngineError> {
        // Free-run greedy REFERENCE generation: plain serial evals, stopping
        // where the golden stopped (an EOS id), never forcing an exact length.
        let mut s = self.lock();
        let eos = eos_ids(s.eos_token());
        Self::fault(s.sync(prompt))?;
        let mut out = Vec::with_capacity(steps.max(0) as usize);
        let mut next = s.argmax();
        for _ in 0..steps {
            out.push(next);
            if eos.contains(&next) {
                break;
            }
            Self::fault(s.eval(next))?;
            next = s.argmax();
        }
        Ok(out)
    }

    fn free_decode_begin(
        &mut self,
        seed: &[i64],
        route: Route,
        requested_depth: Option<i64>,
    ) -> Result<(i64, EffectiveSpec), EngineError> {
        let mut s = self.lock();
        let depth = Self::resolve(s.mtp_armed(), route, requested_depth)?;
        Self::fault(s.sync(seed))?;
        let seed_token = s.argmax();
        drop(s);
        self.free_run = Some(FreeRun {
            route,
            pending: seed_token,
        });
        let effective = EffectiveSpec {
            mode: route.as_str().to_string(),
            mtp: match route {
                Route::Serial => None,
                Route::Mtp => Some(EffectiveSpecMTP { depth }),
            },
        };
        Ok((seed_token, effective))
    }

    fn free_decode_run(&mut self, count: i64) -> Result<FreeRunResult, EngineError> {
        let (route, mut pending) = match self.free_run.as_ref() {
            Some(f) => (f.route, f.pending),
            None => {
                return Err(EngineError::Fault(
                    "free_decode_run with no open free-run phase".into(),
                ))
            }
        };
        if count <= 0 {
            return Err(EngineError::Fault(format!(
                "free_decode_run count {count} is not positive"
            )));
        }
        let mut tokens: Vec<i64> = Vec::with_capacity(count as usize);
        let mut acceptance_lengths: Vec<i64> = Vec::new();
        let (drafted_total, accepted_total, verify_replay_disagreements) = {
            let mut s = self.lock();
            match route {
                Route::Serial => {
                    for _ in 0..count {
                        Self::fault(s.eval(pending))?;
                        pending = s.argmax();
                        tokens.push(pending);
                        acceptance_lengths.push(1);
                    }
                    // The serial route reads no engine counters at all, so it
                    // reports no disagreement count rather than reporting one
                    // it never measured.
                    (0, 0, None)
                }
                Route::Mtp => {
                    let before = s.spec_counters();
                    while (tokens.len() as i64) < count {
                        let budget = count - tokens.len() as i64;
                        let committed = Self::fault(s.eval_speculative(pending, budget))?;
                        if committed.first() != Some(&pending) {
                            return Err(EngineError::Fault(format!(
                                "ds4 speculative cycle committed {committed:?}, expected it to \
                                 start with the fed token {pending}"
                            )));
                        }
                        // The cycle's outputs are every committed token after
                        // the fed one, plus the frontier argmax it left behind.
                        // The cycle contributes every committed token after
                        // the fed token, followed by the frontier argmax. That
                        // is exactly `committed.len()` outputs. Extend the
                        // destination directly instead of allocating and
                        // copying a second temporary Vec on every MTP round.
                        let produced_len = committed.len();
                        if (tokens.len() + produced_len) as i64 > count {
                            return Err(EngineError::Fault(format!(
                                "ds4 speculative cycle produced {} tokens with only {budget} wanted",
                                produced_len
                            )));
                        }
                        tokens.extend_from_slice(&committed[1..]);
                        pending = s.argmax();
                        tokens.push(pending);
                        acceptance_lengths.push(produced_len as i64);
                    }
                    let after = s.spec_counters();
                    let drafted = after.drafts.saturating_sub(before.drafts) as i64;
                    let accepted = after.hits.saturating_sub(before.hits) as i64;
                    let disagreements = after
                        .verify_replay_disagreements
                        .saturating_sub(before.verify_replay_disagreements)
                        as i64;
                    // The leg must run its declared configuration end to end.
                    // A quench, or a drafter that proposed nothing over a leg
                    // long enough to draft (the first cycle is the engine's
                    // plain baseline measurement), is a fault, not a result.
                    // The pinned port enters no adaptive disable and refuses
                    // a non-zero DS4_QWEN_MTP_QUENCH at open, so the shim's
                    // counter never moves; the tripwire stays armed for an
                    // engine that does report one.
                    if after.quenches > before.quenches {
                        return Err(EngineError::Fault(
                            "the engine quenched speculation during the mtp leg".into(),
                        ));
                    }
                    if drafted == 0 && count >= 3 {
                        return Err(EngineError::Fault(format!(
                            "the mtp drafter proposed nothing over {count} tokens; the engine \
                             disabled its MTP block (see stderr) or the drafter is not armed"
                        )));
                    }
                    (drafted, accepted, Some(disagreements))
                }
            }
        };
        if let Some(f) = self.free_run.as_mut() {
            f.pending = pending;
        }
        Ok(FreeRunResult {
            committed_total: tokens.len() as i64,
            tokens,
            acceptance_lengths,
            drafted_total,
            accepted_total,
            verify_replay_disagreements,
        })
    }

    /// The audit-only RSS figure, reported exactly as the unprobed engine
    /// reports it.  The first probe overrode this on a free-run phase; it is
    /// left alone here so no RAM audit can see an engine-authored number.
    fn peak_ram_gb(&self) -> f64 {
        peak_rss_gb().unwrap_or(0.0)
    }

    /// DIAGNOSTIC PROBE.  The dense runtime has no expert streaming, so these
    /// six counters are otherwise the zero struct.  They carry the engine's
    /// packed profile words instead, so the sealed artifact carries the on-box
    /// split.
    ///
    /// The FIRST probe gated this on `free_run.is_some()`, and every counter
    /// came back zero: this track's decode window is a speculative decode
    /// phase, never the correctness free-run, so the gate never opened. The
    /// gate is gone. The engine's counters are CUMULATIVE and split by forward
    /// width class, and the reported class is the two-row class -- the
    /// speculative verify, which exists only inside the decode window -- so an
    /// ungated snapshot is still an attribution of decode alone, whichever
    /// phase close the benchmarker happens to seal.
    fn expert_stats(&self) -> ExpertStreamingStats {
        let mut s = self.lock();
        match s.profile_words() {
            Some((w, f)) => ExpertStreamingStats {
                cache_hits: w[0],
                cache_misses: w[1],
                cache_evictions: w[2],
                bytes_read: w[3],
                read_seconds: f[0],
                peak_cached_tensors: w[4],
            },
            None => ExpertStreamingStats::zero(),
        }
    }
}

/// Peak resident set size of this process in GB, from `/proc/self/status`
/// (`VmHWM`). Audit only; benchd never scores it.
fn peak_rss_gb() -> Option<f64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|l| l.starts_with("VmHWM:"))?;
    let kib: f64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kib * 1024.0 / 1e9)
}

// =========================================================================== //
// The factory
// =========================================================================== //

/// What the box exports for the real engine.
#[derive(Debug, Clone)]
pub struct Ds4Config {
    /// The first GGUF shard of the pinned target (`DS4_MODEL`). The engine
    /// resolves the remaining shards from the path embedded in the GGUF.
    pub model_path: String,
    /// The native MTP draft head GGUF (`DS4_MTP_PATH`), or `None` for a serve
    /// with no draft head. ds4 takes the head as a SEPARATE model.
    pub mtp_head_path: Option<String>,
    /// ds4 `mtp_draft_tokens`: 1 = serial, 2 = one MTP draft per cycle
    /// (`DS4_MTP_DRAFT_TOKENS`, default 1).
    pub mtp_draft_tokens: i32,
    /// Session context size in tokens (`DS4_CTX_SIZE`, default 8192).
    pub ctx_size: i32,
    /// Host threads (`DS4_THREADS`, default 0 = engine default).
    pub n_threads: i32,
}

impl Ds4Config {
    pub fn from_env() -> Result<Self, String> {
        let model_path = std::env::var("DS4_MODEL")
            .ok()
            .filter(|s| !s.is_empty())
            .ok_or_else(|| {
                "DS4_MODEL is unset; export the path of the first GGUF shard of the pinned target"
                    .to_string()
            })?;
        let int_env = |name: &str, default: i32| -> Result<i32, String> {
            match std::env::var(name) {
                Ok(v) if !v.is_empty() => v
                    .trim()
                    .parse::<i32>()
                    .map_err(|e| format!("{name}={v:?} is not an integer: {e}")),
                _ => Ok(default),
            }
        };
        let mtp_draft_tokens = int_env("DS4_MTP_DRAFT_TOKENS", 1)?;
        if !(1..=16).contains(&mtp_draft_tokens) {
            return Err(format!(
                "DS4_MTP_DRAFT_TOKENS={mtp_draft_tokens} is outside 1..16 (1 = serial, 2 = one draft)"
            ));
        }
        let ctx_size = int_env("DS4_CTX_SIZE", 8192)?;
        if ctx_size < 256 {
            return Err(format!("DS4_CTX_SIZE={ctx_size} is too small"));
        }
        Ok(Self {
            model_path,
            mtp_head_path: std::env::var("DS4_MTP_PATH").ok().filter(|s| !s.is_empty()),
            mtp_draft_tokens,
            ctx_size,
            n_threads: int_env("DS4_THREADS", 0)?,
        })
    }
}

/// Mints one [`Ds4Engine`] per phase over the session opened at startup.
pub struct Ds4Factory {
    session: Arc<Mutex<dyn Ds4Session>>,
}

impl Ds4Factory {
    pub fn with_session(session: Arc<Mutex<dyn Ds4Session>>) -> Self {
        Self { session }
    }

    /// Open the real engine from the environment IN THIS PROCESS. This LOADS
    /// THE MODEL, so it runs before the startup hello and outside every timed
    /// window.
    ///
    /// It is NOT the scored path any more. benchd spawns a worker per phase,
    /// so an in-process load is a load per phase; the scored serve boots one
    /// `ds4-resident` per window and every worker uses [`Self::from_resident`]
    /// instead. This stays for a single-shot diagnostic run with no serve.
    #[cfg(feature = "ds4-engine")]
    pub fn from_env() -> Result<Self, String> {
        let config = Ds4Config::from_env()?;
        let session = ffi::FfiSession::open(&config)?;
        Ok(Self::with_session(Arc::new(Mutex::new(session))))
    }

    /// Attach to the window's resident engine over its Unix socket. NO LOAD
    /// HAPPENS HERE: the weights were loaded once when `tools/serve-up.sh`
    /// booted `ds4-resident`, and this connects to that process. Returns the
    /// factory and the resident's own hello, whose `ident` is the engine
    /// identity benchd records.
    #[cfg(unix)]
    pub fn from_resident(
        socket_path: &str,
    ) -> Result<(Self, crate::resident::ResidentHello, std::time::Duration), String> {
        let session = crate::resident::ResidentSession::connect(socket_path)?;
        let hello = session.hello().clone();
        let connect = session.connect;
        Ok((
            Self::with_session(Arc::new(Mutex::new(session))),
            hello,
            connect,
        ))
    }
}

impl EngineFactory for Ds4Factory {
    fn create(&self) -> Box<dyn Engine> {
        Box::new(Ds4Engine::new(Arc::clone(&self.session)))
    }
}

// =========================================================================== //
// The FFI session (box build only)
// =========================================================================== //

#[cfg(feature = "ds4-engine")]
mod ffi {
    use std::ffi::{c_char, c_int, c_void, CStr, CString};

    use super::{Ds4Config, Ds4Session, SpecCounters};
    use crate::protocol::CorrectnessTraceLogit;

    extern "C" {
        fn ds4s_open(
            model_path: *const c_char,
            mtp_head_path: *const c_char,
            mtp_draft_tokens: c_int,
            ctx_size: c_int,
            n_threads: c_int,
        ) -> *mut c_void;
        fn ds4s_close(h: *mut c_void);
        fn ds4s_open_error() -> *const c_char;
        fn ds4s_last_error(h: *const c_void) -> *const c_char;
        fn ds4s_sync(h: *mut c_void, tokens: *const i32, n: usize) -> c_int;
        fn ds4s_eval(h: *mut c_void, token: i32) -> c_int;
        fn ds4s_argmax(h: *const c_void) -> i32;
        fn ds4s_top_logits(h: *const c_void, k: c_int, ids: *mut i32, logits: *mut f32) -> c_int;
        fn ds4s_eval_speculative(
            h: *mut c_void,
            first_token: i32,
            budget: c_int,
            out: *mut i32,
            cap: c_int,
        ) -> c_int;
        fn ds4s_spec_counters(
            h: *const c_void,
            drafts: *mut u64,
            hits: *mut u64,
            quenches: *mut u64,
            disagreements: *mut u64,
        );
        fn ds4s_eos_token(h: *const c_void) -> i32;
        fn ds4s_invalidate(h: *mut c_void);
    }

    pub struct FfiSession {
        handle: *mut c_void,
        mtp_armed: bool,
    }

    // The handle is only ever used behind the factory's Mutex.
    unsafe impl Send for FfiSession {}

    impl FfiSession {
        pub fn open(config: &Ds4Config) -> Result<Self, String> {
            let path = CString::new(config.model_path.as_str())
                .map_err(|e| format!("DS4_MODEL contains a NUL byte: {e}"))?;
            let head = match config.mtp_head_path.as_deref() {
                Some(p) => Some(
                    CString::new(p)
                        .map_err(|e| format!("DS4_MTP_PATH contains a NUL byte: {e}"))?,
                ),
                None => None,
            };
            // SAFETY: the C shim copies what it needs from the paths before returning.
            let handle = unsafe {
                ds4s_open(
                    path.as_ptr(),
                    head.as_ref().map_or(std::ptr::null(), |h| h.as_ptr()),
                    config.mtp_draft_tokens,
                    config.ctx_size,
                    config.n_threads,
                )
            };
            if handle.is_null() {
                // SAFETY: the shim returns a NUL-terminated static string.
                let why = unsafe { CStr::from_ptr(ds4s_open_error()) }
                    .to_string_lossy()
                    .into_owned();
                return Err(format!(
                    "ds4 engine open failed for {}: {}",
                    config.model_path,
                    if why.is_empty() {
                        "see stderr for the engine's reason"
                    } else {
                        &why
                    }
                ));
            }
            Ok(Self {
                handle,
                mtp_armed: config.mtp_draft_tokens >= 2,
            })
        }

        fn last_error(&self) -> String {
            // SAFETY: the shim returns a NUL-terminated string owned by the handle.
            let msg = unsafe { CStr::from_ptr(ds4s_last_error(self.handle)) };
            msg.to_string_lossy().into_owned()
        }

        fn narrow(token: i64) -> Result<i32, String> {
            i32::try_from(token)
                .map_err(|_| format!("token id {token} does not fit the engine's i32"))
        }
    }

    impl Drop for FfiSession {
        fn drop(&mut self) {
            // SAFETY: the handle came from ds4s_open and is closed exactly once.
            unsafe { ds4s_close(self.handle) };
        }
    }

    impl Ds4Session for FfiSession {
        fn mtp_armed(&self) -> bool {
            self.mtp_armed
        }

        fn sync(&mut self, tokens: &[i64]) -> Result<(), String> {
            let narrow: Vec<i32> = tokens
                .iter()
                .map(|&t| Self::narrow(t))
                .collect::<Result<_, _>>()?;
            // SAFETY: the slice outlives the call; the shim copies it.
            let rc = unsafe { ds4s_sync(self.handle, narrow.as_ptr(), narrow.len()) };
            if rc != 0 {
                return Err(format!("ds4 sync failed: {}", self.last_error()));
            }
            Ok(())
        }

        fn eval(&mut self, token: i64) -> Result<(), String> {
            // SAFETY: plain call on a live handle.
            let rc = unsafe { ds4s_eval(self.handle, Self::narrow(token)?) };
            if rc != 0 {
                return Err(format!("ds4 eval failed: {}", self.last_error()));
            }
            Ok(())
        }

        fn argmax(&mut self) -> i64 {
            // SAFETY: plain call on a live handle.
            unsafe { ds4s_argmax(self.handle) as i64 }
        }

        fn top_logits(&mut self, k: usize) -> Vec<CorrectnessTraceLogit> {
            let mut ids = vec![0i32; k];
            let mut logits = vec![0f32; k];
            // SAFETY: both buffers hold `k` elements, the cap passed to the shim.
            let n = unsafe {
                ds4s_top_logits(
                    self.handle,
                    k as c_int,
                    ids.as_mut_ptr(),
                    logits.as_mut_ptr(),
                )
            };
            let n = n.clamp(0, k as c_int) as usize;
            ids[..n]
                .iter()
                .zip(&logits[..n])
                .map(|(&id, &logit)| CorrectnessTraceLogit::new(id as i64, logit as f64))
                .collect()
        }

        fn eval_speculative(&mut self, first_token: i64, budget: i64) -> Result<Vec<i64>, String> {
            let mut out = [0i32; 17];
            let budget = c_int::try_from(budget.max(1)).unwrap_or(c_int::MAX);
            // SAFETY: `out` holds 17 elements, the cap passed to the shim.
            let n = unsafe {
                ds4s_eval_speculative(
                    self.handle,
                    Self::narrow(first_token)?,
                    budget,
                    out.as_mut_ptr(),
                    out.len() as c_int,
                )
            };
            if n < 0 {
                return Err(format!(
                    "ds4 speculative cycle failed: {}",
                    self.last_error()
                ));
            }
            Ok(out[..n as usize].iter().map(|&t| t as i64).collect())
        }

        fn spec_counters(&mut self) -> SpecCounters {
            let mut c = SpecCounters::default();
            // SAFETY: a live handle, and four out-pointers valid for the call.
            unsafe {
                ds4s_spec_counters(
                    self.handle,
                    &mut c.drafts,
                    &mut c.hits,
                    &mut c.quenches,
                    &mut c.verify_replay_disagreements,
                )
            };
            c
        }

        fn eos_token(&self) -> Option<i64> {
            // SAFETY: plain call on a live handle.
            let eos = unsafe { ds4s_eos_token(self.handle) };
            (eos >= 0).then_some(eos as i64)
        }

        fn invalidate(&mut self) {
            // SAFETY: plain call on a live handle.
            unsafe { ds4s_invalidate(self.handle) };
        }
    }
}
