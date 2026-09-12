//! The client half of the resident-server topology: a [`Ds4Session`] that
//! lives in the per-phase `cuda-engine` process and drives the model held by
//! `ds4-resident`.
//!
//! ## WHY THERE IS A SOCKET AT ALL
//!
//! benchd's CUDA residency is `FreshPerPhase`: warmup, timed prefill, timed
//! decode and correctness each get a fresh worker process. With the engine
//! linked in-process every one of those loaded the checkpoint again. The
//! weights now live in ONE process for the whole window
//! (`tools/serve-up.sh` boots it under the GPU lock) and each phase's worker
//! connects to it. WEIGHTS LOAD ONCE; a phase costs a connect.
//!
//! ## THE WIRE IS THE C SURFACE, ONE VERB PER LINE
//!
//! NDJSON over `AF_UNIX`. Every verb is one `ds4_shim.h` function, so nothing
//! is mapped, approximated or reconstructed on the way across
//! (`docs/ds4-resident.md` carries the table and the evidence for why upstream
//! `ds4-server`'s chat wire could not carry it).
//!
//! ## ONE ROUND TRIP PER DECODED TOKEN
//!
//! Every state-advancing verb replies with the frontier argmax, which
//! [`ResidentSession`] caches, so [`Ds4Session::argmax`] costs nothing. That
//! matters because `free_decode_run`'s loop sits INSIDE benchd's timed window:
//! a decoded token is one `eval` (or one `eval_speculative`) and no more. The
//! argmax is a pure read of logits the same call left ready, so returning it
//! eagerly changes no semantics.
//!
//! ## FAIL-CLOSED
//!
//! [`Ds4Session`] has infallible methods (`argmax`, `top_logits`,
//! `spec_counters`, `invalidate`) which cannot report a dead socket. A failure
//! in one of those POISONS the session: it is remembered and returned by the
//! next fallible verb, and meanwhile `top_logits` returns nothing, which the
//! backend already treats as a fault. A broken resident can therefore never
//! read as a fast phase.

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use crate::ds4_backend::{Ds4Session, SpecCounters};
use crate::protocol::CorrectnessTraceLogit;

/// The environment variable `tools/serve-up.sh` exports with the resident
/// server's socket. Its presence is what selects this backend.
pub const SOCKET_ENV: &str = "DS4_RESIDENT_SOCKET";

/// What a worker attached to the resident announces as its backend. It names
/// the TOPOLOGY, not just the engine build: the process that holds the weights
/// is the resident, and this worker loaded nothing.
pub const RESIDENT_BACKEND: &str = "ds4-resident";

/// How long a single verb may take before the client gives up on the resident.
/// Generous: one `sync` of a long prompt is a real prefill on the box. The
/// resident has its own, shorter, idle ceiling in the other direction.
const VERB_TIMEOUT: Duration = Duration::from_secs(1800);

/// What the resident answered `hello` with: the identity of the process that
/// actually holds the weights.
#[derive(Debug, Clone, PartialEq)]
pub struct ResidentHello {
    pub vocab_size: i64,
    pub eos_token: i64,
    pub mtp_armed: bool,
    pub draft_tokens: i64,
    pub ctx_size: i64,
    /// The resident's pid; the same value across every phase of one window, so
    /// an artifact can show the load was not repeated.
    pub load_epoch: i64,
    pub ident: String,
    /// DIAGNOSTIC: the engine's cumulative profile text at hello time (empty
    /// unless the resident armed its profiling switches).
    pub profile: String,
    pub model_path: String,
    pub mtp_head_path: String,
}

impl ResidentHello {
    /// The backend string the attached worker announces in its OWN hello, which
    /// benchd seals as `engine_backend` (`crates/bench-runner/src/session.rs`
    /// reads `hello.backend`; `crates/benchctl/src/official.rs`
    /// `seal_engine_identity` copies it into the score).
    ///
    /// It carries three things a reader of a sealed artifact needs: the
    /// topology (`ds4-resident`), the `load_epoch`, and the resident's own
    /// engine identity. THE `load_epoch` IS THE ONE-LOAD RECORD, NOT A PROOF.
    /// It is the resident's pid, composed by the worker and reported by it:
    /// every phase of one window seals the same value, so the artifact records
    /// that this serve booted no second load, and a differing value across
    /// phases would show one. It is PROVENANCE the worker states about itself
    /// -- it checks nothing independently, and an assertion that must hold
    /// against a hostile worker needs a supervisor observation instead. Before
    /// this, even the record lived only in the resident log and
    /// `serve-identity.json`, neither of which the score carries.
    pub fn backend_string(&self) -> String {
        if self.profile.is_empty() {
            format!(
                "{RESIDENT_BACKEND} load_epoch={} {}",
                self.load_epoch, self.ident
            )
        } else {
            format!(
                "{RESIDENT_BACKEND} load_epoch={} {} prof[{}]",
                self.load_epoch, self.ident, self.profile
            )
        }
    }
}

/// One phase's connection to the resident server.
pub struct ResidentSession {
    reader: BufReader<UnixStream>,
    writer: UnixStream,
    hello: ResidentHello,
    /// The argmax of the current frontier, carried by the last state-advancing
    /// reply. `None` only before the first `sync`.
    frontier: Option<i64>,
    /// The first failure seen by an infallible verb, returned by the next
    /// fallible one.
    poison: Option<String>,
    /// Connect + `hello`, measured by the client. Milliseconds, not seconds,
    /// is the whole claim of this topology, so it is reported rather than
    /// asserted from the outside.
    pub connect: Duration,
}

impl ResidentSession {
    /// Connect to the resident at `path` and complete the `hello` handshake.
    pub fn connect(path: &str) -> Result<Self, String> {
        let started = Instant::now();
        let stream = UnixStream::connect(path)
            .map_err(|e| format!("cannot reach the resident engine at {path}: {e}"))?;
        stream
            .set_read_timeout(Some(VERB_TIMEOUT))
            .and_then(|()| stream.set_write_timeout(Some(VERB_TIMEOUT)))
            .map_err(|e| format!("cannot set the resident socket timeouts: {e}"))?;
        let writer = stream
            .try_clone()
            .map_err(|e| format!("cannot clone the resident socket: {e}"))?;
        let mut session = Self {
            reader: BufReader::new(stream),
            writer,
            hello: ResidentHello {
                vocab_size: 0,
                eos_token: -1,
                mtp_armed: false,
                draft_tokens: 1,
                ctx_size: 0,
                load_epoch: 0,
                ident: String::new(),
                profile: String::new(),
                model_path: String::new(),
                mtp_head_path: String::new(),
            },
            frontier: None,
            poison: None,
            connect: Duration::ZERO,
        };
        let reply = session.call(json!({"op": "hello"}))?;
        session.hello = ResidentHello {
            vocab_size: int_field(&reply, "vocab_size")?,
            eos_token: int_field(&reply, "eos_token")?,
            mtp_armed: reply
                .get("mtp_armed")
                .and_then(Value::as_bool)
                .ok_or_else(|| {
                    "the resident's hello carries no boolean \"mtp_armed\"".to_string()
                })?,
            draft_tokens: int_field(&reply, "draft_tokens")?,
            ctx_size: int_field(&reply, "ctx_size")?,
            load_epoch: int_field(&reply, "load_epoch")?,
            ident: string_field(&reply, "ident")?,
            profile: reply
                .get("profile")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
            model_path: string_field(&reply, "model_path")?,
            mtp_head_path: string_field(&reply, "mtp_head_path")?,
        };
        session.connect = started.elapsed();
        Ok(session)
    }

    /// What the resident said it is serving.
    pub fn hello(&self) -> &ResidentHello {
        &self.hello
    }

    /// Send one request line and read one reply line.
    fn call(&mut self, request: Value) -> Result<Value, String> {
        if let Some(poison) = &self.poison {
            return Err(poison.clone());
        }
        match self.call_inner(request) {
            Ok(reply) => Ok(reply),
            Err(err) => {
                self.poison.get_or_insert(err.clone());
                Err(err)
            }
        }
    }

    fn call_inner(&mut self, request: Value) -> Result<Value, String> {
        let mut line = serde_json::to_string(&request)
            .map_err(|e| format!("cannot serialize a resident request: {e}"))?;
        line.push('\n');
        self.writer
            .write_all(line.as_bytes())
            .and_then(|()| self.writer.flush())
            .map_err(|e| {
                format!("the resident engine closed while a request was going out: {e}")
            })?;
        let mut reply = String::new();
        let read = self
            .reader
            .read_line(&mut reply)
            .map_err(|e| format!("cannot read the resident engine's reply: {e}"))?;
        if read == 0 {
            return Err("the resident engine closed the connection mid-phase".to_string());
        }
        let value: Value = serde_json::from_str(reply.trim_end())
            .map_err(|e| format!("the resident engine sent a line that is not JSON: {e}"))?;
        if value.get("ok").and_then(Value::as_bool) != Some(true) {
            let why = value
                .get("error")
                .and_then(Value::as_str)
                .unwrap_or("the resident engine refused without a reason");
            return Err(format!("resident engine: {why}"));
        }
        Ok(value)
    }

    /// Take the frontier argmax a state-advancing reply carried.
    fn take_frontier(&mut self, reply: &Value) -> Result<(), String> {
        self.frontier = Some(int_field(reply, "token")?);
        Ok(())
    }

    /// Record a failure an infallible verb cannot return.
    fn poison_with(&mut self, err: String) {
        self.poison.get_or_insert(err);
    }
}

fn int_field(value: &Value, key: &str) -> Result<i64, String> {
    value
        .get(key)
        .and_then(Value::as_i64)
        .ok_or_else(|| format!("the resident engine's reply carries no integer \"{key}\""))
}

fn string_field(value: &Value, key: &str) -> Result<String, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| format!("the resident engine's reply carries no string \"{key}\""))
}

impl Ds4Session for ResidentSession {
    fn mtp_armed(&self) -> bool {
        self.hello.mtp_armed
    }

    fn sync(&mut self, tokens: &[i64]) -> Result<(), String> {
        let reply = self.call(json!({"op": "sync", "tokens": tokens}))?;
        self.take_frontier(&reply)
    }

    fn eval(&mut self, token: i64) -> Result<(), String> {
        let reply = self.call(json!({"op": "eval", "token": token}))?;
        self.take_frontier(&reply)
    }

    fn argmax(&mut self) -> i64 {
        // Free: the last state-advancing reply carried it.
        if let Some(token) = self.frontier {
            return token;
        }
        match self.call(json!({"op": "argmax"})) {
            Ok(reply) => match int_field(&reply, "token") {
                Ok(token) => {
                    self.frontier = Some(token);
                    token
                }
                Err(err) => {
                    self.poison_with(err);
                    -1
                }
            },
            Err(err) => {
                self.poison_with(err);
                -1
            }
        }
    }

    fn top_logits(&mut self, k: usize) -> Vec<CorrectnessTraceLogit> {
        let reply = match self.call(json!({"op": "top_logits", "k": k})) {
            Ok(reply) => reply,
            Err(err) => {
                self.poison_with(err);
                // The backend requires exactly TOP_LOGITS_K entries and faults
                // on anything else, so an empty vec is the fail-closed answer.
                return Vec::new();
            }
        };
        let (ids, logits) = match (
            reply.get("ids").and_then(Value::as_array),
            reply.get("logits").and_then(Value::as_array),
        ) {
            (Some(ids), Some(logits)) => (ids.clone(), logits.clone()),
            _ => {
                self.poison_with(
                    "the resident engine's top_logits reply carries no \"ids\"/\"logits\" arrays"
                        .to_string(),
                );
                return Vec::new();
            }
        };
        if ids.len() != logits.len() {
            self.poison_with(format!(
                "the resident engine returned {} ids and {} logits",
                ids.len(),
                logits.len()
            ));
            return Vec::new();
        }
        // THE WIRE PATH MUST WIDEN LIKE THE LINKED PATH. The engine's logits
        // are `float`; the linked backend widens each one `logit as f64`
        // (ds4_backend.rs, FfiSession::top_logits). The resident writes them as
        // `%.9g`, which round-trips a `float` exactly, so the decimal here
        // names a representable f32 -- but parsing it straight into f64 lands
        // on the NEAREST DOUBLE to that decimal, which is not the same number.
        // Narrowing to f32 first recovers the engine's own float, and widening
        // that gives the identical f64 the linked path produces.
        ids.iter()
            .zip(&logits)
            .filter_map(|(id, logit)| Some((id.as_i64()?, logit.as_f64()? as f32 as f64)))
            .map(|(id, logit)| CorrectnessTraceLogit::new(id, logit))
            .collect()
    }

    fn eval_speculative(&mut self, first_token: i64, budget: i64) -> Result<Vec<i64>, String> {
        let reply = self.call(
            json!({"op": "eval_speculative", "first_token": first_token, "budget": budget}),
        )?;
        self.take_frontier(&reply)?;
        let committed = reply
            .get("tokens")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                "the resident engine's speculative reply carries no \"tokens\" array".to_string()
            })?
            .iter()
            .map(|t| {
                t.as_i64()
                    .ok_or_else(|| "the resident engine committed a non-integer token".to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(committed)
    }

    fn spec_counters(&mut self) -> SpecCounters {
        match self.call(json!({"op": "spec_counters"})) {
            Ok(reply) => {
                let read = |key: &str| reply.get(key).and_then(Value::as_u64);
                match (
                    read("drafts"),
                    read("hits"),
                    read("quenches"),
                    read("disagreements"),
                ) {
                    (Some(drafts), Some(hits), Some(quenches), Some(disagreements)) => {
                        SpecCounters {
                            drafts,
                            hits,
                            quenches,
                            verify_replay_disagreements: disagreements,
                        }
                    }
                    _ => {
                        self.poison_with(
                            "the resident engine's counters reply is missing a field".to_string(),
                        );
                        SpecCounters::default()
                    }
                }
            }
            Err(err) => {
                self.poison_with(err);
                SpecCounters::default()
            }
        }
    }

    fn eos_token(&self) -> Option<i64> {
        (self.hello.eos_token >= 0).then_some(self.hello.eos_token)
    }

    fn profile_words(&mut self) -> Option<([u64; 6], [f64; 2])> {
        let reply = self.call(json!({"op": "profile"})).ok()?;
        let w = reply.get("w")?.as_array()?;
        let f = reply.get("f")?.as_array()?;
        if w.len() != 6 || f.len() != 2 {
            return None;
        }
        let mut words = [0u64; 6];
        for (i, v) in w.iter().enumerate() {
            words[i] = v.as_u64()?;
        }
        let mut fl = [0f64; 2];
        for (i, v) in f.iter().enumerate() {
            fl[i] = v.as_f64()?;
        }
        Some((words, fl))
    }

    fn invalidate(&mut self) {
        // The resident invalidates on accept too, so a phase is reset even if
        // this never runs. This is the backend's own drain contract.
        self.frontier = None;
        if let Err(err) = self.call(json!({"op": "invalidate"})) {
            self.poison_with(err);
        }
    }
}

impl Drop for ResidentSession {
    fn drop(&mut self) {
        // Say goodbye so the resident logs a clean phase close and goes back
        // to accepting immediately, rather than waiting for the EOF.
        let _ = self.call_inner(json!({"op": "bye"}));
    }
}
