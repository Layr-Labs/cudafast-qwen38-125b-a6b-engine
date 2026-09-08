//! Adapter tests, driven entirely against the deterministic mock (no GPU).
//!
//! Three layers:
//!  1. BEHAVIOR — message framing/ordering, unsolicited hello, phase-close
//!     barrier + `completed_work`, fresh-engine-per-phase lifecycle, error /
//!     early-EOF / fail-closed handling.
//!  2. WIRE SHAPE — real-shaped requests (`id` + `kind` + `token`), `id` echo,
//!     nonce echo, flat responses carrying NO `kind`.
//!  3. CONFORMANCE — every emitted response line validates against the pinned
//!     copy of the authoritative JSON Schema, with a negative control proving
//!     the validator bites.

use std::io::Cursor;

use serde_json::Value;

use crate::adapter::Adapter;
use crate::mock::{
    Event, Method, MockConfig, MockFactory, MOCK_BACKEND, MOCK_DEVICE, MOCK_PEAK_RAM_GB,
    PREFILL_BASE, SEED_BASE,
};
use crate::protocol::JSON_SCHEMA;

const NONCE: &str = "testnonce";

/// Run the adapter (pinned nonce) over `lines` and return parsed response lines.
fn run(lines: &[&str]) -> Vec<Value> {
    let (factory, _log) = MockFactory::new();
    run_with(Adapter::with_session(factory, "cuda", "cuda", NONCE), lines).0
}

fn run_with<F: crate::engine::EngineFactory>(
    mut adapter: Adapter<F>,
    lines: &[&str],
) -> (Vec<Value>, ()) {
    let input = lines.join("\n");
    let mut out: Vec<u8> = Vec::new();
    adapter
        .run(Cursor::new(input), &mut out)
        .expect("no I/O error");
    (parse_lines(&out), ())
}

fn parse_lines(bytes: &[u8]) -> Vec<Value> {
    String::from_utf8(bytes.to_vec())
        .unwrap()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).expect("each response line is valid JSON"))
        .collect()
}

// ---------------------------------------------------------------------------
// Schema conformance validator — interprets the subset of JSON Schema Draft
// 2020-12 the vendored schema uses ($ref, type, properties, required,
// additionalProperties:false, items, enum, minimum). Validates an emitted
// response against $defs/WorkerResponse.
//
// SCOPE: this is a conformance oracle for the lines THIS adapter EMITS, not a
// general benchd-deserialization oracle. Two fidelity edges exist but are
// unreachable from emitted lines: (a) it does not enforce `minimum`/unsignedness,
// so a hypothetical negative `protocol_version` would pass here yet be rejected by
// bench-protocol's `Option<u32>` serde — the adapter only ever emits `1`; (b) the
// schema marks nested `ExpertStreamingStats`/`CorrectnessTraceLogit` closed, which
// is stricter than those serde structs (no `deny_unknown_fields`) — the adapter
// never emits extra nested keys. Both are noted here should this kit be reused.
// ---------------------------------------------------------------------------

fn schema_root() -> Value {
    serde_json::from_str(JSON_SCHEMA).expect("vendored schema is valid JSON")
}

fn resolve<'a>(root: &'a Value, node: &'a Value) -> &'a Value {
    if let Some(r) = node.get("$ref").and_then(Value::as_str) {
        let mut cur = root;
        for seg in r.trim_start_matches("#/").split('/') {
            cur = &cur[seg];
        }
        cur
    } else {
        node
    }
}

fn validate(root: &Value, schema: &Value, inst: &Value, path: &str) -> Result<(), String> {
    let schema = resolve(root, schema);
    let Some(ty) = schema.get("type").and_then(Value::as_str) else {
        return Ok(());
    };
    match ty {
        "object" => {
            let obj = inst
                .as_object()
                .ok_or_else(|| format!("{path}: expected object"))?;
            if let Some(req) = schema.get("required").and_then(Value::as_array) {
                for r in req {
                    let k = r.as_str().unwrap();
                    if !obj.contains_key(k) {
                        return Err(format!("{path}: missing required '{k}'"));
                    }
                }
            }
            let props = schema.get("properties").and_then(Value::as_object);
            let allow_additional =
                !matches!(schema.get("additionalProperties"), Some(Value::Bool(false)));
            for (k, v) in obj {
                match props.and_then(|p| p.get(k)) {
                    Some(sub) => validate(root, sub, v, &format!("{path}.{k}"))?,
                    None if allow_additional => {}
                    None => return Err(format!("{path}: additional property '{k}' not allowed")),
                }
            }
        }
        "array" => {
            let arr = inst
                .as_array()
                .ok_or_else(|| format!("{path}: expected array"))?;
            if let Some(items) = schema.get("items") {
                for (i, el) in arr.iter().enumerate() {
                    validate(root, items, el, &format!("{path}[{i}]"))?;
                }
            }
        }
        "integer" => {
            if !(inst.is_i64() || inst.is_u64()) {
                return Err(format!("{path}: expected integer, got {inst}"));
            }
        }
        "number" => {
            if !inst.is_number() {
                return Err(format!("{path}: expected number, got {inst}"));
            }
        }
        "string" => {
            let s = inst
                .as_str()
                .ok_or_else(|| format!("{path}: expected string"))?;
            if let Some(en) = schema.get("enum").and_then(Value::as_array) {
                if !en.iter().any(|e| e.as_str() == Some(s)) {
                    return Err(format!("{path}: '{s}' not in enum"));
                }
            }
        }
        "boolean" => {
            if !inst.is_boolean() {
                return Err(format!("{path}: expected boolean"));
            }
        }
        other => return Err(format!("{path}: unhandled schema type {other}")),
    }
    Ok(())
}

fn validate_response(root: &Value, resp: &Value) -> Result<(), String> {
    validate(root, &root["$defs"]["WorkerResponse"], resp, "response")
}

/// The response fields the v1.1 extension adds. The vendored schema is BASE
/// V1 and marks the response envelope `additionalProperties: false`, so it
/// rejects every one of them.
///
/// THE SCHEMA IS NOT EDITED TO ADMIT THEM. It is read-only evidence of the base
/// contract (reference/PROVENANCE.md); a schema this repository amended would
/// stop being evidence of anything. So conformance is checked on the BASE-V1
/// PROJECTION of each response: strip the extension fields, and what remains
/// must still be a valid base-v1 line. That is the property that actually
/// matters -- a v1-only reader must be able to parse what this adapter emits
/// apart from the fields it does not know.
///
/// The extension fields themselves are checked by the free-run tests, against
/// the Swift reference worker's `CodingKeys` (cited in src/protocol.rs).
const V1_1_EXTENSION_FIELDS: &[&str] = &[
    "spec_modes",
    "capabilities",
    "effective_spec",
    "acceptance_lengths",
    "drafted_total",
    "accepted_total",
    "committed_total",
    "verify_replay_disagreements",
];

/// Assert every response line's BASE-V1 PROJECTION conforms to the vendored
/// schema (see [`V1_1_EXTENSION_FIELDS`]).
fn assert_all_conform(resp: &[Value]) {
    let root = schema_root();
    for (i, r) in resp.iter().enumerate() {
        let mut projected = r.clone();
        if let Some(obj) = projected.as_object_mut() {
            for field in V1_1_EXTENSION_FIELDS {
                obj.remove(*field);
            }
        }
        validate_response(&root, &projected)
            .unwrap_or_else(|e| panic!("response[{i}] {r} fails schema: {e}"));
    }
}

fn kind_field(v: &Value) -> Option<&str> {
    v.get("kind").and_then(Value::as_str)
}

// ---------------------------------------------------------------------------
// 1. BEHAVIOR
// ---------------------------------------------------------------------------

#[test]
fn startup_emits_unsolicited_hello_id_zero() {
    let resp = run(&[]);
    assert_eq!(resp.len(), 1, "only the unsolicited hello");
    let hello = &resp[0];
    assert_eq!(hello["id"], 0);
    assert_eq!(hello["ok"], true);
    assert_eq!(hello["nonce"], NONCE);
    assert_eq!(hello["protocol_version"], 1);
    assert_eq!(hello["backend"], "cuda");
    assert_eq!(hello["device"], "cuda");
    assert!(hello.get("expert_stats").is_some());
    // hello is a WorkerResponse, NOT a request kind — it carries no `kind`.
    assert!(kind_field(hello).is_none());

    // THE TWO ADVERTISEMENTS ARE REQUIRED, NOT OPTIONAL, and this assertion
    // used to say the opposite: it asserted `capabilities` was ABSENT, which
    // is base-v1 behaviour and is exactly what makes benchd refuse the scored
    // series. benchd will not issue a free-run verb without
    // `free_run_decode` in `capabilities`, and refuses a spec whose mode is
    // absent from `spec_modes` before the timed seed forward.
    let capabilities = hello["capabilities"]
        .as_array()
        .expect("the hello advertises capabilities");
    assert!(
        capabilities.iter().any(|c| c == "free_run_decode"),
        "the hello must advertise free_run_decode or benchd refuses the free-run verbs: {capabilities:?}"
    );
    let modes: Vec<&str> = hello["spec_modes"]
        .as_array()
        .expect("the hello advertises spec_modes")
        .iter()
        .map(|m| m.as_str().expect("mode is a string"))
        .collect();
    assert_eq!(
        modes,
        vec!["serial", "mtp"],
        "the hello must advertise both legs of the paired measurement"
    );

    // Conformance runs on the BASE-V1 PROJECTION -- assert_all_conform strips
    // the v1.1 fields, because the vendored schema is base v1 and is read-only
    // evidence rather than a design surface.
    assert_all_conform(&resp);
}

/// THE MOCK MUST NAME ITSELF IN THE SEALED IDENTITY.
///
/// benchd seals the hello's `backend`/`device` as `engine_backend` /
/// `engine_device` (`crates/benchctl/src/official.rs`, `seal_engine_identity`).
/// The mock announced `cuda`/`cuda` until 2026-09-03, so a mock artifact was
/// byte-identical to a CUDA one in the only fields that say which engine ran --
/// and on the official path the worker's stderr, where the mock says so in
/// words, is not forwarded. The two strings below are the whole remedy, so they
/// are pinned here, together with the property that carries the point: neither
/// of them names cuda.
#[test]
fn the_mock_announces_itself_in_the_sealed_identity() {
    let (factory, _log) = MockFactory::new();
    let resp = run_with(
        Adapter::with_session(factory, MOCK_BACKEND, MOCK_DEVICE, NONCE),
        &[],
    )
    .0;
    let hello = &resp[0];
    assert_eq!(hello["backend"], "mock");
    assert_eq!(hello["device"], "none");
    for field in ["backend", "device"] {
        let value = hello[field].as_str().expect("the hello carries a string");
        assert!(
            !value.contains("cuda"),
            "the mock's sealed {field} must not read as a CUDA measurement (got {value:?})"
        );
    }
}

#[test]
fn happy_path_decode_phase_completed_work_is_one_plus_n() {
    let n = 4i64;
    let mut lines = vec![r#"{"id":1,"kind":"decode_begin","seed_tokens":[7,8,9]}"#.to_string()];
    for step in 0..n {
        lines.push(format!(
            r#"{{"id":{},"kind":"decode_step","token":{}}}"#,
            step + 2,
            1000 + step
        ));
    }
    lines.push(format!(r#"{{"id":{},"kind":"phase_diagnostics"}}"#, n + 2));
    let refs: Vec<&str> = lines.iter().map(String::as_str).collect();

    let resp = run(&refs);
    assert_all_conform(&resp);

    // hello, decode_begin, N decode_step, barrier
    assert_eq!(resp.len(), 1 + 1 + n as usize + 1);

    let begin = &resp[1];
    assert_eq!(begin["id"], 1);
    assert_eq!(begin["seed_token"], SEED_BASE + 3);
    assert!(
        begin.get("token").is_none(),
        "decode_begin carries seed_token, not token"
    );

    for step in 0..n as usize {
        let r = &resp[2 + step];
        assert_eq!(r["id"], step as i64 + 2);
        // decode_step response is token-only (shared forward, but no top_logits on the wire)
        assert_eq!(r["token"], 1000 + step as i64 + 1);
        assert!(
            r.get("top_logits").is_none(),
            "decode_step response must NOT carry top_logits"
        );
        assert_eq!(r["nonce"], NONCE);
    }

    let barrier = resp.last().unwrap();
    assert_eq!(barrier["completed_work"], 1 + n);
    assert_eq!(barrier["peak_ram_gb"], MOCK_PEAK_RAM_GB);
    assert!(barrier.get("expert_stats").is_some());
}

/// Per is_timed_step, prefill is NOT a timed step: a prefill-only phase reports
/// completed_work == 0.
#[test]
fn prefill_phase_completed_work_is_zero() {
    let resp = run(&[
        r#"{"id":1,"kind":"prefill","prompt_tokens":[1,2,3,4,5]}"#,
        r#"{"id":2,"kind":"phase_diagnostics"}"#,
    ]);
    assert_all_conform(&resp);
    assert_eq!(resp[1]["token"], PREFILL_BASE + 5);
    assert_eq!(resp[2]["completed_work"], 0, "prefill is NOT a timed step");
}

/// A correctness ANCHOR phase (correctness_begin + N correctness_step) is timed:
/// completed_work == 1 + N. Each step carries token + top_logits[8] + expert_stats.
#[test]
fn correctness_anchor_phase_completed_work_is_one_plus_n() {
    let resp = run(&[
        r#"{"id":1,"kind":"correctness_begin","prompt_tokens":[7,8,9]}"#,
        r#"{"id":2,"kind":"correctness_step","token":50}"#,
        r#"{"id":3,"kind":"correctness_step","token":51}"#,
        r#"{"id":4,"kind":"phase_diagnostics"}"#,
    ]);
    assert_all_conform(&resp);

    let begin = &resp[1];
    assert_eq!(begin["token"], PREFILL_BASE + 3);
    assert_eq!(begin["top_logits"].as_array().unwrap().len(), 8);
    assert!(begin.get("expert_stats").is_some());
    assert!(begin.get("peak_ram_gb").is_some());

    let step = &resp[2];
    assert_eq!(step["token"], 51); // 50 + 1
    assert_eq!(step["top_logits"].as_array().unwrap().len(), 8);
    assert!(step.get("expert_stats").is_some());

    assert_eq!(
        resp[4]["completed_work"], 3,
        "correctness_begin + 2 steps = 1 + 2"
    );
}

/// A free-run `correctness` request returns tokens[] + peak_ram_gb, is NOT timed
/// (completed_work == 0), and carries NO expert_stats (per schema note).
#[test]
fn correctness_freerun_returns_tokens_untimed() {
    let resp = run(&[
        r#"{"id":1,"kind":"correctness","prompt_tokens":[1,2],"steps":3}"#,
        r#"{"id":2,"kind":"phase_diagnostics"}"#,
    ]);
    assert_all_conform(&resp);
    let cor = &resp[1];
    assert_eq!(
        cor["tokens"],
        serde_json::json!([PREFILL_BASE, PREFILL_BASE + 1, PREFILL_BASE + 2])
    );
    assert!(cor.get("peak_ram_gb").is_some());
    assert!(
        cor.get("expert_stats").is_none(),
        "plain correctness carries no expert_stats"
    );
    assert_eq!(
        resp[2]["completed_work"], 0,
        "free-run correctness is NOT timed"
    );
}

#[test]
fn fresh_engine_per_phase_lifecycle() {
    let (factory, log) = MockFactory::new();
    let adapter = Adapter::with_session(factory, "cuda", "cuda", NONCE);
    let (resp, _) = run_with(
        adapter,
        &[
            r#"{"id":1,"kind":"decode_begin","seed_tokens":[1]}"#,
            r#"{"id":2,"kind":"decode_step","token":50}"#,
            r#"{"id":3,"kind":"phase_diagnostics"}"#,
            r#"{"id":4,"kind":"decode_begin","seed_tokens":[1,2]}"#,
            r#"{"id":5,"kind":"decode_step","token":60}"#,
            r#"{"id":6,"kind":"decode_step","token":61}"#,
            r#"{"id":7,"kind":"phase_diagnostics"}"#,
        ],
    );
    assert_all_conform(&resp);

    assert_eq!(log.created_count(), 2, "fresh engine per timed phase");
    let barriers: Vec<i64> = resp
        .iter()
        .filter(|r| r.get("completed_work").is_some())
        .map(|r| r["completed_work"].as_i64().unwrap())
        .collect();
    assert_eq!(barriers, vec![2, 3], "completed_work resets between phases");

    let events = log.events();
    assert_eq!(events[0], Event::Created(0));
    assert_eq!(events[1], Event::Drained(0));
    let created1 = events.iter().position(|e| *e == Event::Created(1)).unwrap();
    let dropped0 = events.iter().position(|e| *e == Event::Dropped(0)).unwrap();
    assert!(
        dropped0 < created1,
        "phase-1 engine dropped at its barrier before phase-2 is minted"
    );
    assert!(
        matches!(events[created1 + 1], Event::Drained(1)),
        "every fresh engine drains before any forward"
    );
}

// ---------------------------------------------------------------------------
// 2. WIRE SHAPE — id echo, nonce echo, flat responses
// ---------------------------------------------------------------------------

#[test]
fn real_shaped_request_echoes_id_and_produces_flat_shape() {
    let resp = run(&[r#"{"id":42,"kind":"prefill","prompt_tokens":[1,2,3]}"#]);
    assert_all_conform(&resp);
    let r = &resp[1];
    assert_eq!(r["id"], 42, "id is echoed");
    assert_eq!(r["ok"], true);
    assert_eq!(r["nonce"], NONCE);
    assert_eq!(r["token"], PREFILL_BASE + 3);
    assert!(kind_field(r).is_none(), "responses carry NO kind tag");
}

#[test]
fn every_response_echoes_nonce() {
    let resp = run(&[
        r#"{"id":1,"kind":"decode_begin","seed_tokens":[1]}"#,
        r#"{"id":2,"kind":"decode_step","token":5}"#,
        r#"{"id":3,"kind":"phase_diagnostics"}"#,
    ]);
    for r in &resp {
        assert_eq!(r["nonce"], NONCE, "response {r} missing session nonce");
    }
}

// ---------------------------------------------------------------------------
// 3. CONFORMANCE — schema validator + negative control
// ---------------------------------------------------------------------------

/// The negative control: the schema validator must REJECT a response with a
/// renamed field (additionalProperties:false). If this passed, the conformance
/// assertions above would be worthless.
#[test]
fn conformance_validator_bites_on_renamed_field() {
    let root = schema_root();
    // A genuine, valid emitted response.
    let resp = run(&[r#"{"id":7,"kind":"prefill","prompt_tokens":[1]}"#]);
    let good = resp[1].clone();
    validate_response(&root, &good).expect("the real emitted response conforms");

    // Mutate one field name: token -> tokenX. Now it violates the closed envelope.
    let mut bad = good.as_object().unwrap().clone();
    let v = bad.remove("token").unwrap();
    bad.insert("tokenX".to_string(), v);
    let bad = Value::Object(bad);
    let err = validate_response(&root, &bad).expect_err("renamed field MUST fail the schema");
    assert!(
        err.contains("tokenX"),
        "error should name the offending key: {err}"
    );
}

/// A wrong-typed field is also caught (token as a string).
#[test]
fn conformance_validator_bites_on_wrong_type() {
    let root = schema_root();
    let bad = serde_json::json!({"id": 1, "ok": true, "token": "not-an-int"});
    let err = validate_response(&root, &bad).expect_err("string token MUST fail");
    assert!(err.contains("token"), "{err}");
}

// ---------------------------------------------------------------------------
// Error handling + fail-closed hardening
// ---------------------------------------------------------------------------

#[test]
fn unparseable_line_answers_id_minus_one() {
    let resp = run(&[r#"{ this is not json"#]);
    assert_all_conform(&resp);
    let err = &resp[1];
    assert_eq!(err["id"], -1);
    assert_eq!(err["ok"], false);
    assert_eq!(err["nonce"], NONCE);
    assert!(err["error"].is_string());
}

#[test]
fn unknown_kind_errors_with_echoed_id() {
    let resp = run(&[r#"{"id":9,"kind":"teleport"}"#]);
    assert_all_conform(&resp);
    assert_eq!(resp[1]["id"], 9);
    assert_eq!(resp[1]["ok"], false);
}

#[test]
fn unknown_field_is_rejected_closed_envelope() {
    // deny_unknown_fields: a smuggled field fails the parse (id = -1).
    let resp = run(&[r#"{"id":3,"kind":"prefill","prompt_tokens":[1],"smuggled":true}"#]);
    assert_eq!(resp[1]["id"], -1);
    assert_eq!(resp[1]["ok"], false);
}

/// Malformed line mid-phase discards the session (fail-closed).
#[test]
fn malformed_line_discards_session() {
    let (factory, log) = MockFactory::new();
    let (resp, _) = run_with(
        Adapter::with_session(factory, "cuda", "cuda", NONCE),
        &[
            r#"{"id":1,"kind":"decode_begin","seed_tokens":[1,2]}"#,
            r#"{ not json"#,
            // this step now has no open phase -> fail-closed
            r#"{"id":3,"kind":"decode_step","token":9}"#,
        ],
    );
    assert_all_conform(&resp);
    assert!(
        log.events().contains(&Event::Dropped(0)),
        "session discarded on malformed line"
    );
    // The trailing decode_step gets a no-open-phase error, not a success.
    let last = resp.last().unwrap();
    assert_eq!(last["ok"], false);
    assert_eq!(last["id"], 3);
}

/// N1: a timed step with no open phase fails closed.
#[test]
fn decode_step_without_phase_fails_closed() {
    let resp = run(&[r#"{"id":1,"kind":"decode_step","token":1}"#]);
    assert_all_conform(&resp);
    assert_eq!(resp[1]["ok"], false);
    assert!(resp[1]["error"].as_str().unwrap().contains("no open phase"));
}

/// N1: a decode_step must not run on a correctness-minted engine — the step's
/// opener must match. Fails closed, no timed count accrued.
#[test]
fn decode_step_on_correctness_phase_fails_closed() {
    let resp = run(&[
        r#"{"id":1,"kind":"correctness_begin","prompt_tokens":[1,2]}"#,
        r#"{"id":2,"kind":"decode_step","token":5}"#,
        r#"{"id":3,"kind":"phase_diagnostics"}"#,
    ]);
    assert_all_conform(&resp);
    // decode_step rejected (wrong opener) and session discarded.
    assert_eq!(resp[2]["id"], 2);
    assert_eq!(resp[2]["ok"], false);
    assert!(resp[2]["error"]
        .as_str()
        .unwrap()
        .contains("no matching opener"));
    // The barrier then has no open phase to close -> also fails closed.
    assert_eq!(resp[3]["ok"], false);
}

/// N2: a double-open (two openers with no barrier between) fails closed.
#[test]
fn double_open_fails_closed() {
    let (factory, log) = MockFactory::new();
    let (resp, _) = run_with(
        Adapter::with_session(factory, "cuda", "cuda", NONCE),
        &[
            r#"{"id":1,"kind":"prefill","prompt_tokens":[1,2]}"#,
            // second opener with no phase_diagnostics between -> double-open
            r#"{"id":2,"kind":"decode_begin","seed_tokens":[3]}"#,
        ],
    );
    assert_all_conform(&resp);
    assert_eq!(resp[2]["id"], 2);
    assert_eq!(resp[2]["ok"], false);
    assert!(resp[2]["error"].as_str().unwrap().contains("double-open"));
    // The prefill engine was discarded fail-closed.
    assert!(log.events().contains(&Event::Dropped(0)));
}

/// Early EOF mid-phase: no barrier synthesized; the in-flight engine is dropped.
#[test]
fn early_eof_discards_session_without_barrier() {
    let (factory, log) = MockFactory::new();
    let (resp, _) = run_with(
        Adapter::with_session(factory, "cuda", "cuda", NONCE),
        &[
            r#"{"id":1,"kind":"decode_begin","seed_tokens":[1,2,3]}"#,
            r#"{"id":2,"kind":"decode_step","token":10}"#,
        ],
    );
    assert_all_conform(&resp);
    assert!(
        resp.iter().all(|r| r.get("completed_work").is_none()),
        "no barrier fabricated"
    );
    assert_eq!(
        *log.events().last().unwrap(),
        Event::Dropped(0),
        "session dropped at EOF"
    );
}

/// Fail-closed drain: a non-zero allocator residual aborts the phase before any
/// forward runs.
#[test]
fn nonzero_drain_fails_closed() {
    let (factory, log) = MockFactory::with_config(MockConfig {
        drain_residual: 4096,
        ..Default::default()
    });
    let (resp, _) = run_with(
        Adapter::with_session(factory, "cuda", "cuda", NONCE),
        &[r#"{"id":1,"kind":"decode_begin","seed_tokens":[1]}"#],
    );
    assert_all_conform(&resp);
    assert_eq!(resp[1]["ok"], false);
    assert!(resp[1]["error"].as_str().unwrap().contains("residual"));
    let events = log.events();
    assert!(events.contains(&Event::Drained(0)));
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, Event::Forward(_, Method::DecodeBegin, _))),
        "no forward ran after a failed drain"
    );
}

/// A mid-phase engine fault surfaces as an error and discards the session.
#[test]
fn engine_fault_mid_phase_discards_session() {
    let (factory, log) = MockFactory::with_config(MockConfig {
        fault_on_input: Some(42),
        ..Default::default()
    });
    let (resp, _) = run_with(
        Adapter::with_session(factory, "cuda", "cuda", NONCE),
        &[
            r#"{"id":1,"kind":"decode_begin","seed_tokens":[1]}"#,
            r#"{"id":2,"kind":"decode_step","token":42}"#,
        ],
    );
    assert_all_conform(&resp);
    assert_eq!(resp[2]["ok"], false);
    assert!(log.events().contains(&Event::Dropped(0)));
}

/// After an error discards a session, a fresh opener starts cleanly (the loop
/// keeps serving).
#[test]
fn recovers_after_error_with_fresh_phase() {
    let resp = run(&[
        r#"{"id":1,"kind":"decode_step","token":1}"#, // error: no phase
        r#"{"id":2,"kind":"prefill","prompt_tokens":[1,2]}"#, // fresh phase, ok
        r#"{"id":3,"kind":"phase_diagnostics"}"#,
    ]);
    assert_all_conform(&resp);
    assert_eq!(resp[1]["ok"], false);
    assert_eq!(resp[2]["ok"], true);
    assert_eq!(resp[2]["token"], PREFILL_BASE + 2);
    assert_eq!(resp[3]["completed_work"], 0);
}

#[test]
fn blank_lines_are_skipped() {
    let resp = run(&[
        "",
        r#"{"id":1,"kind":"prefill","prompt_tokens":[1]}"#,
        "   ",
        r#"{"id":2,"kind":"phase_diagnostics"}"#,
    ]);
    assert_all_conform(&resp);
    assert_eq!(resp.last().unwrap()["completed_work"], 0);
}

// ---------------------------------------------------------------------------
// 4. THE v1.1 FREE-RUN PAIR — this track's SCORED path
// ---------------------------------------------------------------------------
//
// Three things are proven here, and they are the three the free-run pair adds
// over base v1:
//
//   * THE VERB SEQUENCE. begin -> run -> barrier, with the phase counting
//     R + 1 where R is the number of ROUNDS (not the token count), and every
//     out-of-order form failing closed.
//   * THE SPEC ECHO. What comes back is what will RUN: an absent spec is
//     serial, an absent depth resolves to the measured default, an over-ceiling
//     depth is CLAMPED (not refused), and a mode this track does not declare is
//     REFUSED BY NAME.
//   * NO DERIVED METRIC. The response carries raw counters and nothing that
//     divides two of them.

/// The keys a response is allowed to carry on the free-run path. Anything else
/// on a `free_decode_run` response is either a derived metric or an accident,
/// and both are worth failing on.
const FREE_RUN_RESPONSE_KEYS: &[&str] = &[
    "id",
    "nonce",
    "ok",
    "tokens",
    "acceptance_lengths",
    "drafted_total",
    "accepted_total",
    "committed_total",
    "verify_replay_disagreements",
];

fn free_begin(id: i64, spec: Option<&str>) -> String {
    match spec {
        Some(s) => {
            format!(r#"{{"id":{id},"kind":"free_decode_begin","seed_tokens":[1,2,3],"spec":{s}}}"#)
        }
        None => format!(r#"{{"id":{id},"kind":"free_decode_begin","seed_tokens":[1,2,3]}}"#),
    }
}

fn free_run_line(id: i64, count: i64) -> String {
    format!(r#"{{"id":{id},"kind":"free_decode_run","count":{count}}}"#)
}

#[test]
fn free_run_phase_verb_sequence_and_completed_work() {
    let resp = run(&[
        &free_begin(1, Some(r#"{"mode":"mtp","mtp":{"depth":2}}"#)),
        &free_run_line(2, 16),
        r#"{"id":3,"kind":"phase_diagnostics"}"#,
    ]);
    assert_eq!(resp.len(), 4, "hello + three answers");

    // begin: seed token + the resolved spec, and NO counters yet.
    let begin = &resp[1];
    assert_eq!(begin["id"], 1);
    assert_eq!(begin["ok"], true);
    assert_eq!(begin["seed_token"], SEED_BASE + 3);
    assert_eq!(begin["effective_spec"]["mode"], "mtp");
    assert_eq!(begin["effective_spec"]["mtp"]["depth"], 2);
    assert!(begin.get("tokens").is_none(), "the opener commits nothing");

    // run: exactly `count` tokens, and the consistency triple holds.
    let run_resp = &resp[2];
    assert_eq!(run_resp["id"], 2);
    assert_eq!(run_resp["ok"], true);
    assert_eq!(run_resp["committed_total"], 16);
    let tokens = run_resp["tokens"].as_array().expect("tokens array");
    assert_eq!(tokens.len(), 16);
    let lengths: Vec<i64> = run_resp["acceptance_lengths"]
        .as_array()
        .expect("acceptance_lengths array")
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect();
    assert_eq!(
        lengths.iter().sum::<i64>(),
        16,
        "sum(acceptance_lengths) == N"
    );
    assert!(
        lengths.iter().all(|&n| n > 0),
        "no round commits zero tokens"
    );
    assert!(
        run_resp["drafted_total"].as_i64().unwrap() >= run_resp["accepted_total"].as_i64().unwrap(),
        "drafted_total >= accepted_total"
    );
    assert!(
        run_resp["drafted_total"].as_i64().unwrap() > 0,
        "the mtp leg must actually draft, or this case proves nothing about drafting"
    );

    // THE BARRIER COUNTS ROUNDS, NOT TOKENS: benchd requires
    // `completed_work == R + 1`, where R is the number of ROUNDS. This case is
    // the one that discriminates, because the mtp leg commits 16 tokens over
    // FEWER rounds -- an adapter counting tokens would report 17 here and be
    // refused by benchd.
    let rounds = lengths.len() as i64;
    assert!(
        rounds < 16,
        "this case only proves the rule if R < N; got R={rounds}"
    );
    assert_eq!(resp[3]["completed_work"], rounds + 1);
}

#[test]
fn free_run_serial_leg_drafts_nothing_and_commits_one_per_round() {
    let resp = run(&[
        &free_begin(1, Some(r#"{"mode":"serial"}"#)),
        &free_run_line(2, 5),
        r#"{"id":3,"kind":"phase_diagnostics"}"#,
    ]);
    assert_eq!(resp[1]["effective_spec"]["mode"], "serial");
    assert!(
        resp[1]["effective_spec"].get("mtp").is_none(),
        "a serial echo must not carry an mtp block"
    );
    assert_eq!(resp[2]["drafted_total"], 0);
    assert_eq!(resp[2]["accepted_total"], 0);
    assert_eq!(resp[2]["committed_total"], 5);
    let lengths: Vec<i64> = resp[2]["acceptance_lengths"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect();
    assert_eq!(lengths, vec![1, 1, 1, 1, 1]);
    // R == N on the serial leg, because every round commits exactly one token.
    // That coincidence is why a token-counting adapter passes this case and
    // fails the mtp one above.
    assert_eq!(resp[3]["completed_work"], 6);
}

#[test]
fn a_backend_that_measures_no_disagreement_omits_the_key() {
    // `verify_replay_disagreements` is OPTIONAL and ABSENT MEANS NOT REPORTED,
    // which is a different statement from `0`. The mock runs no batched verify
    // and no one-row replay, so it has no two argmaxes to compare and says
    // nothing -- on the mtp leg as well as the serial one.
    for spec in [None, Some(r#"{"mode":"mtp","mtp":{"depth":1}}"#)] {
        let resp = run(&[&free_begin(1, spec), &free_run_line(2, 6)]);
        assert_eq!(resp[2]["ok"], true, "{}", resp[2]);
        assert!(
            resp[2].get("verify_replay_disagreements").is_none(),
            "the mock must omit the key, not zero it: {}",
            resp[2]
        );
    }
}

#[test]
fn absent_spec_is_the_serial_control_leg() {
    let resp = run(&[&free_begin(1, None), &free_run_line(2, 3)]);
    assert_eq!(resp[1]["ok"], true);
    assert_eq!(resp[1]["effective_spec"]["mode"], "serial");
    assert_eq!(resp[2]["drafted_total"], 0);
}

#[test]
fn absent_mtp_depth_resolves_to_the_measured_default_and_is_echoed() {
    let resp = run(&[&free_begin(1, Some(r#"{"mode":"mtp"}"#))]);
    assert_eq!(resp[1]["ok"], true);
    assert_eq!(
        resp[1]["effective_spec"]["mtp"]["depth"],
        crate::mock::MTP_DEFAULT_DEPTH
    );
}

#[test]
fn an_explicitly_requested_depth_is_echoed_verbatim() {
    // benchd holds the echo of an EXPLICIT depth to the request. Every value
    // inside the envelope must therefore come back unchanged.
    for depth in [crate::mock::MTP_MIN_DEPTH, 2, crate::mock::MTP_MAX_DEPTH] {
        let spec = format!(r#"{{"mode":"mtp","mtp":{{"depth":{depth}}}}}"#);
        let resp = run(&[&free_begin(1, Some(&spec))]);
        assert_eq!(resp[1]["ok"], true, "depth {depth} is inside the envelope");
        assert_eq!(resp[1]["effective_spec"]["mtp"]["depth"], depth);
    }
}

#[test]
fn an_out_of_envelope_explicit_depth_is_refused_not_clamped() {
    // CLAMPING AN EXPLICIT REQUEST IS THE BUG THIS CATCHES. benchd requires an
    // explicitly requested depth to be echoed verbatim, so an engine that
    // clamped 9 to 3 and echoed 3 would produce an echo DIVERGENCE: benchd
    // discards the leg with a message about the echo, not about the depth, and
    // the operator is left with an opaque failure on a paired measurement.
    // Refusing by name is what makes the fault readable.
    for depth in [0i64, crate::mock::MTP_MAX_DEPTH + 1, 9, -1] {
        let spec = format!(r#"{{"mode":"mtp","mtp":{{"depth":{depth}}}}}"#);
        let resp = run(&[&free_begin(1, Some(&spec))]);
        assert_eq!(resp[1]["ok"], false, "depth {depth} must be refused");
        let err = resp[1]["error"].as_str().unwrap();
        assert!(
            err.contains(&depth.to_string()) && err.contains("envelope"),
            "the refusal names the depth and the envelope: {err}"
        );
        assert!(
            resp[1].get("effective_spec").is_none(),
            "a refused begin echoes no spec"
        );
    }
}

#[test]
fn an_undeclared_mode_is_refused_by_name() {
    // `dflash` is the mode this track most plausibly gets asked for by a stale
    // caller, and it must be refused rather than quietly run as serial.
    let resp = run(&[&free_begin(1, Some(r#"{"mode":"dflash"}"#))]);
    assert_eq!(resp[1]["ok"], false);
    assert_eq!(resp[1]["id"], 1, "the refusal answers the request's own id");
    let err = resp[1]["error"].as_str().unwrap();
    assert!(err.contains("dflash"), "the refusal names the mode: {err}");
    assert!(
        err.contains("serial") && err.contains("mtp"),
        "the refusal names what IS runnable: {err}"
    );
}

#[test]
fn a_real_retired_arm_spec_still_gets_the_named_refusal() {
    // THE SHAPE A STALE CALLER ACTUALLY SENDS. A retired-arm spec carries its
    // own block -- `{"mode":"dflash","dflash":{...}}` -- not a bare mode. With
    // a closed `Spec` envelope that block died in serde while the whole
    // request line was being parsed, so the adapter answered `id = -1` with
    // "not a valid WorkerRequest" and the named refusal was never reached: the
    // operator saw a parse error about a line that was perfectly well formed.
    let resp = run(&[&free_begin(
        1,
        Some(r#"{"mode":"dflash","dflash":{"depth":4,"draft":{"artifact":"x","sha256":"y"}}}"#),
    )]);
    assert_eq!(resp[1]["ok"], false);
    assert_eq!(
        resp[1]["id"], 1,
        "the refusal carries the request's id, not -1: the line parsed, the MODE is what is wrong"
    );
    let err = resp[1]["error"].as_str().unwrap();
    assert!(err.contains("dflash"), "the refusal names the mode: {err}");
    assert!(
        !err.contains("not a valid WorkerRequest"),
        "a well-formed line naming a retired mode must not be reported as a parse error: {err}"
    );
}

#[test]
fn a_serial_spec_carrying_an_mtp_block_is_refused() {
    let resp = run(&[&free_begin(
        1,
        Some(r#"{"mode":"serial","mtp":{"depth":2}}"#),
    )]);
    assert_eq!(resp[1]["ok"], false);
    assert!(resp[1]["error"].as_str().unwrap().contains("mtp"));
}

#[test]
fn an_engine_that_cannot_run_mtp_refuses_by_name() {
    // The capability half of the same rule: the SPEC parsed fine, and the
    // ENGINE is what cannot run it.
    let (factory, _log) = MockFactory::with_config(MockConfig {
        runnable: Some(vec![crate::engine::Route::Serial]),
        ..MockConfig::default()
    });
    let (resp, _) = run_with(
        Adapter::with_session(factory, "cuda", "cuda", NONCE)
            .advertising(vec![crate::engine::Route::Serial]),
        &[&free_begin(1, Some(r#"{"mode":"mtp"}"#))],
    );
    assert_eq!(resp[1]["ok"], false);
    let err = resp[1]["error"].as_str().unwrap();
    assert!(err.contains("mtp") && err.contains("serial"), "{err}");

    // AND THE HELLO SAID SO FIRST. benchd refuses a mode absent from
    // `spec_modes` before it ever issues the request, so an engine that cannot
    // run mtp must not advertise it -- otherwise the refusal lands mid-session
    // instead of at the handshake.
    let modes: Vec<&str> = resp[0]["spec_modes"]
        .as_array()
        .expect("the hello advertises spec_modes")
        .iter()
        .map(|m| m.as_str().unwrap())
        .collect();
    assert_eq!(modes, vec!["serial"]);
}

#[test]
fn free_decode_run_without_an_opener_fails_closed() {
    let resp = run(&[&free_run_line(1, 4)]);
    assert_eq!(resp[1]["ok"], false);
    assert!(resp[1]["error"]
        .as_str()
        .unwrap()
        .contains("free_decode_run"));
}

#[test]
fn free_decode_run_on_a_teacher_forced_phase_fails_closed() {
    // The two decode regimes must not be crossable: a free-run request inside a
    // teacher-forced phase would commit tokens the caller is timing under the
    // other regime's rules.
    let resp = run(&[
        r#"{"id":1,"kind":"decode_begin","seed_tokens":[1,2]}"#,
        &free_run_line(2, 4),
    ]);
    assert_eq!(resp[2]["ok"], false);
    let err = resp[2]["error"].as_str().unwrap();
    assert!(err.contains("Decode") && err.contains("FreeRun"), "{err}");
}

#[test]
fn decode_step_inside_a_free_run_phase_fails_closed() {
    let resp = run(&[
        &free_begin(1, None),
        r#"{"id":2,"kind":"decode_step","token":7}"#,
    ]);
    assert_eq!(resp[2]["ok"], false);
}

#[test]
fn free_decode_begin_while_a_phase_is_open_is_a_double_open() {
    let resp = run(&[&free_begin(1, None), &free_begin(2, None)]);
    assert_eq!(resp[2]["ok"], false);
    assert!(resp[2]["error"].as_str().unwrap().contains("double-open"));
}

#[test]
fn a_missing_or_out_of_range_count_is_refused() {
    let resp = run(&[&free_begin(1, None), r#"{"id":2,"kind":"free_decode_run"}"#]);
    assert_eq!(resp[2]["ok"], false);
    assert!(resp[2]["error"].as_str().unwrap().contains("count"));

    for bad in [0i64, -3, crate::adapter::FREE_RUN_MAX_COUNT + 1] {
        let resp = run(&[&free_begin(1, None), &free_run_line(2, bad)]);
        assert_eq!(resp[2]["ok"], false, "count {bad} must be refused");
        assert!(resp[2]["error"].as_str().unwrap().contains("count"));
    }
}

#[test]
fn a_refused_free_run_discards_the_session() {
    // Fail-closed: the phase's half-advanced state must not survive to be
    // reused by the next request.
    let resp = run(&[
        &free_begin(1, None),
        &free_run_line(2, 0), // refused
        r#"{"id":3,"kind":"phase_diagnostics"}"#,
    ]);
    assert_eq!(resp[2]["ok"], false);
    assert_eq!(
        resp[3]["ok"], false,
        "the barrier finds no open phase, because the refusal discarded it"
    );
}

#[test]
fn free_run_response_carries_no_derived_metric() {
    // THE MEASUREMENT BOUNDARY, as a tripwire. The engine reports raw counters;
    // benchd times the request and does every division. A field whose name
    // implies a rate, a ratio, a duration or a speedup means the engine started
    // measuring itself.
    let resp = run(&[&free_begin(1, None), &free_run_line(2, 8)]);
    let run_resp = resp[2].as_object().expect("object");

    let mut unexpected: Vec<&String> = run_resp
        .keys()
        .filter(|k| !FREE_RUN_RESPONSE_KEYS.contains(&k.as_str()))
        .collect();
    unexpected.sort();
    assert!(
        unexpected.is_empty(),
        "free_decode_run response carries keys outside the raw-counter set: {unexpected:?}"
    );

    // Belt and braces, over the TOP-LEVEL keys of EVERY response the adapter
    // can emit rather than just this one.
    //
    // TOP-LEVEL, and the scope is deliberate: `expert_stats` is a
    // benchd-DEFINED v1 sub-struct and one of its members is
    // `expert_read_seconds`. That is benchd's own counter shape, not a number
    // this engine invented about its own speed, so the check does not descend
    // into it. What this refuses is a derived quantity appearing where the
    // ENGINE decides the key.
    for (i, r) in resp.iter().enumerate() {
        for key in r.as_object().expect("object").keys() {
            let lower = key.to_ascii_lowercase();
            for needle in DERIVED_METRIC_NEEDLES {
                assert!(
                    !lower.contains(needle),
                    "response[{i}] carries key {key:?}, which names a DERIVED quantity; \
                     measurement lives in benchd"
                );
            }
        }
    }
}

/// Name fragments that mean "the engine divided two numbers". Used by the
/// tripwire above and by its negative control below.
const DERIVED_METRIC_NEEDLES: &[&str] = &[
    "speedup",
    "ratio",
    "rate",
    "tps",
    "tokens_per_second",
    "seconds",
    "elapsed",
    "duration",
    "throughput",
    "latency",
    "score",
    "gain",
    "median",
    "composite",
];

#[test]
fn the_derived_metric_tripwire_bites() {
    // NEGATIVE CONTROL for the check above: a response that DID carry a derived
    // key must be caught by both halves. Built by hand, because the adapter
    // cannot emit one.
    let forged: Value = serde_json::from_str(
        r#"{"id":2,"nonce":"testnonce","ok":true,"tokens":[1],"decode_tps":42.0}"#,
    )
    .unwrap();
    let obj = forged.as_object().unwrap();

    let outside: Vec<&String> = obj
        .keys()
        .filter(|k| !FREE_RUN_RESPONSE_KEYS.contains(&k.as_str()))
        .collect();
    assert_eq!(
        outside.len(),
        1,
        "the key-set half must flag the forged derived key"
    );

    let caught = obj.keys().any(|k| {
        let lower = k.to_ascii_lowercase();
        DERIVED_METRIC_NEEDLES.iter().any(|n| lower.contains(n))
    });
    assert!(caught, "the needle half must flag the forged derived key");
}

#[test]
fn a_free_run_phase_mints_its_own_engine_and_drops_it_at_the_barrier() {
    let (factory, log) = MockFactory::new();
    let (_resp, _) = run_with(
        Adapter::with_session(factory, "cuda", "cuda", NONCE),
        &[
            &free_begin(1, None),
            &free_run_line(2, 4),
            r#"{"id":3,"kind":"phase_diagnostics"}"#,
            &free_begin(4, None),
            &free_run_line(5, 4),
            r#"{"id":6,"kind":"phase_diagnostics"}"#,
        ],
    );
    assert_eq!(
        log.created_count(),
        2,
        "one fresh engine per free-run phase"
    );
    let events = log.events();
    assert!(
        events.contains(&Event::Forward(0, Method::FreeDecodeRun, 4)),
        "the first phase's run reached the engine"
    );
    assert_eq!(
        events
            .iter()
            .filter(|e| matches!(e, Event::Dropped(_)))
            .count(),
        2,
        "both engines were dropped at their barriers"
    );
}

// ---------------------------------------------------------------------------
// 5. THE ds4 BACKEND — driven through a SCRIPTED SESSION (no GPU)
// ---------------------------------------------------------------------------
//
// These drive the ACTUAL ds4 backend (crate::ds4_backend) through the actual
// adapter loop; only the engine is stubbed, by a scripted Ds4Session. What
// they prove: the verb->session translation, the free-run counter accounting
// (serial and mtp), the spec resolution / refusals, and that NO derived metric
// is emitted. What they CANNOT prove is the real GPU drive — that is box
// validation (tools/ds4/mtp-exactness-gate.py).

mod ds4 {
    use std::sync::{Arc, Mutex};

    use serde_json::Value;

    use crate::adapter::Adapter;
    use crate::ds4_backend::{
        Ds4Engine, Ds4Factory, Ds4Session, SpecCounters, DS4_IMPLEMENTED_DEPTH, MTP_MAX_DEPTH,
        MTP_MIN_DEPTH,
    };
    use crate::engine::{Engine, EngineError, Route};
    use crate::protocol::CorrectnessTraceLogit;

    /// A deterministic fake engine. The "model" always predicts `last + 1`.
    /// Like the real engine, the FIRST speculative cycle of a session is a
    /// plain step (the engine measures its serial baseline there); after that
    /// the MTP draft is right on every cycle except the ones listed in
    /// `reject_cycles`, so acceptance is scripted. `quench_at_cycle` models the
    /// engine giving up on speculation. A rejecting cycle listed in
    /// `disagree_cycles` also bumps the engine's verify/replay disagreement
    /// counter, which the port raises on a rejecting round only.
    #[derive(Default)]
    struct Scripted {
        mtp_armed: bool,
        context: Vec<i64>,
        drafts: u64,
        hits: u64,
        quenches: u64,
        disagreements: u64,
        cycles: u64,
        reject_cycles: Vec<u64>,
        disagree_cycles: Vec<u64>,
        /// Cycles on which the engine's counter rises although the round did
        /// NOT reject. Only a rejecting round can disagree, so this models an
        /// engine whose counter overstates itself -- the one shape benchd
        /// refuses the leg for.
        phantom_disagree_cycles: Vec<u64>,
        quench_at_cycle: Option<u64>,
        invalidated: u64,
        log: Vec<String>,
    }

    impl Scripted {
        fn frontier(&self) -> i64 {
            self.context.last().copied().unwrap_or(0) + 1
        }
    }

    impl Ds4Session for Scripted {
        fn mtp_armed(&self) -> bool {
            self.mtp_armed
        }
        fn sync(&mut self, tokens: &[i64]) -> Result<(), String> {
            self.log.push(format!("sync({})", tokens.len()));
            self.context = tokens.to_vec();
            Ok(())
        }
        fn eval(&mut self, token: i64) -> Result<(), String> {
            self.log.push(format!("eval({token})"));
            self.context.push(token);
            Ok(())
        }
        fn argmax(&mut self) -> i64 {
            self.frontier()
        }
        fn top_logits(&mut self, k: usize) -> Vec<CorrectnessTraceLogit> {
            let top = self.frontier();
            (0..k as i64)
                .map(|i| CorrectnessTraceLogit::new(top + i, 10.0 - i as f64))
                .collect()
        }
        fn eval_speculative(&mut self, first_token: i64, budget: i64) -> Result<Vec<i64>, String> {
            self.log.push(format!("spec({first_token},{budget})"));
            self.cycles += 1;
            self.context.push(first_token);
            // The engine's own gates: the first cycle is the baseline
            // measurement, and no draft when only one token is wanted.
            if self.cycles == 1 || budget < 2 {
                return Ok(vec![first_token]);
            }
            if self.quench_at_cycle == Some(self.cycles) {
                self.quenches += 1;
            }
            if self.quenches > 0 {
                return Ok(vec![first_token]);
            }
            self.drafts += 1;
            if self.phantom_disagree_cycles.contains(&self.cycles) {
                self.disagreements += 1;
            }
            if self.reject_cycles.contains(&self.cycles) {
                if self.disagree_cycles.contains(&self.cycles) {
                    self.disagreements += 1;
                }
                return Ok(vec![first_token]);
            }
            self.hits += 1;
            let draft = self.frontier();
            self.context.push(draft);
            Ok(vec![first_token, draft])
        }
        fn spec_counters(&mut self) -> SpecCounters {
            SpecCounters {
                drafts: self.drafts,
                hits: self.hits,
                quenches: self.quenches,
                verify_replay_disagreements: self.disagreements,
            }
        }
        fn invalidate(&mut self) {
            self.invalidated += 1;
            self.context.clear();
        }
    }

    fn engine(session: &Arc<Mutex<Scripted>>) -> Ds4Engine {
        let dynamic: Arc<Mutex<dyn Ds4Session>> = session.clone();
        Ds4Engine::new(dynamic)
    }

    fn run(session: Arc<Mutex<Scripted>>, lines: &[&str]) -> Vec<Value> {
        let dynamic: Arc<Mutex<dyn Ds4Session>> = session;
        let mut adapter = Adapter::with_session(
            Ds4Factory::with_session(dynamic),
            "cuda",
            "cuda",
            super::NONCE,
        );
        let input = lines.join("\n") + "\n";
        let mut out = Vec::new();
        adapter.run(input.as_bytes(), &mut out).unwrap();
        String::from_utf8(out)
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect()
    }

    #[test]
    fn step_is_teacher_forced_and_carries_top8() {
        let session = Arc::new(Mutex::new(Scripted::default()));
        let mut e = engine(&session);
        assert_eq!(e.prefill(&[10, 11, 12]).unwrap(), 13);
        let step = e.step(100).unwrap();
        assert_eq!(
            step.token, 101,
            "the fed token, not the model's, is the new context"
        );
        assert_eq!(step.top_logits.len(), 8);
        assert_eq!(step.top_logits[0].token, 101);
        let step = e.step(200).unwrap();
        assert_eq!(step.token, 201);
        assert_eq!(
            session.lock().unwrap().log,
            vec!["sync(3)", "eval(100)", "eval(200)"]
        );
    }

    #[test]
    fn serial_free_run_commits_exactly_count_with_unit_lengths() {
        let session = Arc::new(Mutex::new(Scripted::default()));
        let mut e = engine(&session);
        let (seed_token, spec) = e
            .free_decode_begin(&[1, 2, 3], Route::Serial, None)
            .unwrap();
        assert_eq!(seed_token, 4);
        assert_eq!(spec.mode, "serial");
        assert!(spec.mtp.is_none());
        let r = e.free_decode_run(5).unwrap();
        assert_eq!(r.tokens, vec![5, 6, 7, 8, 9]);
        assert_eq!(r.acceptance_lengths, vec![1; 5]);
        assert_eq!(
            (r.drafted_total, r.accepted_total, r.committed_total),
            (0, 0, 5)
        );
    }

    #[test]
    fn mtp_free_run_counts_cycles_drafts_and_hits() {
        let session = Arc::new(Mutex::new(Scripted {
            mtp_armed: true,
            reject_cycles: vec![2],
            ..Default::default()
        }));
        let mut e = engine(&session);
        let (seed_token, spec) = e
            .free_decode_begin(&[1, 2, 3], Route::Mtp, Some(1))
            .unwrap();
        assert_eq!(seed_token, 4);
        assert_eq!(spec.mtp.unwrap().depth, 1);
        let r = e.free_decode_run(6).unwrap();
        // cycle 1: feed 4, baseline plain step, frontier 5 -> [5]
        // cycle 2: feed 5, draft rejected, frontier 6      -> [6]
        // cycle 3: feed 6, draft 7 accepted, frontier 8    -> [7, 8]
        // cycle 4: feed 8, draft 9 accepted, frontier 10   -> [9, 10]
        assert_eq!(r.tokens, vec![5, 6, 7, 8, 9, 10]);
        assert_eq!(r.acceptance_lengths, vec![1, 1, 2, 2]);
        assert_eq!(r.committed_total, 6);
        assert_eq!(r.drafted_total, 3, "three cycles proposed a draft");
        assert_eq!(r.accepted_total, 2, "two drafts matched the target");
        assert!(r.drafted_total >= r.accepted_total);
        assert_eq!(r.acceptance_lengths.iter().sum::<i64>(), 6);
    }

    #[test]
    fn verify_replay_disagreements_reach_the_result_and_never_fail_the_leg() {
        // The port raises this counter on a REJECTING round where the batched
        // verify's row-0 argmax and the one-row replay of the same position
        // chose different tokens. The replay stands, so the leg is a result,
        // not a fault -- benchd seals a ceiling on the rate later, not here.
        let session = Arc::new(Mutex::new(Scripted {
            mtp_armed: true,
            reject_cycles: vec![2, 3],
            disagree_cycles: vec![3],
            ..Default::default()
        }));
        let mut e = engine(&session);
        e.free_decode_begin(&[1, 2, 3], Route::Mtp, Some(1))
            .unwrap();
        let r = e.free_decode_run(5).unwrap();
        assert_eq!(
            r.verify_replay_disagreements,
            Some(1),
            "the one disagreeing round is carried as a delta over the leg"
        );
        assert_eq!(r.drafted_total, 3, "the disagreement did not lose a draft");
        assert_eq!(r.accepted_total, 1);
        assert_eq!(r.committed_total, 5, "the leg still committed its count");
    }

    #[test]
    fn a_leg_with_no_disagreement_reports_zero() {
        let session = Arc::new(Mutex::new(Scripted {
            mtp_armed: true,
            reject_cycles: vec![2],
            ..Default::default()
        }));
        let mut e = engine(&session);
        e.free_decode_begin(&[1, 2, 3], Route::Mtp, Some(1))
            .unwrap();
        let r = e.free_decode_run(6).unwrap();
        assert_eq!(
            r.verify_replay_disagreements,
            Some(0),
            "a leg that WATCHED and saw none reports zero, not nothing"
        );
    }

    #[test]
    fn a_serial_leg_reports_no_disagreements() {
        let session = Arc::new(Mutex::new(Scripted::default()));
        let mut e = engine(&session);
        e.free_decode_begin(&[1, 2, 3], Route::Serial, None)
            .unwrap();
        let r = e.free_decode_run(4).unwrap();
        assert_eq!(
            (
                r.drafted_total,
                r.accepted_total,
                r.verify_replay_disagreements
            ),
            (0, 0, None),
            "the serial route reads no engine counters at all, so it reports \
             no disagreement count rather than a zero it never measured"
        );
    }

    #[test]
    fn mtp_stream_equals_serial_stream() {
        // Same scripted model, same seed: the mtp leg must commit the tokens the
        // serial leg commits. This is the bit-exactness the box gate measures.
        let serial = Arc::new(Mutex::new(Scripted::default()));
        let mut s = engine(&serial);
        s.free_decode_begin(&[7, 8], Route::Serial, None).unwrap();
        let serial_tokens = s.free_decode_run(7).unwrap().tokens;

        let mtp = Arc::new(Mutex::new(Scripted {
            mtp_armed: true,
            reject_cycles: vec![1, 3],
            ..Default::default()
        }));
        let mut m = engine(&mtp);
        m.free_decode_begin(&[7, 8], Route::Mtp, None).unwrap();
        let mtp_tokens = m.free_decode_run(7).unwrap().tokens;
        assert_eq!(serial_tokens, mtp_tokens);
    }

    #[test]
    fn quench_during_the_mtp_leg_is_a_fault() {
        let session = Arc::new(Mutex::new(Scripted {
            mtp_armed: true,
            quench_at_cycle: Some(3),
            ..Default::default()
        }));
        let mut e = engine(&session);
        e.free_decode_begin(&[1, 2], Route::Mtp, Some(1)).unwrap();
        let err = e.free_decode_run(8).unwrap_err();
        assert!(
            matches!(err, EngineError::Fault(ref m) if m.contains("quenched")),
            "{err:?}"
        );
    }

    #[test]
    fn a_drafter_that_never_drafts_is_a_fault() {
        // Armed, but every cycle after the baseline is "quenched" from the start:
        // no draft is ever proposed, which the counters expose as drafted == 0.
        let session = Arc::new(Mutex::new(Scripted {
            mtp_armed: true,
            quench_at_cycle: Some(2),
            ..Default::default()
        }));
        let mut e = engine(&session);
        e.free_decode_begin(&[1, 2], Route::Mtp, Some(1)).unwrap();
        let err = e.free_decode_run(8).unwrap_err();
        assert!(matches!(err, EngineError::Fault(_)), "{err:?}");
    }

    #[test]
    fn phase_start_invalidates_the_live_prefix() {
        let session = Arc::new(Mutex::new(Scripted::default()));
        let mut e = engine(&session);
        e.drain_to_zero().unwrap();
        e.prefill(&[1, 2, 3]).unwrap();
        e.drain_to_zero().unwrap();
        assert_eq!(session.lock().unwrap().invalidated, 2);
    }

    #[test]
    fn mtp_refused_by_name_when_drafter_not_armed() {
        let session = Arc::new(Mutex::new(Scripted::default()));
        let mut e = engine(&session);
        let err = e.free_decode_begin(&[1], Route::Mtp, Some(1)).unwrap_err();
        match err {
            EngineError::UnsupportedMode { requested, .. } => assert_eq!(requested, "mtp"),
            other => panic!("expected UnsupportedMode, got {other:?}"),
        }
    }

    #[test]
    fn implemented_depth_agrees_with_the_vendored_engine() {
        // THE DRIFT GUARD, and the only test here that can catch a wrong
        // DS4_IMPLEMENTED_DEPTH. Every other depth assertion derives its range
        // from that constant, so it stays green whatever the constant says.
        //
        // The engine is VENDORED, so its own envelope is a file in this
        // repository and can simply be read. A vendor-sync that moves
        // DS4_QWEN4EXP_IMPLEMENTED_DEPTH now fails here until the adapter is
        // moved with it -- which is the point: a constant lower than the
        // engine's refuses depths it serves, and higher accepts depths it
        // refuses at open.
        let header =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../ds4/ds4_qwen4exp_mtp.h");
        let text = std::fs::read_to_string(&header)
            .unwrap_or_else(|e| panic!("cannot read the vendored engine header {header:?}: {e}"));
        let engine_depth: i64 = text
            .lines()
            .find_map(|l| {
                l.strip_prefix("#define DS4_QWEN4EXP_IMPLEMENTED_DEPTH")
                    .and_then(|rest| rest.split_whitespace().next())
                    .and_then(|n| n.parse().ok())
            })
            .expect("the vendored engine header defines DS4_QWEN4EXP_IMPLEMENTED_DEPTH");
        assert_eq!(
            DS4_IMPLEMENTED_DEPTH, engine_depth,
            "the adapter implements depth {DS4_IMPLEMENTED_DEPTH} but the vendored engine \
             implements {engine_depth}; move DS4_IMPLEMENTED_DEPTH with the vendor-sync"
        );
        assert!(
            engine_depth <= MTP_MAX_DEPTH,
            "the engine implements depth {engine_depth}, past the track envelope's {MTP_MAX_DEPTH}"
        );
    }

    #[test]
    fn every_depth_the_engine_implements_is_accepted_and_echoed() {
        // THE POSITIVE CONTROL for the envelope. The refusal test below only
        // proves that out-of-range depths are rejected; without this, setting
        // DS4_IMPLEMENTED_DEPTH to 1 by mistake would still pass it while
        // silently refusing depths the vendored engine serves.
        for depth in MTP_MIN_DEPTH..=DS4_IMPLEMENTED_DEPTH {
            let session = Arc::new(Mutex::new(Scripted {
                mtp_armed: true,
                ..Default::default()
            }));
            let mut e = engine(&session);
            let (_, spec) = e
                .free_decode_begin(&[1], Route::Mtp, Some(depth))
                .unwrap_or_else(|err| panic!("depth {depth} was refused: {err:?}"));
            assert_eq!(
                spec.mtp.expect("an mtp leg reports its depth").depth,
                depth,
                "depth {depth} must be echoed as itself, never clamped"
            );
        }
    }

    #[test]
    fn deeper_than_implemented_depth_is_refused_not_clamped() {
        let session = Arc::new(Mutex::new(Scripted {
            mtp_armed: true,
            ..Default::default()
        }));
        let mut e = engine(&session);
        // Every value OUTSIDE the envelope. `DS4_IMPLEMENTED_DEPTH + 1` is the
        // boundary and moves with the engine; 0 and 9 bracket it from both
        // sides. A depth the engine implements must NOT be in this list, or the
        // test would be asserting that a working depth is refused.
        for depth in [DS4_IMPLEMENTED_DEPTH + 1, DS4_IMPLEMENTED_DEPTH + 6, 0, -1] {
            let err = e
                .free_decode_begin(&[1], Route::Mtp, Some(depth))
                .unwrap_err();
            assert!(
                matches!(err, EngineError::DepthOutOfEnvelope { requested, .. } if requested == depth),
                "depth {depth}: {err:?}"
            );
        }
        // An unnamed depth is the engine's to pick: it picks what it implements.
        let (_, spec) = e.free_decode_begin(&[1], Route::Mtp, None).unwrap();
        assert_eq!(spec.mtp.unwrap().depth, DS4_IMPLEMENTED_DEPTH);
    }

    #[test]
    fn correctness_freerun_stops_at_eos() {
        let session = Arc::new(Mutex::new(Scripted::default()));
        let mut e = engine(&session);
        // The scripted model predicts last+1, so start just below an EOS id.
        let out = e.correctness_freerun(&[248043], 10).unwrap();
        assert_eq!(out, vec![248044], "stopped on the first EOS id");
        let out = e.correctness_freerun(&[1, 2], 3).unwrap();
        assert_eq!(out, vec![3, 4, 5]);
    }

    #[test]
    fn adapter_loop_emits_raw_counters_only() {
        let session = Arc::new(Mutex::new(Scripted {
            mtp_armed: true,
            ..Default::default()
        }));
        let responses = run(
            session,
            &[
                r#"{"id":1,"kind":"free_decode_begin","seed_tokens":[1,2,3],"spec":{"mode":"mtp","mtp":{"depth":1}}}"#,
                r#"{"id":2,"kind":"free_decode_run","count":4}"#,
            ],
        );
        let run_resp = responses
            .iter()
            .find(|r| r["id"] == 2)
            .expect("free_decode_run response");
        assert_eq!(run_resp["ok"], true, "{run_resp}");
        assert_eq!(run_resp["tokens"].as_array().unwrap().len(), 4);
        let text = run_resp.to_string();
        for forbidden in ["tok/s", "seconds", "elapsed", "speedup", "rate"] {
            assert!(
                !text.contains(forbidden),
                "derived metric leaked: {forbidden} in {text}"
            );
        }
        let begin = responses.iter().find(|r| r["id"] == 1).unwrap();
        assert_eq!(begin["effective_spec"]["mode"], "mtp");
        assert_eq!(begin["effective_spec"]["mtp"]["depth"], 1);
    }

    #[test]
    fn the_disagreement_count_reaches_the_wire_on_the_mtp_leg() {
        // bench db3b73e (pull request #258) seals this as
        // `spec_verify_replay_disagreements`. It is AUDIT-ONLY: it never fails
        // a leg here and nothing derives a rate from it.
        let session = Arc::new(Mutex::new(Scripted {
            mtp_armed: true,
            reject_cycles: vec![2, 3],
            disagree_cycles: vec![3],
            ..Default::default()
        }));
        let responses = run(
            session,
            &[
                r#"{"id":1,"kind":"free_decode_begin","seed_tokens":[1,2,3],"spec":{"mode":"mtp","mtp":{"depth":1}}}"#,
                r#"{"id":2,"kind":"free_decode_run","count":5}"#,
            ],
        );
        let run_resp = responses.iter().find(|r| r["id"] == 2).unwrap();
        assert_eq!(run_resp["ok"], true, "{run_resp}");
        assert_eq!(
            run_resp["verify_replay_disagreements"], 1,
            "the leg's delta reaches the wire: {run_resp}"
        );
        // THE INVARIANT benchd enforces, holding on a real leg.
        let rejected = run_resp["drafted_total"].as_i64().unwrap()
            - run_resp["accepted_total"].as_i64().unwrap();
        assert!(
            run_resp["verify_replay_disagreements"].as_i64().unwrap() <= rejected,
            "disagreements must be within the rejected drafts: {run_resp}"
        );
    }

    #[test]
    fn the_serial_leg_reports_no_disagreement_key_at_all() {
        // Absent is NOT `0`. The serial route reads no engine counters, so it
        // makes no claim about a comparison it never ran.
        let session = Arc::new(Mutex::new(Scripted::default()));
        let responses = run(
            session,
            &[
                r#"{"id":1,"kind":"free_decode_begin","seed_tokens":[1,2,3]}"#,
                r#"{"id":2,"kind":"free_decode_run","count":4}"#,
            ],
        );
        let run_resp = responses.iter().find(|r| r["id"] == 2).unwrap();
        assert_eq!(run_resp["ok"], true, "{run_resp}");
        assert!(
            run_resp.get("verify_replay_disagreements").is_none(),
            "the serial leg must omit the key, not zero it: {run_resp}"
        );
    }

    #[test]
    fn a_counter_that_overstates_itself_is_refused_by_name() {
        // Only a REJECTING round can disagree, so the count cannot exceed
        // `drafted_total - accepted_total`. benchd refuses such a leg
        // (VerifyReplayDisagreementsExceedRejected); the adapter names it
        // first, so the operator reads which counter is wrong instead of a
        // benchd consistency error.
        let session = Arc::new(Mutex::new(Scripted {
            mtp_armed: true,
            phantom_disagree_cycles: vec![2, 3, 4],
            ..Default::default()
        }));
        let responses = run(
            session,
            &[
                r#"{"id":1,"kind":"free_decode_begin","seed_tokens":[1,2,3],"spec":{"mode":"mtp","mtp":{"depth":1}}}"#,
                r#"{"id":2,"kind":"free_decode_run","count":5}"#,
            ],
        );
        let run_resp = responses.iter().find(|r| r["id"] == 2).unwrap();
        assert_eq!(run_resp["ok"], false, "{run_resp}");
        let err = run_resp["error"].as_str().unwrap();
        assert!(
            err.contains("verify_replay_disagreements") && err.contains("rejected drafts"),
            "the refusal must name the counter and the bound: {err}"
        );
    }
}

/// The RESIDENT-SERVER client: the session that reaches the model held by
/// `ds4-resident` instead of loading it. Driven against a scripted server on a
/// real Unix socket, so the framing, the argmax caching that keeps one round
/// trip per decoded token, and the fail-closed poisoning are all proven with
/// no GPU and no resident binary.
///
/// The resident binary itself, its one-load-per-window lifecycle and its phase
/// reset are proven end to end by `tools/test-ds4-resident.sh`, which runs the
/// REAL server source against a synthetic engine.
#[cfg(unix)]
mod resident {
    use std::io::{BufRead, BufReader, Write};
    use std::os::unix::net::{UnixListener, UnixStream};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::thread;

    use serde_json::{json, Value};

    use crate::ds4_backend::{Ds4Session, SpecCounters};
    use crate::resident::ResidentSession;

    /// A scripted resident: answers every request from `replies` keyed by op,
    /// counting what it was asked. Serves one connection, then exits.
    struct Server {
        path: std::path::PathBuf,
        requests: Arc<AtomicUsize>,
        handle: Option<thread::JoinHandle<Vec<Value>>>,
    }

    impl Server {
        fn start(reply_for: fn(&Value) -> Value) -> Self {
            let dir = std::env::temp_dir().join(format!(
                "ds4-resident-test-{}-{:?}",
                std::process::id(),
                thread::current().id()
            ));
            std::fs::create_dir_all(&dir).unwrap();
            let path = dir.join("sock");
            let _ = std::fs::remove_file(&path);
            let listener = UnixListener::bind(&path).unwrap();
            let requests = Arc::new(AtomicUsize::new(0));
            let counter = Arc::clone(&requests);
            let handle = thread::spawn(move || {
                let (stream, _) = listener.accept().unwrap();
                let mut writer = stream.try_clone().unwrap();
                let reader = BufReader::new(stream);
                let mut seen = Vec::new();
                for line in reader.lines() {
                    let line = match line {
                        Ok(line) => line,
                        Err(_) => break,
                    };
                    let request: Value = serde_json::from_str(&line).unwrap();
                    counter.fetch_add(1, Ordering::SeqCst);
                    let op = request["op"].as_str().unwrap_or("").to_string();
                    seen.push(request.clone());
                    let reply = reply_for(&request);
                    let mut out = reply.to_string();
                    out.push('\n');
                    if writer.write_all(out.as_bytes()).is_err() {
                        break;
                    }
                    let _ = writer.flush();
                    if op == "bye" {
                        break;
                    }
                }
                seen
            });
            Self {
                path,
                requests,
                handle: Some(handle),
            }
        }

        fn connect(&self) -> ResidentSession {
            ResidentSession::connect(self.path.to_str().unwrap()).unwrap()
        }

        fn requests(&self) -> usize {
            self.requests.load(Ordering::SeqCst)
        }

        fn finish(mut self) -> Vec<Value> {
            self.handle.take().unwrap().join().unwrap()
        }
    }

    /// The whole `ds4_shim.h` surface, answered with fixed values.
    fn scripted(request: &Value) -> Value {
        match request["op"].as_str().unwrap_or("") {
            "hello" => json!({
                "ok": true, "vocab_size": 1024, "eos_token": 99, "mtp_armed": true,
                "draft_tokens": 2, "ctx_size": 8192, "load_epoch": 4242,
                "ident": "ds4@abc123 nvcc=V13.0.88 mtp1", "model_path": "/w/target-00001-of-3.gguf",
                "mtp_head_path": "/w/mtp-head.gguf"
            }),
            "sync" => json!({"ok": true, "token": 501}),
            "eval" => json!({"ok": true, "token": 777}),
            "argmax" => json!({"ok": true, "token": 111}),
            "top_logits" => json!({
                "ok": true,
                "ids": [7, 8, 9, 10, 11, 12, 13, 14],
                "logits": [8.5, 7.5, 6.5, 5.5, 4.5, 3.5, 2.5, 1.5]
            }),
            "eval_speculative" => json!({"ok": true, "tokens": [40, 41], "token": 42}),
            "spec_counters" => json!({
                "ok": true, "drafts": 12, "hits": 5, "quenches": 0, "disagreements": 3
            }),
            _ => json!({"ok": true}),
        }
    }

    #[test]
    fn hello_carries_the_resident_engine_identity() {
        let server = Server::start(scripted);
        let session = server.connect();
        let hello = session.hello().clone();
        assert_eq!(hello.ident, "ds4@abc123 nvcc=V13.0.88 mtp1");
        assert_eq!(hello.load_epoch, 4242);
        assert!(hello.mtp_armed);
        assert_eq!(hello.eos_token, 99);
        assert_eq!(session.eos_token(), Some(99));
        assert!(
            session.connect.as_millis() < 2_000,
            "a resident reconnect must be milliseconds, took {:?}",
            session.connect
        );
        drop(session);
        server.finish();
    }

    /// THE ONE-LOAD PROOF REACHES THE SEALED ARTIFACT.
    ///
    /// The worker announces this string as its hello `backend`, and benchd
    /// seals it as `engine_backend`. It names the topology and carries the
    /// resident's `load_epoch` (the resident's pid), so every phase of one
    /// window seals the SAME value and a repeated load would seal a different
    /// one. Before this, `load_epoch` appeared nowhere in benchd
    /// (`git grep load_epoch` at the pinned dist tip returns nothing), so "the
    /// weights loaded once" was provable only from the resident log and
    /// `serve-identity.json`.
    #[test]
    fn the_attached_worker_announces_the_resident_and_its_load_epoch() {
        let server = Server::start(scripted);
        let session = server.connect();
        assert_eq!(
            session.hello().backend_string(),
            "ds4-resident load_epoch=4242 ds4@abc123 nvcc=V13.0.88 mtp1"
        );
        drop(session);
        server.finish();
    }

    #[test]
    fn every_verb_maps_to_one_shim_call() {
        let server = Server::start(scripted);
        let mut session = server.connect();
        session.sync(&[1, 2, 3]).unwrap();
        session.eval(9).unwrap();
        assert_eq!(session.top_logits(8).len(), 8);
        assert_eq!(session.eval_speculative(40, 4).unwrap(), vec![40, 41]);
        assert_eq!(
            session.spec_counters(),
            SpecCounters {
                drafts: 12,
                hits: 5,
                quenches: 0,
                verify_replay_disagreements: 3,
            },
            "every counter field crosses the resident socket, the diagnostic one included"
        );
        session.invalidate();
        drop(session);

        let seen = server.finish();
        let ops: Vec<&str> = seen.iter().map(|r| r["op"].as_str().unwrap()).collect();
        assert_eq!(
            ops,
            vec![
                "hello",
                "sync",
                "eval",
                "top_logits",
                "eval_speculative",
                "spec_counters",
                "invalidate",
                "bye",
            ]
        );
        assert_eq!(seen[1]["tokens"], json!([1, 2, 3]));
        assert_eq!(seen[2]["token"], 9);
        assert_eq!(seen[3]["k"], 8);
        assert_eq!(seen[4]["first_token"], 40);
        assert_eq!(seen[4]["budget"], 4);
    }

    /// ONE ROUND TRIP PER DECODED TOKEN. `free_decode_run`'s serial loop is
    /// `eval` then `argmax`; the eager argmax in the eval reply must make the
    /// second one free, or the timed window pays two round trips a token.
    #[test]
    fn argmax_after_a_state_advancing_verb_costs_no_round_trip() {
        let server = Server::start(scripted);
        let mut session = server.connect();
        assert_eq!(server.requests(), 1); // hello

        session.sync(&[1, 2, 3]).unwrap();
        assert_eq!(session.argmax(), 501);
        assert_eq!(session.argmax(), 501);
        assert_eq!(server.requests(), 2, "sync's argmax was re-requested");

        for _ in 0..10 {
            session.eval(5).unwrap();
            assert_eq!(session.argmax(), 777);
        }
        assert_eq!(
            server.requests(),
            12,
            "a decoded token cost more than one request"
        );

        session.eval_speculative(40, 4).unwrap();
        assert_eq!(
            session.argmax(),
            42,
            "the speculative frontier must come back with the cycle"
        );
        assert_eq!(server.requests(), 13);
        drop(session);
        server.finish();
    }

    /// An invalidate drops the cached frontier, so the next phase can never
    /// answer from the previous one's.
    #[test]
    fn invalidate_drops_the_cached_frontier() {
        let server = Server::start(scripted);
        let mut session = server.connect();
        session.sync(&[1, 2, 3]).unwrap();
        assert_eq!(session.argmax(), 501);
        session.invalidate();
        assert_eq!(
            session.argmax(),
            111,
            "argmax after invalidate must ask the engine"
        );
        drop(session);
        server.finish();
    }

    /// A refusal is the engine's own text, and it is not swallowed.
    #[test]
    fn a_refusal_carries_the_engine_text() {
        fn refusing(request: &Value) -> Value {
            if request["op"] == "sync" {
                json!({"ok": false, "error": "ds4_session_sync failed: context is full"})
            } else {
                scripted(request)
            }
        }
        let server = Server::start(refusing);
        let mut session = server.connect();
        let err = session.sync(&[1, 2, 3]).unwrap_err();
        assert!(err.contains("context is full"), "{err}");
        drop(session);
        server.finish();
    }

    /// THE TWO BACKEND PATHS MUST PRODUCE THE SAME f64. The linked backend
    /// widens the engine's `float` with `logit as f64`. The resident writes
    /// that same float as `%.9g` and this parses it back. A straight
    /// `as_f64()` would land on the nearest DOUBLE to the decimal, which is a
    /// different number from the float the engine held, and the correctness
    /// gate would then see one value over the socket and another in-process.
    #[test]
    fn wire_logits_widen_exactly_as_the_linked_backend_does() {
        // 0.1f32 is not representable in binary, so its %.9g form ("0.100000001")
        // and its widened f64 are different numbers. That makes it the case
        // that separates a correct parse from a plausible one.
        const ENGINE_FLOAT: f32 = 0.1;
        fn nine_g(request: &Value) -> Value {
            if request["op"] == "top_logits" {
                json!({
                    "ok": true,
                    "ids": [7, 8],
                    // What the resident's printf("%.9g") emits for 0.1f32.
                    "logits": [0.100000001_f64, 0.100000001_f64]
                })
            } else {
                scripted(request)
            }
        }
        let server = Server::start(nine_g);
        let mut session = server.connect();
        let got = session.top_logits(2);
        assert_eq!(got.len(), 2);
        let linked = ENGINE_FLOAT as f64;
        assert_eq!(
            got[0].logit, linked,
            "the wire path must give the same f64 the linked path gives"
        );
        // The negative control: parsing the decimal straight into f64 is a
        // DIFFERENT number, so the test above is not passing by accident.
        assert_ne!(
            0.100000001_f64, linked,
            "this case no longer separates the two widenings; pick another value"
        );
        drop(session);
        server.finish();
    }

    /// FAIL-CLOSED. A resident that dies mid-phase must never read as a fast
    /// phase: the infallible verbs return nothing usable and the failure comes
    /// back from the next fallible one.
    #[test]
    fn a_dead_resident_poisons_the_session() {
        let dir = std::env::temp_dir().join(format!("ds4-resident-dead-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("sock");
        let _ = std::fs::remove_file(&path);
        let listener = UnixListener::bind(&path).unwrap();
        let server = thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let mut writer = stream.try_clone().unwrap();
            let mut reader = BufReader::new(stream);
            let mut line = String::new();
            reader.read_line(&mut line).unwrap();
            let hello = scripted(&json!({"op": "hello"})).to_string();
            writer.write_all(format!("{hello}\n").as_bytes()).unwrap();
            writer.flush().unwrap();
            // and then the resident dies, mid-phase
        });
        let mut session = ResidentSession::connect(path.to_str().unwrap()).unwrap();
        server.join().unwrap();

        assert!(
            session.top_logits(8).is_empty(),
            "a dead resident must not produce logits"
        );
        let err = session.eval(1).unwrap_err();
        assert!(
            err.contains("resident") || err.contains("closed"),
            "the poison must name the resident: {err}"
        );
        let _ = std::fs::remove_file(&path);
    }

    /// A stream of half-lines is reassembled, and a reply is never split
    /// across two `read_line`s.
    #[test]
    fn replies_are_read_one_line_at_a_time() {
        let dir = std::env::temp_dir().join(format!("ds4-resident-chunk-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("sock");
        let _ = std::fs::remove_file(&path);
        let listener = UnixListener::bind(&path).unwrap();
        let server = thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let mut writer: UnixStream = stream.try_clone().unwrap();
            let mut reader = BufReader::new(stream);
            for reply in [
                scripted(&json!({"op": "hello"})).to_string(),
                json!({"ok": true, "token": 55}).to_string(),
            ] {
                let mut line = String::new();
                if reader.read_line(&mut line).unwrap() == 0 {
                    return;
                }
                let bytes = format!("{reply}\n");
                let (head, tail) = bytes.split_at(bytes.len() / 2);
                writer.write_all(head.as_bytes()).unwrap();
                writer.flush().unwrap();
                writer.write_all(tail.as_bytes()).unwrap();
                writer.flush().unwrap();
            }
        });
        let mut session = ResidentSession::connect(path.to_str().unwrap()).unwrap();
        session.eval(3).unwrap();
        assert_eq!(session.argmax(), 55);
        drop(session);
        server.join().unwrap();
        let _ = std::fs::remove_file(&path);
    }
}
