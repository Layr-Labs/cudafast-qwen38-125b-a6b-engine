#!/usr/bin/env bash
# benchmark.sh — the benchd FACADE impersonating the Swift reference benchmark.sh.
#
# RELOCATED COPY. This file used to live at benchd/scripts/benchmark.sh inside the
# benchd SOURCE submodule. The submodule is gone: benchd now ships as a pinned
# PREBUILT benchd (./tools/fetch-benchd.sh, channel-resolved), so there is no
# benchd/ checkout for the facade to live in and it is vendored here instead.
# Its UPSTREAM is mlxfast-bench scripts/benchmark.sh at the pinned commit
# (the dist manifest's `source_commit`); the body below is that file verbatim apart from three
# edits near "Dispatch to benchd": the BENCHD resolution, which now resolves the
# pinned binary instead of trusting whatever `benchd` is on PATH; the
# MLXFAST_QWEN_MTP_TRACK_ID export, which benchd iterate requires and which this
# engine repository exported nowhere; and the DECLARED SPEC + RESIDENT dispatch, because
# this track's engine arms its drafter at open and benchd spawns a worker per phase.
# Port changes from
# upstream rather than diverging here — the reference-parity comments below cite
# the Swift reference by line and are only true of the upstream bytes.
#
# This drop-in stands in, byte-for-byte on the observable shell contract, for the
# Swift reference:
#   ~/projects/layr-labs/mlxfast-challenge-dev/benchmark.sh
# (canonical, byte-identical across sibling checkouts). It does NOT re-implement the
# reference's transform / sandbox / weights-staging / cool-gate-polling machinery —
# `benchd iterate` owns the real run (spawns the MLX engine, runs correctness +
# timing, seals the score.json payload to stdout, writes <score>.sha256 and the
# per-mode benchmark-integrity sidecar). The facade only reproduces the observable
# SHELL contract around that run and then ADDS the human summary to stderr.
#
# Observable behaviors reproduced byte-for-byte from the reference (line cites are
# into the reference benchmark.sh):
#   - Arg parsing / mode selection, ref lines 25-138: the --weights/--golden/
#     --score-path shell-level rejection, --official / --local-cool-gate-only shell
#     selectors, the --official+local and --local-iterate+--local-submit combination
#     errors, and the "no mode given; defaulting to --local-iterate" STDOUT line.
#   - Golden preflight, ref lines 81-106: the missing-MLXFAST_CORRECTNESS_GOLDEN_PATH
#     heredoc and the golden-file-not-found two-liner. SHAPE is reproduced (same
#     conditions, same exit codes, same stream); the TEXT deliberately diverges --
#     the upstream strings name Qwen artifacts and told a participant of THIS track
#     to go find a "provisioned Qwen3.6 golden outside correctness_prompts/", which
#     is wrong here. They now name this track's target and the checked-in public
#     goldens. Nothing diffs these two messages against the reference (the byte-parity
#     bar applies to the no-mode STDOUT line at ref 75, noted below).
#   - Score/integrity path derivation, ref lines 92-138.
#   - The end-of-run human summary on STDERR: report_local_baseline_context
#     (ref 179-205) + report_local_score_summary (ref 219-295), reusing the reference
#     jq programs VERBATIM so the summary block is byte-identical.
#
# Every message keeps the literal `benchmark.sh:` prefix on purpose — the facade
# impersonates benchmark.sh; it does NOT rename messages to its own filename.
#
# --official is a FULL-PARITY surface (R22): like the reference, the facade runs
# --official anywhere it is given a pinned correctness golden. official mode routes to
# `benchd iterate --mode official` (transform-verified weights, correctness vs the
# supplied golden, cool gate ON operator-side, timed, sealed) — the SAME methodology the
# reference runs. `MLXFAST_BENCHMARK_SKIP_TIMED=1` (with `MLXFAST_BENCHMARK_CHECK_GATES=1`)
# makes it a GATES-ONLY seam-1 run producing a `partial_result=true` gates-score. The
# "private oracle" is just the supplied correctness golden INPUT (real ones come from R2 —
# read-only, sha256+bytes pin-verified before use); `MLXFAST_PRIVATE_DIR` is only a sandbox
# DENY rule, never a dependency. official mode's hard requirements are sandbox-on +
# runtime-worker (enforce_official_sandbox, byte-matched to the reference).
#
# Facade-specific behaviors (NOT in the reference; the reference had its own engine):
#   - MLXFAST_ENGINE_BIN must point at the MLX engine binary benchd spawns.
#   - Exit-code mapping: benchd usage/bad-args is native 2; the facade maps 2 -> 1
#     to match the reference (which uses exit 1 for usage). 0 and 1 pass through.
set -euo pipefail

LOCAL_ITERATE=0
LOCAL_SUBMIT=0
OFFICIAL=0
LOCAL_COOL_GATE_ONLY=0
# Arguments are consumed by benchd (not forwarded to a Swift CLI). --official is a
# shell-level mode selector only, so it is filtered out here, matching the reference.
for arg in "$@"; do
  case "${arg}" in
    --weights|--weights=*|--golden|--golden=*|--score-path|--score-path=*)
      echo "benchmark.sh: use MLXFAST_WEIGHTS_PATH, MLXFAST_CORRECTNESS_GOLDEN_PATH, or MLXFAST_SCORE_PATH for shell path overrides" >&2
      echo "benchmark.sh: pass --weights/--golden/--score-path only to .build/release/mlxfast-swift benchmark" >&2
      exit 1
      ;;
    --official)
      OFFICIAL=1
      continue
      ;;
    --local-cool-gate-only)
      LOCAL_COOL_GATE_ONLY=1
      continue
      ;;
  esac
  if [[ "${arg}" == "--local-iterate" ]]; then
    LOCAL_ITERATE=1
  fi
  if [[ "${arg}" == "--local-submit" ]]; then
    LOCAL_SUBMIT=1
  fi
done

if [[ "${OFFICIAL}" == "1" && ( "${LOCAL_ITERATE}" == "1" || "${LOCAL_SUBMIT}" == "1" ) ]]; then
  echo "benchmark.sh: --official cannot be combined with --local-iterate/--local-submit" >&2
  exit 1
fi

if [[ "${LOCAL_ITERATE}" == "1" && "${LOCAL_SUBMIT}" == "1" ]]; then
  echo "benchmark.sh: --local-iterate and --local-submit cannot be used together" >&2
  exit 1
fi

# Bare invocations default to the participant-friendly local edit loop (ref 67-79).
# The ranked full benchmark must be requested explicitly: with --official, or via the
# trusted workflow env (MLXFAST_OFFICIAL_BENCHMARK_RUN=1).
if [[ "${LOCAL_COOL_GATE_ONLY}" == "0" && "${LOCAL_ITERATE}" == "0" && "${LOCAL_SUBMIT}" == "0" && "${OFFICIAL}" == "0" ]]; then
  if [[ "${MLXFAST_OFFICIAL_BENCHMARK_RUN:-0}" == "1" ]]; then
    OFFICIAL=1
  else
    echo "benchmark.sh: no mode given; defaulting to --local-iterate (use --official for the ranked entrypoint, which requires the private oracle)"
    LOCAL_ITERATE=1
  fi
fi
# NOTE (R22): the no-mode STDOUT line above is byte-matched to the reference benchmark.sh
# (ref 75), including its "requires the private oracle" wording — compat-matrix.sh Part 1b
# diffs it against the reference, so byte-parity is the bar and the facade must not diverge
# from the reference bytes here. "private oracle" = the supplied correctness golden INPUT.

# --official is a FULL-PARITY surface (R22): no refusal. It routes to the benchd official
# backend below (MODE=official), exactly as the reference runs its own official path — a
# supplied pinned correctness golden is all official mode needs, plus sandbox-on +
# runtime-worker (enforce_official_sandbox, byte-matched to the reference).

# The --local-cool-gate-only selector is a bare cool-gate probe in the reference;
# the facade does not run engine work in that mode, and benchd exposes its own
# `--local-cool-gate-only` entrypoint, so there is nothing further for the wrapper
# to dispatch here.
if [[ "${LOCAL_COOL_GATE_ONLY}" == "1" ]]; then
  exit 0
fi

if [[ -z "${MLXFAST_CORRECTNESS_GOLDEN_PATH:-}" ]]; then
  cat >&2 <<'EOF'
benchmark.sh: this track requires an explicit correctness golden.

Set MLXFAST_CORRECTNESS_GOLDEN_PATH. There is no default: a golden is never
selected implicitly. For a local run, use a checked-in public fixture --

  --local-iterate  correctness_prompts/public_longcopy_gate_english_1024_256.json
  --local-submit   correctness_prompts/public_longcopy_gate_english_1024_1024.json

Both public fixtures were generated under the GEMMA 4 tokenizer and carry
Gemma provenance. The pinned target is
the Qwen 3.8 Flash Next GGUF (model_type qwen4_exp_text), so the
model-identity golden loader REJECTS them. That refusal is correct and it
stands until the goldens are regenerated on ranked hardware
(docs/qwen38-125b-a6b-port-notes.md section 5).
Official (--official) runs take an organizer-provisioned hidden golden
instead; those are not in this repository.
EOF
  exit 1
fi

if [[ "${LOCAL_ITERATE}" == "1" && -z "${MLXFAST_SCORE_PATH:-}" ]]; then
  SCORE_PATH="score.local-iterate.json"
else
  SCORE_PATH="${MLXFAST_SCORE_PATH:-score.json}"
fi
WEIGHTS_PATH="${MLXFAST_WEIGHTS_PATH:-weights}"
GOLDEN_PATH="${MLXFAST_CORRECTNESS_GOLDEN_PATH:-}"
if [[ "${LOCAL_ITERATE}" == "1" && -z "${MLXFAST_INTEGRITY_PATH:-}" ]]; then
  INTEGRITY_PATH="benchmark-integrity.local-iterate.json"
else
  INTEGRITY_PATH="${MLXFAST_INTEGRITY_PATH:-benchmark-integrity.json}"
fi

# Fail fast with actionable guidance when the golden fixture is missing, BEFORE any
# run work (ref 100-106).
if [[ ! -f "${GOLDEN_PATH}" ]]; then
  echo "benchmark.sh: correctness golden not found at ${GOLDEN_PATH}" >&2
  echo "benchmark.sh: check MLXFAST_CORRECTNESS_GOLDEN_PATH; there is no fallback golden." >&2
  exit 1
fi

# R22 / R2-mirror pin (fail-closed): the supplied correctness golden is the official INPUT.
# The REAL official goldens come from R2 (read-only, integrity-pinned) — pending: R2/.env
# location (David). When the caller pins the golden (sha256 and/or byte count), VERIFY the
# supplied file against the pin BEFORE any run trusts it; a mismatch is fatal (never run
# against an unpinned/altered golden). When BOTH sha256 and bytes are pinned, the official
# dispatch also forwards them to benchd as --golden-sha256/--golden-bytes, which re-verifies
# the raw bytes before parsing — so this shell check is a loud early gate, not the only one.
if [[ -n "${MLXFAST_CORRECTNESS_GOLDEN_SHA256:-}" ]]; then
  if ! command -v shasum >/dev/null 2>&1; then
    echo "benchmark.sh: shasum is required to verify MLXFAST_CORRECTNESS_GOLDEN_SHA256" >&2
    exit 1
  fi
  golden_actual_sha256="$(shasum -a 256 "${GOLDEN_PATH}" | awk '{print $1}')"
  if [[ "${golden_actual_sha256}" != "${MLXFAST_CORRECTNESS_GOLDEN_SHA256}" ]]; then
    echo "benchmark.sh: correctness golden sha256 mismatch — pin=${MLXFAST_CORRECTNESS_GOLDEN_SHA256} actual=${golden_actual_sha256} (${GOLDEN_PATH}); refusing to run against an unpinned golden" >&2
    exit 1
  fi
fi
if [[ -n "${MLXFAST_CORRECTNESS_GOLDEN_BYTES:-}" ]]; then
  golden_actual_bytes="$(wc -c < "${GOLDEN_PATH}" | tr -d '[:space:]')"
  if [[ "${golden_actual_bytes}" != "${MLXFAST_CORRECTNESS_GOLDEN_BYTES}" ]]; then
    echo "benchmark.sh: correctness golden byte count mismatch — pin=${MLXFAST_CORRECTNESS_GOLDEN_BYTES} actual=${golden_actual_bytes} (${GOLDEN_PATH}); refusing to run against an unpinned golden" >&2
    exit 1
  fi
fi

# Local modes print the same-machine baseline snapshot (when one exists) BEFORE the
# run starts. Copied VERBATIM from reference benchmark.sh lines 179-205 (jq program
# 188-200) so the "baseline to beat" line is byte-identical. Diagnostic only: any
# failure here must never fail the benchmark run.
report_local_baseline_context() {
  if [[ "${LOCAL_ITERATE}" != "1" && "${LOCAL_SUBMIT}" != "1" ]]; then
    return 0
  fi
  local baseline_path="${SCORE_PATH%.json}.baseline.json"
  if [[ ! -f "${baseline_path}" ]]; then
    return 0
  fi
  local context
  context="$(jq -r '
    def r3: . * 1000 | round / 1000;
    def r6: . * 1000000 | round / 1000000;
    (.metrics // {}) as $m
    | ($m.prefill_seconds_per_token // 0) as $p
    | ($m.decode_seconds_per_token // 0) as $d
    | select($p > 0 and $d > 0)
    | ($m.prefill_speedup // 0) as $ps
    | ($m.decode_speedup // 0) as $ds
    | (if $ps > 0 and $ds > 0 then pow($ds; 0.75) * pow($ps; 0.25) else 0 end) as $est
    | "prefill \($p | r6) s/token, decode \($d | r6) s/token"
      + (if $est > 0 then ", est score \($est | r3)" else "" end)
  ' "${baseline_path}" 2>/dev/null || true)"
  if [[ -n "${context}" ]]; then
    echo "benchmark.sh: local baseline to beat (${baseline_path}): ${context}" >&2
  fi
  return 0
}

# Local modes end with a compact human-readable summary on stderr. Copied VERBATIM
# from reference benchmark.sh lines 219-295 (jq programs 229-244 + 261-287 and the
# echo/printf framing 248-294) so the summary block is byte-identical. Diagnostic
# only: any failure here must never fail the benchmark run.
report_local_score_summary() {
  if [[ "${LOCAL_ITERATE}" != "1" && "${LOCAL_SUBMIT}" != "1" ]]; then
    return 0
  fi
  local mode_name="local-submit"
  if [[ "${LOCAL_ITERATE}" == "1" ]]; then
    mode_name="local-iterate"
  fi

  local summary
  summary="$(jq -r '
    def r3: . * 1000 | round / 1000;
    def r6: . * 1000000 | round / 1000000;
    (.metrics // {}) as $m
    | ($m.prefill_seconds_per_token // 0) as $p
    | ($m.decode_seconds_per_token // 0) as $d
    | select($p > 0 and $d > 0)
    | ($m.prefill_speedup // 0) as $ps
    | ($m.decode_speedup // 0) as $ds
    | (if $ps > 0 and $ds > 0 then pow($ds; 0.75) * pow($ps; 0.25) else 0 end) as $est
    | "  prefill \($p | r6) s/token  speedup \($ps | r3)x\n"
      + "  decode  \($d | r6) s/token  speedup \($ds | r3)x"
      + (if $est > 0
         then "\n  est score \($est | r3) (decode_speedup^0.75 * prefill_speedup^0.25; official score comes from the ranked runner)"
         else "" end)
  ' "${SCORE_PATH}" 2>/dev/null || true)"
  if [[ -z "${summary}" ]]; then
    return 0
  fi
  {
    echo "benchmark.sh: ${mode_name} summary"
    printf '%s\n' "${summary}"
  } >&2

  local baseline_path="${SCORE_PATH%.json}.baseline.json"
  if [[ ! -f "${baseline_path}" ]]; then
    if [[ "${LOCAL_ITERATE}" == "1" ]]; then
      echo "benchmark.sh: no local baseline at ${baseline_path}; run 'cp ${SCORE_PATH} ${baseline_path}' to compare future runs" >&2
    fi
    return 0
  fi
  local compare
  compare="$(jq -r -n --slurpfile cur "${SCORE_PATH}" --slurpfile base "${baseline_path}" '
    def r1: . * 10 | round / 10;
    def r3: . * 1000 | round / 1000;
    def r6: . * 1000000 | round / 1000000;
    def sign: if . >= 0 then "+" else "" end;
    ($cur[0].metrics // {}) as $c
    | ($base[0].metrics // {}) as $b
    | ($c.prefill_seconds_per_token // 0) as $cp
    | ($c.decode_seconds_per_token // 0) as $cd
    | ($b.prefill_seconds_per_token // 0) as $bp
    | ($b.decode_seconds_per_token // 0) as $bd
    | select($cp > 0 and $cd > 0 and $bp > 0 and $bd > 0)
    | (($cp - $bp) / $bp * 100) as $pdelta
    | (($cd - $bd) / $bd * 100) as $ddelta
    | ($c.prefill_speedup // 0) as $cps
    | ($c.decode_speedup // 0) as $cds
    | ($b.prefill_speedup // 0) as $bps
    | ($b.decode_speedup // 0) as $bds
    | (if $cps > 0 and $cds > 0 then pow($cds; 0.75) * pow($cps; 0.25) else 0 end) as $cest
    | (if $bps > 0 and $bds > 0 then pow($bds; 0.75) * pow($bps; 0.25) else 0 end) as $best
    | "    prefill \($bp | r6) -> \($cp | r6) s/token (\($pdelta | sign)\($pdelta | r1)%)\n"
      + "    decode  \($bd | r6) -> \($cd | r6) s/token (\($ddelta | sign)\($ddelta | r1)%)"
      + (if $cest > 0 and $best > 0
         then (((($cest - $best) / $best) * 100) as $edelta
           | "\n    est score \($best | r3) -> \($cest | r3) (\($edelta | sign)\($edelta | r1)%)")
         else "" end)
  ' 2>/dev/null || true)"
  if [[ -z "${compare}" ]]; then
    return 0
  fi
  {
    echo "benchmark.sh: vs ${baseline_path} (negative s/token deltas = faster)"
    printf '%s\n' "${compare}"
  } >&2
}

# official mode (R22) uses the runtime-worker sandbox. Default ON (ref 139); a real official
# run FAILS CLOSED without it (enforce_official_sandbox below + benchd resolve_official_sandbox).
USE_RUNTIME_WORKER="${MLXFAST_USE_RUNTIME_WORKER:-1}"

# official mode hard requirements — byte-matched to the reference enforce_official_sandbox
# (ref 666-678): sandbox ON + runtime worker. The two error strings are byte-identical to the
# reference; the facade fires them for its official mode (OFFICIAL=1, set by --official or the
# trusted MLXFAST_OFFICIAL_BENCHMARK_RUN=1 env). MLXFAST_PRIVATE_DIR is only a sandbox DENY rule
# inside the worker profile, never a dependency, so it is NOT required here.
enforce_official_sandbox() {
  if [[ "${OFFICIAL}" != "1" ]]; then
    return 0
  fi
  if [[ "${MLXFAST_NO_SANDBOX:-0}" == "1" ]]; then
    echo "benchmark.sh: official GitHub benchmark runs must not set MLXFAST_NO_SANDBOX=1" >&2
    exit 1
  fi
  if [[ "${USE_RUNTIME_WORKER}" != "1" ]]; then
    echo "benchmark.sh: official GitHub benchmark runs must use the runtime worker sandbox" >&2
    exit 1
  fi
}

# ---- Dispatch to benchd -------------------------------------------------------
# The MLX engine binary (Engine Protocol v1) benchd spawns. Facade-specific and
# REQUIRED for a real run — the Swift binary was its own engine; benchd is not.
if [[ -z "${MLXFAST_ENGINE_BIN:-}" ]]; then
  echo "benchmark.sh: MLXFAST_ENGINE_BIN must point to the benchd engine binary" >&2
  exit 1
fi

FACADE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# DIVERGENCE FROM UPSTREAM (edit 1 of 3): the TRACK ID. `benchd iterate` REQUIRES
# MLXFAST_QWEN_MTP_TRACK_ID in EVERY mode -- its `-{platform}-v{N}` suffix keys the
# official baseline pair and the acceptance bands, and the official path resolves it
# BEFORE the gates-only branch, so `--local-iterate` and `--official` alike refused with
# "no track_id" while this repository exported it nowhere. The value is the trusted
# manifest's `trackId` (the PLATFORM id, qwen3.8-125b-a6b-cuda-v1 -- NOT the bench
# channel `qwen3.8-125b-a6b-v1`, which names the PROJECT; see tools/fetch-benchd.sh).
# It is read from benchmark.json beside this file's parent, never taken from the
# caller: benchd treats env != contract as a hard error, so a caller that pre-set a
# DIFFERENT value is refused here rather than silently overridden, and a manifest
# with no trackId refuses rather than letting benchd guess a platform.
MANIFEST_TRACK_ID="$(jq -r '.trackId // empty' "${FACADE_DIR}/../benchmark.json" 2>/dev/null || true)"
if [[ -z "${MANIFEST_TRACK_ID}" ]]; then
  echo "benchmark.sh: benchmark.json carries no trackId; benchd iterate requires MLXFAST_QWEN_MTP_TRACK_ID and this facade will not guess one" >&2
  exit 1
fi
if [[ -n "${MLXFAST_QWEN_MTP_TRACK_ID:-}" && "${MLXFAST_QWEN_MTP_TRACK_ID}" != "${MANIFEST_TRACK_ID}" ]]; then
  echo "benchmark.sh: MLXFAST_QWEN_MTP_TRACK_ID is set to '${MLXFAST_QWEN_MTP_TRACK_ID}' but benchmark.json trackId is '${MANIFEST_TRACK_ID}'; the track id is ONE value and this facade will not override either" >&2
  exit 1
fi
export MLXFAST_QWEN_MTP_TRACK_ID="${MANIFEST_TRACK_ID}"

# DIVERGENCE FROM UPSTREAM (edit 2 of 3): the BENCHD resolution. Upstream
# defaults to whatever `benchd` is first on PATH; here the default is the
# binary fetch-benchd.sh has verified against the channel manifest, so a bare run cannot
# score itself with an unpinned harness. fetch-benchd.sh is a no-op when
# benchd-bin/benchd is already present and matching, so this costs one hash on
# the warm path and works fully offline. An explicit BENCHD= override is still
# honoured (benchd development, bisecting a harness change) and is deliberately
# NOT hash-checked — it is the caller saying "I mean this other binary".
if [[ -z "${BENCHD:-}" ]]; then
  BENCHD="$("${FACADE_DIR}/fetch-benchd.sh")"
fi
if [[ "${OFFICIAL}" == "1" ]]; then
  MODE="official"
elif [[ "${LOCAL_ITERATE}" == "1" ]]; then
  MODE="local-iterate"
else
  MODE="local-submit"
fi

# Note the sidecar the facade would use for its own bookkeeping; benchd writes the
# authoritative per-mode integrity sidecar itself, so the facade does not re-seal it.
: "${INTEGRITY_PATH}"

# Print the same-machine baseline snapshot (when one exists) before the run starts,
# matching the reference ordering (ref 1236). Diagnostic only.
report_local_baseline_context || true

# Assemble the benchd iterate argv. official mode routes to benchd's official backend
# (`--mode official`) with the FULL methodology (sandboxed workers, correctness vs the supplied
# golden, official floor/band gating, timed + seal; MLXFAST_BENCHMARK_SKIP_TIMED=1 makes it a
# gates-only seam-1 run). The benchd official backend loads --weights DIRECTLY — no reference-
# checkpoint regeneration — so the facade needs no cached reference weights. Official has no local
# cool gate (its gate is operator-side / thermal), so --cool-gate is LOCAL-only. When the golden
# is fully pinned (sha256 AND bytes), forward the pin so benchd re-verifies the raw bytes.
benchd_args=(
  --engine "${MLXFAST_ENGINE_BIN}"
  --weights "${WEIGHTS_PATH}"
  --golden "${GOLDEN_PATH}"
  --mode "${MODE}"
  --score-path "${SCORE_PATH}"
)
if [[ -n "${MLXFAST_CORRECTNESS_GOLDEN_SHA256:-}" && -n "${MLXFAST_CORRECTNESS_GOLDEN_BYTES:-}" ]]; then
  benchd_args+=(--golden-sha256 "${MLXFAST_CORRECTNESS_GOLDEN_SHA256}" --golden-bytes "${MLXFAST_CORRECTNESS_GOLDEN_BYTES}")
fi
if [[ "${OFFICIAL}" == "1" ]]; then
  enforce_official_sandbox
  export MLXFAST_USE_RUNTIME_WORKER="${USE_RUNTIME_WORKER}"
  # ARM GATE (David 2026-08-26): --mode official REQUIRES the track contract fixture whose
  # official_scoring_enabled arms scoring. benchd refuses to seal an official artifact unless the
  # fixture declares official_scoring_enabled: true, so this is the fail-closed arm surface. The
  # fixture is fixtures/qwen3_8_125b_a6b_track.json (the scoring-semantics authority per CLAUDE.md).
  # It is false today, so official runs refuse until the arming change flips it true.
  CONTRACT_FIXTURE="${FACADE_DIR}/../fixtures/qwen3_8_125b_a6b_track.json"
  if [[ ! -f "${CONTRACT_FIXTURE}" ]]; then
    echo "benchmark.sh: official runs require the track contract fixture at ${CONTRACT_FIXTURE}" >&2
    exit 1
  fi
  benchd_args+=(--contract "${CONTRACT_FIXTURE}")
else
  benchd_args+=(--cool-gate)
fi

# DIVERGENCE FROM UPSTREAM (edit 3 of 3): THE DECLARED SPEC AND THE ONE
# LOAD.
#
# Upstream's participant loop drove an in-process Swift engine: nothing to boot,
# and the draft depth rode the Engine Protocol wire per request. This track's
# engine is ds4. It arms its MTP drafter when the ENGINE OPENS, and benchd spawns
# a fresh worker for every phase. So the facade has to do the same two things
# tools/qwen38-125b-a6b-measure-and-score.sh does on the ranked path, or the
# entry this repository documents as the measured one measures something the
# ranked run does not:
#
#   1. REQUEST THE DECLARED DEPTH. benchd sends free_decode_begin with no spec
#      unless it is told one, and the adapter then resolves SERIAL. Without
#      --mtp-depth N an mtp declaration was timed as a serial leg, and the run
#      still reported itself as that declaration.
#   2. BOOT ONE RESIDENT FOR THE WINDOW. Without tools/serve-up.sh nothing
#      exports DS4_RESIDENT_SOCKET, so every phase's worker opens the artifact
#      again and the local numbers carry a per-phase reload the ranked numbers
#      do not.
#
# The derivation is tools/spec-declaration.sh -- the ONE deriver the serve, the
# ranked script and the preflight all read -- so the depth the engine arms and
# the depth benchd requests cannot disagree.
SPEC_DESC="$("${FACADE_DIR}/spec-declaration.sh" describe)"
SPEC_SPECULATIVE="$("${FACADE_DIR}/spec-declaration.sh" speculative)"
SPEC_DRAFT_LEN="$("${FACADE_DIR}/spec-declaration.sh" draft-len)"
if [[ "${SPEC_DESC}" != "serial" ]]; then
  # A benchd that does not know --mtp-depth cannot run an MTP leg at all.
  # That is a refusal, never a silent serial run under an mtp declaration.
  if "${BENCHD}" iterate --help 2>&1 | grep -q -- '--mtp-depth'; then
    benchd_args+=(--mtp-depth "${SPEC_DRAFT_LEN}")
  else
    echo "benchmark.sh: the pinned benchd does not accept --mtp-depth, so it cannot request the declared ${SPEC_DESC} leg; refusing rather than measuring a serial leg under an ${SPEC_DESC} declaration" >&2
    exit 1
  fi
fi
echo "benchmark.sh: declared spec ${SPEC_DESC} (mtp-head.manifest.json, derived by tools/spec-declaration.sh)" >&2

# Run benchd. Do NOT redirect its stdout: benchd emits the sealed JSON payload on
# stdout and the facade passes it through untouched (no cat / re-emit / re-seal).
# Capture the exit code without tripping `set -e`.
#
# tools/serve-up.sh boots ONE ds4-resident for this window -- the process that
# owns the weights -- and exports the engine environment every spawned
# cuda-engine reads. serve-up.sh runs the command and exits with its code, so the
# exit-code mapping below is unchanged. A caller that already exported DS4_MODEL
# is driving the engine in-process (a single-shot diagnostic) and owns its own
# model, so it is honoured as-is with no resident -- the same rule
# tools/qwen38-125b-a6b-measure-and-score.sh applies on the ranked path.
rc=0
if [[ -n "${DS4_MODEL:-}" ]]; then
  "${BENCHD}" iterate "${benchd_args[@]}" || rc=$?
else
  env SERVE_UP_WEIGHTS_DIR="${WEIGHTS_PATH}" \
      SERVE_UP_SPECULATIVE="${SPEC_SPECULATIVE}" \
      SERVE_UP_SPEC_DRAFT_LEN="${SPEC_DRAFT_LEN}" \
      "${FACADE_DIR}/serve-up.sh" \
      "${BENCHD}" iterate "${benchd_args[@]}" || rc=$?
fi

# Diagnostic-only summary — must NEVER change the run's exit code.
report_local_score_summary || true

# Exit-code mapping: benchd iterate returns 0 = pass, 1 =
# fail/exec-error, 2 = usage/bad-args. Pass 0 and 1 through; map benchd's native
# usage 2 -> 1 (the reference benchmark.sh uses exit 1 for usage).
if [[ "${rc}" == "2" ]]; then
  exit 1
fi
exit "${rc}"
