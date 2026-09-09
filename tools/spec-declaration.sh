#!/usr/bin/env bash
#
# spec-declaration.sh -- the SINGLE trusted source that reads the participant's
# speculative-decode DECLARATION and derives the serve-boot speculative config
# from it. tools/serve-up.sh, .github/workflows/benchmark.yml and
# tools/ranked-box-preflight.sh ALL resolve the serve spec THROUGH this one
# script, so the fail-closed validation and the derivation can never drift
# between the arm gate and the serve that actually boots.
#
# NOT AN EDITABLE PATH. benchmark.json editablePaths is
# {ds4, harness, mtp-head, mtp-head.manifest.json}. The DECLARATION (mtp-head.manifest.json) is editable
# -- a submission EXPRESSES a value there -- but this DERIVATION is trusted, so a
# submission cannot rewrite how the value is interpreted or relax the envelope.
#
# WHY THIS EXISTS ON THE CUDA TRACK. On the MLX sibling the runtime worker is
# in-process, so the draft depth rides the Engine Protocol wire per request
# (Sources/MLXFastHarness/RuntimeWorkerSpecConfig.swift) and the engine echoes
# `effective_spec`. On this CUDA track the ds4 engine arms its embedded MTP
# drafter when it OPENS (`mtp_draft_tokens`, exported as DS4_MTP_DRAFT_TOKENS by
# tools/serve-up.sh), so the declaration -> engine-config bridge that the
# in-process worker did not need is the piece the engine swap left for the
# serve path. This script IS that bridge. The echo half survives
# in-tree unchanged: harness/protocol-adapter (adapter.rs `resolve_spec` ->
# `effective_spec`, engine.rs `EffectiveSpec`/`DepthOutOfEnvelope`) still resolves
# a requested depth against the envelope and echoes what will run, and benchd
# seals that echo.
#
# THE DECLARATION is the serve knob, in mtp-head.manifest.json:
#
#   "spec": { "enabled": <bool>, "num_speculative_tokens": <int> }
#
# ABSENT `spec` block, `enabled: false`, or `num_speculative_tokens: 0` => SERIAL
# (DS4_MTP_DRAFT_TOKENS=1), which is BIT-FOR-BIT the launch-reference behaviour
# (David MTP-0 ruling). `enabled: true` with N in the track's permitted draft
# depths => the engine opens with the drafter armed at N (DS4_MTP_DRAFT_TOKENS=N+1).
#
# THE REST OF THE DECLARATION is validated here too, because this is the one
# trusted reader of the file (issue #24 work item A). docs/participant-contract.md
# 4.1 and 4.3 promise that `pinned` is the only accepted source, that the 2 GiB
# `max_bytes` cap may be LOWERED and never raised, and that there is no `arm`
# key -- there is one arm, so there is nothing to select. Only the `spec` block
# was ever read, so `"source": "remote"`, `max_bytes` past the cap and an
# unknown top-level key all passed a `validate`.
#
# VALIDATION is fail-closed. A REFUSAL (exit 1, message on stderr) for:
#   * a declaration that is not a JSON object, or carries an unknown top-level
#     key (allowed: version, source, max_bytes, bytes, sha256, spec);
#   * a `source` other than "pinned" -- "remote" and "in_branch" by name;
#   * a `max_bytes` that is not an integer in 1..2147483648;
#   * a `spec` block that is not an object, or carries an unknown key;
#   * `enabled` that is not a boolean, or `num_speculative_tokens` not an integer;
#   * a value outside the structural range 0..8 (the a8/David sanity ceiling);
#   * an ENABLED depth that is not in the contract's permitted_draft_depths
#     (fixtures/qwen3_8_125b_a6b_track.json `mtp_head.permitted_draft_depths`,
#     the same envelope harness/protocol-adapter/src/ds4_backend.rs mirrors as
#     MTP_MIN_DEPTH..=MTP_MAX_DEPTH). An out-of-envelope depth is REFUSED, never
#     clamped -- so the serve that boots and the `effective_spec` the adapter
#     seals can never disagree, matching the sibling's resolveDepth discipline.
#
# Usage:
#   tools/spec-declaration.sh speculative   # 0 (serial) or 1 (mtp)
#   tools/spec-declaration.sh draft-len     # num_speculative_tokens (0 = serial)
#   tools/spec-declaration.sh describe      # "serial" or "mtp<N>" (identity label)
#   tools/spec-declaration.sh validate      # exit 0 if valid, else refuse
# Every subcommand validates the declaration first and refuses on an invalid one.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
MANIFEST="${SPEC_DECLARATION_MANIFEST:-${REPO_ROOT}/mtp-head.manifest.json}"
CONTRACT="${SPEC_DECLARATION_CONTRACT:-${REPO_ROOT}/fixtures/qwen3_8_125b_a6b_track.json}"

# The structural sanity ceiling for a declared draft length (a8/David pin: the
# knob accepts a declared integer 0..8, not a hardcoded value). The contract's
# permitted_draft_depths is the tighter, AUTHORITATIVE set enforced below when
# the value is enabled; this is the outer type/range guard around it.
SPEC_MAX_TOKENS=8

# The track's declaration byte cap, 2 GiB (docs/participant-contract.md 4.3). A
# declaration may state a LOWER max_bytes; it may not raise this one.
DECLARATION_MAX_BYTES=2147483648

fail() {
  echo "spec-declaration.sh: REFUSING -- $*" >&2
  exit 1
}

command -v jq >/dev/null 2>&1 || fail "jq is required to read the declaration"

# --- read the declaration ---------------------------------------------------
# An absent manifest OR an absent `spec` block is SERIAL: the pure no-op. It is
# resolved WITHOUT requiring the file to exist, so the stock-repo derivation
# never depends on a spec block being present.
enabled="false"
raw_tokens="0"
if [[ -f "${MANIFEST}" ]]; then
  jq -e . >/dev/null 2>&1 < "${MANIFEST}" || fail "mtp-head.manifest.json is not valid JSON"

  # --- the whole declaration, not just the spec block ------------------------
  # Everything the file promises is checked, in the same fail-closed style as
  # the spec block below. An ABSENT key is fine throughout: an absent
  # declaration, an absent source and an absent spec block are all the default
  # (organizer-pinned head, serial), which is what most submissions carry.
  [[ "$(jq -r 'type' "${MANIFEST}")" == "object" ]] \
    || fail "mtp-head.manifest.json must be a JSON object with keys {version, source, max_bytes, bytes, sha256, spec}"

  # A typo'd or invented top-level key must not read as its default, exactly as
  # for the spec block. `arm` is the one worth naming: this track has a single
  # speculative arm, so a declaration that tries to select one is stating
  # something the runner will not honour, and silence there is the wrong answer.
  unknown_top="$(jq -r 'keys[] | select(. != "version" and . != "source" and . != "max_bytes" and . != "bytes" and . != "sha256" and . != "spec")' "${MANIFEST}")"
  [[ -z "${unknown_top}" ]] \
    || fail "the declaration carries unknown top-level key(s): $(printf '%s' "${unknown_top}" | tr '\n' ' '); allowed keys are version, source, max_bytes, bytes, sha256, spec"

  if [[ "$(jq -r 'has("source")' "${MANIFEST}")" == "true" ]]; then
    declared_source="$(jq -r '.source | if type == "string" then . else tojson end' "${MANIFEST}")"
    [[ "${declared_source}" == "pinned" ]] \
      || fail "source \"${declared_source}\" is not accepted; the head is the organizer-staged pinned head, so the only accepted source is \"pinned\""
  fi

  if [[ "$(jq -r 'has("max_bytes")' "${MANIFEST}")" == "true" ]]; then
    # jq does the range test, not bash arithmetic: a 30-digit literal would wrap
    # silently in `(( ))` and could land back inside the range.
    declared_max="$(jq -r '.max_bytes | if type == "number" then tostring else tojson end' "${MANIFEST}")"
    [[ "$(jq -r --argjson cap "${DECLARATION_MAX_BYTES}" \
          '.max_bytes | (type == "number" and . == floor and . >= 1 and . <= $cap)' "${MANIFEST}")" == "true" ]] \
      || fail "max_bytes ${declared_max} is not an integer in 1..${DECLARATION_MAX_BYTES}; a declaration may lower the track's 2 GiB cap and may not raise it"
  fi

  if [[ "$(jq -r 'has("spec")' "${MANIFEST}")" == "true" ]]; then
    [[ "$(jq -r '.spec | type' "${MANIFEST}")" == "object" ]] \
      || fail "the \"spec\" declaration must be an object with keys {enabled, num_speculative_tokens}"
    # A typo'd key must not silently read as its default: config drift is a
    # refusal, matching the sibling's rejectUnknownSpecKeys.
    unknown="$(jq -r '.spec | keys[] | select(. != "enabled" and . != "num_speculative_tokens")' "${MANIFEST}")"
    [[ -z "${unknown}" ]] \
      || fail "the \"spec\" declaration carries unknown key(s): $(printf '%s' "${unknown}" | tr '\n' ' '); allowed keys are enabled, num_speculative_tokens"
    if [[ "$(jq -r '.spec | has("enabled")' "${MANIFEST}")" == "true" ]]; then
      [[ "$(jq -r '.spec.enabled | type' "${MANIFEST}")" == "boolean" ]] \
        || fail "spec.enabled must be a boolean (got $(jq -r '.spec.enabled | type' "${MANIFEST}"))"
      enabled="$(jq -r '.spec.enabled' "${MANIFEST}")"
    fi
    if [[ "$(jq -r '.spec | has("num_speculative_tokens")' "${MANIFEST}")" == "true" ]]; then
      [[ "$(jq -r '.spec.num_speculative_tokens | type' "${MANIFEST}")" == "number" ]] \
        || fail "spec.num_speculative_tokens must be an integer (got $(jq -r '.spec.num_speculative_tokens | type' "${MANIFEST}"))"
      raw_tokens="$(jq -r '.spec.num_speculative_tokens' "${MANIFEST}")"
    fi
  fi
fi

# integer + structural 0..SPEC_MAX_TOKENS ceiling
printf '%s' "${raw_tokens}" | grep -Eq '^-?[0-9]+$' \
  || fail "spec.num_speculative_tokens must be an integer (got '${raw_tokens}')"
if (( raw_tokens < 0 || raw_tokens > SPEC_MAX_TOKENS )); then
  fail "spec.num_speculative_tokens=${raw_tokens} is outside the permitted range 0..${SPEC_MAX_TOKENS}"
fi

# --- derive the effective serve spec ----------------------------------------
# enabled:false OR num_speculative_tokens 0 => serial (the no-op). enabled:true
# with N>0 must be a CONTRACT-permitted depth.
spec="0"
draft="0"
if [[ "${enabled}" == "true" && "${raw_tokens}" -gt 0 ]]; then
  # permitted_draft_depths is the fixture's authority (the same envelope
  # harness/protocol-adapter/src/ds4_backend.rs mirrors as MTP_MIN_DEPTH..=
  # MTP_MAX_DEPTH). Enforce membership so the serve can never boot a depth the
  # scored envelope would refuse. When the contract declares no such set, the
  # structural 0..SPEC_MAX_TOKENS ceiling above stands alone.
  if jq -e '.mtp_head.permitted_draft_depths | arrays' >/dev/null 2>&1 < "${CONTRACT}"; then
    if [[ "$(jq -r --argjson n "${raw_tokens}" '(.mtp_head.permitted_draft_depths | index($n)) != null' "${CONTRACT}")" != "true" ]]; then
      permitted="$(jq -r '.mtp_head.permitted_draft_depths | map(tostring) | join(", ")' "${CONTRACT}")"
      fail "spec.num_speculative_tokens=${raw_tokens} is not a contract-permitted draft depth (permitted_draft_depths: ${permitted})"
    fi
  fi
  spec="1"
  draft="${raw_tokens}"
fi

case "${1:-}" in
  speculative) echo "${spec}" ;;
  draft-len)   echo "${draft}" ;;
  describe)    if [[ "${spec}" == "1" ]]; then echo "mtp${draft}"; else echo "serial"; fi ;;
  validate)    : ;;  # validation already ran above; a clean exit means valid
  *)
    echo "spec-declaration.sh: usage: spec-declaration.sh {speculative|draft-len|describe|validate}" >&2
    exit 2
    ;;
esac
