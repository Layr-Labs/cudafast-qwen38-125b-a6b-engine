#!/usr/bin/env bash
# A leg that generates NO TOKENS must fail by name.
#
# WHY THIS EXISTS. A zero-output regression on ds4 278b799 read as a PASSING
# gate. Two paths let it: with empty stdout the stream comparison never ran and
# the gate said `stream_mismatch`, which points at divergence when the truth is
# that there is no stream; and two legs that both emitted only a banner compared
# BYTE-IDENTICAL, so the gate said `ok`. A gate that cannot tell "generated
# nothing" from "generated the same thing" is not a gate.
#
# The driver now parses the engine's own counts and refuses by name. This suite
# drives the two functions that decide it, sourced OUT OF THE SHIPPED SCRIPT --
# not copies -- against a synthetic engine's output in both line shapes:
#
#   OLD  ds4: prefill: 40.00 t/s, generation: 12.00 t/s
#   NEW  ds4: prefill: 40.00 t/s, generation: 12.00 t/s (57 prompt tokens in 1.43 s, 128 generated in 10.67 s)
#
# The new shape is L17's b3e83ca on Layr-Labs/ds4 and is not vendored yet, so
# the driver has to read a tree from either side of that vendor-sync.
#
# Hermetic: no GPU, no artifact, no lock, no Linux. It needs sed, grep and bash.
set -uo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null && pwd -P)"
DRIVER="${REPO_ROOT}/tools/qwen4exp-g1-boot.sh"
VERBOSE=0
[[ "${1:-}" == "-v" ]] && VERBOSE=1

EXPECTED_MIN_ASSERTIONS=28
PASSED=0; FAILED=0; FAILURES=()
pass() { PASSED=$((PASSED+1)); [[ "${VERBOSE}" == "1" ]] && echo "ok    $1"; return 0; }
fail() { FAILED=$((FAILED+1)); FAILURES+=("$1"); echo "FAIL  $1" >&2; }
eq() { # eq GOT WANT LABEL
  if [[ "$1" == "$2" ]]; then pass "$3"; else fail "$3 (got '$1', want '$2')"; fi
}

WORK="$(mktemp -d "${TMPDIR:-/tmp}/g1-no-output.XXXXXX")"
trap 'rm -rf "${WORK}"' EXIT

# Source the two deciders out of the shipped driver, by function boundary, so
# this suite tests the code that runs and not a transcription of it.
LIB="${WORK}/lib.sh"
sed -n '/^g1_parse_rates() {/,/^}/p;/^g1_no_output_reason() {/,/^}/p;/^g1_strip_preamble() {/,/^}/p' "${DRIVER}" > "${LIB}"
if [[ "$(grep -c '^}' "${LIB}")" -ne 3 ]]; then
  echo "test-g1-no-output: could not extract all three functions from ${DRIVER}" >&2
  exit 1
fi
# shellcheck source=/dev/null
. "${LIB}"
pass "all three deciders extracted from the shipped driver"

# The synthetic engine. It prints a banner on stdout when it generates, the
# timing line on stderr in the requested shape, and nothing else.
stub() { # stub SHAPE GENERATED -> writes ${WORK}/x.stdout and ${WORK}/x.stderr
  local shape="$1" gen="$2"
  : > "${WORK}/x.stdout"; : > "${WORK}/x.stderr"
  if [[ "${gen}" -gt 0 ]]; then
    printf 'the quick brown fox\n' > "${WORK}/x.stdout"
  fi
  case "${shape}" in
    old) printf 'ds4: prefill: 40.00 t/s, generation: 12.00 t/s\n' > "${WORK}/x.stderr" ;;
    new) printf 'ds4: prefill: 40.00 t/s, generation: 12.00 t/s (57 prompt tokens in 1.43 s, %d generated in 10.67 s)\n' \
           "${gen}" > "${WORK}/x.stderr" ;;
  esac
}

verdict() { # verdict -> prints the reason after parsing
  local prefill decode prompt_tok prefill_s gen_tok decode_s
  eval "$(g1_parse_rates "${WORK}/x.stderr")"
  g1_no_output_reason "${gen_tok}" "${WORK}/x.stdout"
}
counts() { # counts -> "prompt gen prefill_s decode_s prefill decode"
  local prefill decode prompt_tok prefill_s gen_tok decode_s
  eval "$(g1_parse_rates "${WORK}/x.stderr")"
  printf '%s|%s|%s|%s|%s|%s\n' "${prompt_tok}" "${gen_tok}" "${prefill_s}" "${decode_s}" "${prefill}" "${decode}"
}

# --- NEW shape, G > 0: every count is read, and the leg passes ---------------
stub new 128
eq "$(counts)" "57|128|1.43|10.67|40.00|12.00" "new shape: prompt, generated, prefill s, decode s and both rates are read"
eq "$(verdict)" "" "new shape with 128 generated: no refusal"

# --- NEW shape, G == 0: refused BY NAME even though stdout could be anything -
stub new 0
eq "$(counts)" "57|0|1.43|10.67|40.00|12.00" "new shape with 0 generated: the zero is read, not dropped"
eq "$(verdict)" "0 generated tokens" "new shape with 0 generated: refused by name"

# THE REGRESSION, EXACTLY. The engine says zero but stdout carries a banner, so
# two such legs compare byte-identical and the old gate said ok. The count wins.
stub new 0
printf 'ds4: loading model\n' > "${WORK}/x.stdout"
eq "$(verdict)" "0 generated tokens" \
  "a banner on stdout does NOT rescue a leg the engine says generated nothing"

# --- OLD shape: no counts, so the stdout text decides ------------------------
stub old 128
eq "$(counts)" "||||40.00|12.00" "old shape: rates are read, counts stay empty"
eq "$(verdict)" "" "old shape with token text: no refusal"

stub old 0
eq "$(verdict)" "no token text on stdout" "old shape with empty stdout: refused by name"

printf '   \n\t\n' > "${WORK}/x.stdout"
eq "$(verdict)" "no token text on stdout" "whitespace-only stdout is not token text"

# --- neither shape at all ----------------------------------------------------
: > "${WORK}/x.stderr"; : > "${WORK}/x.stdout"
eq "$(counts)" "|||||" "no timing line: every field empty, and the parse does not fail"
eq "$(verdict)" "no token text on stdout" "no timing line and no stdout: refused by name"

# A missing stderr file must not crash the parse.
rm -f "${WORK}/x.stderr"
eq "$(counts)" "|||||" "an absent stderr capture parses to empty rather than failing"

# ===========================================================================
# THE PREAMBLE. The engine prints its memory plan and a budget line carrying a
# LIVE free-memory reading before the first token. The first real run on
# 3751b81 reported stream_mismatch whose entire stream.diff was those two lines
# -- 119.01 GiB against 118.64 GiB -- while every generated token matched.
# ===========================================================================
BUDGET='qwen4exp memory budget: free %s GiB, required 93.46 GiB'
preamble_stdout() { # preamble_stdout FILE FREE_GIB TOKENS...
  local f="$1" free="$2"; shift 2
  {
    printf 'qwen4exp memory plan:\n'
    printf '  body                 103.69 GiB\n'
    # shellcheck disable=SC2059
    printf "${BUDGET}\n" "${free}"
    [ "$#" -gt 0 ] && printf '%s\n' "$@"
  } > "${f}"
}

serial_raw="${WORK}/serial.stdout"; mtp_raw="${WORK}/mtp1.stdout"
serial_st="${WORK}/serial.stream"; mtp_st="${WORK}/mtp1.stream"

# THE EXACT FAILURE: same tokens, different free-memory reading.
preamble_stdout "${serial_raw}" 119.01 'the quick brown fox' 'jumps over'
preamble_stdout "${mtp_raw}"    118.64 'the quick brown fox' 'jumps over'
if cmp -s "${serial_raw}" "${mtp_raw}"; then
  fail "the raw stdouts differ (the regression this fixes)"
else
  pass "the raw stdouts differ on the budget line alone -- the reported mismatch"
fi
g1_strip_preamble "${serial_raw}" "${serial_st}"
g1_strip_preamble "${mtp_raw}" "${mtp_st}"
if cmp -s "${serial_st}" "${mtp_st}"; then
  pass "the stripped streams COMPARE EQUAL: a live memory reading is not a token"
else
  fail "the stripped streams COMPARE EQUAL ($(diff -u "${serial_st}" "${mtp_st}" | head -4 | tr '\n' ' '))"
fi
eq "$(cat "${serial_st}")" "$(printf 'the quick brown fox\njumps over')" \
  "the stripped stream is exactly the generated tokens"

# ONE DIFFERENT TOKEN still mismatches -- the gate must not have been softened.
preamble_stdout "${mtp_raw}" 118.64 'the quick brown fox' 'jumps under'
g1_strip_preamble "${mtp_raw}" "${mtp_st}"
if cmp -s "${serial_st}" "${mtp_st}"; then
  fail "one differing token still MISMATCHES"
else
  pass "one differing token still MISMATCHES: stripping the preamble did not soften the gate"
fi

# The budget line may repeat (a re-plan); the LAST one is the boundary.
{ printf 'qwen4exp memory budget: free 1 GiB\n'
  printf 'qwen4exp memory budget: free 2 GiB\n'
  printf 'tok\n'; } > "${serial_raw}"
g1_strip_preamble "${serial_raw}" "${serial_st}"
eq "$(cat "${serial_st}")" "tok" "the LAST budget line is the boundary, not the first"

# No budget line at all: compared whole, which is the old behaviour.
printf 'just tokens\n' > "${serial_raw}"
g1_strip_preamble "${serial_raw}" "${serial_st}"
eq "$(cat "${serial_st}")" "just tokens" "an engine that prints no budget line is compared whole"

# A leg that generated NOTHING behind a preamble must still refuse. Raw stdout
# is non-empty here, so this is the case the text fallback exists for.
preamble_stdout "${serial_raw}" 119.01
g1_strip_preamble "${serial_raw}" "${serial_st}"
eq "$(g1_no_output_reason "" "${serial_st}")" "no token text on stdout" \
  "zero tokens behind a preamble still refuses by name"
eq "$(g1_no_output_reason "" "${serial_raw}")" "" \
  "and against RAW stdout it would NOT have -- which is why the stream is passed"

# --- the driver wires them in, and the verdict has its own exit code ---------
if grep -q 'eval "$(g1_parse_rates "${err}")"' "${DRIVER}"; then
  pass "run_leg parses through g1_parse_rates"
else
  fail "run_leg parses through g1_parse_rates"
fi
if grep -q 'no_output_reason="$(g1_no_output_reason "${gen_tok:-}" "${stream}")"' "${DRIVER}"; then
  pass "run_leg decides through g1_no_output_reason"
else
  fail "run_leg decides through g1_no_output_reason"
fi
if grep -q 'return 7' "${DRIVER}" && grep -q 'STATUS=no_output; EXIT=7' "${DRIVER}"; then
  pass "a no-output leg exits 7 and the summary says no_output"
else
  fail "a no-output leg exits 7 and the summary says no_output"
fi
# BOTH LEGS MUST SEND THE GOLDEN'S PREFIX. The golden is the first 1024 tokens
# and the prompt file tokenizes to 1054; without --prompt-tokens the CLI decodes
# the full instruction, whose first argmax is EOS, so the leg generates nothing
# and the run compares two empty streams. That is the failure the no-output
# refusal above catches -- this keeps the cause fixed too.
if grep -q -- '--prompt-tokens "${PROMPT_TOKENS}"' "${DRIVER}"; then
  pass "leg_cmd sends --prompt-tokens on every leg"
else
  fail "leg_cmd sends --prompt-tokens on every leg"
fi
if grep -qE '^PROMPT_TOKENS=1024$' "${DRIVER}"; then
  pass "the prefix defaults to the golden's 1024 tokens"
else
  fail "the prefix defaults to the golden's 1024 tokens"
fi

# The comparison must be over the STREAMS, or the fix is not wired in.
if grep -q 'cmp -s "${RUN}/serial.stream" "${RUN}/mtp1.stream"' "${DRIVER}"; then
  pass "the driver compares the stripped streams, not raw stdout"
else
  fail "the driver compares the stripped streams, not raw stdout"
fi
if grep -q 'diff -u "${RUN}/serial.stream" "${RUN}/mtp1.stream"' "${DRIVER}"; then
  pass "stream.diff is over the stripped streams"
else
  fail "stream.diff is over the stripped streams"
fi

# The counts must reach the sealed leg JSON, or the number is not readable.
for field in prompt_tokens generated_tokens prefill_s decode_s no_output_reason stream_sha256 stream_bytes; do
  if grep -q "\"${field}\":" "${DRIVER}"; then
    pass "the leg JSON seals ${field}"
  else
    fail "the leg JSON seals ${field}"
  fi
done

echo "g1-no-output: ${PASSED} passed, ${FAILED} failed"
if [[ "${FAILED}" -ne 0 ]]; then
  printf '  - %s\n' "${FAILURES[@]}" >&2
  exit 1
fi
if [[ "${PASSED}" -lt "${EXPECTED_MIN_ASSERTIONS}" ]]; then
  echo "g1-no-output: ran only ${PASSED} assertions, expected at least ${EXPECTED_MIN_ASSERTIONS}" >&2
  exit 1
fi
exit 0
