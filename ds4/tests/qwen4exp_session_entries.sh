#!/bin/sh
#
# The list of public session entries, derived from ds4.h at test time.
#
# tests/test_qwen4exp_graph.c asserts that every entry a qwen4exp session does
# not serve refuses BY NAME.  Its list used to be hand-written, so a new entry
# added to ds4.h could slip past unguarded.  This prints the header's own list
# and the test compares against it, which is what makes the tripwire real.
#
# SERVED lists what qwen4exp does implement; anything else must refuse.
set -u

ROOT=$(cd "$(dirname "$0")/.." && pwd)
HEADER="$ROOT/ds4.h"

SERVED="ds4_session_create ds4_session_free ds4_session_sync ds4_session_eval
ds4_session_eval_argmax ds4_session_argmax ds4_session_argmax_excluding
ds4_session_argmax_ignoring_eos ds4_session_top_logprobs
ds4_session_token_logprob ds4_session_copy_logits ds4_session_sample
ds4_session_invalidate ds4_session_rewind ds4_session_pos ds4_session_ctx
ds4_session_tokens ds4_session_common_prefix ds4_session_cancelled
ds4_session_set_progress ds4_session_set_display_progress
ds4_session_set_cancel ds4_session_report_progress ds4_session_power
ds4_session_set_power ds4_session_snapshot_free
ds4_session_payload_file_free ds4_session_eval_speculative
ds4_session_eval_speculative_argmax
ds4_session_eval_speculative_argmax_ignoring_eos
ds4_session_qwen4exp_spec_counters"

grep -oE '\<ds4_session[a-z_0-9]*\(' "$HEADER" \
    | sed 's/($//; s/(//' | sort -u | while read -r name; do
    case " $(echo $SERVED) " in
        *" $name "*) continue ;;
    esac
    echo "$name"
done
