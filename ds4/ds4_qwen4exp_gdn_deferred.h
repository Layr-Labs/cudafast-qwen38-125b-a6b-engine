#ifndef DS4_QWEN4EXP_GDN_DEFERRED_H
#define DS4_QWEN4EXP_GDN_DEFERRED_H

#include <stdbool.h>
#include <stdint.h>

#define DS4_QWEN4EXP_GDN_DEFERRED_FLUSH_ROWS 2u
#define DS4_QWEN4EXP_GDN_DEFERRED_BANK_ROWS 4u
#define DS4_QWEN4EXP_GDN_DEFERRED_MAX_PREFIX 3u

/* A two-row verify keeps its final recurrent state as checkpoint + log.
 * When its input prefix reaches two rows, the kernel folds CURRENT row zero
 * into the checkpoint and writes only row one into the opposite log bank.
 * Otherwise it appends both current rows after the input prefix. */
typedef struct {
    bool valid;
    uint32_t bank;
    uint32_t final_rows;
    uint32_t snapshot_rows;
} ds4_qwen4exp_gdn_deferred_output;

static inline ds4_qwen4exp_gdn_deferred_output
ds4_qwen4exp_gdn_deferred_output_plan(uint32_t prefix, uint32_t bank) {
    ds4_qwen4exp_gdn_deferred_output out = {false, 0u, 0u, 0u};
    if (prefix > DS4_QWEN4EXP_GDN_DEFERRED_MAX_PREFIX || bank > 1u)
        return out;
    const bool fold = prefix >= DS4_QWEN4EXP_GDN_DEFERRED_FLUSH_ROWS;
    out.valid = true;
    out.bank = bank ^ (uint32_t)fold;
    out.final_rows = fold ? 1u : prefix + 2u;
    out.snapshot_rows = out.final_rows - 1u;
    return out;
}

typedef struct {
    bool valid;
    bool active;
    bool reuse;
    bool needs_final;
    bool settle;
    bool swap;
    uint32_t prefix;
    uint32_t bank;
} ds4_qwen4exp_gdn_deferred_step;

/* old_prefix/old_bank describe the INPUT of the preceding deferred verify,
 * not its output. recurrent/conv are pending snapshot selectors (row + 1;
 * zero means retain the final state). Metadata is ignored without previous.
 *
 * The caller must check valid before changing anything. On a nonreuse exit:
 * publish final state if needs_final, settle pending selectors if settle,
 * then swap canonical live/checkpoint buffers if swap. Publishing final state
 * must retain the old metadata until snapshot selection has completed. Only
 * then may the caller install this step's input prefix/bank. A diagnostic
 * materialization alone must not retire the previous virtual snapshot. */
static inline ds4_qwen4exp_gdn_deferred_step ds4_qwen4exp_gdn_deferred_plan(
        bool enabled, bool previous, uint32_t old_prefix, uint32_t old_bank,
        uint32_t width, uint32_t snapshots, uint32_t recurrent, uint32_t conv) {
    ds4_qwen4exp_gdn_deferred_step step =
        {false, false, false, false, false, false, 0u, 0u};
    ds4_qwen4exp_gdn_deferred_output out = {false, 0u, 0u, 0u};
    if (previous) {
        out = ds4_qwen4exp_gdn_deferred_output_plan(old_prefix, old_bank);
        if (!out.valid) return step;
    }
    step.valid = true;
    step.active = enabled && width == 2u && snapshots == 1u;
    step.reuse = step.active && previous && recurrent == conv && recurrent <= 1u;
    step.needs_final = previous && !step.reuse;
    step.settle = !step.reuse && (recurrent != 0u || conv != 0u);
    step.swap = step.active && !step.reuse;
    if (step.reuse) {
        step.prefix = out.final_rows - recurrent;
        step.bank = out.bank;
    }
    return step;
}

#endif
