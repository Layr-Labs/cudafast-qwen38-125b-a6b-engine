#ifndef DS4_QWEN4EXP_END_SYNC_H
#define DS4_QWEN4EXP_END_SYNC_H

/* CUDA end_commands already synchronizes the device by default.  The
 * stream-only override still needs the full-device barrier. */
static inline int ds4_qwen4exp_end_and_sync(int (*end)(void),
                                             int (*sync)(void),
                                             int stream_only) {
    if (!end()) return 0;
    return !stream_only || sync();
}

#endif
