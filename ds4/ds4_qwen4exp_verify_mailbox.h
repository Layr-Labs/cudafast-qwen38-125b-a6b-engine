/* ds4_qwen4exp_verify_mailbox.h
 *
 * Zero-copy verify-result mailbox on the decode sync path.
 *
 * The verify epilogue publishes the per-step result block into host-pinned
 * mapped memory through its device alias, then bumps a generation word.
 * The host polls the generation word for a bounded number of spins and
 * only then falls back to a stream synchronisation; either way it reads
 * the result out of its own memory, so the per-step cudaMemcpyAsync(D2H)
 * plus synchronise round-trip leaves the serial gap between the verify
 * graph and the next draft launch.
 */
#ifndef DS4_QWEN4EXP_VERIFY_MAILBOX_H
#define DS4_QWEN4EXP_VERIFY_MAILBOX_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* One generation word, one length word, then the opaque result block the
 * verify epilogue already produces. 256 B keeps a publish inside a
 * handful of device->host writes. */
#define DS4_VERIFY_MAILBOX_PAYLOAD 240

typedef struct ds4_verify_mailbox {
    volatile uint64_t seq;      /* generation counter; device writes it last */
    volatile uint32_t len;      /* payload bytes the last publish produced   */
    volatile uint32_t reserved0;
    volatile uint8_t  payload[DS4_VERIFY_MAILBOX_PAYLOAD];
} ds4_verify_mailbox;

/* Host side, implemented in ds4_cuda_qwen4exp.cu.
 * alloc hands back the host pointer and its device alias over the same
 * pinned pages; free releases them. */
int  ds4_verify_mailbox_alloc(ds4_verify_mailbox **host,
                              ds4_verify_mailbox **device);
void ds4_verify_mailbox_free(ds4_verify_mailbox *host);

/* Bounded spin on the generation word.  Returns non-zero once the mailbox
 * reached `want`; zero when the caller should fall back to a stream sync. */
static inline int ds4_verify_mailbox_poll(const ds4_verify_mailbox *mb,
                                          uint64_t want,
                                          uint64_t spins) {
    while (spins) {
        if (mb->seq >= want) return 1;
        --spins;
    }
    return mb->seq >= want;
}

#ifdef __cplusplus
}
#endif

#endif /* DS4_QWEN4EXP_VERIFY_MAILBOX_H */
