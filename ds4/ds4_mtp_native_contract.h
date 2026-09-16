#ifndef DS4_MTP_NATIVE_CONTRACT_H
#define DS4_MTP_NATIVE_CONTRACT_H
#include <stdint.h>
/* Reserved only by the paired deferred screen/map API. Accepted vocabularies
 * cannot contain this ID; UINT32_MAX remains the malformed-winner error. */
#define DS4_MTP_NATIVE_RETRY_ID (UINT32_MAX - 1u)
#endif
