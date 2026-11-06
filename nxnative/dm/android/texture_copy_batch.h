// Only a standalone upload may initialize a new image in its main CB: a later
// packaged upload in a mixed batch can execute in the upload CB before the main.
#pragma once
#include "../gfx/nvn/nvn_obj.h"
static inline int texture_copy_standalone(const uint64_t *handles, int n) {
    if (n != 1 || !handles || !handles[0]) return 0;
    const uint8_t *p = (const uint8_t *)(uintptr_t)handles[0]; size_t used = 0; int copied = 0;
    for (int records = 0; records < 16; records++) {
        op_hdr h; memcpy(&h, p + used, sizeof h);
        if (h.size < sizeof h || (h.size & 7) || h.size > 4096 - used) return 0;
        if (h.op == OP_END) return h.size == sizeof h && copied;
        if (copied) return 0;
        if (h.op == OP_COPY_B2T) { if (h.size < sizeof h + sizeof(p_copy_b2t)) return 0; copied = 1; }
        else if (h.op == OP_COPY_STRIDE) { if (h.size < sizeof h + 16) return 0; }
        else return 0;
        used += h.size;
        if (used > 4096 - sizeof h) return 0;
    }
    return 0;
}
