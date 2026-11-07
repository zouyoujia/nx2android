// Reflection is the binding interface used by stage_set, including fallback descriptors.
#pragma once
#include <stdint.h>
#include <string.h>
static inline int vk_reflection_needs_images(const void *reflection, uint32_t len) {
    if (!reflection || len < 20) return 1;
    const uint8_t *p = reflection; uint32_t n; memcpy(&n, p, 4);
    if (n > (len - 20) / sizeof(dmsv_bind)) return 1;
    for (uint32_t i = 0; i < n; i++) {
        dmsv_bind b; memcpy(&b, p + 20 + (size_t)i * sizeof b, sizeof b);
        if (b.kind > RK_SAMPD || b.desc > RD_SAMPLER || b.kind == RK_IMG || b.kind == RK_IMG_UBO || b.kind == RK_IMG_TBUF || b.desc == RD_STORAGE || b.desc == RD_STORAGE_TBUF) return 1;
    }
    return 0;
}
