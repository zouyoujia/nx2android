// Bounded CPU metadata for selected GPU samples; no Vulkan calls or guest writes.
#pragma once
#include <stdint.h>
#include <string.h>
enum { VK_GD_RECORDS = 1024, VK_GD_TEXTURES = 16 };
typedef struct { uint64_t vs, fs, target[9], texture[VK_GD_TEXTURES]; uint32_t pass, count, ntex, unknown; } vk_gd_draw;
typedef struct { uint32_t n, dropped; vk_gd_draw draw[VK_GD_RECORDS]; } vk_gd_state;
static inline void vk_gd_add(vk_gd_state *s, const vk_gd_draw *d) {
    if (s->n) { vk_gd_draw *p = &s->draw[s->n - 1]; vk_gd_draw key = *d; key.count = p->count;
        if (!memcmp(p, &key, sizeof key)) { p->count += d->count; return; } }
    if (s->n == VK_GD_RECORDS) { s->dropped += d->count; return; }
    s->draw[s->n++] = *d;
}
