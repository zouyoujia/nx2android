// Bounded recipes for compute variants observed in real dispatches. No driver data.
#pragma once
#include <stdint.h>
#include <stddef.h>
#include <string.h>

enum { VK_COMPUTE_KEYS_MAX = 256, VK_COMPUTE_KEYS_MAGIC = 0x31574d43u,
       VK_COMPUTE_KEYS_VERSION = 1, VK_COMPUTE_KEYS_RECORD = 24,
       VK_COMPUTE_KEYS_HEADER = 16, VK_COMPUTE_KEYS_BYTES = 16 + 256 * 24 };
typedef struct { uint64_t shader, um, im; } vk_compute_key;
typedef struct { vk_compute_key keys[VK_COMPUTE_KEYS_MAX]; uint32_t count; uint64_t generation; } vk_compute_keys;

static inline int vk_compute_key_valid(vk_compute_key k) { return k.shader && (k.um | k.im) && !(k.um & k.im); }
static inline int vk_compute_keys_add(vk_compute_keys *s, vk_compute_key k) {
    if (s->count > VK_COMPUTE_KEYS_MAX || !vk_compute_key_valid(k)) return 0;
    for (uint32_t i = 0; i < s->count; i++) if (s->keys[i].shader == k.shader && s->keys[i].um == k.um && s->keys[i].im == k.im) return 0;
    if (s->count >= VK_COMPUTE_KEYS_MAX) return 0;
    s->keys[s->count++] = k; s->generation++; return 1;
}
static inline uint64_t vk_compute_keys_get_le(const uint8_t *p, unsigned n) {
    uint64_t v = 0; for (unsigned i = 0; i < n; i++) v |= (uint64_t)p[i] << (8 * i); return v;
}
static inline void vk_compute_keys_put_le(uint8_t *p, uint64_t v, unsigned n) {
    for (unsigned i = 0; i < n; i++) p[i] = (uint8_t)(v >> (8 * i));
}
static inline size_t vk_compute_keys_encode(const vk_compute_keys *s, void *data, size_t cap) {
    if (s->count > VK_COMPUTE_KEYS_MAX) return 0;
    size_t size = VK_COMPUTE_KEYS_HEADER + (size_t)s->count * VK_COMPUTE_KEYS_RECORD;
    if (!data || cap < size) return 0;
    uint8_t *p = data;
    vk_compute_keys_put_le(p, VK_COMPUTE_KEYS_MAGIC, 4); vk_compute_keys_put_le(p + 4, VK_COMPUTE_KEYS_VERSION, 4);
    vk_compute_keys_put_le(p + 8, VK_COMPUTE_KEYS_RECORD, 4); vk_compute_keys_put_le(p + 12, s->count, 4);
    for (uint32_t i = 0; i < s->count; i++) {
        if (!vk_compute_key_valid(s->keys[i])) return 0;
        uint8_t *r = p + VK_COMPUTE_KEYS_HEADER + (size_t)i * VK_COMPUTE_KEYS_RECORD;
        vk_compute_keys_put_le(r, s->keys[i].shader, 8); vk_compute_keys_put_le(r + 8, s->keys[i].um, 8); vk_compute_keys_put_le(r + 16, s->keys[i].im, 8);
    }
    return size;
}
static inline int vk_compute_keys_decode(vk_compute_keys *out, const void *data, size_t size) {
    if (!data || size < VK_COMPUTE_KEYS_HEADER) return 0;
    const uint8_t *p = data;
    uint64_t count = vk_compute_keys_get_le(p + 12, 4);
    if (vk_compute_keys_get_le(p, 4) != VK_COMPUTE_KEYS_MAGIC || vk_compute_keys_get_le(p + 4, 4) != VK_COMPUTE_KEYS_VERSION ||
        vk_compute_keys_get_le(p + 8, 4) != VK_COMPUTE_KEYS_RECORD || count > VK_COMPUTE_KEYS_MAX ||
        size != VK_COMPUTE_KEYS_HEADER + (size_t)count * VK_COMPUTE_KEYS_RECORD) return 0;
    vk_compute_keys parsed = {0};
    for (uint32_t i = 0; i < (uint32_t)count; i++) {
        const uint8_t *r = p + VK_COMPUTE_KEYS_HEADER + (size_t)i * VK_COMPUTE_KEYS_RECORD;
        // Bad tuples and duplicates cannot request a compile. Header/size errors reject the whole file.
        vk_compute_key k = {vk_compute_keys_get_le(r, 8), vk_compute_keys_get_le(r + 8, 8), vk_compute_keys_get_le(r + 16, 8)};
        vk_compute_keys_add(&parsed, k);
    }
    *out = parsed; return 1;
}
