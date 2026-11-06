// Conservative full-storage alias checks for a deferred texture-invalidate barrier.
#pragma once
#include <stdint.h>
#include <string.h>
enum { VK_TB_ATTACHMENTS = 9 };
typedef struct { uint64_t lo, hi, pool, off, end; } vk_tb_range;
typedef struct { vk_tb_range att[VK_TB_ATTACHMENTS]; uint32_t n, draws; int pending; } vk_tb_state;
static inline int vk_tb_range_make(vk_tb_range *r, uint64_t mem, uint64_t size, uint64_t pool, int64_t off) {
    if (!mem || !size || !pool || off < 0 || size > UINT64_MAX - mem || size > UINT64_MAX - (uint64_t)off) return 0;
    *r = (vk_tb_range){mem, mem + size, pool, (uint64_t)off, (uint64_t)off + size}; return 1;
}
static inline int vk_tb_overlap(const vk_tb_range *a, const vk_tb_range *b) {
    // Full footprints deliberately reject disjoint mip/layer views of the same
    // storage too. Different pool handles over overlapping addresses still alias.
    return (a->lo < b->hi && b->lo < a->hi) || (a->pool == b->pool && a->off < b->end && b->off < a->end);
}
static inline int vk_tb_texture_hit(const vk_tb_state *s, const vk_tb_range *r) {
    for (uint32_t i = 0; i < s->n; i++) if (vk_tb_overlap(&s->att[i], r)) return 1;
    return 0;
}
static inline int vk_tb_buffer_hit(const vk_tb_state *s, uint64_t lo, uint64_t size) {
    if (!lo || !size || size > UINT64_MAX - lo) return 1;   // unknown bounds are not independent
    for (uint32_t i = 0; i < s->n; i++) if (lo < s->att[i].hi && s->att[i].lo < lo + size) return 1;
    return 0;
}
static inline void vk_tb_clear(vk_tb_state *s) { memset(s, 0, sizeof *s); }
