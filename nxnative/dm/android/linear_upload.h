// Bounded GPU write history and dirty-byte to complete-row projection.
#pragma once
#include <stdint.h>
#include <stddef.h>
#include "nvn_dirty.h"
enum { VK_LW_HISTORY = 1024, VK_LINEAR_REGIONS = 128 };
typedef struct { uint64_t seq, lo, hi, fence; } vk_lw_event;
typedef struct { uint64_t seq, evicted_fence; vk_lw_event event[VK_LW_HISTORY]; } vk_lw_history;
static inline void vk_lw_note(vk_lw_history *s, uint64_t a, uint64_t n, uint64_t fence) {
    if (s->seq == UINT64_MAX) return;
    if (!n || n > UINT64_MAX - a) { a = 0; n = UINT64_MAX; }
    uint64_t q = ++s->seq; vk_lw_event *old = &s->event[q % VK_LW_HISTORY];
    if (old->seq && old->fence > s->evicted_fence) s->evicted_fence = old->fence;
    s->event[q % VK_LW_HISTORY] = (vk_lw_event){q, a, a + n, fence};
}
static inline int vk_lw_overlap(const vk_lw_event *e, uint64_t a, uint64_t n) {
    return !n || n > UINT64_MAX - a || (e->lo < a + n && a < e->hi);
}
static inline int vk_lw_changed(const vk_lw_history *s, uint64_t stamp, uint64_t a, uint64_t n) {
    if (!n || n > UINT64_MAX - a || s->seq == UINT64_MAX || stamp > s->seq || s->seq - stamp > VK_LW_HISTORY) return 1;
    for (uint64_t q = stamp + 1; q && q <= s->seq; q++) {
        const vk_lw_event *e = &s->event[q % VK_LW_HISTORY];
        if (e->seq != q || vk_lw_overlap(e, a, n)) return 1;
    }
    return 0;
}
static inline int vk_lw_pending(const vk_lw_history *s, uint64_t done, uint64_t a, uint64_t n) {
    if (!n || n > UINT64_MAX - a || s->seq == UINT64_MAX || s->evicted_fence > done) return 1;
    // A pending event evicted from the ring must not become a clean range.
    for (unsigned i = 0; i < VK_LW_HISTORY; i++) if (s->event[i].seq && s->event[i].fence > done && vk_lw_overlap(&s->event[i], a, n)) return 1;
    return 0;
}
typedef struct { uint32_t first, count; } vk_linear_rows;
static inline int vk_linear_project(const nvn_dirty_interval *d, int nd, size_t stride, uint32_t height, uint64_t base, vk_linear_rows *rows) {
    if (nd < 0 || nd > VK_LINEAR_REGIONS || !stride || !height || height > SIZE_MAX / stride) return -1;
    int nr = 0;
    for (int i = 0; i < nd; i++) {
        if (!d[i].length || d[i].offset >= stride * height || d[i].length > stride * height - d[i].offset) return -1;
        size_t first = d[i].offset / stride, last = (d[i].offset + d[i].length - 1) / stride;
        if (base > UINT64_MAX - first * stride || (base + first * stride) % 4) return -1;
        if (nr && first <= (size_t)rows[nr-1].first + rows[nr-1].count) {
            size_t end = (size_t)rows[nr-1].first + rows[nr-1].count; if (last + 1 > end) rows[nr-1].count = (uint32_t)(last + 1 - rows[nr-1].first);
        } else rows[nr++] = (vk_linear_rows){(uint32_t)first, (uint32_t)(last - first + 1)};
    }
    return nr;
}
