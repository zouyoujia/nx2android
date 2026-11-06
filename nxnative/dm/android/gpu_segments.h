// Bounded, observation-only GPU completion intervals. No Vulkan or environment dependencies.
#pragma once
#include <stdint.h>
#include <string.h>

enum { VK_GS_PASS, VK_GS_LRZ_READ, VK_GS_LRZ_WRITE, VK_GS_BARRIER, VK_GS_COMPUTE, VK_GS_KINDS };
enum { VK_GS_RECORDS = 4096, VK_GS_PARTS = 32, VK_GS_QUERIES = 2 * (VK_GS_RECORDS + VK_GS_PARTS), VK_GS_NONE = UINT32_MAX };
typedef struct { uint32_t begin, end, kind, part, w, h, count; uint64_t tag; } vk_gs_record;
typedef struct { uint32_t begin, end; } vk_gs_part;
typedef struct {
    uint32_t nq, nr, np, active, part_open, dropped, nested, invalid, upload_barriers;
    vk_gs_record rec[VK_GS_RECORDS]; vk_gs_part part[VK_GS_PARTS];
    uint64_t cpu[VK_GS_PARTS][5]; // monotonic CPU begin/record-end/end-CB/pre-submit/post-submit; separate clock domain
} vk_gs_state;
// A live request skips the current frame, then selects whole distinct frames at
// least 30 presents apart. All submissions of a selected frame share the decision.
typedef struct { uint64_t frame, taken; int known, have_taken, selected, remaining; } vk_gs_sampler;
static inline int vk_gs_budget(int n) { return n < 0 ? 0 : n > 8 ? 8 : n; }
static inline void vk_gs_request(vk_gs_sampler *s, int n, uint64_t frame) {
    *s = (vk_gs_sampler){.frame = frame, .known = 1, .remaining = vk_gs_budget(n)};
}
static inline int vk_gs_sample(vk_gs_sampler *s, uint64_t frame) {
    if (!s->known || s->frame != frame) {
        s->frame = frame; s->known = 1; s->selected = 0;
        if (s->remaining && (!s->have_taken || frame - s->taken >= 30)) {
            s->selected = 1; s->remaining--; s->taken = frame; s->have_taken = 1;
        }
    }
    return s->selected;
}
static inline uint32_t vk_gs_part_begin(vk_gs_state *s) {
    if (s->part_open) { s->invalid++; return VK_GS_NONE; }
    if (s->np == VK_GS_PARTS || s->nq + 2 > VK_GS_QUERIES) { s->dropped++; return VK_GS_NONE; }
    uint32_t q = s->nq++; s->part[s->np++] = (vk_gs_part){q, VK_GS_NONE}; s->part_open = 1; return q;
}
static inline uint32_t vk_gs_part_end(vk_gs_state *s) {
    if (!s->part_open) return VK_GS_NONE;
    if (s->active) { s->invalid++; s->active = 0; }   // keep the unmatched begin out of totals; no unwritten query is allocated
    uint32_t q = s->nq++; s->part[s->np - 1].end = q; s->part_open = 0; return q;
}
static inline uint32_t vk_gs_begin(vk_gs_state *s, uint32_t kind, uint64_t tag, uint32_t w, uint32_t h) {
    if (!s->part_open || kind >= VK_GS_KINDS) { s->invalid++; return 0; }
    if (s->active) { s->nested++; return 0; }   // LRZ/compute owns its internal full barriers
    // Reserve the segment end and the current part end before writing a begin.
    if (s->nr == VK_GS_RECORDS || s->nq + 3 > VK_GS_QUERIES) { s->dropped++; return 0; }
    uint32_t i = s->nr++; s->rec[i] = (vk_gs_record){s->nq++, VK_GS_NONE, kind, s->np - 1, w, h, 0, tag};
    s->active = i + 1; return s->active;
}
static inline uint32_t vk_gs_end(vk_gs_state *s, uint32_t token, uint32_t count) {
    if (!token) return VK_GS_NONE;
    if (token != s->active || !s->part_open) { s->invalid++; return VK_GS_NONE; }
    vk_gs_record *r = &s->rec[token - 1]; r->end = s->nq++; r->count = count; s->active = 0; return r->end;
}
static inline uint64_t vk_gs_delta(uint64_t a, uint64_t b, uint32_t bits) {
    return (b - a) & (bits == 64 ? UINT64_MAX : (UINT64_C(1) << bits) - 1);
}
typedef struct { uint64_t ticks, available; } vk_gs_query;
typedef struct {
    uint64_t span, in_parts, between_parts, unclassified, ticks[VK_GS_KINDS];
    uint32_t counts[VK_GS_KINDS], valid_parts, invalid;
} vk_gs_totals;
static inline int vk_gs_interval(const vk_gs_state *s, const vk_gs_query *q, uint32_t begin, uint32_t end, uint32_t bits, uint64_t *ticks) {
    if (!bits || bits > 64 || begin >= s->nq || end >= s->nq || begin >= end || !q[begin].available || !q[end].available) return 0;
    *ticks = vk_gs_delta(q[begin].ticks, q[end].ticks, bits);
    return *ticks < (UINT64_C(1) << (bits - 1));   // a backwards/ambiguous interval is not silently attributed
}
static inline vk_gs_totals vk_gs_reduce(const vk_gs_state *s, const vk_gs_query *q, uint32_t bits) {
    vk_gs_totals t = {0}; uint64_t last[VK_GS_PARTS] = {0}, spans[VK_GS_PARTS] = {0}; int valid[VK_GS_PARTS] = {0};
    for (uint32_t i = 0; i < s->np; i++) {
        const vk_gs_part *p = &s->part[i];
        if (!vk_gs_interval(s, q, p->begin, p->end, bits, &spans[i])) { t.invalid++; continue; }
        valid[i] = 1; t.valid_parts++; t.in_parts += spans[i];
        if (i) { uint64_t gap; if (valid[i - 1] && vk_gs_interval(s, q, s->part[i - 1].end, p->begin, bits, &gap)) t.between_parts += gap; else t.invalid++; }
    }
    if (s->np && !vk_gs_interval(s, q, s->part[0].begin, s->part[s->np - 1].end, bits, &t.span)) t.invalid++;
    uint64_t classified = 0;
    for (uint32_t i = 0; i < s->nr; i++) {
        const vk_gs_record *r = &s->rec[i]; uint64_t dt;
        if (r->part >= s->np || r->kind >= VK_GS_KINDS || !valid[r->part] ||
            r->begin <= s->part[r->part].begin || r->end >= s->part[r->part].end ||
            !vk_gs_interval(s, q, r->begin, r->end, bits, &dt)) { t.invalid++; continue; }
        uint64_t start = vk_gs_delta(q[s->part[r->part].begin].ticks, q[r->begin].ticks, bits);
        if (start < last[r->part] || start > spans[r->part] || dt > spans[r->part] - start) { t.invalid++; continue; }
        last[r->part] = start + dt; t.ticks[r->kind] += dt; t.counts[r->kind]++; classified += dt;
    }
    if (classified <= t.in_parts) t.unclassified = t.in_parts - classified; else t.invalid++;
    if (t.span != t.in_parts + t.between_parts) t.invalid++;
    return t;
}
