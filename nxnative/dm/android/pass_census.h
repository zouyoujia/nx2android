// Observation-only render-pass counters. Callers serialize updates and snapshots.
#pragma once
#include <stdint.h>
#include <string.h>

enum vk_pass_end_reason {
    VK_PASS_END_EXTERNAL, VK_PASS_END_EXTERNAL_LAND, VK_PASS_END_SET_RT,
    VK_PASS_END_BARRIER, VK_PASS_END_RASTER_DISCARD, VK_PASS_END_PASS_SPLIT,
    VK_PASS_END_TRANSFORM_FEEDBACK, VK_PASS_END_DISPATCH,
    VK_PASS_END_BUFFER_TO_TEXTURE, VK_PASS_END_TEXTURE_TO_BUFFER,
    VK_PASS_END_TEXTURE_COPY, VK_PASS_END_TEXTURE_CLEAR, VK_PASS_END_DEBUG,
    VK_PASS_END_BUFFER_COPY, VK_PASS_END_BUFFER_FILL,
    VK_PASS_END_UNIFORM_LIMIT, VK_PASS_END_SUBMIT_END, VK_PASS_END_TEXTURE_DEPENDENCY, VK_PASS_END_COUNT
};
enum vk_pc_rt { VK_PC_RT_OPEN0, VK_PC_RT_SAME, VK_PC_RT_MISMATCH, VK_PC_RT_FORCED, VK_PC_RT_COUNT };
enum vk_pc_lrz { VK_PC_LRZ_LOAD, VK_PC_LRZ_SWITCH, VK_PC_LRZ_CUT, VK_PC_LRZ_COUNT };
enum { VK_PC_MASK_OTHER = 256, VK_PC_MASK_COUNT = 257 };

typedef struct {
    uint64_t end_calls[VK_PASS_END_COUNT], ends[VK_PASS_END_COUNT][3], clear_only[VK_PASS_END_COUNT];
    uint64_t barriers[VK_PC_MASK_COUNT][3]; // calls / ended an already active pass / ended a clear-only pass
    uint64_t rt[VK_PC_RT_COUNT], lrz[VK_PC_LRZ_COUNT], lrz_writes[VK_PC_LRZ_COUNT];
    uint64_t adjacent_barriers, adjacent_skipped;
} vk_pc_counts;
typedef struct { int enabled; uint64_t since_frame, last_report; vk_pc_counts counts; } vk_pc_state;
typedef struct { uint64_t since_frame, frame; vk_pc_counts counts; } vk_pc_snapshot;

static inline void vk_pc_enable(vk_pc_state *s, int enabled, uint64_t frame) {
    enabled = !!enabled;
    if (enabled != s->enabled) { memset(&s->counts, 0, sizeof s->counts); s->enabled = enabled; s->since_frame = frame; s->last_report = UINT64_MAX; }
}
static inline void vk_pc_end(vk_pc_state *s, enum vk_pass_end_reason reason, uint32_t mask,
                             int ended, int clear_only, uint64_t draws) {
    if (!s->enabled) return;
    if ((unsigned)reason >= VK_PASS_END_COUNT) reason = VK_PASS_END_EXTERNAL;
    s->counts.end_calls[reason]++;
    if (reason == VK_PASS_END_BARRIER) {
        unsigned k = mask < VK_PC_MASK_OTHER ? mask : VK_PC_MASK_OTHER;
        s->counts.barriers[k][0]++;
        if (ended) s->counts.barriers[k][clear_only ? 2 : 1]++;
    }
    if (ended) { s->counts.ends[reason][draws == 0 ? 0 : draws == 1 ? 1 : 2]++; s->counts.clear_only[reason] += !!clear_only; }
}
static inline void vk_pc_rt_note(vk_pc_state *s, enum vk_pc_rt kind) {
    if (s->enabled && (unsigned)kind < VK_PC_RT_COUNT) s->counts.rt[kind]++;
}
static inline void vk_pc_lrz_note(vk_pc_state *s, enum vk_pc_lrz kind, int write) {
    if (s->enabled && (unsigned)kind < VK_PC_LRZ_COUNT) { s->counts.lrz[kind]++; s->counts.lrz_writes[kind] += !!write; }
}
static inline void vk_pc_adjacent_note(vk_pc_state *s, int candidate, int skipped) {
    if (s->enabled && candidate) { s->counts.adjacent_barriers++; s->counts.adjacent_skipped += !!skipped; }
}
static inline int vk_pc_take(vk_pc_state *s, uint64_t frame, vk_pc_snapshot *out) {
    if (!s->enabled || !frame || frame % 600 || frame == s->last_report || frame <= s->since_frame) return 0;
    out->since_frame = s->since_frame; out->frame = frame; out->counts = s->counts;
    memset(&s->counts, 0, sizeof s->counts); s->since_frame = s->last_report = frame;
    return 1;
}

void vk_pass_census_report(uint64_t frame);
void vk_end_pass_land_reason(enum vk_pass_end_reason reason);
