// Bounded, owner-thread-only observation of guest main-thread yield call sites.
#pragma once
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
enum { DM_YC_SITES = 16, DM_YC_MAX = 8192 };
typedef struct {
    uint64_t lr, gate, count, moved, slept, elapsed, max_elapsed, max_run;
    int64_t first_ns, last_ns; uint64_t zero, negative, positive;
    uint64_t first[10], last[10];
} dm_yc_site;
typedef struct {
    unsigned remaining, samples, nsite, overflow, unlocated;
    uint64_t begin, end, elapsed;
    dm_yc_site site[DM_YC_SITES];
} dm_yc_state;
static inline unsigned dm_yc_budget(const char *value) {
    if (!value || *value < '0' || *value > '9') return 0;
    char *end; errno = 0; unsigned long n = strtoul(value, &end, 10);
    if (errno || end == value || *end) return 0;
    return n > DM_YC_MAX ? DM_YC_MAX : (unsigned)n;
}
static inline void dm_yc_start(dm_yc_state *s, unsigned n) {
    memset(s, 0, sizeof *s); s->remaining = n > DM_YC_MAX ? DM_YC_MAX : n;
}
// Only the register array is read: no guest pointer dereference, stack walk,
// allocation, system call, wait or scheduling change. Return true once at end.
static inline int dm_yc_note(dm_yc_state *s, const uint64_t *r, uint64_t caller, uint64_t begin, uint64_t end, uint64_t run, int moved, int slept) {
    if (!s->remaining) return 0;
    uint64_t elapsed = end >= begin ? end - begin : 0;
    if (!s->samples) s->begin = begin;
    s->samples++; s->remaining--; s->end = end; s->elapsed += elapsed; s->unlocated += !caller;
    unsigned i = 0; while (i < s->nsite && (s->site[i].lr != caller || (!caller && s->site[i].gate != r[30]))) i++;
    if (i == s->nsite && i == DM_YC_SITES) s->overflow++;
    else {
        dm_yc_site *p = &s->site[i];
        if (i == s->nsite) { s->nsite++; p->lr = caller; p->gate = r[30]; p->first_ns = (int64_t)r[0]; memcpy(p->first, r + 19, sizeof p->first); }
        memcpy(p->last, r + 19, sizeof p->last);
        p->last_ns = (int64_t)r[0]; if (!p->last_ns) p->zero++; else if (p->last_ns < 0) p->negative++; else p->positive++;
        p->count++; p->moved += !!moved; p->slept += !!slept; p->elapsed += elapsed;
        if (elapsed > p->max_elapsed) p->max_elapsed = elapsed;
        if (run > p->max_run) p->max_run = run;
    }
    return !s->remaining;
}
