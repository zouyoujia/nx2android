// Eligibility for the optional CPU path. Exact copy-only records carry no GPU event/fence side effects.
#pragma once
#include "../gfx/nvn/nvn_obj.h"
#include <stdlib.h>

static int cpu_small_upload_mode(void) {
    static int mode = -1; int value = __atomic_load_n(&mode, __ATOMIC_ACQUIRE);
    if (value < 0) { const char *e = getenv("DM_VK_CPU_SMALL_UPLOADS"); value = e ? atoi(e) : 0; if (value != 1 && value != 2) value = 0; __atomic_store_n(&mode, value, __ATOMIC_RELEASE); }
    return value;
}

typedef struct { uint64_t src, dst, size; } cpu_small_upload;
typedef struct { uint64_t lo, hi, base, end; int copied, zero_copy, flags; } cpu_upload_pool;
static int cpu_small_upload_record(int n, const uint64_t *handles, cpu_small_upload *copy) {
    if (n != 1 || !handles || !handles[0]) return 0;
    const uint8_t *p = (const uint8_t *)(uintptr_t)handles[0]; op_hdr a, end;
    memcpy(&a, p, sizeof a); if (a.op != OP_COPY_B2B || a.size != 32) return 0;
    memcpy(&end, p + 32, sizeof end); if (end.op != OP_END || end.size != 8) return 0;
    memcpy(copy, p + 8, sizeof *copy);
    if (!copy->size || copy->size > (64u << 10) || copy->src > UINT64_MAX - copy->size || copy->dst > UINT64_MAX - copy->size) return 0;
    if (copy->src < copy->dst + copy->size && copy->dst < copy->src + copy->size) return 0;
    return 1;
}
static int cpu_small_upload_pools(const cpu_small_upload *c, const cpu_upload_pool *src, const cpu_upload_pool *dst) {
    return src->copied && !src->zero_copy && (src->flags & 0x2) && src->end >= src->base && src->end - src->base < (64u << 10)
        && dst->zero_copy && c->src >= src->lo && c->src + c->size <= src->hi && c->dst >= dst->lo && c->dst + c->size <= dst->hi;
}
