// Bounded exact-size cache for completed small copied staging allocations.
#pragma once
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
enum { STAGING_CACHE_SLOTS = 8, STAGING_CACHE_ALLOCATION_MAX = 128 << 10 };
typedef struct { VkBuffer buf; VkDeviceMemory mem; void *map; uint64_t size, allocated; } staging_entry;
static struct { pthread_mutex_t mu; staging_entry entries[STAGING_CACHE_SLOTS]; uint64_t hits, misses, returned, full; } g_staging_cache = {.mu = PTHREAD_MUTEX_INITIALIZER};
static inline int staging_cache_enabled(void) {
    static int value = -1; int on = __atomic_load_n(&value, __ATOMIC_ACQUIRE);
    if (on < 0) { const char *e = getenv("DM_VK_STAGING_CACHE"); on = e && atoi(e) == 1; __atomic_store_n(&value, on, __ATOMIC_RELEASE); }
    return on;
}
static inline int staging_cache_eligible(const staging_entry *e) {
    return e->buf && e->mem && e->map && e->size && e->size < (64u << 10) && e->allocated >= e->size && e->allocated <= STAGING_CACHE_ALLOCATION_MAX;
}
static inline int staging_cache_take(uint64_t size, staging_entry *out) {
    if (!staging_cache_enabled() || !size || size >= (64u << 10)) return 0;
    int found = 0; pthread_mutex_lock(&g_staging_cache.mu);
    for (int i = 0; i < STAGING_CACHE_SLOTS; i++) if (g_staging_cache.entries[i].buf && g_staging_cache.entries[i].size == size) {
        *out = g_staging_cache.entries[i]; memset(&g_staging_cache.entries[i], 0, sizeof *out); found = 1; break;
    }
    if (found) g_staging_cache.hits++; else g_staging_cache.misses++;
    pthread_mutex_unlock(&g_staging_cache.mu); return found;
}
// Caller must establish GPU completion and post-effect retirement before return.
static inline int staging_cache_return(const staging_entry *entry) {
    if (!staging_cache_enabled() || !staging_cache_eligible(entry)) return 0;
    int stored = 0; pthread_mutex_lock(&g_staging_cache.mu);
    for (int i = 0; i < STAGING_CACHE_SLOTS; i++) if (!g_staging_cache.entries[i].buf) { g_staging_cache.entries[i] = *entry; stored = 1; break; }
    if (stored) g_staging_cache.returned++; else g_staging_cache.full++;
    pthread_mutex_unlock(&g_staging_cache.mu); return stored;
}
