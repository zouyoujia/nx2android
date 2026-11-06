// Opt-in interposition inside an isolated Vulkan driver's namespace. Use only a context
// returned by that driver on its own fd: no global KGSL power-control or governor changes.
// On Gen7 the context flag makes real submissions apply this request; SETPROPERTY success
// means accepted, not proof that hardware has activated it. Destruction/retirement owns cleanup.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <android/log.h>
#include <errno.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#include "../../third_party/libadrenotools/include/adrenotools/priv.h"

struct kgsl_perf_create { uint32_t flags, id; };
struct kgsl_perf_constraint { uint32_t type, context_id; void *data; size_t size; };
struct kgsl_perf_property { uint32_t type; void *value; size_t size; };
#define KGSL_PERF_CREATE _IOWR(0x09, 0x13, struct kgsl_perf_create)
#define KGSL_PERF_SETPROPERTY _IOW(0x09, 0x32, struct kgsl_perf_property)
#define KGSL_PERF_CONTEXT_CONSTRAINT 0x800u
#define KGSL_PERF_PROP_CONSTRAINT 0x12u
#define KGSL_PERF_CONSTRAINT_PWRLEVEL 1u
#define KGSL_PERF_PWR_MAX 1u
_Static_assert(sizeof(struct kgsl_perf_create) == 8 && sizeof(struct kgsl_perf_constraint) == 24 && sizeof(struct kgsl_perf_property) == 24, "KGSL ARM64 ABI");
#ifdef __ANDROID__
_Static_assert(KGSL_PERF_CREATE == 0xc0080913ul && KGSL_PERF_SETPROPERTY == 0x40180932ul, "KGSL ioctl ABI");
#endif

static struct adrenotools_kgsl_perf_stats *g_stats;
static uint64_t g_any_ioctl, g_exact_create, g_fd_rejected;
static int g_mode;   // DM_GPU_PERF=2 observes the identical namespace/ioctl path without changing it.
__attribute__((visibility("default"))) void init_kgsl_power_hook(struct adrenotools_kgsl_perf_stats *stats) {
    int saved = errno; const char *mode = getenv("DM_GPU_PERF");
    g_stats = stats; g_mode = mode && !strcmp(mode, "2") ? 2 : 1;
    __android_log_print(ANDROID_LOG_INFO, "nx2a-power", "KGSL context hook mode %d: %s", g_mode, g_mode == 2 ? "observe only; context flags and constraints unchanged" : "request MAX only on owned contexts; activation not yet verified");
    errno = saved;
}
__attribute__((visibility("default"))) void diagnose_kgsl_power_hook(void) {
    int saved = errno;
    __android_log_print(ANDROID_LOG_INFO, "nx2a-power", "KGSL hook entry counters: any_ioctl %llu, exact_create %llu, fd_rejected %llu, initialized %d, mode %d", (unsigned long long)__atomic_load_n(&g_any_ioctl, __ATOMIC_RELAXED), (unsigned long long)__atomic_load_n(&g_exact_create, __ATOMIC_RELAXED), (unsigned long long)__atomic_load_n(&g_fd_rejected, __ATOMIC_RELAXED), g_stats != NULL, g_mode);
    errno = saved;
}
#define COUNT(field) __atomic_fetch_add(&g_stats->field, 1, __ATOMIC_RELAXED)

// The ioctl number alone does not identify the device. Do not change a matching request
// made on any unrelated fd; a denied procfs lookup leaves the original operation intact.
static int kgsl_owned_fd(int fd) {
    char path[64], name[64];
    snprintf(path, sizeof path, "/proc/self/fd/%d", fd);
    ssize_t n = readlink(path, name, sizeof name - 1); int error = n < 0 ? errno : 0;
    int valid = n >= 0 && (size_t)n < sizeof name - 1; if (valid) name[n] = 0;
    static uint64_t checks;
    if (__atomic_fetch_add(&checks, 1, __ATOMIC_RELAXED) < 4)
        __android_log_print(ANDROID_LOG_INFO, "nx2a-power", "KGSL fd check: fd %d, readlink %zd, errno %d, path %s", fd, n, error, valid ? name : "unavailable");
    return valid && !strcmp(name, "/dev/kgsl-3d0");
}

#ifdef __ANDROID__
__attribute__((visibility("default"))) int ioctl(int fd, int request, ...) {
    unsigned long op = (uint32_t)request;   // bionic's request is a signed 32-bit ioctl command
#else
__attribute__((visibility("default"))) int ioctl(int fd, unsigned long request, ...) {
    unsigned long op = request;
#endif
    // Like bionic ioctl, the driver-facing ABI passes its third argument through unchanged.
    va_list ap; va_start(ap, request); void *arg = va_arg(ap, void *); va_end(ap);
    int entry_errno = errno;
    if (__atomic_fetch_add(&g_any_ioctl, 1, __ATOMIC_RELAXED) == 0)
        __android_log_print(ANDROID_LOG_INFO, "nx2a-power", "KGSL hook first ioctl: fd %d request %#lx initialized %d", fd, op, g_stats != NULL);
    if (op == KGSL_PERF_CREATE && __atomic_fetch_add(&g_exact_create, 1, __ATOMIC_RELAXED) == 0)
        __android_log_print(ANDROID_LOG_INFO, "nx2a-power", "KGSL hook first exact CREATE: fd %d arg %p initialized %d", fd, arg, g_stats != NULL);
    errno = entry_errno;
    if (!g_stats || op != KGSL_PERF_CREATE || !arg) return (int)syscall(SYS_ioctl, fd, op, arg);
    int ours = kgsl_owned_fd(fd); errno = entry_errno;
    if (!ours) { __atomic_fetch_add(&g_fd_rejected, 1, __ATOMIC_RELAXED); return (int)syscall(SYS_ioctl, fd, op, arg); }
    COUNT(intercepted);
    if (g_mode == 2) return (int)syscall(SYS_ioctl, fd, op, arg);
    struct kgsl_perf_create *create = arg, requested = *create;
    uint32_t original = requested.flags;
    requested.flags |= KGSL_PERF_CONTEXT_CONSTRAINT;
    int result = (int)syscall(SYS_ioctl, fd, op, &requested);
    if (result < 0) {
        // Unsupported added flags report EINVAL. Do not retry EINTR or a general device
        // failure: repeating a possibly completed create could leak or duplicate a context.
        if (errno == EINVAL && !(original & KGSL_PERF_CONTEXT_CONSTRAINT)) {
            COUNT(fallback);
            result = (int)syscall(SYS_ioctl, fd, op, arg);
            int saved = errno;
            __android_log_print(ANDROID_LOG_INFO, "nx2a-power", "KGSL context flag refused; original flags retried (result %d, errno %d); no performance constraint requested", result, result < 0 ? saved : 0);
            errno = saved;
        }
        return result;
    }
    *create = requested;
    int saved = errno;
    if (!(requested.flags & KGSL_PERF_CONTEXT_CONSTRAINT)) {
        __android_log_print(ANDROID_LOG_INFO, "nx2a-power", "KGSL context %u omitted the requested constraint flag (input %#x, returned %#x); no performance constraint requested", requested.id, original, requested.flags);
        errno = saved;
        return result;
    }
    COUNT(flagged);
    uint32_t maximum = KGSL_PERF_PWR_MAX;
    struct kgsl_perf_constraint constraint = {KGSL_PERF_CONSTRAINT_PWRLEVEL, create->id, &maximum, sizeof maximum};
    struct kgsl_perf_property property = {KGSL_PERF_PROP_CONSTRAINT, &constraint, sizeof constraint};
    int rc = (int)syscall(SYS_ioctl, fd, KGSL_PERF_SETPROPERTY, &property);
    if (rc) COUNT(refused); else COUNT(accepted);
    __android_log_print(ANDROID_LOG_INFO, "nx2a-power", "KGSL context %u on driver fd %d: MAX request %s (errno %d); activation requires constrained GPU work", create->id, fd, rc ? "refused" : "accepted", rc ? errno : 0);
    errno = saved;
    return result;   // a failed optional property must never break a successfully created context
}
