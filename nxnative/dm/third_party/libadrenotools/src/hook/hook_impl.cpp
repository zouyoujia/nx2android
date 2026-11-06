#include <initializer_list>
#include <string>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <link.h>
#include <errno.h>
#include "kgsl_binding_elf.h"
#include <fcntl.h>
#include <sys/ioctl.h>
#include <android_linker_ns.h>
#include <android/dlext.h>
#include <android/log.h>
#include "kgsl.h"
#include "hook_impl_params.h"
#include "hook_impl.h"

#define TAG "hook_impl"
#define LOGI(fmt, ...) __android_log_print(ANDROID_LOG_INFO, TAG, fmt, ##__VA_ARGS__)

const HookImplParams *hook_params; //!< Bunch of info needed to load/patch the driver
int (*gsl_memory_alloc_pure_sym)(uint32_t, uint32_t, void *);
int (*gsl_memory_alloc_pure_64_sym)(uint64_t, uint32_t, void *);
int (*gsl_memory_free_pure_sym)(void *);
int kgsl_fd;

// Executed from the driver namespace's own hook_impl instance, after real work.
static void *kgsl_diag_driver, *kgsl_diag_power;
extern "C" __attribute__((visibility("default"))) void init_kgsl_binding_diagnostic(void *driver, void *power) { kgsl_diag_driver = driver; kgsl_diag_power = power; }
struct KgslBindingDiagnostic {
    uintptr_t expected;
    struct Slot { uintptr_t base, slot, target; char module[256]; } slots[16];
    size_t used; const char *module; uintptr_t base;
};
static void collect_kgsl_slot(uintptr_t slot, uintptr_t target, void *opaque) {
    auto *d = static_cast<KgslBindingDiagnostic *>(opaque);
    if (d->used == 16) return;
    auto &s = d->slots[d->used++]; s.base = d->base; s.slot = slot; s.target = target;
    snprintf(s.module, sizeof s.module, "%s", d->module);
}
static int find_kgsl_binding(dl_phdr_info *info, size_t, void *opaque) {
    auto *d = static_cast<KgslBindingDiagnostic *>(opaque);
    const char *name = info->dlpi_name ? info->dlpi_name : "", *base = strrchr(name, '/'); base = base ? base + 1 : name;
    if (strcmp(base, "libgsl.so") && strcmp(base, "vkbgsl.so") && strcmp(base, "notgsl.so") && (!d->expected || info->dlpi_addr != d->expected)) return 0;
    static_assert(sizeof(kgsl_diag_phdr) == sizeof(Elf64_Phdr), "ELF64 program header ABI");
    kgsl_diag_image image{static_cast<uintptr_t>(info->dlpi_addr), reinterpret_cast<const kgsl_diag_phdr *>(info->dlpi_phdr), info->dlpi_phnum};
    d->module = name; d->base = image.base; const char *why;
    int n = kgsl_diag_ioctl_slots(&image, collect_kgsl_slot, d, &why);
    if (n <= 0) LOGI("KGSL binding module %s base %p: %s; uncovered", name, reinterpret_cast<void *>(image.base), why);
    return 0;
}
extern "C" __attribute__((visibility("default"))) bool diagnose_kgsl_binding(void) {
    int saved = errno; Dl_info from{};
    void *context = kgsl_diag_driver ? dlsym(kgsl_diag_driver, "gsl_context_create") : nullptr;
    bool located = context && dladdr(context, &from);
    LOGI("KGSL driver-handle gsl_context_create %p in %s base %p (auxiliary)", context, located && from.dli_fname ? from.dli_fname : "unknown", located ? from.dli_fbase : nullptr);
    void *aux = dlsym(RTLD_DEFAULT, "ioctl"); Dl_info auxiliary{}; bool aux_found = aux && dladdr(aux, &auxiliary);
    LOGI("KGSL namespace dlsym ioctl %p in %s!%s (auxiliary, not PLT proof)", aux, aux_found && auxiliary.dli_fname ? auxiliary.dli_fname : "unknown", aux_found && auxiliary.dli_sname ? auxiliary.dli_sname : "unknown");
    KgslBindingDiagnostic data{}; data.expected = located ? reinterpret_cast<uintptr_t>(from.dli_fbase) : 0;
    dl_iterate_phdr(find_kgsl_binding, &data);
    for (size_t i = 0; i < data.used; i++) {
        const auto &s = data.slots[i]; Dl_info target{}; bool resolved = dladdr(reinterpret_cast<void *>(s.target), &target);
        LOGI("KGSL read-only PLT: %s base %p (driver-handle dependency match %d) ioctl slot %p -> %p in %s!%s", s.module, reinterpret_cast<void *>(s.base), s.base == data.expected, reinterpret_cast<void *>(s.slot), reinterpret_cast<void *>(s.target), resolved && target.dli_fname ? target.dli_fname : "unknown", resolved && target.dli_sname ? target.dli_sname : "unknown");
    }
    if (!data.used) LOGI("KGSL binding: no readable GSL ioctl PLT slot found; coverage unverified");
    if (kgsl_diag_power) {
        auto detail = reinterpret_cast<void (*)(void)>(dlsym(kgsl_diag_power, "diagnose_kgsl_power_hook")); if (detail) detail();
    }
    errno = saved; return data.used > 0;
}

using gsl_memory_alloc_pure_t = decltype(gsl_memory_alloc_pure_sym);
using gsl_memory_alloc_pure_64_t = decltype(gsl_memory_alloc_pure_64_sym);
using gsl_memory_free_pure_t = decltype(gsl_memory_free_pure_sym);

__attribute__((visibility("default"))) void init_hook_param(const void *param) {
    hook_params = reinterpret_cast<const HookImplParams *>(param);
}

__attribute__((visibility("default"))) void init_gsl(void *alloc, void *alloc64, void *free) {
    gsl_memory_alloc_pure_sym = reinterpret_cast<gsl_memory_alloc_pure_t>(alloc);
    gsl_memory_alloc_pure_64_sym = reinterpret_cast<gsl_memory_alloc_pure_64_t>(alloc64);
    gsl_memory_free_pure_sym = reinterpret_cast<gsl_memory_free_pure_t>(free);
}

__attribute__((visibility("default"))) void *hook_android_dlopen_ext(const char *filename, int flags, const android_dlextinfo *extinfo) {
    auto fallback{[&]() {
        LOGI("hook_android_dlopen_ext: falling back!");
        return android_dlopen_ext(filename, flags, extinfo);
    }};

    LOGI("hook_android_dlopen_ext: filename: %s", filename ? filename : "(null)");

    // Ignore non-vulkan libraries
    if (!filename || !strstr(filename, "vulkan."))
        return android_dlopen_ext(filename, flags, extinfo);

    if (!hook_params || !extinfo || extinfo->library_namespace == nullptr || !(extinfo->flags & ANDROID_DLEXT_USE_NAMESPACE)) {
        LOGI("hook_android_dlopen_ext: hook failed: namespace not supplied!");
        return fallback();
    }

    const bool kgslPerf = hook_params->featureFlags & ADRENOTOOLS_DRIVER_KGSL_PERF;
    std::string paths;
    if (kgslPerf) {
        // A shared namespace inherits already loaded vendor libraries, including libgsl's
        // existing ioctl relocations. Use an escaped regular namespace whose parent is the
        // platform default, not the vendor's shared group (even REGULAR inherits GLOBALs).
        // Load the original driver and private dependencies after our interposer. Never
        // link-all back to vendor: that would silently reuse those unhooked instances.
        std::string file = filename;
        auto slash = file.find_last_of('/');
        if (slash != std::string::npos) {
            std::string dir = file.substr(0, slash); paths = dir;
            if (dir.size() >= 3 && dir.compare(dir.size() - 3, 3, "/hw") == 0) paths += ":" + dir.substr(0, dir.size() - 3);
        }
        if (!paths.empty()) paths += ":";
        paths += "/vendor/lib64:/vendor/lib64/hw:/odm/lib64:/odm/lib64/hw:" + hook_params->hookLibDir;
    } else paths = hook_params->hookLibDir;
    // The empty custom path preserves stock-driver selection; no custom binary is needed.
    auto driverNs{kgslPerf
        ? android_create_namespace_escape(filename, hook_params->customDriverDir.c_str(), paths.c_str(), ANDROID_NAMESPACE_TYPE_REGULAR, nullptr, nullptr)
        : android_create_namespace(filename, hook_params->customDriverDir.c_str(), paths.c_str(), ANDROID_NAMESPACE_TYPE_SHARED, nullptr, extinfo->library_namespace)};
    if (!driverNs) {
        LOGI("hook_android_dlopen_ext: hook failed: namespace not supplied!");
        return fallback();
    }

    // Share platform services, but keep the driver and its private libraries local when
    // interposing ioctls. Unavailable vendor dependencies cause a normal-driver fallback.
    if (kgslPerf) {
        // Namespace links are not transitive. Stock Adreno needs VNDK variants of
        // libcutils/C++/mapper interfaces: linking only to sphal cannot follow its
        // own vndk link (observed: libcutils not found, ordinary-driver fallback).
        // Keep GPU-private driver/GSL/compiler libraries out of both allowlists.
        auto vndk{android_get_exported_namespace ? android_get_exported_namespace("vndk") : nullptr};
        if (vndk) {
            if (!android_link_namespaces(driverNs, vndk,
                "libcutils.so:libbase.so:libc++.so:libhardware.so:libhidlbase.so:libutils.so:libz.so:libdmabufheap.so:android.hardware.graphics.mapper@2.0.so:android.hardware.graphics.mapper@2.1.so:android.hardware.graphics.mapper@3.0.so:android.hardware.graphics.mapper@4.0.so:android.hardware.graphics.common@1.0.so:android.hardware.graphics.common@1.1.so:android.hardware.graphics.common@1.2.so")) {
                LOGI("KGSL VNDK dependency link failed; original driver retained without a performance request");
                return fallback();
            }
        } else if (!(hook_params->featureFlags & ADRENOTOOLS_DRIVER_CUSTOM)) {
            LOGI("KGSL stock namespace requires an exported VNDK provider; original driver retained without a performance request");
            return fallback();
        }
        const char *system = "libc.so:libdl.so:libm.so:liblog.so:libandroid.so:libnativewindow.so:libsync.so";
        if (!android_link_namespaces(driverNs, extinfo->library_namespace, system) || !android_link_namespaces(driverNs, nullptr, system)) {
            LOGI("KGSL system dependency link failed; original driver retained without a performance request");
            return fallback();
        }
    } else android_link_namespaces(driverNs, nullptr, "libandroid.so");

    // Preload ourself, a new instance will be created since we have different linker ancestory
    // If we don't preload we get a weird issue where despite being in NEEDED of the hook lib the hook's symbols will overwrite ours and cause an infinite loop
    std::string implPath = hook_params->hookLibDir + "/libhook_impl.so";
    auto hookImpl{linkernsbypass_namespace_dlopen(kgslPerf ? implPath.c_str() : "libhook_impl.so", RTLD_NOW, driverNs)};
    if (!hookImpl) {
        if (kgslPerf) {
            LOGI("KGSL namespace implementation unavailable; original driver retained without a performance request");
            return fallback();
        }
        return nullptr;
    }

    // Pass parameters to ourself
    auto initHookParam{reinterpret_cast<void (*)(const void *)>(dlsym(hookImpl, "init_hook_param"))};
    if (!initHookParam) {
        if (kgslPerf) {
            LOGI("KGSL namespace parameters unavailable; original driver retained without a performance request");
            return fallback();
        }
        return nullptr;
    }

    initHookParam(hook_params);

    void *powerHookForDiagnostic = nullptr;
    if (kgslPerf) {
        std::string powerPath = hook_params->hookLibDir + "/libkgsl_power_hook.so";
        auto powerHook{linkernsbypass_namespace_dlopen(powerPath.c_str(), RTLD_NOW | RTLD_GLOBAL, driverNs)};
        auto initPower{powerHook ? reinterpret_cast<void (*)(adrenotools_kgsl_perf_stats *)>(dlsym(powerHook, "init_kgsl_power_hook")) : nullptr};
        if (!initPower || !hook_params->kgslPerfStats) {
            LOGI("KGSL context hook unavailable; original driver retained without a performance request");
            return fallback();
        }
        initPower(hook_params->kgslPerfStats); powerHookForDiagnostic = powerHook;
        LOGI("KGSL context hook armed in %s driver namespace; actual context interception/acceptance must be checked", hook_params->featureFlags & ADRENOTOOLS_DRIVER_CUSTOM ? "custom" : "stock");
    }

    if (hook_params->featureFlags & ADRENOTOOLS_DRIVER_FILE_REDIRECT) {
        if (!linkernsbypass_namespace_dlopen("libfile_redirect_hook.so", RTLD_GLOBAL, driverNs)) {
            LOGI("hook_android_dlopen_ext: hook failed: failed to apply libfopen_redirect_hook!");
            return fallback();
        }

        LOGI("hook_android_dlopen_ext: applied libfile_redirect_hook");
    }

    // Use our new namespace to load the vulkan driver
    auto newExtinfo{*extinfo};
    newExtinfo.library_namespace = driverNs;

    if (hook_params->featureFlags & ADRENOTOOLS_DRIVER_GPU_MAPPING_IMPORT) {
        if (!linkernsbypass_namespace_dlopen("libgsl_alloc_hook.so", RTLD_GLOBAL, driverNs)) {
            LOGI("hook_android_dlopen_ext: hook failed: failed to apply libgsl_alloc_hook!");
            return fallback();
        }

        auto libgslHandle{android_dlopen_ext("vkbgsl.so", RTLD_NOW, &newExtinfo)};
        if (!libgslHandle) {
            libgslHandle = android_dlopen_ext("notgsl.so", RTLD_NOW, &newExtinfo);
            if (!libgslHandle)
                libgslHandle = android_dlopen_ext("libgsl.so", RTLD_NOW, &newExtinfo);
        }

        if (libgslHandle) {
            gsl_memory_alloc_pure_sym = reinterpret_cast<decltype(gsl_memory_alloc_pure_sym)>(dlsym(libgslHandle, "gsl_memory_alloc_pure"));
            gsl_memory_alloc_pure_64_sym = reinterpret_cast<decltype(gsl_memory_alloc_pure_64_sym)>(dlsym(libgslHandle, "gsl_memory_alloc_pure_64"));
            gsl_memory_free_pure_sym = reinterpret_cast<decltype(gsl_memory_free_pure_sym)>(dlsym(libgslHandle, "gsl_memory_free_pure"));
            if ((gsl_memory_alloc_pure_sym || gsl_memory_alloc_pure_64_sym) && gsl_memory_free_pure_sym) {
                auto initGsl{reinterpret_cast<void (*)(gsl_memory_alloc_pure_t, gsl_memory_alloc_pure_64_t, gsl_memory_free_pure_t)>(dlsym(hookImpl, "init_gsl"))};
                if (!initGsl)
                    return fallback();

                initGsl(gsl_memory_alloc_pure_sym, gsl_memory_alloc_pure_64_sym, gsl_memory_free_pure_sym);
                LOGI("hook_android_dlopen_ext: applied libgsl_alloc_hook");
                hook_params->nextGpuMapping->gpu_addr = ADRENOTOOLS_GPU_MAPPING_SUCCEEDED_MAGIC;
            }
        }

        if (!((gsl_memory_alloc_pure_sym || gsl_memory_alloc_pure_64_sym) && gsl_memory_free_pure_sym))
            LOGI("hook_android_dlopen_ext: hook failed: failed to apply libgsl_alloc_hook!");
    }

    auto registerDiagnostic = [&](void *driver) {
        if (!kgslPerf || !driver || !hook_params->kgslPerfDiagnose) return;
        auto init = reinterpret_cast<void (*)(void *, void *)>(dlsym(hookImpl, "init_kgsl_binding_diagnostic"));
        auto diagnose = reinterpret_cast<bool (*)(void)>(dlsym(hookImpl, "diagnose_kgsl_binding"));
        if (init && diagnose) { init(driver, powerHookForDiagnostic); __atomic_store_n(hook_params->kgslPerfDiagnose, diagnose, __ATOMIC_RELEASE); }
        else LOGI("KGSL read-only binding diagnostic unavailable");
    };

    // TODO: If there is already an instance of a Vulkan driver loaded hooks won't be applied, this will only be the case for skiavk generally
    // To fix this we would need to search /proc/self/maps for the file to a loaded instance of the library in order to read it to patch the soname and load it uniquely
    if (hook_params->featureFlags & ADRENOTOOLS_DRIVER_CUSTOM) {
        LOGI("hook_android_dlopen_ext: loading custom driver: %s%s", hook_params->customDriverDir.c_str(), hook_params->customDriverName.c_str());
        void *handle{android_dlopen_ext(hook_params->customDriverName.c_str(), flags, &newExtinfo)};
        if (!handle) {
            LOGI("hook_android_dlopen_ext: hook failed: failed to load custom driver: %s!", dlerror());
            return fallback();
        }

        registerDiagnostic(handle); return handle;
    } else {
        LOGI("hook_android_dlopen_ext: loading default driver: %s", filename);
        void *handle{android_dlopen_ext(filename, flags, &newExtinfo)};
        if (!handle && kgslPerf) {
            LOGI("isolated stock driver could not load: %s; original driver retained without a performance request", dlerror());
            return fallback();
        }
        registerDiagnostic(handle); return handle;
    }
}

__attribute__((visibility("default"))) void *hook_android_load_sphal_library(const char *filename, int flags) {
    LOGI("hook_android_load_sphal_library: filename: %s", filename);

    // https://android.googlesource.com/platform/system/core/+/master/libvndksupport/linker.cpp
    for (const char *name : {"sphal", "vendor", "default"}) {
        if (auto vendorNs{android_get_exported_namespace(name)}) {
            android_dlextinfo dlextinfo{
                .flags = ANDROID_DLEXT_USE_NAMESPACE,
                .library_namespace = vendorNs,
            };

            return hook_android_dlopen_ext(filename, flags, &dlextinfo);
        }
    }

    return nullptr;
}

__attribute__((visibility("default"))) FILE *hook_fopen(const char *filename, const char *mode) {
    if (!strncmp("/proc", filename, 5) || !strncmp("/sys", filename, 4)) {
        LOGI("hook_fopen: passthrough: %s", filename);
        return fopen(filename, mode);
    }

    auto replacement{hook_params->fileRedirectDir + filename};
    LOGI("hook_fopen: %s -> %s", filename, replacement.c_str());

    return fopen(replacement.c_str(), mode);
}

static constexpr uintptr_t GslMemDescImportedPrivMagic{0xdeadb33f};
struct GslMemDesc {
    void *hostptr;
    uint64_t gpuaddr;
    uint64_t size;
    uint64_t flags;
    uintptr_t priv;
};

__attribute__((visibility("default"))) int hook_gsl_memory_alloc_pure_64(uint64_t size, uint32_t flags, void *memDesc) {
    auto gslMemDesc{reinterpret_cast<GslMemDesc *>(memDesc)};
    if (hook_params->nextGpuMapping && hook_params->nextGpuMapping->size == size && (hook_params->nextGpuMapping->flags & flags) == hook_params->nextGpuMapping->flags) {
        auto &nextMapping{*hook_params->nextGpuMapping};

        gslMemDesc->hostptr = nextMapping.host_ptr;
        gslMemDesc->gpuaddr = nextMapping.gpu_addr;
        gslMemDesc->size = nextMapping.size;
        gslMemDesc->flags = nextMapping.flags;
        gslMemDesc->priv = GslMemDescImportedPrivMagic;
        hook_params->nextGpuMapping->size = 0;
        hook_params->nextGpuMapping->gpu_addr = ADRENOTOOLS_GPU_MAPPING_SUCCEEDED_MAGIC;
        return 0;
    } else {
        if (gsl_memory_alloc_pure_64_sym)
            return gsl_memory_alloc_pure_64_sym(size, flags, gslMemDesc);
        else
            return gsl_memory_alloc_pure_sym((uint32_t)size, flags, gslMemDesc);
    }
}

__attribute__((visibility("default"))) int hook_gsl_memory_free_pure(void *memDesc) {
    auto gslMemDesc{reinterpret_cast<GslMemDesc *>(memDesc)};

    if (gslMemDesc->priv == GslMemDescImportedPrivMagic) {
        if (!kgsl_fd)
            kgsl_fd = open("/dev/kgsl-3d0", O_RDWR);

        kgsl_gpumem_get_info info{
            .gpuaddr = gslMemDesc->gpuaddr
        };

        if (ioctl(kgsl_fd, IOCTL_KGSL_GPUMEM_GET_INFO, &info) < 0) {
            LOGI("IOCTL_KGSL_GPUMEM_GET_INFO failed");
            return 0;
        }

        kgsl_gpuobj_free args{
            .id = info.id,
        };

        if (ioctl(kgsl_fd, IOCTL_KGSL_GPUOBJ_FREE, &args) < 0)
            LOGI("IOCTL_KGSL_GPUOBJ_FREE failed");

        return 0;
    } else {
        return gsl_memory_free_pure_sym(memDesc);
    }
}
