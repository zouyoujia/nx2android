// Read-only ELF64/AArch64 PLT inspection. All dereferenced dynamic data must be
// inside a readable PT_LOAD; malformed metadata is rejected, never repaired.
#pragma once
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

struct kgsl_diag_phdr { uint32_t type, flags; uint64_t offset, vaddr, paddr, filesz, memsz, align; };
struct kgsl_diag_dyn { int64_t tag; uint64_t value; };
struct kgsl_diag_sym { uint32_t name; uint8_t info, other; uint16_t shndx; uint64_t value, size; };
struct kgsl_diag_rela { uint64_t offset, info; int64_t addend; };
struct kgsl_diag_image { uintptr_t base; const struct kgsl_diag_phdr *phdr; size_t phnum; };
typedef void (*kgsl_diag_slot_fn)(uintptr_t slot, uintptr_t target, void *opaque);

static bool kgsl_diag_add(uintptr_t a, uint64_t b, uintptr_t *out) {
    if (b > UINTPTR_MAX - a) return false;
    *out = a + (uintptr_t)b; return true;
}
static bool kgsl_diag_readable(const struct kgsl_diag_image *im, uintptr_t p, uint64_t size) {
    uintptr_t end; if (!size || !kgsl_diag_add(p, size, &end)) return false;
    for (size_t i = 0; i < im->phnum; i++) {
        const struct kgsl_diag_phdr *h = &im->phdr[i]; uintptr_t lo, hi;
        if (h->type == 1 && (h->flags & 4) && kgsl_diag_add(im->base, h->vaddr, &lo) && kgsl_diag_add(lo, h->memsz, &hi) && p >= lo && end <= hi) return true;
    }
    return false;
}
static uintptr_t kgsl_diag_pointer(const struct kgsl_diag_image *im, uint64_t value, uint64_t size) {
    uintptr_t p;
    // Bionic retains relative dynamic pointers. Also accept already relocated
    // pointers so the parser's bounds can be tested on ordinary host ELFs.
    if (kgsl_diag_add(im->base, value, &p) && kgsl_diag_readable(im, p, size)) return p;
    return value <= UINTPTR_MAX && kgsl_diag_readable(im, (uintptr_t)value, size) ? (uintptr_t)value : 0;
}
static int kgsl_diag_ioctl_slots(const struct kgsl_diag_image *im, kgsl_diag_slot_fn emit, void *opaque, const char **why) {
    *why = "invalid module headers";
    if (!im || !im->phdr || !im->phnum || im->phnum > 1024 || !emit) return -1;
    uintptr_t dynamic = 0; uint64_t dynamic_size = 0;
    for (size_t i = 0; i < im->phnum; i++) if (im->phdr[i].type == 2) {
        if (dynamic || !kgsl_diag_add(im->base, im->phdr[i].vaddr, &dynamic)) return -1;
        dynamic_size = im->phdr[i].memsz;
    }
    if (!dynamic || !kgsl_diag_readable(im, dynamic, dynamic_size) || dynamic_size / sizeof(struct kgsl_diag_dyn) > 4096) return -1;
    uint64_t str = 0, strsz = 0, sym = 0, syment = 0, rel = 0, relsz = 0, relkind = 0; bool terminated = false;
    for (size_t i = 0; i < dynamic_size / sizeof(struct kgsl_diag_dyn); i++) {
        struct kgsl_diag_dyn d; memcpy(&d, (const void *)(dynamic + i * sizeof d), sizeof d);
        if (!d.tag) { terminated = true; break; }
        switch (d.tag) {
        case 5: str = d.value; break; case 10: strsz = d.value; break;
        case 6: sym = d.value; break; case 11: syment = d.value; break;
        case 23: rel = d.value; break; case 2: relsz = d.value; break; case 20: relkind = d.value; break;
        }
    }
    *why = "unterminated PT_DYNAMIC"; if (!terminated) return -1;
    *why = "no PLT relocations"; if (!rel || !relsz) return 0;
    *why = "not ELF64 RELA"; if (relkind != 7 || relsz % sizeof(struct kgsl_diag_rela)) return -1;
    *why = "invalid string/symbol/relocation bounds";
    if (!str || !sym || !strsz || syment != sizeof(struct kgsl_diag_sym)) return -1;
    uintptr_t strings = kgsl_diag_pointer(im, str, strsz), symbols = kgsl_diag_pointer(im, sym, syment), relocs = kgsl_diag_pointer(im, rel, relsz);
    if (!strings || !symbols || !relocs) return -1;
    int found = 0;
    for (size_t i = 0; i < relsz / sizeof(struct kgsl_diag_rela); i++) {
        struct kgsl_diag_rela r; memcpy(&r, (const void *)(relocs + i * sizeof r), sizeof r);
        if ((uint32_t)r.info != 1026) continue;   // R_AARCH64_JUMP_SLOT
        uint64_t index = r.info >> 32; uintptr_t address;
        if (index > UINT64_MAX / syment || !kgsl_diag_add(symbols, index * syment, &address) || !kgsl_diag_readable(im, address, syment)) return -1;
        struct kgsl_diag_sym s; memcpy(&s, (const void *)address, sizeof s);
        if (s.name >= strsz || !memchr((const void *)(strings + s.name), 0, (size_t)(strsz - s.name))) return -1;
        if (s.shndx || strcmp((const char *)(strings + s.name), "ioctl")) continue;
        uintptr_t slot, target;
        *why = "unreadable ioctl slot";
        if (!kgsl_diag_add(im->base, r.offset, &slot) || slot % sizeof(uintptr_t) || !kgsl_diag_readable(im, slot, sizeof target)) return -1;
        memcpy(&target, (const void *)slot, sizeof target); emit(slot, target, opaque); found++;
    }
    *why = found ? "read-only ioctl binding found" : "no imported ioctl JUMP_SLOT";
    return found;
}
