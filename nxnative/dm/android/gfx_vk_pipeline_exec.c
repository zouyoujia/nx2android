// Default-off compiler inspection: private pipeline clones are never used for rendering or submitted.
// Adreno 750 exposes pipelineExecutableInfo; SPIR-V instruction counts alone did not predict its GPU time.
#include "gfx_vk.h"
#include <ctype.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

enum { PE_KEYS = 4, PE_STATES = 8, PE_STATE_BYTES = 4096, PE_EXE = 32, PE_STATS = 128, PE_IR = 16, PE_FILE = 2*1024*1024, PE_TOTAL = 4*1024*1024 };
static struct { uint64_t keys[PE_KEYS]; unsigned n; char out[1024], base[1024], opt[1024]; } cfg;
static pthread_once_t once = PTHREAD_ONCE_INIT;
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static PFN_vkGetPipelineExecutablePropertiesKHR properties;
static PFN_vkGetPipelineExecutableStatisticsKHR statistics;
static PFN_vkGetPipelineExecutableInternalRepresentationsKHR representations;
static struct { unsigned len, key; unsigned char data[PE_STATE_BYTES]; } seen[PE_STATES];
static unsigned nseen, per_key[PE_KEYS]; static size_t dumped;
int g_vk_pipeline_exec_enabled;
static int copy_path(char *dst, size_t cap, const char *env, int required) {
    const char *s = getenv(env); if (!s || !*s) return !required;
    size_t n = strlen(s); if (n >= cap || s[0] != '/') return 0; memcpy(dst, s, n+1); return 1;
}
static void configure(void) {
    const char *s = getenv("DM_VK_PIPE_EXEC_KEYS"); if (!s || !*s) return;
    while (*s) {
        while (*s == ',' || isspace((unsigned char)*s)) s++;
        if (!*s) break;
        uint64_t k = 0;
        for (unsigned i = 0; i < 16; i++, s++) {
            unsigned char c = (unsigned char)*s;
            if (!isxdigit(c)) goto invalid;
            k = (k << 4) | (c <= '9' ? c-'0' : (unsigned)(tolower(c)-'a'+10));
        }
        if (*s && *s != ',' && !isspace((unsigned char)*s)) goto invalid;
        if (cfg.n == PE_KEYS) goto invalid;
        for (unsigned i = 0; i < cfg.n; i++) if (cfg.keys[i] == k) goto invalid;
        cfg.keys[cfg.n++] = k;
    }
    if (!cfg.n || !copy_path(cfg.out, sizeof cfg.out, "DM_VK_PIPE_EXEC_OUT", 1) ||
        !copy_path(cfg.base, sizeof cfg.base, "DM_VK_PIPE_EXEC_BASE_DIR", 0) ||
        !copy_path(cfg.opt, sizeof cfg.opt, "DM_VK_PIPE_EXEC_OPT_DIR", 0)) goto invalid;
    return;
invalid:
    memset(&cfg, 0, sizeof cfg); VKLOG("[RED] pipeline executable inspection: invalid startup configuration; disabled");
}
int vk_pipeline_exec_requested(void) { pthread_once(&once, configure); return cfg.n != 0; }
void vk_pipeline_exec_init(int enabled) {
    if (!enabled || !vk_pipeline_exec_requested()) return;
    properties = (PFN_vkGetPipelineExecutablePropertiesKHR)vkGetDeviceProcAddr(V.dev, "vkGetPipelineExecutablePropertiesKHR");
    statistics = (PFN_vkGetPipelineExecutableStatisticsKHR)vkGetDeviceProcAddr(V.dev, "vkGetPipelineExecutableStatisticsKHR");
    representations = (PFN_vkGetPipelineExecutableInternalRepresentationsKHR)vkGetDeviceProcAddr(V.dev, "vkGetPipelineExecutableInternalRepresentationsKHR");
    struct stat st;
    if (!properties || !statistics || !representations || (mkdir(cfg.out, 0700) && errno != EEXIST) || stat(cfg.out, &st) || !S_ISDIR(st.st_mode)) {
        VKLOG("[RED] pipeline executable inspection: entry points or output directory unavailable; disabled"); return;
    }
    g_vk_pipeline_exec_enabled = 1;
    VKLOG("pipeline executable inspection: %u fragment keys, at most two states each; private clones, no GPU work, not a frame-time sample", cfg.n);
}
// Validate instruction lengths before walking; a reference match permits only DescriptorSet operands to differ.
static int spv_valid(const uint32_t *w, size_t n) {
    if (n < 5 || w[0] != 0x07230203 || !w[3]) return 0;
    for (size_t i = 5; i < n;) { size_t k = w[i] >> 16; if (!k || k > n-i) return 0; i += k; }
    return 1;
}
static int spv_same(const uint32_t *a, size_t na, const uint32_t *b, size_t nb) {
    if (na != nb || !spv_valid(a, na) || !spv_valid(b, nb) || memcmp(a, b, 20)) return 0;
    for (size_t i = 5; i < na;) {
        size_t k = a[i] >> 16;
        if (a[i] != b[i]) return 0;
        size_t skip = (a[i] & 0xffff) == 71 && k >= 4 && a[i+2] == 34 ? 3 : k;
        if (memcmp(a+i, b+i, skip*4) || (skip < k && memcmp(a+i+skip+1, b+i+skip+1, (k-skip-1)*4))) return 0;
        i += k;
    }
    return 1;
}
static uint32_t *load_spv(const char *dir, uint64_t key, size_t *nw) {
    char path[1200]; *nw = 0; if (!*dir) return NULL;
    snprintf(path, sizeof path, "%s/%016llx.spv", dir, (unsigned long long)key);
    FILE *f = fopen(path, "rb"); if (!f) return NULL;
    uint32_t *w = NULL;
    if (!fseek(f, 0, SEEK_END)) {
        long n = ftell(f);
        if (n >= 20 && n <= PE_FILE && !(n & 3) && !fseek(f, 0, SEEK_SET)) {
            w = malloc((size_t)n);
            if (w && fread(w, 1, (size_t)n, f) == (size_t)n && spv_valid(w, (size_t)n/4)) *nw = (size_t)n/4;
            else { free(w); w = NULL; }
        }
    }
    fclose(f); return w;
}
static void field(FILE *f, const char *s, size_t cap) {
    for (size_t i = 0; i < cap && s[i]; i++) {
        unsigned char c = (unsigned char)s[i];
        if (c <= 32 || c == '\\' || c == 127) fprintf(f, "\\x%02x", c); else fputc(c, f);
    }
}
static void dump_ir(FILE *f, const VkPipelineExecutableInfoKHR *ei, unsigned capture, unsigned exe) {
    uint32_t count = 0; VkResult r = representations(V.dev, ei, &count, NULL);
    fprintf(f, "ir_count\t%u\t%d\t%u\n", exe, r, count); fflush(f);
    if (r != VK_SUCCESS || !count || count > PE_IR) return;
    uint32_t n = count; VkPipelineExecutableInternalRepresentationKHR *ir = calloc(PE_IR, sizeof *ir); if (!ir) return;
    for (unsigned i = 0; i < PE_IR; i++) ir[i].sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INTERNAL_REPRESENTATION_KHR;
    r = representations(V.dev, ei, &n, ir);
    fprintf(f, "ir_metadata\t%u\t%d\t%u\n", exe, r, n);
    if ((r != VK_SUCCESS && r != VK_INCOMPLETE) || n > PE_IR) { free(ir); return; }
    // All-or-nothing bounded data query: oversize items must not be silently passed back with undersized storage.
    void *buf[PE_IR] = {0}; size_t cap[PE_IR] = {0}, total = 0;
    for (unsigned i = 0; i < n; i++) {
        fprintf(f, "ir\t%u\t%u\t%zu\t%u\t", exe, i, ir[i].dataSize, ir[i].isText);
        field(f, ir[i].name, sizeof ir[i].name); fputc('\t', f); field(f, ir[i].description, sizeof ir[i].description); fputc('\n', f);
        cap[i] = ir[i].dataSize;
        if (cap[i] > PE_FILE || cap[i] > PE_TOTAL-total || total+cap[i] > PE_TOTAL-dumped) goto done;
        total += cap[i];
    }
    for (unsigned i = 0; i < n; i++) {
        if (cap[i] && !(buf[i] = malloc(cap[i]))) goto done;
        ir[i].pData = buf[i];
    }
    { uint32_t count = n; r = representations(V.dev, ei, &count, ir);
      fprintf(f, "ir_data\t%u\t%d\t%u\n", exe, r, count);
      if (r != VK_SUCCESS || count > n) goto done;
      for (unsigned i = 0; i < count; i++) {
          if (ir[i].dataSize > cap[i] || ir[i].pData != buf[i] || !ir[i].dataSize) continue;
          char path[1200]; snprintf(path, sizeof path, "%s/capture-%u-exec-%u-ir-%u.%s", cfg.out, capture, exe, i, ir[i].isText ? "txt" : "bin");
          FILE *o = fopen(path, "wb"); if (!o) continue;
          size_t actual = fwrite(buf[i], 1, ir[i].dataSize, o); fclose(o); dumped += actual;
          fprintf(f, "ir_file\t%u\t%u\t%zu\n", exe, i, actual);
      }
    }
done:
    for (unsigned i = 0; i < PE_IR; i++) free(buf[i]);
    free(ir);
}
static void inspect(FILE *f, VkPipeline p, unsigned capture) {
    VkPipelineInfoKHR pi = {VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR, NULL, p};
    uint32_t count = 0; VkResult r = properties(V.dev, &pi, &count, NULL);
    fprintf(f, "executable_count\t%d\t%u\n", r, count); fflush(f);
    if (r != VK_SUCCESS || !count || count > PE_EXE) return;
    uint32_t n = count; VkPipelineExecutablePropertiesKHR *ex = calloc(PE_EXE, sizeof *ex); if (!ex) return;
    for (unsigned i = 0; i < PE_EXE; i++) ex[i].sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR;
    r = properties(V.dev, &pi, &n, ex); fprintf(f, "executables\t%d\t%u\n", r, n);
    if ((r != VK_SUCCESS && r != VK_INCOMPLETE) || n > PE_EXE) { free(ex); return; }
    for (unsigned i = 0; i < n; i++) {
        // This stock driver left the supplied capacity unchanged after filling two entries.
        // Never query an unfilled executable index, even when the returned count says it exists.
        if (!ex[i].stages || (ex[i].stages & ~31u) || !ex[i].name[0]) { fprintf(f, "invalid_executable\t%u\n", i); continue; }
        fprintf(f, "executable\t%u\t%u\t%u\t", i, ex[i].stages, ex[i].subgroupSize);
        field(f, ex[i].name, sizeof ex[i].name); fputc('\t', f); field(f, ex[i].description, sizeof ex[i].description); fputc('\n', f);
        VkPipelineExecutableInfoKHR ei = {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR, NULL, p, i};
        uint32_t count_stats = 0; r = statistics(V.dev, &ei, &count_stats, NULL);
        fprintf(f, "statistic_count\t%u\t%d\t%u\n", i, r, count_stats); fflush(f);
        if (r != VK_SUCCESS || !count_stats || count_stats > PE_STATS) continue;
        uint32_t ns = count_stats; VkPipelineExecutableStatisticKHR *st = calloc(PE_STATS, sizeof *st); if (!st) continue;
        for (unsigned j = 0; j < PE_STATS; j++) st[j].sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR;
        r = statistics(V.dev, &ei, &ns, st); fprintf(f, "statistics\t%u\t%d\t%u\n", i, r, ns);
        if ((r == VK_SUCCESS || r == VK_INCOMPLETE) && ns <= PE_STATS) for (unsigned j = 0; j < ns; j++) {
            fprintf(f, "stat\t%u\t%u\t", i, st[j].format);
            field(f, st[j].name, sizeof st[j].name); fputc('\t', f);
            switch (st[j].format) {
            case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR: fprintf(f, "%u", st[j].value.b32); break;
            case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR: fprintf(f, "%lld", (long long)st[j].value.i64); break;
            case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR: fprintf(f, "%llu", (unsigned long long)st[j].value.u64); break;
            case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_FLOAT64_KHR: fprintf(f, "%.17g", st[j].value.f64); break;
            default: fputs("unknown", f); break;
            }
            fputc('\t', f); field(f, st[j].description, sizeof st[j].description); fputc('\n', f);
        }
        free(st); fflush(f); dump_ir(f, &ei, capture, i);
    }
    free(ex);
}
static void pipeline_clone(const VkGraphicsPipelineCreateInfo *ci, unsigned capture, unsigned state, uint64_t vs, uint64_t fs, const char *label) {
    char path[1200]; snprintf(path, sizeof path, "%s/capture-%u.tsv", cfg.out, capture); FILE *f = fopen(path, "wb"); if (!f) return;
    fprintf(f, "clone\t%u\t%u\t%016llx\t%016llx\t%s\n", capture, state, (unsigned long long)vs, (unsigned long long)fs, label);
    fprintf(f, "device\t%u\t%u\t%u\t", V.props.vendorID, V.props.deviceID, V.props.driverVersion); field(f, V.props.deviceName, sizeof V.props.deviceName); fputc('\n', f);
    VkGraphicsPipelineCreateInfo copy = *ci; copy.flags |= VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR | VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR;
    VkPipeline p = VK_NULL_HANDLE; VkResult r = vkCreateGraphicsPipelines(V.dev, VK_NULL_HANDLE, 1, &copy, NULL, &p);
    fprintf(f, "create\t%d\n", r);
    if (r == VK_SUCCESS && p) inspect(f, p, capture);
    if (p) vkDestroyPipeline(V.dev, p, NULL);
    fclose(f); VKLOG("pipeline executable inspection: capture %u state %u %s fs %016llx result %d", capture, state, label, (unsigned long long)fs, r);
}
void vk_pipeline_exec_capture(const VkGraphicsPipelineCreateInfo *ci, const void *state, size_t bytes, uint64_t vs, const vkshader_t *fs) {
    if (!g_vk_pipeline_exec_enabled || !fs || !bytes || bytes > PE_STATE_BYTES || ci->stageCount > 3) return;
    const char *stop = getenv("DM_VK_PIPE_EXEC_STOP"); if (stop && atoi(stop)) return;
    unsigned key = 0; while (key < cfg.n && cfg.keys[key] != fs->key) key++; if (key == cfg.n) return;
    pthread_mutex_lock(&mu);
    if (nseen == PE_STATES || per_key[key] == 2) goto done;
    for (unsigned i = 0; i < nseen; i++) if (seen[i].key == key && seen[i].len == bytes && !memcmp(seen[i].data, state, bytes)) goto done;
    unsigned id = nseen++; seen[id].key = key; seen[id].len = (unsigned)bytes; memcpy(seen[id].data, state, bytes); per_key[key]++;
    pipeline_clone(ci, 2*id, id, vs, fs->key, "runtime-clone");
    // Save the exact runtime reference for offline preparation; never silently compare different shader variants.
    { char path[1200]; snprintf(path, sizeof path, "%s/runtime-%016llx.spv", cfg.out, (unsigned long long)fs->key);
      FILE *f = fopen(path, "wb"); if (f) { if (fs->spv) fwrite(fs->spv, 4, fs->nspv, f); fclose(f); } }
    VkPipelineShaderStageCreateInfo stages[3]; memcpy(stages, ci->pStages, ci->stageCount*sizeof *stages);
    unsigned frag = 0; while (frag < ci->stageCount && stages[frag].stage != VK_SHADER_STAGE_FRAGMENT_BIT) frag++;
    if (frag == ci->stageCount || stages[frag].module != fs->mod || !fs->spv) goto done;
    size_t nb = 0, no = 0; uint32_t *base = load_spv(cfg.base, fs->key, &nb), *opt = load_spv(cfg.opt, fs->key, &no);
    if (base && opt && spv_same(base, nb, fs->spv, fs->nspv)) {
        for (size_t i = 5; i < no;) { unsigned n = opt[i] >> 16; if ((opt[i] & 0xffff) == 71 && n >= 4 && opt[i+2] == 34) opt[i+3] = (uint32_t)fs->set; i += n; }
        VkShaderModule m = VK_NULL_HANDLE; VkShaderModuleCreateInfo mi = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, no*4, opt};
        if (vkCreateShaderModule(V.dev, &mi, NULL, &m) == VK_SUCCESS && m) {
            stages[frag].module = m; VkGraphicsPipelineCreateInfo copy = *ci; copy.pStages = stages;
            pipeline_clone(&copy, 2*id+1, id, vs, fs->key, "optimized-clone"); vkDestroyShaderModule(V.dev, m, NULL);
        }
    } else if (*cfg.opt) VKLOG("pipeline executable inspection: fs %016llx optimized clone skipped: reference unavailable or different runtime SPIR-V", (unsigned long long)fs->key);
    free(base); free(opt);
done:
    pthread_mutex_unlock(&mu);
}
