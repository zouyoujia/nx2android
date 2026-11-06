"""Execute production compute recipe serialization, persistence and warm/dispatch paths."""
from pathlib import Path
import os
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
DM = ROOT / 'nxnative/dm'


def function(source, name):
    match = re.search(r'(?m)^[^\n;{}]*\b' + re.escape(name) + r'\s*\([^;{}]*\)\s*\{', source)
    if not match:
        raise AssertionError('missing production function '+name)
    start, pos, depth = match.start(), match.end(), 1
    while depth:
        depth += (source[pos] == '{') - (source[pos] == '}')
        pos += 1
    return source[start:pos]+'\n'


@unittest.skipUnless(os.name != 'nt' and shutil.which('clang'), 'POSIX host clang needed')
class ComputeWarmKeys(unittest.TestCase):
    def compile_run(self, code, modes=('run',)):
        with tempfile.TemporaryDirectory(prefix='nx2a-compute-seen-') as tmp:
            src, exe = Path(tmp)/'check.c', Path(tmp)/'check'
            src.write_text(code)
            result = subprocess.run(['clang', '-std=gnu11', '-O2', '-Wall', '-Wextra', '-Werror',
                                     '-Wno-missing-field-initializers', '-pthread', '-I'+str(DM/'android'),
                                     '-I'+str(DM/'third_party/vulkan-headers/include'), str(src), '-o', str(exe)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            for mode in modes:
                folder = Path(tmp)/mode
                folder.mkdir()
                result = subprocess.run([str(exe), mode, str(folder)], capture_output=True, text=True, timeout=15)
                self.assertEqual(result.returncode, 0, result.stdout+result.stderr)

    def test_real_serialization_bounds_and_bad_files(self):
        self.compile_run(r'''
#include <assert.h>
#include <stdint.h>
#include <string.h>
#include "compute_warm_keys.h"
int main(void){
    vk_compute_keys keys={0},out={0};uint8_t buf[VK_COMPUTE_KEYS_BYTES+1];
    assert(!vk_compute_keys_add(&keys,(vk_compute_key){0,1,0}));
    assert(!vk_compute_keys_add(&keys,(vk_compute_key){1,0,0}));
    assert(!vk_compute_keys_add(&keys,(vk_compute_key){1,1,1}));
    assert(vk_compute_keys_add(&keys,(vk_compute_key){0x7f75318a892b4758ull,0x700,0}));
    assert(!vk_compute_keys_add(&keys,keys.keys[0]) && keys.count==1 && keys.generation==1);
    size_t n=vk_compute_keys_encode(&keys,buf,sizeof buf);assert(n==40 && !memcmp(buf,"CMW1",4));
    assert(buf[4]==1 && buf[8]==24 && buf[12]==1 && buf[16]==0x58 && buf[23]==0x7f && buf[24]==0 && buf[25]==7);
    assert(!vk_compute_keys_encode(&keys,buf,39));
    assert(vk_compute_keys_decode(&out,buf,n) && out.count==1 && !memcmp(&out.keys[0],&keys.keys[0],sizeof(vk_compute_key)));
    for(size_t cut=0;cut<n;cut++){vk_compute_keys prior=out;assert(!vk_compute_keys_decode(&out,buf,cut) && !memcmp(&out,&prior,sizeof out));}
    assert(!vk_compute_keys_decode(&out,buf,n+1));
    for(unsigned off=0;off<16;off+=4){uint8_t old=buf[off];buf[off]^=0x80;assert(!vk_compute_keys_decode(&out,buf,n));buf[off]=old;}
    vk_compute_keys_put_le(buf+12,257,4);assert(!vk_compute_keys_decode(&out,buf,n));vk_compute_keys_put_le(buf+12,1,4);
    // Invalid tuples cannot ask for a variant; a duplicate row is counted only once.
    vk_compute_keys_put_le(buf+32,0x100,8);assert(vk_compute_keys_decode(&out,buf,n) && !out.count);
    vk_compute_keys_put_le(buf+32,0,8);memcpy(buf+40,buf+16,24);vk_compute_keys_put_le(buf+12,2,4);
    assert(vk_compute_keys_decode(&out,buf,64) && out.count==1);
    memset(&keys,0,sizeof keys);
    for(unsigned i=0;i<VK_COMPUTE_KEYS_MAX;i++)assert(vk_compute_keys_add(&keys,(vk_compute_key){i+1,1,0}));
    assert(keys.count==256 && !vk_compute_keys_add(&keys,(vk_compute_key){999,1,0}));
    n=vk_compute_keys_encode(&keys,buf,sizeof buf);assert(n==VK_COMPUTE_KEYS_BYTES && vk_compute_keys_decode(&out,buf,n) && out.count==256);
    keys.count=257;assert(!vk_compute_keys_add(&keys,(vk_compute_key){999,1,0}) && !vk_compute_keys_encode(&keys,buf,sizeof buf));
    return 0;
}
''')

    def test_actual_record_load_save_retry_and_serial_io(self):
        source = (DM/'android/gfx_vk_exec.c').read_text()
        methods = '\n'.join(function(source, x) for x in ('compute_seen_path', 'compute_seen_load', 'compute_seen_record', 'vk_compute_warm_save'))
        state = source[source.index('static vk_compute_keys g_compute_seen;'):source.index('static int compute_seen_path(')]
        self.compile_run(r'''
#include <assert.h>
#include <errno.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "compute_warm_keys.h"
'''+state+r'''
static unsigned opens,renames;static int fail_rename;
static FILE *counted_fopen(const char *p,const char *mode){opens++;return fopen(p,mode);}
static int counted_rename(const char *a,const char *b){renames++;if(fail_rename){errno=EIO;return -1;}return rename(a,b);}
static void log_line(const char *format,...){(void)format;}
#define VKLOG log_line
int vk_shader_stage(uint64_t key){return key==11 || key==12?5:-1;}
#define fopen counted_fopen
#define rename counted_rename
'''+methods+r'''
#undef fopen
#undef rename
static void put_file(const char *path,const void *bytes,size_t n){FILE *f=fopen(path,"wb");assert(f && fwrite(bytes,1,n,f)==n && !fclose(f));}
static vk_compute_keys read_file(const char *path){uint8_t b[VK_COMPUTE_KEYS_BYTES+1];FILE *f=fopen(path,"rb");assert(f);size_t n=fread(b,1,sizeof b,f);assert(!fclose(f));vk_compute_keys s={0};assert(vk_compute_keys_decode(&s,b,n));return s;}
static void *save_thread(void *unused){(void)unused;vk_compute_warm_save();return NULL;}
int main(int argc,char **argv){
    assert(argc==3);setenv("DM_ROOT",argv[2],1);char path[1024];assert(compute_seen_path(path,sizeof path));
    assert(!compute_seen_path(path,2));assert(compute_seen_path(path,sizeof path));
    if(!strcmp(argv[1],"merge")){
        vk_compute_keys seed={0};vk_compute_keys_add(&seed,(vk_compute_key){11,1,0});vk_compute_keys_add(&seed,(vk_compute_key){99,1,0});
        uint8_t b[VK_COMPUTE_KEYS_BYTES];size_t n=vk_compute_keys_encode(&seed,b,sizeof b);put_file(path,b,n);
    }else put_file(path,"bad",3);
    compute_seen_record((vk_compute_key){12,2,0});compute_seen_record((vk_compute_key){12,2,0});
    assert(!opens && !renames && g_compute_seen.count==1); // recording never does filesystem IO
    pthread_once(&g_compute_seen_once,compute_seen_load);
    assert(g_compute_seen.count==(!strcmp(argv[1],"merge")?2u:1u)); // unknown shaders are not retained
    assert(!pthread_mutex_trylock(&g_compute_seen_mu));pthread_mutex_unlock(&g_compute_seen_mu);
    fail_rename=1;vk_compute_warm_save();assert(renames==1 && !g_compute_seen_saved);
    fail_rename=0;vk_compute_warm_save();assert(renames==2 && g_compute_seen_saved==g_compute_seen.generation);
    vk_compute_keys stored=read_file(path);assert(stored.count==g_compute_seen.count);
    unsigned before=opens;vk_compute_warm_save();assert(opens==before && renames==2);
    compute_seen_record((vk_compute_key){11,4,0});uint64_t generation=g_compute_seen.generation;
    pthread_t threads[8];for(int i=0;i<8;i++)assert(!pthread_create(&threads[i],NULL,save_thread,NULL));
    for(int i=0;i<8;i++)assert(!pthread_join(threads[i],NULL));
    assert(renames==3 && g_compute_seen_saved==generation && read_file(path).count==g_compute_seen.count);
    char tmp[1100];snprintf(tmp,sizeof tmp,"%s.tmp",path);assert(access(tmp,F_OK));
    assert(!pthread_mutex_trylock(&g_compute_seen_io));pthread_mutex_unlock(&g_compute_seen_io);
    return 0;
}
''', ('merge', 'corrupt'))

    def test_actual_warm_and_dispatch_keep_all_gpu_work(self):
        source = (DM/'android/gfx_vk_exec.c').read_text()
        names = ('compute_seen_record', 'compute_build', 'compute_seen_matches', 'compute_warm_seen', 'compute_warm', 'dispatch')
        methods = '\n'.join(function(source, x) for x in names)
        self.compile_run(r'''
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan_core.h>
#include <assert.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pass_census.h"
#include "gpu_segments.h"
static uint32_t g_pass_segment __attribute__((unused));
static inline __attribute__((unused)) uint32_t vk_gpu_segment_begin(uint32_t kind,uint64_t tag,uint32_t w,uint32_t h){(void)kind;(void)tag;(void)w;(void)h;return 0;}
static inline __attribute__((unused)) void vk_gpu_segment_end(uint32_t token,uint32_t count){(void)token;(void)count;}

#include "compute_warm_keys.h"
#define pthread_setname_np(...) ((void)0)
#define setpriority(...) ((void)0)
#define PT_DISPATCH 0xffffffffu
enum {RK_SBUF=4,RK_TEX=7,RK_TBUF=8,RK_IMG=9,RK_IMG_TBUF=10,RK_TEX_C2A=11,RK_TEX_UBO=12,RK_TBUF_UBO=13,RK_IMG_UBO=14};
typedef struct {uint8_t kind;uint16_t binding;} dmsv_bind;
typedef struct {uint64_t key;int stage,nbind,ndyn;const dmsv_bind *b;VkDescriptorSetLayout dsl;uint32_t local[3];} vkshader_t;
static dmsv_bind bindings[3]={{RK_TEX,0},{RK_TEX,1},{RK_SBUF,2}};
static vkshader_t shader={11,5,3,0,bindings,(VkDescriptorSetLayout)(uintptr_t)1,{1,1,1}};
static struct {VkDevice dev;VkPipelineCache pcache;} V;
static struct {vkshader_t *sh[6];struct {uint64_t addr,size;} ssbo[6][16];} S;
static struct {uint64_t dispatches;} g_vkst;
static int g_prof_on,g_x_nodispbar,g_x_tracepass=-1,g_sub_pass;static uint64_t g_prof_ns[8],g_dispatch[3],g_num_um,g_num_im;
static struct {char what[256];int phase;} g_build[1];
static pthread_mutex_t g_cpipe_mu=PTHREAD_MUTEX_INITIALIZER,g_compute_seen_mu=PTHREAD_MUTEX_INITIALIZER;
static pthread_once_t g_compute_seen_once=PTHREAD_ONCE_INIT;static vk_compute_keys g_compute_seen;
static unsigned load_calls,build_calls,destroy_calls,dispatch_calls,indirect_calls,slow_logs;
static int driver_fail,driver_null,race_publish,indirect_ok=1;static VkPipeline bound;
static uint64_t want_um=1,want_im,clock_ns,tick_step=60000000,last_um,last_im;
static uint64_t tick(void){uint64_t n=clock_ns;clock_ns+=tick_step;return n;}
static void compute_seen_load(void){load_calls++;}
static void log_line(const char *format,...){if(strstr(format,"compute pipeline build:"))slow_logs++;}
#define VKLOG log_line
#define LOG log_line
#define VKRED_ONCE(id,...) do {(void)(id);} while(0)
typedef struct {int unused;} fake_map;
static fake_map g_cpipes;
static struct {uint64_t key[3];VkPipeline pipeline;} entries[32];static unsigned nentry;
static VkPipeline nvn_hm_get(fake_map *m,const void *key){assert(m==&g_cpipes);for(unsigned i=0;i<nentry;i++)if(!memcmp(key,entries[i].key,24))return entries[i].pipeline;return VK_NULL_HANDLE;}
static void nvn_hm_put(fake_map *m,const void *key,VkPipeline p){assert(m==&g_cpipes && nentry<32);for(unsigned i=0;i<nentry;i++)if(!memcmp(key,entries[i].key,24)){entries[i].pipeline=p;return;}memcpy(entries[nentry].key,key,24);entries[nentry++].pipeline=p;}
int vk_shader_stage(uint64_t key){return key==11?5:key==22?0:-1;}
static vkshader_t *vk_shader(uint64_t key,int stage){assert(key==11 && stage==5);return &shader;}
int vk_shader_keys(uint32_t stage,uint64_t *out,int max){assert(stage==5 && max>=1);out[0]=11;return 1;}
static int build_begin(void){return -1;}
static void build_end(int slot){assert(slot==-1);}
static const void *vk_sg_pin(VkShaderStageFlagBits stage,const uint32_t *local){assert(stage==VK_SHADER_STAGE_COMPUTE_BIT && local==shader.local);return NULL;}
static VkShaderModule vk_shader_module(vkshader_t *cs,uint64_t um,uint64_t im,uint32_t xo,int fd){assert(cs==&shader && !xo && !fd);last_um=um;last_im=im;return (VkShaderModule)(uintptr_t)3;}
static VkPipelineLayout vk_pipeline_layout(VkDescriptorSetLayout *sets,int n){assert(n==1 && sets[0]==shader.dsl);return (VkPipelineLayout)(uintptr_t)4;}
static VkResult vkCreateComputePipelines(VkDevice dev,VkPipelineCache cache,uint32_t n,const VkComputePipelineCreateInfo *ci,const VkAllocationCallbacks *callbacks,VkPipeline *out){
    assert(dev==V.dev && cache==V.pcache && n==1 && !callbacks && ci->stage.stage==VK_SHADER_STAGE_COMPUTE_BIT);
    assert(!pthread_mutex_trylock(&g_cpipe_mu));pthread_mutex_unlock(&g_cpipe_mu); // never compile under the lookup lock
    build_calls++;if(driver_fail)return VK_ERROR_UNKNOWN;if(driver_null){*out=VK_NULL_HANDLE;return VK_SUCCESS;}
    if(race_publish){uint64_t key[3]={11,last_um,last_im};nvn_hm_put(&g_cpipes,key,(VkPipeline)(uintptr_t)77);race_publish=0;}
    *out=(VkPipeline)(uintptr_t)(100+build_calls);return VK_SUCCESS;
}
static void vkDestroyPipeline(VkDevice dev,VkPipeline p,const VkAllocationCallbacks *callbacks){assert(dev==V.dev && p && !callbacks);destroy_calls++;}
void vk_end_pass_land_reason(enum vk_pass_end_reason reason){assert(reason==VK_PASS_END_DISPATCH);}
static VkCommandBuffer vk_cmd(void){return (VkCommandBuffer)(uintptr_t)7;}
static VkDescriptorSet stage_set(int stage,vkshader_t *cs,uint32_t *offsets){assert(stage==5 && cs==&shader && offsets);g_num_um=want_um;g_num_im=want_im;return (VkDescriptorSet)(uintptr_t)5;}
static void vk_img_barrier_all(VkCommandBuffer cb){assert(cb==vk_cmd());}
static void vkCmdBindPipeline(VkCommandBuffer cb,VkPipelineBindPoint point,VkPipeline p){assert(cb==vk_cmd() && point==VK_PIPELINE_BIND_POINT_COMPUTE);bound=p;}
static void vkCmdBindDescriptorSets(VkCommandBuffer cb,VkPipelineBindPoint point,VkPipelineLayout layout,uint32_t first,uint32_t n,const VkDescriptorSet *sets,uint32_t ndyn,const uint32_t *offsets){assert(cb==vk_cmd() && point==VK_PIPELINE_BIND_POINT_COMPUTE && layout && !first && n==1 && sets && !ndyn && offsets);}
void vk_pass_ts(int end,uint32_t w,uint32_t h,uint32_t fmt,uint32_t draws){(void)end;(void)w;(void)h;(void)fmt;(void)draws;}
static int vk_buf_at(uint64_t a,uint64_t n,VkBuffer *buffer,VkDeviceSize *offset){assert(a==9 && n==12);*buffer=(VkBuffer)(uintptr_t)8;*offset=0;return indirect_ok;}
static void vkCmdDispatch(VkCommandBuffer cb,uint32_t x,uint32_t y,uint32_t z){assert(cb==vk_cmd() && x==1 && y==2 && z==3);dispatch_calls++;}
static void vkCmdDispatchIndirect(VkCommandBuffer cb,VkBuffer b,VkDeviceSize off){assert(cb==vk_cmd() && b && !off);indirect_calls++;}
typedef struct {int n;uint64_t k[8][3];} crumb_t;static crumb_t crumb;
static int crumbs_on(void){return 0;}
static crumb_t *crumb_cur(void){return &crumb;}
static void crumb_mark(int kind){assert(kind==1);}
typedef struct {int32_t x,y,z;uint64_t caller,ind;} p_dispatch;
'''+methods+r'''
static void reset(void){nentry=0;build_calls=destroy_calls=dispatch_calls=indirect_calls=slow_logs=0;memset(&g_compute_seen,0,sizeof g_compute_seen);driver_fail=driver_null=race_publish=0;want_um=1;want_im=0;bound=VK_NULL_HANDLE;S.sh[5]=&shader;}
int main(void){
    unsetenv("DM_VK_SPV_ALT");unsetenv("DM_VK_COMPUTE_WARM_SEEN");unsetenv("DM_VK_COMPUTE_WARM_MAX");
    reset();
    const uint8_t storage[]={RK_IMG,RK_IMG_UBO,RK_IMG_TBUF};
    for(unsigned i=0;i<sizeof storage;i++){bindings[2].kind=storage[i];assert(!compute_seen_matches(&shader,(vk_compute_key){11,4,0}));}
    bindings[2].kind=RK_SBUF;
    compute_seen_record((vk_compute_key){11,1,0});compute_seen_record((vk_compute_key){11,2,0});
    compute_warm(NULL);assert(build_calls==1 && !dispatch_calls && !indirect_calls && load_calls==1); // flag off: base only
    reset();compute_seen_record((vk_compute_key){11,1,0});compute_seen_record((vk_compute_key){11,2,0});
    compute_seen_record((vk_compute_key){11,4,0});compute_seen_record((vk_compute_key){22,1,0});compute_seen_record((vk_compute_key){33,1,0});
    setenv("DM_VK_COMPUTE_WARM_SEEN","1",1);compute_warm(NULL);
    assert(build_calls==3 && slow_logs==3 && !dispatch_calls && !indirect_calls); // base then exactly two valid observed variants
    p_dispatch d={1,2,3,0,0};dispatch(&d);assert(dispatch_calls==1 && build_calls==3); // real dispatch is never omitted
    dispatch(&d);assert(dispatch_calls==2 && build_calls==3); // same complete key hits
    unsigned before=build_calls;compute_warm_seen();assert(build_calls==before);
    reset();compute_seen_record((vk_compute_key){11,1,0});setenv("DM_VK_COMPUTE_WARM_MAX","0",1);compute_warm(NULL);
    assert(build_calls==1 && last_um==1); // base cap remains intact; opt-in seen recipes stay bounded separately
    reset();race_publish=1;compute_seen_record((vk_compute_key){11,1,0});compute_warm_seen();
    assert(build_calls==1 && destroy_calls==1);dispatch(&d);assert(dispatch_calls==1 && bound==(VkPipeline)(uintptr_t)77 && build_calls==1);
    // A background failure must leave no sentinel: a later required dispatch retries.
    for(int failure=0;failure<2;failure++){
        reset();compute_seen_record((vk_compute_key){11,1,0});driver_fail=failure==0;driver_null=failure==1;compute_warm_seen();
        uint64_t key[3]={11,1,0};assert(build_calls==1 && !nvn_hm_get(&g_cpipes,key) && !dispatch_calls);
        driver_fail=driver_null=0;dispatch(&d);assert(build_calls==2 && dispatch_calls==1 && bound && bound!=(VkPipeline)(uintptr_t)1);
    }
    reset();driver_fail=1;dispatch(&d);assert(build_calls==1 && !dispatch_calls && !g_compute_seen.count && slow_logs==1); // existing build-failure semantics
    unsigned loads_before=load_calls;reset();dispatch(&d);assert(dispatch_calls==1 && g_compute_seen.count==1 && load_calls==loads_before);
    uint64_t generation=g_compute_seen.generation;dispatch(&d);assert(dispatch_calls==2 && g_compute_seen.generation==generation);
    reset();d.ind=9;indirect_ok=0;dispatch(&d);assert(!indirect_calls && !g_compute_seen.count); // no record for a dispatch not emitted
    indirect_ok=1;dispatch(&d);assert(indirect_calls==1 && g_compute_seen.count==1);
    reset();d.ind=0;want_um=want_im=0;dispatch(&d);assert(dispatch_calls==1 && !g_compute_seen.count); // no baseline recipes
    reset();tick_step=1000000;dispatch(&d);assert(dispatch_calls==1 && !slow_logs); // fast creates do not flood logs
    return 0;
}
''')
