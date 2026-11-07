"""Run production pass-census helpers and pass-end/LRZ paths with host mocks."""
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
        raise AssertionError(f'missing production function {name}')
    start, pos, depth = match.start(), match.end(), 1
    while depth:
        depth += (source[pos] == '{') - (source[pos] == '}')
        pos += 1
    return source[start:pos] + '\n'


@unittest.skipUnless(os.name != 'nt' and shutil.which('clang'), 'POSIX host clang needed')
class PassCensus(unittest.TestCase):
    def compile_run(self, code):
        with tempfile.TemporaryDirectory(prefix='nx2a-pass-census-') as tmp:
            src, exe = Path(tmp) / 'check.c', Path(tmp) / 'check'
            src.write_text(code)
            result = subprocess.run([
                'clang', '-std=gnu11', '-O2', '-Wall', '-Wextra', '-Werror',
                '-Wno-missing-field-initializers', '-pthread',
                '-I'+str(DM/'android'), '-I'+str(DM/'gfx/nvn'), '-I'+str(DM/'gfx/nvn/include'),
                '-I'+str(DM/'third_party/vulkan-headers/include'),
                str(src), '-o', str(exe)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run([str(exe)], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_actual_end_barrier_rt_live_reporting_and_concurrent_snapshot(self):
        source = (DM/'android/gfx_vk_exec.c').read_text()
        implementations = '\n'.join(function(source, name) for name in (
            'pass_census_refresh', 'pass_census_end', 'pass_census_rt',
            'pass_census_lrz', 'pass_census_append', 'vk_pass_census_report',
            'end_pass_reason', 'vk_end_pass', 'vk_end_pass_land_reason', 'vk_end_pass_land', 'game_barrier'))
        # Execute the actual live setting assignment used by x_read().
        setting = re.search(r'(?m)^    g_x_skip_empty_barrier = [^\n]+', source).group().strip()
        implementations += '\nstatic void tested_switch(void) {const char *e; ' + setting + '}\n'
        self.assertRegex(function(source, 'exec_ops'), r'case OP_BARRIER:\s*game_barrier_adjacent\(')
        # Execute the real RT decision before unrelated attachment-state copying.
        rt = function(source, 'set_rt')
        rt_prefix = rt[rt.index('{')+1:rt.index('    S.rt_n =')]
        implementations += '\nstatic void tested_set_rt(const p_set_rt *r) {' + rt_prefix + '\n}\n'
        self.compile_run(r'''
#include <assert.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pass_census.h"
#include "gpu_segments.h"
static inline __attribute__((unused)) void texture_barrier_record(uint32_t op,const void *data){(void)op;(void)data;}
static inline __attribute__((unused)) int texture_barrier_try(uint32_t mask){(void)mask;return 0;}
static inline __attribute__((unused)) int texture_barrier_end(enum vk_pass_end_reason reason,uint32_t mask){(void)reason;(void)mask;return 0;}

static uint32_t g_pass_segment __attribute__((unused));
static inline __attribute__((unused)) uint32_t vk_gpu_segment_begin(uint32_t kind,uint64_t tag,uint32_t w,uint32_t h){(void)kind;(void)tag;(void)w;(void)h;return 0;}
static inline __attribute__((unused)) void vk_gpu_segment_end(uint32_t token,uint32_t count){(void)token;(void)count;}

#define VK_NULL_HANDLE 0
static pthread_mutex_t g_pc_mu = PTHREAD_MUTEX_INITIALIZER;
static vk_pc_state g_pc; static int g_pc_on;
static struct {int pass_open,rp_pass;uint32_t pass_w,pass_h,cfmt[8];uintptr_t bound_pipe;} S;
static int P_cleared,g_x_listpasses,g_x_tracepass=-1,g_sub_pass,g_lin_back=1,g_tf_in_pass,g_x_skip_empty_barrier=1;
static uint64_t g_pass_draws;static int32_t g_pass_bb[4];
static int commit_ok=1,commit_calls,rp_ends,dyn_ends,timestamps,dumps,backs,crumbs,land_calls,sub_open=1;
static int g_x_nokeep,rt_matches=1,rt_checks;static struct {uint64_t rt_kept;} g_vkst;
#define MAGIC(a,b,c,d) ((uint32_t)(a)|((uint32_t)(b)<<8)|((uint32_t)(c)<<16)|((uint32_t)(d)<<24))
typedef struct {uint32_t magic;uint64_t serial;void *mem;} tex_t;
typedef struct {int dummy,n;const tex_t *color[8],*depth;} p_set_rt;
static int rt_same(const p_set_rt *r){assert(r);rt_checks++;return rt_matches;}
typedef struct {void *dli_fbase;} Dl_info;
static int dladdr(void *symbol,Dl_info *info){(void)symbol;info->dli_fbase=NULL;return 0;}
static int log_calls;static char last_log[8192];
static void log_line(const char *format,...){
    // Reporting must not hold the counter mutex during logging.
    assert(!pthread_mutex_trylock(&g_pc_mu));pthread_mutex_unlock(&g_pc_mu);
    va_list ap;va_start(ap,format);int n=vsnprintf(last_log,sizeof last_log,format,ap);va_end(ap);
    assert(n>=0 && (size_t)n<sizeof last_log);log_calls++;
}
#define VKLOG log_line
static int pass_commit(void){commit_calls++;if(commit_ok){S.pass_open=1;g_pass_draws=0;}return commit_ok;}
static int vk_cmd(void){return 7;}
static int full_barriers;static void vk_img_barrier_all(int cb){assert(cb==7);full_barriers++;}
static void vkCmdEndRenderPass(int cb){assert(cb==7);rp_ends++;}
static void vkCmdEndRendering(int cb){assert(cb==7);dyn_ends++;}
void vk_pass_ts(int end,uint32_t w,uint32_t h,uint32_t fmt,uint32_t draws){
    assert(end==1 && w==640 && h==480 && fmt==37);(void)draws;timestamps++;
}
static void pass_dump(void){dumps++;}
static void lin_back(void){backs++;}
static void crumb_mark(int kind){assert(!kind);crumbs++;}
static int vk_sub_open(void){return sub_open;}
static void vk_ubd_land(int cb){assert(cb==7);land_calls++;}
void vk_end_pass(void);
'''+implementations+r'''
static void active(int open,int clear,uint64_t draws,int rp){
    S.pass_open=open;S.rp_pass=rp;S.pass_w=640;S.pass_h=480;S.cfmt[0]=37;S.bound_pipe=123;
    P_cleared=clear;g_pass_draws=draws;g_tf_in_pass=1;
}
static void retired(void){assert(!S.pass_open && !S.bound_pipe && !g_tf_in_pass);}
static void *updates(void *unused){(void)unused;for(int i=0;i<20000;i++)pass_census_end(VK_PASS_END_BARRIER,0x10,1,0,1);return NULL;}
int main(void){
    unsetenv("DM_VK_PASS_CENSUS");pass_census_refresh(0);
    active(1,0,1,1);vk_end_pass();retired();assert(rp_ends==1 && !g_pc.counts.end_calls[0]);
    vk_pass_census_report(600);assert(!log_calls && !g_pc_on);
    setenv("DM_VK_PASS_CENSUS","1",1);pass_census_refresh(100);assert(g_pc.enabled && g_pc.since_frame==100);
    active(0,0,0,0);end_pass_reason(VK_PASS_END_BARRIER,0);retired();
    active(1,0,0,0);end_pass_reason(VK_PASS_END_BARRIER,0);retired();
    active(1,0,1,1);end_pass_reason(VK_PASS_END_BARRIER,0x10);retired();
    active(1,0,9,0);end_pass_reason(VK_PASS_END_BARRIER,0x10);retired();
    active(2,1,99,1);end_pass_reason(VK_PASS_END_BARRIER,0x10);retired();
    assert(commit_calls==1 && g_pc.counts.clear_only[VK_PASS_END_BARRIER]==1);
    commit_ok=0;active(2,1,99,0);end_pass_reason(VK_PASS_END_BARRIER,0x100);retired();
    active(2,0,99,0);end_pass_reason(VK_PASS_END_BARRIER,0x10);retired();
    assert(commit_calls==2 && g_pc.counts.barriers[VK_PC_MASK_OTHER][0]==1);
    assert(!g_pc.counts.barriers[VK_PC_MASK_OTHER][1] && !g_pc.counts.barriers[VK_PC_MASK_OTHER][2]);
    assert(g_pc.counts.barriers[0][0]==2 && g_pc.counts.barriers[0][1]==1);
    assert(g_pc.counts.barriers[0x10][0]==4 && g_pc.counts.barriers[0x10][1]==2 && g_pc.counts.barriers[0x10][2]==1);
    assert(g_pc.counts.ends[VK_PASS_END_BARRIER][0]==2 && g_pc.counts.ends[VK_PASS_END_BARRIER][1]==1 && g_pc.counts.ends[VK_PASS_END_BARRIER][2]==1);
    active(1,1,0,1);end_pass_reason(VK_PASS_END_SET_RT,0);retired(); // active draw0 is not clear-only
    assert(!g_pc.counts.clear_only[VK_PASS_END_SET_RT]);
    assert(rp_ends==4 && dyn_ends==2 && timestamps==6 && dumps==6 && crumbs==6 && backs==3);
    active(0,0,0,0);vk_end_pass_land_reason(VK_PASS_END_BUFFER_COPY);assert(land_calls==1);
    sub_open=0;vk_end_pass_land();assert(land_calls==1);sub_open=1;
    p_set_rt r={0};
    active(0,0,0,0);tested_set_rt(&r);assert(!rt_checks && g_pc.counts.rt[VK_PC_RT_OPEN0]==1);
    active(1,0,1,1);tested_set_rt(&r);assert(S.pass_open==1 && rt_checks==1 && g_vkst.rt_kept==1 && g_pc.counts.rt[VK_PC_RT_SAME]==1);
    rt_matches=0;active(2,0,0,0);tested_set_rt(&r);assert(!S.pass_open && rt_checks==2 && g_pc.counts.rt[VK_PC_RT_MISMATCH]==1);
    g_x_nokeep=1;active(1,0,1,1);tested_set_rt(&r);assert(!S.pass_open && rt_checks==2 && g_pc.counts.rt[VK_PC_RT_FORCED]==1);
    pass_census_lrz(VK_PC_LRZ_LOAD,0);pass_census_lrz(VK_PC_LRZ_SWITCH,1);pass_census_lrz(VK_PC_LRZ_CUT,1);
    assert(g_pc.counts.lrz[0]==1 && !g_pc.counts.lrz_writes[0] && g_pc.counts.lrz_writes[1]==1 && g_pc.counts.lrz_writes[2]==1);
    vk_pc_snapshot snap;assert(!vk_pc_take(&g_pc,0,&snap) && !vk_pc_take(&g_pc,599,&snap));
    vk_pass_census_report(599);assert(!log_calls);
    vk_pass_census_report(600);assert(log_calls==4 && g_pc.since_frame==600 && !g_pc.counts.end_calls[VK_PASS_END_BARRIER]);
    vk_pass_census_report(600);assert(log_calls==4); // one snapshot per checkpoint
    setenv("DM_VK_PASS_CENSUS","0",1);vk_pass_census_report(601);assert(!g_pc_on && !g_pc.enabled);
    pass_census_end(VK_PASS_END_BARRIER,0,1,0,1);assert(!g_pc.counts.end_calls[VK_PASS_END_BARRIER]);
    setenv("DM_VK_PASS_CENSUS","1",1);pass_census_refresh(1300);assert(g_pc.since_frame==1300);
    assert(!vk_pc_take(&g_pc,1200,&snap));
    // Exercise full-width counters and all names without unbounded logging.
    for(int i=0;i<VK_PASS_END_COUNT;i++){
        g_pc.counts.end_calls[i]=g_pc.counts.clear_only[i]=UINT64_MAX;
        for(int j=0;j<3;j++)g_pc.counts.ends[i][j]=UINT64_MAX;
    }
    for(int i=0;i<VK_PC_MASK_COUNT;i++)for(int j=0;j<3;j++)g_pc.counts.barriers[i][j]=UINT64_MAX;
    vk_pass_census_report(1800);assert(log_calls==8 && g_pc.since_frame==1800);
    struct {char left;char b[4];char right;} bounded={'L',"",'R'};size_t pos=0;
    pass_census_append(bounded.b,sizeof bounded.b,&pos,"%s","abcdef");
    pass_census_append(bounded.b,sizeof bounded.b,&pos,"more");
    assert(pos==3 && !strcmp(bounded.b,"abc") && bounded.left=='L' && bounded.right=='R');
    pos=0;pass_census_append(NULL,0,&pos,"unused");assert(!pos);
    // One atomic snapshot may split concurrent updates; their sum must be conserved.
    memset(&g_pc,0,sizeof g_pc);vk_pc_enable(&g_pc,1,0);
    pthread_t worker;assert(!pthread_create(&worker,NULL,updates,NULL));
    pthread_mutex_lock(&g_pc_mu);assert(vk_pc_take(&g_pc,600,&snap));pthread_mutex_unlock(&g_pc_mu);
    assert(!pthread_join(worker,NULL));vk_pc_snapshot rest;
    pthread_mutex_lock(&g_pc_mu);assert(vk_pc_take(&g_pc,1200,&rest));pthread_mutex_unlock(&g_pc_mu);
    assert(snap.counts.end_calls[VK_PASS_END_BARRIER]+rest.counts.end_calls[VK_PASS_END_BARRIER]==20000);
    assert(snap.counts.ends[VK_PASS_END_BARRIER][1]+rest.counts.ends[VK_PASS_END_BARRIER][1]==20000);
    assert(snap.counts.barriers[0x10][1]+rest.counts.barriers[0x10][1]==20000);
    vk_pc_end(&g_pc,(enum vk_pass_end_reason)-1,0,1,0,2);assert(g_pc.counts.ends[VK_PASS_END_EXTERNAL][2]==1);
    // Empty barriers preserve every pass state, including an uncommitted clear.
    memset(&g_pc,0,sizeof g_pc);vk_pc_enable(&g_pc,1,0);
    unsetenv("DM_VK_SKIP_EMPTY_BARRIER");tested_switch();assert(g_x_skip_empty_barrier==1);
    int old_full=full_barriers,old_ends=rp_ends+dyn_ends,old_commit=commit_calls,old_land=land_calls;
    commit_ok=1;
    for(int state=0;state<4;state++){
        active(state==3?2:state,state==3,99,1);unsigned char before[sizeof S];memcpy(before,&S,sizeof S);
        int was_clear=P_cleared;game_barrier(0);
        assert(!memcmp(before,&S,sizeof S) && P_cleared==was_clear && g_pass_draws==99 && g_tf_in_pass==1);
    }
    assert(full_barriers==old_full && rp_ends+dyn_ends==old_ends && commit_calls==old_commit && land_calls==old_land);
    assert(g_pc.counts.barriers[0][0]==4 && !g_pc.counts.barriers[0][1] && !g_pc.counts.barriers[0][2]);
    assert(g_pc.counts.end_calls[VK_PASS_END_BARRIER]==4);
    // A live control restores the complete legacy path for zero masks.
    setenv("DM_VK_SKIP_EMPTY_BARRIER","0",1);tested_switch();assert(!g_x_skip_empty_barrier);
    active(1,0,2,1);game_barrier(0);retired();
    assert(full_barriers==old_full+1 && rp_ends+dyn_ends==old_ends+1 && g_pc.counts.barriers[0][1]==1);
    active(2,1,99,0);game_barrier(0);retired();
    assert(commit_calls==old_commit+1 && full_barriers==old_full+2 && g_pc.counts.barriers[0][2]==1);
    active(2,0,99,0);game_barrier(0);retired();
    assert(commit_calls==old_commit+1 && full_barriers==old_full+3);
    active(0,0,0,0);game_barrier(0);retired();assert(full_barriers==old_full+4);
    // All eight known bits, combinations, unknown bits and signed masks keep
    // the old end+barrier path under both settings; no implicit mask truncation.
    const int32_t masks[]={1,2,4,8,16,32,64,128,3,0x12,0x100,-1};
    for(int mode=0;mode<=1;mode++){
        setenv("DM_VK_SKIP_EMPTY_BARRIER",mode?"1":"0",1);tested_switch();
        for(size_t i=0;i<sizeof masks/sizeof masks[0];i++){
            old_full=full_barriers;old_ends=rp_ends+dyn_ends;
            active(1,0,2,(int)(i&1));game_barrier(masks[i]);retired();
            assert(full_barriers==old_full+1 && rp_ends+dyn_ends==old_ends+1 && land_calls==old_land);
        }
        old_full=full_barriers;old_commit=commit_calls;
        active(2,1,99,1);game_barrier(0x10);retired();
        assert(full_barriers==old_full+1 && commit_calls==old_commit+1);
        old_full=full_barriers;old_ends=rp_ends+dyn_ends;
        active(0,0,0,0);game_barrier(0x10);retired();
        assert(full_barriers==old_full+1 && rp_ends+dyn_ends==old_ends);
    }
    old_full=full_barriers;active(1,0,2,1);game_barrier(0);assert(S.pass_open==1 && S.bound_pipe==123 && full_barriers==old_full);
    unsetenv("DM_VK_SKIP_EMPTY_BARRIER");tested_switch();assert(g_x_skip_empty_barrier==1);
    return 0;
}
''')

    def test_actual_lrz_guards_and_unchanged_transfers(self):
        source = (DM/'android/gfx_vk_exec.c').read_text()
        implementations = '\n'.join(function(source, name) for name in ('pass_census_lrz', 'lrz_touch'))
        self.compile_run(r'''
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan_core.h>
#include <assert.h>
#include <pthread.h>
#include <stdint.h>
#include "pass_census.h"
#include "gpu_segments.h"
static inline __attribute__((unused)) void texture_barrier_record(uint32_t op,const void *data){(void)op;(void)data;}
static inline __attribute__((unused)) int texture_barrier_try(uint32_t mask){(void)mask;return 0;}
static inline __attribute__((unused)) int texture_barrier_end(enum vk_pass_end_reason reason,uint32_t mask){(void)reason;(void)mask;return 0;}

static uint32_t g_pass_segment __attribute__((unused));
static inline __attribute__((unused)) uint32_t vk_gpu_segment_begin(uint32_t kind,uint64_t tag,uint32_t w,uint32_t h){(void)kind;(void)tag;(void)w;(void)h;return 0;}
static inline __attribute__((unused)) void vk_gpu_segment_end(uint32_t token,uint32_t count){(void)token;(void)count;}

static pthread_mutex_t g_pc_mu = PTHREAD_MUTEX_INITIALIZER;
static vk_pc_state g_pc;static int g_pc_on=1;
static struct {VkDevice dev;} V;
typedef struct {VkImage img;VkImageAspectFlags aspect;uint32_t samples;} vkimg_t;
static int creates,allocs,reads,writes,barriers;
#define VKCHK(call) assert((call)==VK_SUCCESS)
static uint32_t vk_mem_type(uint32_t bits,VkMemoryPropertyFlags need,VkMemoryPropertyFlags prefer){assert(bits==1 && need==VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT && !prefer);return 0;}
static VkCommandBuffer vk_cmd(void){return (VkCommandBuffer)(uintptr_t)7;}
static VkResult vkCreateBuffer(VkDevice dev,const VkBufferCreateInfo *info,const VkAllocationCallbacks *callbacks,VkBuffer *out){
    assert(dev==V.dev && !callbacks && info->size==256 && info->usage==(VK_BUFFER_USAGE_TRANSFER_DST_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT));
    creates++;*out=(VkBuffer)(uintptr_t)1;return VK_SUCCESS;
}
static void vkGetBufferMemoryRequirements(VkDevice dev,VkBuffer b,VkMemoryRequirements *out){assert(dev==V.dev && b);*out=(VkMemoryRequirements){256,1,1};}
static VkResult vkAllocateMemory(VkDevice dev,const VkMemoryAllocateInfo *info,const VkAllocationCallbacks *callbacks,VkDeviceMemory *out){
    assert(dev==V.dev && !callbacks && info->allocationSize==256 && !info->memoryTypeIndex);allocs++;*out=(VkDeviceMemory)(uintptr_t)1;return VK_SUCCESS;
}
static VkResult vkBindBufferMemory(VkDevice dev,VkBuffer buffer,VkDeviceMemory memory,VkDeviceSize offset){assert(dev==V.dev && buffer && memory && !offset);return VK_SUCCESS;}
static void region(VkCommandBuffer cb,VkImage img,VkImageLayout layout,VkBuffer buffer,uint32_t count,const VkBufferImageCopy *copy){
    assert(cb==vk_cmd() && img==(VkImage)(uintptr_t)9 && layout==VK_IMAGE_LAYOUT_GENERAL && buffer && count==1);
    assert(!copy->bufferOffset && copy->imageSubresource.aspectMask==VK_IMAGE_ASPECT_DEPTH_BIT && copy->imageExtent.width==1 && copy->imageExtent.height==1 && copy->imageExtent.depth==1);
}
static void vkCmdCopyImageToBuffer(VkCommandBuffer cb,VkImage img,VkImageLayout layout,VkBuffer buffer,uint32_t count,const VkBufferImageCopy *copy){region(cb,img,layout,buffer,count,copy);reads++;}
static void vkCmdCopyBufferToImage(VkCommandBuffer cb,VkBuffer buffer,VkImage img,VkImageLayout layout,uint32_t count,const VkBufferImageCopy *copy){region(cb,img,layout,buffer,count,copy);writes++;}
static void vk_img_barrier_all(VkCommandBuffer cb){assert(cb==vk_cmd());barriers++;}
'''+implementations+r'''
int main(void){
    vk_pc_enable(&g_pc,1,0);vkimg_t image={(VkImage)(uintptr_t)9,VK_IMAGE_ASPECT_COLOR_BIT,1};
    lrz_touch(NULL,1,VK_PC_LRZ_SWITCH);lrz_touch(&image,1,VK_PC_LRZ_SWITCH);
    image.aspect=VK_IMAGE_ASPECT_DEPTH_BIT;image.samples=4;lrz_touch(&image,1,VK_PC_LRZ_SWITCH);
    assert(!creates && !allocs && !reads && !writes && !barriers && !g_pc.counts.lrz[VK_PC_LRZ_SWITCH]);
    image.samples=1;lrz_touch(&image,0,VK_PC_LRZ_LOAD);
    assert(creates==1 && allocs==1 && reads==1 && !writes && barriers==1);
    assert(g_pc.counts.lrz[VK_PC_LRZ_LOAD]==1 && !g_pc.counts.lrz_writes[VK_PC_LRZ_LOAD]);
    lrz_touch(&image,1,VK_PC_LRZ_SWITCH);lrz_touch(&image,1,VK_PC_LRZ_CUT);
    assert(creates==1 && allocs==1 && reads==3 && writes==2 && barriers==5);
    assert(g_pc.counts.lrz[VK_PC_LRZ_SWITCH]==1 && g_pc.counts.lrz_writes[VK_PC_LRZ_SWITCH]==1);
    assert(g_pc.counts.lrz[VK_PC_LRZ_CUT]==1 && g_pc.counts.lrz_writes[VK_PC_LRZ_CUT]==1);
    g_pc_on=0;vk_pc_enable(&g_pc,0,1);lrz_touch(&image,0,VK_PC_LRZ_LOAD);
    assert(reads==4 && writes==2 && barriers==6 && !g_pc.counts.lrz[VK_PC_LRZ_LOAD]); // disabling counts changes no commands
    return 0;
}
''')

    def test_actual_walker_adjacent_barriers_and_all_boundaries(self):
        source = (DM/'android/gfx_vk_exec.c').read_text()
        state = source[source.index('static struct {'):source.index('} S;')+4]
        adjacent = re.search(r'typedef struct \{[^\n]+\} barrier_adjacent;', source).group()
        implementations = '\n'.join(function(source, name) for name in (
            'pass_census_end', 'pass_census_adjacent', 'end_pass_reason',
            'vk_end_pass', 'vk_end_pass_land_reason', 'vk_end_pass_land',
            'game_barrier', 'game_barrier_adjacent', 'exec_ops'))
        setting = re.search(r'(?m)^    g_x_dedup_adj_barrier = [^\n]+', source).group().strip()
        implementations += '\nstatic void tested_switch(void) {const char *e; ' + setting + '}\n'
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
#include "nvn_obj.h"
#include "pass_census.h"
#include "gpu_segments.h"
static inline __attribute__((unused)) void texture_barrier_record(uint32_t op,const void *data){(void)op;(void)data;}
static inline __attribute__((unused)) int texture_barrier_try(uint32_t mask){(void)mask;return 0;}
static inline __attribute__((unused)) int texture_barrier_end(enum vk_pass_end_reason reason,uint32_t mask){(void)reason;(void)mask;return 0;}

static uint32_t g_pass_segment __attribute__((unused));
static inline __attribute__((unused)) uint32_t vk_gpu_segment_begin(uint32_t kind,uint64_t tag,uint32_t w,uint32_t h){(void)kind;(void)tag;(void)w;(void)h;return 0;}
static inline __attribute__((unused)) void vk_gpu_segment_end(uint32_t token,uint32_t count){(void)token;(void)count;}

typedef struct vkshader vkshader_t;
'''+state+'\n'+adjacent+r'''
static pthread_mutex_t g_pc_mu=PTHREAD_MUTEX_INITIALIZER;
static vk_pc_state g_pc;static int g_pc_on=1,g_x_dedup_adj_barrier,g_x_skip_empty_barrier=1;
static int g_x_listpasses,g_x_tracepass=-1,g_sub_pass,g_lin_back,g_tf_in_pass,P_cleared;
static int g_rc_skip;static uint32_t g_sub_ops[64];static uint64_t g_ubo_gen[6],g_bind_gen[6];
static p_tf_bind g_tf[4];static struct {int has_xfb;} V={1};
static uint64_t g_pass_draws;static int32_t g_pass_bb[4];
static unsigned ends,fulls,commits,lands,cmd_queries;static uintptr_t cb_id=7;
static char events[1024];static size_t nevents;
static void note(char event){assert(nevents+1<sizeof events);events[nevents++]=event;events[nevents]=0;}
static VkCommandBuffer vk_cmd(void){cmd_queries++;return (VkCommandBuffer)cb_id;}
static void vk_img_barrier_all(VkCommandBuffer cb){assert(cb==vk_cmd());fulls++;note('B');}
static void vkCmdEndRenderPass(VkCommandBuffer cb){assert(cb==vk_cmd());ends++;note('E');}
static void vkCmdEndRendering(VkCommandBuffer cb){vkCmdEndRenderPass(cb);}
void vk_pass_ts(int end,uint32_t w,uint32_t h,uint32_t fmt,uint32_t draws){assert(end==1);(void)w;(void)h;(void)fmt;(void)draws;}
static int pass_commit(void){commits++;note('K');S.pass_open=1;g_pass_draws=0;return 1;}
static void pass_dump(void){}
static void lin_back(void){note('L');}
static void crumb_mark(int kind){assert(!kind);}
static int vk_sub_open(void){return 1;}
static void vk_ubd_land(VkCommandBuffer cb){assert(cb==vk_cmd());lands++;note('U');}
typedef struct {void *dli_fbase;} Dl_info;
static int dladdr(void *symbol,Dl_info *info){(void)symbol;info->dli_fbase=NULL;return 0;}
static void log_line(const char *format,...){(void)format;}
#define VKLOG log_line
#define VKRED_ONCE(id,...) do {(void)(id);} while(0)
#define DM_GAME_RENDER_COND 1
void vk_end_pass(void);
// Mock unrelated record operations. Their position and the real barrier/pass
// commands are recorded together, so the production walker determines order.
void mtl_sub_event(volatile uint32_t *v,uint32_t mode,uint32_t value){(void)v;(void)mode;(void)value;note('e');}
void rc_note(uint64_t a,int kind){(void)a;(void)kind;}
void mtl_sub_counter(uint64_t a,uint32_t kind){(void)a;(void)kind;note('q');}
uint64_t mtl_last_fence(void){return 11;}
void mtl_sub_copy(uint64_t s,uint64_t d,uint64_t n){assert(s==1 && d==2 && n==3);note('C');}
void mtl_sub_fill(uint64_t a,uint64_t n,uint32_t v){assert(a==2 && n==4 && v==3);note('F');}
void vk_sub_update(uint64_t a,const void *p,uint64_t n){assert(a==2 && p && n==4);note('u');}
void set_rt(const p_set_rt *p){assert(p);note('R');}
void clear_color(const p_clear_color *p){assert(p);note('c');S.pass_open=2;P_cleared=1;}
void clear_ds(const p_clear_ds *p){assert(p);note('z');S.pass_open=2;P_cleared=1;}
static int g_pool_tex_base,g_pool_samp_base;
static int nvn_dpool_base(const void *p,int t){(void)p;(void)t;return 0;}
static void bind_chmask(uint32_t mask){S.chmask.mask=mask;}
void bind_program(const p_bind_program *p){assert(p);note('p');}
void bind_handles_rec(uint32_t op,const void *p){(void)op;assert(p);note('h');}
void draw(const p_draw *p){assert(p);note('D');S.pass_open=1;g_pass_draws++;}
void draw_idx(const p_draw_idx *p){assert(p);note('I');S.pass_open=1;g_pass_draws++;}
void dispatch(const p_dispatch *p){assert(p);note('X');}
void copy_b2t(const p_copy_b2t *p,int to){assert(p);note(to?'a':'b');}
void copy_t2t(const p_copy_t2t *p){assert(p);note('t');}
void clear_tex(const p_clear_tex *p){assert(p);note('x');}
void render_cond(const p_render_cond *p){assert(p);note('r');}
void draw_tf(const p_draw_tf *p){assert(p);note('f');}
void tf_state(const p_tf_state *p){assert(p);note('s');}
'''+implementations+r'''
typedef struct {uint64_t words[512];size_t used;} record;
static void append(record *r,uint32_t op,const void *payload,size_t size){
    size_t n=(8+size+7)&~(size_t)7;assert(r->used+n<=sizeof r->words);
    uint8_t *p=(uint8_t *)r->words+r->used;memset(p,0,n);op_hdr h={op,(uint32_t)n};memcpy(p,&h,8);
    if(size)memcpy(p+8,payload,size);r->used+=n;
}
static void barrier(record *r,int32_t mask){append(r,OP_BARRIER,&mask,sizeof mask);}
static void finish(record *r){append(r,OP_END,NULL,0);}
static void run(record *r){exec_ops((const uint8_t *)r->words);}
static void reset(int enabled){
    memset(&g_pc,0,sizeof g_pc);vk_pc_enable(&g_pc,1,0);g_pc_on=1;
    memset(&S,0,sizeof S);S.pass_open=1;S.rp_pass=1;S.bound_pipe=(VkPipeline)(uintptr_t)123;
    g_pass_draws=2;g_tf_in_pass=1;P_cleared=0;ends=fulls=commits=lands=cmd_queries=0;nevents=0;events[0]=0;cb_id=7;
    g_x_dedup_adj_barrier=enabled;g_x_skip_empty_barrier=1;
}
static void between(record *r,uint32_t op){
    switch(op){
    case OP_DRAW:{p_draw x={.count=3,.instances=1};append(r,op,&x,sizeof x);break;}
    case OP_DRAW_IDX:{p_draw_idx x={.count=3,.instances=1};append(r,op,&x,sizeof x);break;}
    case OP_COPY_B2B:{uint64_t x[3]={1,2,3};append(r,op,x,sizeof x);break;}
    case OP_UPDATE_UB:{uint64_t x[3]={2,4,0};append(r,op,x,sizeof x);break;}
    case OP_CLEAR_BUF:{p_clear_buf x={.addr=2,.size=4,.value=3};append(r,op,&x,sizeof x);break;}
    case OP_CLEAR_COLOR:{p_clear_color x={.mask=15};append(r,op,&x,sizeof x);break;}
    case OP_CLEAR_DS:{p_clear_ds x={0};append(r,op,&x,sizeof x);break;}
    case OP_DISPATCH:{p_dispatch x={.x=1,.y=1,.z=1};append(r,op,&x,sizeof x);break;}
    case OP_SET_RT:{p_set_rt x={0};append(r,op,&x,sizeof x);break;}
    case OP_COPY_B2T:case OP_COPY_T2B:{p_copy_b2t x={0};append(r,op,&x,sizeof x);break;}
    case OP_COPY_T2T:{p_copy_t2t x={0};append(r,op,&x,sizeof x);break;}
    case OP_CLEAR_TEX:{p_clear_tex x={0};append(r,op,&x,sizeof x);break;}
    default:append(r,op,NULL,0);break;
    }
}
int main(void){
    unsetenv("DM_VK_SKIP_OPS");unsetenv("DM_VK_DEDUP_ADJ_BARRIER");tested_switch();assert(!g_x_dedup_adj_barrier);
    const int32_t masks[]={1,2,4,8,16,32,64,128,3,0x12,0x100,-1};
    for(size_t i=0;i<sizeof masks/sizeof masks[0];i++)for(int mode=0;mode<=1;mode++){
        record r={0};barrier(&r,0x10);barrier(&r,masks[i]);finish(&r);reset(mode);run(&r);
        assert(fulls==(mode?1u:2u) && ends==1 && !strcmp(events,mode?"EB":"EBB"));
        assert(g_pc.counts.adjacent_barriers==1 && g_pc.counts.adjacent_skipped==(uint64_t)mode);
        assert(g_pc.counts.end_calls[VK_PASS_END_BARRIER]==2 && !S.pass_open && !S.bound_pipe && !g_tf_in_pass && !lands);
        unsigned bucket=(uint32_t)masks[i]<VK_PC_MASK_OTHER?(uint32_t)masks[i]:VK_PC_MASK_OTHER;
        assert(g_pc.counts.barriers[bucket][0]==(masks[i]==0x10?2u:1u));
    }
    // A chain retains its first full scope and can omit only subsequent adjacent duplicates.
    record chain={0};barrier(&chain,1);barrier(&chain,2);barrier(&chain,4);finish(&chain);reset(1);run(&chain);
    assert(fulls==1 && ends==1 && g_pc.counts.adjacent_barriers==2 && g_pc.counts.adjacent_skipped==2);
    vk_pc_snapshot snap;assert(vk_pc_take(&g_pc,600,&snap));assert(snap.counts.adjacent_barriers==2 && snap.counts.adjacent_skipped==2);
    assert(!g_pc.counts.adjacent_barriers && !g_pc.counts.adjacent_skipped);
    // Every other opcode, even state-only/no-op/unknown ones, breaks adjacency.
    const uint32_t separators[]={OP_DRAW,OP_DRAW_IDX,OP_COPY_B2B,OP_UPDATE_UB,OP_CLEAR_BUF,OP_CLEAR_COLOR,OP_CLEAR_DS,OP_DISPATCH,OP_SET_RT,OP_COPY_B2T,OP_COPY_T2B,OP_COPY_T2T,OP_CLEAR_TEX,OP_BEGIN,OP_SET_TEX_POOL,OP_SET_SAMP_POOL,0x7fffffff};
    for(size_t i=0;i<sizeof separators/sizeof separators[0];i++){
        record r={0};barrier(&r,0x10);between(&r,separators[i]);barrier(&r,0x12);finish(&r);
        reset(0);run(&r);char original[sizeof events];strcpy(original,events);unsigned old_ends=ends,old_commits=commits;
        reset(1);run(&r);assert(fulls==2 && ends==old_ends && commits==old_commits && !strcmp(events,original));
        assert(!g_pc.counts.adjacent_barriers && !g_pc.counts.adjacent_skipped);
    }
    // The existing empty-barrier no-op must not even request a command buffer.
    record empty_only={0};barrier(&empty_only,0);finish(&empty_only);reset(1);run(&empty_only);
    assert(!cmd_queries && !fulls && !ends && S.pass_open==1 && S.bound_pipe==(VkPipeline)(uintptr_t)123);
    // Zero always breaks the chain, whether its separate no-op mode is on or off.
    for(int empty=0;empty<=1;empty++){
        record r={0};barrier(&r,1);barrier(&r,0);barrier(&r,2);finish(&r);reset(1);g_x_skip_empty_barrier=empty;run(&r);
        assert(fulls==(empty?2u:3u) && !g_pc.counts.adjacent_barriers && !g_pc.counts.adjacent_skipped);
        assert(g_pc.counts.barriers[0][0]==1 && !g_pc.counts.barriers[0][1] && !g_pc.counts.barriers[0][2]);
    }
    record child={0};barrier(&child,2);finish(&child);
    for(int mode=0;mode<=1;mode++){
        record r={0};barrier(&r,1);uint64_t target=(uint64_t)(uintptr_t)child.words;append(&r,OP_CALL,&target,8);barrier(&r,4);finish(&r);
        reset(mode);run(&r);assert(fulls==3 && ends==1 && !strcmp(events,"EBBB") && !g_pc.counts.adjacent_barriers);
        record jump={0};barrier(&jump,1);append(&jump,OP_JUMP,&target,8);finish(&jump);
        reset(mode);run(&jump);assert(fulls==2 && ends==1 && !strcmp(events,"EBB") && !g_pc.counts.adjacent_barriers);
    }
    // Adjacency inside a nested record is independent of both sides of CALL.
    record child_pair={0};barrier(&child_pair,2);barrier(&child_pair,4);finish(&child_pair);
    record caller={0};barrier(&caller,1);uint64_t target=(uint64_t)(uintptr_t)child_pair.words;append(&caller,OP_CALL,&target,8);barrier(&caller,8);finish(&caller);
    reset(1);run(&caller);assert(fulls==3 && g_pc.counts.adjacent_barriers==1 && g_pc.counts.adjacent_skipped==1);
    record one={0};barrier(&one,1);finish(&one);reset(1);run(&one);run(&one);
    assert(fulls==2 && !g_pc.counts.adjacent_barriers); // separate exec_ops invocations are record boundaries
    // A different CB, or an unexpectedly open pass, must retain a real barrier.
    reset(1);barrier_adjacent adjacent={0};game_barrier_adjacent(1,&adjacent);cb_id=8;game_barrier_adjacent(2,&adjacent);
    assert(fulls==2 && !g_pc.counts.adjacent_barriers);S.pass_open=1;game_barrier_adjacent(4,&adjacent);
    assert(fulls==3 && ends==2 && !g_pc.counts.adjacent_skipped);
    reset(1);cb_id=0;adjacent=(barrier_adjacent){0};game_barrier_adjacent(1,&adjacent);game_barrier_adjacent(2,&adjacent);
    assert(fulls==2 && !g_pc.counts.adjacent_barriers && !g_pc.counts.adjacent_skipped); // no claimed coverage from a missing CB
    // Rendering never depends on census; default-off still observes candidates.
    reset(0);run(&chain);assert(fulls==3 && g_pc.counts.adjacent_barriers==2 && !g_pc.counts.adjacent_skipped);
    reset(1);g_pc_on=0;vk_pc_enable(&g_pc,0,0);run(&chain);assert(fulls==1 && !g_pc.counts.adjacent_barriers && !g_pc.counts.adjacent_skipped);
    setenv("DM_VK_DEDUP_ADJ_BARRIER","1",1);tested_switch();assert(g_x_dedup_adj_barrier==1);
    setenv("DM_VK_DEDUP_ADJ_BARRIER","0",1);tested_switch();assert(!g_x_dedup_adj_barrier);
    unsetenv("DM_VK_DEDUP_ADJ_BARRIER");tested_switch();assert(!g_x_dedup_adj_barrier);
    return 0;
}
''')
