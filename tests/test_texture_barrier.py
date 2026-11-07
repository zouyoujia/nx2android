"Execute production conservative texture-barrier checks and actual pass-end paths."
from pathlib import Path
import os
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
DM=ROOT/'nxnative/dm'


def function(source,name):
    match=re.search(r'(?m)^[^\n;{}]*\b'+re.escape(name)+r'\s*\([^;{}]*\)\s*\{',source)
    if not match:raise AssertionError(f'missing production function {name}')
    start,pos,depth=match.start(),match.end(),1
    while depth:
        depth+=(source[pos]=='{')-(source[pos]=='}');pos+=1
    return source[start:pos]+'\n'


@unittest.skipUnless(os.name!='nt' and shutil.which('clang'),'POSIX host clang needed')
class TextureBarrier(unittest.TestCase):
    def compile_run(self,code):
        with tempfile.TemporaryDirectory(prefix='nx2a-texture-barrier-') as tmp:
            src,exe=Path(tmp)/'check.c',Path(tmp)/'check';src.write_text(code)
            run=subprocess.run(['clang','-std=gnu11','-O2','-Wall','-Wextra','-Werror',
                '-Wno-missing-field-initializers','-pthread',
                '-I'+str(DM/'android'),'-I'+str(DM/'gfx/nvn'),'-I'+str(DM/'gfx/nvn/include'),
                '-I'+str(DM/'third_party/vulkan-headers/include'),str(src),'-o',str(exe)],capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr)
            run=subprocess.run([str(exe)],capture_output=True,text=True,timeout=10)
            self.assertEqual(run.returncode,0,run.stdout+run.stderr)

    def test_full_storage_aliases_overflow_and_unknown_buffer_bounds(self):
        self.compile_run(r'''
#include <assert.h>
#include "texture_barrier.h"
int main(void){
    vk_tb_range a,b;assert(vk_tb_range_make(&a,0x10000,0x1000,1,0x2000));
    assert(vk_tb_range_make(&b,0x20000,0x1000,2,0x2000) && !vk_tb_overlap(&a,&b));
    b.pool=1;assert(vk_tb_overlap(&a,&b)); // same guest storage at different CPU addresses
    b.pool=2;b.lo=0x10800;b.hi=0x11800;assert(vk_tb_overlap(&a,&b)); // distinct pool handles over same memory
    b.lo=0x11000;b.hi=0x12000;assert(!vk_tb_overlap(&a,&b)); // touching endpoints
    assert(!vk_tb_range_make(&b,0,1,1,0));assert(!vk_tb_range_make(&b,1,0,1,0));
    assert(!vk_tb_range_make(&b,1,1,0,0));assert(!vk_tb_range_make(&b,1,1,1,-1));
    assert(!vk_tb_range_make(&b,UINT64_MAX-4,8,1,0));
    assert(!vk_tb_range_make(&b,1,UINT64_MAX-1,1,8));
    vk_tb_state s={.n=1,.pending=1};s.att[0]=a;
    assert(vk_tb_buffer_hit(&s,0x10800,16));assert(!vk_tb_buffer_hit(&s,0x11000,16));
    assert(vk_tb_buffer_hit(&s,UINT64_MAX,2));assert(vk_tb_buffer_hit(&s,0,16));assert(vk_tb_buffer_hit(&s,0x9000,0));
    b=a;assert(vk_tb_texture_hit(&s,&b)); // the full footprint includes all mip/layer views
    vk_tb_clear(&s);assert(!s.n && !s.pending && !s.draws);return 0;
}
''')

    def test_actual_pending_dependency_guards_every_draw_and_restores_sync_before_hazards(self):
        source=(DM/'android/gfx_vk_exec.c').read_text();header=(DM/'android/gfx_vk.h').read_text()
        block=source[source.index('static int g_tb_mode,'):source.index('// Before a pass that goes on with a depth buffer')]
        kinds=re.search(r'enum \{ RK_SUPPORT[^\n]+',header).group()
        desc=re.search(r'enum \{ RD_UBO[^\n]+',header).group()
        binding=re.search(r'typedef struct __attribute__\(\(packed\)\) \{[^\n]+\} dmsv_bind;',header).group()
        end=function(source,'end_pass_reason');barrier=function(source,'game_barrier')
        self.assertLess(function(source,'draw').index('texture_barrier_before_draw'),function(source,'draw').index('prepare_draw('))
        self.assertLess(function(source,'draw_idx').index('texture_barrier_before_draw'),function(source,'draw_idx').index('prepare_draw('))
        self.assertLess(function(source,'exec_ops').index('texture_barrier_record('),function(source,'exec_ops').index('switch (h.op)'))
        prepare=function(source,'prepare_draw')
        self.assertLess(prepare.index('stage_set(st,'),prepare.index('texture_barrier_recheck()'))
        self.assertLess(prepare.index('texture_barrier_recheck()'),prepare.index('g_vkst.draws++'))
        self.compile_run(r'''
#define _GNU_SOURCE
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan_core.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "nvn_obj.h"
#include "texture_barrier.h"
#include "pass_census.h"
'''+kinds+'\n'+desc+'\n'+binding+r'''
#define VKLOG(...) do {if(0)fprintf(stderr,__VA_ARGS__);} while(0)
typedef struct {int nbind,xfb;const dmsv_bind *b;uint32_t local[3];} vkshader_t;
typedef struct {uint64_t pool;int64_t off,size;} vkimg_t;
typedef struct {uint64_t lo,hi;VkBuffer buf;int flags,lazy;} vkpool_t;
static struct {int pass_open,rast_discard,rp_pass;VkPipeline bound_pipe;uint32_t pass_w,pass_h,cfmt[8];
    vkshader_t *sh[6];tex_t rt_col[8],rt_dep;p_bind_buf ubo[6][16];NVNbufferRange vbuf[16];uint32_t vattr[16];uint64_t tex[6][256];} S;
static struct {vkimg_t *im;} P_att[9];
static struct {struct {struct {uint64_t minUniformBufferOffsetAlignment;} limits;} props;} V;
static int g_tf_on,g_tf_in_pass,g_x_listpasses,g_x_skip_empty_barrier=1,g_x_tracepass=-1,g_sub_pass,g_lin_back,P_cleared;
static uint64_t g_pass_draws,g_set_epoch=1,g_tex_frame=1,g_tex_gen=1,g_pkg_n,dirty=1;static uint32_t pool_generation=1;
static int32_t g_pass_bb[4];
static uint64_t nvn_dirty_now(void){return dirty;}
static uint32_t vk_pool_gen(void){return pool_generation;}
static vkpool_t pools[3];static tex_t registry[8];static int ready[8],lookup_calls,ready_calls,barriers,ends,events[64],nevent;
static vkpool_t *vk_pool_of(uint64_t addr,uint64_t len){for(int i=0;i<3;i++)if(addr>=pools[i].lo && len<=pools[i].hi-addr && addr<pools[i].hi)return &pools[i];return NULL;}
static vkimg_t *vk_tex_by_id(int tid,const view_t **v,const tex_t **t){lookup_calls++;*v=NULL;if(tid<1 || tid>7 || !registry[tid].serial)return NULL;*t=&registry[tid];return (vkimg_t *)(uintptr_t)1;}
static int vk_tex_barrier_ready(const tex_t *t){ready_calls++;return ready[t-registry];}
static int vk_cmd(void){return 7;}
static void vk_img_barrier_all(int cb){assert(cb==7);barriers++;events[nevent++]=2;}
static void vkCmdEndRenderPass(int cb){assert(cb==7);ends++;events[nevent++]=1;}
static void vkCmdEndRendering(int cb){vkCmdEndRenderPass(cb);}
static uint32_t g_pass_segment;
static void vk_gpu_segment_end(uint32_t token,uint32_t count){(void)token;(void)count;}
void vk_pass_ts(int end,uint32_t w,uint32_t h,uint32_t fmt,uint32_t draws){(void)end;(void)w;(void)h;(void)fmt;(void)draws;}
static int pass_commit(void){S.pass_open=1;return 1;}
static void pass_census_end(enum vk_pass_end_reason reason,uint32_t mask,int ended,int clear,uint64_t draws){(void)reason;(void)mask;(void)ended;(void)clear;(void)draws;}
static void pass_dump(void){}
static void lin_back(void){}
static void crumb_mark(int n){(void)n;}
typedef struct {void *dli_fbase;} Dl_info;
static int dladdr(void *p,Dl_info *d){(void)p;(void)d;return 0;}
static void vk_end_pass(void){}
'''+block+'\n'+end+'\n'+barrier+r'''
static dmsv_bind vsbind[1]={{.kind=RK_CBUF,.desc=RD_UBO,.reg=0,.count=1}};
static dmsv_bind fsbind[1]={{.kind=RK_TEX,.desc=RD_SAMPLED,.reg=0,.count=1}};
static vkshader_t vs={.nbind=1,.b=vsbind},fs={.nbind=1,.b=fsbind};static vkimg_t attachment;
static tex_t texture(uint64_t mem,uint64_t pool,int64_t off){tex_t t={0};t.magic=MAGIC('T','E','X',0);t.serial=1;t.mem=(void *)(uintptr_t)mem;t.pool=(void *)(uintptr_t)pool;t.off=off;t.L.size=0x1000;return t;}
static void reset(int mode){
    memset(&S,0,sizeof S);memset(P_att,0,sizeof P_att);memset(g_tb_stats,0,sizeof g_tb_stats);vk_tb_clear(&g_tb);
    g_tb_mode=mode;g_tb_pass_unsafe=g_tb_this_unsafe=g_tf_on=g_tf_in_pass=0;g_pass_draws=2;g_set_epoch++;g_tex_gen++;dirty++;pool_generation++;
    pools[0]=(vkpool_t){0x100000,0x200000,(VkBuffer)(uintptr_t)1,0x85,0};
    pools[1]=(vkpool_t){0x300000,0x400000,(VkBuffer)(uintptr_t)2,0x85,0};
    pools[2]=(vkpool_t){0x500000,0x600000,(VkBuffer)(uintptr_t)3,0x85,0};
    S.rt_col[0]=texture(0x100000,11,0x2000);attachment=(vkimg_t){11,0x2000,0x1000};P_att[0].im=&attachment;
    registry[1]=texture(0x310000,12,0x10000);registry[2]=texture(0x100800,99,0);registry[3]=texture(0x320000,11,0x2800);
    registry[4]=texture(0x330000,12,0x30000);registry[5]=texture(0x340000,12,0x40000);registry[6]=texture(0x350000,12,0x50000);
    memset(ready,0,sizeof ready);for(int i=1;i<8;i++)ready[i]=1;
    S.sh[0]=&vs;S.sh[1]=&fs;S.ubo[0][0]=(p_bind_buf){0,0,0x360000,1024};S.tex[1][0]=1;
    vs.xfb=0;vs.local[0]=0;fsbind[0]=(dmsv_bind){.kind=RK_TEX,.desc=RD_SAMPLED,.reg=0,.count=1};
    S.pass_open=1;S.rp_pass=1;V.props.limits.minUniformBufferOffsetAlignment=256;
    barriers=ends=nevent=lookup_calls=ready_calls=0;
}
static void deferred(void){assert(texture_barrier_try(0x10));assert(g_tb.pending && !ends && !barriers);}
static void flushed(int reason){assert(!g_tb.pending && ends==1 && barriers==1 && events[0]==1 && events[1]==2 && g_tb_stats[reason]==1);}
int main(void){
    // Disabled mode retains complete legacy behavior and never resolves metadata.
    reset(0);game_barrier(0x10);assert(ends==1 && barriers==1 && !lookup_calls && !g_tb.pending);
    texture_barrier_before_draw(0,0,0);assert(!lookup_calls);
    // Observe mode changes no commands. It can see a hazard after several later
    // independent draws, rather than considering only the immediately next draw.
    reset(1);game_barrier(0x10);assert(ends==1 && barriers==1 && g_tb.pending && !g_tb_stats[TB_KEPT]);
    game_barrier(0x10);game_barrier(0x10);assert(ends==1 && barriers==3 && g_tb.pending && g_tb_stats[TB_INACTIVE]==2);
    texture_barrier_before_draw(0,0,0);texture_barrier_before_draw(0,0,0);assert(g_tb.pending && !g_tb_stats[TB_TEX_ALIAS]);
    S.tex[1][0]=2;texture_barrier_before_draw(0,0,0);assert(!g_tb.pending && g_tb_stats[TB_TEX_ALIAS]==1 && ends==1 && barriers==3);
    // Pending dependency survives repeated barriers, state records, calls/jumps
    // and descriptor memo hits. Only an actual dependent consumer flushes it.
    reset(2);deferred();texture_barrier_before_draw(0,0,0);int looked=lookup_calls;
    texture_barrier_record(OP_BIND_UBO,NULL);texture_barrier_record(OP_CALL,NULL);texture_barrier_record(OP_JUMP,NULL);
    texture_barrier_before_draw(0,0,0);assert(lookup_calls==looked && !ends && g_tb.pending);
    assert(texture_barrier_try(0x10));game_barrier(0);assert(g_tb.pending && !ends && !barriers);
    S.tex[1][0]=2;texture_barrier_before_draw(0,0,0);flushed(TB_TEX_ALIAS);
    reset(2);deferred();S.tex[1][0]=3;texture_barrier_before_draw(0,0,0);flushed(TB_TEX_ALIAS); // pool alias at another address
    // Storage declarations reject the pass, even without a reflection write bit.
    reset(2);g_tb_pass_unsafe=1;assert(!texture_barrier_try(0x10) && !g_tb.pending);
    reset(2);deferred();fsbind[0].kind=RK_IMG;fsbind[0].desc=RD_STORAGE;fsbind[0].flags=0;
    texture_barrier_before_draw(0,0,0);flushed(TB_WRITER);
    reset(2);deferred();S.rast_discard=1;texture_barrier_before_draw(0,0,0);flushed(TB_WRITER);
    reset(2);deferred();vs.xfb=1;texture_barrier_before_draw(0,0,0);flushed(TB_WRITER);
    reset(2);deferred();g_tf_on=1;texture_barrier_before_draw(0,0,0);flushed(TB_WRITER);
    reset(2);deferred();fsbind[0].kind=RK_TEX_UBO;texture_barrier_before_draw(0,0,0);flushed(TB_WRITER);
    reset(2);deferred();fsbind[0].count=0;texture_barrier_before_draw(0,0,0);flushed(TB_WRITER);
    // Missing/fresh/dirty/changed metadata falls back before resolution/uploads.
    reset(2);deferred();S.tex[1][0]=0;texture_barrier_before_draw(0,0,0);flushed(TB_UNKNOWN);
    reset(2);deferred();ready[1]=0;texture_barrier_before_draw(0,0,0);flushed(TB_UPLOAD);
    reset(2);deferred();g_tex_gen++;texture_barrier_before_draw(0,0,0);flushed(TB_EPOCH);
    reset(2);deferred();dirty++;texture_barrier_before_draw(0,0,0);flushed(TB_EPOCH);
    reset(2);deferred();pool_generation++;texture_barrier_before_draw(0,0,0);flushed(TB_EPOCH);
    reset(2);deferred();g_pkg_n++;texture_barrier_before_draw(0,0,0);flushed(TB_EPOCH);
    reset(2);deferred();texture_barrier_before_draw(0,0,0);g_tex_gen++;assert(texture_barrier_recheck());flushed(TB_EPOCH);
    // Buffer reads include descriptor alignment prefixes and the entire vertex
    // backing buffer, because indexed fetches are not size-bounded by Vulkan.
    reset(2);deferred();texture_barrier_before_draw(0x100100,64,0);flushed(TB_BUF_ALIAS);
    reset(2);deferred();vs.local[0]=1;S.vbuf[0]=(NVNbufferRange){0x180000,32};texture_barrier_before_draw(0,0,0);flushed(TB_BUF_ALIAS);
    reset(2);vs.local[0]=1;S.vbuf[0]=(NVNbufferRange){0x510000,32};deferred();texture_barrier_before_draw(0,0,0);assert(g_tb.pending && !ends);
    reset(2);S.rt_col[0].mem=(void *)(uintptr_t)0x100120;S.rt_col[0].L.size=32;attachment.size=32;
    S.ubo[0][0]=(p_bind_buf){0,0,0x100140,32};deferred();texture_barrier_before_draw(0,0,0);flushed(TB_BUF_ALIAS);
    reset(2);deferred();texture_barrier_before_draw(0,0,1);flushed(TB_WRITER);
    // No sparse, virtual, unknown-size or mismatched-image source is retained.
    reset(2);S.rt_col[0].d.flags=NVN_TEXTURE_FLAGS_SPARSE;assert(!texture_barrier_try(0x10));
    reset(2);pools[0].flags=NVN_MEMORY_POOL_FLAGS_VIRTUAL;assert(!texture_barrier_try(0x10));
    reset(2);attachment.size++;assert(!texture_barrier_try(0x10));
    reset(2);pools[0].lazy=1;assert(!texture_barrier_try(0x10));
    reset(2);S.pass_open=2;P_cleared=1;assert(!texture_barrier_try(0x10));P_cleared=0;
    // Every other mask stays on the original path, including mixed/unknown bits.
    const int32_t masks[]={1,2,4,8,0x12,0x21,0x40,0x80,0x100,-1};
    for(unsigned i=0;i<sizeof masks/sizeof masks[0];i++){reset(2);deferred();game_barrier(masks[i]);assert(!g_tb.pending && ends==1 && barriers==1);}
    const uint32_t opaque[]={OP_DISPATCH,OP_COPY_B2B,OP_UPDATE_UB,OP_CLEAR_COLOR,OP_CLEAR_DS,OP_SIGNAL_EVENT,OP_REPORT_COUNTER,OP_FENCE_SYNC,OP_TF_STATE,OP_DRAW_TF,UINT32_MAX};
    for(unsigned i=0;i<sizeof opaque/sizeof opaque[0];i++){reset(2);deferred();texture_barrier_record(opaque[i],NULL);flushed(TB_OP);}
    reset(2);deferred();p_bind_handles handles={.n=1};texture_barrier_record(OP_BIND_TEX,&handles);assert(g_tb.pending && !ends);
    handles.defer=1;handles.src=0x123456;texture_barrier_record(OP_BIND_TEX,&handles);flushed(TB_OP); // flush before deferred CPU handle read
    reset(2);S.rt_col[1]=S.rt_col[0];P_att[1].im=&attachment;assert(!texture_barrier_try(0x10)); // aliased attachments are outside this path
    reset(2);deferred();g_tb.draws=128;texture_barrier_before_draw(0,0,0);flushed(TB_LIMIT);
    reset(2);g_pass_draws=128;assert(!texture_barrier_try(0x10) && !g_tb.pending);
    reset(2);deferred();end_pass_reason(VK_PASS_END_SUBMIT_END,0);assert(ends==1 && barriers==1 && events[0]==1 && events[1]==2 && !g_tb.pending);
    reset(2);deferred();setenv("DM_VK_TEXTURE_BARRIER","0",1);texture_barrier_config();assert(!g_tb_mode && !g_tb.pending && ends==1 && barriers==1);
    setenv("DM_VK_TEXTURE_BARRIER","9",1);texture_barrier_config();assert(!g_tb_mode);
    setenv("DM_VK_TEXTURE_BARRIER","1",1);texture_barrier_config();assert(g_tb_mode==1);
    g_tex_frame=600;texture_barrier_config();assert(g_tb_report==600);texture_barrier_config();
    // A live limit change flushes an existing dependency before changing policy.
    reset(2);deferred();setenv("DM_VK_TEXTURE_BARRIER","2",1);setenv("DM_VK_TEXTURE_BARRIER_DRAWS","256",1);
    texture_barrier_config();assert(g_tb_draw_limit==256 && !g_tb.pending && ends==1 && barriers==1);
    reset(2);g_pass_draws=127;deferred();
    for(unsigned i=127;i<256;i++){g_pass_draws=i;texture_barrier_before_draw(0,0,0);assert(g_tb.pending && !ends);}
    texture_barrier_before_draw(0,0,0);flushed(TB_LIMIT);
    reset(2);g_pass_draws=256;assert(!texture_barrier_try(0x10));
    reset(2);g_pass_draws=200;deferred();S.tex[1][0]=2;texture_barrier_before_draw(0,0,0);flushed(TB_TEX_ALIAS);
    reset(2);g_pass_draws=200;deferred();dirty++;texture_barrier_before_draw(0,0,0);flushed(TB_EPOCH);
    reset(2);g_pass_draws=200;deferred();S.rast_discard=1;texture_barrier_before_draw(0,0,0);flushed(TB_WRITER);
    reset(2);g_pass_draws=200;deferred();texture_barrier_record(OP_DISPATCH,NULL);flushed(TB_OP);
    setenv("DM_VK_TEXTURE_BARRIER_DRAWS","512",1);texture_barrier_config();assert(g_tb_draw_limit==512);
    const char *invalid[]={"", "0", "-1", "127", "513", "999999999999999999999999", "256junk"};
    for(unsigned i=0;i<sizeof invalid/sizeof invalid[0];i++){setenv("DM_VK_TEXTURE_BARRIER_DRAWS",invalid[i],1);texture_barrier_config();assert(g_tb_draw_limit==128);}
    // Identical 1024 independent draws and 1023 invalidations: production guard
    // and pass-end paths reduce physical ends, not the number of guest draws.
    for(unsigned limit=128;limit<=256;limit*=2){
        setenv("DM_VK_TEXTURE_BARRIER_DRAWS",limit==128?"128":"256",1);texture_barrier_config();
        reset(2);g_pass_draws=1;unsigned draws=1;
        for(unsigned i=1;i<1024;i++){
            game_barrier(0x10);texture_barrier_before_draw(0,0,0);
            if(!S.pass_open){pass_commit();g_pass_draws=0;}
            g_pass_draws++;draws++;
        }
        end_pass_reason(VK_PASS_END_SUBMIT_END,0);
        assert(draws==1024 && ends==(int)(1024/limit) && barriers==ends && !g_tb.pending);
    }
    unsetenv("DM_VK_TEXTURE_BARRIER_DRAWS");unsetenv("DM_VK_TEXTURE_BARRIER");texture_barrier_config();assert(!g_tb_mode && g_tb_draw_limit==128);
    return 0;
}
''')

    def test_guarded_mode_rebinds_actual_index_state_on_every_draw(self):
        source=(DM/'android/gfx_vk_exec.c').read_text()
        start=source.index('static struct { uint64_t gen; VkViewport vp;')
        end=source.index('} g_bc;',start)+len('} g_bc;')
        self.compile_run(r'''
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan_core.h>
#include <assert.h>
#include <stdint.h>
#include <string.h>
#include "nvn_obj.h"
static uint64_t g_cmd_gen=7;
static int g_x_nobc,g_tb_mode,g_bc_stale,g_x_rebind_fast,binds;
static unsigned cache_clears;
static void *counted_clear(void *p,int c,size_t n){cache_clears++;return memset(p,c,n);}
#undef memset
#define memset counted_clear
static void vkCmdBindIndexBuffer(VkCommandBuffer cb,VkBuffer b,VkDeviceSize o,VkIndexType t){
    assert(cb==(VkCommandBuffer)(uintptr_t)1 && b==(VkBuffer)(uintptr_t)2 && o==32 && t==VK_INDEX_TYPE_UINT16);binds++;
}
'''+source[start:end]+'\n'+function(source,'bc_fresh')+'\n'+function(source,'bc_index')+r'''
static void bind(void){bc_index((VkCommandBuffer)(uintptr_t)1,(VkBuffer)(uintptr_t)2,32,VK_INDEX_TYPE_UINT16);}
int main(void){
    // The original/observational paths still suppress identical bindings.
    bind();bind();assert(binds==1);g_tb_mode=1;bind();assert(binds==1);
    // Guarded merging must emit the complete state even across identical draws.
    g_tb_mode=2;g_bc.vp.width=1920;g_bc.sets[0]=(VkDescriptorSet)(uintptr_t)3;
    bind();assert(binds==2 && !g_bc.vp.width && !g_bc.sets[0] && g_bc.gen==g_cmd_gen);
    bind();assert(binds==3);
    // Returning to the original mode restores caching; existing invalidation
    // paths still work independently of the merged-pass feature.
    g_tb_mode=0;bind();assert(binds==3);
    g_cmd_gen++;bind();assert(binds==4);
    g_bc_stale=1;bind();assert(binds==5 && !g_bc_stale);
    g_x_nobc=1;bind();bind();assert(binds==7);
    unsigned before=cache_clears;
    g_x_rebind_fast=1;g_bc.vp.width=1920;bind();bind();assert(binds==9 && g_bc.vp.width==1920 && cache_clears==before);
    g_cmd_gen++;bind();assert(binds==10 && !g_bc.vp.width && cache_clears==before+1);
    g_bc.vp.width=1920;g_bc_stale=1;bind();assert(binds==11 && !g_bc.vp.width && cache_clears==before+2);
    g_x_nobc=0;g_tb_mode=2;bind();bind();assert(binds==13);
    g_tb_mode=0;bind();assert(binds==13);

    return 0;
}
''')

    def test_actual_readiness_is_metadata_only_and_rejects_new_dirty_linear_or_sparse_images(self):
        source=(DM/'android/gfx_vk_tex.c').read_text()
        ready=function(source,'vk_tex_barrier_ready')
        self.compile_run(r'''
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan_core.h>
#include <assert.h>
#include <stdint.h>
#include <pthread.h>
#include "nvn_obj.h"
typedef struct {int depth,stencil;} vkfmt_t;
typedef struct {VkImageLayout layout;int64_t size;int packaged,pkg_released;void *pkg_span;uint64_t pkg_stamp;} vkimg_t;
typedef struct {uint64_t pool;int64_t off;nvn_texdesc d;} akey;
static int g_by_key;static pthread_mutex_t g_img_mu=PTHREAD_MUTEX_INITIALIZER;
static vkimg_t image,*found=&image;static int package_required,dirty,lookups;
static vkfmt_t fmt;
static const vkfmt_t *vk_fmt(int f){return f==1?&fmt:NULL;}
static void *nvn_hm_get(void *map,const void *key){assert(map==&g_by_key);const akey *k=key;assert(k->pool==11 && k->off==0x2000);lookups++;return found;}
static int packaged_tex(const tex_t *t){(void)t;return package_required;}
static int nvn_dirty_since_span(const void *span,const void *mem,size_t n,uint64_t stamp){assert(span==(void *)(uintptr_t)3 && mem==(void *)(uintptr_t)0x100000 && n==0x1000 && stamp==7);return dirty;}
'''+ready+r'''
int main(void){
    tex_t t={0};t.magic=MAGIC('T','E','X',0);t.serial=1;t.pool=(void *)(uintptr_t)11;t.mem=(void *)(uintptr_t)0x100000;t.off=0x2000;t.L.size=0x1000;t.d.format=1;t.d.target=1;
    image=(vkimg_t){.layout=VK_IMAGE_LAYOUT_GENERAL,.size=0x1000,.pkg_span=(void *)(uintptr_t)3,.pkg_stamp=7};
    assert(vk_tex_barrier_ready(&t));found=NULL;assert(!vk_tex_barrier_ready(&t));found=&image;
    image.layout=VK_IMAGE_LAYOUT_UNDEFINED;assert(!vk_tex_barrier_ready(&t));image.layout=VK_IMAGE_LAYOUT_GENERAL;
    image.size++;assert(!vk_tex_barrier_ready(&t));image.size--;
    package_required=1;assert(!vk_tex_barrier_ready(&t));image.packaged=1;assert(vk_tex_barrier_ready(&t));
    dirty=1;assert(!vk_tex_barrier_ready(&t));dirty=0;image.pkg_released=1;assert(!vk_tex_barrier_ready(&t));image.pkg_released=0;
    const uint32_t flags[]={NVN_TEXTURE_FLAGS_LINEAR,NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET,NVN_TEXTURE_FLAGS_SPARSE};
    int before=lookups;for(unsigned i=0;i<3;i++){t.d.flags=flags[i];assert(!vk_tex_barrier_ready(&t));}assert(before==lookups);t.d.flags=0;
    t.d.target=NVN_TEXTURE_TARGET_BUFFER;assert(!vk_tex_barrier_ready(&t));t.d.target=1;
    t.mem=NULL;assert(!vk_tex_barrier_ready(&t));assert(!vk_tex_barrier_ready(NULL));return 0;
}
''')
