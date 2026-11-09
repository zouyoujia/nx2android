"Execute the production GPU interval planner, reducer and Vulkan query hooks."
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
class GpuSegments(unittest.TestCase):
    def compile_run(self, code):
        with tempfile.TemporaryDirectory(prefix='nx2a-gpu-segments-') as tmp:
            src, exe = Path(tmp)/'check.c', Path(tmp)/'check'
            src.write_text(code)
            run = subprocess.run([
                'clang', '-std=gnu11', '-O2', '-Wall', '-Wextra', '-Werror',
                '-Wno-missing-field-initializers', '-I'+str(DM/'android'),
                '-I'+str(DM/'third_party/vulkan-headers/include'),
                str(src), '-o', str(exe)], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=10)
            self.assertEqual(run.returncode, 0, run.stdout+run.stderr)

    def test_whole_frame_sampling_spacing_budget_and_cancellation(self):
        self.compile_run(r'''
#include <assert.h>
#include "gpu_segments.h"
int main(void){
    vk_gs_sampler s={0};
    for(uint64_t f=0;f<100;f++)assert(!vk_gs_sample(&s,f));
    vk_gs_request(&s,99,100);assert(s.remaining==8 && !vk_gs_sample(&s,100));
    int frames=0;
    for(uint64_t f=101;f<350;f++){
        int selected=vk_gs_sample(&s,f);frames+=selected;
        assert(vk_gs_sample(&s,f)==selected && vk_gs_sample(&s,f)==selected);
        assert(selected==(f<=311 && (f-101)%30==0));
    }
    assert(frames==8 && !s.remaining);
    vk_gs_request(&s,1,350);assert(!vk_gs_sample(&s,350));
    assert(vk_gs_sample(&s,351) && !s.remaining && vk_gs_sample(&s,351));
    vk_gs_request(&s,0,351);assert(!vk_gs_sample(&s,351) && !vk_gs_sample(&s,352));
    vk_gs_request(&s,-2,400);assert(!s.remaining);
    vk_gs_request(&s,2,UINT64_MAX-2);assert(!vk_gs_sample(&s,UINT64_MAX-2));
    assert(vk_gs_sample(&s,UINT64_MAX-1));assert(!vk_gs_sample(&s,0));assert(vk_gs_sample(&s,28));
    return 0;
}
''')

    def test_cpu_section_epochs_bounds_disabled_and_snapshot(self):
        self.compile_run(r'''
#include <assert.h>
#include "gpu_segments.h"
static vk_gs_state s;
int main(void){
 vk_gs_part_begin(&s);vk_gs_cpu_add(&s,VK_GS_CPU_DESC,100,1,1);assert(!s.work[0][VK_GS_CPU_DESC].calls);
 s.cpu_sections=1;vk_gs_cpu_add(&s,VK_GS_CPU_DESC,100,1,1);vk_gs_cpu_add(&s,VK_GS_CPU_DESC,23,1,1);
 assert(s.work[0][VK_GS_CPU_DESC].calls==2 && s.work[0][VK_GS_CPU_DESC].ns==123);
 vk_gs_cpu_add(&s,VK_GS_CPU_OP+25,200,1,1);assert(s.op[0][25].ns==200 && s.op[0][25].calls==1);
 vk_gs_cpu_add(&s,VK_GS_CPU_DESC,9999,1,2);assert(s.cpu_crossed[0]==1 && s.work[0][VK_GS_CPU_DESC].ns==123);
 vk_gs_cpu_add(&s,VK_GS_CPU_OP+64,10,2,2);assert(!s.op[0][63].calls && s.work[0][VK_GS_CPU_DESC].calls==2);
 vk_gs_part_end(&s);vk_gs_cpu_add(&s,VK_GS_CPU_DESC,9999,2,2);assert(s.work[0][VK_GS_CPU_DESC].ns==123);
 vk_gs_part_begin(&s);vk_gs_cpu_add(&s,VK_GS_CPU_VERTEX,10,2,2);assert(s.work[1][VK_GS_CPU_VERTEX].ns==10 && !s.work[0][VK_GS_CPU_VERTEX].calls);
 vk_gs_state copy=s;memset(&s,0,sizeof s);assert(copy.work[0][VK_GS_CPU_DESC].ns==123 && copy.op[0][25].ns==200 && copy.work[1][VK_GS_CPU_VERTEX].ns==10);
 return 0;
}
''')

    def test_completion_intervals_partition_gaps_and_wraparound(self):
        self.compile_run(r'''
#include <assert.h>
#include "gpu_segments.h"
static vk_gs_state s;static vk_gs_query q[VK_GS_QUERIES];
static void stamp(uint32_t i,uint64_t tick){assert(i<VK_GS_QUERIES);q[i]=(vk_gs_query){tick,1};}
int main(void){
    stamp(vk_gs_part_begin(&s),100);
    uint32_t r=vk_gs_begin(&s,VK_GS_LRZ_READ,0,1,1);stamp(s.rec[r-1].begin,110);
    assert(!vk_gs_begin(&s,VK_GS_BARRIER,0,0,0));assert(vk_gs_end(&s,0,1)==VK_GS_NONE);
    stamp(vk_gs_end(&s,r,1),130);
    r=vk_gs_begin(&s,VK_GS_PASS,9,1920,1080);stamp(s.rec[r-1].begin,140);stamp(vk_gs_end(&s,r,20),200);
    stamp(vk_gs_part_end(&s),220);
    stamp(vk_gs_part_begin(&s),250);
    r=vk_gs_begin(&s,VK_GS_COMPUTE,11,30,17);stamp(s.rec[r-1].begin,260);stamp(vk_gs_end(&s,r,1),280);
    stamp(vk_gs_part_end(&s),300);
    assert(s.nq==10 && s.nr==3 && s.np==2 && s.nested==1 && !s.invalid);
    vk_gs_totals t=vk_gs_reduce(&s,q,64);
    assert(!t.invalid && t.span==200 && t.in_parts==170 && t.between_parts==30 && t.unclassified==70);
    assert(t.ticks[VK_GS_PASS]==60 && t.ticks[VK_GS_LRZ_READ]==20 && t.ticks[VK_GS_COMPUTE]==20 && !t.ticks[VK_GS_BARRIER]);
    assert(t.counts[VK_GS_PASS]==1 && s.rec[1].count==20 && s.rec[1].w==1920);
    for(unsigned i=0;i<s.nq;i++)q[i].ticks=(q[i].ticks+150)&255;
    t=vk_gs_reduce(&s,q,8);assert(t.invalid); // full span exceeds half the modulus: ambiguous
    const uint64_t ticks[]={100,110,130,140,200,220,250,260,280,300};
    for(unsigned i=0;i<s.nq;i++)q[i].ticks=(ticks[i]+850)&1023;
    t=vk_gs_reduce(&s,q,10);assert(!t.invalid && t.span==200 && t.between_parts==30 && t.unclassified==70);
    q[s.rec[1].end].available=0;t=vk_gs_reduce(&s,q,10);assert(t.invalid==1 && !t.counts[VK_GS_PASS]);
    q[s.rec[1].end].available=1;q[s.rec[1].begin].ticks=q[s.rec[0].begin].ticks;
    t=vk_gs_reduce(&s,q,10);assert(t.invalid==1 && !t.counts[VK_GS_PASS]);
    t=vk_gs_reduce(&s,q,0);assert(t.invalid && !t.valid_parts);
    t=vk_gs_reduce(&s,q,65);assert(t.invalid && !t.valid_parts);
    return 0;
}
''')

    def test_bounds_missing_ends_and_stale_tokens_never_allocate_holes(self):
        self.compile_run(r'''
#include <assert.h>
#include "gpu_segments.h"
static vk_gs_state s;static unsigned writes[VK_GS_QUERIES];
static void mark(uint32_t q){assert(q<VK_GS_QUERIES && !writes[q]++);}
int main(void){
    mark(vk_gs_part_begin(&s));
    for(unsigned i=0;i<VK_GS_RECORDS;i++){
        uint32_t token=vk_gs_begin(&s,VK_GS_PASS,0,1,1);assert(token);mark(s.rec[token-1].begin);mark(vk_gs_end(&s,token,1));
    }
    unsigned nq=s.nq;assert(!vk_gs_begin(&s,VK_GS_PASS,0,0,0) && s.nq==nq && s.dropped==1);
    mark(vk_gs_part_end(&s));
    for(unsigned i=1;i<VK_GS_PARTS;i++){mark(vk_gs_part_begin(&s));mark(vk_gs_part_end(&s));}
    assert(s.nq==VK_GS_QUERIES && !s.invalid);
    assert(vk_gs_part_begin(&s)==VK_GS_NONE && s.dropped==2);
    for(unsigned i=0;i<s.nq;i++)assert(writes[i]==1);
    memset(&s,0,sizeof s);
    assert(vk_gs_begin(&s,VK_GS_PASS,0,0,0)==0 && s.invalid==1);
    assert(vk_gs_part_begin(&s)==0);
    uint32_t token=vk_gs_begin(&s,VK_GS_PASS,0,0,0);assert(token==1 && s.nq==2);
    assert(vk_gs_part_end(&s)==2 && s.invalid==2 && s.nq==3 && s.rec[0].end==VK_GS_NONE);
    assert(vk_gs_end(&s,token,1)==VK_GS_NONE && s.invalid==3 && s.nq==3);
    vk_gs_query q[VK_GS_QUERIES]={0};for(unsigned i=0;i<s.nq;i++)q[i]=(vk_gs_query){i*10,1};
    vk_gs_totals t=vk_gs_reduce(&s,q,64);assert(t.invalid==1 && !t.counts[VK_GS_PASS]);
    return 0;
}
''')

    def test_actual_vulkan_hooks_default_off_nesting_slot_snapshot_and_failed_submission(self):
        source=(DM/'android/gfx_vk_core.c').read_text()
        names=('vk_gpu_segments_request','gpu_segment_stamp','vk_gpu_segment_begin',
               'vk_gpu_segment_end','vk_gpu_segment_active','gpu_segment_calibrate','gpu_segment_cpu','gpu_segment_part_begin','gpu_segment_part_end',
               'gpu_segment_sub_begin','gpu_segment_snapshot','gpu_segment_collect','vk_img_barrier_all')
        implementations='\n'.join(function(source,name) for name in names)
        capture=re.search(r'typedef struct \{[^\n]+\} gpu_capture;',source).group()
        completion=function(source,'completion_main')
        self.assertLess(completion.index('gpu_segment_collect('),completion.index('completion_retire(c)'))
        self.assertLess(completion.index('completion_retire(c)'),completion.index('gpu_segment_report('))
        split=function(source,'vk_cmd_split_')
        self.assertLess(split.index('gpu_segment_part_end()'),split.index('vkEndCommandBuffer(s->main)'))
        self.assertGreater(split.index('gpu_segment_part_begin()'),split.index('vkBeginCommandBuffer(s->main'))
        commit=function(source,'mtl_sub_commit')
        self.assertLess(commit.index('gpu_segment_part_end()'),commit.index('vkEndCommandBuffer(s->main)'))
        self.assertLess(split.index('gpu_segment_cpu(2)'),split.index('gpu_segment_cpu(3)'))
        self.assertLess(split.index('gpu_segment_cpu(3)'),split.index('vkQueueSubmit('))
        self.assertGreater(split.index('gpu_segment_cpu(4)'),split.index('vkQueueSubmit('))
        self.assertLess(commit.index('gpu_segment_cpu(2)'),commit.index('gpu_segment_cpu(3)'))
        self.assertGreater(commit.index('gpu_segment_cpu(4)'),commit.index('vkQueueSubmit('))

        self.compile_run(r'''
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan_core.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include "gpu_segments.h"
#include <time.h>
#define NSLOT 6
#define VKLOG(...) do {if(0)fprintf(stderr,__VA_ARGS__);} while(0)
static PFN_vkGetCalibratedTimestampsEXT g_calibrate;
static VkQueryPool g_gsq,g_gsuq;static vk_gs_state g_gs_slot[NSLOT];static int g_gs_active,g_cur=2;static int g_vk_cpu_sections;
static int g_gs_request=-1;static vk_gs_sampler g_gs_sampler;static uint32_t g_gs_captures;
static uint64_t g_cur_frame=100;static struct {VkDevice dev;} V;
static unsigned writes,resets,barriers,reads,query_next;static int final_fence_passed;
static VkCommandBuffer main_cb=(VkCommandBuffer)(uintptr_t)7;
static VkCommandBuffer g_slot_main_cb(void){assert(g_cur>=0);return main_cb;}
static void vkCmdWriteTimestamp(VkCommandBuffer cb,VkPipelineStageFlagBits stage,VkQueryPool pool,uint32_t q){
    assert(cb==main_cb && stage==VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT && pool==g_gsq && q==(uint32_t)g_cur*VK_GS_QUERIES+query_next++);writes++;
}
static void vkCmdResetQueryPool(VkCommandBuffer cb,VkQueryPool pool,uint32_t base,uint32_t count){
    assert(cb==main_cb && pool==g_gsq && base==(uint32_t)g_cur*VK_GS_QUERIES && (count==VK_GS_QUERIES || count==2*VK_GS_PARTS || count==2*(VK_GS_PARTS+256)));resets++;query_next=0;
}
static void vkCmdPipelineBarrier(VkCommandBuffer cb,VkPipelineStageFlags src,VkPipelineStageFlags dst,VkDependencyFlags dep,uint32_t nm,const VkMemoryBarrier *m,uint32_t nb,const VkBufferMemoryBarrier *b,uint32_t ni,const VkImageMemoryBarrier *i){
    assert(cb && src==VK_PIPELINE_STAGE_ALL_COMMANDS_BIT && dst==VK_PIPELINE_STAGE_ALL_COMMANDS_BIT && !dep && nm==1 && !nb && !b && !ni && !i);
    assert(m->srcAccessMask==(VK_ACCESS_MEMORY_READ_BIT|VK_ACCESS_MEMORY_WRITE_BIT) && m->dstAccessMask==m->srcAccessMask);barriers++;
}
static VkResult vkGetQueryPoolResults(VkDevice dev,VkQueryPool pool,uint32_t first,uint32_t count,size_t bytes,void *data,VkDeviceSize stride,VkQueryResultFlags flags){
    assert(final_fence_passed && dev==V.dev && pool==g_gsq && first==2*VK_GS_QUERIES && count==8 && bytes==count*sizeof(vk_gs_query) && stride==sizeof(vk_gs_query));
    assert(flags==(VK_QUERY_RESULT_64_BIT|VK_QUERY_RESULT_WITH_AVAILABILITY_BIT));
    vk_gs_query *q=data;for(unsigned i=0;i<count;i++)q[i]=(vk_gs_query){100+i*10,1};reads++;return VK_SUCCESS;
}
'''+capture+'\n'+implementations+r'''
int main(void){
    gpu_segment_sub_begin();assert(!g_gs_active && !writes && !resets);
    assert(!vk_gpu_segment_begin(VK_GS_PASS,0,0,0));vk_gpu_segment_end(0,1);
    vk_img_barrier_all(main_cb);assert(barriers==1 && !writes);
    g_gsq=(VkQueryPool)(uintptr_t)9;vk_gpu_segments_request(2);gpu_segment_sub_begin();assert(!g_gs_active && !writes);
    g_cur_frame=101;gpu_segment_sub_begin();assert(g_gs_active && writes==1 && resets==1);
    uint32_t token=vk_gpu_segment_begin(VK_GS_LRZ_WRITE,2,1,1);assert(token && writes==2);
    vk_img_barrier_all(main_cb);vk_img_barrier_all(main_cb);assert(writes==2 && barriers==3);
    vk_gpu_segment_end(token,1);assert(writes==3 && g_gs_slot[2].nested==2);
    vk_img_barrier_all((VkCommandBuffer)(uintptr_t)8);assert(writes==3 && g_gs_slot[2].upload_barriers==1);
    vk_img_barrier_all(main_cb);assert(writes==5 && barriers==5);
    gpu_segment_part_end();assert(writes==6);main_cb=(VkCommandBuffer)(uintptr_t)17;gpu_segment_part_begin();assert(writes==7);
    gpu_segment_part_end();assert(writes==8);
    gpu_segment_cpu(2);gpu_segment_cpu(3);gpu_segment_cpu(4);
    assert(g_gs_slot[2].cpu[0][0] && g_gs_slot[2].cpu[0][1]>=g_gs_slot[2].cpu[0][0]);
    for(int i=1;i<5;i++)assert(g_gs_slot[2].cpu[1][i]>=g_gs_slot[2].cpu[1][i-1]);

    gpu_capture *c=gpu_segment_snapshot(0);assert(c && !g_gs_active && c->state.nq==8 && c->state.np==2);
    memset(&g_gs_slot[2],0,sizeof g_gs_slot[2]);
    assert(c->state.nr==2 && c->state.nested==2);
    assert(c->state.cpu[0][0] && c->state.cpu[1][4]>=c->state.cpu[1][0]);
    final_fence_passed=1;gpu_segment_collect(c,2);assert(reads==1 && !c->result);
    vk_gs_totals t=vk_gs_reduce(&c->state,c->query,64);assert(!t.invalid && t.span==70 && t.between_parts==10 && t.in_parts==60);
    free(c);assert(!gpu_segment_snapshot(0));
    gpu_segment_sub_begin();assert(g_gs_active);gpu_segment_part_end();assert(!gpu_segment_snapshot(1) && !g_gs_active && reads==1);
    g_cur_frame=102;gpu_segment_sub_begin();assert(!g_gs_active);
    vk_gpu_segments_request(0);gpu_segment_sub_begin();g_cur_frame=200;gpu_segment_sub_begin();assert(!g_gs_active);
    vk_gpu_segments_request(8);gpu_segment_sub_begin();g_cur_frame=201;
    for(int i=0;i<64;i++){gpu_segment_sub_begin();assert(g_gs_active);gpu_segment_part_end();free(gpu_segment_snapshot(0));}
    gpu_segment_sub_begin();assert(!g_gs_active && !g_gs_sampler.remaining && !g_gs_sampler.selected);
    setenv("DM_VK_GPU_PARTS_ONLY","1",1);setenv("DM_VK_CPU_SECTIONS","1",1);
    vk_gpu_segments_request(1);gpu_segment_sub_begin();g_cur_frame=401;gpu_segment_sub_begin();
    assert(g_gs_active && g_vk_cpu_sections && g_gs_slot[2].parts_only && g_gs_slot[2].cpu_sections);
    unsigned old=writes;assert(!vk_gpu_segment_begin(VK_GS_PASS,0,1920,1080));assert(writes==old);
    vk_img_barrier_all(main_cb);assert(writes==old);gpu_segment_part_end();assert(writes==old+1);
    c=gpu_segment_snapshot(0);assert(c && c->state.nq==2 && !c->state.nr && !g_vk_cpu_sections);free(c);
    unsetenv("DM_VK_GPU_PARTS_ONLY");unsetenv("DM_VK_CPU_SECTIONS");
    setenv("DM_VK_GPU_PASSES_ONLY","1",1);vk_gpu_segments_request(1);gpu_segment_sub_begin();g_cur_frame=601;gpu_segment_sub_begin();
    assert(g_gs_active && g_gs_slot[2].passes_only && !g_gs_slot[2].parts_only && !vk_gpu_segment_active() && !g_vk_cpu_sections);
    old=writes;vk_img_barrier_all(main_cb);assert(writes==old);
    assert(!vk_gpu_segment_begin(VK_GS_LRZ_READ,0,1,1));assert(!vk_gpu_segment_begin(VK_GS_LRZ_WRITE,0,1,1));
    token=vk_gpu_segment_begin(VK_GS_PASS,0,1920,1080);assert(token);vk_gpu_segment_end(token,1460);
    token=vk_gpu_segment_begin(VK_GS_COMPUTE,0,1,1);assert(token);vk_gpu_segment_end(token,1);gpu_segment_part_end();
    c=gpu_segment_snapshot(0);assert(c && c->state.nq==6 && c->state.nr==2 && !c->state.detail.n);free(c);
    unsetenv("DM_VK_GPU_PASSES_ONLY");
    return 0;
}
''')


    def test_upload_hooks_query_ownership_partial_coverage_and_queue_order(self):
        source=(DM/'android/gfx_vk_core.c').read_text()
        names=('gpu_segment_upload','gpu_segment_queue','vk_gpu_upload_note',
               'vk_gpu_segment_active','vk_gpu_segment_draw','gpu_segment_collect')
        implementations='\n'.join(function(source,name) for name in names)
        capture=re.search(r'typedef struct \{[^\n]+\} gpu_capture;',source).group()
        self.compile_run(r'''
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan_core.h>
#include <assert.h>
#include <stdlib.h>
#include "gpu_segments.h"
#define NSLOT 6
static VkQueryPool g_gsq=(VkQueryPool)1,g_gsuq=(VkQueryPool)2;
static vk_gs_state g_gs_slot[NSLOT];static int g_gs_active,g_cur=2;
static uint64_t g_queue_serial=21,g_present_serial=18,g_present_ticket=77;
static struct {VkDevice dev;} V;
static unsigned resets,writes,reads;static int retired;
static VkCommandBuffer upload=(VkCommandBuffer)9;
static void vkCmdResetQueryPool(VkCommandBuffer cb,VkQueryPool pool,uint32_t first,uint32_t count){
 assert(cb==upload && pool==g_gsuq && first==2*VK_GS_PARTS*3 && count==3 && !writes);resets++;
}
static void vkCmdWriteTimestamp(VkCommandBuffer cb,VkPipelineStageFlagBits stage,VkQueryPool pool,uint32_t q){
 assert(cb==upload && pool==g_gsuq && resets==1 && q==2*VK_GS_PARTS*3+writes);
 assert(stage==(writes?VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT:VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT));writes++;
}
static VkResult vkGetQueryPoolResults(VkDevice dev,VkQueryPool pool,uint32_t first,uint32_t count,size_t bytes,void *data,VkDeviceSize stride,VkQueryResultFlags flags){
 assert(retired && dev==V.dev && stride==sizeof(vk_gs_query));
 assert(flags==(VK_QUERY_RESULT_64_BIT|VK_QUERY_RESULT_WITH_AVAILABILITY_BIT));
 assert((pool==g_gsq && count==2 && first==2*VK_GS_QUERIES) || (pool==g_gsuq && count==3 && first==2*VK_GS_PARTS*3));
 assert(bytes==count*sizeof(vk_gs_query));
 vk_gs_query *q=data;for(unsigned i=0;i<count;i++)q[i]=(vk_gs_query){100+i*20,1};reads++;return VK_SUCCESS;
}
'''+capture+implementations+r'''
int main(void){
 gpu_segment_upload(upload,0);gpu_segment_queue(1);vk_gpu_upload_note(2,4096);
 assert(!resets && !writes && !vk_gpu_segment_active());
 g_gs_active=1;vk_gs_state *s=&g_gs_slot[2];assert(vk_gs_part_begin(s)==0);
 gpu_segment_upload(upload,0);gpu_segment_upload(upload,1);gpu_segment_upload(upload,2);
 assert(resets==1 && writes==3 && s->upload_flags[0]==7);
 gpu_segment_queue(1);assert(s->queue[0][0]==21 && s->queue[0][1]==18 && s->queue[0][2]==77 && s->queue[0][3]==1);
 vk_gpu_upload_note(2,4096);vk_gpu_upload_note(2,2048);assert(s->upload_causes[0][2][0]==2 && s->upload_causes[0][2][1]==6144);
 vk_gs_part_end(s);vk_gs_part_begin(s);g_queue_serial=22;gpu_segment_queue(0);
 assert(s->queue[1][0]==22 && s->queue[1][1]==18 && !s->queue[1][3]);
 gpu_segment_upload(upload,1);gpu_segment_upload(upload,2);assert(s->upload_flags[1]==8 && writes==3);
 // Upload already begun before sampling: no made-up start, reset or stale-query read.
 gpu_capture *c=calloc(1,sizeof *c);c->state=*s;c->state.nq=2;g_gs_active=0;retired=1;
 gpu_segment_collect(c,2);assert(reads==2 && c->upload[0][2].available && !c->upload[1][0].available);
 vk_gd_state *d=&c->state.detail;vk_gd_draw v={.vs=1,.fs=2,.pass=4,.count=1};
 for(int i=0;i<100;i++)vk_gd_add(d,&v);assert(d->n==1 && d->draw[0].count==100);
 v.texture[0]=8;vk_gd_add(d,&v);assert(d->n==2); // shader same, resource dependency differs
 v.pass=5;vk_gd_add(d,&v);assert(d->n==3); // do not merge across passes
 for(unsigned i=3;i<VK_GD_RECORDS;i++){v.pass=i+10;vk_gd_add(d,&v);}
 v.pass=5000;vk_gd_add(d,&v);assert(d->n==VK_GD_RECORDS && d->dropped==1);
 free(c);return 0;
}
''')

    def test_actual_clock_calibration_requires_sparse_sample_and_valid_result(self):
        src=(DM/'android/gfx_vk_core.c').read_text()
        self.compile_run(r'''
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan_core.h>
#include <assert.h>
#include "gpu_segments.h"
static vk_gs_state g_gs_slot[6];static int g_gs_active,g_cur=2;
static struct{VkDevice dev;}V;static unsigned calls;static VkResult result;static uint64_t err=500;
static PFN_vkGetCalibratedTimestampsEXT g_calibrate;
static VkResult fake(VkDevice dev,uint32_t n,const VkCalibratedTimestampInfoEXT *i,uint64_t *v,uint64_t *e){
 assert(dev==V.dev && n==2 && i[0].sType==VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_EXT && !i[0].pNext);
 assert(i[0].timeDomain==VK_TIME_DOMAIN_DEVICE_EXT && i[1].timeDomain==VK_TIME_DOMAIN_CLOCK_MONOTONIC_EXT);
 calls++;v[0]=100;v[1]=123000;*e=err;return result;
}
'''+function(src,'gpu_segment_calibrate')+r'''
int main(void){
 gpu_segment_calibrate(0);assert(!calls);g_calibrate=fake;
 gpu_segment_calibrate(0);assert(!calls);g_gs_active=1;
 gpu_segment_calibrate(VK_GS_PARTS);assert(!calls);
 result=VK_ERROR_DEVICE_LOST;gpu_segment_calibrate(0);assert(calls==1 && !g_gs_slot[2].calibration[0][1]);
 result=VK_SUCCESS;err=0;gpu_segment_calibrate(0);assert(calls==2 && !g_gs_slot[2].calibration[0][1]);
 err=500;gpu_segment_calibrate(1);assert(calls==3 && g_gs_slot[2].calibration[1][0]==100 && g_gs_slot[2].calibration[1][1]==123000 && g_gs_slot[2].calibration[1][2]==500);
 assert(!g_gs_slot[2].calibration[0][0]);g_cur=-1;gpu_segment_calibrate(1);assert(calls==3);
 return 0;
}
''')

    def test_pass_only_query_budget_keeps_part_boundaries_when_records_overflow(self):
        self.compile_run(r'''
#include <assert.h>
#include "gpu_segments.h"
static vk_gs_state s;
int main(void){
 s.passes_only=1;assert(vk_gs_query_cap(&s)==576);assert(vk_gs_part_begin(&s)==0);
 for(unsigned i=0;i<256;i++){uint32_t t=vk_gs_begin(&s,VK_GS_PASS,0,1,1);assert(t);assert(vk_gs_end(&s,t,1)!=VK_GS_NONE);}
 assert(!vk_gs_begin(&s,VK_GS_PASS,0,1,1)&&s.dropped==1);assert(vk_gs_part_end(&s)!=VK_GS_NONE);
 for(unsigned i=1;i<VK_GS_PARTS;i++){assert(vk_gs_part_begin(&s)!=VK_GS_NONE);assert(vk_gs_part_end(&s)!=VK_GS_NONE);}
 assert(s.nq==576 && s.nr==256 && s.np==VK_GS_PARTS && !s.invalid);
 s.parts_only=1;assert(vk_gs_query_cap(&s)==64);s.parts_only=s.passes_only=0;assert(vk_gs_query_cap(&s)==VK_GS_QUERIES);return 0;
}
''')
