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
               'vk_gpu_segment_end','gpu_segment_cpu','gpu_segment_part_begin','gpu_segment_part_end',
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
static VkQueryPool g_gsq;static vk_gs_state g_gs_slot[NSLOT];static int g_gs_active,g_cur=2;
static int g_gs_request=-1;static vk_gs_sampler g_gs_sampler;static uint32_t g_gs_captures;
static uint64_t g_cur_frame=100;static struct {VkDevice dev;} V;
static unsigned writes,resets,barriers,reads,query_next;static int final_fence_passed;
static VkCommandBuffer main_cb=(VkCommandBuffer)(uintptr_t)7;
static VkCommandBuffer g_slot_main_cb(void){assert(g_cur>=0);return main_cb;}
static void vkCmdWriteTimestamp(VkCommandBuffer cb,VkPipelineStageFlagBits stage,VkQueryPool pool,uint32_t q){
    assert(cb==main_cb && stage==VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT && pool==g_gsq && q==(uint32_t)g_cur*VK_GS_QUERIES+query_next++);writes++;
}
static void vkCmdResetQueryPool(VkCommandBuffer cb,VkQueryPool pool,uint32_t base,uint32_t count){
    assert(cb==main_cb && pool==g_gsq && base==(uint32_t)g_cur*VK_GS_QUERIES && count==VK_GS_QUERIES);resets++;query_next=0;
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
    return 0;
}
''')
