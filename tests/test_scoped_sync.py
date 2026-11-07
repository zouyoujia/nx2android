"""Check real operation input scopes and retained full output dependencies."""
from pathlib import Path
import os, shutil, subprocess, tempfile, unittest
from tests.test_lazy_small_pools import function
ROOT=Path(__file__).resolve().parents[1]
DM=ROOT/'nxnative/dm'
@unittest.skipUnless(os.name!='nt' and shutil.which('clang'),'POSIX clang needed')
class ScopedSync(unittest.TestCase):
    def test_input_scopes_full_fallback_and_copy_output_order(self):
        source=(DM/'android/gfx_vk_core.c').read_text()
        exec_=(DM/'android/gfx_vk_exec.c').read_text()
        dispatch=function(exec_,'dispatch')
        self.assertIn('if (g_x_nodispbar) vk_img_barrier_all(cb);',dispatch)
        self.assertIn('d->ind ? VK_BEFORE_COMPUTE_INDIRECT : VK_BEFORE_COMPUTE',dispatch)
        self.assertLess(dispatch.index('vkCmdDispatch('),dispatch.index('if (!g_x_nodispbar) vk_img_barrier_all(cb);'))
        for name in ('mtl_sub_copy','mtl_sub_fill'):
            body=function(source,name)
            self.assertLess(body.index('vk_barrier_before('),body.index('vkCmdCopyBuffer(') if name=='mtl_sub_copy' else body.index('vkCmdFillBuffer('))
        code=r'''
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan_core.h>
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "scoped_sync.h"
#include "pass_census.h"
static uint64_t counts[VK_EFF_N];
static void efficiency_add(int k,uint64_t n){counts[k]+=n;}
static uint32_t vk_gpu_segment_begin(uint32_t k,uint64_t tag,uint32_t w,uint32_t h){(void)k;(void)tag;(void)w;(void)h;return 0;}
static void vk_gpu_segment_end(uint32_t token,uint32_t n){assert(!token && n==1);}
#define VK_GS_BARRIER 4
#define VKRED_ONCE(...) ((void)0)
static struct {VkPipelineStageFlags src,dst;VkAccessFlags reads,writes;} seen[16];static unsigned nseen,copies,fills,ends;static char order[32];static unsigned no;
static void vkCmdPipelineBarrier(VkCommandBuffer cb,VkPipelineStageFlags src,VkPipelineStageFlags dst,VkDependencyFlags dep,uint32_t nm,const VkMemoryBarrier *m,uint32_t nb,const VkBufferMemoryBarrier *b,uint32_t ni,const VkImageMemoryBarrier *i){
 assert(cb==(VkCommandBuffer)(uintptr_t)7 && !dep && nm==1 && m && !nb && !b && !ni && !i);
 seen[nseen].src=src;seen[nseen].dst=dst;seen[nseen].reads=m->srcAccessMask;seen[nseen++].writes=m->dstAccessMask;order[no++]='B';
}
static void vk_img_barrier_all(VkCommandBuffer cb){VkMemoryBarrier b={VK_STRUCTURE_TYPE_MEMORY_BARRIER,NULL,VK_ACCESS_MEMORY_READ_BIT|VK_ACCESS_MEMORY_WRITE_BIT,VK_ACCESS_MEMORY_READ_BIT|VK_ACCESS_MEMORY_WRITE_BIT};vkCmdPipelineBarrier(cb,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,0,1,&b,0,NULL,0,NULL);}
static VkCommandBuffer vk_cmd(void){return (VkCommandBuffer)(uintptr_t)7;}
static int source_valid=1;static uint64_t wrote_a,wrote_n;
static int vk_buf_at(uint64_t a,uint64_t n,VkBuffer *b,VkDeviceSize *o){assert(n==64);*b=(VkBuffer)(uintptr_t)a;*o=16;return source_valid;}
void vk_end_pass_land_reason(enum vk_pass_end_reason r){assert(r==VK_PASS_END_BUFFER_COPY || r==VK_PASS_END_BUFFER_FILL);ends++;}
static void vkCmdCopyBuffer(VkCommandBuffer cb,VkBuffer a,VkBuffer b,uint32_t n,const VkBufferCopy *r){assert(cb==vk_cmd() && a==(VkBuffer)(uintptr_t)1 && b==(VkBuffer)(uintptr_t)2 && n==1 && r->srcOffset==16 && r->dstOffset==16 && r->size==64);copies++;order[no++]='C';}
static void vkCmdFillBuffer(VkCommandBuffer cb,VkBuffer b,VkDeviceSize o,VkDeviceSize n,uint32_t v){assert(cb==vk_cmd() && b==(VkBuffer)(uintptr_t)2 && o==16 && n==64 && v==9);fills++;order[no++]='F';}
void vk_gpu_wrote(uint64_t a,uint64_t n){wrote_a=a;wrote_n=n;}
'''+function(source,'vk_barrier_before')+function(source,'mtl_sub_copy')+function(source,'mtl_sub_fill')+r'''
static void reset(void){nseen=no=copies=fills=ends=0;wrote_a=wrote_n=0;memset(order,0,sizeof order);}
static void full(unsigned n){assert(seen[n].src==VK_PIPELINE_STAGE_ALL_COMMANDS_BIT && seen[n].dst==VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);assert(seen[n].reads==(VK_ACCESS_MEMORY_READ_BIT|VK_ACCESS_MEMORY_WRITE_BIT) && seen[n].writes==seen[n].reads);}
int main(void){
 unsetenv("DM_VK_SCOPED_BARRIERS");mtl_sub_copy(1,2,64);assert(nseen==2 && copies==1 && ends==1 && !strcmp(order,"BCB") && wrote_a==2 && wrote_n==64);full(0);full(1);assert(counts[VK_EFF_FULL_INPUT]==1);
 setenv("DM_VK_SCOPED_BARRIERS","1",1);reset();mtl_sub_copy(1,2,64);assert(!strcmp(order,"BCB") && nseen==2 && copies==1 && wrote_a==2 && wrote_n==64);assert(seen[0].src==VK_PIPELINE_STAGE_ALL_COMMANDS_BIT && seen[0].dst==VK_PIPELINE_STAGE_TRANSFER_BIT);assert(seen[0].writes==(VK_ACCESS_TRANSFER_READ_BIT|VK_ACCESS_TRANSFER_WRITE_BIT));full(1);
 reset();mtl_sub_fill(2,64,9);assert(!strcmp(order,"BFB") && fills==1 && ends==1 && wrote_n==64);assert(seen[0].dst==VK_PIPELINE_STAGE_TRANSFER_BIT && seen[0].writes==VK_ACCESS_TRANSFER_WRITE_BIT);full(1);
 reset();vk_barrier_before(vk_cmd(),VK_BEFORE_COMPUTE);assert(nseen==1 && seen[0].dst==VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);assert(seen[0].writes==(VK_ACCESS_UNIFORM_READ_BIT|VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT));assert(seen[0].reads==(VK_ACCESS_MEMORY_READ_BIT|VK_ACCESS_MEMORY_WRITE_BIT));
 reset();vk_barrier_before(vk_cmd(),VK_BEFORE_COMPUTE_INDIRECT);assert(seen[0].dst==(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT|VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT));assert(seen[0].writes&VK_ACCESS_INDIRECT_COMMAND_READ_BIT);
 reset();vk_barrier_before(vk_cmd(),999);full(0);assert(nseen==1);
 reset();source_valid=0;mtl_sub_copy(1,2,64);mtl_sub_fill(2,64,9);assert(!nseen && !copies && !fills && !ends);source_valid=1;mtl_sub_copy(1,2,0);assert(!nseen && !copies);
 setenv("DM_VK_SCOPED_BARRIERS","0",1);reset();vk_barrier_before(vk_cmd(),VK_BEFORE_COMPUTE_INDIRECT);full(0);return 0;
}
'''
        with tempfile.TemporaryDirectory(prefix='nx2a-scoped-sync-') as tmp:
            src,exe=Path(tmp)/'check.c',Path(tmp)/'check';src.write_text(code)
            p=subprocess.run(['clang','-std=gnu11','-O2','-Wall','-Wextra','-Werror','-Wno-missing-field-initializers','-I'+str(DM/'android'),'-I'+str(DM/'third_party/vulkan-headers/include'),str(src),'-o',str(exe)],capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stderr)
            p=subprocess.run([str(exe)],capture_output=True,text=True,timeout=10);self.assertEqual(p.returncode,0,p.stdout+p.stderr)
