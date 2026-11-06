"""Execute bounded staging reuse and production retirement paths with host mocks."""
from pathlib import Path
import os,re,shutil,subprocess,tempfile,unittest
from tests.test_lazy_small_pools import function
ROOT=Path(__file__).resolve().parents[1]
DM=ROOT/'nxnative/dm'
@unittest.skipUnless(os.name!='nt' and shutil.which('clang'),'POSIX clang needed')
class StagingCache(unittest.TestCase):
    def test_actual_retirement_precedes_reuse_and_cache_is_bounded(self):
        source=(DM/'android/gfx_vk_core.c').read_text()
        recycle=function(source,'slot_recycle')
        self.assertLess(recycle.index('completion_wait('),recycle.index('staging_cache_retire('))
        allocation=function(source,'pool_allocate_copy')
        self.assertIn('p->synced = 0',allocation)
        self.assertIn('p->span = NULL',allocation)
        code=r'''
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan_core.h>
#include <assert.h>
#include <stdio.h>
#include "staging_cache.h"
static struct {VkDevice dev;} V;
static int g_cur=-1,completed,waits,destroyed,hardware_waits;
static uint64_t g_fence=17;
typedef struct {int kind;void *a,*b;} dfree_t;
typedef struct {int ndf,capdf;dfree_t *df;} vksub_t;
static vksub_t slot;
static vksub_t *cur(void){return &slot;}
static void completion_wait(uint64_t fence,uint64_t timeout,int reuse){assert(fence==17 && timeout==UINT64_MAX && reuse);assert(!g_staging_cache.returned);waits++;completed=1;}
static void wait_fence(uint64_t f){assert(f==17);hardware_waits++;}
static void vkDestroyBuffer(VkDevice d,VkBuffer b,const void *a){(void)d;(void)a;assert(b);destroyed++;}
static void vkFreeMemory(VkDevice d,VkDeviceMemory m,const void *a){(void)d;(void)a;assert(m);}
static void vkDestroyImage(VkDevice d,VkImage b,const void *a){(void)d;(void)b;(void)a;}
static void vkDestroyImageView(VkDevice d,VkImageView b,const void *a){(void)d;(void)b;(void)a;}
static void vkDestroyBufferView(VkDevice d,VkBufferView b,const void *a){(void)d;(void)b;(void)a;}
void img_sub_free_(void *p){(void)p;}
'''+function(source,'staging_cache_retire')+'\n'+function(source,'df_add')+r'''
static staging_entry entry(uintptr_t id,uint64_t size){return (staging_entry){(VkBuffer)id,(VkDeviceMemory)id,(void *)id,size,65536};}
static staging_entry *owned(uintptr_t id,uint64_t size){staging_entry *e=malloc(sizeof *e);*e=entry(id,size);return e;}
int main(int argc,char **argv){
 assert(argc==2);setenv("DM_VK_STAGING_CACHE",argv[1],1);int on=atoi(argv[1]);
 staging_entry out={0};assert(!staging_cache_take(4096,&out));
 df_add(5,owned(1,4096),NULL);assert(waits==1 && completed && !hardware_waits);
 if(!on){assert(destroyed==1 && !staging_cache_take(4096,&out));return 0;}
 assert(!destroyed && !staging_cache_take(8192,&out));assert(staging_cache_take(4096,&out) && out.buf==(VkBuffer)1 && out.map==(void *)1);
 assert(!staging_cache_take(4096,&out));
 // A resource held by an in-flight slot cannot enter the reusable cache yet.
 g_cur=0;df_add(5,owned(2,4096),NULL);assert(slot.ndf==1 && !staging_cache_take(4096,&out));
 assert(slot.df[0].kind==5);staging_cache_retire(slot.df[0].a);slot.ndf=0;
 assert(staging_cache_take(4096,&out) && out.buf==(VkBuffer)2);free(slot.df);
 // Exact sizes only; cache has eight slots and at most one MiB allocation memory.
 for(int i=0;i<8;i++){staging_entry e=entry(10+i,4096+i*256);assert(staging_cache_return(&e));}
 staging_entry e=entry(30,4096);assert(!staging_cache_return(&e));
 e=entry(31,65536);assert(!staging_cache_return(&e));e=entry(32,4096);e.allocated=131073;assert(!staging_cache_return(&e));
 e=entry(33,4096);e.map=NULL;assert(!staging_cache_return(&e));
 for(int i=0;i<8;i++){assert(staging_cache_take(4096+i*256,&out));assert(out.buf==(VkBuffer)(uintptr_t)(10+i));}
 assert(!staging_cache_take(4096,&out));return 0;
}
'''
        with tempfile.TemporaryDirectory(prefix='nx2a-staging-') as tmp:
            src,exe=Path(tmp)/'check.c',Path(tmp)/'check';src.write_text(code)
            run=subprocess.run(['clang','-std=gnu11','-O2','-Wall','-Wextra','-Werror','-pthread','-I'+str(DM/'android'),'-I'+str(DM/'third_party/vulkan-headers/include'),str(src),'-o',str(exe)],capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr)
            for mode in ('0','1'):
                run=subprocess.run([str(exe),mode],capture_output=True,text=True,timeout=10)
                self.assertEqual(run.returncode,0,run.stdout+run.stderr)
