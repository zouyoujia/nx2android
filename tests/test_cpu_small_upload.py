"""Run the actual guarded upload and import-type selector against host mocks."""
from pathlib import Path
from tests.test_lazy_small_pools import function
import os
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
DM = ROOT / 'nxnative/dm'


@unittest.skipUnless(os.name != 'nt' and shutil.which('clang'), 'POSIX host clang needed')
class CpuSmallUpload(unittest.TestCase):
    def test_actual_upload_guards_and_observation(self):
        source = (DM / 'android/gfx_vk_core.c').read_text()
        upload = source[source.index('int mtl_cpu_upload(const uint64_t *handles, int n) {'):
                        source.index('\nint mtl_wait(uint64_t fence, uint64_t timeout_ns) {')]
        mem_type = function(source, 'vk_mem_type')
        choose_type = function(source, 'cpu_small_upload_memory_type')
        code = r'''
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <pthread.h>
#include <string.h>
#include <time.h>
#include "cpu_small_upload.h"
#define VK_SUCCESS 0
#define VK_WHOLE_SIZE UINT64_MAX
#define VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT 2u
#define VK_MEMORY_PROPERTY_HOST_COHERENT_BIT 4u
typedef int VkResult;
typedef uint32_t VkMemoryPropertyFlags;
typedef struct {uint64_t lo,hi,base,end;void *shadow;int zero_copy,lazy,flags,mem;uint32_t memory_type;VkMemoryPropertyFlags memory_flags;uint8_t *cpu_map;int cpu_map_tried;} vkpool_t;
static pthread_mutex_t g_sub_lock=PTHREAD_MUTEX_INITIALIZER,g_vk_queue_mu=PTHREAD_MUTEX_INITIALIZER;
static pthread_rwlock_t g_pool_mu=PTHREAD_RWLOCK_INITIALIZER;
static vkpool_t g_pool[4];static int g_npool=2,g_cur=-1;
static uint64_t g_fence,completed,g_queue_idle_serial,g_queue_serial;
static struct {int dev,q;struct {uint32_t memoryTypeCount;struct {uint32_t propertyFlags;} memoryTypes[5];} mem;} V={7,1,{5,{{1},{1},{1},{0xb},{7}}}};
static int idle_calls,idle_result,prepare_calls,map_calls,map_result;
static uint8_t src[256],dst[256],other_map[256];
static void *map_address=dst,*expected_guest=dst;
static uint64_t vk_fence_done(void){return completed;}
static VkResult vkQueueWaitIdle(int q){assert(q==1 && pthread_mutex_trylock(&g_vk_queue_mu));idle_calls++;return idle_result;}
void dm_guest_write_prepare(void *p,size_t n){assert(p==expected_guest && n==64 && g_pool[1].cpu_map && pthread_mutex_trylock(&g_sub_lock));prepare_calls++;}
static VkResult vkMapMemory(int dev,int mem,uint64_t offset,uint64_t size,uint32_t flags,void **mapped){
    assert(dev==V.dev && mem==g_pool[1].mem && !offset && size==VK_WHOLE_SIZE && !flags);
    assert((g_pool[1].memory_flags&6)==6 && pthread_mutex_trylock(&g_sub_lock) && pthread_mutex_trylock(&g_vk_queue_mu));
    map_calls++;if(map_result)return map_result;*mapped=map_address;return 0;
}
static void log_line(const char *f,...){(void)f;}
#define VKLOG log_line
'''+mem_type+choose_type+upload+r'''
static void record(uint8_t *r,uint64_t s,uint64_t d,uint64_t n) {
    op_hdr h={OP_COPY_B2B,32},e={OP_END,8};memcpy(r,&h,8);memcpy(r+8,&s,8);memcpy(r+16,&d,8);memcpy(r+24,&n,8);memcpy(r+32,&e,8);
}
static void clear_dst(void){memset(dst,0,sizeof dst);}
static void assert_unlocked(void){
    assert(!pthread_mutex_trylock(&g_sub_lock));pthread_mutex_unlock(&g_sub_lock);
    assert(!pthread_rwlock_trywrlock(&g_pool_mu));pthread_rwlock_unlock(&g_pool_mu);
    assert(!pthread_mutex_trylock(&g_vk_queue_mu));pthread_mutex_unlock(&g_vk_queue_mu);
}
static void fresh_allocation(void *mapped){g_pool[1].cpu_map=NULL;g_pool[1].cpu_map_tried=0;g_pool[1].mem++;map_address=mapped;}
int main(int argc,char **argv){
    assert(argc==2);int mode=atoi(argv[1]);setenv("DM_VK_CPU_SMALL_UPLOADS",argv[1],1);
    assert(cpu_small_upload_memory_type(0x12)==(mode==1 || mode==2 ? 4u : 1u));
    assert(cpu_small_upload_memory_type(0x2)==1); // no coherent compatible type: retain original import type
    assert(cpu_small_upload_memory_type(0)==UINT32_MAX);
    for(int i=0;i<256;i++)src[i]=(uint8_t)(i+1);
    g_pool[0]=(vkpool_t){.lo=(uint64_t)src,.hi=(uint64_t)src+256,.base=(uint64_t)src,.end=(uint64_t)src+256,.shadow=(void *)1,.flags=2,.mem=1};
    g_pool[1]=(vkpool_t){.lo=(uint64_t)dst,.hi=(uint64_t)dst+256,.base=(uint64_t)dst,.end=(uint64_t)dst+256,.zero_copy=1,.flags=1,.mem=2,.memory_type=4,.memory_flags=7};
    uint8_t r[48]={0};record(r,(uint64_t)src,(uint64_t)dst,64);uint64_t h[2]={(uint64_t)r,(uint64_t)r};
    g_queue_serial=1;
    if(mode!=1){assert(!mtl_cpu_upload(h,1) && !prepare_calls && !idle_calls && !g_fence && !map_calls && !g_pool[1].cpu_map_tried);assert_unlocked();return 0;}
    assert(!mtl_cpu_upload(h,2)); // multiple records may carry ordering or event effects
    op_hdr event={OP_FENCE_SYNC,16};memcpy(r+32,&event,8);assert(!mtl_cpu_upload(h,1));record(r,(uint64_t)src,(uint64_t)dst,64);
    g_pool[2]=(vkpool_t){.lo=(uint64_t)src+16,.hi=(uint64_t)src+32};g_npool=3;
    assert(!mtl_cpu_upload(h,1) && !prepare_calls); // partial alias, not just a second covering pool
    g_pool[2].lo=(uint64_t)dst+16;g_pool[2].hi=(uint64_t)dst+32;
    assert(!mtl_cpu_upload(h,1) && !prepare_calls);g_npool=2;
    g_fence=10;completed=9;assert(!mtl_cpu_upload(h,1) && !prepare_calls && !idle_calls);completed=10;
    pthread_mutex_lock(&g_sub_lock);assert(!mtl_cpu_upload(h,1) && !prepare_calls);pthread_mutex_unlock(&g_sub_lock);
    g_cur=0;assert(!mtl_cpu_upload(h,1));g_cur=-1;
    g_pool[0].flags=4;assert(!mtl_cpu_upload(h,1));g_pool[0].flags=2;
    g_pool[1].zero_copy=0;assert(!mtl_cpu_upload(h,1));g_pool[1].zero_copy=1;
    for(uint32_t flags=1;flags<=5;flags+=2){g_pool[1].memory_flags=flags;assert(!mtl_cpu_upload(h,1) && !map_calls && !idle_calls);}
    g_pool[1].memory_flags=7;
    record(r,UINT64_MAX-31,(uint64_t)dst,64);assert(!mtl_cpu_upload(h,1));
    record(r,(uint64_t)src,(uint64_t)src+16,64);assert(!mtl_cpu_upload(h,1));
    record(r,(uint64_t)src,(uint64_t)dst,65537);assert(!mtl_cpu_upload(h,1));
    record(r,(uint64_t)src,(uint64_t)dst,0);assert(!mtl_cpu_upload(h,1));
    record(r,(uint64_t)src,(uint64_t)dst,64);
    idle_result=-1;assert(!mtl_cpu_upload(h,1) && !prepare_calls && !g_queue_idle_serial && !map_calls);
    idle_result=0;map_result=-1;assert(!mtl_cpu_upload(h,1) && map_calls==1 && !prepare_calls && !g_pool[1].cpu_map);
    map_result=0;g_queue_serial++;assert(!mtl_cpu_upload(h,1) && map_calls==1 && !prepare_calls && idle_calls==2);assert_unlocked(); // failed pool neither retries nor idles the queue
    fresh_allocation(dst);assert(mtl_cpu_upload(h,1) && prepare_calls==1 && idle_calls==3 && map_calls==2 && g_fence==10);
    assert(!memcmp(src,dst,64) && g_queue_idle_serial==g_queue_serial && !g_pool[1].shadow);
    clear_dst();assert(mtl_cpu_upload(h,1) && idle_calls==3 && map_calls==2 && !memcmp(src,dst,64));
    g_queue_serial++;clear_dst();assert(mtl_cpu_upload(h,1) && idle_calls==4 && map_calls==2 && !memcmp(src,dst,64));assert_unlocked();
    g_pool[0].shadow=NULL;g_pool[0].lazy=1;assert(mtl_cpu_upload(h,1) && !memcmp(src,dst,64));g_pool[0].shadow=(void *)1;g_pool[0].lazy=0;
    int before_prepare=prepare_calls,before_map=map_calls;
    fresh_allocation(NULL);assert(!mtl_cpu_upload(h,1) && map_calls==before_map+1 && prepare_calls==before_prepare);
    assert(!mtl_cpu_upload(h,1) && map_calls==before_map+1);assert_unlocked();
    expected_guest=dst+16;record(r,(uint64_t)src,(uint64_t)dst+16,64);
    fresh_allocation((void *)(UINTPTR_MAX-7));before_prepare=prepare_calls;before_map=map_calls;
    assert(!mtl_cpu_upload(h,1) && prepare_calls==before_prepare && map_calls==before_map+1);
    assert(!mtl_cpu_upload(h,1) && map_calls==before_map+1);assert_unlocked();
    fresh_allocation(other_map);memset(other_map,0,sizeof other_map);clear_dst();
    assert(mtl_cpu_upload(h,1) && !memcmp(src,other_map+16,64));
    for(int i=0;i<256;i++){assert(!dst[i]);if(i<16 || i>=80)assert(!other_map[i]);}
    before_map=map_calls;assert(mtl_cpu_upload(h,1) && map_calls==before_map && !g_pool[1].shadow);
    assert(g_fence==10);assert_unlocked();puts("guarded upload passed");return 0;
}
'''
        with tempfile.TemporaryDirectory(prefix='nx2a-small-upload-') as tmp:
            src, exe = Path(tmp) / 'check.c', Path(tmp) / 'check'
            src.write_text(code)
            result = subprocess.run(['clang', '-std=gnu11', '-O2', '-Wall', '-Wextra', '-Werror', '-pthread', '-I'+str(DM/'android'), '-I'+str(DM/'gfx/nvn/include'), str(src), '-o', str(exe)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            for mode in ('0', '2', '1', '3'):
                result = subprocess.run([str(exe), mode], capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
