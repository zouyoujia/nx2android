"""Execute production lazy-pool functions with host Vulkan and completion mocks."""
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
        char = source[pos]
        depth += (char == '{') - (char == '}')
        pos += 1
    return source[start:pos] + '\n'


@unittest.skipUnless(os.name != 'nt' and shutil.which('clang'), 'POSIX host clang needed')
class LazySmallPools(unittest.TestCase):
    def test_actual_registration_materialization_and_lifetime(self):
        source = (DM / 'android/gfx_vk_core.c').read_text()
        header = (DM / 'android/gfx_vk.h').read_text()
        pool_type = re.search(r'typedef struct \{[^\n]+\} vkpool_t;', header).group()
        stats = re.search(r'static struct \{[^\n]+\} g_pool_stats;', source).group()
        # Execute the actual begin sequence through the old-fence wait and pool sync.
        # The remaining query/command execution is unrelated to pool preparation.
        begin = function(source, 'mtl_sub_begin')
        prefix = begin[begin.index('{') + 1:begin.index('    if (g_tsq)')]
        begin_test = 'static void tested_begin(void) {' + prefix + '\n    g_cur = -1; pthread_mutex_unlock(&g_sub_lock);\n}\n'
        names = ('lazy_small_pools', 'zc_over', 'zc_under', 'vk_mem_type',
                 'pool_allocate_copy', 'pool_add_one', 'pool_add_impl',
                 'mtl_pool_add', 'mtl_pool_add_flags', 'pool_prepare_submission',
                 'mtl_pool_del', 'vk_pool_remap_ro', 'vk_pools_sync_')
        implementations = '\n'.join(function(source, name) for name in names)
        code = r'''
#define _GNU_SOURCE
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan_core.h>
#include <assert.h>
#include <errno.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>
#include "cpu_small_upload.h"
#ifndef SOCK_CLOEXEC
#define SOCK_CLOEXEC 0
#endif
'''+pool_type+r'''
#define MAXPOOL 16
#define ZC_CHUNK 0x10000000ull
#define NSLOT 6
#define POOL_USAGE (VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT)
static vkpool_t g_pool[MAXPOOL]; static int g_npool;
static pthread_rwlock_t g_pool_mu = PTHREAD_RWLOCK_INITIALIZER;
static pthread_mutex_t g_sub_lock = PTHREAD_MUTEX_INITIALIZER;
static uint32_t g_pool_gen;
'''+stats+r'''
static struct {VkDevice dev; VkPhysicalDeviceMemoryProperties mem;} V;
static uint64_t g_sync_bytes, g_sync_full;
static int create_calls, allocate_calls, map_calls, defer_calls, wait_calls, protect_calls;
static int zero_copy_success, zero_copy_calls, zero_copy_back, ahb_releases;
static size_t buffer_size[128];
static struct {size_t size; uint8_t *bytes;} memories[128];
static uint8_t *guest_a,*guest_b;
static void log_line(const char *format,...){(void)format;}
#define VKLOG log_line
#define VKCHK(call) do { VkResult status=(call); assert(status==VK_SUCCESS); } while(0)
static _Noreturn void die(const char *format,...){(void)format;abort();}
static void vk_init(void){}
static void pool_ord_make(void){}
static void pool_profile_report(void){}
static VkResult vkCreateBuffer(VkDevice device,const VkBufferCreateInfo *info,const VkAllocationCallbacks *a,VkBuffer *out){
    (void)a;assert(device==V.dev && create_calls<127);*out=(VkBuffer)(uintptr_t)++create_calls;buffer_size[create_calls]=info->size;return VK_SUCCESS;
}
static void vkGetBufferMemoryRequirements(VkDevice device,VkBuffer buffer,VkMemoryRequirements *out){
    assert(device==V.dev);*out=(VkMemoryRequirements){buffer_size[(uintptr_t)buffer],1,1};
}
static VkResult vkAllocateMemory(VkDevice device,const VkMemoryAllocateInfo *info,const VkAllocationCallbacks *a,VkDeviceMemory *out){
    (void)a;assert(device==V.dev && info->memoryTypeIndex==0 && allocate_calls<127);int id=++allocate_calls;
    memories[id].size=info->allocationSize;memories[id].bytes=calloc(1,info->allocationSize);assert(memories[id].bytes);
    *out=(VkDeviceMemory)(uintptr_t)id;return VK_SUCCESS;
}
static VkResult vkBindBufferMemory(VkDevice device,VkBuffer buffer,VkDeviceMemory memory,VkDeviceSize offset){
    assert(device==V.dev && buffer && memory && !offset);return VK_SUCCESS;
}
static VkResult vkMapMemory(VkDevice device,VkDeviceMemory memory,VkDeviceSize offset,VkDeviceSize size,VkMemoryMapFlags flags,void **out){
    assert(device==V.dev && !offset && size==VK_WHOLE_SIZE && !flags);map_calls++;*out=memories[(uintptr_t)memory].bytes;return VK_SUCCESS;
}
static int pool_zero_copy(vkpool_t *p,uint8_t *mem,uint64_t size){
    (void)mem;(void)size;zero_copy_calls++;
    if(!zero_copy_success){p->buf=(VkBuffer)(uintptr_t)127;return -2;} // failed AHB path may leave a destroyed handle
    p->zero_copy=1;p->buf=(VkBuffer)(uintptr_t)127;p->ahb=(void *)(uintptr_t)1;return 0;
}
static void pool_copy_back(vkpool_t *p){assert(p->zero_copy);zero_copy_back++;}
static void vk_defer_free_buffer(VkBuffer buffer,VkDeviceMemory memory){
    assert(buffer || memory);defer_calls++;wait_calls++; // outside a submission, old deferred-free waits
    if(memory){free(memories[(uintptr_t)memory].bytes);memories[(uintptr_t)memory].bytes=NULL;}
}
static void AHardwareBuffer_release(void *buffer){assert(buffer);ahb_releases++;}
static int AHardwareBuffer_sendHandleToUnixSocket(void *buffer,int socket){(void)buffer;(void)socket;return -1;}
static int recv_fds(int socket,int *fds,int count){(void)socket;(void)fds;(void)count;return -1;}
static uint64_t nvn_dirty_now(void){return 41;}
static const void *nvn_dirty_span(const void *address,size_t size){assert(address && size);return address;}
static void nvn_dirty_protect(const void *address,size_t size){assert(address && size);protect_calls++;}
static int nvn_dirty_since_span(const void *span,const void *address,size_t size,uint64_t stamp){
    assert(span && address && size);(void)stamp;return 0;
}
'''+implementations+r'''
static int g_cur=-1,g_shadow_pools;static uint64_t g_fence=17,g_cmd_gen;
typedef struct {int npost,up_used,ngw,nmain_used,cb_err,nup_used,nmain_sent;VkCommandBuffer main,up,mains[1],ups[1];} vksub_t;
static vksub_t g_slot[NSLOT];
static void slot_recycle(vksub_t *slot){assert(slot);}
static VkResult vkBeginCommandBuffer(VkCommandBuffer buffer,const VkCommandBufferBeginInfo *info){assert(buffer && info);return VK_SUCCESS;}
static int mtl_wait(uint64_t fence,uint64_t timeout){
    assert(fence==g_fence && timeout==~0ull && pthread_mutex_trylock(&g_sub_lock));
    // Completion copyback must be able to obtain the pool lock at this wait.
    assert(!pthread_rwlock_trywrlock(&g_pool_mu));pthread_rwlock_unlock(&g_pool_mu);wait_calls++;return 1;
}
static void vk_pools_sync(VkCommandBuffer buffer){vk_pools_sync_(buffer);}
'''+begin_test+r'''
static void pool_locks_free(void){
    assert(!pthread_rwlock_trywrlock(&g_pool_mu));pthread_rwlock_unlock(&g_pool_mu);
    assert(!pthread_mutex_trylock(&g_sub_lock));pthread_mutex_unlock(&g_sub_lock);
}
static void cleanup(void){while(g_npool)mtl_pool_del((void *)(uintptr_t)g_pool[0].base);pool_locks_free();}
static pthread_mutex_t worker_mu=PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t worker_cv=PTHREAD_COND_INITIALIZER;
static int worker_started,worker_done;
static void *register_worker(void *arg){
    pthread_mutex_lock(&worker_mu);worker_started=1;pthread_cond_broadcast(&worker_cv);pthread_mutex_unlock(&worker_mu);
    mtl_pool_add_flags(arg,4096,2);
    pthread_mutex_lock(&worker_mu);worker_done=1;pthread_cond_broadcast(&worker_cv);pthread_mutex_unlock(&worker_mu);return NULL;
}
int main(int argc,char **argv){
    (void)cpu_small_upload_record;(void)cpu_small_upload_pools; // covered by the small-copy tests
    assert(argc==4);int lazy=atoi(argv[1]),cpu=atoi(argv[2]),minimum=atoi(argv[3]);
    setenv("DM_VK_LAZY_SMALL_POOLS",argv[1],1);setenv("DM_VK_CPU_SMALL_UPLOADS",argv[2],1);setenv("DM_VK_ZEROCOPY_MIN_KB",argv[3],1);
    unsetenv("DM_VK_NO_ZEROCOPY");unsetenv("DM_VK_ZEROCOPY_MAX_MB");unsetenv("DM_VK_POOL_PROF");
    V.dev=(VkDevice)(uintptr_t)1;V.mem.memoryTypeCount=1;V.mem.memoryTypes[0].propertyFlags=VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    assert(!posix_memalign((void **)&guest_a,4096,131072));assert(!posix_memalign((void **)&guest_b,4096,131072));
    memset(guest_a,0x13,131072);memset(guest_b,0x47,131072);
    for(int i=0;i<NSLOT;i++){g_slot[i].mains[0]=(VkCommandBuffer)(uintptr_t)1;g_slot[i].ups[0]=(VkCommandBuffer)(uintptr_t)2;}
    int enabled=lazy>0 && cpu==1;
    if(!minimum){ // A pool using the zero-copy policy must never be lazily converted to copied backing.
        zero_copy_success=1;mtl_pool_add_flags(guest_a,4096,2);
        assert(g_npool==1 && g_pool[0].zero_copy && !g_pool[0].lazy && !g_pool[0].shadow && !create_calls);
        cleanup();assert(zero_copy_calls==1 && zero_copy_back==1 && ahb_releases==1);
        zero_copy_success=0;mtl_pool_add_flags(guest_a,4096,2);
        assert(zero_copy_calls==2 && !g_pool[0].lazy && g_pool[0].shadow && create_calls==1);
        assert(g_pool[0].buf==(VkBuffer)(uintptr_t)1 && !g_pool_stats.deferred); // stale failed-AHB handle was replaced eagerly
        cleanup();free(guest_a);free(guest_b);return 0;
    }
    mtl_pool_add_flags(guest_a,4096,2);assert(g_npool==1 && g_pool[0].flags==2);
    if(!enabled){assert(!g_pool[0].lazy && g_pool[0].buf && g_pool[0].shadow && create_calls==1);cleanup();free(guest_a);free(guest_b);return 0;}
    assert(g_pool[0].lazy && !g_pool[0].buf && !g_pool[0].mem && !g_pool[0].shadow && !create_calls && !allocate_calls && !map_calls && !wait_calls);
    assert(!vk_pool_remap_ro(guest_a,4096)); // anonymous metadata backing needs no remap
    mtl_pool_del(guest_a);assert(!g_npool && !create_calls && !defer_calls && !wait_calls);pool_locks_free();
    // A real pthread holding backend ownership prevents candidate registration.
    pthread_mutex_lock(&g_sub_lock);pthread_t thread;assert(!pthread_create(&thread,NULL,register_worker,guest_a));
    pthread_mutex_lock(&worker_mu);while(!worker_started)pthread_cond_wait(&worker_cv,&worker_mu);
    struct timespec deadline;clock_gettime(CLOCK_REALTIME,&deadline);deadline.tv_nsec+=100000000;
    if(deadline.tv_nsec>=1000000000){deadline.tv_nsec-=1000000000;deadline.tv_sec++;}
    int timed=0;while(!worker_done && !timed)timed=pthread_cond_timedwait(&worker_cv,&worker_mu,&deadline);
    assert(!worker_done && timed==ETIMEDOUT);pthread_mutex_unlock(&worker_mu);
    pthread_rwlock_rdlock(&g_pool_mu);assert(!g_npool);pthread_rwlock_unlock(&g_pool_mu);
    pthread_mutex_unlock(&g_sub_lock);assert(!pthread_join(thread,NULL));assert(g_npool==1 && g_pool[0].lazy && !create_calls);
    mtl_pool_add_flags(guest_b,8192,2);assert(g_npool==2 && g_pool[1].lazy && !create_calls);
    pthread_mutex_lock(&g_sub_lock);assert(pool_prepare_submission());pthread_mutex_unlock(&g_sub_lock);pool_locks_free();
    assert(create_calls==2 && allocate_calls==2 && map_calls==2);
    for(int i=0;i<2;i++)assert(!g_pool[i].lazy && g_pool[i].buf && g_pool[i].shadow && !g_pool[i].synced && !g_pool[i].span);
    memset(guest_a,0x29,4096);memset(guest_b,0x68,8192); // latest guest bytes, including writes before dirty protection
    tested_begin();assert(wait_calls==1 && g_shadow_pools && g_sync_full==2 && g_sync_bytes==12288 && protect_calls==2);
    assert(!memcmp(g_pool[0].shadow,guest_a,4096) && !memcmp(g_pool[1].shadow,guest_b,8192));
    tested_begin();assert(create_calls==2 && allocate_calls==2 && map_calls==2 && g_sync_full==2);pool_locks_free();cleanup();
    // Non-staging, unknown flags, and the exact 64 KiB boundary remain eager.
    int before=create_calls;mtl_pool_add_flags(guest_a,4096,0);assert(!g_pool[0].lazy && g_pool[0].shadow && create_calls==before+1);cleanup();
    before=create_calls;mtl_pool_add_flags(guest_a,4096,4);assert(!g_pool[0].lazy && g_pool[0].shadow && create_calls==before+1);cleanup();
    before=create_calls;mtl_pool_add(guest_a,4096);assert(!g_pool[0].lazy && g_pool[0].shadow && create_calls==before+1);cleanup();
    before=create_calls;mtl_pool_add_flags(guest_a,65536,2);assert(!g_pool[0].lazy && g_pool[0].shadow && create_calls==before+1);cleanup();
    before=create_calls;mtl_pool_add_flags(guest_a+1,4096,2);assert(!g_pool[0].lazy && g_pool[0].shadow && create_calls==before+1);cleanup(); // alignment fallback is not the explicit small-copy branch
    free(guest_a);free(guest_b);puts("lazy pool lifecycle passed");return 0;
}
'''
        with tempfile.TemporaryDirectory(prefix='nx2a-lazy-pools-') as tmp:
            src, exe = Path(tmp) / 'check.c', Path(tmp) / 'check'
            src.write_text(code)
            result = subprocess.run(['clang', '-std=gnu11', '-O2', '-Wall', '-Wextra', '-Werror', '-Wno-missing-field-initializers', '-pthread',
                                     '-I'+str(DM/'third_party/vulkan-headers/include'), '-I'+str(DM/'android'),
                                     '-I'+str(DM/'gfx/nvn/include'), str(src), '-o', str(exe)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            for lazy, cpu, minimum in (('1', '1', '64'), ('0', '1', '64'), ('1', '0', '64'), ('1', '2', '64'), ('1', '1', '0')):
                with self.subTest(lazy=lazy, cpu=cpu, minimum=minimum):
                    result = subprocess.run([str(exe), lazy, cpu, minimum], capture_output=True, text=True, timeout=10)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
