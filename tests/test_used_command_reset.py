"""Exercise actual retirement after high-water allocation and smaller reuse."""
from pathlib import Path
import os,shutil,subprocess,tempfile,unittest
from tests.test_texture_barrier import function
ROOT=Path(__file__).resolve().parents[1]
DM=ROOT/'nxnative/dm'
@unittest.skipUnless(os.name!='nt' and shutil.which('clang'),'POSIX clang needed')
class UsedCommandReset(unittest.TestCase):
    def test_actual_reset_prefix_preserves_publication_reuse_and_failure(self):
        source=(DM/'android/gfx_vk_core.c').read_text()
        main=function(source,'completion_main')
        self.assertLess(main.index('pools_copy_back('),main.index('completion_retire(c)'))
        self.assertLess(main.index('for (int i = 0; i < c->n;'),main.index('completion_retire(c)'))
        code=r'''
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan_core.h>
#include <assert.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "scoped_sync.h"
typedef struct {uint64_t fence;int nmain_alloc,nmain_used,nup_alloc,nup_used;VkCommandBuffer mains[8],ups[8];} vksub_t;
typedef struct {uint64_t fence;int rslot;} cpend;
static vksub_t g_slot[2];static int g_prereset=1,g_early_complete,flip,fail;
static uint64_t done,ready,stats[VK_EFF_N];static int calls,publishes;static unsigned state[32];
static void efficiency_add(int k,uint64_t n){stats[k]+=n;}
static uint64_t sync_clock(void){return 0;}
#define SP_RESET 6
static void sync_add(int stage,uint64_t begin){assert(stage==SP_RESET && !begin);}
static void completion_publish(uint64_t fence,int complete,int reuse){
 assert(fence>ready);if(complete)done=fence;if(reuse)ready=fence;publishes++;
}
#define VKCHK(e) do {if((e)!=VK_SUCCESS)exit(77);} while(0)
static VkResult vkResetCommandBuffer(VkCommandBuffer cb,VkCommandBufferResetFlags flags){
 unsigned id=(unsigned)(uintptr_t)cb;assert(id && id<32 && !flags && ready<g_slot[0].fence);
 assert(done==(g_early_complete || flip ? g_slot[0].fence : g_slot[0].fence-1));
 calls++;if(fail)return VK_ERROR_DEVICE_LOST;state[id]=0;if(flip)g_early_complete=0;return VK_SUCCESS;
}
'''+function(source,'completion_retire')+r'''
static void setup(uint64_t fence,int ma,int mu,int ua,int uu){
 vksub_t *s=&g_slot[0];s->fence=fence;s->nmain_alloc=ma;s->nmain_used=mu;s->nup_alloc=ua;s->nup_used=uu;
 for(int i=0;i<ma;i++)s->mains[i]=(VkCommandBuffer)(uintptr_t)(i+1);
 for(int i=0;i<ua;i++)s->ups[i]=(VkCommandBuffer)(uintptr_t)(i+17);
 for(int i=0;i<mu;i++)state[i+1]=2;
 for(int i=0;i<uu;i++)state[i+17]=1; // includes a recorded-but-refused trailing upload
}
int main(int argc,char **argv){
 assert(argc==2);if(!strcmp(argv[1],"fail")){fail=1;g_early_complete=1;setup(1,2,2,2,2);cpend c={1,0};completion_retire(&c);return 1;}
 for(int mode=0;mode<2;mode++){
  done=ready=0;g_early_complete=mode;flip=0;calls=publishes=0;memset(state,0,sizeof state);memset(stats,0,sizeof stats);
  setup(1,6,6,4,4);cpend c={1,0};completion_retire(&c);assert(done==1 && ready==1 && calls==10 && publishes==(mode?2:1));
  // The same pool keeps its high-water allocation, then records a small submission.
  setup(2,6,1,4,1);c.fence=2;completion_retire(&c);assert(done==2 && ready==2 && calls==12);
  assert(stats[VK_EFF_RESETS]==12 && stats[VK_EFF_UNUSED_RESETS]==8);
  for(int i=1;i<32;i++)assert(!state[i]);
  // Disabled pre-reset leaves reset/reuse policy unchanged; no driver call.
  setup(3,6,2,4,2);c.fence=3;g_prereset=0;int old=calls;completion_retire(&c);assert(done==3 && ready==3 && calls==old);g_prereset=1;
 }
 // The early flag is sampled once, including a live mode change during reset.
 done=ready=0;g_early_complete=1;flip=1;setup(1,2,1,2,1);cpend c={1,0};completion_retire(&c);assert(done==1 && ready==1);
 return 0;
}
'''
        with tempfile.TemporaryDirectory(prefix='nx2a-used-reset-') as tmp:
            src,exe=Path(tmp)/'check.c',Path(tmp)/'check';src.write_text(code)
            p=subprocess.run(['clang','-std=gnu11','-O2','-Wall','-Wextra','-Werror','-I'+str(DM/'android'),'-I'+str(DM/'third_party/vulkan-headers/include'),str(src),'-o',str(exe)],capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stderr)
            for arg,expected in [('normal',0),('fail',77)]:
                p=subprocess.run([str(exe),arg],capture_output=True,text=True,timeout=10)
                self.assertEqual(p.returncode,expected,p.stdout+p.stderr)
