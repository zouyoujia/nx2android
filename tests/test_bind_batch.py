"Verify within-draw command coalescing against the unbatched final Vulkan state."
from tests import test_gpu_segments as gs
import unittest

class BindBatch(unittest.TestCase):
    compile_run=gs.GpuSegments.compile_run

    def test_stencil_pairs_preserve_both_faces_and_reemit_every_draw(self):
        self.compile_run(r'''
#define VK_NO_PROTOTYPES
#include <assert.h>
#include <stdint.h>
#include "bind_batch.h"
static uint32_t state[2];static unsigned calls;
static void set(VkCommandBuffer cb,VkStencilFaceFlags faces,uint32_t value){assert(cb==(VkCommandBuffer)1);calls++;if(faces&VK_STENCIL_FACE_FRONT_BIT)state[0]=value;if(faces&VK_STENCIL_FACE_BACK_BIT)state[1]=value;}
int main(void){
 for(unsigned i=0;i<128;i++)for(unsigned equal=0;equal<2;equal++){
  uint32_t f=i*0x12345u,b=equal?f:~f;calls=0;
  assert(vk_stencil_pair((VkCommandBuffer)1,set,f,b,0)==2 && calls==2 && state[0]==f && state[1]==b);
  calls=0;assert(vk_stencil_pair((VkCommandBuffer)1,set,f,b,1)==(equal?1:2));assert(calls==(equal?1u:2u) && state[0]==f && state[1]==b);
  unsigned old=calls;vk_stencil_pair((VkCommandBuffer)1,set,f,b,1);assert(calls==2*old); // complete state again, no cross-draw omission
 }
 return 0;
}
''')

    def test_vertex_runs_preserve_holes_offsets_and_full_binding(self):
        self.compile_run(r'''
#define VK_NO_PROTOTYPES
#include <assert.h>
#include <stdint.h>
#include <string.h>
#include "bind_batch.h"
static VkBuffer state[17];static VkDeviceSize pos[17];static unsigned calls,seen;
static void bind(VkCommandBuffer cb,uint32_t first,uint32_t n,const VkBuffer *b,const VkDeviceSize *o){
 assert(cb==(VkCommandBuffer)1 && n && first+n<=17);calls++;
 for(unsigned i=0;i<n;i++){assert(!(seen&(1u<<(first+i))));seen|=1u<<(first+i);state[first+i]=b[i];pos[first+i]=o[i];}
}
int main(void){
 VkBuffer b[17];VkDeviceSize o[17];for(unsigned i=0;i<17;i++){b[i]=(VkBuffer)(uintptr_t)(100+i);o[i]=37*i;}
 unsigned masks[]={0,1,0x1ffff,0x15555,0x10017,0x1c0,0x11001};
 for(unsigned k=0;k<sizeof masks/sizeof *masks;k++){
  uint32_t m=masks[k];for(unsigned i=0;i<17;i++){state[i]=(VkBuffer)(uintptr_t)55;pos[i]=777;}
  calls=seen=0;unsigned actual=vk_vertex_runs((VkCommandBuffer)1,m,b,o,bind);assert(actual==calls && seen==m);
  assert(actual<=(unsigned)__builtin_popcount(m));if(m==0x1ffff)assert(actual==1);if(m==0)assert(!calls);
  for(unsigned i=0;i<17;i++){assert(state[i]==((m&(1u<<i))?b[i]:(VkBuffer)(uintptr_t)55));assert(pos[i]==((m&(1u<<i))?o[i]:777));}
  unsigned prior=calls;seen=0;vk_vertex_runs((VkCommandBuffer)1,m,b,o,bind);assert(calls==2*prior && seen==m);
 }
 return 0;
}
''')
