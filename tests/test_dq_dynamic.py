"Exercise the production dynamic packet through the concurrent FIFO."
from pathlib import Path
import os,re,shutil,subprocess,tempfile,unittest
from tests import test_dq_live as dq
from tests import test_gpu_segments as gs

DM=dq.DM

def program(tail):
    source=(DM/'android/gfx_vk_dq.c').read_text()
    header=(DM/'android/gfx_vk.h').read_text()
    state=re.search(r'typedef struct \{ VkViewport vp;.*?\} vk_dq_dynamic_state;',header,re.S).group()
    extra=r'''
#include "bind_batch.h"
static unsigned nc;static uint32_t code[128];static vk_dq_dynamic_state observed;
static void note(unsigned op,VkCommandBuffer cb){assert(cb==(VkCommandBuffer)1 && nc<128);code[nc++]=op;}
static void viewport(VkCommandBuffer cb,uint32_t first,uint32_t n,const VkViewport *p){note(1,cb);assert(!first && n==1);observed.vp=*p;}
static void scissor(VkCommandBuffer cb,uint32_t first,uint32_t n,const VkRect2D *p){note(2,cb);assert(!first && n==1);observed.sc=*p;}
static void blend(VkCommandBuffer cb,const float *p){note(3,cb);memcpy(observed.blend,p,sizeof observed.blend);}
static void stencil(VkCommandBuffer cb,VkStencilFaceFlags f,uint32_t v,unsigned op,uint32_t *s){note(op*100+f,cb);if(f&1)s[0]=v;if(f&2)s[1]=v;}
static void sm(VkCommandBuffer cb,VkStencilFaceFlags f,uint32_t v){stencil(cb,f,v,4,observed.sm);}
static void sw(VkCommandBuffer cb,VkStencilFaceFlags f,uint32_t v){stencil(cb,f,v,5,observed.sw);}
static void sr(VkCommandBuffer cb,VkStencilFaceFlags f,uint32_t v){stencil(cb,f,v,6,observed.sr);}
static void bias(VkCommandBuffer cb,float b,float c,float s){note(7,cb);observed.bias=b;observed.clamp=c;observed.slope=s;}
static void line(VkCommandBuffer cb,float w){note(8,cb);assert(w==1);}
static PFN_vkCmdSetViewport r_vkCmdSetViewport=viewport;
static PFN_vkCmdSetScissor r_vkCmdSetScissor=scissor;
static PFN_vkCmdSetBlendConstants r_vkCmdSetBlendConstants=blend;
static PFN_vkCmdSetStencilCompareMask r_vkCmdSetStencilCompareMask=sm;
static PFN_vkCmdSetStencilWriteMask r_vkCmdSetStencilWriteMask=sw;
static PFN_vkCmdSetStencilReference r_vkCmdSetStencilReference=sr;
static PFN_vkCmdSetDepthBias r_vkCmdSetDepthBias=bias;
static PFN_vkCmdSetLineWidth r_vkCmdSetLineWidth=line;
'''
    # Insert after FIFO helpers, before its dispatch function. Its blocking bind
    # gives a deterministic consumer barrier to test caller-owned stack lifetime.
    base=dq.program(tail).replace('OP_WRAP,OP_vkCmdBindVertexBuffers','OP_WRAP,OP_DYNAMIC_STATE,OP_vkCmdBindVertexBuffers')
    marker='static void run(const rec_hdr *h){'
    helper=state+extra+'\n'+gs.function(source,'vk_dq_dynamic')+'\n'+gs.function(source,'run_dynamic')+'\n'
    return base.replace(marker,helper+marker+'\n if(h->op==OP_DYNAMIC_STATE){run_dynamic(h->cb,(const void*)(h+1));return;}')

@unittest.skipUnless(os.name!='nt' and shutil.which('clang'),'POSIX clang needed')
class DqDynamic(unittest.TestCase):
    def test_owned_packet_wraparound_bypass_and_complete_state(self):
        tail=r'''
int main(void){
 setup();for(unsigned i=0;i<256;i++)atomic_store(&allowed[i],1);
 vk_dq_dynamic_state a={.vp={2,3,1920,-1080,.2f,.9f},.sc={{7,9},{1500,800}},.blend={.1f,.2f,.3f,.4f},
  .sm={0x12345678,0x12345678},.sw={0xaabbccdd,0x11223344},.sr={3,5},.bias=-1,.clamp=2,.slope=3,.stencil_batch=1};
 assert(!vk_dq_dynamic((VkCommandBuffer)1,&a)); // disabled: caller uses original wrappers
 setenv("DM_VK_MT_STATE_BATCH","1",1);setenv("DM_VK_MT_SPACE_PROF","1",1);vk_dq_refresh();
 for(unsigned k=0;k<4;k++){unsigned batch=k&1;
  a.stencil_batch=batch;vk_dq_dynamic_state want=a;atomic_store(&allowed[100],0);one(100);WAIT(atomic_load(&started[100]));
  uint64_t n=g_dq_recs;assert(vk_dq_dynamic((VkCommandBuffer)1,&a));assert(g_dq_recs==n+1);memset(&a,0xef,sizeof a);
  atomic_store(&allowed[100],1);vk_dq_drain();assert(nc==(batch?10:11));
  assert(code[0]==1 && code[1]==2 && code[2]==3 && code[nc-2]==7 && code[nc-1]==8);
  assert(code[3]==(batch?403u:401u));unsigned j=batch?4:5;assert(code[j]==501 && code[j+1]==502 && code[j+2]==601 && code[j+3]==602);
  want.stencil_batch=0;assert(!memcmp(&observed,&want,sizeof want));a=want;nc=0;memset(&observed,0,sizeof observed);atomic_store(&started[100],0);
 }
 assert(g_w>RING && g_space_stats[DQ_SPACE_PACKETS]==4 && g_space_stats[DQ_SPACE_STATE_CALLS]==42); // packet wrap was consumed safely
 setenv("DM_VK_MT_RECORD","0",1);vk_dq_refresh();uint64_t before=g_w;assert(!vk_dq_dynamic((VkCommandBuffer)1,&a) && g_w==before);
 setenv("DM_VK_MT_RECORD","1",1);vk_dq_refresh();setenv("DM_VK_MT_STATE_BATCH","0",1);vk_dq_refresh();assert(!vk_dq_dynamic((VkCommandBuffer)1,&a));
 assert(!warnings && atomic_load(&g_rd)==atomic_load(&g_pub));return 0;
}
'''
        with tempfile.TemporaryDirectory(prefix='nx2a-dynamic-') as tmp:
            src,exe=Path(tmp)/'check.c',Path(tmp)/'check';src.write_text(program(tail))
            p=subprocess.run(['clang','-std=gnu11','-O2','-pthread','-Wall','-Wextra','-Werror','-Wno-missing-field-initializers','-I'+str(DM/'third_party/vulkan-headers/include'),'-I'+str(DM/'android'),str(src),'-o',str(exe)],capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stderr);p=subprocess.run([str(exe)],capture_output=True,text=True,timeout=10);self.assertEqual(p.returncode,0,p.stdout+p.stderr)

    def test_production_set_dynamic_matches_original_calls_and_bounds(self):
        source=(DM/'android/gfx_vk_exec.c').read_text();rec=(DM/'android/gfx_vk_dq.c').read_text();header=(DM/'android/gfx_vk.h').read_text()
        state=re.search(r'typedef struct \{ VkViewport vp;.*?\} vk_dq_dynamic_state;',header,re.S).group()
        bc=source[source.index('static struct { uint64_t gen; VkViewport vp;'):source.index('// Merged-pass validation')]
        macro=re.search(r'(?m)^#define BC_SAME[^\n]*',source).group()
        code=r'''
#define VK_NO_PROTOTYPES
#include <assert.h>
#include <stdint.h>
#include <limits.h>
#include <math.h>
#include <string.h>
#include "bind_batch.h"
typedef struct{float factor,units,clamp;}p_poly_offset;
static struct{float vp[1][4],dnear,dfar;uint32_t pass_w,pass_h;uint8_t vp_sw[1][4];float blendc[4];int smask[2],swmask[2],sref[2];p_poly_offset pofs;}S;
static uint64_t g_cmd_gen=17;static int g_bc_stale,g_x_nobc,g_tb_mode=2,g_x_rebind_fast=1,g_x_bind_batch,g_lower_left,g_flipy,g_depth_zero=1;
static struct{int has_depth_clip_ctl;}V;
static int32_t g_pass_bb[4];static VkRect2D sc;
static VkRect2D scissor_rect(void){return sc;}
#define VKRED_ONCE(...) ((void)0)
static unsigned logical,calls;
static void bind_batch_count(unsigned kind,unsigned old,unsigned n){assert(kind==0);logical+=old;calls+=n;}
static unsigned nbytes,emit_count,helper_calls;static int packet;
static unsigned char bytes[2048];
static void emit(unsigned op,VkCommandBuffer cb,const void *p,size_t n){assert(cb==(VkCommandBuffer)1 && nbytes+4+n<=sizeof bytes);memcpy(bytes+nbytes,&op,4);nbytes+=4;memcpy(bytes+nbytes,p,n);nbytes+=n;emit_count++;}
static void vkCmdSetViewport(VkCommandBuffer cb,uint32_t first,uint32_t n,const VkViewport *p){assert(!first && n==1);emit(1,cb,p,sizeof *p);}
static void vkCmdSetScissor(VkCommandBuffer cb,uint32_t first,uint32_t n,const VkRect2D *p){assert(!first && n==1);emit(2,cb,p,sizeof *p);}
static void vkCmdSetBlendConstants(VkCommandBuffer cb,const float *p){emit(3,cb,p,4*sizeof *p);}
static void vkCmdSetStencilCompareMask(VkCommandBuffer cb,VkStencilFaceFlags f,uint32_t v){uint32_t a[]={f,v};emit(4,cb,a,sizeof a);}
static void vkCmdSetStencilWriteMask(VkCommandBuffer cb,VkStencilFaceFlags f,uint32_t v){uint32_t a[]={f,v};emit(5,cb,a,sizeof a);}
static void vkCmdSetStencilReference(VkCommandBuffer cb,VkStencilFaceFlags f,uint32_t v){uint32_t a[]={f,v};emit(6,cb,a,sizeof a);}
static void vkCmdSetDepthBias(VkCommandBuffer cb,float b,float c,float s){float a[]={b,c,s};emit(7,cb,a,sizeof a);}
static void vkCmdSetLineWidth(VkCommandBuffer cb,float w){emit(8,cb,&w,sizeof w);}
#define r_vkCmdSetViewport vkCmdSetViewport
#define r_vkCmdSetScissor vkCmdSetScissor
#define r_vkCmdSetBlendConstants vkCmdSetBlendConstants
#define r_vkCmdSetStencilCompareMask vkCmdSetStencilCompareMask
#define r_vkCmdSetStencilWriteMask vkCmdSetStencilWriteMask
#define r_vkCmdSetStencilReference vkCmdSetStencilReference
#define r_vkCmdSetDepthBias vkCmdSetDepthBias
#define r_vkCmdSetLineWidth vkCmdSetLineWidth
'''+state+'\n'+gs.function(rec,'run_dynamic')+r'''
static int vk_dq_dynamic(VkCommandBuffer cb,const vk_dq_dynamic_state *a){helper_calls++;if(!packet)return 0;run_dynamic(cb,a);return 1;}
'''+bc+gs.function(source,'bc_fresh')+macro+'\n'+gs.function(source,'set_dynamic')+r'''
static void reset(void){memset(&g_bc,0,sizeof g_bc);g_bc_stale=1;g_pass_bb[0]=g_pass_bb[1]=INT_MAX;g_pass_bb[2]=g_pass_bb[3]=INT_MIN;nbytes=emit_count=helper_calls=logical=calls=0;}
int main(void){
 for(unsigned k=0;k<256;k++){
  S.pass_w=1920;S.pass_h=1080;S.vp[0][0]=k%8;S.vp[0][1]=k%13;S.vp[0][2]=(k&1)?1920:0;S.vp[0][3]=(k&1)?1080:0;
  S.dnear=(k&2)?-.5f:.25f;S.dfar=(k&2)?2:.75f;S.vp_sw[0][0]=0;S.vp_sw[0][1]=(k&4)?1:0;
  S.blendc[0]=.1f;S.blendc[1]=.2f;S.blendc[2]=.3f;S.blendc[3]=.4f;sc=(VkRect2D){{(int32_t)k,7},{1000,800}};
  for(unsigned j=0;j<2;j++){S.smask[j]=(k&8)?(int)(0x80000000u+k):k+j;S.swmask[j]=(k&16)?-1:k+j;S.sref[j]=(k&32)?19:k-j;}
  S.pofs=(p_poly_offset){1.5f,-.5f,.2f};g_lower_left=k&64;g_flipy=k&128;g_x_bind_batch=k&1;
  unsigned char expected[2048];int32_t bounds[4];
  packet=0;reset();set_dynamic((VkCommandBuffer)1);unsigned n=nbytes,count=emit_count,old=logical,actual=calls;
  memcpy(expected,bytes,n);memcpy(bounds,g_pass_bb,sizeof bounds);__typeof__(g_bc) cache=g_bc;
  packet=1;reset();set_dynamic((VkCommandBuffer)1);
  assert(helper_calls==1 && emit_count==count && nbytes==n && !memcmp(expected,bytes,n));
  assert(!memcmp(bounds,g_pass_bb,sizeof bounds) && !memcmp(&cache,&g_bc,sizeof cache) && old==logical && actual==calls);
  // Unforced unchanged state must keep the original cross-draw cache path.
  g_tb_mode=0;helper_calls=nbytes=0;set_dynamic((VkCommandBuffer)1);assert(!helper_calls && !nbytes);g_tb_mode=2;
 }
 return 0;
}
'''
        gs.GpuSegments.compile_run(self,code)
