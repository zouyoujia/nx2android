"Execute production dirty-page planning and linear upload commands with bounded write hazards."
from pathlib import Path
import os,re,shutil,subprocess,tempfile,unittest
from tests.test_gpu_segments import function
ROOT=Path(__file__).resolve().parents[1];DM=ROOT/'nxnative/dm'
@unittest.skipUnless(os.name!='nt' and shutil.which('clang'),'POSIX clang needed')
class LinearUpload(unittest.TestCase):
    def run_c(self,code):
        with tempfile.TemporaryDirectory(prefix='nx2a-linear-') as tmp:
            src,exe=Path(tmp)/'test.c',Path(tmp)/'test';src.write_text(code)
            p=subprocess.run(['clang','-std=gnu11','-O2','-Wall','-Wextra','-Werror','-Wno-missing-field-initializers',
                '-I'+str(DM/'android'),'-I'+str(DM/'gfx/nvn'),'-I'+str(DM/'gfx/nvn/include'),'-I'+str(DM/'third_party/vulkan-headers/include'),str(src),'-o',str(exe)],capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stderr)
            p=subprocess.run([str(exe)],capture_output=True,text=True,timeout=15);self.assertEqual(p.returncode,0,p.stdout+p.stderr)

    def test_write_history_eviction_pending_unknown_and_row_projection(self):
        self.run_c(r'''
#include <assert.h>
#include "linear_upload.h"
int main(void){
 vk_lw_history h={0};assert(!vk_lw_changed(&h,0,100,100));
 vk_lw_note(&h,20,10,8);assert(!vk_lw_changed(&h,0,100,100));
 vk_lw_note(&h,150,3,8);assert(vk_lw_changed(&h,1,100,100) && vk_lw_pending(&h,7,100,100));
 assert(!vk_lw_pending(&h,8,100,100));
 uint64_t seq=h.seq;vk_lw_note(&h,150,3,8);assert(vk_lw_changed(&h,seq,100,100)); // repeated same-fence writes stay observable
 vk_lw_note(&h,0,UINT64_MAX,9);assert(vk_lw_changed(&h,seq,100,100));
 for(unsigned i=0;i<VK_LW_HISTORY+1;i++)vk_lw_note(&h,10000,10,20);
 assert(vk_lw_changed(&h,0,100,100));assert(vk_lw_pending(&h,8,100,100)); // evicted pending opaque writer
 assert(!vk_lw_pending(&h,20,100,100));
 assert(vk_lw_changed(&h,h.seq,UINT64_MAX-1,4));
 h.seq=UINT64_MAX;assert(vk_lw_pending(&h,UINT64_MAX,100,100));
 vk_linear_rows rows[VK_LINEAR_REGIONS];nvn_dirty_interval d[]={{3,1000},{4096,4096},{8192,4096}};
 int n=vk_linear_project(d,3,1024,32,0,rows);assert(n==2 && rows[0].first==0 && rows[0].count==1 && rows[1].first==4 && rows[1].count==8);
 assert(vk_linear_project(d,-1,1024,32,0,rows)==-1);
 assert(vk_linear_project(d,3,1024,32,1,rows)==-1); // unaligned Vulkan buffer offset
 d[0]=(nvn_dirty_interval){32000,1000};assert(vk_linear_project(d,1,1024,32,0,rows)==-1);
 assert(vk_linear_project(d,0,1024,32,0,rows)==0);
 return 0;
}
''')

    def test_actual_dirty_collector_clips_merges_and_never_silently_truncates(self):
        code=function((DM/'gfx/nvn/nvn_dirty.c').read_text(),'nvn_dirty_collect')
        self.run_c(r'''
#include <assert.h>
#include <stdatomic.h>
#include <stdint.h>
#include "nvn_dirty.h"
typedef struct{uintptr_t base,end;_Atomic int live;_Atomic uint64_t last;_Atomic uint64_t gen[8];}range_t;
static range_t range;static int g_shift=12,found=1;
static range_t *find_span(uintptr_t a,size_t n){return found && a>=range.base && n<=range.end-a?&range:0;}
'''+code+r'''
int main(void){
 range.base=0x10000;range.end=0x18000;range.live=1;range.last=20;
 range.gen[0]=12;range.gen[1]=13;range.gen[4]=19;nvn_dirty_interval d[8];
 int n=nvn_dirty_collect(&range,(void*)0x10800,0x5000,10,d,8);
 assert(n==2 && d[0].offset==0 && d[0].length==0x1800 && d[1].offset==0x3800 && d[1].length==0x1000);
 assert(nvn_dirty_collect(&range,(void*)0x10800,0x5000,10,d,1)==-1);
 assert(nvn_dirty_collect(&range,(void*)0x10800,0x5000,20,d,8)==0);
 found=0;range.live=0;assert(nvn_dirty_collect(&range,(void*)0x10800,0x5000,10,d,8)==-1);
 assert(nvn_dirty_collect(0,(void*)UINTPTR_MAX,2,0,d,8)==-1);
 return 0;
}
''')

    def test_actual_shader_write_classifier(self):
        h=(DM/'android/gfx_vk.h').read_text();defs='\n'.join(re.search(p,h).group() for p in [r'enum \{ RK_SUPPORT[^\n]+',r'enum \{ RD_UBO[^\n]+',r'typedef struct __attribute__\(\(packed\)\) \{[^\n]+\} dmsv_bind;'])
        self.run_c('#include <stdint.h>\n#include <assert.h>\n'+defs+r'''
#include "linear_shader_writes.h"
int main(void){
 dmsv_bind b[2]={{.kind=RK_TEX,.desc=RD_SAMPLED},{.kind=RK_CBUF,.desc=RD_UBO}};
 assert(!vk_linear_shader_class(b,2));b[1].kind=RK_SBUF;b[1].desc=RD_SSBO;assert(vk_linear_shader_class(b,2)==1);
 b[0].kind=RK_IMG;b[0].desc=RD_STORAGE;assert(vk_linear_shader_class(b,2)==3);
 b[0].kind=RK_UNKNOWN;assert(vk_linear_shader_class(b,1)==2);
 assert(vk_linear_shader_class(0,1)==2);return 0;
}
''')

    def test_actual_linear_copy_audit_reuse_partial_and_safe_fallbacks(self):
        s=(DM/'android/gfx_vk_tex.c').read_text();funcs='\n'.join(function(s[s.rindex('static int lin_copy('):] if n=='lin_copy' else s,n) for n in ['linear_storage','linear_cache_forget','linear_cache_protect','lin_copy'])
        enums=re.search(r'enum \{ LC_CALL[^\n]+',s).group()
        self.run_c(r'''
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan_core.h>
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "nvn_obj.h"
#include "linear_upload.h"
#define VKLOG(...) do{}while(0)
typedef struct{unsigned bw,bh,bpb;}vkfmt_t;
typedef struct{
 unsigned levels,d,layers,w,h,aspect;int gpu_w,novl,nvn_format;VkImage img;VkImageLayout layout;
 const void *lin_mem,*lin_span,*lc_mem,*lc_span;size_t lin_len,lc_len;uint64_t lin_stamp,lin_gen,lc_stamp,lc_writes;int lc_valid;
}vkimg_t;
static int mode,tracked=1,pending,changed,deferred,dirty_n,write_race;
static uint64_t generation=10,journal,g_up_gen;
static unsigned calls,uploads,barriers,protects,releases;static int upload_open;
static nvn_dirty_interval dirty[VK_LINEAR_REGIONS];static VkBufferImageCopy last[VK_LINEAR_REGIONS];static unsigned last_count;
static uint64_t g_lc[32];
static int vk_linear_cache_mode(void){return mode;}
static int vk_buf_at(uint64_t a,uint64_t n,VkBuffer *b,VkDeviceSize *o){assert(a && n);*b=(VkBuffer)1;*o=0;return 1;}
int nvn_dirty_tracked(const void *a,size_t n){assert(a && n);return tracked;}
void nvn_dirty_protect(const void *a,size_t n){assert(a && n);protects++;}
void nvn_dirty_release(const void *a,size_t n){assert(a && n);releases++;}
uint64_t nvn_dirty_now(void){return generation;}
const void *nvn_dirty_span(const void *a,size_t n){assert(a && n);return tracked?(void*)1:0;}
int nvn_dirty_since_span(const void *sp,const void *a,size_t n,uint64_t stamp){(void)sp;(void)a;(void)n;return generation>stamp;}
int nvn_dirty_collect(const void *sp,const void *a,size_t n,uint64_t stamp,nvn_dirty_interval *out,int cap){
 assert(sp && a && n && cap==VK_LINEAR_REGIONS && stamp<=generation);if(write_race)generation++;if(dirty_n<0)return -1;memcpy(out,dirty,(size_t)dirty_n*sizeof *out);return dirty_n;
}
static uint64_t vk_linear_write_stamp(void){return journal;}
static int vk_linear_writes_changed(uint64_t stamp,uint64_t a,uint64_t n){assert(stamp<=journal && a && n);return changed;}
static int vk_linear_writes_pending(uint64_t a,uint64_t n){assert(a && n);return pending;}
static int vk_ubd_overlaps(uint64_t a,uint64_t n){assert(a && n);return deferred;}
static VkCommandBuffer vk_upload_cmd(void){uploads++;if(!upload_open){g_up_gen++;upload_open=1;}return (VkCommandBuffer)2;}
static void vk_gpu_upload_note(unsigned k,uint64_t n){assert(k==2 && n);}
static void vk_img_barrier_all(VkCommandBuffer cb){assert(cb==(VkCommandBuffer)2);barriers++;}
static void vk_tex_prepare(VkCommandBuffer cb,vkimg_t *im){assert(cb);im->layout=VK_IMAGE_LAYOUT_GENERAL;}
static void vkCmdCopyBufferToImage(VkCommandBuffer cb,VkBuffer b,VkImage im,VkImageLayout layout,uint32_t n,const VkBufferImageCopy *r){
 assert(cb && b && im && layout==VK_IMAGE_LAYOUT_GENERAL && n>0 && n<=VK_LINEAR_REGIONS);calls++;last_count=n;memcpy(last,r,n*sizeof *r);
}
'''+enums+'\n'+funcs+r'''
int main(void){
 pool_t pool={.flags=0};tex_t t={.pool=&pool,.mem=(void*)0x100000};t.d.flags=NVN_TEXTURE_FLAGS_LINEAR;t.d.stride=1024;
 vkfmt_t f={1,1,1};vkimg_t im={.levels=1,.d=1,.layers=1,.w=1024,.h=2048,.aspect=VK_IMAGE_ASPECT_COLOR_BIT,.img=(VkImage)3};
 mode=2;assert(lin_copy(&im,&t,&f));assert(calls==1 && im.lc_valid && protects==2 && last[0].imageExtent.height==2048);
 upload_open=0;mode=1;journal=5;unsigned up=uploads;assert(lin_copy(&im,&t,&f));assert(calls==1 && uploads==up && !upload_open && im.lc_writes==5 && im.lc_stamp==10); // clean avoids CB and barrier
 dirty_n=1;dirty[0]=(nvn_dirty_interval){4096,4096};generation=20;write_race=1;
 assert(lin_copy(&im,&t,&f));assert(calls==2 && last_count==1 && last[0].bufferOffset==4096 && last[0].imageOffset.y==4 && last[0].imageExtent.height==4);
 assert(im.lc_stamp==20 && generation==21);write_race=0; // write during scan remains newer than snapshot
 upload_open=0;changed=1;journal++;assert(lin_copy(&im,&t,&f));assert(calls==3 && last[0].imageExtent.height==2048);changed=0;
 upload_open=0;dirty_n=0;pending=1;assert(lin_copy(&im,&t,&f));assert(calls==4 && !im.lc_valid);pending=0;
 upload_open=0;mode=2;assert(lin_copy(&im,&t,&f));assert(im.lc_valid);upload_open=0;unsigned c=calls;assert(lin_copy(&im,&t,&f));assert(calls==c+1 && last[0].imageExtent.height==2048); // audit preserves full transfer
 upload_open=0;mode=1;dirty_n=-1;assert(lin_copy(&im,&t,&f));assert(last[0].imageExtent.height==2048);
 upload_open=0;im.novl=1;c=calls;assert(lin_copy(&im,&t,&f));assert(calls==c+1 && !im.lc_valid && releases);im.novl=0;
 upload_open=0;pool.flags=NVN_MEMORY_POOL_FLAGS_VIRTUAL;assert(lin_copy(&im,&t,&f));assert(!im.lc_valid);pool.flags=0;
 upload_open=0;tracked=0;assert(lin_copy(&im,&t,&f));assert(!im.lc_valid);tracked=1;
 im.gpu_w=1;c=calls;assert(lin_copy(&im,&t,&f));assert(calls==c && im.lin_mem && !im.lc_valid); // existing GPU-image rule
 mode=0;im.gpu_w=0;upload_open=0;assert(lin_copy(&im,&t,&f));assert(calls==c+1);
 f.bw=4;assert(!lin_copy(&im,&t,&f));return 0;
}
''')

    def test_actual_storage_write_ranges_are_recorded_before_descriptor_memo(self):
        src=(DM/'android/gfx_vk_exec.c').read_text();h=(DM/'android/gfx_vk.h').read_text()
        enums='\n'.join(re.search(p,h).group() for p in [r'enum \{ RK_SUPPORT[^\n]+',r'enum \{ RD_UBO[^\n]+',r'typedef struct __attribute__\(\(packed\)\) \{[^\n]+\} dmsv_bind;'])
        self.assertLess(function(src,'stage_set').index('linear_shader_writes(st, s)'),function(src,'stage_set').index('set_memo_get('))
        self.run_c('#include <assert.h>\n#include <stdint.h>\n#include <stddef.h>\n'+enums+r'''
#define SB_ALIGN 256
static int mode=1,unknown,notes;static uint64_t lo,size,ptr;
typedef struct{int nbind;unsigned linear_writes;dmsv_bind *b;}vkshader_t;
typedef struct{uint64_t lo,hi;}vkpool_t;
static vkpool_t pool={0x10000,0x20000};static struct{struct{uint64_t addr,size;}ssbo[6][16];}S;
static int vk_linear_cache_mode(void){return mode;}
static void vk_linear_unknown_write(void){unknown++;}
static uint32_t ubo_word(int st,int reg,int off){(void)st;(void)reg;return off?ptr>>32:ptr;}
static vkpool_t *vk_pool_of(uint64_t a,uint64_t n){return a>=pool.lo && a+n<=pool.hi?&pool:0;}
static void vk_linear_buffer_wrote(uint64_t a,uint64_t n){lo=a;size=n;notes++;}
'''+function(src,'linear_shader_writes')+r'''
int main(void){
 dmsv_bind b={.kind=RK_SBUF,.desc=RD_SSBO,.reg=1};vkshader_t sh={1,1,&b};
 S.ssbo[5][1].addr=0x10181;S.ssbo[5][1].size=100;linear_shader_writes(5,&sh);
 assert(notes==1 && lo==0x10100 && size==229); // include actual descriptor alignment prefix
 b.kind=RK_SBUF_UBO;b.off=0;ptr=0x1ff80;linear_shader_writes(5,&sh);assert(notes==2 && lo==0x1ff00 && size==256);
 sh.linear_writes=2;linear_shader_writes(0,&sh);assert(unknown==1);
 mode=0;linear_shader_writes(0,&sh);assert(unknown==1);return 0;
}
''')
