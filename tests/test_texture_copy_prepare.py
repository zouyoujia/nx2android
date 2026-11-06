"""Execute real image resolution and transfer commands with Vulkan call mocks."""
from pathlib import Path
import os,shutil,subprocess,tempfile,unittest
from tests.test_lazy_small_pools import function
ROOT=Path(__file__).resolve().parents[1];DM=ROOT/'nxnative/dm'
@unittest.skipUnless(os.name!='nt' and shutil.which('clang'),'POSIX clang needed')
class TextureCopyPrepare(unittest.TestCase):
    def test_copy_destination_avoids_upload_cb_and_preserves_package_linear_and_readback(self):
        tex=(DM/'android/gfx_vk_tex.c').read_text();exec_=(DM/'android/gfx_vk_exec.c').read_text()
        funcs='\n'.join(function(tex,n) for n in ('vk_tex_prepare','vk_tex_','vk_tex_timed','vk_tex','vk_tex_copy_dst','vk_tex_copy_profile'))
        funcs+='\n'+function(exec_,'copy_b2t')
        code=r'''
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan_core.h>
#include <stdint.h>
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include "nvn_obj.h"
#include "pass_census.h"
#include "texture_copy_batch.h"
#define VKLOG(...) do {if(0)fprintf(stderr,__VA_ARGS__);}while(0)
#define VKRED_ONCE(n,...) do {(void)(n);if(0)fprintf(stderr,__VA_ARGS__);}while(0)
typedef struct {int depth,stencil;unsigned bw,bh,bpb,dbpp;int dec;} vkfmt_t;
typedef struct {VkImageLayout layout;VkImage img;uint64_t pool,used,pkg_used,pkg_stamp,pkg_seen,pkg_hash;int64_t off,size;int flags,packaged,pkg_released,gpu_w;const void *pkg_span;unsigned layers,d,aspect;} vkimg_t;
typedef struct {uint64_t pool;int64_t off;nvn_texdesc d;} akey;
typedef struct {uint64_t hi;} vkpool_t;
static pthread_mutex_t g_img_mu=PTHREAD_MUTEX_INITIALIZER;
static int g_by_key,g_by_serial,exists,sub_open=1,package,cpu_written,linear,source_ok=1;
static int up_calls,up_prepares,main_prepares,copies,readbacks,package_calls,linear_calls;
static uint64_t g_tex_frame=3,g_img_n,g_img_bytes,g_copy_dst_preparations;
int g_prof_on;uint64_t g_prof_ns[8];
static vkimg_t image;static vkfmt_t format={.bw=1,.bh=1,.bpb=4,.dbpp=4};
static struct {int row_stride,image_stride;} S;
static int64_t g_watch_off=-1;static int g_copy_dst_main=1;
static vkimg_t *vk_tex_(const tex_t *,int);
static VkCommandBuffer vk_upload_cmd(void){up_calls++;return (VkCommandBuffer)(uintptr_t)9;}
static VkCommandBuffer vk_cmd(void){return (VkCommandBuffer)(uintptr_t)7;}
static void vkCmdPipelineBarrier(VkCommandBuffer cb,VkPipelineStageFlags src,VkPipelineStageFlags dst,VkDependencyFlags dep,uint32_t nm,const VkMemoryBarrier *m,uint32_t nb,const VkBufferMemoryBarrier *b,uint32_t ni,const VkImageMemoryBarrier *i){
 (void)m;(void)b;assert(!dep && !nm && !nb && ni==1 && i && src==VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT && dst==VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
 assert(i->oldLayout==VK_IMAGE_LAYOUT_UNDEFINED && i->newLayout==VK_IMAGE_LAYOUT_GENERAL && i->image==image.img);
 assert(!i->srcAccessMask && i->dstAccessMask==(VK_ACCESS_MEMORY_READ_BIT|VK_ACCESS_MEMORY_WRITE_BIT));
 if(cb==(VkCommandBuffer)(uintptr_t)9)up_prepares++;else{assert(cb==(VkCommandBuffer)(uintptr_t)7);main_prepares++;}
}
static void vk_tex_prepare(VkCommandBuffer cb,vkimg_t *im);
static const vkfmt_t *vk_fmt(int f){return f==1?&format:NULL;}
static void *nvn_hm_get(void *map,const void *key){(void)key;assert(map==&g_by_key || map==&g_by_serial);return exists?&image:NULL;}
static void nvn_hm_put(void *map,const void *key,void *v){(void)map;(void)key;assert(v==&image);exists=1;}
static void evict_overlaps(uint64_t p,int64_t o,int64_t n){(void)p;(void)o;(void)n;}
static vkimg_t *img_new(const tex_t *t,const vkfmt_t *f){(void)f;memset(&image,0,sizeof image);image.img=(VkImage)(uintptr_t)10;image.layers=1;image.aspect=VK_IMAGE_ASPECT_COLOR_BIT;image.d=t->d.depth;return &image;}
static void alias_link(vkimg_t *im){assert(im==&image);}
static void pool_image_add(vkimg_t *im,const akey *k){assert(im==&image && k->pool==im->pool);}
static int vk_sub_open(void){return sub_open;}
static int lin_copy(vkimg_t *im,const tex_t *t,const vkfmt_t *f){(void)t;(void)f;if(!linear)return 0;linear_calls++;vk_tex_prepare(vk_upload_cmd(),im);return 1;}
static int packaged_tex(const tex_t *t){(void)t;return package;}
int nvn_ft_written(const tex_t *t){(void)t;return cpu_written;}
static void pkg_upload(vkimg_t *im,const tex_t *t,const vkfmt_t *f){(void)t;(void)f;package_calls++;vk_tex_prepare(vk_upload_cmd(),im);}
static uint64_t nvn_dirty_now(void){return 7;}
static int nvn_dirty_since_span(const void *sp,const void *m,size_t n,uint64_t s){(void)sp;(void)m;(void)n;(void)s;return 0;}
static uint64_t pkg_hash(const void *p,size_t n){(void)p;(void)n;return 0;}
static void nvn_dirty_protect(const void *p,size_t n){(void)p;(void)n;}
void vk_end_pass_land_reason(enum vk_pass_end_reason r){(void)r;}
static int vk_buf_at(uint64_t a,uint64_t n,VkBuffer *b,VkDeviceSize *o){(void)n;assert(a==0x10000);*b=(VkBuffer)(uintptr_t)11;*o=16;return source_ok;}
static void vk_img_barrier_all(VkCommandBuffer cb){assert(cb==(VkCommandBuffer)(uintptr_t)7);}
static VkImageSubresourceLayers sub_of(vkimg_t *im,const view_t *v,int has,int z){(void)v;(void)has;(void)z;return (VkImageSubresourceLayers){im->aspect,0,0,1};}
static uint8_t *vk_scratch(size_t n,int al,VkBuffer *b,VkDeviceSize *o){(void)n;(void)al;(void)b;(void)o;abort();}
static void vk_bc_decode(const vkfmt_t *f,const void *a,size_t p,uint32_t w,uint32_t h,int d,void *o){(void)f;(void)a;(void)p;(void)w;(void)h;(void)d;(void)o;abort();}
static void vk_watch(const vkimg_t *im,const char *w){(void)im;(void)w;}
static void vkCmdCopyBufferToImage(VkCommandBuffer cb,VkBuffer b,VkImage im,VkImageLayout l,uint32_t n,const VkBufferImageCopy *r){
 assert(cb==(VkCommandBuffer)(uintptr_t)7 && b==(VkBuffer)(uintptr_t)11 && im==image.img && l==VK_IMAGE_LAYOUT_GENERAL && image.layout==l && n==1 && r->bufferOffset==16 && r->imageExtent.width==8 && r->imageExtent.height==8);copies++;
}
static void vkCmdCopyImageToBuffer(VkCommandBuffer cb,VkImage im,VkImageLayout l,VkBuffer b,uint32_t n,const VkBufferImageCopy *r){(void)cb;(void)im;(void)b;(void)r;assert(l==VK_IMAGE_LAYOUT_GENERAL && image.layout==l && n==1);readbacks++;}
void vk_gpu_wrote(uint64_t a,uint64_t n){assert(a==0x10000 && n==4096);}
static vkpool_t pool={.hi=0x11000};
static vkpool_t *vk_pool_of(uint64_t a,uint64_t n){(void)a;(void)n;return &pool;}
'''+funcs+r'''
static void reset(void){exists=package=cpu_written=linear=up_calls=up_prepares=main_prepares=copies=readbacks=package_calls=linear_calls=0;g_copy_dst_preparations=0;sub_open=source_ok=1;}
int main(void){
 tex_t t={0};t.magic=MAGIC('T','E','X',0);t.serial=1;t.mem=(void *)(uintptr_t)0x20000;t.pool=(void *)(uintptr_t)0x30000;t.L.size=4096;t.d.format=1;t.d.target=NVN_TEXTURE_TARGET_2D;t.d.width=t.d.height=8;t.d.depth=1;
 p_copy_b2t c={.src=0x10000,.tex=&t,.r={.width=8,.height=8,.depth=1}};
 reset();copy_b2t(&c,1);assert(copies==1 && main_prepares==1 && !up_calls && image.gpu_w && g_copy_dst_preparations==1);
 copy_b2t(&c,1);assert(copies==2 && main_prepares==1 && !up_calls && g_copy_dst_preparations==1);
 reset();copy_b2t(&c,0);assert(readbacks==1 && up_calls==1 && up_prepares==1 && !main_prepares && !g_copy_dst_preparations);
 reset();package=1;copy_b2t(&c,1);assert(copies==1 && package_calls==1 && up_calls==1 && !main_prepares && !g_copy_dst_preparations);
 reset();cpu_written=1;copy_b2t(&c,1);assert(copies==1 && package_calls==1 && up_calls==1 && !g_copy_dst_preparations);
 reset();linear=1;copy_b2t(&c,1);assert(copies==1 && linear_calls==1 && up_calls==1 && !g_copy_dst_preparations);
 reset();source_ok=0;copy_b2t(&c,1);assert(!copies && !up_calls && !main_prepares);source_ok=1;copy_b2t(&c,1);assert(copies==1 && main_prepares==1 && !up_calls);
 reset();sub_open=0;assert(!vk_tex_copy_dst(&t) && !up_calls);sub_open=1;copy_b2t(&c,1);assert(copies==1 && main_prepares==1 && !up_calls);
 reset();g_copy_dst_main=0;copy_b2t(&c,1);assert(copies==1 && up_calls==1 && up_prepares==1 && !g_copy_dst_preparations);
 reset();assert(vk_tex(&t) && up_calls==1 && up_prepares==1);
 // Batch admission accepts one direct copy (optionally preceded by stride state).
 uint8_t record[2048]={0};uint64_t handles[2]={(uint64_t)(uintptr_t)record,0};
 uint32_t size=(uint32_t)((sizeof(op_hdr)+sizeof(p_copy_b2t)+7)&~7u);
 op_hdr h={.op=OP_COPY_B2T,.size=size},end={.op=OP_END,.size=8};memcpy(record,&h,8);memcpy(record+size,&end,8);
 assert(texture_copy_standalone(handles,1));assert(!texture_copy_standalone(handles,2));
 op_hdr stride={.op=OP_COPY_STRIDE,.size=24};memmove(record+24,record,size+8);memcpy(record,&stride,8);assert(texture_copy_standalone(handles,1));
 h.op=OP_COPY_B2B;memcpy(record+24,&h,8);assert(!texture_copy_standalone(handles,1));
 h.op=OP_COPY_B2T;memcpy(record+24,&h,8);end.op=OP_DRAW;memcpy(record+24+size,&end,8);assert(!texture_copy_standalone(handles,1));
 stride.op=OP_CALL;memcpy(record,&stride,8);assert(!texture_copy_standalone(handles,1));
 stride.op=OP_COPY_STRIDE;stride.size=7;memcpy(record,&stride,8);assert(!texture_copy_standalone(handles,1));return 0;
}
'''
        with tempfile.TemporaryDirectory(prefix='nx2a-copy-prepare-') as tmp:
            src,exe=Path(tmp)/'check.c',Path(tmp)/'check';src.write_text(code)
            p=subprocess.run(['clang','-std=gnu11','-O2','-Wall','-Wextra','-Werror','-Wno-missing-field-initializers','-pthread','-I'+str(DM/'android'),'-I'+str(DM/'gfx/nvn'),'-I'+str(DM/'gfx/nvn/include'),'-I'+str(DM/'third_party/vulkan-headers/include'),str(src),'-o',str(exe)],capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stderr)
            p=subprocess.run([str(exe)],capture_output=True,text=True,timeout=10)
            self.assertEqual(p.returncode,0,p.stdout+p.stderr)
