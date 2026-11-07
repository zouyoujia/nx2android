"Exercise fused present geometry and actual recording with mocked Vulkan."
from pathlib import Path
import unittest
from tests import test_texture_barrier as tb
DM=tb.DM

class PresentFused(unittest.TestCase):
    def test_rotation_crop_letterbox_and_push_abi(self):
        tb.TextureBarrier.compile_run(self,r'''
#include <assert.h>
#include <math.h>
#include "present_geometry.h"
static int closef(float a,float b){return fabsf(a-b)<.000002f;}
int main(void){
 vk_present_quad q;
 for(unsigned turn=0;turn<=3;turn++){
  if(turn==2)continue;
  assert(vk_present_quad_make(&q,3168,1440,1920,1080,NULL,0,turn));
  assert(closef(q.pos[0],-2560.0f/3168) && closef(q.pos[2],2560.0f/3168));
  assert(q.pos[1]==-1 && q.pos[3]==1 && q.uv[0]==0 && q.uv[3]==1 && q.turn==turn);
  // Shader's logical->physical map is inverse of the legacy rotation sampler.
  float x=.37f,y=-.23f,px=turn==1?-y:turn==3?y:x,py=turn==1?x:turn==3?-x:y;
  float bx=turn==1?py:turn==3?-py:px,by=turn==1?-px:turn==3?px:py;
  assert(closef(bx,x) && closef(by,y));
 }
 int crop[4]={100,50,800,400};assert(vk_present_quad_make(&q,2000,1000,1920,1080,crop,1,1));
 assert(q.pos[0]==-1 && q.pos[3]==1 && closef(q.uv[0],100.0f/1920) && closef(q.uv[1],450.0f/1080) && closef(q.uv[3],50.0f/1080));
 assert(!vk_present_quad_make(&q,0,1,1,1,NULL,0,0));assert(!vk_present_quad_make(&q,1,1,1,1,NULL,0,2));
 crop[0]=-1;assert(!vk_present_quad_make(&q,2000,1000,1920,1080,crop,0,0));crop[0]=1800;assert(!vk_present_quad_make(&q,2000,1000,1920,1080,crop,0,0));
 return 0;
}
''')

    def test_actual_fused_record_has_one_rendering_pass_and_rotates_overlay(self):
        s=(DM/'android/gfx_vk_present.c').read_text()
        # Compilation fails if the direct recorder gains a transfer blit/clear;
        # only actual colour-pass operations below are available to it.
        code=r'''
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan_core.h>
#include <assert.h>
#include <stdint.h>
#include "present_geometry.h"
typedef vk_present_quad ov_quad;
static struct {VkDevice dev;} V;
static struct {VkSampler samp;VkDescriptorSet set[8];VkPipeline pipe;VkPipelineLayout layout;} g_direct;
static struct {VkPipeline pipe;VkPipelineLayout layout;VkDescriptorSet set;} g_ov;
static VkImage g_simg[8];static VkImageView g_sview[8];static VkExtent2D g_next={1440,3168};static uint32_t g_ov_turn;
static int active,begin_count,end_count,draws,updates,overlay_calls,barriers,ready;
static void barrier(VkCommandBuffer cb,VkImage im,VkImageLayout old,VkImageLayout now){assert(cb && im==g_simg[5] && !active);if(!barriers)assert(old==VK_IMAGE_LAYOUT_UNDEFINED && now==VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);else assert(old==VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL && now==VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);barriers++;}
static int overlay_ready(VkCommandBuffer cb,uint32_t k,int menu){(void)menu;assert(cb && k==3 && !active && !updates);return ready;}
static void vkUpdateDescriptorSets(VkDevice dev,uint32_t n,const VkWriteDescriptorSet *w,uint32_t nc,const VkCopyDescriptorSet *copies){(void)copies;assert(dev==V.dev && n==1 && !nc && w->dstSet==g_direct.set[3] && w->pImageInfo->imageLayout==VK_IMAGE_LAYOUT_GENERAL && !active);updates++;}
static void vkCmdBeginRendering(VkCommandBuffer cb,const VkRenderingInfo *r){assert(cb && !active && updates==1 && barriers==1 && r->renderArea.extent.width==1440 && r->renderArea.extent.height==3168 && r->colorAttachmentCount==1);const VkRenderingAttachmentInfo *a=r->pColorAttachments;assert(a->imageView==g_sview[5] && a->loadOp==VK_ATTACHMENT_LOAD_OP_CLEAR && a->storeOp==VK_ATTACHMENT_STORE_OP_STORE && a->clearValue.color.float32[3]==1);active=1;begin_count++;}
static void vkCmdEndRendering(VkCommandBuffer cb){assert(cb && active);active=0;end_count++;}
static void vkCmdSetViewport(VkCommandBuffer cb,uint32_t at,uint32_t n,const VkViewport *p){assert(cb && active && !at && n==1 && p->width==1440 && p->height==3168);}
static void vkCmdSetScissor(VkCommandBuffer cb,uint32_t at,uint32_t n,const VkRect2D *p){assert(cb && active && !at && n==1 && p->extent.width==1440);}
static void vkCmdBindPipeline(VkCommandBuffer cb,VkPipelineBindPoint p,VkPipeline pipe){assert(cb && active && p==VK_PIPELINE_BIND_POINT_GRAPHICS && pipe);}
static void vkCmdBindDescriptorSets(VkCommandBuffer cb,VkPipelineBindPoint p,VkPipelineLayout l,uint32_t first,uint32_t n,const VkDescriptorSet *set,uint32_t nd,const uint32_t *offs){(void)offs;assert(cb && active && p==VK_PIPELINE_BIND_POINT_GRAPHICS && l && !first && n==1 && *set && !nd);}
static void vkCmdPushConstants(VkCommandBuffer cb,VkPipelineLayout l,VkShaderStageFlags st,uint32_t off,uint32_t size,const void *p){assert(cb && active && l && (st&VK_SHADER_STAGE_VERTEX_BIT) && !off && size==40);const vk_present_quad *q=p;assert(q->turn==1);}
static void vkCmdDraw(VkCommandBuffer cb,uint32_t n,uint32_t ni,uint32_t f,uint32_t fi){assert(cb && active && n==6 && ni==1 && !f && !fi);draws++;}
''' + tb.function(s,'ov_draw') + r'''
static void ov_render(VkCommandBuffer cb,int ovm,int menu,int fps){assert(ready && (ovm || menu>=0 || fps) && g_ov_turn==1 && active);ov_quad q={.alpha=.6f};ov_draw(cb,&q);overlay_calls++;}
''' + tb.function(s,'direct_record') + r'''
int main(void){
 g_direct.samp=(VkSampler)(uintptr_t)1;g_direct.pipe=(VkPipeline)(uintptr_t)2;g_direct.layout=(VkPipelineLayout)(uintptr_t)3;g_direct.set[3]=(VkDescriptorSet)(uintptr_t)4;
 g_ov.pipe=(VkPipeline)(uintptr_t)5;g_ov.layout=(VkPipelineLayout)(uintptr_t)6;g_ov.set=(VkDescriptorSet)(uintptr_t)7;g_simg[5]=(VkImage)(uintptr_t)8;g_sview[5]=(VkImageView)(uintptr_t)9;
 for(int mode=0;mode<4;mode++){
  active=begin_count=end_count=draws=updates=overlay_calls=barriers=0;ready=1;vk_present_quad q={.alpha=1,.turn=1};
  direct_record((VkCommandBuffer)(uintptr_t)1,3,5,(VkImageView)(uintptr_t)10,&q,mode==1,mode==2?0:-1,mode==3);
  assert(!active && begin_count==1 && end_count==1 && draws==(mode?2:1) && overlay_calls==(mode?1:0) && !g_ov_turn && barriers==2);
 }
 return 0;
}
'''
        tb.TextureBarrier.compile_run(self,code)
        window=tb.function(s,'present_window')
        self.assertLess(window.index('direct_source('),window.index('vkAcquireNextImageKHR('))
        self.assertIn('fused ? VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT : VK_PIPELINE_STAGE_TRANSFER_BIT',window)

    def test_present_view_is_cached_owned_and_rejects_unsupported_sources(self):
        source=(DM/'android/gfx_vk_tex.c').read_text()
        tb.TextureBarrier.compile_run(self,r'''
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan_core.h>
#include <assert.h>
#include <stdint.h>
#include <pthread.h>
#include <stdlib.h>
typedef struct{VkImage img;VkImageView full;VkFormat fmt;VkImageViewType vtype;VkImageAspectFlags aspect;VkImageUsageFlags usage;int samples,layers,d;}vkimg_t;
static struct{VkDevice dev;VkPhysicalDevice pd;}V;static pthread_mutex_t g_view_mu=PTHREAD_MUTEX_INITIALIZER;static int creates,fail,linear=1,retired;static uint64_t g_tex_gen;
static void vkGetPhysicalDeviceFormatProperties(VkPhysicalDevice pd,VkFormat fmt,VkFormatProperties *p){assert(pd==V.pd && fmt==VK_FORMAT_R8G8B8A8_UNORM);*p=(VkFormatProperties){.optimalTilingFeatures=linear?VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT:0};}
static VkResult vkCreateImageView(VkDevice dev,const VkImageViewCreateInfo *c,const VkAllocationCallbacks *a,VkImageView *v){assert(dev==V.dev && !a && c->viewType==VK_IMAGE_VIEW_TYPE_2D && c->subresourceRange.levelCount==1 && c->subresourceRange.layerCount==1);const VkImageViewUsageCreateInfo *u=c->pNext;assert(u && u->usage==VK_IMAGE_USAGE_SAMPLED_BIT);creates++;if(fail)return VK_ERROR_OUT_OF_HOST_MEMORY;*v=(VkImageView)(uintptr_t)99;return VK_SUCCESS;}
typedef struct vview{VkImageView v;struct vview *next;}vview;static int g_views;
static void *nvn_hm_get(void *h,const void *key){(void)h;(void)key;return NULL;}static void nvn_hm_del(void *h,const void *key){(void)h;(void)key;}
static void vk_defer_free_view(VkImageView v){assert(v==(VkImageView)(uintptr_t)99);retired++;}
''' + tb.function(source,'vk_present_view') + tb.function(source,'vk_tex_view_drop_all') + r'''
int main(void){
 vkimg_t im={.img=(VkImage)(uintptr_t)1,.fmt=VK_FORMAT_R8G8B8A8_UNORM,.vtype=VK_IMAGE_VIEW_TYPE_2D,.aspect=VK_IMAGE_ASPECT_COLOR_BIT,.usage=VK_IMAGE_USAGE_SAMPLED_BIT,.samples=1,.layers=1,.d=1};
 assert(!vk_present_view(NULL));im.samples=2;assert(!vk_present_view(&im) && !creates);im.samples=1;
 im.vtype=VK_IMAGE_VIEW_TYPE_3D;assert(!vk_present_view(&im));im.vtype=VK_IMAGE_VIEW_TYPE_2D;
 linear=0;assert(!vk_present_view(&im) && !creates);linear=1;fail=1;assert(!vk_present_view(&im) && creates==1 && !im.full);fail=0;
 assert(vk_present_view(&im) && creates==2);assert(vk_present_view(&im) && creates==2);
 vk_tex_view_drop_all(&im);assert(!im.full && retired==1 && g_tex_gen==1);
 return 0;
}
''')
