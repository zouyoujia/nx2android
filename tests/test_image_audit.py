"Execute read-only attachment compression queries without altering image policy."
from pathlib import Path
import os
import shutil
import unittest
from tests.test_texture_barrier import function, DM
from tests import test_texture_barrier as tb


@unittest.skipUnless(os.name != 'nt' and shutil.which('clang'), 'POSIX host clang needed')
class ImageAudit(unittest.TestCase):
    def test_query_is_opt_in_bounded_and_unknown_without_a_driver_report(self):
        source = (DM/'android/gfx_vk_tex.c').read_text()
        code = r'''
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan_core.h>
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
static char line[2048];static unsigned logs,requirements,queries;static int leave_unknown;
#define VKLOG(...) do {snprintf(line,sizeof line,__VA_ARGS__);logs++;} while(0)
typedef struct {VkImage img;VkFormat fmt,fmt2,audit_view_fmt;uint32_t w,h,samples,levels,layers;int fixed_fmt;VkImageUsageFlags usage;VkImageCreateFlags cflags;VkImageAspectFlags aspect;VkImageLayout layout;} vkimg_t;
static struct {int image_audit;VkDevice dev;PFN_vkGetImageSubresourceLayout2EXT compression_query;} V;
static void vkGetImageMemoryRequirements(VkDevice dev,VkImage image,VkMemoryRequirements *rq){assert(dev==V.dev && image);requirements++;*rq=(VkMemoryRequirements){.size=999999};}
static void query(VkDevice dev,VkImage image,const VkImageSubresource2EXT *sr,VkSubresourceLayout2EXT *lo){
 assert(dev==V.dev && image && sr->sType==VK_STRUCTURE_TYPE_IMAGE_SUBRESOURCE_2_EXT && !sr->pNext);
 assert(sr->imageSubresource.aspectMask && !(sr->imageSubresource.aspectMask & (sr->imageSubresource.aspectMask-1)) && !sr->imageSubresource.mipLevel && !sr->imageSubresource.arrayLayer);
 assert(lo->sType==VK_STRUCTURE_TYPE_SUBRESOURCE_LAYOUT_2_EXT && lo->pNext);queries++;
 VkImageCompressionPropertiesEXT *cp=lo->pNext;assert(cp->sType==VK_STRUCTURE_TYPE_IMAGE_COMPRESSION_PROPERTIES_EXT && !cp->pNext);
 if(!leave_unknown){cp->imageCompressionFlags=queries==1?VK_IMAGE_COMPRESSION_DISABLED_EXT:VK_IMAGE_COMPRESSION_DEFAULT_EXT;cp->imageCompressionFixedRateFlags=0;}
}
''' + function(source, 'vk_image_audit') + r'''
int main(void){
 vkimg_t im={.img=(VkImage)(uintptr_t)1,.fmt=VK_FORMAT_R8G8B8A8_UNORM,.w=1920,.h=1080,.aspect=VK_IMAGE_ASPECT_COLOR_BIT,.samples=1,.layers=1,.levels=1};
 vkimg_t original=im;vk_image_audit(&im,im.fmt);assert(!memcmp(&im,&original,sizeof im) && !logs && !queries && !requirements);
 V.image_audit=1;vk_image_audit(NULL,im.fmt);assert(!logs);
 vk_image_audit(&im,im.fmt);assert(logs==1 && requirements==1 && !queries && strstr(line,"compression unknown") && strstr(line,"999999"));
 vk_image_audit(&im,im.fmt);assert(logs==1); // unchanged attachment is not repeatedly queried
 V.compression_query=query;im.audit_view_fmt=0;vk_image_audit(&im,im.fmt);assert(queries==1 && strstr(line,"compression disabled"));
 im.audit_view_fmt=0;vk_image_audit(&im,im.fmt);assert(queries==2 && strstr(line,"compression default"));
 leave_unknown=1;im.audit_view_fmt=0;vk_image_audit(&im,im.fmt);assert(queries==3 && strstr(line,"compression unknown"));leave_unknown=0;
 im.aspect=VK_IMAGE_ASPECT_DEPTH_BIT|VK_IMAGE_ASPECT_STENCIL_BIT;im.audit_view_fmt=0;vk_image_audit(&im,im.fmt);assert(queries==3 && strstr(line,"compression unknown"));
 im.aspect=VK_IMAGE_ASPECT_DEPTH_BIT;im.audit_view_fmt=0;vk_image_audit(&im,im.fmt);assert(queries==4);
 // Only the diagnostic stamp changed; creation flags, usage and layout stay intact.
 im.aspect=original.aspect;im.audit_view_fmt=0;assert(!memcmp(&im,&original,sizeof im));
 for(int i=0;i<200;i++){im.audit_view_fmt=0;vk_image_audit(&im,im.fmt);}assert(logs==96 && requirements==96);
 return 0;
}
'''
        tb.TextureBarrier.compile_run(self, code)
