"Conservative storage-image capability from the immutable shader binding interface."
import re
import unittest
from tests import test_texture_barrier as tb

class ShaderImageUsage(unittest.TestCase):
    def test_complete_reflection_and_unknown_interfaces(self):
        h=(tb.DM/'android/gfx_vk.h').read_text()
        defs='\n'.join(re.search(p,h).group() for p in [r'enum \{ RK_SUPPORT[^\n]+',r'enum \{ RD_UBO[^\n]+',r'typedef struct __attribute__\(\(packed\)\) \{[^\n]+\} dmsv_bind;'])
        tb.TextureBarrier.compile_run(self,r'''
#include <assert.h>
#include <stdint.h>
#include <string.h>
''' + defs + r'''
#include "shader_image_usage.h"
int main(void){
 uint8_t b[60]={0};uint32_t n=1;memcpy(b,&n,4);dmsv_bind d={.kind=RK_TEX,.desc=RD_SAMPLED};memcpy(b+20,&d,sizeof d);
 assert(!vk_reflection_needs_images(b,40));d.kind=RK_SBUF;d.desc=RD_SSBO;memcpy(b+20,&d,sizeof d);assert(!vk_reflection_needs_images(b,40));
 const unsigned kinds[]={RK_IMG,RK_IMG_UBO,RK_IMG_TBUF,RK_UNKNOWN};
 for(unsigned i=0;i<sizeof kinds/sizeof kinds[0];i++){d.kind=kinds[i];memcpy(b+20,&d,sizeof d);assert(vk_reflection_needs_images(b,40));}
 d.kind=RK_TEX;d.desc=RD_STORAGE;memcpy(b+20,&d,sizeof d);assert(vk_reflection_needs_images(b,40));d.desc=99;memcpy(b+20,&d,sizeof d);assert(vk_reflection_needs_images(b,40));
 assert(vk_reflection_needs_images(NULL,40));assert(vk_reflection_needs_images(b,19));n=UINT32_MAX;memcpy(b,&n,4);assert(vk_reflection_needs_images(b,40));
 n=2;memcpy(b,&n,4);d.kind=RK_TEX;d.desc=RD_SAMPLED;memcpy(b+20,&d,sizeof d);d.kind=RK_IMG;d.desc=RD_STORAGE;memcpy(b+40,&d,sizeof d);assert(vk_reflection_needs_images(b,60));assert(vk_reflection_needs_images(b,59));
 return 0;
}
''')
        source=(tb.DM/'android/gfx_vk_shader.c').read_text()
        self.assertIn('static void vk_shaders_open(',source)
        self.assertIn('pthread_once(&once, bundle_open_once)',source)
        self.assertIn('static int g_storage_images = 1',source)

    def test_actual_image_policy_retains_dummy_unknown_and_storage_bundles(self):
        source=(tb.DM/'android/gfx_vk_tex.c').read_text()
        tb.TextureBarrier.compile_run(self,r'''
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan_core.h>
#include <assert.h>
#include <stdlib.h>
#include <stdint.h>
#define NVN_TEXTURE_FLAGS_IMAGE 8
#define VKLOG(...) do{}while(0)
typedef struct{struct{int flags;}d;void *pool;}tex_t;
static int needed=1,calls;static int vk_shader_storage_images(void){calls++;return needed;}
''' + tb.function(source,'image_storage_usage') + r'''
int main(void){
 tex_t t={.d.flags=8,.pool=(void *)(uintptr_t)1};VkFormatProperties fp={.optimalTilingFeatures=VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT};
 unsetenv("DM_VK_IMAGE_USAGE_TRIM");assert(image_storage_usage(&t,&fp) && !calls);
 setenv("DM_VK_IMAGE_USAGE_TRIM","1",1);assert(image_storage_usage(&t,&fp) && calls==1);
 needed=0;assert(!image_storage_usage(&t,&fp) && calls==2);
 t.pool=NULL;assert(image_storage_usage(&t,&fp) && calls==2);t.pool=(void *)(uintptr_t)1;
 fp.optimalTilingFeatures=0;assert(!image_storage_usage(&t,&fp) && calls==2);
 fp.optimalTilingFeatures=VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT;t.d.flags=0;assert(!image_storage_usage(&t,&fp));
 return 0;
}
''')
