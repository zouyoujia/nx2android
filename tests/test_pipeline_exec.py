"Verify optional driver compiler inspection using production capture code and mocked Vulkan entry points."
from tests import test_gpu_segments as gs
import unittest

ROOT = gs.ROOT
SOURCE = (ROOT/'nxnative/dm/android/gfx_vk_pipeline_exec.c').read_text()
PREFIX = r'''#define VK_NO_PROTOTYPES
#include <vulkan/vulkan_core.h>
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <pthread.h>
#include <dirent.h>
#include <unistd.h>
#define VKLOG(...) do { if (0) fprintf(stderr, __VA_ARGS__); } while(0)
typedef struct {uint64_t key; VkShaderModule mod; int set; uint32_t *spv,nspv;} vkshader_t;
static struct {VkDevice dev; VkPhysicalDeviceProperties props;} V;
static unsigned creates,destroys,modules,module_destroys,ir_reads; static int fail_create,ir_mode;
static int missing_entry_points, unchanged_count;
static VkShaderModule expected_base=(VkShaderModule)(uintptr_t)7;
static VkResult make(VkDevice d,VkPipelineCache cache,uint32_t n,const VkGraphicsPipelineCreateInfo *c,const VkAllocationCallbacks *a,VkPipeline *p){
 (void)d;(void)a;assert(!cache && n==1);assert(c->flags&VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR);assert(c->flags&VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR);
 assert(c->stageCount==2);assert(c->pStages[0].module==(VkShaderModule)(uintptr_t)8);
 assert(c->pStages[1].module==(modules?(VkShaderModule)(uintptr_t)9:expected_base));creates++;*p=(VkPipeline)(uintptr_t)(100+creates);return fail_create?VK_ERROR_OUT_OF_HOST_MEMORY:VK_SUCCESS;
}
static void destroy(VkDevice d,VkPipeline p,const VkAllocationCallbacks *a){(void)d;(void)a;assert(p);destroys++;}
static VkResult make_module(VkDevice d,const VkShaderModuleCreateInfo *c,const VkAllocationCallbacks *a,VkShaderModule *m){(void)d;(void)a;assert(c->codeSize==36 && c->pCode[8]==1);modules++;*m=(VkShaderModule)(uintptr_t)9;return VK_SUCCESS;}
static void destroy_module(VkDevice d,VkShaderModule m,const VkAllocationCallbacks *a){(void)d;(void)a;assert(m==(VkShaderModule)(uintptr_t)9);module_destroys++;}
static VkResult props(VkDevice d,const VkPipelineInfoKHR *p,uint32_t *n,VkPipelineExecutablePropertiesKHR *e){
 (void)d;assert(p->pipeline);if(!e){*n=unchanged_count?32:1;return VK_SUCCESS;}
 assert(*n==(unchanged_count?32u:1u) && e[0].sType==VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR);if(!unchanged_count)*n=1;e[0].stages=VK_SHADER_STAGE_FRAGMENT_BIT;e[0].subgroupSize=64;strcpy(e[0].name,"fragment\nname");return VK_SUCCESS;
}
static VkResult stats(VkDevice d,const VkPipelineExecutableInfoKHR *p,uint32_t *n,VkPipelineExecutableStatisticKHR *s){
 (void)d;assert(p->executableIndex==0);if(!s){*n=1;return VK_SUCCESS;}
 assert(*n==1 && s[0].sType==VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR);*n=1;strcpy(s->name,"instructions");s->format=VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR;s->value.u64=777;return VK_SUCCESS;
}
static VkResult ir(VkDevice d,const VkPipelineExecutableInfoKHR *p,uint32_t *n,VkPipelineExecutableInternalRepresentationKHR *s){
 (void)d;assert(p->executableIndex==0);if(!s){*n=1;return VK_SUCCESS;}
 assert(*n>=1 && s[0].sType==VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INTERNAL_REPRESENTATION_KHR);*n=1;
 if(!s[0].pData){s[0].dataSize=ir_mode==1?3*1024*1024:4;s[0].isText=1;strcpy(s[0].name,"../not-a-file");return ir_mode==4?VK_INCOMPLETE:VK_SUCCESS;}
 ir_reads++;assert(s[0].dataSize==4);memcpy(s[0].pData,"asm\n",4);if(ir_mode==2)s[0].dataSize=100;return ir_mode==3?VK_INCOMPLETE:VK_SUCCESS;
}
static PFN_vkVoidFunction get(VkDevice d,const char *s){(void)d;if(missing_entry_points)return NULL;
 if(!strcmp(s,"vkGetPipelineExecutablePropertiesKHR"))return (PFN_vkVoidFunction)props;
 if(!strcmp(s,"vkGetPipelineExecutableStatisticsKHR"))return (PFN_vkVoidFunction)stats;
 if(!strcmp(s,"vkGetPipelineExecutableInternalRepresentationsKHR"))return (PFN_vkVoidFunction)ir;
 return NULL;
}
static PFN_vkGetDeviceProcAddr vkGetDeviceProcAddr=get;
static PFN_vkCreateGraphicsPipelines vkCreateGraphicsPipelines=make;
static PFN_vkDestroyPipeline vkDestroyPipeline=destroy;
static PFN_vkCreateShaderModule vkCreateShaderModule=make_module;
static PFN_vkDestroyShaderModule vkDestroyShaderModule=destroy_module;
'''
COMMON = r'''static char tmp[1024];
static __attribute__((unused)) void start(void){strcpy(tmp,"/tmp/nx2a-pipeline-exec-XXXXXX");assert(mkdtemp(tmp));setenv("DM_VK_PIPE_EXEC_KEYS","0000000000000011",1);setenv("DM_VK_PIPE_EXEC_OUT",tmp,1);unsetenv("DM_VK_PIPE_EXEC_BASE_DIR");unsetenv("DM_VK_PIPE_EXEC_OPT_DIR");unsetenv("DM_VK_PIPE_EXEC_STOP");assert(vk_pipeline_exec_requested());vk_pipeline_exec_init(1);assert(g_vk_pipeline_exec_enabled);}
static __attribute__((unused)) void finish(void){DIR *d=opendir(tmp);assert(d);struct dirent *e;while((e=readdir(d)))if(strcmp(e->d_name,".")&&strcmp(e->d_name,"..")){char p[1400];snprintf(p,sizeof p,"%s/%s",tmp,e->d_name);assert(!unlink(p));}closedir(d);assert(!rmdir(tmp));}
static __attribute__((unused)) void capture(unsigned state,int variant){
 uint32_t spv[]={0x07230203,0x10300,0,5,0,4u<<16|71,2,34,1};vkshader_t fs={.key=17,.mod=(VkShaderModule)(uintptr_t)7,.set=1,.spv=spv,.nspv=9};
 VkPipelineShaderStageCreateInfo stages[2]={{.stage=VK_SHADER_STAGE_VERTEX_BIT,.module=(VkShaderModule)(uintptr_t)8},{.stage=VK_SHADER_STAGE_FRAGMENT_BIT,.module=variant?(VkShaderModule)(uintptr_t)10:fs.mod}};
 expected_base=stages[1].module;VkGraphicsPipelineCreateInfo c={.sType=VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,.stageCount=2,.pStages=stages};VkGraphicsPipelineCreateInfo before=c;
 vk_pipeline_exec_capture(&c,&state,sizeof state,123,&fs);assert(!memcmp(&c,&before,sizeof c)&&stages[1].module==expected_base);
}
static __attribute__((unused)) unsigned files(const char *suffix){unsigned count=0;DIR *d=opendir(tmp);assert(d);struct dirent *e;while((e=readdir(d))){size_t a=strlen(e->d_name),b=strlen(suffix);if(a>=b&&!strcmp(e->d_name+a-b,suffix))count++;}closedir(d);return count;}
'''

class PipelineExec(unittest.TestCase):
    compile_run=gs.GpuSegments.compile_run

    def run_c(self, main):
        self.compile_run(PREFIX+SOURCE.replace('#include "gfx_vk.h"','')+COMMON+main)

    def test_default_disabled_strict_startup_filter_and_missing_entry_points(self):
        self.run_c(r'''int main(void){
 unsetenv("DM_VK_PIPE_EXEC_KEYS");assert(!vk_pipeline_exec_requested());capture(1,0);assert(!creates);
 const char *bad[]={"1","000000000000001g","00000000000000111","0000000000000011,0000000000000011","0000000000000011/"};
 setenv("DM_VK_PIPE_EXEC_OUT","/tmp",1);for(unsigned i=0;i<sizeof bad/sizeof *bad;i++){memset(&cfg,0,sizeof cfg);setenv("DM_VK_PIPE_EXEC_KEYS",bad[i],1);configure();assert(!cfg.n);}
 setenv("DM_VK_PIPE_EXEC_KEYS","0000000000000011, 0000000000000022",1);configure();assert(cfg.n==2);vk_pipeline_exec_init(0);assert(!g_vk_pipeline_exec_enabled);missing_entry_points=1;vk_pipeline_exec_init(1);assert(!g_vk_pipeline_exec_enabled);return 0;
}
''')

    def test_spirv_validation_and_descriptor_only_reference_match(self):
        self.run_c(r'''int main(void){
 uint32_t a[]={0x07230203,0x10300,0,5,0,4u<<16|71,2,34,0},b[9];memcpy(b,a,sizeof a);b[8]=1;assert(spv_same(a,9,b,9));
 b[6]=3;assert(!spv_same(a,9,b,9));b[6]=2;b[5]=5u<<16|71;assert(!spv_valid(b,9));b[5]=0;assert(!spv_valid(b,9));assert(!spv_valid(a,4));a[0]=0;assert(!spv_valid(a,9));return 0;
}
''')

    def test_private_pairs_dedup_state_limits_and_no_render_mutation(self):
        self.run_c(r'''int main(void){
 start();setenv("DM_VK_PIPE_EXEC_STOP","1",1);capture(99,0);assert(!creates);setenv("DM_VK_PIPE_EXEC_STOP","0",1);strcpy(cfg.base,tmp);strcpy(cfg.opt,tmp);char path[1200];snprintf(path,sizeof path,"%s/0000000000000011.spv",tmp);
 uint32_t w[]={0x07230203,0x10300,0,5,0,4u<<16|71,2,34,0};FILE *f=fopen(path,"wb");assert(f&&fwrite(w,4,9,f)==9);fclose(f);
 capture(1,0);assert(creates==2&&destroys==2&&modules==1&&module_destroys==1&&files(".txt")==2);capture(1,0);assert(creates==2);
 modules=0;capture(2,0);assert(creates==4&&destroys==4&&modules==1&&module_destroys==2);capture(3,0);assert(creates==4&&nseen==2);finish();return 0;
}
''')

    def test_variant_and_reference_mismatch_skip_optimized_clone(self):
        self.run_c(r'''int main(void){start();setenv("DM_VK_PIPE_EXEC_STOP","1",1);capture(99,0);assert(!creates);setenv("DM_VK_PIPE_EXEC_STOP","0",1);strcpy(cfg.base,tmp);strcpy(cfg.opt,tmp);char path[1200];snprintf(path,sizeof path,"%s/0000000000000011.spv",tmp);
 uint32_t w[]={0x07230203,0x10300,0,6,0,4u<<16|71,2,34,0};FILE *f=fopen(path,"wb");assert(f&&fwrite(w,4,9,f)==9);fclose(f);
 capture(1,0);capture(2,1);assert(creates==2&&destroys==2&&!modules&&!module_destroys);finish();return 0;}
''')

    def test_oversized_changed_and_incomplete_ir_do_not_dump_untrusted_data(self):
        for mode in [1,2,3,4]:
            with self.subTest(mode=mode):
                self.run_c('int main(void){start();ir_mode='+str(mode)+';capture(1,0);assert(creates==1&&destroys==1&&files(".txt")=='+('1' if mode==4 else '0')+');finish();return 0;}')

    def test_failed_clone_is_destroyed_without_queries(self):
        self.run_c(r'''int main(void){start();fail_create=1;capture(1,0);assert(creates==1&&destroys==1&&!ir_reads);finish();return 0;}''')

    def test_unfilled_executables_are_never_queried_when_driver_keeps_capacity(self):
        self.run_c(r'''int main(void){start();unchanged_count=1;capture(1,0);assert(creates==1&&destroys==1&&ir_reads==1);finish();return 0;}''')
