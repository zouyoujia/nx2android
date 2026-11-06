"""Execute the actual KGSL hook and namespace loader with denied/unsupported operations."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent.parent
AT = ROOT / 'nxnative/dm/third_party/libadrenotools'
HOOK = ROOT / 'nxnative/dm/android/driver/kgsl_power_hook.c'


@unittest.skipUnless(shutil.which('clang'), 'host clang needed')
class ContextPower(unittest.TestCase):
    def test_actual_context_hook(self):
        code = r'''
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <assert.h>
#include <errno.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <adrenotools/priv.h>
int kgsl_hook_ioctl(int,unsigned long,...);
void init_kgsl_power_hook(struct adrenotools_kgsl_perf_stats *);
struct create {uint32_t flags,id;};
struct constraint {uint32_t type,context_id;void *data;size_t size;};
struct property {uint32_t type;void *value;size_t size;};
static int mode,calls,logs,sets,links;
static struct create *original_arg;
static const uint32_t original_flags=0x40010041u;
int __android_log_print(int priority,const char *tag,const char *fmt,...) {
    (void)priority;assert(tag && fmt);logs++;errno=EIO;return 0;
}
ssize_t power_test_readlink(const char *path,char *buf,size_t size) {
    assert(!strcmp(path,"/proc/self/fd/7"));assert(size>=16);links++;
    if(mode==9){errno=EACCES;return -1;}
    const char *name=mode==8?"/dev/null":"/dev/kgsl-3d0";
    size_t n=strlen(name);memcpy(buf,name,n);errno=EAGAIN;return (ssize_t)n;
}
#ifdef __APPLE__
int power_test_syscall(int nr,...) {
#else
long power_test_syscall(long nr,...) {
#endif
    assert(nr==SYS_ioctl);va_list ap;va_start(ap,nr);
    int fd=va_arg(ap,int);unsigned long req=va_arg(ap,unsigned long);void *arg=va_arg(ap,void *);va_end(ap);
    assert(fd==7);calls++;
    if(mode==6){assert(req==0x123456789ul && arg==(void *)1);errno=ENOTTY;return 17;}
    if(mode==7){assert(!arg);errno=EFAULT;return -1;}
    if(req==_IOWR(0x09,0x13,struct create)){
        struct create *c=arg;
        if(mode==0 || mode==8 || mode==9 || mode==12 || mode==13){
            assert(arg==original_arg && c->flags==original_flags);
            if(mode==13){errno=EACCES;return -1;}
            c->id=123;errno=EBUSY;return 0;
        }
        if(mode==2 && calls==2){
            assert(arg==original_arg && c->flags==original_flags && c->id==0);
            c->id=124;errno=EALREADY;return 0;
        }
        assert(arg!=original_arg && c->flags==(original_flags|0x800u));
        if(mode==2){c->id=999;errno=EINVAL;return -1;}
        if(mode==3){errno=EACCES;return -1;}
        if(mode==4){errno=EINTR;return -1;}
        if(mode==10){errno=EINVAL;return -1;}
        c->id=123;if(mode==11)c->flags&=~0x800u;
        errno=EBUSY;return mode==14?17:0;
    }
    assert(req==_IOW(0x09,0x32,struct property));sets++;
    struct property *p=arg;assert(p->type==0x12 && p->size==24);
    struct constraint *c=p->value;
    assert(c->type==1 && c->context_id==123 && c->size==4 && *(uint32_t *)c->data==1);
    if(mode==5){errno=EPERM;return -1;}errno=0;return 0;
}
int main(int argc,char **argv) {
    assert(argc==2);mode=atoi(argv[1]);struct create c={original_flags,0};original_arg=&c;
    struct adrenotools_kgsl_perf_stats stats={0};
    if(mode!=0){
        if(mode==12 || mode==13)setenv("DM_GPU_PERF","2",1);else unsetenv("DM_GPU_PERF");
        errno=EDOM;init_kgsl_power_hook(&stats);assert(errno==EDOM);
    }
    if(mode==6){assert(kgsl_hook_ioctl(7,0x123456789ul,(void *)1)==17 && calls==1 && errno==ENOTTY && !links);return 0;}
    if(mode==10)c.flags|=0x800u;
    int result=kgsl_hook_ioctl(7,_IOWR(0x09,0x13,struct create),mode==7?NULL:&c);
    if(mode==13){
        assert(result==-1 && c.id==0 && c.flags==original_flags && calls==1 && !sets && errno==EACCES);
        assert(stats.intercepted==1 && !stats.flagged && !stats.fallback && !stats.accepted && !stats.refused);
    }else if(mode==0 || mode==8 || mode==9 || mode==12){
        assert(!result && c.id==123 && c.flags==original_flags && calls==1 && !sets && errno==EBUSY);
        assert(stats.intercepted==(uint64_t)(mode==12 || mode==13));
        assert(!stats.flagged && !stats.fallback && !stats.accepted && !stats.refused);
    }else if(mode==2){
        assert(!result && c.id==124 && c.flags==original_flags && calls==2 && !sets && errno==EALREADY);
        assert(stats.intercepted==1 && stats.fallback==1 && !stats.flagged && !stats.accepted && !stats.refused);
    }else if(mode==3 || mode==4 || mode==10){
        assert(result==-1 && c.id==0 && c.flags==(original_flags|(mode==10?0x800u:0)) && calls==1 && !sets);
        assert(errno==(mode==3?EACCES:mode==4?EINTR:EINVAL));
        assert(stats.intercepted==1 && !stats.flagged && !stats.fallback && !stats.accepted && !stats.refused);
    }else if(mode==7){assert(result==-1 && errno==EFAULT && calls==1 && !sets && !links && !stats.intercepted);}
    else if(mode==11){
        assert(!result && c.id==123 && c.flags==original_flags && calls==1 && !sets && errno==EBUSY);
        assert(stats.intercepted==1 && !stats.flagged && !stats.accepted && !stats.refused);
    }else{
        assert(result==(mode==14?17:0) && c.id==123 && c.flags==(original_flags|0x800u) && calls==2 && sets==1 && errno==EBUSY);
        assert(stats.intercepted==1 && stats.flagged==1 && !stats.fallback);
        assert(stats.accepted==(uint64_t)(mode!=5) && stats.refused==(uint64_t)(mode==5));
    }
    return 0;
}
'''
        with tempfile.TemporaryDirectory(prefix='nx2a-kgsl-power-') as tmp:
            p = Path(tmp)
            (p / 'android').mkdir()
            (p / 'android/log.h').write_text('#define ANDROID_LOG_INFO 4\nint __android_log_print(int,const char *,const char *,...);\n')
            (p / 'check.c').write_text(code)
            result = subprocess.run([
                'clang', '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror',
                '-Wno-deprecated-declarations', '-I' + str(p), '-I' + str(AT / 'include'),
                '-Dioctl=kgsl_hook_ioctl', '-Dsyscall=power_test_syscall',
                '-Dreadlink=power_test_readlink', str(HOOK), str(p / 'check.c'),
                '-o', str(p / 'check')], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            for mode in range(15):
                subprocess.run([str(p / 'check'), str(mode)], check=True, timeout=5)

    @unittest.skipUnless(shutil.which('clang++'), 'host clang++ needed')
    def test_actual_stock_namespace_loader(self):
        # Use the production loader functions, stopping before unrelated GSL allocation
        # adapters which depend on Linux-only headers. No copy of the loader's logic.
        source = (AT / 'src/hook/hook_impl.cpp').read_text()
        source = source.split('__attribute__((visibility("default"))) FILE *hook_fopen')[0]
        source = source.replace('#include "kgsl.h"', '')
        code = r'''
#include <cassert>
#include <cstdarg>
#include <cstring>
#include <string>
#include <vector>
#include <dlfcn.h>
#include <android_linker_ns.h>
#include <link.h>
#include "hook_impl_params.h"
#include "hook_impl.h"
struct android_namespace_t {};
static android_namespace_t vendor, isolated, vndk;
static int mode, plain_creates, escaped_creates, hook_loads, init_power, calls;
static std::vector<std::string> order;
static const HookImplParams *expected;
static adrenotools_kgsl_perf_stats stats{};
int __android_log_print(int,const char *,const char *,...) { return 0; }
extern "C" int dl_iterate_phdr(int (*)(dl_phdr_info *,size_t,void *),void *) { return 0; }
static bool link_names(android_namespace_t *to,android_namespace_t *from,const char *names) {
    assert(to==&isolated && (from==&vendor || from==&vndk || from==nullptr));
    assert(!strstr(names,"libgsl") && !strstr(names,"vulkan.adreno") && !strstr(names,"libllvm") && !strstr(names,"libadreno"));
    if(mode==0){assert(from==nullptr && !strcmp(names,"libandroid.so"));return true;}
    if(from==&vndk){assert(strstr(names,"libcutils.so") && strstr(names,"libdmabufheap.so") && strstr(names,"mapper@2.0.so") && strstr(names,"mapper@3.0.so") && strstr(names,"mapper@4.0.so"));}
    else{assert(!strstr(names,"libcutils.so") && !strstr(names,"libc++.so") && strstr(names,"libsync.so"));}
    return true;
}
android_link_namespaces_t android_link_namespaces=link_names;
android_link_namespaces_all_libs_t android_link_namespaces_all_libs=nullptr;
static android_namespace_t *get_ns(const char *name) { return !strcmp(name,"vndk") ? (mode==13?nullptr:&vndk) : &vendor; }
android_get_exported_namespace_t android_get_exported_namespace=get_ns;
android_namespace_t *android_create_namespace(const char *,const char *custom,const char *paths,uint64_t type,const char *,android_namespace_t *parent) {
    plain_creates++;assert(mode==0 && !*custom && !strcmp(paths,"/app/libs") && type==ANDROID_NAMESPACE_TYPE_SHARED && parent==&vendor);return &isolated;
}
android_namespace_t *android_create_namespace_escape(const char *,const char *custom,const char *paths,uint64_t type,const char *,android_namespace_t *parent) {
    escaped_creates++;assert(mode!=0 && type==ANDROID_NAMESPACE_TYPE_REGULAR && parent==nullptr);
    assert(!strncmp(paths,"/vendor/lib64/hw:/vendor/lib64:",31));
    assert(strstr(paths,"/odm/lib64") && strstr(paths,"/app/libs"));
    assert(!strcmp(custom,mode==7?"/custom/":""));return mode==4?nullptr:&isolated;
}
static void power_init(adrenotools_kgsl_perf_stats *s) {assert(s==&stats);init_power++;}
extern "C" void *dlsym(void *handle,const char *symbol) {
    if(handle==(void *)1 && !strcmp(symbol,"init_hook_param"))return mode==12?nullptr:(void *)&init_hook_param;
    if(handle==(void *)2 && !strcmp(symbol,"init_kgsl_power_hook"))return mode==5?nullptr:(void *)&power_init;
    return nullptr;
}
void *linkernsbypass_namespace_dlopen(const char *name,int flags,android_namespace_t *ns) {
    assert(ns==&isolated);order.emplace_back(name);
    if(strstr(name,"libhook_impl.so")){assert(flags==RTLD_NOW);return mode==11?nullptr:(void *)1;}
    assert(!strcmp(name,"/app/libs/libkgsl_power_hook.so"));assert(flags==(RTLD_NOW|RTLD_GLOBAL));hook_loads++;
    return mode==2?nullptr:(void *)2;
}
extern "C" void *android_dlopen_ext(const char *name,int,const android_dlextinfo *info) {
    calls++;
    if(mode==8 || mode==9){assert(info==nullptr);return (void *)3;}
    if(mode==10){assert(!name);return (void *)3;}
    assert(info && (info->library_namespace==&isolated || info->library_namespace==&vendor));
    if(info->library_namespace==&isolated){
        assert(mode==0 || (hook_loads==1 && init_power==1));
        assert(!strcmp(name,mode==7?"vulkan.custom.so":"/vendor/lib64/hw/vulkan.adreno.so"));
        return mode==3?nullptr:(void *)3;
    }
    assert(mode==2 || mode==3 || mode==4 || mode==5 || mode==6 || mode==11 || mode==12 || mode==13);
    return (void *)3;
}
int main(int argc,char **argv) {
    assert(argc==2);mode=atoi(argv[1]);
    HookImplParams params(mode==0?0:ADRENOTOOLS_DRIVER_KGSL_PERF | (mode==7?ADRENOTOOLS_DRIVER_CUSTOM:0),nullptr,"/app/libs",mode==7?"/custom/":nullptr,mode==7?"vulkan.custom.so":nullptr,nullptr,nullptr,&stats);
    expected=&params;init_hook_param(mode==6?nullptr:&params);
    android_dlextinfo ext{};ext.flags=ANDROID_DLEXT_USE_NAMESPACE;ext.library_namespace=&vendor;
    const char *filename=mode==10?nullptr:"/vendor/lib64/hw/vulkan.adreno.so";
    const android_dlextinfo *info=mode==8 || mode==9?nullptr:&ext;
    if(mode==9)filename="libunrelated.so";
    assert(hook_android_dlopen_ext(filename,RTLD_NOW,info)==(void *)3);
    if(mode==0)assert(plain_creates==1 && !escaped_creates && !hook_loads && !init_power && calls==1);
    else if(mode>=6 && mode<=10 && mode!=7)assert(!plain_creates && !escaped_creates && !hook_loads && calls==1);
    else assert(!plain_creates && escaped_creates==1);
    if(mode==1 || mode==7)assert(hook_loads==1 && init_power==1 && calls==1);
    if(mode==3)assert(calls==2);
    return 0;
}
'''
        with tempfile.TemporaryDirectory(prefix='nx2a-kgsl-loader-') as tmp:
            p = Path(tmp)
            (p / 'android').mkdir()
            (p / 'android/log.h').write_text('#define ANDROID_LOG_INFO 4\nint __android_log_print(int,const char *,const char *,...);\n')
            (p / 'android/dlext.h').write_text('''#pragma once
#include <stdint.h>
struct android_namespace_t;
struct android_dlextinfo {uint64_t flags; struct android_namespace_t *library_namespace;};
#define ANDROID_DLEXT_USE_NAMESPACE 0x200
#ifdef __cplusplus
extern "C" {
#endif
void *android_dlopen_ext(const char *,int,const struct android_dlextinfo *);
#ifdef __cplusplus
}
#endif
''')
            (p / 'link.h').write_text('''#pragma once
#include <stddef.h>
#include <stdint.h>
struct Elf64_Phdr {uint32_t p_type,p_flags; uint64_t p_offset,p_vaddr,p_paddr,p_filesz,p_memsz,p_align;};
struct dl_phdr_info {uintptr_t dlpi_addr; const char *dlpi_name; const Elf64_Phdr *dlpi_phdr; uint16_t dlpi_phnum;};
extern "C" int dl_iterate_phdr(int (*)(dl_phdr_info *,size_t,void *),void *);
''')
            (p / 'loader.cpp').write_text(source)
            (p / 'check.cpp').write_text(code)
            result = subprocess.run([
                'clang++', '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
                '-Wno-unused-parameter', '-I' + str(p), '-I' + str(AT / 'include'),
                '-I' + str(AT / 'src/hook'), '-I' + str(AT / 'lib/linkernsbypass'),
                str(p / 'loader.cpp'), str(p / 'check.cpp'), '-o', str(p / 'check')],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            for mode in range(14):
                subprocess.run([str(p / 'check'), str(mode)], check=True, timeout=5)


    def test_read_only_elf_binding_bounds(self):
        code = r"""
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "kgsl_binding_elf.h"
static uintptr_t slots[4], targets[4]; static int count;
static void emit(uintptr_t slot,uintptr_t target,void *opaque) {
    assert(opaque==(void *)123 && count<4); slots[count]=slot;targets[count++]=target;
}
int main(int argc,char **argv) {
    assert(argc==2);int mode=atoi(argv[1]);
    _Alignas(8) unsigned char image[4096]={0}, before[4096];
    struct kgsl_diag_phdr headers[3]={{1,6,0,0,0,sizeof image,sizeof image,4096},{2,6,0,0x100,0,128,128,8},{0}};
    struct kgsl_diag_image im={(uintptr_t)image,headers,2};
    struct kgsl_diag_dyn *d=(void *)(image+0x100);
    const struct kgsl_diag_dyn initial[]={{5,0x300},{10,14},{6,0x400},{11,24},{23,0x500},{2,24},{20,7},{0,0}};
    memcpy(d,initial,sizeof initial);memcpy(image+0x300,"\0ioctl\0other\0",14);
    struct kgsl_diag_sym *syms=(void *)(image+0x400);syms[1].name=1;
    struct kgsl_diag_rela *r=(void *)(image+0x500);r[0]=(struct kgsl_diag_rela){0x600,((uint64_t)1<<32)|1026,0};
    uintptr_t target=0x1234;memcpy(image+0x600,&target,sizeof target);
    switch(mode){
    case 1:d[1].value=UINT64_MAX;break;
    case 2:syms[1].name=14;break;
    case 3:memset(image+0x300,'X',14);break;
    case 4:d[4].value=0x10000;break;
    case 5:d[5].value=23;break;
    case 6:r[0].info=((uint64_t)0xffffffff<<32)|1026;break;
    case 7:r[0].offset=0x10000;break;
    case 8:r[0].offset=0x601;break;
    case 9:headers[0].flags=2;break;
    case 10:d[7].tag=5;break;
    case 11:headers[2]=headers[1];im.phnum=3;break;
    case 12:d[0].value+=im.base;d[2].value+=im.base;d[4].value+=im.base;break;
    case 13:d[6].value=17;break;
    case 14:syms[1].name=7;break;
    case 15:r[0].info=((uint64_t)1<<32)|257;break;
    case 16:syms[1].shndx=1;break;
    case 18:im.phnum=1025;break;
    case 19:d[5].value=0;break;
    case 20:headers[1].vaddr=UINT64_MAX;break;
    case 21:headers[0].memsz=UINT64_MAX;break;
    case 22:d[2].value=0;break;
    case 23:d[0].value=0;break;
    case 24:d[3].value=16;break;
    case 25:r[1]=r[0];r[1].offset=0x608;d[5].value=48;target=0x5678;memcpy(image+0x608,&target,sizeof target);break;
    case 27:d[0].value=0xfff;break;
    }
    memcpy(before,image,sizeof image);const char *why=NULL;
    int n=kgsl_diag_ioctl_slots(mode==26?NULL:&im,mode==17?NULL:emit,(void *)123,&why);
    assert(why && !memcmp(image,before,sizeof image));
    if(mode==0 || mode==12){assert(n==1 && count==1 && slots[0]==im.base+0x600 && targets[0]==0x1234);}
    else if(mode==25){assert(n==2 && count==2 && targets[1]==0x5678);}
    else if(mode==14 || mode==15 || mode==16 || mode==19){assert(n==0 && count==0);}
    else assert(n==-1 && count==0);
    return 0;
}
"""
        with tempfile.TemporaryDirectory(prefix='nx2a-kgsl-elf-') as tmp:
            p = Path(tmp)
            (p / 'check.c').write_text(code)
            result = subprocess.run([
                'clang', '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror',
                '-I' + str(AT / 'src/hook'), str(p / 'check.c'), '-o', str(p / 'check')],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            for mode in range(28):
                subprocess.run([str(p / 'check'), str(mode)], check=True, timeout=5)


if __name__ == '__main__':
    unittest.main()
