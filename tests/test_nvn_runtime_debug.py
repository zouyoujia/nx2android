"""Run NVN debug control helpers with portable thunk/wrapper and fence stubs."""
from pathlib import Path
import os
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def function(source, name):
    pattern = r'(?m)^[^\n;{}]*\b'+re.escape(name)+r'\s*\([^;{}]*\)\s*\{'
    match = re.search(pattern, source)
    if not match:
        raise AssertionError('missing production function '+name)
    pos, depth = match.end(), 1
    while depth:
        depth += (source[pos] == '{')-(source[pos] == '}')
        pos += 1
    return source[match.start():pos]+'\n'


@unittest.skipUnless(os.name != 'nt' and shutil.which('clang'), 'POSIX host clang needed')
class NvnRuntimeDebug(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = (ROOT/'nxnative/dm/gfx/nvn/hle_nvn.c').read_text()
        slots = source[source.index('typedef struct { _Atomic uint64_t cnt;'):source.index('extern void *nvn_thunk')]
        state = source[source.index('static _Atomic int g_nvn_prof;'):source.index('uint64_t nvn_unimpl_c')]
        seq = source[source.index('static _Atomic uint64_t g_frame;'):source.index('NVN(nvnQueueFenceSync,')]
        nops = source[source.index('static void nvn_nop(void)'):source.index('__attribute__((constructor)) static void nvn_nops_register')]
        names = re.findall(r'"(nvn[^"\n]+)"', nops)
        names = ['nvnCommandBufferDrawArrays', 'nvnQueueSubmitCommands', 'nvnQueuePresentTexture', 'nvnMissing']+names
        public_names = 'static const char *nvn_names[NVN_NFUNC] = {'+', '.join('"'+x+'"' for x in names)+'};\n'
        methods = '\n'.join(function(source, x) for x in ('nvn_init_slots', 'nvn_prof_prepare', 'nvn_prof_set', 'nvn_prof_install', 'nvn_lookup'))
        registration = function(source, 'nvn_nops_register').replace('__attribute__((constructor)) ', '')
        sync_at = source.index('NVN(nvnSyncWait,')
        sync_tail = source.index('NVN(nvnEventBuilderSetDefaults,', sync_at)
        sync = source[sync_at:sync_tail]
        cls.tmp = tempfile.TemporaryDirectory(prefix='nx2a-nvn-debug-')
        cls.exe = Path(cls.tmp.name)/'check'
        src = Path(cls.tmp.name)/'check.c'
        src.write_text(r'''
#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdatomic.h>
#include <pthread.h>
#include <sched.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define NVN_NFUNC '''+str(len(names))+r'''
'''+slots+state+public_names+r'''
void *nvn_real[NVN_NFUNC], *nvn_big_real[NVN_NFUNC], *nvn_thunk[NVN_NFUNC];
static _Atomic unsigned logs, env_prof_reads, env_seq_reads, clock_reads, wrappers;
static const char *env_prof, *env_seq;
static char *observed_getenv(const char *name) {
    if(!strcmp(name,"DM_NVN_PROF")){atomic_fetch_add(&env_prof_reads,1);return (char *)env_prof;}
    if(!strcmp(name,"DM_NVN_SEQ")){atomic_fetch_add(&env_seq_reads,1);return (char *)env_seq;}
    assert(!strcmp(name,"DM_NVN_LOG"));return NULL;
}
static void log_line(const char *format,...) {(void)format;atomic_fetch_add(&logs,1);}
#define LOG log_line
#define getenv observed_getenv
static uint64_t now_ns(void) {return (uint64_t)(atomic_fetch_add(&clock_reads,1)+1)*1000;}
static _Thread_local unsigned current_slot;
static _Atomic int pause_wrapper, entered_wrapper;
static uint64_t draw_impl(uint64_t x) {return x+7;}
static uint64_t submit_impl(uint64_t x) {return x+11;}
static uint64_t present_impl(uint64_t x) {return x+13;}
static uint64_t nvn_unimpl_entry(uint64_t x) {(void)x;return 0;}
static uint64_t nvn_big_entry(uint64_t x) {return ((uint64_t (*)(uint64_t))nvn_big_real[current_slot])(x);}
static uint64_t nvn_prof_entry(uint64_t x) {
    atomic_fetch_add(&wrappers,1);
    if(current_slot==0 && atomic_load(&pause_wrapper)){
        atomic_store(&entered_wrapper,1);while(atomic_load(&pause_wrapper))sched_yield();
    }
    return ((uint64_t (*)(uint64_t))nvn_real[current_slot])(x);
}
static uint64_t call_slot(unsigned slot,uint64_t x) {
    current_slot=slot;atomic_fetch_add(&nvn_slot[slot].cnt,1);
    void *fn=atomic_load_explicit(&nvn_slot[slot].fn,memory_order_acquire);
    return ((uint64_t (*)(uint64_t))fn)(x);
}
typedef struct {const char *name;void *fn;} nvn_impl;
static const nvn_impl implementations[] = {
    {"nvnCommandBufferDrawArrays",(void *)draw_impl},
    {"nvnQueueSubmitCommands",(void *)submit_impl},
    {"nvnQueuePresentTexture",(void *)present_impl}
};
static int _mh_execute_header,g_log;static unsigned g_nsem;
static void *getsectiondata(const void *header,const char *segment,const char *section,unsigned long *size){
    assert(header==&_mh_execute_header && !strcmp(segment,"__DATA") && !strcmp(section,"swnvn"));
    *size=sizeof implementations;return (void *)implementations;
}
static void die(const char *format,...) {(void)format;abort();}
'''+methods+nops+registration+seq+r'''
typedef struct {uint32_t magic,fenced;uint64_t fence;} sync_t;
typedef sync_t NVNsync;
typedef int NVNsyncWaitResult;
enum {NVN_SYNC_WAIT_RESULT_ALREADY_SIGNALED=0,NVN_SYNC_WAIT_RESULT_CONDITION_SATISFIED=1,NVN_SYNC_WAIT_RESULT_TIMEOUT_EXPIRED=2};
static int answers[2];static unsigned waits;static uint64_t timeouts[2];
static int nvn_fence_wait(uint64_t f,uint64_t timeout) {
    assert(f==17 && waits<2);timeouts[waits]=timeout;return answers[waits++];
}
#define NVN(name,ret,...) static ret h_##name(__VA_ARGS__)
'''+sync+r'''
static void assert_registration(int profiling) {
    assert(sizeof(nvn_slot_t)==16 && offsetof(nvn_slot_t,fn)==8);
    assert(g_nsem==3 && g_nvn_prof_ready);
    assert(nvn_real[0]==(void *)draw_impl && nvn_real[1]==(void *)nvn_big_entry);
    assert(nvn_big_real[1]==(void *)submit_impl && nvn_big_real[2]==(void *)present_impl);
    assert(!nvn_real[2] && !nvn_real[3]);
    assert(atomic_load(&nvn_slot[2].fn)==(void *)nvn_big_entry && atomic_load(&nvn_slot[3].fn)==(void *)nvn_unimpl_entry);
    for(int i=4;i<NVN_NFUNC;i++){
        assert(nvn_real[i]==(void *)nvn_nop && !nvn_big_real[i]);
        assert(atomic_load(&nvn_slot[i].fn)==(profiling?(void *)nvn_prof_entry:(void *)nvn_nop));
    }
    assert(atomic_load(&nvn_slot[0].fn)==(profiling?(void *)nvn_prof_entry:(void *)draw_impl));
    assert(atomic_load(&nvn_slot[1].fn)==(profiling?(void *)nvn_prof_entry:(void *)nvn_big_entry));
    assert(call_slot(0,9)==16 && call_slot(1,9)==20 && call_slot(2,9)==22 && call_slot(3,9)==0);
}
static void test_prepare_and_startup_off(void) {
    env_prof="0";nvn_nops_register();assert_registration(0);assert(!atomic_load(&g_nvn_prof));
    void *saved[NVN_NFUNC];memcpy(saved,nvn_real,sizeof saved);
    nvn_prof_prepare();assert(!memcmp(saved,nvn_real,sizeof saved));
    // Constructor ran before env.txt; a later lookup may enable startup profiling.
    env_prof="1";nvn_prof_install();assert_registration(1);assert(atomic_load(&g_nvn_prof));
    nvn_prof_prepare();assert(!memcmp(saved,nvn_real,sizeof saved));
}
static void test_constructor_on_launch_env_off(void) {
    env_prof="1";nvn_nops_register();assert_registration(1);
    void *saved[NVN_NFUNC];memcpy(saved,nvn_real,sizeof saved);nvn_thunk[0]=(void *)draw_impl;
    // Android reads env.txt after constructors: the next real proc lookup must
    // remove earlier wrappers when the launch config explicitly selects zero.
    env_prof="0";assert(nvn_lookup(nvn_names[0])==nvn_thunk[0]);assert_registration(0);
    assert(!atomic_load(&g_nvn_prof) && !g_nvn_prof_live && !memcmp(saved,nvn_real,sizeof saved));
    env_prof="1";assert(nvn_lookup(nvn_names[0])==nvn_thunk[0]);assert_registration(1);
    env_prof=NULL;assert(nvn_lookup(nvn_names[0])==nvn_thunk[0]);assert_registration(0);
    assert(!memcmp(saved,nvn_real,sizeof saved));
}
static void test_startup_on_and_live_override(void) {
    env_prof="1";nvn_nops_register();assert_registration(1);
    dm_nvn_runtime_env_changed("DM_NVN_PROF","0");assert_registration(0);
    // Even a stale startup environment cannot undo an explicit live choice.
    for(int i=0;i<100;i++)nvn_prof_install();assert_registration(0);assert(!atomic_load(&g_nvn_prof));
    dm_nvn_runtime_env_changed("DM_NVN_PROF","1");assert_registration(1);
    dm_nvn_runtime_env_changed("DM_NVN_PROF","0");assert_registration(0);
    dm_nvn_runtime_env_changed(NULL,"1");dm_nvn_runtime_env_changed("DM_NVN_PROF",NULL);dm_nvn_runtime_env_changed("IGNORED","1");
    assert_registration(0);
}
static void *caller(void *unused) {
    (void)unused;for(uint64_t x=0;x<30000;x++){assert(call_slot(0,x)==x+7);assert(call_slot(1,x)==x+11);}return NULL;
}
static void *toggle(void *unused) {
    (void)unused;for(int i=0;i<3000;i++)dm_nvn_runtime_env_changed("DM_NVN_PROF",i&1?"0":"1");return NULL;
}
static void *startup_install(void *unused) {(void)unused;for(int i=0;i<3000;i++)nvn_prof_install();return NULL;}
static void test_concurrent_toggling(void) {
    env_prof="1";nvn_nops_register();void *saved[NVN_NFUNC];memcpy(saved,nvn_real,sizeof saved);
    pthread_t rd[4],wr,install;for(int i=0;i<4;i++)assert(!pthread_create(&rd[i],NULL,caller,NULL));
    assert(!pthread_create(&wr,NULL,toggle,NULL));assert(!pthread_create(&install,NULL,startup_install,NULL));
    for(int i=0;i<4;i++)assert(!pthread_join(rd[i],NULL));assert(!pthread_join(wr,NULL));assert(!pthread_join(install,NULL));
    dm_nvn_runtime_env_changed("DM_NVN_PROF","0");assert_registration(0);nvn_prof_install();assert_registration(0);
    assert(!memcmp(saved,nvn_real,sizeof saved));
}
static void *held_caller(void *unused) {(void)unused;assert(call_slot(0,42)==49);return NULL;}
static void test_inflight_wrapper_target(void) {
    env_prof="1";nvn_nops_register();atomic_store(&pause_wrapper,1);pthread_t th;
    assert(!pthread_create(&th,NULL,held_caller,NULL));while(!atomic_load(&entered_wrapper))sched_yield();
    dm_nvn_runtime_env_changed("DM_NVN_PROF","0");assert(atomic_load(&nvn_slot[0].fn)==(void *)draw_impl && nvn_real[0]==(void *)draw_impl);
    atomic_store(&pause_wrapper,0);assert(!pthread_join(th,NULL));assert_registration(0);
}
static void test_seq_off_hotpath(void) {
    env_seq=NULL;atomic_store(&g_frame,100);assert(!seq_on());assert(atomic_load(&env_seq_reads)==1);
    for(int i=0;i<100000;i++)assert(!seq_on());
    assert(atomic_load(&env_seq_reads)==1 && !atomic_load(&clock_reads));
}
static void test_seq_startup_and_live_frames(void) {
    env_seq="3";atomic_store(&g_frame,100);assert(seq_on());assert(atomic_load(&g_seq_until)==103 && !atomic_load(&g_seq_request));
    assert(seq_on());atomic_store(&g_frame,102);assert(seq_on());atomic_store(&g_frame,103);assert(!seq_on());
    dm_nvn_runtime_env_changed("DM_NVN_SEQ","5");assert(seq_on());assert(atomic_load(&g_seq_until)==108);
    // 0 clears pending activation; it does not truncate an already-armed window.
    dm_nvn_runtime_env_changed("DM_NVN_SEQ","0");atomic_store(&g_frame,107);assert(seq_on());atomic_store(&g_frame,108);assert(!seq_on());
    dm_nvn_runtime_env_changed("DM_NVN_SEQ","4");dm_nvn_runtime_env_changed("DM_NVN_SEQ","0");assert(!seq_on());
    dm_nvn_runtime_env_changed("DM_NVN_SEQ","2");assert(seq_on());assert(atomic_load(&g_seq_until)==110);
    dm_nvn_runtime_env_changed("DM_NVN_SEQ","-1");atomic_store(&g_frame,110);assert(!seq_on());
    assert(atomic_load(&env_seq_reads)==1 && !atomic_load(&clock_reads));
}
static void test_seq_notification_before_first_call(void) {
    env_seq="100";atomic_store(&g_frame,20);dm_nvn_runtime_env_changed("DM_NVN_SEQ","2");
    assert(seq_on() && atomic_load(&g_seq_until)==22 && atomic_load(&env_seq_reads)==1);
    atomic_store(&g_frame,22);assert(!seq_on());
}
static void fence_answers(int first,int second) {answers[0]=first;answers[1]=second;waits=0;memset(timeouts,0xff,sizeof timeouts);}
static void test_syncwait_without_trace(void) {
    env_seq=NULL;sync_t s={0,0,17};fence_answers(0,0);
    assert(h_nvnSyncWait(&s,0)==NVN_SYNC_WAIT_RESULT_TIMEOUT_EXPIRED && !waits);
    s.fenced=1;fence_answers(1,0);assert(h_nvnSyncWait(&s,123)==NVN_SYNC_WAIT_RESULT_ALREADY_SIGNALED && waits==1 && !timeouts[0]);
    fence_answers(0,0);assert(h_nvnSyncWait(&s,0)==NVN_SYNC_WAIT_RESULT_TIMEOUT_EXPIRED && waits==2 && !timeouts[0] && !timeouts[1]);
    fence_answers(0,1);assert(h_nvnSyncWait(&s,0)==NVN_SYNC_WAIT_RESULT_CONDITION_SATISFIED && waits==2 && !timeouts[0] && !timeouts[1]);
    fence_answers(0,0);assert(h_nvnSyncWait(&s,123)==NVN_SYNC_WAIT_RESULT_TIMEOUT_EXPIRED && waits==2 && !timeouts[0] && timeouts[1]==123);
    fence_answers(0,1);assert(h_nvnSyncWait(&s,123)==NVN_SYNC_WAIT_RESULT_CONDITION_SATISFIED && waits==2 && !timeouts[0] && timeouts[1]==123);
    assert(!atomic_load(&clock_reads) && !atomic_load(&logs) && atomic_load(&env_seq_reads)==1);
}
static void test_syncwait_trace_and_disable(void) {
    env_seq="2";sync_t s={0,1,17};fence_answers(0,1);
    assert(h_nvnSyncWait(&s,99)==NVN_SYNC_WAIT_RESULT_CONDITION_SATISFIED && waits==2 && timeouts[1]==99);
    assert(atomic_load(&clock_reads)>=2 && atomic_load(&logs)==1);
    atomic_store(&g_frame,2);unsigned old_clock=atomic_load(&clock_reads),old_logs=atomic_load(&logs);
    for(int i=0;i<1000;i++){fence_answers(0,0);assert(h_nvnSyncWait(&s,0)==NVN_SYNC_WAIT_RESULT_TIMEOUT_EXPIRED);}
    assert(atomic_load(&clock_reads)==old_clock && atomic_load(&logs)==old_logs && atomic_load(&env_seq_reads)==1);
}
int main(int argc,char **argv) {
    assert(argc==2);
    if(!strcmp(argv[1],"prepare"))test_prepare_and_startup_off();
    else if(!strcmp(argv[1],"ctorenv"))test_constructor_on_launch_env_off();
    else if(!strcmp(argv[1],"startup"))test_startup_on_and_live_override();
    else if(!strcmp(argv[1],"concurrent"))test_concurrent_toggling();
    else if(!strcmp(argv[1],"inflight"))test_inflight_wrapper_target();
    else if(!strcmp(argv[1],"seqoff"))test_seq_off_hotpath();
    else if(!strcmp(argv[1],"seqframes"))test_seq_startup_and_live_frames();
    else if(!strcmp(argv[1],"seqfirst"))test_seq_notification_before_first_call();
    else if(!strcmp(argv[1],"syncoff"))test_syncwait_without_trace();
    else if(!strcmp(argv[1],"syncon"))test_syncwait_trace_and_disable();
    else assert(0);
    return 0;
}
''')
        result = subprocess.run(['clang','-std=gnu11','-O2','-Wall','-Wextra','-Werror','-Wno-unused-function','-pthread',str(src),'-o',str(cls.exe)],capture_output=True,text=True)
        if result.returncode:
            cls.tmp.cleanup()
            raise AssertionError(result.stderr)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def check(self, mode):
        result = subprocess.run([str(self.exe),mode],capture_output=True,text=True,timeout=15)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_prepare_after_registration_and_explicit_zero(self):
        self.check('prepare')

    def test_constructor_profile_on_then_launch_env_off_at_lookup(self):
        self.check('ctorenv')

    def test_startup_wrapper_and_live_override(self):
        self.check('startup')

    def test_concurrent_direct_wrapper_switching(self):
        self.check('concurrent')

    def test_inflight_wrapper_keeps_target(self):
        self.check('inflight')

    def test_seq_off_no_repeated_env_or_clock(self):
        self.check('seqoff')

    def test_seq_startup_live_n_frames_and_zero(self):
        self.check('seqframes')

    def test_seq_notification_before_first_use(self):
        self.check('seqfirst')

    def test_syncwait_results_and_no_clock_when_off(self):
        self.check('syncoff')

    def test_syncwait_trace_window_and_expiration(self):
        self.check('syncon')
