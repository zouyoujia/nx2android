"""Exercise the production wait debug gate without changing sleep semantics."""
from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.name != 'nt' and shutil.which('clang'), 'POSIX host clang needed')
class RuntimeWaitDebug(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = (ROOT/'nxnative/dm/host/hle_thread.c').read_text()
        start = source.index('static pthread_mutex_t g_wait_bt_mu')
        end = source.index('static int svc_thread_impl(', start)
        production = source[start:end]
        cls.tmp = tempfile.TemporaryDirectory(prefix='nx2a-wait-debug-')
        cls.exe = Path(cls.tmp.name)/'check'
        src = Path(cls.tmp.name)/'check.c'
        src.write_text(r'''
#include <assert.h>
#include <stdint.h>
#include <stdatomic.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static const char *startup;
static void yield_census_request(const char *value) {(void)value;}
static _Atomic unsigned env_reads, locks, unlocks, logs, backtraces;
static _Atomic uint64_t last_fp; static _Thread_local unsigned held_config;
static char last_what[64]; static double last_ms;
static char *observed_getenv(const char *name) {
    assert(!strcmp(name,"DM_SLEEP_BT"));atomic_fetch_add(&env_reads,1);return (char *)startup;
}
static int observed_lock(pthread_mutex_t *m) {atomic_fetch_add(&locks,1);int r=pthread_mutex_lock(m);if(!r)held_config++;return r;}
static int observed_unlock(pthread_mutex_t *m) {atomic_fetch_add(&unlocks,1);assert(held_config);held_config--;return pthread_mutex_unlock(m);}
static void observed_log(const char *format,...);
void dm_backtrace(uint64_t fp) {atomic_fetch_add(&backtraces,1);atomic_store(&last_fp,fp);}
#define getenv observed_getenv
#define pthread_mutex_lock observed_lock
#define pthread_mutex_unlock observed_unlock
#define LOG observed_log
'''+production+r'''
#undef getenv
#undef pthread_mutex_lock
#undef pthread_mutex_unlock
static void observed_log(const char *format,...) {
    // Never hold the configuration lock while invoking diagnostics.
    assert(!held_config);
    va_list ap;va_start(ap,format);const char *what=va_arg(ap,const char *);double ms=va_arg(ap,double);va_end(ap);
    assert(!strcmp(format,"%s %.2f ms"));assert(what && ms>=0);
    // Only single-thread cases inspect these last-value fields.
    if (!strcmp(what,"single")) {snprintf(last_what,sizeof last_what,"%s",what);last_ms=ms;}
    atomic_fetch_add(&logs,1);
}
static unsigned nlogs(void) {return atomic_load(&logs);}
static void one(int64_t ns) {wait_bt("single",ns,0x1234);}
static void test_startup_off(void) {
    startup=NULL;one(2000000);assert(atomic_load(&env_reads)==1 && atomic_load(&g_wait_bt_state)==1);
    unsigned l=atomic_load(&locks),u=atomic_load(&unlocks);
    for(int i=0;i<100000;i++)one(i);
    assert(atomic_load(&env_reads)==1 && atomic_load(&locks)==l && atomic_load(&unlocks)==u && !nlogs());
    dm_thread_runtime_env_changed(NULL,"1-2");dm_thread_runtime_env_changed("IGNORED","1-2");
    assert(atomic_load(&locks)==l && atomic_load(&g_wait_bt_state)==1);
}
static void test_startup_range(void) {
    startup="1.5-2.5";one(1499999);assert(!nlogs());one(1500000);one(2500000);one(2500001);
    assert(nlogs()==2 && atomic_load(&env_reads)==1 && !strcmp(last_what,"single") && last_ms==2.5 && atomic_load(&last_fp)==0x1234);
    for(int i=0;i<100;i++)one(2000000);
    assert(nlogs()==64 && atomic_load(&backtraces)==64 && g_wait_bt_count==64 && atomic_load(&env_reads)==1);
    // Default upper bound when no explicit max is supplied.
    dm_thread_runtime_env_changed("DM_SLEEP_BT","3");one(2999999);one(3000000);one(1000000000000000ll);one(1000000000000001ll);
    assert(nlogs()==66 && g_wait_bt_count==2);
}
static void test_live_ranges_and_reset(void) {
    startup="9-10";
    // A notification before first use wins over the launch environment.
    dm_thread_runtime_env_changed("DM_SLEEP_BT","1-2");one(1000000);assert(nlogs()==1 && !atomic_load(&env_reads));
    for(int i=0;i<100;i++)one(2000000);assert(nlogs()==64);
    dm_thread_runtime_env_changed("DM_SLEEP_BT","1-2");one(1500000);assert(nlogs()==64); // same range keeps its budget
    dm_thread_runtime_env_changed("DM_SLEEP_BT","2-3");one(1999999);one(2000000);one(3000000);one(3000001);assert(nlogs()==66 && g_wait_bt_count==2);
    dm_thread_runtime_env_changed("DM_SLEEP_BT","");unsigned l=atomic_load(&locks),u=atomic_load(&unlocks);
    for(int i=0;i<100000;i++)one(2500000);
    assert(nlogs()==66 && atomic_load(&locks)==l && atomic_load(&unlocks)==u && !atomic_load(&env_reads));
    dm_thread_runtime_env_changed("DM_SLEEP_BT","2-3");one(2500000);assert(nlogs()==67 && g_wait_bt_count==3); // empty off preserves old last-range semantics
    dm_thread_runtime_env_changed("DM_SLEEP_BT","0");one(1000000);assert(nlogs()==67 && atomic_load(&g_wait_bt_state)==1);
    dm_thread_runtime_env_changed("DM_SLEEP_BT","1-2");one(1000000);assert(nlogs()==68 && g_wait_bt_count==1);
    dm_thread_runtime_env_changed("IGNORED","5-6");one(1500000);assert(nlogs()==69);
    dm_thread_runtime_env_changed("DM_SLEEP_BT",NULL);one(1500000);assert(nlogs()==69 && atomic_load(&g_wait_bt_state)==1);
}
static void *initial_reader(void *unused) {(void)unused;for(int i=0;i<10000;i++)wait_bt("worker",1000000,7);return NULL;}
static void test_concurrent_initialization(void) {
    startup=NULL;pthread_t th[8];for(int i=0;i<8;i++)assert(!pthread_create(&th[i],NULL,initial_reader,NULL));
    for(int i=0;i<8;i++)assert(!pthread_join(th[i],NULL));
    assert(atomic_load(&env_reads)==1 && !nlogs());
}
static _Atomic int stop;
static void *live_reader(void *unused) {(void)unused;while(!atomic_load(&stop)) {wait_bt("worker",1000000,9);wait_bt("worker",2000000,9);wait_bt("worker",3000000,9);}return NULL;}
static void *config_writer(void *unused) {
    (void)unused;const char *values[]={"1-1","3-3","", "2-2", "0"};
    for(int i=0;i<5000;i++)dm_thread_runtime_env_changed("DM_SLEEP_BT",values[i%5]);
    return NULL;
}
static void test_concurrent_live_config(void) {
    // All range/count access shares one lock, and logging never runs under it.
    dm_thread_runtime_env_changed("DM_SLEEP_BT","1-1");pthread_t rd[4],wr[2];
    for(int i=0;i<4;i++)assert(!pthread_create(&rd[i],NULL,live_reader,NULL));
    for(int i=0;i<2;i++)assert(!pthread_create(&wr[i],NULL,config_writer,NULL));
    for(int i=0;i<2;i++)assert(!pthread_join(wr[i],NULL));atomic_store(&stop,1);
    for(int i=0;i<4;i++)assert(!pthread_join(rd[i],NULL));
    assert(!atomic_load(&env_reads) && atomic_load(&backtraces)==nlogs());
    dm_thread_runtime_env_changed("DM_SLEEP_BT","4-4");unsigned before=nlogs();
    one(3000000);one(5000000);for(int i=0;i<100;i++)one(4000000);
    assert(nlogs()==before+64 && g_wait_bt_count==64);
    dm_thread_runtime_env_changed("DM_SLEEP_BT","");unsigned l=atomic_load(&locks);
    for(int i=0;i<100000;i++)one(4000000);assert(atomic_load(&locks)==l && nlogs()==before+64);
}
int main(int argc,char **argv) {
    assert(argc==2);
    if(!strcmp(argv[1],"off"))test_startup_off();
    else if(!strcmp(argv[1],"range"))test_startup_range();
    else if(!strcmp(argv[1],"live"))test_live_ranges_and_reset();
    else if(!strcmp(argv[1],"init"))test_concurrent_initialization();
    else if(!strcmp(argv[1],"concurrent"))test_concurrent_live_config();
    else assert(0);
    return 0;
}
''')
        result = subprocess.run(['clang','-std=gnu11','-O2','-Wall','-Wextra','-Werror','-pthread',str(src),'-o',str(cls.exe)],capture_output=True,text=True)
        if result.returncode:
            cls.tmp.cleanup()
            raise AssertionError(result.stderr)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def check(self, mode):
        result = subprocess.run([str(self.exe), mode], capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)

    def test_startup_off_hotpath(self):
        self.check('off')

    def test_startup_range_and_limit(self):
        self.check('range')

    def test_live_ranges_and_counter_reset(self):
        self.check('live')

    def test_concurrent_startup_once(self):
        self.check('init')

    def test_concurrent_live_configuration(self):
        self.check('concurrent')
