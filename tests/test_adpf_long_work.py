"""Check real long-cycle reporting: CPU execution, lifecycle epochs and API fallback."""
from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.name != 'nt' and shutil.which('clang'), 'POSIX host clang needed')
class LongWorkHints(unittest.TestCase):
    def test_completed_cycles_and_lifecycle(self):
        source = (ROOT / 'nxnative/dm/android/perf_android.c').read_text()
        epoch = source[source.index('void dm_perf_activity_changed(void)'):source.index('// The long-work experiment')]
        functions = source[source.index('typedef struct { int32_t tid; uint64_t ns; } hint_cpu_sample;'):
                           source.index('static void *hint_long_watch(void *arg)')]
        harness = r'''
#include <assert.h>
#include <stdint.h>
#include <limits.h>
#include <pthread.h>
#include <time.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
static pthread_mutex_t g_long_mu=PTHREAD_MUTEX_INITIALIZER;
static uint64_t g_activity_epoch;
static int32_t g_hint_tids[8]={7,8};static int g_hint_n=2;
static int focus=1,menu=-1,window=1,detclock,fail_tid,bad_nsec,reports;
static uint64_t cpu[3]={100000000,200000000,10000000000ull};
typedef struct {int unused;} ANativeWindow;
int dm_focus_now(void){return focus;}int dm_menu_state(void){return menu;}
ANativeWindow *dm_front_window(void){return window?(ANativeWindow *)1:NULL;}
int dm_detclock_enabled(void){return detclock;}
static void test_log(const char *fmt,...){(void)fmt;}
#define LOG test_log
static int test_clock_gettime(clockid_t c,struct timespec *t) {
    int tid=0;for(int i=7;i<=9;i++)if(c==(clockid_t)(((~(uint32_t)i)<<3)|6u))tid=i;
    assert(tid>=7 && tid<=9);if(tid==fail_tid)return -1;
    t->tv_sec=(time_t)(cpu[tid-7]/1000000000ull);t->tv_nsec=bad_nsec?-1:(long)(cpu[tid-7]%1000000000ull);return 0;
}
#define clock_gettime test_clock_gettime
#define gettid() 7
typedef struct {int64_t start,total,cpu,gpu;} AWorkDuration;
static AWorkDuration work,*g_wd=&work;
static void *g_hint=(void *)1;
static void set_start(AWorkDuration*d,int64_t n){d->start=n;}
static void set_total(AWorkDuration*d,int64_t n){d->total=n;}
static void set_cpu(AWorkDuration*d,int64_t n){d->cpu=n;}
static void set_gpu(AWorkDuration*d,int64_t n){d->gpu=n;}
static int report(void*h,AWorkDuration*d){assert(h==g_hint && d==g_wd);reports++;return 0;}
static struct {int(*report2)(void*,AWorkDuration*);void(*wd_start)(AWorkDuration*,int64_t);void(*wd_total)(AWorkDuration*,int64_t);void(*wd_cpu)(AWorkDuration*,int64_t);void(*wd_gpu)(AWorkDuration*,int64_t);} H={report,set_start,set_total,set_cpu,set_gpu};
'''+epoch+functions+r'''
static void end(int64_t now){pthread_mutex_lock(&g_long_mu);hint_long_end(now);pthread_mutex_unlock(&g_long_mu);}
static void begin(void){focus=1;menu=-1;window=1;detclock=0;fail_tid=bad_nsec=0;reports=0;H.report2=report;hint_long_begin(1000000000);}
static void progress(void){cpu[0]+=200000000;cpu[1]+=300000000;}
int main(void){
    begin();progress();end(2000000000);
    assert(reports==1 && work.start==1000000000 && work.total==1000000000 && work.cpu==500000000 && work.gpu==0);
    begin();end(2000000000);assert(!reports); // blocked/idle wall time is not CPU work
    begin();cpu[0]+=49000000;end(2000000000);assert(!reports);
    begin();cpu[0]+=50000000;end(2000000000);assert(reports==1);
    begin();progress();dm_perf_activity_changed();end(2000000000);assert(!reports); // even a completed pause/resume is rejected
    begin();progress();menu=0;end(2000000000);assert(!reports);
    begin();progress();focus=2;end(2000000000);assert(!reports);
    begin();progress();window=0;end(2000000000);assert(!reports);
    begin();progress();detclock=1;end(2000000000);assert(!reports);
    begin();progress();H.report2=NULL;end(2000000000);assert(!reports); // no wall-time-only fallback
    begin();progress();fail_tid=8;end(2000000000);assert(!reports); // unknown CPU must not become fabricated CPU
    begin();progress();bad_nsec=1;end(2000000000);assert(!reports);
    begin();progress();end(1249999999);assert(!reports); // ordinary frame path remains separate
    begin();progress();end(1250000000);assert(reports==1);
    hint_cpu_sample old[]={{7,100},{8,200}},cur[]={{7,150},{8,180},{9,UINT64_MAX}};
    assert(hint_cpu_delta(old,2,cur,3)==50); // regressions and new TIDs do not credit lifetime CPU
    hint_cpu_sample ov[]={{7,0},{8,0}},nc[]={{7,UINT64_MAX},{8,1}};
    assert(hint_cpu_delta(ov,2,nc,2)==UINT64_MAX);
    assert(!hint_long_allowed(1,100,0,0,1,100));
    assert(!hint_long_allowed(1000000000,2000000000,1,2,1,500000000));
    int episode=0;assert(!hint_long_notify(&episode,0));assert(hint_long_notify(&episode,1));
    for(int i=0;i<20;i++)assert(!hint_long_notify(&episode,1));
    uint64_t ns=123;assert(!hint_cpu_time(0,&ns) && ns==123);
    fail_tid=0;bad_nsec=0;assert(hint_cpu_time(7,&ns) && ns==cpu[0]);
    puts("long-work reporting passed");return 0;
}
'''
        with tempfile.TemporaryDirectory(prefix='nx2a-long-work-') as tmp:
            src, exe = Path(tmp) / 'check.c', Path(tmp) / 'check'
            src.write_text(harness)
            result = subprocess.run(['clang', '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror', '-pthread', str(src), '-o', str(exe)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run([str(exe)], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
