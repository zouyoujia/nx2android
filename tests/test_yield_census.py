"""Bounded yield-site observation with unchanged production yield decisions."""
from pathlib import Path
import os, shutil, subprocess, tempfile, unittest
from tests.test_texture_barrier import function
ROOT=Path(__file__).resolve().parents[1]
HOST=ROOT/'nxnative/dm/host'

@unittest.skipUnless(os.name!='nt' and shutil.which('clang'),'POSIX clang needed')
class YieldCensus(unittest.TestCase):
    def run_c(self, code):
        with tempfile.TemporaryDirectory(prefix='nx2a-yield-census-') as tmp:
            src,exe=Path(tmp)/'check.c',Path(tmp)/'check';src.write_text(code)
            p=subprocess.run(['clang','-std=gnu11','-O2','-Wall','-Wextra','-Werror','-pthread','-I'+str(HOST),str(src),'-o',str(exe)],capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stderr)
            p=subprocess.run([str(exe)],capture_output=True,text=True,timeout=10)
            self.assertEqual(p.returncode,0,p.stdout+p.stderr)

    def test_budget_histogram_overflow_snapshots_and_stop(self):
        self.run_c(r'''
#include <assert.h>
#include "yield_census.h"
int main(void){
 assert(!dm_yc_budget(NULL) && !dm_yc_budget("") && !dm_yc_budget("-1") && !dm_yc_budget(" -1") && !dm_yc_budget("4x") && !dm_yc_budget("9999999999999999999999999"));
 assert(dm_yc_budget("4096")==4096 && dm_yc_budget("99999")==DM_YC_MAX);
 dm_yc_state s={0};assert(!dm_yc_note(&s,NULL,0,0,0,0,0,0));
 uint64_t r[31]={0};r[30]=0x1234;r[19]=9;dm_yc_start(&s,3);
 assert(!dm_yc_note(&s,r,r[30],10,20,4,0,1));r[19]=10;r[0]=(uint64_t)-1;
 assert(!dm_yc_note(&s,r,r[30],25,40,7,1,0));assert(dm_yc_note(&s,r,r[30],45,70,8,0,0));
 assert(s.samples==3 && s.nsite==1 && !s.remaining && s.begin==10 && s.end==70 && s.elapsed==50);
 assert(s.site[0].count==3 && s.site[0].moved==1 && s.site[0].slept==1 && s.site[0].max_elapsed==25 && s.site[0].max_run==8);
 assert(s.site[0].first[0]==9 && s.site[0].last[0]==10 && s.site[0].first_ns==0 && s.site[0].last_ns==-1 && s.site[0].zero==1 && s.site[0].negative==2 && !s.site[0].positive);
 for(int i=0;i<10000;i++)assert(!dm_yc_note(&s,NULL,0,0,0,0,0,0));assert(s.samples==3);
 dm_yc_start(&s,20);for(unsigned i=0;i<20;i++){r[30]=i+1;assert(dm_yc_note(&s,r,r[30],i*2,i*2+1,0,0,0)==(i==19));}
 assert(s.samples==20 && s.nsite==DM_YC_SITES && s.overflow==4 && s.elapsed==20);
 dm_yc_start(&s,0);assert(!s.samples && !s.nsite && !s.remaining && !dm_yc_note(&s,NULL,0,0,0,0,0,0));
 dm_yc_start(&s,~0u);assert(s.remaining==DM_YC_MAX);return 0;
}
''')

    def test_live_request_rearm_cancel_and_original_yield_behavior(self):
        source=(HOST/'hle_thread.c').read_text()
        block=source[source.index('static _Atomic int g_yc_request'):source.index('static uint64_t g_next_id')]
        self.assertIn('if (me->handle == 0x1001)',function(source,'thr_yield'))
        self.assertIn('getenv("DM_YIELD_CENSUS")',function(source,'svc_thread_main_init'))
        self.assertIn('yield_census_request(value)',function(source,'dm_thread_runtime_env_changed'))
        self.assertIn('sub sp, sp, #800',(HOST/'entry.S').read_text())
        self.assertIn('emit(STP_PRE(29, 30, 31, -16))',(ROOT/'nxnative/dm/tools/patch_nso.py').read_text())
        self.run_c(r'''
#include <assert.h>
#include <stdatomic.h>
#include <stdarg.h>
#include <stdio.h>
#include <pthread.h>
#include "yield_census.h"
typedef struct {const char *name;void *base;} module_t;
static unsigned lookups,logs,complete;
static module_t *module_of(void *p){(void)p;lookups++;return NULL;}
static void logger(const char *format,...){logs++;if(strstr(format,"complete:"))complete++;}
#define LOG logger
typedef struct {uint32_t handle;uint64_t nyield,nbackoff,nsvc,yrun_svc,yrun_sig,yrun_t0;uint64_t *_Atomic svc_regs;} thr_t;
'''+block+r'''
static unsigned yields,clock_reads;static uint64_t g_main_yield_ns,g_main_yield_n,clock_ns;
static uint64_t now_ns(void){clock_reads++;return clock_ns+=100;}
static int observed_yield(void){yields++;return 0;}
#define sched_yield observed_yield
static void yield_for(thr_t *t,uint64_t run){(void)run;t->nbackoff++;yields++;}
'''+function(source,'thr_yield')+r'''
int main(void){
 // Disabled observation neither reads registers nor resolves modules/logs.
 for(int i=0;i<100000;i++)yield_census_note(NULL,NULL,0,0,0,0,0);
 assert(!logs && !lookups && !g_yc.samples);
 uint64_t r[102]={0};r[30]=0x1234;r[101]=0x4567;thr_t t={.handle=0x1001,.nsvc=1,.svc_regs=r};
 // Run the actual scheduler function twice off and twice on from identical state.
 thr_yield(&t,r);t.nsvc++;thr_yield(&t,r);unsigned off_yields=yields,off_clocks=clock_reads;thr_t off=t;
 memset(&t,0,sizeof t);t.handle=0x1001;t.nsvc=1;atomic_store(&t.svc_regs,r);yields=clock_reads=0;clock_ns=0;
 yield_census_request("2");thr_yield(&t,r);t.nsvc++;thr_yield(&t,r);
 assert(complete==1 && g_yc.samples==2 && !g_yc.remaining && g_yc.nsite==1 && !g_yc.unlocated && g_yc.site[0].lr==0x4567 && g_yc.site[0].gate==0x1234);
 assert(yields==off_yields && clock_reads==off_clocks && !memcmp(&off,&t,sizeof t));
 unsigned done=complete;yield_census_note(&t,r,0,1,0,0,0);assert(complete==done);
 yield_census_request("3");yield_census_note(&t,r,0,1,0,0,0);assert(g_yc.remaining==2);
 yield_census_request("0");yield_census_note(NULL,NULL,0,0,0,0,0);assert(!g_yc.remaining && !g_yc.samples && complete==done);
 yield_census_request("1");yield_census_note(&t,r,0,1,0,0,0);assert(complete==done+1);
 // Worker yields are unchanged and never enter the main-thread census.
 yield_census_request("1");t.handle=0x1002;t.nsvc++;thr_yield(&t,r);assert(atomic_load(&g_yc_request)==1);
 yield_census_note(&t,r,0,1,0,0,0);assert(complete==done+2);
    // An arbitrary array or stale frame cannot be read as an extended gate frame.
    atomic_store(&t.svc_regs,NULL);uint64_t short_regs[31]={0};short_regs[30]=0x9999;
    yield_census_request("1");yield_census_note(&t,short_regs,0,1,0,0,0);
    assert(g_yc.unlocated==1 && !g_yc.site[0].lr && g_yc.site[0].gate==0x9999);
    return 0;
}
''')
