"""Exercise production completion queries, publication, and safe slot retirement."""
from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.name != 'nt' and shutil.which('clang'), 'POSIX host clang needed')
class CompletionProgress(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = (ROOT / 'nxnative/dm/android/gfx_vk_core.c').read_text()
        progress = source[source.index('static uint64_t g_done, g_done_at, g_slot_ready;'):
                          source.index('static void slots_init(void) {')]
        retire = source[source.index('static void completion_retire(cpend *c)'):
                        source.index('static void *completion_main(void *a)')]
        waits = source[source.index('int mtl_wait(uint64_t fence, uint64_t timeout_ns) {'):]
        harness = r'''
#include <assert.h>
#include <stdint.h>
#include <pthread.h>
#include <stdatomic.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sched.h>
#define NSLOT 6
#define VKCHK(x) assert((x)==0)
#define VKLOG(...) do {if(0)fprintf(stderr,__VA_ARGS__);} while(0)
enum {SP_SLOT,SP_WAIT,SP_WAKE,SP_RESET};
static _Atomic unsigned observed_clocks,observed_locks,observed_unlocks,observed_waits,observed_broadcasts;
static int counted_clock(clockid_t c,struct timespec *t) {atomic_fetch_add(&observed_clocks,1);return clock_gettime(c,t);}
static int counted_lock(pthread_mutex_t *m) {atomic_fetch_add(&observed_locks,1);return pthread_mutex_lock(m);}
static int counted_unlock(pthread_mutex_t *m) {atomic_fetch_add(&observed_unlocks,1);return pthread_mutex_unlock(m);}
static int counted_wait(pthread_cond_t *c,pthread_mutex_t *m,const struct timespec *t) {atomic_fetch_add(&observed_waits,1);return pthread_cond_timedwait(c,m,t);}
static int counted_broadcast(pthread_cond_t *c) {atomic_fetch_add(&observed_broadcasts,1);return pthread_cond_broadcast(c);}
static int g_sync_prof=1;
static uint64_t sync_clock(void) {if(!g_sync_prof)return 0;struct timespec t;counted_clock(CLOCK_MONOTONIC,&t);return (uint64_t)t.tv_sec*1000000000+t.tv_nsec;}
static void sync_add(int kind,uint64_t begin) {(void)kind;(void)begin;}
static pthread_mutex_t g_done_mu=PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_done_cv=PTHREAD_COND_INITIALIZER;
static pthread_mutex_t gate_mu=PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t gate_cv=PTHREAD_COND_INITIALIZER;
static int post_allowed,reset_entered,reset_allowed,resets;
static _Atomic int visible,cache_saved;
static uint64_t g_fence;
static int g_prereset;
static void vk_gpu_segments_request(int n){(void)n;}
typedef struct {uint64_t fence;int nmain_alloc,nup_alloc,nmain_used,nup_used;int mains[2],ups[2];} vksub_t;
typedef struct {uint64_t fence;int rslot;} cpend;
static vksub_t g_slot[NSLOT];
enum {VK_EFF_RESETS,VK_EFF_UNUSED_RESETS};
static void efficiency_add(int k,uint64_t n){(void)k;(void)n;}
static int vkResetCommandBuffer(int cb,int flags) {
    assert(cb>0 && !flags);
    pthread_mutex_lock(&gate_mu);resets++;reset_entered=1;pthread_cond_broadcast(&gate_cv);
    while(!reset_allowed)pthread_cond_wait(&gate_cv,&gate_mu);
    pthread_mutex_unlock(&gate_mu);return 0;
}
static void vk_pipeline_cache_save(void) {atomic_store(&cache_saved,1);}
/* Instrument only the extracted production helpers, not fixture synchronization. */
#define clock_gettime counted_clock
#define pthread_mutex_lock counted_lock
#define pthread_mutex_unlock counted_unlock
#define pthread_cond_timedwait counted_wait
#define pthread_cond_broadcast counted_broadcast
'''+progress+retire+waits+r'''
#undef clock_gettime
#undef pthread_mutex_lock
#undef pthread_mutex_unlock
#undef pthread_cond_timedwait
#undef pthread_cond_broadcast
typedef struct {unsigned clocks,locks,unlocks,waits,broadcasts;} observed;
static observed observations(void) {
    return (observed){atomic_load(&observed_clocks),atomic_load(&observed_locks),
        atomic_load(&observed_unlocks),atomic_load(&observed_waits),atomic_load(&observed_broadcasts)};
}
static void assert_no_waiting_calls(observed before) {
    observed after=observations();
    assert(before.clocks==after.clocks && before.locks==after.locks &&
        before.unlocks==after.unlocks && before.waits==after.waits &&
        before.broadcasts==after.broadcasts);
}
typedef struct {uint64_t fence;int reusable,ok;_Atomic int returned;} waiter;
static void *wait_thread(void *p) {waiter *w=p;w->ok=completion_wait(w->fence,2000000000ull,w->reusable);atomic_store(&w->returned,1);return NULL;}
static void *drain_thread(void *p) {(void)p;mtl_drain();return NULL;}
static void *complete_thread(void *p) {
    pthread_mutex_lock(&gate_mu);while(!post_allowed)pthread_cond_wait(&gate_cv,&gate_mu);pthread_mutex_unlock(&gate_mu);
    atomic_store(&visible,42); // copyback/query/event/counter phase precedes retirement
    completion_retire(p);return NULL;
}
static void test_fast_queries(void) {
    assert(g_poll_fast==1);
    observed before=observations();
    assert(!completion_wait(1,0,0) && !completion_wait(1,0,1));
    assert(completion_wait(0,0,0) && completion_wait(0,0,1));
    assert_no_waiting_calls(before);
    completion_publish(5,1,0);
    before=observations();
    assert(completion_wait(5,0,0) && !completion_wait(5,0,1));
    assert_no_waiting_calls(before);
    completion_publish(5,0,1);
    before=observations();
    assert(completion_wait(5,0,0) && completion_wait(5,0,1));
    assert_no_waiting_calls(before);
    assert(g_poll_zero==8 && g_poll_fast_queries==8 && g_poll_pending==3);
    assert(!g_poll_legacy_cond && !g_poll_positive && !g_poll_zero_ns && !g_poll_positive_ns);
}
static void test_diagnostic_off(void) {
    g_sync_prof=0;observed before=observations();
    for(int i=0;i<100000;i++){assert(!completion_wait(1,0,0));assert(completion_wait(0,0,1));}
    assert_no_waiting_calls(before);
    assert(!g_poll_zero && !g_poll_fast_queries && !g_poll_pending && !g_poll_positive);
    completion_publish(5,1,0);before=observations();assert(completion_wait(5,0,0) && !completion_wait(5,0,1));assert_no_waiting_calls(before);
    assert(!g_poll_zero && !g_poll_fast_queries && !g_poll_pending);
    // Restoring the legacy path changes waits, while diagnostics remain off.
    dm_runtime_env_changed("DM_VK_POLL_FAST","0");assert(!completion_wait(6,0,0));
    assert(!g_poll_zero && !g_poll_pending && !g_poll_legacy_cond);
    assert(!completion_wait(6,1000,0));assert(!g_poll_positive && !g_poll_positive_blocked && !g_poll_positive_ns);
}
static void test_legacy_rollback(void) {
    dm_runtime_env_changed("DM_VK_POLL_FAST","0");
    observed before=observations();
    assert(!completion_wait(1,0,0));
    observed after=observations();
    assert(after.clocks>before.clocks && after.locks>before.locks &&
        after.unlocks>before.unlocks && after.waits>before.waits);
    assert(g_poll_zero==1 && g_poll_pending==1 && g_poll_legacy_cond==1 && !g_poll_fast_queries);
    completion_publish(1,1,0);
    before=observations();
    assert(completion_wait(1,0,0));
    assert(atomic_load(&observed_waits)==before.waits);
    dm_runtime_env_changed("DM_VK_POLL_FAST","1");
    before=observations();
    assert(!completion_wait(2,0,0));
    assert_no_waiting_calls(before);
    assert(g_poll_zero==3 && g_poll_pending==2 && g_poll_legacy_cond==1 && g_poll_fast_queries==1);
    assert(!g_poll_positive);
}
static void test_positive_wait(void) {
    waiter game={.fence=17};pthread_t thread;
    unsigned before=atomic_load(&observed_waits);
    assert(!pthread_create(&thread,NULL,wait_thread,&game));
    while(atomic_load(&observed_waits)==before && !atomic_load(&game.returned))sched_yield();
    assert(atomic_load(&observed_waits)>before && !atomic_load(&game.returned));
    completion_publish(17,1,0);
    assert(!pthread_join(thread,NULL) && game.ok);
    assert(g_poll_positive==1 && g_poll_positive_blocked==1);
    assert(!g_poll_zero && !g_poll_fast_queries && !g_poll_legacy_cond);
    before=atomic_load(&observed_waits);
    assert(!completion_wait(18,1000000,0));
    assert(atomic_load(&observed_waits)>before);
    assert(g_poll_positive==2 && g_poll_positive_blocked==2);
}
static int done_payload,ready_payload;
static _Atomic int done_ack,ready_ack;
static void *publication_writer(void *unused) {
    (void)unused;
    for(int i=1;i<=1000;i++) {
        while(atomic_load(&ready_ack)<i-1)sched_yield();
        done_payload=i;completion_publish((uint64_t)i,1,0);
        while(atomic_load(&done_ack)<i)sched_yield();
        ready_payload=i;completion_publish((uint64_t)i,0,1);
    }
    return NULL;
}
static void test_publication_visibility(void) {
    pthread_t writer;assert(!pthread_create(&writer,NULL,publication_writer,NULL));
    for(int i=1;i<=1000;i++) {
        while(!completion_wait((uint64_t)i,0,0))sched_yield();
        assert(done_payload==i);
        /* No independent atomic payload/notification supplies the reader's visibility. */
        assert(!completion_wait((uint64_t)i,0,1));
        atomic_store(&done_ack,i);
        while(!completion_wait((uint64_t)i,0,1))sched_yield();
        assert(ready_payload==i);
        atomic_store(&ready_ack,i);
    }
    assert(!pthread_join(writer,NULL));
    assert(g_done==1000 && g_slot_ready==1000);
}
static void run(int early,int prereset,int slot,uint64_t f,int flip_early) {
    g_early_complete=early;g_prereset=prereset;g_fence=f;
    g_done=g_slot_ready=0;g_done_at=0;
    post_allowed=reset_entered=resets=0;reset_allowed=0;
    atomic_store(&visible,0);atomic_store(&cache_saved,0);
    if(slot>=0){g_slot[slot]=(vksub_t){.fence=f,.nmain_alloc=1,.nup_alloc=1,.nmain_used=1,.nup_used=1,.mains={1},.ups={2}};}
    cpend c={f,slot};waiter game={.fence=f},reuse={.fence=f,.reusable=1};pthread_t complete,game_t,reuse_t,drain;
    assert(!pthread_create(&complete,NULL,complete_thread,&c));
    assert(!pthread_create(&game_t,NULL,wait_thread,&game));
    assert(!pthread_create(&reuse_t,NULL,wait_thread,&reuse));
    // Neither wait can return while CPU-visible effects are still unpublished.
    assert(!completion_wait(f,0,0) && !completion_wait(f,0,1));assert(!atomic_load(&visible));
    pthread_mutex_lock(&gate_mu);post_allowed=1;pthread_cond_broadcast(&gate_cv);
    if(prereset && slot>=0)while(!reset_entered)pthread_cond_wait(&gate_cv,&gate_mu);
    pthread_mutex_unlock(&gate_mu);
    if(prereset && slot>=0) {
        assert(completion_wait(f,0,0)==early);
        assert(!completion_wait(f,0,1) && atomic_load(&visible)==42);
        if(early) {assert(!pthread_join(game_t,NULL) && game.ok);}
        // An in-progress retirement owns its early snapshot across live changes.
        if(flip_early)dm_runtime_env_changed("DM_VK_EARLY_COMPLETE",early?"0":"1");
        assert(!pthread_create(&drain,NULL,drain_thread,NULL));
        assert(!atomic_load(&cache_saved));
        // Other free slots can proceed, but ring wrap cannot reuse this one.
        for(int i=0;i<NSLOT;i++)if(i!=slot)assert(completion_wait(0,0,1));
        assert(!completion_wait(c.fence,0,1));
        pthread_mutex_lock(&gate_mu);reset_allowed=1;pthread_cond_broadcast(&gate_cv);pthread_mutex_unlock(&gate_mu);
        assert(!pthread_join(drain,NULL));assert(atomic_load(&cache_saved));
    }
    assert(!pthread_join(complete,NULL));
    if(!(early && prereset && slot>=0))assert(!pthread_join(game_t,NULL));
    assert(!pthread_join(reuse_t,NULL));assert(game.ok && reuse.ok);
    assert(atomic_load(&visible)==42 && g_done==f && g_slot_ready==f);
    assert(resets==(prereset && slot>=0?2:0));
    completion_publish(f-1,1,1);assert(g_done==f && g_slot_ready==f);
    assert(!completion_wait(f+1,1000000,0) && !completion_wait(f+1,1000000,1));
}
static void test_ordering(void) {
    for(int early=0;early<2;early++) {
        for(int i=0;i<NSLOT*2;i++)run(early,1,i%NSLOT,(uint64_t)i+1,0);
        run(early,0,0,50,0); // no-prereset still waits for visible effects
        run(early,1,-1,51,0); // no slot to reset
        run(early,1,0,52,1); // flip the live policy while reset is deliberately blocked
    }
}
int main(int argc,char **argv) {
    assert(argc==2);
    if(!strcmp(argv[1],"off"))test_diagnostic_off();
    else if(!strcmp(argv[1],"fast"))test_fast_queries();
    else if(!strcmp(argv[1],"legacy"))test_legacy_rollback();
    else if(!strcmp(argv[1],"positive"))test_positive_wait();
    else if(!strcmp(argv[1],"visibility"))test_publication_visibility();
    else if(!strcmp(argv[1],"ordering"))test_ordering();
    else assert(!"unknown case");
    completion_poll_report();
    puts("completion ordering passed");return 0;
}
'''
        cls.tmp = tempfile.TemporaryDirectory(prefix='nx2a-completion-')
        cls.addClassCleanup(cls.tmp.cleanup)
        src, cls.exe = Path(cls.tmp.name) / 'check.c', Path(cls.tmp.name) / 'check'
        src.write_text(harness)
        result = subprocess.run(['clang', '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror', '-pthread', str(src), '-o', str(cls.exe)], capture_output=True, text=True)
        if result.returncode:
            raise AssertionError(result.stderr)

    def run_case(self, name):
        result = subprocess.run([str(self.exe), name], capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_fast_zero_queries_avoid_waiting_calls(self):
        self.run_case('fast')

    def test_diagnostic_off_skips_counters_without_changing_results(self):
        self.run_case('off')

    def test_live_control_can_restore_legacy_poll(self):
        self.run_case('legacy')

    def test_positive_timeout_still_waits(self):
        self.run_case('positive')

    def test_release_acquire_publishes_visible_data(self):
        self.run_case('visibility')

    def test_visible_completion_and_safe_reuse(self):
        self.run_case('ordering')
