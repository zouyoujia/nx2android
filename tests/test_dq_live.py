"Execute the production recorder/FIFO with real concurrent drain waiters."
from pathlib import Path
import os,re,shutil,subprocess,tempfile,unittest
from tests import test_gpu_segments as gs
DM=Path(__file__).resolve().parents[1]/'nxnative/dm'

def program(tail):
    source=(DM/'android/gfx_vk_dq.c').read_text()
    functions='\n'.join(gs.function(source,n) for n in ['wake','vk_dq_drain','vk_dq_refresh','space_for','rec_begin','rec_end','put'])
    macros='\n'.join(re.search(r'(?m)^#define '+n+r'\b[^\n]*',source).group() for n in ['ARR','DIRECT','BEGIN'])
    wrapper=gs.function(source,'w_vkCmdBindVertexBuffers');worker=gs.function(source,'recorder_main')
    cursors=source[source.index('typedef struct { _Alignas(128)'):source.index('static _Atomic int g_sleeping')]
    credit=source[source.index('static uint64_t g_cached_rd;'):source.index('uint64_t g_dq_recs,')]
    return r'''
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan_core.h>
#include <assert.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sched.h>
#include <time.h>
#ifdef __APPLE__
#define pthread_spinlock_t pthread_mutex_t
#define pthread_spin_init(p,shared) pthread_mutex_init(p,NULL)
#define pthread_spin_lock pthread_mutex_lock
#define pthread_spin_unlock pthread_mutex_unlock
#define pthread_setname_np(thread,name) ((void)(thread),(void)(name))
#endif
static _Atomic unsigned warnings,waits_live,waits_total;
static void log_line(const char *fmt,...){if(strstr(fmt,"[RED]"))atomic_fetch_add(&warnings,1);}
#define VKLOG log_line
static int watched_wait(pthread_cond_t *c,pthread_mutex_t *m,const struct timespec *t){
 atomic_fetch_add(&waits_live,1);atomic_fetch_add(&waits_total,1);int rc=pthread_cond_timedwait(c,m,t);atomic_fetch_sub(&waits_live,1);return rc;
}
#define pthread_cond_timedwait watched_wait
static _Atomic unsigned wake_signals;
static int watched_signal(pthread_cond_t *c){atomic_fetch_add(&wake_signals,1);return pthread_cond_signal(c);}
#define pthread_cond_signal watched_signal
static _Thread_local int test_producer;static _Atomic unsigned backpressure;
static int watched_yield(void){if(test_producer)atomic_fetch_add(&backpressure,1);return sched_yield();}
#define sched_yield watched_yield
static uint64_t now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return (uint64_t)t.tv_sec*1000000000+t.tv_nsec;}
#define WAIT(c) do{uint64_t limit=now()+5000000000ull;while(!(c)){assert(now()<limit);sched_yield();}}while(0)
enum {RING=512,OP_WRAP,OP_vkCmdBindVertexBuffers};
typedef struct{uint32_t op,size;VkCommandBuffer cb;}rec_hdr;
typedef struct{uint32_t first,n;}a_range;
static uint8_t *g_ring;static uint64_t g_w,g_dq_recs,g_dq_drains,g_dq_direct;
'''+cursors+credit+r'''
static _Atomic int g_sleeping,g_draining,g_record_on=1,g_dynamic_batch;
static pthread_spinlock_t g_pl;static pthread_mutex_t g_mu=PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_cv=PTHREAD_COND_INITIALIZER,g_dcv=PTHREAD_COND_INITIALIZER;
static int g_on;
static _Atomic unsigned started[256],allowed[256],events;
typedef struct{uint32_t first,n;VkBuffer b[2];VkDeviceSize o[2];}event;
static event ev[256];static pthread_mutex_t ev_mu=PTHREAD_MUTEX_INITIALIZER;
static void bind_real(VkCommandBuffer cb,uint32_t first,uint32_t n,const VkBuffer *b,const VkDeviceSize *o){
 assert(cb==(VkCommandBuffer)1 && n && n<=2);unsigned id=(unsigned)(uintptr_t)b[0];assert(id<256);
 atomic_store(&started[id],1);WAIT(atomic_load(&allowed[id]));pthread_mutex_lock(&ev_mu);unsigned at=atomic_load(&events);assert(at<256);
 ev[at].first=first;ev[at].n=n;memcpy(ev[at].b,b,n*sizeof *b);memcpy(ev[at].o,o,n*sizeof *o);atomic_store(&events,at+1);pthread_mutex_unlock(&ev_mu);
}
static PFN_vkCmdBindVertexBuffers r_vkCmdBindVertexBuffers=bind_real;
'''+functions+'\n'+macros+'\n'+wrapper+r'''
static void run(const rec_hdr *h){
 if(h->op==OP_WRAP)return;assert(h->op==OP_vkCmdBindVertexBuffers);const a_range *a=(const void*)(h+1);const VkBuffer *b=(const void*)(a+1);
 r_vkCmdBindVertexBuffers(h->cb,a->first,a->n,b,(const VkDeviceSize*)(b+a->n));
}
'''+worker+r'''
static void setup(void){
 setenv("DM_VK_MT_RECORD","0",1);vk_dq_refresh();assert(atomic_load(&g_record_on)==1 && !g_ring); // startup-disabled: no locks or mode changes
 g_ring=calloc(1,RING);assert(g_ring);pthread_spin_init(&g_pl,PTHREAD_PROCESS_PRIVATE);g_on=1;setenv("DM_VK_MT_RECORD","1",1);
 pthread_t t;assert(!pthread_create(&t,NULL,recorder_main,NULL));pthread_detach(t);
}
static void one(unsigned id){VkBuffer b=(VkBuffer)(uintptr_t)id;VkDeviceSize o=id*5;w_vkCmdBindVertexBuffers((VkCommandBuffer)1,0,1,&b,&o);}
static __attribute__((unused)) void *drainer(void *p){vk_dq_drain();atomic_store((_Atomic unsigned*)p,1);return NULL;}
'''+tail

@unittest.skipUnless(os.name!='nt' and shutil.which('clang'),'POSIX clang needed')
class DqLive(unittest.TestCase):
    def run_c(self,tail,ring=512):
        with tempfile.TemporaryDirectory(prefix='nx2a-dq-') as tmp:
            src,exe=Path(tmp)/'check.c',Path(tmp)/'check';src.write_text(program(tail).replace('RING=512','RING='+str(ring)))
            p=subprocess.run(['clang','-std=gnu11','-O2','-pthread','-Wall','-Wextra','-Werror','-Wno-missing-field-initializers','-I'+str(DM/'third_party/vulkan-headers/include'),str(src),'-o',str(exe)],capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stderr);p=subprocess.run([str(exe)],capture_output=True,text=True,timeout=10);self.assertEqual(p.returncode,0,p.stdout+p.stderr)
    def test_live_bypass_preserves_fifo_copies_and_wraparound(self):
        self.run_c(r'''
int main(void){
 setup();for(unsigned i=0;i<256;i++)atomic_store(&allowed[i],1);atomic_store(&allowed[100],0);
 VkBuffer b[2]={(VkBuffer)100,(VkBuffer)101};VkDeviceSize o[2]={10,11};w_vkCmdBindVertexBuffers((VkCommandBuffer)1,3,2,b,o);
 WAIT(atomic_load(&started[100]));b[0]=(VkBuffer)200;o[0]=999;atomic_store(&allowed[100],1);vk_dq_drain();
 assert(atomic_load(&events)==1 && ev[0].b[0]==(VkBuffer)100 && ev[0].b[1]==(VkBuffer)101 && ev[0].o[0]==10 && ev[0].o[1]==11);
 setenv("DM_VK_MT_RECORD","0",1);vk_dq_refresh();uint64_t w=g_w;one(2);assert(g_w==w && atomic_load(&events)==2 && !atomic_load(&g_record_on));
 for(unsigned i=3;i<=70;i++){setenv("DM_VK_MT_RECORD",i%7?"1":"0",1);vk_dq_refresh();one(i);}
 vk_dq_drain();assert(atomic_load(&events)==70 && g_w>RING && !atomic_load(&warnings));
 for(unsigned i=2;i<70;i++)assert(ev[i].b[0]==(VkBuffer)(uintptr_t)(i+1) && ev[i].o[0]==(i+1)*5);
 assert(atomic_load(&g_rd)==atomic_load(&g_pub));return 0;
}
''')
    def test_space_credit_removes_repeated_acquires_without_changing_records(self):
        self.run_c(r'''
int main(void){
 setup();for(unsigned i=0;i<256;i++)atomic_store(&allowed[i],1);
 assert(sizeof(vk_dq_cursor)>=128 && _Alignof(vk_dq_cursor)>=128);
 assert(((uintptr_t)&g_pub>>7)!=((uintptr_t)&g_rd>>7));
 setenv("DM_VK_MT_SPACE_PROF","1",1);setenv("DM_VK_MT_SPACE_CACHE","0",1);vk_dq_refresh();
 for(unsigned i=1;i<=64;i++)one(i);vk_dq_drain();assert(g_space_stats[DQ_SPACE_RECS]==64 && g_space_stats[DQ_SPACE_CHECKS]==64 && g_space_stats[DQ_SPACE_LOADS]==64);
 setenv("DM_VK_MT_SPACE_CACHE","1",1);vk_dq_refresh();
 for(unsigned i=1;i<=64;i++)one(i);vk_dq_drain();assert(g_space_stats[DQ_SPACE_RECS]==64 && g_space_stats[DQ_SPACE_CHECKS]==64 && !g_space_stats[DQ_SPACE_LOADS]);
 assert(atomic_load(&events)==128 && !atomic_load(&warnings));for(unsigned i=0;i<128;i++)assert(ev[i].b[0]==(VkBuffer)(uintptr_t)(i%64+1));
 return 0;
}
''',ring=1<<20)
    def test_cached_credit_backpressure_and_wrap_never_overwrite_unread_data(self):
        self.run_c(r'''
static void *fill(void *p){(void)p;test_producer=1;for(unsigned i=2;i<=80;i++)one(i);return NULL;}
int main(void){
 setup();setenv("DM_VK_MT_SPACE_CACHE","1",1);setenv("DM_VK_MT_SPACE_PROF","1",1);vk_dq_refresh();
 for(unsigned i=2;i<256;i++)atomic_store(&allowed[i],1);one(1);WAIT(atomic_load(&started[1]));
 pthread_t p;assert(!pthread_create(&p,NULL,fill,NULL));WAIT(atomic_load(&backpressure));atomic_store(&allowed[1],1);pthread_join(p,NULL);vk_dq_drain();
 assert(g_w>RING && g_space_stats[DQ_SPACE_WAITS] && atomic_load(&events)==80 && !atomic_load(&warnings));
 for(unsigned i=0;i<80;i++)assert(ev[i].b[0]==(VkBuffer)(uintptr_t)(i+1));assert(atomic_load(&g_pub)==atomic_load(&g_rd));return 0;
}
''')
    def test_two_producers_share_only_conservative_credit(self):
        self.run_c(r'''
static _Atomic unsigned ready,go;
static void *produce(void *p){unsigned base=(unsigned)(uintptr_t)p;test_producer=1;atomic_fetch_add(&ready,1);WAIT(atomic_load(&go));for(unsigned i=1;i<=70;i++)one(base+i);return NULL;}
int main(void){
 setup();setenv("DM_VK_MT_SPACE_CACHE","1",1);vk_dq_refresh();for(unsigned i=0;i<256;i++)atomic_store(&allowed[i],1);
 pthread_t a,b;assert(!pthread_create(&a,NULL,produce,(void*)0));assert(!pthread_create(&b,NULL,produce,(void*)100));WAIT(atomic_load(&ready)==2);atomic_store(&go,1);pthread_join(a,NULL);pthread_join(b,NULL);vk_dq_drain();
 assert(atomic_load(&events)==140 && g_w>RING && !atomic_load(&warnings));unsigned next[2]={1,101};
 for(unsigned i=0;i<140;i++){unsigned id=(unsigned)(uintptr_t)ev[i].b[0],n=id>100;assert(id==next[n]++ && ev[i].o[0]==id*5);}
 assert(next[0]==71 && next[1]==171 && atomic_load(&g_pub)==atomic_load(&g_rd));return 0;
}
''')
    def test_one_notification_per_sleep_and_empty_wake_rearms(self):
        self.run_c(WAKE_REARM,ring=1<<20)
    def test_two_drain_waiters_keep_notifications_until_both_finish(self):
        self.run_c(WAITERS)

WAITERS=r'''
static _Atomic unsigned done1,done2;
int main(void){
 setup();one(1);WAIT(atomic_load(&started[1]));pthread_t a,b;assert(!pthread_create(&a,NULL,drainer,&done1));WAIT(atomic_load(&waits_total)>=1);
 one(2);assert(!pthread_create(&b,NULL,drainer,&done2));WAIT(atomic_load(&waits_total)>=2);
 atomic_store(&allowed[1],1);WAIT(atomic_load(&done1));WAIT(atomic_load(&started[2]) && atomic_load(&waits_live)==1);
 atomic_store(&allowed[2],1);pthread_join(a,NULL);pthread_join(b,NULL);
 assert(atomic_load(&done1) && atomic_load(&done2) && atomic_load(&events)==2 && !atomic_load(&warnings) && !atomic_load(&g_draining));return 0;
}
'''

WAKE_REARM=r'''
int main(void){
 (void)setup;g_ring=calloc(1,RING);assert(g_ring);pthread_spin_init(&g_pl,PTHREAD_PROCESS_PRIVATE);g_on=1;
 for(unsigned i=0;i<256;i++)atomic_store(&allowed[i],1);
 // Hold the reader off the CPU: all records are published before its first run.
 atomic_store(&g_sleeping,1);for(unsigned i=1;i<=64;i++)one(i);
 assert(atomic_load(&wake_signals)==1 && atomic_load(&g_pub)==g_w && !atomic_load(&g_rd));
 pthread_t worker;assert(!pthread_create(&worker,NULL,recorder_main,NULL));pthread_detach(worker);
 WAIT(atomic_load(&events)==64 && atomic_load(&g_sleeping));
 for(unsigned i=0;i<16;i++){
  unsigned before=atomic_load(&wake_signals);wake();
  WAIT(atomic_load(&g_sleeping));assert(atomic_load(&wake_signals)==before+1);
 }
 one(65);vk_dq_drain();assert(atomic_load(&events)==65 && !atomic_load(&warnings));
 for(unsigned i=0;i<65;i++)assert(ev[i].b[0]==(VkBuffer)(uintptr_t)(i+1)&&ev[i].o[0]==(i+1)*5);
 assert(atomic_load(&g_rd)==atomic_load(&g_pub));return 0;
}
'''
