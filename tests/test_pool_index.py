"""Exercise production pool index mutation and lookup against a linear oracle."""
from pathlib import Path
import os,shutil,subprocess,tempfile,unittest
from tests.test_lazy_small_pools import function
ROOT=Path(__file__).resolve().parents[1]
@unittest.skipUnless(os.name!='nt' and shutil.which('clang'),'POSIX clang needed')
class PoolIndex(unittest.TestCase):
    def test_random_dense_removal_overlap_duplicates_and_capacity(self):
        source=(ROOT/'nxnative/dm/android/gfx_vk_core.c').read_text()
        self.assertNotIn('pool_ord_make',source)
        code=r'''
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>
#define MAXPOOL 4096
#define LOG(...) do {if(0)fprintf(stderr,__VA_ARGS__);}while(0)
static void die(const char *fmt,int n){(void)fmt;(void)n;abort();}
typedef struct {uint64_t lo,hi,end;} vkpool_t;
static vkpool_t g_pool[MAXPOOL];static int g_npool;static uint16_t g_pool_ord[MAXPOOL];static uint32_t g_pool_gen;
static pthread_rwlock_t g_pool_mu=PTHREAD_RWLOCK_INITIALIZER;
'''+function(source,'pool_ord_add')+function(source,'pool_ord_del')+function(source,'vk_pool_of')+r'''
static uint32_t rnd=17;
static uint32_t next(void){rnd=rnd*1664525+1013904223;return rnd;}
static void verify(void){
 unsigned char seen[MAXPOOL]={0};
 for(int i=0;i<g_npool;i++){int j=g_pool_ord[i];assert(j>=0 && j<g_npool && !seen[j]);seen[j]=1;if(i)assert(g_pool[g_pool_ord[i-1]].lo<=g_pool[j].lo);}
 for(int k=0;k<10;k++){
  uint64_t a=0x1000+(next()%1000)*32,n=(next()%128)+1;int have=0;
  for(int i=0;i<g_npool;i++)have|=a>=g_pool[i].lo && a+n<=g_pool[i].hi;
  vkpool_t *p=vk_pool_of(a,n);assert((p!=NULL)==have);if(p)assert(a>=p->lo && a+n<=p->hi);
 }
}
static void add(uint64_t a,uint64_t n){g_pool[g_npool++]=(vkpool_t){a,a+n,a+n};pool_ord_add();g_pool_gen++;}
static void del(int i){g_pool[i]=g_pool[--g_npool];pool_ord_del(i);g_pool_gen++;}
int main(void){
 // Full capacity, equal starts, nested/overlapping ranges, and deletion of the last slot.
 for(int i=0;i<MAXPOOL;i++)add(0x1000+(uint64_t)(i%100)*32,32+(uint64_t)(i%8)*64);
 verify();del(g_npool-1);verify();del(0);verify();
 while(g_npool){del((int)(next()%(unsigned)g_npool));if(!(g_npool%127))verify();}
 verify();
 for(int i=0;i<50000;i++){
  if(!g_npool || (g_npool<200 && (next()&1)))add(0x1000+(next()%1000)*32,1+next()%512);
  else del((int)(next()%(unsigned)g_npool));
  verify();
 }
 return 0;
}
'''
        with tempfile.TemporaryDirectory(prefix='nx2a-pool-index-') as tmp:
            src,exe=Path(tmp)/'check.c',Path(tmp)/'check';src.write_text(code)
            p=subprocess.run(['clang','-std=gnu11','-O2','-Wall','-Wextra','-Werror','-Wno-dangling-else','-pthread',str(src),'-o',str(exe)],capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stderr)
            p=subprocess.run([str(exe)],capture_output=True,text=True,timeout=20)
            self.assertEqual(p.returncode,0,p.stdout+p.stderr)
