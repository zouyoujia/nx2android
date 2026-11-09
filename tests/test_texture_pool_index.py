"""Exercise production per-pool image indexing, aliasing, eviction and retirement."""
from pathlib import Path
import os, shutil, subprocess, tempfile, unittest
from tests.test_lazy_small_pools import function
ROOT = Path(__file__).resolve().parents[1]
DM = ROOT / 'nxnative/dm'

@unittest.skipUnless(os.name != 'nt' and shutil.which('clang'), 'POSIX clang needed')
class TexturePoolIndex(unittest.TestCase):
    def test_pool_lifecycle_overlap_limits_and_hash_growth(self):
        source = (DM / 'android/gfx_vk_tex.c').read_text()
        index = source[source.index('typedef struct pool_image'):source.index('static pthread_mutex_t g_img_mu')]
        funcs = '\n'.join(function(source, n) for n in
            ('alias_link', 'alias_unlink', 'img_drop', 'evict_overlaps', 'vk_tex_drop_pool'))
        # These paths must not traverse the global storage hash table, even on deletion.
        for name in ('alias_link', 'evict_overlaps', 'vk_tex_drop_pool'):
            self.assertNotIn('g_by_key.cap', function(source, name))
        code = r'''
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include "nvn_hmap.h"
#define VKLOG(...) do {if(0)fprintf(stderr,__VA_ARGS__);}while(0)
typedef struct vkimg vkimg_t;
struct vkimg {
 uint64_t pool,used,img,mem;int64_t off,size;int flags;
 vkimg_t *ovl[8];int novl;void *sub;
 const void *lin_mem,*pkg_mem;size_t lin_len,pkg_len;int packaged,pkg_released;
};
typedef struct {uint64_t pool;int64_t off;uint8_t d[24];} akey;
static nvn_hmap g_by_key={.klen=sizeof(akey)};
static uint64_t g_tex_frame=100,g_img_n,g_img_bytes,g_img_evicted;
enum {EVICT_KEEP=60};
static pthread_mutex_t g_img_mu=PTHREAD_MUTEX_INITIALIZER;
static unsigned char freed[20000];
static unsigned next_id=1,forgotten,view_drops,dirty_releases,defer_regular,defer_sub;
static void linear_cache_forget(vkimg_t *im){(void)im;}
static void vk_lrz_forget(vkimg_t *im){assert(!freed[im->img]);forgotten++;}
static void vk_tex_view_drop_all(vkimg_t *im){assert(!freed[im->img]);view_drops++;}
static void nvn_dirty_release(const void *p,size_t n){assert(p && n);dirty_releases++;}
static void vk_defer_free_image(uint64_t img,uint64_t mem){assert(mem==img+100000 && !freed[img]);freed[img]=1;defer_regular++;}
static void vk_defer_free_image_sub(uint64_t img,void *sub){assert(sub && !freed[img]);freed[img]=1;defer_sub++;}
''' + index + funcs + r'''
static akey key_of(uint64_t pool,int64_t off,unsigned variant){akey k;memset(&k,0,sizeof k);k.pool=pool;k.off=off;memcpy(k.d,&variant,sizeof variant);return k;}
static vkimg_t *add(uint64_t pool,int64_t off,int64_t size,unsigned variant,uint64_t used){
 akey k=key_of(pool,off,variant);assert(!nvn_hm_get(&g_by_key,&k));
 pool_image *e=calloc(1,sizeof *e);assert(e);vkimg_t *im=&e->im;
 im->pool=pool;im->off=off;im->size=size;im->used=used;im->img=next_id++;im->mem=im->img+100000;
 if(im->img&1)im->sub=(void *)(uintptr_t)im->img;
 g_img_n++;g_img_bytes+=size;alias_link(im);nvn_hm_put(&g_by_key,&k,im);pool_image_add(im,&k);return im;
}
static void check(void){
 unsigned count=0,pools=0;uint64_t bytes=0;
 for(unsigned i=0;i<g_by_pool.cap;i++){
  pool_image *head=g_by_pool.vals[i];if(!head)continue;pools++;assert(!head->prev);
  uint64_t pool=0;memcpy(&pool,g_by_pool.keys+(size_t)i*g_by_pool.klen,sizeof pool);
  pool_image *prev=NULL;
  for(pool_image *e=head;e;e=e->next){
   assert(e->prev==prev && e->im.pool==pool && e->key.pool==pool);
   assert(nvn_hm_get(&g_by_key,&e->key)==&e->im && !freed[e->im.img]);
   assert(e->im.novl>=0 && e->im.novl<=8);
   for(int j=0;j<e->im.novl;j++){
    vkimg_t *o=e->im.ovl[j];assert(!freed[o->img] && o!=&e->im && o->pool==pool);
    assert(o->off<e->im.off+e->im.size && e->im.off<o->off+o->size);
    int back=0;for(int k=0;k<o->novl;k++)back+=o->ovl[k]==&e->im;assert(back==1);
   }
   count++;bytes+=e->im.size;prev=e;
  }
 }
 assert(pools==g_by_pool.n && count==g_by_key.n && count==g_img_n && bytes==g_img_bytes);
}
static void drop(uint64_t pool){vk_tex_drop_pool((void *)(uintptr_t)pool);check();}
int main(void){
 // Many unrelated pools: alias and eviction work must remain local after both maps grow.
 for(uint64_t p=1;p<=512;p++)for(int j=0;j<4;j++)add(p,j*128,64,0,g_tex_frame);
 check();assert(g_tex_pool_stats.alias_visits==512*6);
 uint64_t before=g_tex_pool_stats.evict_visits;evict_overlaps(17,0,1024);
 assert(g_tex_pool_stats.evict_visits-before==4 && g_img_n==2048);
 before=g_tex_pool_stats.alias_visits;add(17,1000,64,0,g_tex_frame);
 assert(g_tex_pool_stats.alias_visits-before==4);check();
 uint64_t alias=g_tex_pool_stats.alias_visits,evict=g_tex_pool_stats.evict_visits;
 // Empty staging pools never visit images or disturb live pool heads.
 for(uint64_t p=1000;p<36000;p++)vk_tex_drop_pool((void *)(uintptr_t)p);
 assert(g_tex_pool_stats.empty_drops==35000 && g_tex_pool_stats.alias_visits==alias && g_tex_pool_stats.evict_visits==evict);check();
 // Delete middle, head, and tail after hash growth/backward shifts; exact keys still resolve.
 uint64_t p=17;pool_image *head=nvn_hm_get(&g_by_pool,&p);assert(head && head->next && head->next->next);
 pool_image *mid=head->next;akey gone=mid->key;img_drop(&mid->im);assert(!nvn_hm_get(&g_by_key,&gone));check();
 img_drop(&head->im);check();head=nvn_hm_get(&g_by_pool,&p);while(head->next)head=head->next;img_drop(&head->im);check();
 for(uint64_t q=1;q<=512;q++)drop(q);assert(!g_img_n && !g_by_key.n && !g_by_pool.n);
 // Reused pool address and same exact key receive a new, independent image.
 vkimg_t *a=add(7,0,100,3,100);unsigned id=a->img;drop(7);assert(freed[id]);
 a=add(7,0,100,3,100);assert(a->img!=id && !a->novl);check();drop(7);
 // Half-open overlap boundaries, fresh retention, age boundary and exact descriptions.
 a=add(9,100,100,1,100);vkimg_t *b=add(9,200,50,1,100);assert(!a->novl && !b->novl);
 vkimg_t *c=add(9,150,70,2,100);assert(c->novl==2 && a->novl==1 && b->novl==1);check();
 evict_overlaps(9,150,1);assert(g_img_n==3); // all recently used
 a->used=c->used=40;evict_overlaps(9,150,1);assert(g_img_n==1 && !b->novl);check();drop(9);
 // The 8-neighbor cap and 64-eviction cap remain bounded and bidirectional.
 for(unsigned i=0;i<100;i++)add(10,0,128,i,0);check();
 before=g_img_n;evict_overlaps(10,0,128);assert(before-g_img_n==64);check();drop(10);
 // Existing dirty releases, view invalidation and both deferred-GPU-free routes survive.
 a=add(11,0,128,0,0);a->lin_mem=(void *)1;a->lin_len=128;a->pkg_mem=(void *)2;a->pkg_len=128;a->packaged=1;
 unsigned dirty=dirty_releases;drop(11);assert(dirty_releases==dirty+2);
 a=add(11,0,128,0,0);a->pkg_mem=(void *)2;a->pkg_len=128;a->packaged=1;a->pkg_released=1;drop(11);assert(dirty_releases==dirty+2);
 assert(defer_regular && defer_sub && forgotten==next_id-1 && view_drops==forgotten && defer_regular+defer_sub==forgotten);
 assert(g_tex_pool_stats.added==g_tex_pool_stats.dropped && !g_img_n);
 nvn_hm_clear(&g_by_pool,NULL);free(g_by_pool.keys);free(g_by_pool.vals);
 nvn_hm_clear(&g_by_key,NULL);free(g_by_key.keys);free(g_by_key.vals);
 return 0;
}
'''
        with tempfile.TemporaryDirectory(prefix='nx2a-texture-pool-') as tmp:
            src, exe = Path(tmp)/'check.c', Path(tmp)/'check'
            src.write_text(code)
            p = subprocess.run(['clang', '-std=gnu11', '-O1', '-g', '-Wall', '-Wextra', '-Werror',
                '-fsanitize=address,undefined', '-fno-omit-frame-pointer', '-pthread',
                '-I'+str(DM/'gfx/nvn'), str(src), '-o', str(exe)], capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            p = subprocess.run([str(exe)], capture_output=True, text=True, timeout=30)
            self.assertEqual(p.returncode, 0, p.stdout+p.stderr)
