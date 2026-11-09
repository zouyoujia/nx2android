"Verify production overlay metrics preserve geometry with bounded window queries."
from pathlib import Path
import os,shutil,subprocess,tempfile,unittest
from tests import test_gpu_segments as gs
DM=Path(__file__).resolve().parents[1]/'nxnative/dm'

def program(tail):
    source=(DM/'android/front_android.c').read_text();header=(DM/'android/overlay_layout.h').read_text()
    state=source[source.index('static _Thread_local unsigned g_ov_metrics_depth;'):source.index('static void overlay_window_size')]
    funcs='\n'.join(gs.function(source,n) for n in ['overlay_window_size','dm_overlay_metrics_begin','dm_overlay_metrics_end','dm_overlay_layout','dm_overlay_scale'])
    return r'''
#include <assert.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
typedef struct{int w,h;}ANativeWindow;
static ANativeWindow *window;
static _Thread_local unsigned queries;
static ANativeWindow *phone_window(void){return window;}
static int ANativeWindow_getWidth(ANativeWindow *w){queries++;return w->w;}
static int ANativeWindow_getHeight(ANativeWindow *w){queries++;return w->h;}
#define LOG(...) ((void)0)
enum{OV_STICK_L,OV_STICK_R,OV_BUTTON,OV_DPAD,OV_MENU,DM_BTN_A=1};
typedef struct{int kind;float ax,x,y,w,h;uint64_t btn;const char *label,*name;}ov_ctl;
enum{OV_N=2};
static const ov_ctl OV_LAYOUTS[2][OV_N]={
 {{OV_BUTTON,1,-.2f,.7f,.12f,.12f,DM_BTN_A,0,0},{OV_MENU,1,-.1f,.1f,.08f,.04f,0,0,0}},
 {{OV_BUTTON,1,-.3f,.7f,.14f,.14f,DM_BTN_A,0,0},{OV_MENU,1,-.1f,.1f,.08f,.04f,0,0,0}}};
#define OV_CTLS (OV_LAYOUTS[dm_overlay_layout()])
#define OV_PHONE_ASPECT 1.9f
#define OV_MIN_MM 10.0f
static int g_dpi=400,custom;
static int ov_layout_for_size(int w,int h){return h>0 && (float)w/(float)h>=OV_PHONE_ASPECT;}
static int dm_overlay_custom(int l,int i,float *x,float *y,float *s){assert(l>=0&&l<2&&i>=0&&i<OV_N);*s=custom?1.25f:1;if(custom){*x=.8f;*y=.6f;}return custom;}
'''+state+funcs+gs.function(header,'ov_index')+gs.function(header,'ov_box')+r'''
typedef struct{int layout;float scale,box[2][4];}sample;
static __attribute__((unused)) sample measure(void){sample s={0};s.layout=dm_overlay_layout();s.scale=dm_overlay_scale();for(int i=0;i<OV_N;i++)ov_box(&OV_CTLS[i],(float)(window?window->w:0),(float)(window?window->h:0),s.box[i]);return s;}
'''+tail

@unittest.skipUnless(os.name!='nt' and shutil.which('clang'),'POSIX clang needed')
class OverlayMetrics(unittest.TestCase):
    def run_c(self,tail):
        with tempfile.TemporaryDirectory(prefix='nx2a-overlay-metrics-') as tmp:
            src,exe=Path(tmp)/'check.c',Path(tmp)/'check';src.write_text(program(tail))
            p=subprocess.run(['clang','-std=gnu11','-O2','-pthread','-Wall','-Wextra','-Werror','-Wno-missing-field-initializers',str(src),'-o',str(exe)],capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stderr);p=subprocess.run([str(exe)],capture_output=True,text=True,timeout=10);self.assertEqual(p.returncode,0,p.stdout+p.stderr)
    def test_same_geometry_custom_positions_and_scale_with_two_queries(self):
        self.run_c(r'''
int main(void){
 ANativeWindow w={3168,1440};window=&w;
 for(int hi=0;hi<2;hi++)for(int layout=0;layout<2;layout++)for(custom=0;custom<2;custom++){
  w.h=hi?1440:720;w.w=layout?w.h*22/10:w.h*12/10;setenv("DM_OVERLAY_WINDOW_CACHE","0",1);queries=0;dm_overlay_metrics_begin();sample a=measure();for(int i=0;i<100;i++)a=measure();dm_overlay_metrics_end();assert(queries>=400);
  setenv("DM_OVERLAY_WINDOW_CACHE","1",1);queries=0;dm_overlay_metrics_begin();sample b=measure();for(int i=0;i<100;i++)b=measure();dm_overlay_metrics_end();assert(queries==2);
  assert(a.layout==b.layout && a.scale==b.scale && !memcmp(a.box,b.box,sizeof a.box));
 }
 return 0;
}
''')
    def test_resize_scope_nesting_null_window_and_live_disable(self):
        self.run_c(r'''
int main(void){
 ANativeWindow w={1600,1000};window=&w;unsetenv("DM_OVERLAY_WINDOW_CACHE");queries=0;
 dm_overlay_metrics_begin();dm_overlay_metrics_end();assert(queries==0); // no geometry requested by a hidden overlay
 dm_overlay_metrics_begin();assert(!dm_overlay_layout() && queries==2);w.w=3200;dm_overlay_metrics_begin();assert(!dm_overlay_layout()&&queries==2);dm_overlay_metrics_end();assert(!dm_overlay_layout());dm_overlay_metrics_end();
 unsigned q=queries;assert(dm_overlay_layout() && queries==q+2);dm_overlay_metrics_begin();assert(dm_overlay_layout());dm_overlay_metrics_end();dm_overlay_metrics_end();assert(!g_ov_metrics_depth && !g_ov_metrics_active);
 setenv("DM_OVERLAY_WINDOW_CACHE","0",1);dm_overlay_metrics_begin();w.w=1600;assert(!dm_overlay_layout());w.w=3200;assert(dm_overlay_layout());dm_overlay_metrics_end();
 window=NULL;setenv("DM_OVERLAY_WINDOW_CACHE","1",1);dm_overlay_metrics_begin();assert(!dm_overlay_layout() && dm_overlay_scale()==1);dm_overlay_metrics_end();return 0;
}
''')
    def test_input_thread_is_not_frozen_by_present_thread_snapshot(self):
        self.run_c(r'''
static void *input(void *p){(void)p;assert(!g_ov_metrics_active && dm_overlay_layout()==1 && queries==2);dm_overlay_metrics_begin();assert(dm_overlay_layout()==1);dm_overlay_metrics_end();return 0;}
int main(void){
 ANativeWindow w={1600,1000};window=&w;setenv("DM_OVERLAY_WINDOW_CACHE","1",1);dm_overlay_metrics_begin();assert(!dm_overlay_layout());w.w=3200;
 pthread_t t;assert(!pthread_create(&t,NULL,input,NULL));pthread_join(t,NULL);assert(!dm_overlay_layout());dm_overlay_metrics_end();assert(dm_overlay_layout()==1);return 0;
}
''')
    def test_present_scope_closes_before_submission_and_has_no_early_return(self):
        source=(DM/'android/gfx_vk_present.c').read_text();fn=gs.function(source,'present_window')
        self.assertEqual(fn.count('dm_overlay_metrics_begin();'),1);self.assertEqual(fn.count('dm_overlay_metrics_end();'),1)
        a,b=fn.index('dm_overlay_metrics_begin();'),fn.index('dm_overlay_metrics_end();')
        self.assertNotIn('return',fn[a:b]);self.assertLess(a,b);self.assertLess(b,fn.index('vkEndCommandBuffer(cb)'));self.assertLess(b,fn.index('vkQueueSubmit('))
