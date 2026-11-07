// Logical window coordinates stay unchanged for touch/UI; only vertex positions rotate.
#pragma once
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <limits.h>
typedef struct { float pos[4], uv[4], alpha; uint32_t turn; } vk_present_quad;
_Static_assert(offsetof(vk_present_quad, alpha) == 32 && offsetof(vk_present_quad, turn) == 36 && sizeof(vk_present_quad) == 40, "present push ABI");
static inline int vk_present_quad_make(vk_present_quad *q, uint32_t ww, uint32_t wh, uint32_t iw, uint32_t ih, const int crop[4], int bottom, uint32_t turn) {
    if (!ww || !wh || !iw || !ih || ww > INT_MAX || wh > INT_MAX || iw > INT_MAX || ih > INT_MAX || (turn != 0 && turn != 1 && turn != 3)) return 0;
    int cx = crop ? crop[0] : 0, cy = crop ? crop[1] : 0, cw = crop && crop[2] ? crop[2] : (int)iw, ch = crop && crop[3] ? crop[3] : (int)ih;
    if (cx < 0 || cy < 0 || cw <= 0 || ch <= 0 || (int64_t)cx + cw > iw || (int64_t)cy + ch > ih) return 0;
    double s = fmin((double)ww / cw, (double)wh / ch); int dw = (int)(cw * s), dh = (int)(ch * s), dx = ((int)ww - dw) / 2, dy = ((int)wh - dh) / 2;
    if (!dw || !dh) return 0;
    *q = (vk_present_quad){{2.0f * dx / ww - 1, 2.0f * dy / wh - 1, 2.0f * (dx + dw) / ww - 1, 2.0f * (dy + dh) / wh - 1},
        {(float)cx / iw, (float)(bottom ? cy + ch : cy) / ih, (float)(cx + cw) / iw, (float)(bottom ? cy : cy + ch) / ih}, 1, turn};
    return 1;
}
