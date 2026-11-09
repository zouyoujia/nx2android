// Reflection semantics, independent of game and device. Unknown is an opaque writer.
#pragma once
static inline unsigned vk_linear_shader_class(const dmsv_bind *b, int n) {
    unsigned flags = 0; if (n < 0 || (n && !b)) return 2;
    for (int i = 0; i < n; i++) {
        if ((b[i].kind == RK_SBUF || b[i].kind == RK_SBUF_UBO) && b[i].desc == RD_SSBO) { flags |= 1; continue; }
        if ((b[i].kind == RK_SUPPORT || b[i].kind == RK_CBUF || b[i].kind == RK_C1 || b[i].kind == RK_DRV || b[i].kind == RK_UBO_PTR) && (b[i].desc == RD_UBO || b[i].desc == RD_SSBO)) continue;
        if ((b[i].kind == RK_TEX || b[i].kind == RK_TEX_UBO || b[i].kind == RK_TEX_C2A || b[i].kind == RK_TEXD) && (b[i].desc == RD_SAMPLED || b[i].desc == RD_IMAGE)) continue;
        if ((b[i].kind == RK_TBUF || b[i].kind == RK_TBUF_UBO) && b[i].desc == RD_TBUF) continue;
        if (b[i].kind == RK_SAMPD && b[i].desc == RD_SAMPLER) continue;
        flags |= 2;
    }
    return flags;
}
