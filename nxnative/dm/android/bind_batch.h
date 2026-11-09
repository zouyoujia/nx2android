// Coalesce commands within one draw; no state is elided across draws.
#pragma once
#include <vulkan/vulkan_core.h>
typedef void (VKAPI_PTR *vk_stencil_set)(VkCommandBuffer, VkStencilFaceFlags, uint32_t);
static inline unsigned vk_stencil_pair(VkCommandBuffer cb, vk_stencil_set emit, uint32_t front, uint32_t back, int batch) {
    if (batch && front == back) { emit(cb, VK_STENCIL_FACE_FRONT_AND_BACK, front); return 1; }
    emit(cb, VK_STENCIL_FACE_FRONT_BIT, front); emit(cb, VK_STENCIL_FACE_BACK_BIT, back); return 2;
}
static inline unsigned vk_vertex_runs(VkCommandBuffer cb, uint32_t mask, const VkBuffer buffers[17], const VkDeviceSize offsets[17], PFN_vkCmdBindVertexBuffers emit) {
    unsigned calls = 0;
    for (unsigned i = 0; i < 17;) {
        if (!(mask & (1u << i))) { i++; continue; }
        unsigned first = i++; while (i < 17 && (mask & (1u << i))) i++;
        emit(cb, first, i - first, buffers + first, offsets + first); calls++;
    }
    return calls;
}
