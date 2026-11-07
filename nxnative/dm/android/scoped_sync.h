// A narrowed input dependency paired with an unchanged full output barrier.
#pragma once
// Only the immediate operation's consumer stages are narrowed. Its following
// full barrier still publishes ALL earlier writes to ALL subsequent consumers.
// Do not reuse this helper when that output barrier is disabled or conditional.
enum { VK_BEFORE_COMPUTE, VK_BEFORE_COMPUTE_INDIRECT, VK_BEFORE_TRANSFER, VK_BEFORE_FILL };
typedef struct { VkPipelineStageFlags stage; VkAccessFlags access; } vk_before_scope;
static inline vk_before_scope vk_before_scope_for(int kind) {
    switch (kind) {
    case VK_BEFORE_COMPUTE: return (vk_before_scope){VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
    case VK_BEFORE_COMPUTE_INDIRECT: return (vk_before_scope){VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT};
    case VK_BEFORE_TRANSFER: return (vk_before_scope){VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT};
    case VK_BEFORE_FILL: return (vk_before_scope){VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT};
    default: return (vk_before_scope){0, 0};
    }
}
enum { VK_EFF_EARLY, VK_EFF_FINAL, VK_EFF_MAIN, VK_EFF_UPLOAD, VK_EFF_FULL_INPUT, VK_EFF_SCOPED_INPUT, VK_EFF_RESETS, VK_EFF_UNUSED_RESETS, VK_EFF_N };
