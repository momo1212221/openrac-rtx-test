// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 the OpenRAC contributors
//
// The Metal objects behind the Vulkan renderer on Apple's systems (MoltenVK, VK_EXT_metal_objects),
// for work the renderer's GL does not express: enhanced/renderer/rtx_metal.mm traces rays with
// Metal's own intersector on the device and buffers the frames are drawn with. Objective-C objects
// are given as void*, bridged by an Objective-C++ caller. Elsewhere every call answers "none".
#pragma once

#include <cstddef>
#include <cstdint>

namespace openrac::vkgl {

// A GL buffer's bytes as Metal sees them: `buffer` (id<MTLBuffer>) from `offset`, `size` bytes.
// The MTLBuffer is the whole memory block the allocator placed the buffer in.
struct MetalBuffer {
    void* buffer = nullptr;
    std::size_t offset = 0;
    std::size_t size = 0;
};

// Whether the device runs on Metal and lends its objects (MoltenVK with VK_EXT_metal_objects).
bool metal_available();
// The renderer's id<MTLDevice>; null without metal_available().
void* metal_device();
// The memory of GL buffer `name`, which must be a storage buffer (bound to
// GL_SHADER_STORAGE_BUFFER) the GPU has given an allocation of its own: false otherwise, and
// while it has none yet (before its first frame as storage).
bool metal_buffer(unsigned name, MetalBuffer& out);
// The frame being recorded, and the last one the GPU has finished (its fence signalled), as
// counted by the renderer: work a frame queued is done once completed_frame() reaches its number.
std::uint64_t recording_frame();
std::uint64_t completed_frame();

}  // namespace openrac::vkgl
