// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 the OpenRAC contributors
//
// native_metal.h: MoltenVK's Metal objects through VK_EXT_metal_objects.
#include "renderer/vulkan/native_metal.h"

#include "renderer/vulkan/vkgl_internal.h"

#if defined(__APPLE__)
#define VK_USE_PLATFORM_METAL_EXT 1
#include <vulkan/vulkan_metal.h>
#endif

namespace openrac::vkgl {
using namespace detail;

namespace {

#if defined(__APPLE__)
// volk is built without the Metal platform: the one entry point is loaded from the device.
PFN_vkExportMetalObjectsEXT export_objects() {
    static VkDevice loaded_for = VK_NULL_HANDLE;
    static PFN_vkExportMetalObjectsEXT fn = nullptr;
    if (loaded_for != g->device) {
        loaded_for = g->device;
        fn = reinterpret_cast<PFN_vkExportMetalObjectsEXT>(
            vkGetDeviceProcAddr(g->device, "vkExportMetalObjectsEXT")
        );
    }
    return fn;
}
#endif

}  // namespace

bool metal_available() {
#if defined(__APPLE__)
    return g && g->open && g->metal_objects && export_objects() != nullptr;
#else
    return false;
#endif
}

void* metal_device() {
#if defined(__APPLE__)
    if (!metal_available()) {
        return nullptr;
    }
    VkExportMetalDeviceInfoEXT device{VK_STRUCTURE_TYPE_EXPORT_METAL_DEVICE_INFO_EXT};
    VkExportMetalObjectsInfoEXT objects{VK_STRUCTURE_TYPE_EXPORT_METAL_OBJECTS_INFO_EXT};
    objects.pNext = &device;
    export_objects()(g->device, &objects);
    return device.mtlDevice;
#else
    return nullptr;
#endif
}

bool metal_buffer(unsigned name, MetalBuffer& out) {
    out = {};
#if defined(__APPLE__)
    if (!metal_available()) {
        return false;
    }
    Buffer* b = g->buffers.find(name);
    if (b == nullptr || !b->storage || b->own == VK_NULL_HANDLE || b->own_allocation == nullptr
        || b->own_version != b->version) {
        return false;
    }
    VmaAllocationInfo info{};
    vmaGetAllocationInfo(g->allocator, b->own_allocation, &info);
    VkExportMetalBufferInfoEXT buffer{VK_STRUCTURE_TYPE_EXPORT_METAL_BUFFER_INFO_EXT};
    buffer.memory = info.deviceMemory;
    VkExportMetalObjectsInfoEXT objects{VK_STRUCTURE_TYPE_EXPORT_METAL_OBJECTS_INFO_EXT};
    objects.pNext = &buffer;
    export_objects()(g->device, &objects);
    if (buffer.mtlBuffer == nullptr) {
        return false;
    }
    out = {buffer.mtlBuffer, static_cast<std::size_t>(info.offset), b->size};
    return true;
#else
    (void)name;
    return false;
#endif
}

std::uint64_t recording_frame() {
    return g ? g->frame_count : 0;
}

std::uint64_t completed_frame() {
    if (!g || !g->open) {
        return 0;
    }
    // The queue runs frames in order: the newest finished one says all before it are done too.
    std::uint64_t done = 0;
    std::uint64_t oldest_pending = UINT64_MAX;
    for (Frame& f : g->frames) {
        if (!f.submitted) {
            continue;
        }
        if (vkGetFenceStatus(g->device, f.fence) == VK_SUCCESS) {
            done = std::max(done, f.submitted_count);
        } else {
            oldest_pending = std::min(oldest_pending, f.submitted_count);
        }
    }
    // Frames no longer in a slot were waited for when their slot was reused.
    if (oldest_pending != UINT64_MAX) {
        done = std::max(done, oldest_pending - 1);
    } else {
        done = std::max(done, g->frame_count > 0 ? g->frame_count - 1 : 0);
    }
    return done;
}

}  // namespace openrac::vkgl
