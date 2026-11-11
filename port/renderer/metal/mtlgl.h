// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 the OpenRAC contributors
//
// The renderer's GL (renderer/gl.h, renderer/world/gl_world.h) carried out with Metal, as
// renderer/vulkan/vkgl.h carries it out with Vulkan: chosen with OPENRAC_GL_API=metal.
//
// The GL model kept: rows are counted from the bottom as GL counts them. A texture's row 0 is GL's
// row 0, the shaders flip y (shader_translate.h) so that a draw lands on the rows GL would put it
// on, and the default framebuffer is a texture of the backend's that present() copies to the
// screen upside down, the right way up there. GL's objects are Metal objects held by reference:
// the command buffers keep what they use alive, so deleting a GL object never waits for the GPU.

#pragma once

#include <functional>
#include <string>

struct SDL_Window;

namespace openrac::mtlgl {

struct Config {
    // The window to present to (made with SDL_WINDOW_METAL), or null: drawing without a screen,
    // for the tests (glReadPixels reads the default framebuffer).
    SDL_Window* window = nullptr;
    int width = 0;  // the default framebuffer without a window
    int height = 0;
    bool vsync = true;
};

// Opens the device and the context. False, with why in `error`, when Metal cannot be used: the
// caller falls back to another API.
bool open(const Config& config, std::string& error);
void close();
bool active();

// renderer/gl.h's loader: the GL functions by name, null for one this does not carry out.
using Proc = void (*)();
Proc get_proc(const char* name);

// Ends the frame: the default framebuffer to the screen, the frame's commands submitted.
void present();
void set_swap_interval(int interval);

// The GPU's name, for the log.
std::string device_name();

// ---- Native Metal, for what the GL does not express (enhanced/renderer/rtx_metal.mm's ray tracing) ----
// Objective-C objects are given as void*: an Objective-C++ caller bridges them with
// (__bridge id<MTLDevice>) and the like. They are the context's own, alive while it is open.

struct Native {
    void* device = nullptr;  // id<MTLDevice>: the GPU the GL draws with
    void* queue = nullptr;   // id<MTLCommandQueue>
};
// The context's device and queue; nulls while no context is open.
Native native();
// The MTLBuffer behind GL buffer `name`, counted as used by this frame (so that a later
// glBufferSubData copies it instead of writing what the GPU may still read); null for none.
void* native_buffer(unsigned name);
// The MTLTexture behind GL texture `name`; null for none or a texture buffer.
void* native_texture(unsigned name);
// Ends the open encoder and gives `record` the frame's command buffer (id<MTLCommandBuffer>), so
// that what it encodes runs after the GL's commands so far and before those that follow. False,
// without calling `record`, while no context is open.
bool encode_native(const std::function<void(void* commands)>& record);
// The serial of the frame whose commands are being recorded, and the last one the GPU has finished:
// work the frame records may be read by another queue once completed_frame() reaches its serial.
unsigned long long recording_frame();
unsigned long long completed_frame();

}  // namespace openrac::mtlgl
