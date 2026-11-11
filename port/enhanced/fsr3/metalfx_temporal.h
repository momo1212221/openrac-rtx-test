// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 the OpenRAC contributors
//
// MetalFX's temporal scaler as the temporal upscaler on the Metal GL (renderer/metal): the same
// jittered world, depth and motion the FSR 3.1 path makes, encoded on the Metal GL's own frame.
#pragma once

#include <array>
#include <memory>
#include <string>

#include "enhanced/fsr3/sdk_context.h"

namespace openrac::enhanced::fsr3_detail {

class MetalFxTemporal {
public:
    MetalFxTemporal();
    ~MetalFxTemporal();
    MetalFxTemporal(const MetalFxTemporal&) = delete;
    MetalFxTemporal& operator=(const MetalFxTemporal&) = delete;

    // Whether the window draws with the Metal GL and its device has MetalFX's temporal scaler.
    static bool supported();
    // Notes the render and display sizes; the scaler is made at the first dispatch, from the
    // textures' own formats.
    [[nodiscard]] bool configure(
        std::array<int, 2> render_size, std::array<int, 2> display_size, std::string& error
    );
    // Upscales GL textures colour, float depth (reversed, R32F), motion (render pixels, previous
    // minus current), the reactive mask (R8) into the output, on the Metal GL's frame.
    [[nodiscard]] bool dispatch(
        std::array<unsigned, 5> textures, const Dispatch& frame, std::string& error
    );
    void release();

private:
    struct Impl;
    std::unique_ptr<Impl> m;
};

}  // namespace openrac::enhanced::fsr3_detail
