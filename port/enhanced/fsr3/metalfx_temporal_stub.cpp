// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 the OpenRAC contributors
//
// MetalFxTemporal where the build has no Metal GL: never supported.
#include "enhanced/fsr3/metalfx_temporal.h"

namespace openrac::enhanced::fsr3_detail {

struct MetalFxTemporal::Impl {};

MetalFxTemporal::MetalFxTemporal() : m(std::make_unique<Impl>()) {}
MetalFxTemporal::~MetalFxTemporal() = default;

bool MetalFxTemporal::supported() {
    return false;
}

bool MetalFxTemporal::configure(std::array<int, 2>, std::array<int, 2>, std::string& error) {
    error = "MetalFX needs the Metal renderer";
    return false;
}

bool MetalFxTemporal::dispatch(std::array<unsigned, 5>, const Dispatch&, std::string& error) {
    error = "MetalFX needs the Metal renderer";
    return false;
}

void MetalFxTemporal::release() {}

}  // namespace openrac::enhanced::fsr3_detail
