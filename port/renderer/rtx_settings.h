// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 the OpenRAC contributors
//
// Settings of the native hardware ray-tracing pass (enhanced/renderer/rtx.h): the frame carries
// them (FrameInput::rtx_settings), and the player's settings keep them (game/common/settings.h).
// The renderer owns the type so that neither it nor the settings include enhanced/.

#pragma once

namespace openrac::renderer {

// Hardware ray-tracing backend preference. Auto selects Apple Metal on macOS, NVIDIA CUDA/OptiX
// when the CUDA driver is present, or Vulkan Ray Query / Compute BVH on Vulkan.
enum class RtxBackend {
    Auto,
    Metal,
    Cuda,
    VulkanRayQuery,
    Count
};

inline constexpr const char* kRtxBackendKeys[] = {"auto", "metal", "cuda", "vulkan"};
inline constexpr const char* kRtxBackendNames[] = {
    "Auto (Native)", "Apple Metal RT", "NVIDIA CUDA / RTX", "Vulkan Ray Query"
};

// Ray-tracing resolution and ray budget preset.
enum class RtxQuality {
    Performance,
    Balanced,
    Quality,
    Count
};

inline constexpr const char* kRtxQualityKeys[] = {"performance", "balanced", "quality"};
inline constexpr const char* kRtxQualityNames[] = {"Performance", "Balanced", "Quality"};

// Persistent and per-frame RTX ray-tracing configuration.
struct RtxSettings {
    RtxBackend backend = RtxBackend::Auto;
    RtxQuality quality = RtxQuality::Balanced;
    bool shadows = true;                     // Hardware ray-traced sun soft shadows.
    bool ambient_occlusion = true;           // Hardware ray-traced ambient occlusion.
    bool reflections = true;                 // Hardware ray-traced specular reflections.
    bool global_illumination = true;         // Single-bounce ray-traced indirect diffuse boost.
    bool denoise = true;                     // Edge-preserving bilateral spatial filter.
    bool half_resolution = true;             // Trace on 2x2 half-resolution grid matching EarlyLighting.
    int samples_per_pixel = 2;               // Rays per effect per pixel (1..8).
    float ray_distance = 120.0f;             // Maximum world-space trace distance in game units.
    float ao_radius = 2.5f;                  // World-space ambient occlusion ray reach.
    float shadow_softness = 0.035f;          // Sun angular cone tangent for soft penumbrae.
    float reflection_roughness_cutoff = 0.65f;  // Max surface roughness that traces reflection rays.
    float gi_strength = 1.0f;                // Multiplier for ray-traced indirect bounce radiance.
    bool operator==(const RtxSettings&) const = default;
};

}  // namespace openrac::renderer
