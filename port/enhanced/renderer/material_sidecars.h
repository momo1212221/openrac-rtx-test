// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 the OpenRAC contributors
//
// A texture pack's material maps (Materials/<hash>/normal.png, metallic.png, emissive.png...; see
// docs/port/TEXTURE_PACKS.md) for the textures of the loaded level, packed side by side into one
// RGBA8 texture per colour texture, and the emission masks the ray tracer turns into lights.
//
//   left half  (surface):  tangent normal x, y; occlusion (AO darkened by cavity); emission mask
//   right half (response): roughness; metallic; specular F0 (0: the material's own); edge wear
//
// One sampler on unit 30: the mesh shader is at MoltenVK's 16 samplers a stage without it, and
// vkgl has no texture arrays.

#pragma once

#include <array>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace openrac::renderer {
class Shader;
}

namespace openrac::scene {
struct LevelData;
}

namespace openrac::enhanced {

class MaterialSidecars {
public:
    static constexpr unsigned kUnit = 30;

    MaterialSidecars() = default;
    MaterialSidecars(const MaterialSidecars&) = delete;
    MaterialSidecars& operator=(const MaterialSidecars&) = delete;
    ~MaterialSidecars();

    // The maps of every material of `level` whose texture a pack has maps for.
    void load(const scene::LevelData& level);
    void release();

    // The maps of the texture of `hash` (0 or one without maps: neutral, and switched off).
    void bind(const renderer::Shader& shader, std::uint64_t hash) const;

    // Whether the texture of `hash` has an emission mask that lights anything.
    bool glows(std::uint64_t hash) const;
    // Its emission at a texture coordinate (repeating), 0..1; 0 without a mask.
    float emission(std::uint64_t hash, float u, float v) const;
    // Its response at a texture coordinate (repeating): roughness, metallic and specular F0
    // (0: none), 0..1 each. False without maps.
    bool response(std::uint64_t hash, float u, float v, std::array<float, 3>& out) const;

    std::size_t size() const { return m_sets.size(); }

private:
    struct Set {
        unsigned texture = 0;
        // The emission mask at a reduced size (at most 64 x 64), for the ray tracer's lights.
        int mask_width = 0, mask_height = 0;
        std::vector<std::uint8_t> mask;
        // Roughness, metallic and F0 averaged to 16 x 16, for the ray tracer's materials.
        int response_width = 0, response_height = 0;
        std::vector<std::uint8_t> response;
    };
    void ensure_neutral() const;

    std::unordered_map<std::uint64_t, Set> m_sets;
    mutable unsigned m_neutral = 0;
};

}  // namespace openrac::enhanced
