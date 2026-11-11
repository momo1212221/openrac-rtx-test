// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 the OpenRAC contributors

#include "enhanced/renderer/material_sidecars.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>

#include "common/log.h"
#include "renderer/gl.h"
#include "renderer/shader.h"
#include "renderer/texture_pack.h"
#include "scene/level.h"

namespace openrac::enhanced {
namespace {
using renderer::Rgba8Image;
using renderer::texture_pack::MaterialMap;

// Below this an emission mask lights nothing (8-bit noise of a dark map).
constexpr std::uint8_t kGlowFloor = 40;
constexpr int kMaskSize = 64;

unsigned upload(const Rgba8Image& image, unsigned unit) {
    unsigned texture = 0;
    gl::glGenTextures(1, &texture);
    gl::glActiveTexture(gl::GL_TEXTURE0 + unit);
    gl::glBindTexture(gl::GL_TEXTURE_2D, texture);
    gl::glTexImage2D(
        gl::GL_TEXTURE_2D,
        0,
        gl::GL_RGBA8,
        image.width,
        image.height,
        0,
        gl::GL_RGBA,
        gl::GL_UNSIGNED_BYTE,
        image.pixels.data()
    );
    const bool mipmapped = image.width > 1 || image.height > 1;
    gl::glTexParameteri(
        gl::GL_TEXTURE_2D,
        gl::GL_TEXTURE_MIN_FILTER,
        mipmapped ? gl::GL_LINEAR_MIPMAP_LINEAR : gl::GL_LINEAR
    );
    gl::glTexParameteri(gl::GL_TEXTURE_2D, gl::GL_TEXTURE_MAG_FILTER, gl::GL_LINEAR);
    // The two halves are separate maps: mesh.frag repeats within each one itself.
    gl::glTexParameteri(gl::GL_TEXTURE_2D, gl::GL_TEXTURE_WRAP_S, gl::GL_CLAMP_TO_EDGE);
    gl::glTexParameteri(gl::GL_TEXTURE_2D, gl::GL_TEXTURE_WRAP_T, gl::GL_REPEAT);
    if (mipmapped) {
        gl::glGenerateMipmap(gl::GL_TEXTURE_2D);
    }
    gl::glBindTexture(gl::GL_TEXTURE_2D, 0);
    gl::glActiveTexture(gl::GL_TEXTURE0);
    return texture;
}

Rgba8Image filled(int width, int height, std::array<std::uint8_t, 4> value) {
    Rgba8Image image(width, height);
    for (std::size_t i = 0; i < image.pixels.size(); i += 4) {
        std::copy(value.begin(), value.end(), image.pixels.begin() + static_cast<std::ptrdiff_t>(i));
    }
    return image;
}

// One map of a set, when the pack has it at the set's size.
std::optional<Rgba8Image> map_of(std::uint64_t hash, MaterialMap map, int width, int height) {
    auto image = renderer::texture_pack::load_material_map(hash, map);
    if (image && (image->width != width || image->height != height)) {
        log::warn(
            "material sidecar {:016x}: {} is {}x{}, not {}x{} as the others; it is left out",
            hash,
            renderer::texture_pack::material_map_file(map),
            image->width,
            image->height,
            width,
            height
        );
        image.reset();
    }
    return image;
}

std::uint8_t at(const std::optional<Rgba8Image>& image, std::size_t i, std::uint8_t fallback) {
    return image ? image->pixels[i] : fallback;
}
}  // namespace

MaterialSidecars::~MaterialSidecars() {
    release();
}

void MaterialSidecars::ensure_neutral() const {
    if (m_neutral != 0) {
        return;
    }
    // Every declared sampler needs an image under Vulkan, maps or not.
    Rgba8Image neutral = filled(2, 1, {128, 128, 255, 0});
    neutral.pixels[4] = 128;
    neutral.pixels[5] = neutral.pixels[6] = neutral.pixels[7] = 0;
    m_neutral = upload(neutral, kUnit);
}

void MaterialSidecars::load(const scene::LevelData& level) {
    release();
    std::size_t glowing = 0, metallic = 0;
    for (const auto& material : level.materials) {
        const std::uint64_t hash = material.texture_pack_hash;
        if (hash == 0 || m_sets.contains(hash)
            || !renderer::texture_pack::has_material_maps(hash)) {
            continue;
        }
        // The size of the first map found is the set's.
        std::optional<Rgba8Image> first;
        for (std::size_t m = 0; m < renderer::texture_pack::kMaterialMapCount && !first; ++m) {
            first = renderer::texture_pack::load_material_map(hash, static_cast<MaterialMap>(m));
        }
        if (!first) {
            continue;
        }
        const int width = first->width, height = first->height;
        const auto normal = map_of(hash, MaterialMap::Normal, width, height);
        const auto ao = map_of(hash, MaterialMap::Ao, width, height);
        const auto cavity = map_of(hash, MaterialMap::Cavity, width, height);
        const auto emissive = map_of(hash, MaterialMap::Emissive, width, height);
        const auto roughness = map_of(hash, MaterialMap::Roughness, width, height);
        const auto metal = map_of(hash, MaterialMap::Metallic, width, height);
        const auto specular = map_of(hash, MaterialMap::Specular, width, height);
        const auto wear = map_of(hash, MaterialMap::EdgeWear, width, height);

        // Side by side: surface on the left, response on the right.
        Rgba8Image packed(width * 2, height);
        Rgba8Image surface(width, height), response(width, height);
        bool lights = false, any_metal = false;
        for (std::size_t i = 0; i < surface.pixels.size(); i += 4) {
            surface.pixels[i] = at(normal, i, 128);
            surface.pixels[i + 1] = at(normal, i + 1, 128);
            const float occlusion = at(ao, i, 255) / 255.0f
                                    * (1.0f - 0.18f * (at(cavity, i, 0) / 255.0f));
            surface.pixels[i + 2] = static_cast<std::uint8_t>(std::lround(occlusion * 255.0f));
            // The mask is the brightest channel: the hue comes from the colour texture.
            const std::uint8_t glow =
                emissive ? std::max({emissive->pixels[i], emissive->pixels[i + 1],
                                     emissive->pixels[i + 2]})
                         : std::uint8_t{0};
            surface.pixels[i + 3] = glow;
            lights = lights || glow >= kGlowFloor;
            response.pixels[i] = at(roughness, i, 128);
            response.pixels[i + 1] = at(metal, i, 0);
            // 0 keeps the material's own F0; a real specular map never goes that low.
            response.pixels[i + 2] = specular ? std::max<std::uint8_t>(specular->pixels[i], 1) : 0;
            response.pixels[i + 3] = at(wear, i, 0);
            any_metal = any_metal || response.pixels[i + 1] >= 128;
        }

        for (int y = 0; y < height; ++y) {
            const auto row = static_cast<std::size_t>(y) * width * 4;
            const auto out = static_cast<std::size_t>(y) * width * 8;
            std::copy_n(surface.pixels.begin() + row, width * 4, packed.pixels.begin() + out);
            std::copy_n(
                response.pixels.begin() + row, width * 4, packed.pixels.begin() + out + width * 4
            );
        }
        Set set;
        set.texture = upload(packed, kUnit);
        {
            constexpr int kResponseSize = 16;
            set.response_width = std::min(width, kResponseSize);
            set.response_height = std::min(height, kResponseSize);
            const std::size_t cells =
                static_cast<std::size_t>(set.response_width) * set.response_height;
            std::vector<std::uint32_t> sums(cells * 4, 0);
            for (int y = 0; y < height; ++y) {
                const int ry = y * set.response_height / height;
                for (int x = 0; x < width; ++x) {
                    const int rx = x * set.response_width / width;
                    const std::size_t at = static_cast<std::size_t>(ry) * set.response_width + rx;
                    const std::size_t i = (static_cast<std::size_t>(y) * width + x) * 4;
                    for (int c = 0; c < 3; ++c) {
                        sums[at * 4 + c] += response.pixels[i + c];
                    }
                    ++sums[at * 4 + 3];
                }
            }
            set.response.resize(cells * 3);
            for (std::size_t at = 0; at < cells; ++at) {
                const std::uint32_t n = std::max<std::uint32_t>(sums[at * 4 + 3], 1);
                for (int c = 0; c < 3; ++c) {
                    set.response[at * 3 + c] = static_cast<std::uint8_t>(sums[at * 4 + c] / n);
                }
            }
        }
        if (lights) {
            set.mask_width = std::min(width, kMaskSize);
            set.mask_height = std::min(height, kMaskSize);
            set.mask.assign(static_cast<std::size_t>(set.mask_width) * set.mask_height, 0);
            // The brightest texel of each cell, so that a small bulb survives the reduction.
            for (int y = 0; y < height; ++y) {
                const int my = y * set.mask_height / height;
                for (int x = 0; x < width; ++x) {
                    const int mx = x * set.mask_width / width;
                    auto& cell = set.mask[static_cast<std::size_t>(my) * set.mask_width + mx];
                    cell = std::max(
                        cell, surface.pixels[(static_cast<std::size_t>(y) * width + x) * 4 + 3]
                    );
                }
            }
            ++glowing;
        }
        metallic += any_metal ? 1 : 0;
        m_sets.emplace(hash, std::move(set));
    }
    if (!m_sets.empty()) {
        log::info(
            "material sidecars: {} texture(s) with maps, {} glow, {} metal",
            m_sets.size(),
            glowing,
            metallic
        );
    }
}

void MaterialSidecars::release() {
    for (auto& [hash, set] : m_sets) {
        (void)hash;
        gl::glDeleteTextures(1, &set.texture);
    }
    m_sets.clear();
    if (m_neutral != 0) {
        gl::glDeleteTextures(1, &m_neutral);
        m_neutral = 0;
    }
}

void MaterialSidecars::bind(const renderer::Shader& shader, std::uint64_t hash) const {
    const int enabled = shader.uniform("material_sidecars_enabled");
    if (enabled < 0) {
        return;
    }
    ensure_neutral();
    const auto found = hash != 0 ? m_sets.find(hash) : m_sets.end();
    const bool has = found != m_sets.end();
    gl::glUniform1i(enabled, has ? 1 : 0);
    gl::glUniform1i(shader.uniform("material_sidecar_maps"), static_cast<int>(kUnit));
    gl::glActiveTexture(gl::GL_TEXTURE0 + kUnit);
    gl::glBindTexture(gl::GL_TEXTURE_2D, has ? found->second.texture : m_neutral);
    gl::glActiveTexture(gl::GL_TEXTURE0);
}

bool MaterialSidecars::glows(std::uint64_t hash) const {
    const auto found = hash != 0 ? m_sets.find(hash) : m_sets.end();
    return found != m_sets.end() && !found->second.mask.empty();
}

float MaterialSidecars::emission(std::uint64_t hash, float u, float v) const {
    const auto found = hash != 0 ? m_sets.find(hash) : m_sets.end();
    if (found == m_sets.end() || found->second.mask.empty()) {
        return 0.0f;
    }
    const Set& set = found->second;
    const float fu = u - std::floor(u), fv = v - std::floor(v);
    const int x = std::clamp(static_cast<int>(fu * set.mask_width), 0, set.mask_width - 1);
    const int y = std::clamp(static_cast<int>(fv * set.mask_height), 0, set.mask_height - 1);
    const std::uint8_t value = set.mask[static_cast<std::size_t>(y) * set.mask_width + x];
    return value >= kGlowFloor ? value / 255.0f : 0.0f;
}

bool MaterialSidecars::response(
    std::uint64_t hash, float u, float v, std::array<float, 3>& out
) const {
    const auto found = hash != 0 ? m_sets.find(hash) : m_sets.end();
    if (found == m_sets.end() || found->second.response.empty()) {
        return false;
    }
    const Set& set = found->second;
    const float fu = u - std::floor(u), fv = v - std::floor(v);
    const int x = std::clamp(static_cast<int>(fu * set.response_width), 0, set.response_width - 1);
    const int y = std::clamp(static_cast<int>(fv * set.response_height), 0, set.response_height - 1);
    const std::size_t at = (static_cast<std::size_t>(y) * set.response_width + x) * 3;
    for (int c = 0; c < 3; ++c) {
        out[c] = set.response[at + c] / 255.0f;
    }
    return true;
}

}  // namespace openrac::enhanced
