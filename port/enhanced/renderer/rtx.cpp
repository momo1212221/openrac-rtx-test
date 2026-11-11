// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 the OpenRAC contributors
//
// Native hardware ray-tracing pipeline (RTX) for Enhanced Graphics:
// SAH BVH construction, dynamic moby instance updates, Vulkan compute/ray-query dispatch,
// and hardware backend orchestration (Apple Metal RT + NVIDIA CUDA RTX).

#include "enhanced/renderer/rtx.h"
#include "enhanced/renderer/material_sidecars.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <numeric>
#include <optional>
#include <unordered_map>

#include "common/log.h"
#include "common/stats.h"
#include "enhanced/renderer/compute_graph.h"
#include "enhanced_viewer_shaders.h"
#include "renderer/gl.h"
#include "renderer/texture_pool.h"
#if defined(OPENRAC_VULKAN)
#include "renderer/vulkan/native_compute.h"
#include "renderer/vulkan/native_metal.h"
#endif
#if defined(OPENRAC_METAL)
#include "renderer/metal/mtlgl.h"
#endif
#include "renderer/world/gl_world.h"
#include "renderer_shaders.h"

namespace openrac::enhanced {
namespace {
using namespace openrac::gl;
namespace wgl = renderer::world::gl;

#if defined(OPENRAC_VULKAN) && defined(__APPLE__)
// The Metal objects of whichever renderer draws: Metal's own GL (renderer/metal), natively, or
// Vulkan through MoltenVK, which lends its device and buffers (renderer/vulkan/native_metal.h).
namespace native {
bool available() {
#if defined(OPENRAC_METAL)
    if (is_metal()) {
        return mtlgl::native().device != nullptr;
    }
#endif
    return is_vulkan() && vkgl::metal_available();
}

void* device() {
#if defined(OPENRAC_METAL)
    if (is_metal()) {
        return mtlgl::native().device;
    }
#endif
    return vkgl::metal_device();
}

// The MTLBuffer behind a GL buffer. Metal's GL gives each GL buffer an MTLBuffer of its own.
bool buffer(GLuint name, vkgl::MetalBuffer& out) {
#if defined(OPENRAC_METAL)
    if (is_metal()) {
        out = {mtlgl::native_buffer(name), 0, 0};
        return out.buffer != nullptr;
    }
#endif
    return vkgl::metal_buffer(name, out);
}

std::uint64_t recording_frame() {
#if defined(OPENRAC_METAL)
    if (is_metal()) {
        return mtlgl::recording_frame();
    }
#endif
    return vkgl::recording_frame();
}

std::uint64_t completed_frame() {
#if defined(OPENRAC_METAL)
    if (is_metal()) {
        return mtlgl::completed_frame();
    }
#endif
    return vkgl::completed_frame();
}
}  // namespace native
#endif

std::mutex g_rtx_mutex;
RtxStatus g_rtx_status{};
RtxDebugView g_rtx_debug_view = RtxDebugView::Off;

float uint_bits_to_float(std::uint32_t u) {
    float f = 0.0f;
    std::memcpy(&f, &u, sizeof(float));
    return f;
}

float int_bits_to_float(std::int32_t i) {
    float f = 0.0f;
    std::memcpy(&f, &i, sizeof(float));
    return f;
}

std::uint32_t float_bits_to_uint(float f) {
    std::uint32_t u = 0;
    std::memcpy(&u, &f, sizeof(std::uint32_t));
    return u;
}

std::uint32_t pack_rgb8(float r, float g, float b) {
    const auto cr = static_cast<std::uint32_t>(std::clamp(r, 0.0f, 1.0f) * 255.0f + 0.5f);
    const auto cg = static_cast<std::uint32_t>(std::clamp(g, 0.0f, 1.0f) * 255.0f + 0.5f);
    const auto cb = static_cast<std::uint32_t>(std::clamp(b, 0.0f, 1.0f) * 255.0f + 0.5f);
    return cr | (cg << 8u) | (cb << 16u) | 0xFF000000u;
}

std::array<float, 3> transform_point(const renderer::Mat4& m, const float p[3]) {
    return {
        m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12],
        m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13],
        m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14]
    };
}

// Samples average colour from a level texture or material tint for ray-traced secondary bounces.
std::uint32_t sample_material_albedo(
    const scene::LevelData& level, const scene::Material& mat, const float vcol[4]
) {
    float r = mat.colour[0] * std::clamp(vcol[0], 0.25f, 1.5f);
    float g = mat.colour[1] * std::clamp(vcol[1], 0.25f, 1.5f);
    float b = mat.colour[2] * std::clamp(vcol[2], 0.25f, 1.5f);
    if (mat.image >= 0 && static_cast<std::size_t>(mat.image) < level.images.size()) {
        const auto& img = level.images[static_cast<std::size_t>(mat.image)];
        if (img.width > 0 && img.height > 0 && img.pixels.size() >= 4) {
            const std::size_t cx = static_cast<std::size_t>(img.width / 2);
            const std::size_t cy = static_cast<std::size_t>(img.height / 2);
            const std::size_t offset = (cy * static_cast<std::size_t>(img.width) + cx) * 4;
            if (offset + 3 < img.pixels.size()) {
                r *= static_cast<float>(img.pixels[offset + 0]) * (1.0f / 255.0f);
                g *= static_cast<float>(img.pixels[offset + 1]) * (1.0f / 255.0f);
                b *= static_cast<float>(img.pixels[offset + 2]) * (1.0f / 255.0f);
            }
        }
    }
    return pack_rgb8(r, g, b);
}

struct BvhBuildEntry {
    std::array<float, 3> bmin{};
    std::array<float, 3> bmax{};
    std::array<float, 3> centroid{};
};

void build_sah_bvh_recursive(
    std::vector<RtxBvhNode>& nodes,
    std::vector<std::uint32_t>& tri_indices,
    std::span<const BvhBuildEntry> entries,
    std::uint32_t node_idx,
    std::uint32_t start,
    std::uint32_t count
) {
    std::array<float, 3> bmin{
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max()
    };
    std::array<float, 3> bmax{
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest()
    };
    std::array<float, 3> cmin = bmin;
    std::array<float, 3> cmax = bmax;

    for (std::uint32_t i = start; i < start + count; ++i) {
        const auto& e = entries[tri_indices[i]];
        for (int a = 0; a < 3; ++a) {
            bmin[a] = std::min(bmin[a], e.bmin[a]);
            bmax[a] = std::max(bmax[a], e.bmax[a]);
            cmin[a] = std::min(cmin[a], e.centroid[a]);
            cmax[a] = std::max(cmax[a], e.centroid[a]);
        }
    }

    nodes[node_idx].bbox_min = bmin;
    nodes[node_idx].bbox_max = bmax;

    constexpr std::uint32_t kMaxLeafTris = 4;
    if (count <= kMaxLeafTris) {
        nodes[node_idx].left_or_first = ~static_cast<std::int32_t>(start);
        nodes[node_idx].count = static_cast<std::int32_t>(count);
        return;
    }

    int axis = 0;
    float extent = cmax[0] - cmin[0];
    if (cmax[1] - cmin[1] > extent) {
        axis = 1;
        extent = cmax[1] - cmin[1];
    }
    if (cmax[2] - cmin[2] > extent) {
        axis = 2;
        extent = cmax[2] - cmin[2];
    }

    std::uint32_t mid = start + count / 2;
    if (extent > 1e-5f) {
        const float split = 0.5f * (cmin[axis] + cmax[axis]);
        auto it = std::partition(
            tri_indices.begin() + start,
            tri_indices.begin() + start + count,
            [&](std::uint32_t idx) { return entries[idx].centroid[axis] < split; }
        );
        mid = static_cast<std::uint32_t>(std::distance(tri_indices.begin(), it));
        if (mid == start || mid == start + count) {
            mid = start + count / 2;
            std::nth_element(
                tri_indices.begin() + start,
                tri_indices.begin() + mid,
                tri_indices.begin() + start + count,
                [&](std::uint32_t a, std::uint32_t b) {
                    return entries[a].centroid[axis] < entries[b].centroid[axis];
                }
            );
        }
    }

    const std::uint32_t left_idx = static_cast<std::uint32_t>(nodes.size());
    nodes.emplace_back();
    nodes.emplace_back();
    nodes[node_idx].left_or_first = static_cast<std::int32_t>(left_idx);
    nodes[node_idx].count = 0;

    build_sah_bvh_recursive(nodes, tri_indices, entries, left_idx, start, mid - start);
    build_sah_bvh_recursive(nodes, tri_indices, entries, left_idx + 1, mid, start + count - mid);
}

// CPU SAH-BVH ray traversal fallback when neither Metal nor CUDA is requested/present: one tree,
// keeping `out_hit` when it is nearer (the static tree's, then the mobys'), its triangles
// numbered from `first_triangle`.
void trace_cpu_bvh(
    std::span<const RtxBvhNode> nodes,
    std::span<const RtxTriangle> tris,
    const RtxRay& ray,
    RtxHit& out_hit,
    std::int32_t first_triangle = 0
) {
    if (nodes.empty() || tris.empty()) {
        return;
    }
    const std::array<float, 3> inv_dir{
        1.0f / (std::fabs(ray.direction[0]) > 1e-8f ? ray.direction[0]
                                                    : (ray.direction[0] >= 0.0f ? 1e-8f : -1e-8f)),
        1.0f / (std::fabs(ray.direction[1]) > 1e-8f ? ray.direction[1]
                                                    : (ray.direction[1] >= 0.0f ? 1e-8f : -1e-8f)),
        1.0f / (std::fabs(ray.direction[2]) > 1e-8f ? ray.direction[2]
                                                    : (ray.direction[2] >= 0.0f ? 1e-8f : -1e-8f))
    };

    std::int32_t stack[64];
    int sp = 0;
    stack[sp++] = 0;
    float closest = out_hit.t >= 0.0f ? std::min(out_hit.t, ray.t_max) : ray.t_max;

    while (sp > 0) {
        const std::int32_t idx = stack[--sp];
        const auto& node = nodes[static_cast<std::size_t>(idx)];
        float tmin = ray.t_min;
        float tmax = closest;
        for (int a = 0; a < 3; ++a) {
            const float t0 = (node.bbox_min[a] - ray.origin[a]) * inv_dir[a];
            const float t1 = (node.bbox_max[a] - ray.origin[a]) * inv_dir[a];
            tmin = std::max(tmin, std::min(t0, t1));
            tmax = std::min(tmax, std::max(t0, t1));
        }
        if (tmax < tmin) {
            continue;
        }
        if (node.count > 0) {
            const std::int32_t first = ~node.left_or_first;
            for (std::int32_t i = 0; i < node.count; ++i) {
                const std::int32_t tri_idx = first + i;
                const auto& tri = tris[static_cast<std::size_t>(tri_idx)];
                const float e1[3]{tri.e1_layer[0], tri.e1_layer[1], tri.e1_layer[2]};
                const float e2[3]{tri.e2_flags[0], tri.e2_flags[1], tri.e2_flags[2]};
                const float pvec[3]{
                    ray.direction[1] * e2[2] - ray.direction[2] * e2[1],
                    ray.direction[2] * e2[0] - ray.direction[0] * e2[2],
                    ray.direction[0] * e2[1] - ray.direction[1] * e2[0]
                };
                const float det = e1[0] * pvec[0] + e1[1] * pvec[1] + e1[2] * pvec[2];
                if (std::fabs(det) < 1e-8f) {
                    continue;
                }
                const float inv_det = 1.0f / det;
                const float tvec[3]{
                    ray.origin[0] - tri.v0_mat[0],
                    ray.origin[1] - tri.v0_mat[1],
                    ray.origin[2] - tri.v0_mat[2]
                };
                const float u =
                    (tvec[0] * pvec[0] + tvec[1] * pvec[1] + tvec[2] * pvec[2]) * inv_det;
                if (u < 0.0f || u > 1.0f) {
                    continue;
                }
                const float qvec[3]{
                    tvec[1] * e1[2] - tvec[2] * e1[1],
                    tvec[2] * e1[0] - tvec[0] * e1[2],
                    tvec[0] * e1[1] - tvec[1] * e1[0]
                };
                const float v = (ray.direction[0] * qvec[0] + ray.direction[1] * qvec[1]
                                 + ray.direction[2] * qvec[2])
                                * inv_det;
                if (v < 0.0f || u + v > 1.0f) {
                    continue;
                }
                const float t =
                    (e2[0] * qvec[0] + e2[1] * qvec[1] + e2[2] * qvec[2]) * inv_det;
                if (t >= ray.t_min && t < closest) {
                    closest = t;
                    out_hit.t = t;
                    out_hit.triangle_index = first_triangle + tri_idx;
                    out_hit.u = u;
                    out_hit.v = v;
                    out_hit.normal = {
                        tri.normal_albedo[0], tri.normal_albedo[1], tri.normal_albedo[2]
                    };
                    out_hit.packed_albedo = float_bits_to_uint(tri.normal_albedo[3]);
                }
            }
        } else if (sp + 2 <= 64) {
            stack[sp++] = node.left_or_first + 1;
            stack[sp++] = node.left_or_first;
        }
    }
}

// Whether a material glows (probe-emission.json's runtime emission).
bool glows(const scene::Material& m) {
    return std::max({m.emission[0], m.emission[1], m.emission[2]}) > 1e-4f;
}

// How much of a triangle a texture pack's emission mask marks (0..1): the mask at its centre and
// half-way to each corner, averaged, so that a wall with one bright texel stays a wall.
float sidecar_glow(
    const MaterialSidecars* sidecars, const scene::Material& mat, const scene::Vertex& a,
    const scene::Vertex& b, const scene::Vertex& c
) {
    if (sidecars == nullptr || !sidecars->glows(mat.texture_pack_hash)) {
        return 0.0f;
    }
    const float cu = (a.uv[0] + b.uv[0] + c.uv[0]) / 3.0f;
    const float cv = (a.uv[1] + b.uv[1] + c.uv[1]) / 3.0f;
    float sum = sidecars->emission(mat.texture_pack_hash, cu, cv);
    for (const auto* corner : {&a, &b, &c}) {
        sum += sidecars->emission(
            mat.texture_pack_hash, 0.5f * (cu + corner->uv[0]), 0.5f * (cv + corner->uv[1])
        );
    }
    const float glow = sum * 0.25f;
    return glow >= 0.3f ? glow : 0.0f;
}

// The colour texture's texel at a texture coordinate (repeating), times the material colour.
std::array<float, 3> texel_at(const scene::LevelData& level, const scene::Material& mat, float u,
                              float v) {
    std::array<float, 3> rgb{mat.colour[0], mat.colour[1], mat.colour[2]};
    if (mat.image < 0 || static_cast<std::size_t>(mat.image) >= level.images.size()) {
        return rgb;
    }
    const auto& img = level.images[static_cast<std::size_t>(mat.image)];
    if (img.width <= 0 || img.height <= 0) {
        return rgb;
    }
    const int x = std::clamp(static_cast<int>((u - std::floor(u)) * img.width), 0, img.width - 1);
    const int y = std::clamp(static_cast<int>((v - std::floor(v)) * img.height), 0, img.height - 1);
    const std::size_t offset = (static_cast<std::size_t>(y) * img.width + x) * 4;
    for (int a = 0; a < 3; ++a) {
        rgb[a] *= img.pixels[offset + a] / 255.0f;
    }
    return rgb;
}

// Sums one instance's glowing triangles (area-weighted centre, normal and colour) under `matrix`.
struct GlowSum {
    std::array<float, 3> centre{}, normal{}, colour{};
    float area = 0;
};

void sum_glow(
    const scene::LevelData& level, const scene::Model& model, const renderer::Mat4& matrix,
    const MaterialSidecars* sidecars, GlowSum& sum
) {
    for (const auto& prim : model.primitives) {
        if (prim.material < 0 || static_cast<std::size_t>(prim.material) >= level.materials.size()) {
            continue;
        }
        const auto& mat = level.materials[static_cast<std::size_t>(prim.material)];
        const bool emissive = glows(mat);
        for (std::uint32_t i = 0; i + 2 < prim.index_count; i += 3) {
            const std::size_t at = static_cast<std::size_t>(prim.first_index + i);
            if (at + 2 >= level.indices.size()) {
                break;
            }
            const auto i0 = level.indices[at], i1 = level.indices[at + 1], i2 = level.indices[at + 2];
            if (i0 >= level.vertices.size() || i1 >= level.vertices.size()
                || i2 >= level.vertices.size()) {
                continue;
            }
            // The game's own glow (a glow packet's vertices, light slot -2: drawn unlit and
            // bright) counts as well as probe-emission.json's emissive materials.
            const bool packet = level.vertices[i0].light_slot == -2
                                || level.vertices[i1].light_slot == -2
                                || level.vertices[i2].light_slot == -2;
            // And what a texture pack's emission mask marks (material_sidecars.h).
            const float mask = emissive || packet
                                   ? 0.0f
                                   : sidecar_glow(sidecars, mat, level.vertices[i0],
                                                  level.vertices[i1], level.vertices[i2]);
            if (!emissive && !packet && mask <= 0.0f) {
                continue;
            }
            const auto p0 = transform_point(matrix, level.vertices[i0].position);
            const auto p1 = transform_point(matrix, level.vertices[i1].position);
            const auto p2 = transform_point(matrix, level.vertices[i2].position);
            const std::array<float, 3> e1{p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]};
            const std::array<float, 3> e2{p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]};
            const std::array<float, 3> n{
                e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2],
                e1[0] * e2[1] - e1[1] * e2[0]
            };
            const float area = 0.5f * std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
            if (area < 1e-6f) {
                continue;
            }
            const std::uint32_t albedo = sample_material_albedo(level, mat, level.vertices[i0].colour);
            float rgb[3]{
                float(albedo & 0xFFu) / 255.0f, float((albedo >> 8u) & 0xFFu) / 255.0f,
                float((albedo >> 16u) & 0xFFu) / 255.0f
            };
            if (mask > 0.0f) {
                // The glowing texels' own colour, normalised to the mask's brightness (a white
                // bulb on a dark lamp shade glows white, not the shade's average).
                const auto& va = level.vertices[i0];
                const auto& vb = level.vertices[i1];
                const auto& vc = level.vertices[i2];
                const auto texel = texel_at(level, mat, (va.uv[0] + vb.uv[0] + vc.uv[0]) / 3.0f,
                                            (va.uv[1] + vb.uv[1] + vc.uv[1]) / 3.0f);
                const float peak = std::max({texel[0], texel[1], texel[2], 1.0f / 255.0f});
                for (int a = 0; a < 3; ++a) {
                    rgb[a] = texel[a] / peak * mask;
                }
            }
            for (int a = 0; a < 3; ++a) {
                sum.centre[a] += (p0[a] + p1[a] + p2[a]) / 3.0f * area;
                sum.normal[a] += n[a] * 0.5f;
                sum.colour[a] += rgb[a] * (emissive ? mat.emission[a] : 1.0f) * area;
            }
            sum.area += area;
        }
    }
}

// A glowing part as a point light: a little off the surface it glows from, its colour the
// surface's, brighter and reaching further the larger it is.
LocalLight glow_light(const std::array<float, 3>& centre, const std::array<float, 3>& normal,
                      const std::array<float, 3>& colour, float area) {
    const float n = std::sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
    const float size = std::sqrt(area);
    LocalLight light;
    for (int a = 0; a < 3; ++a) {
        light.position[a] = centre[a] + (n > 1e-6f ? normal[a] / n * 0.2f : 0.0f);
        light.colour[a] = colour[a] * 1.5f * std::clamp(size, 0.5f, 2.5f);
    }
    light.range = std::clamp(4.0f + 4.0f * size, 5.0f, 16.0f);
    return light;
}

// A triangle's material for the paths, in e2_flags.w's bits: roughness, metallic, specular F0
// (0: none), a byte each from the low one.
std::uint32_t pack_material(const std::array<float, 3>& response) {
    std::uint32_t packed = 0;
    for (int k = 0; k < 3; ++k) {
        packed |= static_cast<std::uint32_t>(std::lround(std::clamp(response[k], 0.0f, 1.0f) * 255.0f))
                  << (8u * k);
    }
    return packed;
}

// A glowing triangle for the emitter structure, under `matrix`: its glow's colour (peak 1,
// packed) and strength, as sum_glow finds them. OPENRAC_RTX_GLOW scales the strength (default 2).
std::optional<RtxTriangle> emitter_triangle(
    const scene::LevelData& level, const scene::Material& mat, const renderer::Mat4& matrix,
    const scene::Vertex& a, const scene::Vertex& b, const scene::Vertex& c, bool emissive,
    float mask
) {
    static const float strength = [] {
        const char* v = std::getenv("OPENRAC_RTX_GLOW");
        const float f = v != nullptr ? std::strtof(v, nullptr) : 0.0f;
        return f > 0.0f ? f : 2.0f;
    }();
    const auto p0 = transform_point(matrix, a.position);
    const auto p1 = transform_point(matrix, b.position);
    const auto p2 = transform_point(matrix, c.position);
    const std::array<float, 3> e1{p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]};
    const std::array<float, 3> e2{p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]};
    std::array<float, 3> n{
        e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]
    };
    const float len = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    if (len < 1e-6f) {
        return std::nullopt;
    }
    std::array<float, 3> rgb{};
    if (mask > 0.0f) {
        // The glowing texels' own colour (a white bulb on a dark shade glows white).
        const auto texel = texel_at(level, mat, (a.uv[0] + b.uv[0] + c.uv[0]) / 3.0f,
                                    (a.uv[1] + b.uv[1] + c.uv[1]) / 3.0f);
        const float peak = std::max({texel[0], texel[1], texel[2], 1.0f / 255.0f});
        for (int k = 0; k < 3; ++k) {
            rgb[k] = texel[k] / peak * mask;
        }
    } else {
        const std::uint32_t albedo = sample_material_albedo(level, mat, a.colour);
        for (int k = 0; k < 3; ++k) {
            rgb[k] = float((albedo >> (8u * k)) & 0xFFu) / 255.0f * (emissive ? mat.emission[k] : 1.0f);
        }
    }
    const float peak = std::max({rgb[0], rgb[1], rgb[2]});
    if (peak <= 1e-3f) {
        return std::nullopt;
    }
    std::uint32_t packed = 0xFF000000u;
    for (int k = 0; k < 3; ++k) {
        packed |= static_cast<std::uint32_t>(std::lround(std::clamp(rgb[k] / peak, 0.0f, 1.0f) * 255.0f))
                  << (8u * k);
    }
    RtxTriangle tri{};
    tri.v0_mat = {p0[0], p0[1], p0[2], 0.0f};
    tri.e1_layer = {e1[0], e1[1], e1[2], 0.0f};
    tri.e2_flags = {e2[0], e2[1], e2[2], peak * strength};
    tri.normal_albedo = {n[0] / len, n[1] / len, n[2] / len, uint_bits_to_float(packed)};
    return tri;
}

// Builds the SAH tree of `tris` into `nodes` and reorders `tris` to its leaves' order.
void build_tree(std::vector<RtxTriangle>& tris, std::vector<RtxBvhNode>& nodes) {
    nodes.clear();
    if (tris.empty()) {
        return;
    }
    std::vector<BvhBuildEntry> entries(tris.size());
    std::vector<std::uint32_t> order(tris.size());
    for (std::size_t i = 0; i < tris.size(); ++i) {
        order[i] = static_cast<std::uint32_t>(i);
        const auto& t = tris[i];
        const std::array<float, 3> v0{t.v0_mat[0], t.v0_mat[1], t.v0_mat[2]};
        const std::array<float, 3> v1{
            v0[0] + t.e1_layer[0], v0[1] + t.e1_layer[1], v0[2] + t.e1_layer[2]
        };
        const std::array<float, 3> v2{
            v0[0] + t.e2_flags[0], v0[1] + t.e2_flags[1], v0[2] + t.e2_flags[2]
        };
        for (int a = 0; a < 3; ++a) {
            entries[i].bmin[a] = std::min({v0[a], v1[a], v2[a]}) - 1e-4f;
            entries[i].bmax[a] = std::max({v0[a], v1[a], v2[a]}) + 1e-4f;
            entries[i].centroid[a] = (v0[a] + v1[a] + v2[a]) * (1.0f / 3.0f);
        }
    }
    nodes.reserve(tris.size() * 2);
    nodes.emplace_back();
    build_sah_bvh_recursive(nodes, order, entries, 0, 0, static_cast<std::uint32_t>(tris.size()));
    std::vector<RtxTriangle> reordered(tris.size());
    for (std::size_t i = 0; i < order.size(); ++i) {
        reordered[i] = tris[order[i]];
    }
    tris = std::move(reordered);
}

// Nodes a ray may visit before its answer counts as unknown (rtx_trace.comp), by quality.
int step_budget(renderer::RtxQuality quality) {
    switch (quality) {
        case renderer::RtxQuality::Performance:
            return 192;
        case renderer::RtxQuality::Quality:
            return 1024;
        default:
            return 448;
    }
}

}  // namespace

#if !defined(__APPLE__)
bool metal_rtx_supported() {
    return false;
}

std::unique_ptr<RtxHardwareBackend> create_metal_rtx_backend() {
    return nullptr;
}

std::unique_ptr<RtxHardwareBackend> create_metal_rtx_backend_on(void*) {
    return nullptr;
}
#endif

RtxStatus rtx_status() {
    std::lock_guard lock(g_rtx_mutex);
    return g_rtx_status;
}

RtxDebugView rtx_debug_view() {
    std::lock_guard lock(g_rtx_mutex);
    return g_rtx_debug_view;
}

void set_rtx_debug_view(RtxDebugView view) {
    std::lock_guard lock(g_rtx_mutex);
    g_rtx_debug_view = view;
}

RtxPipeline::RtxPipeline() {
    // OPENRAC_RTX_VIEW=N starts in debug view N (1 shadows, 2 AO, 3 reflections, 4 bounce,
    // 5 the trace's step heat map), for hidden runs and screenshots.
    if (const char* view = std::getenv("OPENRAC_RTX_VIEW")) {
        const int n = std::atoi(view);
        if (n > 0 && n < static_cast<int>(RtxDebugView::Count)) {
            set_rtx_debug_view(static_cast<RtxDebugView>(n));
        }
    }
    m_metal_backend = create_metal_rtx_backend();
    m_cuda_backend = create_cuda_rtx_backend();

    bool vk_rq = false;
#if defined(OPENRAC_VULKAN)
    vkgl::NativeDevice dev{};
    if (vkgl::native_device(dev)) {
        vk_rq = dev.ray_query;
    }
#endif

    std::lock_guard lock(g_rtx_mutex);
    g_rtx_status.metal_supported = m_metal_backend != nullptr;
    g_rtx_status.cuda_supported = m_cuda_backend != nullptr;
    g_rtx_status.vulkan_ray_query_supported = vk_rq;
    g_rtx_status.available =
        g_rtx_status.metal_supported || g_rtx_status.cuda_supported || enhanced_rendering_available();
    if (m_metal_backend) {
        g_rtx_status.active_backend = renderer::RtxBackend::Metal;
        g_rtx_status.device_name = std::string(m_metal_backend->device_name());
    } else if (m_cuda_backend) {
        g_rtx_status.active_backend = renderer::RtxBackend::Cuda;
        g_rtx_status.device_name = std::string(m_cuda_backend->device_name());
    } else {
        g_rtx_status.active_backend = renderer::RtxBackend::VulkanRayQuery;
        g_rtx_status.device_name = vk_rq ? "Vulkan Hardware Ray Query" : "Vulkan Compute BVH";
    }
}

RtxPipeline::~RtxPipeline() {
    release();
}

RtxHardwareBackend* RtxPipeline::select_hardware_backend(renderer::RtxBackend requested) const {
    if (requested == renderer::RtxBackend::Metal && m_metal_backend) {
        return m_metal_backend.get();
    }
    if (requested == renderer::RtxBackend::Cuda && m_cuda_backend) {
        return m_cuda_backend.get();
    }
    if (requested == renderer::RtxBackend::Auto) {
        if (m_metal_backend) {
            return m_metal_backend.get();
        }
        if (m_cuda_backend) {
            return m_cuda_backend.get();
        }
    }
    return nullptr;
}

void RtxPipeline::upload(const scene::LevelData& level) {
    m_triangles.clear();
    m_emitter_triangles.clear();
    m_nodes.clear();
    m_moby_meshes.clear();
    m_dynamic_instance_count = 0;

    // `glowing`, when given, takes the glowing triangles (the emitter structure); else they are
    // left out as before.
    auto extract_instance = [&](const scene::Instance& inst,
                                scene::Layer layer,
                                std::vector<RtxTriangle>& out,
                                std::vector<RtxTriangle>* glowing = nullptr) {
        if (inst.stand_in || inst.lighting_only || inst.model >= level.models.size()) {
            return;
        }
        const auto& model = level.models[inst.model];
        for (const auto& prim : model.primitives) {
            if (prim.material >= 0
                && static_cast<std::size_t>(prim.material) < level.materials.size()) {
                const auto& mat = level.materials[static_cast<std::size_t>(prim.material)];
                // Blended and alpha-tested (cutout) surfaces are left out: the tree has no
                // texture to test, so a leaf card or a vine's quad would block as a solid wall,
                // blackening the ambient and casting square shadows. The shadow maps keep their
                // alpha-tested shadows (early_lighting.comp takes the darker of both).
                // Glowing surfaces are light sources (emissive_lights and the emitter structure), not
                // occluders: a lamp's bulb would shadow its own light. A glow is often blended.
                if (glows(mat) ? glowing == nullptr : (mat.blend || mat.cutout)) {
                    continue;
                }
            }
            const scene::Material default_mat{};
            const auto& mat =
                (prim.material >= 0
                 && static_cast<std::size_t>(prim.material) < level.materials.size())
                    ? level.materials[static_cast<std::size_t>(prim.material)]
                    : default_mat;
            // The game's chrome and glass maps (metal_*) are mirrors; else the pack's maps say,
            // per triangle (below); else a rough dielectric.
            const bool chrome = mat.metal;

            for (std::uint32_t i = 0; i + 2 < prim.index_count; i += 3) {
                const std::size_t idx_base = static_cast<std::size_t>(prim.first_index + i);
                if (idx_base + 2 >= level.indices.size()) {
                    break;
                }
                const std::uint32_t i0 = level.indices[idx_base + 0];
                const std::uint32_t i1 = level.indices[idx_base + 1];
                const std::uint32_t i2 = level.indices[idx_base + 2];
                if (i0 >= level.vertices.size() || i1 >= level.vertices.size()
                    || i2 >= level.vertices.size()) {
                    continue;
                }
                const auto& v0 = level.vertices[i0];
                const auto& v1 = level.vertices[i1];
                const auto& v2 = level.vertices[i2];
                // A glow packet is a light (emissive_lights), not an occluder, nor is what a
                // texture pack's emission mask marks. With `glowing` they go there instead.
                const bool emissive = glows(mat);
                const bool packet =
                    v0.light_slot == -2 || v1.light_slot == -2 || v2.light_slot == -2;
                const float mask =
                    emissive || packet ? 0.0f : sidecar_glow(m_sidecars, mat, v0, v1, v2);
                if (emissive || packet || mask > 0.0f) {
                    if (glowing != nullptr) {
                        if (auto tri = emitter_triangle(level, mat, inst.matrix, v0, v1, v2,
                                                        emissive, mask)) {
                            glowing->push_back(*tri);
                        }
                    }
                    continue;
                }
                const auto p0 = transform_point(inst.matrix, v0.position);
                const auto p1 = transform_point(inst.matrix, v1.position);
                const auto p2 = transform_point(inst.matrix, v2.position);
                const std::array<float, 3> e1{p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]};
                const std::array<float, 3> e2{p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]};
                std::array<float, 3> n{
                    e1[1] * e2[2] - e1[2] * e2[1],
                    e1[2] * e2[0] - e1[0] * e2[2],
                    e1[0] * e2[1] - e1[1] * e2[0]
                };
                const float len = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
                if (len < 1e-6f) {
                    continue;
                }
                n[0] /= len;
                n[1] /= len;
                n[2] /= len;

                const std::uint32_t albedo = sample_material_albedo(level, mat, v0.colour);
                std::array<float, 3> response{chrome ? 0.12f : 0.55f, chrome ? 1.0f : 0.0f, 0.0f};
                if (!chrome && m_sidecars != nullptr) {
                    m_sidecars->response(mat.texture_pack_hash, (v0.uv[0] + v1.uv[0] + v2.uv[0]) / 3.0f,
                                         (v0.uv[1] + v1.uv[1] + v2.uv[1]) / 3.0f, response);
                }
                RtxTriangle tri{};
                tri.v0_mat = {
                    p0[0], p0[1], p0[2], int_bits_to_float(static_cast<std::int32_t>(prim.material))
                };
                tri.e1_layer = {
                    e1[0],
                    e1[1],
                    e1[2],
                    int_bits_to_float(static_cast<std::int32_t>(layer))
                };
                tri.e2_flags = {e2[0], e2[1], e2[2], uint_bits_to_float(pack_material(response))};
                tri.normal_albedo = {n[0], n[1], n[2], uint_bits_to_float(albedo)};
                out.push_back(tri);
            }
        }
    };

    for (const scene::Layer layer :
         {scene::Layer::Terrain, scene::Layer::Ties, scene::Layer::Shrubs}) {
        const auto& list = level.instances[static_cast<std::size_t>(layer)];
        for (const auto& inst : list) {
            extract_instance(inst, layer, m_triangles, &m_emitter_triangles);
        }
    }

    // Cache bind-pose moby models for fast per-frame instance updates.
    m_moby_meshes.resize(level.models.size());
    for (std::size_t m = 0; m < level.models.size(); ++m) {
        scene::Instance identity_inst{};
        identity_inst.model = static_cast<std::uint32_t>(m);
        identity_inst.matrix = renderer::identity();
        extract_instance(identity_inst, scene::Layer::Mobys, m_moby_meshes[m].local_triangles);
    }

    // Each model's colour, for the light of an additive moby of it (an explosion, a shot).
    m_model_colours.assign(level.models.size(), {1.0f, 1.0f, 1.0f});
    for (std::size_t m = 0; m < level.models.size(); ++m) {
        std::array<double, 3> sum{};
        double weight = 0.0;
        for (const auto& prim : level.models[m].primitives) {
            if (prim.material < 0 || static_cast<std::size_t>(prim.material) >= level.materials.size()) {
                continue;
            }
            const auto& mat = level.materials[static_cast<std::size_t>(prim.material)];
            const std::size_t at = static_cast<std::size_t>(prim.first_index);
            if (prim.index_count == 0 || at >= level.indices.size()
                || level.indices[at] >= level.vertices.size()) {
                continue;
            }
            const std::uint32_t albedo =
                sample_material_albedo(level, mat, level.vertices[level.indices[at]].colour);
            for (int k = 0; k < 3; ++k) {
                sum[k] += float((albedo >> (8u * k)) & 0xFFu) / 255.0f * prim.index_count;
            }
            weight += prim.index_count;
        }
        if (weight > 0.0) {
            for (int k = 0; k < 3; ++k) {
                m_model_colours[m][k] = static_cast<float>(sum[k] / weight);
            }
        }
    }

    // The skinned models (characters), in bind pose with their joints, for posing each frame.
    m_skinned_meshes.assign(level.models.size(), {});
    std::size_t skinned_models = 0;
    for (std::size_t m = 0; m < level.models.size(); ++m) {
        const auto& model = level.models[m];
        auto& mesh = m_skinned_meshes[m];
        std::unordered_map<std::uint32_t, std::uint32_t> local;  // level vertex -> mesh vertex
        bool skinned = false;
        for (const auto& prim : model.primitives) {
            for (std::uint32_t i = 0; i < prim.index_count && !skinned; ++i) {
                const std::size_t at = static_cast<std::size_t>(prim.first_index + i);
                if (at < level.indices.size() && level.indices[at] < level.vertices.size()) {
                    const auto& w = level.vertices[level.indices[at]].weights;
                    skinned = w[0] + w[1] + w[2] + w[3] > 0.0f;
                }
            }
        }
        if (!skinned) {
            continue;
        }
        for (const auto& prim : model.primitives) {
            if (prim.material < 0 || static_cast<std::size_t>(prim.material) >= level.materials.size()) {
                continue;
            }
            const auto& mat = level.materials[static_cast<std::size_t>(prim.material)];
            const bool emissive = glows(mat);
            // Blended and alpha-tested parts are left out, as for the level (no texture test).
            if (!emissive && (mat.blend || mat.cutout)) {
                continue;
            }
            for (std::uint32_t i = 0; i + 2 < prim.index_count; i += 3) {
                const std::size_t at = static_cast<std::size_t>(prim.first_index + i);
                if (at + 2 >= level.indices.size()) {
                    break;
                }
                const std::uint32_t idx[3]{level.indices[at], level.indices[at + 1], level.indices[at + 2]};
                if (idx[0] >= level.vertices.size() || idx[1] >= level.vertices.size()
                    || idx[2] >= level.vertices.size()) {
                    continue;
                }
                const auto& v0 = level.vertices[idx[0]];
                const auto& v1 = level.vertices[idx[1]];
                const auto& v2 = level.vertices[idx[2]];
                SkinTriangle tri;
                const bool packet = v0.light_slot == -2 || v1.light_slot == -2 || v2.light_slot == -2;
                const float mask = emissive || packet ? 0.0f : sidecar_glow(m_sidecars, mat, v0, v1, v2);
                if (emissive || packet || mask > 0.0f) {
                    // A glowing part (Clank's antenna bulb, his eyes): seen and reflected as light.
                    const auto glow = emitter_triangle(level, mat, renderer::identity(), v0, v1, v2,
                                                       emissive, mask);
                    if (!glow) {
                        continue;
                    }
                    std::uint32_t bits = 0;
                    std::memcpy(&bits, &glow->normal_albedo[3], sizeof bits);
                    tri.albedo = bits;
                    const auto strength = static_cast<std::uint32_t>(
                        std::clamp(std::lround(glow->e2_flags[3] * 16.0f), 1L, 255L)
                    );
                    tri.material = pack_material({0.5f, 0.0f, 0.0f}) | (strength << 24u);
                } else {
                    tri.albedo = sample_material_albedo(level, mat, v0.colour);
                    std::array<float, 3> response{mat.metal ? 0.12f : 0.55f, mat.metal ? 1.0f : 0.0f, 0.0f};
                    if (!mat.metal && m_sidecars != nullptr) {
                        m_sidecars->response(mat.texture_pack_hash, (v0.uv[0] + v1.uv[0] + v2.uv[0]) / 3.0f,
                                             (v0.uv[1] + v1.uv[1] + v2.uv[1]) / 3.0f, response);
                    }
                    tri.material = pack_material(response);
                }
                for (int k = 0; k < 3; ++k) {
                    const auto [found, added] =
                        local.try_emplace(idx[k], static_cast<std::uint32_t>(mesh.vertices.size()));
                    if (added) {
                        const auto& v = level.vertices[idx[k]];
                        SkinVertex sv;
                        sv.position = {v.position[0], v.position[1], v.position[2]};
                        sv.joints = {v.joints[0], v.joints[1], v.joints[2], v.joints[3]};
                        sv.weights = {v.weights[0], v.weights[1], v.weights[2], v.weights[3]};
                        mesh.vertices.push_back(sv);
                    }
                    tri.vertices[k] = found->second;
                }
                mesh.triangles.push_back(tri);
            }
        }
        skinned_models += mesh.triangles.empty() ? 0 : 1;
    }

    // The glowing parts: a light per static instance, and per model for the mobys.
    m_emissive_static.clear();
    m_emissive_mobys.clear();
    m_moby_emissive.assign(level.models.size(), {});
    for (const scene::Layer layer :
         {scene::Layer::Terrain, scene::Layer::Ties, scene::Layer::Shrubs}) {
        for (const auto& inst : level.instances[static_cast<std::size_t>(layer)]) {
            if (inst.stand_in || inst.lighting_only || inst.model >= level.models.size()) {
                continue;
            }
            GlowSum sum;
            sum_glow(level, level.models[inst.model], inst.matrix, m_sidecars, sum);
            if (sum.area <= 1e-4f) {
                continue;
            }
            std::array<float, 3> centre{}, colour{};
            for (int a = 0; a < 3; ++a) {
                centre[a] = sum.centre[a] / sum.area;
                colour[a] = sum.colour[a] / sum.area;
            }
            m_emissive_static.push_back(glow_light(centre, sum.normal, colour, sum.area));
        }
    }
    // The largest first when a level has more than LocalLights takes beside the others.
    constexpr std::size_t kStaticGlows = 2048;
    if (m_emissive_static.size() > kStaticGlows) {
        std::partial_sort(
            m_emissive_static.begin(), m_emissive_static.begin() + kStaticGlows,
            m_emissive_static.end(),
            [](const LocalLight& a, const LocalLight& b) { return a.range > b.range; }
        );
        m_emissive_static.resize(kStaticGlows);
    }
    for (std::size_t m = 0; m < level.models.size(); ++m) {
        GlowSum sum;
        sum_glow(level, level.models[m], renderer::identity(), m_sidecars, sum);
        if (sum.area <= 1e-4f) {
            continue;
        }
        auto& part = m_moby_emissive[m];
        for (int a = 0; a < 3; ++a) {
            part.centre[a] = sum.centre[a] / sum.area;
            part.normal[a] = sum.normal[a];
            part.colour[a] = sum.colour[a] / sum.area;
        }
        part.area = sum.area;
    }
    ++m_emissive_revision;
    const auto glowing_models = std::count_if(
        m_moby_emissive.begin(), m_moby_emissive.end(), [](const auto& p) { return p.area > 1e-4f; }
    );
    log::info("rtx: {} glowing surfaces become lights; {} models glow when placed as mobys; "
              "{} glowing triangles traced; {} skinned models posed for the trace",
              m_emissive_static.size(), glowing_models, m_emitter_triangles.size(), skinned_models);

    build_bvh();
}

bool RtxPipeline::upload_triangles(std::span<const RtxTriangle> triangles, std::string& error) {
    m_triangles.assign(triangles.begin(), triangles.end());
    clear_dynamic();
    build_tree(m_triangles, m_nodes);
    m_static_dirty = true;
    publish_counts();
    if (m_metal_backend && !m_metal_backend->upload_scene(m_triangles, m_nodes, error)) {
        return false;
    }
    if (m_cuda_backend && !m_cuda_backend->upload_scene(m_triangles, m_nodes, error)) {
        return false;
    }
    return true;
}

void RtxPipeline::clear_dynamic() {
    if (!m_dynamic_triangles.empty() || !m_dynamic_nodes.empty()) {
        m_dynamic_dirty = true;
    }
    m_dynamic_triangles.clear();
    m_dynamic_nodes.clear();
    m_dynamic_key = 0;
}

void RtxPipeline::publish_counts() {
    std::lock_guard lock(g_rtx_mutex);
    g_rtx_status.triangle_count = m_triangles.size() + m_dynamic_triangles.size();
    g_rtx_status.node_count = m_nodes.size() + m_dynamic_nodes.size();
    g_rtx_status.dynamic_instance_count = m_dynamic_instance_count;
}

// The mobys get a tree of their own, beside the level's static one: the static tree is built and
// uploaded once a level, the mobys' (a few thousand triangles at most) only when one has moved.
// Bind poses: animation does not reach the tree yet.
std::span<const LocalLight> RtxPipeline::moving_lights(
    std::span<const renderer::EffectQuad> effects, const renderer::TexturePool* textures
) {
    m_moving_lights.assign(m_emissive_mobys.begin(), m_emissive_mobys.end());
    // Added quads (GS ALPHA (Cs - 0) x As + Cd: B 2, D 1) gathered in cells of 3 units: each cell's
    // particles become one light at their weighted centre, their colours (0x80 = 1) times their
    // alpha summed (a burst of many is brighter, its peak at most 4), reaching further the larger they are
    // (3 to 8). The colour is the quads' vertex colour times their texture's hue.
    struct Cell {
        std::array<float, 3> centre{}, colour{};
        float weight = 0, size = 0;
    };
    std::unordered_map<std::uint64_t, Cell> cells;
    for (const auto& q : effects) {
        const std::uint64_t alpha = q.alpha;
        if (((alpha >> 2) & 3u) != 2u || ((alpha >> 6) & 3u) != 1u) {
            continue;
        }
        std::array<float, 3> centre{}, colour{};
        float a = 0.0f;
        for (int c = 0; c < 4; ++c) {
            for (int k = 0; k < 3; ++k) {
                centre[k] += q.corner[c][k] * 0.25f;
                colour[k] += float((q.rgba[c] >> (8u * k)) & 0xFFu) / 128.0f * 0.25f;
            }
            a += float((q.rgba[c] >> 24u) & 0xFFu) / 128.0f * 0.25f;
        }
        const float dx = q.corner[3][0] - q.corner[0][0], dy = q.corner[3][1] - q.corner[0][1],
                    dz = q.corner[3][2] - q.corner[0][2];
        const float size = std::sqrt(dx * dx + dy * dy + dz * dz);
        // The texture's hue, as the effects drew it (a frame ago): GS MODULATE takes vertex colour
        // times texel, and an explosion's orange is in its texture, its vertices near white.
        if (textures != nullptr) {
            if (const auto* hue = textures->effect_hue(
                    renderer::TexturePool::effect_key(q.tex0, q.tbp, q.cbp))) {
                for (int k = 0; k < 3; ++k) {
                    colour[k] *= (*hue)[k];
                }
            }
        }
        const float weight = (colour[0] * 0.2126f + colour[1] * 0.7152f + colour[2] * 0.0722f) * a;
        if (!(weight > 0.01f) || !std::isfinite(size)) {
            continue;
        }
        std::uint64_t key = 0;
        for (int k = 0; k < 3; ++k) {
            key = key * 1000003u
                  + static_cast<std::uint64_t>(static_cast<std::int64_t>(std::floor(centre[k] / 3.0f)) + (1 << 20));
        }
        Cell& cell = cells[key];
        for (int k = 0; k < 3; ++k) {
            cell.centre[k] += centre[k] * weight;
            cell.colour[k] += colour[k] * a;
        }
        cell.weight += weight;
        cell.size = std::max(cell.size, size);
    }
    // Bursts only: a cell lit 45 frames in a row fades out over the next 30 (a lamp's corona or a
    // light shaft is added light that lasts; as a light it brightened the whole level).
    std::unordered_map<std::uint64_t, int> ages;
    std::vector<LocalLight> lights;
    for (const auto& [key, cell] : cells) {
        const auto old = m_effect_ages.find(key);
        const int age = old != m_effect_ages.end() ? old->second + 1 : 0;
        ages[key] = age;
        const float fade = age < 45 ? 1.0f : std::max(0.0f, 1.0f - float(age - 45) / 30.0f);
        if (fade <= 0.0f) {
            continue;
        }
        LocalLight light;
        // At most 4, scaled as a whole: clamping each channel turned a burst's orange (red over
        // the cap, green reaching it) into yellowish white.
        const float peak = std::max({cell.colour[0], cell.colour[1], cell.colour[2]});
        const float scale = peak > 4.0f ? 4.0f / peak : 1.0f;
        for (int k = 0; k < 3; ++k) {
            light.position[k] = cell.centre[k] / cell.weight;
            light.colour[k] = cell.colour[k] * scale * fade;
        }
        light.range = std::clamp(3.0f + cell.size * 2.0f, 3.0f, 8.0f);
        lights.push_back(light);
    }
    // The brightest 8: an explosion's 30 cells of reach 12 filled the screen's light tiles and
    // cost ~9 ms at its worst frames.
    const auto brightness = [](const LocalLight& l) { return l.colour[0] + l.colour[1] + l.colour[2]; };
    if (lights.size() > 8) {
        std::partial_sort(lights.begin(), lights.begin() + 8, lights.end(),
                          [&](const auto& x, const auto& y) { return brightness(x) > brightness(y); });
        lights.resize(8);
    }
    m_effect_ages = std::move(ages);
    stats::gauge("rtx effect quad lights", static_cast<double>(lights.size()));
    m_moving_lights.insert(m_moving_lights.end(), lights.begin(), lights.end());
    return m_moving_lights;
}

void RtxPipeline::prepare_instances(
    scene::Layer layer, std::span<const scene::Instance> instances,
    std::span<const renderer::Mat4> palette
) {
    if (layer != scene::Layer::Mobys) {
        return;
    }
    // A skinned instance's vertex in world space: its joints' matrices weighted (mesh.vert), then
    // the instance's matrix. Joints out of the palette leave the bind pose.
    const auto skinned_mesh = [&](const scene::Instance& inst) -> const SkinnedMesh* {
        if (inst.stand_in || inst.lighting_only || inst.palette < 0
            || inst.model >= m_skinned_meshes.size() || m_skinned_meshes[inst.model].triangles.empty()) {
            return nullptr;
        }
        return &m_skinned_meshes[inst.model];
    };
    const auto pose = [&](const scene::Instance& inst, const SkinVertex& v) {
        float p[3]{v.position[0], v.position[1], v.position[2]};
        const float total = v.weights[0] + v.weights[1] + v.weights[2] + v.weights[3];
        if (total > 0.0f) {
            float out[3]{0.0f, 0.0f, 0.0f};
            float used = 0.0f;
            for (int k = 0; k < 4; ++k) {
                const std::size_t joint = static_cast<std::size_t>(inst.palette) + v.joints[k];
                if (v.weights[k] <= 0.0f || joint >= palette.size()) {
                    continue;
                }
                const auto j = transform_point(palette[joint], v.position.data());
                for (int a = 0; a < 3; ++a) {
                    out[a] += v.weights[k] * j[a];
                }
                used += v.weights[k];
            }
            if (used > 0.0f) {
                for (int a = 0; a < 3; ++a) {
                    p[a] = out[a] / used;
                }
            }
        }
        return transform_point(inst.matrix, p);
    };
    // Which characters are traced: the hero and what he carries first, then the others nearest him
    // within 30 units, at most 12000 triangles. A level holds a hundred skinned models; posing and
    // rebuilding them all each frame cost more than the frame had.
    std::vector<const scene::Instance*> characters;
    {
        std::array<float, 3> hero{};
        bool hero_known = false;
        for (const auto& inst : instances) {
            if (inst.hero_shadow_anchor) {
                hero = {inst.matrix[12], inst.matrix[13], inst.matrix[14]};
                hero_known = true;
                break;
            }
        }
        std::vector<std::pair<float, const scene::Instance*>> ranked;
        for (const auto& inst : instances) {
            if (skinned_mesh(inst) == nullptr) {
                continue;
            }
            const float dx = inst.matrix[12] - hero[0], dy = inst.matrix[13] - hero[1],
                        dz = inst.matrix[14] - hero[2];
            const float d2 = dx * dx + dy * dy + dz * dz;
            const bool carried = inst.hero_shadow_caster || inst.hero_shadow_anchor;
            if (!carried && (!hero_known || d2 > 30.0f * 30.0f)) {
                continue;
            }
            ranked.emplace_back(carried ? -1.0f : d2, &inst);
        }
        std::sort(ranked.begin(), ranked.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        std::size_t budget = 12000;
        for (const auto& [d2, inst] : ranked) {
            const std::size_t count = m_skinned_meshes[inst->model].triangles.size();
            if (count > budget) {
                continue;
            }
            budget -= count;
            characters.push_back(inst);
        }
    }
    m_dynamic_instance_count = instances.size();
    // The glowing mobys' lights follow them. Only rigid ones (lamps, beacons, machines): a
    // character's glow (Clank's eyes, an antenna, an enemy's visor) is a detail, not a lamp, and
    // made the hero light up everything around him. At most 256.
    m_emissive_mobys.clear();
    for (const auto& inst : instances) {
        if (inst.stand_in || inst.lighting_only || inst.palette >= 0 || inst.hero_shadow_caster
            || inst.hero_shadow_anchor || inst.model >= m_moby_emissive.size()) {
            continue;
        }
        const auto& part = m_moby_emissive[inst.model];
        if (part.area <= 1e-4f || m_emissive_mobys.size() >= 256) {
            continue;
        }
        const auto centre = transform_point(inst.matrix, part.centre.data());
        const auto& m = inst.matrix;
        const std::array<float, 3> normal{
            m[0] * part.normal[0] + m[4] * part.normal[1] + m[8] * part.normal[2],
            m[1] * part.normal[0] + m[5] * part.normal[1] + m[9] * part.normal[2],
            m[2] * part.normal[0] + m[6] * part.normal[1] + m[10] * part.normal[2]
        };
        m_emissive_mobys.push_back(glow_light(centre, normal, part.colour, part.area));
    }
    // A character's glowing parts (Clank's antenna bulb and eyes): small lights that follow the
    // pose, one per hue (the red bulb and the green eyes stay apart instead of mixing to a
    // yellowish white), short reach, so that they colour what is close.
    for (const scene::Instance* character : characters) {
        const auto& inst = *character;
        const SkinnedMesh* mesh = skinned_mesh(inst);
        if (mesh == nullptr) {
            continue;
        }
        std::array<GlowSum, 3> sums{};  // by the glow's strongest channel
        for (const auto& tri : mesh->triangles) {
            if ((tri.material >> 24u) == 0u) {
                continue;
            }
            const auto p0 = pose(inst, mesh->vertices[tri.vertices[0]]);
            const auto p1 = pose(inst, mesh->vertices[tri.vertices[1]]);
            const auto p2 = pose(inst, mesh->vertices[tri.vertices[2]]);
            const std::array<float, 3> e1{p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]};
            const std::array<float, 3> e2{p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]};
            const std::array<float, 3> n{e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2],
                                         e1[0] * e2[1] - e1[1] * e2[0]};
            const float area = 0.5f * std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
            if (area < 1e-7f) {
                continue;
            }
            const float strength = float(tri.material >> 24u) / 16.0f;
            std::array<float, 3> rgb{};
            for (int a = 0; a < 3; ++a) {
                rgb[a] = float((tri.albedo >> (8u * a)) & 0xFFu) / 255.0f;
            }
            const auto hue = static_cast<std::size_t>(
                std::max_element(rgb.begin(), rgb.end()) - rgb.begin()
            );
            auto& sum = sums[hue];
            for (int a = 0; a < 3; ++a) {
                sum.centre[a] += (p0[a] + p1[a] + p2[a]) / 3.0f * area;
                sum.normal[a] += n[a] * 0.5f;
                sum.colour[a] += rgb[a] * strength * area;
            }
            sum.area += area;
        }
        for (const auto& sum : sums) {
            if (sum.area <= 1e-6f || m_emissive_mobys.size() >= 256) {
                continue;
            }
            std::array<float, 3> centre{}, colour{};
            for (int a = 0; a < 3; ++a) {
                centre[a] = sum.centre[a] / sum.area;
                colour[a] = sum.colour[a] / sum.area * 1.2f;
            }
            LocalLight light = glow_light(centre, sum.normal, colour, sum.area);
            light.range = 3.0f;
            m_emissive_mobys.push_back(light);
        }
    }
    // Additive mobys (MobyProc's added blend: explosions' flashes, fireballs, shots) light their
    // surroundings while they show: their model's colour times their tint, as bright as the tint's
    // alpha, reaching four times their bound (4 to 14). At most 64; Ratchet's bombs and the
    // blaster's bolts are such mobys.
    std::size_t effect_lights = 0;
    for (const auto& inst : instances) {
        if (inst.blend != 2 || inst.stand_in || inst.lighting_only || inst.moby_light[3] > 1.5f
            || inst.tint[3] <= 0.02f || inst.model >= m_model_colours.size() || effect_lights >= 64
            || m_emissive_mobys.size() >= 512) {
            continue;
        }
        const float radius = inst.world_sphere[3];
        LocalLight light;
        for (int a = 0; a < 3; ++a) {
            light.position[a] = radius > 0.0f ? inst.world_sphere[a] : inst.matrix[12 + a];
            light.colour[a] = m_model_colours[inst.model][a] * inst.tint[a] * std::min(inst.tint[3], 2.0f) * 3.0f;
        }
        light.range = std::clamp(radius * 4.0f, 4.0f, 14.0f);
        m_emissive_mobys.push_back(light);
        ++effect_lights;
    }
    stats::gauge("rtx effect lights", static_cast<double>(effect_lights));
    stats::gauge("rtx glowing mobys", static_cast<double>(m_emissive_mobys.size()));
    if (m_moby_meshes.empty()) {
        publish_counts();
        return;
    }

    // What the tree depends on: each instance's model and matrix. Unchanged, nothing is rebuilt.
    std::uint64_t key = 1469598103934665603ull;
    const auto mix = [&key](const void* data, std::size_t size) {
        const auto* bytes = static_cast<const unsigned char*>(data);
        for (std::size_t i = 0; i < size; ++i) {
            key = (key ^ bytes[i]) * 1099511628211ull;
        }
    };
    // Skinned mobys (Ratchet, enemies: a joint palette) are left out: the tree holds bind poses,
    // and a T-posed body shadowed the animated one and cast a second, wrong shadow beside the
    // hero shadow map's. Rigid mobys (crates, props, machines) stay.
    const auto traced = [&](const scene::Instance& inst) {
        return !inst.stand_in && !inst.lighting_only && inst.palette < 0
               && inst.model < m_moby_meshes.size();
    };
    const bool any_skinned = !characters.empty();
    for (const scene::Instance* character : characters) {
        mix(&character->model, sizeof character->model);
        mix(&character->palette, sizeof character->palette);
        mix(character->matrix.data(), sizeof(float) * 16);
    }
    for (const auto& inst : instances) {
        if (!traced(inst)) {
            continue;
        }
        mix(&inst.model, sizeof inst.model);
        mix(inst.matrix.data(), sizeof(float) * 16);
    }
    if (any_skinned && !palette.empty()) {
        mix(palette.data(), palette.size_bytes());
    }
    if (key == m_dynamic_key && (m_dynamic_key != 0 || m_dynamic_triangles.empty())) {
        publish_counts();
        return;
    }
    m_dynamic_key = key;

    // Rigid mobys and the posed characters (at most 12000 of them, chosen above).
    constexpr std::size_t kMaxDynamicMobyTriangles = 20480;
    std::vector<RtxTriangle> dyn_tris;
    dyn_tris.reserve(std::min<std::size_t>(kMaxDynamicMobyTriangles, instances.size() * 64));
    for (const auto& inst : instances) {
        if (!traced(inst)) {
            continue;
        }
        const auto& slice = m_moby_meshes[inst.model].local_triangles;
        if (slice.empty() || dyn_tris.size() + slice.size() > kMaxDynamicMobyTriangles) {
            continue;
        }
        for (const auto& lt : slice) {
            const std::array<float, 3> lp0{lt.v0_mat[0], lt.v0_mat[1], lt.v0_mat[2]};
            const std::array<float, 3> lp1{
                lp0[0] + lt.e1_layer[0], lp0[1] + lt.e1_layer[1], lp0[2] + lt.e1_layer[2]
            };
            const std::array<float, 3> lp2{
                lp0[0] + lt.e2_flags[0], lp0[1] + lt.e2_flags[1], lp0[2] + lt.e2_flags[2]
            };
            const auto p0 = transform_point(inst.matrix, lp0.data());
            const auto p1 = transform_point(inst.matrix, lp1.data());
            const auto p2 = transform_point(inst.matrix, lp2.data());
            const std::array<float, 3> e1{p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]};
            const std::array<float, 3> e2{p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]};
            std::array<float, 3> n{
                e1[1] * e2[2] - e1[2] * e2[1],
                e1[2] * e2[0] - e1[0] * e2[2],
                e1[0] * e2[1] - e1[1] * e2[0]
            };
            const float len = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
            if (len < 1e-6f) {
                continue;
            }
            n[0] /= len;
            n[1] /= len;
            n[2] /= len;
            RtxTriangle wt = lt;
            wt.v0_mat = {p0[0], p0[1], p0[2], lt.v0_mat[3]};
            wt.e1_layer = {e1[0], e1[1], e1[2], lt.e1_layer[3]};
            wt.e2_flags = {e2[0], e2[1], e2[2], lt.e2_flags[3]};
            wt.normal_albedo = {n[0], n[1], n[2], lt.normal_albedo[3]};
            dyn_tris.push_back(wt);
        }
    }
    // The characters, posed: their glowing parts carry the glow (material's top byte).
    for (const scene::Instance* character : characters) {
        const auto& inst = *character;
        const SkinnedMesh* mesh = skinned_mesh(inst);
        if (mesh == nullptr || dyn_tris.size() + mesh->triangles.size() > kMaxDynamicMobyTriangles) {
            continue;
        }
        std::vector<std::array<float, 3>> posed(mesh->vertices.size());
        for (std::size_t v = 0; v < posed.size(); ++v) {
            posed[v] = pose(inst, mesh->vertices[v]);
        }
        for (const auto& tri : mesh->triangles) {
            const auto& p0 = posed[tri.vertices[0]];
            const auto& p1 = posed[tri.vertices[1]];
            const auto& p2 = posed[tri.vertices[2]];
            const std::array<float, 3> e1{p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]};
            const std::array<float, 3> e2{p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]};
            std::array<float, 3> n{e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2],
                                   e1[0] * e2[1] - e1[1] * e2[0]};
            const float len = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
            if (len < 1e-7f) {
                continue;
            }
            RtxTriangle wt{};
            wt.v0_mat = {p0[0], p0[1], p0[2], 0.0f};
            wt.e1_layer = {e1[0], e1[1], e1[2], int_bits_to_float(static_cast<std::int32_t>(scene::Layer::Mobys))};
            wt.e2_flags = {e2[0], e2[1], e2[2], uint_bits_to_float(tri.material)};
            wt.normal_albedo = {n[0] / len, n[1] / len, n[2] / len, uint_bits_to_float(tri.albedo)};
            dyn_tris.push_back(wt);
        }
    }
    m_dynamic_triangles = std::move(dyn_tris);
    build_tree(m_dynamic_triangles, m_dynamic_nodes);
    m_dynamic_dirty = true;
    publish_counts();
}

void RtxPipeline::build_bvh() {
    clear_dynamic();
    build_tree(m_triangles, m_nodes);
    std::string backend_error;
    if (auto* hw = select_hardware_backend(renderer::RtxBackend::Auto)) {
        if (!hw->upload_scene(m_triangles, m_nodes, backend_error)
            || !hw->upload_emitters(m_emitter_triangles, backend_error)) {
            log::warn("rtx: {}: {}", hw->device_name(), backend_error);
        }
    }
    m_static_dirty = true;
    publish_counts();
}

bool RtxPipeline::trace_rays(
    std::span<const RtxRay> rays,
    std::span<RtxHit> hits,
    renderer::RtxBackend preferred
) {
    if (rays.size() != hits.size()) {
        return false;
    }
    // The hardware backends hold the static tree only: with mobys in the scene the CPU answers.
    if (m_dynamic_triangles.empty()) {
        if (auto* hw = select_hardware_backend(preferred)) {
            std::string error;
            if (hw->trace_rays(rays, hits, error)) {
                return true;
            }
        }
    }
    const auto first_dynamic = static_cast<std::int32_t>(m_triangles.size());
    for (std::size_t i = 0; i < rays.size(); ++i) {
        hits[i] = RtxHit{};
        trace_cpu_bvh(m_nodes, m_triangles, rays[i], hits[i]);
        trace_cpu_bvh(m_dynamic_nodes, m_dynamic_triangles, rays[i], hits[i], first_dynamic);
    }
    return true;
}

namespace {

// Uploads `data` (at least one element: a texel buffer may not be empty) into `buffer` and views
// it as RGBA32F texels in `texture`.
template <typename T>
void upload_texels(unsigned buffer, unsigned texture, std::span<const T> data) {
    const T dummy{};
    glBindBuffer(wgl::GL_TEXTURE_BUFFER, buffer);
    glBufferData(
        wgl::GL_TEXTURE_BUFFER,
        std::max<std::size_t>(1, data.size()) * sizeof(T),
        data.empty() ? &dummy : data.data(),
        GL_STATIC_DRAW
    );
    glBindTexture(wgl::GL_TEXTURE_BUFFER, texture);
    glTexBuffer(wgl::GL_TEXTURE_BUFFER, wgl::GL_RGBA32F, buffer);
}

}  // namespace

void RtxPipeline::ensure_gpu_buffers() {
    if (!m_bvh_buffer) {
        glGenBuffers(1, &m_bvh_buffer);
        glGenTextures(1, &m_bvh_texture);
        glGenBuffers(1, &m_tri_buffer);
        glGenTextures(1, &m_tri_texture);
        glGenBuffers(1, &m_dyn_bvh_buffer);
        glGenTextures(1, &m_dyn_bvh_texture);
        glGenBuffers(1, &m_dyn_tri_buffer);
        glGenTextures(1, &m_dyn_tri_texture);
        glGenBuffers(1, &m_output_buffer);
        glGenTextures(1, &m_output_texture);
        glGenBuffers(1, &m_history_buffer);
        glGenTextures(1, &m_history_texture);
        m_static_dirty = true;
        m_dynamic_dirty = true;
    }
    glActiveTexture(GL_TEXTURE0 + 23);
    if (m_static_dirty) {
        // The level's tree: once a level, megabytes.
        upload_texels<RtxBvhNode>(m_bvh_buffer, m_bvh_texture, m_nodes);
        glActiveTexture(GL_TEXTURE0 + 24);
        upload_texels<RtxTriangle>(m_tri_buffer, m_tri_texture, m_triangles);
        m_static_dirty = false;
    }
    if (m_dynamic_dirty) {
        // The mobys' tree: when one moved, kilobytes.
        glActiveTexture(GL_TEXTURE0 + 26);
        upload_texels<RtxBvhNode>(m_dyn_bvh_buffer, m_dyn_bvh_texture, m_dynamic_nodes);
        glActiveTexture(GL_TEXTURE0 + 27);
        upload_texels<RtxTriangle>(m_dyn_tri_buffer, m_dyn_tri_texture, m_dynamic_triangles);
        m_dynamic_dirty = false;
    }
    glActiveTexture(GL_TEXTURE0);
}

bool RtxPipeline::render(
    unsigned depth,
    const renderer::FrameInput& input,
    const renderer::RenderTarget& target,
    const renderer::Vec3& sun_direction,
    const renderer::Vec3& sun_colour,
    const renderer::Vec3& sky_colour,
    std::string& error
) {
    m_ready = false;
    // What runs is the GPU compute pass over the two trees (the native backends trace only
    // tests' and bakes' batches): the status says that, not the backend chosen in the menu.
    const char* runs = "GPU compute BVH";
    {
        std::lock_guard lock(g_rtx_mutex);
        g_rtx_status.active = input.rtx;
        g_rtx_status.active_backend = renderer::RtxBackend::VulkanRayQuery;
        g_rtx_status.device_name = runs;
    }
    // Once a session, why tracing does not run, or that it does and how.
    const auto why_not = [&](const char* reason) {
        if (input.rtx && m_logged != reason) {
            log::info("rtx: not tracing: {}", reason);
            m_logged = reason;
        }
        return false;
    };
    if (!input.rtx) {
        return false;
    }
    if (m_compute_failed) {
        return why_not("its compute program failed to build");
    }
    if (!enhanced_rendering_available()) {
        return why_not("enhanced rendering is off or this GL has none");
    }
    if (!glDispatchCompute) {
        return why_not("this GL has no compute");
    }
    if (!depth || target.width < 2 || target.height < 2) {
        return false;
    }
    m_reproject = false;
    if (metal_frames(input)) {
        return render_metal(depth, input, target, sun_direction, sun_colour, sky_colour, error);
    }

    const int width = target.width;
    const int height = target.height;
    const std::size_t cells =
        static_cast<std::size_t>((width + 1) / 2) * static_cast<std::size_t>((height + 1) / 2);
    GLint limit = 0;
    glGetIntegerv(GL_MAX_TEXTURE_BUFFER_SIZE, &limit);
    if (cells * 3 > static_cast<std::size_t>(limit)) {
        return false;
    }

    if (!m_compute.id()) {
        std::string source = lighting::shaders::rtx_trace_comp;
        source.insert(source.find('\n') + 1, renderer::shaders::surface_normal_glsl);
        if (!m_compute.build_compute("rtx ray tracing", source, error)) {
            m_compute_failed = true;
            return false;
        }
    }

    ensure_gpu_buffers();
    // Last frame's result becomes the history this frame reads; this frame writes the other.
    std::swap(m_output_buffer, m_history_buffer);
    std::swap(m_output_texture, m_history_texture);
    if (width != m_width || height != m_height) {
        for (unsigned buffer : {m_output_buffer, m_history_buffer}) {
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
            glBufferData(
                GL_SHADER_STORAGE_BUFFER, cells * 3 * sizeof(std::uint32_t), nullptr, GL_DYNAMIC_DRAW
            );
        }
        m_history_valid = false;
    }
    // History holds only across consecutive frames of one camera: a cut, a teleport or a frame
    // skipped starts over.
    const auto& cam = input.camera.position;
    const float dx = cam[0] - m_prev_camera[0], dy = cam[1] - m_prev_camera[1],
                dz = cam[2] - m_prev_camera[2];
    const bool history =
        m_history_valid && input.frame >= m_frame && input.frame - m_frame <= 1
        && dx * dx + dy * dy + dz * dz < 4.0f;
    glActiveTexture(GL_TEXTURE0 + 28);
    glBindTexture(wgl::GL_TEXTURE_BUFFER, m_history_texture);
    glTexBuffer(wgl::GL_TEXTURE_BUFFER, GL_RGBA8, m_history_buffer);
    glActiveTexture(GL_TEXTURE0);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_output_buffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 14, m_output_buffer);

    glActiveTexture(GL_TEXTURE0 + 23);
    glBindTexture(wgl::GL_TEXTURE_BUFFER, m_bvh_texture);
    glActiveTexture(GL_TEXTURE0 + 24);
    glBindTexture(wgl::GL_TEXTURE_BUFFER, m_tri_texture);
    glActiveTexture(GL_TEXTURE0 + 25);
    glBindTexture(wgl::GL_TEXTURE_BUFFER, m_output_texture);
    glTexBuffer(wgl::GL_TEXTURE_BUFFER, GL_RGBA8, m_output_buffer);
    glActiveTexture(GL_TEXTURE0 + 26);
    glBindTexture(wgl::GL_TEXTURE_BUFFER, m_dyn_bvh_texture);
    glActiveTexture(GL_TEXTURE0 + 27);
    glBindTexture(wgl::GL_TEXTURE_BUFFER, m_dyn_tri_texture);

    const auto& s = input.rtx_settings;
    const int flags = effect_flags(s);
    const int quality_spp = s.quality == renderer::RtxQuality::Performance ? 1
                            : s.quality == renderer::RtxQuality::Balanced  ? std::min(2, s.samples_per_pixel)
                                                                           : s.samples_per_pixel;

    m_compute.use();
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, depth);
    glUniform1i(m_compute.uniform("rtx_depth"), 0);
    glUniform1i(m_compute.uniform("rtx_bvh_nodes"), 23);
    glUniform1i(m_compute.uniform("rtx_triangles"), 24);
    glUniform1i(m_compute.uniform("surface_normals"), 19);
    glUniform1i(m_compute.uniform("surface_normals_enabled"), target.normal_texture != 0 ? 1 : 0);
    glActiveTexture(GL_TEXTURE0 + 19);
    glBindTexture(GL_TEXTURE_2D, target.normal_texture);
    glActiveTexture(GL_TEXTURE0);

    glUniformMatrix4fv(
        m_compute.uniform("rtx_projection"), 1, GL_FALSE, input.camera.projection.data()
    );
    glUniformMatrix4fv(m_compute.uniform("rtx_view"), 1, GL_FALSE, input.camera.view.data());
    glUniform3f(
        m_compute.uniform("rtx_camera_pos"),
        input.camera.position[0],
        input.camera.position[1],
        input.camera.position[2]
    );
    glUniform3f(
        m_compute.uniform("rtx_sun_dir"), sun_direction[0], sun_direction[1], sun_direction[2]
    );
    glUniform3f(
        m_compute.uniform("rtx_sun_colour"), sun_colour[0], sun_colour[1], sun_colour[2]
    );
    glUniform3f(
        m_compute.uniform("rtx_sky_colour"), sky_colour[0], sky_colour[1], sky_colour[2]
    );
    glUniform1i(m_compute.uniform("rtx_node_count"), static_cast<int>(m_nodes.size()));
    glUniform1i(m_compute.uniform("rtx_dynamic_nodes"), 26);
    glUniform1i(m_compute.uniform("rtx_dynamic_triangles"), 27);
    glUniform1i(
        m_compute.uniform("rtx_dynamic_node_count"), static_cast<int>(m_dynamic_nodes.size())
    );
    glUniform1i(m_compute.uniform("rtx_step_budget"), step_budget(s.quality));
    glUniform1i(m_compute.uniform("rtx_history"), 28);
    glUniform1i(m_compute.uniform("rtx_history_valid"), history ? 1 : 0);
    glUniformMatrix4fv(m_compute.uniform("rtx_prev_view"), 1, GL_FALSE, m_prev_view.data());
    glUniformMatrix4fv(
        m_compute.uniform("rtx_prev_projection"), 1, GL_FALSE, m_prev_projection.data()
    );
    // A quarter of each frame's own sample: about the last eight frames' rays together.
    glUniform1f(m_compute.uniform("rtx_history_blend"), history ? 0.25f : 1.0f);
    glUniform1i(m_compute.uniform("rtx_samples_per_pixel"), std::clamp(quality_spp, 1, 8));
    glUniform1i(m_compute.uniform("rtx_flags"), flags);
    glUniform1i(m_compute.uniform("rtx_frame_index"), static_cast<int>(input.frame));
    glUniform1i(m_compute.uniform("rtx_debug_view"), static_cast<int>(rtx_debug_view()));
    glUniform1f(m_compute.uniform("rtx_ray_distance"), s.ray_distance);
    glUniform1f(m_compute.uniform("rtx_ao_radius"), s.ao_radius);
    glUniform1f(m_compute.uniform("rtx_shadow_softness"), s.shadow_softness);
    glUniform1f(m_compute.uniform("rtx_gi_strength"), s.gi_strength);
    glUniform1f(m_compute.uniform("rtx_roughness_cutoff"), s.reflection_roughness_cutoff);

    const std::array<ComputeOutput, 1> outputs{
        {{m_output_buffer, cells * 3 * sizeof(std::uint32_t)}}
    };
    const int cells_w = (width + 1) / 2;
    const int cells_h = (height + 1) / 2;
    if (!run_compute_graph(
            "rtx ray tracing",
            outputs,
            [&] {
                const stats::Scope pass("rtx ray tracing", stats::kGpu);
                glDispatchCompute(
                    static_cast<unsigned>((cells_w + 7) / 8),
                    static_cast<unsigned>((cells_h + 7) / 8),
                    1
                );
            },
            error
        )) {
        return false;
    }

    m_width = width;
    m_height = height;
    m_target = target.framebuffer;
    m_frame = input.frame;
    m_history_valid = true;
    m_prev_view = input.camera.view;
    m_prev_projection = input.camera.projection;
    m_prev_camera = {cam[0], cam[1], cam[2]};
    m_ready = true;
    glBindTexture(GL_TEXTURE_2D, 0);
    stats::gauge("rtx ray tracing cells", static_cast<double>(cells));
    if (m_logged != runs) {
        log::info("rtx: tracing {}x{} cells on {}, {} static and {} moby triangles",
                  cells_w, cells_h, runs, m_triangles.size(), m_dynamic_triangles.size());
        m_logged = runs;
    }
    return true;
}

bool RtxPipeline::metal_frames(const renderer::FrameInput& input) {
#if defined(OPENRAC_VULKAN) && defined(__APPLE__)
    const auto wanted = input.rtx_settings.backend;
    if (!native::available()
        || (wanted != renderer::RtxBackend::Auto && wanted != renderer::RtxBackend::Metal)) {
        return false;
    }
    // The backend must run on the renderer's own MTLDevice, whose buffers it reads.
    void* device = native::device();
    if (device == nullptr) {
        return false;
    }
    if (!m_metal_backend || m_metal_backend->native_device() != device) {
        m_metal_backend = create_metal_rtx_backend_on(device);
        if (!m_metal_backend) {
            return false;
        }
        std::string error;
        if (!m_metal_backend->upload_scene(m_triangles, m_nodes, error)
            || !m_metal_backend->upload_emitters(m_emitter_triangles, error)) {
            log::warn("rtx: Metal: {}", error);
            m_metal_backend.reset();
            return false;
        }
        release_metal_slots();
    }
    return true;
#else
    (void)input;
    return false;
#endif
}

void RtxPipeline::release_metal_slots() {
    for (auto& slot : m_slots) {
        if (slot.cells) {
            glDeleteBuffers(1, &slot.cells);
        }
        if (slot.result) {
            glDeleteBuffers(1, &slot.result);
        }
        if (slot.result_texture) {
            glDeleteTextures(1, &slot.result_texture);
        }
        if (slot.cells_texture) {
            glDeleteTextures(1, &slot.cells_texture);
        }
        slot = MetalSlot{};
    }
    m_next_slot = 0;
    m_latest_slot = -1;
    m_last_submitted = -1;
}

// Metal's intersector on the renderer's device, a frame behind. A Vulkan pass writes the cells'
// positions and normals (rtx_gbuffer.comp) into one of three slots; once the GPU has finished that
// frame, the slot goes to Metal, which traces into the slot's result; early lighting reads the
// newest finished result through the camera it was traced from. The three slots let Vulkan write
// one while Metal traces another and early lighting reads a third, without the CPU waiting.
bool RtxPipeline::render_metal(
    unsigned depth,
    const renderer::FrameInput& input,
    const renderer::RenderTarget& target,
    const renderer::Vec3& sun_direction,
    const renderer::Vec3& sun_colour,
    const renderer::Vec3& sky_colour,
    std::string& error
) {
#if defined(OPENRAC_VULKAN) && defined(__APPLE__)
    using State = MetalSlot::State;
    auto* hw = m_metal_backend.get();
    const int cells_w = (target.width + 1) / 2;
    const int cells_h = (target.height + 1) / 2;
    const std::size_t cells = static_cast<std::size_t>(cells_w) * static_cast<std::size_t>(cells_h);
    GLint limit = 0;
    glGetIntegerv(GL_MAX_TEXTURE_BUFFER_SIZE, &limit);
    if (cells * 3 > static_cast<std::size_t>(limit)) {
        return false;
    }

    // 1. Slots whose Vulkan frame the GPU has finished go to Metal.
    const std::uint64_t done = native::completed_frame();
    for (int i = 0; i < static_cast<int>(m_slots.size()); ++i) {
        MetalSlot& slot = m_slots[static_cast<std::size_t>(i)];
        if (slot.state != State::Written || slot.vk_frame > done) {
            continue;
        }
        vkgl::MetalBuffer c, r, h, hc;
        if (!native::buffer(slot.cells, c) || !native::buffer(slot.result, r)) {
            slot.state = State::Free;
            continue;
        }
        RtxFrameJob job = slot.params;
        job.cells = c.buffer;
        job.cells_offset = c.offset;
        job.result = r.buffer;
        job.result_offset = r.offset;
        job.dynamic_triangles = slot.mobys;
        job.lights = slot.lights;
        // The previous job's result blends in through its camera, when it is of the same grid
        // and its camera did not jump.
        if (m_last_submitted >= 0 && m_last_submitted != i) {
            const MetalSlot& prev = m_slots[static_cast<std::size_t>(m_last_submitted)];
            const auto& a = prev.params.camera_pos;
            const auto& b = job.camera_pos;
            const float dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
            if ((prev.state == State::Submitted || prev.state == State::Done)
                && prev.width == slot.width && prev.height == slot.height
                && dx * dx + dy * dy + dz * dz < 4.0f && native::buffer(prev.result, h)
                && native::buffer(prev.cells, hc)) {
                job.history = h.buffer;
                job.history_offset = h.offset;
                job.history_cells = hc.buffer;
                job.history_cells_offset = hc.offset;
                job.prev_view_projection = prev.view_projection;
            }
        }
        slot.job = hw->submit_frame(job, error);
        if (slot.job == 0) {
            slot.state = State::Free;
            continue;
        }
        slot.state = State::Submitted;
        m_last_submitted = i;
    }

    // 2. Finished traces: the newest becomes what early lighting reads.
    for (int i = 0; i < static_cast<int>(m_slots.size()); ++i) {
        MetalSlot& slot = m_slots[static_cast<std::size_t>(i)];
        if (slot.state == State::Submitted && hw->finished(slot.job)) {
            slot.state = State::Done;
            if (m_latest_slot < 0
                || slot.serial > m_slots[static_cast<std::size_t>(m_latest_slot)].serial) {
                m_latest_slot = i;
            }
        }
    }

    // 3. This frame's cells into a slot nobody reads or traces.
    MetalSlot& next = m_slots[static_cast<std::size_t>(m_next_slot)];
    const bool free = next.state == State::Free
                      || (next.state == State::Done && m_next_slot != m_latest_slot
                          && m_next_slot != m_last_submitted);
    if (free && !m_gbuffer_failed) {
        if (!m_gbuffer_compute.id()) {
            std::string source = lighting::shaders::rtx_gbuffer_comp;
            source.insert(source.find('\n') + 1, renderer::shaders::surface_normal_glsl);
            if (!m_gbuffer_compute.build_compute("rtx cells", source, error)) {
                m_gbuffer_failed = true;
                return false;
            }
        }
        if (!next.cells) {
            glGenBuffers(1, &next.cells);
            glGenBuffers(1, &next.result);
            glGenTextures(1, &next.result_texture);
            glGenTextures(1, &next.cells_texture);
        }
        if (next.width != cells_w || next.height != cells_h) {
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, next.cells);
            glBufferData(GL_SHADER_STORAGE_BUFFER, cells * 2 * 16, nullptr, GL_DYNAMIC_DRAW);
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, next.result);
            glBufferData(GL_SHADER_STORAGE_BUFFER, cells * 3 * 4, nullptr, GL_DYNAMIC_DRAW);
            glActiveTexture(GL_TEXTURE0 + 25);
            glBindTexture(wgl::GL_TEXTURE_BUFFER, next.result_texture);
            glTexBuffer(wgl::GL_TEXTURE_BUFFER, GL_RGBA8, next.result);
            glBindTexture(wgl::GL_TEXTURE_BUFFER, next.cells_texture);
            glTexBuffer(wgl::GL_TEXTURE_BUFFER, wgl::GL_RGBA32F, next.cells);
            glActiveTexture(GL_TEXTURE0);
            next.width = cells_w;
            next.height = cells_h;
        }
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 14, next.cells);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 15, next.result);
        m_gbuffer_compute.use();
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, depth);
        glUniform1i(m_gbuffer_compute.uniform("rtx_depth"), 0);
        glUniform1i(m_gbuffer_compute.uniform("surface_normals"), 19);
        glUniform1i(m_gbuffer_compute.uniform("surface_normals_enabled"), target.normal_texture != 0 ? 1 : 0);
        glActiveTexture(GL_TEXTURE0 + 19);
        glBindTexture(GL_TEXTURE_2D, target.normal_texture);
        glActiveTexture(GL_TEXTURE0);
        glUniformMatrix4fv(m_gbuffer_compute.uniform("rtx_projection"), 1, GL_FALSE, input.camera.projection.data());
        glUniformMatrix4fv(m_gbuffer_compute.uniform("rtx_view"), 1, GL_FALSE, input.camera.view.data());
        glUniform1f(
            m_gbuffer_compute.uniform("rtx_roughness"),
            std::clamp(input.rtx_settings.reflection_roughness_cutoff * 0.5f, 0.04f, 1.0f)
        );
        const std::array<ComputeOutput, 2> outputs{
            {{next.cells, cells * 2 * 16}, {next.result, cells * 3 * 4}}
        };
        if (!run_compute_graph(
                "rtx cells",
                outputs,
                [&] {
                    const stats::Scope pass("rtx cells", stats::kGpu);
                    glDispatchCompute(
                        static_cast<unsigned>((cells_w + 7) / 8),
                        static_cast<unsigned>((cells_h + 7) / 8),
                        1
                    );
                },
                error
            )) {
            return false;
        }
        glBindTexture(GL_TEXTURE_2D, 0);
        next.vk_frame = native::recording_frame();
        next.serial = ++m_slot_serial;
        next.view_projection = renderer::multiply(input.camera.projection, input.camera.view);
        RtxFrameJob& p = next.params;
        p = RtxFrameJob{};
        p.width = cells_w;
        p.height = cells_h;
        p.camera_pos = {input.camera.position[0], input.camera.position[1], input.camera.position[2]};
        p.sun_dir = {sun_direction[0], sun_direction[1], sun_direction[2]};
        p.sun_colour = {sun_colour[0], sun_colour[1], sun_colour[2]};
        p.sky_colour = {sky_colour[0], sky_colour[1], sky_colour[2]};
        p.settings = input.rtx_settings;
        p.flags = effect_flags(input.rtx_settings);
        p.frame_index = static_cast<std::uint32_t>(input.frame);
        p.dynamic_key = m_dynamic_key;
        next.mobys = m_dynamic_triangles;
        // The local lights with a radius, nearest the camera first: the paths sample up to 512
        // of them (next-event estimation), the 64 nearest also have their shadows traced.
        next.lights.clear();
        const auto& eye = input.camera.position;
        for (const auto& light : m_local_lights) {
            if (light[0][3] > 0.0f) {
                next.lights.push_back(light);
            }
        }
        const auto distance = [&](const RtxLightRecord& l) {
            const float dx = l[0][0] - eye[0], dy = l[0][1] - eye[1], dz = l[0][2] - eye[2];
            return dx * dx + dy * dy + dz * dz;
        };
        constexpr std::size_t kTracedLights = 512;
        const std::size_t kept = std::min(next.lights.size(), kTracedLights);
        std::partial_sort(
            next.lights.begin(), next.lights.begin() + static_cast<std::ptrdiff_t>(kept),
            next.lights.end(), [&](const auto& a, const auto& b) { return distance(a) < distance(b); }
        );
        next.lights.resize(kept);
        if (input.rtx_settings.shadows && !next.lights.empty()) {
            p.flags |= 16;
        }
        next.state = State::Written;
        m_next_slot = (m_next_slot + 1) % static_cast<int>(m_slots.size());
    }

    // 4. Early lighting reads the newest finished result.
    if (m_latest_slot < 0) {
        return false;
    }
    const MetalSlot& latest = m_slots[static_cast<std::size_t>(m_latest_slot)];
    if (latest.width != cells_w || latest.height != cells_h) {
        return false;
    }
    m_metal_texture = latest.result_texture;
    m_metal_cells_texture = latest.cells_texture;
    m_reproject = true;
    m_local_traced = (latest.params.flags & 16) != 0;
    m_source_view_projection = latest.view_projection;
    m_source_cells = {latest.width, latest.height};
    m_width = target.width;
    m_height = target.height;
    m_target = target.framebuffer;
    m_frame = input.frame;
    m_ready = true;
    {
        std::lock_guard lock(g_rtx_mutex);
        g_rtx_status.active_backend = renderer::RtxBackend::Metal;
        g_rtx_status.device_name = std::string(hw->device_name());
    }
    stats::gauge("rtx ray tracing cells", static_cast<double>(cells));
    static const char* const kMetal = "Apple Metal RT";
    if (m_logged != kMetal) {
        log::info("rtx: tracing {}x{} cells on Apple Metal RT ({}), {} static and {} moby triangles",
                  cells_w, cells_h, hw->device_name(), m_triangles.size(), m_dynamic_triangles.size());
        m_logged = kMetal;
    }
    return true;
#else
    (void)depth;
    (void)input;
    (void)target;
    (void)sun_direction;
    (void)sun_colour;
    (void)sky_colour;
    (void)error;
    return false;
#endif
}

void RtxPipeline::bind_for_early_lighting(
    const renderer::Shader& shader, const renderer::FrameInput& input
) const {
    const bool enabled = m_ready && input.rtx && m_frame == input.frame && (m_reproject ? m_metal_texture : m_output_texture) != 0;
    glUniform1i(shader.uniform("rtx_enabled"), enabled ? 1 : 0);
    glUniform1i(shader.uniform("rtx_values"), 25);
    glActiveTexture(GL_TEXTURE0 + 25);
    if (!enabled) {
        glBindTexture(wgl::GL_TEXTURE_BUFFER, 0);
        glActiveTexture(GL_TEXTURE0);
        return;
    }
    const auto& s = input.rtx_settings;
    const int flags = effect_flags(s) | (m_reproject && m_local_traced ? 16 : 0);
    glUniform1i(shader.uniform("rtx_flags"), flags);
    glUniform1i(shader.uniform("rtx_reproject"), m_reproject ? 1 : 0);
    glUniformMatrix4fv(
        shader.uniform("rtx_source_view_projection"), 1, GL_FALSE, m_source_view_projection.data()
    );
    glUniform2f(
        shader.uniform("rtx_source_cells"), float(m_source_cells[0]), float(m_source_cells[1])
    );
    glUniform1i(shader.uniform("rtx_denoise"), s.denoise ? 1 : 0);
    glUniform1i(shader.uniform("rtx_debug_view"), static_cast<int>(rtx_debug_view()));
    glBindTexture(wgl::GL_TEXTURE_BUFFER, m_reproject ? m_metal_texture : m_output_texture);
    // Where the result in use was traced from, to keep it off the surfaces it was not.
    glUniform1i(shader.uniform("rtx_source_positions"), 29);
    glActiveTexture(GL_TEXTURE0 + 29);
    glBindTexture(wgl::GL_TEXTURE_BUFFER, m_reproject ? m_metal_cells_texture : 0);
    glActiveTexture(GL_TEXTURE0);
}

// The effects worth their rays this frame. The bounce only fills where no probe bake reaches
// (early_lighting.comp), so with a bake its long rays and their shadow rays would be thrown away;
// without it the AO rays also stop at the AO radius instead of the bounce's reach. Performance
// leaves out reflections, a long ray and a shadow ray a cell for a Fresnel-weighted touch.
int RtxPipeline::effect_flags(const renderer::RtxSettings& s) const {
    const bool reflections = s.reflections && s.quality != renderer::RtxQuality::Performance;
    const bool bounce = s.global_illumination && !m_probes_cover;
    return (s.shadows ? 1 : 0) | (s.ambient_occlusion ? 2 : 0) | (reflections ? 4 : 0)
           | (bounce ? 8 : 0);
}

bool RtxPipeline::active(
    const renderer::FrameInput& input, const renderer::RenderState& state
) const {
    return m_ready && input.rtx && m_frame == input.frame
           && m_target == state.target.framebuffer && m_width == state.target.width
           && m_height == state.target.height;
}

void RtxPipeline::bind_for_mesh(
    const renderer::Shader& shader,
    const renderer::FrameInput& input,
    const renderer::RenderState& state
) const {
    const bool is_active = active(input, state);
    glUniform1i(
        shader.uniform("rtx_reflections_enabled"),
        (is_active && input.rtx_settings.reflections) ? 1 : 0
    );
    glUniform1i(shader.uniform("rtx_debug_view"), is_active ? static_cast<int>(rtx_debug_view()) : 0);
}

void RtxPipeline::release() {
    m_compute.release();
    if (m_bvh_buffer) {
        glDeleteBuffers(1, &m_bvh_buffer);
        m_bvh_buffer = 0;
    }
    if (m_bvh_texture) {
        glDeleteTextures(1, &m_bvh_texture);
        m_bvh_texture = 0;
    }
    if (m_tri_buffer) {
        glDeleteBuffers(1, &m_tri_buffer);
        m_tri_buffer = 0;
    }
    if (m_tri_texture) {
        glDeleteTextures(1, &m_tri_texture);
        m_tri_texture = 0;
    }
    if (m_output_buffer) {
        glDeleteBuffers(1, &m_output_buffer);
        m_output_buffer = 0;
    }
    if (m_output_texture) {
        glDeleteTextures(1, &m_output_texture);
        m_output_texture = 0;
    }
    for (unsigned* buffer : {&m_dyn_bvh_buffer, &m_dyn_tri_buffer, &m_history_buffer}) {
        if (*buffer) {
            glDeleteBuffers(1, buffer);
            *buffer = 0;
        }
    }
    for (unsigned* texture : {&m_dyn_bvh_texture, &m_dyn_tri_texture, &m_history_texture}) {
        if (*texture) {
            glDeleteTextures(1, texture);
            *texture = 0;
        }
    }
    m_gbuffer_compute.release();
    release_metal_slots();
    m_width = m_height = 0;
    m_history_valid = false;
    m_ready = false;
    m_compute_failed = false;
}

}  // namespace openrac::enhanced
