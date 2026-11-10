// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 the OpenRAC contributors
//
// Native hardware ray-tracing pipeline (RTX) for Enhanced Graphics:
// SAH BVH construction, dynamic moby instance updates, Vulkan compute/ray-query dispatch,
// and hardware backend orchestration (Apple Metal RT + NVIDIA CUDA RTX).

#include "enhanced/renderer/rtx.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <numeric>

#include "common/log.h"
#include "common/stats.h"
#include "enhanced/renderer/compute_graph.h"
#include "enhanced_viewer_shaders.h"
#include "renderer/gl.h"
#if defined(OPENRAC_VULKAN)
#include "renderer/vulkan/native_compute.h"
#endif
#include "renderer/world/gl_world.h"
#include "renderer_shaders.h"

namespace openrac::enhanced {
namespace {
using namespace openrac::gl;
namespace wgl = renderer::world::gl;

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

// CPU SAH-BVH ray traversal fallback when neither Metal nor CUDA is requested/present.
void trace_cpu_bvh(
    std::span<const RtxBvhNode> nodes,
    std::span<const RtxTriangle> tris,
    const RtxRay& ray,
    RtxHit& out_hit
) {
    out_hit = RtxHit{};
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
    float closest = ray.t_max;

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
                    out_hit.triangle_index = tri_idx;
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

}  // namespace

#if !defined(__APPLE__)
bool metal_rtx_supported() {
    return false;
}

std::unique_ptr<RtxHardwareBackend> create_metal_rtx_backend() {
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
    m_static_triangles.clear();
    m_triangles.clear();
    m_nodes.clear();
    m_moby_meshes.clear();
    m_dynamic_instance_count = 0;

    auto extract_instance = [&](const scene::Instance& inst,
                                scene::Layer layer,
                                std::vector<RtxTriangle>& out) {
        if (inst.stand_in || inst.lighting_only || inst.model >= level.models.size()) {
            return;
        }
        const auto& model = level.models[inst.model];
        for (const auto& prim : model.primitives) {
            if (prim.material >= 0
                && static_cast<std::size_t>(prim.material) < level.materials.size()) {
                const auto& mat = level.materials[static_cast<std::size_t>(prim.material)];
                if (mat.blend) {
                    continue;
                }
            }
            const scene::Material default_mat{};
            const auto& mat =
                (prim.material >= 0
                 && static_cast<std::size_t>(prim.material) < level.materials.size())
                    ? level.materials[static_cast<std::size_t>(prim.material)]
                    : default_mat;
            const float roughness = mat.metal ? 0.12f : 0.55f;

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
                tri.e2_flags = {e2[0], e2[1], e2[2], roughness};
                tri.normal_albedo = {n[0], n[1], n[2], uint_bits_to_float(albedo)};
                out.push_back(tri);
            }
        }
    };

    for (const scene::Layer layer :
         {scene::Layer::Terrain, scene::Layer::Ties, scene::Layer::Shrubs}) {
        const auto& list = level.instances[static_cast<std::size_t>(layer)];
        for (const auto& inst : list) {
            extract_instance(inst, layer, m_static_triangles);
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

    m_triangles = m_static_triangles;
    build_bvh();
    m_static_triangles = m_triangles;
    m_static_nodes = m_nodes;
}

bool RtxPipeline::upload_triangles(std::span<const RtxTriangle> triangles, std::string& error) {
    m_static_triangles.assign(triangles.begin(), triangles.end());
    m_triangles = m_static_triangles;
    build_bvh();
    m_static_triangles = m_triangles;
    m_static_nodes = m_nodes;
    if (m_metal_backend && !m_metal_backend->upload_scene(m_triangles, m_nodes, error)) {
        return false;
    }
    if (m_cuda_backend && !m_cuda_backend->upload_scene(m_triangles, m_nodes, error)) {
        return false;
    }
    return true;
}

void RtxPipeline::prepare_instances(
    scene::Layer layer, std::span<const scene::Instance> instances
) {
    if (layer != scene::Layer::Mobys) {
        return;
    }
    m_dynamic_instance_count = instances.size();
    {
        std::lock_guard lock(g_rtx_mutex);
        g_rtx_status.dynamic_instance_count = m_dynamic_instance_count;
    }
    if (m_moby_meshes.empty()) {
        return;
    }

    constexpr std::size_t kMaxDynamicMobyTriangles = 2048;
    std::vector<RtxTriangle> dyn_tris;
    dyn_tris.reserve(std::min<std::size_t>(kMaxDynamicMobyTriangles, instances.size() * 64));
    for (const auto& inst : instances) {
        if (inst.stand_in || inst.lighting_only || inst.model >= m_moby_meshes.size()) {
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

    if (dyn_tris.empty()) {
        if (m_triangles.size() != m_static_triangles.size()) {
            m_triangles = m_static_triangles;
            m_nodes = m_static_nodes;
            m_gpu_scene_dirty = true;
            std::lock_guard lock(g_rtx_mutex);
            g_rtx_status.triangle_count = m_triangles.size();
            g_rtx_status.node_count = m_nodes.size();
        }
        return;
    }

    // Build a fast SAH BVH over only the dynamic moby triangles, then merge with m_static_nodes.
    std::vector<BvhBuildEntry> entries(dyn_tris.size());
    std::vector<std::uint32_t> order(dyn_tris.size());
    for (std::size_t i = 0; i < dyn_tris.size(); ++i) {
        order[i] = static_cast<std::uint32_t>(i);
        const auto& t = dyn_tris[i];
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
    std::vector<RtxBvhNode> dyn_nodes;
    dyn_nodes.reserve(dyn_tris.size() * 2);
    dyn_nodes.emplace_back();
    build_sah_bvh_recursive(
        dyn_nodes, order, entries, 0, 0, static_cast<std::uint32_t>(dyn_tris.size())
    );
    std::vector<RtxTriangle> reordered_dyn(dyn_tris.size());
    for (std::size_t i = 0; i < order.size(); ++i) {
        reordered_dyn[i] = dyn_tris[order[i]];
    }

    const std::int32_t static_tri_offset = static_cast<std::int32_t>(m_static_triangles.size());
    m_triangles = m_static_triangles;
    m_triangles.insert(m_triangles.end(), reordered_dyn.begin(), reordered_dyn.end());

    if (m_static_nodes.empty()) {
        m_nodes = std::move(dyn_nodes);
    } else {
        const std::int32_t S = static_cast<std::int32_t>(m_static_nodes.size());
        const std::int32_t D = static_cast<std::int32_t>(dyn_nodes.size());
        m_nodes.resize(static_cast<std::size_t>(1 + S + D));

        // Node 0: combined root whose children are node 1 (static root) and node 2 (dynamic root).
        RtxBvhNode root{};
        for (int a = 0; a < 3; ++a) {
            root.bbox_min[a] = std::min(m_static_nodes[0].bbox_min[a], dyn_nodes[0].bbox_min[a]);
            root.bbox_max[a] = std::max(m_static_nodes[0].bbox_max[a], dyn_nodes[0].bbox_max[a]);
        }
        root.left_or_first = 1;
        root.count = 0;
        m_nodes[0] = root;

        auto adjust_static = [](RtxBvhNode n) {
            if (n.count == 0) {
                n.left_or_first += 2;
            }
            return n;
        };
        auto adjust_dynamic = [S, static_tri_offset](RtxBvhNode n) {
            if (n.count == 0) {
                n.left_or_first += (S + 1);
            } else {
                const std::int32_t first = ~n.left_or_first;
                n.left_or_first = ~(first + static_tri_offset);
            }
            return n;
        };

        m_nodes[1] = adjust_static(m_static_nodes[0]);
        m_nodes[2] = adjust_dynamic(dyn_nodes[0]);
        for (std::int32_t i = 1; i < S; ++i) {
            m_nodes[static_cast<std::size_t>(i + 2)] =
                adjust_static(m_static_nodes[static_cast<std::size_t>(i)]);
        }
        for (std::int32_t j = 1; j < D; ++j) {
            m_nodes[static_cast<std::size_t>(j + S + 1)] =
                adjust_dynamic(dyn_nodes[static_cast<std::size_t>(j)]);
        }
    }

    m_gpu_scene_dirty = true;
    std::lock_guard lock(g_rtx_mutex);
    g_rtx_status.triangle_count = m_triangles.size();
    g_rtx_status.node_count = m_nodes.size();
}

void RtxPipeline::build_bvh() {
    m_nodes.clear();
    if (!m_triangles.empty()) {
        std::vector<BvhBuildEntry> entries(m_triangles.size());
        std::vector<std::uint32_t> order(m_triangles.size());
        for (std::size_t i = 0; i < m_triangles.size(); ++i) {
            order[i] = static_cast<std::uint32_t>(i);
            const auto& t = m_triangles[i];
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
        m_nodes.reserve(m_triangles.size() * 2);
        m_nodes.emplace_back();
        build_sah_bvh_recursive(
            m_nodes, order, entries, 0, 0, static_cast<std::uint32_t>(m_triangles.size())
        );

        std::vector<RtxTriangle> reordered(m_triangles.size());
        for (std::size_t i = 0; i < order.size(); ++i) {
            reordered[i] = m_triangles[order[i]];
        }
        m_triangles = std::move(reordered);
    }

    std::string backend_error;
    if (m_metal_backend) {
        m_metal_backend->upload_scene(m_triangles, m_nodes, backend_error);
    }
    if (m_cuda_backend) {
        m_cuda_backend->upload_scene(m_triangles, m_nodes, backend_error);
    }
    m_gpu_scene_dirty = true;

    std::lock_guard lock(g_rtx_mutex);
    g_rtx_status.triangle_count = m_triangles.size();
    g_rtx_status.node_count = m_nodes.size();
}

bool RtxPipeline::trace_rays(
    std::span<const RtxRay> rays,
    std::span<RtxHit> hits,
    renderer::RtxBackend preferred
) {
    if (rays.size() != hits.size()) {
        return false;
    }
    if (auto* hw = select_hardware_backend(preferred)) {
        std::string error;
        if (hw->trace_rays(rays, hits, error)) {
            return true;
        }
    }
    for (std::size_t i = 0; i < rays.size(); ++i) {
        trace_cpu_bvh(m_nodes, m_triangles, rays[i], hits[i]);
    }
    return true;
}

void RtxPipeline::ensure_gpu_buffers() {
    if (!m_bvh_buffer) {
        glGenBuffers(1, &m_bvh_buffer);
        glGenTextures(1, &m_bvh_texture);
        glGenBuffers(1, &m_tri_buffer);
        glGenTextures(1, &m_tri_texture);
        glGenBuffers(1, &m_output_buffer);
        glGenTextures(1, &m_output_texture);
        m_gpu_scene_dirty = true;
    }
    if (m_gpu_scene_dirty) {
        const RtxBvhNode dummy_node{};
        const RtxTriangle dummy_tri{};
        glBindBuffer(wgl::GL_TEXTURE_BUFFER, m_bvh_buffer);
        glBufferData(
            wgl::GL_TEXTURE_BUFFER,
            std::max<std::size_t>(1, m_nodes.size()) * sizeof(RtxBvhNode),
            m_nodes.empty() ? &dummy_node : m_nodes.data(),
            GL_STATIC_DRAW
        );
        glActiveTexture(GL_TEXTURE0 + 23);
        glBindTexture(wgl::GL_TEXTURE_BUFFER, m_bvh_texture);
        glTexBuffer(wgl::GL_TEXTURE_BUFFER, wgl::GL_RGBA32F, m_bvh_buffer);

        glBindBuffer(wgl::GL_TEXTURE_BUFFER, m_tri_buffer);
        glBufferData(
            wgl::GL_TEXTURE_BUFFER,
            std::max<std::size_t>(1, m_triangles.size()) * sizeof(RtxTriangle),
            m_triangles.empty() ? &dummy_tri : m_triangles.data(),
            GL_STATIC_DRAW
        );
        glActiveTexture(GL_TEXTURE0 + 24);
        glBindTexture(wgl::GL_TEXTURE_BUFFER, m_tri_texture);
        glTexBuffer(wgl::GL_TEXTURE_BUFFER, wgl::GL_RGBA32F, m_tri_buffer);
        glActiveTexture(GL_TEXTURE0);
        m_gpu_scene_dirty = false;
    }
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
    {
        std::lock_guard lock(g_rtx_mutex);
        g_rtx_status.active = input.rtx;
        if (auto* hw = select_hardware_backend(input.rtx_settings.backend)) {
            g_rtx_status.active_backend = hw->kind();
            g_rtx_status.device_name = std::string(hw->device_name());
        } else {
            g_rtx_status.active_backend = renderer::RtxBackend::VulkanRayQuery;
            g_rtx_status.device_name = g_rtx_status.vulkan_ray_query_supported
                                           ? "Vulkan Hardware Ray Query"
                                           : "Vulkan Compute BVH";
        }
    }
    if (!input.rtx || m_compute_failed || !enhanced_rendering_available() || !glDispatchCompute
        || !depth || target.width < 2 || target.height < 2) {
        return false;
    }

    const int width = target.width;
    const int height = target.height;
    const std::size_t cells =
        static_cast<std::size_t>((width + 1) / 2) * static_cast<std::size_t>((height + 1) / 2);
    GLint limit = 0;
    glGetIntegerv(GL_MAX_TEXTURE_BUFFER_SIZE, &limit);
    if (cells * 2 > static_cast<std::size_t>(limit)) {
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
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_output_buffer);
    if (width != m_width || height != m_height) {
        glBufferData(
            GL_SHADER_STORAGE_BUFFER,
            cells * 2 * sizeof(std::uint32_t),
            nullptr,
            GL_DYNAMIC_DRAW
        );
    }
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 14, m_output_buffer);

    glActiveTexture(GL_TEXTURE0 + 23);
    glBindTexture(wgl::GL_TEXTURE_BUFFER, m_bvh_texture);
    glActiveTexture(GL_TEXTURE0 + 24);
    glBindTexture(wgl::GL_TEXTURE_BUFFER, m_tri_texture);
    glActiveTexture(GL_TEXTURE0 + 25);
    glBindTexture(wgl::GL_TEXTURE_BUFFER, m_output_texture);
    glTexBuffer(wgl::GL_TEXTURE_BUFFER, GL_RGBA8, m_output_buffer);

    const auto& s = input.rtx_settings;
    const int flags = (s.shadows ? 1 : 0) | (s.ambient_occlusion ? 2 : 0)
                      | (s.reflections ? 4 : 0) | (s.global_illumination ? 8 : 0);
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
        {{m_output_buffer, cells * 2 * sizeof(std::uint32_t)}}
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
    m_ready = true;
    glBindTexture(GL_TEXTURE_2D, 0);
    stats::gauge("rtx ray tracing cells", static_cast<double>(cells));
    return true;
}

void RtxPipeline::bind_for_early_lighting(
    const renderer::Shader& shader, const renderer::FrameInput& input
) const {
    const bool enabled = m_ready && input.rtx && m_frame == input.frame && m_output_texture != 0;
    glUniform1i(shader.uniform("rtx_enabled"), enabled ? 1 : 0);
    glUniform1i(shader.uniform("rtx_values"), 25);
    glActiveTexture(GL_TEXTURE0 + 25);
    if (!enabled) {
        glBindTexture(wgl::GL_TEXTURE_BUFFER, 0);
        glActiveTexture(GL_TEXTURE0);
        return;
    }
    const auto& s = input.rtx_settings;
    const int flags = (s.shadows ? 1 : 0) | (s.ambient_occlusion ? 2 : 0)
                      | (s.reflections ? 4 : 0) | (s.global_illumination ? 8 : 0);
    glUniform1i(shader.uniform("rtx_flags"), flags);
    glUniform1i(shader.uniform("rtx_denoise"), s.denoise ? 1 : 0);
    glUniform1i(shader.uniform("rtx_debug_view"), static_cast<int>(rtx_debug_view()));
    glBindTexture(wgl::GL_TEXTURE_BUFFER, m_output_texture);
    glActiveTexture(GL_TEXTURE0);
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
    m_width = m_height = 0;
    m_ready = false;
    m_compute_failed = false;
}

}  // namespace openrac::enhanced
