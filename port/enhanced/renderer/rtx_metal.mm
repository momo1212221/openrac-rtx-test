// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 the OpenRAC contributors
//
// Native Apple Metal hardware ray-tracing backend (MTLAccelerationStructure + MSL raytracing).
// Uses Apple Silicon / AMD Metal ray-tracing hardware to build bottom-level acceleration
// structures and dispatch hardware ray queries for shadows, AO, reflections, GI, and probe baking.

#include "enhanced/renderer/rtx.h"

#if defined(__APPLE__)
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace openrac::enhanced {
namespace {

constexpr const char* kMetalRtxShaderSource = R"msl(
#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace metal::raytracing;

struct RtxTriangleGpu {
    float4 v0_mat;
    float4 e1_layer;
    float4 e2_flags;
    float4 normal_albedo;
};

struct RtxRayGpu {
    packed_float3 origin;
    float t_min;
    packed_float3 direction;
    float t_max;
};

struct RtxHitGpu {
    float t;
    int triangle_index;
    float u;
    float v;
    packed_float3 normal;
    uint packed_albedo;
};

struct ScreenParams {
    uint width;
    uint height;
    uint samples_per_pixel;
    uint flags; // bit0: shadows, bit1: ao, bit2: reflections, bit3: gi
    packed_float3 camera_pos;
    float ray_distance;
    packed_float3 sun_dir;
    float ao_radius;
    packed_float3 sun_colour;
    float shadow_softness;
    packed_float3 sky_colour;
    float gi_strength;
    uint frame_index;
    float roughness_cutoff;
    float2 pad;
};

inline uint pcg_hash(uint v) {
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

inline float rand01(thread uint& seed) {
    seed = pcg_hash(seed);
    return float(seed & 0x00FFFFFFu) * (1.0f / 16777216.0f);
}

inline void make_orthonormal_basis(float3 n, thread float3& t, thread float3& b) {
    float s = n.z >= 0.0f ? 1.0f : -1.0f;
    float a = -1.0f / (s + n.z);
    float d = n.x * n.y * a;
    t = float3(1.0f + s * n.x * n.x * a, s * d, -s * n.x);
    b = float3(d, s + n.y * n.y * a, -n.y);
}

inline float3 sample_cosine_hemisphere(float3 n, float u1, float u2) {
    float r = sqrt(u1);
    float phi = 6.28318530718f * u2;
    float x = r * cos(phi);
    float y = r * sin(phi);
    float z = sqrt(max(0.0f, 1.0f - u1));
    float3 t, b;
    make_orthonormal_basis(n, t, b);
    return normalize(t * x + b * y + n * z);
}

inline float3 unpack_rgb8(uint packed) {
    return float3(
        float(packed & 0xFFu),
        float((packed >> 8u) & 0xFFu),
        float((packed >> 16u) & 0xFFu)
    ) * (1.0f / 255.0f);
}

inline uint pack_rgba8(float4 c) {
    uint4 u = uint4(clamp(c, 0.0f, 1.0f) * 255.0f + 0.5f);
    return u.x | (u.y << 8u) | (u.z << 16u) | (u.w << 24u);
}

// Encodes linear GI irradiance using OpenRAC's EarlyLighting log2 encoding.
inline float3 encode_early_gi(float3 gi) {
    constexpr float kEarlyGiLogRange = 12.0f;
    constexpr float kEarlyGiMax = 16.0f;
    if (all(gi <= float3(0.0f))) {
        return float3(0.0f);
    }
    return clamp((log2(max(gi, float3(1e-6f))) + (kEarlyGiLogRange - log2(kEarlyGiMax))) / kEarlyGiLogRange, 0.0f, 1.0f);
}

kernel void rtx_trace_rays_kernel(
    primitive_acceleration_structure accel [[buffer(0)]],
    device const RtxTriangleGpu* triangles [[buffer(1)]],
    device const RtxRayGpu* rays [[buffer(2)]],
    device RtxHitGpu* hits [[buffer(3)]],
    constant uint& ray_count [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= ray_count) {
        return;
    }
    RtxRayGpu r = rays[tid];
    ray q;
    q.origin = float3(r.origin);
    q.direction = float3(r.direction);
    q.min_distance = r.t_min;
    q.max_distance = r.t_max;

    intersector<triangle_data> inter;
    inter.assume_geometry_type(geometry_type::triangle);
    auto result = inter.intersect(q, accel);

    RtxHitGpu out_hit;
    if (result.type == intersection_type::triangle) {
        uint prim = result.primitive_id;
        RtxTriangleGpu tri = triangles[prim];
        out_hit.t = result.distance;
        out_hit.triangle_index = int(prim);
        out_hit.u = result.triangle_barycentric_coord.x;
        out_hit.v = result.triangle_barycentric_coord.y;
        out_hit.normal = tri.normal_albedo.xyz;
        out_hit.packed_albedo = as_type<uint>(tri.normal_albedo.w);
    } else {
        out_hit.t = -1.0f;
        out_hit.triangle_index = -1;
        out_hit.u = 0.0f;
        out_hit.v = 0.0f;
        out_hit.normal = float3(0.0f, 0.0f, 1.0f);
        out_hit.packed_albedo = 0u;
    }
    hits[tid] = out_hit;
}

kernel void rtx_trace_screen_kernel(
    primitive_acceleration_structure accel [[buffer(0)]],
    device const RtxTriangleGpu* triangles [[buffer(1)]],
    device const float4* world_positions [[buffer(2)]],
    device const float4* world_normals [[buffer(3)]],
    constant ScreenParams& params [[buffer(4)]],
    device uint* out_rgba8_pairs [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }
    uint cell = gid.y * params.width + gid.x;
    float4 pos_valid = world_positions[cell];
    if (pos_valid.w <= 0.0f) {
        out_rgba8_pairs[cell * 2u + 0u] = 0xFF000000u;
        out_rgba8_pairs[cell * 2u + 1u] = 0u;
        return;
    }

    float3 P = pos_valid.xyz;
    float4 norm_rough = world_normals[cell];
    float3 N = normalize(norm_rough.xyz);
    float roughness = clamp(norm_rough.w, 0.04f, 1.0f);
    float3 origin = P + N * 0.025f;
    uint seed = pcg_hash(cell ^ (params.frame_index * 1664525u + 1013904223u));

    intersector<triangle_data> closest_inter;
    closest_inter.assume_geometry_type(geometry_type::triangle);
    intersector<triangle_data> shadow_inter;
    shadow_inter.assume_geometry_type(geometry_type::triangle);
    shadow_inter.accept_any_intersection(true);

    uint spp = max(1u, min(params.samples_per_pixel, 8u));
    float3 sun_dir = float3(params.sun_dir);
    float3 sun_colour = float3(params.sun_colour);
    float3 sky_colour = float3(params.sky_colour);
    float3 camera_pos = float3(params.camera_pos);
    float3 to_sun = normalize(-sun_dir);

    // 1. Hardware ray-traced soft sun shadows.
    float shadow_vis = 1.0f;
    if ((params.flags & 1u) != 0u) {
        float ndotl = dot(N, to_sun);
        if (ndotl > 0.0f) {
            float3 st, sb;
            make_orthonormal_basis(to_sun, st, sb);
            uint unoccluded = 0u;
            for (uint s = 0u; s < spp; ++s) {
                float u1 = rand01(seed);
                float u2 = rand01(seed);
                float r = sqrt(u1) * params.shadow_softness;
                float phi = 6.28318530718f * u2;
                float3 dir = normalize(to_sun + (st * cos(phi) + sb * sin(phi)) * r);
                ray sray(origin, dir, 0.02f, params.ray_distance);
                auto sres = shadow_inter.intersect(sray, accel);
                if (sres.type == intersection_type::none) {
                    unoccluded += 1u;
                }
            }
            shadow_vis = float(unoccluded) / float(spp);
        } else {
            shadow_vis = 0.0f;
        }
    }

    // 2. Hardware ray-traced ambient occlusion & 1-bounce diffuse global illumination.
    float ao_vis = 1.0f;
    float3 gi_radiance = float3(0.0f);
    if ((params.flags & 10u) != 0u) {
        uint ao_hits = 0u;
        for (uint s = 0u; s < spp; ++s) {
            float3 dir = sample_cosine_hemisphere(N, rand01(seed), rand01(seed));
            float max_t = ((params.flags & 8u) != 0u) ? params.ray_distance : params.ao_radius;
            ray b_ray(origin, dir, 0.02f, max_t);
            auto b_res = closest_inter.intersect(b_ray, accel);
            if (b_res.type == intersection_type::triangle) {
                if (b_res.distance <= params.ao_radius) {
                    ao_hits += 1u;
                }
                if ((params.flags & 8u) != 0u) {
                    RtxTriangleGpu hit_tri = triangles[b_res.primitive_id];
                    float3 hit_n = hit_tri.normal_albedo.xyz;
                    if (dot(hit_n, dir) > 0.0f) {
                        hit_n = -hit_n;
                    }
                    float3 albedo = unpack_rgb8(as_type<uint>(hit_tri.normal_albedo.w));
                    float3 hit_p = origin + dir * b_res.distance + hit_n * 0.03f;
                    float sun_ndotl = max(0.0f, dot(hit_n, to_sun));
                    float hit_sun = 0.0f;
                    if (sun_ndotl > 0.0f) {
                        ray sray2(hit_p, to_sun, 0.02f, params.ray_distance);
                        if (shadow_inter.intersect(sray2, accel).type == intersection_type::none) {
                            hit_sun = sun_ndotl;
                        }
                    }
                    gi_radiance += albedo * (sun_colour * hit_sun + sky_colour * 0.25f);
                }
            } else if ((params.flags & 8u) != 0u) {
                gi_radiance += sky_colour * max(0.2f, 0.5f + 0.5f * dir.z);
            }
        }
        if ((params.flags & 2u) != 0u) {
            ao_vis = 1.0f - (float(ao_hits) / float(spp));
        }
        if ((params.flags & 8u) != 0u) {
            gi_radiance = (gi_radiance / float(spp)) * params.gi_strength;
        }
    }

    // 3. Hardware ray-traced specular reflections.
    float3 refl_radiance = float3(0.0f);
    float refl_confidence = 0.0f;
    if ((params.flags & 4u) != 0u && roughness <= params.roughness_cutoff) {
        float3 V = normalize(P - camera_pos);
        float3 R = reflect(V, N);
        float3 rt, rb;
        make_orthonormal_basis(R, rt, rb);
        float cone = roughness * roughness * 0.35f;
        float u1 = rand01(seed);
        float u2 = rand01(seed);
        float3 dir = normalize(R + (rt * cos(6.2831853f * u2) + rb * sin(6.2831853f * u2)) * (sqrt(u1) * cone));
        if (dot(dir, N) > 0.01f) {
            ray r_ray(origin, dir, 0.025f, params.ray_distance);
            auto r_res = closest_inter.intersect(r_ray, accel);
            if (r_res.type == intersection_type::triangle) {
                RtxTriangleGpu hit_tri = triangles[r_res.primitive_id];
                float3 hit_n = hit_tri.normal_albedo.xyz;
                if (dot(hit_n, dir) > 0.0f) {
                    hit_n = -hit_n;
                }
                float3 albedo = unpack_rgb8(as_type<uint>(hit_tri.normal_albedo.w));
                float3 hit_p = origin + dir * r_res.distance + hit_n * 0.03f;
                float sun_ndotl = max(0.0f, dot(hit_n, to_sun));
                float hit_sun = 0.0f;
                if (sun_ndotl > 0.0f) {
                    ray sray2(hit_p, to_sun, 0.02f, params.ray_distance);
                    if (shadow_inter.intersect(sray2, accel).type == intersection_type::none) {
                        hit_sun = sun_ndotl;
                    }
                }
                refl_radiance = albedo * (sun_colour * hit_sun + sky_colour * 0.35f);
                refl_confidence = clamp(1.0f - r_res.distance / params.ray_distance, 0.15f, 1.0f);
            } else {
                refl_radiance = sky_colour * max(0.25f, 0.5f + 0.5f * dir.z);
                refl_confidence = 0.5f;
            }
        }
    }

    float combined_vis = shadow_vis * ao_vis;
    float3 encoded_gi = encode_early_gi(gi_radiance);
    out_rgba8_pairs[cell * 2u + 0u] = pack_rgba8(float4(encoded_gi, combined_vis));
    out_rgba8_pairs[cell * 2u + 1u] = pack_rgba8(float4(clamp(refl_radiance, 0.0f, 1.0f), refl_confidence));
}
)msl";

struct alignas(16) MetalScreenParams {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t samples_per_pixel = 1;
    std::uint32_t flags = 0;
    float camera_pos[3]{};
    float ray_distance = 120.0f;
    float sun_dir[3]{};
    float ao_radius = 2.5f;
    float sun_colour[3]{};
    float shadow_softness = 0.035f;
    float sky_colour[3]{};
    float gi_strength = 1.0f;
    std::uint32_t frame_index = 0;
    float roughness_cutoff = 0.65f;
    float pad[2]{};
};

class MetalRtxBackend final : public RtxHardwareBackend {
public:
    MetalRtxBackend(
        id<MTLDevice> device,
        id<MTLCommandQueue> queue,
        id<MTLComputePipelineState> trace_rays_pso,
        id<MTLComputePipelineState> trace_screen_pso
    )
        : m_device(device),
          m_queue(queue),
          m_trace_rays_pso(trace_rays_pso),
          m_trace_screen_pso(trace_screen_pso) {
        if (m_device && [m_device name]) {
            m_device_name = std::string([[m_device name] UTF8String]) + " (Metal RT)";
        } else {
            m_device_name = "Apple Metal Ray Tracing";
        }
    }

    renderer::RtxBackend kind() const override { return renderer::RtxBackend::Metal; }
    std::string_view device_name() const override { return m_device_name; }

    bool upload_scene(
        std::span<const RtxTriangle> triangles,
        std::span<const RtxBvhNode>,
        std::string& error
    ) override {
        @autoreleasepool {
            std::lock_guard lock(m_mutex);
            m_accel = nil;
            m_triangle_buffer = nil;
            m_vertex_buffer = nil;
            if (triangles.empty()) {
                return true;
            }

            // Pack 3 float3 vertices per triangle for MTLAccelerationStructureTriangleGeometryDescriptor.
            std::vector<std::array<float, 3>> vertices(triangles.size() * 3);
            for (std::size_t i = 0; i < triangles.size(); ++i) {
                const auto& t = triangles[i];
                const std::array<float, 3> v0{t.v0_mat[0], t.v0_mat[1], t.v0_mat[2]};
                const std::array<float, 3> v1{
                    v0[0] + t.e1_layer[0], v0[1] + t.e1_layer[1], v0[2] + t.e1_layer[2]
                };
                const std::array<float, 3> v2{
                    v0[0] + t.e2_flags[0], v0[1] + t.e2_flags[1], v0[2] + t.e2_flags[2]
                };
                vertices[i * 3 + 0] = v0;
                vertices[i * 3 + 1] = v1;
                vertices[i * 3 + 2] = v2;
            }

            m_vertex_buffer = [m_device newBufferWithBytes:vertices.data()
                                                    length:vertices.size() * sizeof(float) * 3
                                                   options:MTLResourceStorageModeShared];
            m_triangle_buffer = [m_device newBufferWithBytes:triangles.data()
                                                      length:triangles.size() * sizeof(RtxTriangle)
                                                     options:MTLResourceStorageModeShared];
            if (!m_vertex_buffer || !m_triangle_buffer) {
                error = "Metal RT: failed to allocate geometry buffers";
                return false;
            }

            MTLAccelerationStructureTriangleGeometryDescriptor* geom =
                [MTLAccelerationStructureTriangleGeometryDescriptor descriptor];
            geom.vertexBuffer = m_vertex_buffer;
            geom.vertexStride = sizeof(float) * 3;
            geom.triangleCount = triangles.size();
            geom.opaque = YES;

            MTLPrimitiveAccelerationStructureDescriptor* accel_desc =
                [MTLPrimitiveAccelerationStructureDescriptor descriptor];
            accel_desc.geometryDescriptors = @[ geom ];

            MTLAccelerationStructureSizes sizes =
                [m_device accelerationStructureSizesWithDescriptor:accel_desc];
            m_accel = [m_device newAccelerationStructureWithSize:sizes.accelerationStructureSize];
            id<MTLBuffer> scratch =
                [m_device newBufferWithLength: std::max<NSUInteger>(sizes.buildScratchBufferSize, 256)
                                      options:MTLResourceStorageModePrivate];
            if (!m_accel || !scratch) {
                error = "Metal RT: failed to allocate MTLAccelerationStructure";
                return false;
            }

            id<MTLCommandBuffer> cmd = [m_queue commandBuffer];
            id<MTLAccelerationStructureCommandEncoder> enc =
                [cmd accelerationStructureCommandEncoder];
            [enc buildAccelerationStructure:m_accel
                                 descriptor:accel_desc
                              scratchBuffer:scratch
                        scratchBufferOffset:0];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
            if (cmd.status == MTLCommandBufferStatusError) {
                error = cmd.error ? [[cmd.error localizedDescription] UTF8String]
                                  : "Metal RT: acceleration structure build failed";
                m_accel = nil;
                return false;
            }
            return true;
        }
    }

    bool trace_rays(
        std::span<const RtxRay> rays,
        std::span<RtxHit> hits,
        std::string& error
    ) override {
        @autoreleasepool {
            std::lock_guard lock(m_mutex);
            if (rays.size() != hits.size()) {
                error = "Metal RT: ray and hit buffer sizes must match";
                return false;
            }
            if (rays.empty()) {
                return true;
            }
            if (!m_accel || !m_triangle_buffer) {
                for (auto& h : hits) {
                    h = RtxHit{};
                }
                return true;
            }

            id<MTLBuffer> ray_buf = [m_device newBufferWithBytes:rays.data()
                                                          length:rays.size() * sizeof(RtxRay)
                                                         options:MTLResourceStorageModeShared];
            id<MTLBuffer> hit_buf = [m_device newBufferWithLength:hits.size() * sizeof(RtxHit)
                                                          options:MTLResourceStorageModeShared];
            const std::uint32_t count = static_cast<std::uint32_t>(rays.size());

            id<MTLCommandBuffer> cmd = [m_queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:m_trace_rays_pso];
            [enc setAccelerationStructure:m_accel atBufferIndex:0];
            [enc setBuffer:m_triangle_buffer offset:0 atIndex:1];
            [enc setBuffer:ray_buf offset:0 atIndex:2];
            [enc setBuffer:hit_buf offset:0 atIndex:3];
            [enc setBytes:&count length:sizeof(count) atIndex:4];

            const NSUInteger threadgroup_size =
                std::min<NSUInteger>(64, m_trace_rays_pso.maxTotalThreadsPerThreadgroup);
            const NSUInteger groups = (rays.size() + threadgroup_size - 1) / threadgroup_size;
            [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(threadgroup_size, 1, 1)];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
            if (cmd.status == MTLCommandBufferStatusError) {
                error = cmd.error ? [[cmd.error localizedDescription] UTF8String]
                                  : "Metal RT: ray dispatch failed";
                return false;
            }
            std::memcpy(hits.data(), [hit_buf contents], hits.size() * sizeof(RtxHit));
            return true;
        }
    }

    bool trace_screen(
        int cells_width,
        int cells_height,
        std::span<const std::array<float, 4>> world_positions,
        std::span<const std::array<float, 4>> world_normals,
        const std::array<float, 3>& camera_pos,
        const std::array<float, 3>& sun_dir,
        const std::array<float, 3>& sun_colour,
        const std::array<float, 3>& sky_colour,
        const renderer::RtxSettings& settings,
        std::uint32_t frame_index,
        std::span<std::uint32_t> out_rgba8_pairs,
        std::string& error
    ) override {
        @autoreleasepool {
            std::lock_guard lock(m_mutex);
            const std::size_t cells =
                static_cast<std::size_t>(std::max(0, cells_width)) * std::max(0, cells_height);
            if (cells == 0 || out_rgba8_pairs.size() < cells * 2) {
                return false;
            }
            if (!m_accel || !m_triangle_buffer) {
                for (std::size_t i = 0; i < cells; ++i) {
                    out_rgba8_pairs[i * 2 + 0] = 0xFF000000u;
                    out_rgba8_pairs[i * 2 + 1] = 0u;
                }
                return true;
            }

            MetalScreenParams params{};
            params.width = static_cast<std::uint32_t>(cells_width);
            params.height = static_cast<std::uint32_t>(cells_height);
            params.samples_per_pixel =
                static_cast<std::uint32_t>(std::clamp(settings.samples_per_pixel, 1, 8));
            params.flags = (settings.shadows ? 1u : 0u)
                           | (settings.ambient_occlusion ? 2u : 0u)
                           | (settings.reflections ? 4u : 0u)
                           | (settings.global_illumination ? 8u : 0u);
            std::memcpy(params.camera_pos, camera_pos.data(), sizeof(float) * 3);
            params.ray_distance = settings.ray_distance;
            std::memcpy(params.sun_dir, sun_dir.data(), sizeof(float) * 3);
            params.ao_radius = settings.ao_radius;
            std::memcpy(params.sun_colour, sun_colour.data(), sizeof(float) * 3);
            params.shadow_softness = settings.shadow_softness;
            std::memcpy(params.sky_colour, sky_colour.data(), sizeof(float) * 3);
            params.gi_strength = settings.gi_strength;
            params.frame_index = frame_index;
            params.roughness_cutoff = settings.reflection_roughness_cutoff;

            id<MTLBuffer> pos_buf =
                [m_device newBufferWithBytes:world_positions.data()
                                      length:cells * sizeof(float) * 4
                                     options:MTLResourceStorageModeShared];
            id<MTLBuffer> norm_buf =
                [m_device newBufferWithBytes:world_normals.data()
                                      length:cells * sizeof(float) * 4
                                     options:MTLResourceStorageModeShared];
            id<MTLBuffer> out_buf =
                [m_device newBufferWithLength:cells * 2 * sizeof(std::uint32_t)
                                      options:MTLResourceStorageModeShared];

            id<MTLCommandBuffer> cmd = [m_queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:m_trace_screen_pso];
            [enc setAccelerationStructure:m_accel atBufferIndex:0];
            [enc setBuffer:m_triangle_buffer offset:0 atIndex:1];
            [enc setBuffer:pos_buf offset:0 atIndex:2];
            [enc setBuffer:norm_buf offset:0 atIndex:3];
            [enc setBytes:&params length:sizeof(params) atIndex:4];
            [enc setBuffer:out_buf offset:0 atIndex:5];

            const MTLSize group_size = MTLSizeMake(8, 8, 1);
            const MTLSize groups = MTLSizeMake(
                (static_cast<NSUInteger>(cells_width) + 7) / 8,
                (static_cast<NSUInteger>(cells_height) + 7) / 8,
                1
            );
            [enc dispatchThreadgroups:groups threadsPerThreadgroup:group_size];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
            if (cmd.status == MTLCommandBufferStatusError) {
                error = cmd.error ? [[cmd.error localizedDescription] UTF8String]
                                  : "Metal RT: screen trace dispatch failed";
                return false;
            }
            std::memcpy(
                out_rgba8_pairs.data(),
                [out_buf contents],
                cells * 2 * sizeof(std::uint32_t)
            );
            return true;
        }
    }

private:
    std::mutex m_mutex;
    id<MTLDevice> m_device = nil;
    id<MTLCommandQueue> m_queue = nil;
    id<MTLComputePipelineState> m_trace_rays_pso = nil;
    id<MTLComputePipelineState> m_trace_screen_pso = nil;
    id<MTLAccelerationStructure> m_accel = nil;
    id<MTLBuffer> m_vertex_buffer = nil;
    id<MTLBuffer> m_triangle_buffer = nil;
    std::string m_device_name;
};

}  // namespace

bool metal_rtx_supported() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        return device != nil && [device supportsRaytracing];
    }
}

std::unique_ptr<RtxHardwareBackend> create_metal_rtx_backend() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device || ![device supportsRaytracing]) {
            return nullptr;
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) {
            return nullptr;
        }
        MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
        opts.languageVersion = MTLLanguageVersion2_4;
        NSError* err = nil;
        NSString* src = [NSString stringWithUTF8String:kMetalRtxShaderSource];
        id<MTLLibrary> lib = [device newLibraryWithSource:src options:opts error:&err];
        if (!lib) {
            return nullptr;
        }
        id<MTLFunction> rays_fn = [lib newFunctionWithName:@"rtx_trace_rays_kernel"];
        id<MTLFunction> screen_fn = [lib newFunctionWithName:@"rtx_trace_screen_kernel"];
        if (!rays_fn || !screen_fn) {
            return nullptr;
        }
        id<MTLComputePipelineState> rays_pso =
            [device newComputePipelineStateWithFunction:rays_fn error:&err];
        id<MTLComputePipelineState> screen_pso =
            [device newComputePipelineStateWithFunction:screen_fn error:&err];
        if (!rays_pso || !screen_pso) {
            return nullptr;
        }
        return std::make_unique<MetalRtxBackend>(device, queue, rays_pso, screen_pso);
    }
}

}  // namespace openrac::enhanced
#endif
