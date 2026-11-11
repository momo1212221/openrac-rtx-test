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
#include <array>
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
// ---- One frame on the GPU (RtxFrameJob): rtx_gbuffer.comp's cells in, rtx_trace.comp's words out.

struct FrameParams {
    float4x4 prev_view_projection;
    float4 camera_pos;
    float4 sun_dir;      // the light's travel direction
    float4 sun_colour;
    float4 sky_colour;
    uint width;
    uint height;
    uint samples;
    uint flags;          // bit0 shadows, bit1 AO, bit2 reflections, bit3 bounce
    uint frame_index;
    uint has_dynamic;
    uint history_valid;
    uint light_count;    // local lights in `lights`, four float4 each (local_lights.cpp's records)
    float ray_distance;
    float ao_radius;
    float shadow_softness;
    float gi_strength;
    float roughness_cutoff;
    float history_blend;
    uint has_emitters;   // the glowing surfaces' own structure (buffers 10 and 11)
    uint local_count;    // lights the local-light visibility walks (the first ones, at most 64)
};

struct FrameHit {
    bool hit;
    float t;
    float3 normal;
    float3 albedo;
    uint material;  // e2_flags.w's bits: roughness, metallic, F0 (rtx.cpp pack_material)
};

// The nearer hit of the level's structure and the mobys'.
inline FrameHit frame_closest(ray r, primitive_acceleration_structure level,
                              device const RtxTriangleGpu* level_tris,
                              primitive_acceleration_structure mobys,
                              device const RtxTriangleGpu* moby_tris, bool has_mobys) {
    intersector<triangle_data> it;
    it.assume_geometry_type(geometry_type::triangle);
    FrameHit h;
    h.hit = false;
    h.t = r.max_distance;
    h.normal = float3(0.0f, 0.0f, 1.0f);
    h.albedo = float3(0.5f);
    h.material = 140u;  // roughness 0.55
    auto a = it.intersect(r, level);
    if (a.type == intersection_type::triangle) {
        RtxTriangleGpu tri = level_tris[a.primitive_id];
        h.hit = true;
        h.t = a.distance;
        h.normal = tri.normal_albedo.xyz;
        h.albedo = unpack_rgb8(as_type<uint>(tri.normal_albedo.w));
        h.material = as_type<uint>(tri.e2_flags.w);
        r.max_distance = a.distance;
    }
    if (has_mobys) {
        auto b = it.intersect(r, mobys);
        if (b.type == intersection_type::triangle) {
            RtxTriangleGpu tri = moby_tris[b.primitive_id];
            h.hit = true;
            h.t = b.distance;
            h.normal = tri.normal_albedo.xyz;
            h.albedo = unpack_rgb8(as_type<uint>(tri.normal_albedo.w));
            h.material = as_type<uint>(tri.e2_flags.w);
        }
    }
    return h;
}

inline bool frame_blocked(ray r, primitive_acceleration_structure level,
                          primitive_acceleration_structure mobys, bool has_mobys) {
    intersector<triangle_data> it;
    it.assume_geometry_type(geometry_type::triangle);
    it.accept_any_intersection(true);
    if (it.intersect(r, level).type != intersection_type::none) {
        return true;
    }
    return has_mobys && it.intersect(r, mobys).type != intersection_type::none;
}

inline float4 unpack_rgba8(uint u) {
    return float4(float(u & 0xFFu), float((u >> 8u) & 0xFFu), float((u >> 16u) & 0xFFu),
                  float(u >> 24u)) * (1.0f / 255.0f);
}

// At most `limit` in luminance, the colour kept: one path that meets a bright glow through a
// gap is a firefly the few frames of history cannot average away.
inline float3 clamp_luminance(float3 c, float limit) {
    float l = dot(c, float3(0.2126f, 0.7152f, 0.0722f));
    return l > limit ? c * (limit / l) : c;
}

// A point of the R2 sequence (the plastic number's lattice) for sample `n` of a cell: over the
// frames the history averages, a cell's rays cover the sun's disc, the hemisphere and a
// reflection's lobe evenly instead of at random, so the same few frames leave far less noise to
// flicker. `salt` gives each use and each cell its own offset; integer steps keep the fraction
// exact however long the game runs.
inline float2 r2_sample(uint salt, uint n) {
    uint a = pcg_hash(salt), b = pcg_hash(a ^ 0x9E3779B9u);
    uint x = a + n * 3242174889u, y = b + n * 2447445414u;
    return float2(float(x >> 8u), float(y >> 8u)) * (1.0f / 16777216.0f);
}

// encode_early_gi's inverse, a channel at a time (0 stays no light).
inline float3 decode_early_gi(float3 e) {
    return select(float3(0.0f), exp2(e * 12.0f - 8.0f), e > float3(0.0f));
}

// pack_rgba8 rounded at random in proportion to the remainder: a history that takes a twelfth of
// each new frame would never move a byte by less than half a step, and stuck in visible bands.
inline uint pack_rgba8_dithered(float4 c, thread uint& seed) {
    float4 r = float4(rand01(seed), rand01(seed), rand01(seed), rand01(seed));
    uint4 u = uint4(min(clamp(c, 0.0f, 1.0f) * 255.0f + r, 255.0f));
    return u.x | (u.y << 8u) | (u.z << 16u) | (u.w << 24u);
}

// A path's hit: the level, the mobys, or a glowing surface (its own structure: it lights, it is
// seen and reflected, but it never shadows, so a bulb does not darken its own light).
struct PathHit {
    bool hit;
    bool emitter;
    float t;
    float3 normal;
    float3 albedo;
    float3 emission;
};

inline PathHit path_closest(ray r, primitive_acceleration_structure level,
                            device const RtxTriangleGpu* level_tris,
                            primitive_acceleration_structure mobys,
                            device const RtxTriangleGpu* moby_tris, bool has_mobys,
                            primitive_acceleration_structure emitters,
                            device const RtxTriangleGpu* emitter_tris, bool has_emitters) {
    FrameHit h = frame_closest(r, level, level_tris, mobys, moby_tris, has_mobys);
    PathHit o;
    o.hit = h.hit;
    o.emitter = false;
    o.t = h.t;
    o.normal = h.normal;
    o.albedo = h.albedo;
    o.emission = float3(0.0f);
    // A posed character's glowing part (Clank's antenna bulb, his eyes) is in the mobys'
    // structure: its strength x16 in the material's top byte, its colour as the albedo.
    if (h.hit && (h.material >> 24u) != 0u) {
        o.emitter = true;
        o.emission = h.albedo * (float(h.material >> 24u) / 16.0f);
    }
    if (has_emitters) {
        r.max_distance = h.hit ? h.t : r.max_distance;
        intersector<triangle_data> it;
        it.assume_geometry_type(geometry_type::triangle);
        auto e = it.intersect(r, emitters);
        if (e.type == intersection_type::triangle) {
            RtxTriangleGpu tri = emitter_tris[e.primitive_id];
            o.hit = true;
            o.emitter = true;
            o.t = e.distance;
            o.normal = tri.normal_albedo.xyz;
            // The glow's colour (peak 1) and its strength (e2_flags.w).
            o.emission = unpack_rgb8(as_type<uint>(tri.normal_albedo.w)) * tri.e2_flags.w;
        }
    }
    return o;
}

// One local light's light at a point, as the mesh shader sums it (local_lights.glsl's falloff
// and cone), unshadowed.
inline float3 local_light_at(device const float4* lights, uint i, float3 pos, float3 n,
                             thread float3& toward, thread float& d) {
    float4 lp = lights[i * 4u + 0u];
    float4 lc = lights[i * 4u + 1u];
    float radius = lp.w;
    float3 delta = lp.xyz - pos;
    d = length(delta);
    if (radius <= 0.0f || d >= radius || d <= 1e-4f) {
        return float3(0.0f);
    }
    toward = delta / d;
    float ndotl = dot(n, toward);
    if (ndotl <= 0.0f) {
        return float3(0.0f);
    }
    float x = d / radius, x2 = x * x;
    float window = max(0.0f, 1.0f - x2 * x2);
    float falloff = window * window / (1.0f + 3.03125f * x2);
    float cone = 1.0f;
    float4 ld = lights[i * 4u + 2u];
    float4 lpar = lights[i * 4u + 3u];
    if (lpar.y != 0.0f) {
        float cosine = dot(-toward, ld.xyz);
        cone = lpar.x == ld.w ? float(cosine >= ld.w) : smoothstep(ld.w, lpar.x, cosine);
    }
    return max(lc.rgb, float3(0.0f)) * ndotl * falloff * cone;
}

// Next-event estimation of the local lights at a path vertex: eight lights drawn at random, one
// of them kept in proportion to its unshadowed light (resampled importance sampling), then one
// shadow ray to it. Unbiased for all `count` lights, with one ray instead of `count`.
inline float3 sample_local_lights(device const float4* lights, uint count, float3 pos, float3 n,
                                  thread uint& seed, primitive_acceleration_structure level,
                                  primitive_acceleration_structure mobys, bool has_mobys) {
    if (count == 0u) {
        return float3(0.0f);
    }
    constexpr uint kCandidates = 8u;
    float total = 0.0f, kept_weight = 0.0f, kept_distance = 0.0f;
    float3 kept = float3(0.0f), kept_toward = float3(0.0f, 0.0f, 1.0f);
    for (uint k = 0u; k < kCandidates; ++k) {
        uint i = min(uint(rand01(seed) * float(count)), count - 1u);
        float3 toward;
        float d;
        float3 c = local_light_at(lights, i, pos, n, toward, d);
        float w = dot(c, float3(0.2126f, 0.7152f, 0.0722f));
        if (w <= 0.0f) {
            continue;
        }
        total += w;
        if (rand01(seed) * total < w) {
            kept = c;
            kept_weight = w;
            kept_toward = toward;
            kept_distance = d;
        }
    }
    if (total <= 0.0f || kept_weight <= 0.0f) {
        return float3(0.0f);
    }
    ray lr(pos, kept_toward, 0.025f, max(kept_distance - 0.1f, 0.03f));
    if (frame_blocked(lr, level, mobys, has_mobys)) {
        return float3(0.0f);
    }
    return kept / kept_weight * (total / float(kCandidates)) * float(count);
}

kernel void rtx_trace_frame_kernel(
    primitive_acceleration_structure level [[buffer(0)]],
    device const RtxTriangleGpu* level_tris [[buffer(1)]],
    primitive_acceleration_structure mobys [[buffer(2)]],
    device const RtxTriangleGpu* moby_tris [[buffer(3)]],
    device float4* cells [[buffer(4)]],
    constant FrameParams& p [[buffer(5)]],
    device uint* result [[buffer(6)]],
    device const uint* history [[buffer(7)]],
    device const float4* lights [[buffer(8)]],
    device const float4* history_cells [[buffer(9)]],
    primitive_acceleration_structure emitters [[buffer(10)]],
    device const RtxTriangleGpu* emitter_tris [[buffer(11)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= p.width || gid.y >= p.height) {
        return;
    }
    uint cell = gid.y * p.width + gid.x;
    float4 position = cells[cell * 2u];
    // Nothing traced past 70 units: the level's far backdrop (Blarg's stations and asteroids
    // behind the windows) took the trace's shadows, AO and noise, and costs rays for nothing.
    // early_lighting.comp fades the trace out from 40 units on, so this edge never shows.
    if (position.w <= 0.0f || distance(position.xyz, p.camera_pos.xyz) > 70.0f) {
        result[cell * 3u + 0u] = 0xFF000000u;
        result[cell * 3u + 1u] = 0xFF000000u;
        result[cell * 3u + 2u] = 0x000000FFu;  // local lights unblocked, no glow light
        cells[cell * 2u + 1u].w = 0.0f;        // no frames gathered here
        return;
    }
    float4 normal_rough = cells[cell * 2u + 1u];
    float3 N = normalize(normal_rough.xyz);
    float3 origin = position.xyz + N * 0.03f;
    uint seed = pcg_hash(cell ^ (p.frame_index * 1664525u + 1013904223u));
    uint spp = clamp(p.samples, 1u, 8u);
    bool mobys_on = p.has_dynamic != 0u;
    bool glow_on = p.has_emitters != 0u;
    float3 to_sun = normalize(-p.sun_dir.xyz);

    // Sun: soft shadow rays toward a cone around the sun.
    float sun_visible = 1.0f;
    if ((p.flags & 1u) != 0u) {
        if (dot(N, to_sun) > 0.0f) {
            float3 st, sb;
            make_orthonormal_basis(to_sun, st, sb);
            uint lit = 0u;
            for (uint s = 0u; s < spp; ++s) {
                float2 u = r2_sample(cell * 3u, p.frame_index * spp + s);
                float r = sqrt(u.x) * p.shadow_softness;
                float phi = 6.28318530718f * u.y;
                ray sr(origin, normalize(to_sun + (st * cos(phi) + sb * sin(phi)) * r), 0.025f,
                       p.ray_distance);
                lit += frame_blocked(sr, level, mobys, mobys_on) ? 0u : 1u;
            }
            sun_visible = float(lit) / float(spp);
        } else {
            sun_visible = 0.0f;
        }
    }

    // Ambient occlusion, and path-traced indirect light: two bounces from cosine-weighted
    // directions, at each vertex the sun and the local lights sampled (next-event estimation),
    // glowing surfaces met on the way as light. `bounce` is the sun and the sky (it fills only
    // where no probe bake lights the level); `glow` is what the local lights and the glowing
    // surfaces send around, which the bake does not hold and which is added everywhere.
    float ambient = 1.0f;
    float3 bounce = float3(0.0f);
    float3 glow = float3(0.0f);
    if ((p.flags & 10u) != 0u) {
        bool gi = (p.flags & 8u) != 0u;
        uint blocked = 0u;
        for (uint s = 0u; s < spp; ++s) {
            float2 u = r2_sample(cell * 3u + 1u, p.frame_index * spp + s);
            float3 dir = sample_cosine_hemisphere(N, u.x, u.y);
            if (!gi) {
                ray br(origin, dir, 0.025f, p.ao_radius);
                blocked += frame_blocked(br, level, mobys, mobys_on) ? 1u : 0u;
                continue;
            }
            float3 at = origin;
            float3 throughput = float3(1.0f);
            for (uint b = 0u; b < 2u; ++b) {
                ray pr(at, dir, 0.025f, p.ray_distance);
                PathHit h = path_closest(pr, level, level_tris, mobys, moby_tris, mobys_on,
                                         emitters, emitter_tris, glow_on);
                if (b == 0u && h.hit && h.t <= p.ao_radius && !h.emitter) {
                    ++blocked;
                }
                if (!h.hit) {
                    bounce += throughput * p.sky_colour.xyz * max(0.2f, 0.5f + 0.5f * dir.z);
                    break;
                }
                if (h.emitter) {
                    glow += clamp_luminance(throughput * h.emission, 1.5f);
                    break;
                }
                float3 hn = dot(h.normal, dir) > 0.0f ? -h.normal : h.normal;
                float3 hp = at + dir * h.t + hn * 0.03f;
                float ndotl = max(0.0f, dot(hn, to_sun));
                float sun_at_hit = 0.0f;
                if (ndotl > 0.0f) {
                    ray sr(hp, to_sun, 0.025f, p.ray_distance);
                    sun_at_hit = frame_blocked(sr, level, mobys, mobys_on) ? 0.0f : ndotl;
                }
                bounce += throughput * h.albedo
                          * (p.sun_colour.xyz * sun_at_hit + p.sky_colour.xyz * 0.25f);
                glow += clamp_luminance(throughput * h.albedo
                                            * sample_local_lights(lights, p.light_count, hp, hn,
                                                                  seed, level, mobys, mobys_on),
                                        1.5f);
                throughput *= h.albedo;
                if (max(throughput.x, max(throughput.y, throughput.z)) < 0.02f) {
                    break;
                }
                at = hp;
                dir = sample_cosine_hemisphere(hn, rand01(seed), rand01(seed));
            }
        }
        if ((p.flags & 2u) != 0u) {
            ambient = 1.0f - float(blocked) / float(spp);
        }
        bounce = gi ? bounce / float(spp) * p.gi_strength : float3(0.0f);
        glow = gi ? glow / float(spp) * p.gi_strength : float3(0.0f);
    }

    // A reflection where the surface is glossy. Its material is the triangle the camera sees at
    // this point (roughness, metallic, F0 from the texture pack's maps, or the game's chrome);
    // the ray leaves along a GGX-sampled half vector of that roughness. What it meets is lit by
    // the sun, the sky and a sampled local light, or is a glowing surface's own light: radiance,
    // not weighted (the mesh shader applies its own Fresnel). Nothing traced stays 0, which the
    // early lighting reads as "use the probes' environment".
    float3 reflection = float3(0.0f);
    if ((p.flags & 4u) != 0u && p.roughness_cutoff > 0.05f) {
        float3 to_point = position.xyz - p.camera_pos.xyz;
        float distance_to_point = length(to_point);
        float3 V = to_point / max(distance_to_point, 1e-4f);
        float roughness = 0.55f, metallic = 0.0f, f0 = 0.0f;
        ray cr(p.camera_pos.xyz, V, 0.0f, distance_to_point + 0.5f);
        FrameHit seen_here = frame_closest(cr, level, level_tris, mobys, moby_tris, mobys_on);
        if (seen_here.hit
            && abs(seen_here.t - distance_to_point) < max(0.3f, distance_to_point * 0.03f)) {
            roughness = float(seen_here.material & 0xFFu) / 255.0f;
            metallic = float((seen_here.material >> 8u) & 0xFFu) / 255.0f;
            f0 = float((seen_here.material >> 16u) & 0xFFu) / 255.0f;
        }
        // Smooth surfaces only: a rougher one's reflection from one path a cell is noise (Blarg's
        // window frames came out grainy white); it keeps the probes' calm environment.
        bool glossy = (metallic > 0.05f && roughness < 0.4f) || roughness < 0.25f;
        if (glossy && roughness <= p.roughness_cutoff) {
            // A third of the surface's lobe: one ray a cell from the full lobe of a half-rough metal
            // left blotches the history could not smooth (Blarg's ceiling ring). The mesh shader
            // gives a rougher surface more of the probes' blurred environment instead.
            float alpha = max(roughness * roughness * 0.33f, 0.002f);
            float2 lobe = r2_sample(cell * 3u + 2u, p.frame_index);
            float u1 = lobe.x, u2 = lobe.y;
            float cos_theta = sqrt((1.0f - u1) / (1.0f + (alpha * alpha - 1.0f) * u1));
            float sin_theta = sqrt(max(0.0f, 1.0f - cos_theta * cos_theta));
            float phi = 6.28318530718f * u2;
            float3 t, b;
            make_orthonormal_basis(N, t, b);
            float3 H = normalize(t * (sin_theta * cos(phi)) + b * (sin_theta * sin(phi)) + N * cos_theta);
            float3 R = reflect(V, H);
            if (dot(R, N) <= 0.01f) {
                R = reflect(V, N);
            }
            if (dot(R, N) > 0.01f) {
                ray rr(origin, R, 0.03f, p.ray_distance);
                PathHit h = path_closest(rr, level, level_tris, mobys, moby_tris, mobys_on, emitters,
                                         emitter_tris, glow_on);
                float3 seen;
                if (h.emitter) {
                    seen = h.emission;
                } else if (h.hit) {
                    float3 hn = dot(h.normal, R) > 0.0f ? -h.normal : h.normal;
                    float3 hp = origin + R * h.t + hn * 0.03f;
                    float ndotl = max(0.0f, dot(hn, to_sun));
                    float sun_at_hit = 0.0f;
                    if (ndotl > 0.0f) {
                        ray sr(hp, to_sun, 0.025f, p.ray_distance);
                        sun_at_hit = frame_blocked(sr, level, mobys, mobys_on) ? 0.0f : ndotl;
                    }
                    seen = h.albedo * (p.sun_colour.xyz * sun_at_hit + p.sky_colour.xyz * 0.35f
                                       + sample_local_lights(lights, p.light_count, hp, hn, seed,
                                                             level, mobys, mobys_on));
                } else {
                    seen = p.sky_colour.xyz * max(0.25f, 0.5f + 0.5f * R.z) * 1.5f;
                }
                // At least 0.01: a traced reflection never reads as "nothing traced"; at most 3.
                reflection = max(clamp_luminance(seen, 3.0f), float3(0.01f));
            }
        }
    }

    // The local lights: what share of their unshadowed light (as the mesh shader sums it, with
    // local_lights.glsl's falloff and cone) reaches this point past the level and the mobys.
    float local_visible = 1.0f;
    if ((p.flags & 16u) != 0u && p.local_count > 0u) {
        float open_sum = 0.0f, lit_sum = 0.0f;
        for (uint i = 0u; i < p.local_count; ++i) {
            float4 lp = lights[i * 4u + 0u];
            float4 lc = lights[i * 4u + 1u];
            float radius = lp.w;
            float3 delta = lp.xyz - position.xyz;
            float d = length(delta);
            if (radius <= 0.0f || d >= radius || d <= 1e-4f) {
                continue;
            }
            float3 toward = delta / d;
            float ndotl = dot(N, toward);
            if (ndotl <= 0.0f) {
                continue;
            }
            float x = d / radius, x2 = x * x;
            float window = max(0.0f, 1.0f - x2 * x2);
            float falloff = window * window / (1.0f + 3.03125f * x2);
            float cone = 1.0f;
            float4 ld = lights[i * 4u + 2u];
            float4 lpar = lights[i * 4u + 3u];
            if (lpar.y != 0.0f) {
                float cosine = dot(-toward, ld.xyz);
                cone = lpar.x == ld.w ? float(cosine >= ld.w) : smoothstep(ld.w, lpar.x, cosine);
            }
            float weight = dot(max(lc.rgb, float3(0.0f)), float3(0.2126f, 0.7152f, 0.0722f))
                           * ndotl * falloff * cone;
            if (weight <= 1e-5f) {
                continue;
            }
            open_sum += weight;
            ray lr(origin, toward, 0.025f, max(d - 0.1f, 0.03f));
            lit_sum += frame_blocked(lr, level, mobys, mobys_on) ? 0.0f : weight;
        }
        local_visible = open_sum > 1e-5f ? lit_sum / open_sum : 1.0f;
    }

    // The previous job's result at this point through its camera: many frames' rays together.
    // Its four cells around the point, each only where it traced this same surface (near this
    // point's plane and not far along it: a plain distance threw the ground's history away while
    // walking, where a cell seen at a grazing angle spans far more than the camera's step, and
    // what was left was each frame's raw rays, flickering). The light is averaged as light, not
    // as its log encoding. `frames` counts what a cell has gathered: a fresh cell takes the new
    // rays whole, then a half, a third, and so on down to history_blend, so nothing stays on its
    // first noisy sample.
    float frames = 1.0f;
    if (p.history_valid != 0u) {
        float4 clip = p.prev_view_projection * float4(position.xyz, 1.0f);
        float2 uv = clip.w > 1e-4f ? clip.xy / clip.w * 0.5f + 0.5f : float2(-1.0f);
        if (all(uv >= float2(0.0f)) && all(uv < float2(1.0f))) {
            float2 at = uv * float2(p.width, p.height) - 0.5f;
            int2 base = int2(floor(at));
            float2 f = at - float2(base);
            float reach = max(0.3f, distance(position.xyz, p.camera_pos.xyz) * 0.02f);
            float weight = 0.0f, reflected_weight = 0.0f, was_frames = 0.0f;
            float3 was_visible = float3(0.0f), was_bounce = float3(0.0f);
            float3 was_glow = float3(0.0f), was_reflection = float3(0.0f);
            for (uint i = 0u; i < 4u; ++i) {
                int2 c = base + int2(int(i & 1u), int(i >> 1u));
                if (c.x < 0 || c.y < 0 || c.x >= int(p.width) || c.y >= int(p.height)) {
                    continue;
                }
                uint pi = uint(c.y) * p.width + uint(c.x);
                float4 was = history_cells[pi * 2u];
                float3 offset = was.xyz - position.xyz;
                if (was.w <= 0.0f || abs(dot(offset, N)) > reach || length(offset) > reach * 8.0f) {
                    continue;
                }
                float w = max(((i & 1u) != 0u ? f.x : 1.0f - f.x)
                                  * ((i >> 1u) != 0u ? f.y : 1.0f - f.y),
                              0.001f);
                float4 h0 = unpack_rgba8(history[pi * 3u]);
                float4 h1 = unpack_rgba8(history[pi * 3u + 1u]);
                float4 h2 = unpack_rgba8(history[pi * 3u + 2u]);
                was_visible += w * float3(h0.w, h1.w, h2.x);
                was_bounce += w * decode_early_gi(h0.xyz);
                was_glow += w * decode_early_gi(h2.yzw);
                if (max(h1.x, max(h1.y, h1.z)) > 0.05f) {  // a traced reflection is at least 0.01
                    was_reflection += w * decode_early_gi(h1.xyz);
                    reflected_weight += w;
                }
                was_frames += w * history_cells[pi * 2u + 1u].w;
                weight += w;
            }
            if (weight > 0.05f) {
                float inv = 1.0f / weight;
                frames = min(was_frames * inv, 64.0f) + 1.0f;
                float blend = max(1.0f / frames, p.history_blend);
                sun_visible = mix(was_visible.x * inv, sun_visible, blend);
                ambient = mix(was_visible.y * inv, ambient, blend);
                // The local lights' shadows are exact each frame: half, to follow a moving light.
                local_visible = mix(was_visible.z * inv, local_visible, max(blend, 0.5f));
                bounce = mix(was_bounce * inv, bounce, blend);
                glow = mix(was_glow * inv, glow, blend);
                if (reflected_weight > 0.0f && max(reflection.x, max(reflection.y, reflection.z)) > 0.0f) {
                    reflection = mix(was_reflection / reflected_weight, reflection, blend);
                }
            }
        }
    }
    // Word 0: the bounce (log encoded) and the sun's visibility; word 1: the reflection (log
    // encoded) and the ambient visibility; word 2: the local lights' visibility, then the glow
    // light (log encoded). The cell's normal slot keeps its frame count for the next job.
    result[cell * 3u + 0u] = pack_rgba8_dithered(float4(encode_early_gi(bounce), sun_visible), seed);
    result[cell * 3u + 1u] = pack_rgba8_dithered(float4(encode_early_gi(reflection), ambient), seed);
    result[cell * 3u + 2u] = pack_rgba8_dithered(float4(local_visible, encode_early_gi(glow)), seed);
    cells[cell * 2u + 1u].w = frames;
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

// FrameParams in the kernel, field for field.
struct alignas(16) MetalFrameParams {
    float prev_view_projection[16]{};
    float camera_pos[4]{};
    float sun_dir[4]{};
    float sun_colour[4]{};
    float sky_colour[4]{};
    std::uint32_t width = 0, height = 0, samples = 1, flags = 0;
    std::uint32_t frame_index = 0, has_dynamic = 0, history_valid = 0, light_count = 0;
    float ray_distance = 120.0f, ao_radius = 2.5f, shadow_softness = 0.035f, gi_strength = 1.0f;
    float roughness_cutoff = 0.65f, history_blend = 1.0f;
    std::uint32_t has_emitters = 0, local_count = 0;
};
static_assert(sizeof(MetalFrameParams) == 192);  // FrameParams: 64 + 4 x 16 + 8 x 4 + 8 x 4

class MetalRtxBackend final : public RtxHardwareBackend {
public:
    MetalRtxBackend(
        id<MTLDevice> device,
        id<MTLCommandQueue> queue,
        id<MTLComputePipelineState> trace_rays_pso,
        id<MTLComputePipelineState> trace_screen_pso,
        id<MTLComputePipelineState> trace_frame_pso
    )
        : m_device(device),
          m_queue(queue),
          m_trace_rays_pso(trace_rays_pso),
          m_trace_screen_pso(trace_screen_pso),
          m_trace_frame_pso(trace_frame_pso) {
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

    void* native_device() const override { return (__bridge void*)m_device; }

    // Builds the mobys' structure into `cmd` when they changed; false without any.
    bool prepare_mobys(id<MTLCommandBuffer> cmd, const RtxFrameJob& job) {
        if (job.dynamic_key == m_moby_key && (m_moby_accel != nil || job.dynamic_triangles.empty())) {
            return m_moby_accel != nil;
        }
        m_moby_key = job.dynamic_key;
        m_moby_accel = nil;
        m_moby_triangles = nil;
        const auto tris = job.dynamic_triangles;
        if (tris.empty()) {
            return false;
        }
        std::vector<float> vertices(tris.size() * 9);
        for (std::size_t i = 0; i < tris.size(); ++i) {
            const auto& t = tris[i];
            for (int a = 0; a < 3; ++a) {
                vertices[i * 9 + a] = t.v0_mat[a];
                vertices[i * 9 + 3 + a] = t.v0_mat[a] + t.e1_layer[a];
                vertices[i * 9 + 6 + a] = t.v0_mat[a] + t.e2_flags[a];
            }
        }
        id<MTLBuffer> vertex = [m_device newBufferWithBytes:vertices.data()
                                                     length:vertices.size() * sizeof(float)
                                                    options:MTLResourceStorageModeShared];
        m_moby_triangles = [m_device newBufferWithBytes:tris.data()
                                                 length:tris.size() * sizeof(RtxTriangle)
                                                options:MTLResourceStorageModeShared];
        MTLAccelerationStructureTriangleGeometryDescriptor* geom =
            [MTLAccelerationStructureTriangleGeometryDescriptor descriptor];
        geom.vertexBuffer = vertex;
        geom.vertexStride = sizeof(float) * 3;
        geom.triangleCount = tris.size();
        geom.opaque = YES;
        MTLPrimitiveAccelerationStructureDescriptor* desc =
            [MTLPrimitiveAccelerationStructureDescriptor descriptor];
        desc.geometryDescriptors = @[ geom ];
        // Rebuilt as the mobys move: built fast, traced a little slower.
        desc.usage = MTLAccelerationStructureUsagePreferFastBuild;
        const MTLAccelerationStructureSizes sizes =
            [m_device accelerationStructureSizesWithDescriptor:desc];
        id<MTLAccelerationStructure> accel =
            [m_device newAccelerationStructureWithSize:sizes.accelerationStructureSize];
        id<MTLBuffer> scratch =
            [m_device newBufferWithLength:std::max<NSUInteger>(sizes.buildScratchBufferSize, 256)
                                  options:MTLResourceStorageModePrivate];
        if (accel == nil || scratch == nil || vertex == nil || m_moby_triangles == nil) {
            m_moby_triangles = nil;
            return false;
        }
        id<MTLAccelerationStructureCommandEncoder> enc = [cmd accelerationStructureCommandEncoder];
        [enc buildAccelerationStructure:accel descriptor:desc scratchBuffer:scratch scratchBufferOffset:0];
        [enc endEncoding];
        m_moby_accel = accel;
        return true;
    }

    bool upload_emitters(std::span<const RtxTriangle> tris, std::string& error) override {
        @autoreleasepool {
            std::lock_guard lock(m_mutex);
            m_emitter_accel = nil;
            m_emitter_triangles = nil;
            if (tris.empty()) {
                return true;
            }
            std::vector<float> vertices(tris.size() * 9);
            for (std::size_t i = 0; i < tris.size(); ++i) {
                const auto& t = tris[i];
                for (int a = 0; a < 3; ++a) {
                    vertices[i * 9 + a] = t.v0_mat[a];
                    vertices[i * 9 + 3 + a] = t.v0_mat[a] + t.e1_layer[a];
                    vertices[i * 9 + 6 + a] = t.v0_mat[a] + t.e2_flags[a];
                }
            }
            id<MTLBuffer> vertex = [m_device newBufferWithBytes:vertices.data()
                                                         length:vertices.size() * sizeof(float)
                                                        options:MTLResourceStorageModeShared];
            id<MTLBuffer> triangles = [m_device newBufferWithBytes:tris.data()
                                                            length:tris.size() * sizeof(RtxTriangle)
                                                           options:MTLResourceStorageModeShared];
            MTLAccelerationStructureTriangleGeometryDescriptor* geom =
                [MTLAccelerationStructureTriangleGeometryDescriptor descriptor];
            geom.vertexBuffer = vertex;
            geom.vertexStride = sizeof(float) * 3;
            geom.triangleCount = tris.size();
            geom.opaque = YES;
            MTLPrimitiveAccelerationStructureDescriptor* desc =
                [MTLPrimitiveAccelerationStructureDescriptor descriptor];
            desc.geometryDescriptors = @[ geom ];
            const MTLAccelerationStructureSizes sizes =
                [m_device accelerationStructureSizesWithDescriptor:desc];
            id<MTLAccelerationStructure> accel =
                [m_device newAccelerationStructureWithSize:sizes.accelerationStructureSize];
            id<MTLBuffer> scratch =
                [m_device newBufferWithLength:std::max<NSUInteger>(sizes.buildScratchBufferSize, 256)
                                      options:MTLResourceStorageModePrivate];
            if (accel == nil || scratch == nil || vertex == nil || triangles == nil) {
                error = "cannot allocate the glowing surfaces' acceleration structure";
                return false;
            }
            id<MTLCommandBuffer> cmd = [m_queue commandBuffer];
            id<MTLAccelerationStructureCommandEncoder> enc = [cmd accelerationStructureCommandEncoder];
            [enc buildAccelerationStructure:accel descriptor:desc scratchBuffer:scratch scratchBufferOffset:0];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
            m_emitter_accel = accel;
            m_emitter_triangles = triangles;
            return true;
        }
    }

    std::uint64_t submit_frame(const RtxFrameJob& job, std::string& error) override {
        @autoreleasepool {
            std::lock_guard lock(m_mutex);
            if (m_accel == nil || m_triangle_buffer == nil || m_trace_frame_pso == nil
                || job.cells == nullptr || job.result == nullptr || job.width <= 0
                || job.height <= 0) {
                return 0;
            }
            id<MTLBuffer> cells = (__bridge id<MTLBuffer>)job.cells;
            id<MTLBuffer> result = (__bridge id<MTLBuffer>)job.result;
            id<MTLBuffer> history = job.history ? (__bridge id<MTLBuffer>)job.history : result;
            id<MTLCommandBuffer> cmd = [m_queue commandBuffer];
            cmd.label = @"OpenRAC ray tracing";
            const bool mobys = prepare_mobys(cmd, job);

            MetalFrameParams p{};
            std::memcpy(p.prev_view_projection, job.prev_view_projection.data(), sizeof(float) * 16);
            std::memcpy(p.camera_pos, job.camera_pos.data(), sizeof(float) * 3);
            std::memcpy(p.sun_dir, job.sun_dir.data(), sizeof(float) * 3);
            std::memcpy(p.sun_colour, job.sun_colour.data(), sizeof(float) * 3);
            std::memcpy(p.sky_colour, job.sky_colour.data(), sizeof(float) * 3);
            p.width = static_cast<std::uint32_t>(job.width);
            p.height = static_cast<std::uint32_t>(job.height);
            const auto& s = job.settings;
            p.samples = static_cast<std::uint32_t>(
                s.quality == renderer::RtxQuality::Performance ? 1
                : s.quality == renderer::RtxQuality::Balanced ? std::min(2, s.samples_per_pixel)
                                                               : std::clamp(s.samples_per_pixel, 1, 8)
            );
            p.flags = static_cast<std::uint32_t>(job.flags);
            p.frame_index = job.frame_index;
            p.has_dynamic = mobys ? 1u : 0u;
            p.history_valid = job.history != nullptr && job.history_cells != nullptr ? 1u : 0u;
            // All lights for the paths' sampling (at most 512); the first 64 for the visibility.
            const std::size_t light_total = std::min<std::size_t>(job.lights.size(), 512);
            p.light_count = static_cast<std::uint32_t>(light_total);
            p.local_count = static_cast<std::uint32_t>(std::min<std::size_t>(light_total, 64));
            p.has_emitters = m_emitter_accel != nil ? 1u : 0u;
            p.ray_distance = s.ray_distance;
            p.ao_radius = s.ao_radius;
            p.shadow_softness = s.shadow_softness;
            p.gi_strength = s.gi_strength;
            p.roughness_cutoff = s.reflection_roughness_cutoff;
            // A twelfth of each new trace once a cell has gathered that many frames (the kernel
            // counts them, and takes a fresh cell's rays whole): at a sixth the one or two paths
            // a cell still flickered where it stood.
            p.history_blend = job.history != nullptr ? 0.08f : 1.0f;

            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:m_trace_frame_pso];
            [enc setAccelerationStructure:m_accel atBufferIndex:0];
            [enc setBuffer:m_triangle_buffer offset:0 atIndex:1];
            [enc setAccelerationStructure:(mobys ? m_moby_accel : m_accel) atBufferIndex:2];
            [enc setBuffer:(mobys ? m_moby_triangles : m_triangle_buffer) offset:0 atIndex:3];
            [enc setBuffer:cells offset:job.cells_offset atIndex:4];
            [enc setBytes:&p length:sizeof p atIndex:5];
            [enc setBuffer:result offset:job.result_offset atIndex:6];
            [enc setBuffer:history offset:(job.history ? job.history_offset : job.result_offset) atIndex:7];
            // The local lights, a buffer of their own this frame (one zeroed light without any).
            static const RtxLightRecord kNone{};
            id<MTLBuffer> light_buffer =
                [m_device newBufferWithBytes:(light_total != 0 ? static_cast<const void*>(job.lights.data()) : &kNone)
                                      length:(light_total != 0 ? light_total * sizeof(RtxLightRecord) : sizeof kNone)
                                     options:MTLResourceStorageModeShared];
            [enc setBuffer:light_buffer offset:0 atIndex:8];
            // The glowing surfaces' structure, or the level's in its place (has_emitters 0).
            [enc setAccelerationStructure:(m_emitter_accel ? m_emitter_accel : m_accel) atBufferIndex:10];
            [enc setBuffer:(m_emitter_accel ? m_emitter_triangles : m_triangle_buffer) offset:0 atIndex:11];
            if (mobys) {
                [enc useResource:m_moby_accel usage:MTLResourceUsageRead];
            }
            if (m_emitter_accel) {
                [enc useResource:m_emitter_accel usage:MTLResourceUsageRead];
            }
            [enc setBuffer:(job.history_cells ? (__bridge id<MTLBuffer>)job.history_cells : cells)
                    offset:(job.history_cells ? job.history_cells_offset : job.cells_offset)
                   atIndex:9];
            [enc dispatchThreadgroups:MTLSizeMake((p.width + 7) / 8, (p.height + 7) / 8, 1)
                threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
            [enc endEncoding];
            [cmd commit];
            const std::uint64_t number = m_next_job++;
            m_jobs[number % m_jobs.size()] = {number, cmd};
            (void)error;
            return number;
        }
    }

    bool finished(std::uint64_t job) const override {
        const auto& slot = m_jobs[job % m_jobs.size()];
        if (slot.number != job || slot.commands == nil) {
            return true;  // long gone: done
        }
        const MTLCommandBufferStatus status = slot.commands.status;
        return status == MTLCommandBufferStatusCompleted || status == MTLCommandBufferStatusError;
    }

private:
    struct Job {
        std::uint64_t number = 0;
        id<MTLCommandBuffer> commands = nil;
    };
    std::array<Job, 8> m_jobs{};
    std::uint64_t m_next_job = 1;
    id<MTLComputePipelineState> m_trace_frame_pso = nil;
    id<MTLAccelerationStructure> m_moby_accel = nil;
    id<MTLBuffer> m_moby_triangles = nil;
    id<MTLAccelerationStructure> m_emitter_accel = nil;  // the level's glowing surfaces
    id<MTLBuffer> m_emitter_triangles = nil;
    std::uint64_t m_moby_key = 0;
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

namespace {

std::unique_ptr<RtxHardwareBackend> make_metal_backend(id<MTLDevice> device) {
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
        NSLog(@"OpenRAC Metal RT: %@", err.localizedDescription);
        return nullptr;
    }
    id<MTLFunction> rays_fn = [lib newFunctionWithName:@"rtx_trace_rays_kernel"];
    id<MTLFunction> screen_fn = [lib newFunctionWithName:@"rtx_trace_screen_kernel"];
    id<MTLFunction> frame_fn = [lib newFunctionWithName:@"rtx_trace_frame_kernel"];
    if (!rays_fn || !screen_fn || !frame_fn) {
        return nullptr;
    }
    id<MTLComputePipelineState> rays_pso = [device newComputePipelineStateWithFunction:rays_fn error:&err];
    id<MTLComputePipelineState> screen_pso =
        [device newComputePipelineStateWithFunction:screen_fn error:&err];
    id<MTLComputePipelineState> frame_pso =
        [device newComputePipelineStateWithFunction:frame_fn error:&err];
    if (!rays_pso || !screen_pso || !frame_pso) {
        return nullptr;
    }
    return std::make_unique<MetalRtxBackend>(device, queue, rays_pso, screen_pso, frame_pso);
}

}  // namespace

std::unique_ptr<RtxHardwareBackend> create_metal_rtx_backend() {
    @autoreleasepool {
        return make_metal_backend(MTLCreateSystemDefaultDevice());
    }
}

std::unique_ptr<RtxHardwareBackend> create_metal_rtx_backend_on(void* device) {
    @autoreleasepool {
        return make_metal_backend((__bridge id<MTLDevice>)device);
    }
}

}  // namespace openrac::enhanced
#endif
