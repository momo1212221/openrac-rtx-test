// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 the OpenRAC contributors
//
// Hardware verification & benchmark suite for the native RTX ray-tracing pipeline:
// 1. LevelData geometry extraction (Terrain, Ties, Shrubs, Moby meshes, material albedo sampling)
// 2. SAH BVH construction & 4,096-ray CPU-vs-GPU hardware equivalence (Apple Metal / NVIDIA CUDA)
// 3. Hardware screen-space RTX pass (soft shadows, cosine-hemisphere AO, 1-bounce GI, reflections)
// 4. Live GPU ray-traced shadow/lighting map & throughput telemetry

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "enhanced/renderer/rtx.h"
#include "tests/check.h"

using namespace openrac;

namespace {

std::uint32_t pack_rgb8_test(float r, float g, float b) {
    const auto cr = static_cast<std::uint32_t>(std::clamp(r, 0.0f, 1.0f) * 255.0f + 0.5f);
    const auto cg = static_cast<std::uint32_t>(std::clamp(g, 0.0f, 1.0f) * 255.0f + 0.5f);
    const auto cb = static_cast<std::uint32_t>(std::clamp(b, 0.0f, 1.0f) * 255.0f + 0.5f);
    return cr | (cg << 8u) | (cb << 16u) | 0xFF000000u;
}

float uint_bits_test(std::uint32_t u) {
    float f = 0.0f;
    std::memcpy(&f, &u, sizeof(float));
    return f;
}

enhanced::RtxTriangle make_triangle(
    std::array<float, 3> v0,
    std::array<float, 3> v1,
    std::array<float, 3> v2,
    std::array<float, 3> normal,
    std::uint32_t albedo,
    float roughness = 0.1f
) {
    enhanced::RtxTriangle t{};
    t.v0_mat = {v0[0], v0[1], v0[2], 0.0f};
    t.e1_layer = {v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2], 0.0f};
    t.e2_flags = {v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2], roughness};
    t.normal_albedo = {normal[0], normal[1], normal[2], uint_bits_test(albedo)};
    return t;
}

void check_level_data_extraction() {
    scene::LevelData level{};
    // Create a 2x2 texture image (bright cyan-blue)
    renderer::Rgba8Image img{};
    img.width = 2;
    img.height = 2;
    img.pixels = {
        40, 180, 255, 255, 40, 180, 255, 255,
        40, 180, 255, 255, 40, 180, 255, 255,
    };
    level.images.push_back(img);

    // Material 0: opaque metallic surface using texture 0
    scene::Material mat_opaque{};
    mat_opaque.image = 0;
    mat_opaque.metal = true;
    mat_opaque.blend = false;
    level.materials.push_back(mat_opaque);

    // Material 1: transparent/blended surface (must be ignored by RTX blocker extraction)
    scene::Material mat_blend{};
    mat_blend.image = 0;
    mat_blend.blend = true;
    level.materials.push_back(mat_blend);

    // 4 vertices forming 2 triangles in a quad
    for (int y = 0; y < 2; ++y) {
        for (int x = 0; x < 2; ++x) {
            scene::Vertex v{};
            v.position[0] = float(x) * 4.0f;
            v.position[1] = float(y) * 4.0f;
            v.position[2] = 0.0f;
            v.normal[2] = 1.0f;
            v.colour[0] = v.colour[1] = v.colour[2] = v.colour[3] = 1.0f;
            level.vertices.push_back(v);
        }
    }
    level.indices = {0, 1, 2, 1, 3, 2};

    scene::Model model{};
    model.primitives.push_back({0, 6, 0});  // 2 opaque triangles
    model.primitives.push_back({0, 6, 1});  // 2 blended triangles (skipped)
    level.models.push_back(model);

    scene::Instance terrain_inst{};
    terrain_inst.model = 0;
    terrain_inst.matrix = renderer::identity();
    level.instances[static_cast<std::size_t>(scene::Layer::Terrain)].push_back(terrain_inst);

    scene::Instance tie_inst{};
    tie_inst.model = 0;
    tie_inst.matrix = renderer::translation({10.0f, 0.0f, 0.0f});
    level.instances[static_cast<std::size_t>(scene::Layer::Ties)].push_back(tie_inst);

    enhanced::RtxPipeline pipeline;
    pipeline.upload(level);

    // 2 opaque triangles from Terrain + 2 opaque triangles from Ties = 4 static triangles total
    CHECK(pipeline.triangles().size() == 4);
    CHECK(!pipeline.bvh_nodes().empty());

    // Place a dynamic moby overhead at z = 3.0 (x in 0..4, y in 0..4)
    scene::Instance overhead_moby{};
    overhead_moby.model = 0;
    overhead_moby.matrix = renderer::translation({0.0f, 0.0f, 3.0f});

    // Before dynamic moby injection: upward ray from (1, 1, 0.5) misses overhead
    enhanced::RtxRay upward_ray{{1.0f, 1.0f, 0.5f}, 0.01f, {0.0f, 0.0f, 1.0f}, 100.0f};
    enhanced::RtxHit upward_hit{};
    CHECK(pipeline.trace_rays(std::span(&upward_ray, 1), std::span(&upward_hit, 1), renderer::RtxBackend::VulkanRayQuery));
    CHECK(upward_hit.t < 0.0f);

    // Inject 3 dynamic moby instances (6 dynamic triangles -> 10 total triangles in combined 2-level BVH)
    const std::array<scene::Instance, 3> dynamic_mobys{{overhead_moby, tie_inst, terrain_inst}};
    pipeline.prepare_instances(scene::Layer::Mobys, dynamic_mobys);

    const auto st = enhanced::rtx_status();
    CHECK(st.triangle_count == 10);
    CHECK(st.node_count == pipeline.bvh_nodes().size());
    CHECK(st.dynamic_instance_count == 3);

    // Now the upward ray hits the dynamic moby at z = 3.0 (t = 2.5)
    CHECK(pipeline.trace_rays(std::span(&upward_ray, 1), std::span(&upward_hit, 1), renderer::RtxBackend::VulkanRayQuery));
    CHECK(std::fabs(upward_hit.t - 2.5f) < 1e-4f);

    // Clearing dynamic mobys restores the static 4-triangle BVH without rebuilding static nodes
    pipeline.prepare_instances(scene::Layer::Mobys, {});
    CHECK(pipeline.triangles().size() == 4);

    std::printf("[Test 1] LevelData & Dynamic Moby 2-Level BVH verified: 4 static + 6 dynamic = %zu tris, "
                "%zu BVH nodes, dynamic hit t=%.2f.\n",
                st.triangle_count,
                st.node_count,
                upward_hit.t);
}

void verify_hardware_backend(
    enhanced::RtxHardwareBackend& hw,
    enhanced::RtxPipeline& pipeline,
    std::uint32_t floor_albedo,
    std::uint32_t blocker_albedo,
    std::uint32_t wall_albedo
) {
    std::string error;
    const auto t_upload_start = std::chrono::steady_clock::now();
    CHECK(hw.upload_scene(pipeline.triangles(), pipeline.bvh_nodes(), error));
    const auto t_upload_end = std::chrono::steady_clock::now();
    const double upload_ms =
        std::chrono::duration<double, std::milli>(t_upload_end - t_upload_start).count();

    std::printf("\n====================================================================\n");
    std::printf("HARDWARE RTX BACKEND PROOF: %.*s\n",
                int(hw.device_name().size()),
                hw.device_name().data());
    std::printf("Acceleration Structure Build Time: %.3f ms (%zu triangles, %zu BVH nodes)\n",
                upload_ms,
                pipeline.triangles().size(),
                pipeline.bvh_nodes().size());
    std::printf("====================================================================\n");

    // 1. 4,096-Ray (64x64) Hardware vs. CPU SAH-BVH Exact Equivalence Test
    constexpr int kGrid = 64;
    constexpr std::size_t kRayCount = static_cast<std::size_t>(kGrid * kGrid);
    std::vector<enhanced::RtxRay> rays(kRayCount);
    for (int y = 0; y < kGrid; ++y) {
        for (int x = 0; x < kGrid; ++x) {
            const float wx = (float(x) / float(kGrid - 1)) * 12.0f - 6.0f;
            const float wy = (float(y) / float(kGrid - 1)) * 12.0f - 6.0f;
            auto& r = rays[static_cast<std::size_t>(y * kGrid + x)];
            r.origin = {wx, wy, 5.0f};
            r.t_min = 0.001f;
            r.direction = {0.0f, 0.0f, -1.0f};
            r.t_max = 100.0f;
        }
    }

    std::vector<enhanced::RtxHit> cpu_hits(kRayCount);
    std::vector<enhanced::RtxHit> gpu_hits(kRayCount);
    CHECK(pipeline.trace_rays(rays, cpu_hits, renderer::RtxBackend::VulkanRayQuery));

    const auto t_trace_start = std::chrono::steady_clock::now();
    CHECK(hw.trace_rays(rays, gpu_hits, error));
    const auto t_trace_end = std::chrono::steady_clock::now();
    const double trace_ms =
        std::chrono::duration<double, std::milli>(t_trace_end - t_trace_start).count();
    const double mrays_per_sec = (double(kRayCount) / 1e6) / (trace_ms / 1e3);

    std::size_t blocker_hits = 0;
    std::size_t floor_hits = 0;
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < kRayCount; ++i) {
        if (std::fabs(cpu_hits[i].t - gpu_hits[i].t) > 1e-3f
            || cpu_hits[i].packed_albedo != gpu_hits[i].packed_albedo) {
            ++mismatches;
        }
        if (gpu_hits[i].packed_albedo == blocker_albedo) {
            ++blocker_hits;
        } else if (gpu_hits[i].packed_albedo == floor_albedo) {
            ++floor_hits;
        }
    }
    CHECK(mismatches == 0);
    CHECK(blocker_hits > 100);
    CHECK(floor_hits > 3000);
    std::printf("[Test 2] 4,096 Hardware Closest-Hit Rays verified on GPU:\n"
                "         - GPU vs. CPU BVH mismatches : %zu / %zu (0.00%% error)\n"
                "         - Overhead Blocker hits      : %zu rays (t = 3.000)\n"
                "         - Floor Receiver hits        : %zu rays (t = 5.000)\n"
                "         - Hardware Ray Dispatch Time : %.3f ms (%.2f MRays/s)\n",
                mismatches,
                kRayCount,
                blocker_hits,
                floor_hits,
                trace_ms,
                mrays_per_sec);

    // 2. Hardware Screen-Space RTX Pass (Soft Shadows, AO, 1-Bounce GI, Specular Reflections)
    // Render a 32x16 grid of floor points (-4..+4 in x, -4..+4 in y) on the hardware backend.
    constexpr std::uint32_t kScreenW = 32;
    constexpr std::uint32_t kScreenH = 16;
    constexpr std::size_t kCells = static_cast<std::size_t>(kScreenW * kScreenH);
    std::vector<std::array<float, 4>> positions(kCells);
    std::vector<std::array<float, 4>> normals(kCells);
    for (std::uint32_t y = 0; y < kScreenH; ++y) {
        for (std::uint32_t x = 0; x < kScreenW; ++x) {
            const float wx = (float(x) / float(kScreenW - 1)) * 8.0f - 4.0f;
            const float wy = (float(y) / float(kScreenH - 1)) * 8.0f - 4.0f;
            const std::size_t idx = static_cast<std::size_t>(y * kScreenW + x);
            positions[idx] = {wx, wy, 0.0f, 1.0f};
            normals[idx] = {0.0f, 0.0f, 1.0f, 0.05f};  // mirror-like roughness 0.05
        }
    }

    std::vector<std::uint32_t> out_pairs(kCells * 2, 0u);
    renderer::RtxSettings settings{};
    settings.shadows = true;
    settings.ambient_occlusion = true;
    settings.reflections = true;
    settings.global_illumination = true;
    settings.samples_per_pixel = 8;
    settings.ray_distance = 100.0f;
    settings.ao_radius = 3.5f;
    settings.shadow_softness = 0.04f;
    settings.gi_strength = 1.5f;
    settings.reflection_roughness_cutoff = 0.65f;

    const auto t_screen_start = std::chrono::steady_clock::now();
    CHECK(hw.trace_screen(
        kScreenW,
        kScreenH,
        positions,
        normals,
        {0.0f, 0.0f, 5.0f},    // camera above floor looking down so V=(0,0,-1) reflects to R=(0,0,+1)
        {0.0f, 0.0f, -1.0f},   // sun straight overhead
        {1.0f, 1.0f, 1.0f},    // white sun
        {0.2f, 0.3f, 0.5f},    // blue sky
        settings,
        1,
        out_pairs,
        error
    ));
    const auto t_screen_end = std::chrono::steady_clock::now();
    const double screen_ms =
        std::chrono::duration<double, std::milli>(t_screen_end - t_screen_start).count();

    // Centre cell (16, 8) is at (~0.13, ~0.27, 0) directly under the red blocker at z = 2 (-1.5..1.5):
    // - Shadow visibility must be 0 (occluded by blocker)
    // - Reflection must hit the red blocker overhead (red channel > green/blue channels, high confidence)
    const std::size_t center_cell = static_cast<std::size_t>(8 * kScreenW + 16);
    const std::size_t corner_lit_cell = static_cast<std::size_t>(1 * kScreenW + 30);
    const std::size_t wall_adjacent_cell = static_cast<std::size_t>(8 * kScreenW + 0);

    const std::uint32_t center_val = out_pairs[center_cell * 2 + 0];
    const std::uint32_t center_refl = out_pairs[center_cell * 2 + 1];
    const std::uint32_t lit_val = out_pairs[corner_lit_cell * 2 + 0];
    const std::uint32_t wall_val = out_pairs[wall_adjacent_cell * 2 + 0];

    const std::uint32_t vis_center = (center_val >> 24u) & 0xFFu;
    const std::uint32_t vis_lit = (lit_val >> 24u) & 0xFFu;
    const std::uint32_t vis_wall = (wall_val >> 24u) & 0xFFu;

    const std::uint32_t refl_r = center_refl & 0xFFu;
    const std::uint32_t refl_g = (center_refl >> 8u) & 0xFFu;
    const std::uint32_t refl_a = (center_refl >> 24u) & 0xFFu;

    const std::uint32_t wall_gi_r = wall_val & 0xFFu;
    (void)wall_albedo;

    CHECK(vis_center == 0u);
    CHECK(vis_lit > 200u);
    // Wall at x = -4.2 occludes half the hemisphere at x = -4.0, so AO reduces visibility compared to open corner!
    CHECK(vis_wall < vis_lit);
    // Reflection at center hits the bright red overhead blocker (R > G and confidence alpha > 180)
    CHECK(refl_r > refl_g);
    CHECK(refl_a > 180u);
    // 1-bounce GI near the lit wall/sky produces non-zero encoded GI radiance
    CHECK(wall_gi_r > 0u);

    std::printf("[Test 3] Hardware Screen-Space RTX Shading (%ux%u cells, %d spp, %.3f ms):\n"
                "         - Umbra Shadow Visibility (under blocker) : %u / 255 (expected 0)\n"
                "         - Open Floor Visibility (unoccluded)      : %u / 255 (expected > 200)\n"
                "         - Wall-Corner AO Visibility (occluded AO) : %u / 255 (expected < %u)\n"
                "         - Specular Reflection RGBA8 (red blocker) : R=%u G=%u B=%u Conf=%u/255\n"
                "         - 1-Bounce GI Encoded Radiance (near wall): R=%u G=%u B=%u\n",
                kScreenW,
                kScreenH,
                settings.samples_per_pixel,
                screen_ms,
                vis_center,
                vis_lit,
                vis_wall,
                vis_lit,
                refl_r,
                refl_g,
                (center_refl >> 16u) & 0xFFu,
                refl_a,
                wall_gi_r,
                (wall_val >> 8u) & 0xFFu,
                (wall_val >> 16u) & 0xFFu);

    std::printf("\n[Test 4] Visual Hardware Ray-Traced Shadow + AO Map (32x16 from GPU memory):\n");
    std::printf("         Legend: '#' = Full Sun+Sky, '+' = AO darkened near wall, '.' = Soft Penumbra, ' ' = Umbra Shadow\n");
    for (std::uint32_t y = 0; y < kScreenH; ++y) {
        std::printf("         |");
        for (std::uint32_t x = 0; x < kScreenW; ++x) {
            const std::uint32_t v =
                (out_pairs[static_cast<std::size_t>(y * kScreenW + x) * 2 + 0] >> 24u) & 0xFFu;
            const char ch = (v == 0u) ? ' ' : (v < 140u) ? '.' : (v < 225u) ? '+' : '#';
            std::putchar(ch);
        }
        std::printf("|\n");
    }
    std::printf("====================================================================\n\n");
}

void check_bvh_and_hardware_backends() {
    // Build a 3D test scene:
    // 1. Floor quad at z = 0 (-10..10 in x, y)
    // 2. Overhead red blocker quad at z = 2 (-1.5..1.5 in x, y), facing both down and up for reflections/shadows
    // 3. Vertical gold-orange wall at x = -4.2 (-6..6 in y, 0..4 in z) for AO and GI colour bleeding
    const std::uint32_t floor_albedo = pack_rgb8_test(0.35f, 0.75f, 0.45f);
    const std::uint32_t blocker_albedo = pack_rgb8_test(0.95f, 0.15f, 0.10f);
    const std::uint32_t wall_albedo = pack_rgb8_test(0.95f, 0.65f, 0.15f);

    std::vector<enhanced::RtxTriangle> tris = {
        // Floor (z = 0)
        make_triangle({-10, -10, 0}, {10, -10, 0}, {10, 10, 0}, {0, 0, 1}, floor_albedo),
        make_triangle({-10, -10, 0}, {10, 10, 0}, {-10, 10, 0}, {0, 0, 1}, floor_albedo),
        // Overhead blocker (z = 2, x in -1.5..1.5, y in -1.5..1.5)
        make_triangle({-1.5f, -1.5f, 2}, {1.5f, -1.5f, 2}, {1.5f, 1.5f, 2}, {0, 0, -1}, blocker_albedo),
        make_triangle({-1.5f, -1.5f, 2}, {1.5f, 1.5f, 2}, {-1.5f, 1.5f, 2}, {0, 0, -1}, blocker_albedo),
        // Vertical wall at x = -4.2 (z in 0..4, y in -6..6)
        make_triangle({-4.2f, -6, 0}, {-4.2f, 6, 0}, {-4.2f, 6, 4}, {1, 0, 0}, wall_albedo),
        make_triangle({-4.2f, -6, 0}, {-4.2f, 6, 4}, {-4.2f, -6, 4}, {1, 0, 0}, wall_albedo),
    };

    enhanced::RtxPipeline pipeline;
    std::string error;
    CHECK(pipeline.upload_triangles(tris, error));
    CHECK(error.empty());
    CHECK(pipeline.triangles().size() == 6);
    CHECK(!pipeline.bvh_nodes().empty());

    // Verify Apple Metal Hardware Ray Tracing on macOS when supported.
    if (enhanced::metal_rtx_supported()) {
        auto metal = enhanced::create_metal_rtx_backend();
        CHECK(metal != nullptr);
        verify_hardware_backend(*metal, pipeline, floor_albedo, blocker_albedo, wall_albedo);
    }

    // Verify NVIDIA CUDA RTX backend when an NVIDIA GPU + driver is present.
    if (enhanced::cuda_rtx_supported()) {
        auto cuda = enhanced::create_cuda_rtx_backend();
        CHECK(cuda != nullptr);
        verify_hardware_backend(*cuda, pipeline, floor_albedo, blocker_albedo, wall_albedo);
    }
}

}  // namespace

int main() {
    check_level_data_extraction();
    check_bvh_and_hardware_backends();
    return test::result();
}
