// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 the OpenRAC contributors
//
// Native hardware ray-tracing pipeline (RTX) for Enhanced Graphics:
// - Apple Silicon native Metal Ray Tracing (MTLAccelerationStructure + MSL raytracing::intersector)
// - NVIDIA native CUDA / RTX Ray Tracing (CUDA Driver API + NVRTC PTX BVH kernel)
// - Vulkan Compute / Ray-Query graph pass (enhanced::run_compute_graph + rtx_trace.comp)

#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "renderer/renderer.h"
#include "renderer/rtx_settings.h"
#include "renderer/shader.h"
#include "scene/level.h"

namespace openrac::enhanced {

// Diagnostic view modes for the Enhanced Settings -> Diagnostics / RTX inspector.
enum class RtxDebugView {
    Off,
    Shadows,
    AmbientOcclusion,
    Reflections,
    GlobalIllumination,
    BvhHeatmap,
    Count
};

// 64-byte aligned triangle record shared across Vulkan SSBO, CUDA device memory, and Metal buffers.
struct alignas(16) RtxTriangle {
    std::array<float, 4> v0_mat{};       // xyz: vertex 0 in world space, w: bitcast material index
    std::array<float, 4> e1_layer{};     // xyz: edge 1 (v1 - v0), w: bitcast scene::Layer
    std::array<float, 4> e2_flags{};     // xyz: edge 2 (v2 - v0), w: roughness / flags
    std::array<float, 4> normal_albedo{};  // xyz: unit geometric normal, w: packed RGB8 albedo
};
static_assert(sizeof(RtxTriangle) == 64);

// 32-byte aligned flattened SAH BVH node for GPU stackless/short-stack traversal.
struct alignas(16) RtxBvhNode {
    std::array<float, 3> bbox_min{};
    std::int32_t left_or_first = 0;  // >= 0: left child node index; < 0: ~(first triangle index)
    std::array<float, 3> bbox_max{};
    std::int32_t count = 0;          // 0: internal node (right child is left_or_first + 1); > 0: leaf triangle count
};
static_assert(sizeof(RtxBvhNode) == 32);

// Hardware ray query input for batch tracing (baker transport, tests, and hybrid passes).
struct alignas(16) RtxRay {
    std::array<float, 3> origin{};
    float t_min = 0.001f;
    std::array<float, 3> direction{};
    float t_max = 1000.0f;
};
static_assert(sizeof(RtxRay) == 32);

// Hardware ray intersection result.
struct alignas(16) RtxHit {
    float t = -1.0f;                     // < 0 on miss, >= t_min on hit
    std::int32_t triangle_index = -1;
    float u = 0.0f;
    float v = 0.0f;
    std::array<float, 3> normal{};
    std::uint32_t packed_albedo = 0;
};
static_assert(sizeof(RtxHit) == 32);

// Runtime status snapshot for UI diagnostics and logging.
struct RtxStatus {
    bool available = false;
    bool active = false;
    renderer::RtxBackend active_backend = renderer::RtxBackend::Auto;
    std::string device_name;
    std::size_t triangle_count = 0;
    std::size_t node_count = 0;
    std::size_t dynamic_instance_count = 0;
    bool metal_supported = false;
    bool cuda_supported = false;
    bool vulkan_ray_query_supported = false;
};

// One frame's tracing on the GPU, with no copy through the CPU: the cells' positions and normals
// (rtx_gbuffer.comp's two vec4 a cell) and the result (rtx_trace.comp's two RGBA8 words a cell)
// are the renderer's own buffers, given as the backend's native objects. Metal: id<MTLBuffer>.
struct RtxFrameJob {
    void* cells = nullptr;
    std::size_t cells_offset = 0;
    void* result = nullptr;
    std::size_t result_offset = 0;
    void* history = nullptr;  // the previous job's result, blended in through its camera
    std::size_t history_offset = 0;
    std::array<float, 16> prev_view_projection{};
    int width = 0, height = 0;  // cells
    std::array<float, 3> camera_pos{}, sun_dir{}, sun_colour{}, sky_colour{};
    renderer::RtxSettings settings;
    int flags = 0;  // RtxPipeline::effect_flags
    std::uint32_t frame_index = 0;
    // The mobys' world-space triangles, rebuilt into their own structure when `dynamic_key` changes.
    std::span<const RtxTriangle> dynamic_triangles;
    std::uint64_t dynamic_key = 0;
};

// Native hardware backend interface (implemented by Metal on macOS and CUDA on NVIDIA).
class RtxHardwareBackend {
public:
    // Queues `job` on the GPU without waiting; its number for finished(), 0 when it cannot.
    virtual std::uint64_t submit_frame(const RtxFrameJob& /*job*/, std::string& /*error*/) {
        return 0;
    }
    // Whether the job numbered `job` is done (its result may be read); never blocks.
    virtual bool finished(std::uint64_t /*job*/) const { return true; }
    // The native device it runs on (Metal: id<MTLDevice>), to match the renderer's.
    virtual void* native_device() const { return nullptr; }

    virtual ~RtxHardwareBackend() = default;
    virtual renderer::RtxBackend kind() const = 0;
    virtual std::string_view device_name() const = 0;
    virtual bool upload_scene(
        std::span<const RtxTriangle> triangles,
        std::span<const RtxBvhNode> nodes,
        std::string& error
    ) = 0;
    virtual bool trace_rays(
        std::span<const RtxRay> rays,
        std::span<RtxHit> hits,
        std::string& error
    ) = 0;
    // Traces a half-resolution screen grid of primary world positions + normals and writes
    // 2 RGBA8 words per cell: word 0 = (GI.rgb, shadow_ao), word 1 = (reflection.rgb, confidence).
    virtual bool trace_screen(
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
    ) = 0;
};

// Factory functions for the native Apple Metal and NVIDIA CUDA backends.
std::unique_ptr<RtxHardwareBackend> create_metal_rtx_backend();
// The Metal backend on a given id<MTLDevice> (the renderer's, vkgl::metal_device()).
std::unique_ptr<RtxHardwareBackend> create_metal_rtx_backend_on(void* device);
std::unique_ptr<RtxHardwareBackend> create_cuda_rtx_backend();
bool metal_rtx_supported();
bool cuda_rtx_supported();

// Session-wide RTX status and debug view controls for the Enhanced Settings window.
RtxStatus rtx_status();
RtxDebugView rtx_debug_view();
void set_rtx_debug_view(RtxDebugView view);

// Per-level hardware ray-tracing pipeline owned by lighting::SceneLighting.
class RtxPipeline {
public:
    RtxPipeline();
    ~RtxPipeline();
    RtxPipeline(const RtxPipeline&) = delete;
    RtxPipeline& operator=(const RtxPipeline&) = delete;

    // Extracts static level geometry (terrain, ties, shrubs) and moby meshes, builds the SAH BVH,
    // and uploads acceleration structures to the active hardware backends.
    void upload(const scene::LevelData& level);

    // Uploads an explicit triangle list (used by unit/GPU tests and custom scenes) and builds the BVH.
    bool upload_triangles(std::span<const RtxTriangle> triangles, std::string& error);

    // Captures live moby instances for dynamic TLAS/BVH refit.
    void prepare_instances(scene::Layer layer, std::span<const scene::Instance> instances);

    // Executes the per-frame RTX ray-tracing pass before EarlyLighting.
    bool render(
        unsigned depth,
        const renderer::FrameInput& input,
        const renderer::RenderTarget& target,
        const renderer::Vec3& sun_direction,
        const renderer::Vec3& sun_colour,
        const renderer::Vec3& sky_colour,
        std::string& error
    );

    // Binds the ray-traced buffer (unit 20) to early_lighting.comp so cached GI and sun shadows
    // incorporate hardware ray-traced shadows, AO, and 1-bounce GI.
    void bind_for_early_lighting(
        const renderer::Shader& shader, const renderer::FrameInput& input
    ) const;

    // Binds ray-traced reflections and debug visualization uniforms to mesh.frag.
    void bind_for_mesh(
        const renderer::Shader& shader,
        const renderer::FrameInput& input,
        const renderer::RenderState& state
    ) const;

    // Traces arbitrary ray batches using the selected hardware backend (Metal / CUDA / CPU SAH BVH).
    bool trace_rays(
        std::span<const RtxRay> rays,
        std::span<RtxHit> hits,
        renderer::RtxBackend preferred = renderer::RtxBackend::Auto
    );

    bool active(const renderer::FrameInput& input, const renderer::RenderState& state) const;
    // Whether a verified probe bake lights this level (its bounce then needs no rays).
    void set_probe_coverage(bool covers) { m_probes_cover = covers; }
    int effect_flags(const renderer::RtxSettings& settings) const;
    unsigned output_buffer() const { return m_output_buffer; }
    unsigned output_texture() const { return m_output_texture; }
    // The static tree; the mobys' is apart (dynamic_*).
    std::span<const RtxTriangle> triangles() const { return m_triangles; }
    std::span<const RtxBvhNode> bvh_nodes() const { return m_nodes; }
    std::span<const RtxTriangle> dynamic_triangles() const { return m_dynamic_triangles; }
    std::span<const RtxBvhNode> dynamic_nodes() const { return m_dynamic_nodes; }
    void release();

private:
    void build_bvh();
    void ensure_gpu_buffers();
    // Metal's own ray tracing on the renderer's device (Vulkan through MoltenVK): whether it runs
    // this frame, and its frame (render's work when it does).
    bool metal_frames(const renderer::FrameInput& input);
    bool render_metal(
        unsigned depth,
        const renderer::FrameInput& input,
        const renderer::RenderTarget& target,
        const renderer::Vec3& sun_direction,
        const renderer::Vec3& sun_colour,
        const renderer::Vec3& sky_colour,
        std::string& error
    );
    void release_metal_slots();

    // One frame's cells and result on their way through Metal: Free; Written (the Vulkan frame
    // that fills `cells` is queued); Submitted (Metal traces them); Done (`result` is readable).
    struct MetalSlot {
        enum class State { Free, Written, Submitted, Done } state = State::Free;
        unsigned cells = 0, result = 0, result_texture = 0;
        int width = 0, height = 0;
        std::uint64_t vk_frame = 0, job = 0, serial = 0;
        renderer::Mat4 view_projection{};
        RtxFrameJob params;  // all but the buffers, taken when the slot is written
        std::vector<RtxTriangle> mobys;
    };
    std::array<MetalSlot, 3> m_slots{};
    int m_next_slot = 0, m_latest_slot = -1, m_last_submitted = -1;
    std::uint64_t m_slot_serial = 0;
    renderer::Shader m_gbuffer_compute;
    bool m_gbuffer_failed = false;
    // Early lighting reads the result through the camera it was traced from (render_metal), or
    // this frame's own cells (the compute pass).
    bool m_reproject = false;
    unsigned m_metal_texture = 0;  // the newest finished slot's result (not owned)
    renderer::Mat4 m_source_view_projection{};
    std::array<int, 2> m_source_cells{};
    RtxHardwareBackend* select_hardware_backend(renderer::RtxBackend requested) const;

    void clear_dynamic();
    void publish_counts();

    // The level's static tree (terrain, ties, shrubs), built and uploaded once a level.
    std::vector<RtxTriangle> m_triangles;
    std::vector<RtxBvhNode> m_nodes;
    // The mobys' tree, rebuilt when one moves (m_dynamic_key: their models and matrices).
    std::vector<RtxTriangle> m_dynamic_triangles;
    std::vector<RtxBvhNode> m_dynamic_nodes;
    std::uint64_t m_dynamic_key = 0;
    struct MobyMeshSlice {
        std::vector<RtxTriangle> local_triangles;
    };
    std::vector<MobyMeshSlice> m_moby_meshes;
    std::size_t m_dynamic_instance_count = 0;

    std::unique_ptr<RtxHardwareBackend> m_metal_backend;
    std::unique_ptr<RtxHardwareBackend> m_cuda_backend;

    renderer::Shader m_compute;
    unsigned m_bvh_buffer = 0;
    unsigned m_bvh_texture = 0;
    unsigned m_tri_buffer = 0;
    unsigned m_tri_texture = 0;
    unsigned m_dyn_bvh_buffer = 0;
    unsigned m_dyn_bvh_texture = 0;
    unsigned m_dyn_tri_buffer = 0;
    unsigned m_dyn_tri_texture = 0;
    unsigned m_output_buffer = 0;   // this frame's result
    unsigned m_output_texture = 0;
    unsigned m_history_buffer = 0;  // the previous frame's, swapped each frame
    unsigned m_history_texture = 0;
    bool m_history_valid = false;
    renderer::Mat4 m_prev_view{};
    renderer::Mat4 m_prev_projection{};
    std::array<float, 3> m_prev_camera{};
    bool m_static_dirty = false;
    bool m_dynamic_dirty = false;
    bool m_compute_failed = false;

    int m_width = 0;
    int m_height = 0;
    unsigned m_target = 0;
    std::uint64_t m_frame = 0;
    bool m_ready = false;
    bool m_probes_cover = false;
    const char* m_logged = nullptr;  // the last state logged (render's reasons), once each
};

}  // namespace openrac::enhanced
