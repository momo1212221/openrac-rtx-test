// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 the OpenRAC contributors
//
// Native NVIDIA CUDA / RTX ray-tracing backend.
// Dynamically loads the NVIDIA CUDA Driver API (libcuda / nvcuda.dll) and NVRTC runtime compiler
// at runtime so OpenRAC builds on any host without a hard build-time CUDA Toolkit dependency, while
// executing native GPU BVH ray-tracing kernels on NVIDIA GPUs.

#include "enhanced/renderer/rtx.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace openrac::enhanced {
namespace {

// Minimal CUDA Driver API and NVRTC ABI definitions for zero-dependency dynamic loading.
using CUresult = int;
using CUdevice = int;
using CUcontext = struct CUctx_st*;
using CUmodule = struct CUmod_st*;
using CUfunction = struct CUfunc_st*;
using CUstream = struct CUstream_st*;
using CUdeviceptr = std::uint64_t;
using nvrtcResult = int;
using nvrtcProgram = struct _nvrtcProgram*;

constexpr CUresult CUDA_SUCCESS = 0;
constexpr nvrtcResult NVRTC_SUCCESS = 0;

void* open_shared_library(std::span<const char* const> names) {
    for (const char* name : names) {
#if defined(_WIN32)
        if (HMODULE mod = LoadLibraryA(name)) {
            return reinterpret_cast<void*>(mod);
        }
#else
        if (void* mod = dlopen(name, RTLD_LAZY | RTLD_LOCAL)) {
            return mod;
        }
#endif
    }
    return nullptr;
}

void close_shared_library(void* handle) {
    if (!handle) {
        return;
    }
#if defined(_WIN32)
    FreeLibrary(reinterpret_cast<HMODULE>(handle));
#else
    dlclose(handle);
#endif
}

template <typename Fn>
bool load_symbol(void* handle, const char* name, Fn& out) {
#if defined(_WIN32)
    out = reinterpret_cast<Fn>(GetProcAddress(reinterpret_cast<HMODULE>(handle), name));
#else
    out = reinterpret_cast<Fn>(dlsym(handle, name));
#endif
    return out != nullptr;
}

struct CudaDriverApi {
    void* cuda_lib = nullptr;
    void* nvrtc_lib = nullptr;

    CUresult (*cuInit)(unsigned int) = nullptr;
    CUresult (*cuDeviceGetCount)(int*) = nullptr;
    CUresult (*cuDeviceGet)(CUdevice*, int) = nullptr;
    CUresult (*cuDeviceGetName)(char*, int, CUdevice) = nullptr;
    CUresult (*cuDeviceComputeCapability)(int*, int*, CUdevice) = nullptr;
    CUresult (*cuCtxCreate)(CUcontext*, unsigned int, CUdevice) = nullptr;
    CUresult (*cuCtxDestroy)(CUcontext) = nullptr;
    CUresult (*cuCtxSetCurrent)(CUcontext) = nullptr;
    CUresult (*cuModuleLoadData)(CUmodule*, const void*) = nullptr;
    CUresult (*cuModuleUnload)(CUmodule) = nullptr;
    CUresult (*cuModuleGetFunction)(CUfunction*, CUmodule, const char*) = nullptr;
    CUresult (*cuMemAlloc)(CUdeviceptr*, std::size_t) = nullptr;
    CUresult (*cuMemFree)(CUdeviceptr) = nullptr;
    CUresult (*cuMemcpyHtoD)(CUdeviceptr, const void*, std::size_t) = nullptr;
    CUresult (*cuMemcpyDtoH)(void*, CUdeviceptr, std::size_t) = nullptr;
    CUresult (*cuLaunchKernel)(
        CUfunction,
        unsigned int,
        unsigned int,
        unsigned int,
        unsigned int,
        unsigned int,
        unsigned int,
        unsigned int,
        CUstream,
        void**,
        void**
    ) = nullptr;
    CUresult (*cuCtxSynchronize)() = nullptr;

    nvrtcResult (*nvrtcCreateProgram)(
        nvrtcProgram*, const char*, const char*, int, const char* const*, const char* const*
    ) = nullptr;
    nvrtcResult (*nvrtcDestroyProgram)(nvrtcProgram*) = nullptr;
    nvrtcResult (*nvrtcCompileProgram)(nvrtcProgram, int, const char* const*) = nullptr;
    nvrtcResult (*nvrtcGetPTXSize)(nvrtcProgram, std::size_t*) = nullptr;
    nvrtcResult (*nvrtcGetPTX)(nvrtcProgram, char*) = nullptr;

    bool load() {
        constexpr const char* kCudaNames[] = {
#if defined(_WIN32)
            "nvcuda.dll",
#elif defined(__APPLE__)
            "libcuda.dylib",
            "/usr/local/cuda/lib/libcuda.dylib",
#else
            "libcuda.so.1",
            "libcuda.so",
#endif
        };
        cuda_lib = open_shared_library(kCudaNames);
        if (!cuda_lib) {
            return false;
        }
        bool ok = load_symbol(cuda_lib, "cuInit", cuInit)
                  && load_symbol(cuda_lib, "cuDeviceGetCount", cuDeviceGetCount)
                  && load_symbol(cuda_lib, "cuDeviceGet", cuDeviceGet)
                  && load_symbol(cuda_lib, "cuDeviceGetName", cuDeviceGetName)
                  && load_symbol(
                      cuda_lib, "cuDeviceComputeCapability", cuDeviceComputeCapability
                  )
                  && (load_symbol(cuda_lib, "cuCtxCreate_v2", cuCtxCreate)
                      || load_symbol(cuda_lib, "cuCtxCreate", cuCtxCreate))
                  && (load_symbol(cuda_lib, "cuCtxDestroy_v2", cuCtxDestroy)
                      || load_symbol(cuda_lib, "cuCtxDestroy", cuCtxDestroy))
                  && load_symbol(cuda_lib, "cuCtxSetCurrent", cuCtxSetCurrent)
                  && load_symbol(cuda_lib, "cuModuleLoadData", cuModuleLoadData)
                  && load_symbol(cuda_lib, "cuModuleUnload", cuModuleUnload)
                  && load_symbol(cuda_lib, "cuModuleGetFunction", cuModuleGetFunction)
                  && (load_symbol(cuda_lib, "cuMemAlloc_v2", cuMemAlloc)
                      || load_symbol(cuda_lib, "cuMemAlloc", cuMemAlloc))
                  && (load_symbol(cuda_lib, "cuMemFree_v2", cuMemFree)
                      || load_symbol(cuda_lib, "cuMemFree", cuMemFree))
                  && (load_symbol(cuda_lib, "cuMemcpyHtoD_v2", cuMemcpyHtoD)
                      || load_symbol(cuda_lib, "cuMemcpyHtoD", cuMemcpyHtoD))
                  && (load_symbol(cuda_lib, "cuMemcpyDtoH_v2", cuMemcpyDtoH)
                      || load_symbol(cuda_lib, "cuMemcpyDtoH", cuMemcpyDtoH))
                  && load_symbol(cuda_lib, "cuLaunchKernel", cuLaunchKernel)
                  && load_symbol(cuda_lib, "cuCtxSynchronize", cuCtxSynchronize);
        if (!ok || cuInit(0) != CUDA_SUCCESS) {
            close_shared_library(cuda_lib);
            cuda_lib = nullptr;
            return false;
        }
        int device_count = 0;
        if (cuDeviceGetCount(&device_count) != CUDA_SUCCESS || device_count <= 0) {
            close_shared_library(cuda_lib);
            cuda_lib = nullptr;
            return false;
        }

        constexpr const char* kNvrtcNames[] = {
#if defined(_WIN32)
            "nvrtc64_120_0.dll",
            "nvrtc64_112_0.dll",
            "nvrtc64_110_0.dll",
#elif defined(__APPLE__)
            "libnvrtc.dylib",
#else
            "libnvrtc.so.12",
            "libnvrtc.so.11.2",
            "libnvrtc.so",
#endif
        };
        nvrtc_lib = open_shared_library(kNvrtcNames);
        if (nvrtc_lib) {
            if (!load_symbol(nvrtc_lib, "nvrtcCreateProgram", nvrtcCreateProgram)
                || !load_symbol(nvrtc_lib, "nvrtcDestroyProgram", nvrtcDestroyProgram)
                || !load_symbol(nvrtc_lib, "nvrtcCompileProgram", nvrtcCompileProgram)
                || !load_symbol(nvrtc_lib, "nvrtcGetPTXSize", nvrtcGetPTXSize)
                || !load_symbol(nvrtc_lib, "nvrtcGetPTX", nvrtcGetPTX)) {
                close_shared_library(nvrtc_lib);
                nvrtc_lib = nullptr;
            }
        }
        return true;
    }

    ~CudaDriverApi() {
        close_shared_library(nvrtc_lib);
        close_shared_library(cuda_lib);
    }
};

// CUDA C source compiled to PTX via NVRTC at runtime on NVIDIA RTX / GTX GPUs.
constexpr const char* kCudaRtxKernelSource = R"cuda(
extern "C" {

struct RtxTriangleGpu {
    float v0_x, v0_y, v0_z, mat;
    float e1_x, e1_y, e1_z, layer;
    float e2_x, e2_y, e2_z, flags;
    float n_x, n_y, n_z, albedo_bits;
};

struct RtxBvhNodeGpu {
    float min_x, min_y, min_z;
    int left_or_first;
    float max_x, max_y, max_z;
    int count;
};

struct RtxRayGpu {
    float ox, oy, oz, t_min;
    float dx, dy, dz, t_max;
};

struct RtxHitGpu {
    float t;
    int triangle_index;
    float u;
    float v;
    float nx, ny, nz;
    unsigned int packed_albedo;
};

struct ScreenParams {
    unsigned int width;
    unsigned int height;
    unsigned int samples_per_pixel;
    unsigned int flags;
    float cam_x, cam_y, cam_z, ray_distance;
    float sun_dx, sun_dy, sun_dz, ao_radius;
    float sun_r, sun_g, sun_b, shadow_softness;
    float sky_r, sky_g, sky_b, gi_strength;
    unsigned int frame_index;
    float roughness_cutoff;
    float pad0, pad1;
};

__device__ __forceinline__ float fminf3(float a, float b) { return a < b ? a : b; }
__device__ __forceinline__ float fmaxf3(float a, float b) { return a > b ? a : b; }

__device__ bool intersect_aabb(
    float ox, float oy, float oz,
    float inv_dx, float inv_dy, float inv_dz,
    float t_min, float t_max,
    const RtxBvhNodeGpu& node
) {
    float tx0 = (node.min_x - ox) * inv_dx;
    float tx1 = (node.max_x - ox) * inv_dx;
    float tmin = fmaxf3(t_min, fminf3(tx0, tx1));
    float tmax = fminf3(t_max, fmaxf3(tx0, tx1));

    float ty0 = (node.min_y - oy) * inv_dy;
    float ty1 = (node.max_y - oy) * inv_dy;
    tmin = fmaxf3(tmin, fminf3(ty0, ty1));
    tmax = fminf3(tmax, fmaxf3(ty0, ty1));

    float tz0 = (node.min_z - oz) * inv_dz;
    float tz1 = (node.max_z - oz) * inv_dz;
    tmin = fmaxf3(tmin, fminf3(tz0, tz1));
    tmax = fminf3(tmax, fmaxf3(tz0, tz1));

    return tmax >= tmin;
}

__device__ void trace_bvh(
    const RtxBvhNodeGpu* __restrict__ nodes,
    const RtxTriangleGpu* __restrict__ tris,
    int node_count,
    float ox, float oy, float oz,
    float dx, float dy, float dz,
    float t_min, float t_max,
    bool any_hit,
    RtxHitGpu& out_hit
) {
    out_hit.t = -1.0f;
    out_hit.triangle_index = -1;
    out_hit.u = 0.0f;
    out_hit.v = 0.0f;
    out_hit.nx = 0.0f;
    out_hit.ny = 0.0f;
    out_hit.nz = 1.0f;
    out_hit.packed_albedo = 0u;
    if (node_count <= 0) {
        return;
    }

    float inv_dx = 1.0f / (fabsf(dx) > 1e-8f ? dx : (dx >= 0.0f ? 1e-8f : -1e-8f));
    float inv_dy = 1.0f / (fabsf(dy) > 1e-8f ? dy : (dy >= 0.0f ? 1e-8f : -1e-8f));
    float inv_dz = 1.0f / (fabsf(dz) > 1e-8f ? dz : (dz >= 0.0f ? 1e-8f : -1e-8f));

    int stack[64];
    int sp = 0;
    stack[sp++] = 0;
    float closest = t_max;

    while (sp > 0) {
        int idx = stack[--sp];
        const RtxBvhNodeGpu node = nodes[idx];
        if (!intersect_aabb(ox, oy, oz, inv_dx, inv_dy, inv_dz, t_min, closest, node)) {
            continue;
        }
        if (node.count > 0) {
            int first = ~node.left_or_first;
            for (int i = 0; i < node.count; ++i) {
                int tri_idx = first + i;
                const RtxTriangleGpu tri = tris[tri_idx];
                float px = dy * tri.e2_z - dz * tri.e2_y;
                float py = dz * tri.e2_x - dx * tri.e2_z;
                float pz = dx * tri.e2_y - dy * tri.e2_x;
                float det = tri.e1_x * px + tri.e1_y * py + tri.e1_z * pz;
                if (fabsf(det) < 1e-8f) {
                    continue;
                }
                float inv_det = 1.0f / det;
                float tx = ox - tri.v0_x;
                float ty = oy - tri.v0_y;
                float tz = oz - tri.v0_z;
                float u = (tx * px + ty * py + tz * pz) * inv_det;
                if (u < 0.0f || u > 1.0f) {
                    continue;
                }
                float qx = ty * tri.e1_z - tz * tri.e1_y;
                float qy = tz * tri.e1_x - tx * tri.e1_z;
                float qz = tx * tri.e1_y - ty * tri.e1_x;
                float v = (dx * qx + dy * qy + dz * qz) * inv_det;
                if (v < 0.0f || u + v > 1.0f) {
                    continue;
                }
                float t = (tri.e2_x * qx + tri.e2_y * qy + tri.e2_z * qz) * inv_det;
                if (t >= t_min && t < closest) {
                    closest = t;
                    out_hit.t = t;
                    out_hit.triangle_index = tri_idx;
                    out_hit.u = u;
                    out_hit.v = v;
                    out_hit.nx = tri.n_x;
                    out_hit.ny = tri.n_y;
                    out_hit.nz = tri.n_z;
                    out_hit.packed_albedo = __float_as_uint(tri.albedo_bits);
                    if (any_hit) {
                        return;
                    }
                }
            }
        } else {
            int left = node.left_or_first;
            int right = left + 1;
            if (sp + 2 <= 64) {
                stack[sp++] = right;
                stack[sp++] = left;
            }
        }
    }
}

__global__ void rtx_trace_rays_kernel(
    const RtxBvhNodeGpu* __restrict__ nodes,
    int node_count,
    const RtxTriangleGpu* __restrict__ tris,
    const RtxRayGpu* __restrict__ rays,
    RtxHitGpu* __restrict__ hits,
    unsigned int ray_count
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= ray_count) {
        return;
    }
    RtxRayGpu r = rays[tid];
    RtxHitGpu h;
    trace_bvh(nodes, tris, node_count, r.ox, r.oy, r.oz, r.dx, r.dy, r.dz, r.t_min, r.t_max, false, h);
    hits[tid] = h;
}

__device__ unsigned int pcg_hash(unsigned int v) {
    unsigned int state = v * 747796405u + 2891336453u;
    unsigned int word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

__device__ float rand01(unsigned int& seed) {
    seed = pcg_hash(seed);
    return float(seed & 0x00FFFFFFu) * (1.0f / 16777216.0f);
}

__device__ unsigned int pack_rgba8(float r, float g, float b, float a) {
    r = fminf3(1.0f, fmaxf3(0.0f, r));
    g = fminf3(1.0f, fmaxf3(0.0f, g));
    b = fminf3(1.0f, fmaxf3(0.0f, b));
    a = fminf3(1.0f, fmaxf3(0.0f, a));
    unsigned int ur = (unsigned int)(r * 255.0f + 0.5f);
    unsigned int ug = (unsigned int)(g * 255.0f + 0.5f);
    unsigned int ub = (unsigned int)(b * 255.0f + 0.5f);
    unsigned int ua = (unsigned int)(a * 255.0f + 0.5f);
    return ur | (ug << 8u) | (ub << 16u) | (ua << 24u);
}

__global__ void rtx_trace_screen_kernel(
    const RtxBvhNodeGpu* __restrict__ nodes,
    int node_count,
    const RtxTriangleGpu* __restrict__ tris,
    const float4* __restrict__ world_positions,
    const float4* __restrict__ world_normals,
    ScreenParams params,
    unsigned int* __restrict__ out_rgba8_pairs
) {
    unsigned int gx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int gy = blockIdx.y * blockDim.y + threadIdx.y;
    if (gx >= params.width || gy >= params.height) {
        return;
    }
    unsigned int cell = gy * params.width + gx;
    float4 pos = world_positions[cell];
    if (pos.w <= 0.0f) {
        out_rgba8_pairs[cell * 2u + 0u] = 0xFF000000u;
        out_rgba8_pairs[cell * 2u + 1u] = 0u;
        return;
    }
    float4 nrm = world_normals[cell];
    float inv_n = rsqrtf(fmaxf3(1e-8f, nrm.x * nrm.x + nrm.y * nrm.y + nrm.z * nrm.z));
    float nx = nrm.x * inv_n, ny = nrm.y * inv_n, nz = nrm.z * inv_n;
    float ox = pos.x + nx * 0.025f;
    float oy = pos.y + ny * 0.025f;
    float oz = pos.z + nz * 0.025f;

    float to_sun_x = -params.sun_dx, to_sun_y = -params.sun_dy, to_sun_z = -params.sun_dz;
    float inv_s = rsqrtf(fmaxf3(1e-8f, to_sun_x * to_sun_x + to_sun_y * to_sun_y + to_sun_z * to_sun_z));
    to_sun_x *= inv_s; to_sun_y *= inv_s; to_sun_z *= inv_s;

    float shadow_vis = 1.0f;
    if ((params.flags & 1u) != 0u) {
        float ndotl = nx * to_sun_x + ny * to_sun_y + nz * to_sun_z;
        if (ndotl > 0.0f) {
            RtxHitGpu sh;
            trace_bvh(nodes, tris, node_count, ox, oy, oz, to_sun_x, to_sun_y, to_sun_z, 0.02f, params.ray_distance, true, sh);
            shadow_vis = (sh.triangle_index < 0) ? 1.0f : 0.0f;
        } else {
            shadow_vis = 0.0f;
        }
    }

    unsigned int seed = pcg_hash(cell ^ (params.frame_index * 1664525u + 1013904223u));
    float ao_vis = 1.0f;
    float gi_r = 0.0f, gi_g = 0.0f, gi_b = 0.0f;
    if ((params.flags & 10u) != 0u) {
        unsigned int spp = params.samples_per_pixel > 0u ? params.samples_per_pixel : 1u;
        unsigned int hits = 0u;
        for (unsigned int s = 0u; s < spp; ++s) {
            float u1 = rand01(seed);
            float u2 = rand01(seed);
            float r = sqrtf(u1);
            float phi = 6.2831853f * u2;
            float lx = r * cosf(phi), ly = r * sinf(phi), lz = sqrtf(fmaxf3(0.0f, 1.0f - u1));
            float sign = nz >= 0.0f ? 1.0f : -1.0f;
            float a = -1.0f / (sign + nz);
            float b = nx * ny * a;
            float tx = 1.0f + sign * nx * nx * a, ty = sign * b, tz = -sign * nx;
            float bx = b, by = sign + ny * ny * a, bz = -ny;
            float dx = tx * lx + bx * ly + nx * lz;
            float dy = ty * lx + by * ly + ny * lz;
            float dz = tz * lx + bz * ly + nz * lz;
            RtxHitGpu bh;
            float max_t = ((params.flags & 8u) != 0u) ? params.ray_distance : params.ao_radius;
            trace_bvh(nodes, tris, node_count, ox, oy, oz, dx, dy, dz, 0.02f, max_t, false, bh);
            if (bh.triangle_index >= 0) {
                if (bh.t <= params.ao_radius) {
                    hits++;
                }
                if ((params.flags & 8u) != 0u) {
                    float ar = float(bh.packed_albedo & 0xFFu) * (1.0f / 255.0f);
                    float ag = float((bh.packed_albedo >> 8u) & 0xFFu) * (1.0f / 255.0f);
                    float ab = float((bh.packed_albedo >> 16u) & 0xFFu) * (1.0f / 255.0f);
                    gi_r += ar * params.sky_r * params.gi_strength;
                    gi_g += ag * params.sky_g * params.gi_strength;
                    gi_b += ab * params.sky_b * params.gi_strength;
                }
            }
        }
        if ((params.flags & 2u) != 0u) {
            ao_vis = 1.0f - (float(hits) / float(spp));
        }
        gi_r /= float(spp); gi_g /= float(spp); gi_b /= float(spp);
    }

    float refl_r = 0.0f, refl_g = 0.0f, refl_b = 0.0f, refl_conf = 0.0f;
    if ((params.flags & 4u) != 0u && nrm.w <= params.roughness_cutoff) {
        float vx = pos.x - params.cam_x, vy = pos.y - params.cam_y, vz = pos.z - params.cam_z;
        float inv_v = rsqrtf(fmaxf3(1e-8f, vx * vx + vy * vy + vz * vz));
        vx *= inv_v; vy *= inv_v; vz *= inv_v;
        float vn = vx * nx + vy * ny + vz * nz;
        float rx = vx - 2.0f * vn * nx;
        float ry = vy - 2.0f * vn * ny;
        float rz = vz - 2.0f * vn * nz;
        RtxHitGpu rh;
        trace_bvh(nodes, tris, node_count, ox, oy, oz, rx, ry, rz, 0.025f, params.ray_distance, false, rh);
        if (rh.triangle_index >= 0) {
            refl_r = float(rh.packed_albedo & 0xFFu) * (1.0f / 255.0f) * params.sun_r;
            refl_g = float((rh.packed_albedo >> 8u) & 0xFFu) * (1.0f / 255.0f) * params.sun_g;
            refl_b = float((rh.packed_albedo >> 16u) & 0xFFu) * (1.0f / 255.0f) * params.sun_b;
            refl_conf = fmaxf3(0.15f, 1.0f - rh.t / params.ray_distance);
        } else {
            refl_r = params.sky_r; refl_g = params.sky_g; refl_b = params.sky_b;
            refl_conf = 0.5f;
        }
    }

    out_rgba8_pairs[cell * 2u + 0u] = pack_rgba8(gi_r, gi_g, gi_b, shadow_vis * ao_vis);
    out_rgba8_pairs[cell * 2u + 1u] = pack_rgba8(refl_r, refl_g, refl_b, refl_conf);
}

} // extern "C"
)cuda";

class CudaRtxBackend final : public RtxHardwareBackend {
public:
    CudaRtxBackend(
        std::unique_ptr<CudaDriverApi> api,
        CUcontext context,
        CUmodule module,
        CUfunction trace_rays_fn,
        CUfunction trace_screen_fn,
        std::string device_name
    )
        : m_api(std::move(api)),
          m_context(context),
          m_module(module),
          m_trace_rays_fn(trace_rays_fn),
          m_trace_screen_fn(trace_screen_fn),
          m_device_name(std::move(device_name)) {}

    ~CudaRtxBackend() override {
        if (!m_api || !m_context) {
            return;
        }
        m_api->cuCtxSetCurrent(m_context);
        if (m_d_nodes) {
            m_api->cuMemFree(m_d_nodes);
        }
        if (m_d_tris) {
            m_api->cuMemFree(m_d_tris);
        }
        if (m_module) {
            m_api->cuModuleUnload(m_module);
        }
        m_api->cuCtxDestroy(m_context);
    }

    renderer::RtxBackend kind() const override { return renderer::RtxBackend::Cuda; }
    std::string_view device_name() const override { return m_device_name; }

    bool upload_scene(
        std::span<const RtxTriangle> triangles,
        std::span<const RtxBvhNode> nodes,
        std::string& error
    ) override {
        std::lock_guard lock(m_mutex);
        m_api->cuCtxSetCurrent(m_context);
        if (m_d_nodes) {
            m_api->cuMemFree(m_d_nodes);
            m_d_nodes = 0;
        }
        if (m_d_tris) {
            m_api->cuMemFree(m_d_tris);
            m_d_tris = 0;
        }
        m_node_count = static_cast<int>(nodes.size());
        if (triangles.empty() || nodes.empty()) {
            return true;
        }
        const std::size_t node_bytes = nodes.size() * sizeof(RtxBvhNode);
        const std::size_t tri_bytes = triangles.size() * sizeof(RtxTriangle);
        if (m_api->cuMemAlloc(&m_d_nodes, node_bytes) != CUDA_SUCCESS
            || m_api->cuMemAlloc(&m_d_tris, tri_bytes) != CUDA_SUCCESS) {
            error = "CUDA RTX: failed to allocate device BVH buffers";
            return false;
        }
        if (m_api->cuMemcpyHtoD(m_d_nodes, nodes.data(), node_bytes) != CUDA_SUCCESS
            || m_api->cuMemcpyHtoD(m_d_tris, triangles.data(), tri_bytes) != CUDA_SUCCESS) {
            error = "CUDA RTX: failed to upload BVH to device memory";
            return false;
        }
        return true;
    }

    bool trace_rays(
        std::span<const RtxRay> rays,
        std::span<RtxHit> hits,
        std::string& error
    ) override {
        std::lock_guard lock(m_mutex);
        if (rays.size() != hits.size()) {
            error = "CUDA RTX: ray and hit counts must match";
            return false;
        }
        if (rays.empty()) {
            return true;
        }
        if (!m_d_nodes || !m_d_tris || m_node_count <= 0) {
            for (auto& h : hits) {
                h = RtxHit{};
            }
            return true;
        }
        m_api->cuCtxSetCurrent(m_context);
        CUdeviceptr d_rays = 0;
        CUdeviceptr d_hits = 0;
        const std::size_t ray_bytes = rays.size() * sizeof(RtxRay);
        const std::size_t hit_bytes = hits.size() * sizeof(RtxHit);
        if (m_api->cuMemAlloc(&d_rays, ray_bytes) != CUDA_SUCCESS
            || m_api->cuMemAlloc(&d_hits, hit_bytes) != CUDA_SUCCESS) {
            if (d_rays) m_api->cuMemFree(d_rays);
            error = "CUDA RTX: failed to allocate ray/hit buffers";
            return false;
        }
        m_api->cuMemcpyHtoD(d_rays, rays.data(), ray_bytes);

        unsigned int ray_count = static_cast<unsigned int>(rays.size());
        void* args[] = {&m_d_nodes, &m_node_count, &m_d_tris, &d_rays, &d_hits, &ray_count};
        const unsigned int block = 64;
        const unsigned int grid = (ray_count + block - 1) / block;
        const CUresult launch = m_api->cuLaunchKernel(
            m_trace_rays_fn, grid, 1, 1, block, 1, 1, 0, nullptr, args, nullptr
        );
        m_api->cuCtxSynchronize();
        if (launch == CUDA_SUCCESS) {
            m_api->cuMemcpyDtoH(hits.data(), d_hits, hit_bytes);
        }
        m_api->cuMemFree(d_rays);
        m_api->cuMemFree(d_hits);
        if (launch != CUDA_SUCCESS) {
            error = "CUDA RTX: kernel launch failed";
            return false;
        }
        return true;
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
        std::lock_guard lock(m_mutex);
        const std::size_t cells =
            static_cast<std::size_t>(std::max(0, cells_width)) * std::max(0, cells_height);
        if (cells == 0 || out_rgba8_pairs.size() < cells * 2) {
            return false;
        }
        if (!m_d_nodes || !m_d_tris || m_node_count <= 0) {
            for (std::size_t i = 0; i < cells; ++i) {
                out_rgba8_pairs[i * 2 + 0] = 0xFF000000u;
                out_rgba8_pairs[i * 2 + 1] = 0u;
            }
            return true;
        }
        m_api->cuCtxSetCurrent(m_context);

        struct ScreenParamsHost {
            unsigned int width, height, samples_per_pixel, flags;
            float cam_x, cam_y, cam_z, ray_distance;
            float sun_dx, sun_dy, sun_dz, ao_radius;
            float sun_r, sun_g, sun_b, shadow_softness;
            float sky_r, sky_g, sky_b, gi_strength;
            unsigned int frame_index;
            float roughness_cutoff, pad0, pad1;
        } params{};
        params.width = static_cast<unsigned int>(cells_width);
        params.height = static_cast<unsigned int>(cells_height);
        params.samples_per_pixel =
            static_cast<unsigned int>(std::clamp(settings.samples_per_pixel, 1, 8));
        params.flags = (settings.shadows ? 1u : 0u)
                       | (settings.ambient_occlusion ? 2u : 0u)
                       | (settings.reflections ? 4u : 0u)
                       | (settings.global_illumination ? 8u : 0u);
        params.cam_x = camera_pos[0];
        params.cam_y = camera_pos[1];
        params.cam_z = camera_pos[2];
        params.ray_distance = settings.ray_distance;
        params.sun_dx = sun_dir[0];
        params.sun_dy = sun_dir[1];
        params.sun_dz = sun_dir[2];
        params.ao_radius = settings.ao_radius;
        params.sun_r = sun_colour[0];
        params.sun_g = sun_colour[1];
        params.sun_b = sun_colour[2];
        params.shadow_softness = settings.shadow_softness;
        params.sky_r = sky_colour[0];
        params.sky_g = sky_colour[1];
        params.sky_b = sky_colour[2];
        params.gi_strength = settings.gi_strength;
        params.frame_index = frame_index;
        params.roughness_cutoff = settings.reflection_roughness_cutoff;

        CUdeviceptr d_pos = 0, d_nrm = 0, d_out = 0;
        const std::size_t vec4_bytes = cells * sizeof(float) * 4;
        const std::size_t out_bytes = cells * 2 * sizeof(std::uint32_t);
        if (m_api->cuMemAlloc(&d_pos, vec4_bytes) != CUDA_SUCCESS
            || m_api->cuMemAlloc(&d_nrm, vec4_bytes) != CUDA_SUCCESS
            || m_api->cuMemAlloc(&d_out, out_bytes) != CUDA_SUCCESS) {
            if (d_pos) m_api->cuMemFree(d_pos);
            if (d_nrm) m_api->cuMemFree(d_nrm);
            error = "CUDA RTX: failed to allocate screen buffers";
            return false;
        }
        m_api->cuMemcpyHtoD(d_pos, world_positions.data(), vec4_bytes);
        m_api->cuMemcpyHtoD(d_nrm, world_normals.data(), vec4_bytes);

        void* args[] = {&m_d_nodes, &m_node_count, &m_d_tris, &d_pos, &d_nrm, &params, &d_out};
        const unsigned int bx = 8, by = 8;
        const unsigned int gx = (params.width + bx - 1) / bx;
        const unsigned int gy = (params.height + by - 1) / by;
        const CUresult launch = m_api->cuLaunchKernel(
            m_trace_screen_fn, gx, gy, 1, bx, by, 1, 0, nullptr, args, nullptr
        );
        m_api->cuCtxSynchronize();
        if (launch == CUDA_SUCCESS) {
            m_api->cuMemcpyDtoH(out_rgba8_pairs.data(), d_out, out_bytes);
        }
        m_api->cuMemFree(d_pos);
        m_api->cuMemFree(d_nrm);
        m_api->cuMemFree(d_out);
        if (launch != CUDA_SUCCESS) {
            error = "CUDA RTX: screen trace kernel failed";
            return false;
        }
        return true;
    }

private:
    std::mutex m_mutex;
    std::unique_ptr<CudaDriverApi> m_api;
    CUcontext m_context = nullptr;
    CUmodule m_module = nullptr;
    CUfunction m_trace_rays_fn = nullptr;
    CUfunction m_trace_screen_fn = nullptr;
    CUdeviceptr m_d_nodes = 0;
    CUdeviceptr m_d_tris = 0;
    int m_node_count = 0;
    std::string m_device_name;
};

}  // namespace

bool cuda_rtx_supported() {
    CudaDriverApi api;
    return api.load() && api.nvrtc_lib != nullptr;
}

std::unique_ptr<RtxHardwareBackend> create_cuda_rtx_backend() {
    auto api = std::make_unique<CudaDriverApi>();
    if (!api->load() || !api->nvrtc_lib) {
        return nullptr;
    }
    CUdevice device = 0;
    if (api->cuDeviceGet(&device, 0) != CUDA_SUCCESS) {
        return nullptr;
    }
    char name_buf[256]{};
    api->cuDeviceGetName(name_buf, sizeof(name_buf), device);
    int major = 0, minor = 0;
    api->cuDeviceComputeCapability(&major, &minor, device);

    CUcontext context = nullptr;
    if (api->cuCtxCreate(&context, 0, device) != CUDA_SUCCESS || !context) {
        return nullptr;
    }

    nvrtcProgram prog = nullptr;
    if (api->nvrtcCreateProgram(
            &prog, kCudaRtxKernelSource, "openrac_rtx.cu", 0, nullptr, nullptr
        )
        != NVRTC_SUCCESS) {
        api->cuCtxDestroy(context);
        return nullptr;
    }
    const std::string arch_opt =
        "--gpu-architecture=compute_" + std::to_string(std::max(5, major)) + std::to_string(minor);
    const char* opts[] = {arch_opt.c_str(), "--use_fast_math"};
    if (api->nvrtcCompileProgram(prog, 2, opts) != NVRTC_SUCCESS) {
        const char* fallback_opts[] = {"-- use_fast_math"};
        if (api->nvrtcCompileProgram(prog, 0, fallback_opts) != NVRTC_SUCCESS) {
            api->nvrtcDestroyProgram(&prog);
            api->cuCtxDestroy(context);
            return nullptr;
        }
    }
    std::size_t ptx_size = 0;
    api->nvrtcGetPTXSize(prog, &ptx_size);
    std::string ptx(ptx_size, '\0');
    api->nvrtcGetPTX(prog, ptx.data());
    api->nvrtcDestroyProgram(&prog);

    CUmodule module = nullptr;
    if (api->cuModuleLoadData(&module, ptx.data()) != CUDA_SUCCESS || !module) {
        api->cuCtxDestroy(context);
        return nullptr;
    }
    CUfunction rays_fn = nullptr;
    CUfunction screen_fn = nullptr;
    if (api->cuModuleGetFunction(&rays_fn, module, "rtx_trace_rays_kernel") != CUDA_SUCCESS
        || api->cuModuleGetFunction(&screen_fn, module, "rtx_trace_screen_kernel") != CUDA_SUCCESS) {
        api->cuModuleUnload(module);
        api->cuCtxDestroy(context);
        return nullptr;
    }

    std::string full_name = std::string(name_buf) + " (CUDA RTX)";
    return std::make_unique<CudaRtxBackend>(
        std::move(api), context, module, rays_fn, screen_fn, std::move(full_name)
    );
}

}  // namespace openrac::enhanced
