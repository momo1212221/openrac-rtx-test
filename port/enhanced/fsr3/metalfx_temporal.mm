// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 the OpenRAC contributors
//
// MetalFX temporal upscaling on the Metal GL (metalfx_temporal.h).
//
// The Metal GL keeps a GL image's row 0 at the bottom, as OpenGL does, in the texture's row 0. A
// GL pixel's y is therefore its Metal row, and the motion (render pixels) and the jitter keep
// their signs. The float depth is copied into a Depth32Float image, the depth format MetalFX
// takes; it is reversed (1 near), as the FSR path's.

#include "enhanced/fsr3/metalfx_temporal.h"

#import <Metal/Metal.h>
#import <MetalFX/MetalFX.h>

#include <cstdlib>
#include <format>

#include "common/log.h"
#include "renderer/gl.h"
#include "renderer/metal/mtlgl.h"

namespace openrac::enhanced::fsr3_detail {

namespace {

constexpr const char* kDepthSource = R"(
#include <metal_stdlib>
using namespace metal;
struct Out { float4 position [[position]]; };
vertex Out depth_vertex(uint id [[vertex_id]]) {
    const float2 p = float2((id << 1) & 2, id & 2);
    return {float4(p * 2.0 - 1.0, 0.0, 1.0)};
}
struct Depth { float value [[depth(any)]]; };
fragment Depth depth_fragment(Out in [[stage_in]], texture2d<float> source [[texture(0)]]) {
    return {source.read(uint2(in.position.xy)).r};
}
)";

// OPENRAC_METALFX_JITTER_SIGN=-1 hands MetalFX the opposite jitter, for checking the convention.
float jitter_sign() {
    static const float sign = [] {
        const char* v = std::getenv("OPENRAC_METALFX_JITTER_SIGN");
        return v != nullptr && v[0] == '-' ? -1.0f : 1.0f;
    }();
    return sign;
}

}  // namespace

struct MetalFxTemporal::Impl {
    std::array<int, 2> render{}, display{};
    id scaler;  // id<MTLFXTemporalScaler>
    id<MTLTexture> depth, output;
    id<MTLRenderPipelineState> depth_copy;
    MTLPixelFormat colour_format = MTLPixelFormatInvalid, motion_format = MTLPixelFormatInvalid,
                   output_format = MTLPixelFormatInvalid, reactive_format = MTLPixelFormatInvalid;
};

MetalFxTemporal::MetalFxTemporal() : m(std::make_unique<Impl>()) {}

MetalFxTemporal::~MetalFxTemporal() = default;

bool MetalFxTemporal::supported() {
    if (!gl::is_metal()) {
        return false;
    }
    id<MTLDevice> device = (__bridge id<MTLDevice>)mtlgl::native().device;
    if (device == nil) {
        return false;
    }
    if (@available(macOS 13.0, *)) {
        return [MTLFXTemporalScalerDescriptor supportsDevice:device];
    }
    return false;
}

bool MetalFxTemporal::configure(
    std::array<int, 2> render_size, std::array<int, 2> display_size, std::string& error
) {
    if (render_size[0] <= 0 || render_size[1] <= 0 || display_size[0] < render_size[0]
        || display_size[1] < render_size[1]) {
        error = "invalid MetalFX render or display size";
        return false;
    }
    if (render_size != m->render || display_size != m->display) {
        release();
        m->render = render_size;
        m->display = display_size;
    }
    return true;
}

bool MetalFxTemporal::dispatch(
    std::array<unsigned, 5> textures, const Dispatch& frame, std::string& error
) {
    if (@available(macOS 13.0, *)) {
        id<MTLDevice> device = (__bridge id<MTLDevice>)mtlgl::native().device;
        id<MTLTexture> colour = (__bridge id<MTLTexture>)mtlgl::native_texture(textures[0]);
        id<MTLTexture> float_depth = (__bridge id<MTLTexture>)mtlgl::native_texture(textures[1]);
        id<MTLTexture> motion = (__bridge id<MTLTexture>)mtlgl::native_texture(textures[2]);
        id<MTLTexture> reactive = (__bridge id<MTLTexture>)mtlgl::native_texture(textures[3]);
        id<MTLTexture> target = (__bridge id<MTLTexture>)mtlgl::native_texture(textures[4]);
        if (device == nil || colour == nil || float_depth == nil || motion == nil || target == nil) {
            error = "MetalFX temporal upscaling: an input is not a Metal texture";
            return false;
        }
        const auto [rw, rh] = m->render;
        const auto [dw, dh] = m->display;
        if (static_cast<int>(target.width) != dw || static_cast<int>(target.height) != dh) {
            error = std::format(
                "MetalFX temporal upscaling: output {}x{}, configured {}x{}", target.width,
                target.height, dw, dh
            );
            return false;
        }

        if (m->scaler == nil || m->colour_format != colour.pixelFormat
            || m->motion_format != motion.pixelFormat || m->output_format != target.pixelFormat
            || m->reactive_format != (reactive != nil ? reactive.pixelFormat : MTLPixelFormatInvalid)) {
            m->scaler = nil;
            MTLFXTemporalScalerDescriptor* d = [MTLFXTemporalScalerDescriptor new];
            d.colorTextureFormat = colour.pixelFormat;
            d.depthTextureFormat = MTLPixelFormatDepth32Float;
            d.motionTextureFormat = motion.pixelFormat;
            d.outputTextureFormat = target.pixelFormat;
            d.inputWidth = static_cast<NSUInteger>(rw);
            d.inputHeight = static_cast<NSUInteger>(rh);
            d.outputWidth = static_cast<NSUInteger>(dw);
            d.outputHeight = static_cast<NSUInteger>(dh);
            d.autoExposureEnabled = NO;
            // The particle and new-object mask (reactive.frag): there the current frame wins, so a
            // moving figure leaves no trail of earlier frames.
            if (@available(macOS 14.4, *)) {
                if (reactive != nil) {
                    d.reactiveMaskTextureEnabled = YES;
                    d.reactiveMaskTextureFormat = reactive.pixelFormat;
                }
            }
            id<MTLFXTemporalScaler> scaler = [d newTemporalScalerWithDevice:device];
            if (scaler == nil) {
                error = std::format(
                    "MetalFX cannot make a temporal scaler for {}x{} to {}x{}", rw, rh, dw, dh
                );
                return false;
            }
            MTLTextureDescriptor* t =
                [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float
                                                                   width:static_cast<NSUInteger>(rw)
                                                                  height:static_cast<NSUInteger>(rh)
                                                               mipmapped:NO];
            t.storageMode = MTLStorageModePrivate;
            t.usage = MTLTextureUsageRenderTarget | scaler.depthTextureUsage;
            m->depth = [device newTextureWithDescriptor:t];
            t = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:target.pixelFormat
                                                                   width:static_cast<NSUInteger>(dw)
                                                                  height:static_cast<NSUInteger>(dh)
                                                               mipmapped:NO];
            t.storageMode = MTLStorageModePrivate;
            t.usage = scaler.outputTextureUsage;
            m->output = [device newTextureWithDescriptor:t];
            if (m->depth_copy == nil) {
                NSError* failure = nil;
                id<MTLLibrary> library = [device newLibraryWithSource:@(kDepthSource)
                                                              options:nil
                                                                error:&failure];
                MTLRenderPipelineDescriptor* p = [MTLRenderPipelineDescriptor new];
                p.vertexFunction = [library newFunctionWithName:@"depth_vertex"];
                p.fragmentFunction = [library newFunctionWithName:@"depth_fragment"];
                p.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
                m->depth_copy = library != nil ? [device newRenderPipelineStateWithDescriptor:p
                                                                                        error:&failure]
                                               : nil;
                if (m->depth_copy == nil) {
                    error = std::format(
                        "MetalFX depth copy: {}",
                        failure != nil ? failure.localizedDescription.UTF8String : "no pipeline"
                    );
                    return false;
                }
            }
            if (m->depth == nil || m->output == nil) {
                error = "MetalFX temporal upscaling: no memory for its images";
                return false;
            }
            m->scaler = scaler;
            m->colour_format = colour.pixelFormat;
            m->motion_format = motion.pixelFormat;
            m->output_format = target.pixelFormat;
            m->reactive_format = reactive != nil ? reactive.pixelFormat : MTLPixelFormatInvalid;
            log::info("graphics: MetalFX temporal upscaling, {}x{} to {}x{}", rw, rh, dw, dh);
        }

        id<MTLFXTemporalScaler> scaler = m->scaler;
        id<MTLTexture> depth = m->depth, output = m->output;
        id<MTLRenderPipelineState> depth_copy = m->depth_copy;
        const float sign = jitter_sign();
        return mtlgl::encode_native([&](void* raw) {
            id<MTLCommandBuffer> commands = (__bridge id<MTLCommandBuffer>)raw;
            MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
            pass.depthAttachment.texture = depth;
            pass.depthAttachment.loadAction = MTLLoadActionDontCare;
            pass.depthAttachment.storeAction = MTLStoreActionStore;
            id<MTLRenderCommandEncoder> copy = [commands renderCommandEncoderWithDescriptor:pass];
            [copy setRenderPipelineState:depth_copy];
            [copy setFragmentTexture:float_depth atIndex:0];
            [copy drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
            [copy endEncoding];

            scaler.colorTexture = colour;
            scaler.depthTexture = depth;
            scaler.motionTexture = motion;
            scaler.outputTexture = output;
            if (@available(macOS 14.4, *)) {
                if (scaler.reactiveMaskTextureFormat != MTLPixelFormatInvalid) {
                    scaler.reactiveMaskTexture = reactive;
                }
            }
            scaler.inputContentWidth = static_cast<NSUInteger>(rw);
            scaler.inputContentHeight = static_cast<NSUInteger>(rh);
            scaler.jitterOffsetX = sign * frame.jitter[0];
            scaler.jitterOffsetY = sign * frame.jitter[1];
            scaler.motionVectorScaleX = 1.0f;
            scaler.motionVectorScaleY = 1.0f;
            scaler.depthReversed = YES;
            scaler.reset = frame.reset;
            [scaler encodeToCommandBuffer:commands];

            id<MTLBlitCommandEncoder> blit = [commands blitCommandEncoder];
            [blit copyFromTexture:output toTexture:target];
            [blit endEncoding];
        });
    }
    error = "MetalFX temporal upscaling needs macOS 13";
    return false;
}

void MetalFxTemporal::release() {
    m->scaler = nil;
    m->depth = nil;
    m->output = nil;
    m->colour_format = m->motion_format = m->output_format = m->reactive_format =
        MTLPixelFormatInvalid;
}

}  // namespace openrac::enhanced::fsr3_detail
