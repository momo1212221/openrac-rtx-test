#version 410 core
// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 the OpenRAC contributors
// Particle colour changes and new opaque objects reduce temporal history confidence.
uniform sampler2D tex_T0;  // the world's colour
uniform sampler2D tex_T1;  // the same before the blended effects (the opaque snapshot)
uniform sampler2D tex_T2;  // motion (previous minus current, render pixels), z: no history
uniform sampler2D tex_T3;  // stage 2: what the effects changed this frame (stage 1's output)
uniform sampler2D tex_T4;  // stage 2: the same of the previous frame
// 0: the mask from this frame's colour change alone. 1: writes the colour change (the effects'
// layer) for the next frame. 2: the mask from how that layer differs from the previous frame's.
uniform int stage;
uniform vec2 history_offset;  // the previous frame's raster jitter minus this one's
layout(location=0) out vec4 result;

float largest(vec3 v) {
    return max(v.r, max(v.g, v.b));
}

void main() {
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    if (stage == 1) {
        result = vec4(texelFetch(tex_T0, pixel, 0).rgb - texelFetch(tex_T1, pixel, 0).rgb, 1);
        return;
    }
    vec4 motion = texelFetch(tex_T2, pixel, 0);
    float reactive;
    if (stage == 2) {
        // A blended surface that stays as it was (a pane of glass, a haze, a lasting corona) keeps
        // its history: taken as reactive for its colour change alone, everything seen through
        // Blarg station's windows had none, and every edge behind the glass flickered with the
        // raster jitter. Reactive is only what differs from the previous frame's layer where this
        // pixel was: a spark, a shot or smoke that came, moved or went. Both ways against the
        // other frame's 3x3 neighbours, so that the jitter's one-pixel wander of an edge behind
        // the layer does not count and a small spark that left still does.
        ivec2 last = textureSize(tex_T3, 0) - 1;
        ivec2 was = clamp(ivec2(floor(gl_FragCoord.xy + motion.xy + history_offset)), ivec2(0), last);
        vec3 now = texelFetch(tex_T3, pixel, 0).rgb;
        vec3 then = texelFetch(tex_T4, was, 0).rgb;
        vec3 then_low = then, then_high = then, now_low = now, now_high = now;
        for (int y = -1; y <= 1; ++y) {
            for (int x = -1; x <= 1; ++x) {
                vec3 a = texelFetch(tex_T4, clamp(was + ivec2(x, y), ivec2(0), last), 0).rgb;
                vec3 b = texelFetch(tex_T3, clamp(pixel + ivec2(x, y), ivec2(0), last), 0).rgb;
                then_low = min(then_low, a);
                then_high = max(then_high, a);
                now_low = min(now_low, b);
                now_high = max(now_high, b);
            }
        }
        float appeared = largest(max(max(then_low - now, now - then_high), vec3(0)));
        float left = largest(max(max(now_low - then, then - now_high), vec3(0)));
        reactive = clamp(4 * (max(appeared, left) - 0.02), 0, 1);
    } else {
        vec3 difference = abs(texelFetch(tex_T0, pixel, 0).rgb - texelFetch(tex_T1, pixel, 0).rgb);
        // Conservative coverage for alpha-blended and additive effects without opaque depth.
        reactive = clamp(4 * largest(difference), 0, 1);
    }
    reactive = max(reactive, motion.z);
    result = vec4(reactive, 0, 0, 1);
}
