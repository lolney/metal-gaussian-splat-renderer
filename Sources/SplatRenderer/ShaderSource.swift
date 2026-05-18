import Foundation

enum ShaderSource {
    static let metal = #"""
#include <metal_stdlib>
using namespace metal;

struct PackedSplat {
    float4 positionAndOpacity;
    float4 scaleAndFlags;
    float4 rotation;
    float4 color;
};

struct CameraUniforms {
    float4x4 viewMatrix;
    float4x4 projectionMatrix;
    float4x4 viewProjectionMatrix;
    float4 viewportAndRadius;
};

struct SortPair {
    uint key;
    uint index;
};

struct SortConstants {
    uint count;
    uint paddedCount;
    uint j;
    uint k;
    uint sourceCount;
};

struct ProjectedSplat {
    float4 clipCenter;
    float4 axis0Opacity;
    float4 axis1;
    float4 color;
};

struct VertexOut {
    float4 position [[position]];
    float2 local;
    float4 color;
    float opacity;
};

static float3x3 quatToMatrix(float4 q) {
    float w = q.x;
    float x = q.y;
    float y = q.z;
    float z = q.w;
    float xx = x * x;
    float yy = y * y;
    float zz = z * z;
    float xy = x * y;
    float xz = x * z;
    float yz = y * z;
    float wx = w * x;
    float wy = w * y;
    float wz = w * z;
    return float3x3(
        float3(1.0 - 2.0 * (yy + zz), 2.0 * (xy + wz), 2.0 * (xz - wy)),
        float3(2.0 * (xy - wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz + wx)),
        float3(2.0 * (xz + wy), 2.0 * (yz - wx), 1.0 - 2.0 * (xx + yy))
    );
}

static float2 projectNDC(constant CameraUniforms &camera, float3 position) {
    float4 clip = camera.viewProjectionMatrix * float4(position, 1.0);
    return clip.xy / max(abs(clip.w), 0.0001);
}

static void covarianceEllipse(
    constant CameraUniforms &camera,
    PackedSplat splat,
    thread float2 &axis0,
    thread float2 &axis1
) {
    float3 position = splat.positionAndOpacity.xyz;
    float3 scale = max(splat.scaleAndFlags.xyz, float3(0.0001));
    float3x3 rotation = quatToMatrix(splat.rotation);
    float2 viewport = max(camera.viewportAndRadius.xy, float2(1.0));
    float2 center = projectNDC(camera, position);

    float3 worldAxis0 = rotation[0] * scale.x;
    float3 worldAxis1 = rotation[1] * scale.y;
    float3 worldAxis2 = rotation[2] * scale.z;
    float2 p0 = (projectNDC(camera, position + worldAxis0) - center) * viewport * 0.5;
    float2 p1 = (projectNDC(camera, position + worldAxis1) - center) * viewport * 0.5;
    float2 p2 = (projectNDC(camera, position + worldAxis2) - center) * viewport * 0.5;

    float a = dot(float3(p0.x, p1.x, p2.x), float3(p0.x, p1.x, p2.x));
    float b = dot(float3(p0.x, p1.x, p2.x), float3(p0.y, p1.y, p2.y));
    float d = dot(float3(p0.y, p1.y, p2.y), float3(p0.y, p1.y, p2.y));
    float trace = a + d;
    float delta = sqrt(max((a - d) * (a - d) * 0.25 + b * b, 0.0));
    float lambda0 = max(trace * 0.5 + delta, 1.0);
    float lambda1 = max(trace * 0.5 - delta, 1.0);
    float2 major = normalize(abs(b) > 0.00001 ? float2(lambda0 - d, b) : (a >= d ? float2(1.0, 0.0) : float2(0.0, 1.0)));
    float2 minor = float2(-major.y, major.x);

    float2 radii = sqrt(float2(lambda0, lambda1)) * 3.0;
    radii = clamp(radii, float2(1.0), float2(camera.viewportAndRadius.z));
    axis0 = major * radii.x / viewport * 2.0;
    axis1 = minor * radii.y / viewport * 2.0;
}

kernel void depthKeyKernel(
    device const PackedSplat *splats [[buffer(0)]],
    device SortPair *pairs [[buffer(1)]],
    constant CameraUniforms &camera [[buffer(2)]],
    constant SortConstants &constants [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= constants.paddedCount) {
        return;
    }
    if (gid >= constants.count) {
        pairs[gid] = SortPair{0u, 0xffffffffu};
        return;
    }
    uint splatIndex = gid;
    if (constants.sourceCount > constants.count && constants.count > 0) {
        splatIndex = min(constants.sourceCount - 1, (uint)(((float)gid * (float)constants.sourceCount) / (float)constants.count));
    }
    float4 world = float4(splats[splatIndex].positionAndOpacity.xyz, 1.0);
    float4 view = camera.viewMatrix * world;
    float depth = max(-view.z, 0.0);
    uint key = min((uint)(depth * 100000.0), 0xfffffffeu);
    pairs[gid] = SortPair{key, splatIndex};
}

kernel void projectSplatsKernel(
    device const PackedSplat *splats [[buffer(0)]],
    device const SortPair *pairs [[buffer(1)]],
    device ProjectedSplat *projected [[buffer(2)]],
    constant CameraUniforms &camera [[buffer(3)]],
    constant SortConstants &constants [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= constants.count) {
        return;
    }

    uint splatIndex = pairs[gid].index;
    if (splatIndex == 0xffffffffu) {
        projected[gid].clipCenter = float4(2.0, 2.0, 0.0, 1.0);
        projected[gid].axis0Opacity = float4(0.0);
        projected[gid].axis1 = float4(0.0);
        projected[gid].color = float4(0.0);
        return;
    }

    PackedSplat splat = splats[splatIndex];
    float3 position = splat.positionAndOpacity.xyz;
    float4 viewCenter = camera.viewMatrix * float4(position, 1.0);
    float4 clipCenter = camera.projectionMatrix * viewCenter;
    bool culled = false;
    if (camera.viewportAndRadius.w > 0.5) {
        float margin = 1.25;
        culled = clipCenter.w <= 0.001 ||
            clipCenter.x < -clipCenter.w * margin ||
            clipCenter.x > clipCenter.w * margin ||
            clipCenter.y < -clipCenter.w * margin ||
            clipCenter.y > clipCenter.w * margin;
    }

    if (culled) {
        projected[gid].clipCenter = float4(2.0, 2.0, 0.0, 1.0);
        projected[gid].axis0Opacity = float4(0.0);
        projected[gid].axis1 = float4(0.0);
        projected[gid].color = float4(0.0);
        return;
    }

    float2 axis0;
    float2 axis1;
    covarianceEllipse(camera, splat, axis0, axis1);
    projected[gid].clipCenter = clipCenter;
    projected[gid].axis0Opacity = float4(axis0, splat.positionAndOpacity.w, 0.0);
    projected[gid].axis1 = float4(axis1, 0.0, 0.0);
    projected[gid].color = splat.color;
}

kernel void bitonicSortKernel(
    device SortPair *pairs [[buffer(0)]],
    constant SortConstants &constants [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    uint i = gid;
    if (i >= constants.paddedCount) {
        return;
    }

    uint ixj = i ^ constants.j;
    if (ixj <= i || ixj >= constants.paddedCount) {
        return;
    }

    SortPair a = pairs[i];
    SortPair b = pairs[ixj];
    bool descendingPhase = (i & constants.k) == 0;
    bool shouldSwap = descendingPhase ? (a.key < b.key) : (a.key > b.key);
    if (a.key == b.key) {
        shouldSwap = descendingPhase ? (a.index < b.index) : (a.index > b.index);
    }

    if (shouldSwap) {
        pairs[i] = b;
        pairs[ixj] = a;
    }
}

vertex VertexOut projectedSplatVertex(
    uint vertexID [[vertex_id]],
    uint instanceID [[instance_id]],
    device const ProjectedSplat *projected [[buffer(0)]]
) {
    constexpr float2 corners[6] = {
        float2(-1.0, -1.0),
        float2( 1.0, -1.0),
        float2(-1.0,  1.0),
        float2( 1.0, -1.0),
        float2( 1.0,  1.0),
        float2(-1.0,  1.0)
    };

    ProjectedSplat splat = projected[instanceID];
    float2 corner = corners[vertexID];
    float2 ndcOffset = splat.axis0Opacity.xy * corner.x + splat.axis1.xy * corner.y;
    VertexOut out;
    out.position = splat.clipCenter + float4(ndcOffset * splat.clipCenter.w, 0.0, 0.0);
    out.local = corner * 3.0;
    out.color = splat.color;
    out.opacity = splat.axis0Opacity.z;
    return out;
}

vertex VertexOut splatVertex(
    uint vertexID [[vertex_id]],
    uint instanceID [[instance_id]],
    device const PackedSplat *splats [[buffer(0)]],
    constant CameraUniforms &camera [[buffer(1)]],
    device const SortPair *pairs [[buffer(2)]]
) {
    constexpr float2 corners[6] = {
        float2(-1.0, -1.0),
        float2( 1.0, -1.0),
        float2(-1.0,  1.0),
        float2( 1.0, -1.0),
        float2( 1.0,  1.0),
        float2(-1.0,  1.0)
    };

    uint splatIndex = pairs[instanceID].index;
    VertexOut out;
    if (splatIndex == 0xffffffffu) {
        out.position = float4(2.0, 2.0, 0.0, 1.0);
        out.local = float2(8.0, 8.0);
        out.color = float4(0.0);
        out.opacity = 0.0;
        return out;
    }

    PackedSplat splat = splats[splatIndex];
    float3 position = splat.positionAndOpacity.xyz;
    float opacity = splat.positionAndOpacity.w;
    float3 scale = max(splat.scaleAndFlags.xyz, float3(0.0001));
    float2 corner = corners[vertexID];

    float4 viewCenter = camera.viewMatrix * float4(position, 1.0);
    float4 clipCenter = camera.projectionMatrix * viewCenter;
    float depth = max(-viewCenter.z, 0.001);

    if (camera.viewportAndRadius.w > 0.5) {
        float margin = 1.25;
        if (clipCenter.w <= 0.001 ||
            clipCenter.x < -clipCenter.w * margin ||
            clipCenter.x > clipCenter.w * margin ||
            clipCenter.y < -clipCenter.w * margin ||
            clipCenter.y > clipCenter.w * margin) {
            out.position = float4(2.0, 2.0, 0.0, 1.0);
            out.local = float2(8.0, 8.0);
            out.color = float4(0.0);
            out.opacity = 0.0;
            return out;
        }
    }

    float3x3 rotation = quatToMatrix(splat.rotation);
    float3 axis0 = rotation[0] * scale.x;
    float3 axis1 = rotation[1] * scale.y;
    float3 viewAxis0 = (camera.viewMatrix * float4(axis0, 0.0)).xyz;
    float3 viewAxis1 = (camera.viewMatrix * float4(axis1, 0.0)).xyz;

    float focalPixels = abs(camera.projectionMatrix[1][1]) * camera.viewportAndRadius.y * 0.5;
    float2 radiusPixels = float2(length(viewAxis0.xy), length(viewAxis1.xy)) * focalPixels / depth * 3.0;
    radiusPixels = clamp(radiusPixels, float2(1.0), float2(camera.viewportAndRadius.z));
    float2 ndcOffset = corner * radiusPixels / max(camera.viewportAndRadius.xy, float2(1.0)) * 2.0;

    out.position = clipCenter + float4(ndcOffset * clipCenter.w, 0.0, 0.0);
    out.local = corner * 3.0;
    out.color = splat.color;
    out.opacity = opacity;
    return out;
}

fragment float4 splatFragment(VertexOut in [[stage_in]]) {
    float falloff = exp(-0.5 * dot(in.local, in.local));
    float alpha = clamp(in.opacity * falloff, 0.0, 1.0);
    if (alpha < 0.003) {
        discard_fragment();
    }
    return float4(in.color.rgb * alpha, alpha);
}
"""#
}
