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
    float4 world = float4(splats[gid].positionAndOpacity.xyz, 1.0);
    float4 view = camera.viewMatrix * world;
    float depth = max(-view.z, 0.0);
    uint key = min((uint)(depth * 100000.0), 0xfffffffeu);
    pairs[gid] = SortPair{key, gid};
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
