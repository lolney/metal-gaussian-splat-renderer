import Foundation

enum ShaderSource {
    static let metal = #"""
#include <metal_stdlib>
using namespace metal;

struct PackedSplat {
    float4 positionAndOpacity;
    float4 covarianceA;
    float4 covarianceB;
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

struct RadixConstants {
    uint count;
    uint blockSize;
    uint blockCount;
    uint shift;
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

static float covMul(float3 lhs, float3 rhs, float xx, float xy, float xz, float yy, float yz, float zz) {
    float3 product = float3(
        xx * rhs.x + xy * rhs.y + xz * rhs.z,
        xy * rhs.x + yy * rhs.y + yz * rhs.z,
        xz * rhs.x + yz * rhs.y + zz * rhs.z
    );
    return dot(lhs, product);
}

static void covarianceEllipse(
    constant CameraUniforms &camera,
    PackedSplat splat,
    thread float2 &axis0,
    thread float2 &axis1
) {
    float2 viewport = max(camera.viewportAndRadius.xy, float2(1.0));
    float4 viewCenter = camera.viewMatrix * float4(splat.positionAndOpacity.xyz, 1.0);
    float depth = max(-viewCenter.z, 0.0001);
    float fx = abs(camera.projectionMatrix[0][0]) * viewport.x * 0.5;
    float fy = abs(camera.projectionMatrix[1][1]) * viewport.y * 0.5;
    float3 jacobianX = float3(fx / depth, 0.0, fx * viewCenter.x / (depth * depth));
    float3 jacobianY = float3(0.0, fy / depth, fy * viewCenter.y / (depth * depth));

    float3 viewRow0 = float3(camera.viewMatrix[0][0], camera.viewMatrix[1][0], camera.viewMatrix[2][0]);
    float3 viewRow1 = float3(camera.viewMatrix[0][1], camera.viewMatrix[1][1], camera.viewMatrix[2][1]);
    float3 viewRow2 = float3(camera.viewMatrix[0][2], camera.viewMatrix[1][2], camera.viewMatrix[2][2]);
    float xx = splat.covarianceA.x;
    float xy = splat.covarianceA.y;
    float xz = splat.covarianceA.z;
    float yy = splat.covarianceA.w;
    float yz = splat.covarianceB.x;
    float zz = splat.covarianceB.y;
    float c00 = covMul(viewRow0, viewRow0, xx, xy, xz, yy, yz, zz);
    float c01 = covMul(viewRow0, viewRow1, xx, xy, xz, yy, yz, zz);
    float c02 = covMul(viewRow0, viewRow2, xx, xy, xz, yy, yz, zz);
    float c11 = covMul(viewRow1, viewRow1, xx, xy, xz, yy, yz, zz);
    float c12 = covMul(viewRow1, viewRow2, xx, xy, xz, yy, yz, zz);
    float c22 = covMul(viewRow2, viewRow2, xx, xy, xz, yy, yz, zz);
    float a = covMul(jacobianX, jacobianX, c00, c01, c02, c11, c12, c22);
    float b = covMul(jacobianX, jacobianY, c00, c01, c02, c11, c12, c22);
    float d = covMul(jacobianY, jacobianY, c00, c01, c02, c11, c12, c22);

    float mid = (a + d) * 0.5;
    float radius = length(float2((a - d) * 0.5, b));
    float lambda0 = mid + radius;
    float lambda1 = mid - radius;
    if (lambda1 < 0.0 || !isfinite(lambda0) || !isfinite(lambda1)) {
        axis0 = float2(0.0);
        axis1 = float2(0.0);
        return;
    }

    float2 diagonal = normalize(abs(b) > 0.000001 || abs(lambda0 - a) > 0.000001
        ? float2(b, lambda0 - a)
        : (a >= d ? float2(1.0, 0.0) : float2(0.0, 1.0)));
    float majorRadius = min(sqrt(2.0 * max(lambda0, 0.0)), camera.viewportAndRadius.z);
    float minorRadius = min(sqrt(2.0 * max(lambda1, 0.0)), camera.viewportAndRadius.z);
    float2 majorAxis = majorRadius * diagonal;
    float2 minorAxis = minorRadius * float2(diagonal.y, -diagonal.x);
    axis0 = majorAxis / viewport;
    axis1 = minorAxis / viewport;
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

kernel void radixHistogramKernel(
    device const SortPair *source [[buffer(0)]],
    device atomic_uint *histograms [[buffer(1)]],
    constant RadixConstants &constants [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint blockID [[threadgroup_position_in_grid]]
) {
    if (gid >= constants.count) {
        return;
    }
    uint digit = (source[gid].key >> constants.shift) & 0xffu;
    atomic_fetch_add_explicit(&histograms[digit * constants.blockCount + blockID], 1u, memory_order_relaxed);
}

kernel void radixPrefixKernel(
    device const atomic_uint *histograms [[buffer(0)]],
    device uint *blockOffsets [[buffer(1)]],
    device uint *bucketTotals [[buffer(2)]],
    constant RadixConstants &constants [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= 256) {
        return;
    }

    uint bucketOffset = 0;
    for (uint block = 0; block < constants.blockCount; ++block) {
        uint histogramIndex = gid * constants.blockCount + block;
        blockOffsets[histogramIndex] = bucketOffset;
        bucketOffset += atomic_load_explicit(&histograms[histogramIndex], memory_order_relaxed);
    }
    bucketTotals[gid] = bucketOffset;
}

kernel void radixBucketStartKernel(
    device const uint *bucketTotals [[buffer(0)]],
    device uint *bucketStarts [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0) {
        return;
    }
    uint globalOffset = 0;
    for (int bucket = 255; bucket >= 0; --bucket) {
        bucketStarts[bucket] = globalOffset;
        globalOffset += bucketTotals[bucket];
    }
}

kernel void radixScatterKernel(
    device const SortPair *source [[buffer(0)]],
    device SortPair *destination [[buffer(1)]],
    device const uint *blockOffsets [[buffer(2)]],
    device const uint *bucketStarts [[buffer(3)]],
    constant RadixConstants &constants [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint localID [[thread_index_in_threadgroup]],
    uint blockID [[threadgroup_position_in_grid]]
) {
    threadgroup uint localDigits[256];
    threadgroup SortPair localPairs[256];

    uint blockStart = blockID * constants.blockSize;
    uint activeCount = min(constants.blockSize, constants.count - min(blockStart, constants.count));
    bool active = localID < activeCount && gid < constants.count;
    SortPair pair = active ? source[gid] : SortPair{0u, 0xffffffffu};
    uint digit = (pair.key >> constants.shift) & 0xffu;
    localPairs[localID] = pair;
    localDigits[localID] = digit;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (!active) {
        return;
    }

    uint localRank = 0;
    for (uint i = 0; i < localID; ++i) {
        localRank += localDigits[i] == digit ? 1u : 0u;
    }
    uint bucketIndex = digit * constants.blockCount + blockID;
    uint outputIndex = bucketStarts[digit] + blockOffsets[bucketIndex] + localRank;
    destination[outputIndex] = pair;
}

vertex VertexOut projectedSplatVertex(
    uint vertexID [[vertex_id]],
    uint instanceID [[instance_id]],
    device const ProjectedSplat *projected [[buffer(0)]]
) {
    constexpr float2 corners[6] = {
        float2(-2.0, -2.0),
        float2( 2.0, -2.0),
        float2(-2.0,  2.0),
        float2( 2.0, -2.0),
        float2( 2.0,  2.0),
        float2(-2.0,  2.0)
    };

    ProjectedSplat splat = projected[instanceID];
    float2 corner = corners[vertexID];
    float2 ndcOffset = splat.axis0Opacity.xy * corner.x + splat.axis1.xy * corner.y;
    VertexOut out;
    out.position = splat.clipCenter + float4(ndcOffset * splat.clipCenter.w, 0.0, 0.0);
    out.local = corner;
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
        float2(-2.0, -2.0),
        float2( 2.0, -2.0),
        float2(-2.0,  2.0),
        float2( 2.0, -2.0),
        float2( 2.0,  2.0),
        float2(-2.0,  2.0)
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
    float2 corner = corners[vertexID];

    float4 viewCenter = camera.viewMatrix * float4(position, 1.0);
    float4 clipCenter = camera.projectionMatrix * viewCenter;

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

    float2 axis0;
    float2 axis1;
    covarianceEllipse(camera, splat, axis0, axis1);
    float2 ndcOffset = axis0 * corner.x + axis1 * corner.y;

    out.position = clipCenter + float4(ndcOffset * clipCenter.w, 0.0, 0.0);
    out.local = corner;
    out.color = splat.color;
    out.opacity = opacity;
    return out;
}

fragment float4 splatFragment(VertexOut in [[stage_in]]) {
    float exponent = -dot(in.local, in.local);
    if (exponent < -4.0) {
        discard_fragment();
    }
    float alpha = clamp(exp(exponent) * in.opacity, 0.0, 1.0);
    return float4(in.color.rgb * alpha, alpha);
}
"""#
}
