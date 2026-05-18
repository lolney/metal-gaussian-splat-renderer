import Foundation
import simd

public extension simd_float4x4 {
    static func perspective(fx: Float, fy: Float, sourceWidth: Float, sourceHeight: Float, nearZ: Float, farZ: Float) -> simd_float4x4 {
        let x = 2 * fx / max(sourceWidth, 1)
        let y = 2 * fy / max(sourceHeight, 1)
        let z = farZ / (nearZ - farZ)
        return simd_float4x4(columns: (
            SIMD4<Float>(x, 0, 0, 0),
            SIMD4<Float>(0, y, 0, 0),
            SIMD4<Float>(0, 0, z, -1),
            SIMD4<Float>(0, 0, z * nearZ, 0)
        ))
    }

    static func perspective(fovyRadians: Float, aspect: Float, nearZ: Float, farZ: Float) -> simd_float4x4 {
        let y = 1 / tan(fovyRadians * 0.5)
        let x = y / aspect
        let z = farZ / (nearZ - farZ)
        return simd_float4x4(columns: (
            SIMD4<Float>(x, 0, 0, 0),
            SIMD4<Float>(0, y, 0, 0),
            SIMD4<Float>(0, 0, z, -1),
            SIMD4<Float>(0, 0, z * nearZ, 0)
        ))
    }

    static func lookAt(eye: SIMD3<Float>, center: SIMD3<Float>, up: SIMD3<Float>) -> simd_float4x4 {
        let z = simd_normalize(eye - center)
        let x = simd_normalize(simd_cross(up, z))
        let y = simd_cross(z, x)
        return simd_float4x4(columns: (
            SIMD4<Float>(x.x, y.x, z.x, 0),
            SIMD4<Float>(x.y, y.y, z.y, 0),
            SIMD4<Float>(x.z, y.z, z.z, 0),
            SIMD4<Float>(-simd_dot(x, eye), -simd_dot(y, eye), -simd_dot(z, eye), 1)
        ))
    }
}

public func clamp<T: Comparable>(_ value: T, _ lower: T, _ upper: T) -> T {
    min(max(value, lower), upper)
}

func sigmoid(_ value: Float) -> Float {
    1 / (1 + exp(-value))
}

func normalizedQuaternion(_ q: SIMD4<Float>) -> SIMD4<Float> {
    let length = simd_length(q)
    guard length.isFinite, length > 0 else {
        return SIMD4<Float>(1, 0, 0, 0)
    }
    return q / length
}
