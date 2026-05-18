import Foundation
import simd

public enum SplatError: Error, LocalizedError {
    case invalidPLY(String)
    case unsupportedPLY(String)
    case missingRequiredFields([String])
    case metalUnavailable
    case shaderBuildFailed(String)
    case noDrawable

    public var errorDescription: String? {
        switch self {
        case .invalidPLY(let message): "Invalid PLY: \(message)"
        case .unsupportedPLY(let message): "Unsupported PLY: \(message)"
        case .missingRequiredFields(let fields): "PLY is missing required fields: \(fields.joined(separator: ", "))"
        case .metalUnavailable: "No Metal device is available."
        case .shaderBuildFailed(let message): "Metal shader build failed: \(message)"
        case .noDrawable: "No render drawable was available."
        }
    }
}

public enum SortMode: String, CaseIterable, Sendable, Identifiable, Codable {
    case unsorted
    case tiled
    case radix
    case bitonic
    case cpu
    case gpu

    public var id: String { rawValue }
    public var displayName: String {
        switch self {
        case .unsorted: "UNSORTED"
        case .tiled: "TILED"
        case .radix: "CPU RADIX"
        case .bitonic: "BITONIC"
        case .cpu: "CPU"
        case .gpu: "GPU RADIX"
        }
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        let rawValue = try container.decode(String.self)
        if rawValue == "none" {
            self = .unsorted
        } else if let mode = SortMode(rawValue: rawValue) {
            self = mode
        } else {
            throw DecodingError.dataCorruptedError(in: container, debugDescription: "Unknown sort mode \(rawValue)")
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(rawValue)
    }
}

public struct RenderOptions: Codable, Sendable, Equatable {
    public var sortMode: SortMode
    public var resolutionScale: Float
    public var sphericalHarmonicsDegree: Int
    public var maxSplatRadius: Float
    public var enableProfiling: Bool
    public var waitForGPU: Bool
    public var enableCulling: Bool
    public var maxVisibleSplats: Int
    public var useProjectionCache: Bool

    public init(
        sortMode: SortMode = .gpu,
        resolutionScale: Float = 1,
        sphericalHarmonicsDegree: Int = 0,
        maxSplatRadius: Float = 72,
        enableProfiling: Bool = true,
        waitForGPU: Bool = false,
        enableCulling: Bool = true,
        maxVisibleSplats: Int = 0,
        useProjectionCache: Bool = false
    ) {
        self.sortMode = sortMode
        self.resolutionScale = resolutionScale
        self.sphericalHarmonicsDegree = sphericalHarmonicsDegree
        self.maxSplatRadius = maxSplatRadius
        self.enableProfiling = enableProfiling
        self.waitForGPU = waitForGPU
        self.enableCulling = enableCulling
        self.maxVisibleSplats = maxVisibleSplats
        self.useProjectionCache = useProjectionCache
    }
}

public struct FrameStats: Codable, Sendable, Equatable, Identifiable {
    public var id: Int
    public var timestamp: TimeInterval
    public var cpuEncodeMilliseconds: Double
    public var gpuFrameMilliseconds: Double?
    public var depthKeyMilliseconds: Double?
    public var sortMilliseconds: Double?
    public var projectionMilliseconds: Double?
    public var drawMilliseconds: Double?
    public var presentMilliseconds: Double?
    public var totalFrameMilliseconds: Double
    public var visibleSplats: Int
    public var totalSplats: Int
    public var sortMode: SortMode
    public var estimatedMemoryBytes: UInt64
    public var estimatedMemoryBandwidthGBps: Double?

    public init(
        id: Int,
        timestamp: TimeInterval = Date().timeIntervalSince1970,
        cpuEncodeMilliseconds: Double,
        gpuFrameMilliseconds: Double?,
        depthKeyMilliseconds: Double?,
        sortMilliseconds: Double?,
        projectionMilliseconds: Double? = nil,
        drawMilliseconds: Double?,
        presentMilliseconds: Double?,
        totalFrameMilliseconds: Double,
        visibleSplats: Int,
        totalSplats: Int,
        sortMode: SortMode,
        estimatedMemoryBytes: UInt64 = 0,
        estimatedMemoryBandwidthGBps: Double? = nil
    ) {
        self.id = id
        self.timestamp = timestamp
        self.cpuEncodeMilliseconds = cpuEncodeMilliseconds
        self.gpuFrameMilliseconds = gpuFrameMilliseconds
        self.depthKeyMilliseconds = depthKeyMilliseconds
        self.sortMilliseconds = sortMilliseconds
        self.projectionMilliseconds = projectionMilliseconds
        self.drawMilliseconds = drawMilliseconds
        self.presentMilliseconds = presentMilliseconds
        self.totalFrameMilliseconds = totalFrameMilliseconds
        self.visibleSplats = visibleSplats
        self.totalSplats = totalSplats
        self.sortMode = sortMode
        self.estimatedMemoryBytes = estimatedMemoryBytes
        self.estimatedMemoryBandwidthGBps = estimatedMemoryBandwidthGBps
    }
}

public struct Splat: Sendable, Equatable {
    public var position: SIMD3<Float>
    public var scale: SIMD3<Float>
    public var rotation: SIMD4<Float>
    public var opacity: Float
    public var color: SIMD3<Float>

    public init(
        position: SIMD3<Float>,
        scale: SIMD3<Float>,
        rotation: SIMD4<Float>,
        opacity: Float,
        color: SIMD3<Float>
    ) {
        self.position = position
        self.scale = scale
        self.rotation = rotation
        self.opacity = opacity
        self.color = color
    }
}

public struct SplatBounds: Codable, Sendable, Equatable {
    public var minimum: [Float]
    public var maximum: [Float]
    public var center: [Float]
    public var radius: Float
}

public struct SplatFieldAvailability: Codable, Sendable, Equatable {
    public var hasSHDC: Bool
    public var hasRGB: Bool
    public var hasScale: Bool
    public var hasRotation: Bool
    public var hasOpacity: Bool
}

public struct SplatDiagnostics: Codable, Sendable, Equatable {
    public var sourceURL: URL?
    public var format: String
    public var vertexCount: Int
    public var fieldAvailability: SplatFieldAvailability
    public var bounds: SplatBounds
    public var warnings: [String]
}

public struct PackedSplat {
    public var positionAndOpacity: SIMD4<Float>
    public var scaleAndFlags: SIMD4<Float>
    public var rotation: SIMD4<Float>
    public var color: SIMD4<Float>

    public init(_ splat: Splat) {
        positionAndOpacity = SIMD4<Float>(splat.position.x, splat.position.y, splat.position.z, splat.opacity)
        scaleAndFlags = SIMD4<Float>(splat.scale.x, splat.scale.y, splat.scale.z, 0)
        rotation = splat.rotation
        color = SIMD4<Float>(splat.color.x, splat.color.y, splat.color.z, 1)
    }
}

public struct Camera: Sendable, Equatable {
    public var viewMatrix: simd_float4x4
    public var projectionMatrix: simd_float4x4
    public var viewportSize: SIMD2<Float>

    public init(viewMatrix: simd_float4x4, projectionMatrix: simd_float4x4, viewportSize: SIMD2<Float>) {
        self.viewMatrix = viewMatrix
        self.projectionMatrix = projectionMatrix
        self.viewportSize = viewportSize
    }
}

public struct CameraUniforms {
    public var viewMatrix: simd_float4x4
    public var projectionMatrix: simd_float4x4
    public var viewProjectionMatrix: simd_float4x4
    public var viewportAndRadius: SIMD4<Float>

    public init(camera: Camera, maxSplatRadius: Float, enableCulling: Bool = true) {
        viewMatrix = camera.viewMatrix
        projectionMatrix = camera.projectionMatrix
        viewProjectionMatrix = camera.projectionMatrix * camera.viewMatrix
        viewportAndRadius = SIMD4<Float>(camera.viewportSize.x, camera.viewportSize.y, maxSplatRadius, enableCulling ? 1 : 0)
    }
}

public struct SortPair {
    public var key: UInt32
    public var index: UInt32

    public init(key: UInt32, index: UInt32) {
        self.key = key
        self.index = index
    }
}
