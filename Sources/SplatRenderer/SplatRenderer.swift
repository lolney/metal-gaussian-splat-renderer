import Foundation
import Metal
import MetalKit
import QuartzCore
import simd

public final class SplatRenderer {
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue

    private let renderPipeline: MTLRenderPipelineState
    private let depthKeyPipeline: MTLComputePipelineState
    private let sortPipeline: MTLComputePipelineState
    private var splatBuffer: MTLBuffer?
    private var orderBuffer: MTLBuffer?
    private var uniformBuffers: [MTLBuffer] = []
    private var packedSplats: [PackedSplat] = []
    private var sourceSplats: [Splat] = []
    private var frameIndex = 0
    private var frameID = 0
    private var offscreenTexture: MTLTexture?
    private var offscreenSize = SIMD2<Int32>(0, 0)
    private let maxInflightBuffers = 3

    public private(set) var sceneDiagnostics: SplatDiagnostics?

    public init(device: MTLDevice = MTLCreateSystemDefaultDevice()!) throws {
        self.device = device
        guard let queue = device.makeCommandQueue() else {
            throw SplatError.metalUnavailable
        }
        commandQueue = queue
        commandQueue.label = "SplatRenderer.commandQueue"

        let library: MTLLibrary
        do {
            library = try device.makeLibrary(source: ShaderSource.metal, options: nil)
        } catch {
            throw SplatError.shaderBuildFailed(error.localizedDescription)
        }

        guard
            let vertex = library.makeFunction(name: "splatVertex"),
            let fragment = library.makeFunction(name: "splatFragment"),
            let depth = library.makeFunction(name: "depthKeyKernel"),
            let sort = library.makeFunction(name: "bitonicSortKernel")
        else {
            throw SplatError.shaderBuildFailed("missing shader entry point")
        }

        let renderDescriptor = MTLRenderPipelineDescriptor()
        renderDescriptor.label = "SplatRenderer.splatRenderPipeline"
        renderDescriptor.vertexFunction = vertex
        renderDescriptor.fragmentFunction = fragment
        renderDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
        renderDescriptor.colorAttachments[0].isBlendingEnabled = true
        renderDescriptor.colorAttachments[0].rgbBlendOperation = .add
        renderDescriptor.colorAttachments[0].alphaBlendOperation = .add
        renderDescriptor.colorAttachments[0].sourceRGBBlendFactor = .one
        renderDescriptor.colorAttachments[0].sourceAlphaBlendFactor = .one
        renderDescriptor.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        renderDescriptor.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
        renderPipeline = try device.makeRenderPipelineState(descriptor: renderDescriptor)
        depthKeyPipeline = try device.makeComputePipelineState(function: depth)
        sortPipeline = try device.makeComputePipelineState(function: sort)

        uniformBuffers = (0..<maxInflightBuffers).compactMap { index in
            let buffer = device.makeBuffer(length: MemoryLayout<CameraUniforms>.stride, options: .storageModeShared)
            buffer?.label = "SplatRenderer.cameraUniforms.\(index)"
            return buffer
        }
    }

    public func load(scene: SplatScene) throws {
        sourceSplats = scene.splats
        packedSplats = scene.packedSplats()
        sceneDiagnostics = scene.diagnostics

        let splatLength = max(1, packedSplats.count) * MemoryLayout<PackedSplat>.stride
        splatBuffer = device.makeBuffer(bytes: packedSplats, length: splatLength, options: .storageModeShared)
        splatBuffer?.label = "SplatRenderer.splats"

        let paddedCount = max(1, nextPowerOfTwo(packedSplats.count))
        let pairs = (0..<paddedCount).map { SortPair(key: 0, index: $0 < packedSplats.count ? UInt32($0) : UInt32.max) }
        orderBuffer = device.makeBuffer(bytes: pairs, length: pairs.count * MemoryLayout<SortPair>.stride, options: .storageModeShared)
        orderBuffer?.label = "SplatRenderer.order"
    }

    public func draw(
        camera: Camera,
        renderPassDescriptor: MTLRenderPassDescriptor,
        drawable: MTLDrawable?,
        viewportSize: SIMD2<Int32>,
        options inputOptions: RenderOptions
    ) throws -> FrameStats {
        var options = inputOptions
        options.resolutionScale = clamp(options.resolutionScale, 0.1, 2)
        let totalStart = CFAbsoluteTimeGetCurrent()
        let encodeStart = CFAbsoluteTimeGetCurrent()
        frameID += 1

        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw SplatError.metalUnavailable
        }
        commandBuffer.label = "SplatRenderer.frame.\(frameID)"

        let splatCount = packedSplats.count
        let paddedCount = max(1, nextPowerOfTwo(splatCount))
        let selectedSortMode = splatCount == 0 ? .cpu : options.sortMode

        guard let splatBuffer, let orderBuffer else {
            clearOnly(commandBuffer: commandBuffer, descriptor: renderPassDescriptor, drawable: drawable)
            return FrameStats(
                id: frameID,
                cpuEncodeMilliseconds: milliseconds(since: encodeStart),
                gpuFrameMilliseconds: nil,
                depthKeyMilliseconds: nil,
                sortMilliseconds: nil,
                drawMilliseconds: nil,
                presentMilliseconds: nil,
                totalFrameMilliseconds: milliseconds(since: totalStart),
                visibleSplats: 0,
                totalSplats: 0,
                sortMode: selectedSortMode
            )
        }

        let uniformBuffer = uniformBuffers[frameIndex % max(1, uniformBuffers.count)]
        frameIndex += 1
        var uniforms = CameraUniforms(camera: camera, maxSplatRadius: options.maxSplatRadius)
        memcpy(uniformBuffer.contents(), &uniforms, MemoryLayout<CameraUniforms>.stride)

        var depthKeyMilliseconds: Double?
        var sortMilliseconds: Double?
        if selectedSortMode == .gpu {
            let depthStart = CFAbsoluteTimeGetCurrent()
            encodeDepthKeys(
                commandBuffer: commandBuffer,
                splatBuffer: splatBuffer,
                orderBuffer: orderBuffer,
                uniformBuffer: uniformBuffer,
                count: splatCount,
                paddedCount: paddedCount
            )
            depthKeyMilliseconds = milliseconds(since: depthStart)

            let sortStart = CFAbsoluteTimeGetCurrent()
            encodeBitonicSort(commandBuffer: commandBuffer, orderBuffer: orderBuffer, count: splatCount, paddedCount: paddedCount)
            sortMilliseconds = milliseconds(since: sortStart)
        } else {
            let sortStart = CFAbsoluteTimeGetCurrent()
            updateCPUOrder(camera: camera, orderBuffer: orderBuffer, paddedCount: paddedCount)
            sortMilliseconds = milliseconds(since: sortStart)
        }

        let drawStart = CFAbsoluteTimeGetCurrent()
        let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor)
        encoder?.label = "SplatRenderer.drawSplats"
        encoder?.setRenderPipelineState(renderPipeline)
        encoder?.setViewport(MTLViewport(originX: 0, originY: 0, width: Double(viewportSize.x), height: Double(viewportSize.y), znear: 0, zfar: 1))
        encoder?.setVertexBuffer(splatBuffer, offset: 0, index: 0)
        encoder?.setVertexBuffer(uniformBuffer, offset: 0, index: 1)
        encoder?.setVertexBuffer(orderBuffer, offset: 0, index: 2)
        encoder?.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6, instanceCount: paddedCount)
        encoder?.endEncoding()
        let drawMilliseconds = milliseconds(since: drawStart)

        let presentStart = CFAbsoluteTimeGetCurrent()
        if let drawable {
            commandBuffer.present(drawable)
        }
        let presentMilliseconds = milliseconds(since: presentStart)

        commandBuffer.commit()
        if options.enableProfiling || options.waitForGPU || drawable == nil {
            commandBuffer.waitUntilCompleted()
        }

        let gpuMilliseconds: Double?
        if commandBuffer.gpuEndTime > commandBuffer.gpuStartTime {
            gpuMilliseconds = (commandBuffer.gpuEndTime - commandBuffer.gpuStartTime) * 1000
        } else {
            gpuMilliseconds = nil
        }

        return FrameStats(
            id: frameID,
            cpuEncodeMilliseconds: milliseconds(since: encodeStart),
            gpuFrameMilliseconds: gpuMilliseconds,
            depthKeyMilliseconds: depthKeyMilliseconds,
            sortMilliseconds: sortMilliseconds,
            drawMilliseconds: drawMilliseconds,
            presentMilliseconds: presentMilliseconds,
            totalFrameMilliseconds: milliseconds(since: totalStart),
            visibleSplats: splatCount,
            totalSplats: splatCount,
            sortMode: selectedSortMode
        )
    }

    public func drawOffscreen(size: SIMD2<Int32>, camera: Camera, options: RenderOptions) throws -> FrameStats {
        let texture = makeOffscreenTexture(size: size)
        let descriptor = MTLRenderPassDescriptor()
        descriptor.colorAttachments[0].texture = texture
        descriptor.colorAttachments[0].loadAction = .clear
        descriptor.colorAttachments[0].storeAction = .store
        descriptor.colorAttachments[0].clearColor = MTLClearColor(red: 0.02, green: 0.025, blue: 0.03, alpha: 1)
        var blockingOptions = options
        blockingOptions.waitForGPU = true
        blockingOptions.enableProfiling = true
        return try draw(camera: camera, renderPassDescriptor: descriptor, drawable: nil, viewportSize: size, options: blockingOptions)
    }

    public func makeDefaultCamera(width: Int, height: Int) -> Camera {
        let bounds = sceneDiagnostics?.bounds
        let centerArray = bounds?.center ?? [0, 0, 0]
        let center = SIMD3<Float>(centerArray[0], centerArray[1], centerArray[2])
        let radius = max(bounds?.radius ?? 1, 0.1)
        let eye = center + SIMD3<Float>(0, 0, radius * 3)
        let view = simd_float4x4.lookAt(eye: eye, center: center, up: SIMD3<Float>(0, 1, 0))
        let aspect = Float(max(width, 1)) / Float(max(height, 1))
        let projection = simd_float4x4.perspective(fovyRadians: 60 * .pi / 180, aspect: aspect, nearZ: 0.001, farZ: radius * 100)
        return Camera(viewMatrix: view, projectionMatrix: projection, viewportSize: SIMD2<Float>(Float(width), Float(height)))
    }

    private func clearOnly(commandBuffer: MTLCommandBuffer, descriptor: MTLRenderPassDescriptor, drawable: MTLDrawable?) {
        let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: descriptor)
        encoder?.label = "SplatRenderer.clear"
        encoder?.endEncoding()
        if let drawable {
            commandBuffer.present(drawable)
        }
        commandBuffer.commit()
    }

    private func encodeDepthKeys(
        commandBuffer: MTLCommandBuffer,
        splatBuffer: MTLBuffer,
        orderBuffer: MTLBuffer,
        uniformBuffer: MTLBuffer,
        count: Int,
        paddedCount: Int
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "SplatRenderer.depthKeys"
        encoder.setComputePipelineState(depthKeyPipeline)
        encoder.setBuffer(splatBuffer, offset: 0, index: 0)
        encoder.setBuffer(orderBuffer, offset: 0, index: 1)
        encoder.setBuffer(uniformBuffer, offset: 0, index: 2)
        var constants = SortConstants(count: UInt32(count), paddedCount: UInt32(paddedCount), j: 0, k: 0)
        encoder.setBytes(&constants, length: MemoryLayout<SortConstants>.stride, index: 3)
        dispatch(encoder: encoder, pipeline: depthKeyPipeline, count: paddedCount)
        encoder.endEncoding()
    }

    private func encodeBitonicSort(commandBuffer: MTLCommandBuffer, orderBuffer: MTLBuffer, count: Int, paddedCount: Int) {
        var k = 2
        while k <= paddedCount {
            var j = k / 2
            while j > 0 {
                guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
                encoder.label = "SplatRenderer.bitonicSort.k\(k).j\(j)"
                encoder.setComputePipelineState(sortPipeline)
                encoder.setBuffer(orderBuffer, offset: 0, index: 0)
                var constants = SortConstants(count: UInt32(count), paddedCount: UInt32(paddedCount), j: UInt32(j), k: UInt32(k))
                encoder.setBytes(&constants, length: MemoryLayout<SortConstants>.stride, index: 1)
                dispatch(encoder: encoder, pipeline: sortPipeline, count: paddedCount)
                encoder.endEncoding()
                j /= 2
            }
            k *= 2
        }
    }

    private func updateCPUOrder(camera: Camera, orderBuffer: MTLBuffer, paddedCount: Int) {
        let sorted = SplatScene.sortedIndices(for: sourceSplats, camera: camera)
        let pairsPointer = orderBuffer.contents().bindMemory(to: SortPair.self, capacity: paddedCount)
        for offset in 0..<paddedCount {
            if offset < sorted.count {
                let index = sorted[offset]
                pairsPointer[offset] = SortPair(key: UInt32(sorted.count - offset), index: index)
            } else {
                pairsPointer[offset] = SortPair(key: 0, index: UInt32.max)
            }
        }
    }

    private func dispatch(encoder: MTLComputeCommandEncoder, pipeline: MTLComputePipelineState, count: Int) {
        let width = min(max(pipeline.threadExecutionWidth, 1), 256)
        let threadsPerGroup = MTLSize(width: width, height: 1, depth: 1)
        let groups = MTLSize(width: (count + width - 1) / width, height: 1, depth: 1)
        encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: threadsPerGroup)
    }

    private func makeOffscreenTexture(size: SIMD2<Int32>) -> MTLTexture {
        if let offscreenTexture, offscreenSize == size {
            return offscreenTexture
        }
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm,
            width: Int(size.x),
            height: Int(size.y),
            mipmapped: false
        )
        descriptor.usage = [.renderTarget, .shaderRead]
        descriptor.storageMode = .private
        let texture = device.makeTexture(descriptor: descriptor)!
        texture.label = "SplatRenderer.offscreen"
        offscreenTexture = texture
        offscreenSize = size
        return texture
    }

    private func nextPowerOfTwo(_ value: Int) -> Int {
        guard value > 1 else { return 1 }
        var power = 1
        while power < value {
            power <<= 1
        }
        return power
    }

    private func milliseconds(since start: CFAbsoluteTime) -> Double {
        (CFAbsoluteTimeGetCurrent() - start) * 1000
    }
}

private struct SortConstants {
    var count: UInt32
    var paddedCount: UInt32
    var j: UInt32
    var k: UInt32
}
