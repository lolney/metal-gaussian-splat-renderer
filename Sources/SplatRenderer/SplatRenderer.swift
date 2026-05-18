import Foundation
import Metal
import MetalKit
import QuartzCore
import simd

public final class SplatRenderer {
    private static let maxInflightBuffers = 3

    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue

    private let renderPipeline: MTLRenderPipelineState
    private let depthKeyPipeline: MTLComputePipelineState
    private let sortPipeline: MTLComputePipelineState
    private var sceneResources: SceneResources?
    private var cpuSortSplats: [Splat] = []
    private var frameResources: [FrameResources] = []
    private var frameIndex = 0
    private var frameID = 0
    private var offscreenTexture: MTLTexture?
    private var offscreenSize = SIMD2<Int32>(0, 0)
    private let inflightSemaphore = DispatchSemaphore(value: SplatRenderer.maxInflightBuffers)

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

        var resources: [FrameResources] = []
        for index in 0..<Self.maxInflightBuffers {
            guard let buffer = device.makeBuffer(length: MemoryLayout<CameraUniforms>.stride, options: .storageModeShared) else {
                throw SplatError.metalUnavailable
            }
            buffer.label = "SplatRenderer.cameraUniforms.\(index)"
            resources.append(FrameResources(uniformBuffer: buffer))
        }
        frameResources = resources
    }

    public func load(scene: SplatScene) throws {
        waitForInflightResources()
        cpuSortSplats = scene.splats
        sceneDiagnostics = scene.diagnostics

        sceneResources = try makeSceneResources(packedSplats: scene.packedSplats())
        resetFrameResourceState()
        frameIndex = 0
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

        inflightSemaphore.wait()
        var commandBufferCommitted = false
        defer {
            if !commandBufferCommitted {
                inflightSemaphore.signal()
            }
        }

        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw SplatError.metalUnavailable
        }
        commandBuffer.label = "SplatRenderer.frame.\(frameID)"
        let semaphore = inflightSemaphore
        commandBuffer.addCompletedHandler { _ in
            semaphore.signal()
        }

        let splatCount = sceneResources?.splatCount ?? 0
        let requestedDrawCount = options.maxVisibleSplats > 0 ? min(options.maxVisibleSplats, splatCount) : splatCount
        let drawCount = max(0, requestedDrawCount)
        let sortPaddedCount = max(1, nextPowerOfTwo(max(drawCount, 1)))
        let selectedSortMode: SortMode
        if splatCount == 0 {
            selectedSortMode = .unsorted
        } else {
            selectedSortMode = options.sortMode
        }

        guard let sceneResources else {
            clearOnly(commandBuffer: commandBuffer, descriptor: renderPassDescriptor, drawable: drawable)
            commandBufferCommitted = true
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
                sortMode: selectedSortMode,
                estimatedMemoryBytes: 0,
                estimatedMemoryBandwidthGBps: nil
            )
        }
        let splatBuffer = sceneResources.splatBuffer

        let resourceIndex = frameIndex % max(1, frameResources.count)
        frameIndex += 1
        var frameResource = frameResources[resourceIndex]
        let requestedOrderCapacity = selectedSortMode == .gpu ? sortPaddedCount : max(drawCount, 1)
        let orderAllocation = try orderBuffer(for: selectedSortMode, paddedCount: requestedOrderCapacity, frameResource: &frameResource)
        let orderBuffer = orderAllocation.buffer
        let orderCapacity = orderAllocation.capacity

        let uniformBuffer = frameResource.uniformBuffer
        var uniforms = CameraUniforms(camera: camera, maxSplatRadius: options.maxSplatRadius, enableCulling: options.enableCulling)
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
                count: drawCount,
                paddedCount: sortPaddedCount,
                sourceCount: splatCount
            )
            depthKeyMilliseconds = milliseconds(since: depthStart)

            let sortStart = CFAbsoluteTimeGetCurrent()
            encodeBitonicSort(commandBuffer: commandBuffer, orderBuffer: orderBuffer, count: drawCount, paddedCount: sortPaddedCount, sourceCount: splatCount)
            sortMilliseconds = milliseconds(since: sortStart)
        } else {
            let sortStart = CFAbsoluteTimeGetCurrent()
            if selectedSortMode == .cpu {
                updateCPUOrder(camera: camera, orderBuffer: orderBuffer, drawCount: drawCount, paddedCount: orderCapacity)
            } else {
                updateUnsortedOrderIfNeeded(
                    mode: selectedSortMode,
                    orderBuffer: orderBuffer,
                    splatCount: splatCount,
                    drawCount: drawCount,
                    paddedCount: orderCapacity,
                    frameResource: &frameResource
                )
            }
            sortMilliseconds = milliseconds(since: sortStart)
        }
        frameResources[resourceIndex] = frameResource

        let drawStart = CFAbsoluteTimeGetCurrent()
        let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor)
        encoder?.label = "SplatRenderer.drawSplats"
        encoder?.setRenderPipelineState(renderPipeline)
        encoder?.setViewport(MTLViewport(originX: 0, originY: 0, width: Double(viewportSize.x), height: Double(viewportSize.y), znear: 0, zfar: 1))
        encoder?.setVertexBuffer(splatBuffer, offset: 0, index: 0)
        encoder?.setVertexBuffer(uniformBuffer, offset: 0, index: 1)
        encoder?.setVertexBuffer(orderBuffer, offset: 0, index: 2)
        encoder?.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6, instanceCount: drawCount)
        encoder?.endEncoding()
        let drawMilliseconds = milliseconds(since: drawStart)

        let presentStart = CFAbsoluteTimeGetCurrent()
        if let drawable {
            commandBuffer.present(drawable)
        }
        let presentMilliseconds = milliseconds(since: presentStart)

        commandBuffer.commit()
        commandBufferCommitted = true
        if options.enableProfiling || options.waitForGPU || drawable == nil {
            commandBuffer.waitUntilCompleted()
        }

        let gpuMilliseconds: Double?
        if commandBuffer.gpuEndTime > commandBuffer.gpuStartTime {
            gpuMilliseconds = (commandBuffer.gpuEndTime - commandBuffer.gpuStartTime) * 1000
        } else {
            gpuMilliseconds = nil
        }
        let estimatedMemoryBytes = estimateMemoryBytes(mode: selectedSortMode, drawCount: drawCount, sortPaddedCount: sortPaddedCount)
        let estimatedBandwidthGBps: Double?
        if let gpuMilliseconds, gpuMilliseconds > 0 {
            estimatedBandwidthGBps = Double(estimatedMemoryBytes) / (gpuMilliseconds / 1000) / 1_000_000_000
        } else {
            estimatedBandwidthGBps = nil
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
            visibleSplats: drawCount,
            totalSplats: splatCount,
            sortMode: selectedSortMode,
            estimatedMemoryBytes: estimatedMemoryBytes,
            estimatedMemoryBandwidthGBps: estimatedBandwidthGBps
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
        paddedCount: Int,
        sourceCount: Int
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "SplatRenderer.depthKeys"
        encoder.setComputePipelineState(depthKeyPipeline)
        encoder.setBuffer(splatBuffer, offset: 0, index: 0)
        encoder.setBuffer(orderBuffer, offset: 0, index: 1)
        encoder.setBuffer(uniformBuffer, offset: 0, index: 2)
        var constants = SortConstants(count: UInt32(count), paddedCount: UInt32(paddedCount), j: 0, k: 0, sourceCount: UInt32(sourceCount))
        encoder.setBytes(&constants, length: MemoryLayout<SortConstants>.stride, index: 3)
        dispatch(encoder: encoder, pipeline: depthKeyPipeline, count: paddedCount)
        encoder.endEncoding()
    }

    private func encodeBitonicSort(commandBuffer: MTLCommandBuffer, orderBuffer: MTLBuffer, count: Int, paddedCount: Int, sourceCount: Int) {
        var k = 2
        while k <= paddedCount {
            var j = k / 2
            while j > 0 {
                guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
                encoder.label = "SplatRenderer.bitonicSort.k\(k).j\(j)"
                encoder.setComputePipelineState(sortPipeline)
                encoder.setBuffer(orderBuffer, offset: 0, index: 0)
                var constants = SortConstants(count: UInt32(count), paddedCount: UInt32(paddedCount), j: UInt32(j), k: UInt32(k), sourceCount: UInt32(sourceCount))
                encoder.setBytes(&constants, length: MemoryLayout<SortConstants>.stride, index: 1)
                dispatch(encoder: encoder, pipeline: sortPipeline, count: paddedCount)
                encoder.endEncoding()
                j /= 2
            }
            k *= 2
        }
    }

    private func updateCPUOrder(camera: Camera, orderBuffer: MTLBuffer, drawCount: Int, paddedCount: Int) {
        let sorted = SplatScene.sortedIndices(for: cpuSortSplats, camera: camera)
        let pairsPointer = orderBuffer.contents().bindMemory(to: SortPair.self, capacity: paddedCount)
        let writableCount = min(drawCount, sorted.count, paddedCount)
        for offset in 0..<paddedCount {
            if offset < writableCount {
                let index = sorted[offset]
                pairsPointer[offset] = SortPair(key: UInt32(sorted.count - offset), index: index)
            } else {
                pairsPointer[offset] = SortPair(key: 0, index: UInt32.max)
            }
        }
    }

    private func updateUnsortedOrderIfNeeded(
        mode: SortMode,
        orderBuffer: MTLBuffer,
        splatCount: Int,
        drawCount: Int,
        paddedCount: Int,
        frameResource: inout FrameResources
    ) {
        guard frameResource.modeStates[mode]?.preparedDrawCount != drawCount else { return }
        writeUnsortedOrder(orderBuffer: orderBuffer, splatCount: splatCount, drawCount: drawCount, paddedCount: paddedCount)
        frameResource.modeStates[mode]?.preparedDrawCount = drawCount
    }

    private func writeUnsortedOrder(orderBuffer: MTLBuffer, splatCount: Int, drawCount: Int, paddedCount: Int) {
        let pairsPointer = orderBuffer.contents().bindMemory(to: SortPair.self, capacity: paddedCount)
        let writableCount = min(drawCount, paddedCount)
        if splatCount > 0, writableCount > 0 {
            let stride = Double(splatCount) / Double(writableCount)
            for offset in 0..<writableCount {
                let index = min(splatCount - 1, Int(Double(offset) * stride))
                pairsPointer[offset] = SortPair(key: UInt32(splatCount - index), index: UInt32(index))
            }
        }
        if writableCount < paddedCount {
            for offset in writableCount..<paddedCount {
                pairsPointer[offset] = SortPair(key: 0, index: UInt32.max)
            }
        }
    }

    private func orderBuffer(for mode: SortMode, paddedCount: Int, frameResource: inout FrameResources) throws -> OrderAllocation {
        if frameResource.activeMode != mode {
            frameResource.modeStates.removeAll(keepingCapacity: true)
            frameResource.activeMode = mode
        }
        if let state = frameResource.modeStates[mode], state.paddedCount >= paddedCount {
            return OrderAllocation(buffer: state.orderBuffer, capacity: state.paddedCount)
        }
        guard let buffer = device.makeBuffer(length: paddedCount * MemoryLayout<SortPair>.stride, options: .storageModeShared) else {
            throw SplatError.metalUnavailable
        }
        buffer.label = "SplatRenderer.order.\(mode.rawValue).frame"
        frameResource.modeStates[mode] = ModeState(orderBuffer: buffer, paddedCount: paddedCount, preparedDrawCount: nil)
        return OrderAllocation(buffer: buffer, capacity: paddedCount)
    }

    private func makeSceneResources(packedSplats: [PackedSplat]) throws -> SceneResources {
        let splatCount = packedSplats.count
        let splatLength = max(1, splatCount) * MemoryLayout<PackedSplat>.stride
        guard let splatBuffer = device.makeBuffer(length: splatLength, options: .storageModePrivate) else {
            throw SplatError.metalUnavailable
        }
        splatBuffer.label = "SplatRenderer.scene.splats.private"

        guard let stagingBuffer = device.makeBuffer(length: splatLength, options: .storageModeShared) else {
            throw SplatError.metalUnavailable
        }
        stagingBuffer.label = "SplatRenderer.scene.splats.staging"
        if !packedSplats.isEmpty {
            packedSplats.withUnsafeBytes { bytes in
                _ = memcpy(stagingBuffer.contents(), bytes.baseAddress, bytes.count)
            }
        }

        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw SplatError.metalUnavailable
        }
        commandBuffer.label = "SplatRenderer.sceneUpload"
        if let encoder = commandBuffer.makeBlitCommandEncoder() {
            encoder.label = "SplatRenderer.uploadSplats"
            encoder.copy(from: stagingBuffer, sourceOffset: 0, to: splatBuffer, destinationOffset: 0, size: splatLength)
            encoder.endEncoding()
        }
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        return SceneResources(splatBuffer: splatBuffer, splatCount: splatCount, splatBufferLength: splatLength)
    }

    private func resetFrameResourceState() {
        for index in frameResources.indices {
            frameResources[index].modeStates.removeAll(keepingCapacity: false)
            frameResources[index].activeMode = nil
        }
    }

    private func waitForInflightResources() {
        for _ in 0..<Self.maxInflightBuffers {
            inflightSemaphore.wait()
        }
        for _ in 0..<Self.maxInflightBuffers {
            inflightSemaphore.signal()
        }
    }

    private func estimateMemoryBytes(mode: SortMode, drawCount: Int, sortPaddedCount: Int) -> UInt64 {
        let splatStride = UInt64(MemoryLayout<PackedSplat>.stride)
        let pairStride = UInt64(MemoryLayout<SortPair>.stride)
        let vertexShaderReads = UInt64(drawCount) * 6 * (splatStride + pairStride)
        let colorWrites = UInt64(drawCount) * 6 * 4
        var total = vertexShaderReads + colorWrites
        if mode == .gpu {
            let sortCount = UInt64(sortPaddedCount)
            let depthKeyBytes = sortCount * (splatStride + pairStride)
            let stages = UInt64(bitonicStageCount(sortPaddedCount))
            let sortBytes = stages * sortCount * pairStride * 2
            total += depthKeyBytes + sortBytes
        }
        return total
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

    private func bitonicStageCount(_ value: Int) -> Int {
        guard value > 1 else { return 0 }
        var levels = 0
        var n = value
        while n > 1 {
            levels += 1
            n >>= 1
        }
        return levels * (levels + 1) / 2
    }

    private func milliseconds(since start: CFAbsoluteTime) -> Double {
        (CFAbsoluteTimeGetCurrent() - start) * 1000
    }
}

private struct ModeState {
    var orderBuffer: MTLBuffer
    var paddedCount: Int
    var preparedDrawCount: Int?
}

private struct OrderAllocation {
    var buffer: MTLBuffer
    var capacity: Int
}

private struct FrameResources {
    var uniformBuffer: MTLBuffer
    var modeStates: [SortMode: ModeState] = [:]
    var activeMode: SortMode?
}

private struct SceneResources {
    var splatBuffer: MTLBuffer
    var splatCount: Int
    var splatBufferLength: Int
}

private struct SortConstants {
    var count: UInt32
    var paddedCount: UInt32
    var j: UInt32
    var k: UInt32
    var sourceCount: UInt32
}
