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
    private let projectedRenderPipeline: MTLRenderPipelineState
    private let depthKeyPipeline: MTLComputePipelineState
    private let sortPipeline: MTLComputePipelineState
    private let radixHistogramPipeline: MTLComputePipelineState
    private let radixPrefixPipeline: MTLComputePipelineState
    private let radixBucketStartPipeline: MTLComputePipelineState
    private let radixScatterPipeline: MTLComputePipelineState
    private let projectionPipeline: MTLComputePipelineState
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
            let projectedVertex = library.makeFunction(name: "projectedSplatVertex"),
            let fragment = library.makeFunction(name: "splatFragment"),
            let project = library.makeFunction(name: "projectSplatsKernel"),
            let depth = library.makeFunction(name: "depthKeyKernel"),
            let sort = library.makeFunction(name: "bitonicSortKernel"),
            let radixHistogram = library.makeFunction(name: "radixHistogramKernel"),
            let radixPrefix = library.makeFunction(name: "radixPrefixKernel"),
            let radixBucketStart = library.makeFunction(name: "radixBucketStartKernel"),
            let radixScatter = library.makeFunction(name: "radixScatterKernel")
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

        let projectedDescriptor = MTLRenderPipelineDescriptor()
        projectedDescriptor.label = "SplatRenderer.projectedRenderPipeline"
        projectedDescriptor.vertexFunction = projectedVertex
        projectedDescriptor.fragmentFunction = fragment
        projectedDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
        projectedDescriptor.colorAttachments[0].isBlendingEnabled = true
        projectedDescriptor.colorAttachments[0].rgbBlendOperation = .add
        projectedDescriptor.colorAttachments[0].alphaBlendOperation = .add
        projectedDescriptor.colorAttachments[0].sourceRGBBlendFactor = .one
        projectedDescriptor.colorAttachments[0].sourceAlphaBlendFactor = .one
        projectedDescriptor.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        projectedDescriptor.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
        projectedRenderPipeline = try device.makeRenderPipelineState(descriptor: projectedDescriptor)

        depthKeyPipeline = try device.makeComputePipelineState(function: depth)
        sortPipeline = try device.makeComputePipelineState(function: sort)
        radixHistogramPipeline = try device.makeComputePipelineState(function: radixHistogram)
        radixPrefixPipeline = try device.makeComputePipelineState(function: radixPrefix)
        radixBucketStartPipeline = try device.makeComputePipelineState(function: radixBucketStart)
        radixScatterPipeline = try device.makeComputePipelineState(function: radixScatter)
        projectionPipeline = try device.makeComputePipelineState(function: project)

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

        let plan = makeFramePlan(sceneSplatCount: sceneResources?.splatCount ?? 0, options: options)

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
                sortMode: plan.sortMode,
                estimatedMemoryBytes: 0,
                estimatedMemoryBandwidthGBps: nil
            )
        }
        let splatBuffer = sceneResources.splatBuffer

        let resourceIndex = frameIndex % max(1, frameResources.count)
        frameIndex += 1
        var frameResource = frameResources[resourceIndex]
        let requestedOrderCapacity = plan.sortMode == .bitonic ? plan.sortPaddedCount : max(plan.drawCount, 1)
        let orderAllocation = try orderBuffer(for: plan.sortMode, paddedCount: requestedOrderCapacity, frameResource: &frameResource)
        let orderBuffer = orderAllocation.buffer
        let orderCapacity = orderAllocation.capacity
        let radixResources = try plan.sortMode == .gpu
            ? radixResources(drawCount: max(plan.drawCount, 1), frameResource: &frameResource)
            : nil
        let projectionBuffer = try options.useProjectionCache
            ? projectedBuffer(drawCount: max(plan.drawCount, 1), frameResource: &frameResource)
            : nil

        let uniformBuffer = frameResource.uniformBuffer
        var uniforms = CameraUniforms(camera: camera, maxSplatRadius: options.maxSplatRadius, enableCulling: options.enableCulling)
        memcpy(uniformBuffer.contents(), &uniforms, MemoryLayout<CameraUniforms>.stride)

        var depthKeyMilliseconds: Double?
        var sortMilliseconds: Double?
        if plan.sortMode == .gpu || plan.sortMode == .bitonic {
            let depthStart = CFAbsoluteTimeGetCurrent()
            encodeDepthKeys(
                commandBuffer: commandBuffer,
                splatBuffer: splatBuffer,
                orderBuffer: orderBuffer,
                uniformBuffer: uniformBuffer,
                count: plan.drawCount,
                paddedCount: plan.sortMode == .bitonic ? plan.sortPaddedCount : plan.drawCount,
                sourceCount: plan.totalSplats
            )
            depthKeyMilliseconds = milliseconds(since: depthStart)

            let sortStart = CFAbsoluteTimeGetCurrent()
            if plan.sortMode == .bitonic {
                encodeBitonicSort(commandBuffer: commandBuffer, orderBuffer: orderBuffer, count: plan.drawCount, paddedCount: plan.sortPaddedCount, sourceCount: plan.totalSplats)
            } else if let radixResources {
                encodeRadixSort(commandBuffer: commandBuffer, orderBuffer: orderBuffer, radixResources: radixResources, count: plan.drawCount)
            }
            sortMilliseconds = milliseconds(since: sortStart)
        } else {
            let sortStart = CFAbsoluteTimeGetCurrent()
            if plan.sortMode == .cpu {
                updateCPUOrder(camera: camera, orderBuffer: orderBuffer, drawCount: plan.drawCount, paddedCount: orderCapacity)
            } else if plan.sortMode == .radix {
                updateRadixOrder(camera: camera, orderBuffer: orderBuffer, splatCount: plan.totalSplats, drawCount: plan.drawCount, paddedCount: orderCapacity)
            } else if plan.sortMode == .tiled {
                updateTiledOrder(camera: camera, orderBuffer: orderBuffer, splatCount: plan.totalSplats, drawCount: plan.drawCount, paddedCount: orderCapacity)
            } else {
                updateUnsortedOrderIfNeeded(
                    mode: plan.sortMode,
                    orderBuffer: orderBuffer,
                    splatCount: plan.totalSplats,
                    drawCount: plan.drawCount,
                    paddedCount: orderCapacity,
                    frameResource: &frameResource
                )
            }
            sortMilliseconds = milliseconds(since: sortStart)
        }
        frameResources[resourceIndex] = frameResource

        var projectionMilliseconds: Double?
        if let projectionBuffer {
            let projectionStart = CFAbsoluteTimeGetCurrent()
            encodeProjection(
                commandBuffer: commandBuffer,
                splatBuffer: splatBuffer,
                orderBuffer: orderBuffer,
                projectedBuffer: projectionBuffer,
                uniformBuffer: uniformBuffer,
                count: plan.drawCount,
                sourceCount: plan.totalSplats
            )
            projectionMilliseconds = milliseconds(since: projectionStart)
        }

        let drawStart = CFAbsoluteTimeGetCurrent()
        encodeRaster(
            commandBuffer: commandBuffer,
            renderPassDescriptor: renderPassDescriptor,
            viewportSize: viewportSize,
            splatBuffer: splatBuffer,
            uniformBuffer: uniformBuffer,
            orderBuffer: orderBuffer,
            projectionBuffer: projectionBuffer,
            drawCount: plan.drawCount
        )
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
        let estimatedMemoryBytes = estimateMemoryBytes(mode: plan.sortMode, drawCount: plan.drawCount, sortPaddedCount: plan.sortPaddedCount, usesProjectionCache: projectionBuffer != nil)
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
            projectionMilliseconds: projectionMilliseconds,
            drawMilliseconds: drawMilliseconds,
            presentMilliseconds: presentMilliseconds,
            totalFrameMilliseconds: milliseconds(since: totalStart),
            visibleSplats: plan.drawCount,
            totalSplats: plan.totalSplats,
            sortMode: plan.sortMode,
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

    private func makeFramePlan(sceneSplatCount: Int, options: RenderOptions) -> FramePlan {
        let requestedDrawCount = options.maxVisibleSplats > 0 ? min(options.maxVisibleSplats, sceneSplatCount) : sceneSplatCount
        let drawCount = max(0, requestedDrawCount)
        let sortPaddedCount = max(1, nextPowerOfTwo(max(drawCount, 1)))
        let sortMode: SortMode = sceneSplatCount == 0 ? .unsorted : options.sortMode
        return FramePlan(sortMode: sortMode, drawCount: drawCount, totalSplats: sceneSplatCount, sortPaddedCount: sortPaddedCount)
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

    private func encodeRadixSort(commandBuffer: MTLCommandBuffer, orderBuffer: MTLBuffer, radixResources: RadixResources, count: Int) {
        guard count > 1 else { return }

        let radixBuckets = 256
        let blockSize = 256
        let blockCount = max(1, (count + blockSize - 1) / blockSize)
        let histogramLength = radixBuckets * blockCount * MemoryLayout<UInt32>.stride
        let threadsPerGroup = MTLSize(width: blockSize, height: 1, depth: 1)
        let groups = MTLSize(width: blockCount, height: 1, depth: 1)

        for pass in 0..<4 {
            if let blit = commandBuffer.makeBlitCommandEncoder() {
                blit.label = "SplatRenderer.radix.clear.\(pass)"
                blit.fill(buffer: radixResources.histogramBuffer, range: 0..<histogramLength, value: 0)
                blit.endEncoding()
            }

            let sourceBuffer = pass.isMultiple(of: 2) ? orderBuffer : radixResources.scratchBuffer
            let destinationBuffer = pass.isMultiple(of: 2) ? radixResources.scratchBuffer : orderBuffer
            var constants = RadixConstants(
                count: UInt32(count),
                blockSize: UInt32(blockSize),
                blockCount: UInt32(blockCount),
                shift: UInt32(pass * 8)
            )

            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.label = "SplatRenderer.radix.histogram.\(pass)"
                encoder.setComputePipelineState(radixHistogramPipeline)
                encoder.setBuffer(sourceBuffer, offset: 0, index: 0)
                encoder.setBuffer(radixResources.histogramBuffer, offset: 0, index: 1)
                encoder.setBytes(&constants, length: MemoryLayout<RadixConstants>.stride, index: 2)
                encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: threadsPerGroup)
                encoder.endEncoding()
            }

            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.label = "SplatRenderer.radix.prefix.\(pass)"
                encoder.setComputePipelineState(radixPrefixPipeline)
                encoder.setBuffer(radixResources.histogramBuffer, offset: 0, index: 0)
                encoder.setBuffer(radixResources.offsetBuffer, offset: 0, index: 1)
                encoder.setBuffer(radixResources.bucketTotalBuffer, offset: 0, index: 2)
                encoder.setBytes(&constants, length: MemoryLayout<RadixConstants>.stride, index: 3)
                encoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                encoder.endEncoding()
            }

            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.label = "SplatRenderer.radix.bucketStarts.\(pass)"
                encoder.setComputePipelineState(radixBucketStartPipeline)
                encoder.setBuffer(radixResources.bucketTotalBuffer, offset: 0, index: 0)
                encoder.setBuffer(radixResources.bucketStartBuffer, offset: 0, index: 1)
                encoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
                encoder.endEncoding()
            }

            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.label = "SplatRenderer.radix.scatter.\(pass)"
                encoder.setComputePipelineState(radixScatterPipeline)
                encoder.setBuffer(sourceBuffer, offset: 0, index: 0)
                encoder.setBuffer(destinationBuffer, offset: 0, index: 1)
                encoder.setBuffer(radixResources.offsetBuffer, offset: 0, index: 2)
                encoder.setBuffer(radixResources.bucketStartBuffer, offset: 0, index: 3)
                encoder.setBytes(&constants, length: MemoryLayout<RadixConstants>.stride, index: 4)
                encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: threadsPerGroup)
                encoder.endEncoding()
            }
        }
    }

    private func encodeProjection(
        commandBuffer: MTLCommandBuffer,
        splatBuffer: MTLBuffer,
        orderBuffer: MTLBuffer,
        projectedBuffer: MTLBuffer,
        uniformBuffer: MTLBuffer,
        count: Int,
        sourceCount: Int
    ) {
        guard count > 0, let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "SplatRenderer.projectSplats"
        encoder.setComputePipelineState(projectionPipeline)
        encoder.setBuffer(splatBuffer, offset: 0, index: 0)
        encoder.setBuffer(orderBuffer, offset: 0, index: 1)
        encoder.setBuffer(projectedBuffer, offset: 0, index: 2)
        encoder.setBuffer(uniformBuffer, offset: 0, index: 3)
        var constants = SortConstants(count: UInt32(count), paddedCount: UInt32(count), j: 0, k: 0, sourceCount: UInt32(sourceCount))
        encoder.setBytes(&constants, length: MemoryLayout<SortConstants>.stride, index: 4)
        dispatch(encoder: encoder, pipeline: projectionPipeline, count: count)
        encoder.endEncoding()
    }

    private func encodeRaster(
        commandBuffer: MTLCommandBuffer,
        renderPassDescriptor: MTLRenderPassDescriptor,
        viewportSize: SIMD2<Int32>,
        splatBuffer: MTLBuffer,
        uniformBuffer: MTLBuffer,
        orderBuffer: MTLBuffer,
        projectionBuffer: MTLBuffer?,
        drawCount: Int
    ) {
        let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor)
        encoder?.label = "SplatRenderer.rasterComposite"
        encoder?.setViewport(MTLViewport(originX: 0, originY: 0, width: Double(viewportSize.x), height: Double(viewportSize.y), znear: 0, zfar: 1))
        if let projectionBuffer {
            encoder?.setRenderPipelineState(projectedRenderPipeline)
            encoder?.setVertexBuffer(projectionBuffer, offset: 0, index: 0)
        } else {
            encoder?.setRenderPipelineState(renderPipeline)
            encoder?.setVertexBuffer(splatBuffer, offset: 0, index: 0)
            encoder?.setVertexBuffer(uniformBuffer, offset: 0, index: 1)
            encoder?.setVertexBuffer(orderBuffer, offset: 0, index: 2)
        }
        encoder?.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6, instanceCount: drawCount)
        encoder?.endEncoding()
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

    private func updateRadixOrder(camera: Camera, orderBuffer: MTLBuffer, splatCount: Int, drawCount: Int, paddedCount: Int) {
        let pairsPointer = orderBuffer.contents().bindMemory(to: SortPair.self, capacity: paddedCount)
        let writableCount = min(drawCount, paddedCount)
        guard splatCount > 0, writableCount > 0 else {
            for offset in 0..<paddedCount {
                pairsPointer[offset] = SortPair(key: 0, index: UInt32.max)
            }
            return
        }

        var pairs = Array(repeating: SortPair(key: 0, index: 0), count: writableCount)
        var scratch = pairs
        let sampleStride = Double(splatCount) / Double(writableCount)
        for offset in 0..<writableCount {
            let sourceIndex = min(splatCount - 1, Int(Double(offset) * sampleStride))
            let position = cpuSortSplats[sourceIndex].position
            let world = SIMD4<Float>(position.x, position.y, position.z, 1)
            let view = camera.viewMatrix * world
            let depth = max(-view.z, 0)
            pairs[offset] = SortPair(key: min(UInt32(depth * 100_000), 0xffff_fffe), index: UInt32(sourceIndex))
        }

        for shift in stride(from: 0, to: 32, by: 8) {
            var counts = Array(repeating: 0, count: 256)
            for pair in pairs {
                counts[Int((pair.key >> UInt32(shift)) & 0xff)] += 1
            }
            var offsets = Array(repeating: 0, count: 256)
            var cursor = 0
            for bucket in stride(from: 255, through: 0, by: -1) {
                offsets[bucket] = cursor
                cursor += counts[bucket]
            }
            var writeOffsets = offsets
            for pair in pairs {
                let bucket = Int((pair.key >> UInt32(shift)) & 0xff)
                scratch[writeOffsets[bucket]] = pair
                writeOffsets[bucket] += 1
            }
            swap(&pairs, &scratch)
        }

        for offset in 0..<writableCount {
            pairsPointer[offset] = pairs[offset]
        }
        if writableCount < paddedCount {
            for offset in writableCount..<paddedCount {
                pairsPointer[offset] = SortPair(key: 0, index: UInt32.max)
            }
        }
    }

    private func updateTiledOrder(camera: Camera, orderBuffer: MTLBuffer, splatCount: Int, drawCount: Int, paddedCount: Int) {
        let pairsPointer = orderBuffer.contents().bindMemory(to: SortPair.self, capacity: paddedCount)
        let writableCount = min(drawCount, paddedCount)
        guard splatCount > 0, writableCount > 0 else {
            for offset in 0..<paddedCount {
                pairsPointer[offset] = SortPair(key: 0, index: UInt32.max)
            }
            return
        }

        let tileSize = 32
        let depthBucketCount = 16
        let viewport = camera.viewportSize
        let tilesX = max(1, Int(ceil(Double(viewport.x) / Double(tileSize))))
        let tilesY = max(1, Int(ceil(Double(viewport.y) / Double(tileSize))))
        let tileCount = tilesX * tilesY
        let binCount = tileCount * depthBucketCount
        let viewProjectionMatrix = camera.projectionMatrix * camera.viewMatrix
        var counts = Array(repeating: 0, count: binCount)
        var sources = Array(repeating: UInt32.max, count: writableCount)
        var bins = Array(repeating: 0, count: writableCount)
        var depths = Array(repeating: UInt32(0), count: writableCount)
        let stride = Double(splatCount) / Double(writableCount)

        for offset in 0..<writableCount {
            let sourceIndex = min(splatCount - 1, Int(Double(offset) * stride))
            sources[offset] = UInt32(sourceIndex)
            let position = cpuSortSplats[sourceIndex].position
            let world = SIMD4<Float>(position.x, position.y, position.z, 1)
            let clip = viewProjectionMatrix * world
            let view = camera.viewMatrix * world
            let depth = max(-view.z, 0)
            depths[offset] = min(UInt32(depth * 100_000), 0xffff_fffe)

            let ndcX = clip.x / max(abs(clip.w), 0.0001)
            let ndcY = clip.y / max(abs(clip.w), 0.0001)
            let pixelX = min(max(Int((ndcX * 0.5 + 0.5) * viewport.x), 0), max(tilesX * tileSize - 1, 0))
            let pixelY = min(max(Int((1 - (ndcY * 0.5 + 0.5)) * viewport.y), 0), max(tilesY * tileSize - 1, 0))
            let tileX = min(max(pixelX / tileSize, 0), tilesX - 1)
            let tileY = min(max(pixelY / tileSize, 0), tilesY - 1)
            let tile = tileY * tilesX + tileX
            let depthBucket = min(depthBucketCount - 1, Int(depth / max(sceneDiagnostics?.bounds.radius ?? 1, 0.001) * Float(depthBucketCount) / 6))
            let bin = (depthBucketCount - 1 - depthBucket) * tileCount + tile
            bins[offset] = bin
            counts[bin] += 1
        }

        var offsets = Array(repeating: 0, count: binCount)
        var cursor = 0
        for bin in 0..<binCount {
            offsets[bin] = cursor
            cursor += counts[bin]
        }
        var writeOffsets = offsets
        for offset in 0..<writableCount {
            let bin = bins[offset]
            let destination = writeOffsets[bin]
            writeOffsets[bin] += 1
            pairsPointer[destination] = SortPair(key: depths[offset], index: sources[offset])
        }
        if writableCount < paddedCount {
            for offset in writableCount..<paddedCount {
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

    private func projectedBuffer(drawCount: Int, frameResource: inout FrameResources) throws -> MTLBuffer {
        let length = drawCount * MemoryLayout<ProjectedSplat>.stride
        if let buffer = frameResource.projectedBuffer, frameResource.projectedCapacity >= drawCount {
            return buffer
        }
        guard let buffer = device.makeBuffer(length: length, options: .storageModePrivate) else {
            throw SplatError.metalUnavailable
        }
        buffer.label = "SplatRenderer.projectedSplats.frame"
        frameResource.projectedBuffer = buffer
        frameResource.projectedCapacity = drawCount
        return buffer
    }

    private func radixResources(drawCount: Int, frameResource: inout FrameResources) throws -> RadixResources {
        let blockSize = 256
        let radixBuckets = 256
        let blockCount = max(1, (drawCount + blockSize - 1) / blockSize)
        if let resources = frameResource.radixResources,
           resources.pairCapacity >= drawCount,
           resources.blockCapacity >= blockCount {
            return resources
        }

        let pairLength = drawCount * MemoryLayout<SortPair>.stride
        let histogramLength = radixBuckets * blockCount * MemoryLayout<UInt32>.stride
        let bucketStartLength = radixBuckets * MemoryLayout<UInt32>.stride
        guard
            let scratch = device.makeBuffer(length: pairLength, options: .storageModePrivate),
            let histogram = device.makeBuffer(length: histogramLength, options: .storageModePrivate),
            let offsets = device.makeBuffer(length: histogramLength, options: .storageModePrivate),
            let bucketTotals = device.makeBuffer(length: bucketStartLength, options: .storageModePrivate),
            let bucketStarts = device.makeBuffer(length: bucketStartLength, options: .storageModePrivate)
        else {
            throw SplatError.metalUnavailable
        }
        scratch.label = "SplatRenderer.radix.scratch"
        histogram.label = "SplatRenderer.radix.histograms"
        offsets.label = "SplatRenderer.radix.offsets"
        bucketTotals.label = "SplatRenderer.radix.bucketTotals"
        bucketStarts.label = "SplatRenderer.radix.bucketStarts"
        let resources = RadixResources(
            scratchBuffer: scratch,
            histogramBuffer: histogram,
            offsetBuffer: offsets,
            bucketTotalBuffer: bucketTotals,
            bucketStartBuffer: bucketStarts,
            pairCapacity: drawCount,
            blockCapacity: blockCount
        )
        frameResource.radixResources = resources
        return resources
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
            frameResources[index].projectedBuffer = nil
            frameResources[index].projectedCapacity = 0
            frameResources[index].radixResources = nil
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

    private func estimateMemoryBytes(mode: SortMode, drawCount: Int, sortPaddedCount: Int, usesProjectionCache: Bool) -> UInt64 {
        let splatStride = UInt64(MemoryLayout<PackedSplat>.stride)
        let pairStride = UInt64(MemoryLayout<SortPair>.stride)
        let projectedStride = UInt64(MemoryLayout<ProjectedSplat>.stride)
        let vertexShaderReads: UInt64
        if usesProjectionCache {
            vertexShaderReads = UInt64(drawCount) * 6 * projectedStride
        } else {
            vertexShaderReads = UInt64(drawCount) * 6 * (splatStride + pairStride)
        }
        let colorWrites = UInt64(drawCount) * 6 * 4
        var total = vertexShaderReads + colorWrites
        if usesProjectionCache {
            total += UInt64(drawCount) * (splatStride + pairStride + projectedStride)
        }
        if mode == .gpu {
            let sortCount = UInt64(drawCount)
            let blockCount = UInt64(max(1, (drawCount + 255) / 256))
            let histogramBytes = UInt64(256) * blockCount * UInt64(MemoryLayout<UInt32>.stride)
            let depthKeyBytes = sortCount * (splatStride + pairStride)
            let sortBytes = UInt64(4) * (sortCount * pairStride * 2 + histogramBytes * 3)
            total += depthKeyBytes + sortBytes
        } else if mode == .bitonic {
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

private struct FramePlan {
    var sortMode: SortMode
    var drawCount: Int
    var totalSplats: Int
    var sortPaddedCount: Int
}

private struct OrderAllocation {
    var buffer: MTLBuffer
    var capacity: Int
}

private struct FrameResources {
    var uniformBuffer: MTLBuffer
    var modeStates: [SortMode: ModeState] = [:]
    var activeMode: SortMode?
    var projectedBuffer: MTLBuffer?
    var projectedCapacity: Int = 0
    var radixResources: RadixResources?
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

private struct RadixConstants {
    var count: UInt32
    var blockSize: UInt32
    var blockCount: UInt32
    var shift: UInt32
}

private struct RadixResources {
    var scratchBuffer: MTLBuffer
    var histogramBuffer: MTLBuffer
    var offsetBuffer: MTLBuffer
    var bucketTotalBuffer: MTLBuffer
    var bucketStartBuffer: MTLBuffer
    var pairCapacity: Int
    var blockCapacity: Int
}

private struct ProjectedSplat {
    var clipCenter: SIMD4<Float>
    var axis0Opacity: SIMD4<Float>
    var axis1: SIMD4<Float>
    var color: SIMD4<Float>
}
