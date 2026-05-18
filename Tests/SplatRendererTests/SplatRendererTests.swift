import Foundation
import Metal
import Testing
import simd
@testable import SplatRenderer

@Suite("PLY loader")
struct PLYLoaderTests {
    @Test("loads ASCII Gaussian PLY with SH color")
    func loadsASCIIPLY() throws {
        let url = try temporaryPLY("""
        ply
        format ascii 1.0
        element vertex 2
        property float x
        property float y
        property float z
        property float scale_0
        property float scale_1
        property float scale_2
        property float rot_0
        property float rot_1
        property float rot_2
        property float rot_3
        property float opacity
        property float f_dc_0
        property float f_dc_1
        property float f_dc_2
        end_header
        0 0 0 -4 -4 -4 1 0 0 0 2 0.1 0.2 0.3
        1 2 3 -5 -5 -5 1 0 0 0 -2 0.3 0.2 0.1
        """)

        let scene = try SplatScene.load(url: url)
        #expect(scene.count == 2)
        #expect(scene.diagnostics.fieldAvailability.hasSHDC)
        #expect(scene.bounds.radius > 0)
        #expect(scene.splats[0].opacity > scene.splats[1].opacity)
    }

    @Test("loads binary little-endian Gaussian PLY")
    func loadsBinaryLittleEndianPLY() throws {
        let header = [
            "ply",
            "format binary_little_endian 1.0",
            "element vertex 1",
            "property float x",
            "property float y",
            "property float z",
            "property float f_dc_0",
            "property float f_dc_1",
            "property float f_dc_2",
            "property float opacity",
            "property float scale_0",
            "property float scale_1",
            "property float scale_2",
            "property float rot_0",
            "property float rot_1",
            "property float rot_2",
            "property float rot_3",
            "end_header\n"
        ].joined(separator: "\n")
        var data = Data(header.utf8)
        for value in [1, 2, 3, 0.1, 0.2, 0.3, 1, -4, -4, -4, 1, 0, 0, 0] as [Float] {
            var bits = value.bitPattern.littleEndian
            withUnsafeBytes(of: &bits) { data.append(contentsOf: $0) }
        }
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString).appendingPathExtension("ply")
        try data.write(to: url)

        let scene = try SplatScene.load(url: url)
        #expect(scene.count == 1)
        #expect(scene.diagnostics.format == "binary_little_endian")
        #expect(scene.splats[0].position == SIMD3<Float>(1, 2, 3))
        #expect(scene.diagnostics.fieldAvailability.hasSHDC)
    }

    @Test("reports missing position fields")
    func reportsMissingFields() throws {
        let url = try temporaryPLY("""
        ply
        format ascii 1.0
        element vertex 1
        property float x
        property float y
        end_header
        0 0
        """)

        #expect(throws: SplatError.self) {
            _ = try SplatScene.load(url: url)
        }
    }

    @Test("reference depth ordering sorts far to near")
    func referenceDepthOrderSortsFarToNear() {
        let splats = [
            Splat(position: SIMD3<Float>(0, 0, -1), scale: .one, rotation: SIMD4<Float>(1, 0, 0, 0), opacity: 1, color: .one),
            Splat(position: SIMD3<Float>(0, 0, -3), scale: .one, rotation: SIMD4<Float>(1, 0, 0, 0), opacity: 1, color: .one)
        ]
        let camera = Camera(viewMatrix: matrix_identity_float4x4, projectionMatrix: matrix_identity_float4x4, viewportSize: SIMD2<Float>(100, 100))
        let sorted = SplatScene.sortedIndices(for: splats, camera: camera)
        #expect(sorted == [1, 0])
    }

    @Test("packed splats store reference-scaled covariance")
    func packedSplatStoresReferenceCovariance() {
        let splat = Splat(
            position: .zero,
            scale: SIMD3<Float>(2, 3, 4),
            rotation: SIMD4<Float>(1, 0, 0, 0),
            opacity: 1,
            color: .one
        )
        let packed = PackedSplat(splat)
        #expect(packed.covarianceA.x == 16)
        #expect(packed.covarianceA.y == 0)
        #expect(packed.covarianceA.z == 0)
        #expect(packed.covarianceA.w == 36)
        #expect(packed.covarianceB.x == 0)
        #expect(packed.covarianceB.y == 64)
    }

    @Test("renderer can draw tiny offscreen scene when Metal is available")
    func rendererSmokeTest() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { return }
        let splat = Splat(position: SIMD3<Float>(0, 0, 0), scale: SIMD3<Float>(repeating: 0.05), rotation: SIMD4<Float>(1, 0, 0, 0), opacity: 1, color: SIMD3<Float>(1, 0, 0))
        let diagnostics = SplatDiagnostics(
            sourceURL: nil,
            format: "synthetic",
            vertexCount: 1,
            fieldAvailability: SplatFieldAvailability(hasSHDC: false, hasRGB: true, hasScale: true, hasRotation: true, hasOpacity: true),
            bounds: SplatBounds(minimum: [0, 0, 0], maximum: [0, 0, 0], center: [0, 0, 0], radius: 1),
            warnings: []
        )
        let scene = SplatScene(splats: [splat], diagnostics: diagnostics)
        let renderer = try SplatRenderer(device: device)
        try renderer.load(scene: scene)
        let camera = renderer.makeDefaultCamera(width: 64, height: 64)
        let stats = try renderer.drawOffscreen(size: SIMD2<Int32>(64, 64), camera: camera, options: RenderOptions(sortMode: .radix, waitForGPU: true))
        #expect(stats.totalSplats == 1)
    }

    @Test("renderer can switch sort modes without sharing order state")
    func rendererSortModeSwitching() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { return }
        let splats = (0..<8).map { index in
            Splat(
                position: SIMD3<Float>(Float(index) * 0.01, 0, Float(index) * -0.05),
                scale: SIMD3<Float>(repeating: 0.03),
                rotation: SIMD4<Float>(1, 0, 0, 0),
                opacity: 1,
                color: SIMD3<Float>(1, 0.5, 0.25)
            )
        }
        let diagnostics = SplatDiagnostics(
            sourceURL: nil,
            format: "synthetic",
            vertexCount: splats.count,
            fieldAvailability: SplatFieldAvailability(hasSHDC: false, hasRGB: true, hasScale: true, hasRotation: true, hasOpacity: true),
            bounds: SplatBounds(minimum: [0, 0, -0.35], maximum: [0.07, 0, 0], center: [0.035, 0, -0.175], radius: 1),
            warnings: []
        )
        let renderer = try SplatRenderer(device: device)
        try renderer.load(scene: SplatScene(splats: splats, diagnostics: diagnostics))
        let camera = renderer.makeDefaultCamera(width: 64, height: 64)
        let modes: [SortMode] = [.unsorted, .radix, .gpu, .bitonic, .unsorted]
        for mode in modes {
            let stats = try renderer.drawOffscreen(size: SIMD2<Int32>(64, 64), camera: camera, options: RenderOptions(sortMode: mode, waitForGPU: true, maxVisibleSplats: 4))
            #expect(stats.sortMode == mode)
            #expect(stats.visibleSplats == 4)
            #expect(stats.estimatedMemoryBytes > 0)
        }
    }

    private func temporaryPLY(_ contents: String) throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString).appendingPathExtension("ply")
        try contents.data(using: .utf8)!.write(to: url)
        return url
    }
}

private extension SIMD3 where Scalar == Float {
    static var one: SIMD3<Float> { SIMD3<Float>(repeating: 1) }
}
