import AppKit
import MetalKit
import SplatRenderer
import SwiftUI
import simd

struct MetalViewport: NSViewRepresentable {
    @EnvironmentObject private var store: ViewerStore

    func makeCoordinator() -> Coordinator {
        Coordinator(store: store)
    }

    func makeNSView(context: Context) -> InteractiveMTKView {
        let device = MTLCreateSystemDefaultDevice()
        let view = InteractiveMTKView()
        view.device = device
        view.colorPixelFormat = .bgra8Unorm
        view.clearColor = MTLClearColor(red: 0.02, green: 0.025, blue: 0.03, alpha: 1)
        view.preferredFramesPerSecond = 120
        view.enableSetNeedsDisplay = false
        view.isPaused = false
        view.framebufferOnly = true
        view.delegate = context.coordinator
        view.cameraController = context.coordinator.cameraController
        view.onCameraPresetSelected = { preset in
            Task { @MainActor in
                context.coordinator.store.statusMessage = "Bicycle camera \(preset.id) (\(preset.imageName)) selected. Drag, pan, or zoom to move from this camera."
                context.coordinator.store.markEvent("Camera preset \(preset.id)")
            }
        }
        context.coordinator.configure(device: device, view: view)
        return view
    }

    func updateNSView(_ nsView: InteractiveMTKView, context: Context) {
        context.coordinator.store = store
        context.coordinator.loadSceneIfNeeded()
    }

    final class Coordinator: NSObject, MTKViewDelegate {
        @MainActor var store: ViewerStore
        let cameraController = OrbitCameraController()
        private var renderer: SplatRenderer?
        private weak var view: MTKView?
        private var loadedVertexCount: Int?

        @MainActor
        init(store: ViewerStore) {
            self.store = store
        }

        @MainActor
        func configure(device: MTLDevice?, view: MTKView) {
            self.view = view
            guard let device else { return }
            do {
                renderer = try SplatRenderer(device: device)
            } catch {
                store.loadError = error.localizedDescription
            }
        }

        @MainActor
        func loadSceneIfNeeded() {
            guard let scene = store.scene, loadedVertexCount != scene.count else { return }
            do {
                try renderer?.load(scene: scene)
                cameraController.focus(bounds: scene.bounds)
                if let url = scene.diagnostics.sourceURL, BicycleCameraPreset.isBicycleScene(url), let preset = BicycleCameraPreset.preset(for: 0) {
                    cameraController.useSavedCamera(preset)
                    store.statusMessage = "Using saved bicycle camera 0 (\(preset.imageName)). Press 0-9 to switch saved bicycle cameras; drag, pan, or zoom to move from this camera."
                }
                loadedVertexCount = scene.count
            } catch {
                store.loadError = error.localizedDescription
            }
        }

        func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

        func draw(in view: MTKView) {
            Task { @MainActor in
                drawOnMain(in: view)
            }
        }

        @MainActor
        private func drawOnMain(in view: MTKView) {
            loadSceneIfNeeded()
            guard
                let renderer,
                let descriptor = view.currentRenderPassDescriptor,
                let drawable = view.currentDrawable
            else { return }

            let width = max(Int32(view.drawableSize.width), 1)
            let height = max(Int32(view.drawableSize.height), 1)
            let camera = cameraController.camera(width: Int(width), height: Int(height))
            do {
                let stats = try renderer.draw(
                    camera: camera,
                    renderPassDescriptor: descriptor,
                    drawable: drawable,
                    viewportSize: SIMD2<Int32>(width, height),
                    options: store.options
                )
                store.record(stats)
            } catch {
                store.loadError = error.localizedDescription
            }
        }
    }
}

final class InteractiveMTKView: MTKView {
    var cameraController: OrbitCameraController?
    var onCameraPresetSelected: ((BicycleCameraPreset) -> Void)?

    override var acceptsFirstResponder: Bool { true }

    override func viewDidMoveToWindow() {
        window?.makeFirstResponder(self)
    }

    override func mouseDragged(with event: NSEvent) {
        cameraController?.orbit(deltaX: Float(event.deltaX), deltaY: Float(event.deltaY))
    }

    override func rightMouseDragged(with event: NSEvent) {
        cameraController?.pan(deltaX: Float(event.deltaX), deltaY: Float(event.deltaY))
    }

    override func scrollWheel(with event: NSEvent) {
        cameraController?.zoom(delta: Float(event.scrollingDeltaY))
    }

    override func keyDown(with event: NSEvent) {
        guard
            let characters = event.charactersIgnoringModifiers,
            characters.count == 1,
            let key = characters.first?.wholeNumberValue,
            let preset = BicycleCameraPreset.preset(for: key)
        else {
            super.keyDown(with: event)
            return
        }
        cameraController?.useSavedCamera(preset)
        onCameraPresetSelected?(preset)
    }
}

final class OrbitCameraController {
    private var center = SIMD3<Float>(0, 0, 0)
    private var radius: Float = 1
    private var yaw: Float = 0
    private var pitch: Float = 0
    private var distance: Float = 3
    private var savedCamera: BicycleCameraPreset?
    private var referenceProjection: BicycleCameraPreset?
    private var freeCamera: FreeCameraState?

    func focus(bounds: SplatBounds) {
        center = SIMD3<Float>(bounds.center[0], bounds.center[1], bounds.center[2])
        radius = max(bounds.radius, 0.1)
        distance = radius * 3
        yaw = 0
        pitch = 0
        savedCamera = nil
        referenceProjection = nil
        freeCamera = nil
    }

    func useSavedCamera(_ preset: BicycleCameraPreset) {
        savedCamera = preset
        referenceProjection = preset
        freeCamera = nil
    }

    func orbit(deltaX: Float, deltaY: Float) {
        if useFreeCameraFromSavedCameraIfNeeded() || freeCamera != nil {
            rotateFreeCamera(deltaX: deltaX, deltaY: deltaY)
            return
        }
        yaw -= deltaX * 0.008
        pitch = clamp(pitch - deltaY * 0.008, -1.45, 1.45)
    }

    func pan(deltaX: Float, deltaY: Float) {
        if useFreeCameraFromSavedCameraIfNeeded() || freeCamera != nil {
            panFreeCamera(deltaX: deltaX, deltaY: deltaY)
            return
        }
        let scale = distance * 0.0015
        let right = SIMD3<Float>(cos(yaw), 0, -sin(yaw))
        let up = SIMD3<Float>(0, 1, 0)
        center -= right * deltaX * scale
        center += up * deltaY * scale
    }

    func zoom(delta: Float) {
        if useFreeCameraFromSavedCameraIfNeeded() || freeCamera != nil {
            zoomFreeCamera(delta: delta)
            return
        }
        distance = clamp(distance * (1 - delta * 0.0015), radius * 0.05, radius * 50)
    }

    func camera(width: Int, height: Int) -> Camera {
        if let savedCamera {
            return savedCamera.camera(width: width, height: height)
        }
        if let freeCamera {
            return Camera(
                viewMatrix: freeCamera.viewMatrix,
                projectionMatrix: referenceProjection?.projectionMatrix(width: width, height: height) ?? defaultProjection(width: width, height: height),
                viewportSize: SIMD2<Float>(Float(width), Float(height))
            )
        }
        let cp = cos(pitch)
        let direction = SIMD3<Float>(sin(yaw) * cp, sin(pitch), cos(yaw) * cp)
        let eye = center + direction * distance
        return Camera(
            viewMatrix: .lookAt(eye: eye, center: center, up: SIMD3<Float>(0, 1, 0)),
            projectionMatrix: defaultProjection(width: width, height: height),
            viewportSize: SIMD2<Float>(Float(width), Float(height))
        )
    }

    private func defaultProjection(width: Int, height: Int) -> simd_float4x4 {
        let aspect = Float(max(width, 1)) / Float(max(height, 1))
        return .perspective(fovyRadians: 60 * .pi / 180, aspect: aspect, nearZ: max(radius * 0.001, 0.0001), farZ: max(radius * 100, 10))
    }

    @discardableResult
    private func useFreeCameraFromSavedCameraIfNeeded() -> Bool {
        guard let savedCamera else { return false }
        freeCamera = FreeCameraState(viewMatrix: savedCamera.viewMatrix(), moveScale: max(radius, 0.1))
        self.savedCamera = nil
        return true
    }

    private func rotateFreeCamera(deltaX: Float, deltaY: Float) {
        guard var state = freeCamera else { return }
        var pose = CameraPose(viewMatrix: state.viewMatrix)
        let yawRotation = simd_quatf(angle: -deltaX * 0.004, axis: SIMD3<Float>(0, 1, 0))
        pose.forward = yawRotation.act(pose.forward)
        pose.up = yawRotation.act(pose.up)
        pose.orthonormalize()

        let pitchRotation = simd_quatf(angle: -deltaY * 0.004, axis: pose.right)
        pose.forward = pitchRotation.act(pose.forward)
        pose.up = pitchRotation.act(pose.up)
        pose.orthonormalize()
        state.viewMatrix = pose.viewMatrix
        freeCamera = state
    }

    private func panFreeCamera(deltaX: Float, deltaY: Float) {
        guard var state = freeCamera else { return }
        var pose = CameraPose(viewMatrix: state.viewMatrix)
        let scale = state.moveScale * 0.0015
        pose.eye -= pose.right * deltaX * scale
        pose.eye += pose.up * deltaY * scale
        state.viewMatrix = pose.viewMatrix
        freeCamera = state
    }

    private func zoomFreeCamera(delta: Float) {
        guard var state = freeCamera else { return }
        var pose = CameraPose(viewMatrix: state.viewMatrix)
        let step = state.moveScale * delta * 0.003
        pose.eye += pose.forward * step
        state.moveScale = clamp(state.moveScale * (1 - delta * 0.001), radius * 0.02, radius * 10)
        state.viewMatrix = pose.viewMatrix
        freeCamera = state
    }
}

private struct FreeCameraState {
    var viewMatrix: simd_float4x4
    var moveScale: Float
}

private struct CameraPose {
    var eye: SIMD3<Float>
    var forward: SIMD3<Float>
    var up: SIMD3<Float>
    var right: SIMD3<Float>

    init(viewMatrix: simd_float4x4) {
        let cameraToWorld = simd_inverse(viewMatrix)
        eye = cameraToWorld.columns.3.xyz
        right = simd_normalize(cameraToWorld.columns.0.xyz)
        up = simd_normalize(cameraToWorld.columns.1.xyz)
        forward = -simd_normalize(cameraToWorld.columns.2.xyz)
        orthonormalize()
    }

    mutating func orthonormalize() {
        forward = simd_normalize(forward)
        right = simd_normalize(simd_cross(forward, up))
        up = simd_normalize(simd_cross(right, forward))
    }

    var viewMatrix: simd_float4x4 {
        simd_float4x4.lookAt(eye: eye, center: eye + forward, up: up)
    }
}

private extension SIMD4 where Scalar == Float {
    var xyz: SIMD3<Float> {
        SIMD3<Float>(x, y, z)
    }
}
