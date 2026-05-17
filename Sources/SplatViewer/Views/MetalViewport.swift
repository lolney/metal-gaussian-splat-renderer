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
}

final class OrbitCameraController {
    private var center = SIMD3<Float>(0, 0, 0)
    private var radius: Float = 1
    private var yaw: Float = 0
    private var pitch: Float = 0
    private var distance: Float = 3

    func focus(bounds: SplatBounds) {
        center = SIMD3<Float>(bounds.center[0], bounds.center[1], bounds.center[2])
        radius = max(bounds.radius, 0.1)
        distance = radius * 3
        yaw = 0
        pitch = 0
    }

    func orbit(deltaX: Float, deltaY: Float) {
        yaw -= deltaX * 0.008
        pitch = clamp(pitch - deltaY * 0.008, -1.45, 1.45)
    }

    func pan(deltaX: Float, deltaY: Float) {
        let scale = distance * 0.0015
        let right = SIMD3<Float>(cos(yaw), 0, -sin(yaw))
        let up = SIMD3<Float>(0, 1, 0)
        center -= right * deltaX * scale
        center += up * deltaY * scale
    }

    func zoom(delta: Float) {
        distance = clamp(distance * (1 - delta * 0.0015), radius * 0.05, radius * 50)
    }

    func camera(width: Int, height: Int) -> Camera {
        let cp = cos(pitch)
        let direction = SIMD3<Float>(sin(yaw) * cp, sin(pitch), cos(yaw) * cp)
        let eye = center + direction * distance
        let aspect = Float(max(width, 1)) / Float(max(height, 1))
        return Camera(
            viewMatrix: .lookAt(eye: eye, center: center, up: SIMD3<Float>(0, 1, 0)),
            projectionMatrix: .perspective(fovyRadians: 60 * .pi / 180, aspect: aspect, nearZ: max(radius * 0.001, 0.0001), farZ: max(radius * 100, 10)),
            viewportSize: SIMD2<Float>(Float(width), Float(height))
        )
    }
}
