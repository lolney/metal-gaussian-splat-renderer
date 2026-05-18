import Foundation
import Metal
import SplatRenderer
import simd

struct BenchOptions {
    var input: URL?
    var frames = 120
    var warmupFrames = 2
    var width = 1280
    var height = 720
    var sortMode: SortMode = .gpu
    var maxVisibleSplats = 0
    var maxSplatRadius: Float = 72
    var enableCulling = true
    var output: URL?
    var capture: URL?
}

@main
enum SplatBench {
    static func main() throws {
        let arguments = Array(CommandLine.arguments.dropFirst())
        let options = try parseArguments(arguments)
        guard let input = options.input else {
            printUsage()
            return
        }

        guard let device = MTLCreateSystemDefaultDevice() else {
            throw SplatError.metalUnavailable
        }

        print("Device: \(device.name)")
        print("Input: \(input.path)")
        print("Frames: \(options.frames), warmup: \(options.warmupFrames), size: \(options.width)x\(options.height), sort: \(options.sortMode.rawValue), budget: \(options.maxVisibleSplats == 0 ? "all" : "\(options.maxVisibleSplats)")")

        let loadStart = CFAbsoluteTimeGetCurrent()
        let scene = try SplatScene.load(url: input)
        let loadMilliseconds = (CFAbsoluteTimeGetCurrent() - loadStart) * 1000
        let renderer = try SplatRenderer(device: device)
        let rendererLoadStart = CFAbsoluteTimeGetCurrent()
        try renderer.load(scene: scene)
        let rendererLoadMilliseconds = (CFAbsoluteTimeGetCurrent() - rendererLoadStart) * 1000

        if let capture = options.capture {
            try startCapture(device: device, destination: capture)
        }

        var stats: [FrameStats] = []
        let renderOptions = RenderOptions(
            sortMode: options.sortMode,
            maxSplatRadius: options.maxSplatRadius,
            enableProfiling: true,
            waitForGPU: true,
            enableCulling: options.enableCulling,
            maxVisibleSplats: options.maxVisibleSplats
        )
        let size = SIMD2<Int32>(Int32(options.width), Int32(options.height))
        for frame in 0..<options.frames {
            let camera = orbitCamera(scene: scene, frame: frame, total: options.frames, width: options.width, height: options.height)
            stats.append(try renderer.drawOffscreen(size: size, camera: camera, options: renderOptions))
        }

        if options.capture != nil {
            MTLCaptureManager.shared().stopCapture()
        }

        let summary = BenchmarkSummary(
            device: device.name,
            input: input.path,
            loadMilliseconds: loadMilliseconds,
            rendererLoadMilliseconds: rendererLoadMilliseconds,
            warmupFrames: options.warmupFrames,
            frames: stats
        )
        let encoded = try JSONEncoder.pretty.encode(summary)
        if let output = options.output {
            try FileManager.default.createDirectory(at: output.deletingLastPathComponent(), withIntermediateDirectories: true)
            try encoded.write(to: output)
            try csv(frames: stats).write(to: output.deletingPathExtension().appendingPathExtension("csv"), atomically: true, encoding: .utf8)
            print("Wrote \(output.path)")
        } else if let json = String(data: encoded, encoding: .utf8) {
            print(json)
        }
    }

    private static func parseArguments(_ args: [String]) throws -> BenchOptions {
        var options = BenchOptions()
        var index = 0
        while index < args.count {
            let arg = args[index]
            func next() throws -> String {
                guard index + 1 < args.count else {
                    throw SplatError.invalidPLY("missing value for \(arg)")
                }
                index += 1
                return args[index]
            }
            switch arg {
            case "--input":
                options.input = URL(fileURLWithPath: try next())
            case "--frames":
                options.frames = Int(try next()) ?? options.frames
            case "--warmup":
                options.warmupFrames = Int(try next()) ?? options.warmupFrames
            case "--width":
                options.width = Int(try next()) ?? options.width
            case "--height":
                options.height = Int(try next()) ?? options.height
            case "--sort":
                let rawMode = try next()
                options.sortMode = rawMode == "none" ? .unsorted : (SortMode(rawValue: rawMode) ?? options.sortMode)
            case "--max-splats":
                options.maxVisibleSplats = Int(try next()) ?? options.maxVisibleSplats
            case "--radius":
                options.maxSplatRadius = Float(try next()) ?? options.maxSplatRadius
            case "--no-culling":
                options.enableCulling = false
            case "--output":
                options.output = URL(fileURLWithPath: try next())
            case "--capture":
                options.capture = URL(fileURLWithPath: try next())
            case "-h", "--help":
                printUsage()
                Foundation.exit(0)
            default:
                if options.input == nil {
                    options.input = URL(fileURLWithPath: arg)
                } else {
                    throw SplatError.invalidPLY("unknown argument \(arg)")
                }
            }
            index += 1
        }
        return options
    }

    private static func printUsage() {
        print("""
        Usage: splatbench --input scene.ply [--frames N] [--warmup N] [--width W] [--height H] [--sort unsorted|gpu|cpu] [--max-splats N] [--radius px] [--no-culling] [--output results.json] [--capture trace.gputrace]
        """)
    }

    private static func orbitCamera(scene: SplatScene, frame: Int, total: Int, width: Int, height: Int) -> Camera {
        let bounds = scene.bounds
        let center = SIMD3<Float>(bounds.center[0], bounds.center[1], bounds.center[2])
        let radius = max(bounds.radius, 0.1)
        let angle = Float(frame) / Float(max(total, 1)) * 2 * .pi
        let eye = center + SIMD3<Float>(sin(angle) * radius * 3, radius * 0.35, cos(angle) * radius * 3)
        let aspect = Float(max(width, 1)) / Float(max(height, 1))
        return Camera(
            viewMatrix: .lookAt(eye: eye, center: center, up: SIMD3<Float>(0, 1, 0)),
            projectionMatrix: .perspective(fovyRadians: 60 * .pi / 180, aspect: aspect, nearZ: max(radius * 0.001, 0.0001), farZ: max(radius * 100, 10)),
            viewportSize: SIMD2<Float>(Float(width), Float(height))
        )
    }

    private static func startCapture(device: MTLDevice, destination: URL) throws {
        let descriptor = MTLCaptureDescriptor()
        descriptor.captureObject = device
        descriptor.destination = .gpuTraceDocument
        descriptor.outputURL = destination
        try MTLCaptureManager.shared().startCapture(with: descriptor)
    }

    private static func csv(frames: [FrameStats]) -> String {
        var rows = ["id,total_ms,cpu_encode_ms,gpu_ms,depth_key_ms,sort_ms,draw_ms,estimated_memory_bytes,estimated_bandwidth_gbps,visible_splats,total_splats,sort_mode"]
        rows += frames.map { frame in
            [
                "\(frame.id)",
                "\(frame.totalFrameMilliseconds)",
                "\(frame.cpuEncodeMilliseconds)",
                "\(frame.gpuFrameMilliseconds ?? 0)",
                "\(frame.depthKeyMilliseconds ?? 0)",
                "\(frame.sortMilliseconds ?? 0)",
                "\(frame.drawMilliseconds ?? 0)",
                "\(frame.estimatedMemoryBytes)",
                "\(frame.estimatedMemoryBandwidthGBps ?? 0)",
                "\(frame.visibleSplats)",
                "\(frame.totalSplats)",
                frame.sortMode.rawValue
            ].joined(separator: ",")
        }
        return rows.joined(separator: "\n")
    }
}

struct BenchmarkSummary: Codable {
    var device: String
    var input: String
    var loadMilliseconds: Double
    var rendererLoadMilliseconds: Double
    var warmupFrames: Int
    var frames: [FrameStats]
    var averageFrameMilliseconds: Double
    var p95FrameMilliseconds: Double
    var measuredAverageFrameMilliseconds: Double
    var measuredP95FrameMilliseconds: Double

    init(
        device: String,
        input: String,
        loadMilliseconds: Double,
        rendererLoadMilliseconds: Double,
        warmupFrames: Int,
        frames: [FrameStats]
    ) {
        self.device = device
        self.input = input
        self.loadMilliseconds = loadMilliseconds
        self.rendererLoadMilliseconds = rendererLoadMilliseconds
        self.warmupFrames = warmupFrames
        self.frames = frames
        let totals = frames.map(\.totalFrameMilliseconds)
        averageFrameMilliseconds = totals.isEmpty ? 0 : totals.reduce(0, +) / Double(totals.count)
        let sorted = totals.sorted()
        p95FrameMilliseconds = sorted.isEmpty ? 0 : sorted[min(sorted.count - 1, Int(Double(sorted.count - 1) * 0.95))]
        let measuredTotals = Array(totals.dropFirst(min(warmupFrames, totals.count)))
        measuredAverageFrameMilliseconds = measuredTotals.isEmpty ? averageFrameMilliseconds : measuredTotals.reduce(0, +) / Double(measuredTotals.count)
        let measuredSorted = measuredTotals.sorted()
        measuredP95FrameMilliseconds = measuredSorted.isEmpty ? p95FrameMilliseconds : measuredSorted[min(measuredSorted.count - 1, Int(Double(measuredSorted.count - 1) * 0.95))]
    }
}

extension JSONEncoder {
    static var pretty: JSONEncoder {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return encoder
    }
}
