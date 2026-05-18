import Foundation
import SplatRenderer
import SwiftUI

struct ProfilingPanel: View {
    @EnvironmentObject private var store: ViewerStore

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                Text("Profiling")
                    .font(.headline)
                Spacer()
                Button {
                    store.profilingPaused.toggle()
                } label: {
                    Image(systemName: store.profilingPaused ? "play.fill" : "pause.fill")
                }
                .help(store.profilingPaused ? "Resume metrics" : "Pause metrics")

                Button {
                    store.clearHistory()
                } label: {
                    Image(systemName: "trash")
                }
                .help("Clear history")

                Button {
                    store.exportProfilingData()
                } label: {
                    Image(systemName: "square.and.arrow.up")
                }
                .help("Export profiling data")
            }

            if let diagnostics = store.diagnostics {
                SceneStatsView(diagnostics: diagnostics)
            }

            if let statusMessage = store.statusMessage {
                Text(statusMessage)
                    .font(.caption)
                    .foregroundStyle(.orange)
                    .fixedSize(horizontal: false, vertical: true)
            }

            MetricSummary(title: "Frame", values: store.frameHistory.map(\.totalFrameMilliseconds), suffix: "ms")
            FrameGraph(frames: store.frameHistory)
                .frame(height: 150)

            MetricSummary(title: "GPU", values: store.frameHistory.compactMap(\.gpuFrameMilliseconds), suffix: "ms")
            MetricSummary(title: "Memory est.", values: store.frameHistory.compactMap(\.estimatedMemoryBandwidthGBps), suffix: " GB/s")
            MemoryBandwidthGraph(frames: Array(store.frameHistory.suffix(120)))
                .frame(height: 76)

            VStack(alignment: .leading, spacing: 3) {
                HStack {
                    Text("CPU Encode Stages")
                        .font(.subheadline.weight(.semibold))
                    Spacer()
                    if let scale = MetricBars.scaleValue(frames: Array(store.frameHistory.suffix(80))) {
                        Text(String(format: "p95 scale %.3fms", scale))
                            .font(.system(.caption2, design: .monospaced))
                            .foregroundStyle(.secondary)
                    }
                }
                Text("These bars are command encoding time. GPU work is summarized above; memory bandwidth is estimated from bytes touched per frame.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            HStack(spacing: 12) {
                LegendItem(color: .blue, label: "Depth encode")
                LegendItem(color: .orange, label: "Sort encode")
                LegendItem(color: .purple, label: "Projection")
                LegendItem(color: .green, label: "Draw encode")
            }
            .font(.caption2)
            .foregroundStyle(.secondary)
            MetricBars(frames: Array(store.frameHistory.suffix(80)))
                .frame(height: 120)

            ControlsView()

            Spacer()
        }
        .padding(14)
        .background(.thinMaterial)
    }
}

private struct SceneStatsView: View {
    var diagnostics: SplatDiagnostics

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(diagnostics.sourceURL?.lastPathComponent ?? "Loaded scene")
                .font(.subheadline)
                .lineLimit(1)
            Text("\(diagnostics.vertexCount) splats  \(diagnostics.format)")
                .font(.caption)
                .foregroundStyle(.secondary)
            ForEach(diagnostics.warnings, id: \.self) { warning in
                Text(warning)
                    .font(.caption2)
                    .foregroundStyle(.orange)
                    .lineLimit(2)
            }
        }
    }
}

private struct MetricSummary: View {
    var title: String
    var values: [Double]
    var suffix: String

    var body: some View {
        let stats = Summary(values: values)
        HStack {
            Text(title)
                .font(.subheadline.weight(.semibold))
            Spacer()
            Text(String(format: "avg %.2f%@  p95 %.2f%@  min %.2f%@  max %.2f%@", stats.average, suffix, stats.p95, suffix, stats.minimum, suffix, stats.maximum, suffix))
                .font(.system(.caption, design: .monospaced))
                .foregroundStyle(.secondary)
        }
    }
}

private struct FrameGraph: View {
    var frames: [FrameStats]

    var body: some View {
        Canvas { context, size in
            let values = frames.suffix(240).map(\.totalFrameMilliseconds)
            guard values.count > 1 else { return }
            let maxValue = max(values.max() ?? 16.67, 33.33)
            drawBudgetLine(8.33, label: "120", context: &context, size: size, maxValue: maxValue)
            drawBudgetLine(11.11, label: "90", context: &context, size: size, maxValue: maxValue)
            drawBudgetLine(16.67, label: "60", context: &context, size: size, maxValue: maxValue)

            var path = Path()
            for (index, value) in values.enumerated() {
                let x = size.width * CGFloat(index) / CGFloat(values.count - 1)
                let y = size.height - size.height * CGFloat(value / maxValue)
                if index == 0 {
                    path.move(to: CGPoint(x: x, y: y))
                } else {
                    path.addLine(to: CGPoint(x: x, y: y))
                }
            }
            context.stroke(path, with: .color(.accentColor), lineWidth: 2)
        }
        .background(.quaternary.opacity(0.25), in: RoundedRectangle(cornerRadius: 8))
    }

    private func drawBudgetLine(_ milliseconds: Double, label: String, context: inout GraphicsContext, size: CGSize, maxValue: Double) {
        let y = size.height - size.height * CGFloat(milliseconds / maxValue)
        var path = Path()
        path.move(to: CGPoint(x: 0, y: y))
        path.addLine(to: CGPoint(x: size.width, y: y))
        context.stroke(path, with: .color(.secondary.opacity(0.45)), style: StrokeStyle(lineWidth: 1, dash: [4, 4]))
        context.draw(Text(label).font(.caption2).foregroundStyle(.secondary), at: CGPoint(x: size.width - 16, y: y - 8))
    }
}

private struct MetricBars: View {
    var frames: [FrameStats]

    var body: some View {
        Canvas { context, size in
            guard !frames.isEmpty else { return }
            let maxValue = Self.scaleValue(frames: frames) ?? 1
            let barWidth = size.width / CGFloat(frames.count)
            for (index, frame) in frames.enumerated() {
                var y = size.height
                let segments: [(Double, Color)] = [
                    (frame.depthKeyMilliseconds ?? 0, .blue),
                    (frame.sortMilliseconds ?? 0, .orange),
                    (frame.projectionMilliseconds ?? 0, .purple),
                    (frame.drawMilliseconds ?? 0, .green)
                ]
                for (value, color) in segments {
                    let clippedValue = min(value, maxValue)
                    let height = max(value > 0 ? 1 : 0, size.height * CGFloat(clippedValue / maxValue))
                    y -= height
                    let rect = CGRect(x: CGFloat(index) * barWidth, y: y, width: max(1, barWidth - 1), height: height)
                    context.fill(Path(rect), with: .color(color.opacity(0.75)))
                }
            }
        }
        .background(.quaternary.opacity(0.25), in: RoundedRectangle(cornerRadius: 8))
    }

    static func scaleValue(frames: [FrameStats]) -> Double? {
        let totals = frames.map {
            ($0.depthKeyMilliseconds ?? 0) + ($0.sortMilliseconds ?? 0) + ($0.projectionMilliseconds ?? 0) + ($0.drawMilliseconds ?? 0)
        }.filter { $0 > 0 }
        guard !totals.isEmpty else { return nil }
        let sorted = totals.sorted()
        return max(sorted[min(sorted.count - 1, Int(Double(sorted.count - 1) * 0.95))], 0.001)
    }
}

private struct MemoryBandwidthGraph: View {
    var frames: [FrameStats]

    var body: some View {
        Canvas { context, size in
            let values = frames.map { $0.estimatedMemoryBandwidthGBps ?? 0 }
            guard values.count > 1 else { return }
            let sorted = values.filter { $0 > 0 }.sorted()
            let scale = max(sorted.isEmpty ? 1 : sorted[min(sorted.count - 1, Int(Double(sorted.count - 1) * 0.95))], 1)
            let barWidth = size.width / CGFloat(values.count)
            for (index, value) in values.enumerated() {
                let clippedValue = min(value, scale)
                let height = size.height * CGFloat(clippedValue / scale)
                let rect = CGRect(x: CGFloat(index) * barWidth, y: size.height - height, width: max(1, barWidth - 1), height: height)
                context.fill(Path(rect), with: .color(.indigo.opacity(0.72)))
            }
            context.draw(Text(String(format: "%.1f GB/s p95", scale)).font(.caption2).foregroundStyle(.secondary), at: CGPoint(x: size.width - 42, y: 10))
        }
        .background(.quaternary.opacity(0.25), in: RoundedRectangle(cornerRadius: 8))
        .overlay(alignment: .topLeading) {
            if let last = frames.last {
                Text("\(formatBytes(last.estimatedMemoryBytes))/frame")
                    .font(.caption2.monospacedDigit())
                    .foregroundStyle(.secondary)
                    .padding(6)
            }
        }
    }
}

private struct LegendItem: View {
    var color: Color
    var label: String

    var body: some View {
        HStack(spacing: 4) {
            Circle()
                .fill(color)
                .frame(width: 7, height: 7)
            Text(label)
        }
    }
}

private func formatBytes(_ bytes: UInt64) -> String {
    let formatter = ByteCountFormatter()
    formatter.countStyle = .memory
    formatter.allowedUnits = bytes > 1_000_000_000 ? [.useGB] : [.useMB]
    return formatter.string(fromByteCount: Int64(bytes))
}

private struct ControlsView: View {
    @EnvironmentObject private var store: ViewerStore

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Toggle("Enable profiling", isOn: $store.options.enableProfiling)
            Toggle("Wait for GPU timings", isOn: $store.options.waitForGPU)
            Toggle("Projection cache", isOn: $store.options.useProjectionCache)
            HStack {
                Text("Budget")
                Stepper(value: Binding(
                    get: { store.options.maxVisibleSplats },
                    set: { store.options.maxVisibleSplats = $0 }
                ), in: 0...max(store.scene?.count ?? 0, 1), step: 100_000) {
                    Text(store.options.maxVisibleSplats == 0 ? "All" : store.options.maxVisibleSplats.formatted())
                        .font(.system(.caption, design: .monospaced))
                        .frame(width: 86, alignment: .trailing)
                }
            }
            HStack {
                Text("Max radius")
                Slider(value: Binding(
                    get: { Double(store.options.maxSplatRadius) },
                    set: { store.options.maxSplatRadius = Float($0) }
                ), in: 4...256)
                Text("\(Int(store.options.maxSplatRadius))")
                    .font(.system(.caption, design: .monospaced))
                    .frame(width: 36, alignment: .trailing)
            }
        }
        .font(.caption)
    }
}

private struct Summary {
    var minimum: Double = 0
    var maximum: Double = 0
    var average: Double = 0
    var p95: Double = 0

    init(values: [Double]) {
        guard !values.isEmpty else { return }
        let sorted = values.sorted()
        minimum = sorted.first ?? 0
        maximum = sorted.last ?? 0
        average = values.reduce(0, +) / Double(values.count)
        p95 = sorted[min(sorted.count - 1, Int(Double(sorted.count - 1) * 0.95))]
    }
}
