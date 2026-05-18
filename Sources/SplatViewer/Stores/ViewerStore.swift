import AppKit
import Combine
import Foundation
import SplatRenderer
import SwiftUI

@MainActor
final class ViewerStore: ObservableObject {
    @Published var scene: SplatScene?
    @Published var loadError: String?
    @Published var statusMessage: String?
    @Published var isLoading = false
    @Published var options = RenderOptions(enableProfiling: true, waitForGPU: true)
    @Published var profilingVisible = true
    @Published var profilingPaused = false
    @Published var frameHistory: [FrameStats] = []
    @Published var eventMarks: [ProfilingEvent] = []

    let maxHistory = 600
    let interactiveGPUSortLimit = 1_000_000
    private var loadGeneration = 0

    var diagnostics: SplatDiagnostics? {
        scene?.diagnostics
    }

    func openPLY() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.init(filenameExtension: "ply")!]
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        guard panel.runModal() == .OK, let url = panel.url else { return }
        load(url: url)
    }

    func load(url: URL) {
        loadGeneration += 1
        let generation = loadGeneration
        isLoading = true
        loadError = nil
        statusMessage = "Loading \(url.lastPathComponent)..."

        Task {
            do {
                let loaded = try await Task.detached(priority: .userInitiated) {
                    try SplatScene.load(url: url)
                }.value
                guard generation == loadGeneration else { return }
                applyLoadedScene(loaded, filename: url.lastPathComponent)
            } catch {
                guard generation == loadGeneration else { return }
                isLoading = false
                loadError = error.localizedDescription
                statusMessage = nil
            }
        }
    }

    private func applyLoadedScene(_ loaded: SplatScene, filename: String) {
        scene = loaded
        if loaded.count > interactiveGPUSortLimit, options.sortMode == .gpu {
            options.sortMode = .unsorted
            options.maxVisibleSplats = min(max(options.maxVisibleSplats, 500_000), loaded.count)
            statusMessage = "Large scene loaded in unsorted mode with a 500k budget. GPU radix, CPU radix, and bitonic reference modes remain available, but they may be slow."
        } else {
            statusMessage = nil
        }
        isLoading = false
        loadError = nil
        markEvent("Loaded \(filename)")
    }

    func record(_ stats: FrameStats) {
        guard options.enableProfiling, !profilingPaused else { return }
        frameHistory.append(stats)
        if frameHistory.count > maxHistory {
            frameHistory.removeFirst(frameHistory.count - maxHistory)
        }
    }

    func clearHistory() {
        frameHistory.removeAll()
        eventMarks.removeAll()
    }

    func markEvent(_ label: String) {
        eventMarks.append(ProfilingEvent(label: label, frameID: frameHistory.last?.id))
    }

    func selectSortMode(_ mode: SortMode) {
        options.sortMode = mode
        if let scene, scene.count > interactiveGPUSortLimit, mode != .unsorted {
            statusMessage = "\(mode.displayName) requested for a large scene. The renderer will reinitialize that mode's order buffer and may stall while sorting."
        } else {
            statusMessage = nil
        }
        markEvent("Sort: \(mode.rawValue)")
    }

    func exportProfilingData() {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.json]
        panel.nameFieldStringValue = "splat-profile.json"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            let export = ProfilingExport(events: eventMarks, frames: frameHistory)
            let data = try JSONEncoder.pretty.encode(export)
            try data.write(to: url)
        } catch {
            loadError = error.localizedDescription
        }
    }
}

struct ProfilingEvent: Codable, Identifiable {
    var id = UUID()
    var label: String
    var frameID: Int?
    var timestamp = Date()
}

struct ProfilingExport: Codable {
    var events: [ProfilingEvent]
    var frames: [FrameStats]
}

extension JSONEncoder {
    static var pretty: JSONEncoder {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        return encoder
    }
}
