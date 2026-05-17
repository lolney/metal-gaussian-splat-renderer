import AppKit
import Combine
import Foundation
import SplatRenderer
import SwiftUI

@MainActor
final class ViewerStore: ObservableObject {
    @Published var scene: SplatScene?
    @Published var loadError: String?
    @Published var options = RenderOptions(enableProfiling: true, waitForGPU: true)
    @Published var profilingVisible = true
    @Published var profilingPaused = false
    @Published var frameHistory: [FrameStats] = []
    @Published var eventMarks: [ProfilingEvent] = []

    let maxHistory = 600

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
        do {
            let loaded = try SplatScene.load(url: url)
            scene = loaded
            loadError = nil
            markEvent("Loaded \(url.lastPathComponent)")
        } catch {
            loadError = error.localizedDescription
        }
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
