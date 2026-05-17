import AppKit
import SwiftUI

@main
struct SplatViewerApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @StateObject private var store = ViewerStore()

    var body: some Scene {
        WindowGroup("Splat Viewer") {
            ContentView()
                .environmentObject(store)
                .frame(minWidth: 1100, minHeight: 720)
        }
        .commands {
            CommandGroup(after: .newItem) {
                Button("Open PLY...") {
                    store.openPLY()
                }
                .keyboardShortcut("o")

                Button("Export Profiling Data...") {
                    store.exportProfilingData()
                }
                .keyboardShortcut("e", modifiers: [.command, .shift])
            }
        }
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
    }
}
