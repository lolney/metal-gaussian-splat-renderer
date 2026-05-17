import SplatRenderer
import SwiftUI

struct ContentView: View {
    @EnvironmentObject private var store: ViewerStore

    var body: some View {
        HSplitView {
            ZStack(alignment: .topLeading) {
                MetalViewport()
                    .environmentObject(store)

                TimingOverlay()
                    .environmentObject(store)
                    .padding(12)
            }
            .frame(minWidth: 680)

            if store.profilingVisible {
                ProfilingPanel()
                    .environmentObject(store)
                    .frame(minWidth: 330, idealWidth: 380, maxWidth: 460)
            }
        }
        .toolbar {
            ToolbarItemGroup {
                Button {
                    store.openPLY()
                } label: {
                    Label("Open", systemImage: "folder")
                }

                Picker("Sort", selection: $store.options.sortMode) {
                    ForEach(SortMode.allCases) { mode in
                        Text(mode.rawValue.uppercased()).tag(mode)
                    }
                }
                .pickerStyle(.segmented)
                .frame(width: 130)
                .onChange(of: store.options.sortMode) { _, mode in
                    store.markEvent("Sort: \(mode.rawValue)")
                }

                Toggle(isOn: $store.profilingVisible) {
                    Label("Profiling", systemImage: "chart.xyaxis.line")
                }
            }
        }
        .alert("Load Error", isPresented: Binding(get: { store.loadError != nil }, set: { _ in store.loadError = nil })) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(store.loadError ?? "")
        }
    }
}

private struct TimingOverlay: View {
    @EnvironmentObject private var store: ViewerStore

    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            if let last = store.frameHistory.last {
                Text(String(format: "%.2f ms  %.0f FPS", last.totalFrameMilliseconds, 1000 / max(last.totalFrameMilliseconds, 0.001)))
                    .font(.system(.headline, design: .monospaced))
                Text("\(last.totalSplats) splats  \(last.sortMode.rawValue.uppercased()) sort")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                Text("Drop or open a Gaussian .ply")
                    .font(.headline)
                Text("No profiling frames yet")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(10)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 8))
    }
}
