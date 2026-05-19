import SplatRenderer
import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @EnvironmentObject private var store: ViewerStore
    @State private var didLoadLaunchArgument = false

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
            .onDrop(of: [UTType.fileURL.identifier], isTargeted: nil) { providers in
                loadDroppedFile(providers: providers)
            }

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

                Picker("Sort", selection: Binding(
                    get: { store.options.sortMode },
                    set: { store.selectSortMode($0) }
                )) {
                    ForEach(SortMode.allCases) { mode in
                        Text(mode.displayName).tag(mode)
                    }
                }
                .pickerStyle(.segmented)
                .frame(width: 300)

                Toggle(isOn: Binding(
                    get: { store.options.useProjectionCache },
                    set: {
                        store.options.useProjectionCache = $0
                        store.markEvent($0 ? "Projection cache on" : "Projection cache off")
                    }
                )) {
                    Label("Projection Cache", systemImage: "rectangle.3.group")
                }
                .help("Project and cull splats once per frame before rasterization")

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
        .onAppear {
            loadLaunchArgumentIfNeeded()
        }
    }

    private func loadLaunchArgumentIfNeeded() {
        guard !didLoadLaunchArgument else { return }
        didLoadLaunchArgument = true
        guard let path = CommandLine.arguments.dropFirst().first, path.hasSuffix(".ply") else { return }
        store.load(url: URL(fileURLWithPath: path))
    }

    private func loadDroppedFile(providers: [NSItemProvider]) -> Bool {
        guard let provider = providers.first(where: { $0.hasItemConformingToTypeIdentifier(UTType.fileURL.identifier) }) else {
            return false
        }
        provider.loadItem(forTypeIdentifier: UTType.fileURL.identifier, options: nil) { item, _ in
            let url: URL?
            if let data = item as? Data {
                url = URL(dataRepresentation: data, relativeTo: nil)
            } else {
                url = item as? URL
            }
            guard let url, url.pathExtension.lowercased() == "ply" else { return }
            Task { @MainActor in
                store.load(url: url)
            }
        }
        return true
    }
}

private struct TimingOverlay: View {
    @EnvironmentObject private var store: ViewerStore

    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            if let last = store.frameHistory.last {
                Text(String(format: "%.2f ms  %.0f FPS", last.totalFrameMilliseconds, 1000 / max(last.totalFrameMilliseconds, 0.001)))
                    .font(.system(.headline, design: .monospaced))
                Text("\(last.visibleSplats.formatted()) / \(last.totalSplats.formatted()) splats  \(last.sortMode.displayName) sort")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                Text(store.isLoading ? "Loading scene..." : "Drop or open a Gaussian .ply")
                    .font(.headline)
                Text(store.statusMessage ?? "No profiling frames yet")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(10)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 8))
    }
}
