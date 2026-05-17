# Metal Gaussian Splat Renderer

Apple Silicon-first Gaussian splat renderer scaffold with:

- `SplatRenderer`: reusable Swift/Metal renderer library.
- `SplatViewer`: macOS SwiftUI + MetalKit viewer for Gaussian `.ply` files.
- `splatbench`: headless benchmark runner with JSON/CSV timing output and optional Metal capture.

## Build

```bash
swift build
```

## Run Viewer

```bash
script/build_and_run.sh
```

Use **Open** or `Command-O` to load a Gaussian `.ply`.

To launch directly into a scene:

```bash
script/build_and_run.sh /path/to/scene.ply
```

## Run Benchmarks

```bash
swift run splatbench --input /path/to/scene.ply --frames 120 --width 1280 --height 720 --sort gpu --output results.json
```

This writes `results.json` and a sibling `results.csv`.
