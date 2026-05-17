// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "MetalGaussianSplatRenderer",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .library(name: "SplatRenderer", targets: ["SplatRenderer"]),
        .executable(name: "SplatViewer", targets: ["SplatViewer"]),
        .executable(name: "splatbench", targets: ["splatbench"])
    ],
    targets: [
        .target(
            name: "SplatRenderer",
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ],
            linkerSettings: [
                .linkedFramework("Metal"),
                .linkedFramework("MetalKit"),
                .linkedFramework("QuartzCore")
            ]
        ),
        .executableTarget(
            name: "SplatViewer",
            dependencies: ["SplatRenderer"],
            path: "Sources/SplatViewer",
            linkerSettings: [
                .linkedFramework("AppKit"),
                .linkedFramework("SwiftUI"),
                .linkedFramework("MetalKit")
            ]
        ),
        .executableTarget(
            name: "splatbench",
            dependencies: ["SplatRenderer"],
            path: "Sources/splatbench",
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ],
            linkerSettings: [
                .linkedFramework("Metal"),
                .linkedFramework("MetalKit")
            ]
        ),
        .testTarget(
            name: "SplatRendererTests",
            dependencies: ["SplatRenderer"]
        )
    ]
)
