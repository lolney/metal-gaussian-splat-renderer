import Foundation
import SplatRenderer
import simd

struct BicycleCameraPreset {
    var id: Int
    var imageName: String
    var sourceWidth: Float
    var sourceHeight: Float
    var position: SIMD3<Float>
    var rotation: simd_float3x3
    var fx: Float
    var fy: Float

    func camera(width: Int, height: Int) -> Camera {
        let view = viewMatrix()
        let projection = projectionMatrix(width: width, height: height)
        return Camera(
            viewMatrix: view,
            projectionMatrix: projection,
            viewportSize: SIMD2<Float>(Float(width), Float(height))
        )
    }

    func projectionMatrix(width: Int, height: Int) -> simd_float4x4 {
        let projection = simd_float4x4.perspective(
            fx: fx,
            fy: fy,
            sourceWidth: Float(max(width, 1)),
            sourceHeight: Float(max(height, 1)),
            nearZ: 0.2,
            farZ: 200
        )
        return projection
    }

    func viewMatrix() -> simd_float4x4 {
        let r00 = rotation.columns.0.x
        let r01 = rotation.columns.1.x
        let r02 = rotation.columns.2.x
        let r10 = rotation.columns.0.y
        let r11 = rotation.columns.1.y
        let r12 = rotation.columns.2.y
        let r20 = rotation.columns.0.z
        let r21 = rotation.columns.1.z
        let r22 = rotation.columns.2.z
        let t0 = -(position.x * r00 + position.y * r10 + position.z * r20)
        let t1 = -(position.x * r01 + position.y * r11 + position.z * r21)
        let t2 = -(position.x * r02 + position.y * r12 + position.z * r22)

        return simd_float4x4(columns: (
            SIMD4<Float>(r00, -r01, -r02, 0),
            SIMD4<Float>(r10, -r11, -r12, 0),
            SIMD4<Float>(r20, -r21, -r22, 0),
            SIMD4<Float>(t0, -t1, -t2, 1)
        ))
    }
}

extension BicycleCameraPreset {
    static func preset(for key: Int) -> BicycleCameraPreset? {
        presets.first { $0.id == key }
    }

    static func isBicycleScene(_ url: URL?) -> Bool {
        url?.lastPathComponent.localizedCaseInsensitiveContains("bicycle") == true
    }

    static let presets: [BicycleCameraPreset] = [
        .init(
            id: 0,
            imageName: "00001",
            sourceWidth: 1959,
            sourceHeight: 1090,
            position: SIMD3<Float>(-3.0089893, -0.1108649, -3.752764),
            rotation: simd_float3x3(rows: [
                SIMD3<Float>(0.8761342, 0.06925962, 0.477066),
                SIMD3<Float>(-0.04747422, 0.9972111, -0.05758674),
                SIMD3<Float>(-0.47972393, 0.02780538, 0.8769788)
            ]),
            fx: 1159.5881,
            fy: 1164.6602
        ),
        .init(
            id: 1,
            imageName: "00009",
            sourceWidth: 1959,
            sourceHeight: 1090,
            position: SIMD3<Float>(-2.5199776, -0.09704736, -3.6247725),
            rotation: simd_float3x3(rows: [
                SIMD3<Float>(0.99827313, -0.01192871, -0.05751927),
                SIMD3<Float>(0.00650614, 0.99559283, -0.09355534),
                SIMD3<Float>(0.05838177, 0.09301955, 0.9939512)
            ]),
            fx: 1159.5881,
            fy: 1164.6602
        ),
        .init(
            id: 2,
            imageName: "00017",
            sourceWidth: 1959,
            sourceHeight: 1090,
            position: SIMD3<Float>(-0.77375335, -0.3364272, -2.935897),
            rotation: simd_float3x3(rows: [
                SIMD3<Float>(0.9998813, 0.01374238, -0.00696055),
                SIMD3<Float>(-0.01426837, 0.99651295, -0.08220929),
                SIMD3<Float>(0.00580653, 0.08229885, 0.9965908)
            ]),
            fx: 1159.5881,
            fy: 1164.6602
        ),
        .init(
            id: 3,
            imageName: "00025",
            sourceWidth: 1959,
            sourceHeight: 1090,
            position: SIMD3<Float>(1.2198222, -0.21966879, -2.3183162),
            rotation: simd_float3x3(rows: [
                SIMD3<Float>(0.9208649, 0.00120106, 0.38988),
                SIMD3<Float>(-0.06298204, 0.9873195, 0.14571694),
                SIMD3<Float>(-0.38476112, -0.15874104, 0.90926355)
            ]),
            fx: 1159.5881,
            fy: 1164.6602
        ),
        .init(
            id: 4,
            imageName: "00033",
            sourceWidth: 1959,
            sourceHeight: 1090,
            position: SIMD3<Float>(1.7423879, -0.13848226, -2.056637),
            rotation: simd_float3x3(rows: [
                SIMD3<Float>(0.24669889, -0.08370189, -0.9654707),
                SIMD3<Float>(0.11343748, 0.99190825, -0.05700815),
                SIMD3<Float>(0.96243006, -0.09545671, 0.2541976)
            ]),
            fx: 1159.5881,
            fy: 1164.6602
        ),
        .init(
            id: 5,
            imageName: "00041",
            sourceWidth: 1959,
            sourceHeight: 1090,
            position: SIMD3<Float>(3.656731, -0.16470991, -1.3458085),
            rotation: simd_float3x3(rows: [
                SIMD3<Float>(0.23412931, -0.0296833, -0.9717522),
                SIMD3<Float>(0.10270824, 0.99469554, -0.00563811),
                SIMD3<Float>(0.9667649, -0.09848691, 0.2359361)
            ]),
            fx: 1159.5881,
            fy: 1164.6602
        ),
        .init(
            id: 6,
            imageName: "00049",
            sourceWidth: 1959,
            sourceHeight: 1090,
            position: SIMD3<Float>(3.9013555, -0.2597501, -0.8106154),
            rotation: simd_float3x3(rows: [
                SIMD3<Float>(0.67172354, -0.01571816, -0.74063516),
                SIMD3<Float>(0.05562735, 0.99802244, 0.02927099),
                SIMD3<Float>(0.7387104, -0.06086159, 0.67126954)
            ]),
            fx: 1159.5881,
            fy: 1164.6602
        ),
        .init(
            id: 7,
            imageName: "00057",
            sourceWidth: 1959,
            sourceHeight: 1090,
            position: SIMD3<Float>(4.7429946, -0.05591661, 0.9500366),
            rotation: simd_float3x3(rows: [
                SIMD3<Float>(-0.17042656, 0.01207081, -0.9852964),
                SIMD3<Float>(0.11650904, 0.9931575, -0.00798543),
                SIMD3<Float>(0.97845817, -0.11615687, -0.17066678)
            ]),
            fx: 1159.5881,
            fy: 1164.6602
        ),
        .init(
            id: 8,
            imageName: "00065",
            sourceWidth: 1959,
            sourceHeight: 1090,
            position: SIMD3<Float>(4.346763, 0.08168161, 1.0876222),
            rotation: simd_float3x3(rows: [
                SIMD3<Float>(-0.00357545, -0.0447925, -0.99898994),
                SIMD3<Float>(0.10770153, 0.9931681, -0.04491694),
                SIMD3<Float>(0.99417686, -0.10775334, 0.0012732)
            ]),
            fx: 1159.5881,
            fy: 1164.6602
        ),
        .init(
            id: 9,
            imageName: "00073",
            sourceWidth: 1959,
            sourceHeight: 1090,
            position: SIMD3<Float>(3.2649844, 0.07897494, 1.01172),
            rotation: simd_float3x3(rows: [
                SIMD3<Float>(-0.02691999, -0.1565891, -0.9872969),
                SIMD3<Float>(0.08444552, 0.9837682, -0.15833198),
                SIMD3<Float>(0.99606436, -0.0876351, -0.01325979)
            ]),
            fx: 1159.5881,
            fy: 1164.6602
        )
    ]
}
