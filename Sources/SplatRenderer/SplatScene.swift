import Foundation
import simd

public struct SplatScene: Sendable {
    public var splats: [Splat]
    public var diagnostics: SplatDiagnostics

    public var bounds: SplatBounds { diagnostics.bounds }
    public var count: Int { splats.count }

    public init(splats: [Splat], diagnostics: SplatDiagnostics) {
        self.splats = splats
        self.diagnostics = diagnostics
    }

    public static func load(url: URL) throws -> SplatScene {
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        return try PLYLoader.load(data: data, sourceURL: url)
    }

    public func packedSplats() -> [PackedSplat] {
        splats.map(PackedSplat.init)
    }

    public func sortedIndices(camera: Camera) -> [UInt32] {
        Self.sortedIndices(for: splats, camera: camera)
    }

    public static func sortedIndices(for splats: [Splat], camera: Camera) -> [UInt32] {
        let view = camera.viewMatrix
        return splats.indices.sorted { lhs, rhs in
            let lp = view * SIMD4<Float>(splats[lhs].position, 1)
            let rp = view * SIMD4<Float>(splats[rhs].position, 1)
            return -lp.z > -rp.z
        }.map(UInt32.init)
    }
}

private enum PLYFormat: String {
    case ascii
    case binaryLittleEndian = "binary_little_endian"
}

private enum PLYScalarType: String {
    case char, uchar, short, ushort, int, uint, float, double

    var byteCount: Int {
        switch self {
        case .char, .uchar: 1
        case .short, .ushort: 2
        case .int, .uint, .float: 4
        case .double: 8
        }
    }
}

private struct PLYProperty {
    var name: String
    var type: PLYScalarType
}

private struct PLYHeader {
    var format: PLYFormat
    var vertexCount: Int
    var properties: [PLYProperty]
    var dataOffset: Int
}

private enum PLYLoader {
    static func load(data: Data, sourceURL: URL?) throws -> SplatScene {
        let header = try parseHeader(data: data)
        if header.format == .binaryLittleEndian {
            return try parseBinaryLittleEndianScene(data: data, header: header, sourceURL: sourceURL)
        }
        let rows: [[String: Float]]
        switch header.format {
        case .ascii:
            rows = try parseASCII(data: data, header: header)
        case .binaryLittleEndian:
            rows = try parseBinaryLittleEndian(data: data, header: header)
        }
        return try makeScene(rows: rows, header: header, sourceURL: sourceURL)
    }

    private static func parseBinaryLittleEndianScene(data: Data, header: PLYHeader, sourceURL: URL?) throws -> SplatScene {
        let names = Set(header.properties.map(\.name))
        let required = ["x", "y", "z"]
        let missing = required.filter { !names.contains($0) }
        guard missing.isEmpty else {
            throw SplatError.missingRequiredFields(missing)
        }

        let hasSHDC = names.isSuperset(of: ["f_dc_0", "f_dc_1", "f_dc_2"])
        let hasRGB = names.isSuperset(of: ["red", "green", "blue"])
        let hasScale = names.isSuperset(of: ["scale_0", "scale_1", "scale_2"])
        let hasRotation = names.isSuperset(of: ["rot_0", "rot_1", "rot_2", "rot_3"])
        let hasOpacity = names.contains("opacity")
        var warnings: [String] = []
        if !hasSHDC && !hasRGB {
            warnings.append("No f_dc_* or RGB color fields found; using white.")
        }
        if !hasScale {
            warnings.append("No scale_* fields found; using small isotropic splats.")
        }
        if !hasRotation {
            warnings.append("No rot_* fields found; using identity rotation.")
        }
        if !hasOpacity {
            warnings.append("No opacity field found; using opaque splats.")
        }

        var offsets: [String: (Int, PLYScalarType)] = [:]
        var stride = 0
        for property in header.properties {
            offsets[property.name] = (stride, property.type)
            stride += property.type.byteCount
        }
        let requiredBytes = header.dataOffset + stride * header.vertexCount
        guard data.count >= requiredBytes else {
            throw SplatError.invalidPLY("binary payload is shorter than vertex count requires")
        }

        func value(_ name: String, rowStart: Int, default defaultValue: Float = 0) throws -> Float {
            guard let (offset, type) = offsets[name] else {
                return defaultValue
            }
            return try readScalar(data: data, offset: rowStart + offset, type: type)
        }

        var splats: [Splat] = []
        splats.reserveCapacity(header.vertexCount)
        for rowIndex in 0..<header.vertexCount {
            let rowStart = header.dataOffset + rowIndex * stride
            let position = SIMD3<Float>(
                try value("x", rowStart: rowStart),
                try value("y", rowStart: rowStart),
                try value("z", rowStart: rowStart)
            )
            let scale = hasScale
                ? SIMD3<Float>(
                    exp(try value("scale_0", rowStart: rowStart)),
                    exp(try value("scale_1", rowStart: rowStart)),
                    exp(try value("scale_2", rowStart: rowStart))
                )
                : SIMD3<Float>(repeating: 0.01)
            let rotation = hasRotation
                ? normalizedQuaternion(SIMD4<Float>(
                    try value("rot_0", rowStart: rowStart, default: 1),
                    try value("rot_1", rowStart: rowStart),
                    try value("rot_2", rowStart: rowStart),
                    try value("rot_3", rowStart: rowStart)
                ))
                : SIMD4<Float>(1, 0, 0, 0)
            let opacity = hasOpacity ? sigmoid(try value("opacity", rowStart: rowStart)) : 1
            let color: SIMD3<Float>
            if hasSHDC {
                let c0: Float = 0.28209479177387814
                color = SIMD3<Float>(
                    clamp(0.5 + c0 * (try value("f_dc_0", rowStart: rowStart)), 0, 1),
                    clamp(0.5 + c0 * (try value("f_dc_1", rowStart: rowStart)), 0, 1),
                    clamp(0.5 + c0 * (try value("f_dc_2", rowStart: rowStart)), 0, 1)
                )
            } else if hasRGB {
                color = SIMD3<Float>(
                    clamp((try value("red", rowStart: rowStart, default: 255)) / 255, 0, 1),
                    clamp((try value("green", rowStart: rowStart, default: 255)) / 255, 0, 1),
                    clamp((try value("blue", rowStart: rowStart, default: 255)) / 255, 0, 1)
                )
            } else {
                color = SIMD3<Float>(repeating: 1)
            }
            splats.append(Splat(position: position, scale: scale, rotation: rotation, opacity: opacity, color: color))
        }
        sortByRenderImportance(&splats)

        let bounds = makeBounds(splats: splats)
        let diagnostics = SplatDiagnostics(
            sourceURL: sourceURL,
            format: header.format.rawValue,
            vertexCount: splats.count,
            fieldAvailability: SplatFieldAvailability(
                hasSHDC: hasSHDC,
                hasRGB: hasRGB,
                hasScale: hasScale,
                hasRotation: hasRotation,
                hasOpacity: hasOpacity
            ),
            bounds: bounds,
            warnings: warnings
        )
        return SplatScene(splats: splats, diagnostics: diagnostics)
    }

    private static func parseHeader(data: Data) throws -> PLYHeader {
        guard let headerRange = data.range(of: Data("end_header\n".utf8)) ?? data.range(of: Data("end_header\r\n".utf8)) else {
            throw SplatError.invalidPLY("missing end_header")
        }
        guard let headerText = String(data: data[..<headerRange.upperBound], encoding: .utf8) else {
            throw SplatError.invalidPLY("header is not UTF-8")
        }

        let lines = headerText
            .split(whereSeparator: \.isNewline)
            .map(String.init)
        guard lines.first == "ply" else {
            throw SplatError.invalidPLY("missing ply magic")
        }

        var format: PLYFormat?
        var vertexCount: Int?
        var properties: [PLYProperty] = []
        var readingVertex = false

        for line in lines.dropFirst() {
            let tokens = line.split(separator: " ").map(String.init)
            guard !tokens.isEmpty else { continue }
            switch tokens[0] {
            case "format":
                guard tokens.count >= 2, let parsed = PLYFormat(rawValue: tokens[1]) else {
                    throw SplatError.unsupportedPLY("only ascii and binary_little_endian are supported")
                }
                format = parsed
            case "element":
                readingVertex = tokens.count >= 3 && tokens[1] == "vertex"
                if readingVertex, let count = Int(tokens[2]) {
                    vertexCount = count
                }
            case "property" where readingVertex:
                guard tokens.count == 3 else {
                    throw SplatError.unsupportedPLY("list properties are not supported for Gaussian vertices")
                }
                guard let type = PLYScalarType(rawValue: tokens[1]) else {
                    throw SplatError.unsupportedPLY("unsupported scalar type \(tokens[1])")
                }
                properties.append(PLYProperty(name: tokens[2], type: type))
            default:
                continue
            }
        }

        guard let format, let vertexCount else {
            throw SplatError.invalidPLY("missing format or vertex element")
        }
        guard !properties.isEmpty else {
            throw SplatError.invalidPLY("vertex element has no scalar properties")
        }

        return PLYHeader(format: format, vertexCount: vertexCount, properties: properties, dataOffset: headerRange.upperBound)
    }

    private static func parseASCII(data: Data, header: PLYHeader) throws -> [[String: Float]] {
        guard let body = String(data: data[header.dataOffset...], encoding: .utf8) else {
            throw SplatError.invalidPLY("ASCII body is not UTF-8")
        }
        let lines = body.split(whereSeparator: \.isNewline)
        guard lines.count >= header.vertexCount else {
            throw SplatError.invalidPLY("vertex count exceeds ASCII row count")
        }
        return try lines.prefix(header.vertexCount).map { line in
            let values = line.split(separator: " ")
            guard values.count >= header.properties.count else {
                throw SplatError.invalidPLY("ASCII vertex row has too few values")
            }
            var row: [String: Float] = [:]
            for (property, raw) in zip(header.properties, values) {
                guard let value = Float(raw) else {
                    throw SplatError.invalidPLY("could not parse float value \(raw)")
                }
                row[property.name] = value
            }
            return row
        }
    }

    private static func parseBinaryLittleEndian(data: Data, header: PLYHeader) throws -> [[String: Float]] {
        let stride = header.properties.reduce(0) { $0 + $1.type.byteCount }
        let requiredBytes = header.dataOffset + stride * header.vertexCount
        guard data.count >= requiredBytes else {
            throw SplatError.invalidPLY("binary payload is shorter than vertex count requires")
        }

        return try (0..<header.vertexCount).map { rowIndex in
            var offset = header.dataOffset + rowIndex * stride
            var row: [String: Float] = [:]
            for property in header.properties {
                row[property.name] = try readScalar(data: data, offset: offset, type: property.type)
                offset += property.type.byteCount
            }
            return row
        }
    }

    private static func readScalar(data: Data, offset: Int, type: PLYScalarType) throws -> Float {
        switch type {
        case .char:
            return Float(Int8(bitPattern: data[offset]))
        case .uchar:
            return Float(data[offset])
        case .short:
            return Float(Int16(bitPattern: loadInteger(data: data, offset: offset)))
        case .ushort:
            return Float(UInt16(littleEndian: loadInteger(data: data, offset: offset)))
        case .int:
            return Float(Int32(bitPattern: loadInteger(data: data, offset: offset)))
        case .uint:
            return Float(UInt32(littleEndian: loadInteger(data: data, offset: offset)))
        case .float:
            let bits: UInt32 = loadInteger(data: data, offset: offset)
            return Float(bitPattern: UInt32(littleEndian: bits))
        case .double:
            let bits: UInt64 = loadInteger(data: data, offset: offset)
            return Float(Double(bitPattern: UInt64(littleEndian: bits)))
        }
    }

    private static func loadInteger<T: FixedWidthInteger>(data: Data, offset: Int) -> T {
        data.withUnsafeBytes { rawBuffer in
            rawBuffer.loadUnaligned(fromByteOffset: offset, as: T.self)
        }
    }

    private static func makeScene(rows: [[String: Float]], header: PLYHeader, sourceURL: URL?) throws -> SplatScene {
        let names = Set(header.properties.map(\.name))
        let required = ["x", "y", "z"]
        let missing = required.filter { !names.contains($0) }
        guard missing.isEmpty else {
            throw SplatError.missingRequiredFields(missing)
        }

        let hasSHDC = names.isSuperset(of: ["f_dc_0", "f_dc_1", "f_dc_2"])
        let hasRGB = names.isSuperset(of: ["red", "green", "blue"])
        let hasScale = names.isSuperset(of: ["scale_0", "scale_1", "scale_2"])
        let hasRotation = names.isSuperset(of: ["rot_0", "rot_1", "rot_2", "rot_3"])
        let hasOpacity = names.contains("opacity")
        var warnings: [String] = []
        if !hasSHDC && !hasRGB {
            warnings.append("No f_dc_* or RGB color fields found; using white.")
        }
        if !hasScale {
            warnings.append("No scale_* fields found; using small isotropic splats.")
        }
        if !hasRotation {
            warnings.append("No rot_* fields found; using identity rotation.")
        }
        if !hasOpacity {
            warnings.append("No opacity field found; using opaque splats.")
        }

        var splats = rows.map { row in
            let position = SIMD3<Float>(row["x"] ?? 0, row["y"] ?? 0, row["z"] ?? 0)
            let scale = hasScale
                ? SIMD3<Float>(exp(row["scale_0"] ?? -5), exp(row["scale_1"] ?? -5), exp(row["scale_2"] ?? -5))
                : SIMD3<Float>(repeating: 0.01)
            let rotation = hasRotation
                ? normalizedQuaternion(SIMD4<Float>(row["rot_0"] ?? 1, row["rot_1"] ?? 0, row["rot_2"] ?? 0, row["rot_3"] ?? 0))
                : SIMD4<Float>(1, 0, 0, 0)
            let opacity = hasOpacity ? sigmoid(row["opacity"] ?? 0) : 1
            let color: SIMD3<Float>
            if hasSHDC {
                let c0: Float = 0.28209479177387814
                color = SIMD3<Float>(
                    clamp(0.5 + c0 * (row["f_dc_0"] ?? 0), 0, 1),
                    clamp(0.5 + c0 * (row["f_dc_1"] ?? 0), 0, 1),
                    clamp(0.5 + c0 * (row["f_dc_2"] ?? 0), 0, 1)
                )
            } else if hasRGB {
                color = SIMD3<Float>(
                    clamp((row["red"] ?? 255) / 255, 0, 1),
                    clamp((row["green"] ?? 255) / 255, 0, 1),
                    clamp((row["blue"] ?? 255) / 255, 0, 1)
                )
            } else {
                color = SIMD3<Float>(repeating: 1)
            }
            return Splat(position: position, scale: scale, rotation: rotation, opacity: opacity, color: color)
        }
        sortByRenderImportance(&splats)

        let bounds = makeBounds(splats: splats)
        let diagnostics = SplatDiagnostics(
            sourceURL: sourceURL,
            format: header.format.rawValue,
            vertexCount: splats.count,
            fieldAvailability: SplatFieldAvailability(
                hasSHDC: hasSHDC,
                hasRGB: hasRGB,
                hasScale: hasScale,
                hasRotation: hasRotation,
                hasOpacity: hasOpacity
            ),
            bounds: bounds,
            warnings: warnings
        )
        return SplatScene(splats: splats, diagnostics: diagnostics)
    }

    private static func makeBounds(splats: [Splat]) -> SplatBounds {
        guard let first = splats.first else {
            return SplatBounds(minimum: [0, 0, 0], maximum: [0, 0, 0], center: [0, 0, 0], radius: 1)
        }
        var minimum = first.position
        var maximum = first.position
        for splat in splats.dropFirst() {
            minimum = simd_min(minimum, splat.position)
            maximum = simd_max(maximum, splat.position)
        }
        let center = (minimum + maximum) * 0.5
        let radius = max(simd_length(maximum - center), 0.01)
        return SplatBounds(
            minimum: [minimum.x, minimum.y, minimum.z],
            maximum: [maximum.x, maximum.y, maximum.z],
            center: [center.x, center.y, center.z],
            radius: radius
        )
    }

    private static func sortByRenderImportance(_ splats: inout [Splat]) {
        splats.sort { lhs, rhs in
            renderImportance(lhs) > renderImportance(rhs)
        }
    }

    private static func renderImportance(_ splat: Splat) -> Float {
        splat.scale.x * splat.scale.y * splat.scale.z * splat.opacity
    }
}
