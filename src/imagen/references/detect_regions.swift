import Foundation
import ImageIO
import Vision

// Emit geometry only. No image previews or recognized text leave this helper.
guard CommandLine.arguments.count > 1 else {
    fputs("Usage: detect_regions.swift IMAGE [IMAGE ...]\n", stderr)
    exit(2)
}

func candidate(_ rectangle: CGRect, kind: String, confidence: Float,
               detector: String) -> [String: Any] {
    return ["kind": kind, "confidence": Double(confidence), "detector": detector,
            "box": [rectangle.minX, 1 - rectangle.maxY,
                    rectangle.maxX, 1 - rectangle.minY]]
}

do {
    var images: [[String: Any]] = []
    for (index, path) in CommandLine.arguments.dropFirst().enumerated() {
        let started = Date()
        let url = URL(fileURLWithPath: path)
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              let storedWidth = properties[kCGImagePropertyPixelWidth] as? NSNumber,
              let storedHeight = properties[kCGImagePropertyPixelHeight] as? NSNumber
        else {
            throw NSError(domain: "GaonRegions", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "Cannot read input image properties"])
        }
        let rawOrientation = (properties[kCGImagePropertyOrientation] as? NSNumber)?.uint32Value ?? 1
        let orientation = CGImagePropertyOrientation(rawValue: rawOrientation) ?? .up
        let swapsAxes = [UInt32(5), 6, 7, 8].contains(rawOrientation)
        let width = swapsAxes ? storedHeight.intValue : storedWidth.intValue
        let height = swapsAxes ? storedWidth.intValue : storedHeight.intValue
        let handler = VNImageRequestHandler(url: url, orientation: orientation, options: [:])
        var candidates: [[String: Any]] = []
        var errors: [[String: String]] = []
        var detectorSeconds: [String: Double] = [:]

        let faces = VNDetectFaceRectanglesRequest()
        var detectorStarted = Date()
        do {
            try handler.perform([faces])
            for face in faces.results ?? [] {
                candidates.append(candidate(face.boundingBox, kind: "face",
                                            confidence: face.confidence, detector: "face"))
            }
        } catch {
            errors.append(["detector": "face", "error": error.localizedDescription])
        }
        detectorSeconds["face"] = Date().timeIntervalSince(detectorStarted)

        let text = VNRecognizeTextRequest()
        detectorStarted = Date()
        text.recognitionLevel = .accurate
        text.automaticallyDetectsLanguage = true
        text.usesLanguageCorrection = false
        text.minimumTextHeight = 0.006
        do {
            try handler.perform([text])
            for observation in text.results ?? [] {
                guard let recognized = observation.topCandidates(1).first,
                      !recognized.string.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                else { continue }
                candidates.append(candidate(observation.boundingBox, kind: "text",
                                            confidence: recognized.confidence, detector: "text"))
            }
        } catch {
            errors.append(["detector": "text", "error": error.localizedDescription])
        }
        detectorSeconds["text"] = Date().timeIntervalSince(detectorStarted)

        let objects = VNGenerateObjectnessBasedSaliencyImageRequest()
        detectorStarted = Date()
        do {
            try handler.perform([objects])
            for object in objects.results?.first?.salientObjects ?? [] {
                candidates.append(candidate(object.boundingBox, kind: "object",
                                            confidence: object.confidence, detector: "objectness"))
            }
        } catch {
            errors.append(["detector": "objectness", "error": error.localizedDescription])
        }
        detectorSeconds["objectness"] = Date().timeIntervalSince(detectorStarted)

        let attention = VNGenerateAttentionBasedSaliencyImageRequest()
        detectorStarted = Date()
        do {
            try handler.perform([attention])
            for object in attention.results?.first?.salientObjects ?? [] {
                candidates.append(candidate(object.boundingBox, kind: "object",
                                            confidence: object.confidence, detector: "attention"))
            }
        } catch {
            errors.append(["detector": "attention", "error": error.localizedDescription])
        }
        detectorSeconds["attention"] = Date().timeIntervalSince(detectorStarted)
        images.append(["source": index + 1, "size": [width, height],
                       "candidates": candidates, "errors": errors,
                       "detector_seconds": detectorSeconds,
                       "seconds": Date().timeIntervalSince(started)])
    }
    let data = try JSONSerialization.data(withJSONObject: ["version": 1, "images": images],
                                          options: [.sortedKeys])
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write(Data("\n".utf8))
} catch {
    fputs("Region detection failed: \(error.localizedDescription)\n", stderr)
    exit(1)
}
