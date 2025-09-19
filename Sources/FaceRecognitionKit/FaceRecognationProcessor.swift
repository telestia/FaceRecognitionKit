//
//  FaceRecognationProcessoe.swift
//  Swix
//
//  Created by Harun SUBAŞI on 17.09.2025.
//

import AVFoundation
import CoreImage
import CoreML
import PhotosUI
import SwiftUI
import Vision

enum ProcessingState {
    case processing(progress: Float, stage: String)
    case completed(result: [FaceCluster])
    case error(message: String)
}

public class FaceRecognationProcessor: ObservableObject {
    let edgeFace: EdgeFaceWrapper

    public init(edgeFace: EdgeFaceWrapper = EdgeFaceWrapper()) {
        self.edgeFace = edgeFace
    }

    public func processVideo(url: URL, progressHandler: ((ProcessingState) -> Void)? = nil) {
        DispatchQueue.global(qos: .userInitiated).async {
            self._processVideoSync(url: url, progressHandler: progressHandler)
        }
    }

    private func _processVideoSync(
        url: URL,
        progressHandler: ((ProcessingState) -> Void)? = nil
    ) {
        let asset = AVAsset(url: url)
        guard let reader = try? AVAssetReader(asset: asset),
            let videoTrack = asset.tracks(withMediaType: .video).first
        else {
            progressHandler?(.error(message: "Video dosyası okunamadı"))
            return
        }

        let duration = asset.duration
        let totalFrames = Int(duration.seconds * Double(videoTrack.nominalFrameRate))

        progressHandler?(.processing(progress: 0.0, stage: "Video analiz ediliyor..."))

        let readerOutputSettings: [String: Any] = [
            (kCVPixelBufferPixelFormatTypeKey as String): kCVPixelFormatType_32BGRA
        ]
        let readerOutput = AVAssetReaderTrackOutput(track: videoTrack, outputSettings: readerOutputSettings)
        reader.add(readerOutput)
        reader.startReading()

        var embeddings: [[Float]] = []
        var faceCrops: [FaceCrop] = []
        var frameIndex = 0

        while let sampleBuffer = readerOutput.copyNextSampleBuffer() {
            if frameIndex % 30 == 0 && totalFrames > 0 {
                let progress = Float(frameIndex) / Float(totalFrames) * 0.8
                progressHandler?(.processing(progress: progress, stage: "Yüzler tespit ediliyor... (\(embeddings.count) yüz bulundu)"))
            }

            if frameIndex % 5 == 0, let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) {
                let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
                let request = VNDetectFaceRectanglesRequest()
                let handler = VNImageRequestHandler(ciImage: ciImage, options: [:])
                try? handler.perform([request])

                if let results = request.results {
                    // Sort faces by area (largest first) to assign frame face index
                    let sortedResults = results.enumerated().sorted { (first, second) in
                        let firstArea = first.element.boundingBox.width * first.element.boundingBox.height
                        let secondArea = second.element.boundingBox.width * second.element.boundingBox.height
                        return firstArea > secondArea
                    }

                    for (frameFaceIndex, (_, face)) in sortedResults.enumerated() {
                        let rect = face.boundingBox
                        if let faceCropBuffer = cropFace(pixelBuffer: pixelBuffer, rect: rect),
                            let resizedBuffer = resizePixelBuffer(faceCropBuffer, width: 112, height: 112)
                        {
                            let embedding = edgeFace.getEmbedding(facePixelBuffer: resizedBuffer, normalize: false)
                            if !embedding.isEmpty {
                                let faceCrop = FaceCrop(
                                    frameIndex: frameIndex,
                                    frameFaceIndex: frameFaceIndex,
                                    rect: rect,
                                    pixelBuffer: faceCropBuffer,
                                    embedding: embedding,
                                    confidence: face.confidence
                                )
                                embeddings.append(embedding)
                                faceCrops.append(faceCrop)
                            }
                        }
                    }
                }
            }
            frameIndex += 1
        }

        if embeddings.isEmpty {
            progressHandler?(.error(message: "Videoda hiç yüz bulunamadı"))
            return
        }

        progressHandler?(.processing(progress: 0.8, stage: "Yüzler kümeleniyor..."))

        let params = DBSCAN.autoTune(data: embeddings, metric: .angular)
        print("DBSCAN Params: \(params)")
        let clustering = DBSCAN(eps: params.eps, minSamples: params.minSamples, metric: .angular).fit(embeddings)
        let labels = clustering.labels_

        progressHandler?(.processing(progress: 0.9, stage: "Kümeler oluşturuluyor..."))

        var clustersDict: [Int: FaceCluster] = [:]
        for (i, label) in labels.enumerated() {
            if label == -1 { continue }
            if clustersDict[label] == nil {
                clustersDict[label] = FaceCluster(id: label)
            }
            clustersDict[label]?.add(face: faceCrops[i])
        }

        let result = clustersDict.values.sorted { $0.getClusterQuality() > $1.getClusterQuality() }
        progressHandler?(.completed(result: result))
    }

    private func cropFace(pixelBuffer: CVPixelBuffer, rect: CGRect) -> CVPixelBuffer? {
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let width = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
        let height = CGFloat(CVPixelBufferGetHeight(pixelBuffer))

        // VNFaceObservation Vision koordinat sistemi (sol alt origin) -> Core Image koordinat sistemi (sol üst origin)
        // Vision: Y=0 altta, Y=1 üstte
        // Core Image: Y=0 üstte, Y=height altta
        let cropRect = CGRect(
            x: rect.minX * width,
            y: rect.minY * height,  // Doğru dönüşüm: rect.minY kullan
            width: rect.width * width,
            height: rect.height * height
        )

        // Sınırları kontrol et (güvenlik için)
        let safeCropRect = cropRect.intersection(CGRect(x: 0, y: 0, width: width, height: height))

        // Crop işlemi
        let cropped = ciImage.cropped(to: safeCropRect)

        // Origin'i (0,0)'a taşı
        let translated = cropped.transformed(
            by: CGAffineTransform(
                translationX: -cropped.extent.origin.x,
                y: -cropped.extent.origin.y
            )
        )

        // 112x112'ye resize et (aspect ratio'yu koru)
        let targetSize: CGFloat = 112
        let scaleX = targetSize / translated.extent.width
        let scaleY = targetSize / translated.extent.height
        let scale = min(scaleX, scaleY)

        // Lanczos filter ile yüksek kaliteli resize
        guard let lanczosFilter = CIFilter(name: "CILanczosScaleTransform") else { return nil }
        lanczosFilter.setValue(translated, forKey: kCIInputImageKey)
        lanczosFilter.setValue(scale, forKey: kCIInputScaleKey)
        lanczosFilter.setValue(1.0, forKey: kCIInputAspectRatioKey)

        guard let scaled = lanczosFilter.outputImage else { return nil }

        // Merkeze yerleştir (aspect ratio korunduğu için gerekli olabilir)
        let scaledWidth = translated.extent.width * scale
        let scaledHeight = translated.extent.height * scale
        let offsetX = (targetSize - scaledWidth) / 2
        let offsetY = (targetSize - scaledHeight) / 2

        let finalImage = scaled.transformed(
            by: CGAffineTransform(translationX: offsetX, y: offsetY)
        )

        // Output buffer oluştur
        var outputBuffer: CVPixelBuffer?
        let attrs: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true,
            kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_32BGRA,
        ]

        let status = CVPixelBufferCreate(
            nil,
            Int(targetSize),
            Int(targetSize),
            kCVPixelFormatType_32BGRA,
            attrs as CFDictionary,
            &outputBuffer
        )

        guard status == kCVReturnSuccess, let outBuffer = outputBuffer else { return nil }

        // Render
        let context = CIContext(options: [
            .workingColorSpace: CGColorSpaceCreateDeviceRGB(),
            .outputColorSpace: CGColorSpaceCreateDeviceRGB(),
            .useSoftwareRenderer: false,
        ])

        context.render(
            finalImage,
            to: outBuffer,
            bounds: CGRect(x: 0, y: 0, width: targetSize, height: targetSize),
            colorSpace: CGColorSpaceCreateDeviceRGB()
        )

        return outBuffer
    }

    func resizePixelBuffer(_ pixelBuffer: CVPixelBuffer, width: Int, height: Int) -> CVPixelBuffer? {
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)

        // Calculate scale to maintain aspect ratio and center crop if needed
        let sourceWidth = ciImage.extent.width
        let sourceHeight = ciImage.extent.height
        let targetWidth = CGFloat(width)
        let targetHeight = CGFloat(height)

        let scaleX = targetWidth / sourceWidth
        let scaleY = targetHeight / sourceHeight
        let scale = max(scaleX, scaleY)  // Use max to ensure the image fills the target size

        // Scale the image
        let scaled = ciImage.transformed(by: CGAffineTransform(scaleX: scale, y: scale))

        // Center crop if needed
        let scaledWidth = sourceWidth * scale
        let scaledHeight = sourceHeight * scale
        let offsetX = (scaledWidth - targetWidth) / 2
        let offsetY = (scaledHeight - targetHeight) / 2

        let cropped = scaled.cropped(
            to: CGRect(
                x: offsetX,
                y: offsetY,
                width: targetWidth,
                height: targetHeight
            )
        )

        // Ensure the final image is positioned at origin
        let final = cropped.transformed(
            by: CGAffineTransform(
                translationX: -cropped.extent.origin.x,
                y: -cropped.extent.origin.y
            )
        )

        let attrs: [String: Any] = [
            kCVPixelBufferCGImageCompatibilityKey as String: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey as String: true,
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        ]

        var outputBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(nil, width, height, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &outputBuffer)

        guard status == kCVReturnSuccess, let outBuffer = outputBuffer else { return nil }

        // Use RGB color space consistently
        let context = CIContext(options: [
            .workingColorSpace: CGColorSpaceCreateDeviceRGB(),
            .outputColorSpace: CGColorSpaceCreateDeviceRGB(),
        ])

        context.render(final, to: outBuffer, bounds: CGRect(x: 0, y: 0, width: width, height: height), colorSpace: CGColorSpaceCreateDeviceRGB())
        return outBuffer
    }

}
