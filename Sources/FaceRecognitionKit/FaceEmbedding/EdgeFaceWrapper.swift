//
//  EdgeFaceWrapper.swift
//  FaceReferenceArge
//
//  Created by Harun SUBAŞI on 16.09.2025.
//

import CoreImage
import CoreVideo
import Foundation
import Vision

public class EdgeFaceWrapper {
    let model: EdgeFaceXS

    public init() {
        model = try! EdgeFaceXS()
    }

    public func getEmbedding(facePixelBuffer: CVPixelBuffer, normalize: Bool = false) -> [Float] {
        let input = EdgeFaceXSInput(input_image: facePixelBuffer)
        guard let prediction = try? model.prediction(input: input) else { return [] }

        guard let featureValue = prediction.featureValue(for: "output"),
            featureValue.type == .multiArray,
            let embeddingArray = featureValue.multiArrayValue
        else { return [] }

        let floatEmbedding = (0..<embeddingArray.count).map { i in Float(truncating: embeddingArray[i]) }

        if normalize {
            let norm = sqrt(floatEmbedding.map { $0 * $0 }.reduce(0, +))
            guard norm > 0 else { return floatEmbedding }
            return floatEmbedding.map { $0 / norm }
        }

        return floatEmbedding
    }

}
