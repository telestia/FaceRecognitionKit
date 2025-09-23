//
//  FaceCrop.swift
//  Swix
//
//  Created by Harun SUBAŞI on 17.09.2025.
//

import Foundation
import Vision

public struct FaceCrop: Identifiable {
    public let id: UUID = UUID()  // Unique identifier for SwiftUI
    public let frameIndex: Int
    public let frameFaceIndex: Int  // Index of this face in the frame (0 = largest face in frame)
    public let rect: CGRect
    public let pixelBuffer: CVPixelBuffer
    public let embedding: [Float]
    public let confidence: Float  // Face detection confidence
    public var clusterRank: Int = -1  // Rank within cluster (0 = best, largest face)
    public var qualityScore: Float = 0.0  // Overall quality score for this face

    public init(frameIndex: Int, frameFaceIndex: Int, rect: CGRect, pixelBuffer: CVPixelBuffer, embedding: [Float], confidence: Float) {
        self.frameIndex = frameIndex
        self.frameFaceIndex = frameFaceIndex
        self.rect = rect
        self.pixelBuffer = pixelBuffer
        self.embedding = embedding
        self.confidence = confidence
        self.qualityScore = calculateQualityScore()
    }

    private func calculateQualityScore() -> Float {
        // Calculate quality based on face size and detection confidence
        let faceArea = rect.width * rect.height
        let sizeScore = min(faceArea * 4.0, 1.0)  // Larger faces get higher scores
        let confidenceScore = confidence
        return (Float(sizeScore) + confidenceScore) / 2.0
    }
}
