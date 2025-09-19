//
//  DistanceMetric.swift
//  Swix
//
//  Created by Harun SUBAŞI on 17.09.2025.
//

import Foundation
import Accelerate

enum DistanceMetric {
    case euclidean
    case cosine
    case manhattan
    case normalizedEuclidean  // L2 normalized vectors + Euclidean
    case angular  // Angular distance (better for high-dim)
    case correlation  // Pearson correlation distance

    func distance(_ a: [Float], _ b: [Float]) -> Float {
        switch self {
        case .euclidean:
            return euclideanDistance(a, b)
        case .cosine:
            return cosineDistance(a, b)
        case .manhattan:
            return manhattanDistance(a, b)
        case .normalizedEuclidean:
            return normalizedEuclideanDistance(a, b)
        case .angular:
            return angularDistance(a, b)
        case .correlation:
            return correlationDistance(a, b)
        }
    }

    private func euclideanDistance(_ a: [Float], _ b: [Float]) -> Float {
        var diff = [Float](repeating: 0, count: a.count)
        vDSP_vsub(b, 1, a, 1, &diff, 1, vDSP_Length(a.count))

        var squaredDiff = [Float](repeating: 0, count: a.count)
        vDSP_vsq(diff, 1, &squaredDiff, 1, vDSP_Length(a.count))

        var sum: Float = 0.0
        vDSP_sve(squaredDiff, 1, &sum, vDSP_Length(a.count))

        return sqrt(sum)
    }

    private func cosineDistance(_ a: [Float], _ b: [Float]) -> Float {
        // Cosine distance = 1 - cosine similarity
        var dotProduct: Float = 0.0
        vDSP_dotpr(a, 1, b, 1, &dotProduct, vDSP_Length(a.count))

        var normA: Float = 0.0
        vDSP_svesq(a, 1, &normA, vDSP_Length(a.count))
        normA = sqrt(normA)

        var normB: Float = 0.0
        vDSP_svesq(b, 1, &normB, vDSP_Length(b.count))
        normB = sqrt(normB)

        if normA == 0.0 || normB == 0.0 {
            return 1.0
        }

        let cosineSimilarity = dotProduct / (normA * normB)
        return 1.0 - cosineSimilarity
    }

    private func manhattanDistance(_ a: [Float], _ b: [Float]) -> Float {
        var diff = [Float](repeating: 0, count: a.count)
        vDSP_vsub(b, 1, a, 1, &diff, 1, vDSP_Length(a.count))

        var absDiff = [Float](repeating: 0, count: a.count)
        vDSP_vabs(diff, 1, &absDiff, 1, vDSP_Length(a.count))

        var sum: Float = 0.0
        vDSP_sve(absDiff, 1, &sum, vDSP_Length(a.count))

        return sum
    }

    private func normalizedEuclideanDistance(_ a: [Float], _ b: [Float]) -> Float {
        // L2 normalize both vectors first, then compute Euclidean
        var normA: Float = 0.0
        vDSP_svesq(a, 1, &normA, vDSP_Length(a.count))
        normA = sqrt(normA)

        var normB: Float = 0.0
        vDSP_svesq(b, 1, &normB, vDSP_Length(b.count))
        normB = sqrt(normB)

        var normalizedA = [Float](repeating: 0, count: a.count)
        var normalizedB = [Float](repeating: 0, count: b.count)

        if normA > 0 {
            var scale = 1.0 / normA
            vDSP_vsmul(a, 1, &scale, &normalizedA, 1, vDSP_Length(a.count))
        } else {
            normalizedA = a
        }

        if normB > 0 {
            var scale = 1.0 / normB
            vDSP_vsmul(b, 1, &scale, &normalizedB, 1, vDSP_Length(b.count))
        } else {
            normalizedB = b
        }

        return euclideanDistance(normalizedA, normalizedB)
    }

    private func angularDistance(_ a: [Float], _ b: [Float]) -> Float {
        // Angular distance = arccos(cosine_similarity) / π
        // More sensitive to differences than cosine distance
        var dotProduct: Float = 0.0
        vDSP_dotpr(a, 1, b, 1, &dotProduct, vDSP_Length(a.count))

        var normA: Float = 0.0
        vDSP_svesq(a, 1, &normA, vDSP_Length(a.count))
        normA = sqrt(normA)

        var normB: Float = 0.0
        vDSP_svesq(b, 1, &normB, vDSP_Length(b.count))
        normB = sqrt(normB)

        if normA == 0.0 || normB == 0.0 {
            return Float.pi / 2.0  // 90 degrees for orthogonal
        }

        let cosineSimilarity = dotProduct / (normA * normB)
        // Clamp to [-1, 1] to avoid numerical errors
        let clampedSimilarity = max(-1.0, min(1.0, cosineSimilarity))
        return acos(clampedSimilarity) / Float.pi
    }

    private func correlationDistance(_ a: [Float], _ b: [Float]) -> Float {
        // Pearson correlation distance = 1 - correlation
        // Good for pattern matching regardless of scale
        let n = Float(a.count)

        var meanA: Float = 0.0
        vDSP_sve(a, 1, &meanA, vDSP_Length(a.count))
        meanA /= n

        var meanB: Float = 0.0
        vDSP_sve(b, 1, &meanB, vDSP_Length(b.count))
        meanB /= n

        var centeredA = [Float](repeating: 0, count: a.count)
        var negMeanA = -meanA
        vDSP_vsadd(a, 1, &negMeanA, &centeredA, 1, vDSP_Length(a.count))

        var centeredB = [Float](repeating: 0, count: b.count)
        var negMeanB = -meanB
        vDSP_vsadd(b, 1, &negMeanB, &centeredB, 1, vDSP_Length(b.count))

        var numerator: Float = 0.0
        vDSP_dotpr(centeredA, 1, centeredB, 1, &numerator, vDSP_Length(a.count))

        var stdA: Float = 0.0
        vDSP_svesq(centeredA, 1, &stdA, vDSP_Length(a.count))
        stdA = sqrt(stdA)

        var stdB: Float = 0.0
        vDSP_svesq(centeredB, 1, &stdB, vDSP_Length(b.count))
        stdB = sqrt(stdB)

        if stdA == 0.0 || stdB == 0.0 {
            return 1.0
        }

        let correlation = numerator / (stdA * stdB)
        return 1.0 - correlation
    }
}
