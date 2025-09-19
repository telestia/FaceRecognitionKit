//
//  FaceCluster.swift
//  Swix
//
//  Created by Harun SUBAŞI on 17.09.2025.
//

import Foundation

class FaceCluster: Identifiable {
    let id: Int
    private var _faces: [FaceCrop] = []
    private var _thumbnailFace: FaceCrop!

    var faces: [FaceCrop] {
        return _faces
    }

    var thumbnailFace: FaceCrop {
        return _thumbnailFace
    }

    init(id: Int, faces: [FaceCrop] = []) {
        self.id = id
        for face in faces {
            add(face: face)
        }
    }

    func add(face: FaceCrop) {
        _faces.append(face)
        updateRankingsAndThumbnail()
    }

    private func updateRankingsAndThumbnail() {
        // Sort faces by quality score (descending)
        let sortedFaces = _faces.sorted { $0.qualityScore > $1.qualityScore }

        // Update rankings
        for (index, face) in sortedFaces.enumerated() {
            if let faceIndex = _faces.firstIndex(where: { $0.id == face.id }) {
                _faces[faceIndex].clusterRank = index
            }
        }

        // Set best face as thumbnail
        _thumbnailFace = sortedFaces.first

    }

    func getAverageEmbedding() -> [Float] {
        guard !_faces.isEmpty else { return [] }

        let embeddingSize = _faces[0].embedding.count
        var averageEmbedding = Array(repeating: Float(0), count: embeddingSize)

        for face in _faces {
            for i in 0..<embeddingSize {
                averageEmbedding[i] += face.embedding[i]
            }
        }

        let faceCount = Float(_faces.count)
        for i in 0..<embeddingSize {
            averageEmbedding[i] /= faceCount
        }

        return averageEmbedding
    }

    func getClusterQuality() -> Float {
        guard !_faces.isEmpty else { return 0.0 }
        return _faces.map { $0.qualityScore }.reduce(0, +) / Float(_faces.count)
    }
}
