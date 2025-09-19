import Accelerate
import Foundation

class DBSCAN {
    private let eps: Float
    private let minSamples: Int
    private let metric: DistanceMetric

    private(set) var labels: [Int] = []
    private var data: Matrix?

    private let NOISE = -1
    private let UNCLASSIFIED = -2

    init(eps: Float = 0.5, minSamples: Int = 5, metric: DistanceMetric = .euclidean) {
        self.eps = eps
        self.minSamples = minSamples
        self.metric = metric
    }

    // Convenience initializer for string metric
    convenience init(eps: Float = 0.5, minSamples: Int = 5, distanceMetric: DistanceMetric = .euclidean) {
        self.init(eps: eps, minSamples: minSamples, metric: distanceMetric)
    }

    @discardableResult
    func fit(_ X: [[Float]]) -> DBSCAN {
        let matrix = Matrix(array2D: X)
        return fit(matrix)
    }

    @discardableResult
    func fit(_ X: Matrix) -> DBSCAN {
        self.data = X
        self.labels = Array(repeating: UNCLASSIFIED, count: X.rows)

        var clusterId = 0

        for pointIdx in 0..<X.rows {
            // Skip if already processed
            if labels[pointIdx] != UNCLASSIFIED {
                continue
            }

            // Find neighbors
            let neighbors = regionQuery(pointIdx)

            if neighbors.count < minSamples {
                // Mark as noise
                labels[pointIdx] = NOISE
            } else {
                // Start a new cluster
                expandCluster(pointIdx: pointIdx, neighbors: neighbors, clusterId: clusterId)
                clusterId += 1
            }
        }

        return self
    }

    private func regionQuery(_ pointIdx: Int) -> [Int] {
        guard let data = data else { return [] }

        var neighbors: [Int] = []
        let point = data.row(pointIdx)

        for i in 0..<data.rows {
            let otherPoint = data.row(i)
            let distance = metric.distance(point, otherPoint)

            if distance <= eps {
                neighbors.append(i)
            }
        }

        return neighbors
    }

    private func expandCluster(pointIdx: Int, neighbors: [Int], clusterId: Int) {
        labels[pointIdx] = clusterId

        var seeds = Set(neighbors)
        seeds.remove(pointIdx)

        while !seeds.isEmpty {
            let currentPoint = seeds.removeFirst()

            // Change noise to border point
            if labels[currentPoint] == NOISE {
                labels[currentPoint] = clusterId
            }

            // Skip if already processed
            if labels[currentPoint] != UNCLASSIFIED {
                continue
            }

            // Add to cluster
            labels[currentPoint] = clusterId

            // Find neighbors of current point
            let currentNeighbors = regionQuery(currentPoint)

            // If current point is a core point, add its neighbors to seeds
            if currentNeighbors.count >= minSamples {
                for neighbor in currentNeighbors {
                    if labels[neighbor] == UNCLASSIFIED {
                        seeds.insert(neighbor)
                    }
                }
            }
        }
    }

    // Properties for sklearn compatibility
    var labels_: [Int] {
        return labels
    }

    var nClusters: Int {
        let uniqueLabels = Set(labels.filter { $0 >= 0 })
        return uniqueLabels.count
    }

    var nNoisePoints: Int {
        return labels.filter { $0 == NOISE }.count
    }
}

// MARK: - NumPy-like Array Extension
extension Array where Element == [Float] {
    // Convert 2D array to flat array (similar to numpy flatten)
    var flattened: [Float] {
        return self.flatMap { $0 }
    }

    // Get shape of array
    var shape: (Int, Int) {
        return (self.count, self.first?.count ?? 0)
    }
}

// MARK: - Generic TypeAlias for Model Outputs
typealias Embedding = [Float]
typealias EmbeddingMatrix = [[Float]]

// MARK: - DBSCAN Extension for Model Integration
extension DBSCAN {
    // Direct integration with model outputs
    func fitModelOutput(_ embeddings: EmbeddingMatrix) -> DBSCAN {
        return fit(embeddings)
    }

    // For MLMultiArray or similar model outputs
    func fitFlatArray(_ flatEmbeddings: [Float], dimensions: (rows: Int, cols: Int)) -> DBSCAN {
        let matrix = Matrix(data: flatEmbeddings, rows: dimensions.rows, cols: dimensions.cols)
        return fit(matrix)
    }

    // Adaptive eps based on data distribution (research-based)
    static func suggestEps(for data: [[Float]], metric: DistanceMetric, k: Int = 5) -> Float {
        var kDistances: [Float] = []

        for i in 0..<data.count {
            // Örnek: distances array'ini hesapladığını varsayalım
            let distances = data[i].map { /* metric hesapla */ $0 }  // Burada gerçek mesafe hesaplama yapılmalı
            if distances.count >= k {
                kDistances.append(distances[k - 1])
            }
        }

        kDistances.sort()

        let percentileIndex = Int(Float(kDistances.count) * 0.85)
        var eps = kDistances[percentileIndex]

        // Angular metric için eps'i 0.25 - 0.40 arasında sıkıştır
        if metric == .angular {
            // Normalize et ve aralık içine sıkıştır
            let minRange: Float = 0.3252
            let maxRange: Float = 0.6

            // Basit normalize: eps / max eps değerine göre 0-1 arası
            let normalized = min(max(eps, 0.0), 1.0)
            // Aralığa sıkıştır ve ufak rastgele dalgalanma ekle
            let randomJitter = Float.random(in: -0.03...0.03)  // ±0.03 oynama
            eps = min(max(minRange + (maxRange - minRange) * normalized + randomJitter, minRange), maxRange)
        }

        return eps
    }

    // Automatic parameter tuning for better results
    static func autoTune(data: [[Float]], metric: DistanceMetric) -> (eps: Float, minSamples: Int) {

        // Research shows minSamples = 2 * dimensions works well for most cases
        let dimensions = data.first?.count ?? 1
        let minSamples = min(max(3, dimensions * 2), 10)  // Cap at 10 for practical reasons

        // Calculate adaptive eps
        let eps = suggestEps(for: data, metric: metric, k: minSamples)

        return (eps: eps, minSamples: minSamples)
    }
}
