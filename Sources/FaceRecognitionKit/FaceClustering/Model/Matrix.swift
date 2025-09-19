//
//  Matrix.swift
//  Swix
//
//  Created by Harun SUBAŞI on 17.09.2025.
//

import Foundation

struct Matrix {
    let data: [Float]
    let rows: Int
    let cols: Int

    init(data: [Float], rows: Int, cols: Int) {
        self.data = data
        self.rows = rows
        self.cols = cols
    }

    init(array2D: [[Float]]) {
        self.rows = array2D.count
        self.cols = array2D.first?.count ?? 0
        self.data = array2D.flatMap { $0 }
    }

    subscript(row: Int, col: Int) -> Float {
        get {
            return data[row * cols + col]
        }
    }

    func row(_ index: Int) -> [Float] {
        let start = index * cols
        return Array(data[start..<start + cols])
    }
}
