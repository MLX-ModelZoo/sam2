import MLX
import MLXNN
import MLXRandom

class PositionEmbeddingSine: MLXNN.Module {
    let numPosFeats: Int
    let temperature: Float
    let normalize: Bool
    let scale: Float
    var cache: [String: MLXArray] = [:]

    init(
        numPosFeats: Int,
        temperature: Float = 10000,
        normalize: Bool = true,
        scale: Float? = nil
    ) {
        precondition(numPosFeats % 2 == 0, "Expecting even model width")
        self.numPosFeats = numPosFeats / 2
        self.temperature = temperature
        self.normalize = normalize
        if let scale = scale, !normalize {
            preconditionFailure("normalize should be True if scale is passed")
        }
        self.scale = scale ?? (2 * Float.pi)
        super.init()
    }

    func encodeXY(_ x: MLXArray, _ y: MLXArray) -> (MLXArray, MLXArray) {
        precondition(x.shape == y.shape && x.ndim == 1 && y.ndim == 1)
        let xEmbed = x * scale
        let yEmbed = y * scale

        let dimT = MLXArray.arange(Float(numPosFeats))
        let dimT = MLXArray.pow(temperature, 2 * (dimT / 2) / Float(numPosFeats))

        let posX = xEmbed.expandedDimensions(axis: 1) / dimT
        let posY = yEmbed.expandedDimensions(axis: 1) / dimT
        
        let posXSin = MLXArray.sin(posX[..., 0...(.strided(2))])
        let posXCos = MLXArray.cos(posX[..., 1...(.strided(2))])
        let posYSin = MLXArray.sin(posY[.., 0...(.strided(2))])
        let posYCos = MLXArray.cos(posY[.., 1...(.strided(2))])
        
        return (MLXArray.concatenate([posXSin, posXCos], axis: 1),
                MLXArray.concatenate([posYSin, posYCos], axis: 1))
    }

    func encodeBoxes(_ x: MLXArray, _ y: MLXArray, _ w: MLXArray, _ h: MLXArray) -> MLXArray {
        let (posX, posY) = encodeXY(x, y)
        return MLXArray.concatenate([posY, posX, h.expandedDimensions(axis: 1), w.expandedDimensions(axis: 1)], axis: 1)
    }

    func encodePoints(_ x: MLXArray, _ y: MLXArray, _ labels: MLXArray) -> MLXArray {
        let (bx, nx) = (x.shape[0], x.shape[1])
        let (by, ny) = (y.shape[0], y.shape[1])
        let (bl, nl) = (labels.shape[0], labels.shape[1])
        precondition(bx == by && nx == ny && bx == bl && nx == nl)
        
        let (posX, posY) = encodeXY(x.flattened(), y.flattened())
        let posXReshaped = posX.reshaped([bx, nx, -1])
        let posYReshaped = posY.reshaped([by, ny, -1])
        
        return MLXArray.concatenate([posYReshaped, posXReshaped, labels.expandedDimensions(axis: 2)], axis: 2)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let cacheKey = "\(x.shape[-2]),\(x.shape[-1])"
        if let cached = cache[cacheKey] {
            return cached.expandedDimensions(axis: 0).repeated(x.shape[0], axis: 0)
        }

        let yEmbed = MLXArray.arange(1, x.shape[-2] + 1, dtype: .float32)
            .reshaped([1, -1, 1])
            .repeated(x.shape[-1], axis: 2)
        
        let xEmbed = MLXArray.arange(1, x.shape[-1] + 1, dtype: .float32)
            .reshaped([1, 1, -1])
            .repeated(x.shape[-2], axis: 1)

        var yEmbedNorm = yEmbed
        var xEmbedNorm = xEmbed

        if normalize {
            let eps: Float = 1e-6
            yEmbedNorm = yEmbed / (yEmbed[.., -1..., ..] + eps) * scale
            xEmbedNorm = xEmbed / (xEmbed[.., .., -1...] + eps) * scale
        }

        let dimT = MLXArray.arange(Float(numPosFeats))
        let dimT = MLXArray.pow(temperature, 2 * (dimT / 2) / Float(numPosFeats))

        let posX = xEmbedNorm.expandedDimensions(axis: 3) / dimT
        let posY = yEmbedNorm.expandedDimensions(axis: 3) / dimT

        let posXSinCos = MLXArray.concatenate([
            MLXArray.sin(posX[.., .., .., 0...(.strided(2))]),
            MLXArray.cos(posX[.., .., .., 1...(.strided(2))])
        ], axis: 3).reshaped(posX.shape[0..<3] + [-1])

        let posYSinCos = MLXArray.concatenate([
            MLXArray.sin(posY[.., .., .., 0...(.strided(2))]),
            MLXArray.cos(posY[.., .., .., 1...(.strided(2))])
        ], axis: 3).reshaped(posY.shape[0..<3] + [-1])

        let pos = MLXArray.concatenate([posYSinCos, posXSinCos], axis: 3).transposed(0, 3, 1, 2)
        cache[cacheKey] = pos[0]
        return pos
    }
}

class PositionEmbeddingRandom: MLXNN.Module {
    var positionalEncodingGaussianMatrix: MLXArray

    init(numPosFeats: Int = 64, scale: Float? = nil) {
        let scale = scale.map { $0 > 0 ? $0 : 1.0 } ?? 1.0
        self.positionalEncodingGaussianMatrix = scale * MLXRandom.normal([2, numPosFeats])
        super.init()
    }

    func peEncoding(_ coords: MLXArray) -> MLXArray {
        let coords = 2 * coords - 1
        let encodedCoords = coords.matmul(positionalEncodingGaussianMatrix)
        let encodedCoords = 2 * Float.pi * encodedCoords
        return MLXArray.concatenate([MLXArray.sin(encodedCoords), MLXArray.cos(encodedCoords)], axis: -1)
    }

    public func callAsFunction(_ size: (Int, Int)) -> MLXArray {
        let (h, w) = size
        let grid = MLXArray.ones([h, w])
        let yEmbed = grid.cumulativeSum(axis: 0) - 0.5
        let xEmbed = grid.cumulativeSum(axis: 1) - 0.5
        let yEmbedNorm = yEmbed / Float(h)
        let xEmbedNorm = xEmbed / Float(w)

        let pe = peEncoding(MLXArray.stack([xEmbedNorm, yEmbedNorm], axis: -1))
        return pe.transposed(2, 0, 1)
    }

    func forwardWithCoords(_ coordsInput: MLXArray, imageSize: (Int, Int)) -> MLXArray {
        var coords = coordsInput
        coords[.., .., 0] = coords[.., .., 0] / Float(imageSize.1)
        coords[.., .., 1] = coords[.., .., 1] / Float(imageSize.0)
        return peEncoding(coords.asType(.float32))
    }
}
