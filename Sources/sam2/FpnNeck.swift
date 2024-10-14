import MLX
import MLXNN
import MLXRandom

class FpnNeck: MLXNN.Module {
    let positionEncoding: MLXNN.Module
    let convs: [MLXNN.Conv2d]
    let backboneChannelList: [Int]
    let fpnInterpModel: String
    let fuseType: String
    let fpnTopDownLevels: [Int]
    let dModel: Int

    init(positionEncoding: MLXNN.Module,
         dModel: Int,
         backboneChannelList: [Int],
         kernelSize: Int = 1,
         stride: Int = 1,
         padding: Int = 0,
         fpnInterpModel: String = "bilinear",
         fuseType: String = "sum",
         fpnTopDownLevels: [Int]? = nil) {
        
        self.positionEncoding = positionEncoding
        self.dModel = dModel
        self.backboneChannelList = backboneChannelList
        self.fpnInterpModel = fpnInterpModel
        self.fuseType = fuseType
        
        // Initialize convs
        self.convs = backboneChannelList.map { dim in
            MLXNN.Conv2d(inputChannels: dim, outputChannels: dModel, kernelSize: [kernelSize, kernelSize], stride: [stride, stride], padding: [padding, padding])
        }
        
        // Set fpnTopDownLevels
        if let levels = fpnTopDownLevels {
            self.fpnTopDownLevels = levels
        } else {
            self.fpnTopDownLevels = Array(0..<self.convs.count)
        }
        
        super.init()
    }
    
    public func callAsFunction(_ xs: [MLXArray]) -> ([MLXArray], [MLXArray]) {
        var out = [MLXArray](repeating: MLXArray(), count: convs.count)
        var pos = [MLXArray](repeating: MLXArray(), count: convs.count)
        
        assert(xs.count == convs.count, "Input count must match convs count")
        
        var prevFeatures: MLXArray?
        
        for i in stride(from: convs.count - 1, through: 0, by: -1) {
            let x = xs[i]
            let lateralFeatures = convs[convs.count - 1 - i](x)
            
            if fpnTopDownLevels.contains(i) && prevFeatures != nil {
//                let topDownFeatures = MLXNN.interpolate(prevFeatures!.asType(.float32),
//                                                        scale: [2.0, 2.0],
//                                                        mode: fpnInterpModel)
                let topDownFeatures = prevFeatures!
                
                prevFeatures = lateralFeatures + topDownFeatures
                if fuseType == "avg" {
                    prevFeatures = prevFeatures! / MLXArray(2)
                }
            } else {
                prevFeatures = lateralFeatures
            }
            
            let xOut = prevFeatures!
            out[i] = xOut
//            pos[i] = positionEncoding.callAsFunction(xOut).asType(xOut.dtype)
        }
        
        return (out, pos)
    }
}
