// swift-tools-version: 5.10
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "sam2",
    platforms: [
        .iOS(.v14),
        .macOS(.v14)
    ],
    products: [
        // Products define the executables and libraries a package produces, making them visible to other packages.
        .library(
            name: "sam2",
            targets: ["sam2"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.1.0")
    ],
    targets: [
        // Targets are the basic building blocks of a package, defining a module or a test suite.
        // Targets can depend on other targets in this package and products from dependencies.
        .target(
            name: "sam2",
            dependencies: [.product(name: "MLX", package: "mlx-swift")]),
        .testTarget(
            name: "sam2Tests",
            dependencies: ["sam2"]),
    ]
)