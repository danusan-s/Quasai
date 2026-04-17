# Quasai

A machine learning framework implemented in C++ with support for neural networks, traditional ML algorithms, and linear algebra operations.

## Structure

```
include/quasai/
├── core/           # Core: Tensor, Array, Types, Error
├── linalg/         # Linear algebra: Matrix, Vector, Ops
├── nn/            # Neural networks: Layers, Activation, Loss, Optimizer, Model
├── ml/            # Traditional ML: SVM, Decision Tree, K-Means, Regression
└── utils/         # Utilities (if needed)

src/
├── core/           # Core implementations
├── linalg/         # Linear algebra implementations
├── nn/            # Neural network implementations
├── ml/            # ML algorithm implementations
└── tests/         # Test files

tests/             # Additional test files
examples/          # Example programs
bindings/          # Language bindings (future)
tools/             # Build tools
```

## Getting Started

1. **Implement Array** - see `include/quasai/core/array.h`
2. **Implement Tensor** - see `include/quasai/core/tensor.h`  
3. **Implement Linear Algebra** - Matrix and Vector classes
4. **Implement Neural Networks** - Layers, activations, loss, optimizer
5. **Implement ML Algorithms** - SVM, Decision Tree, K-Means, Regression

See `QUICKSTART.md` for detailed implementation guide.

## Build

```bash
bazel build //:quasai
bazel test //tests/...
```

## License

MIT - See LICENSE file
