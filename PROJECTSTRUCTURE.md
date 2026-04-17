# Quasai - Machine Learning Framework in C++

A machine learning framework implemented in C++ with support for neural networks, traditional ML algorithms, and linear algebra operations.

## Project Structure

```
Quasai/
├── include/quasai/     # Public API headers
│   ├── core/           # Core components (Tensor, Array, Types)
│   ├── linalg/         # Linear algebra operations
│   ├── nn/            # Neural network components
│   ├── ml/            # Traditional ML algorithms
│   └── utils/         # Utilities
├── src/               # Implementation files
│   ├── core/
│   ├── linalg/
│   ├── nn/
│   ├── ml/
│   └── utils/
├── tests/             # Unit tests (Google Test)
├── examples/          # Usage examples
│   ├── cpp/          # C++ examples
│   └── python/       # Python bindings examples
├── bindings/          # Language bindings (pybind11)
├── tools/            # Build tools and scripts
├── BUILD.bazel       # Bazel build configuration
├── WORKSPACE         # Bazel workspace
└── README.md         # This file
```

## Build System

This project uses **Bazel** as its build system.

### Building

```bash
bazel build //:quasai
```

### Testing

```bash
bazel test //tests/...
```

### Examples

```bash
bazel run //examples/cpp:example_name
```

## Components

### Core
- **Tensor**: N-dimensional tensor with memory management
- **Array**: Dynamic array with RAII semantics
- **Types**: Type definitions and constants
- **Error**: Custom exception handling

### Linear Algebra
- **Matrix**: 2D matrix operations
- **Vector**: 1D vector operations
- **Ops**: Mathematical operations and utilities

### Neural Networks
- **Layer**: Base class for neural network layers
- **Activation**: Activation functions (ReLU, Sigmoid, etc.)
- **Loss**: Loss functions (MSE, Cross-Entropy, etc.)
- **Optimizer**: Gradient-based optimizers (SGD, Adam, etc.)
- **Model**: High-level model interface

### Machine Learning
- **SVM**: Support Vector Machines
- **Decision Tree**: CART decision trees
- **K-Means**: Clustering algorithm
- **Regression**: Linear, Ridge, Lasso regression

## C++ Standard

This project uses **C++20** features including:
- Concepts and constraints
- Modules (when ready)
- Coroutines (for async operations)
- Ranges library
- 20-bit static_assert improvements

## Getting Started

### Step 1: Implement Array Class (`include/quasai/core/array.h`)

Implement a dynamic array with:
- Dynamic capacity growth
- RAII memory management
- Iterator support
- Move semantics

### Step 2: Implement Tensor Class (`include/quasai/core/tensor.h`)

Implement N-dimensional tensor with:
- Shape storage
- Multi-dimensional indexing
- Memory layout management
- Row-major ordering

### Step 3: Implement Linear Algebra

Implement Matrix and Vector classes with:
- Basic arithmetic operations
- Matrix multiplication
- Vector operations (dot product, norm)

### Step 4: Implement Neural Network Components

Implement layers, activations, loss functions, and optimizers following the base class interfaces.

### Step 5: Implement Traditional ML Algorithms

Implement classification, regression, and clustering algorithms.

## Performance Considerations

- Cache-friendly memory access patterns
- SIMD vectorization support
- Multi-threading (OpenMP/TBB)
- Zero-copy operations where possible
- Lazy evaluation where beneficial

## Future Work

- [ ] CUDA GPU backend
- [ ] Automatic differentiation
- [ ] Model serialization
- [ ] Python bindings
- [ ] Distributed training support

## License

MIT License - see LICENSE file for details.
