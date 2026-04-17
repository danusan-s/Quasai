# Project Structure - Quasai ML Framework

## Overview

This framework provides a comprehensive machine learning implementation in C++ with:
- **Neural network components** ( layers, activations, optimizers, models)
- **Traditional ML algorithms** (SVM, decision trees, k-means, regression)
- **Linear algebra** (Matrix, Vector operations)
- **Core utilities** (Tensor, Array, error handling)

## Build System

- **Bazel** - for building, testing, and examples

## Directory Structure

### Include Headers (`include/quasai/`)

**core/**
- `types.h` - Fundamental type definitions
- `error.h` - Custom exception classes
- `array.h` - Dynamic array with RAII
- `tensor.h` - N-dimensional tensor

**linalg/**
- `matrix.h` - 2D matrix operations
- `vector.h` - 1D vector operations
- `ops.h` - Mathematical operations

**nn/**
- `layer.h` - Base layer interface
- `activation.h` - Activation functions
- `loss.h` - Loss/cost functions
- `optimizer.h` - Optimization algorithms
- `model.h` - Model interface

**ml/**
- `svm.h` - Support Vector Machines
- `decision_tree.h` - Decision Trees
- `kmeans.h` - K-Means clustering
- `regression.h` - Regression models

### Source Files (`src/`)

Contains `.cpp` implementation files matching the include structure:
- `src/core/`
- `src/linalg/`
- `src/nn/`
- `src/ml/`
- `src/tests/`
- `src/examples/`

### Tests (`tests/`)

Unit tests using Google Test framework

### Examples (`examples/cpp/`)

Demonstration programs showing framework usage

## Getting Started Tasks

### Phase 1: Core Implementation
1. **Array** (`include/quasai/core/array.h`): Implement dynamic array
   - Memory management (reserve, resize)
   - Iterator support
   - Move semantics
   - RAII cleanup

2. **Tensor** (`include/quasai/core/tensor.h`): Implement N-dimensional tensor
   - Shape storage and indexing
   - Multi-dimensional to 1D conversion
   - Memory layout management

### Phase 2: Linear Algebra
3. **Matrix** (`include/quasai/linalg/matrix.h`): Implement 2D matrix
4. **Vector** (`include/quasai/linalg/vector.h`): Implement 1D vector
5. **Ops** (`include/quasai/linalg/ops.h`): Implement operations

### Phase 3: Neural Networks
6. Implement Layer base class and subclasses
7. Implement activation functions
8. Implement loss functions
9. Implement optimizers
10. Implement Model interface

### Phase 4: Traditional ML
11. Implement SVM
12. Implement Decision Tree
13. Implement K-Means
14. Implement Regression models

## Implementation Approach

### Recommended Order

1. **Start with Array** - it's the foundation for everything else
2. **Build Tensor** on top of Array
3. **Implement basic operations** - simple tests can validate each step
4. **Add complexity gradually** - test after each major feature

### Testing Strategy

- Write unit tests as you implement each class
- Test edge cases: empty arrays, reshape operations, out-of-bounds
- Verify memory management (no leaks, proper copy/move)

## C++ Features Used

- C++20 support
- Template metaprogramming
- RAII for memory management
- Move semantics
- Exception handling

## Performance Optimizations

Consider these when implementing:
- Cache-friendly memory access (row-major ordering)
- SIMD vectorization (AVX2/AVX-512)
- Multi-threading for parallel operations
- Memory pooling to reduce allocations
- Lazy evaluation for complex operations

## Notes

- All implementations should be placed in corresponding `.cpp` files in `src/`
- Headers should only contain declarations (with comments explaining what to implement)
- License is MIT - see LICENSE file
- Build with Bazel (configure WORKSPACE and BUILD files as needed)
