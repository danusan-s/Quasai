# Quasai

[![CI](https://github.com/danusan-s/Quasai/actions/workflows/ci.yml/badge.svg)](https://github.com/danusan-s/Quasai/actions/workflows/ci.yml)

A from-scratch C++20 machine learning framework built with minimal external dependencies to demonstrate automatic differentiation, tensor operations, neural network systems and data processing utilities.
This framework is on par with PyTorch in terms of tensor operations performance. The project will get more features to make it a fully fledged ML framework in the future.

## Features

### Implemented Features

#### Core
- **Tensor** — Multi-dimensional array with shape, strides, dtype (INT32, INT64, FLOAT32, FLOAT64), and device (CPU) support
- **Storage System** — Custom allocator with reference-counted buffer management
- **Shape & Strides** — Full broadcasting support for element-wise operations

#### Autograd
- **Automatic Differentiation** — Computes gradients via dynamic computation graph
- **Backward Pass** — Supports add, sub, mul, div, matmul, and all unary/reduction operations

#### Operations
- **Binary Operations** — add, sub, mul, div, matmul (with OpenMP parallelization)
- **Unary Operations** — neg, abs, relu, heaviside, signum, sigmoid, tanh
- **Reduction Operations** — sum, mean, sum_to_shape, broadcast_to_shape
- **View Operations** — transpose, expand, reshape, make_contiguous, slice
- **Scalar Broadcasting** — All binary ops support scalar-tensor operations

#### Neural Networks
- **Layers** — Linear (fully-connected)
- **Initializers** — Glorot uniform and normal, He uniform and normal, Xavier uniform and normal
- **Activations** — ReLU, Sigmoid, Tanh, Heaviside
- **Containers** — Sequential model composition
- **Loss Functions** — MSE loss, L1 loss
- **Model Class** — High-level API for training and inference

#### Optimizers
- **SGD** — Stochastic gradient descent with momentum

#### Data Processing
- **CSV Parser** — Load tabular data from CSV files
- **StandardScaler** — Feature normalization with fit/transform/inverse_transform

#### Utilities
- **Random** — Seed control, tensor initialization (randn, rand)
- **Logger** — Logging utilities with timestamps

### Planned Features
- CUDA support
- More layers (Conv2D, LSTM, etc.)
- More optimizers (Adam, RMSProp)
- Serialization/persistence
- More comprehensive benchmarks
- API documentation

---

## Technical Decisions & Tradeoffs

This project prioritizes demonstrating systems ML concepts over production readiness. Key tradeoffs made:

- **CSV Parser**: Functional but acknowledged as "hacked together" (`include/quasai/data/csv_parser.hpp`). A rewrite is planned for Phase 3.
- **Gradient Function Allocation**: Uses raw `new` in `src/quasai/ops/binary_ops.cpp`. Ownership is transferred via `std::unique_ptr` in `set_grad_fn()`, but a factory pattern is planned.
- **Logger Path Trimming**: Contains a brittle path-trimming hack in `include/quasai/utils/logger.hpp` that depends on repo folder name.

---

## Build Instructions

### Prerequisites
- CMake >= 3.16
- C++20 compatible compiler (GCC, Clang)
- OpenMP (optional, for parallel operations on large tensors)

```bash
# Clone the repository
git clone https://github.com/danusan-s/Quasai.git
cd Quasai

# Build the library
make build

# Run tests
make run_tests

# Open documentation
make open_docs
```

See `examples/cpp/` for C++ examples 

---

## Benchmarks

Quasai includes benchmarks for large tensor operations that trigger OpenMP parallelization (tested with up to 64M element tensors):

```bash
# Build with benchmarks and run them
make run_benchmarks
```

**OpenMP Speedup**: Benchmarks show ~5x speedup on 16-core systems for large tensor operations (8192x8192 float32 addition: 125ms vs 624ms single-threaded).

| Operation           | Quasai (ms) | PyTorch (ms) | Speedup (Q/P) |
| ------------------- | ----------: | -----------: | ------------: |
| Add 2048×2048       |       1.299 |        1.266 |         1.03× |
| Add 4096×4096       |       7.076 |       10.196 |         1.44× |
| Matmul 2048×2048    |      24.267 |       21.744 |         0.90× |
| Transpose 2048×2048 |    0.000041 |     0.000354 |         8.70× |
| Transpose 4096×4096 |    0.000040 |     0.000348 |         8.75× |
| Sum 2048×2048       |      0.0450 |       0.0577 |         1.28× |
| Sum 4096×4096       |       1.304 |        1.300 |         1.00× |
| ScalarAdd 5,000,000 |       1.202 |        1.134 |         0.94× |
| ReLU 2048×2048      |       0.893 |        0.785 |         0.88× |
| ReLU 4096×4096      |       5.169 |        7.922 |         1.53× |

Note: The speedup column <1 means PyTorch is faster, >1 means Quasai is faster.
Some timings may vary due to system load and other factors, but the general trends hold.
The speedups in transpose could simply be because Quasai is less bloated and therefore less overhead for now.

### Does this mean Quasai is better than PyTorch?

Not necessarily. PyTorch is a mature, highly optimized framework with CUDA support, a large ecosystem, and many features. Quasai is a demonstration of core ML concepts in C++.
Another tradeoff is that Quasai relies on compiler optimizations for native architectures so it has to be built from source.  This means it is not really portable as compared to pytorch's hand optimized kernels for different platforms which are dispatched at runtime.

---

## Example Usage

```cpp
#include "quasai/nn/model.hpp"
#include "quasai/nn/modules/activations.hpp"
#include "quasai/nn/modules/linear.hpp"
#include "quasai/nn/sequential_builder.hpp"
#include "quasai/optim/sgd.hpp"
#include <iostream>

int main() {
  // Create a simple MLP using SequentialBuilder
  auto network = quasai::nn::SequentialBuilder()
                     .add<quasai::nn::Linear>(64, 32)
                     .add<quasai::nn::ReLU>()
                     .add<quasai::nn::Linear>(32, 10)
                     .build_ptr();

  quasai::nn::Model ml_model(std::move(network));
  ml_model.set_loss(quasai::nn::Loss::MSE);
  ml_model.set_optimizer<quasai::optim::SGD>(0.01f, 0.9f);

  // Training data (batch_size, 64)
  quasai::core::Tensor X = quasai::core::Tensor::zeros({32, 64});
  quasai::core::Tensor y = quasai::core::Tensor::zeros({32, 10});

  std::cout << "Training simple MLP..." << std::endl;
  ml_model.train(X, y, 10, 32);

  std::cout << "Training complete!" << std::endl;

  return 0;
}
```

---

**Disclaimer:** This is **not** a production-ready framework and is intended as a skills demonstration. The API is subject to change as the project evolves. Feedback is welcome. Changes made in other branches are experimental and may be broken. Please refer to the main branch for the most stable version.

---

## License

MIT License - see LICENSE file for details.

---
