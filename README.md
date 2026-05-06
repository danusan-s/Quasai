# Quasai

[![CI](https://github.com/danusan-s/Quasai/actions/workflows/ci.yml/badge.svg)](https://github.com/danusan-s/Quasai/actions/workflows/ci.yml)

A from-scratch C++20 machine learning framework built with minimal external dependencies to demonstrate automatic differentiation, tensor operations, neural network systems and data processing utilities.

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

| Op         | Size      | Quasai (ns) | PyTorch (ns) | Speedup (PT / Q) |
| ---------- | --------- | ----------- | ------------ | ---------------- |
| Add        | 2048      | 1,459,870   | 1,237,418    | 0.85×            |
| Add        | 4096      | 7,479,059   | 10,355,398   | 1.38×            |
| Matmul     | 2048      | 26,882,458  | 24,090,533   | 0.90×            |
| Transpose  | 2048      | 59.6        | 503          | 8.44×            |
| Transpose  | 4096      | 59.6        | 501          | 8.41×            |
| Sum        | 2048      | 47,794      | 57,489       | 1.20×            |
| Sum        | 4096      | 1,299,329   | 1,216,361    | 0.94×            |
| Scalar Add | 5,000,000 | 7,439,729   | 1,043,395    | 0.14×            |
| ReLU       | 2048      | 1,613,666   | 752,993      | 0.47×            |
| ReLU       | 4096      | 6,943,953   | 7,809,385    | 1.12×            |

Note: The speedup column <1 means PyTorch is faster, >1 means Quasai is faster.
Some timings may vary due to system load and other factors, but the general trends hold.
The speedups in transpose and ReLU could simply be because Quasai is less bloated and therefore less overhead for now.

---

## Example Usage

```cpp
#include "quasai/nn/layers/linear.hpp"
#include "quasai/nn/layers/activations.hpp"
#include "quasai/nn/sequential.hpp"
#include "quasai/nn/model.hpp"
#include "quasai/optim/sgd.hpp"
#include <iostream>

int main() {
    // Create a simple MLP
    auto linear1 = std::make_shared<quasai::nn::Linear>(64, 32);
    auto relu = std::make_shared<quasai::nn::ReLU>();
    auto linear2 = std::make_shared<quasai::nn::Linear>(32, 10);

    auto model = std::make_shared<quasai::nn::Sequential>(
        std::vector<std::shared_ptr<quasai::nn::Module>>{linear1, relu, linear2});

    quasai::nn::Model ml_model(model);

    float learning_rate = 0.01f;
    float momentum = 0.9f;
    auto optimizer = std::make_shared<quasai::optim::SGD>(learning_rate, momentum);

    ml_model.compile(quasai::nn::Loss::MSE, optimizer);

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

**Disclaimer:** This is **not** a production-ready framework and is intended as a skills demonstration. The API is subject to change as the project evolves. Feedback is welcome.

---

## License

MIT License - see LICENSE file for details.

---
