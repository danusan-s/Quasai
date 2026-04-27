# Quasai

A C++ machine learning framework with automatic differentiation, neural network layers, optimizers, and data processing utilities.

## Features

### Core
- **Tensor** — Multi-dimensional array with shape, strides, dtype (INT32, INT64, FLOAT32, FLOAT64), and device (CPU) support
- **Storage System** — Custom allocator with reference-counted buffer management
- **Shape & Strides** — Full broadcasting support for element-wise operations

### Autograd
- **Automatic Differentiation** — Computes gradients via dynamic computation graph
- **Backward Pass** — Supports add, sub, mul, div, matmul, and all unary/reduction operations

### Operations
- **Binary Operations** — add, sub, mul, div, matmul (with OpenMP parallelization)
- **Unary Operations** — neg, abs, relu, heaviside, signum, sigmoid, tanh
- **Reduction Operations** — sum, mean, sum_to_shape, broadcast_to_shape
- **View Operations** — transpose, expand, reshape, make_contiguous, slice
- **Scalar Broadcasting** — All binary ops support scalar-tensor operations

### Neural Networks
- **Layers** — Linear (fully-connected)
- **Initializers** — Glorot uniform and normal, He uniform and normal, Xavier uniform and normal
- **Activations** — ReLU, Sigmoid, Tanh, Heaviside
- **Containers** — Sequential model composition
- **Loss Functions** — MSE loss, L1 loss
- **Model Class** — High-level API for training and inference

### Optimizers
- **SGD** — Stochastic gradient descent with momentum

### Data Processing
- **CSV Parser** — Load tabular data from CSV files (sort of hacked together, but it works)
- **StandardScaler** — Feature normalization with fit/transform/inverse_transform

### Utilities
- **Random** — Seed control, tensor initialization (randn, rand)
- **Logger** — Logging utilities with timestamps

---

## Build Instructions

### Prerequisites
- CMake >= 3.16
- C++20 compatible compiler (GCC, Clang)
- OpenMP (optional, for parallel operations)

```bash
# Clone the repository
git clone https://github.com/yourusername/Quasai.git
cd Quasai

# Create build directory and configure
cmake -S . -B build

# Build the library
cmake --build build

# Run tests
ctest --test-dir build --output-on-failure
```

### Example Usage

```cpp
#include "quasai/core/tensor.hpp"
#include "quasai/nn/linear.hpp"
#include "quasai/nn/sequential.hpp"
#include "quasai/nn/activations.hpp"
#include "quasai/optim/sgd.hpp"

int main() {
    // Create a simple MLP
    auto linear1 = std::make_shared<quasai::Linear>(64, 32);
    auto relu = std::make_shared<quasai::ReLU>();
    auto linear2 = std::make_shared<quasai::Linear>(32, 10);
    
    auto model = std::make_shared<quasai::Sequential>(std::vector<std::shared_ptr<quasai::Module>>{
        linear1, relu, linear2
    });
    
    quasai::Model ml_model(model);
    
    quasai::SGD optimizer(ml_model.parameters(), 0.01f, 0.9f);
    
    // Training data (batch_size, 64)
    quasai::Tensor X = quasai::Tensor::zeros({32, 64});
    quasai::Tensor y = quasai::Tensor::zeros({32, 10});
    
    ml_model.train(X, y, quasai::Loss::MSE, optimizer, 10, 32);
    
    return 0;
}
```

See `examples/cpp/` for more examples.

---

## License

MIT License - see LICENSE file for details.

---

**Disclaimer:** This project is still in active development. The API may change.
