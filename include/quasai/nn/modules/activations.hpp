#pragma once

#include "quasai/nn/module.hpp"
#include "quasai/ops/tensor_ops.hpp"

namespace quasai::nn {

/// @brief ReLU activation module: y = max(0, x).
class ReLU : public Module {
public:
  ReLU() = default;

  /// @brief Forward pass: applies ReLU activation.
  core::Tensor forward(const core::Tensor &input) override {
    return ops::relu(input);
  }
};

/// @brief Sigmoid activation module: y = 1 / (1 + exp(-x)).
class Sigmoid : public Module {
public:
  Sigmoid() = default;

  /// @brief Forward pass: applies Sigmoid activation.
  core::Tensor forward(const core::Tensor &input) override {
    return ops::sigmoid(input);
  }
};

/// @brief Tanh activation module: y = tanh(x).
class Tanh : public Module {
public:
  Tanh() = default;

  /// @brief Forward pass: applies Tanh activation.
  core::Tensor forward(const core::Tensor &input) override {
    return ops::tanh(input);
  }
};

/// @brief Heaviside step module: y = 0 if x<0 else 1.
class Heaviside : public Module {
public:
  Heaviside() = default;

  /// @brief Forward pass: applies Heaviside step.
  core::Tensor forward(const core::Tensor &input) override {
    return ops::heaviside(input);
  }
};

} // namespace quasai::nn
