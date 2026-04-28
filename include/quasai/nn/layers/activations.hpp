#pragma once

#include "quasai/nn/module.hpp"
#include "quasai/ops/tensor_ops.hpp"

namespace quasai::nn {

class ReLU : public Module {
public:
  ReLU() = default;

  core::Tensor forward(const core::Tensor &input) override {
    return relu(input);
  }
};

class Sigmoid : public Module {
public:
  Sigmoid() = default;

  core::Tensor forward(const core::Tensor &input) override {
    return sigmoid(input);
  }
};

class Tanh : public Module {
public:
  Tanh() = default;

  core::Tensor forward(const core::Tensor &input) override {
    return tanh(input);
  }
};

class Heaviside : public Module {
public:
  Heaviside() = default;

  core::Tensor forward(const core::Tensor &input) override {
    return heaviside(input);
  }
};

} // namespace quasai::nn
