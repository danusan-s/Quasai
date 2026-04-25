#pragma once

#include "quasai/nn/module.hpp"
#include "quasai/ops/tensor_ops.hpp"

namespace quasai {

class ReLU : public Module {
public:
  ReLU() = default;

  Tensor forward(const Tensor &input) override {
    return relu(input);
  }
};

class Sigmoid : public Module {
public:
  Sigmoid() = default;

  Tensor forward(const Tensor &input) override {
    return sigmoid(input);
  }
};

class Tanh : public Module {
public:
  Tanh() = default;

  Tensor forward(const Tensor &input) override {
    return tanh(input);
  }
};

class Heaviside : public Module {
public:
  Heaviside() = default;

  Tensor forward(const Tensor &input) override {
    return step(input);
  }
};

} // namespace quasai
