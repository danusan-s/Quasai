#pragma once

#include "quasai/nn/module.hpp"
#include "quasai/ops/tensor_ops.hpp"

namespace quasai::nn {

class Flatten : public Module {
public:
  Flatten() = default;

  core::Tensor forward(const core::Tensor &input) override {
    size_t batch_size = input.shape()[0];
    size_t total = core::total_size(input.shape());
    size_t feature_size = total / batch_size;

    return quasai::reshape(input, core::Shape{input.shape()[0], feature_size});
  }
};

} // namespace quasai::nn
