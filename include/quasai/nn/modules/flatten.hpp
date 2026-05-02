#pragma once

#include "quasai/nn/module.hpp"
#include "quasai/ops/tensor_ops.hpp"

namespace quasai::nn {

/**
 * Flatten layer that reshapes the input tensor to have a shape of
 * (batch_size, feature_size). The feature_size is calculated as the total
 * number of elements in the input tensor divided by the batch size.
 */
class Flatten : public Module {
public:
  Flatten() = default;

  core::Tensor forward(const core::Tensor &input) override {
    if (input.shape().dimensions() < 2) {
      throw std::invalid_argument(
          "Input tensor must have at least 2 dimensions to flatten");
    }

    size_t batch_size = input.shape()[0];
    size_t total = core::total_size(input.shape());
    size_t feature_size = total / batch_size;

    return ops::reshape(input, core::Shape{input.shape()[0], feature_size});
  }
};

} // namespace quasai::nn
