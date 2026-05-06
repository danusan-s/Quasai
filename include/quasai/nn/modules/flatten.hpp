#pragma once

#include "quasai/nn/module.hpp"
#include "quasai/ops/tensor_ops.hpp"

namespace quasai::nn {

/**
 * @brief Flatten layer: reshapes (batch, ...) to (batch, features).
 * @details Computes feature_size = total_elements / batch_size.
 */
class Flatten : public Module {
public:
  /// @brief Construct a Flatten layer.
  Flatten() = default;

  /// @brief Forward pass: flattens all but the first dimension.
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
