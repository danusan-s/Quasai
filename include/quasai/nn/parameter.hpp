#pragma once

#include "quasai/core/tensor.hpp"

namespace quasai::nn {

/**
 * @brief Tensor that automatically requires gradient.
 * @details Wraps a Tensor and sets requires_grad(true) on construction.
 */
class Parameter : public core::Tensor {
public:
  /**
   * @brief Construct a Parameter from a Tensor.
   * @param tensor Source tensor (will have requires_grad set to true).
   */
  Parameter(const core::Tensor &tensor) : Tensor(tensor) {
    this->requires_grad(true);
  }
};

} // namespace quasai::nn
