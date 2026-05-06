#pragma once

#include "quasai/autograd/function.hpp"
#include "quasai/core/tensor.hpp"

namespace quasai::autograd {

/**
 * @brief Metadata attached to tensors that require gradient tracking.
 */
struct AutoGradMeta {
  core::Tensor grad;                 ///< Accumulated gradient.
  std::unique_ptr<Function> grad_fn; ///< Function that produced this tensor.
  bool requires_grad;                ///< Whether to track gradients.

  AutoGradMeta() : grad_fn(nullptr), requires_grad(false) {
  }
};

/**
 * @brief Check if a tensor requires gradient computation.
 * @param tensor Tensor to check.
 * @return true if tensor has autograd_meta and requires_grad is set.
 */
inline bool tensor_requires_grad(const core::Tensor &tensor) {
  return tensor.autograd_meta() && tensor.autograd_meta()->requires_grad;
}

} // namespace quasai::autograd
