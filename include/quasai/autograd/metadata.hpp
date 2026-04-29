#pragma once

#include "quasai/autograd/function.hpp"
#include "quasai/core/tensor.hpp"

namespace quasai::autograd {

struct AutoGradMeta {
  core::Tensor grad;
  std::unique_ptr<Function> grad_fn;
  bool requires_grad;

  AutoGradMeta() : grad_fn(nullptr), requires_grad(false) {
  }
};

inline bool tensor_requires_grad(const core::Tensor &tensor) {
  return tensor.autograd_meta() && tensor.autograd_meta()->requires_grad;
}

} // namespace quasai::autograd
