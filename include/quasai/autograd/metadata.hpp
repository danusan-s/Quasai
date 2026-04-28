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

} // namespace quasai::autograd
