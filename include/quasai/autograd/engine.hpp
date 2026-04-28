#pragma once

#include "quasai/core/tensor.hpp"

namespace quasai::autograd {

class AutoGradEngine {
public:
  static void backward(const core::Tensor &tensor);
};

} // namespace quasai::autograd
