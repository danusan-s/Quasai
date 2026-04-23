#pragma once

#include "quasai/core/tensor.hpp"

namespace quasai {

class AutoGradEngine {
public:
  static void backward(const Tensor &tensor, const Tensor &grad = Tensor());
};

} // namespace quasai
