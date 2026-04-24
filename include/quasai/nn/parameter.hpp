#pragma once

#include "quasai/core/tensor.hpp"

namespace quasai {

class Parameter : public Tensor {
public:
  Parameter(const Tensor &tensor) : Tensor(tensor) {
    this->requires_grad(true);
  }
};

} // namespace quasai
