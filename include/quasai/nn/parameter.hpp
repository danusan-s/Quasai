#pragma once

#include "quasai/core/tensor.hpp"

namespace quasai::nn {

class Parameter : public core::Tensor {
public:
  Parameter(const core::Tensor &tensor) : Tensor(tensor) {
    this->requires_grad(true);
  }
};

} // namespace quasai::nn
