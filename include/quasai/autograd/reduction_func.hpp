#pragma once

#include "quasai/autograd/function.hpp"

namespace quasai::autograd {

class SumFunction : public Function {
public:
  std::vector<core::Tensor> backward(const core::Tensor &grad_output) override;
};

class SumToShapeFunction : public Function {
public:
  std::vector<core::Tensor> backward(const core::Tensor &grad_output) override;
};

} // namespace quasai::autograd
