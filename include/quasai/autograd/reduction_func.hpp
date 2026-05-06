#pragma once

#include "quasai/autograd/function.hpp"

namespace quasai::autograd {

/// @brief Function node for sum reduction.
class SumFunction : public Function {
public:
  std::vector<core::Tensor> backward(const core::Tensor &grad_output) override;
};

/// @brief Function node for sum-to-shape reduction.
class SumToShapeFunction : public Function {
public:
  std::vector<core::Tensor> backward(const core::Tensor &grad_output) override;
};

} // namespace quasai::autograd
