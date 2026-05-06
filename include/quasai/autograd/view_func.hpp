#pragma once

#include "quasai/autograd/function.hpp"

namespace quasai::autograd {

/// @brief Function node for matrix transpose.
class TransposeFunction : public Function {
public:
  std::vector<core::Tensor> backward(const core::Tensor &grad_output) override;
};

/// @brief Function node for reshape operation.
class ReshapeFunction : public Function {
public:
  std::vector<core::Tensor> backward(const core::Tensor &grad_output) override;
};

/// @brief Function node for expand (broadcast view) operation.
class ExpandFunction : public Function {
public:
  std::vector<core::Tensor> backward(const core::Tensor &grad_output) override;
};

/// @brief Function node for make_contiguous operation.
class MakeContiguousFunction : public Function {
public:
  std::vector<core::Tensor> backward(const core::Tensor &grad_output) override;
};

} // namespace quasai::autograd
