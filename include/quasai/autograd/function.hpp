#pragma once

#include "quasai/core/tensor.hpp"
#include <vector>

namespace quasai::autograd {

class Function {
public:
  // Input tensors that produced this function's output. Populated by the
  // forward pass before backward() is called. Used by the engine to traverse
  // the compute graph and propagate gradients to these inputs.
  std::vector<core::Tensor> inputs;

  virtual std::vector<core::Tensor>
  backward(const core::Tensor
               &grad_output) = 0; // Pure virtual function for backward pass

  virtual ~Function() = default; // Virtual destructor for proper cleanup
};

class MatMulFunction : public Function {
public:
  std::vector<core::Tensor> backward(const core::Tensor &grad_output) override;
};

} // namespace quasai::autograd
