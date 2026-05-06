#pragma once

#include "quasai/core/tensor.hpp"
#include <vector>

namespace quasai::autograd {

class Function {
public:
  /// @brief Virtual method for backward pass of the autograd function.
  virtual std::vector<core::Tensor>
  backward(const core::Tensor &grad_output) = 0;

  /// @brief Virtual destructor for proper cleanup in derived classes.
  virtual ~Function() = default;

  /**
   *  @brief Input tensors that produced this function's output.
   *  Populated by the forward pass before backward() is called.
   *  Used by the engine to traverse the compute graph and propagate
   *  gradients to these inputs.
   */
  std::vector<core::Tensor> inputs;
};

/**
 * @brief Function node for matrix multiplication in the autograd graph.
 */
class MatMulFunction : public Function {
public:
  /**
   * @brief Compute gradients for matmul.
   * @param grad_output Gradient flowing back from the output.
   * @return Pair of gradients for input1 and input2.
   */
  std::vector<core::Tensor> backward(const core::Tensor &grad_output) override;
};

} // namespace quasai::autograd
