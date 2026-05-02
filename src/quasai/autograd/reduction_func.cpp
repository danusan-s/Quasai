#include "quasai/autograd/reduction_func.hpp"
#include "quasai/autograd/metadata.hpp"
#include "quasai/ops/core_ops.hpp"
#include "quasai/ops/tensor_ops.hpp"

namespace quasai::autograd {

std::vector<core::Tensor>
SumFunction::backward(const core::Tensor &grad_output) {
  const core::Tensor &input = inputs[0];
  LOG_DEBUG(("SumFunction backward: grad_output shape = " +
             grad_output.shape().to_string() +
             ", input shape = " + input.shape().to_string())
                .c_str());
  core::Tensor grad_input;
  if (tensor_requires_grad(input)) {
    grad_input = ops::expand(grad_output, input.shape());
  }
  return {grad_input};
}

std::vector<core::Tensor>
SumToShapeFunction::backward(const core::Tensor &grad_output) {
  const core::Tensor &input = inputs[0];
  LOG_DEBUG(("SumToShapeFunction backward: grad_output shape = " +
             grad_output.shape().to_string() +
             ", input shape = " + input.shape().to_string())
                .c_str());
  core::Tensor grad_input;
  if (tensor_requires_grad(input)) {
    grad_input = ops::expand(grad_output, input.shape());
  }
  return {grad_input};
}

} // namespace quasai::autograd
