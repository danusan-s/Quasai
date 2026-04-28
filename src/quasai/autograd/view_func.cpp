#include "quasai/autograd/view_func.hpp"
#include "quasai/ops/tensor_ops.hpp"

namespace quasai::autograd {

std::vector<core::Tensor> TransposeFunction::backward(const core::Tensor &grad_output) {
  const core::Tensor &input = inputs[0];
  LOG_DEBUG(("TransposeFunction backward: grad_output shape = " +
             grad_output.shape().to_string() +
             ", input shape = " + input.shape().to_string())
                .c_str());

  core::Tensor grad_input = ops::transpose(grad_output);
  return {grad_input};
}

std::vector<core::Tensor> ReshapeFunction::backward(const core::Tensor &grad_output) {
  const core::Tensor &input = inputs[0];
  LOG_DEBUG(("ReshapeFunction backward: grad_output shape = " +
             grad_output.shape().to_string() +
             ", input shape = " + input.shape().to_string())
                .c_str());

  core::Tensor grad_input = ops::reshape(grad_output, input.shape());
  return {grad_input};
}

std::vector<core::Tensor> ExpandFunction::backward(const core::Tensor &grad_output) {
  const core::Tensor &input = inputs[0];
  LOG_DEBUG(("ExpandFunction backward: grad_output shape = " +
             grad_output.shape().to_string() +
             ", input shape = " + input.shape().to_string())
                .c_str());

  core::Tensor grad_input = ops::sum_to_shape(grad_output, input.shape());
  return {grad_input};
}

std::vector<core::Tensor>
MakeContiguousFunction::backward(const core::Tensor &grad_output) {
  const core::Tensor &input = inputs[0];
  LOG_DEBUG(("MakeContiguousFunction backward: grad_output shape = " +
             grad_output.shape().to_string() +
             ", input shape = " + input.shape().to_string())
                .c_str());

  return {grad_output};
}

} // namespace quasai::autograd
