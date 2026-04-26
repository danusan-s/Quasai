
#include "quasai/autograd/function.hpp"
#include "quasai/ops/tensor_ops.hpp"

namespace quasai {

std::vector<Tensor> TransposeFunction::backward(const Tensor &grad_output) {
  // grad for input is just the transpose of grad_output
  const Tensor &input = inputs[0];
  LOG_DEBUG(("TransposeFunction backward: grad_output shape = " +
             grad_output.shape().to_string() +
             ", input shape = " + input.shape().to_string())
                .c_str());

  Tensor grad_input = transpose(grad_output);
  return {grad_input};
}

std::vector<Tensor> ReshapeFunction::backward(const Tensor &grad_output) {
  // grad for input is just grad_output reshaped to input shape
  const Tensor &input = inputs[0];
  LOG_DEBUG(("ReshapeFunction backward: grad_output shape = " +
             grad_output.shape().to_string() +
             ", input shape = " + input.shape().to_string())
                .c_str());

  Tensor grad_input = reshape(grad_output, input.shape());
  return {grad_input};
}

std::vector<Tensor> ExpandFunction::backward(const Tensor &grad_output) {
  // grad for input is grad_output summed to input shape
  const Tensor &input = inputs[0];
  LOG_DEBUG(("ExpandFunction backward: grad_output shape = " +
             grad_output.shape().to_string() +
             ", input shape = " + input.shape().to_string())
                .c_str());

  Tensor grad_input = sum_to_shape(grad_output, input.shape());
  return {grad_input};
}

std::vector<Tensor>
MakeContiguousFunction::backward(const Tensor &grad_output) {
  // grad for input is just grad_output since making contiguous is a view
  const Tensor &input = inputs[0];
  LOG_DEBUG(("MakeContiguousFunction backward: grad_output shape = " +
             grad_output.shape().to_string() +
             ", input shape = " + input.shape().to_string())
                .c_str());

  return {grad_output};
}

} // namespace quasai
