#include "quasai/autograd/binary_func.hpp"
#include "quasai/ops/tensor_ops.hpp"

namespace quasai::autograd {

std::vector<core::Tensor>
AddFunction::backward(const core::Tensor &grad_output) {
  // same gradient for both inputs and is just the grad_output
  const core::Tensor &input1 = inputs[0];
  const core::Tensor &input2 = inputs[1];
  LOG_DEBUG(("AddFunction backward: grad_output shape = " +
             grad_output.shape().to_string() +
             ", input1 shape = " + input1.shape().to_string() +
             ", input2 shape = " + input2.shape().to_string())
                .c_str());
  return {sum_to_shape(grad_output, input1.shape()),
          sum_to_shape(grad_output, input2.shape())};
}

std::vector<core::Tensor>
SubFunction::backward(const core::Tensor &grad_output) {
  // grad for input 1 is grad_output
  // grad for input 2 is -grad_output
  const core::Tensor &input1 = inputs[0];
  const core::Tensor &input2 = inputs[1];
  LOG_DEBUG(("SubFunction backward: grad_output shape = " +
             grad_output.shape().to_string() +
             ", input1 shape = " + input1.shape().to_string() +
             ", input2 shape = " + input2.shape().to_string())
                .c_str());
  return {sum_to_shape(grad_output, input1.shape()),
          sum_to_shape(neg(grad_output), input2.shape())};
}

std::vector<core::Tensor>
MulFunction::backward(const core::Tensor &grad_output) {
  // grad for input 1 is grad_output * input 2
  // grad for input 2 is grad_output * input 1
  const core::Tensor &input1 = inputs[0];
  const core::Tensor &input2 = inputs[1];
  LOG_DEBUG(("MulFunction backward: grad_output shape = " +
             grad_output.shape().to_string() +
             ", input1 shape = " + input1.shape().to_string() +
             ", input2 shape = " + input2.shape().to_string())
                .c_str());
  core::Tensor grad_input1 = mul(grad_output, input2);
  core::Tensor grad_input2 = mul(grad_output, input1);
  return {sum_to_shape(grad_input1, input1.shape()),
          sum_to_shape(grad_input2, input2.shape())};
}

std::vector<core::Tensor>
DivFunction::backward(const core::Tensor &grad_output) {
  // grad for input 1 is grad_output / input 2
  // grad for input 2 is -grad_output * input 1 / (input 2^2)
  const core::Tensor &input1 = inputs[0];
  const core::Tensor &input2 = inputs[1];
  LOG_DEBUG(("DivFunction backward: grad_output shape = " +
             grad_output.shape().to_string() +
             ", input1 shape = " + input1.shape().to_string() +
             ", input2 shape = " + input2.shape().to_string())
                .c_str());
  core::Tensor grad_input1 = div(grad_output, input2);
  core::Tensor grad_input2 =
      neg(mul(grad_output, div(input1, mul(input2, input2))));
  return {sum_to_shape(grad_input1, input1.shape()),
          sum_to_shape(grad_input2, input2.shape())};
}

} // namespace quasai::autograd
