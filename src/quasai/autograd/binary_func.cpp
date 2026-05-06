#include "quasai/autograd/binary_func.hpp"

#include "quasai/autograd/metadata.hpp"
#include "quasai/ops/tensor_ops.hpp"

namespace quasai::autograd {

std::vector<core::Tensor>
AddFunction::backward(const core::Tensor &grad_output) {
  const core::Tensor &input1 = inputs[0];
  const core::Tensor &input2 = inputs[1];
  LOG_DEBUG(("AddFunction backward: grad_output shape = " +
             grad_output.shape().to_string() +
             ", input1 shape = " + input1.shape().to_string() +
             ", input2 shape = " + input2.shape().to_string())
                .c_str());

  core::Tensor grad_input1;
  core::Tensor grad_input2;

  if (tensor_requires_grad(input1)) {
    grad_input1 = ops::sum_to_shape(grad_output, input1.shape());
  }
  if (tensor_requires_grad(input2)) {
    grad_input2 = ops::sum_to_shape(grad_output, input2.shape());
  }

  return {grad_input1, grad_input2};
}

std::vector<core::Tensor>
SubFunction::backward(const core::Tensor &grad_output) {
  const core::Tensor &input1 = inputs[0];
  const core::Tensor &input2 = inputs[1];
  LOG_DEBUG(("SubFunction backward: grad_output shape = " +
             grad_output.shape().to_string() +
             ", input1 shape = " + input1.shape().to_string() +
             ", input2 shape = " + input2.shape().to_string())
                .c_str());

  core::Tensor grad_input1;
  core::Tensor grad_input2;

  if (tensor_requires_grad(input1)) {
    grad_input1 = ops::sum_to_shape(grad_output, input1.shape());
  }
  if (tensor_requires_grad(input2)) {
    grad_input2 = ops::sum_to_shape(ops::neg(grad_output), input2.shape());
  }

  return {grad_input1, grad_input2};
}

std::vector<core::Tensor>
MulFunction::backward(const core::Tensor &grad_output) {
  const core::Tensor &input1 = inputs[0];
  const core::Tensor &input2 = inputs[1];
  LOG_DEBUG(("MulFunction backward: grad_output shape = " +
             grad_output.shape().to_string() +
             ", input1 shape = " + input1.shape().to_string() +
             ", input2 shape = " + input2.shape().to_string())
                .c_str());
  core::Tensor grad_input1;
  core::Tensor grad_input2;

  if (tensor_requires_grad(input1)) {
    grad_input1 =
        ops::sum_to_shape(ops::mul(grad_output, input2), input1.shape());
  }
  if (tensor_requires_grad(input2)) {
    grad_input2 =
        ops::sum_to_shape(ops::mul(grad_output, input1), input2.shape());
  }

  return {grad_input1, grad_input2};
}

std::vector<core::Tensor>
DivFunction::backward(const core::Tensor &grad_output) {
  const core::Tensor &input1 = inputs[0];
  const core::Tensor &input2 = inputs[1];
  LOG_DEBUG(("DivFunction backward: grad_output shape = " +
             grad_output.shape().to_string() +
             ", input1 shape = " + input1.shape().to_string() +
             ", input2 shape = " + input2.shape().to_string())
                .c_str());

  core::Tensor grad_input1;
  core::Tensor grad_input2;

  if (tensor_requires_grad(input1)) {
    grad_input1 =
        ops::sum_to_shape(ops::div(grad_output, input2), input1.shape());
  }
  if (tensor_requires_grad(input2)) {
    grad_input2 = ops::sum_to_shape(
        ops::neg(
            ops::mul(grad_output, ops::div(input1, ops::mul(input2, input2)))),
        input2.shape());
  }
  return {grad_input1, grad_input2};
}

} // namespace quasai::autograd
