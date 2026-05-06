#include "quasai/ops/tensor_ops.hpp"

#include "quasai/autograd/metadata.hpp"

namespace quasai::autograd {

std::vector<core::Tensor>
MatMulFunction::backward(const core::Tensor &grad_output) {
  const core::Tensor &input1 = inputs[0];
  const core::Tensor &input2 = inputs[1];
  LOG_DEBUG(("MatMulFunction backward: grad_output shape = " +
             grad_output.shape().to_string() +
             ", input1 shape = " + input1.shape().to_string() +
             ", input2 shape = " + input2.shape().to_string())
                .c_str());

  core::Tensor grad_input1;
  core::Tensor grad_input2;

  if (tensor_requires_grad(input1)) {
    grad_input1 = ops::matmul(grad_output, ops::transpose(input2));
    if (input1.shape().dimensions() == 1) {
      grad_input1 = ops::reshape(grad_input1, input1.shape());
    }
  }
  if (tensor_requires_grad(input2)) {
    grad_input2 = ops::matmul(ops::transpose(input1), grad_output);
    if (input2.shape().dimensions() == 1) {
      grad_input2 = ops::reshape(grad_input2, input2.shape());
    }
  }

  return {grad_input1, grad_input2};
}

} // namespace quasai::autograd
