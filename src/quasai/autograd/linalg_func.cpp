#include "quasai/autograd/function.hpp"
#include "quasai/ops/tensor_ops.hpp"

namespace quasai {

std::vector<Tensor> MatMulFunction::backward(const Tensor &grad_output) {
  // grad for input 1 is grad_output @ input 2^T
  // grad for input 2 is input 1^T @ grad_output
  const Tensor &input1 = inputs[0];
  const Tensor &input2 = inputs[1];
  LOG_DEBUG(("MatMulFunction backward: grad_output shape = " +
             grad_output.shape().to_string() +
             ", input1 shape = " + input1.shape().to_string() +
             ", input2 shape = " + input2.shape().to_string())
                .c_str());

  Tensor grad_input1 = matmul(grad_output, transpose(input2));
  Tensor grad_input2 = matmul(transpose(input1), grad_output);
  return {grad_input1, grad_input2};
}

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

} // namespace quasai
