#include "quasai/autograd/function.hpp"
#include "quasai/ops/tensor_ops.hpp"

namespace quasai {

std::vector<Tensor> MatMulFunction::backward(const Tensor &grad_output) {
  // grad for input 1 is grad_output @ input 2^T
  // grad for input 2 is input 1^T @ grad_output
  const Tensor &input1 = inputs[0];
  const Tensor &input2 = inputs[1];
  Tensor grad_input1 = matmul(grad_output, transpose(input2));
  Tensor grad_input2 = matmul(transpose(input1), grad_output);
  return {grad_input1, grad_input2};
}

} // namespace quasai
