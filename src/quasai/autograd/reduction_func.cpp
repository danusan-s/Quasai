#include "quasai/autograd/function.hpp"
#include "quasai/ops/tensor_ops.hpp"

namespace quasai {

std::vector<Tensor> SumFunction::backward(const Tensor &grad_output) {
  // grad for input is just grad_output broadcasted to input shape
  const Tensor &input = inputs[0];
  Tensor grad_input = broadcast_to_shape(grad_output, input.shape());
  return {grad_input};
}

} // namespace quasai
