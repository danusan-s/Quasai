#include "quasai/autograd/function.hpp"
#include "quasai/ops/tensor_ops.hpp"

namespace quasai {

std::vector<Tensor> AddFunction::backward(const Tensor &grad_output) {
  // same gradient for both inputs and is just the grad_output
  return {sum_to_shape(grad_output, inputs[0].shape()),
          sum_to_shape(grad_output, inputs[1].shape())};
}

std::vector<Tensor> SubFunction::backward(const Tensor &grad_output) {
  // grad for input 1 is grad_output
  // grad for input 2 is -grad_output
  return {sum_to_shape(grad_output, inputs[0].shape()),
          sum_to_shape(neg(grad_output), inputs[1].shape())};
}

std::vector<Tensor> MulFunction::backward(const Tensor &grad_output) {
  // grad for input 1 is grad_output * input 2
  // grad for input 2 is grad_output * input 1
  const Tensor &input1 = inputs[0];
  const Tensor &input2 = inputs[1];
  Tensor grad_input1 = mul(grad_output, input2);
  Tensor grad_input2 = mul(grad_output, input1);
  return {sum_to_shape(grad_input1, input1.shape()),
          sum_to_shape(grad_input2, input2.shape())};
}

std::vector<Tensor> DivFunction::backward(const Tensor &grad_output) {
  // grad for input 1 is grad_output / input 2
  // grad for input 2 is -grad_output * input 1 / (input 2^2)
  const Tensor &input1 = inputs[0];
  const Tensor &input2 = inputs[1];
  Tensor grad_input1 = div(grad_output, input2);
  Tensor grad_input2 = neg(mul(grad_output, div(input1, mul(input2, input2))));
  return {sum_to_shape(grad_input1, input1.shape()),
          sum_to_shape(grad_input2, input2.shape())};
}

} // namespace quasai
