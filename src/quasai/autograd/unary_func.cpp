#include "quasai/autograd/function.hpp"
#include "quasai/ops/tensor_ops.hpp"

namespace quasai {

std::vector<Tensor> NegFunction::backward(const Tensor &grad_output) {
  // grad for input is just -grad_output
  LOG_DEBUG(("NegFunction backward: grad_output shape = " +
             grad_output.shape().to_string())
                .c_str());
  return {neg(grad_output)};
}

std::vector<Tensor> ReluFunction::backward(const Tensor &grad_output) {
  // grad for input is grad_output where input > 0 and 0 otherwise
  const Tensor &input = inputs[0];
  LOG_DEBUG(
      ("ReluFunction backward: input shape = " + input.shape().to_string() +
       ", grad_output shape = " + grad_output.shape().to_string())
          .c_str());
  Tensor grad_input = mul(grad_output, step(input));
  return {grad_input};
}

std::vector<Tensor> SigmoidFunction::backward(const Tensor &grad_output) {
  // grad for input is grad_output * sigmoid(input) * (1 - sigmoid(input))
  const Tensor &input = inputs[0];
  LOG_DEBUG(
      ("SigmoidFunction backward: input shape = " + input.shape().to_string() +
       ", grad_output shape = " + grad_output.shape().to_string())
          .c_str());
  Tensor sigmoid_input = sigmoid(input);
  Tensor grad_input =
      mul(grad_output,
          mul(sigmoid_input, sub(Tensor::ones(input.shape()), sigmoid_input)));
  return {grad_input};
}

std::vector<Tensor> TanhFunction::backward(const Tensor &grad_output) {
  // grad for input is grad_output * (1 - tanh(input)^2)
  const Tensor &input = inputs[0];
  LOG_DEBUG(
      ("TanhFunction backward: input shape = " + input.shape().to_string() +
       ", grad_output shape = " + grad_output.shape().to_string())
          .c_str());
  Tensor tanh_input = tanh(input);
  Tensor grad_input = mul(grad_output, sub(Tensor::ones(input.shape()),
                                           mul(tanh_input, tanh_input)));
  return {grad_input};
}

} // namespace quasai
