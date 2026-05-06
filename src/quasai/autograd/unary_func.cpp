#include "quasai/autograd/unary_func.hpp"

#include "quasai/autograd/metadata.hpp"
#include "quasai/ops/tensor_ops.hpp"

namespace quasai::autograd {

std::vector<core::Tensor>
NegFunction::backward(const core::Tensor &grad_output) {
  const core::Tensor &input = inputs[0];
  LOG_DEBUG(("NegFunction backward: grad_output shape = " +
             grad_output.shape().to_string())
                .c_str());
  core::Tensor grad_input;
  if (tensor_requires_grad(input)) {
    grad_input = ops::neg(grad_output);
  }
  return {grad_input};
}

std::vector<core::Tensor>
AbsFunction::backward(const core::Tensor &grad_output) {
  const core::Tensor &input = inputs[0];
  LOG_DEBUG(
      ("AbsFunction backward: input shape = " + input.shape().to_string() +
       ", grad_output shape = " + grad_output.shape().to_string())
          .c_str());
  core::Tensor grad_input;
  if (tensor_requires_grad(input)) {
    grad_input = ops::mul(grad_output, ops::signum(input));
  }
  return {grad_input};
}

std::vector<core::Tensor>
HeavisideFunction::backward(const core::Tensor &grad_output) {
  const core::Tensor &input = inputs[0];
  LOG_DEBUG(("HeavisideFunction backward: input shape = " +
             input.shape().to_string() +
             ", grad_output shape = " + grad_output.shape().to_string())
                .c_str());
  core::Tensor grad_input;
  if (tensor_requires_grad(input)) {
    grad_input =
        core::Tensor::zeros(input.shape(), input.dtype(), input.device());
  }
  return {grad_input};
}

std::vector<core::Tensor>
SignumFunction::backward(const core::Tensor &grad_output) {
  const core::Tensor &input = inputs[0];
  LOG_DEBUG(
      ("SignumFunction backward: input shape = " + input.shape().to_string() +
       ", grad_output shape = " + grad_output.shape().to_string())
          .c_str());
  core::Tensor grad_input;
  if (tensor_requires_grad(input)) {
    grad_input =
        core::Tensor::zeros(input.shape(), input.dtype(), input.device());
  }
  return {grad_input};
}

std::vector<core::Tensor>
ReluFunction::backward(const core::Tensor &grad_output) {
  const core::Tensor &input = inputs[0];
  LOG_DEBUG(
      ("ReluFunction backward: input shape = " + input.shape().to_string() +
       ", grad_output shape = " + grad_output.shape().to_string())
          .c_str());
  core::Tensor grad_input;
  if (tensor_requires_grad(input)) {
    grad_input = ops::mul(grad_output, ops::heaviside(input));
  }
  return {grad_input};
}

std::vector<core::Tensor>
SigmoidFunction::backward(const core::Tensor &grad_output) {
  const core::Tensor &input = inputs[0];
  LOG_DEBUG(
      ("SigmoidFunction backward: input shape = " + input.shape().to_string() +
       ", grad_output shape = " + grad_output.shape().to_string())
          .c_str());
  core::Tensor grad_input;
  if (tensor_requires_grad(input)) {
    core::Tensor sigmoid_input = ops::sigmoid(input);
    grad_input = ops::mul(
        grad_output,
        ops::mul(sigmoid_input,
                 ops::sub(core::Tensor::ones(input.shape()), sigmoid_input)));
  }
  return {grad_input};
}

std::vector<core::Tensor>
TanhFunction::backward(const core::Tensor &grad_output) {
  const core::Tensor &input = inputs[0];
  LOG_DEBUG(
      ("TanhFunction backward: input shape = " + input.shape().to_string() +
       ", grad_output shape = " + grad_output.shape().to_string())
          .c_str());
  core::Tensor grad_input;
  if (tensor_requires_grad(input)) {
    core::Tensor tanh_input = ops::tanh(input);
    grad_input =
        ops::mul(grad_output, ops::sub(core::Tensor::ones(input.shape()),
                                       ops::mul(tanh_input, tanh_input)));
  }
  return {grad_input};
}

std::vector<core::Tensor>
PowFunction::backward(const core::Tensor &grad_output) {
  const core::Tensor &input = inputs[0];
  LOG_DEBUG(
      ("PowFunction backward: input shape = " + input.shape().to_string() +
       ", grad_output shape = " + grad_output.shape().to_string())
          .c_str());
  core::Tensor grad_input;
  if (tensor_requires_grad(input)) {
    if (exponent == 0) {
      // If exponent is 0, the output is always 1, so the gradient w.r.t input
      // is 0
      grad_input =
          core::Tensor::zeros(input.shape(), input.dtype(), input.device());
    } else if (exponent == 1) {
      // If exponent is 1, the output is the same as input, so the gradient
      // w.r.t input is just grad_output
      grad_input = grad_output;
    } else {
      grad_input = ops::mul(grad_output,
                            ops::mul(exponent, ops::pow(input, exponent - 1)));
    }
  }
  return {grad_input};
}

} // namespace quasai::autograd
