#include "quasai/autograd/unary_func.hpp"

#include "quasai/autograd/metadata.hpp"
#include "quasai/core/constants.hpp"
#include "quasai/ops/tensor_ops.hpp"

#include <cmath>

namespace quasai::autograd {

std::vector<core::Tensor>
NegFunction::backward(const core::Tensor &grad_output) {
  const core::Tensor &input = inputs_[0];
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
  const core::Tensor &input = inputs_[0];
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
  const core::Tensor &input = inputs_[0];
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
  const core::Tensor &input = inputs_[0];
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
  const core::Tensor &input = inputs_[0];
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
  const core::Tensor &input = inputs_[0];
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
  const core::Tensor &input = inputs_[0];
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
  const core::Tensor &input = inputs_[0];
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

std::vector<core::Tensor>
ExpFunction::backward(const core::Tensor &grad_output) {
  const core::Tensor &input = inputs_[0];
  LOG_DEBUG(
      ("ExpFunction backward: input shape = " + input.shape().to_string() +
       ", grad_output shape = " + grad_output.shape().to_string())
          .c_str());
  core::Tensor grad_input;
  if (tensor_requires_grad(input)) {
    core::Tensor exp_input = ops::exp(input);
    grad_input = ops::mul(grad_output, exp_input);
  }
  return {grad_input};
}

std::vector<core::Tensor>
LogFunction::backward(const core::Tensor &grad_output) {
  const core::Tensor &input = inputs_[0];
  LOG_DEBUG(
      ("LogFunction backward: input shape = " + input.shape().to_string() +
       ", grad_output shape = " + grad_output.shape().to_string())
          .c_str());
  core::Tensor grad_input;
  if (tensor_requires_grad(input)) {
    const core::Tensor eps = core::Tensor::from_scalar(
        core::LOG_EPSILON, input.dtype(), input.device());
    const core::Tensor log_base = core::Tensor::from_scalar(
        std::log(base), input.dtype(), input.device());
    grad_input =
        ops::div(grad_output, ops::mul(ops::add(input, eps), log_base));
  }
  return {grad_input};
}

std::vector<core::Tensor>
GeluFunction::backward(const core::Tensor &grad_output) {
  const core::Tensor &input = inputs_[0];
  LOG_DEBUG(
      ("GeluFunction backward: input shape = " + input.shape().to_string() +
       ", grad_output shape = " + grad_output.shape().to_string())
          .c_str());
  core::Tensor grad_input;
  if (tensor_requires_grad(input)) {
    // GELU derivative (tanh approximation):
    // GELU(x) = 0.5 * x * (1 + tanh(k * (x + a * x^3)))
    // where k = sqrt(2/pi), a = 0.044715
    // dGELU/dx = 0.5 * (1 + tanh(u)) + 0.5 * x * (1 - tanh^2(u)) * k * (1 +
    // 3*a*x^2) where u = k * (x + a * x^3)
    constexpr float k = std::sqrt(2.0f / M_PI);
    constexpr float a = 0.044715f;
    constexpr float three_a = 3.0f * a;
    constexpr float half = 0.5f;
    constexpr float one = 1.0f;

    core::Tensor x = input;
    core::Tensor x2 = ops::mul(x, x);
    core::Tensor x3 = ops::mul(x2, x);
    core::Tensor inner = ops::add(x, ops::mul(a, x3));
    core::Tensor u = ops::mul(k, inner);
    core::Tensor tanh_u = ops::tanh(u);
    core::Tensor tanh_u2 = ops::mul(tanh_u, tanh_u);
    core::Tensor one_minus_tanh2 = ops::sub(one, tanh_u2);
    core::Tensor du_dx = ops::mul(k, ops::add(one, ops::mul(three_a, x2)));
    core::Tensor term1 = ops::mul(half, ops::add(one, tanh_u));
    core::Tensor term2 =
        ops::mul(half, ops::mul(x, ops::mul(one_minus_tanh2, du_dx)));
    core::Tensor gelu_grad = ops::add(term1, term2);
    grad_input = ops::mul(grad_output, gelu_grad);
  }
  return {grad_input};
}

} // namespace quasai::autograd
