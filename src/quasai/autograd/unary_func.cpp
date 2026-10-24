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
    const core::Tensor k = core::Tensor::from_scalar(
        std::sqrt(2.0 / core::PI), input.dtype(), input.device());
    const core::Tensor l =
        core::Tensor::from_scalar(0.044715, input.dtype(), input.device());
    const core::Tensor one =
        core::Tensor::from_scalar(1.0, input.dtype(), input.device());
    const core::Tensor half =
        core::Tensor::from_scalar(0.5, input.dtype(), input.device());
    const core::Tensor three_l = core::Tensor::from_scalar(
        3.0 * 0.044715, input.dtype(), input.device());

    core::Tensor x2 = ops::pow(input, 2);
    core::Tensor u =
        ops::mul(k, ops::add(input, ops::mul(l, ops::pow(input, 3))));

    core::Tensor t = ops::tanh(u);

    // GELU'(x) = 0.5 * (1 + tanh(u))
    //          + 0.5 * x * (1 - tanh(u)^2) * k * (1 + 3*l*x^2)
    core::Tensor first_term = ops::mul(half, ops::add(one, t));

    core::Tensor tanh_derivative = ops::sub(one, ops::pow(t, 2));

    core::Tensor du_dx = ops::mul(k, ops::add(one, ops::mul(three_l, x2)));

    core::Tensor second_term =
        ops::mul(half, ops::mul(input, ops::mul(tanh_derivative, du_dx)));

    core::Tensor derivative = ops::add(first_term, second_term);

    grad_input = ops::mul(grad_output, derivative);
  }

  return {grad_input};
}

} // namespace quasai::autograd
