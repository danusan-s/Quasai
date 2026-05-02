#include "quasai/autograd/metadata.hpp"
#include "quasai/autograd/unary_func.hpp"
#include "quasai/ops/cpu_kernel.hpp"
#include "quasai/ops/tensor_ops.hpp"
#include <cmath>

namespace quasai::ops {

inline void add_unary_gradient(
    const core::Tensor &a, core::Tensor &result,
    std::function<std::unique_ptr<autograd::Function>()> grad_fn_constructor) {
  std::shared_ptr<autograd::AutoGradMeta> meta_a = a.autograd_meta();

  if (meta_a && meta_a->requires_grad) {
    auto grad_fn = grad_fn_constructor();
    if (!grad_fn) {
      throw std::runtime_error("Gradient function constructor returned nullptr "
                               "or not implemented for this operation");
    }
    grad_fn->inputs = {a};

    result.requires_grad(true);
    result.set_grad_fn(std::move(grad_fn));
  }
}

core::Tensor neg(const core::Tensor &a) {
  core::Tensor result = core::Tensor::empty(a.shape(), a.dtype(), a.device());
  add_unary_gradient(
      a, result, []() { return std::make_unique<autograd::NegFunction>(); });
  dispatch_by_dtype(a.dtype(), [&]<typename T>() {
    do_unary_op<T>(a, result, [](auto x) { return -x; });
  });
  return result;
}

core::Tensor abs(const core::Tensor &a) {
  core::Tensor result = core::Tensor::empty(a.shape(), a.dtype(), a.device());
  add_unary_gradient(
      a, result, []() { return std::make_unique<autograd::AbsFunction>(); });
  dispatch_by_dtype(a.dtype(), [&]<typename T>() {
    do_unary_op<T>(a, result, [](auto x) { return x >= 0 ? x : -x; });
  });
  return result;
}

core::Tensor relu(const core::Tensor &a) {
  core::Tensor result = core::Tensor::empty(a.shape(), a.dtype(), a.device());
  add_unary_gradient(
      a, result, []() { return std::make_unique<autograd::ReluFunction>(); });
  dispatch_by_dtype(a.dtype(), [&]<typename T>() {
    do_unary_op<T>(a, result, [](auto x) { return x > 0 ? x : 0; });
  });
  return result;
}

core::Tensor heaviside(const core::Tensor &a) {
  core::Tensor result = core::Tensor::empty(a.shape(), a.dtype(), a.device());
  add_unary_gradient(a, result, []() {
    return std::make_unique<autograd::HeavisideFunction>();
  });
  dispatch_by_dtype(a.dtype(), [&]<typename T>() {
    do_unary_op<T>(a, result, [](auto x) { return x > 0 ? 1 : 0; });
  });
  return result;
}

core::Tensor signum(const core::Tensor &a) {
  core::Tensor result = core::Tensor::empty(a.shape(), a.dtype(), a.device());
  add_unary_gradient(
      a, result, []() { return std::make_unique<autograd::SignumFunction>(); });
  dispatch_by_dtype(a.dtype(), [&]<typename T>() {
    do_unary_op<T>(a, result,
                   [](auto x) { return x > 0 ? 1 : (x < 0 ? -1 : 0); });
  });
  return result;
}

core::Tensor sigmoid(const core::Tensor &a) {
  core::Tensor result = core::Tensor::empty(a.shape(), a.dtype(), a.device());
  add_unary_gradient(a, result, []() {
    return std::make_unique<autograd::SigmoidFunction>();
  });
  dispatch_by_dtype(a.dtype(), [&]<typename T>() {
    do_unary_op<T>(a, result,
                   [](auto x) { return 1.0 / (1.0 + std::exp(-x)); });
  });
  return result;
}

core::Tensor tanh(const core::Tensor &a) {
  core::Tensor result = core::Tensor::empty(a.shape(), a.dtype(), a.device());
  add_unary_gradient(
      a, result, []() { return std::make_unique<autograd::TanhFunction>(); });
  dispatch_by_dtype(a.dtype(), [&]<typename T>() {
    do_unary_op<T>(a, result, [](auto x) { return std::tanh(x); });
  });
  return result;
}

core::Tensor pow(const core::Tensor &a, float exponent) {
  core::Tensor result = core::Tensor::empty(a.shape(), a.dtype(), a.device());
  add_unary_gradient(a, result, [exponent]() {
    return std::make_unique<autograd::PowFunction>(exponent);
  });
  dispatch_by_dtype(a.dtype(), [&]<typename T>() {
    do_unary_op<T>(a, result,
                   [exponent](auto x) { return std::pow(x, exponent); });
  });
  return result;
}

} // namespace quasai::ops
