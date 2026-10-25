#include "quasai/ops/tensor_ops.hpp"

#include "quasai/autograd/metadata.hpp"
#include "quasai/autograd/unary_func.hpp"
#include "quasai/ops/cpu_kernel.hpp"

#include <cassert>
#include <cmath>

namespace quasai::ops {

inline void add_unary_gradient(
    const core::Tensor &a, core::Tensor &result,
    std::function<std::unique_ptr<autograd::Function>()> grad_fn_constructor) {

  if (autograd::tensor_requires_grad(a)) {
    auto grad_fn = grad_fn_constructor();
    if (!grad_fn) {
      throw std::runtime_error("Gradient function constructor returned nullptr "
                               "or not implemented for this operation");
    }
    grad_fn->inputs_ = {a};

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

core::Tensor exp(const core::Tensor &a) {
  core::Tensor result = core::Tensor::empty(a.shape(), a.dtype(), a.device());
  add_unary_gradient(
      a, result, []() { return std::make_unique<autograd::ExpFunction>(); });
  dispatch_by_dtype(a.dtype(), [&]<typename T>() {
    do_unary_op<T>(a, result, [](auto x) { return std::exp(x); });
  });
  return result;
}

core::Tensor log(const core::Tensor &a, float base) {
  if (base <= 0.0f || base == 1.0f) {
    throw std::runtime_error("Logarithm base must be positive and not equal 1");
  }

  core::Tensor result = core::Tensor::empty(a.shape(), a.dtype(), a.device());
  const float log_base = std::log(base);

  add_unary_gradient(a, result, [base]() {
    return std::make_unique<autograd::LogFunction>(base);
  });

  dispatch_by_dtype(a.dtype(), [&]<typename T>() {
    do_unary_op<T>(a, result, [log_base](auto x) {
      return std::log(x + core::LOG_EPSILON) / log_base;
    });
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

core::Tensor gelu(const core::Tensor &a) {
  core::Tensor result = core::Tensor::empty(a.shape(), a.dtype(), a.device());
  add_unary_gradient(
      a, result, []() { return std::make_unique<autograd::GeluFunction>(); });

  if (a.is_contiguous() && result.is_contiguous()) {
    dispatch_by_dtype(a.dtype(), [&]<typename T>() {
      if constexpr (std::is_floating_point_v<T>) {
        do_gelu<T>(a, result);
      } else {
        // Fallback for non-floating types (shouldn't happen in practice)
        do_unary_op<T>(a, result, [](auto x) {
          using U = decltype(x);
          constexpr U kAlpha = U(0.044715);
          constexpr U kHalf = U(0.5);
          constexpr U kOne = U(1.0);
          // sqrt(2/pi) using float-precision PI for consistency
          constexpr U kBeta = U(0.7978845834732055757);
          U x3 = x * x * x;
          U inner = x + kAlpha * x3;
          U tanh_val = std::tanh(kBeta * inner);
          return kHalf * x * (kOne + tanh_val);
        });
      }
    });
  } else {
    // Fallback to decomposed implementation for non-contiguous tensors
    core::Tensor u = ops::mul(std::sqrt(2.0 / core::PI),
                              ops::add(a, ops::mul(0.044715, ops::pow(a, 3))));
    result = ops::mul(0.5, ops::mul(a, ops::add(1.0, ops::tanh(u))));
  }

  return result;
}

} // namespace quasai::ops
