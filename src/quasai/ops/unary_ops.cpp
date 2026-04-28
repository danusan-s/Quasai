#include "quasai/autograd/metadata.hpp"
#include "quasai/autograd/unary_func.hpp"
#include "quasai/ops/tensor_ops.hpp"
#include <cmath>

#define DISPATCH_UNARY_OP(a, result, OP)                                       \
  switch (a.dtype()) {                                                         \
    case core::DType::FLOAT32:                                                   \
      do_unary_op<float>(a, result, OP);                                         \
      break;                                                                   \
    case core::DType::FLOAT64:                                                   \
      do_unary_op<double>(a, result, OP);                                          \
      break;                                                                   \
    case core::DType::INT32:                                                    \
      do_unary_op<int32_t>(a, result, OP);                                      \
      break;                                                                   \
    case core::DType::INT64:                                                      \
      do_unary_op<int64_t>(a, result, OP);                                      \
      break;                                                                   \
    default:                                                                   \
      throw std::runtime_error("Unsupported data type for unary operation");   \
  }

namespace quasai::ops {

inline void add_unary_gradient(const core::Tensor &a, core::Tensor &result,
                        std::function<autograd::Function *()> grad_fn_constructor) {
  std::shared_ptr<autograd::AutoGradMeta> meta_a = a.autograd_meta();

  if (meta_a && meta_a->requires_grad) {
    autograd::Function *grad_fn = grad_fn_constructor();
    if (!grad_fn) {
      throw std::runtime_error("Gradient function constructor returned nullptr "
                           "or not implemented for this operation");
    }
    grad_fn->inputs = {a};

    result.requires_grad(true);
    result.set_grad_fn(std::unique_ptr<autograd::Function>(grad_fn));
  }
}

core::Tensor neg(const core::Tensor &a) {
  core::Tensor result = core::Tensor::empty(a.shape(), a.dtype(), a.device());
  add_unary_gradient(a, result, []() { return new autograd::NegFunction(); });
  DISPATCH_UNARY_OP(a, result, [](auto x) { return -x; });
  return result;
}

core::Tensor abs(const core::Tensor &a) {
  core::Tensor result = core::Tensor::empty(a.shape(), a.dtype(), a.device());
  add_unary_gradient(a, result, []() { return new autograd::AbsFunction(); });
  DISPATCH_UNARY_OP(a, result, [](auto x) { return x >= 0 ? x : -x; });
  return result;
}

core::Tensor relu(const core::Tensor &a) {
  core::Tensor result = core::Tensor::empty(a.shape(), a.dtype(), a.device());
  add_unary_gradient(a, result, []() { return new autograd::ReluFunction(); });
  DISPATCH_UNARY_OP(a, result, [](auto x) { return x > 0 ? x : 0; });
  return result;
}

core::Tensor heaviside(const core::Tensor &a) {
  core::Tensor result = core::Tensor::empty(a.shape(), a.dtype(), a.device());
  add_unary_gradient(a, result, []() { return new autograd::HeavisideFunction(); });
  DISPATCH_UNARY_OP(a, result, [](auto x) { return x > 0 ? 1 : 0; });
  return result;
}

core::Tensor signum(const core::Tensor &a) {
  core::Tensor result = core::Tensor::empty(a.shape(), a.dtype(), a.device());
  add_unary_gradient(a, result, []() { return new autograd::SignumFunction(); });
  DISPATCH_UNARY_OP(a, result,
                    [](auto x) { return x > 0 ? 1 : (x < 0 ? -1 : 0); });
  return result;
}

core::Tensor sigmoid(const core::Tensor &a) {
  core::Tensor result = core::Tensor::empty(a.shape(), a.dtype(), a.device());
  add_unary_gradient(a, result, []() { return new autograd::SigmoidFunction(); });
  DISPATCH_UNARY_OP(a, result,
                    [](auto x) { return 1.0 / (1.0 + std::exp(-x)); });
  return result;
}

core::Tensor tanh(const core::Tensor &a) {
  core::Tensor result = core::Tensor::empty(a.shape(), a.dtype(), a.device());
  add_unary_gradient(a, result, []() { return new autograd::TanhFunction(); });
  DISPATCH_UNARY_OP(a, result, [](auto x) { return std::tanh(x); });
  return result;
}

} // namespace quasai::ops
