#include "quasai/autograd/metadata.hpp"
#include "quasai/ops/tensor_ops.hpp"
#include <cmath>

#define DISPATCH_UNARY_OP(a, result, OP)                                       \
  switch (a.dtype()) {                                                         \
    case DType::FLOAT32:                                                       \
      do_unary_op<float>(a, result, OP);                                       \
      break;                                                                   \
    case DType::FLOAT64:                                                       \
      do_unary_op<double>(a, result, OP);                                      \
      break;                                                                   \
    case DType::INT32:                                                         \
      do_unary_op<int32_t>(a, result, OP);                                     \
      break;                                                                   \
    case DType::INT64:                                                         \
      do_unary_op<int64_t>(a, result, OP);                                     \
      break;                                                                   \
    default:                                                                   \
      throw std::runtime_error("Unsupported data type for unary operation");   \
  }

namespace quasai {

void add_unary_gradient(const Tensor &a, Tensor &result,
                        std::function<Function *()> grad_fn_constructor) {
  std::shared_ptr<AutoGradMeta> meta_a = a.autograd_meta();

  if (meta_a && meta_a->requires_grad) {
    Function *grad_fn = grad_fn_constructor();
    if (!grad_fn) {
      throw std::runtime_error("Gradient function constructor returned nullptr "
                               "or not implemented for this operation");
    }
    grad_fn->inputs = {a};

    result.requires_grad(true);
    result.set_grad_fn(std::unique_ptr<Function>(grad_fn));
  }
}

Tensor neg(const Tensor &a) {
  Tensor result = Tensor::empty(a.shape(), a.dtype(), a.device());
  add_unary_gradient(a, result, []() { return new NegFunction(); });
  DISPATCH_UNARY_OP(a, result, [](auto x) { return -x; });
  return result;
}

Tensor relu(const Tensor &a) {
  Tensor result = Tensor::empty(a.shape(), a.dtype(), a.device());
  add_unary_gradient(a, result, []() { return new ReluFunction(); });
  DISPATCH_UNARY_OP(a, result, [](auto x) { return x > 0 ? x : 0; });
  return result;
}

Tensor step(const Tensor &a) {
  Tensor result = Tensor::empty(a.shape(), a.dtype(), a.device());
  DISPATCH_UNARY_OP(a, result, [](auto x) { return x > 0 ? 1 : 0; });
  return result;
}

Tensor sigmoid(const Tensor &a) {
  Tensor result = Tensor::empty(a.shape(), a.dtype(), a.device());
  add_unary_gradient(a, result, []() { return new SigmoidFunction(); });
  DISPATCH_UNARY_OP(a, result,
                    [](auto x) { return 1.0 / (1.0 + std::exp(-x)); });
  return result;
}

Tensor tanh(const Tensor &a) {
  Tensor result = Tensor::empty(a.shape(), a.dtype(), a.device());
  add_unary_gradient(a, result, []() { return new TanhFunction(); });
  DISPATCH_UNARY_OP(a, result, [](auto x) { return std::tanh(x); });
  return result;
}

} // namespace quasai
