#include "quasai/autograd/metadata.hpp"
#include "quasai/core/shape.hpp"
#include "quasai/ops/tensor_ops.hpp"

#define DISPATCH_BINARY_OP(a, b, result, OP)                                   \
  switch (a.dtype()) {                                                         \
    case DType::FLOAT32:                                                       \
      do_binary_op<float>(a, b, result, OP);                                   \
      break;                                                                   \
    case DType::FLOAT64:                                                       \
      do_binary_op<double>(a, b, result, OP);                                  \
      break;                                                                   \
    case DType::INT32:                                                         \
      do_binary_op<int32_t>(a, b, result, OP);                                 \
      break;                                                                   \
    case DType::INT64:                                                         \
      do_binary_op<int64_t>(a, b, result, OP);                                 \
      break;                                                                   \
    default:                                                                   \
      throw std::runtime_error("Unsupported data type for binary operation");  \
  }

namespace quasai {

static Tensor create_result_tensor(const Tensor &a, const Tensor &b) {
  if (a.dtype() != b.dtype()) {
    throw std::runtime_error(
        "Data types of input tensors must match for binary operations");
  }
  Shape result_shape = broadcast_shape(a.shape(), b.shape());
  DType result_dtype = a.dtype();
  return Tensor::empty(result_shape, result_dtype, a.device());
}

void add_binary_gradient(const Tensor &a, const Tensor &b, Tensor &result,
                         std::function<Function *()> grad_fn_constructor) {

  std::shared_ptr<AutoGradMeta> meta_a = a.autograd_meta();
  std::shared_ptr<AutoGradMeta> meta_b = b.autograd_meta();

  if ((meta_a && meta_a->requires_grad) || (meta_b && meta_b->requires_grad)) {
    Function *grad_fn = grad_fn_constructor();
    if (!grad_fn) {
      throw std::runtime_error("Gradient function constructor returned nullptr "
                               "or not implemented for this operation");
    }
    grad_fn->inputs = {a, b};
    result.requires_grad(true);
    result.set_grad_fn(std::unique_ptr<Function>(grad_fn));
  }
}

Tensor add(const Tensor &a, const Tensor &b) {
  Tensor result = create_result_tensor(a, b);
  add_binary_gradient(a, b, result, []() { return new AddFunction(); });
  DISPATCH_BINARY_OP(a, b, result, [](auto x, auto y) { return x + y; });
  return result;
}

Tensor sub(const Tensor &a, const Tensor &b) {
  Tensor result = create_result_tensor(a, b);
  add_binary_gradient(a, b, result, []() { return new SubFunction(); });
  DISPATCH_BINARY_OP(a, b, result, [](auto x, auto y) { return x - y; });
  return result;
}

Tensor mul(const Tensor &a, const Tensor &b) {
  Tensor result = create_result_tensor(a, b);
  add_binary_gradient(a, b, result, []() { return new MulFunction(); });
  DISPATCH_BINARY_OP(a, b, result, [](auto x, auto y) { return x * y; });
  return result;
}

Tensor div(const Tensor &a, const Tensor &b) {
  Tensor result = create_result_tensor(a, b);
  add_binary_gradient(a, b, result, []() { return new DivFunction(); });
  DISPATCH_BINARY_OP(a, b, result, [](auto x, auto y) { return x / y; });
  return result;
}

} // namespace quasai
