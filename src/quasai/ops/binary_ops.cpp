#include "quasai/autograd/binary_func.hpp"
#include "quasai/autograd/metadata.hpp"
#include "quasai/core/shape.hpp"
#include "quasai/ops/tensor_ops.hpp"

#define DISPATCH_BINARY_OP(a, b, result, OP)                                   \
  switch (a.dtype()) {                                                         \
    case core::DType::FLOAT32:                                                 \
      do_binary_op<float>(a, b, result, OP);                                   \
      break;                                                                   \
    case core::DType::FLOAT64:                                                 \
      do_binary_op<double>(a, b, result, OP);                                  \
      break;                                                                   \
    case core::DType::INT32:                                                   \
      do_binary_op<int32_t>(a, b, result, OP);                                 \
      break;                                                                   \
    case core::DType::INT64:                                                   \
      do_binary_op<int64_t>(a, b, result, OP);                                 \
      break;                                                                   \
    default:                                                                   \
      throw std::runtime_error("Unsupported data type for binary operation");  \
  }

namespace quasai::ops {

static core::Tensor create_result_tensor(const core::Tensor &a,
                                         const core::Tensor &b) {
  if (a.dtype() != b.dtype()) {
    throw std::runtime_error(
        "Data types of input tensors must match for binary operations");
  }
  core::Shape result_shape = broadcast_shape(a.shape(), b.shape());
  core::DType result_dtype = a.dtype();
  return core::Tensor::empty(result_shape, result_dtype, a.device());
}

void add_binary_gradient(
    const core::Tensor &a, const core::Tensor &b, core::Tensor &result,
    std::function<autograd::Function *()> grad_fn_constructor) {

  std::shared_ptr<autograd::AutoGradMeta> meta_a = a.autograd_meta();
  std::shared_ptr<autograd::AutoGradMeta> meta_b = b.autograd_meta();

  if ((meta_a && meta_a->requires_grad) || (meta_b && meta_b->requires_grad)) {
    autograd::Function *grad_fn = grad_fn_constructor();
    if (!grad_fn) {
      throw std::runtime_error("Gradient function constructor returned nullptr "
                               "or not implemented for this operation");
    }
    grad_fn->inputs = {a, b};
    result.requires_grad(true);
    result.set_grad_fn(std::unique_ptr<autograd::Function>(grad_fn));
  }
}

core::Tensor add(const core::Tensor &a, const core::Tensor &b) {
  core::Tensor result = create_result_tensor(a, b);
  add_binary_gradient(a, b, result,
                      []() { return new autograd::AddFunction(); });
  DISPATCH_BINARY_OP(a, b, result, [](auto x, auto y) { return x + y; });
  return result;
}

core::Tensor sub(const core::Tensor &a, const core::Tensor &b) {
  core::Tensor result = create_result_tensor(a, b);
  add_binary_gradient(a, b, result,
                      []() { return new autograd::SubFunction(); });
  DISPATCH_BINARY_OP(a, b, result, [](auto x, auto y) { return x - y; });
  return result;
}

core::Tensor mul(const core::Tensor &a, const core::Tensor &b) {
  core::Tensor result = create_result_tensor(a, b);
  add_binary_gradient(a, b, result,
                      []() { return new autograd::MulFunction(); });
  DISPATCH_BINARY_OP(a, b, result, [](auto x, auto y) { return x * y; });
  return result;
}

core::Tensor div(const core::Tensor &a, const core::Tensor &b) {
  core::Tensor result = create_result_tensor(a, b);
  add_binary_gradient(a, b, result,
                      []() { return new autograd::DivFunction(); });
  DISPATCH_BINARY_OP(a, b, result, [](auto x, auto y) { return x / y; });
  return result;
}

} // namespace quasai::ops
