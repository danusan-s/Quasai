#include "quasai/autograd/metadata.hpp"
#include "quasai/ops/tensor_ops.hpp"

namespace quasai {

Tensor binary_operation(const Tensor &a, const Tensor &b,
                        std::function<Function *()> grad_fn_constructor,
                        std::function<float(float, float)> op) {
  if (a.shape() != b.shape()) {
    throw std::runtime_error("Shape mismatch for binary operation");
  }

  Tensor result = Tensor::empty(a.shape(), a.dtype(), a.device());

  std::shared_ptr<AutoGradMeta> meta_a = a.autograd_meta();
  std::shared_ptr<AutoGradMeta> meta_b = b.autograd_meta();

  if (meta_a && meta_a->requires_grad && meta_b && meta_b->requires_grad) {
    Function *grad_fn = grad_fn_constructor();
    if (!grad_fn) {
      throw std::runtime_error("Gradient function constructor returned nullptr "
                               "or not implemented for this operation");
    }
    grad_fn->inputs = {a, b};
    result.requires_grad(true);
    result.set_grad_fn(std::unique_ptr<Function>(grad_fn));
  }

  const size_t num_elements = total_size(a.shape());

  const float *data_a = a.data<float>();
  const float *data_b = b.data<float>();
  float *data_result = result.data<float>();

  // Naive implementation assuming float on cpu
  for (size_t i = 0; i < num_elements; ++i) {
    data_result[i] = op(data_a[i], data_b[i]);
  }

  return result;
}

Tensor add(const Tensor &a, const Tensor &b) {
  return binary_operation(
      a, b, []() { return new AddFunction(); },
      [](float x, float y) { return x + y; });
}

Tensor sub(const Tensor &a, const Tensor &b) {
  return binary_operation(
      a, b, []() { return new SubFunction(); },
      [](float x, float y) { return x - y; });
}

Tensor mul(const Tensor &a, const Tensor &b) {
  return binary_operation(
      a, b, []() { return new MulFunction(); },
      [](float x, float y) { return x * y; });
}

Tensor div(const Tensor &a, const Tensor &b) {
  return binary_operation(
      a, b, []() { return new DivFunction(); },
      [](float x, float y) { return x / y; });
}

} // namespace quasai
