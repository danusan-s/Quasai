#include "quasai/autograd/metadata.hpp"
#include "quasai/core/shape.hpp"
#include "quasai/ops/tensor_ops.hpp"

namespace quasai {

Tensor binary_operation(const Tensor &a, const Tensor &b,
                        std::function<Function *()> grad_fn_constructor,
                        std::function<float(float, float)> op) {
  Shape out_shape = broadcast_shape(a.shape(), b.shape());

  Tensor result = Tensor::empty(out_shape, a.dtype(), a.device());

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

  const size_t num_elements = total_size(out_shape);

  const float *data_a = a.data<float>();
  const float *data_b = b.data<float>();
  float *data_result = result.data<float>();

  std::size_t ndim_a = a.shape().dimensions();
  std::size_t ndim_b = b.shape().dimensions();

  // Naive implementation assuming float on cpu
  for (size_t i = 0; i < num_elements; ++i) {
    Index idx = unravel_index(i, out_shape);
    Index idx_a = get_broadcast_index(idx, a.shape());
    Index idx_b = get_broadcast_index(idx, b.shape());

    data_result[i] = op(data_a[ravel_index(idx_a, a.shape())],
                        data_b[ravel_index(idx_b, b.shape())]);
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
