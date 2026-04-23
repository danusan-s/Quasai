#include "quasai/autograd/metadata.hpp"
#include "quasai/ops/tensor_ops.hpp"
#include <cmath>

namespace quasai {

Tensor unary_operation(const Tensor &a,
                       std::function<Function *()> grad_fn_constructor,
                       std::function<float(float)> op) {
  Tensor result = Tensor::empty(a.shape(), a.dtype(), a.device());

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

  const size_t num_elements = total_size(a.shape());
  const float *data_a = a.data<float>();
  float *data_result = result.data<float>();

  for (size_t i = 0; i < num_elements; ++i) {
    data_result[i] = op(data_a[i]);
  }

  return result;
}

Tensor neg(const Tensor &a) {
  return unary_operation(
      a, []() { return new NegFunction(); }, [](float x) { return -x; });
}

Tensor relu(const Tensor &a) {
  return unary_operation(
      a, []() { return new ReluFunction(); },
      [](float x) { return x > 0.0f ? x : 0.0f; });
}

Tensor step(const Tensor &a) {
  return unary_operation(
      a, []() { return nullptr; },
      [](float x) { return x > 0.0f ? 1.0f : 0.0f; });
}

Tensor sigmoid(const Tensor &a) {
  return unary_operation(
      a, []() { return new SigmoidFunction(); },
      [](float x) { return 1.0f / (1.0f + std::exp(-x)); });
}

Tensor tanh(const Tensor &a) {
  return unary_operation(
      a, []() { return new TanhFunction(); },
      [](float x) { return std::tanh(x); });
}

} // namespace quasai
