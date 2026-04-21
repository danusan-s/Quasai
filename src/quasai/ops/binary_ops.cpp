#include "quasai/ops/tensor_ops.hpp"

namespace quasai {

Tensor add(const Tensor &a, const Tensor &b) {
  const TensorImpl impl_a = a.get_impl();
  const TensorImpl impl_b = b.get_impl();
  if (impl_a.shape != impl_b.shape) {
    throw std::runtime_error("Shape mismatch for addition");
  }

  Tensor result = Tensor::empty(impl_a.shape, impl_a.dtype, impl_a.device);

  const size_t num_elements = total_size(impl_a.shape);

  const float *data_a = a.data<float>();
  const float *data_b = b.data<float>();
  const TensorImpl impl_result = result.get_impl();
  float *data_result = result.data<float>();

  // Naive implementation assuming float on cpu
  for (size_t i = 0; i < num_elements; ++i) {
    data_result[i] = data_a[i] + data_b[i];
  }

  return result;
}

Tensor sub(const Tensor &a, const Tensor &b) {
  const TensorImpl impl_a = a.get_impl();
  const TensorImpl impl_b = b.get_impl();
  if (impl_a.shape != impl_b.shape) {
    throw std::runtime_error("Shape mismatch for subtraction");
  }

  Tensor result = Tensor::empty(impl_a.shape, impl_a.dtype, impl_a.device);

  const size_t num_elements = total_size(impl_a.shape);

  const float *data_a = a.data<float>();
  const float *data_b = b.data<float>();
  const TensorImpl impl_result = result.get_impl();
  float *data_result = result.data<float>();

  // Naive implementation assuming float on cpu
  for (size_t i = 0; i < num_elements; ++i) {
    data_result[i] = data_a[i] - data_b[i];
  }

  return result;
}

Tensor mul(const Tensor &a, const Tensor &b) {
  const TensorImpl impl_a = a.get_impl();
  const TensorImpl impl_b = b.get_impl();
  if (impl_a.shape != impl_b.shape) {
    throw std::runtime_error("Shape mismatch for multiplication");
  }

  Tensor result = Tensor::empty(impl_a.shape, impl_a.dtype, impl_a.device);

  const size_t num_elements = total_size(impl_a.shape);

  const float *data_a = a.data<float>();
  const float *data_b = b.data<float>();
  const TensorImpl impl_result = result.get_impl();
  float *data_result = result.data<float>();

  // Naive implementation assuming float on cpu
  for (size_t i = 0; i < num_elements; ++i) {
    data_result[i] = data_a[i] * data_b[i];
  }

  return result;
}

Tensor div(const Tensor &a, const Tensor &b) {
  const TensorImpl impl_a = a.get_impl();
  const TensorImpl impl_b = b.get_impl();
  if (impl_a.shape != impl_b.shape) {
    throw std::runtime_error("Shape mismatch for division");
  }

  Tensor result = Tensor::empty(impl_a.shape, impl_a.dtype, impl_a.device);

  const size_t num_elements = total_size(impl_a.shape);

  const float *data_a = a.data<float>();
  const float *data_b = b.data<float>();
  const TensorImpl impl_result = result.get_impl();
  float *data_result = result.data<float>();

  // Naive implementation assuming float on cpu
  for (size_t i = 0; i < num_elements; ++i) {
    if (data_b[i] == 0.0f) {
      throw std::runtime_error("Division by zero");
    }
    data_result[i] = data_a[i] / data_b[i];
  }

  return result;
}

} // namespace quasai
