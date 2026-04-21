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

  float *data_a = static_cast<float *>(impl_a.buffer->raw_data());
  float *data_b = static_cast<float *>(impl_b.buffer->raw_data());
  const TensorImpl impl_result = result.get_impl();
  float *data_result = static_cast<float *>(impl_result.buffer->raw_data());

  // Naive implementation assuming float on cpu
  for (size_t i = 0; i < num_elements; ++i) {
    data_result[i] = data_a[i] + data_b[i];
  }

  return a;
}

Tensor sub(const Tensor &a, const Tensor &b) {
  const TensorImpl impl_a = a.get_impl();
  const TensorImpl impl_b = b.get_impl();
  if (impl_a.shape != impl_b.shape) {
    throw std::runtime_error("Shape mismatch for subtraction");
  }

  Tensor result = Tensor::empty(impl_a.shape, impl_a.dtype, impl_a.device);

  const size_t num_elements = total_size(impl_a.shape);

  float *data_a = static_cast<float *>(impl_a.buffer->raw_data());
  float *data_b = static_cast<float *>(impl_b.buffer->raw_data());
  const TensorImpl impl_result = result.get_impl();
  float *data_result = static_cast<float *>(impl_result.buffer->raw_data());

  // Naive implementation assuming float on cpu
  for (size_t i = 0; i < num_elements; ++i) {
    data_result[i] = data_a[i] - data_b[i];
  }

  return a;
}

Tensor mul(const Tensor &a, const Tensor &b) {
  const TensorImpl impl_a = a.get_impl();
  const TensorImpl impl_b = b.get_impl();
  if (impl_a.shape != impl_b.shape) {
    throw std::runtime_error("Shape mismatch for multiplication");
  }

  Tensor result = Tensor::empty(impl_a.shape, impl_a.dtype, impl_a.device);

  const size_t num_elements = total_size(impl_a.shape);

  float *data_a = static_cast<float *>(impl_a.buffer->raw_data());
  float *data_b = static_cast<float *>(impl_b.buffer->raw_data());
  const TensorImpl impl_result = result.get_impl();
  float *data_result = static_cast<float *>(impl_result.buffer->raw_data());

  // Naive implementation assuming float on cpu
  for (size_t i = 0; i < num_elements; ++i) {
    data_result[i] = data_a[i] * data_b[i];
  }

  return a;
}

Tensor div(const Tensor &a, const Tensor &b) {
  const TensorImpl impl_a = a.get_impl();
  const TensorImpl impl_b = b.get_impl();
  if (impl_a.shape != impl_b.shape) {
    throw std::runtime_error("Shape mismatch for division");
  }

  Tensor result = Tensor::empty(impl_a.shape, impl_a.dtype, impl_a.device);

  const size_t num_elements = total_size(impl_a.shape);

  float *data_a = static_cast<float *>(impl_a.buffer->raw_data());
  float *data_b = static_cast<float *>(impl_b.buffer->raw_data());
  const TensorImpl impl_result = result.get_impl();
  float *data_result = static_cast<float *>(impl_result.buffer->raw_data());

  // Naive implementation assuming float on cpu
  for (size_t i = 0; i < num_elements; ++i) {
    if (data_b[i] == 0.0f) {
      throw std::runtime_error("Division by zero");
    }
    data_result[i] = data_a[i] / data_b[i];
  }

  return a;
}

} // namespace quasai
