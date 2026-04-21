#include "quasai/ops/tensor_ops.hpp"
#include <cmath>

namespace quasai {

Tensor neg(const Tensor &a) {
  const TensorImpl impl_a = a.get_impl();
  Tensor result = Tensor::empty(impl_a.shape, impl_a.dtype, impl_a.device);

  const size_t num_elements = total_size(impl_a.shape);
  float *data_a = static_cast<float *>(impl_a.buffer->raw_data());
  const TensorImpl impl_result = result.get_impl();
  float *data_result = static_cast<float *>(impl_result.buffer->raw_data());

  for (size_t i = 0; i < num_elements; ++i) {
    data_result[i] = -data_a[i];
  }

  return result;
}

Tensor relu(const Tensor &a) {
  const TensorImpl impl_a = a.get_impl();
  Tensor result = Tensor::empty(impl_a.shape, impl_a.dtype, impl_a.device);

  const size_t num_elements = total_size(impl_a.shape);
  float *data_a = static_cast<float *>(impl_a.buffer->raw_data());
  const TensorImpl impl_result = result.get_impl();
  float *data_result = static_cast<float *>(impl_result.buffer->raw_data());

  for (size_t i = 0; i < num_elements; ++i) {
    data_result[i] = std::max(0.0f, data_a[i]);
  }

  return result;
}

Tensor sigmoid(const Tensor &a) {
  const TensorImpl impl_a = a.get_impl();
  Tensor result = Tensor::empty(impl_a.shape, impl_a.dtype, impl_a.device);

  const size_t num_elements = total_size(impl_a.shape);
  float *data_a = static_cast<float *>(impl_a.buffer->raw_data());
  const TensorImpl impl_result = result.get_impl();
  float *data_result = static_cast<float *>(impl_result.buffer->raw_data());

  for (size_t i = 0; i < num_elements; ++i) {
    data_result[i] = 1.0f / (1.0f + std::exp(-data_a[i]));
  }

  return result;
}

Tensor tanh(const Tensor &a) {
  const TensorImpl impl_a = a.get_impl();
  Tensor result = Tensor::empty(impl_a.shape, impl_a.dtype, impl_a.device);

  const size_t num_elements = total_size(impl_a.shape);
  float *data_a = static_cast<float *>(impl_a.buffer->raw_data());
  const TensorImpl impl_result = result.get_impl();
  float *data_result = static_cast<float *>(impl_result.buffer->raw_data());

  for (size_t i = 0; i < num_elements; ++i) {
    data_result[i] = std::tanh(data_a[i]);
  }

  return result;
}

} // namespace quasai
