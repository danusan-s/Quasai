#include "quasai/ops/tensor_ops.hpp"

namespace quasai {

Tensor sum(const Tensor &a) {
  const TensorImpl impl_a = a.get_impl();
  // Scalar result so empty shape
  Tensor result = Tensor::empty(Shape{}, impl_a.dtype, impl_a.device);

  const size_t num_elements = total_size(impl_a.shape);
  const float *data_a = a.data<float>();
  const TensorImpl impl_result = result.get_impl();
  float *data_result = result.data<float>();

  float sum = 0.0f;
  for (size_t i = 0; i < num_elements; ++i) {
    sum += data_a[i];
  }
  data_result[0] = sum;

  return result;
}

Tensor mean(const Tensor &a) {
  const TensorImpl impl_a = a.get_impl();
  // Scalar result so empty shape
  Tensor result = Tensor::empty(Shape{}, impl_a.dtype, impl_a.device);

  const size_t num_elements = total_size(impl_a.shape);
  const float *data_a = a.data<float>();
  const TensorImpl impl_result = result.get_impl();
  float *data_result = result.data<float>();

  float sum = 0.0f;
  for (size_t i = 0; i < num_elements; ++i) {
    sum += data_a[i];
  }
  data_result[0] = sum / num_elements;

  return result;
}

}; // namespace quasai
