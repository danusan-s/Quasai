#include "quasai/ops/tensor_ops.hpp"

namespace quasai {

Tensor sum(const Tensor &a) {
  const TensorImpl impl_a = a.get_impl();
  // Scalar result so empty shape
  Tensor result = Tensor::empty(Shape{}, impl_a.dtype, impl_a.device);

  const size_t num_elements = total_size(impl_a.shape);
  float *data_a = static_cast<float *>(impl_a.buffer->raw_data());
  const TensorImpl impl_result = result.get_impl();
  float *data_result = static_cast<float *>(impl_result.buffer->raw_data());

  float sum = 0.0f;
  for (size_t i = 0; i < num_elements; ++i) {
    sum += data_a[i];
  }
  data_result[0] = sum;

  return result;
}

}; // namespace quasai
