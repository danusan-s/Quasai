#include "quasai/ops/tensor_ops.hpp"

namespace quasai {

Tensor sum(const Tensor &a) {
  // Scalar result so empty shape
  Tensor result = Tensor::empty(Shape{}, a.dtype(), a.device());

  const size_t num_elements = total_size(a.shape());
  const float *data_a = a.data<float>();
  float *data_result = result.data<float>();

  float sum = 0.0f;
  for (size_t i = 0; i < num_elements; ++i) {
    sum += data_a[i];
  }
  data_result[0] = sum;

  return result;
}

Tensor mean(const Tensor &a) {
  // Scalar result so empty shape
  Tensor result = Tensor::empty(Shape{}, a.dtype(), a.device());

  const size_t num_elements = total_size(a.shape());
  const float *data_a = a.data<float>();
  float *data_result = result.data<float>();

  float sum = 0.0f;
  for (size_t i = 0; i < num_elements; ++i) {
    sum += data_a[i];
  }
  data_result[0] = sum / num_elements;

  return result;
}

}; // namespace quasai
