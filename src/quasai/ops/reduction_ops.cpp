#include "quasai/core/shape.hpp"
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

Tensor sum_to_shape(const Tensor &a, const Shape &target) {
  Shape a_shape = a.shape();
  size_t ndim_a = a_shape.dimensions();
  size_t ndim_t = target.dimensions();

  if (ndim_t > ndim_a) {
    throw std::runtime_error("Target shape cannot have more dimensions than "
                             "input tensor, got input" +
                             a_shape.to_string() + " and target " +
                             target.to_string());
  }

  // pad target shape
  Index padded(ndim_a, 1);
  for (size_t i = 0; i < ndim_t; ++i) {
    padded[ndim_a - ndim_t + i] = target[i];
  }

  Tensor out = Tensor::zeros(target, a.dtype(), a.device());

  auto *a_data = a.data<float>();
  auto *out_data = out.data<float>();

  for (size_t i = 0; i < total_size(a_shape); ++i) {
    Index idx = unravel_index(i, a_shape);
    Index out_idx = get_broadcast_index(idx, target);
    out_data[ravel_index(out_idx, target)] += a_data[i];
  }

  return out;
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
