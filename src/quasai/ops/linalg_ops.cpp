#include "quasai/ops/tensor_ops.hpp"

namespace quasai {

Tensor matmul(const Tensor &a, const Tensor &b) {
  const TensorImpl impl_a = a.get_impl();
  const TensorImpl impl_b = b.get_impl();

  if (impl_a.shape.dimensions() != 2 || impl_b.shape.dimensions() != 2) {
    throw std::runtime_error("matmul requires 2D tensors");
  }
  if (impl_a.shape[1] != impl_b.shape[0]) {
    throw std::runtime_error("Inner dimensions must match for matmul");
  }

  Shape result_shape{impl_a.shape[0], impl_b.shape[1]};
  Tensor result = Tensor::empty(result_shape, impl_a.dtype, impl_a.device);

  const size_t M = impl_a.shape[0];
  const size_t K = impl_a.shape[1];
  const size_t N = impl_b.shape[1];

  float *data_a = static_cast<float *>(impl_a.buffer->raw_data());
  float *data_b = static_cast<float *>(impl_b.buffer->raw_data());
  const TensorImpl impl_result = result.get_impl();
  float *data_result = static_cast<float *>(impl_result.buffer->raw_data());

  // Naive matrix multiplication
  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      float sum = 0.0f;
      for (size_t k = 0; k < K; ++k) {
        sum += data_a[i * K + k] * data_b[k * N + j];
      }
      data_result[i * N + j] = sum;
    }
  }

  return result;
}

// No new buffer allocation
// just create a new view with swapped shape and strides
Tensor transpose(const Tensor &a) {
  TensorImpl impl_a = a.get_impl();

  if (impl_a.shape.dimensions() != 2) {
    throw std::runtime_error("transpose requires a 2D tensor");
  }

  impl_a.shape = Shape{impl_a.shape[1], impl_a.shape[0]};
  impl_a.strides = Strides{impl_a.strides[1], impl_a.strides[0]};

  return Tensor::from_impl(impl_a);
}

} // namespace quasai
