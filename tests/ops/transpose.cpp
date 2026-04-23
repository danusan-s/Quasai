#include "quasai/ops/tensor_ops.hpp"
#include <gtest/gtest.h>

TEST(Transpose, Float32) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  quasai::Shape shape{2, 3};
  quasai::Tensor tensor =
      quasai::Tensor::from_data(data.data(), shape, quasai::DType::FLOAT32);

  quasai::Tensor result = quasai::transpose(tensor);
  const auto impl = result.get_impl_copy();
  const auto ref_impl = tensor.get_impl_copy();

  EXPECT_EQ(impl.shape[0], ref_impl.shape[1]);
  EXPECT_EQ(impl.shape[1], ref_impl.shape[0]);
  EXPECT_EQ(impl.strides[0], ref_impl.strides[1]);
  EXPECT_EQ(impl.strides[1], ref_impl.strides[0]);
  EXPECT_EQ(impl.dtype, ref_impl.dtype);
  EXPECT_EQ(impl.device.type, ref_impl.device.type);
}
