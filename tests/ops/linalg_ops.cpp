#include "quasai/ops/tensor_ops.hpp"
#include <gtest/gtest.h>

TEST(TensorOpsTest, MatrixMultiplication) {
  std::vector<float> data_a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<float> data_b = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
  quasai::Shape shape_a{2, 3};
  quasai::Shape shape_b{3, 2};
  quasai::Tensor tensor_a =
      quasai::Tensor::from_data(data_a.data(), shape_a, quasai::DType::FLOAT32);
  quasai::Tensor tensor_b =
      quasai::Tensor::from_data(data_b.data(), shape_b, quasai::DType::FLOAT32);

  quasai::Tensor result = quasai::matmul(tensor_a, tensor_b);

  const auto impl = result.get_impl();
  const auto ref_impl = tensor_a.get_impl();

  EXPECT_EQ(impl.shape[0], 2);
  EXPECT_EQ(impl.shape[1], 2);
  EXPECT_EQ(impl.dtype, ref_impl.dtype);
  EXPECT_EQ(impl.device.type, ref_impl.device.type);

  float *result_data = result.data<float>();
  EXPECT_FLOAT_EQ(result_data[0], 58.0f);
  EXPECT_FLOAT_EQ(result_data[1], 64.0f);
  EXPECT_FLOAT_EQ(result_data[2], 139.0f);
  EXPECT_FLOAT_EQ(result_data[3], 154.0f);
}

TEST(TensorOpsTest, MatMulShapeMismatch) {
  quasai::Tensor tensor_a = quasai::Tensor::zeros(quasai::Shape{2, 3});
  quasai::Tensor tensor_b = quasai::Tensor::zeros(quasai::Shape{4, 2});

  EXPECT_THROW(quasai::matmul(tensor_a, tensor_b), std::runtime_error);
}

TEST(TensorOpsTest, Transpose) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  quasai::Shape shape{2, 3};
  quasai::Tensor tensor =
      quasai::Tensor::from_data(data.data(), shape, quasai::DType::FLOAT32);

  quasai::Tensor result = quasai::transpose(tensor);

  const auto impl = result.get_impl();
  const auto ref_impl = tensor.get_impl();

  EXPECT_EQ(impl.shape[0], ref_impl.shape[1]);
  EXPECT_EQ(impl.shape[1], ref_impl.shape[0]);
  EXPECT_EQ(impl.strides[0], ref_impl.strides[1]);
  EXPECT_EQ(impl.strides[1], ref_impl.strides[0]);
  EXPECT_EQ(impl.dtype, ref_impl.dtype);
  EXPECT_EQ(impl.device.type, ref_impl.device.type);
}
