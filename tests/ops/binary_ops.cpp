#include "quasai/ops/tensor_ops.hpp"
#include <gtest/gtest.h>

TEST(TensorOpsTest, Add) {
  std::vector<float> data_a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<float> data_b = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
  quasai::Shape shape{3, 2};
  quasai::Tensor tensor_a =
      quasai::Tensor::from_data(data_a.data(), shape, quasai::DType::FLOAT32);
  quasai::Tensor tensor_b =
      quasai::Tensor::from_data(data_b.data(), shape, quasai::DType::FLOAT32);

  quasai::Tensor result = quasai::add(tensor_a, tensor_b);

  const auto impl = result.get_impl_copy();
  const auto ref_impl = tensor_a.get_impl_copy();

  EXPECT_EQ(impl.shape, ref_impl.shape);
  EXPECT_EQ(impl.dtype, ref_impl.dtype);
  EXPECT_EQ(impl.device.type, ref_impl.device.type);

  float *result_data = result.data<float>();

  EXPECT_FLOAT_EQ(result_data[0], 8.0f);
  EXPECT_FLOAT_EQ(result_data[1], 10.0f);
  EXPECT_FLOAT_EQ(result_data[2], 12.0f);
  EXPECT_FLOAT_EQ(result_data[3], 14.0f);
  EXPECT_FLOAT_EQ(result_data[4], 16.0f);
  EXPECT_FLOAT_EQ(result_data[5], 18.0f);
}

TEST(TensorOpsTest, AddShapeMismatch) {
  quasai::Tensor tensor_a = quasai::Tensor::zeros(quasai::Shape{2, 3});
  quasai::Tensor tensor_b = quasai::Tensor::zeros(quasai::Shape{3, 2});

  EXPECT_THROW(quasai::add(tensor_a, tensor_b), std::runtime_error);
}

TEST(TensorOpsTest, Subtract) {
  std::vector<float> data_a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<float> data_b = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
  quasai::Shape shape{3, 2};
  quasai::Tensor tensor_a =
      quasai::Tensor::from_data(data_a.data(), shape, quasai::DType::FLOAT32);
  quasai::Tensor tensor_b =
      quasai::Tensor::from_data(data_b.data(), shape, quasai::DType::FLOAT32);

  quasai::Tensor result = quasai::sub(tensor_a, tensor_b);

  const auto impl = result.get_impl_copy();
  const auto ref_impl = tensor_a.get_impl_copy();

  EXPECT_EQ(impl.shape, ref_impl.shape);
  EXPECT_EQ(impl.dtype, ref_impl.dtype);
  EXPECT_EQ(impl.device.type, ref_impl.device.type);

  float *result_data = result.data<float>();

  EXPECT_FLOAT_EQ(result_data[0], -6.0f);
  EXPECT_FLOAT_EQ(result_data[1], -6.0f);
  EXPECT_FLOAT_EQ(result_data[2], -6.0f);
  EXPECT_FLOAT_EQ(result_data[3], -6.0f);
  EXPECT_FLOAT_EQ(result_data[4], -6.0f);
  EXPECT_FLOAT_EQ(result_data[5], -6.0f);
}

TEST(TensorOpsTest, SubtractShapeMismatch) {
  quasai::Tensor tensor_a = quasai::Tensor::zeros(quasai::Shape{2, 3});
  quasai::Tensor tensor_b = quasai::Tensor::zeros(quasai::Shape{3, 2});

  EXPECT_THROW(quasai::sub(tensor_a, tensor_b), std::runtime_error);
}

TEST(TensorOpsTest, Multiply) {
  std::vector<float> data_a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<float> data_b = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
  quasai::Shape shape{3, 2};
  quasai::Tensor tensor_a =
      quasai::Tensor::from_data(data_a.data(), shape, quasai::DType::FLOAT32);
  quasai::Tensor tensor_b =
      quasai::Tensor::from_data(data_b.data(), shape, quasai::DType::FLOAT32);

  quasai::Tensor result = quasai::mul(tensor_a, tensor_b);

  const auto impl = result.get_impl_copy();
  const auto ref_impl = tensor_a.get_impl_copy();

  EXPECT_EQ(impl.shape, ref_impl.shape);
  EXPECT_EQ(impl.dtype, ref_impl.dtype);
  EXPECT_EQ(impl.device.type, ref_impl.device.type);

  float *result_data = result.data<float>();

  EXPECT_FLOAT_EQ(result_data[0], 7.0f);
  EXPECT_FLOAT_EQ(result_data[1], 16.0f);
  EXPECT_FLOAT_EQ(result_data[2], 27.0f);
  EXPECT_FLOAT_EQ(result_data[3], 40.0f);
  EXPECT_FLOAT_EQ(result_data[4], 55.0f);
  EXPECT_FLOAT_EQ(result_data[5], 72.0f);
}

TEST(TensorOpsTest, MultiplyShapeMismatch) {
  quasai::Tensor tensor_a = quasai::Tensor::zeros(quasai::Shape{2, 3});
  quasai::Tensor tensor_b = quasai::Tensor::zeros(quasai::Shape{3, 2});

  EXPECT_THROW(quasai::mul(tensor_a, tensor_b), std::runtime_error);
}

TEST(TensorOpsTest, Divide) {
  std::vector<float> data_a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<float> data_b = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
  quasai::Shape shape{3, 2};
  quasai::Tensor tensor_a =
      quasai::Tensor::from_data(data_a.data(), shape, quasai::DType::FLOAT32);
  quasai::Tensor tensor_b =
      quasai::Tensor::from_data(data_b.data(), shape, quasai::DType::FLOAT32);

  quasai::Tensor result = quasai::div(tensor_a, tensor_b);

  const auto impl = result.get_impl_copy();
  const auto ref_impl = tensor_a.get_impl_copy();

  EXPECT_EQ(impl.shape, ref_impl.shape);
  EXPECT_EQ(impl.dtype, ref_impl.dtype);
  EXPECT_EQ(impl.device.type, ref_impl.device.type);

  float *result_data = result.data<float>();

  EXPECT_FLOAT_EQ(result_data[0], 1.0f / 7.0f);
  EXPECT_FLOAT_EQ(result_data[1], 2.0f / 8.0f);
  EXPECT_FLOAT_EQ(result_data[2], 3.0f / 9.0f);
  EXPECT_FLOAT_EQ(result_data[3], 4.0f / 10.0f);
  EXPECT_FLOAT_EQ(result_data[4], 5.0f / 11.0f);
  EXPECT_FLOAT_EQ(result_data[5], 6.0f / 12.0f);
}

TEST(TensorOpsTest, DivideShapeMismatch) {
  quasai::Tensor tensor_a = quasai::Tensor::zeros(quasai::Shape{2, 3});
  quasai::Tensor tensor_b = quasai::Tensor::zeros(quasai::Shape{3, 2});

  EXPECT_THROW(quasai::div(tensor_a, tensor_b), std::runtime_error);
}

TEST(TensorOpsTest, BroadcastLeading) {
  std::vector<float> data_a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<float> data_b = {10.0f, 20.0f};
  quasai::Shape shape_a{3, 2};
  quasai::Shape shape_b{2};
  quasai::Tensor tensor_a =
      quasai::Tensor::from_data(data_a.data(), shape_a, quasai::DType::FLOAT32);
  quasai::Tensor tensor_b =
      quasai::Tensor::from_data(data_b.data(), shape_b, quasai::DType::FLOAT32);

  quasai::Tensor result = quasai::add(tensor_a, tensor_b);

  const auto impl = result.get_impl_copy();
  const auto ref_impl = tensor_a.get_impl_copy();

  EXPECT_EQ(impl.shape, ref_impl.shape);
  EXPECT_EQ(impl.dtype, ref_impl.dtype);
  EXPECT_EQ(impl.device.type, ref_impl.device.type);

  float *result_data = result.data<float>();

  EXPECT_FLOAT_EQ(result_data[0], 11.0f);
  EXPECT_FLOAT_EQ(result_data[1], 22.0f);
  EXPECT_FLOAT_EQ(result_data[2], 13.0f);
  EXPECT_FLOAT_EQ(result_data[3], 24.0f);
  EXPECT_FLOAT_EQ(result_data[4], 15.0f);
  EXPECT_FLOAT_EQ(result_data[5], 26.0f);
}

TEST(TensorOpsTest, BroadcastOnes) {
  std::vector<float> data_a = {1.0f};
  std::vector<float> data_b = {10.0f, 20.0f, 30.0f};
  quasai::Shape shape_a{1};
  quasai::Shape shape_b{3};
  quasai::Tensor tensor_a =
      quasai::Tensor::from_data(data_a.data(), shape_a, quasai::DType::FLOAT32);
  quasai::Tensor tensor_b =
      quasai::Tensor::from_data(data_b.data(), shape_b, quasai::DType::FLOAT32);

  quasai::Tensor result = quasai::add(tensor_a, tensor_b);

  const auto impl = result.get_impl_copy();
  const auto ref_impl = tensor_b.get_impl_copy();

  EXPECT_EQ(impl.shape, ref_impl.shape);
  EXPECT_EQ(impl.dtype, ref_impl.dtype);
  EXPECT_EQ(impl.device.type, ref_impl.device.type);

  float *result_data = result.data<float>();

  EXPECT_FLOAT_EQ(result_data[0], 11.0f);
  EXPECT_FLOAT_EQ(result_data[1], 21.0f);
  EXPECT_FLOAT_EQ(result_data[2], 31.0f);
}
