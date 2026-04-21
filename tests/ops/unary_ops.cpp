#include "quasai/ops/tensor_ops.hpp"
#include <cmath>
#include <gtest/gtest.h>

TEST(TensorOpsTest, Negate) {
  std::vector<float> data = {1.0f, -2.0f, 3.0f, -4.0f};
  quasai::Shape shape{2, 2};
  quasai::Tensor tensor =
      quasai::Tensor::from_data(data.data(), shape, quasai::DType::FLOAT32);

  quasai::Tensor result = quasai::neg(tensor);

  const auto impl = result.get_impl();
  const auto ref_impl = tensor.get_impl();

  EXPECT_EQ(impl.shape, ref_impl.shape);
  EXPECT_EQ(impl.dtype, ref_impl.dtype);
  EXPECT_EQ(impl.device.type, ref_impl.device.type);

  float *result_data = result.data<float>();

  EXPECT_FLOAT_EQ(result_data[0], -1.0f);
  EXPECT_FLOAT_EQ(result_data[1], 2.0f);
  EXPECT_FLOAT_EQ(result_data[2], -3.0f);
  EXPECT_FLOAT_EQ(result_data[3], 4.0f);
}

TEST(TensorOpsTest, ReLU) {
  std::vector<float> data = {1.0f, -2.0f, 3.0f, -4.0f};
  quasai::Shape shape{2, 2};
  quasai::Tensor tensor =
      quasai::Tensor::from_data(data.data(), shape, quasai::DType::FLOAT32);

  quasai::Tensor result = quasai::relu(tensor);

  const auto impl = result.get_impl();
  const auto ref_impl = tensor.get_impl();

  EXPECT_EQ(impl.shape, ref_impl.shape);
  EXPECT_EQ(impl.dtype, ref_impl.dtype);
  EXPECT_EQ(impl.device.type, ref_impl.device.type);

  float *result_data = result.data<float>();

  EXPECT_FLOAT_EQ(result_data[0], 1.0f);
  EXPECT_FLOAT_EQ(result_data[1], 0.0f);
  EXPECT_FLOAT_EQ(result_data[2], 3.0f);
  EXPECT_FLOAT_EQ(result_data[3], 0.0f);
}

TEST(TensorOpsTest, Sigmoid) {
  std::vector<float> data = {0.0f, 1.0f, -1.0f, 2.0f};
  quasai::Shape shape{2, 2};
  quasai::Tensor tensor =
      quasai::Tensor::from_data(data.data(), shape, quasai::DType::FLOAT32);

  quasai::Tensor result = quasai::sigmoid(tensor);

  const auto impl = result.get_impl();
  const auto ref_impl = tensor.get_impl();

  EXPECT_EQ(impl.shape, ref_impl.shape);
  EXPECT_EQ(impl.dtype, ref_impl.dtype);
  EXPECT_EQ(impl.device.type, ref_impl.device.type);

  float *result_data = result.data<float>();

  EXPECT_FLOAT_EQ(result_data[0], 0.5f);
  EXPECT_FLOAT_EQ(result_data[1], 1.0f / (1.0f + std::exp(-1.0f)));
  EXPECT_FLOAT_EQ(result_data[2], 1.0f / (1.0f + std::exp(1.0f)));
  EXPECT_FLOAT_EQ(result_data[3], 1.0f / (1.0f + std::exp(-2.0f)));
}

TEST(TensorOpsTest, Tanh) {
  std::vector<float> data = {0.0f, 1.0f, -1.0f, 2.0f};
  quasai::Shape shape{2, 2};
  quasai::Tensor tensor =
      quasai::Tensor::from_data(data.data(), shape, quasai::DType::FLOAT32);

  quasai::Tensor result = quasai::tanh(tensor);

  const auto impl = result.get_impl();
  const auto ref_impl = tensor.get_impl();

  EXPECT_EQ(impl.shape, ref_impl.shape);
  EXPECT_EQ(impl.dtype, ref_impl.dtype);
  EXPECT_EQ(impl.device.type, ref_impl.device.type);

  float *result_data = result.data<float>();

  EXPECT_FLOAT_EQ(result_data[0], std::tanh(0.0f));
  EXPECT_FLOAT_EQ(result_data[1], std::tanh(1.0f));
  EXPECT_FLOAT_EQ(result_data[2], std::tanh(-1.0f));
  EXPECT_FLOAT_EQ(result_data[3], std::tanh(2.0f));
}
