#include "quasai/ops/tensor_ops.hpp"
#include <gtest/gtest.h>

TEST(BroadcastToShape, Float32) {
  std::vector<float> data = {1.0f, 2.0f};
  quasai::Shape shape{2};
  quasai::Tensor tensor =
      quasai::Tensor::from_data(data.data(), shape, quasai::DType::FLOAT32);
  quasai::Shape target{3, 2};

  quasai::Tensor result = quasai::broadcast_to_shape(tensor, target);
  float *result_data = result.data<float>();
  EXPECT_FLOAT_EQ(result_data[0], 1.0f);
  EXPECT_FLOAT_EQ(result_data[1], 2.0f);
  EXPECT_FLOAT_EQ(result_data[2], 1.0f);
  EXPECT_FLOAT_EQ(result_data[3], 2.0f);
  EXPECT_FLOAT_EQ(result_data[4], 1.0f);
  EXPECT_FLOAT_EQ(result_data[5], 2.0f);
}

TEST(BroadcastToShape, Float64) {
  std::vector<double> data = {1.0, 2.0};
  quasai::Shape shape{2};
  quasai::Tensor tensor =
      quasai::Tensor::from_data(data.data(), shape, quasai::DType::FLOAT64);
  quasai::Shape target{3, 2};

  quasai::Tensor result = quasai::broadcast_to_shape(tensor, target);
  double *result_data = result.data<double>();
  EXPECT_DOUBLE_EQ(result_data[0], 1.0);
  EXPECT_DOUBLE_EQ(result_data[1], 2.0);
  EXPECT_DOUBLE_EQ(result_data[2], 1.0);
  EXPECT_DOUBLE_EQ(result_data[3], 2.0);
  EXPECT_DOUBLE_EQ(result_data[4], 1.0);
  EXPECT_DOUBLE_EQ(result_data[5], 2.0);
}

TEST(BroadcastToShape, Int32) {
  std::vector<int32_t> data = {1, 2};
  quasai::Shape shape{2};
  quasai::Tensor tensor =
      quasai::Tensor::from_data(data.data(), shape, quasai::DType::INT32);
  quasai::Shape target{3, 2};

  quasai::Tensor result = quasai::broadcast_to_shape(tensor, target);
  int32_t *result_data = result.data<int32_t>();
  EXPECT_EQ(result_data[0], 1);
  EXPECT_EQ(result_data[1], 2);
  EXPECT_EQ(result_data[2], 1);
  EXPECT_EQ(result_data[3], 2);
  EXPECT_EQ(result_data[4], 1);
  EXPECT_EQ(result_data[5], 2);
}

TEST(BroadcastToShape, Int64) {
  std::vector<int64_t> data = {1, 2};
  quasai::Shape shape{2};
  quasai::Tensor tensor =
      quasai::Tensor::from_data(data.data(), shape, quasai::DType::INT64);
  quasai::Shape target{3, 2};

  quasai::Tensor result = quasai::broadcast_to_shape(tensor, target);
  int64_t *result_data = result.data<int64_t>();
  EXPECT_EQ(result_data[0], 1);
  EXPECT_EQ(result_data[1], 2);
  EXPECT_EQ(result_data[2], 1);
  EXPECT_EQ(result_data[3], 2);
  EXPECT_EQ(result_data[4], 1);
  EXPECT_EQ(result_data[5], 2);
}

TEST(BroadcastToShape, ScalarToShape) {
  std::vector<float> data = {5.0f};
  quasai::Shape shape{};
  quasai::Tensor tensor =
      quasai::Tensor::from_data(data.data(), shape, quasai::DType::FLOAT32);
  quasai::Shape target{3, 2};

  quasai::Tensor result = quasai::broadcast_to_shape(tensor, target);
  float *result_data = result.data<float>();
  for (size_t i = 0; i < 6; ++i) {
    EXPECT_FLOAT_EQ(result_data[i], 5.0f);
  }
}
