#include "quasai/ops/tensor_ops.hpp"
#include <gtest/gtest.h>

TEST(BroadcastToShape, Float32) {
  std::vector<float> data = {1.0f, 2.0f};
  quasai::core::Shape shape{2};
  quasai::core::Tensor tensor =
      quasai::core::Tensor::from_data(data.data(), shape, quasai::core::DType::FLOAT32);
  quasai::core::Shape target{3, 2};

  quasai::core::Tensor result = quasai::ops::broadcast_to_shape(tensor, target);
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
  quasai::core::Shape shape{2};
  quasai::core::Tensor tensor =
      quasai::core::Tensor::from_data(data.data(), shape, quasai::core::DType::FLOAT64);
  quasai::core::Shape target{3, 2};

  quasai::core::Tensor result = quasai::ops::broadcast_to_shape(tensor, target);
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
  quasai::core::Shape shape{2};
  quasai::core::Tensor tensor =
      quasai::core::Tensor::from_data(data.data(), shape, quasai::core::DType::INT32);
  quasai::core::Shape target{3, 2};

  quasai::core::Tensor result = quasai::ops::broadcast_to_shape(tensor, target);
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
  quasai::core::Shape shape{2};
  quasai::core::Tensor tensor =
      quasai::core::Tensor::from_data(data.data(), shape, quasai::core::DType::INT64);
  quasai::core::Shape target{3, 2};

  quasai::core::Tensor result = quasai::ops::broadcast_to_shape(tensor, target);
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
  quasai::core::Shape shape{};
  quasai::core::Tensor tensor =
      quasai::core::Tensor::from_data(data.data(), shape, quasai::core::DType::FLOAT32);
  quasai::core::Shape target{3, 2};

  quasai::core::Tensor result = quasai::ops::broadcast_to_shape(tensor, target);
  float *result_data = result.data<float>();
  for (size_t i = 0; i < 6; ++i) {
    EXPECT_FLOAT_EQ(result_data[i], 5.0f);
  }
}
