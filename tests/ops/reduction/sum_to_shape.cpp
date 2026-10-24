#include "quasai/ops/tensor_ops.hpp"
#include <gtest/gtest.h>

TEST(SumToShape, Float32) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  quasai::core::Shape shape{2, 3};
  quasai::core::Tensor tensor = quasai::core::Tensor::from_data(
      data.data(), shape, quasai::core::DType::FLOAT32);
  quasai::core::Shape target{3};

  quasai::core::Tensor result = quasai::ops::sum_to_shape(tensor, target);
  float *result_data = result.data<float>();
  EXPECT_FLOAT_EQ(result_data[0], 5.0f);
  EXPECT_FLOAT_EQ(result_data[1], 7.0f);
  EXPECT_FLOAT_EQ(result_data[2], 9.0f);
}

TEST(SumToShape, Float64) {
  std::vector<double> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  quasai::core::Shape shape{2, 3};
  quasai::core::Tensor tensor = quasai::core::Tensor::from_data(
      data.data(), shape, quasai::core::DType::FLOAT64);
  quasai::core::Shape target{3};

  quasai::core::Tensor result = quasai::ops::sum_to_shape(tensor, target);
  double *result_data = result.data<double>();
  EXPECT_FLOAT_EQ(result_data[0], 5.0);
  EXPECT_FLOAT_EQ(result_data[1], 7.0);
  EXPECT_FLOAT_EQ(result_data[2], 9.0);
}

TEST(SumToShape, Int32) {
  std::vector<int32_t> data = {1, 2, 3, 4, 5, 6};
  quasai::core::Shape shape{2, 3};
  quasai::core::Tensor tensor = quasai::core::Tensor::from_data(
      data.data(), shape, quasai::core::DType::INT32);
  quasai::core::Shape target{3};

  quasai::core::Tensor result = quasai::ops::sum_to_shape(tensor, target);
  int32_t *result_data = result.data<int32_t>();
  EXPECT_EQ(result_data[0], 5);
  EXPECT_EQ(result_data[1], 7);
  EXPECT_EQ(result_data[2], 9);
}

TEST(SumToShape, Int64) {
  std::vector<int64_t> data = {1, 2, 3, 4, 5, 6};
  quasai::core::Shape shape{2, 3};
  quasai::core::Tensor tensor = quasai::core::Tensor::from_data(
      data.data(), shape, quasai::core::DType::INT64);
  quasai::core::Shape target{3};

  quasai::core::Tensor result = quasai::ops::sum_to_shape(tensor, target);
  int64_t *result_data = result.data<int64_t>();
  EXPECT_EQ(result_data[0], 5);
  EXPECT_EQ(result_data[1], 7);
  EXPECT_EQ(result_data[2], 9);
}

TEST(SumToShape, NonContiguousTranspose) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  quasai::core::Tensor tensor = quasai::core::Tensor::from_data(
      data.data(), quasai::core::Shape{2, 3}, quasai::core::DType::FLOAT32);
  quasai::core::Tensor transposed = quasai::ops::transpose(tensor);

  quasai::core::Tensor result = quasai::ops::sum_to_shape(transposed, {2});
  const float *result_data = result.data<float>();
  EXPECT_FLOAT_EQ(result_data[0], 6.0f);
  EXPECT_FLOAT_EQ(result_data[1], 15.0f);
}

TEST(SumToShape, ExpandedBroadcastView) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f};
  quasai::core::Tensor tensor = quasai::core::Tensor::from_data(
      data.data(), quasai::core::Shape{3}, quasai::core::DType::FLOAT32);
  quasai::core::Tensor expanded = quasai::ops::expand(tensor, {2, 3});

  quasai::core::Tensor result = quasai::ops::sum_to_shape(expanded, {3});
  const float *result_data = result.data<float>();
  EXPECT_FLOAT_EQ(result_data[0], 2.0f);
  EXPECT_FLOAT_EQ(result_data[1], 4.0f);
  EXPECT_FLOAT_EQ(result_data[2], 6.0f);
}

TEST(SumToShape, Broadcast) {
  std::vector<float> data = {1.0f, 2.0f};
  quasai::core::Shape shape{2};
  quasai::core::Tensor tensor = quasai::core::Tensor::from_data(
      data.data(), shape, quasai::core::DType::FLOAT32);
  quasai::core::Shape target{2, 3};

  EXPECT_THROW(quasai::ops::sum_to_shape(tensor, target), std::runtime_error);
}
