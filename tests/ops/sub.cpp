#include "quasai/ops/tensor_ops.hpp"
#include <gtest/gtest.h>

TEST(Sub, Float32) {
  std::vector<float> data_a = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> data_b = {5.0f, 6.0f, 7.0f, 8.0f};
  quasai::Shape shape{2, 2};
  quasai::Tensor tensor_a =
      quasai::Tensor::from_data(data_a.data(), shape, quasai::DType::FLOAT32);
  quasai::Tensor tensor_b =
      quasai::Tensor::from_data(data_b.data(), shape, quasai::DType::FLOAT32);

  quasai::Tensor result = quasai::sub(tensor_a, tensor_b);
  float *result_data = result.data<float>();
  EXPECT_FLOAT_EQ(result_data[0], -4.0f);
  EXPECT_FLOAT_EQ(result_data[1], -4.0f);
  EXPECT_FLOAT_EQ(result_data[2], -4.0f);
  EXPECT_FLOAT_EQ(result_data[3], -4.0f);
}

TEST(Sub, Float64) {
  std::vector<double> data_a = {1.0, 2.0, 3.0, 4.0};
  std::vector<double> data_b = {5.0, 6.0, 7.0, 8.0};
  quasai::Shape shape{2, 2};
  quasai::Tensor tensor_a =
      quasai::Tensor::from_data(data_a.data(), shape, quasai::DType::FLOAT64);
  quasai::Tensor tensor_b =
      quasai::Tensor::from_data(data_b.data(), shape, quasai::DType::FLOAT64);

  quasai::Tensor result = quasai::sub(tensor_a, tensor_b);
  double *result_data = result.data<double>();
  EXPECT_DOUBLE_EQ(result_data[0], -4.0);
  EXPECT_DOUBLE_EQ(result_data[1], -4.0);
  EXPECT_DOUBLE_EQ(result_data[2], -4.0);
  EXPECT_DOUBLE_EQ(result_data[3], -4.0);
}

TEST(Sub, Int32) {
  std::vector<int32_t> data_a = {1, 2, 3, 4};
  std::vector<int32_t> data_b = {5, 6, 7, 8};
  quasai::Shape shape{2, 2};
  quasai::Tensor tensor_a =
      quasai::Tensor::from_data(data_a.data(), shape, quasai::DType::INT32);
  quasai::Tensor tensor_b =
      quasai::Tensor::from_data(data_b.data(), shape, quasai::DType::INT32);

  quasai::Tensor result = quasai::sub(tensor_a, tensor_b);
  int32_t *result_data = result.data<int32_t>();
  EXPECT_EQ(result_data[0], -4);
  EXPECT_EQ(result_data[1], -4);
  EXPECT_EQ(result_data[2], -4);
  EXPECT_EQ(result_data[3], -4);
}

TEST(Sub, Int64) {
  std::vector<int64_t> data_a = {1, 2, 3, 4};
  std::vector<int64_t> data_b = {5, 6, 7, 8};
  quasai::Shape shape{2, 2};
  quasai::Tensor tensor_a =
      quasai::Tensor::from_data(data_a.data(), shape, quasai::DType::INT64);
  quasai::Tensor tensor_b =
      quasai::Tensor::from_data(data_b.data(), shape, quasai::DType::INT64);

  quasai::Tensor result = quasai::sub(tensor_a, tensor_b);
  int64_t *result_data = result.data<int64_t>();
  EXPECT_EQ(result_data[0], -4);
  EXPECT_EQ(result_data[1], -4);
  EXPECT_EQ(result_data[2], -4);
  EXPECT_EQ(result_data[3], -4);
}
