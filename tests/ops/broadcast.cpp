#include "quasai/ops/tensor_ops.hpp"
#include <gtest/gtest.h>

TEST(BroadcastAdd, Float32) {
  std::vector<float> data_a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<float> data_b = {10.0f, 20.0f};
  quasai::core::Shape shape_a{3, 2};
  quasai::core::Shape shape_b{2};
  quasai::core::Tensor tensor_a =
      quasai::core::Tensor::from_data(data_a.data(), shape_a, quasai::core::DType::FLOAT32);
  quasai::core::Tensor tensor_b =
      quasai::core::Tensor::from_data(data_b.data(), shape_b, quasai::core::DType::FLOAT32);

  quasai::core::Tensor result = quasai::ops::add(tensor_a, tensor_b);
  float *result_data = result.data<float>();
  EXPECT_FLOAT_EQ(result_data[0], 11.0f);
  EXPECT_FLOAT_EQ(result_data[1], 22.0f);
  EXPECT_FLOAT_EQ(result_data[2], 13.0f);
  EXPECT_FLOAT_EQ(result_data[3], 24.0f);
  EXPECT_FLOAT_EQ(result_data[4], 15.0f);
  EXPECT_FLOAT_EQ(result_data[5], 26.0f);
}

TEST(BroadcastAdd, Float64) {
  std::vector<double> data_a = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  std::vector<double> data_b = {10.0, 20.0};
  quasai::core::Shape shape_a{3, 2};
  quasai::core::Shape shape_b{2};
  quasai::core::Tensor tensor_a =
      quasai::core::Tensor::from_data(data_a.data(), shape_a, quasai::core::DType::FLOAT64);
  quasai::core::Tensor tensor_b =
      quasai::core::Tensor::from_data(data_b.data(), shape_b, quasai::core::DType::FLOAT64);

  quasai::core::Tensor result = quasai::ops::add(tensor_a, tensor_b);
  double *result_data = result.data<double>();
  EXPECT_DOUBLE_EQ(result_data[0], 11.0);
  EXPECT_DOUBLE_EQ(result_data[1], 22.0);
  EXPECT_DOUBLE_EQ(result_data[2], 13.0);
  EXPECT_DOUBLE_EQ(result_data[3], 24.0);
  EXPECT_DOUBLE_EQ(result_data[4], 15.0);
  EXPECT_DOUBLE_EQ(result_data[5], 26.0);
}

TEST(BroadcastAdd, Int32) {
  std::vector<int32_t> data_a = {1, 2, 3, 4, 5, 6};
  std::vector<int32_t> data_b = {10, 20};
  quasai::core::Shape shape_a{3, 2};
  quasai::core::Shape shape_b{2};
  quasai::core::Tensor tensor_a =
      quasai::core::Tensor::from_data(data_a.data(), shape_a, quasai::core::DType::INT32);
  quasai::core::Tensor tensor_b =
      quasai::core::Tensor::from_data(data_b.data(), shape_b, quasai::core::DType::INT32);

  quasai::core::Tensor result = quasai::ops::add(tensor_a, tensor_b);
  int32_t *result_data = result.data<int32_t>();
  EXPECT_EQ(result_data[0], 11);
  EXPECT_EQ(result_data[1], 22);
  EXPECT_EQ(result_data[2], 13);
  EXPECT_EQ(result_data[3], 24);
  EXPECT_EQ(result_data[4], 15);
  EXPECT_EQ(result_data[5], 26);
}

TEST(BroadcastAdd, Int64) {
  std::vector<int64_t> data_a = {1, 2, 3, 4, 5, 6};
  std::vector<int64_t> data_b = {10, 20};
  quasai::core::Shape shape_a{3, 2};
  quasai::core::Shape shape_b{2};
  quasai::core::Tensor tensor_a =
      quasai::core::Tensor::from_data(data_a.data(), shape_a, quasai::core::DType::INT64);
  quasai::core::Tensor tensor_b =
      quasai::core::Tensor::from_data(data_b.data(), shape_b, quasai::core::DType::INT64);

  quasai::core::Tensor result = quasai::ops::add(tensor_a, tensor_b);
  int64_t *result_data = result.data<int64_t>();
  EXPECT_EQ(result_data[0], 11);
  EXPECT_EQ(result_data[1], 22);
  EXPECT_EQ(result_data[2], 13);
  EXPECT_EQ(result_data[3], 24);
  EXPECT_EQ(result_data[4], 15);
  EXPECT_EQ(result_data[5], 26);
}

TEST(BroadcastAddScalar, Float32) {
  std::vector<float> data_a = {1.0f, 2.0f, 3.0f};
  std::vector<float> data_b = {10.0f};
  quasai::core::Shape shape_a{3};
  quasai::core::Shape shape_b{1};
  quasai::core::Tensor tensor_a =
      quasai::core::Tensor::from_data(data_a.data(), shape_a, quasai::core::DType::FLOAT32);
  quasai::core::Tensor tensor_b =
      quasai::core::Tensor::from_data(data_b.data(), shape_b, quasai::core::DType::FLOAT32);

  quasai::core::Tensor result = quasai::ops::add(tensor_a, tensor_b);
  float *result_data = result.data<float>();
  EXPECT_FLOAT_EQ(result_data[0], 11.0f);
  EXPECT_FLOAT_EQ(result_data[1], 12.0f);
  EXPECT_FLOAT_EQ(result_data[2], 13.0f);
}

TEST(BroadcastSub, Float32) {
  std::vector<float> data_a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<float> data_b = {10.0f, 20.0f};
  quasai::core::Shape shape_a{3, 2};
  quasai::core::Shape shape_b{2};
  quasai::core::Tensor tensor_a =
      quasai::core::Tensor::from_data(data_a.data(), shape_a, quasai::core::DType::FLOAT32);
  quasai::core::Tensor tensor_b =
      quasai::core::Tensor::from_data(data_b.data(), shape_b, quasai::core::DType::FLOAT32);

  quasai::core::Tensor result = quasai::ops::sub(tensor_a, tensor_b);
  float *result_data = result.data<float>();
  EXPECT_FLOAT_EQ(result_data[0], -9.0f);
  EXPECT_FLOAT_EQ(result_data[1], -18.0f);
  EXPECT_FLOAT_EQ(result_data[2], -7.0f);
  EXPECT_FLOAT_EQ(result_data[3], -16.0f);
  EXPECT_FLOAT_EQ(result_data[4], -5.0f);
  EXPECT_FLOAT_EQ(result_data[5], -14.0f);
}

TEST(BroadcastMul, Float32) {
  std::vector<float> data_a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<float> data_b = {2.0f, 3.0f};
  quasai::core::Shape shape_a{3, 2};
  quasai::core::Shape shape_b{2};
  quasai::core::Tensor tensor_a =
      quasai::core::Tensor::from_data(data_a.data(), shape_a, quasai::core::DType::FLOAT32);
  quasai::core::Tensor tensor_b =
      quasai::core::Tensor::from_data(data_b.data(), shape_b, quasai::core::DType::FLOAT32);

  quasai::core::Tensor result = quasai::ops::mul(tensor_a, tensor_b);
  float *result_data = result.data<float>();
  EXPECT_FLOAT_EQ(result_data[0], 2.0f);
  EXPECT_FLOAT_EQ(result_data[1], 6.0f);
  EXPECT_FLOAT_EQ(result_data[2], 6.0f);
  EXPECT_FLOAT_EQ(result_data[3], 12.0f);
  EXPECT_FLOAT_EQ(result_data[4], 10.0f);
  EXPECT_FLOAT_EQ(result_data[5], 18.0f);
}

TEST(BroadcastDiv, Float32) {
  std::vector<float> data_a = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f};
  std::vector<float> data_b = {2.0f, 5.0f};
  quasai::core::Shape shape_a{3, 2};
  quasai::core::Shape shape_b{2};
  quasai::core::Tensor tensor_a =
      quasai::core::Tensor::from_data(data_a.data(), shape_a, quasai::core::DType::FLOAT32);
  quasai::core::Tensor tensor_b =
      quasai::core::Tensor::from_data(data_b.data(), shape_b, quasai::core::DType::FLOAT32);

  quasai::core::Tensor result = quasai::ops::div(tensor_a, tensor_b);
  float *result_data = result.data<float>();
  EXPECT_FLOAT_EQ(result_data[0], 5.0f);
  EXPECT_FLOAT_EQ(result_data[1], 4.0f);
  EXPECT_FLOAT_EQ(result_data[2], 15.0f);
  EXPECT_FLOAT_EQ(result_data[3], 8.0f);
  EXPECT_FLOAT_EQ(result_data[4], 25.0f);
  EXPECT_FLOAT_EQ(result_data[5], 12.0f);
}
