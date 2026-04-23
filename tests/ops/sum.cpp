#include "quasai/ops/tensor_ops.hpp"
#include <gtest/gtest.h>

TEST(Sum, Float32) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  quasai::Shape shape{2, 2};
  quasai::Tensor tensor =
      quasai::Tensor::from_data(data.data(), shape, quasai::DType::FLOAT32);

  quasai::Tensor result = quasai::sum(tensor);
  float *result_data = result.data<float>();
  EXPECT_FLOAT_EQ(result_data[0], 10.0f);
}

TEST(Sum, Float64) {
  std::vector<double> data = {1.0, 2.0, 3.0, 4.0};
  quasai::Shape shape{2, 2};
  quasai::Tensor tensor =
      quasai::Tensor::from_data(data.data(), shape, quasai::DType::FLOAT64);

  quasai::Tensor result = quasai::sum(tensor);
  double *result_data = result.data<double>();
  EXPECT_DOUBLE_EQ(result_data[0], 10.0);
}

TEST(Sum, Int32) {
  std::vector<int32_t> data = {1, 2, 3, 4};
  quasai::Shape shape{2, 2};
  quasai::Tensor tensor =
      quasai::Tensor::from_data(data.data(), shape, quasai::DType::INT32);

  quasai::Tensor result = quasai::sum(tensor);
  int32_t *result_data = result.data<int32_t>();
  EXPECT_EQ(result_data[0], 10);
}

TEST(Sum, Int64) {
  std::vector<int64_t> data = {1, 2, 3, 4};
  quasai::Shape shape{2, 2};
  quasai::Tensor tensor =
      quasai::Tensor::from_data(data.data(), shape, quasai::DType::INT64);

  quasai::Tensor result = quasai::sum(tensor);
  int64_t *result_data = result.data<int64_t>();
  EXPECT_EQ(result_data[0], 10);
}

TEST(Sum, Negative) {
  std::vector<float> data = {-1.0f, 2.0f, -3.0f, 4.0f};
  quasai::Shape shape{2, 2};
  quasai::Tensor tensor =
      quasai::Tensor::from_data(data.data(), shape, quasai::DType::FLOAT32);

  quasai::Tensor result = quasai::sum(tensor);
  float *result_data = result.data<float>();
  EXPECT_FLOAT_EQ(result_data[0], 2.0f);
}
