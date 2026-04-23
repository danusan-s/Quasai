#include "quasai/ops/tensor_ops.hpp"
#include <gtest/gtest.h>
#include <cmath>

TEST(Relu, Float32) {
  std::vector<float> data = {1.0f, -2.0f, 3.0f, -4.0f};
  quasai::Shape shape{2, 2};
  quasai::Tensor tensor =
      quasai::Tensor::from_data(data.data(), shape, quasai::DType::FLOAT32);

  quasai::Tensor result = quasai::relu(tensor);
  float *result_data = result.data<float>();
  EXPECT_FLOAT_EQ(result_data[0], 1.0f);
  EXPECT_FLOAT_EQ(result_data[1], 0.0f);
  EXPECT_FLOAT_EQ(result_data[2], 3.0f);
  EXPECT_FLOAT_EQ(result_data[3], 0.0f);
}

TEST(Relu, Float64) {
  std::vector<double> data = {1.0, -2.0, 3.0, -4.0};
  quasai::Shape shape{2, 2};
  quasai::Tensor tensor =
      quasai::Tensor::from_data(data.data(), shape, quasai::DType::FLOAT64);

  quasai::Tensor result = quasai::relu(tensor);
  double *result_data = result.data<double>();
  EXPECT_DOUBLE_EQ(result_data[0], 1.0);
  EXPECT_DOUBLE_EQ(result_data[1], 0.0);
  EXPECT_DOUBLE_EQ(result_data[2], 3.0);
  EXPECT_DOUBLE_EQ(result_data[3], 0.0);
}

TEST(Relu, Int32) {
  std::vector<int32_t> data = {1, -2, 3, -4};
  quasai::Shape shape{2, 2};
  quasai::Tensor tensor =
      quasai::Tensor::from_data(data.data(), shape, quasai::DType::INT32);

  quasai::Tensor result = quasai::relu(tensor);
  int32_t *result_data = result.data<int32_t>();
  EXPECT_EQ(result_data[0], 1);
  EXPECT_EQ(result_data[1], 0);
  EXPECT_EQ(result_data[2], 3);
  EXPECT_EQ(result_data[3], 0);
}
