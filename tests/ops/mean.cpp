#include "quasai/ops/tensor_ops.hpp"
#include <gtest/gtest.h>

TEST(Mean, Float32) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  quasai::Shape shape{2, 2};
  quasai::Tensor tensor =
      quasai::Tensor::from_data(data.data(), shape, quasai::DType::FLOAT32);

  quasai::Tensor result = quasai::mean(tensor);
  float *result_data = result.data<float>();
  EXPECT_FLOAT_EQ(result_data[0], 2.5f);
}

TEST(Mean, Float64) {
  std::vector<double> data = {1.0, 2.0, 3.0, 4.0};
  quasai::Shape shape{2, 2};
  quasai::Tensor tensor =
      quasai::Tensor::from_data(data.data(), shape, quasai::DType::FLOAT64);

  quasai::Tensor result = quasai::mean(tensor);
  double *result_data = result.data<double>();
  EXPECT_DOUBLE_EQ(result_data[0], 2.5);
}

TEST(Mean, Int32ShouldThrow) {
  std::vector<int32_t> data = {1, 2, 3, 4};
  quasai::Shape shape{2, 2};
  quasai::Tensor tensor =
      quasai::Tensor::from_data(data.data(), shape, quasai::DType::INT32);

  EXPECT_THROW(quasai::mean(tensor), std::runtime_error);
}
