#include "quasai/autograd/metadata.hpp"
#include "quasai/ops/tensor_ops.hpp"
#include <cmath>
#include <gtest/gtest.h>

TEST(Slice, Float32) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  quasai::Shape shape{3, 2};
  quasai::Tensor tensor =
      quasai::Tensor::from_data(data.data(), shape, quasai::DType::FLOAT32);

  quasai::Tensor result = quasai::slice(tensor, 1, 2);
  float *result_data = result.data<float>();
  EXPECT_EQ(result.shape()[0], 1);
  EXPECT_EQ(result.shape()[1], 2);
  EXPECT_FLOAT_EQ(result_data[0], 3.0f);
  EXPECT_FLOAT_EQ(result_data[1], 4.0f);
}

TEST(Slice, Float64) {
  std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  quasai::Shape shape{3, 2};
  quasai::Tensor tensor =
      quasai::Tensor::from_data(data.data(), shape, quasai::DType::FLOAT64);

  quasai::Tensor result = quasai::slice(tensor, 1, 3);
  double *result_data = result.data<double>();
  EXPECT_EQ(result.shape()[0], 2);
  EXPECT_FLOAT_EQ(result_data[0], 3.0);
  EXPECT_FLOAT_EQ(result_data[1], 4.0);
}
