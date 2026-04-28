#include "quasai/autograd/metadata.hpp"
#include "quasai/ops/tensor_ops.hpp"
#include <cmath>
#include <gtest/gtest.h>

TEST(Expand, Float32) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f};
  quasai::core::Shape shape{3};
  quasai::core::Tensor tensor = quasai::core::Tensor::from_data(
      data.data(), shape, quasai::core::DType::FLOAT32);

  quasai::core::Tensor result =
      quasai::ops::expand(tensor, quasai::core::Shape{2, 3});
  EXPECT_EQ(result.shape()[0], 2);
  EXPECT_EQ(result.shape()[1], 3);
  EXPECT_FALSE(result.is_contiguous());

  result = quasai::ops::make_contiguous(result);

  float *result_data = result.data<float>();
  EXPECT_FLOAT_EQ(result_data[0], 1.0f);
  EXPECT_FLOAT_EQ(result_data[1], 2.0f);
  EXPECT_FLOAT_EQ(result_data[2], 3.0f);
  EXPECT_FLOAT_EQ(result_data[3], 1.0f);
  EXPECT_FLOAT_EQ(result_data[4], 2.0f);
  EXPECT_FLOAT_EQ(result_data[5], 3.0f);
}

TEST(Expand, Float64) {
  std::vector<double> data = {1.0, 2.0};
  quasai::core::Shape shape{2};
  quasai::core::Tensor tensor = quasai::core::Tensor::from_data(
      data.data(), shape, quasai::core::DType::FLOAT64);

  quasai::core::Tensor result =
      quasai::ops::expand(tensor, quasai::core::Shape{2, 2});
  EXPECT_FALSE(result.is_contiguous());
  result = quasai::ops::make_contiguous(result);

  EXPECT_EQ(result.shape()[0], 2);
  EXPECT_EQ(result.shape()[1], 2);
}

TEST(Expand, Scalar) {
  float data = 5.0f;
  quasai::core::Shape shape{};
  quasai::core::Tensor tensor = quasai::core::Tensor::from_data(
      &data, shape, quasai::core::DType::FLOAT32);

  quasai::core::Tensor result =
      quasai::ops::expand(tensor, quasai::core::Shape{3});
  EXPECT_EQ(result.shape()[0], 3);
  EXPECT_FALSE(result.is_contiguous());

  result = quasai::ops::make_contiguous(result);
  float *result_data = result.data<float>();

  EXPECT_FLOAT_EQ(result_data[0], 5.0f);
  EXPECT_FLOAT_EQ(result_data[1], 5.0f);
  EXPECT_FLOAT_EQ(result_data[2], 5.0f);
}
