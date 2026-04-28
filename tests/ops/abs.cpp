#include "quasai/autograd/metadata.hpp"
#include "quasai/ops/tensor_ops.hpp"
#include <cmath>
#include <gtest/gtest.h>

TEST(Abs, Float32) {
  std::vector<float> data = {1.0f, -2.0f, 3.0f, -4.0f};
  quasai::core::Shape shape{2, 2};
  quasai::core::Tensor tensor =
      quasai::core::Tensor::from_data(data.data(), shape, quasai::core::DType::FLOAT32);

  quasai::core::Tensor result = quasai::ops::abs(tensor);
  float *result_data = result.data<float>();
  EXPECT_FLOAT_EQ(result_data[0], 1.0f);
  EXPECT_FLOAT_EQ(result_data[1], 2.0f);
  EXPECT_FLOAT_EQ(result_data[2], 3.0f);
  EXPECT_FLOAT_EQ(result_data[3], 4.0f);
}

TEST(Abs, Float64) {
  std::vector<double> data = {1.0, -2.0, 3.0, -4.0};
  quasai::core::Shape shape{2, 2};
  quasai::core::Tensor tensor =
      quasai::core::Tensor::from_data(data.data(), shape, quasai::core::DType::FLOAT64);

  quasai::core::Tensor result = quasai::ops::abs(tensor);
  double *result_data = result.data<double>();
  EXPECT_DOUBLE_EQ(result_data[0], 1.0);
  EXPECT_DOUBLE_EQ(result_data[1], 2.0);
  EXPECT_DOUBLE_EQ(result_data[2], 3.0);
  EXPECT_DOUBLE_EQ(result_data[3], 4.0);
}

TEST(Abs, Int32) {
  std::vector<int32_t> data = {1, -2, 3, -4};
  quasai::core::Shape shape{2, 2};
  quasai::core::Tensor tensor =
      quasai::core::Tensor::from_data(data.data(), shape, quasai::core::DType::INT32);

  quasai::core::Tensor result = quasai::ops::abs(tensor);
  int32_t *result_data = result.data<int32_t>();
  EXPECT_EQ(result_data[0], 1);
  EXPECT_EQ(result_data[1], 2);
  EXPECT_EQ(result_data[2], 3);
  EXPECT_EQ(result_data[3], 4);
}

TEST(Abs, Gradient) {
  float eps = 1e-3f;
  float val = 2.0f;
  quasai::core::Tensor input =
      quasai::core::Tensor::from_data(&val, quasai::core::Shape{}, quasai::core::DType::FLOAT32);
  input.requires_grad(true);

  quasai::core::Tensor output = quasai::ops::abs(input);
  output.backward();

  quasai::core::Tensor grad = input.autograd_meta()->grad;
  float computed_grad = grad.data<float>()[0];

  float f_plus = std::abs(val + eps);
  float f_minus = std::abs(val - eps);
  float finite_diff = (f_plus - f_minus) / (2.0f * eps);
  float rel_err = std::abs(computed_grad - finite_diff) / std::max(1.0f, std::abs(val));

  EXPECT_NEAR(rel_err, 0.0f, 1e-2f);
}

TEST(Abs, GradientNegative) {
  float eps = 1e-3f;
  float val = -2.0f;
  quasai::core::Tensor input =
      quasai::core::Tensor::from_data(&val, quasai::core::Shape{}, quasai::core::DType::FLOAT32);
  input.requires_grad(true);

  quasai::core::Tensor output = quasai::ops::abs(input);
  output.backward();

  quasai::core::Tensor grad = input.autograd_meta()->grad;
  float computed_grad = grad.data<float>()[0];

  float f_plus = std::abs(val + eps);
  float f_minus = std::abs(val - eps);
  float finite_diff = (f_plus - f_minus) / (2.0f * eps);
  float rel_err = std::abs(computed_grad - finite_diff) / std::max(1.0f, std::abs(val));

  EXPECT_NEAR(rel_err, 0.0f, 1e-2f);
}