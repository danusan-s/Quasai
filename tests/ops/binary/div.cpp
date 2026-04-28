#include "quasai/autograd/metadata.hpp"
#include "quasai/ops/tensor_ops.hpp"
#include <cmath>
#include <gtest/gtest.h>

TEST(Div, Float32) {
  std::vector<float> data_a = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> data_b = {5.0f, 6.0f, 7.0f, 8.0f};
  quasai::core::Shape shape{2, 2};
  quasai::core::Tensor tensor_a = quasai::core::Tensor::from_data(
      data_a.data(), shape, quasai::core::DType::FLOAT32);
  quasai::core::Tensor tensor_b = quasai::core::Tensor::from_data(
      data_b.data(), shape, quasai::core::DType::FLOAT32);

  quasai::core::Tensor result = quasai::ops::div(tensor_a, tensor_b);
  float *result_data = result.data<float>();
  EXPECT_FLOAT_EQ(result_data[0], 1.0f / 5.0f);
  EXPECT_FLOAT_EQ(result_data[1], 2.0f / 6.0f);
  EXPECT_FLOAT_EQ(result_data[2], 3.0f / 7.0f);
  EXPECT_FLOAT_EQ(result_data[3], 4.0f / 8.0f);
}

TEST(Div, Float64) {
  std::vector<double> data_a = {1.0, 2.0, 3.0, 4.0};
  std::vector<double> data_b = {5.0, 6.0, 7.0, 8.0};
  quasai::core::Shape shape{2, 2};
  quasai::core::Tensor tensor_a = quasai::core::Tensor::from_data(
      data_a.data(), shape, quasai::core::DType::FLOAT64);
  quasai::core::Tensor tensor_b = quasai::core::Tensor::from_data(
      data_b.data(), shape, quasai::core::DType::FLOAT64);

  quasai::core::Tensor result = quasai::ops::div(tensor_a, tensor_b);
  double *result_data = result.data<double>();
  EXPECT_DOUBLE_EQ(result_data[0], 1.0 / 5.0);
  EXPECT_DOUBLE_EQ(result_data[1], 2.0 / 6.0);
  EXPECT_DOUBLE_EQ(result_data[2], 3.0 / 7.0);
  EXPECT_DOUBLE_EQ(result_data[3], 4.0 / 8.0);
}

TEST(Div, Int32) {
  std::vector<int32_t> data_a = {10, 20, 30, 40};
  std::vector<int32_t> data_b = {3, 4, 5, 8};
  quasai::core::Shape shape{2, 2};
  quasai::core::Tensor tensor_a = quasai::core::Tensor::from_data(
      data_a.data(), shape, quasai::core::DType::INT32);
  quasai::core::Tensor tensor_b = quasai::core::Tensor::from_data(
      data_b.data(), shape, quasai::core::DType::INT32);

  quasai::core::Tensor result = quasai::ops::div(tensor_a, tensor_b);
  int32_t *result_data = result.data<int32_t>();
  EXPECT_EQ(result_data[0], 3);
  EXPECT_EQ(result_data[1], 5);
  EXPECT_EQ(result_data[2], 6);
  EXPECT_EQ(result_data[3], 5);
}

TEST(Div, Int64) {
  std::vector<int64_t> data_a = {10, 20, 30, 40};
  std::vector<int64_t> data_b = {3, 4, 5, 8};
  quasai::core::Shape shape{2, 2};
  quasai::core::Tensor tensor_a = quasai::core::Tensor::from_data(
      data_a.data(), shape, quasai::core::DType::INT64);
  quasai::core::Tensor tensor_b = quasai::core::Tensor::from_data(
      data_b.data(), shape, quasai::core::DType::INT64);

  quasai::core::Tensor result = quasai::ops::div(tensor_a, tensor_b);
  int64_t *result_data = result.data<int64_t>();
  EXPECT_EQ(result_data[0], 3);
  EXPECT_EQ(result_data[1], 5);
  EXPECT_EQ(result_data[2], 6);
  EXPECT_EQ(result_data[3], 5);
}

TEST(Div, GradientA) {
  float eps = 1e-3f;
  float val_a = 6.0f;
  float val_b = 3.0f;
  quasai::core::Tensor input_a = quasai::core::Tensor::from_data(
      &val_a, quasai::core::Shape{}, quasai::core::DType::FLOAT32);
  quasai::core::Tensor input_b = quasai::core::Tensor::from_data(
      &val_b, quasai::core::Shape{}, quasai::core::DType::FLOAT32);
  input_a.requires_grad(true);
  input_b.requires_grad(true);

  quasai::core::Tensor output = quasai::ops::div(input_a, input_b);
  output.backward();

  quasai::core::Tensor grad_a = input_a.autograd_meta()->grad;
  float computed_grad_a = grad_a.data<float>()[0];

  float f_plus = (val_a + eps) / val_b;
  float f_minus = (val_a - eps) / val_b;
  float finite_diff = (f_plus - f_minus) / (2.0f * eps);
  float rel_err =
      std::abs(computed_grad_a - finite_diff) / std::max(1.0f, std::abs(val_a));

  EXPECT_NEAR(rel_err, 0.0f, 1e-2f);
}

TEST(Div, GradientB) {
  float eps = 1e-3f;
  float val_a = 6.0f;
  float val_b = 3.0f;
  quasai::core::Tensor input_a = quasai::core::Tensor::from_data(
      &val_a, quasai::core::Shape{}, quasai::core::DType::FLOAT32);
  quasai::core::Tensor input_b = quasai::core::Tensor::from_data(
      &val_b, quasai::core::Shape{}, quasai::core::DType::FLOAT32);
  input_a.requires_grad(true);
  input_b.requires_grad(true);

  quasai::core::Tensor output = quasai::ops::div(input_a, input_b);
  output.backward();

  quasai::core::Tensor grad_b = input_b.autograd_meta()->grad;
  float computed_grad_b = grad_b.data<float>()[0];

  float f_plus = val_a / (val_b + eps);
  float f_minus = val_a / (val_b - eps);
  float finite_diff = (f_plus - f_minus) / (2.0f * eps);
  float rel_err =
      std::abs(computed_grad_b - finite_diff) / std::max(1.0f, std::abs(val_b));

  EXPECT_NEAR(rel_err, 0.0f, 1e-2f);
}
