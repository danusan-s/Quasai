#include "quasai/ops/tensor_ops.hpp"
#include "quasai/autograd/metadata.hpp"
#include <gtest/gtest.h>
#include <cmath>

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

TEST(Sub, GradientA) {
  float eps = 1e-3f;
  float val_a = 5.0f;
  float val_b = 3.0f;
  quasai::Tensor input_a = quasai::Tensor::from_data(&val_a, quasai::Shape{}, quasai::DType::FLOAT32);
  quasai::Tensor input_b = quasai::Tensor::from_data(&val_b, quasai::Shape{}, quasai::DType::FLOAT32);
  input_a.requires_grad(true);
  input_b.requires_grad(true);

  quasai::Tensor output = quasai::sub(input_a, input_b);
  output.backward();

  quasai::Tensor grad_a = input_a.autograd_meta()->grad;
  float computed_grad_a = grad_a.data<float>()[0];

  float f_plus = (val_a + eps) - val_b;
  float f_minus = (val_a - eps) - val_b;
  float finite_diff = (f_plus - f_minus) / (2.0f * eps);
  float rel_err = std::abs(computed_grad_a - finite_diff) / std::max(1.0f, std::abs(val_a));

  EXPECT_NEAR(rel_err, 0.0f, 1e-2f);
}

TEST(Sub, GradientB) {
  float eps = 1e-3f;
  float val_a = 5.0f;
  float val_b = 3.0f;
  quasai::Tensor input_a = quasai::Tensor::from_data(&val_a, quasai::Shape{}, quasai::DType::FLOAT32);
  quasai::Tensor input_b = quasai::Tensor::from_data(&val_b, quasai::Shape{}, quasai::DType::FLOAT32);
  input_a.requires_grad(true);
  input_b.requires_grad(true);

  quasai::Tensor output = quasai::sub(input_a, input_b);
  output.backward();

  quasai::Tensor grad_b = input_b.autograd_meta()->grad;
  float computed_grad_b = grad_b.data<float>()[0];

  float f_plus = val_a - (val_b + eps);
  float f_minus = val_a - (val_b - eps);
  float finite_diff = (f_plus - f_minus) / (2.0f * eps);
  float rel_err = std::abs(computed_grad_b - finite_diff) / std::max(1.0f, std::abs(val_b));

  EXPECT_NEAR(rel_err, 0.0f, 1e-2f);
}
