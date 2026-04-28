#include "quasai/autograd/metadata.hpp"
#include "quasai/ops/tensor_ops.hpp"
#include <cmath>
#include <gtest/gtest.h>

TEST(Sum, Float32) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  quasai::core::Shape shape{2, 2};
  quasai::core::Tensor tensor = quasai::core::Tensor::from_data(
      data.data(), shape, quasai::core::DType::FLOAT32);

  quasai::core::Tensor result = quasai::ops::sum(tensor);
  float *result_data = result.data<float>();
  EXPECT_FLOAT_EQ(result_data[0], 10.0f);
}

TEST(Sum, Float64) {
  std::vector<double> data = {1.0, 2.0, 3.0, 4.0};
  quasai::core::Shape shape{2, 2};
  quasai::core::Tensor tensor = quasai::core::Tensor::from_data(
      data.data(), shape, quasai::core::DType::FLOAT64);

  quasai::core::Tensor result = quasai::ops::sum(tensor);
  double *result_data = result.data<double>();
  EXPECT_DOUBLE_EQ(result_data[0], 10.0);
}

TEST(Sum, Int32) {
  std::vector<int32_t> data = {1, 2, 3, 4};
  quasai::core::Shape shape{2, 2};
  quasai::core::Tensor tensor = quasai::core::Tensor::from_data(
      data.data(), shape, quasai::core::DType::INT32);

  quasai::core::Tensor result = quasai::ops::sum(tensor);
  int32_t *result_data = result.data<int32_t>();
  EXPECT_EQ(result_data[0], 10);
}

TEST(Sum, Int64) {
  std::vector<int64_t> data = {1, 2, 3, 4};
  quasai::core::Shape shape{2, 2};
  quasai::core::Tensor tensor = quasai::core::Tensor::from_data(
      data.data(), shape, quasai::core::DType::INT64);

  quasai::core::Tensor result = quasai::ops::sum(tensor);
  int64_t *result_data = result.data<int64_t>();
  EXPECT_EQ(result_data[0], 10);
}

TEST(Sum, Negative) {
  std::vector<float> data = {-1.0f, 2.0f, -3.0f, 4.0f};
  quasai::core::Shape shape{2, 2};
  quasai::core::Tensor tensor = quasai::core::Tensor::from_data(
      data.data(), shape, quasai::core::DType::FLOAT32);

  quasai::core::Tensor result = quasai::ops::sum(tensor);
  float *result_data = result.data<float>();
  EXPECT_FLOAT_EQ(result_data[0], 2.0f);
}

TEST(Sum, Gradient) {
  float eps = 1e-3f;
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  quasai::core::Shape shape{2, 2};
  quasai::core::Tensor input = quasai::core::Tensor::from_data(
      data.data(), shape, quasai::core::DType::FLOAT32);
  input.requires_grad(true);

  quasai::core::Tensor output = quasai::ops::sum(input);
  output.backward();

  quasai::core::Tensor grad = input.autograd_meta()->grad;
  float computed_grad_00 = grad.data<float>()[0];

  float f_plus = (1.0f + eps) + 2.0f + 3.0f + 4.0f;
  float f_minus = (1.0f - eps) + 2.0f + 3.0f + 4.0f;
  float finite_diff = (f_plus - f_minus) / (2.0f * eps);
  float rel_err = std::abs(computed_grad_00 - finite_diff) /
                  std::max(1.0f, std::abs(data[0]));

  EXPECT_NEAR(rel_err, 0.0f, 1e-2f);
}
