#include "quasai/autograd/metadata.hpp"
#include "quasai/ops/tensor_ops.hpp"
#include <cmath>
#include <gtest/gtest.h>

TEST(Matmul, Float32) {
  std::vector<float> data_a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<float> data_b = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
  quasai::core::Shape shape_a{2, 3};
  quasai::core::Shape shape_b{3, 2};
  quasai::core::Tensor tensor_a = quasai::core::Tensor::from_data(
      data_a.data(), shape_a, quasai::core::DType::FLOAT32);
  quasai::core::Tensor tensor_b = quasai::core::Tensor::from_data(
      data_b.data(), shape_b, quasai::core::DType::FLOAT32);

  quasai::core::Tensor result = quasai::ops::matmul(tensor_a, tensor_b);
  float *result_data = result.data<float>();
  EXPECT_FLOAT_EQ(result_data[0], 58.0f);
  EXPECT_FLOAT_EQ(result_data[1], 64.0f);
  EXPECT_FLOAT_EQ(result_data[2], 139.0f);
  EXPECT_FLOAT_EQ(result_data[3], 154.0f);
}

TEST(Matmul, GradientA) {
  float eps = 1e-3f;
  std::vector<float> data_a = {1.0f, 2.0f, 3.0f};
  std::vector<float> data_b = {4.0f, 5.0f, 6.0f};
  quasai::core::Shape shape_a{1, 3};
  quasai::core::Shape shape_b{3, 1};
  quasai::core::Tensor input_a = quasai::core::Tensor::from_data(
      data_a.data(), shape_a, quasai::core::DType::FLOAT32);
  quasai::core::Tensor input_b = quasai::core::Tensor::from_data(
      data_b.data(), shape_b, quasai::core::DType::FLOAT32);
  input_a.requires_grad(true);
  input_b.requires_grad(true);

  quasai::core::Tensor output = quasai::ops::matmul(input_a, input_b);
  output.backward();

  quasai::core::Tensor grad_a = input_a.autograd_meta()->grad;
  float computed_grad_00 = grad_a.data<float>()[0];

  std::vector<float> data_a_plus = {1.0f + eps, 2.0f, 3.0f};
  std::vector<float> data_a_minus = {1.0f - eps, 2.0f, 3.0f};
  quasai::core::Tensor input_a_plus = quasai::core::Tensor::from_data(
      data_a_plus.data(), shape_a, quasai::core::DType::FLOAT32);
  quasai::core::Tensor input_a_minus = quasai::core::Tensor::from_data(
      data_a_minus.data(), shape_a, quasai::core::DType::FLOAT32);

  quasai::core::Tensor out_plus = quasai::ops::matmul(input_a_plus, input_b);
  quasai::core::Tensor out_minus = quasai::ops::matmul(input_a_minus, input_b);
  float f_plus = out_plus.data<float>()[0];
  float f_minus = out_minus.data<float>()[0];
  float finite_diff = (f_plus - f_minus) / (2.0f * eps);
  float rel_err = std::abs(computed_grad_00 - finite_diff) /
                  std::max(1.0f, std::abs(data_a[0]));

  EXPECT_NEAR(rel_err, 0.0f, 1e-2f);
}
