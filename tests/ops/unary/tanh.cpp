#include "quasai/autograd/metadata.hpp"
#include "quasai/ops/tensor_ops.hpp"
#include <cmath>
#include <gtest/gtest.h>

TEST(Tanh, Float32) {
  std::vector<float> data = {0.0f, 1.0f, -1.0f, 2.0f};
  quasai::core::Shape shape{2, 2};
  quasai::core::Tensor tensor = quasai::core::Tensor::from_data(
      data.data(), shape, quasai::core::DType::FLOAT32);

  quasai::core::Tensor result = quasai::ops::tanh(tensor);
  float *result_data = result.data<float>();
  EXPECT_FLOAT_EQ(result_data[0], std::tanh(0.0f));
  EXPECT_FLOAT_EQ(result_data[1], std::tanh(1.0f));
  EXPECT_FLOAT_EQ(result_data[2], std::tanh(-1.0f));
  EXPECT_FLOAT_EQ(result_data[3], std::tanh(2.0f));
}

TEST(Tanh, Float64) {
  std::vector<double> data = {0.0, 1.0, -1.0, 2.0};
  quasai::core::Shape shape{2, 2};
  quasai::core::Tensor tensor = quasai::core::Tensor::from_data(
      data.data(), shape, quasai::core::DType::FLOAT64);

  quasai::core::Tensor result = quasai::ops::tanh(tensor);
  double *result_data = result.data<double>();
  EXPECT_DOUBLE_EQ(result_data[0], std::tanh(0.0));
  EXPECT_DOUBLE_EQ(result_data[1], std::tanh(1.0));
  EXPECT_DOUBLE_EQ(result_data[2], std::tanh(-1.0));
  EXPECT_DOUBLE_EQ(result_data[3], std::tanh(2.0));
}

TEST(Tanh, Gradient) {
  float eps = 1e-3f;
  float val = 0.5f;
  quasai::core::Tensor input = quasai::core::Tensor::from_data(
      &val, quasai::core::Shape{}, quasai::core::DType::FLOAT32);
  input.requires_grad(true);

  quasai::core::Tensor output = quasai::ops::tanh(input);
  output.backward();

  quasai::core::Tensor grad = input.autograd_meta()->grad;
  float computed_grad = grad.data<float>()[0];

  float f_plus = std::tanh(val + eps);
  float f_minus = std::tanh(val - eps);
  float finite_diff = (f_plus - f_minus) / (2.0f * eps);
  float rel_err =
      std::abs(computed_grad - finite_diff) / std::max(1.0f, std::abs(val));

  EXPECT_NEAR(rel_err, 0.0f, 1e-2f);
}
