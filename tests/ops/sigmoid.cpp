#include "quasai/ops/tensor_ops.hpp"
#include "quasai/autograd/metadata.hpp"
#include <gtest/gtest.h>
#include <cmath>

TEST(Sigmoid, Float32) {
  std::vector<float> data = {0.0f, 1.0f, -1.0f, 2.0f};
  quasai::Shape shape{2, 2};
  quasai::Tensor tensor =
      quasai::Tensor::from_data(data.data(), shape, quasai::DType::FLOAT32);

  quasai::Tensor result = quasai::sigmoid(tensor);
  float *result_data = result.data<float>();
  EXPECT_FLOAT_EQ(result_data[0], 0.5f);
  EXPECT_FLOAT_EQ(result_data[1], 1.0f / (1.0f + std::exp(-1.0f)));
  EXPECT_FLOAT_EQ(result_data[2], 1.0f / (1.0f + std::exp(1.0f)));
  EXPECT_FLOAT_EQ(result_data[3], 1.0f / (1.0f + std::exp(-2.0f)));
}

TEST(Sigmoid, Float64) {
  std::vector<double> data = {0.0, 1.0, -1.0, 2.0};
  quasai::Shape shape{2, 2};
  quasai::Tensor tensor =
      quasai::Tensor::from_data(data.data(), shape, quasai::DType::FLOAT64);

  quasai::Tensor result = quasai::sigmoid(tensor);
  double *result_data = result.data<double>();
  EXPECT_DOUBLE_EQ(result_data[0], 0.5);
  EXPECT_DOUBLE_EQ(result_data[1], 1.0 / (1.0 + std::exp(-1.0)));
  EXPECT_DOUBLE_EQ(result_data[2], 1.0 / (1.0 + std::exp(1.0)));
  EXPECT_DOUBLE_EQ(result_data[3], 1.0 / (1.0 + std::exp(-2.0)));
}

TEST(Sigmoid, Gradient) {
  float eps = 1e-3f;
  float val = 0.5f;
  quasai::Tensor input = quasai::Tensor::from_data(&val, quasai::Shape{}, quasai::DType::FLOAT32);
  input.requires_grad(true);

  quasai::Tensor output = quasai::sigmoid(input);
  output.backward();

  quasai::Tensor grad = input.autograd_meta()->grad;
  float computed_grad = grad.data<float>()[0];

  auto sigmoid = [](float x) { return 1.0f / (1.0f + std::exp(-x)); };
  float f_plus = sigmoid(val + eps);
  float f_minus = sigmoid(val - eps);
  float finite_diff = (f_plus - f_minus) / (2.0f * eps);
  float rel_err = std::abs(computed_grad - finite_diff) / std::max(1.0f, std::abs(val));

  EXPECT_NEAR(rel_err, 0.0f, 1e-2f);
}
