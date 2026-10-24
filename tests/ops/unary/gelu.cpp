#include "quasai/autograd/metadata.hpp"
#include "quasai/core/constants.hpp"
#include "quasai/ops/tensor_ops.hpp"
#include <cmath>
#include <gtest/gtest.h>

namespace {

float gelu_reference(float x) {
  double xd = static_cast<double>(x);
  return static_cast<float>(
      0.5 * xd *
      (1.0 + std::tanh(std::sqrt(2.0 / quasai::core::PI) *
                       (xd + 0.044715 * std::pow(xd, 3)))));
}

} // namespace

TEST(Gelu, Float32) {
  std::vector<float> data = {0.0f, 1.0f, -1.0f, 2.0f};
  quasai::core::Shape shape{2, 2};
  quasai::core::Tensor tensor = quasai::core::Tensor::from_data(
      data.data(), shape, quasai::core::DType::FLOAT32);

  quasai::core::Tensor result = quasai::ops::gelu(tensor);
  float *result_data = result.data<float>();
  for (std::size_t i = 0; i < data.size(); ++i) {
    EXPECT_FLOAT_EQ(result_data[i], gelu_reference(data[i]));
  }
}

TEST(Gelu, Float64) {
  std::vector<double> data = {0.0, 1.0, -1.0, 2.0};
  quasai::core::Shape shape{2, 2};
  quasai::core::Tensor tensor = quasai::core::Tensor::from_data(
      data.data(), shape, quasai::core::DType::FLOAT64);

  quasai::core::Tensor result = quasai::ops::gelu(tensor);
  double *result_data = result.data<double>();
  for (std::size_t i = 0; i < data.size(); ++i) {
    double x = data[i];
    double expected = 0.5 * x *
                      (1.0 + std::tanh(std::sqrt(2.0 / quasai::core::PI) *
                                       (x + 0.044715 * std::pow(x, 3))));
    EXPECT_DOUBLE_EQ(result_data[i], expected);
  }
}

TEST(Gelu, Gradient) {
  float eps = 1e-3f;
  float val = 0.5f;
  quasai::core::Tensor input = quasai::core::Tensor::from_data(
      &val, quasai::core::Shape{}, quasai::core::DType::FLOAT32);
  input.requires_grad(true);

  quasai::core::Tensor output = quasai::ops::gelu(input);
  output.backward();

  float computed_grad = input.autograd_meta()->grad_.data<float>()[0];
  float finite_diff =
      (gelu_reference(val + eps) - gelu_reference(val - eps)) / (2.0f * eps);
  float rel_err = std::abs(computed_grad - finite_diff) /
                  std::max(1.0f, std::abs(finite_diff));

  EXPECT_NEAR(rel_err, 0.0f, 1e-2f);
}
