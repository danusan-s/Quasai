#include "quasai/autograd/metadata.hpp"
#include "quasai/core/constants.hpp"
#include "quasai/ops/tensor_ops.hpp"
#include <cmath>
#include <gtest/gtest.h>

TEST(Log, Float32) {
  std::vector<float> data = {0.5f, 1.0f, 2.0f, 4.0f};
  quasai::core::Shape shape{2, 2};
  quasai::core::Tensor tensor = quasai::core::Tensor::from_data(
      data.data(), shape, quasai::core::DType::FLOAT32);

  const float log_base = std::log(quasai::core::EULER);
  quasai::core::Tensor result = quasai::ops::log(tensor);
  float *result_data = result.data<float>();
  for (std::size_t i = 0; i < data.size(); ++i) {
    EXPECT_FLOAT_EQ(result_data[i],
                    std::log(data[i] + quasai::core::LOG_EPSILON) / log_base);
  }
}

TEST(Log, Float64) {
  std::vector<double> data = {0.5, 1.0, 2.0, 4.0};
  quasai::core::Shape shape{2, 2};
  quasai::core::Tensor tensor = quasai::core::Tensor::from_data(
      data.data(), shape, quasai::core::DType::FLOAT64);

  const float log_base = std::log(quasai::core::EULER);
  quasai::core::Tensor result = quasai::ops::log(tensor);
  double *result_data = result.data<double>();
  for (std::size_t i = 0; i < data.size(); ++i) {
    EXPECT_DOUBLE_EQ(result_data[i],
                     std::log(data[i] + quasai::core::LOG_EPSILON) / log_base);
  }
}

TEST(Log, CustomBase) {
  float base = 10.0f;
  std::vector<float> data = {1.0f, 10.0f, 100.0f, 1000.0f};
  quasai::core::Shape shape{2, 2};
  quasai::core::Tensor tensor = quasai::core::Tensor::from_data(
      data.data(), shape, quasai::core::DType::FLOAT32);

  quasai::core::Tensor result = quasai::ops::log(tensor, base);
  float *result_data = result.data<float>();
  for (std::size_t i = 0; i < data.size(); ++i) {
    EXPECT_FLOAT_EQ(result_data[i],
                    std::log(data[i] + quasai::core::LOG_EPSILON) /
                        std::log(base));
  }
}

TEST(Log, InvalidBaseShouldThrow) {
  float val = 1.0f;
  quasai::core::Tensor input = quasai::core::Tensor::from_data(
      &val, quasai::core::Shape{}, quasai::core::DType::FLOAT32);

  EXPECT_THROW(quasai::ops::log(input, 0.0f), std::runtime_error);
  EXPECT_THROW(quasai::ops::log(input, -1.0f), std::runtime_error);
  EXPECT_THROW(quasai::ops::log(input, 1.0f), std::runtime_error);
}

TEST(Log, Gradient) {
  float eps = 1e-3f;
  float val = 2.0f;
  quasai::core::Tensor input = quasai::core::Tensor::from_data(
      &val, quasai::core::Shape{}, quasai::core::DType::FLOAT32);
  input.requires_grad(true);

  quasai::core::Tensor output = quasai::ops::log(input);
  output.backward();

  float computed_grad = input.autograd_meta()->grad_.data<float>()[0];
  float f_plus = std::log(val + eps + quasai::core::LOG_EPSILON);
  float f_minus = std::log(val - eps + quasai::core::LOG_EPSILON);
  float finite_diff = (f_plus - f_minus) / (2.0f * eps);
  float rel_err = std::abs(computed_grad - finite_diff) /
                  std::max(1.0f, std::abs(finite_diff));

  EXPECT_NEAR(rel_err, 0.0f, 1e-2f);
}
