#include "quasai/ops/tensor_ops.hpp"
#include "quasai/autograd/metadata.hpp"
#include <gtest/gtest.h>
#include <cmath>

TEST(Signum, Float32) {
  std::vector<float> data = {0.0f, 1.0f, -1.0f, 2.0f};
  quasai::core::Shape shape{2, 2};
  quasai::core::Tensor tensor =
      quasai::core::Tensor::from_data(data.data(), shape, quasai::core::DType::FLOAT32);

  quasai::core::Tensor result = quasai::ops::signum(tensor);
  float *result_data = result.data<float>();
  EXPECT_FLOAT_EQ(result_data[0], 0.0f);
  EXPECT_FLOAT_EQ(result_data[1], 1.0f);
  EXPECT_FLOAT_EQ(result_data[2], -1.0f);
  EXPECT_FLOAT_EQ(result_data[3], 1.0f);
}

TEST(Signum, Float64) {
  std::vector<double> data = {0.0, 1.0, -1.0, 2.0};
  quasai::core::Shape shape{2, 2};
  quasai::core::Tensor tensor =
      quasai::core::Tensor::from_data(data.data(), shape, quasai::core::DType::FLOAT64);

  quasai::core::Tensor result = quasai::ops::signum(tensor);
  double *result_data = result.data<double>();
  EXPECT_DOUBLE_EQ(result_data[0], 0.0);
  EXPECT_DOUBLE_EQ(result_data[1], 1.0);
  EXPECT_DOUBLE_EQ(result_data[2], -1.0);
  EXPECT_DOUBLE_EQ(result_data[3], 1.0);
}

TEST(Signum, Gradient) {
  float val = 2.0f;
  quasai::core::Tensor input = quasai::core::Tensor::from_data(&val, quasai::core::Shape{}, quasai::core::DType::FLOAT32);
  input.requires_grad(true);

  quasai::core::Tensor output = quasai::ops::signum(input);
  output.backward();

  quasai::core::Tensor grad = input.autograd_meta()->grad;
  float computed_grad = grad.data<float>()[0];

  EXPECT_FLOAT_EQ(computed_grad, 0.0f);
}
