#include "quasai/ops/tensor_ops.hpp"
#include "quasai/autograd/metadata.hpp"
#include <gtest/gtest.h>
#include <cmath>

TEST(Heaviside, Float32) {
  std::vector<float> data = {0.0f, 1.0f, -1.0f, 2.0f};
  quasai::Shape shape{2, 2};
  quasai::Tensor tensor =
      quasai::Tensor::from_data(data.data(), shape, quasai::DType::FLOAT32);

  quasai::Tensor result = quasai::heaviside(tensor);
  float *result_data = result.data<float>();
  EXPECT_FLOAT_EQ(result_data[0], 0.0f);
  EXPECT_FLOAT_EQ(result_data[1], 1.0f);
  EXPECT_FLOAT_EQ(result_data[2], 0.0f);
  EXPECT_FLOAT_EQ(result_data[3], 1.0f);
}

TEST(Heaviside, Float64) {
  std::vector<double> data = {0.0, 1.0, -1.0, 2.0};
  quasai::Shape shape{2, 2};
  quasai::Tensor tensor =
      quasai::Tensor::from_data(data.data(), shape, quasai::DType::FLOAT64);

  quasai::Tensor result = quasai::heaviside(tensor);
  double *result_data = result.data<double>();
  EXPECT_DOUBLE_EQ(result_data[0], 0.0);
  EXPECT_DOUBLE_EQ(result_data[1], 1.0);
  EXPECT_DOUBLE_EQ(result_data[2], 0.0);
  EXPECT_DOUBLE_EQ(result_data[3], 1.0);
}

TEST(Heaviside, Gradient) {
  float val = 2.0f;
  quasai::Tensor input = quasai::Tensor::from_data(&val, quasai::Shape{}, quasai::DType::FLOAT32);
  input.requires_grad(true);

  quasai::Tensor output = quasai::heaviside(input);
  output.backward();

  quasai::Tensor grad = input.autograd_meta()->grad;
  float computed_grad = grad.data<float>()[0];

  EXPECT_FLOAT_EQ(computed_grad, 0.0f);
}
