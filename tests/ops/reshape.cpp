#include "quasai/ops/tensor_ops.hpp"
#include "quasai/autograd/metadata.hpp"
#include <gtest/gtest.h>
#include <cmath>

TEST(Reshape, Float32) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  quasai::Shape shape{2, 3};
  quasai::Tensor tensor =
      quasai::Tensor::from_data(data.data(), shape, quasai::DType::FLOAT32);

  quasai::Tensor result = quasai::reshape(tensor, quasai::Shape{3, 2});
  float *result_data = result.data<float>();
  EXPECT_EQ(result.shape()[0], 3);
  EXPECT_EQ(result.shape()[1], 2);
  EXPECT_FLOAT_EQ(result_data[0], 1.0f);
  EXPECT_FLOAT_EQ(result_data[1], 2.0f);
  EXPECT_FLOAT_EQ(result_data[2], 3.0f);
  EXPECT_FLOAT_EQ(result_data[3], 4.0f);
  EXPECT_FLOAT_EQ(result_data[4], 5.0f);
  EXPECT_FLOAT_EQ(result_data[5], 6.0f);
}

TEST(Reshape, Float64) {
  std::vector<double> data = {1.0, 2.0, 3.0, 4.0};
  quasai::Shape shape{2, 2};
  quasai::Tensor tensor =
      quasai::Tensor::from_data(data.data(), shape, quasai::DType::FLOAT64);

  quasai::Tensor result = quasai::reshape(tensor, quasai::Shape{4});
  EXPECT_EQ(result.shape()[0], 4);
}

TEST(Reshape, Int32) {
  std::vector<int32_t> data = {1, 2, 3, 4, 5, 6};
  quasai::Shape shape{2, 3};
  quasai::Tensor tensor =
      quasai::Tensor::from_data(data.data(), shape, quasai::DType::INT32);

  quasai::Tensor result = quasai::reshape(tensor, quasai::Shape{6});
  int32_t *result_data = result.data<int32_t>();
  EXPECT_EQ(result.shape()[0], 6);
  EXPECT_EQ(result_data[0], 1);
  EXPECT_EQ(result_data[5], 6);
}

TEST(Reshape, InvalidShape) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f};
  quasai::Shape shape{1, 3};
  quasai::Tensor tensor =
      quasai::Tensor::from_data(data.data(), shape, quasai::DType::FLOAT32);

  EXPECT_THROW(quasai::reshape(tensor, quasai::Shape{2, 2}), std::runtime_error);
}

TEST(Reshape, Gradient) {
  float eps = 1e-3f;
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  quasai::Shape shape{2, 2};
  quasai::Tensor input = quasai::Tensor::from_data(data.data(), shape, quasai::DType::FLOAT32);
  input.requires_grad(true);

  quasai::Tensor output = quasai::reshape(input, quasai::Shape{4});
  output.backward();

  quasai::Tensor grad = input.autograd_meta()->grad;
  float computed_grad_00 = grad.data<float>()[0];

  std::vector<float> data_plus = {1.0f + eps, 2.0f, 3.0f, 4.0f};
  std::vector<float> data_minus = {1.0f - eps, 2.0f, 3.0f, 4.0f};
  quasai::Tensor input_plus = quasai::Tensor::from_data(data_plus.data(), shape, quasai::DType::FLOAT32);
  quasai::Tensor input_minus = quasai::Tensor::from_data(data_minus.data(), shape, quasai::DType::FLOAT32);

  quasai::Tensor out_plus = quasai::reshape(input_plus, quasai::Shape{4});
  quasai::Tensor out_minus = quasai::reshape(input_minus, quasai::Shape{4});
  float f_plus = out_plus.data<float>()[0];
  float f_minus = out_minus.data<float>()[0];
  float finite_diff = (f_plus - f_minus) / (2.0f * eps);
  float rel_err = std::abs(computed_grad_00 - finite_diff) / std::max(1.0f, std::abs(data[0]));

  EXPECT_NEAR(rel_err, 0.0f, 1e-2f);
}