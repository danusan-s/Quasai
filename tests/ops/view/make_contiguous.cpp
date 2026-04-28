#include "quasai/autograd/metadata.hpp"
#include "quasai/ops/tensor_ops.hpp"
#include <cmath>
#include <gtest/gtest.h>

TEST(MakeContiguous, Float32) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  quasai::core::Shape shape{2, 3};
  quasai::core::Tensor tensor =
      quasai::core::Tensor::from_data(data.data(), shape, quasai::core::DType::FLOAT32);

  quasai::core::Tensor transposed = quasai::ops::transpose(tensor);
  EXPECT_FALSE(transposed.is_contiguous());

  quasai::core::Tensor result = quasai::ops::make_contiguous(transposed);
  EXPECT_TRUE(result.is_contiguous());

  float *result_data = result.data<float>();
  EXPECT_FLOAT_EQ(result_data[0], 1.0f);
  EXPECT_FLOAT_EQ(result_data[1], 4.0f);
  EXPECT_FLOAT_EQ(result_data[2], 2.0f);
  EXPECT_FLOAT_EQ(result_data[3], 5.0f);
  EXPECT_FLOAT_EQ(result_data[4], 3.0f);
  EXPECT_FLOAT_EQ(result_data[5], 6.0f);
}

TEST(MakeContiguous, Float64) {
  std::vector<double> data = {1.0, 2.0, 3.0, 4.0};
  quasai::core::Shape shape{2, 2};
  quasai::core::Tensor tensor =
      quasai::core::Tensor::from_data(data.data(), shape, quasai::core::DType::FLOAT64);

  quasai::core::Tensor transposed = quasai::ops::transpose(tensor);
  EXPECT_FALSE(transposed.is_contiguous());

  quasai::core::Tensor result = quasai::ops::make_contiguous(transposed);
  EXPECT_TRUE(result.is_contiguous());

  double *result_data = result.data<double>();
  EXPECT_DOUBLE_EQ(result_data[0], 1.0);
  EXPECT_DOUBLE_EQ(result_data[1], 3.0);
  EXPECT_DOUBLE_EQ(result_data[2], 2.0);
  EXPECT_DOUBLE_EQ(result_data[3], 4.0);
}

TEST(MakeContiguous, TransposeThenSlice) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  quasai::core::Shape shape{4, 2};
  quasai::core::Tensor tensor =
      quasai::core::Tensor::from_data(data.data(), shape, quasai::core::DType::FLOAT32);

  quasai::core::Tensor transposed = quasai::ops::transpose(tensor);
  EXPECT_FALSE(transposed.is_contiguous());

  quasai::core::Tensor sliced = quasai::ops::slice(transposed, 1, 2);
  EXPECT_TRUE(sliced.is_contiguous());

  float *sliced_data = sliced.data<float>();
  EXPECT_EQ(sliced.shape()[0], 1);
  EXPECT_EQ(sliced.shape()[1], 4);
  EXPECT_FLOAT_EQ(sliced_data[0], 2.0f);
  EXPECT_FLOAT_EQ(sliced_data[1], 4.0f);
  EXPECT_FLOAT_EQ(sliced_data[2], 6.0f);
  EXPECT_FLOAT_EQ(sliced_data[3], 8.0f);
}

TEST(MakeContiguous, Gradient) {
  float eps = 1e-3f;
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  quasai::core::Shape shape{2, 2};
  quasai::core::Tensor input =
      quasai::core::Tensor::from_data(data.data(), shape, quasai::core::DType::FLOAT32);
  input.requires_grad(true);

  quasai::core::Tensor output = quasai::ops::make_contiguous(input);
  output.backward();

  quasai::core::Tensor grad = input.autograd_meta()->grad;
  float computed_grad_00 = grad.data<float>()[0];

  std::vector<float> data_plus = {1.0f + eps, 2.0f, 3.0f, 4.0f};
  std::vector<float> data_minus = {1.0f - eps, 2.0f, 3.0f, 4.0f};
quasai::core::Tensor input_plus = quasai::core::Tensor::from_data(data_plus.data(), shape,
                                                         quasai::core::DType::FLOAT32);
  quasai::core::Tensor input_minus = quasai::core::Tensor::from_data(
      data_minus.data(), shape, quasai::core::DType::FLOAT32);

  quasai::core::Tensor out_plus = quasai::ops::make_contiguous(input_plus);
  quasai::core::Tensor out_minus = quasai::ops::make_contiguous(input_minus);
  float f_plus = out_plus.data<float>()[0];
  float f_minus = out_minus.data<float>()[0];
  float finite_diff = (f_plus - f_minus) / (2.0f * eps);
  float rel_err = std::abs(computed_grad_00 - finite_diff) /
                  std::max(1.0f, std::abs(data[0]));

  EXPECT_NEAR(rel_err, 0.0f, 1e-2f);
}
