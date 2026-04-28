#include "quasai/autograd/metadata.hpp"
#include "quasai/ops/tensor_ops.hpp"
#include <cmath>
#include <gtest/gtest.h>

TEST(Transpose, Float32) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  quasai::core::Shape shape{2, 3};
  quasai::core::Tensor tensor = quasai::core::Tensor::from_data(
      data.data(), shape, quasai::core::DType::FLOAT32);

  quasai::core::Tensor result = quasai::ops::transpose(tensor);
  const auto impl = result.get_impl_copy();
  const auto ref_impl = tensor.get_impl_copy();

  EXPECT_EQ(impl.shape[0], ref_impl.shape[1]);
  EXPECT_EQ(impl.shape[1], ref_impl.shape[0]);
  EXPECT_EQ(impl.strides[0], ref_impl.strides[1]);
  EXPECT_EQ(impl.strides[1], ref_impl.strides[0]);
  EXPECT_EQ(impl.dtype, ref_impl.dtype);
  EXPECT_EQ(impl.device.type, ref_impl.device.type);
}

TEST(Transpose, Float64) {
  std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  quasai::core::Shape shape{2, 3};
  quasai::core::Tensor tensor = quasai::core::Tensor::from_data(
      data.data(), shape, quasai::core::DType::FLOAT64);

  quasai::core::Tensor result = quasai::ops::transpose(tensor);
  const auto impl = result.get_impl_copy();
  const auto ref_impl = tensor.get_impl_copy();

  EXPECT_EQ(impl.shape[0], ref_impl.shape[1]);
  EXPECT_EQ(impl.shape[1], ref_impl.shape[0]);
  EXPECT_EQ(impl.strides[0], ref_impl.strides[1]);
  EXPECT_EQ(impl.strides[1], ref_impl.strides[0]);
  EXPECT_EQ(impl.dtype, ref_impl.dtype);
  EXPECT_EQ(impl.device.type, ref_impl.device.type);
}

TEST(Transpose, AddWithTranspose) {
  std::vector<float> data_a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<float> data_b = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f};
  quasai::core::Shape shape{2, 3};
  quasai::core::Shape transposed_shape{3, 2};
  quasai::core::Tensor tensor_a = quasai::core::Tensor::from_data(
      data_a.data(), shape, quasai::core::DType::FLOAT32);
  quasai::core::Tensor tensor_b = quasai::core::Tensor::from_data(
      data_b.data(), transposed_shape, quasai::core::DType::FLOAT32);

  quasai::core::Tensor result =
      quasai::ops::add(tensor_a, quasai::ops::transpose(tensor_b));
  float *result_data = result.data<float>();
  EXPECT_FLOAT_EQ(result_data[0], 11.0f);
  EXPECT_FLOAT_EQ(result_data[1], 32.0f);
  EXPECT_FLOAT_EQ(result_data[2], 53.0f);
  EXPECT_FLOAT_EQ(result_data[3], 24.0f);
  EXPECT_FLOAT_EQ(result_data[4], 45.0f);
  EXPECT_FLOAT_EQ(result_data[5], 66.0f);
}

TEST(Transpose, Gradient) {
  float eps = 1e-3f;
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  quasai::core::Shape shape{2, 2};
  quasai::core::Tensor input = quasai::core::Tensor::from_data(
      data.data(), shape, quasai::core::DType::FLOAT32);
  input.requires_grad(true);

  quasai::core::Tensor output = quasai::ops::transpose(input);
  output.backward();

  quasai::core::Tensor grad = input.autograd_meta()->grad;
  float computed_grad_00 = grad.data<float>()[0];

  std::vector<float> data_plus = {1.0f + eps, 2.0f, 3.0f, 4.0f};
  std::vector<float> data_minus = {1.0f - eps, 2.0f, 3.0f, 4.0f};
  quasai::core::Tensor input_plus = quasai::core::Tensor::from_data(
      data_plus.data(), shape, quasai::core::DType::FLOAT32);
  quasai::core::Tensor input_minus = quasai::core::Tensor::from_data(
      data_minus.data(), shape, quasai::core::DType::FLOAT32);

  quasai::core::Tensor out_plus = quasai::ops::transpose(input_plus);
  quasai::core::Tensor out_minus = quasai::ops::transpose(input_minus);
  float f_plus = out_plus.data<float>()[0];
  float f_minus = out_minus.data<float>()[0];
  float finite_diff = (f_plus - f_minus) / (2.0f * eps);
  float rel_err = std::abs(computed_grad_00 - finite_diff) /
                  std::max(1.0f, std::abs(data[0]));

  EXPECT_NEAR(rel_err, 0.0f, 1e-2f);
}
