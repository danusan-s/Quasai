#include "quasai/ops/tensor_ops.hpp"
#include "quasai/autograd/metadata.hpp"
#include <gtest/gtest.h>
#include <cmath>

TEST(Transpose, Float32) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  quasai::Shape shape{2, 3};
  quasai::Tensor tensor =
      quasai::Tensor::from_data(data.data(), shape, quasai::DType::FLOAT32);

  quasai::Tensor result = quasai::transpose(tensor);
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
  quasai::Shape shape{2, 3};
  quasai::Tensor tensor =
      quasai::Tensor::from_data(data.data(), shape, quasai::DType::FLOAT64);

  quasai::Tensor result = quasai::transpose(tensor);
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
  quasai::Shape shape{2, 3};
  quasai::Shape transposed_shape{3, 2};
  quasai::Tensor tensor_a =
      quasai::Tensor::from_data(data_a.data(), shape, quasai::DType::FLOAT32);
  quasai::Tensor tensor_b = quasai::Tensor::from_data(
      data_b.data(), transposed_shape, quasai::DType::FLOAT32);

  quasai::Tensor result = quasai::add(tensor_a, quasai::transpose(tensor_b));
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
  quasai::Shape shape{2, 2};
  quasai::Tensor input = quasai::Tensor::from_data(data.data(), shape, quasai::DType::FLOAT32);
  input.requires_grad(true);

  quasai::Tensor output = quasai::transpose(input);
  output.backward();

  quasai::Tensor grad = input.autograd_meta()->grad;
  float computed_grad_00 = grad.data<float>()[0];

  std::vector<float> data_plus = {1.0f + eps, 2.0f, 3.0f, 4.0f};
  std::vector<float> data_minus = {1.0f - eps, 2.0f, 3.0f, 4.0f};
  quasai::Tensor input_plus = quasai::Tensor::from_data(data_plus.data(), shape, quasai::DType::FLOAT32);
  quasai::Tensor input_minus = quasai::Tensor::from_data(data_minus.data(), shape, quasai::DType::FLOAT32);

  quasai::Tensor out_plus = quasai::transpose(input_plus);
  quasai::Tensor out_minus = quasai::transpose(input_minus);
  float f_plus = out_plus.data<float>()[0];
  float f_minus = out_minus.data<float>()[0];
  float finite_diff = (f_plus - f_minus) / (2.0f * eps);
  float rel_err = std::abs(computed_grad_00 - finite_diff) / std::max(1.0f, std::abs(data[0]));

  EXPECT_NEAR(rel_err, 0.0f, 1e-2f);
}
