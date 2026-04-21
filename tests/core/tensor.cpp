#include "quasai/core/tensor.hpp"
#include <gtest/gtest.h>
#include <vector>

TEST(TensorTest, ZerosCPU) {
  quasai::Shape shape{3, 4};
  quasai::Tensor tensor = quasai::Tensor::zeros(shape);

  const auto impl = tensor.get_impl();
  EXPECT_EQ(impl.shape[0], 3);
  EXPECT_EQ(impl.shape[1], 4);
  EXPECT_EQ(impl.dtype, quasai::DType::FLOAT32);
  EXPECT_EQ(impl.device.type, quasai::CPU);

  float *data = static_cast<float *>(impl.buffer->raw_data());
  for (size_t i = 0; i < 12; ++i) {
    EXPECT_FLOAT_EQ(data[i], 0.0f);
  }
}

TEST(TensorTest, ZerosDifferentDType) {
  quasai::Shape shape{2, 3};
  quasai::Tensor tensor = quasai::Tensor::zeros(shape, quasai::DType::FLOAT64);

  const auto impl = tensor.get_impl();
  EXPECT_EQ(impl.dtype, quasai::DType::FLOAT64);
  EXPECT_EQ(impl.shape[0], 2);
  EXPECT_EQ(impl.shape[1], 3);

  double *data = static_cast<double *>(impl.buffer->raw_data());
  for (size_t i = 0; i < 6; ++i) {
    EXPECT_DOUBLE_EQ(data[i], 0.0);
  }
}

TEST(TensorTest, EmptyCPU) {
  quasai::Shape shape{2, 2};
  quasai::Tensor tensor = quasai::Tensor::empty(shape);

  const auto impl = tensor.get_impl();
  EXPECT_EQ(impl.shape[0], 2);
  EXPECT_EQ(impl.shape[1], 2);
  EXPECT_NE(impl.buffer->raw_data(), nullptr);
}

TEST(TensorTest, FromData) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  quasai::Shape shape{2, 3};
  quasai::Tensor tensor =
      quasai::Tensor::from_data(data.data(), shape, quasai::DType::FLOAT32);

  const auto impl = tensor.get_impl();
  EXPECT_EQ(impl.shape[0], 2);
  EXPECT_EQ(impl.shape[1], 3);

  float *tensor_data = static_cast<float *>(impl.buffer->raw_data());
  for (size_t i = 0; i < 6; ++i) {
    EXPECT_FLOAT_EQ(tensor_data[i], data[i]);
  }
}

TEST(TensorTest, Reshape) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  quasai::Shape shape{2, 3};
  quasai::Tensor tensor =
      quasai::Tensor::from_data(data.data(), shape, quasai::DType::FLOAT32);

  tensor.reshape(quasai::Shape{3, 2});

  const auto impl = tensor.get_impl();
  EXPECT_EQ(impl.shape[0], 3);
  EXPECT_EQ(impl.shape[1], 2);

  float *tensor_data = static_cast<float *>(impl.buffer->raw_data());
  for (size_t i = 0; i < 6; ++i) {
    EXPECT_FLOAT_EQ(tensor_data[i], data[i]);
  }
}

TEST(TensorTest, ReshapeInvalid) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  quasai::Shape shape{2, 2};
  quasai::Tensor tensor =
      quasai::Tensor::from_data(data.data(), shape, quasai::DType::FLOAT32);

  EXPECT_THROW(tensor.reshape(quasai::Shape{3, 2}), std::runtime_error);
}

TEST(TensorTest, TensorView) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  quasai::Shape shape{2, 2};
  quasai::Tensor tensor =
      quasai::Tensor::from_data(data.data(), shape, quasai::DType::FLOAT32);

  const auto impl = tensor.get_impl();

  EXPECT_EQ(impl.shape[0], 2);
  EXPECT_EQ(impl.shape[1], 2);
  EXPECT_EQ(impl.dtype, quasai::DType::FLOAT32);
  EXPECT_EQ(impl.device.type, quasai::CPU);
  EXPECT_NE(impl.buffer->raw_data(), nullptr);
}

TEST(TensorTest, OneDimensional) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  quasai::Shape shape{5};
  quasai::Tensor tensor =
      quasai::Tensor::from_data(data.data(), shape, quasai::DType::FLOAT32);

  const auto impl = tensor.get_impl();
  EXPECT_EQ(impl.shape[0], 5);
  EXPECT_EQ(impl.shape.dimensions(), 1);

  float *tensor_data = static_cast<float *>(impl.buffer->raw_data());
  for (size_t i = 0; i < 5; ++i) {
    EXPECT_FLOAT_EQ(tensor_data[i], data[i]);
  }
}

TEST(TensorTest, ThreeDimensional) {
  std::vector<float> data(24, 1.5f);
  quasai::Shape shape{2, 3, 4};
  quasai::Tensor tensor =
      quasai::Tensor::from_data(data.data(), shape, quasai::DType::FLOAT32);

  const auto impl = tensor.get_impl();
  EXPECT_EQ(impl.shape[0], 2);
  EXPECT_EQ(impl.shape[1], 3);
  EXPECT_EQ(impl.shape[2], 4);
  EXPECT_EQ(impl.shape.dimensions(), 3);

  float *tensor_data = static_cast<float *>(impl.buffer->raw_data());
  for (size_t i = 0; i < 24; ++i) {
    EXPECT_FLOAT_EQ(tensor_data[i], 1.5f);
  }
}

TEST(TensorTest, Strides) {
  quasai::Shape shape{2, 3};
  quasai::Tensor tensor = quasai::Tensor::zeros(shape);

  const auto impl = tensor.get_impl();
  quasai::Strides strides = impl.strides;
  EXPECT_EQ(strides[0], 3);
  EXPECT_EQ(strides[1], 1);
}

TEST(TensorTest, ScalarTensor) {
  std::vector<float> data = {42.0f};
  quasai::Shape shape{};
  quasai::Tensor tensor =
      quasai::Tensor::from_data(data.data(), shape, quasai::DType::FLOAT32);

  const auto impl = tensor.get_impl();
  EXPECT_EQ(impl.shape.dimensions(), 0);
  EXPECT_FLOAT_EQ(*static_cast<float *>(impl.buffer->raw_data()), 42.0f);
}
