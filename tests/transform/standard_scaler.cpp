#include "quasai/transform/standard_scaler.hpp"
#include <cmath>
#include <gtest/gtest.h>

TEST(StandardScaler, FitComputesMeanAndStd) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  quasai::core::Shape shape{2, 3};
  quasai::core::Tensor tensor = quasai::core::Tensor::from_data(
      data.data(), shape, quasai::core::DType::FLOAT32);

  quasai::transform::StandardScaler scaler;
  scaler.fit(tensor);

  float *mean = scaler.mean_.data<float>();
  float *std = scaler.std_.data<float>();

  EXPECT_FLOAT_EQ(mean[0], 2.5f);
  EXPECT_FLOAT_EQ(mean[1], 3.5f);
  EXPECT_FLOAT_EQ(mean[2], 4.5f);

  EXPECT_FLOAT_EQ(std[0], 1.5f);
  EXPECT_FLOAT_EQ(std[1], 1.5f);
  EXPECT_FLOAT_EQ(std[2], 1.5f);
}

TEST(StandardScaler, Transform) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  quasai::core::Shape shape{2, 3};
  quasai::core::Tensor tensor = quasai::core::Tensor::from_data(
      data.data(), shape, quasai::core::DType::FLOAT32);

  quasai::transform::StandardScaler scaler;
  scaler.fit(tensor);

  quasai::core::Tensor result = scaler.transform(tensor);
  float *result_data = result.data<float>();

  EXPECT_FLOAT_EQ(result_data[0], (1.0f - 2.5f) / 1.5f);
  EXPECT_FLOAT_EQ(result_data[1], (2.0f - 3.5f) / 1.5f);
  EXPECT_FLOAT_EQ(result_data[2], (3.0f - 4.5f) / 1.5f);
  EXPECT_FLOAT_EQ(result_data[3], (4.0f - 2.5f) / 1.5f);
  EXPECT_FLOAT_EQ(result_data[4], (5.0f - 3.5f) / 1.5f);
  EXPECT_FLOAT_EQ(result_data[5], (6.0f - 4.5f) / 1.5f);
}

TEST(StandardScaler, InverseTransform) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  quasai::core::Shape shape{2, 3};
  quasai::core::Tensor tensor = quasai::core::Tensor::from_data(
      data.data(), shape, quasai::core::DType::FLOAT32);

  quasai::transform::StandardScaler scaler;
  scaler.fit(tensor);

  quasai::core::Tensor transformed = scaler.transform(tensor);
  quasai::core::Tensor restored = scaler.inverse_transform(transformed);
  float *restored_data = restored.data<float>();

  for (size_t i = 0; i < 6; ++i) {
    EXPECT_FLOAT_EQ(restored_data[i], data[i]);
  }
}

TEST(StandardScaler, FitThrowsOnNot2D) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  quasai::core::Shape shape{4};
  quasai::core::Tensor tensor = quasai::core::Tensor::from_data(
      data.data(), shape, quasai::core::DType::FLOAT32);

  quasai::transform::StandardScaler scaler;
  EXPECT_THROW(scaler.fit(tensor), std::runtime_error);
}

TEST(StandardScaler, TransformThrowsIfNotFitted) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  quasai::core::Shape shape{2, 2};
  quasai::core::Tensor tensor = quasai::core::Tensor::from_data(
      data.data(), shape, quasai::core::DType::FLOAT32);

  quasai::transform::StandardScaler scaler;
  EXPECT_THROW(scaler.transform(tensor), std::runtime_error);
}

TEST(StandardScaler, ZeroVarianceFeature) {
  std::vector<float> data = {5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f};
  quasai::core::Shape shape{2, 3};
  quasai::core::Tensor tensor = quasai::core::Tensor::from_data(
      data.data(), shape, quasai::core::DType::FLOAT32);

  quasai::transform::StandardScaler scaler;
  scaler.fit(tensor);

  quasai::core::Tensor result = scaler.transform(tensor);
  float *result_data = result.data<float>();

  for (size_t i = 0; i < 6; ++i) {
    EXPECT_FLOAT_EQ(result_data[i], 0.0f);
  }
}
