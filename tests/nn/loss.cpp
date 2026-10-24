#include "quasai/nn/loss.hpp"
#include "quasai/autograd/metadata.hpp"
#include "quasai/core/constants.hpp"
#include "quasai/ops/tensor_ops.hpp"
#include <cmath>
#include <gtest/gtest.h>

TEST(Loss, CrossEntropy) {
  std::vector<float> predictions_data = {0.7f, 0.2f, 0.1f, 0.1f, 0.8f, 0.1f};
  std::vector<float> targets_data = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
  quasai::core::Shape shape{2, 3};

  quasai::core::Tensor predictions = quasai::core::Tensor::from_data(
      predictions_data.data(), shape, quasai::core::DType::FLOAT32);
  quasai::core::Tensor targets = quasai::core::Tensor::from_data(
      targets_data.data(), shape, quasai::core::DType::FLOAT32);

  quasai::core::Tensor loss =
      quasai::nn::cross_entropy_loss(predictions, targets);

  float expected = 0.0f;
  for (std::size_t i = 0; i < predictions_data.size(); ++i) {
    expected += targets_data[i] *
                std::log(predictions_data[i] + quasai::core::LOG_EPSILON);
  }
  expected = -expected / static_cast<float>(predictions_data.size());

  EXPECT_FLOAT_EQ(loss.data<float>()[0], expected);
}

TEST(Loss, CrossEntropyGradient) {
  float eps = 1e-3f;
  float val = 0.7f;
  float target = 1.0f;

  quasai::core::Tensor predictions = quasai::core::Tensor::from_data(
      &val, quasai::core::Shape{}, quasai::core::DType::FLOAT32);
  predictions.requires_grad(true);
  quasai::core::Tensor targets = quasai::core::Tensor::from_data(
      &target, quasai::core::Shape{}, quasai::core::DType::FLOAT32);

  quasai::core::Tensor loss =
      quasai::nn::cross_entropy_loss(predictions, targets);
  loss.backward();

  float computed_grad = predictions.autograd_meta()->grad_.data<float>()[0];
  float f_plus = -std::log(val + eps + quasai::core::LOG_EPSILON);
  float f_minus = -std::log(val - eps + quasai::core::LOG_EPSILON);
  float finite_diff = (f_plus - f_minus) / (2.0f * eps);
  float rel_err = std::abs(computed_grad - finite_diff) /
                  std::max(1.0f, std::abs(finite_diff));

  EXPECT_NEAR(rel_err, 0.0f, 1e-2f);
}

TEST(Loss, CrossEntropyShapeMismatchShouldThrow) {
  std::vector<float> predictions_data = {0.7f, 0.3f};
  std::vector<float> targets_data = {1.0f};
  quasai::core::Tensor predictions = quasai::core::Tensor::from_data(
      predictions_data.data(), quasai::core::Shape{2},
      quasai::core::DType::FLOAT32);
  quasai::core::Tensor targets = quasai::core::Tensor::from_data(
      targets_data.data(), quasai::core::Shape{1},
      quasai::core::DType::FLOAT32);

  EXPECT_THROW(quasai::nn::cross_entropy_loss(predictions, targets),
               std::runtime_error);
}

TEST(Loss, ComputeLossDispatchesCrossEntropy) {
  std::vector<float> predictions_data = {0.7f, 0.3f};
  std::vector<float> targets_data = {1.0f, 0.0f};
  quasai::core::Tensor predictions = quasai::core::Tensor::from_data(
      predictions_data.data(), quasai::core::Shape{2},
      quasai::core::DType::FLOAT32);
  quasai::core::Tensor targets = quasai::core::Tensor::from_data(
      targets_data.data(), quasai::core::Shape{2},
      quasai::core::DType::FLOAT32);

  EXPECT_NO_THROW(quasai::nn::compute_loss(predictions, targets,
                                           quasai::nn::Loss::CROSS_ENTROPY));
}
