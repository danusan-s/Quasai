#include "quasai/nn/modules/layer_norm.hpp"
#include "quasai/autograd/metadata.hpp"
#include "quasai/ops/tensor_ops.hpp"
#include <cmath>
#include <gtest/gtest.h>

TEST(LayerNorm, ForwardNormalizesEachRow) {
  std::vector<float> data = {0.1f, 0.2f, 0.3f, 0.4f, 1.0f, 2.0f, 3.0f, 4.0f};
  quasai::core::Tensor input = quasai::core::Tensor::from_data(
      data.data(), quasai::core::Shape{2, 4}, quasai::core::DType::FLOAT32);

  quasai::nn::LayerNorm layer_norm(4);
  quasai::core::Tensor output = layer_norm.forward(input);

  ASSERT_EQ(output.shape(), input.shape());
  const float *output_data = output.data<float>();

  for (size_t row = 0; row < 2; ++row) {
    float sum = 0.0f;
    float sum_sq = 0.0f;
    for (size_t col = 0; col < 4; ++col) {
      float value = output_data[row * 4 + col];
      sum += value;
      sum_sq += value * value;
    }
    float mean = sum / 4.0f;
    float var = sum_sq / 4.0f - mean * mean;
    EXPECT_NEAR(mean, 0.0f, 1e-5f);
    EXPECT_NEAR(var, 1.0f, 1e-3f);
  }
}

TEST(LayerNorm, ForwardMatchesReference) {
  std::vector<float> data = {0.1f, 0.2f, 0.3f, 0.4f};
  quasai::core::Tensor input = quasai::core::Tensor::from_data(
      data.data(), quasai::core::Shape{2, 2}, quasai::core::DType::FLOAT32);

  quasai::nn::LayerNorm layer_norm(2);
  quasai::core::Tensor output = layer_norm.forward(input);

  const float *output_data = output.data<float>();
  EXPECT_NEAR(output_data[0], -0.998006f, 1e-4f);
  EXPECT_NEAR(output_data[1], 0.998006f, 1e-4f);
  EXPECT_NEAR(output_data[2], -0.998006f, 1e-4f);
  EXPECT_NEAR(output_data[3], 0.998006f, 1e-4f);
}

TEST(LayerNorm, GradientThroughReductions) {
  // Each row of the normalized output sums to zero, so the gradient of the
  // sum of all outputs with respect to the input is zero.
  std::vector<float> data = {0.1f, 0.2f, 0.3f, 0.4f};
  quasai::core::Tensor input = quasai::core::Tensor::from_data(
      data.data(), quasai::core::Shape{2, 2}, quasai::core::DType::FLOAT32);
  input.requires_grad(true);

  quasai::nn::LayerNorm layer_norm(2);
  quasai::core::Tensor output = layer_norm.forward(input);
  quasai::core::Tensor loss = quasai::ops::sum(output);
  loss.backward();

  const float *grad = input.autograd_meta()->grad_.data<float>();
  for (size_t i = 0; i < data.size(); ++i) {
    EXPECT_NEAR(grad[i], 0.0f, 1e-4f);
  }
}
