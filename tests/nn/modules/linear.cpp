#include "quasai/nn/modules/linear.hpp"
#include "quasai/nn/init.hpp"
#include <gtest/gtest.h>
#include <vector>

TEST(Linear, Constructor) {
  quasai::nn::Linear linear(3, 2, quasai::nn::Initialization::GLOROT_UNIFORM);

  auto params = linear.parameters();
  ASSERT_EQ(params.size(), 2);

  EXPECT_TRUE(params[0].shape() == quasai::core::Shape({3, 2}));
  EXPECT_TRUE(params[1].shape() == quasai::core::Shape({1, 2}));
}

TEST(Linear, ForwardShape) {
  quasai::nn::Linear linear(3, 2, quasai::nn::Initialization::GLOROT_UNIFORM);

  std::vector<float> input_data = {1.0f, 2.0f, 3.0f};
  quasai::core::Tensor input = quasai::core::Tensor::from_data(
      input_data.data(), quasai::core::Shape{3}, quasai::core::DType::FLOAT32);

  quasai::core::Tensor output = linear.forward(input);

  EXPECT_TRUE(output.shape() == quasai::core::Shape({1, 2}));
}

TEST(Linear, ForwardBatchShape) {
  quasai::nn::Linear linear(3, 2, quasai::nn::Initialization::GLOROT_UNIFORM);

  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  quasai::core::Tensor input = quasai::core::Tensor::from_data(
      input_data.data(), quasai::core::Shape{2, 3},
      quasai::core::DType::FLOAT32);

  quasai::core::Tensor output = linear.forward(input);

  EXPECT_EQ(output.shape(), quasai::core::Shape({2, 2}));
}

TEST(Linear, ParametersReturnsWeightsAndBias) {
  quasai::nn::Linear linear(3, 2, quasai::nn::Initialization::GLOROT_UNIFORM);

  auto params = linear.parameters();
  ASSERT_EQ(params.size(), 2);
}

TEST(Linear, TrainEvalMode) {
  quasai::nn::Linear linear(3, 2, quasai::nn::Initialization::GLOROT_UNIFORM);

  EXPECT_TRUE(linear.is_training());

  linear.eval();
  EXPECT_FALSE(linear.is_training());

  linear.train();
  EXPECT_TRUE(linear.is_training());
}

TEST(Linear, ForwardConsistency) {
  quasai::nn::Linear linear(3, 2, quasai::nn::Initialization::GLOROT_UNIFORM);
  linear.eval();

  std::vector<float> input_data = {1.0f, 2.0f, 3.0f};
  quasai::core::Tensor input = quasai::core::Tensor::from_data(
      input_data.data(), quasai::core::Shape{3}, quasai::core::DType::FLOAT32);

  quasai::core::Tensor output1 = linear.forward(input);
  quasai::core::Tensor output2 = linear.forward(input);

  const float *data1 = output1.data<float>();
  const float *data2 = output2.data<float>();
  for (size_t i = 0; i < 2; ++i) {
    EXPECT_FLOAT_EQ(data1[i], data2[i]);
  }
}
