#include "quasai/nn/modules/sequential.hpp"
#include "quasai/nn/init.hpp"
#include "quasai/nn/modules/activations.hpp"
#include "quasai/nn/modules/linear.hpp"
#include <gtest/gtest.h>
#include <memory>
#include <vector>

TEST(Sequential, ConstructorAndForward) {
  auto linear1 = std::make_shared<quasai::nn::Linear>(3, 4);
  auto relu = std::make_shared<quasai::nn::ReLU>();
  auto linear2 = std::make_shared<quasai::nn::Linear>(4, 2);

  quasai::nn::Sequential model({linear1, relu, linear2});

  std::vector<float> input_data = {1.0f, 2.0f, 3.0f};
  quasai::core::Tensor input = quasai::core::Tensor::from_data(
      input_data.data(), quasai::core::Shape{3}, quasai::core::DType::FLOAT32);

  quasai::core::Tensor output = model.forward(input);
  EXPECT_TRUE(output.shape() == quasai::core::Shape({1, 2}));
}

TEST(Sequential, ForwardShape) {
  auto linear1 = std::make_shared<quasai::nn::Linear>(3, 4);
  auto relu = std::make_shared<quasai::nn::ReLU>();
  auto linear2 = std::make_shared<quasai::nn::Linear>(4, 2);

  quasai::nn::Sequential model({linear1, relu, linear2});

  std::vector<float> input_data = {1.0f, 2.0f, 3.0f};
  quasai::core::Tensor input = quasai::core::Tensor::from_data(
      input_data.data(), quasai::core::Shape{3}, quasai::core::DType::FLOAT32);

  quasai::core::Tensor output = model.forward(input);

  EXPECT_TRUE(output.shape() == quasai::core::Shape({1, 2}));
}

TEST(Sequential, Parameters) {
  auto linear1 = std::make_shared<quasai::nn::Linear>(3, 4);
  auto relu = std::make_shared<quasai::nn::ReLU>();
  auto linear2 = std::make_shared<quasai::nn::Linear>(4, 2);

  quasai::nn::Sequential model({linear1, relu, linear2});

  auto params = model.parameters();
  EXPECT_EQ(params.size(), 4);
}

TEST(Sequential, ForwardWithFunctionModules) {
  auto linear = std::make_shared<quasai::nn::Linear>(3, 2);
  auto relu = std::make_shared<quasai::nn::ReLU>();

  quasai::nn::Sequential model({linear, relu});

  std::vector<float> input_data = {-1.0f, 2.0f, -3.0f};
  quasai::core::Tensor input = quasai::core::Tensor::from_data(
      input_data.data(), quasai::core::Shape{3}, quasai::core::DType::FLOAT32);

  quasai::core::Tensor output = model.forward(input);
  const float *output_data = output.data<float>();

  EXPECT_GE(output_data[0], 0.0f);
  EXPECT_GE(output_data[1], 0.0f);
}

TEST(Sequential, TrainEvalMode) {
  auto linear1 = std::make_shared<quasai::nn::Linear>(3, 4);
  auto linear2 = std::make_shared<quasai::nn::Linear>(4, 2);

  quasai::nn::Sequential model({linear1, linear2});

  EXPECT_TRUE(model.is_training());

  model.eval();
  EXPECT_FALSE(model.is_training());
  EXPECT_FALSE(linear1->is_training());
  EXPECT_FALSE(linear2->is_training());

  model.train();
  EXPECT_TRUE(model.is_training());
  EXPECT_TRUE(linear1->is_training());
  EXPECT_TRUE(linear2->is_training());
}

TEST(Sequential, CallableOperator) {
  auto linear = std::make_shared<quasai::nn::Linear>(3, 2);
  quasai::nn::Sequential model({linear});

  std::vector<float> input_data = {1.0f, 2.0f, 3.0f};
  quasai::core::Tensor input = quasai::core::Tensor::from_data(
      input_data.data(), quasai::core::Shape{3}, quasai::core::DType::FLOAT32);

  quasai::core::Tensor output = model(input);

  EXPECT_TRUE(output.shape() == quasai::core::Shape({1, 2}));
}
