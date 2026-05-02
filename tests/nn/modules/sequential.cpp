#include "quasai/nn/modules/activations.hpp"
#include "quasai/nn/modules/linear.hpp"
#include "quasai/nn/sequential_builder.hpp"
#include <gtest/gtest.h>
#include <vector>

TEST(Sequential, ConstructorAndForward) {
  auto model = quasai::nn::SequentialBuilder()
                   .add<quasai::nn::Linear>(3, 4)
                   .add<quasai::nn::ReLU>()
                   .add<quasai::nn::Linear>(4, 2)
                   .build();

  std::vector<float> input_data = {1.0f, 2.0f, 3.0f};
  quasai::core::Tensor input = quasai::core::Tensor::from_data(
      input_data.data(), quasai::core::Shape{3}, quasai::core::DType::FLOAT32);

  quasai::core::Tensor output = model.forward(input);
  EXPECT_TRUE(output.shape() == quasai::core::Shape({1, 2}));
}

TEST(Sequential, ForwardShape) {
  auto model = quasai::nn::SequentialBuilder()
                   .add<quasai::nn::Linear>(3, 4)
                   .add<quasai::nn::ReLU>()
                   .add<quasai::nn::Linear>(4, 2)
                   .build();

  std::vector<float> input_data = {1.0f, 2.0f, 3.0f};
  quasai::core::Tensor input = quasai::core::Tensor::from_data(
      input_data.data(), quasai::core::Shape{3}, quasai::core::DType::FLOAT32);

  quasai::core::Tensor output = model.forward(input);

  EXPECT_TRUE(output.shape() == quasai::core::Shape({1, 2}));
}

TEST(Sequential, Parameters) {
  auto model = quasai::nn::SequentialBuilder()
                   .add<quasai::nn::Linear>(3, 4)
                   .add<quasai::nn::ReLU>()
                   .add<quasai::nn::Linear>(4, 2)
                   .build();

  auto params = model.parameters();
  EXPECT_EQ(params.size(), 4);
}

TEST(Sequential, TrainEvalMode) {
  auto model = quasai::nn::SequentialBuilder()
                   .add<quasai::nn::Linear>(3, 4)
                   .add<quasai::nn::ReLU>()
                   .add<quasai::nn::Linear>(4, 2)
                   .build();

  EXPECT_TRUE(model.is_training());

  model.set_eval();
  EXPECT_FALSE(model.is_training());
  model.set_train();
  EXPECT_TRUE(model.is_training());
}

TEST(Sequential, CallableOperator) {
  auto model =
      quasai::nn::SequentialBuilder().add<quasai::nn::Linear>(3, 2).build();

  std::vector<float> input_data = {1.0f, 2.0f, 3.0f};
  quasai::core::Tensor input = quasai::core::Tensor::from_data(
      input_data.data(), quasai::core::Shape{3}, quasai::core::DType::FLOAT32);

  quasai::core::Tensor output = model(input);

  EXPECT_TRUE(output.shape() == quasai::core::Shape({1, 2}));
}
