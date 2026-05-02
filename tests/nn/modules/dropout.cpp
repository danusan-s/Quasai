#include "quasai/nn/modules/dropout.hpp"
#include "quasai/ops/tensor_ops.hpp"
#include <gtest/gtest.h>
#include <random>
#include <vector>

TEST(Dropout, InvalidProbabilityThrows) {
  EXPECT_THROW(quasai::nn::Dropout(-0.1f), std::invalid_argument);
  EXPECT_THROW(quasai::nn::Dropout(1.0f), std::invalid_argument);
  EXPECT_THROW(quasai::nn::Dropout(1.5f), std::invalid_argument);
}

TEST(Dropout, ValidProbabilityDoesNotThrow) {
  EXPECT_NO_THROW(quasai::nn::Dropout(0.0f));
  EXPECT_NO_THROW(quasai::nn::Dropout(0.5f));
  EXPECT_NO_THROW(quasai::nn::Dropout(0.99f));
}

TEST(Dropout, EvalModeReturnsInputUnchanged) {
  quasai::nn::Dropout dropout(0.5f);
  dropout.set_eval();

  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
  quasai::core::Tensor input = quasai::core::Tensor::from_data(
      input_data.data(), quasai::core::Shape{4}, quasai::core::DType::FLOAT32);

  quasai::core::Tensor output = dropout.forward(input);

  ASSERT_EQ(output.shape(), input.shape());
  const float *output_data = output.data<float>();
  for (size_t i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(output_data[i], input_data[i]);
  }
}

TEST(Dropout, TrainingModeAppliesDropout) {
  quasai::nn::Dropout dropout(0.5f);
  dropout.set_train();

  std::vector<float> input_data(1000, 1.0f);
  quasai::core::Tensor input = quasai::core::Tensor::from_data(
      input_data.data(), quasai::core::Shape{1000},
      quasai::core::DType::FLOAT32);

  quasai::core::Tensor output = dropout.forward(input);

  ASSERT_EQ(output.shape(), input.shape());
  const float *output_data = output.data<float>();

  int zeros = 0;
  int nonzeros = 0;
  for (size_t i = 0; i < 1000; ++i) {
    if (output_data[i] == 0.0f) {
      zeros++;
    } else {
      nonzeros++;
      EXPECT_GT(output_data[i], 1.0f);
    }
  }

  EXPECT_GT(zeros, 0);
  EXPECT_GT(nonzeros, 0);
}

TEST(Dropout, TrainingModeScalesOutput) {
  float p = 0.3f;
  quasai::nn::Dropout dropout(p);
  dropout.set_train();

  std::vector<float> input_data = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                                   1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  quasai::core::Tensor input = quasai::core::Tensor::from_data(
      input_data.data(), quasai::core::Shape{10}, quasai::core::DType::FLOAT32);

  quasai::core::Tensor output = dropout.forward(input);
  const float *output_data = output.data<float>();

  for (size_t i = 0; i < 10; ++i) {
    if (output_data[i] != 0.0f) {
      EXPECT_FLOAT_EQ(output_data[i], 1.0f / (1.0f - p));
    }
  }
}

TEST(Dropout, TrainEvalModeSwitch) {
  quasai::nn::Dropout dropout(0.5f);

  std::vector<float> input_data = {1.0f, 2.0f, 3.0f};
  quasai::core::Tensor input = quasai::core::Tensor::from_data(
      input_data.data(), quasai::core::Shape{3}, quasai::core::DType::FLOAT32);

  dropout.set_train();
  quasai::core::Tensor train_output = dropout.forward(input);

  dropout.set_eval();
  quasai::core::Tensor eval_output = dropout.forward(input);

  const float *eval_data = eval_output.data<float>();
  for (size_t i = 0; i < 3; ++i) {
    EXPECT_FLOAT_EQ(eval_data[i], input_data[i]);
  }
}
