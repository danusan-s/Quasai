#include "quasai/nn/init.hpp"
#include "quasai/nn/loss.hpp"
#include "quasai/nn/modules/activations.hpp"
#include "quasai/nn/modules/dropout.hpp"
#include "quasai/nn/modules/linear.hpp"
#include "quasai/nn/modules/sequential.hpp"
#include "quasai/ops/tensor_ops.hpp"
#include "quasai/optim/sgd.hpp"
#include <gtest/gtest.h>
#include <memory>
#include <vector>

TEST(Model, LinearRegression) {
  size_t in_features = 3;
  size_t out_features = 2;

  std::vector<float> input_data = {1.0f, 2.0f, 3.0f};
  quasai::core::Tensor input = quasai::core::Tensor::from_data(
      input_data.data(), quasai::core::Shape{in_features},
      quasai::core::DType::FLOAT32);

  std::vector<float> target_data = {0.0f, 1.0f};
  quasai::core::Tensor target = quasai::core::Tensor::from_data(
      target_data.data(), quasai::core::Shape{out_features},
      quasai::core::DType::FLOAT32);

  quasai::nn::Linear linear(in_features, out_features,
                            quasai::nn::Initialization::HE_UNIFORM);

  quasai::optim::SGD optimizer(0.005f, 0.9f);
  optimizer.compile(linear.parameters());

  size_t epochs = 10;
  for (size_t epoch = 0; epoch < epochs; ++epoch) {
    quasai::core::Tensor output = linear.forward(input);
    quasai::core::Tensor loss = quasai::nn::mse_loss(output, target);
    loss.backward();
    optimizer.step();
    optimizer.zero_grad();
  }
}

TEST(Model, TwoLayerRegression) {
  size_t in_features = 3;
  size_t hidden_features = 4;
  size_t out_features = 2;

  std::vector<float> input_data = {1.0f, 2.0f, 3.0f};
  quasai::core::Tensor input = quasai::core::Tensor::from_data(
      input_data.data(), quasai::core::Shape{in_features},
      quasai::core::DType::FLOAT32);

  std::vector<float> target_data = {0.0f, 1.0f};
  quasai::core::Tensor target = quasai::core::Tensor::from_data(
      target_data.data(), quasai::core::Shape{out_features},
      quasai::core::DType::FLOAT32);

  auto linear1 = std::make_shared<quasai::nn::Linear>(
      in_features, hidden_features, quasai::nn::Initialization::GLOROT_UNIFORM);
  auto relu = std::make_shared<quasai::nn::ReLU>();
  auto linear2 = std::make_shared<quasai::nn::Linear>(
      hidden_features, out_features,
      quasai::nn::Initialization::GLOROT_UNIFORM);

  quasai::nn::Sequential model({linear1, relu, linear2});

  quasai::optim::SGD optimizer(0.001f, 0.9f);
  optimizer.compile(model.parameters());

  for (size_t epoch = 0; epoch < 10; ++epoch) {
    quasai::core::Tensor output = model.forward(input);
    quasai::core::Tensor loss = quasai::nn::mse_loss(output, target);
    loss.backward();
    optimizer.step();
    optimizer.zero_grad();
  }
}

TEST(Model, MultiSampleTraining) {
  size_t in_features = 3;
  size_t hidden_features = 8;
  size_t out_features = 2;
  size_t num_samples = 100;

  std::vector<quasai::core::Tensor> inputs;
  std::vector<quasai::core::Tensor> targets;

  for (size_t i = 0; i < num_samples; ++i) {
    float x0 = static_cast<float>(i % 10) / 10.0f;
    float x1 = static_cast<float>((i * 2) % 10) / 10.0f;
    float x2 = static_cast<float>((i * 3) % 10) / 10.0f;

    std::vector<float> input_data = {x0, x1, x2};
    std::vector<float> target_data = {x0 + x1, x2};

    inputs.push_back(quasai::core::Tensor::from_data(
        input_data.data(), quasai::core::Shape{in_features},
        quasai::core::DType::FLOAT32));
    targets.push_back(quasai::core::Tensor::from_data(
        target_data.data(), quasai::core::Shape{out_features},
        quasai::core::DType::FLOAT32));
  }

  auto linear1 = std::make_shared<quasai::nn::Linear>(
      in_features, hidden_features, quasai::nn::Initialization::GLOROT_UNIFORM);
  auto relu = std::make_shared<quasai::nn::ReLU>();
  auto linear2 = std::make_shared<quasai::nn::Linear>(
      hidden_features, out_features,
      quasai::nn::Initialization::GLOROT_UNIFORM);

  quasai::nn::Sequential model({linear1, relu, linear2});

  quasai::optim::SGD optimizer(0.001f, 0.9f);
  optimizer.compile(model.parameters());

  float initial_loss = 0.0f;
  for (size_t i = 0; i < num_samples; ++i) {
    quasai::core::Tensor output = model.forward(inputs[i]);
    initial_loss += quasai::nn::mse_loss(output, targets[i]).data<float>()[0];
  }
  initial_loss /= num_samples;

  for (size_t epoch = 0; epoch < 50; ++epoch) {
    float total_loss = 0.0f;
    for (size_t i = 0; i < num_samples; ++i) {
      quasai::core::Tensor output = model(inputs[i]);
      quasai::core::Tensor loss = quasai::nn::mse_loss(output, targets[i]);
      loss.backward();
      total_loss += loss.data<float>()[0];
      optimizer.step();
      optimizer.zero_grad();
    }
  }

  float final_loss = 0.0f;
  for (size_t i = 0; i < num_samples; ++i) {
    quasai::core::Tensor output = model.forward(inputs[i]);
    final_loss += quasai::nn::mse_loss(output, targets[i]).data<float>()[0];
  }
  final_loss /= num_samples;

  EXPECT_LT(final_loss, initial_loss);
  EXPECT_LT(final_loss, 0.1f);
}

TEST(Model, TrainingWithDropout) {
  size_t in_features = 3;
  size_t hidden_features = 8;
  size_t out_features = 2;
  size_t num_samples = 100;

  std::vector<quasai::core::Tensor> inputs;
  std::vector<quasai::core::Tensor> targets;

  for (size_t i = 0; i < num_samples; ++i) {
    float x0 = static_cast<float>(i % 10) / 10.0f;
    float x1 = static_cast<float>((i * 2) % 10) / 10.0f;
    float x2 = static_cast<float>((i * 3) % 10) / 10.0f;

    std::vector<float> input_data = {x0, x1, x2};
    std::vector<float> target_data = {x0 + x1, x2};

    inputs.push_back(quasai::core::Tensor::from_data(
        input_data.data(), quasai::core::Shape{in_features},
        quasai::core::DType::FLOAT32));
    targets.push_back(quasai::core::Tensor::from_data(
        target_data.data(), quasai::core::Shape{out_features},
        quasai::core::DType::FLOAT32));
  }

  auto linear1 = std::make_shared<quasai::nn::Linear>(
      in_features, hidden_features, quasai::nn::Initialization::GLOROT_UNIFORM);
  auto relu = std::make_shared<quasai::nn::ReLU>();
  auto dropout = std::make_shared<quasai::nn::Dropout>(0.2f);
  auto linear2 = std::make_shared<quasai::nn::Linear>(
      hidden_features, out_features,
      quasai::nn::Initialization::GLOROT_UNIFORM);

  quasai::nn::Sequential model({linear1, relu, dropout, linear2});

  quasai::optim::SGD optimizer(0.001f, 0.9f);
  optimizer.compile(model.parameters());

  model.train();

  for (size_t epoch = 0; epoch < 50; ++epoch) {
    float total_loss = 0.0f;
    for (size_t i = 0; i < num_samples; ++i) {
      quasai::core::Tensor output = model(inputs[i]);
      quasai::core::Tensor loss = quasai::nn::mse_loss(output, targets[i]);
      loss.backward();
      total_loss += loss.data<float>()[0];
      optimizer.step();
      optimizer.zero_grad();
    }
  }

  model.eval();
  float eval_loss = 0.0f;
  for (size_t i = 0; i < num_samples; ++i) {
    quasai::core::Tensor output = model(inputs[i]);
    eval_loss += quasai::nn::mse_loss(output, targets[i]).data<float>()[0];
  }
  eval_loss /= num_samples;

  EXPECT_LT(eval_loss, 0.15f);
}

TEST(Model, RegressionWithBatching) {
  size_t in_features = 3;
  size_t hidden_features = 8;
  size_t out_features = 1;
  size_t num_samples = 1000;

  std::vector<float> inputs;
  std::vector<float> targets;

  for (size_t i = 0; i < num_samples; ++i) {
    float x0 = static_cast<float>(i % 10) / 10.0f;
    float x1 = static_cast<float>((i * 2) % 10) / 10.0f;
    float x2 = static_cast<float>((i * 3) % 10) / 10.0f;

    std::vector<float> input_data = {x0, x1, x2};
    std::vector<float> target_data = {x0 + x1 - 0.4f * x2};

    inputs.insert(inputs.end(), input_data.begin(), input_data.end());
    targets.insert(targets.end(), target_data.begin(), target_data.end());
  }

  quasai::core::Tensor input_tensor = quasai::core::Tensor::from_data(
      inputs.data(), quasai::core::Shape{num_samples, in_features},
      quasai::core::DType::FLOAT32);
  quasai::core::Tensor target_tensor = quasai::core::Tensor::from_data(
      targets.data(), quasai::core::Shape{num_samples, out_features},
      quasai::core::DType::FLOAT32);

  auto linear1 = std::make_shared<quasai::nn::Linear>(
      in_features, hidden_features, quasai::nn::Initialization::GLOROT_UNIFORM);
  auto relu = std::make_shared<quasai::nn::ReLU>();
  auto linear2 = std::make_shared<quasai::nn::Linear>(
      hidden_features, out_features,
      quasai::nn::Initialization::GLOROT_UNIFORM);

  quasai::nn::Sequential model({linear1, relu, linear2});

  quasai::optim::SGD optimizer(0.001f, 0.9f);
  optimizer.compile(model.parameters());

  size_t epochs = 50;
  size_t batch_size = 10;

  model.train();

  for (size_t epoch = 0; epoch < epochs; ++epoch) {
    for (size_t i = 0; i < num_samples; i += batch_size) {
      size_t current_batch_size = std::min(batch_size, num_samples - i);
      quasai::core::Tensor batch_input =
          quasai::ops::slice(input_tensor, i, i + current_batch_size);
      quasai::core::Tensor batch_target =
          quasai::ops::slice(target_tensor, i, i + current_batch_size);

      quasai::core::Tensor output = model(batch_input);
      quasai::core::Tensor loss = quasai::nn::mse_loss(output, batch_target);
      loss.backward();
      optimizer.step();
      optimizer.zero_grad();
    }
  }

  model.eval();
  quasai::core::Tensor output = model.forward(input_tensor);
  quasai::core::Tensor final_loss = quasai::nn::mse_loss(output, target_tensor);

  EXPECT_LT(final_loss.data<float>()[0], 1e-2f);
}
