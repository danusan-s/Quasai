#include "quasai/nn/sequential.hpp"
#include "quasai/nn/activations.hpp"
#include "quasai/nn/init.hpp"
#include "quasai/nn/linear.hpp"
#include "quasai/nn/loss.hpp"
#include "quasai/optim/sgd.hpp"
#include <gtest/gtest.h>

TEST(Sequential, TwoLayerRegression) {
  size_t in_features = 3;
  size_t hidden_features = 4;
  size_t out_features = 2;

  std::vector<float> input_data = {1.0f, 2.0f, 3.0f};
  quasai::Tensor input = quasai::Tensor::from_data(
      input_data.data(), quasai::Shape{in_features}, quasai::DType::FLOAT32);

  std::vector<float> target_data = {0.0f, 1.0f};
  quasai::Tensor target = quasai::Tensor::from_data(
      target_data.data(), quasai::Shape{out_features}, quasai::DType::FLOAT32);

  quasai::Initialization init = quasai::Initialization::GLOROT_UNIFORM;

  auto linear1 =
      std::make_shared<quasai::Linear>(in_features, hidden_features, init);
  auto relu = std::make_shared<quasai::ReLU>();
  auto linear2 =
      std::make_shared<quasai::Linear>(hidden_features, out_features, init);

  quasai::Sequential model({linear1, relu, linear2});

  std::vector<quasai::Parameter> params = model.parameters();

  float learning_rate = 0.001f;
  float momentum = 0.9f;
  quasai::SGD optimizer(params, learning_rate, momentum);

  size_t epochs = 10;

  for (size_t epoch = 0; epoch < epochs; ++epoch) {
    quasai::Tensor output = model.forward(input);

    const float *output_data = output.data<float>();
    quasai::Tensor loss = quasai::mse_loss(output, target);

    loss.backward();

    std::cout << "Output: [" << output_data[0] << ", " << output_data[1] << "]"
              << std::endl;

    std::cout << "Epoch " << epoch + 1 << ", Loss: " << loss.data<float>()[0]
              << std::endl;

    optimizer.step();
    optimizer.zero_grad();
  }
}

TEST(Sequential, OneHiddenLayer_MultiSample) {
  size_t in_features = 3;
  size_t hidden_features = 8;
  size_t out_features = 2;

  size_t num_samples = 100;

  std::vector<quasai::Tensor> inputs;
  std::vector<quasai::Tensor> targets;

  // Generate dataset
  for (size_t i = 0; i < num_samples; ++i) {
    float x0 = static_cast<float>(i % 10) / 10.0f;
    float x1 = static_cast<float>((i * 2) % 10) / 10.0f;
    float x2 = static_cast<float>((i * 3) % 10) / 10.0f;

    std::vector<float> input_data = {x0, x1, x2};
    std::vector<float> target_data = {
        x0 + x1, // first output depends on x0, x1
        x2       // second output depends on x2
    };

    quasai::Tensor input = quasai::Tensor::from_data(
        input_data.data(), quasai::Shape{in_features}, quasai::DType::FLOAT32);

    quasai::Tensor target = quasai::Tensor::from_data(
        target_data.data(), quasai::Shape{out_features},
        quasai::DType::FLOAT32);

    targets.push_back(target);
    inputs.push_back(input);
  }

  quasai::Initialization init = quasai::Initialization::GLOROT_UNIFORM;

  auto linear1 =
      std::make_shared<quasai::Linear>(in_features, hidden_features, init);
  auto relu = std::make_shared<quasai::ReLU>();
  auto linear2 =
      std::make_shared<quasai::Linear>(hidden_features, out_features, init);

  quasai::Sequential model({linear1, relu, linear2});

  float learning_rate = 0.001f;
  float momentum = 0.9f;
  quasai::SGD optimizer(model.parameters(), learning_rate, momentum);

  size_t epochs = 50;

  std::cout << "Starting training on " << num_samples << " samples for "
            << epochs << " epochs." << std::endl;

  for (size_t epoch = 0; epoch < epochs; ++epoch) {
    float total_loss = 0.0f;

    for (size_t i = 0; i < num_samples; ++i) {
      quasai::Tensor output = model(inputs[i]);

      quasai::Tensor loss = quasai::mse_loss(output, targets[i]);

      loss.backward();

      total_loss += loss.data<float>()[0];

      optimizer.step();
      optimizer.zero_grad();
    }

    std::cout << "Epoch " << epoch + 1
              << ", Avg Loss: " << total_loss / num_samples << std::endl;
  }

  std::cout << "Starting testing on training data." << std::endl;

  for (size_t i = 0; i < num_samples; ++i) {
    quasai::Tensor output = model.forward(inputs[i]);

    const float *output_data = output.data<float>();
    const float *target_data = targets[i].data<float>();

    std::cout << "Sample " << i + 1 << ": Output: [" << output_data[0] << ", "
              << output_data[1] << "], Target: [" << target_data[0] << ", "
              << target_data[1] << "]" << std::endl;
  }

  // Basic assertion: loss should be small after training
  EXPECT_LT((float)(1e-2),
            1.0f); // replace with real check if you track final loss
}
