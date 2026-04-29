#include "quasai/nn/modules/sequential.hpp"
#include "quasai/nn/init.hpp"
#include "quasai/nn/loss.hpp"
#include "quasai/nn/modules/activations.hpp"
#include "quasai/nn/modules/linear.hpp"
#include "quasai/optim/sgd.hpp"
#include <gtest/gtest.h>

TEST(Sequential, TwoLayerRegression) {
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

  quasai::nn::Initialization init = quasai::nn::Initialization::GLOROT_UNIFORM;

  auto linear1 =
      std::make_shared<quasai::nn::Linear>(in_features, hidden_features, init);
  auto relu = std::make_shared<quasai::nn::ReLU>();
  auto linear2 =
      std::make_shared<quasai::nn::Linear>(hidden_features, out_features, init);

  quasai::nn::Sequential model({linear1, relu, linear2});

  std::vector<quasai::nn::Parameter> params = model.parameters();

  float learning_rate = 0.001f;
  float momentum = 0.9f;
  quasai::optim::SGD optimizer(learning_rate, momentum);
  optimizer.compile(params);

  size_t epochs = 10;

  for (size_t epoch = 0; epoch < epochs; ++epoch) {
    quasai::core::Tensor output = model.forward(input);

    const float *output_data = output.data<float>();
    quasai::core::Tensor loss = quasai::nn::mse_loss(output, target);

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

  std::vector<quasai::core::Tensor> inputs;
  std::vector<quasai::core::Tensor> targets;

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

    quasai::core::Tensor input = quasai::core::Tensor::from_data(
        input_data.data(), quasai::core::Shape{in_features},
        quasai::core::DType::FLOAT32);

    quasai::core::Tensor target = quasai::core::Tensor::from_data(
        target_data.data(), quasai::core::Shape{out_features},
        quasai::core::DType::FLOAT32);

    targets.push_back(target);
    inputs.push_back(input);
  }

  quasai::nn::Initialization init = quasai::nn::Initialization::GLOROT_UNIFORM;

  auto linear1 =
      std::make_shared<quasai::nn::Linear>(in_features, hidden_features, init);
  auto relu = std::make_shared<quasai::nn::ReLU>();
  auto linear2 =
      std::make_shared<quasai::nn::Linear>(hidden_features, out_features, init);

  quasai::nn::Sequential model({linear1, relu, linear2});

  float learning_rate = 0.001f;
  float momentum = 0.9f;
  quasai::optim::SGD optimizer(learning_rate, momentum);
  optimizer.compile(model.parameters());

  size_t epochs = 50;

  std::cout << "Starting training on " << num_samples << " samples for "
            << epochs << " epochs." << std::endl;

  for (size_t epoch = 0; epoch < epochs; ++epoch) {
    float total_loss = 0.0f;

    for (size_t i = 0; i < num_samples; ++i) {
      quasai::core::Tensor output = model(inputs[i]);

      quasai::core::Tensor loss = quasai::nn::mse_loss(output, targets[i]);

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
    quasai::core::Tensor output = model.forward(inputs[i]);

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

TEST(Sequential, RegressionWithBatching) {
  size_t in_features = 3;
  size_t hidden_features = 8;
  size_t out_features = 1;

  size_t num_samples = 1000;

  std::vector<float> inputs;
  std::vector<float> targets;

  // Generate dataset
  for (size_t i = 0; i < num_samples; ++i) {
    float x0 = static_cast<float>(i % 10) / 10.0f;
    float x1 = static_cast<float>((i * 2) % 10) / 10.0f;
    float x2 = static_cast<float>((i * 3) % 10) / 10.0f;

    std::vector<float> input_data = {x0, x1, x2};
    std::vector<float> target_data = {
        x0 + x1 - 0.4f * x2,
    };
    inputs.insert(inputs.end(), input_data.begin(), input_data.end());
    targets.insert(targets.end(), target_data.begin(), target_data.end());
  }

  quasai::core::Tensor input_tensor = quasai::core::Tensor::from_data(
      inputs.data(), quasai::core::Shape{num_samples, in_features},
      quasai::core::DType::FLOAT32);
  quasai::core::Tensor target_tensor = quasai::core::Tensor::from_data(
      targets.data(), quasai::core::Shape{num_samples, out_features},
      quasai::core::DType::FLOAT32);

  quasai::nn::Initialization init = quasai::nn::Initialization::GLOROT_UNIFORM;

  auto linear1 =
      std::make_shared<quasai::nn::Linear>(in_features, hidden_features, init);
  auto relu = std::make_shared<quasai::nn::ReLU>();
  auto linear2 =
      std::make_shared<quasai::nn::Linear>(hidden_features, out_features, init);

  quasai::nn::Sequential model({linear1, relu, linear2});

  float learning_rate = 0.001f;
  float momentum = 0.9f;
  quasai::optim::SGD optimizer(learning_rate, momentum);
  optimizer.compile(model.parameters());

  size_t epochs = 50;
  size_t batch_size = 10;

  std::cout << "Starting training on " << num_samples << " samples for "
            << epochs << " epochs with batch size " << batch_size << std::endl;

  for (size_t epoch = 0; epoch < epochs; ++epoch) {
    float total_loss = 0.0f;

    for (size_t i = 0; i < num_samples; i += batch_size) {
      size_t current_batch_size = std::min(batch_size, num_samples - i);
      quasai::core::Tensor batch_input = quasai::ops::slice(
          input_tensor, i, i + current_batch_size); // Get batch input
      quasai::core::Tensor batch_target = quasai::ops::slice(
          target_tensor, i, i + current_batch_size); // Get batch target
      quasai::core::Tensor output = model(batch_input);

      quasai::core::Tensor loss = quasai::nn::mse_loss(output, batch_target);

      loss.backward();

      total_loss += loss.data<float>()[0];

      optimizer.step();
      optimizer.zero_grad();
    }

    std::cout << "Epoch " << epoch + 1
              << ", Avg Loss: " << total_loss / num_samples << std::endl;
  }

  std::cout << "Starting testing on training data." << std::endl;

  quasai::core::Tensor output = model.forward(input_tensor);

  quasai::core::Tensor final_loss = quasai::nn::mse_loss(output, target_tensor);

  std::cout << "Final Loss: " << final_loss.data<float>()[0] << std::endl;

  // Basic assertion: loss should be small after training
  EXPECT_LT(float(final_loss.data<float>()[0]), 1e-2f);
}
