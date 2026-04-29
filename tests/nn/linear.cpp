#include "quasai/nn/modules/linear.hpp"
#include "quasai/nn/init.hpp"
#include "quasai/nn/loss.hpp"
#include "quasai/ops/tensor_ops.hpp"
#include "quasai/optim/sgd.hpp"
#include <gtest/gtest.h>

TEST(Linear, SimpleRegression) {

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

  quasai::nn::Initialization init = quasai::nn::Initialization::HE_UNIFORM;

  quasai::nn::Linear linear(in_features, out_features, init);

  size_t epochs = 10;

  std::vector<quasai::nn::Parameter> params = linear.parameters();

  float learning_rate = 0.005f;
  float momentum = 0.9f;
  quasai::optim::SGD optimizer(learning_rate, momentum);
  optimizer.compile(params);

  for (size_t epoch = 0; epoch < epochs; ++epoch) {
    quasai::core::Tensor output = linear.forward(input);

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

TEST(Linear, OneHiddenLayer) {
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

  quasai::nn::Linear linear1(in_features, hidden_features, init);
  quasai::nn::Linear linear2(hidden_features, out_features, init);

  size_t epochs = 10;

  std::vector<quasai::nn::Parameter> params;
  for (quasai::nn::Parameter &p : linear1.parameters()) {
    params.push_back(p);
  }
  for (quasai::nn::Parameter &p : linear2.parameters()) {
    params.push_back(p);
  }

  quasai::optim::SGD optimizer(0.01f, 0.9f);
  optimizer.compile(params);

  for (size_t epoch = 0; epoch < epochs; ++epoch) {
    std::cout << "Epoch " << epoch + 1 << std::endl;
    quasai::core::Tensor hidden = linear1.forward(input);
    quasai::core::Tensor activated = quasai::ops::relu(hidden);
    quasai::core::Tensor output = linear2.forward(activated);

    const float *output_data = output.data<float>();
    quasai::core::Tensor loss = quasai::nn::mse_loss(output, target);

    loss.backward();

    std::cout << "Output: [" << output_data[0] << ", " << output_data[1] << "]"
              << std::endl;

    std::cout << "Target: [" << target_data[0] << ", " << target_data[1] << "]"
              << std::endl;

    std::cout << "Epoch " << epoch + 1 << ", Loss: " << loss.data<float>()[0]
              << std::endl;

    optimizer.step();
    optimizer.zero_grad();
  }
}

TEST(Linear, OneHiddenLayer_MultiSample) {
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

  quasai::nn::Linear linear1(in_features, hidden_features, init);
  quasai::nn::Linear linear2(hidden_features, out_features, init);

  std::vector<quasai::nn::Parameter> params;
  for (auto &p : linear1.parameters())
    params.push_back(p);
  for (auto &p : linear2.parameters())
    params.push_back(p);

  float learning_rate = 0.001f;
  float momentum = 0.9f;
  quasai::optim::SGD optimizer(learning_rate, momentum);
  optimizer.compile(params);

  size_t epochs = 50;

  // Track initial loss (before training)
  float initial_loss = 0.0f;
  for (size_t i = 0; i < num_samples; ++i) {
    quasai::core::Tensor hidden = linear1.forward(inputs[i]);
    quasai::core::Tensor activated = quasai::ops::relu(hidden);
    quasai::core::Tensor output = linear2.forward(activated);
    quasai::core::Tensor loss = quasai::nn::mse_loss(output, targets[i]);
    initial_loss += loss.data<float>()[0];
  }
  initial_loss /= num_samples;

  std::cout << "Starting training on " << num_samples << " samples for "
            << epochs << " epochs." << std::endl;

  for (size_t epoch = 0; epoch < epochs; ++epoch) {
    float total_loss = 0.0f;

    for (size_t i = 0; i < num_samples; ++i) {
      quasai::core::Tensor hidden = linear1.forward(inputs[i]);
      quasai::core::Tensor activated = quasai::ops::relu(hidden);
      quasai::core::Tensor output = linear2.forward(activated);

      quasai::core::Tensor loss = quasai::nn::mse_loss(output, targets[i]);

      loss.backward();

      total_loss += loss.data<float>()[0];

      optimizer.step();
      optimizer.zero_grad();
    }

    std::cout << "Epoch " << epoch + 1
              << ", Avg Loss: " << total_loss / num_samples << std::endl;
  }

  // Track final loss (after training)
  float final_loss = 0.0f;
  for (size_t i = 0; i < num_samples; ++i) {
    quasai::core::Tensor hidden = linear1.forward(inputs[i]);
    quasai::core::Tensor activated = quasai::ops::relu(hidden);
    quasai::core::Tensor output = linear2.forward(activated);
    quasai::core::Tensor loss = quasai::nn::mse_loss(output, targets[i]);
    final_loss += loss.data<float>()[0];
  }
  final_loss /= num_samples;

  std::cout << "Initial Avg Loss: " << initial_loss << std::endl;
  std::cout << "Final Avg Loss: " << final_loss << std::endl;

  std::cout << "Starting testing on training data." << std::endl;

  for (size_t i = 0; i < num_samples; ++i) {
    quasai::core::Tensor hidden = linear1.forward(inputs[i]);
    quasai::core::Tensor activated = quasai::ops::relu(hidden);
    quasai::core::Tensor output = linear2.forward(activated);

    const float *output_data = output.data<float>();
    const float *target_data = targets[i].data<float>();

    std::cout << "Sample " << i + 1 << ": Output: [" << output_data[0] << ", "
              << output_data[1] << "], Target: [" << target_data[0] << ", "
              << target_data[1] << "]" << std::endl;
  }

  // Loss should decrease significantly after training
  EXPECT_LT(final_loss, initial_loss);
  // Final loss should be small for this simple problem
  EXPECT_LT(final_loss, 0.1f);
}
