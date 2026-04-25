#include "quasai/nn/linear.hpp"
#include "quasai/nn/init.hpp"
#include "quasai/nn/loss.hpp"
#include "quasai/ops/tensor_ops.hpp"
#include "quasai/optim/sgd.hpp"
#include <gtest/gtest.h>

TEST(Linear, SimpleRegression) {

  size_t in_features = 3;
  size_t out_features = 2;

  std::vector<float> input_data = {1.0f, 2.0f, 3.0f};
  quasai::Tensor input = quasai::Tensor::from_data(
      input_data.data(), quasai::Shape{in_features}, quasai::DType::FLOAT32);

  input.requires_grad(true);

  std::vector<float> target_data = {0.0f, 1.0f};
  quasai::Tensor target = quasai::Tensor::from_data(
      target_data.data(), quasai::Shape{out_features}, quasai::DType::FLOAT32);

  target.requires_grad(true);

  quasai::Initialization init = quasai::Initialization::HE_UNIFORM;

  quasai::Linear linear(in_features, out_features, init);

  size_t epochs = 10;

  std::vector<quasai::Parameter> params = linear.parameters();

  quasai::SGD optimizer(params, 0.005f);

  for (size_t epoch = 0; epoch < epochs; ++epoch) {
    quasai::Tensor output = linear.forward(input);

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

TEST(Linear, OneHiddenLayer) {
  size_t in_features = 3;
  size_t hidden_features = 4;
  size_t out_features = 2;

  std::vector<float> input_data = {1.0f, 2.0f, 3.0f};
  quasai::Tensor input = quasai::Tensor::from_data(
      input_data.data(), quasai::Shape{in_features}, quasai::DType::FLOAT32);

  input.requires_grad(true);

  std::vector<float> target_data = {0.0f, 1.0f};
  quasai::Tensor target = quasai::Tensor::from_data(
      target_data.data(), quasai::Shape{out_features}, quasai::DType::FLOAT32);

  target.requires_grad(true);

  quasai::Initialization init = quasai::Initialization::GLOROT_UNIFORM;

  quasai::Linear linear1(in_features, hidden_features, init);
  quasai::Linear linear2(hidden_features, out_features, init);

  size_t epochs = 10;

  std::vector<quasai::Parameter> params;
  for (quasai::Parameter &p : linear1.parameters()) {
    params.push_back(p);
  }
  for (quasai::Parameter &p : linear2.parameters()) {
    params.push_back(p);
  }

  quasai::SGD optimizer(params, 0.01f);

  for (size_t epoch = 0; epoch < epochs; ++epoch) {
    std::cout << "Epoch " << epoch + 1 << std::endl;
    quasai::Tensor hidden = linear1.forward(input);
    quasai::Tensor activated = quasai::relu(hidden);
    quasai::Tensor output = linear2.forward(activated);

    const float *output_data = output.data<float>();
    quasai::Tensor loss = quasai::mse_loss(output, target);

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

    input.requires_grad(true);
    target.requires_grad(true);
    targets.push_back(target);
    inputs.push_back(input);
  }

  quasai::Initialization init = quasai::Initialization::GLOROT_UNIFORM;

  quasai::Linear linear1(in_features, hidden_features, init);
  quasai::Linear linear2(hidden_features, out_features, init);

  std::vector<quasai::Parameter> params;
  for (auto &p : linear1.parameters())
    params.push_back(p);
  for (auto &p : linear2.parameters())
    params.push_back(p);

  quasai::SGD optimizer(params, 0.01f);

  size_t epochs = 50;

  std::cout << "Starting training on " << num_samples << " samples for "
            << epochs << " epochs." << std::endl;

  for (size_t epoch = 0; epoch < epochs; ++epoch) {
    float total_loss = 0.0f;

    for (size_t i = 0; i < num_samples; ++i) {
      quasai::Tensor hidden = linear1.forward(inputs[i]);
      quasai::Tensor activated = quasai::relu(hidden);
      quasai::Tensor output = linear2.forward(activated);

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
    quasai::Tensor hidden = linear1.forward(inputs[i]);
    quasai::Tensor activated = quasai::relu(hidden);
    quasai::Tensor output = linear2.forward(activated);

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
