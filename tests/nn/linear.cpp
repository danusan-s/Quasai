#include "quasai/nn/linear.hpp"
#include "quasai/nn/loss.hpp"
#include "quasai/ops/tensor_ops.hpp"
#include "quasai/optim/sgd.hpp"
#include <gtest/gtest.h>

TEST(Linear, Simple) {

  size_t in_features = 3;
  size_t out_features = 2;

  std::vector<float> input_data = {1.0f, 2.0f, 3.0f};
  quasai::Tensor input = quasai::Tensor::from_data(
      input_data.data(), quasai::Shape{1, in_features}, quasai::DType::FLOAT32);

  quasai::Linear linear(in_features, out_features);

  quasai::Tensor y = linear.forward(input);

  float *output_data = y.data<float>();
  EXPECT_FLOAT_EQ(output_data[0], 7.0f);
  EXPECT_FLOAT_EQ(output_data[1], 7.0f);
}

TEST(Linear, SimpleRegression) {

  size_t in_features = 3;
  size_t out_features = 2;

  std::vector<float> input_data = {1.0f, 2.0f, 3.0f};
  quasai::Tensor input = quasai::Tensor::from_data(
      input_data.data(), quasai::Shape{1, in_features}, quasai::DType::FLOAT32);

  input.requires_grad(true);

  std::vector<float> target_data = {0.0f, 1.0f};
  quasai::Tensor target = quasai::Tensor::from_data(
      target_data.data(), quasai::Shape{out_features}, quasai::DType::FLOAT32);

  target.requires_grad(true);

  quasai::Linear linear(in_features, out_features);

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

  quasai::Linear linear1(in_features, hidden_features);
  quasai::Linear linear2(hidden_features, out_features);

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
