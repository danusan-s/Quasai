// examples/cpp/simple_example.cpp
// Simple example demonstrating Quasai framework usage

#include "quasai/nn/layers/activations.hpp"
#include "quasai/nn/layers/linear.hpp"
#include "quasai/nn/model.hpp"
#include "quasai/nn/sequential.hpp"
#include "quasai/optim/sgd.hpp"
#include <iostream>

int main() {
  // Create a simple MLP
  auto linear1 = std::make_shared<quasai::nn::Linear>(64, 32);
  auto relu = std::make_shared<quasai::nn::ReLU>();
  auto linear2 = std::make_shared<quasai::nn::Linear>(32, 10);

  auto model = std::make_shared<quasai::nn::Sequential>(
      std::vector<std::shared_ptr<quasai::nn::Module>>{linear1, relu, linear2});

  quasai::nn::Model ml_model(model);

  float learning_rate = 0.01f;
  float momentum = 0.9f;
  auto optimizer =
      std::make_shared<quasai::optim::SGD>(learning_rate, momentum);

  ml_model.compile(quasai::nn::Loss::MSE, optimizer);

  // Training data (batch_size, 64)
  quasai::core::Tensor X = quasai::core::Tensor::zeros({32, 64});
  quasai::core::Tensor y = quasai::core::Tensor::zeros({32, 10});

  std::cout << "Training simple MLP..." << std::endl;
  ml_model.train(X, y, 10, 32);

  std::cout << "Training complete!" << std::endl;

  return 0;
}
