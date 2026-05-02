// examples/cpp/simple_example.cpp
// Simple example demonstrating Quasai framework usage

#include "quasai/nn/model.hpp"
#include "quasai/nn/modules/activations.hpp"
#include "quasai/nn/modules/linear.hpp"
#include "quasai/nn/sequential_builder.hpp"
#include "quasai/optim/sgd.hpp"
#include <iostream>

int main() {
  // Create a simple MLP using SequentialBuilder
  auto network = quasai::nn::SequentialBuilder()
                     .add<quasai::nn::Linear>(64, 32)
                     .add<quasai::nn::ReLU>()
                     .add<quasai::nn::Linear>(32, 10)
                     .build_ptr();

  quasai::nn::Model ml_model(std::move(network));
  ml_model.set_loss(quasai::nn::Loss::MSE);
  ml_model.set_optimizer<quasai::optim::SGD>(0.01f, 0.9f);

  // Training data (batch_size, 64)
  quasai::core::Tensor X = quasai::core::Tensor::zeros({32, 64});
  quasai::core::Tensor y = quasai::core::Tensor::zeros({32, 10});

  std::cout << "Training simple MLP..." << std::endl;
  ml_model.train(X, y, 10, 32);

  std::cout << "Training complete!" << std::endl;

  return 0;
}
