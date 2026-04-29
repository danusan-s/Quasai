#include "quasai/nn/model.hpp"
#include "quasai/nn/init.hpp"
#include "quasai/nn/loss.hpp"
#include "quasai/nn/modules/activations.hpp"
#include "quasai/nn/modules/dropout.hpp"
#include "quasai/nn/modules/linear.hpp"
#include "quasai/nn/modules/sequential.hpp"
#include "quasai/ops/tensor_ops.hpp"
#include "quasai/optim/sgd.hpp"
#include "quasai/utils/logger.hpp"
#include <gtest/gtest.h>
#include <memory>
#include <vector>

TEST(Model, SingleModuleModel) {
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

  auto linear = std::make_shared<quasai::nn::Linear>(
      in_features, out_features, quasai::nn::Initialization::HE_UNIFORM);

  auto optimizer = std::make_shared<quasai::optim::SGD>(0.005f, 0.9f);

  quasai::nn::Model model(linear);
  model.compile(quasai::nn::Loss::MSE, optimizer);

  size_t epochs = 10;

  model.train(input, target, epochs);

  model.evaluate(input, target);
}

TEST(Model, SequentialModel) {
  size_t in_features = 3;
  size_t hidden_features = 8;
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

  auto sequential_model = std::make_shared<quasai::nn::Sequential>(
      std::vector<std::shared_ptr<quasai::nn::Module>>{linear1, relu, linear2});

  quasai::nn::Model model(sequential_model);

  auto optimizer = std::make_shared<quasai::optim::SGD>(0.001f, 0.9f);

  model.compile(quasai::nn::Loss::MSE, optimizer);

  model.train(input, target, 50);
  quasai::core::Tensor final_loss = model.evaluate(input, target);

  std::cout << "Final Loss: " << final_loss.data<float>()[0] << std::endl;

  EXPECT_LT(final_loss.data<float>()[0], 1e-2f);
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

  auto sequential_model = std::make_shared<quasai::nn::Sequential>(
      std::vector<std::shared_ptr<quasai::nn::Module>>{linear1, relu, linear2});

  auto optimizer = std::make_shared<quasai::optim::SGD>(0.001f, 0.9f);

  quasai::nn::Model model(sequential_model);
  model.compile(quasai::nn::Loss::MSE, optimizer);

  size_t epochs = 50;
  size_t batch_size = 10;

  quasai::core::Tensor initial_loss =
      model.evaluate(input_tensor, target_tensor);

  std::cout << "Initial Loss: " << initial_loss.data<float>()[0] << std::endl;

  model.train(input_tensor, target_tensor, epochs, batch_size);

  quasai::core::Tensor final_loss = model.evaluate(input_tensor, target_tensor);

  std::cout << "Final Loss: " << final_loss.data<float>()[0] << std::endl;

  EXPECT_LT(final_loss.data<float>()[0], 1e-2f);
}

TEST(Model, DropoutModel) {
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
  auto dropout = std::make_shared<quasai::nn::Dropout>(0.1f);
  auto linear2 = std::make_shared<quasai::nn::Linear>(
      hidden_features, out_features,
      quasai::nn::Initialization::GLOROT_UNIFORM);

  auto sequential_model = std::make_shared<quasai::nn::Sequential>(
      std::vector<std::shared_ptr<quasai::nn::Module>>{linear1, relu, dropout,
                                                       linear2});

  auto optimizer = std::make_shared<quasai::optim::SGD>(0.001f, 0.9f);

  quasai::nn::Model model(sequential_model);
  model.compile(quasai::nn::Loss::MSE, optimizer);

  size_t epochs = 50;
  size_t batch_size = 10;

  quasai::core::Tensor initial_loss =
      model.evaluate(input_tensor, target_tensor);

  std::cout << "Initial Loss: " << initial_loss.data<float>()[0] << std::endl;

  model.train(input_tensor, target_tensor, epochs, batch_size);

  quasai::core::Tensor final_loss = model.evaluate(input_tensor, target_tensor);

  std::cout << "Final Loss: " << final_loss.data<float>()[0] << std::endl;

  EXPECT_LT(final_loss.data<float>()[0], 1e-2f);
}
