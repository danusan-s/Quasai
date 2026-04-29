// src/examples/cpp/linear_regression.cpp
// Linear regression example

#include "quasai/data/csv_parser.hpp"
#include "quasai/nn/model.hpp"
#include "quasai/nn/modules/activations.hpp"
#include "quasai/nn/modules/linear.hpp"
#include "quasai/nn/modules/sequential.hpp"
#include "quasai/optim/sgd.hpp"
#include "quasai/transform/standard_scaler.hpp"
#include <iostream>
#include <vector>

int main() {
  // - Prepare data
  // - Train model
  // - Evaluate performance

  auto data = quasai::data::parse_csv("../../examples/cpp/housing.csv");

  std::vector<std::string> columns_to_remove;
  for (auto &pair : data) {
    const std::string &feature_name = pair.first;
    std::vector<std::string> &values = pair.second;

    if (!quasai::data::clean_is_float(values)) {
      columns_to_remove.push_back(feature_name);
    }
  }

  for (const auto &col : columns_to_remove) {
    std::cout << "Removing non-numeric column: " << col << std::endl;
    data.erase(col);
  }
  std::string target_column = "median_income";

  std::vector<float> train_input_data;
  std::vector<float> train_target_data;
  std::vector<float> test_input_data;
  std::vector<float> test_target_data;

  std::cout << "Preparing data for training..." << std::endl;
  size_t num_samples = data.begin()->second.size();
  size_t num_features = data.size() - 1; // Exclude target column
                                         //
  std::cout << "Number of samples: " << num_samples << std::endl;
  std::cout << "Number of features: " << num_features << std::endl;

  for (size_t i = 0; i < num_samples; ++i) {
    bool is_test_sample = (rand() % 5 == 0); // 20% chance for test sample
    for (const auto &pair : data) {
      if (pair.first == target_column) {
        const std::vector<std::string> &values = pair.second;
        if (i >= values.size()) {
          throw std::runtime_error(
              "Inconsistent number of samples across features");
        }
        float value = std::stof(values[i]);
        if (is_test_sample) {
          test_target_data.push_back(value);
        } else {
          train_target_data.push_back(value);
        }
      } else {
        const std::vector<std::string> &values = pair.second;
        if (i >= values.size()) {
          throw std::runtime_error(
              "Inconsistent number of samples across features");
        }
        float value = std::stof(values[i]);
        if (is_test_sample) {
          test_input_data.push_back(value);
        } else {
          train_input_data.push_back(value);
        }
      }
    }
  }

  size_t train_samples = train_target_data.size();
  size_t test_samples = test_target_data.size();

  std::cout << "Training samples: " << train_samples << std::endl;
  std::cout << "Testing samples: " << test_samples << std::endl;

  std::vector<float> means(num_features, 0.0f);
  std::vector<float> stddevs(num_features, 0.0f);

  quasai::core::Tensor train_input = quasai::core::Tensor::from_data(
      train_input_data.data(), quasai::core::Shape{train_samples, num_features},
      quasai::core::DType::FLOAT32);

  quasai::core::Tensor train_target = quasai::core::Tensor::from_data(
      train_target_data.data(), quasai::core::Shape{train_samples, 1},
      quasai::core::DType::FLOAT32);

  quasai::core::Tensor test_input = quasai::core::Tensor::from_data(
      test_input_data.data(), quasai::core::Shape{test_samples, num_features},
      quasai::core::DType::FLOAT32);

  quasai::core::Tensor test_target = quasai::core::Tensor::from_data(
      test_target_data.data(), quasai::core::Shape{test_samples, 1},
      quasai::core::DType::FLOAT32);

  quasai::transform::StandardScaler input_scaler;
  input_scaler.fit(train_input);
  train_input = input_scaler.transform(train_input);
  test_input = input_scaler.transform(test_input);

  quasai::transform::StandardScaler target_scaler;
  target_scaler.fit(train_target);
  train_target = target_scaler.transform(train_target);
  test_target = target_scaler.transform(test_target);

  auto linear1 = std::make_shared<quasai::nn::Linear>(num_features, 64);
  auto relu1 = std::make_shared<quasai::nn::ReLU>();
  auto linear2 = std::make_shared<quasai::nn::Linear>(64, 32);
  auto relu2 = std::make_shared<quasai::nn::ReLU>();
  auto linear3 = std::make_shared<quasai::nn::Linear>(32, 1);

  auto sequential = std::make_shared<quasai::nn::Sequential>(
      std::vector<std::shared_ptr<quasai::nn::Module>>{linear1, relu1, linear2,
                                                       relu2, linear3});

  quasai::nn::Model model(sequential);

  size_t num_epochs = 10;
  size_t batch_size = 32;

  float learning_rate = 0.01f;
  float momentum = 0.3f;

  auto optimizer =
      std::make_shared<quasai::optim::SGD>(learning_rate, momentum);

  model.compile(quasai::nn::Loss::MSE, optimizer);

  std::cout << "Starting training..." << std::endl;

  model.train(train_input, train_target, num_epochs, batch_size);

  std::cout << "Evaluating model performance on training data..." << std::endl;

  quasai::core::Tensor loss =
      model.evaluate(test_input, test_target, quasai::nn::Loss::MSE);

  std::cout << "Final training loss: " << loss.data<float>()[0] << std::endl;

  // Pick a few samples to show predictions
  std::cout << "Sample predictions vs targets:" << std::endl;
  quasai::core::Tensor predictions = model.predict(test_input);

  quasai::core::Tensor predictions_unscaled =
      target_scaler.inverse_transform(predictions);
  quasai::core::Tensor targets_unscaled =
      target_scaler.inverse_transform(test_target);

  for (size_t i = 0; i < std::min<size_t>(20, test_samples); ++i) {
    float pred = predictions_unscaled.data<float>()[i];
    float target = targets_unscaled.data<float>()[i];
    std::cout << "Sample " << i + 1 << ": Predicted: " << pred
              << ", Target: " << target << std::endl;
  }

  return 0;
}
