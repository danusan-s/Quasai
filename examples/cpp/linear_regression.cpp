// src/examples/cpp/linear_regression.cpp
// Linear regression example

#include "quasai/nn/activations.hpp"
#include "quasai/nn/linear.hpp"
#include "quasai/nn/model.hpp"
#include "quasai/nn/sequential.hpp"
#include "quasai/optim/sgd.hpp"
#include "quasai/utils/csv_parser.hpp"
#include <iostream>
#include <vector>

int main() {
  // - Prepare data
  // - Train model
  // - Evaluate performance

  auto data = quasai::parse_csv("../../examples/cpp/housing.csv");

  std::vector<std::string> columns_to_remove;
  for (auto &pair : data) {
    const std::string &feature_name = pair.first;
    std::vector<std::string> &values = pair.second;

    if (!quasai::clean_is_float(values)) {
      columns_to_remove.push_back(feature_name);
    }
  }

  for (const auto &col : columns_to_remove) {
    std::cout << "Removing non-numeric column: " << col << std::endl;
    data.erase(col);
  }
  std::string target_column = "median_income";

  std::vector<float> input_data;
  std::vector<float> target_data;

  std::cout << "Preparing data for training..." << std::endl;
  size_t num_samples = data.begin()->second.size();
  size_t num_features = data.size() - 1; // Exclude target column
  std::cout << "Number of samples: " << num_samples << std::endl;
  std::cout << "Number of features: " << num_features << std::endl;

  for (size_t i = 0; i < num_samples; ++i) {
    for (const auto &pair : data) {
      if (pair.first == target_column) {
        const std::vector<std::string> &values = pair.second;
        if (i >= values.size()) {
          throw std::runtime_error(
              "Inconsistent number of samples across features");
        }
        target_data.push_back(std::stof(values[i]));
      } else {
        const std::vector<std::string> &values = pair.second;
        if (i >= values.size()) {
          throw std::runtime_error(
              "Inconsistent number of samples across features");
        }
        float value = std::stof(values[i]);
        input_data.push_back(value);
      }
    }
  }

  std::vector<float> means(num_features, 0.0f);
  std::vector<float> stddevs(num_features, 0.0f);

  for (size_t j = 0; j < num_features; ++j) {
    float sum = 0.0f;
    for (size_t i = 0; i < num_samples; ++i) {
      sum += input_data[i * num_features + j];
    }
    means[j] = sum / num_samples;

    float variance_sum = 0.0f;
    for (size_t i = 0; i < num_samples; ++i) {
      float diff = input_data[i * num_features + j] - means[j];
      variance_sum += diff * diff;
    }
    stddevs[j] = std::sqrt(variance_sum / num_samples);

    for (size_t i = 0; i < num_samples; ++i) {
      input_data[i * num_features + j] =
          (input_data[i * num_features + j] - means[j]) / stddevs[j];
    }
  }

  float target_mean = 0.0f;
  for (size_t i = 0; i < num_samples; ++i) {
    target_mean += target_data[i];
  }
  target_mean /= num_samples;

  float target_variance_sum = 0.0f;
  for (size_t i = 0; i < num_samples; ++i) {
    float diff = target_data[i] - target_mean;
    target_variance_sum += diff * diff;
  }
  float target_stddev = std::sqrt(target_variance_sum / num_samples);

  for (size_t i = 0; i < num_samples; ++i) {
    target_data[i] = (target_data[i] - target_mean) / target_stddev;
  }

  quasai::Tensor input = quasai::Tensor::from_data(
      input_data.data(), quasai::Shape{num_samples, num_features},
      quasai::DType::FLOAT32);

  quasai::Tensor target = quasai::Tensor::from_data(
      target_data.data(), quasai::Shape{num_samples, 1},
      quasai::DType::FLOAT32);

  auto linear1 = std::make_shared<quasai::Linear>(num_features, 10);
  auto relu = std::make_shared<quasai::ReLU>();
  auto linear2 = std::make_shared<quasai::Linear>(10, 1);

  auto sequential = std::make_shared<quasai::Sequential>(
      std::vector<std::shared_ptr<quasai::Module>>{linear1, relu, linear2});

  quasai::Model model(sequential);

  size_t num_epochs = 30;
  size_t batch_size = 32;

  float learning_rate = 0.01f;
  float momentum = 0.0f;
  quasai::SGD optimizer(model.parameters(), learning_rate, momentum);

  std::cout << "Starting training..." << std::endl;

  model.train(input, target, quasai::Loss::MSE, optimizer, num_epochs,
              batch_size);

  std::cout << "Evaluating model performance on training data..." << std::endl;

  quasai::Tensor loss = model.evaluate(input, target, quasai::Loss::MSE);

  std::cout << "Final training loss: " << loss.data<float>()[0] << std::endl;

  return 0;
}
