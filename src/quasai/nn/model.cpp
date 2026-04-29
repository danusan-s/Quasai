#include "quasai/nn/model.hpp"
#include "quasai/ops/tensor_ops.hpp"
#include "quasai/optim/optimizer.hpp"
#include "quasai/utils/logger.hpp"
#include <sstream>

namespace quasai::nn {

Model::Model(std::shared_ptr<Module> module) : module_(module) {
}

void Model::compile(Loss loss_fn, std::shared_ptr<optim::Optimizer> optimizer) {
  loss_fn_ = loss_fn;
  optimizer->compile(parameters());
  optimizer_ = optimizer;
}

void Model::train(const core::Tensor &input, const core::Tensor &targets,
                  size_t epochs, size_t batch_size) {
  if (!optimizer_) {
    throw std::runtime_error(
        "Optimizer not set. Call compile() before training.");
  }

  // Set the module to training mode
  module_->train();

  std::ostringstream oss;
  oss << "Starting training for " << epochs << " epochs with batch size "
      << batch_size << "...\n";
  LOG_INFO(oss.str().c_str());

  core::Tensor train_input = input;
  if (input.shape().dimensions() < 2) {
    train_input = ops::reshape(input, core::Shape{1, input.shape()[0]});
  }

  core::Tensor train_targets = targets;
  if (targets.shape().dimensions() < 2) {
    train_targets = ops::reshape(targets, core::Shape{1, targets.shape()[0]});
  }

  const size_t num_samples = train_input.shape()[0];
  const size_t num_batches = (num_samples + batch_size - 1) / batch_size;
  constexpr size_t log_interval = 10;
  const size_t batches_per_log =
      std::max<size_t>(1, num_batches / log_interval);

  for (size_t epoch = 0; epoch < epochs; ++epoch) {
    float total_loss = 0.0f;
    size_t batch_counter = 0;

    LOG_INFO(("Epoch " + std::to_string(epoch + 1) + "/" +
              std::to_string(epochs) + "\n")
                 .c_str());

    for (size_t i = 0; i < num_samples; i += batch_size) {
      size_t current_batch_size = std::min(batch_size, num_samples - i);
      core::Tensor batch_input =
          ops::slice(train_input, i, i + current_batch_size);
      core::Tensor batch_targets =
          ops::slice(train_targets, i, i + current_batch_size);
      core::Tensor batch_output = module_->forward(batch_input);
      core::Tensor loss = compute_loss(batch_output, batch_targets, loss_fn_);
      total_loss += loss.data<float>()[0];

      if (++batch_counter % batches_per_log == 0 ||
          batch_counter == num_batches) {
        std::ostringstream batch_oss;
        batch_oss << "Batch " << (i / batch_size) + 1 << "/" << num_batches
                  << ", Loss: " << loss.data<float>()[0] << "\n";
        LOG_DEBUG(batch_oss.str().c_str());
      }

      loss.backward();
      optimizer_->step();
      optimizer_->zero_grad();
    }

    std::ostringstream oss;
    oss << "Avg Loss: " << total_loss / num_samples << "\n";
    LOG_INFO(oss.str().c_str());
  }
}

core::Tensor Model::predict(const core::Tensor &input) {
  module_->eval();
  return module_->forward(input);
}

core::Tensor Model::evaluate(const core::Tensor &input,
                             const core::Tensor &targets) {
  return compute_loss(predict(input), targets, loss_fn_);
}

std::vector<Parameter> Model::parameters() const {
  return module_->parameters();
}

} // namespace quasai::nn
