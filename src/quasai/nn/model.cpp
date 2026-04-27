#include "quasai/nn/model.hpp"
#include "quasai/ops/tensor_ops.hpp"
#include "quasai/optim/optimizer.hpp"
#include "quasai/utils/logger.hpp"
#include <sstream>

namespace quasai {

Model::Model(std::shared_ptr<Module> module) : module_(module) {
}

void Model::compile(Loss loss_fn, std::shared_ptr<Optimizer> optimizer) {
  loss_fn_ = loss_fn;
  optimizer->compile(parameters());
  optimizer_ = optimizer;
}

void Model::train(const Tensor &input, const Tensor &targets, size_t epochs,
                  size_t batch_size) {
  if (!optimizer_) {
    throw std::runtime_error(
        "Optimizer not set. Call compile() before training.");
  }

  std::ostringstream oss;
  oss << "Starting training for " << epochs << " epochs with batch size "
      << batch_size << "...\n";
  LOG_INFO(oss.str().c_str());

  const size_t num_samples = input.shape()[0];
  const size_t num_batches = (num_samples + batch_size - 1) / batch_size;
  constexpr size_t log_interval = 10; // Log every 10% of the batches
  const size_t batches_per_log =
      std::max<size_t>(1, num_batches / log_interval);

  for (size_t epoch = 0; epoch < epochs; ++epoch) {
    float total_loss = 0.0f;
    size_t batch_counter = 0;

    LOG_INFO(("Epoch " + std::to_string(epoch + 1) + "/" +
              std::to_string(epochs) + "\n")
                 .c_str());

    for (size_t i = 0; i < num_samples; i += batch_size) {
      size_t current_batch_size = std::min(batch_size, input.shape()[0] - i);
      Tensor batch_input = slice(input, i, i + current_batch_size);
      Tensor batch_targets = slice(targets, i, i + current_batch_size);
      Tensor batch_output = module_->forward(batch_input);
      Tensor loss = compute_loss(batch_output, batch_targets, loss_fn_);
      total_loss += loss.data<float>()[0];

      if (++batch_counter % batches_per_log == 0 ||
          batch_counter == num_batches) {
        std::ostringstream batch_oss;
        batch_oss << "Batch " << (i / batch_size) + 1 << "/" << num_batches
                  << ", Loss: " << loss.data<float>()[0] << "\n";
        LOG_INFO(batch_oss.str().c_str());
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

Tensor Model::predict(const Tensor &input) {
  return module_->forward(input);
}

Tensor Model::evaluate(const Tensor &input, const Tensor &targets,
                       Loss loss_fn) {
  Tensor output = module_->forward(input);
  return compute_loss(output, targets, loss_fn);
}

std::vector<Parameter> Model::parameters() const {
  return module_->parameters();
}

} // namespace quasai
