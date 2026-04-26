#include "quasai/nn/model.hpp"
#include "quasai/ops/tensor_ops.hpp"
#include "quasai/utils/logger.hpp"
#include <sstream>

namespace quasai {

Model::Model(std::shared_ptr<Module> module) : module_(module) {
}

void Model::train(const Tensor &input, const Tensor &targets, Loss loss_fn,
                  Optimizer &optimizer, size_t epochs, size_t batch_size) {
  const size_t num_samples = input.shape()[0];
  for (size_t epoch = 0; epoch < epochs; ++epoch) {
    float total_loss = 0.0f;
    for (size_t i = 0; i < num_samples; i += batch_size) {
      size_t current_batch_size = std::min(batch_size, input.shape()[0] - i);
      Tensor batch_input = slice(input, i, i + current_batch_size);
      Tensor batch_targets = slice(targets, i, i + current_batch_size);
      Tensor batch_output = module_->forward(batch_input);
      Tensor loss = compute_loss(batch_output, batch_targets, loss_fn);
      total_loss += loss.data<float>()[0];

      std::ostringstream batch_oss;
      batch_oss << "Batch " << (i / batch_size) + 1 << "/"
                << (num_samples + batch_size - 1) / batch_size
                << ", Loss: " << loss.data<float>()[0] << "\n";
      LOG_DEBUG(batch_oss.str().c_str());

      loss.backward();
      optimizer.step();
      optimizer.zero_grad();
    }

    std::ostringstream oss;
    oss << "Epoch " << epoch + 1 << ", Avg Loss: " << total_loss / num_samples
        << "\n";
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
