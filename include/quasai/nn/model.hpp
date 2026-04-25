#include "quasai/nn/loss.hpp"
#include "quasai/nn/module.hpp"
#include "quasai/optim/optimizer.hpp"
#include "quasai/utils/logger.hpp"
#include <sstream>

namespace quasai {

class Model {
public:
  Model(std::shared_ptr<Module> module) : module_(module) {
  }

  void train(const Tensor &input, const Tensor &targets, Loss loss_fn,
             Optimizer &optimizer, size_t epochs = 1, size_t batch_size = 1) {
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

  Tensor predict(const Tensor &input) {
    return module_->forward(input);
  }

  Tensor evaluate(const Tensor &input, const Tensor &targets, Loss loss_fn) {
    Tensor output = module_->forward(input);
    return compute_loss(output, targets, loss_fn);
  }

private:
  std::shared_ptr<Module> module_;
};

} // namespace quasai
