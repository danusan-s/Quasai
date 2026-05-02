#pragma once

#include "quasai/nn/loss.hpp"
#include "quasai/nn/module.hpp"
#include "quasai/optim/optimizer.hpp"
#include "quasai/utils/logger.hpp"
#include <memory>

namespace quasai::nn {

class Model {
public:
  Model(std::unique_ptr<Module> module);

  void set_loss(Loss loss_fn);

  template <typename OptimizerType, typename... Args>
  void set_optimizer(Args &&...args) {
    optimizer_ = std::make_unique<OptimizerType>(std::forward<Args>(args)...);
    optimizer_->compile(parameters());
  }

  void train(const core::Tensor &input, const core::Tensor &targets,
             size_t epochs = 1, size_t batch_size = 1);

  core::Tensor predict(const core::Tensor &input);

  core::Tensor evaluate(const core::Tensor &input, const core::Tensor &targets);

  std::vector<Parameter> parameters() const;

private:
  std::unique_ptr<Module> module_;
  Loss loss_fn_;
  std::unique_ptr<optim::Optimizer> optimizer_;
};

} // namespace quasai::nn
