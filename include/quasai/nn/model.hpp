#pragma once

#include "quasai/nn/loss.hpp"
#include "quasai/nn/module.hpp"
#include "quasai/optim/optimizer.hpp"

namespace quasai::nn {

class Model {
public:
  Model(std::shared_ptr<Module> module);

  void compile(Loss loss_fn, std::shared_ptr<optim::Optimizer> optimizer);

  void train(const core::Tensor &input, const core::Tensor &targets,
             size_t epochs = 1, size_t batch_size = 1);

  core::Tensor predict(const core::Tensor &input);

  core::Tensor evaluate(const core::Tensor &input, const core::Tensor &targets,
                        Loss loss_fn);

  std::vector<Parameter> parameters() const;

private:
  std::shared_ptr<Module> module_;
  Loss loss_fn_;
  std::shared_ptr<optim::Optimizer> optimizer_;
};

} // namespace quasai::nn
