#pragma once

#include "quasai/nn/loss.hpp"
#include "quasai/nn/module.hpp"
#include "quasai/optim/optimizer.hpp"

namespace quasai {

class Model {
public:
  Model(std::shared_ptr<Module> module);

  void compile(Loss loss_fn, Optimizer &optimizer);

  void train(const Tensor &input, const Tensor &targets, size_t epochs = 1,
             size_t batch_size = 1);

  Tensor predict(const Tensor &input);

  Tensor evaluate(const Tensor &input, const Tensor &targets, Loss loss_fn);

  std::vector<Parameter> parameters() const;

private:
  std::shared_ptr<Module> module_;
  Loss loss_fn_;
  Optimizer *optimizer_;
};

} // namespace quasai
