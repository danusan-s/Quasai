#pragma once

#include "quasai/nn/loss.hpp"
#include "quasai/nn/module.hpp"
#include "quasai/optim/optimizer.hpp"

namespace quasai {

class Model {
public:
  Model(std::unique_ptr<Module> module);

  void train(const Tensor &input, const Tensor &targets, Loss loss_fn,
             Optimizer &optimizer, size_t epochs = 1, size_t batch_size = 1);

  Tensor predict(const Tensor &input);

  Tensor evaluate(const Tensor &input, const Tensor &targets, Loss loss_fn);

  std::vector<Parameter> parameters() const;

private:
  std::unique_ptr<Module> module_;
};

} // namespace quasai
