#pragma once

#include "quasai/nn/parameter.hpp"
#include "quasai/optim/optimizer.hpp"
#include <vector>

namespace quasai {

class SGD : public Optimizer {
public:
  SGD(float learning_rate, float momentum = 0.0f)
      : learning_rate_(learning_rate), momentum_(momentum) {
    if (momentum >= 1.0f || momentum < 0.0f) {
      throw std::invalid_argument("Momentum must be in the range [0, 1)");
    }
  }

  void compile(const std::vector<Parameter> &parameters) override;

  void step() override;

  void zero_grad() override;

private:
  std::vector<Parameter> parameters_;
  std::vector<Tensor> gradients_;
  float learning_rate_;
  float momentum_;
};

} // namespace quasai
