#pragma once

#include "quasai/nn/parameter.hpp"
#include <iostream>
#include <vector>

namespace quasai {

class SGD {
public:
  SGD(std::vector<Parameter> parameters, float learning_rate,
      float momentum = 0.0f)
      : parameters_(parameters), learning_rate_(learning_rate),
        momentum_(momentum) {
    gradients_.resize(parameters.size());
    if (momentum >= 1.0f || momentum < 0.0f) {
      throw std::invalid_argument("Momentum must be in the range [0, 1)");
    }
  }

  void step();

  void zero_grad();

private:
  std::vector<Parameter> parameters_;
  std::vector<Tensor> gradients_;
  float learning_rate_;
  float momentum_;
};

} // namespace quasai
