#pragma once

#include "quasai/nn/parameter.hpp"
#include <vector>

namespace quasai {

class SGD {
public:
  SGD(std::vector<Parameter> parameters, float learning_rate)
      : parameters_(parameters), learning_rate_(learning_rate) {
  }

  void step();

  void zero_grad();

private:
  std::vector<Parameter> parameters_;
  float learning_rate_;
};

} // namespace quasai
