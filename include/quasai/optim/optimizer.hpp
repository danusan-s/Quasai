#pragma once

#include "quasai/nn/parameter.hpp"

namespace quasai::optim {

class Optimizer {
public:
  virtual void compile(const std::vector<nn::Parameter> &parameters) = 0;
  virtual void step() = 0;
  virtual void zero_grad() = 0;
  virtual ~Optimizer() = default;
};

} // namespace quasai::optim
