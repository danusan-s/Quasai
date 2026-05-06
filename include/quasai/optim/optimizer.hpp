#pragma once

#include "quasai/nn/parameter.hpp"

namespace quasai::optim {

/**
 * @brief Base class for all optimizers.
 */
class Optimizer {
public:
  /// @brief Compile the optimizer with the given parameters.
  virtual void compile(const std::vector<nn::Parameter> &parameters) = 0;
  /// @brief Perform a single optimization step.
  virtual void step() = 0;
  /// @brief Zero out all parameter gradients.
  virtual void zero_grad() = 0;
  virtual ~Optimizer() = default;
};

} // namespace quasai::optim
