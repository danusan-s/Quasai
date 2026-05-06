#pragma once

#include "quasai/nn/parameter.hpp"
#include "quasai/optim/optimizer.hpp"
#include <vector>

namespace quasai::optim {

/**
 * @brief Stochastic Gradient Descent optimizer.
 */
class SGD : public Optimizer {
public:
  /**
   * @brief Construct an SGD optimizer.
   * @param learning_rate Learning rate for parameter updates.
   * @param momentum Momentum factor in [0, 1) (default: 0.0f).
   * @throws std::invalid_argument if momentum is not in [0, 1).
   */
  SGD(float learning_rate, float momentum = 0.0f)
      : learning_rate_(learning_rate), momentum_(momentum) {
    if (momentum >= 1.0f || momentum < 0.0f) {
      throw std::invalid_argument("Momentum must be in the range [0, 1)");
    }
  }

  /// @brief Compile the optimizer with the given parameters.
  void compile(const std::vector<nn::Parameter> &parameters) override;

  /// @brief Perform a single optimization step.
  void step() override;

  /// @brief Zero out all parameter gradients.
  void zero_grad() override;

private:
  std::vector<nn::Parameter> parameters_;
  std::vector<core::Tensor> gradients_;
  float learning_rate_;
  float momentum_;
};

} // namespace quasai::optim
