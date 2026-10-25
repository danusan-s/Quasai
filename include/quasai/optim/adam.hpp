#pragma once

#include "quasai/nn/parameter.hpp"
#include "quasai/optim/optimizer.hpp"
#include <vector>

namespace quasai::optim {

/**
 * @brief Stochastic Gradient Descent optimizer.
 */
class Adam : public Optimizer {
public:
  /**
   * @brief Construct an SGD optimizer.
   * @param learning_rate Learning rate for parameter updates.
   * @param beta1 Exponential decay rate for the first moment estimates.
   * @param beta2 Exponential decay rate for the second moment estimates.
   * @param epsilon Small constant for numerical stability.
   * @throws std::invalid_argument if momentum is not in [0, 1).
   */
  Adam(float learning_rate, float beta1 = 0.9, float beta2 = 0.999,
       float epsilon = 1e-8f)
      : learning_rate_(learning_rate), beta1_(beta1), beta2_(beta2),
        epsilon_(epsilon) {
    if (beta1 < 0.0f || beta1 >= 1.0f) {
      throw std::invalid_argument("Beta1 must be in [0, 1).");
    }
    if (beta2 < 0.0f || beta2 >= 1.0f) {
      throw std::invalid_argument("Beta2 must be in [0, 1).");
    }
  }

  /// @brief Compile the optimizer with the given parameters.
  void compile(const std::vector<nn::Parameter> &parameters) override;

  /// @brief Perform a single optimization step.
  void step() override;

  /// @brief Zero out all parameter gradients.
  void zero_grad() override;

private:
  struct AdamState {
    core::Tensor m; // First moment vector
    core::Tensor v; // Second moment vector
  };

  std::vector<nn::Parameter> parameters_;
  std::vector<AdamState> states_;
  std::vector<core::Tensor> gradients_;
  float learning_rate_;
  float beta1_;
  float beta2_;
  float epsilon_;
  size_t t_ = 1; // Step count
};

} // namespace quasai::optim
