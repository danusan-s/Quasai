#include "quasai/nn/module.hpp"

namespace quasai::nn {

class BatchNorm1D : public Module {
public:
  BatchNorm1D(size_t num_features, float eps = 1e-5f, float momentum = 0.1f);

  core::Tensor forward(const core::Tensor &input) override;

private:
  size_t num_features_;       // Number of features in the input
  float eps_;                 // Small constant for numerical stability
  float momentum_;            // Momentum for running statistics
  Parameter scale_;           // Scale parameter (gamma)
  Parameter shift_;           // Shift parameter (beta)
  core::Tensor running_mean_; // Running mean for inference
  core::Tensor running_var_;  // Running variance for inference
};

} // namespace quasai::nn
