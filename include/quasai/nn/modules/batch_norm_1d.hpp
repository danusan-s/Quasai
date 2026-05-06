#pragma once

#include "quasai/nn/module.hpp"

namespace quasai::nn {

/**
 * @brief 1D Batch Normalization layer.
 * @details Normalizes input over the batch dimension:
 *          y = (x - mean) / sqrt(var + eps) * gamma + beta
 */
class BatchNorm1D : public Module {
public:
  /**
   * @brief Construct a BatchNorm1D layer.
   * @param num_features Number of features (channels).
   * @param eps Small constant for numerical stability (default: 1e-5f).
   * @param momentum Momentum for updating running statistics (default: 0.1f).
   * @param dtype Data type (default: FLOAT32).
   * @param device Device to allocate on (default: CPU).
   */
  BatchNorm1D(size_t num_features, float eps = 1e-5f, float momentum = 0.1f,
              core::DType dtype = core::DType::FLOAT32,
              core::Device device = core::Device::cpu());

  /// @brief Forward pass (uses batch stats in train mode, running stats in
  /// eval).
  core::Tensor forward(const core::Tensor &input) override;

private:
  float eps_;                 ///< Small constant for numerical stability
  float momentum_;            ///< Momentum for running statistics
  Parameter scale_;           ///< Scale parameter (gamma)
  Parameter shift_;           ///< Shift parameter (beta)
  core::Tensor running_mean_; ///< Running mean for inference
  core::Tensor running_var_;  ///< Running variance for inference
};

} // namespace quasai::nn
