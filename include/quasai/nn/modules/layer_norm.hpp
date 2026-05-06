#pragma once

#include "quasai/nn/module.hpp"

namespace quasai::nn {

/**
 * @brief Layer Normalization layer.
 * @details Normalizes across the feature dimension:
 *          y = (x - mean) / sqrt(var + eps) * gamma + beta
 */
class LayerNorm : public Module {
public:
  /**
   * @brief Construct a LayerNorm layer using number of features.
   * @param num_features Number of features to normalize over.
   * @param eps Small constant for numerical stability (default: 1e-5f).
   * @param dtype Data type (default: FLOAT32).
   * @param device Device to allocate on (default: CPU).
   */
  LayerNorm(size_t num_features, float eps = 1e-5f,
            core::DType dtype = core::DType::FLOAT32,
            core::Device device = core::Device::cpu());

  /**
   * @brief Construct a LayerNorm layer using a shape.
   * @param shape Shape specifying the feature dimensions.
   * @param eps Small constant for numerical stability (default: 1e-5f).
   * @param dtype Data type (default: FLOAT32).
   * @param device Device to allocate on (default: CPU).
   */
  LayerNorm(core::Shape shape, float eps = 1e-5f,
            core::DType dtype = core::DType::FLOAT32,
            core::Device device = core::Device::cpu());

  /// @brief Forward pass: normalizes input across feature dimensions.
  core::Tensor forward(const core::Tensor &input) override;

private:
  float eps_;       ///< Small constant for numerical stability
  Parameter scale_; ///< Scale parameter (gamma)
  Parameter shift_; ///< Shift parameter (beta)
};

} // namespace quasai::nn
