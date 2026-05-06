#pragma once

#include "quasai/nn/init.hpp"
#include "quasai/nn/module.hpp"

namespace quasai::nn {

/**
 * @brief Fully connected linear layer.
 * @details Computes output = input @ weight + bias
 */
class Linear : public Module {
public:
  /**
   * @brief Construct a Linear layer.
   * @param in_features Number of input features.
   * @param out_features Number of output features.
   * @param init Weight initialization scheme (default: GLOROT_UNIFORM).
   * @param dtype Data type (default: FLOAT32).
   * @param device Device to allocate on (default: CPU).
   */
  Linear(size_t in_features, size_t out_features,
         Initialization init = Initialization::GLOROT_UNIFORM,
         core::DType dtype = core::DType::FLOAT32,
         core::Device device = core::Device::cpu());

  /// @brief Forward pass: output = input @ weight + bias.
  core::Tensor forward(const core::Tensor &input) override;

private:
  Parameter weight_; ///< Weight matrix of shape (in_features, out_features)
  Parameter bias_;   ///< Bias vector of shape (1, out_features)
};

} // namespace quasai::nn
