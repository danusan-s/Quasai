#pragma once

#include "quasai/nn/module.hpp"

namespace quasai::nn {

/**
 * @brief Dropout layer for regularization.
 * @details During training, randomly zeroes elements with probability p and
 *          scales remaining elements by 1/(1-p) to maintain expected value.
 *          During evaluation, returns input unchanged.
 */
class Dropout : public Module {
public:
  /**
   * @brief Construct a Dropout layer.
   * @param p Dropout probability (default: 0.5f).
   */
  Dropout(float p = 0.5f);

  /// @brief Forward pass (applies dropout in training mode).
  core::Tensor forward(const core::Tensor &input) override;

private:
  float p_; ///< Dropout probability
};

} // namespace quasai::nn
