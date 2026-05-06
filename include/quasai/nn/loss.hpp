#pragma once

#include "quasai/core/tensor.hpp"

namespace quasai::nn {

/// @brief Supported loss functions.
enum class Loss {
  MSE,          ///< Mean Squared Error
  L1,           ///< L1 loss (Mean Absolute Error)
  CROSS_ENTROPY ///< Cross Entropy loss
};

/**
 * @brief Compute Mean Squared Error loss.
 * @param predictions Model predictions.
 * @param targets Ground truth values.
 * @return Tensor containing the MSE loss.
 */
core::Tensor mse_loss(const core::Tensor &predictions,
                      const core::Tensor &targets);

/**
 * @brief Compute loss based on the given Loss type.
 * @param predictions Model predictions.
 * @param targets Ground truth values.
 * @param loss Type of loss to compute.
 * @return Tensor containing the loss.
 */
core::Tensor compute_loss(const core::Tensor &predictions,
                          const core::Tensor &targets, Loss loss);

} // namespace quasai::nn
