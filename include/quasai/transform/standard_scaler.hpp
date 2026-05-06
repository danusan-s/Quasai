#pragma once

#include "quasai/core/tensor.hpp"

namespace quasai::transform {

/**
 * @brief Standardize features by removing mean and scaling to unit variance.
 * @details Implements the standard scaling: (x - mean) / std
 *          for each feature independently.
 */
class StandardScaler {
public:
  /// @brief Construct an unftted StandardScaler.
  StandardScaler() = default;

  /**
   * @brief Compute mean and std from training data.
   * @param data Input data of shape (samples, features).
   */
  void fit(const core::Tensor &data);

  /**
   * @brief Apply standardization to data.
   * @param data Input data to transform.
   * @return Standardized data.
   */
  core::Tensor transform(const core::Tensor &data) const;

  /**
   * @brief Reverse the standardization.
   * @param data Standardized data.
   * @return Data in original scale.
   */
  core::Tensor inverse_transform(const core::Tensor &data) const;

  core::Tensor mean_; ///< Per-feature mean (1D tensor)
  core::Tensor std_;  ///< Per-feature standard deviation (1D tensor)
};

} // namespace quasai::transform
