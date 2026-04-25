#pragma once

#include "quasai/core/tensor.hpp"
#include "quasai/ops/tensor_ops.hpp"

namespace quasai {

enum class Loss { MSE };

inline Tensor mse_loss(const Tensor &predictions, const Tensor &targets) {
  if (predictions.dtype() != targets.dtype()) {
    throw std::runtime_error(
        "Predictions and targets must have the same data type");
  }

  Tensor diff = sub(predictions, targets);
  Tensor squared_diff = mul(diff, diff);
  Tensor mean_squared_diff = mean(squared_diff);
  return mean_squared_diff;
}

inline Tensor compute_loss(const Tensor &predictions, const Tensor &targets,
                           Loss loss) {
  switch (loss) {
    case Loss::MSE:
      return mse_loss(predictions, targets);
    default:
      throw std::runtime_error("Unsupported loss type");
  }
}

} // namespace quasai
