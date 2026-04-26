#include "quasai/nn/loss.hpp"
#include "quasai/ops/tensor_ops.hpp"

namespace quasai {

Tensor mse_loss(const Tensor &predictions, const Tensor &targets) {
  if (predictions.dtype() != targets.dtype()) {
    throw std::runtime_error(
        "Predictions and targets must have the same data type");
  }

  Tensor diff = sub(predictions, targets);
  Tensor squared_diff = mul(diff, diff);
  Tensor mean_squared_diff = mean(squared_diff);
  return mean_squared_diff;
}

Tensor l1_loss(const Tensor &predictions, const Tensor &targets) {
  if (predictions.dtype() != targets.dtype()) {
    throw std::runtime_error(
        "Predictions and targets must have the same data type");
  }

  Tensor diff = sub(predictions, targets);
  Tensor abs_diff = abs(diff);
  Tensor mean_abs_diff = mean(abs_diff);
  return mean_abs_diff;
}

Tensor cross_entropy_loss(const Tensor &predictions, const Tensor &targets) {
  // Placeholder implementation for cross-entropy loss
  throw std::runtime_error("Cross-entropy loss not implemented yet");
}

Tensor compute_loss(const Tensor &predictions, const Tensor &targets,
                    Loss loss) {
  switch (loss) {
    case Loss::MSE:
      return mse_loss(predictions, targets);
    case Loss::L1:
      return l1_loss(predictions, targets);
    default:
      throw std::runtime_error("Unsupported loss type");
  }
}

} // namespace quasai
