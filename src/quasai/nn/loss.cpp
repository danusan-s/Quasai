#include "quasai/nn/loss.hpp"
#include "quasai/ops/tensor_ops.hpp"

namespace quasai::nn {

core::Tensor mse_loss(const core::Tensor &predictions, const core::Tensor &targets) {
  if (predictions.dtype() != targets.dtype()) {
    throw std::runtime_error(
        "Predictions and targets must have the same data type");
  }

  core::Tensor diff = ops::sub(predictions, targets);
  core::Tensor squared_diff = ops::mul(diff, diff);
  core::Tensor mean_squared_diff = ops::mean(squared_diff);
  return mean_squared_diff;
}

core::Tensor l1_loss(const core::Tensor &predictions, const core::Tensor &targets) {
  if (predictions.dtype() != targets.dtype()) {
    throw std::runtime_error(
        "Predictions and targets must have the same data type");
  }

  core::Tensor diff = ops::sub(predictions, targets);
  core::Tensor abs_diff = ops::abs(diff);
  core::Tensor mean_abs_diff = ops::mean(abs_diff);
  return mean_abs_diff;
}

core::Tensor cross_entropy_loss(const core::Tensor &predictions, const core::Tensor &targets) {
  throw std::runtime_error("Cross-entropy loss not implemented yet");
}

core::Tensor compute_loss(const core::Tensor &predictions, const core::Tensor &targets,
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

} // namespace quasai::nn
