#pragma once

#include "quasai/core/tensor.hpp"

namespace quasai::nn {

enum class Loss { MSE, L1, CROSS_ENTROPY };

core::Tensor mse_loss(const core::Tensor &predictions,
                      const core::Tensor &targets);

core::Tensor compute_loss(const core::Tensor &predictions,
                          const core::Tensor &targets, Loss loss);

} // namespace quasai::nn
