#pragma once

#include "quasai/core/tensor.hpp"

namespace quasai {

enum class Loss { MSE, L1, CROSS_ENTROPY };

Tensor mse_loss(const Tensor &predictions, const Tensor &targets);

Tensor compute_loss(const Tensor &predictions, const Tensor &targets,
                    Loss loss);

} // namespace quasai
