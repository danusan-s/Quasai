#pragma once

#include "quasai/core/tensor.hpp"
namespace quasai {
Tensor add(const Tensor &a, const Tensor &b);
Tensor sub(const Tensor &a, const Tensor &b);
Tensor mul(const Tensor &a, const Tensor &b);
Tensor div(const Tensor &a, const Tensor &b);

Tensor matmul(const Tensor &a, const Tensor &b);
Tensor transpose(const Tensor &a);

Tensor neg(const Tensor &a);
Tensor relu(const Tensor &a);
Tensor sigmoid(const Tensor &a);
Tensor tanh(const Tensor &a);

Tensor sum(const Tensor &a);
Tensor mean(const Tensor &a);

} // namespace quasai
