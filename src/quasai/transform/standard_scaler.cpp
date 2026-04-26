#include "quasai/transform/standard_scaler.hpp"
#include "quasai/ops/tensor_ops.hpp"

namespace quasai {

void StandardScaler::fit(const Tensor &data) {
  mean_ = mean(data);
  std_ = mean(abs(sub(data, mean_)));
}

Tensor StandardScaler::transform(const Tensor &data) const {
  if (mean_.shape().dimensions() == 0 || std_.shape().dimensions() == 0) {
    throw std::runtime_error(
        "StandardScaler must be fitted before calling transform");
  }
  return div(sub(data, mean_), std_);
}

Tensor StandardScaler::inverse_transform(const Tensor &data) const {
  if (mean_.shape().dimensions() == 0 || std_.shape().dimensions() == 0) {
    throw std::runtime_error(
        "StandardScaler must be fitted before calling inverse_transform");
  }
  return add(mul(data, std_), mean_);
}

} // namespace quasai
