#pragma once

#include "quasai/core/tensor.hpp"

namespace quasai::transform {

class StandardScaler {
public:
  StandardScaler() = default;

  void fit(const core::Tensor &data);

  core::Tensor transform(const core::Tensor &data) const;

  core::Tensor inverse_transform(const core::Tensor &data) const;

  core::Tensor mean_;
  core::Tensor std_;
};

} // namespace quasai::transform
