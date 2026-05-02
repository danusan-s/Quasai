#pragma once

#include "quasai/nn/modules/sequential.hpp"

namespace quasai::nn {

class SequentialBuilder {
public:
  SequentialBuilder() {
    model_ = std::make_unique<Sequential>();
  }

  template <typename ModuleType, typename... Args>
  SequentialBuilder &&add(Args &&...args) && {
    auto module = std::make_unique<ModuleType>(std::forward<Args>(args)...);
    model_->add_module(std::move(module));
    return std::move(*this);
  }

  Sequential build() && {
    return std::move(*model_);
  }

  std::unique_ptr<Sequential> build_ptr() && {
    return std::move(model_);
  }

private:
  std::unique_ptr<Sequential> model_;
};

} // namespace quasai::nn
