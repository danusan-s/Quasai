#pragma once

#include "quasai/nn/init.hpp"
#include "quasai/nn/module.hpp"

namespace quasai::nn {

class Linear : public Module {
public:
  Linear(size_t in_features, size_t out_features,
         Initialization init = Initialization::GLOROT_UNIFORM);

  core::Tensor forward(const core::Tensor &input) override;

private:
  Parameter weight_;
  Parameter bias_;
};

} // namespace quasai::nn
