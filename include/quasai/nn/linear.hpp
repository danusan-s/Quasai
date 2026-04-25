#pragma once

#include "quasai/nn/init.hpp"
#include "quasai/nn/module.hpp"

namespace quasai {

class Linear : public Module {
public:
  Linear(size_t in_features, size_t out_features,
         Initialization init = Initialization::GLOROT_UNIFORM);

  Tensor forward(const Tensor &input) override;

private:
  Parameter weight_;
  Parameter bias_;
};

} // namespace quasai
