#pragma once

#include "quasai/nn/module.hpp"
#include <memory>
#include <vector>

namespace quasai::nn {

class Sequential : public Module {
public:
  Sequential();

  void add_module(std::unique_ptr<Module> module);

  core::Tensor forward(const core::Tensor &input) override;

  void set_train() override;

  void set_eval() override;

private:
  std::vector<std::unique_ptr<Module>> modules_;
};

} // namespace quasai::nn
