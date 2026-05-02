#include "quasai/nn/modules/sequential.hpp"
#include <memory>

namespace quasai::nn {

Sequential::Sequential() {
}

void Sequential::add_module(std::unique_ptr<Module> module) {
  modules_.push_back(std::move(module));
  auto added_module = modules_.back().get();
  for (const auto &param : added_module->parameters()) {
    params_.push_back(param);
  }
}

core::Tensor Sequential::forward(const core::Tensor &input) {
  core::Tensor output = input;
  for (const auto &module : modules_) {
    output = module->forward(output);
  }
  return output;
}

void Sequential::set_train() {
  training_ = true;
  for (const auto &module : modules_) {
    module->set_train();
  }
}

void Sequential::set_eval() {
  training_ = false;
  for (const auto &module : modules_) {
    module->set_eval();
  }
}

} // namespace quasai::nn
