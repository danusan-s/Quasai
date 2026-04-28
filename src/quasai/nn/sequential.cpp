#include "quasai/nn/sequential.hpp"

namespace quasai::nn {

Sequential::Sequential(const std::vector<std::shared_ptr<Module>> &modules)
    : modules_(modules) {
  for (const auto &module : modules_) {
    std::vector<Parameter> module_params = module->parameters();
    for (Parameter &param : module_params) {
      params_.push_back(param);
    }
  }
}

core::Tensor Sequential::forward(const core::Tensor &input) {
  core::Tensor output = input;
  for (const auto &module : modules_) {
    output = module->forward(output);
  }
  return output;
}

} // namespace quasai::nn
