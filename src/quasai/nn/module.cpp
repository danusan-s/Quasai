#include "quasai/nn/module.hpp"

namespace quasai {

std::vector<Parameter> Module::parameters() {
  std::vector<Parameter> params;
  for (Parameter *param : params_) {
    params.push_back(*param);
  }
  for (Module *submodule : submodules_) {
    std::vector<Parameter> sub_params = submodule->parameters();
    params.insert(params.end(), sub_params.begin(), sub_params.end());
  }
  return params;
}

} // namespace quasai
