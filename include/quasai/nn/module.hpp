#pragma once

#include "quasai/nn/parameter.hpp"

namespace quasai {

class Module {
public:
  // Forward method to be implemented by derived classes
  virtual Tensor forward(const Tensor &input) = 0;

  std::vector<Parameter> parameters() {
    std::vector<Parameter> params;
    for (Parameter param : params_) {
      params.push_back(param);
    }
    return params;
  }

  // Call operator to allow using the module like a function
  Tensor operator()(const Tensor &input) {
    return forward(input);
  }

  virtual ~Module() = default;

protected:
  // Store parameters in a vector for easy access
  std::vector<Parameter> params_;
};

} // namespace quasai
