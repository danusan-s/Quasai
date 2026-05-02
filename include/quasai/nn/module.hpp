#pragma once

#include "quasai/nn/parameter.hpp"

namespace quasai::nn {

class Module {
public:
  // Forward method to be implemented by derived classes
  virtual core::Tensor forward(const core::Tensor &input) = 0;

  virtual void set_train() {
    training_ = true;
  }

  virtual void set_eval() {
    training_ = false;
  }

  std::vector<Parameter> parameters() {
    std::vector<Parameter> params;
    for (Parameter param : params_) {
      params.push_back(param);
    }
    return params;
  }

  // Call operator to allow using the module like a function
  core::Tensor operator()(const core::Tensor &input) {
    return forward(input);
  }

  bool is_training() const {
    return training_;
  }

  virtual ~Module() = default;

protected:
  bool training_ = true; // Flag to indicate training or evaluation mode
  // Store parameters in a vector for easy access
  std::vector<Parameter> params_;
};

} // namespace quasai::nn
