#pragma once

#include "quasai/nn/parameter.hpp"

namespace quasai::nn {

/**
 * @brief Base class for all neural network modules.
 */
class Module {
public:
  /// @brief Forward pass to be implemented by derived classes.
  virtual core::Tensor forward(const core::Tensor &input) = 0;

  /// @brief Set module to training mode (enables dropout, etc.).
  virtual void set_train() {
    training_ = true;
  }

  /// @brief Set module to evaluation mode (disables dropout, etc.).
  virtual void set_eval() {
    training_ = false;
  }

  /// @brief Get a copy of all parameters in this module and its children.
  std::vector<Parameter> parameters() {
    std::vector<Parameter> params;
    for (Parameter param : params_) {
      params.push_back(param);
    }
    return params;
  }

  /// @brief Convenience operator to call forward().
  core::Tensor operator()(const core::Tensor &input) {
    return forward(input);
  }

  /// @brief Check if the module is in training mode.
  bool is_training() const {
    return training_;
  }

  virtual ~Module() = default;

protected:
  bool training_ = true; ///< Flag to indicate training or evaluation mode
  std::vector<Parameter> params_; ///< Store parameters for easy access
};

} // namespace quasai::nn
