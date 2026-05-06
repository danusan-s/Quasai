#pragma once

#include "quasai/nn/module.hpp"
#include <memory>
#include <vector>

namespace quasai::nn {

/**
 * @brief Sequential container for stacking modules.
 * @details Applies modules in order: output =
 * module[n](...(module[1](module[0](input))...)
 */
class Sequential : public Module {
public:
  /// @brief Construct an empty Sequential container.
  Sequential();

  /// @brief Add a module to the end of the sequence.
  void add_module(std::unique_ptr<Module> module);

  /// @brief Forward pass: runs input through all modules in order.
  core::Tensor forward(const core::Tensor &input) override;

  /// @brief Set all contained modules to training mode.
  void set_train() override;

  /// @brief Set all contained modules to evaluation mode.
  void set_eval() override;

private:
  std::vector<std::unique_ptr<Module>> modules_; ///< Ordered list of modules
};

} // namespace quasai::nn
