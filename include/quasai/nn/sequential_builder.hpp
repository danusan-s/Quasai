#pragma once

#include "quasai/nn/modules/sequential.hpp"

#include <concepts>

template <typename T>
concept ModuleDerived = std::derived_from<T, quasai::nn::Module>;

namespace quasai::nn {

/**
 * @brief Builder pattern for constructing Sequential models fluently.
 * @details Allows chaining: builder.add<Linear>(...).add<ReLU>(...).build()
 */
class SequentialBuilder {
public:
  /// @brief Construct a new builder with an empty Sequential.
  SequentialBuilder() {
    model_ = std::make_unique<Sequential>();
  }

  /**
   * @brief Add a module to the sequence (fluent interface).
   * @tparam ModuleType Type of module to add (must derive from Module).
   * @tparam Args Constructor argument types.
   * @param args Arguments to forward to the module constructor.
   * @return Builder reference for chaining.
   */
  template <ModuleDerived ModuleType, typename... Args>
  SequentialBuilder &&add(Args &&...args) && {
    auto module = std::make_unique<ModuleType>(std::forward<Args>(args)...);
    model_->add_module(std::move(module));
    return std::move(*this);
  }

  /// @brief Build and return the Sequential model (by value).
  Sequential build() && {
    return std::move(*model_);
  }

  /// @brief Build and return the Sequential model (as unique_ptr).
  std::unique_ptr<Sequential> build_ptr() && {
    return std::move(model_);
  }

private:
  std::unique_ptr<Sequential> model_; ///< The sequential being built
};

} // namespace quasai::nn
