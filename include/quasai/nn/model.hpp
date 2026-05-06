#pragma once

#include "quasai/nn/loss.hpp"
#include "quasai/nn/module.hpp"
#include "quasai/optim/optimizer.hpp"
#include "quasai/utils/logger.hpp"
#include <memory>

namespace quasai::nn {

/**
 * @brief High-level model wrapping a Module, loss, and optimizer.
 */
class Model {
public:
  /**
   * @brief Construct a Model with the given module.
   * @param module Unique pointer to the module (takes ownership).
   */
  Model(std::unique_ptr<Module> module);

  /// @brief Set the loss function for training.
  void set_loss(Loss loss_fn);

  /**
   * @brief Set the optimizer with the given arguments.
   * @tparam OptimizerType Type of optimizer (e.g., SGD).
   * @tparam Args Constructor argument types.
   * @param args Arguments to forward to the optimizer constructor.
   */
  template <typename OptimizerType, typename... Args>
  void set_optimizer(Args &&...args) {
    optimizer_ = std::make_unique<OptimizerType>(std::forward<Args>(args)...);
    optimizer_->compile(parameters());
  }

  /**
   * @brief Train the model with the given data.
   * @param input Training input data.
   * @param targets Training target values.
   * @param epochs Number of training epochs (default: 1).
   * @param batch_size Batch size (default: 1).
   */
  void train(const core::Tensor &input, const core::Tensor &targets,
             size_t epochs = 1, size_t batch_size = 1);

  /// @brief Make predictions (runs forward in eval mode).
  core::Tensor predict(const core::Tensor &input);

  /// @brief Evaluate loss on given input and targets.
  core::Tensor evaluate(const core::Tensor &input, const core::Tensor &targets);

  /// @brief Get all parameters from the underlying module.
  std::vector<Parameter> parameters() const;

private:
  std::unique_ptr<Module> module_;              ///< Wrapped module
  Loss loss_fn_;                                ///< Loss function
  std::unique_ptr<optim::Optimizer> optimizer_; ///< Optimizer
};

} // namespace quasai::nn
