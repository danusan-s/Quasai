#pragma once

#include "quasai/autograd/function.hpp"

namespace quasai::autograd {

/// @brief Function node for element-wise negation.
class NegFunction : public Function {
public:
  std::vector<core::Tensor> backward(const core::Tensor &grad_output) override;
};

/// @brief Function node for element-wise absolute value.
class AbsFunction : public Function {
public:
  std::vector<core::Tensor> backward(const core::Tensor &grad_output) override;
};

/// @brief Function node for Heaviside step function.
class HeavisideFunction : public Function {
public:
  std::vector<core::Tensor> backward(const core::Tensor &grad_output) override;
};

/// @brief Function node for sign function.
class SignumFunction : public Function {
public:
  std::vector<core::Tensor> backward(const core::Tensor &grad_output) override;
};

/// @brief Function node for ReLU activation.
class ReluFunction : public Function {
public:
  std::vector<core::Tensor> backward(const core::Tensor &grad_output) override;
};

/// @brief Function node for Sigmoid activation.
class SigmoidFunction : public Function {
public:
  std::vector<core::Tensor> backward(const core::Tensor &grad_output) override;
};

/// @brief Function node for Tanh activation.
class TanhFunction : public Function {
public:
  std::vector<core::Tensor> backward(const core::Tensor &grad_output) override;
};

/// @brief Function node for element-wise power.
class PowFunction : public Function {
public:
  float exponent; ///< Exponent used in the forward pass.

  /// @brief Construct with the given exponent.
  PowFunction(float exp) : exponent(exp) {
  }

  std::vector<core::Tensor> backward(const core::Tensor &grad_output) override;
};

/// @brief Function node for element-wise exponential.
class ExpFunction : public Function {
public:
  std::vector<core::Tensor> backward(const core::Tensor &grad_output) override;
};

/// @brief Function node for element-wise logarithm.
class LogFunction : public Function {
public:
  float base; ///< Base of the logarithm used in the forward pass.

  /// @brief Construct with the given base.
  LogFunction(float base) : base(base) {
  }

  std::vector<core::Tensor> backward(const core::Tensor &grad_output) override;
};

/// @brief Function node for GELU activation (tanh approximation).
class GeluFunction : public Function {
public:
  std::vector<core::Tensor> backward(const core::Tensor &grad_output) override;
};

} // namespace quasai::autograd
