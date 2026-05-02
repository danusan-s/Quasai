#pragma once

#include "quasai/autograd/function.hpp"

namespace quasai::autograd {

class NegFunction : public Function {
public:
  std::vector<core::Tensor> backward(const core::Tensor &grad_output) override;
};

class AbsFunction : public Function {
public:
  std::vector<core::Tensor> backward(const core::Tensor &grad_output) override;
};

class HeavisideFunction : public Function {
public:
  std::vector<core::Tensor> backward(const core::Tensor &grad_output) override;
};

class SignumFunction : public Function {
public:
  std::vector<core::Tensor> backward(const core::Tensor &grad_output) override;
};

class ReluFunction : public Function {
public:
  std::vector<core::Tensor> backward(const core::Tensor &grad_output) override;
};

class SigmoidFunction : public Function {
public:
  std::vector<core::Tensor> backward(const core::Tensor &grad_output) override;
};

class TanhFunction : public Function {
public:
  std::vector<core::Tensor> backward(const core::Tensor &grad_output) override;
};

class PowFunction : public Function {
public:
  float exponent; // Store the exponent for backward computation

  PowFunction(float exp) : exponent(exp) {
  }

  std::vector<core::Tensor> backward(const core::Tensor &grad_output) override;
};

} // namespace quasai::autograd
