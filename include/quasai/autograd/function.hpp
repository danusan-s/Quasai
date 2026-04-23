
#pragma once
#include "quasai/core/tensor.hpp"
#include <vector>

namespace quasai {

class Function {
public:
  std::vector<Tensor> inputs;

  virtual std::vector<Tensor> backward(
      const Tensor &grad_output) = 0; // Pure virtual function for backward pass

  virtual ~Function() = default; // Virtual destructor for proper cleanup
};

class AddFunction : public Function {
public:
  std::vector<Tensor> backward(const Tensor &grad_output) override;
};

class SubFunction : public Function {
public:
  std::vector<Tensor> backward(const Tensor &grad_output) override;
};

class MulFunction : public Function {
public:
  std::vector<Tensor> backward(const Tensor &grad_output) override;
};

class DivFunction : public Function {
public:
  std::vector<Tensor> backward(const Tensor &grad_output) override;
};

class NegFunction : public Function {
public:
  std::vector<Tensor> backward(const Tensor &grad_output) override;
};

class ReluFunction : public Function {
public:
  std::vector<Tensor> backward(const Tensor &grad_output) override;
};

class SigmoidFunction : public Function {
public:
  std::vector<Tensor> backward(const Tensor &grad_output) override;
};

class TanhFunction : public Function {
public:
  std::vector<Tensor> backward(const Tensor &grad_output) override;
};

class MatMulFunction : public Function {
public:
  std::vector<Tensor> backward(const Tensor &grad_output) override;
};

} // namespace quasai
