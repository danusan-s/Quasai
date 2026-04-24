#pragma once

#include "quasai/core/tensor.hpp"
#include "quasai/utils/logger.hpp"
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

class SumFunction : public Function {
public:
  std::vector<Tensor> backward(const Tensor &grad_output) override;
};

class SumToShapeFunction : public Function {
public:
  std::vector<Tensor> backward(const Tensor &grad_output) override;
};

class BroadcastToShapeFunction : public Function {
public:
  std::vector<Tensor> backward(const Tensor &grad_output) override;
};

class MeanFunction : public Function {
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

class TransposeFunction : public Function {
public:
  std::vector<Tensor> backward(const Tensor &grad_output) override;
};

class ReshapeFunction : public Function {
public:
  std::vector<Tensor> backward(const Tensor &grad_output) override;
};

} // namespace quasai
