#include "quasai/optim/adam.hpp"
#include "quasai/autograd/metadata.hpp"
#include "quasai/ops/tensor_ops.hpp"
#include "quasai/utils/logger.hpp"
#include <cmath>

namespace quasai::optim {

void Adam::compile(const std::vector<nn::Parameter> &parameters) {
  parameters_ = parameters;
  gradients_.resize(parameters.size());
  states_.resize(parameters.size());
  t_ = 1;

  LOG_DEBUG(("Adam optimizer compiled with " +
             std::to_string(parameters.size()) + " parameters.")
                .c_str());
}

void Adam::step() {
  if (parameters_.empty()) {
    throw std::runtime_error(
        "Adam optimizer not compiled with parameters. Call compile() first.");
  }

  LOG_DEBUG(("Performing Adam optimization step with learning rate " +
             std::to_string(learning_rate_) + ", beta1 " +
             std::to_string(beta1_) + ", beta2 " + std::to_string(beta2_) +
             ", epsilon " + std::to_string(epsilon_))
                .c_str());

  for (size_t i = 0; i < parameters_.size(); ++i) {
    nn::Parameter &param = parameters_[i];
    if (!autograd::tensor_requires_grad(param)) {
      continue;
    }
    core::Tensor grad = param.autograd_meta()->grad_;
    if (!grad.buffer()) {
      continue;
    }

    AdamState &state = states_[i];

    if (!state.m.buffer() || state.m.buffer()->size() == 0) {
      state.m = core::Tensor::zeros(grad.shape(), grad.dtype(), grad.device());
      state.v = core::Tensor::zeros(grad.shape(), grad.dtype(), grad.device());
    }

    state.m =
        ops::add(ops::mul(state.m, beta1_), ops::mul(grad, 1.0f - beta1_));

    state.v = ops::add(ops::mul(state.v, beta2_),
                       ops::mul(ops::pow(grad, 2), 1.0f - beta2_));

    core::Tensor m_hat = ops::div(state.m, 1.0f - std::pow(beta1_, t_));
    core::Tensor v_hat = ops::div(state.v, 1.0f - std::pow(beta2_, t_));

    const float *m_hat_data = m_hat.data<float>();
    const float *v_hat_data = v_hat.data<float>();

    for (size_t j = 0; j < core::total_size(param.shape()); ++j) {
      float update = m_hat_data[j] / (std::sqrt(v_hat_data[j]) + epsilon_);
      param.data<float>()[j] -= learning_rate_ * update;
    }
  }

  ++t_; // Increment step count
}

void Adam::zero_grad() {
  if (parameters_.empty()) {
    throw std::runtime_error(
        "Adam optimizer not compiled with parameters. Call compile() first.");
  }

  for (nn::Parameter &param : parameters_) {
    if (autograd::tensor_requires_grad(param)) {
      core::Tensor grad = param.autograd_meta()->grad_;
      if (!grad.buffer()) {
        continue;
      }
      std::memset(grad.buffer()->raw_data(), 0, grad.buffer()->size());
    }
  }
}

} // namespace quasai::optim
