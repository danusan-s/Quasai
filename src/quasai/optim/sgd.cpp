#include "quasai/optim/sgd.hpp"

#include "quasai/autograd/metadata.hpp"
#include "quasai/ops/tensor_ops.hpp"

namespace quasai::optim {

void SGD::compile(const std::vector<nn::Parameter> &parameters) {
  parameters_ = parameters;
  gradients_.resize(parameters.size());
}

void SGD::step() {
  if (parameters_.empty()) {
    throw std::runtime_error(
        "SGD optimizer not compiled with parameters. Call compile() first.");
  }

  const size_t num_params = parameters_.size();
  for (size_t i = 0; i < num_params; ++i) {
    nn::Parameter &param = parameters_[i];
    if (!param.autograd_meta() || !param.autograd_meta()->requires_grad) {
      continue;
    }
    core::Tensor grad = param.autograd_meta()->grad;

    core::Tensor &velocity = gradients_[i];

    if (!velocity.buffer() || velocity.buffer()->size() == 0) {
      velocity = core::Tensor::zeros(grad.shape(), grad.dtype(), grad.device());
    }

    velocity = ops::add(ops::mul(velocity, momentum_), grad);

    const float *grad_data = grad.data<float>();
    const float *velocity_data = velocity.data<float>();

    for (size_t j = 0; j < core::total_size(param.shape()); ++j) {
      float update = momentum_ * velocity_data[j] + grad_data[j];

      param.data<float>()[j] -= learning_rate_ * update;
    }
  }
}

void SGD::zero_grad() {
  if (parameters_.empty()) {
    throw std::runtime_error(
        "SGD optimizer not compiled with parameters. Call compile() first.");
  }

  for (nn::Parameter &param : parameters_) {
    if (param.autograd_meta() && param.autograd_meta()->requires_grad) {
      core::Tensor grad = param.autograd_meta()->grad;
      std::memset(grad.buffer()->raw_data(), 0, grad.buffer()->size());
    }
  }
}

} // namespace quasai::optim
