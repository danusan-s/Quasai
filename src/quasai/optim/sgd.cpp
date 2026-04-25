#include "quasai/optim/sgd.hpp"
#include "quasai/autograd/metadata.hpp"
#include "quasai/ops/tensor_ops.hpp"

namespace quasai {

void SGD::step() {
  const size_t num_params = parameters_.size();
  Tensor momentum_scalar =
      Tensor::from_data(&momentum_, Shape{}, DType::FLOAT32);
  for (size_t i = 0; i < num_params; ++i) {
    Parameter &param = parameters_[i];
    if (!param.autograd_meta() || !param.autograd_meta()->requires_grad) {
      continue; // Skip parameters that don't require gradients
    }
    Tensor grad = param.autograd_meta()->grad;

    Tensor &velocity = gradients_[i];

    if (!velocity.buffer() || velocity.buffer()->size() == 0) {
      velocity = Tensor::zeros(param.shape(), param.dtype(), param.device());
    }

    velocity = add(mul(velocity, momentum_scalar), grad);

    const float *grad_data = grad.data<float>();
    const float *velocity_data = velocity.data<float>();

    for (size_t j = 0; j < total_size(param.shape()); ++j) {
      // Nesterov momentum
      float update = momentum_ * velocity_data[j] + grad_data[j];

      param.data<float>()[j] -= learning_rate_ * update;
    }
  }
}

void SGD::zero_grad() {
  for (Parameter &param : parameters_) {
    if (param.autograd_meta() && param.autograd_meta()->requires_grad) {
      Tensor grad = param.autograd_meta()->grad;
      std::memset(grad.buffer()->raw_data(), 0, grad.buffer()->size());
    }
  }
}

} // namespace quasai
