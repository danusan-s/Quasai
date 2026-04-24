#include "quasai/optim/sgd.hpp"
#include "quasai/autograd/metadata.hpp"

namespace quasai {

void SGD::step() {
  for (Parameter &param : parameters_) {
    if (!param.autograd_meta() || !param.autograd_meta()->grad_fn) {
      continue; // Skip parameters that don't require gradients
    }
    Tensor grad = param.autograd_meta()->grad;

    // Update parameter in-place: param = param - learning_rate * grad
    // This is a simplified version and does not handle all edge cases
    // like different data types, devices, etc.
    for (size_t i = 0; i < total_size(param.shape()); ++i) {
      param.data<float>()[i] -= learning_rate_ * grad.data<float>()[i];
    }
  }
}

void SGD::zero_grad() {
  for (Parameter &param : parameters_) {
    if (param.autograd_meta() && param.autograd_meta()->grad_fn) {
      Tensor grad = param.autograd_meta()->grad;
      std::memset(grad.buffer()->raw_data(), 0, grad.buffer()->size());
    }
  }
}

} // namespace quasai
