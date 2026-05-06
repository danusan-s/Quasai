#include "quasai/nn/modules/batch_norm_1d.hpp"

#include "quasai/nn/init.hpp"
#include "quasai/ops/tensor_ops.hpp"

namespace quasai::nn {

BatchNorm1D::BatchNorm1D(size_t num_features, float eps, float momentum,
                         core::DType dtype, core::Device device)
    : eps_(eps), momentum_(momentum),
      scale_(initialize({num_features}, dtype, device, Initialization::ONES)),
      shift_(initialize({num_features}, dtype, device, Initialization::ZEROS)),
      running_mean_(core::Tensor::zeros({num_features}, dtype, device)),
      running_var_(core::Tensor::ones({num_features}, dtype, device)) {
  params_.push_back(scale_);
  params_.push_back(shift_);
}

core::Tensor BatchNorm1D::forward(const core::Tensor &input) {
  if (input.shape().dimensions() != 2) {
    throw std::invalid_argument(
        "BatchNorm1D expects input of shape (batch_size, num_features)");
  }

  core::Tensor mean = ops::mean(input, 0); // Mean across batch dimension
  core::Tensor var = ops::mean(ops::pow(ops::sub(input, mean), 2),
                               0); // Variance across batch dimension

  // Update running statistics
  float coeff = 1.0f - momentum_;
  running_mean_ =
      ops::add(ops::mul(running_mean_, coeff), ops::mul(mean, momentum_));
  running_var_ =
      ops::add(ops::mul(running_var_, coeff), ops::mul(var, momentum_));

  core::Tensor normalized =
      ops::div(ops::sub(input, mean), ops::pow(ops::add(var, eps_), 0.5f));

  core::Tensor output = ops::add(ops::mul(normalized, scale_), shift_);

  return core::Tensor();
}

} // namespace quasai::nn
