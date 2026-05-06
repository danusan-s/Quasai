#include "quasai/nn/modules/layer_norm.hpp"

#include "quasai/nn/init.hpp"
#include "quasai/ops/tensor_ops.hpp"

namespace quasai::nn {

LayerNorm::LayerNorm(size_t num_features, float eps, core::DType dtype,
                     core::Device device)
    : eps_(eps),
      scale_(initialize({num_features}, dtype, device, Initialization::ONES)),
      shift_(initialize({num_features}, dtype, device, Initialization::ZEROS)) {
  params_.push_back(scale_);
  params_.push_back(shift_);
}

LayerNorm::LayerNorm(core::Shape shape, float eps, core::DType dtype,
                     core::Device device)
    : eps_(eps), scale_(initialize(shape, dtype, device, Initialization::ONES)),
      shift_(initialize(shape, dtype, device, Initialization::ZEROS)) {
  params_.push_back(scale_);
  params_.push_back(shift_);
}

core::Tensor LayerNorm::forward(const core::Tensor &input) {
  if (input.shape().dimensions() != 2) {
    throw std::invalid_argument(
        "LayerNorm expects input of shape (batch_size, num_features)");
  }

  core::Tensor mean = ops::mean(input, 1); // Mean across feature dimension
  core::Tensor var = ops::mean(ops::pow(ops::sub(input, mean), 2),
                               0); // Variance across feature dimension

  core::Tensor normalized =
      ops::div(ops::sub(input, mean), ops::pow(ops::add(var, eps_), 0.5f));

  core::Tensor output = ops::add(ops::mul(normalized, scale_), shift_);

  return core::Tensor();
}

} // namespace quasai::nn
