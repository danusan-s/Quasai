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

  core::Tensor mean = ops::mean(input, 1, true); // (batch, 1)
  core::Tensor centered = ops::sub(input, mean);
  core::Tensor var = ops::mean(ops::pow(centered, 2), 1, true); // (batch, 1)

  core::Tensor normalized =
      ops::div(centered, ops::pow(ops::add(var, eps_), 0.5f));

  return ops::add(ops::mul(normalized, scale_), shift_);
}

} // namespace quasai::nn
