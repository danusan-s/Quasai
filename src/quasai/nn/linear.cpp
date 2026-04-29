#include "quasai/nn/modules/linear.hpp"
#include "quasai/nn/init.hpp"
#include "quasai/ops/tensor_ops.hpp"

namespace quasai::nn {

Linear::Linear(size_t in_features, size_t out_features, Initialization init)
    : weight_(initialize({in_features, out_features}, core::DType::FLOAT32,
                         core::Device::cpu(), init)),
      bias_(initialize({1, out_features}, core::DType::FLOAT32,
                       core::Device::cpu(), init)) {
  params_.push_back(weight_);
  params_.push_back(bias_);
}

core::Tensor Linear::forward(const core::Tensor &input) {
  core::Tensor output = ops::add(ops::matmul(input, weight_), bias_);
  return output;
}

} // namespace quasai::nn
