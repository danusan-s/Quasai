#include "quasai/nn/linear.hpp"
#include "quasai/nn/init.hpp"
#include "quasai/ops/tensor_ops.hpp"

namespace quasai {

Linear::Linear(size_t in_features, size_t out_features, Initialization init)
    : weight_(initialize({out_features, in_features}, DType::FLOAT32,
                         Device::cpu(), init)),
      bias_(
          initialize({1, out_features}, DType::FLOAT32, Device::cpu(), init)) {
  params_.push_back(&weight_);
  params_.push_back(&bias_);
}

Tensor Linear::forward(const Tensor &input) {
  Tensor output = add(matmul(input, transpose(weight_)), bias_);
  return sum_to_shape(output, Shape{weight_.shape()[0]});
}

} // namespace quasai
