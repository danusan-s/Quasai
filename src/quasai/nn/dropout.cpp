#include "quasai/nn/modules/dropout.hpp"

#include "quasai/ops/cpu_kernel.hpp"
#include "quasai/utils/random.hpp"

namespace quasai::nn {

Dropout::Dropout(float p) : p_(p) {
  if (p < 0.0f || p >= 1.0f) {
    throw std::invalid_argument(
        "Dropout probability must be in the range [0, 1)");
  }
}

core::Tensor Dropout::forward(const core::Tensor &input) {
  if (training_) {
    core::Tensor mask =
        core::Tensor::empty(input.shape(), input.dtype(), input.device());

    std::bernoulli_distribution dist(1.0f - p_);
    auto &engine = utils::RNG::instance().engine();

    dispatch_by_dtype(input.dtype(), [&]<typename T>() {
      ops::do_unary_op<T>(input, mask, [&dist, &engine, this](T x) {
        return dist(engine) ? static_cast<T>(1.0f / (1.0f - p_))
                            : static_cast<T>(0);
      });
    });

    return ops::mul(input, mask);
  }

  return input; // No dropout during evaluation
}

}; // namespace quasai::nn
