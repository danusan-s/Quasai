#pragma once

#include "quasai/core/dtype.hpp"
#include "quasai/nn/module.hpp"
#include "quasai/ops/scalar_ops.hpp"
#include "quasai/utils/random.hpp"

namespace quasai::nn {

class Dropout : public Module {
public:
  Dropout(float p = 0.5f) : p_(p) {
    if (p_ < 0.0f || p_ >= 1.0f) {
      throw std::invalid_argument(
          "Dropout probability must be in the range [0, 1)");
    }
  }

  template <typename T> void populate_mask(core::Tensor &mask) {
    std::bernoulli_distribution dist(1.0f - p_);
    T *mask_data = mask.data<T>();
    size_t num_elements = core::total_size(mask.shape());
    for (size_t i = 0; i < num_elements; ++i) {
      mask_data[i] = dist(utils::RNG::instance().engine()) ? static_cast<T>(1)
                                                           : static_cast<T>(0);
    }
  }

  core::Tensor forward(const core::Tensor &input) override {
    if (training_) {
      if (mask_.buffer() == nullptr) {
        mask_ =
            core::Tensor::empty(input.shape(), input.dtype(), input.device());
      }
      core::dispatch_by_dtype(input.dtype(),
                              [&]<typename T>() { populate_mask<T>(mask_); });

      core::Tensor out = ops::mul(input, mask_);

      return core::dispatch_by_dtype(input.dtype(),
                                     [&]<typename T>() -> core::Tensor {
                                       T denom = static_cast<T>(1 - p_);
                                       return ops::div(out, denom);
                                     });
    }

    return input; // No dropout during evaluation
  }

private:
  core::Tensor mask_; // Mask tensor for dropout
  float p_;           // Dropout probability
};

} // namespace quasai::nn
