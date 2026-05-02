#pragma once

#include "quasai/nn/module.hpp"

namespace quasai::nn {

/**
 * Dropout layer for regularization. Takes in a dropout probability 'p' on
 * construction.
 * During training, it randomly zeroes out elements of the input tensor with
 * probability 'p' or effectively keep elements with a (1-p) probability and
 * scales the remaining elements by 1/(1-p) to maintain the expected value.
 * During evaluation, it returns the input tensor unchanged.
 */
class Dropout : public Module {
public:
  Dropout(float p = 0.5f);

  core::Tensor forward(const core::Tensor &input) override;

private:
  float p_; // Dropout probability
};

} // namespace quasai::nn
