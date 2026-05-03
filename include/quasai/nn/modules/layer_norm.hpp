#pragma once

#include "quasai/nn/module.hpp"

namespace quasai::nn {

class LayerNorm : public Module {
public:
  LayerNorm(size_t num_features, float eps = 1e-5f,
            core::DType dtype = core::DType::FLOAT32,
            core::Device device = core::Device::cpu());

  LayerNorm(core::Shape shape, float eps = 1e-5f,
            core::DType dtype = core::DType::FLOAT32,
            core::Device device = core::Device::cpu());

  core::Tensor forward(const core::Tensor &input) override;

private:
  float eps_;       // Small constant for numerical stability
  Parameter scale_; // Scale parameter (gamma)
  Parameter shift_; // Shift parameter (beta)
};

} // namespace quasai::nn
