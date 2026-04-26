#include "quasai/transform/standard_scaler.hpp"
#include "quasai/ops/core_ops.hpp"
#include "quasai/ops/tensor_ops.hpp"
#include <cmath>

namespace quasai {

void StandardScaler::fit(const Tensor &data) {
  if (data.shape().dimensions() != 2) {
    throw std::runtime_error("StandardScaler::fit expects 2D input");
  }

  size_t M = data.shape()[0]; // samples
  size_t N = data.shape()[1]; // features

  mean_ = Tensor::zeros({N}, data.dtype(), data.device());
  std_ = Tensor::zeros({N}, data.dtype(), data.device());

  const float *x = data.data<float>();
  auto s = data.strides();

  float *mean_ptr = mean_.data<float>();
  float *std_ptr = std_.data<float>();

  // ---- compute mean ----
  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      size_t idx = i * s[0] + j * s[1];
      mean_ptr[j] += x[idx];
    }
  }

  for (size_t j = 0; j < N; ++j) {
    mean_ptr[j] /= M;
  }

  // ---- compute std ----
  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      size_t idx = i * s[0] + j * s[1];
      float diff = x[idx] - mean_ptr[j];
      std_ptr[j] += diff * diff;
    }
  }

  for (size_t j = 0; j < N; ++j) {
    std_ptr[j] = std::sqrt(std_ptr[j] / M);
    if (std_ptr[j] == 0.0f) {
      std_ptr[j] = 1.0f; // Avoid division by zero, feature has zero variance
    }
  }
}

Tensor StandardScaler::transform(const Tensor &data) const {
  if (mean_.shape().dimensions() == 0 || std_.shape().dimensions() == 0) {
    throw std::runtime_error("StandardScaler not fitted");
  }

  size_t M = data.shape()[0];
  size_t N = data.shape()[1];

  if (N != mean_.shape()[0]) {
    throw std::runtime_error("Feature mismatch in transform");
  }

  Tensor out = Tensor::empty(data.shape(), data.dtype(), data.device());

  const float *x = data.data<float>();
  float *y = out.data<float>();

  auto xs = data.strides();
  auto ys = out.strides();

  const float *mean_ptr = mean_.data<float>();
  const float *std_ptr = std_.data<float>();

  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      size_t x_idx = i * xs[0] + j * xs[1];
      size_t y_idx = i * ys[0] + j * ys[1];

      y[y_idx] = (x[x_idx] - mean_ptr[j]) / std_ptr[j];
    }
  }

  return out;
}

Tensor StandardScaler::inverse_transform(const Tensor &data) const {
  if (mean_.shape().dimensions() == 0 || std_.shape().dimensions() == 0) {
    throw std::runtime_error("StandardScaler not fitted");
  }

  size_t M = data.shape()[0];
  size_t N = data.shape()[1];

  Tensor out = Tensor::empty(data.shape(), data.dtype(), data.device());

  const float *x = data.data<float>();
  float *y = out.data<float>();

  auto xs = data.strides();
  auto ys = out.strides();

  const float *mean_ptr = mean_.data<float>();
  const float *std_ptr = std_.data<float>();

  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      size_t x_idx = i * xs[0] + j * xs[1];
      size_t y_idx = i * ys[0] + j * ys[1];

      y[y_idx] = x[x_idx] * std_ptr[j] + mean_ptr[j];
    }
  }

  return out;
}

} // namespace quasai
